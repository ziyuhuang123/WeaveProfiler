---
name: weaveprofiler
description: Instrument and profile GPU kernels across NVIDIA H20/B300/Ampere sm_86 (RTX 30xx/A10/A40/A6000) and AMD MI308X in CUDA C++, HIP C++, Triton, or CuTeDSL, with validated software and hardware timelines. Use for source recorder edits, profiler capture, and delivery evidence; generic GPU benchmarking does not need this workflow.
---

# WeaveProfiler

WeaveProfiler is a portable agent skill and deterministic Python backend (`weaveprof`) for automated, verified fine-grained GPU kernel profiling across NVIDIA H20 (`sm_90a`), NVIDIA B300 (`sm_100a` / `sm_103a`), NVIDIA Ampere (`sm_86`, RTX 30xx / A10 / A40 / A6000), and AMD Instinct MI308X (`gfx942`) in CUDA C++, HIP C++, Triton, and CuTeDSL.

## Architecture & Python Skill Interface (`weaveprof`)

WeaveProfiler couples two modules in an iterative verification loop without modifying user kernel files or caller signatures:

1. **Knowledge-Guided Stage Identifier (`weaveprof/pipeline/scan/llm_probe_placer.py` + Warp-Level Exemplar Knowledge Base in `configs/`)**:
   - The LLM-based **Stage Identifier** (`wp.identify`, alias `wp.locate`) has **read-only** access to the target kernel source and consults a **Warp-Level Exemplar Knowledge Base** across `configs/languages/{cuda_cpp,hip_cpp,triton,cutedsl}/`, `configs/cards/`, and `configs/operators/operators.json`:
     - **Per-Warp Action Exemplars**: Bracket warp-level actions (`load`, `store`, `mma`/`wgmma`/`tcgen05`/`mfma`/`tl.dot`/`cute.gemm`, barrier waits strictly outside polling spin loops, and cross-GPU `pull`/`scatter`).
     - **Cross-Warp & Multi-Slot Split-Endpoint Pairing**: Pair asynchronous producer DMA issue (`begin` on DMA role) with consumer barrier wait return (`end` on Math role), or same-warp multi-slot async groups (`wgmma` / `cp.async`), via an arbitrary dynamic pipeline `slot` expression (e.g., `"k%4"` or `"stage"`).
   - Outputs `<run>/probe.json` declaring intra-CTA warp `roles` and stage `probes` (`id`, `begin: {role, line}`, `end: {role, line}`, optional `slot`).

2. **Deterministic Trace Verifier (`weaveprof/` + `weaveprof/pipeline/instrument/` + `weaveprof/pipeline/validate/`)**:
   - **Call-Stack-Intact Probe Injection (`weaveprof/pipeline/instrument/rule_engine.py`)**: Builds isolated baseline (cached across iterations) and instrumented copies in an isolated temp workspace (`weaveprof/workspace.py`) via language-native mechanisms:
     - *(a) CUDA C++ & HIP C++*: Declares a translation-unit-internal descriptor (`wp_fi_c`, macro alias `wp_g` on CUDA; `fi_g` / `wp_g` on HIP) bound from the host via `cudaMemcpyToSymbol` / `hipMemcpyToSymbol` (`recorders/cuda_cpp/wp_lite.cuh` & `recorders/hip_cpp/wp_lite_hip.h`), where `lane 0` per warp/wave writes 16-byte events via an RAII register cursor (`WpScope`) with zero atomics on the hot path.
     - *(b) Triton & CuTeDSL*: Because Python DSLs lack mutable device globals, `weaveprof/pipeline/instrument/rule_engine.py` AST-clones the kernel with trailing defaulted buffer arguments (stripping `@triton.autotune` from the instrumented clone so the Triton proxy reuses the baseline winning `best_config` without `do_bench` buffer overflow; and threading buffer descriptors through `@cute.jit` launchers into `@cute.kernel` via `recorders/cutedsl/wp_cute.py`).
   - **Two-Step Software-Hardware Timeline Alignment (`weaveprof/pipeline/validate/tools/fi_merge_perfetto.py`)**: Maps per-warp events onto a chip-wide nanosecond axis (`%globaltimer` on NVIDIA; `s_memrealtime` 100 MHz OPEN/CLOSE anchors calibrating `s_memtime` on AMD MI308X), then rigidly shifts software events into the external hardware profiler window (`[T_start, T_end]` from `nsys` or `rocprof-sys` / `rocprofv3`) via `ΔT = T_start - t_0` with non-negative `1–4 us` boundary slack verification, exporting native Perfetto TrackEvent protobuf (`.perfetto-trace` via `weaveprof/pipeline/validate/tools/fi_perfetto_proto.py`).
   - **Four Verification Criteria + Product-Level Hard Gates (`weaveprof/backend.py`)**:
     1. *Assembly Check (`assembly`)*: Inspects compiled machine instructions (NVIDIA SASS / AMD CDNA3 ISA) and resource metadata to verify core compute/memory instruction multiset and structure preservation (`core_ratio >= 0.95` or (`core_multiset_ratio >= 0.98` and `core_ratio >= 0.85`), with `lcs_ratio >= 0.60`), prevent extra shared-memory usage (`extra_smem_bytes <= 0`), and ensure local-memory stack spills and `STL`/`LDL` counts do not regress (`delta_spill_bytes <= 0`).
     2. *Output Check (`output`)*: Exact bitwise output tensor equality against the unmodified baseline on identical inputs.
     3. *Event Check & Product-Level Topology/Curve Gate (`events` / `event`)*:
        - Every `begin` timestamp closes with a matching `end` timestamp (`t_end >= t_begin`, zero unclosed re-entry, zero cross-nesting, zero orphan ends), `overflow == 0`, and `fires[id] > 0` for every declared probe (supporting multi-warp `producer -> consumer` `slot` FIFO handoffs).
        - **Disjoint-Sibling & Operator-Stage Gate**: Rejects outer macro-wrappers enclosing inner stages (`NESTED_WRAPPER_COLLAPSE`), single-stage/coarse-phase collapse, and missing core operator stages (e.g., attention kernels with online softmax MUST expose `load`, `mma`, `softmax`, and `epilogue`/`store`).
        - **All-Warp Completeness Hard Gate (`verify_warp_completeness`)**: At warp granularity (`WP_GRAN=0`, the default for CUDA C++, HIP C++, CuTeDSL, and Triton), **EVERY warp `0..warps_per_cta-1` MUST be recorded and plotted** (`WP_KERNEL_SCOPE(role)`, NEVER filtering to one representative warp per warpgroup like `warp_idx % 4 == 0`); partial warp coverage fails closed with `WARP_COVERAGE_INCOMPLETE`.
        - **Product-Level Hard Gate (`validate_trace_product`)**: Rejects single-CTA truncation (`ctas_observed == grid_ctas` across physical SMs/CUs), out-of-range warp IDs (`0 <= warp < warps_per_cta`), collapsed stages missing `load`/`mma`/`softmax`/`epilogue`, missing cross-warp `slot` handoffs on async `mbarrier` pipelines, or missing hardware `Compute Util (%)`, `Memory Util (%)`, and `Active SM/CU (%)` curves.
     4. *Overhead Check (`overhead`)*: Warm-launch median runtime change stays within `|t_inst / t_base - 1| <= limit` (default `0.10`, i.e., `±10%`; negative speedups beyond `-limit` also fail as scheduling perturbation).
   - **Structured Handoff & PNG-First Visual Delivery (`<run>/overview.png`, `<run>/feedback.json`, `<run>/verify.json`, `<run>/seal.json`)**:
     - By default, `run_megakernel_audit` (`render_overview_png=True`, `write_perfetto_gz=False`) and `wp.profile` auto-render a single-run **4-panel SW+HW overview PNG (`overview.png`, ~250–600 KB, ~1.4–2.5 s locally)** via `weaveprof/viz.py` on one locked microsecond axis across **both NVIDIA (`SM`/`warp`, `nsys`) and AMD MI308X (`CU`/`wave`, `rocprofiler-sdk` / `rocprof-sys`)**: **(A)** same-run real hardware utilization curves (`Compute / MFMA / Tensor Active`, `HBM Read+Write BW` summed across the shared DRAM bus, separate full-duplex `NVLink TX` and `NVLink RX` curves — NEVER summed or merged — or `xGMI` on AMD, optional `Inter-Node RDMA NIC` curve or honest HCA unavailable footnote, `Active SMs / CUs`), **(B)** all physical SMs/CUs dense multi-stage / dual-fabric waterfall, **(C)** single-SM / single-CU per-warp/wave timeline (upper main execution track + 0/1/2 on-demand thin async engine depth sub-tracks for `MEM` green / `MMA` blue with fixed-pixel `▼` issue pins), and **(D)** automatic deterministic single-CTA micro-zoom when full-kernel spacing is denser than `28 px` (`--zoom-cta` / `--zoom-window` overrides supported).
     - **Rich Stage Naming & Intra-Node vs Inter-Node Dual-Color Fabrics**: Every probe in `<run>/probe.json` should carry a descriptive algorithmic/fabric `id`, human-readable `label` (e.g., `"KV Tile Load (TMA)"`, `"QK^T Score MMA"`, `"Online Softmax (exp/rescale)"`, `"GEMM1 (FC1+SwiGLU)"`, `"GEMM2 (FC2)"`), and `fabric` scope (`on_chip`, `intra_node_nvlink`, `intra_node_xgmi`, `inter_node_rdma`, `inter_node_ib`) — never lazy bare `"load"` / `"mma"` labels. On multi-node traces (e.g., 2-node EP16 MI308X or H20/B300), `weaveprof/viz.py` automatically separates **Inter-Node RDMA / IB** (warm crimson `#D62728` / amber `#F28E2B`) from **Intra-Node xGMI / NVLink** (cool teal `#009FB7` / royal blue `#2563EB`) with side-by-side Panel B & C fabric legends and `>= 2 px` crisp rendering for streaming Combine/Reduce slices.
     - **Async MMA vs. Main-Track Row Separation Rule (Hopper `wgmma` / Blackwell `tcgen05` + Softmax/Epilogue)**: Whenever a kernel issues asynchronous Tensor Core MMA (`wgmma.mma_async`, `tcgen05.mma`)—especially when followed by or overlapped with non-MMA main-track stages like `Online Softmax`, `Rescale`, `Exposed Wait`, or `Epilogue Store`—register the async MMA event in `<run>/names.json` with `"async_engines": {"<mma_ev>": "mma"}` and `"slotted_events": {"<mma_ev>": {"id": "...", "slot": "k", "cross_warp": false}}` (marking distinct concurrent groups e.g. `QK=0u`, `PV=1u`), while keeping `Softmax` / `Wait` / `Store` on the upper main track. This renders **two clearly separated aligned rows per warp**: the upper main execution row (`Softmax` / `Wait` / `Store`) and the lower `MMA` async depth row (dark-blue `▼` issue pins + 5-shade blue in-flight depth strip; `MEM` uses matching dark-green `▼` issue pins + 5-shade green depth strip), directly exposing both MMA in-flight depth (`d=1/2/...`) and true MMA↔Softmax overlap.
     - **Panel (D) Minimum-Event Density Rule & No K0-Only Peeling**: Do NOT peel `k=0` outside a Triton K-loop and leave the main loop uninstrumented (which yields empty single-CTA zoom shells); on NVIDIA Triton GEMM/attention kernels, use `triton_ptx_async_hook` so every K-loop `cp.async` / `wgmma` / `mma.sync` / `wait_group` is captured cleanly at PTX level. `weaveprof/viz.py` enforces `>= 6` per-warp events per CTA for single-CTA micro-zoom and automatically falls back to a multi-CTA wave window (`>= 12` events) when individual CTAs have fewer events.
     - **Long-Running Kernels (`> 10 ms`) & Desktop WDDM Preemption**: Always bracket `nsys profile` to a single steady-state launch with `--capture-range=cudaProfilerApi --capture-range-end=stop` (avoiding 50M+ metric rows from warmup runs). On multi-millisecond kernels or laptop/desktop GPUs subject to DVFS frequency ramps or Windows WDDM context preemption (`~1 ms` display preempts), prefer `WP_TIMER=0` (`%globaltimer` wall clock) over `%clock64` linear interpolation so clock-rate changes or preemption pauses never skew software-to-hardware axis alignment.
     - No `.perfetto.json.gz` is written by default (pass `write_perfetto_gz=True` only when explicitly requesting raw Perfetto JSON export, or render offline via `weaveprof render <trace> -o overview.png`).
     - Alongside `overview.png`, `run_megakernel_audit` returns a `<2 KB` structured causal JSON report containing `hw_quartiles` (`[Q1..Q4]`), `stage_summary`, `megakernel_skew`, `causal_opportunities`, and `fidelity_gates` so both multimodal and text-only agents can diagnose bottlenecks directly.

### Fast <10 Min RSI Instrumentation Workflow (`weaveprof.fast_search`)

Avoid slow 30-minute sequential trial-and-error loops by following these four rules:
1. **Minimal `-I` header overlay (`build_minimal_include_overlay`)**: Write ONLY the 1–2 modified headers into a tiny overlay directory and compile with `-I <overlay_dir> -I <orig_include>` instead of copying hundreds of megabytes of upstream headers.
2. **Static stage checklist first (`detect_kernel_stage_requirements`)**: Verify every required stage (`load`, `mma`, `softmax`, `epilogue`/`store` for attention) as **disjoint sibling intervals** (never wrap the whole device call in an outer `attention`/`mma` macro-probe!) BEFORE spending time compiling.
3. **Parallel CPU compile + SASS pre-screen (`run_rsi_placement_search`)**: Compile candidate `(placement_strategy, recorder_mode)` arms in parallel on CPU cores, reject spills / SASS regressions before touching the GPU, and persist outcomes to `<cache>/rsi_priors.json` so winning strategies run first on subsequent rounds and kernels. Keep `-O3` intact (the scheduling-aware multiset+LCS assembly gate tolerates `ptxas -O3` instruction interleaving when core opcodes and spills are clean).
4. **Keep CPU reference & warmup out of `nsys`**: Cache baseline `.cubin` + baseline output + CPU reference ONCE during baseline bring-up; run `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-frequency=200000` around a single steady-state launch only, keeping `.sqlite` small (<10 MB) and export under a few seconds.

```python
import weaveprof as wp

# AI Coding Agent / MCP (`run.py install-agent --workspace ... --protect ...`) or Python SDK
spec = wp.identify("kernels/fa3_fwd.py", out="probe.json", card="h20")

# 1. Run Trace Verifier on stage specification (probe.json -> feedback.json + 4-panel PNG)
res = wp.verify("probe.json", feedback="feedback.json", out="overview.png",
                bench=["python", "bench.py"])

# 2. End-to-end closed-loop Stage Identifier + Trace Verifier (unchanged kernel & launch)
with wp.profile(source="kernels/fa3_fwd.py", limit=0.10, out="overview.png"):
    out = flash_attn_v3(q, k, v)
```

### AMD Instinct MI308X (`gfx942`, CDNA3) Hardware Utilization & Co-Axial Alignment Playbook

Always follow these verified rules when capturing software + hardware timelines on AMD MI308X (`gfx942`, 80 CUs across 4 XCCs × 20 CUs/XCC × 4 SIMDs/CU):
1. **Same-Run Hardware Counter Command**:
   ```bash
   HIP_VISIBLE_DEVICES=<idle_gpu> /usr/bin/rocprof-sys-run --use-rocm \
     --rocm-events SQ_VALU_MFMA_BUSY_CYCLES,GRBM_GUI_ACTIVE,SIMD_UTILIZATION,GPU_UTIL,TCC_EA0_RDREQ_DRAM,TCC_EA0_WRREQ_DRAM \
     --use-process-sampling false --use-sampling false -o rocprof_out -- python3 -B run_capture.py
   ```
   *(Unset `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` if set simultaneously with `HIP_VISIBLE_DEVICES` to prevent ROCm runtime abort.)*
2. **Exact SQLite/Perfetto Counter Extraction (`trace_processor_shell`)**:
   - Query the warm dispatch slice from `slice JOIN track` (`dur > 0`) to obtain `(ts, dur)`.
   - Query `counter JOIN counter_track` at `c.ts = <dispatch_ts>` (`GPU <EVENT> [0]`).
3. **Verified CDNA3 Utilization Formulas (`weaveprof/pipeline/validate/tools/fi_merge_perfetto.py`)**:
   - **Matrix Core MFMA Utilization (%)**: `100.0 * SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE * 80.0)` — because `GRBM_GUI_ACTIVE` sums active cycles across the 4 XCCs while each XCC has `20 CUs × 4 SIMDs = 80` SIMD pipelines.
   - **HBM3 Read+Write Bandwidth Utilization (%)**: `100.0 * (TCC_EA0_RDREQ_DRAM + TCC_EA0_WRREQ_DRAM) * 64.0 / (dur_ns * 3684.0)` (64 B per DRAM request against MI308X measured stream peak `3684 GB/s`).
   - **CU Residency & Physical Topology**: Decode `HW_ID` (`xcc_id`, `se_id`, `cu_id`, `simd_id`) into physical CUs `0..79` (`CU 000 [XCC0 SE0 CU0]`), and convert `s_memrealtime` (100 MHz wall-clock ticks) + `s_memtime` anchors to `ns` before applying `ΔT = T_start - t_0`.
   - **Kernel Performance Pitfall**: Naive scalar FP16 loads or unpadded `BM × 64` LDS arrays suffer 32-way LDS bank conflicts on CDNA3 (`MFMA % < 15%`); always use 128-bit vectorized loads (`uint4` / `global_load_dwordx4`) and padded LDS strides (`BK + 8`) when writing reference HIP MFMA kernels.

### Legacy Multi-Stage CLI (`run.py`)

Resolve the card, exact architecture, and language before reading mode-specific knowledge. B300 `sm_100a` and `sm_103a` are separate routes. On MI308X, CUDA-style `.cu` source resolves to `hip_cpp`; there is no separate HIP project category.

| Card / architecture | `cuda_cpp` | `hip_cpp` | `triton` | `cutedsl` | Capture |
|---|---|---|---|---|---|
| [H20 / `sm_90`](configs/cards/h20.json) | yes | no | yes | yes | `nsys` (200 kHz) |
| [B300 / `sm_100a`](configs/cards/b300_sm_100a.json) | yes | no | yes | yes | `nsys` (200 kHz) |
| [B300 / `sm_103a`](configs/cards/b300_sm_103a.json) | yes | no | yes | yes | `nsys` (200 kHz) |
| [MI308X / `gfx942`](configs/cards/mi308x.json) | `.cu` -> `hip_cpp` | yes | AMD Triton | no | same-run `rocprof-sys-run --rocm-events`, then `rocprofv3` fallback |
| [Ampere / `sm_86`](configs/cards/sm86.json) | yes | no | yes | yes | `nsys` (200 kHz on Linux; software-anchored fallback on WSL2; auto `WP_TIMER=1` clock64 ~27 ns) |

Language knowledge: [CUDA C++](configs/languages/cuda_cpp/patterns.json), [HIP C++](configs/languages/hip_cpp/patterns.json) and its [gfx942 recorder](configs/languages/hip_cpp/recorder_gfx942.json), [Triton on NVIDIA](configs/languages/triton/nvidia_patterns.json), [Triton on AMD](configs/languages/triton/amd_patterns.json), and [CuTeDSL](configs/languages/cutedsl/patterns.json).

## CLI

All commands enter through [run.py](run.py). Use absolute project, run, and evidence paths.

```bash
python3 -B run.py capture --card mi308x [--arch gfx942] --project P --out RUN [--collector auto|rocprof-sys|rocprofv3] [--deadline ISO8601_UTC] [--heartbeat PATH] --payload CMD [ARG...]
python3 -B run.py gemm-probe --card h20 --arch sm_90 --profile full|quick [--shape MxNxK] --gpu auto --out RUN --deadline-sec 1800 [--trace-processor PATH]
```
`run.py capture` is AMD MI308X only; on NVIDIA (`h20`/`b300`/`sm86`), use MCP tool `weaveprof_megakernel_audit` with `"capture_nsys": true` (`python3 -B weaveprof/aka_adapter.py megakernel-audit < tmp/request.json`) or run `nsys profile` directly (lines 39, 49).

B300 commands must pass the exact `--arch`; H20 and MI308X may omit it when `--card` resolves one architecture. Repeat `--source`/`--glob` to select multiple inputs. All staged and agent commands route through `run.py` (18 registered subcommands).

## GEMM probe fast path

For a one-command GEMM software/hardware measurement, use `gemm-probe`. It owns the full path: idle H20 selection, hash-cached off/entry/full builds, randomized repeated benchmarking, full-output verification, NCU HMMA comparison, 200 kHz NSYS capture, software decode, counter extraction, one-time-base alignment, and a real Perfetto import. The default 30-minute deadline is a hard process-tree deadline, not a suggestion.

```bash
# Fast integration feedback: 512x512x512, 3 rounds, 2 capture launches,
# NCU skipped, full-output checks and real Perfetto import retained.
python3 -B run.py gemm-probe --card h20 --arch sm_90 --profile quick --gpu auto \
  --out /absolute/new/quick-run --deadline-sec 300

# Final measurement: full shape, 35 full-output runs, NCU, 200 kHz NSYS.
python3 -B run.py gemm-probe --card h20 --arch sm_90 --profile full --gpu auto \
  --out /absolute/new/full-run --deadline-sec 1800
```

Run `quick` first and `full` once after quick passes. `quick` uses `512x512x512`, 3 benchmark rounds, 2 hardware-capture launches, and 15 independent full-output checks. It skips NCU but still performs the real Perfetto import. Its overhead percentage is diagnostic only: fixed per-CTA probe cost is amplified by the small shape, so quick does not enforce the delivery overhead threshold. `full` uses `2048x2048x4096`, 7 rounds (35 runs), NCU, and the complete overhead gate. Neither profile is a delivery receipt (`delivery_evidence=false`).

Measured on H20 on 2026-11-14 with a warm build cache: `quick=20.448 s`; `full=63.701 s`. The previous full path was `98.446 s`. The improvement comes from using gzip level 3 for large Chrome JSON debug traces (about 1 s instead of 12 s at level 9), optional `msgspec` encoding, and decoding a dump once before alignment. The level-3 debug trace is 18.2 MB instead of 14.9 MB and imports the same 1,050,704 slices with zero forbidden stats; do not restore level 9 without measuring end-to-end cost.

### Two-pass integration loop

Use this loop for every GEMM instrumentation attempt; it is the default way to spend the time budget.

1. **Survey with one `quick` path.** Start a fresh quick run and record every failure it exposes in one ledger: build/compile, ABI, correctness, software trace, decode, hardware capture, alignment, and trace import. The probe may stop at a blocking failure. Fix only that blocker, rerun from the last reusable cached stage, and continue the same survey until trace import succeeds or every reachable failure class is recorded. Do not start `full`, NCU, or a long NSYS capture while quick is red, and do not polish downstream code speculatively.
2. **Batch repairs.** Once quick reaches trace import, or every reachable stage has produced evidence, group failures by stage and fix independent issues in one edit batch. Use CPU-only checks such as compile preflight, decoder roundtrip, alignment fixtures, and trace import while preparing the batch. Do not pay for a GPU capture after each small edit.
3. **Rerun from the earliest changed dependency.** Use artifact hashes and stage manifests to choose the first invalid stage. Reuse unchanged builds, dumps, decoded traces, and counters. A second quick run is justified to verify the repaired integration path; further runs need a named, new failure.
4. **Promote once.** Run `full` after quick passes, as the final correctness and performance measurement. If full exposes a condition that quick could not exercise, preserve that run, make a concrete fix, and rerun only the affected full stages when their caches remain valid; do not return to exploratory all-stage loops.

Keep a short failure ledger with the stage, first failing command, artifact path, input hashes, whether the failure blocks downstream stages, and the earliest reusable stage. Choose the next action from that ledger rather than retrying the whole chain by habit.

- Generate FP16 inputs with `curandGenerateNormal`, compute an independent FP32 reference, and compare every element of every output on every benchmark variant. A pass requires `mismatches_total=0` and `nonfinite_total=0`; never sample the output or compare the kernel with itself.
- Decode each immutable dump once. `alignment` must consume the decoder's software trace and summary through `fi_align_nsys.py --decoded-trace ... --decoded-summary ...`. The summary binds the dump SHA, names/semantics SHA, and decoder SHA; reject stale caches instead of silently mixing runs.
- For trace-only failures, reuse `hardware/gemm_nsys.fi.bin`, `hardware/capture.sqlite`, and the decoded software artifacts. Rerun alignment offline; rerun only decode if the decoder or names contract changed. Do not rerun benchmark, NCU, or NSYS for a formatting/merge fix:

```bash
python3 -B "$SKILL/weaveprof/pipeline/validate/tools/fi_align_nsys.py" \
  "$RUN/hardware/gemm_nsys.fi.bin" \
  --names "$SKILL/configs/templates/gemm_probe/names.json" \
  --sqlite "$RUN/hardware/capture.sqlite" --kernel gemm_wmma --instance -1 \
  --metrics 'SMs Active|SM Issue|Tensor Active|SM Tensor Pipe Active|DRAM Read|DRAM Write' \
  --min-metrics-samples 3 --min-metric-curves 6 \
  --decoded-trace "$RUN/software/gemm_sw.perfetto.json.gz" \
  --decoded-summary "$RUN/software/gemm_sw.summary.json" \
  -o "$RUN/alignment/gemm_sw_hw.perfetto.json"
```

- Use hard iteration checkpoints: quick integration should finish within 10 minutes, and the full evidence path within 30 minutes. If either slips, preserve the failed run and stop repeating the whole pipeline; identify the last reusable artifact and fix only the failing downstream stage. A two-hour debugging loop followed by a sub-two-minute final run is a workflow failure, not a fast result.
- The bundled `gemm_wmma` kernel is a self-contained `nvcuda::wmma` correctness/capture fixture, not an H20 performance reference. When instrumenting a real operand GEMM, edit and measure that target's actual mainloop (including WGMMA/CUTLASS where present). Never relabel a WMMA fixture as WGMMA or infer WGMMA utilization from it.
- Keep workflow profiles and recorder capture settings separate. `--profile quick|full` changes the benchmark and validation matrix. WeaveProfiler enforces full-audit profiling without ad-hoc loop sampling: when inner-loop density triggers an unrolling cliff or pushes overhead > 10%, closed-loop Scheme A coarsening deterministically hoists/merges probes outward to enclosing stage boundaries.
- Report `detached` and `attached` measurements separately. Detached isolates the entry-side probe cost; only `full:attached` includes the recorder's actual write cost. Never present detached overhead as the recorder write cost.
- Calibrate registers from the target build's own `ptxas -v` evidence, using the maximum observed register count among the spill-free baseline variants as the candidate seed. Compile one candidate at a time and select the smallest cap with zero spill loads, zero spill stores, and zero stack. An H20 run may select 80 registers; that is measured evidence, never a default and never a rule to set `maxrregcount=80` everywhere.
- Read `%globaltimer` only on the thread that records the event. Hoisting the timer read across the whole block changes the measurement itself.
- Cache builds by source hash, recorder hash, exact architecture, NVCC identity, defines, and register-calibration policy. A warm cache should skip compilation and ptxas searches; retain the cache metadata so the selected variant remains auditable.
- Select only an idle GPU whose `nvidia-smi` model matches the requested card. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, keep physical index/UUID mappings in the manifest, and never evict or kill another user's process.
- Verify the trace by importing it with `trace_processor_shell`, checking forbidden import stats, minimum slice count, and real counter tracks. Successful JSON parsing alone is not trace-import evidence.
- Preserve failing manifests and logs. `gemm-probe` is a measurement probe and always sets `delivery_evidence=false`; it cannot substitute for `deliver` or a delivery receipt.

## Knowledge routing

Route through card + language: read the selected card and language first, then only the recipe for that route. Never merge recipes across cards or architectures.

- Cards hold architecture, recorder, clock, and profiler constraints. Recipe roots are `configs/recipes/nv/h20`, `configs/recipes/nv/b300`, and `configs/recipes/amd/mi308x`; when a route recipe exists, read only that recipe. If it is missing, report the knowledge gap instead of borrowing another route.
- Recipes hold language-specific operator writing patterns. [Operators](configs/operators/operators.json) hold algorithm semantics. [Metrics](configs/metrics/hardware.json) hold hardware-counter and curve definitions; use the [nsys metric definitions](configs/metrics/nsys_ep_moe.json) or [gfx942 counters](configs/metrics/rocprof_gfx942_moe.json) for the selected route.
- Recorder examples: [CUDA C++](examples/cuda_cpp/profiler.cuh), [Triton](examples/triton/profiler.py), [CuTeDSL](examples/cute_dsl/profiler.py), [HIP C++](examples/demo/demo_two_roles_hip.cpp), and the verified [MI308X Triton GEMM reference](examples/triton/card-mi308x/gemm_triton_instrumented_card-mi308x.py) with its [card-specific run notes](examples/triton/card-mi308x/README_card-mi308x.md).
- Templates: [capture harness](configs/templates/capture/capture_harness_reference.py), [ABI](configs/templates/capture/abi_reference.json), and [full-output comparison](configs/templates/compare_raw_arrays.py).
- Validation schemas and guidance: [scan schema](configs/validation/scan_schema.json), [contract example](configs/validation/contract_lock_example.json), and [failure fixes](configs/validation/fixes.json).

## Shared workflow

**scan.** Read only files selected by `--source` or `--glob` under `--project`. Emit canonical line-level candidates from [semantic_ids.json](configs/semantic_ids.json), with the exact path and line, and bind the result to each selected file's source SHA. Never scan the whole repository or modify source during scan.

**instrument.** Require exact source begin/end anchors or explicit line spans for every recorder event. Unknown semantics block the edit. A single line may have multiple scan hints (for example `tl.load(..., mask=...)` can suggest both `load` and `compute`); an explicit selection resolves that ambiguity only when the chosen canonical ID is one of the scan candidates. Selecting two different semantics for the same ambiguous line remains a hard error. Resolve the route's recorder and use the card, language, and recipe patterns rather than a generic insert. Edits are SHA-bound and atomic: reject a stale `base_sha256`, apply all edits or none, and roll back every touched file on failure. Event IDs are stable; `semantic_id` and `display_name` both use the canonical ID, such as `mma`, `gemm`, or `load`. Aliases are match hints only, never drifting display names.

On AMD Triton, generated interval calls use the MI308X wave64 recorder explicitly: `time_ptr, tag_ptr, aux_ptr, overflow_ptr, slot, selected, n`, then `EVENT`, `CAP=FI_CAP`, `THREADS=FI_THREADS`, and `TIMER=FI_TIMER`. The `fi_open`/`fi_close` guard and `slot, selected, n` binding come from the selected route recipe; NVIDIA Triton keeps its `rec, ovf, pid, n, FI` form. Do not copy one platform's call signature onto the other.

**capture.** Use `nsys` for NVIDIA. On MI308X use same-run `rocprof-sys-run --rocm-events` first and `rocprofv3` only when the preferred collector is unavailable or incompatible, recording the fallback reason. Capture must expose the four stable curve IDs exactly: `compute_util`, `memory_util`, `communication_util`, and `sm_active`. Missing evidence is `unavailable` with a reason, never zero and never a fabricated series. Keep software and hardware evidence distinct. Same-host interconnect evidence comes from NVLink or XGMI; cross-host communication requires NIC/RDMA/fabric evidence.

**PNG-Only + Mandatory Hardware-Curves Visual Delivery Policy (Hook- & Renderer-Enforced by Default):** The primary deliverable a user opens and inspects is the **4-panel SW+HW overview PNG** (`overview.png`, rendered directly via `wp.profile(..., out="overview.png")`, `weaveprof_megakernel_audit`, `weaveprof/viz.py`, or `wp_decode.py <dump> --names <names> -o overview.png`).
- **Fail-Closed Panel (A) Hardware Curves Gate**: `render_swhw_overview_png(..., require_hw_curves=True)` **REFUSES** to emit a SW-only PNG when Panel (A) has zero real hardware curves (`MISSING_HARDWARE_UTILIZATION_CURVES`). Always capture same-run `200 kHz nsys` (NVIDIA; on Linux use `sudo -n -E /usr/local/bin/nsys` if needed; on Windows enable NVIDIA Control Panel Developer GPU Performance Counters) or `rocprof-sys-run --rocm-events` (AMD MI308X). NEVER bypass with `--allow-no-hw` for user deliverables.
- **Async Engine Sub-Tracks & Triton PTX Auto-Async Pass**: For async pipelines (`cp.async`, TMA, `wgmma`, `tcgen05`), declare `async_engines` (`"mem"` / `"mma"`) and FIFO `slotted_events` plus explicit `Exposed Wait` (`wait_group` / `bar.sync` / `mbarrier_wait`) so Panels (C) & (D) show Main Track + Thin Async Depth Strip + Fixed-Pixel `▼` Issue Pins + Deterministic Single-CTA Micro-Zoom. For **NVIDIA Triton**, use `weaveprof.pipeline.instrument.triton_ptx_async.triton_ptx_async_hook` + `build_triton_ptx_async_names(stats)` (zero Python K-loop edits!) to extract compiled `cp.async` and `wgmma` depth + exposed waits automatically.
- **No Stray `.gz` Dumps**: Do **NOT** generate `.perfetto.json.gz`, `.trace.json.gz`, or `.perfetto-trace.gz` files by default — `weaveprof/guard_hook.py` actively blocks and cleans up stray `.gz` traces and refuses to let the agent stop until a valid non-empty `.png` deliverable exists (pass `--write-gz` / `write_perfetto_gz=True` / `WEAVEPROF_ALLOW_GZ=1` ONLY if the user explicitly asks for raw Perfetto debug export). `FI_LEVEL` selects recorder verbosity: `0=off`, `1=entry-only`, `2=full-events`. Probing enforces full deterministic event collection without ad-hoc loop sampling.

MI308X uses one canonical same-run rocprof-sys hardware layer, with optional independent checks:

- Primary collection runs `rocprof-sys-run --rocm-events` on the workload. Raw ROCm PM counter tracks and kernel dispatch slices are written to the same Perfetto proto. Pair each selected dispatch with its counter sample by same-run, same-dispatch start timestamp and record the declared timestamp tolerance. Do not splice counters from another run. Extract and merge this layer with [build_rocprof_sys_sw_hw.py](weaveprof/pipeline/capture/build_rocprof_sys_sw_hw.py).
- For the MI308X MFMA/GEMM route, derive both `compute_util` and auxiliary `tensor_util` from `100*SQ_VALU_MFMA_BUSY_CYCLES/(GRBM_GUI_ACTIVE*(CU_NUM*SIMD_PER_CU/XCC_NUM))`. The defaults are 80 CU, 4 SIMD/CU, and 4 XCC, so the denominator uses 80 SIMD lanes per XCC. The canonical event is labeled `compute_util [matrix_core]`; `otherData.hw_merge` records `source=rocprof-sys:SQ_VALU_MFMA_BUSY_CYCLES+GRBM_GUI_ACTIVE`, both native metrics, and `compute_pipe=matrix_core`. This same-run GEMM path has `measured` evidence on MI308X; do not extend that status to other operators, languages, or cards without their own run.
- `simd_util` is separate auxiliary evidence from `SIMD_UTILIZATION*100` with `compute_pipe=simd`; its stable track is `simd_util [SIMD/CU]`. `sm_active` is `GPU_UTIL`. Never substitute `SIMD_UTILIZATION`, `GPU_UTIL`, CUDA-core utilization, `GPU GFX Busy`, or `GPU UMC Busy` for Matrix Core activity. If the required MFMA/GEMM counters are absent or invalid, mark `compute_util` and `tensor_util` `unavailable` with a reason; never fill zero.
- `memory_util` is an explicitly estimated HBM utilization: `(TCC_EA0_RDREQ_DRAM + TCC_EA0_WRREQ_DRAM) * 64 B / dispatch_duration / 3684 GB/s`, with directional read/write series retained. The 3684 GB/s peak is measured on MI308X, and the trace metadata must state both the 64 B/request assumption and the measured peak. Label the result as an estimate, not a directly measured bandwidth percentage.
- If no same-dispatch XGMI/RDMA/fabric counter was collected, `communication_util` is `unavailable` with that reason. Never invent zero. GMI credit-stall cycles, EA-request minus DRAM-request differences, and amd-smi NIC or memory values are not a communication utilization series.
- rocprofv3 `MfmaUtil` is an optional independent calibration/fallback source, not the primary MI308X source. If used, label `rocprofv3:MfmaUtil` explicitly, use it directly as a percent rather than multiplying by 100, and keep fallback or calibration evidence separate from same-run rocprof-sys points.
- Optional amd-smi `gpu_busy`, `memory_busy`, and `gpu_memory_used_mib` polling tracks are coarse debug context only. They retain `canonical_mapping=null` and `auxiliary_only=true`; no amd-smi GPU-busy, memory-busy, or memory-used curve may be presented as `compute_util` or `memory_util`. The kernel selector defaults to the legacy `mfma_kernel`; for GEMM, pass the real name with `--kernel-name-contains`.
- ROCm PM counters are per-dispatch aggregates. Do not claim intra-dispatch resolution. A zero boundary after a measured span is a display break, not a missing-evidence sample.
- [get_trace_processor.py](weaveprof/pipeline/scan/get_trace_processor.py) verifies a real import before accepting a trace processor. On hosts whose glibc is too old for the current Perfetto release, it retries the pinned, SHA256-verified v47 linux-x86_64 binary and records the reason in a fallback metadata file next to the selected binary.

**validate.** Validate the completed capture against the instrumented artifact and its edit evidence:

- Run a full output comparison over every required output key and element, for all selected ranks. Sampling is not correctness evidence.
- Before comparing off, entry-only, and full variants, compile and execute each variant
  in at least one untimed launch so module loading, JIT finalization, allocator growth,
  and first-touch page costs are outside the measurement. Report first-launch and
  steady-state timings separately; the overhead decision uses the median of repeated
  warm launches and the declared limit. Do not relax the limit after a failure.
- Compare SASS/ISA disturbance before and after instrumentation and report instruction, register, shared-memory, or scheduling changes.
- Verify software/hardware alignment: device timestamps, profiler kernel window, and counters must share one documented time base.
- For NVIDIA DRAM utilization, prefer the explicitly combined bandwidth counter. When only directional read/write counters exist, sum them per timestamp; the per-sample maximum understates the shared DRAM interface and is invalid.
- For AMD Tensor/Matrix Core claims, require `tensor_util` and MFMA/GEMM `compute_util` to cite the same-run `rocprof-sys:SQ_VALU_MFMA_BUSY_CYCLES+GRBM_GUI_ACTIVE` derivation and its CU/SIMD/XCC constants. Verify same-run/same-dispatch timestamp pairing, a separate SIMD/CU `simd_util` series, `GPU_UTIL` as `sm_active`, and auxiliary-only GFX/UMC busy overlays. A rocprofv3 `MfmaUtil` may appear only as labeled independent calibration or fallback.
- Verify that an absent same-dispatch interconnect counter leaves `communication_util` explicitly `unavailable` with a reason, and that estimated HBM utilization records the 64 B/request model and 3684 GB/s measured peak.
- Verify buffer completeness and overflow: every begin has an end, unknown and unpaired events are zero, drops are zero, and overflow fails validation.

A native trace is accepted only after `trace_processor_shell` imports that file with no import errors and with non-empty slices. Static JSON parsing, checking only a filename extension, or listing a tar archive is not evidence. Run each SELECT below as a separate `trace_processor_shell -Q` invocation against the final `.perfetto-trace` or `.perfetto-trace.gz`; require exit code 0, `slices > 0`, `import_errors = 0`, and the queried `wave_tracks` equal to both `physical_waves` and `wave_tracks` in the export report:

```sql
SELECT
  (SELECT COUNT(*) FROM slice) AS slices,
  (SELECT COUNT(*) FROM stats
   WHERE value > 0
     AND (severity = 'error'
          OR name IN ('json_parser_failure',
                      'json_tokenizer_failure',
                      'slice_spill_overlapping_complete_event',
                      'slice_drop_overlapping_complete_event',
                      'trace_sorter_negative_timestamp_dropped'))) AS import_errors,
  (SELECT COUNT(*) FROM track
   WHERE name GLOB 'wave *' OR name GLOB 'cta * / warp *') AS wave_tracks;

SELECT name, COUNT(*) AS slices
FROM slice
WHERE name IN ('load', 'mma', 'store', 'gemm')
GROUP BY name
ORDER BY name;

SELECT t.name,
       (SELECT COUNT(*) FROM counter AS c WHERE c.track_id = t.id) AS counter_points,
       (SELECT GROUP_CONCAT(DISTINCT s.name) FROM slice AS s WHERE s.track_id = t.id) AS slice_markers
FROM track AS t
WHERE t.name LIKE '%util%' OR t.name LIKE 'sm_active%'
   OR t.name LIKE 'software_%_active_waves%'
ORDER BY t.name;
```

Reconcile the native `load`, `mma`, and `store` counts with the export report's `semantic_counts`, and require exactly one native `gemm` envelope. Restrict the wave-track slices to those three canonical semantic names. Require every `hardware` entry in the export report to have its named track and counter points, and require the count of those utilization tracks to equal the report's `hardware_tracks`; for missing interconnect evidence, require the `communication_util [interconnect]` track and its `unavailable` marker in `slice_markers`. A preview has reduced counts and cannot be accepted as the full delivery.

The merged Chrome JSON debug input may use `trace__<model>_<stage>__card-<h20|b300|mi308x>__bs<B>_sq<S>[_kv<K>]__<dtype>__grid<G>_warp<W>_ph<P>__<bundle>.perfetto.json.gz`. The final native trace name must use the same path with an explicit `<fi-config>.perfetto-trace` token, or the same name with `.gz`; the receipt name is `<trace>.delivery_receipt.json`. For MI308X, `<fi-config>` is `fi-l2`. Both include the literal canonical card token `card-h20`, `card-b300`, or `card-mi308x`. Keep B300 `sm_100a` and `sm_103a` in separate contracts, runs, and evidence; never collapse or relabel one architecture as the other.

## Evidence levels and speed

`fast` is development feedback only and must emit `delivery_evidence: false`; it may run limited shapes, ranks, and event windows. `check` is a quick validation pass and cannot issue a final receipt. `deliver` is the complete evidence path: it runs the full validation matrix and may issue a receipt only after the complete checks and real trace import pass.

Evidence is per card and exact architecture. A no-card compile may be labeled only `compiled`, `simulated`, or `unverified`, never `measured`. An H20 run does not raise B300 evidence, and a B300 `sm_100a` run does not raise `sm_103a` evidence.

WeaveProfiler enforces full-audit profiling (`FI_LEVEL=2`) across all platforms. Probes are placed at canonical stage boundaries. If fine-grained inner-loop probes cause overhead > 10% or compiler unrolling cliffs, closed-loop Scheme A coarsening deterministically hoists or merges them outward to enclosing stage boundaries until all 4 verification gates pass.

H20 and B300 route selection is unchanged, and MI308X HIP C++ continues through the same `scan -> instrument -> capture -> validate` workflow as MI308X Triton. No HIP-only capture or validation path is introduced.

MI308X recorders use `s_memtime` for semantic intervals and OPEN/CLOSE `s_memrealtime` anchors for the observation window. Report the observation window and the canonical semantic envelope separately. Pairing a hardware dispatch to the software run is an integration placement: record the source clock offset and set `start_placement_measured=false` unless an absolute clock pairing was actually measured. Check alignment with the hardware/software duration ratio; do not assert absolute timestamp equality from a display offset.

The same-run rocprof-sys PM counters are per-dispatch aggregates. A hardware utilization track may therefore be constant across one kernel's displayed duration; that is the measured representation, not an instantaneous intra-kernel series. Never subdivide or interpolate that value to simulate a curve. If intra-kernel hardware resolution is required, use an independently labeled sampling method with its own evidence and limitations.

Reuse one compiled artifact and one GPU lease across validation phases. Cache each phase output by input hashes and reuse it only when those hashes match. Give every profiler process an absolute UTC deadline and heartbeat path; terminate on expiry and retain the failure evidence. Keep `fast` shape limits for iteration while `deliver` uses the complete shape, rank, and required hardware matrix. Decompress, copy, hash, and compress each trace once per immutable input set; share the parsed trace across checks.

Missing `communication_util` evidence stays `unavailable` with the collector-specific reason. A real measured zero and missing evidence are different states; never zero-fill a missing counter or relabel a stall/busy proxy as communication utilization.

Do not introduce or depend on MLIR anywhere in scan, edits, capture, or validation.

## Declarative `<run>/probe.json` & `<run>/feedback.json` Schema Reference

```json
// probe.json
{
  "roles": {"dma": "warps 0-3", "math": "warps 4-7"},
  "probes": [
    {"id": "load", "begin": {"role": "dma", "line": 101}, "end": {"role": "math", "line": 125}, "slot": "stage"},
    {"id": "mma",  "begin": {"role": "math", "line": 126}, "end": {"role": "math", "line": 137}}
  ]
}
```

```json
// feedback.json (on violation -> hoist inner-loop endpoints to enclosing_loop)
{
  "status": "FAIL",
  "failed": ["assembly", "overhead"],
  "fires": {"load": 64, "mma": 4096},
  "enclosing_loop": [123, 138]
}
```

CLI entry points (`weaveprof` / `python -m weaveprof`):
- `weaveprof identify --source <source> -o <out_probe> [--card h20|b300|mi308x] [--feedback <in_feedback>]`
- `weaveprof verify --probe <in_probe> --feedback <out_feedback> -o out.perfetto-trace -- <bench_cmd...>`
- `weaveprof profile --source <source> --card <h20|b300|mi308x> --limit 0.10 -o out.perfetto-trace -- <bench_cmd...>`
- `weaveprof render <trace.perfetto.json[.gz]> -o overview.png [--title T] [--sm 0] [--dpi 220]`
 *(Positional `<source>` / `<in_probe>` and `locate` / `validate` aliases are also accepted; `run.py` maintains backward-compatible CLI routing for automated test harnesses.)*

## LLM vs Deterministic Boundary (Agent-Callable Entry Points)

The LLM decides only **where** probes go (`<run>/probe.json`) and **what to change next** in the kernel. Every accept/reject decision is fixed Python code that the LLM cannot write to or bypass:

| Entry point | Who decides | Deterministic guarantee |
|---|---|---|
| `wp.identify(...)` | LLM (read-only source) | Output is schema-validated `<run>/probe.json`; spin-loop placement rejected by `PolicyGuard`; semantic cue warnings attached before verification. |
| `wp.verify(...)` / `wp.profile_for_agent(...)` | Harness | Re-compiles baseline + instrumented copy itself; 4 gates (scheduling-aware SASS/spill, bitwise output, events + granularity + attention/epilogue coverage, symmetric overhead); HMAC-SHA256 `<run>/seal.json` binding source, `<run>/verify.json`, and artifacts; `verify_seal()` fails closed on any hash mismatch or missing HMAC key. |
| `wp.run_compiler_feedback_selection(compile_fn, src, ...)` | Harness | Compiles single-register bit-packed context (`WP_CTX_REG`, 0 B SMEM, 32,767 records/lane cap) first; only on measured spill / SASS `STL`/`LDL` regression falls back to static shared-memory (`WP_CTX_SMEM`, 512 B at warp granularity or 16 B at CTA granularity) + `__noinline__` (`WP_CTX_REG_NOINLINE`) (and `{pre,post}_setmaxnreg`), ranking by `select_compiler_feedback_arm()`. Returns `NO_CLEAN_ARM` instead of guessing. |
| `wp.run_rsi_placement_search(...)` / `wp.build_minimal_include_overlay(...)` | Harness (RSI) | Minimal `-I` header overlay, static stage-completeness & disjoint-sibling filter (rejecting missing `softmax`/`epilogue` or whole-kernel wrappers before compiling), parallel CPU compile/SASS pre-screen, fast GPU screen, and persistent local `<cache>/rsi_priors.json` ranking. |
| `wp.analyze_sw_hw_causality(...)` | Harness (D3) | Calibrated HW regimes, ranked opportunities with counterfactual upper bounds, negative-ROI `structural_blockers`. |
| `wp.compare_optimization_rounds(prev, curr)` | Harness (D3) | Round-to-round diff: eliminated / persisting / new bottlenecks, `revert_previous_change` on regression, `next_round_directive` (`STOP_SATURATED` / `PURSUE_TOP_OPPORTUNITY`). |

Atrex (`alibaba/atrex-kernel-agent`) integration: `plugin.json` declares an AKA `api_version: 1` local plugin with 6 tools (`weaveprofiler.profile`, `weaveprofiler.megakernel-audit`, `weaveprofiler.analyze`, `weaveprofiler.compare-rounds`, `weaveprofiler.plan-context`, `weaveprofiler.verify-seal` via `weaveprof/aka_adapter.py` + strict JSON schemas, plus this `SKILL.md` and episode instructions) validated directly by `aka.resources.plugin_runtime.PluginRegistry`. In addition, `weaveprof.atrex_plugin.apply(ctx, config)` exposes the Core module contract (`intra_kernel_verifier_profiler`), and GPU exclusivity supports host lock injection via `register_lock_provider`.

Recorder context modes (`recorders/cuda_cpp/wp_lite.cuh`, chosen automatically by the compiler-feedback selector): default context, single-register bit-packed mode (`-DWP_CTX_REG` / `-DWP_CTX_REG_NOINLINE`, 0 B SMEM, 32,767 records/lane cap), and static `__shared__` + `__noinline__` fallback (`-DWP_CTX_SMEM`, 512 B SMEM at warp granularity / 16 B at CTA granularity). For multi-file `.so` megakernels, `-DWP_DEFINE_HOST_API` exports `extern "C" { wp_init, wp_reset, wp_dump, wp_free }` (`WpSoController`); for runtime `.cubin` JIT kernels (`cudaLibrary_t`), `wp_attach_library(lib, &buf)` binds `wp_fi_c` (macro alias `wp_g`) across all recorder modes via `wp_active_buf()` with zero kernel argument changes. CuTeDSL (`recorders/cutedsl/wp_cute.py`), Triton (`recorders/triton/wp_triton.py`, `recorders/triton/wp_triton_amd.py`), HIP (`recorders/hip_cpp/wp_lite_hip.h`), and the `FILITE02` / `WPLITE02` decoder (`weaveprof/pipeline/validate/tools/wp_decode.py`) expose the unified `wp_*` / `WP_*` API.

## Agent Provisioning (`Skill + MCP + Hook`)

Provision any kernel optimization workspace for **Claude Code**, **OpenAI Codex CLI**, **Qwen Code**, and **Antigravity** in one command:

```bash
python3 -B run.py install-agent --workspace /path/to/kernel_dir --protect kernel.cu [--card h20|b300|mi308x]
```

- Writes `<workspace>/.mcp.json` + `<workspace>/.claude/settings.local.json`, `<workspace>/.codex/config.toml`, `<workspace>/.qwen/settings.json`, and `<workspace>/.agents/plugins/weaveprofiler/mcp_config.json` registering [mcp_server.py](weaveprof/mcp_server.py) (`weaveprof_status`, `weaveprof_megakernel_audit`, `weaveprof_profile`, `weaveprof_plan_context`, `weaveprof_analyze`, `weaveprof_compare_rounds`, `weaveprof_verify_seal`).
- Wires [guard_hook.py](weaveprof/guard_hook.py) into `<workspace>/.claude/settings.json` (5 events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `SubagentStop`), `<workspace>/.codex/hooks.json` (3 events: `PreToolUse`, `PostToolUse`, `Stop`), `<workspace>/.qwen/settings.json` (5 events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `SubagentStop`), and `<workspace>/.agents/plugins/weaveprofiler/hooks.json` (4 events: `PreToolUse`, `PostToolUse`, `PostInvocation`, `Stop`), storing `.supervisor.key`, clean kernel backups, config backups, and HMAC-signed `<session>/session.json` outside the workspace: blocks unsealed edits, config tampering, and probe pollution, auto-restores unsealed mutations or tampered hook/MCP configs, and blocks `Stop` until the final kernel matches an HMAC-verified `PASS` seal (with a 3-block safety valve).
- For non-interactive campaigns (with automatic Linux `bwrap` read-only skill mount when available), wrap execution with [agent_installer.py](weaveprof/agent_installer.py):
  ```bash
  python3 -B run.py guard-exec --workspace /path/to/kernel_dir -- codex exec "optimize the kernel"
  ```
