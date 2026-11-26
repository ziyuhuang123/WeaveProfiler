# WeaveProfiler

[简体中文](README.md) | English

> **See what every SM/CU and warp/wave is doing microsecond by microsecond — an automated intra-kernel GPU profiling framework built for modern asynchronous pipelines and agent-driven Recursive Self-Improvement (RSI).**

**WeaveProfiler** aligns **automated intra-kernel software instrumentation** with **same-run hardware utilization counters** in a timeline image. It integrates directly inside AI coding agents (`Claude Code`, `Qwen Code`, `OpenAI Codex CLI`, `Antigravity`) via **`Skill + MCP + Hook`**, and also provides a standalone CLI. Across **CUDA C++, HIP C++, Triton, and CuTeDSL**, semantic stage identification is handled by an LLM, while probe injection and output post-processing are executed by deterministic code to maximize reliability and stability.

---

## 1. Why WeaveProfiler

Modern LLM kernels (MegaMoE, FlashAttention-3, FlashMLA, GDN) increasingly run as **persistent fused megakernels**: warps inside each CTA split up data loading, matrix math, and inter-GPU communication, hiding latency behind multi-stage **asynchronous hardware pipelines** (NVIDIA `cp.async` / TMA / `wgmma` / `tcgen05`, AMD `buffer_load` / `vmcnt` / `MFMA`).

Existing hardware profilers and free-form LLM code edits hit three practical roadblocks:

1. **Free-form code edits break traces without verification**: Vendor profilers (`nsys`, `rocprofv3`) treat each kernel launch as a black box. Asking an LLM to rewrite kernel source for instrumentation routinely misses warps, overflows buffers, or slows down hot loops.
2. **Async instructions get mismeasured as instant synchronous calls**: Source-level probes close right after non-blocking issue, missing lowered commit/wait points (such as Triton's `commit_group` / `wait_group` or AMD's `s_waitcnt vmcnt(N)`), making load durations look `30×–120×` shorter than reality.
3. **Heavy recorder state triggers register spills or occupancy cliffs**: Production fused kernels run right up against register and shared-memory limits. Adding 4–10 live registers causes `STL/LDL` local-memory spills, while kilobyte shared-memory buffers drop occupancy (e.g., `2 → 1 CTA/SM`).

---

## 2. Core Idea & Three Key Designs

> **Core Idea**: Only **recognizing stage boundaries** requires an LLM to read source code; probe injection, async pairing, and trace checks can all run as **deterministic code**.

```text
Protected Source (Read-Only) ──► Read-Only Stage Identification (LLM) ──► probe.json
                                                                              │
                                                                              ▼
             Out-of-Workspace Auto-Injection (C++/CuTe Source · NVIDIA PTX / AMD AMDGCN Async Pass)
                                                                              │
                                                                              ▼
                       Same-Run HW Sampling (200 kHz nsys / rocprof-sys) + Four Hard Gates
                         ├── Any gate fails ──► return offending probes in feedback.json -> coarsen & retry
                         └── All gates pass ──► render aligned 4-panel overview.png + HMAC-SHA256 seal
```

### 2.1 Closed-Loop Auto-Instrumentation & Four Hard Gates
- **Read-only stage identification + automatic injection**: The LLM only reads kernel code and outputs warp roles plus line boundaries (`probe.json`). Deterministic tooling injects probes into a scratch copy—and for Triton kernels, instruments **NVIDIA PTX and AMD AMDGCN assembly directly** without touching Python loops.
- **Four fail-closed checks**: (1) **Assembly & resources** (core instruction order preserved, zero added spill `Δspill ≤ 0 B`, zero added shared memory `Δsmem ≤ 0 B`); (2) **Numerical output** (matches unmodified baseline); (3) **Trace completeness** (every `begin` paired with `end`, zero drops, 100% physical warp/wave64 coverage); (4) **Bidirectional overhead** (steady-state median runtime within `±10%`; over-budget probes are automatically coarsened and re-verified).
- **Out-of-workspace tamper protection**: Validation runs in a separate MCP process signing `HMAC-SHA256` seals, paired with lifecycle hooks and optional `bubblewrap` read-only mounts so agents cannot relax thresholds or leave stray edits.

### 2.2 Asynchronous Pipeline Tracing (Single-Warp & Cross-Warp)
- **Single-warp FIFO counting**: Hardware async memory and matrix engines complete requests in submission order (FIFO). Counting `in++` on issue and closing completed entries on `wait(N)` gives exact in-flight depth `d = in - out` at every cycle, while the time spent blocked inside `wait` is recorded separately as **Exposed Wait** (split into `Load Wait` vs. `MMA Wait`) across both NVIDIA (`cp.async`, `wgmma`, `tcgen05`) and AMD (`vmcnt`, `v_mfma`).
- **Cross-warp producer–consumer pairing**: In warp-specialized ring buffers (such as Hopper FA3 and FlashMLA) where one producer warp loads a slot and multiple compute warps wait on it, pairing each producer issue on `(CTA, slot)` with the **first consumer warp that clears the wait** gives true transfer duration and in-flight TMA depth—without 1-to-N fan-out double-counting or slow-consumer inflation.

### 2.3 Low-Overhead Recording & Unchanged Function Signatures
- **Per-warp buffer slices + single-register state (`WP_CTX_REG`)**: Each `(CTA, warp/wave)` writes 16-byte records sequentially from `lane 0` with **zero atomics and zero cross-warp sync**. `WP_CTX_REG` packs slot ID and write cursor into **one 32-bit register and `0 B` shared memory** (initialized before `setmaxnreg`), with an optional `16 B` shared-memory fallback (`WP_CTX_SMEM`).
- **Cycle-counter calibration + out-of-band binding**: Hot loops read fast local cycle counters (`%clock64` / `s_memtime`) anchored by open/close wall clocks, fusing back-to-back stage boundaries into one clock read; multi-header C++ kernels bind buffers via module-scope device symbols without changing any `__global__` / `__device__` function signature.

### Implementation Details: Silicon-Calibrated Hardware Utilization Curves (Avoiding Static Datasheet Distortion)
Raw vendor profiler percentages can severely misrepresent hardware headroom on regulated or derived GPU SKUs. WeaveProfiler calibrates Panel (A) curves against first-principles silicon microbenchmarks (pure-compute + full-memory GEMM):
- **NVIDIA H20 (`sm_90a`) — `1/4` Die-Width Calibration (`×4.0`)**: `nsys` normalizes `Tensor Active` against the full GH100 die's Tensor pipe width, capping a 100%-saturated H20 kernel at `25%` raw; WeaveProfiler automatically detects H20 and restores `×4.0`.
- **AMD Instinct MI308X (`gfx942`, 80 CUs) — Live Dynamic-Clock Cycle Duty (Eliminating the `398.7T` Denominator Trap)**:
  - Unthrottled CDNA3 peak (`1.85 GHz`) is `606.2 TFLOPS`, while the MI308X datasheet lists `398.7 TFLOPS` FP8—**neither static figure is a valid denominator**. On MI308X silicon (`80×512 = 1 CTA/CU`, `0` spills), a pure `MFMA` microbenchmark (zero memory ops) sustains **`461.3+ TFLOPS`** (`451.0 TFLOPS` under `rocprof-sys`, **`97.6%–99.5%`** pipe duty), and a spill-free AGPR dense FP8 GEMM (`8192×8192×4096`, with HBM3 + blockwise scaling) sustains **`443.4 TFLOPS`** (`SQ_VALU_MFMA_BUSY_CYCLES = 2MNK / 1024` with `0.000%` error). This disproves any `398.7T` hardware wall: dividing by `398.7T` overflows real kernels to **`111.2% > 100%`**, whereas `376.7 TFLOPS` (`torch._scaled_mm`) genuinely reflects `376.7 / 461.3 = 81.7%` pipe duty.
  - **Unified exact formula**: Across both per-dispatch `rocprof-sys` and intra-kernel `device_counting_service` sampling, WeaveProfiler uses **`100 × ΔSQ_VALU_MFMA_BUSY_CYCLES / (80 × ΔGRBM_GUI_ACTIVE)`** (`scale = 1.0`). Because `ΔGRBM_GUI_ACTIVE / 4` counts exact elapsed shader cycles per XCC at the live dynamic clock (`~1.38–1.42 GHz` → `451.5–465.3 TFLOPS` real-time pipe ceiling), frequency throttling cancels out in hardware and utilization stays strictly bounded in `[0, 100]%` (see [amd_mi308x.md](docs/amd_mi308x.md)).
- **Full-Duplex Interconnect Split & Honest Missing-Counter Policy**: `NVLink TX / RX` and `xGMI TX / RX` are independent full-duplex links and are never summed or merged; host-bypassed GPU-initiated RDMA NIC counters are explicitly noted rather than zero-filled.

---

## 3. Reading the Four-Panel Aligned View (`overview.png`)

WeaveProfiler renders one high-resolution PNG readable directly by engineers and multimodal LLMs. Panels (A)–(C) share a locked microsecond (`µs`) axis, and Panel (D) opens automatically when stages are dense:

- **(A) Same-Run Hardware Utilization Curves**: Chip-wide compute (Tensor / Matrix Core), HBM read+write bandwidth, separate full-duplex interconnect (`NVLink TX` and `NVLink RX` plotted separately; `xGMI` on AMD), and active SM/CU %. Fails closed if real same-run hardware curves are missing.
- **(B) All-SM / All-CU Waterfall**: Every physical SM/CU on the vertical axis, showing wave steps, load imbalance, and tail stragglers.
- **(C) Single-SM / Single-CU Per-Warp Timeline**: Each warp shows an **upper main track** (compute stages + amber-hatched `Exposed Wait` + `Store`) and **lower async depth strips** (`MEM` green and `MMA` blue 5-shade heat bands + issue pins `▼` encoding live depth `d`).
- **(D) Single-CTA Micro-Zoom**: Zooms into the CTA with the highest `Exposed Wait` share (marked in dashed magenta across Panels A–C) to show prefetch ramp-up, steady-state overlap, wait bubbles, and drain steps.

![NVIDIA H20 Triton num_stages=5 PTX Auto-Async Four-Panel Aligned View](docs/images/case_h20_triton_s5_4panel.png)

> **NVIDIA H20 (`sm_90a`) `4096×4096×2048` Triton `num_stages=5` GEMM**: Zero Python K-loop edits; automatic PTX extraction of `cp.async` (**peak `d = 8`**, showing `1→8` ramp and `8→6→4→2→0` drain) and `wgmma` (**`d = 2`**) alongside `Exposed Wait` at **`+5.46%`** overhead, `max_diff = 0.0`, locked to 200 kHz `nsys`.

---

## 4. Cross-Platform Gallery & Optimization Case Studies Guided by `(In-Flight Depth d + Exposed Wait)`

### 4.1 Direct Silicon Speedups Guided by `(In-Flight Depth d + Exposed Wait)`

Jointly observing `(d, Exposed Wait %)` distinguishes **under-pipelined memory stalls** (increase depth), **split compute synchronization bottlenecks** (fuse `wgmma` waits instead of crossing SMEM cliffs), and **over-pipelined register/SMEM pressure** (trade redundant depth for larger tiles/higher occupancy):

| Platform | Workload & Shape | Before `(d, Exposed Wait)` | Diagnosis & Guided Optimization | After `(d, Exposed Wait)` | Verified Speedup (`max_diff=0`) |
| --- | --- | :---: | --- | :---: | :---: |
| **8×NVIDIA H20** (`sm_90a`) | DeepGEMM MegaMoE (Qwen3.5-397B, `EP=8, S=1024`) | `65 µs` launch gap + sync `Scatter` stall | Persistent dynamic-queue fusion + async `Scatter` offload to idle `Disp` warps | Comm overlapped with FP8 Tensor Core (`27.9% → 41.8%`) | `613.2 → 416.4 µs` (**`1.47×`**) |
| **NVIDIA H20** (`sm_90a`) | Upstream `vLLM fused_moe` (DeepSeek-V3 `M=128, E=256, H=7168, I=2048`) | `s=3` default: `steady d=2.3 (max 4)`, **`Wait = 33.6%`** | HBM latency exposed with spare SMEM headroom → deepen pipeline to `num_stages=5` | `steady d=6.3 (max 8)`, **`Wait = 5.0%`** (`6.8×` drop) | `10,242.5 → 8,724.5 µs` (**`1.17×`** vs default / **`1.29×`** vs `s=1`) |
| **NVIDIA H20** (`sm_90a`) | Upstream `vLLM FLA FlashKDA` (`chunk_fwd_h`, `T=4096`) | `s=2` default: `max d=5`, **`Wait = 27.5%`** (4 split `wgmma` waits) | Naive `s=3` (`d=10`, `133 KB` SMEM) hits `2→1 CTA/SM` cliff (`0.59×`); fuse `K=128` `RS-WGMMA` (`wgmma_waits: 4→2`) | Keeps `2 CTAs/SM` (`73.7 KB` SMEM), halves sync barriers | `286.2 → 265.3 µs` (**`1.08×`**) |
| **NVIDIA B300** (`sm_103a`) | Blackwell Gather-GEMM (`4096×4096×2048` & `8192×8192×4096`) | `s=2`: `steady d=1.9 (max 2)`, **`Wait = 37.4% / 53.4%`** | Advance to `s=3` (`s=4` `131 KB` SMEM drops occupancy to `0.72×`) | `steady d=2.5 (max 4)`, **`Wait = 4.6% / 16.6%`** (`8.1×` drop) | `102.7 → 80.9 µs` (**`1.27×`**, **`1.63×`** vs `s=1`; **`1,071.7 TFLOPS`** @ 8K) |
| **AMD MI308X** (`gfx942`) | CDNA3 Async VMEM + MFMA GEMM (`4096×4096×2688`) | `s=1` sync: `d=0` (blocking `vmcnt(0)`) | 4-stage `global_load_dwordx4` + `s_waitcnt vmcnt(8)` register–LDS pipeline | `steady d=2.1 (max 3)`, **`Wait = 1.7%`** | `1,572.9 → 810.0 µs` (**`1.94×`**, `111.3 TFLOPS`) |
| **2×8 AMD MI308X** (`gfx942`) | 2-Node EP16 MegaMoE (Qwen3.8-Max, `S=16384, E=512, K=10`) | Split opening dispatch + serial `GEMM1→GEMM2` + exposed combine | Fused opening dispatch + cross-node Top-K dedup/pre-reduce + 3-group `GEMM1↔GEMM2` interleave + `1-CTA/CU (80×512)` partial-wave `4+4` split | Full-duplex `xGMI RX/TX` overlapped with `MFMA` (**`62.2%`** chip / **`73.7%`** compute pool, `280.6T`) | `17,204.0 → 6,981.1 µs` (**`2.46×`** vs RCCL+AITER) |
| **RTX 3050** (`sm_86`) | Ampere CUDA C++ `cp.async` + WMMA GEMM (`2048×2048×1280`) | `6-stage (64×128)`: `steady d=4.0 (max 5)`, **`Wait = 19.8%`** | Redundant depth on small tile → retarget SMEM to `3-stage (128×128)` for 2× data reuse | `steady d=1.1 (max 2)`, **`Wait = 16.6%`**, MMA share `79.7%` | `1,170.2 → 1,005.6 µs` (**`1.16×`**, `88.2%` cuBLAS) |

### 4.2 8×H20 Production DeepGEMM MegaMoE (`1.47×` Speedup)

Aligned timelines on an 8×H20 NVLink node running **Qwen3.5-397B DeepGEMM MegaMoE** (`EP=8, S=1024, E=512`) exposed a `~65 µs` inter-kernel gap and synchronous `Scatter` stalls while `Disp` warps idled; fusing into a persistent dynamic-queue megakernel with async `Scatter` offload yielded **`1.47×`** speedup (see [h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)):

![8×H20 DeepGEMM MegaMoE Before/After Aligned SW+HW View](docs/images/case_h20_megamoe_1p47x.png)

### 4.3 AMD Instinct MI308X (`gfx942`, 80 CUs) 4-Stage Async VMEM (`vmcnt` FIFO, `d=3`) + MFMA Four-Panel View

On AMD CDNA3 GPUs, vector memory loads (`global_load_dwordx4` / `buffer_load_dwordx4`) execute asynchronously and retire in FIFO order via `s_waitcnt vmcnt(N)`, overlapping directly with `v_mfma_*` Matrix Core pipelines. WeaveProfiler supports both **HIP C++ async macros** (`WP_VMEM_COMMIT` / `WP_VMEM_WAIT`) and **ROCm Triton AMDGCN auto-async instrumentation** (`triton_amdgcn_async.py`). Below is a live 80-CU MI308X capture at **`+5.23%`** overhead (`810.4 → 852.8 µs`), **`max_diff = 0.0`**, `0 B` spill, and **`0.37%`** `rocprof-sys` SW–HW delta, showing all 4 `wave64` tracks' `vmcnt` issue pins `▼`, in-flight depth (`steady d = 2.1, max 3`), `Exposed Wait (1.8%)`, and `3→2→1→0` tail drain:

![AMD Instinct MI308X gfx942 4-Stage Async VMEM vmcnt FIFO + MFMA Four-Panel View](docs/images/case_mi308x_async_4panel.png)

### 4.4 Zero-Signature Four-Panel Captures on Open-Source H20 Kernels

Measured on unmodified open-source repositories at long-sequence production shapes (`max_diff = 0.0`, `0` overflow, 100% physical warp coverage):

- **FlashInfer 0.6.8 Hopper FA3** (3 headers, CUDA C++, `B=5, H=16, S=4096, D=128`, 12/12 warps): **`+0.15%`** overhead (`2,618 → 2,622 µs`), `0.08%` SW–HW delta, deduplicated 1-to-8 cross-warp TMA depth `max_d_tma = 4` (2 K + 2 V) and dual-row `Softmax` / async `WGMMA (d=1/2)` overlap.
- **FlashInfer 0.6.8 Hopper FlashMLA** (`mla_hopper.cuh`, `B=64, H=128, KV=2048`, 8/8 warps): **`+9.96%`** overhead, `0 B` spill, `0.49%` SW–HW delta.
- **vLLM FLA FlashKDA** (`chunk_delta_h.py`, Triton 3.6 PTX Auto-Async, `B=1, T=4096, H=64, D=128`, 4/4 warps): **`+8.14%`** overhead, `0 B` spill, `0.39%` SW–HW delta, zero Python K-loop edits.

![FlashInfer 0.6.8 Hopper FA3 H20 S=4096 Four-Panel Aligned View](docs/images/case_h20_flashinfer_fa3.png)
![FlashInfer 0.6.8 Hopper FlashMLA H20 KV=2048 Four-Panel Aligned View](docs/images/case_h20_flashmla.png)
![vLLM FLA FlashKDA H20 T=4096 Triton PTX Auto-Async Four-Panel Aligned View](docs/images/case_h20_flashkda.png)

### 4.5 Desktop Ampere (RTX 3050 `sm_86`) 6-Stage `cp.async` & Windows WDDM Preemption Forensics

For local Ampere desktop/laptop development:
- **6-Stage `cp.async` + WMMA GEMM (`2048×2048×1280`)**: **`+0.70%`** steady-state overhead (`1,178.6 → 1,186.8 µs`, `0 B` spill, `max_diff = 0.0`) under Scheme A head/tail retention + interior coarsening, preserving the full `0→1→2→3→4→5` ramp and `5→4→3→2→1→0` drain in Panel (D).
- **Windows WDDM Preemption Forensics (`8192³` Triton PTX Auto-Async GEMM, see [sm86_wddm_preemption.md](docs/case_studies/sm86_wddm_preemption.md))**: Captures both the `0–25 ms` cold-start GDDR6 ramp and `t ≈ 30–33 ms` OS graphics preemption cliffs.

![RTX 3050 sm_86 CUDA C++ 6-Stage cp.async Low-Overhead Four-Panel View](docs/images/case_sm86_cuda_s6_4panel.png)
![RTX 3050 sm_86 8192³ Triton GEMM Cold-Start DRAM Ramp and WDDM Preemption](docs/images/case_sm86_wddm_preemption.png)

### 4.6 2-Node 16×AMD Instinct MI308X (EP16) Inter-Node RDMA + Full-Duplex xGMI MegaMoE (`2.46×` Speedup)

Captured on a 2-node 16×MI308X cluster (`2×8 GPUs`, full-duplex intra-node xGMI + inter-node RDMA) running a persistent **MegaMoE (Qwen3.8-Max, `EP=16, S=16384, E=512, K=10`)**:
- **Panel (A) Dynamic-Clock Pipe-Duty Calibration & Split xGMI**: On MI308X silicon, dense FP8 GEMM sustains `443.4 TFLOPS` (`96.1%` of the `461.3 TFLOPS` dynamic-clock ceiling at `~1.408 GHz`, which would overflow to `111.2%` under the nominal `398.7T` datasheet figure; see [amd_mi308x.md](docs/amd_mi308x.md)). Using exact cycle-domain pipe duty `100 × ΔSQ_VALU_MFMA_BUSY_CYCLES / (80 × ΔGRBM_GUI_ACTIVE)`, Matrix Core utilization reaches **`62.2%` chip-wide (`280.6 TFLOPS` @ `1.385 GHz`)** and **`73.7%`** across the `540/640` compute-wave pool. `xGMI RX (In)` and `xGMI TX (Out)` are plotted separately, with an explicit note where host NIC RDMA counters are unavailable (never zero-filled).
- **Panels (B)–(D) `1-CTA/CU` Overlapped Pipeline**: Running `1 CTA/CU` (`80 CTAs × 512 threads`) with 4 compute waves and 4 comm waves per CU, fused opening dispatch, cross-node Top-K deduplication (`+15.1%`), and 3-group `GEMM1 ↔ GEMM2` interleaving overlap inter-node RDMA and xGMI reduction underneath full Matrix Core execution, cutting latency from `17,204.0 µs` (`RCCL + AITER`) to **`6,981.1 µs` (`2.46×` speedup, 16-rank FP64 verified)**:

![2×8 AMD MI308X EP16 Qwen3.8-Max S=16384 Inter-Node RDMA and Full-Duplex xGMI Four-Panel View](docs/images/case_mi308x_ep16_megamoe_4panel.png)

---

## 5. Quick Start & Support Matrix

| Dimension | Coverage |
| --- | --- |
| **GPUs** | NVIDIA Hopper H20 (`sm_90`/`sm_90a`), Blackwell B300 (`sm_100a`/`sm_103a`), Ampere (`sm_86`), AMD Instinct MI308X (`gfx942`) |
| **Frontends** | CUDA C++, HIP C++, Triton (NVIDIA PTX Dual-Async & AMD AMDGCN Async), CuTeDSL |
| **Integration** | AI Coding Agents (`Claude Code`, `Qwen Code`, `OpenAI Codex CLI`, `Antigravity` via `Skill + MCP + Hook`), or CLI / Python SDK |
| **Prerequisites** | Python 3.10+, `nvcc` + `nsys` (NVIDIA) or `hipcc` + `rocprof-sys` (AMD), `matplotlib` |

### AI Coding Agents (`Skill + MCP + Hook`)

Install into any kernel repository in two commands (`--protect` locks production files until the initial baseline seal passes):

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20   # h20 | b300 | mi308x | sm86
```

### Standalone CLI & Python SDK

```bash
python3 -B WeaveProfiler/run.py identify kernel.cu -o probe.json   --card h20
python3 -B WeaveProfiler/run.py profile  kernel.cu -o overview.png --card h20 -- python3 bench.py
python3 -B WeaveProfiler/run.py doctor
```

```python
import weaveprof as wp

spec = wp.identify("kernel.cu", out="probe.json", card="h20")
with wp.profile("kernel.cu", out="overview.png", card="h20"):
    run_kernel()
```

### Guarded Batch Execution & Offline Tests

```bash
python3 -B WeaveProfiler/run.py guard-exec --workspace /path/to/your_kernel_repo \
  -- codex exec "Profile exposed waits and cross-SM load skew in this kernel"

# Offline structural & unit tests (no GPU required)
python3 -B WeaveProfiler/tests/package_structure.py
python3 -B -m unittest \
  WeaveProfiler/tests/test_wp_megakernel.py \
  WeaveProfiler/tests/test_mcp_hook_installer.py \
  WeaveProfiler/tests/test_triton_amdgcn_async.py
```

**Core MCP Tools & Known Bounds**: Primary MCP tools are `weaveprof_megakernel_audit` (isolated injection + 4 checks + 4-panel PNG + HMAC seal), `weaveprof_status`, `weaveprof_analyze`, `weaveprof_compare_rounds`, and `weaveprof_verify_seal`. Known bounds: CUDA `WP_CTX_REG` caps per-lane capacity at `32,767` records; AMD `rocprof-sys` PM counters are per-dispatch aggregates. See [SKILL.md](SKILL.md) for full recorder macros, compiler flags, and platform details.
