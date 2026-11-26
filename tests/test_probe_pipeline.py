#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for WeaveProfiler §3.1–§3.3 Probe Placement, Fast Injection, Incremental Reuse & Closed-Loop Verification."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path


def _record_timing(name: str, val: float, thr: float, unit: str = "s") -> bool:
    strict = os.environ.get("WEAVEPROF_STRICT_TIMING") == "1"
    mode = "enforced" if strict else "not enforced"
    print(f"[timing] {name}: {val:.3f} {unit} (threshold {thr} {unit}, {mode})", file=sys.stderr)
    return strict

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLACER = _load("weaveprof/pipeline/scan/llm_probe_placer.py", "test_wp_placer")
REUSE = _load("weaveprof/pipeline/instrument/incremental_reuse.py", "test_wp_reuse")
ENGINE = _load("weaveprof/pipeline/instrument/rule_engine.py", "test_wp_engine")
VERIFIER = _load("weaveprof/pipeline/validate/closed_loop_verifier.py", "test_wp_verifier")


SAMPLE_CUDA_SUBFUNC = """#include <cuda_runtime.h>

__device__ __forceinline__ void mma_mainloop(const float* smem_a, const float* smem_b, float& acc, int k_tiles) {
    for (int kt = 0; kt < k_tiles; ++kt) {
        float a = smem_a[kt];
        float b = smem_b[kt];
        acc = fmaf(a, b, acc);
    }
}

__global__ void ws_gemm_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    __shared__ float smem_a[64];
    __shared__ float smem_b[64];
    smem_a[threadIdx.x & 63] = A[idx];
    smem_b[threadIdx.x & 63] = B[idx];
    __syncthreads();
    float acc = 0.0f;
    mma_mainloop(smem_a, smem_b, acc, 16);
    C[idx] = acc;
}
"""

SAMPLE_SPIN_CUDA = """#include <cuda_runtime.h>
__global__ void ring_kernel(volatile int* flag, float* out) {
    while (*flag == 0) {
        // spin wait
    }
    out[threadIdx.x] = 1.0f;
}
"""

SAMPLE_TRITON = """import triton
import triton.language as tl

@triton.jit
def triton_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :])
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
"""

SAMPLE_CUTEDSL = """import cutlass
import cutlass.cute as cute

@cute.kernel
def cute_mm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    for kt in range(0, 8):
        cute.copy(gA, gB)
        cute.gemm(gA, gB, gC)
    cute.copy(gB, gC)
"""


class ProbePipelineTests(unittest.TestCase):
    def test_kb_retrieval_and_readonly_probe_placement(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "ws_gemm.cu"
            src.write_text(SAMPLE_CUDA_SUBFUNC, encoding="utf-8")

            def fake_llm(model, messages, **kwargs):
                self.assertEqual(kwargs.get("reasoning_effort"), "max")
                return {
                    "ok": True,
                    "model": model,
                    "reasoning_effort": "max",
                    "elapsed_s": 1.25,
                    "usage": {"prompt_tokens": 900, "completion_tokens": 180, "reasoning_tokens": 90, "total_tokens": 1080},
                    "content": json.dumps({
                        "kernel_name": "ws_gemm_kernel",
                        "operator_family": "gemm",
                        "granularity": "sm",
                        "warp_roles": [{"role_id": 0, "name": "cta_worker", "predicate": "true"}],
                        "probes": [
                            {"probe_id": "load_0", "semantic_id": "load", "mode": "sync", "begin_line": 16, "end_line": 17},
                            {"probe_id": "sync_0", "semantic_id": "barrier", "mode": "sync", "begin_line": 18, "end_line": 18},
                            {"probe_id": "mma_0", "semantic_id": "mma", "mode": "sync", "begin_line": 4, "end_line": 8},
                            {"probe_id": "store_0", "semantic_id": "store", "mode": "sync", "begin_line": 21, "end_line": 21},
                        ],
                    }),
                }

            res = PLACER.place_probes(src, card="h20", arch="sm_90a", model="qwen4-max", llm_caller=fake_llm)
            spec = res["probe_spec"]
            self.assertEqual(spec["schema"], "weaveprofiler.probe-spec/v1")
            self.assertEqual(len(spec["probes"]), 4)
            self.assertEqual(res["telemetry"]["total_tokens"], 1080)

            # §3.2 (2a): Lower C++ with BOTH __global__ and __device__ subfunction -> ZERO signature edits!
            inj = ENGINE.inject_probes(SAMPLE_CUDA_SUBFUNC, spec)
            self.assertTrue(inj["ok"], inj["diagnostics"])
            inst_code = inj["instrumented_source"]
            self.assertIn("void mma_mainloop(const float* smem_a, const float* smem_b, float& acc, int k_tiles)", inst_code)
            self.assertIn("void ws_gemm_kernel(const float* A, const float* B, float* C, int N)", inst_code)
            self.assertIn("FI_KERNEL_SCOPE(", inst_code)
            self.assertIn("FI_DEVICE_SCOPE()", inst_code)
            self.assertIn("FI_SYNC_OUT()", inst_code)
            self.assertIn("FI_SYNC_IN()", inst_code)

    def test_spin_loop_static_detection_blocks_bad_probe(self):
        spec = PLACER.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "ring_kernel",
                "probes": [{"probe_id": "bad_wait", "semantic_id": "wait", "begin_line": 4, "end_line": 4}],
            },
            source_text=SAMPLE_SPIN_CUDA,
            source_path=None,
            route=PLACER.resolve_route("h20", "sm_90a", "cuda_cpp"),
        )
        inj = ENGINE.inject_probes(SAMPLE_SPIN_CUDA, spec)
        self.assertFalse(inj["ok"])
        codes = [d["code"] for d in inj["diagnostics"]]
        self.assertIn("SPIN_LOOP_PROBE", codes)

    def test_triton_nv_amd_and_cutedsl_zero_signature_proxies(self):
        for card, arch in (("h20", "sm_90a"), ("mi308x", "gfx942")):
            spec = PLACER.sanitize_and_bind_probe_spec(
                {
                    "kernel_name": "triton_mm_kernel",
                    "operator_family": "gemm",
                    "probes": [
                        {"probe_id": "ld", "semantic_id": "load", "begin_line": 16, "end_line": 17},
                        {"probe_id": "dot", "semantic_id": "mma", "begin_line": 18, "end_line": 18},
                        {"probe_id": "st", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                    ],
                },
                source_text=SAMPLE_TRITON,
                source_path=None,
                route=PLACER.resolve_route(card, arch, "triton"),
            )
            inj = ENGINE.inject_probes(SAMPLE_TRITON, spec)
            self.assertTrue(inj["ok"], inj["diagnostics"])
            ast.parse(inj["instrumented_source"])
            self.assertIn("def _fi_l0_triton_mm_kernel(", inj["instrumented_source"])
            self.assertIn("def _fi_inst_triton_mm_kernel(", inj["instrumented_source"])
            self.assertIn("triton_mm_kernel = _FiTritonProxy(", inj["instrumented_source"])

        cute_spec = PLACER.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "cute_mm_kernel",
                "operator_family": "gemm",
                "probes": [
                    {"probe_id": "ld", "semantic_id": "load", "begin_line": 7, "end_line": 7},
                    {"probe_id": "mma", "semantic_id": "mma", "begin_line": 8, "end_line": 8},
                    {"probe_id": "st", "semantic_id": "store", "begin_line": 9, "end_line": 9},
                ],
            },
            source_text=SAMPLE_CUTEDSL,
            source_path=None,
            route=PLACER.resolve_route("h20", "sm_90a", "cutedsl"),
        )
        cute_inj = ENGINE.inject_probes(SAMPLE_CUTEDSL, cute_spec)
        self.assertTrue(cute_inj["ok"], cute_inj["diagnostics"])
        ast.parse(cute_inj["instrumented_source"])
        self.assertIn("cute_mm_kernel = _FiCuteProxy(", cute_inj["instrumented_source"])

    def test_incremental_zero_llm_reuse_under_param_and_epilogue_edits(self):
        with tempfile.TemporaryDirectory() as td:
            v0 = Path(td) / "v0.py"
            v1 = Path(td) / "v1.py"
            spec0_path = Path(td) / "spec0.json"
            v0.write_text(SAMPLE_TRITON, encoding="utf-8")
            # Insert new lines above the loop + tune tile setup -> line numbers shift by +3, 0 stage primitives changed!
            edited = SAMPLE_TRITON.replace(
                "    pid = tl.program_id(0)\n",
                "    # Step 1 optimization: swizzled tile id\n    raw_pid = tl.program_id(0)\n    num_pid = tl.num_programs(0)\n    pid = raw_pid % num_pid\n",
            )
            v1.write_text(edited, encoding="utf-8")
            spec0 = PLACER.sanitize_and_bind_probe_spec(
                {
                    "kernel_name": "triton_mm_kernel",
                    "probes": [
                        {"probe_id": "ld", "semantic_id": "load", "begin_line": 16, "end_line": 17},
                        {"probe_id": "dot", "semantic_id": "mma", "begin_line": 18, "end_line": 18},
                        {"probe_id": "st", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                    ],
                },
                source_text=SAMPLE_TRITON,
                source_path=v0,
                route=PLACER.resolve_route("h20", "sm_90a", "triton"),
            )
            spec0_path.write_text(json.dumps(spec0), encoding="utf-8")

            inc = REUSE.reuse_probes_incremental(v0, spec0_path, v1, allow_delta_llm=False)
            self.assertTrue(inc["ok"])
            self.assertFalse(inc["used_delta_llm"])
            if _record_timing("ProbePipelineTests.incremental_reuse.elapsed_s", inc["elapsed_s"], 0.1):
                self.assertLess(inc["elapsed_s"], 0.1)
            self.assertEqual(inc["telemetry"]["total_tokens"], 0)
            new_lines = [(p["probe_id"], p["begin_line"], p["end_line"]) for p in inc["probe_spec"]["probes"]]
            self.assertEqual(new_lines, [("ld", 19, 20), ("dot", 21, 21), ("st", 22, 22)])

    def test_closed_loop_scheme_a_coarsening_convergence(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src = td_p / "inner_dense.py"
            src.write_text(SAMPLE_TRITON, encoding="utf-8")

            calls = []

            def fake_llm(model, messages, **kwargs):
                calls.append(messages)
                if len(calls) == 1:
                    # Round 1: overly dense inner-loop probes -> runtime reports 24.5% overhead (> 10%)
                    return {
                        "ok": True, "model": model, "reasoning_effort": "max", "elapsed_s": 2.0,
                        "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620},
                        "content": json.dumps({
                            "kernel_name": "triton_mm_kernel",
                            "probes": [
                                {"probe_id": "inner_ld", "semantic_id": "load", "in_inner_loop": True, "begin_line": 16, "end_line": 17},
                                {"probe_id": "inner_mma", "semantic_id": "mma", "in_inner_loop": True, "begin_line": 18, "end_line": 18},
                                {"probe_id": "ep_store", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                            ],
                        }),
                    }
                # Round 2: LLM receives OVERHEAD_EXCEEDED_COARSEN_SCHEME_A and hoists probes to outer loop boundary!
                self.assertIn("OVERHEAD_EXCEEDED_COARSEN_SCHEME_A", messages[-1]["content"])
                return {
                    "ok": True, "model": model, "reasoning_effort": "max", "elapsed_s": 1.8,
                    "usage": {"prompt_tokens": 700, "completion_tokens": 100, "total_tokens": 800},
                    "content": json.dumps({
                        "kernel_name": "triton_mm_kernel",
                        "probes": [
                            {"probe_id": "hoisted_mainloop_mma", "semantic_id": "mma", "in_inner_loop": False, "begin_line": 15, "end_line": 18},
                            {"probe_id": "ep_store", "semantic_id": "store", "begin_line": 19, "end_line": 19},
                        ],
                    }),
                }

            def fake_runtime_eval(_l0, _inst, _names, spec, round_dir):
                dump = round_dir / "cap.fi02.bin"
                has_inner_ld = any(p["probe_id"] == "inner_ld" for p in spec["probes"])
                # Write a valid FILITE02 dump matching declared stages
                meta_b = json.dumps({"arch": "sm_90a", "backend": "nvidia_ptx", "gran": 1}).encode("utf-8")
                recs = [(1000, 0x40000000, 0)]  # OPEN
                t = 1100
                for p in spec["probes"]:
                    ev = p["event_id"]
                    recs.append((t, ev, 0))
                    recs.append((t + 200, 0x80000000 | ev, 0))
                    t += 300
                recs.append((t + 100, 0x08000000, 0))  # CLOSE
                with open(dump, "wb") as f:
                    f.write(b"FILITE02")
                    f.write(struct.pack("<6I", 1, 32, 1, 0, 0, len(meta_b)))
                    f.write(meta_b)
                    f.write(struct.pack("<1I", len(recs)))
                    for rt, rtag, raux in recs:
                        f.write(struct.pack("<QII", rt, rtag, raux))
                ovh = 24.5 if has_inner_ld else 2.8
                return {"ok": True, "mismatches_total": 0, "nonfinite_total": 0, "overhead_pct": ovh,
                        "binsim": {"assembly_change_rate": 0.0},
                        "resources": {"l0": {"spill_total_bytes": 0}, "inst": {"spill_total_bytes": 0},
                                      "delta_smem_bytes": 0}}, dump

            manifest = VERIFIER.run_closed_loop_profile(
                src,
                td_p / "out_run",
                card="h20",
                arch="sm_90a",
                model="qwen4-max",
                max_rounds=3,
                runtime_evaluator=fake_runtime_eval,
                llm_caller=fake_llm,
            )
            self.assertTrue(manifest["ok"])
            self.assertEqual(manifest["rounds_executed"], 2)
            self.assertFalse(manifest["round_history"][0]["all_gates_passed"])
            self.assertTrue(manifest["round_history"][1]["all_gates_passed"])

    def test_cutedsl_class_method_descriptor_proxy(self):
        sample_oop = (
            "import cutlass\n"
            "import cutlass.cute as cute\n\n"
            "class Sm100SwapABSwigluFp4Fc12Kernel:\n"
            "    @cute.kernel\n"
            "    def fc1fc2_kernel_impl(\n"
            "        self,\n"
            "        tiled_mma: cute.TiledMma,\n"
            "        fc1_weight: cute.Tensor,\n"
            "        fc2_output: cute.Tensor,\n"
            "        token_comm_args=None,\n"
            "    ):\n"
            "        for kt in range(0, 8):\n"
            "            cute.gemm(fc1_weight, fc2_output)\n"
            "        cute.copy(fc1_weight, fc2_output)\n"
        )
        spec = PLACER.sanitize_and_bind_probe_spec(
            {
                "kernel_name": "fc1fc2_kernel_impl",
                "operator_family": "gemm",
                "probes": [
                    {"probe_id": "mma", "semantic_id": "mma", "begin_line": 14, "end_line": 14},
                    {"probe_id": "cp", "semantic_id": "store", "begin_line": 15, "end_line": 15},
                ],
            },
            source_text=sample_oop,
            source_path=None,
            route=PLACER.resolve_route("b300", "sm_103a", "cutedsl"),
        )
        inj = ENGINE.inject_probes(sample_oop, spec)
        self.assertTrue(inj["ok"], inj["diagnostics"])
        inst_code = inj["instrumented_source"]
        ast.parse(inst_code)
        self.assertIn("fc1fc2_kernel_impl = _FiCuteDescriptor(_fi_l0_fc1fc2_kernel_impl, _fi_inst_fc1fc2_kernel_impl, fi_get_attached)", inst_code)
        self.assertIn("def _fi_l0_fc1fc2_kernel_impl(", inst_code)
        self.assertIn("def _fi_inst_fc1fc2_kernel_impl(", inst_code)
        self.assertIn("_fi_rec=None, _fi_ovf=None", inst_code)

    def test_cutedsl_wp_cute_numpy2_bitwise_not_guard(self):
        # CPython 3.14 + NumPy 2.x `~nonzero` temporary elision guard in wp_cute.py (#85 regression)
        fi_cute_src = (ROOT / "recorders" / "cutedsl" / "wp_cute.py").read_text(encoding="utf-8")
        self.assertNotIn("np.any(~nonzero", fi_cute_src)
        self.assertNotIn("np.argmax(~nonzero", fi_cute_src)
        self.assertIn("is_zero = tag == np.uint32(0)", fi_cute_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
