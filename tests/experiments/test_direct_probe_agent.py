#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline unit tests for paper ablation scripts in `experiments/` (`direct_probe_agent.py` & `verify.py`)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.scan import llm_probe_placer as PLACER  # noqa: E402


def _load(rel: str, name: str):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIRECT_AGENT = _load("experiments/direct_probe_agent.py", "test_wp_direct_agent")
VERIFY_PY = _load("experiments/verify.py", "test_wp_verify_py")


class DirectProbeAgentExperimentsTests(unittest.TestCase):
    def test_direct_probe_agent_search_replace_and_incremental_reuse(self):
        src = "void my_gemm() {\n    int tiles = 16;\n    mma_body();\n}\n"
        patch = "<<<<<<< SEARCH\n    mma_body();\n=======\n    FI_B(mma);\n    mma_body();\n    FI_E(mma);\n>>>>>>>"
        inst = DIRECT_AGENT.apply_search_replace_blocks(src, patch)
        self.assertIn("FI_B(mma);", inst)
        self.assertIn("FI_E(mma);", inst)

        # Incremental reuse across tuning edits (tiles 16 -> 32)
        curr = src.replace("int tiles = 16;", "int tiles = 32;")
        inc = DIRECT_AGENT.incremental_inherit_probes(curr, src, inst)
        self.assertIsNotNone(inc)
        self.assertIn("int tiles = 32;", inc[0])
        self.assertIn("FI_B(mma);", inc[0])
        self.assertEqual(inc[0].count("}"), curr.count("}"))

        # Multi-helper C++/HIP file with closing `}` and blank-line `replace` hunks (#84 regression)
        cpp_prev = (
            "__device__ uint64_t get_hw_timer() {\n"
            "    return (uint64_t)clock64();\n"
            "}\n"
            "__device__ void ws_mbar_try_wait_parity(uint64_t* m) {\n"
            "    if (*m) return;\n"
            "}\n"
            "__global__ __launch_bounds__(128) void my_ws_kernel(float* out, int work) {\n"
            "    const int cta = blockIdx.x;\n"
            "    \n"
            "    for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;\n"
            "}\n"
        )
        cpp_inst = (
            '#include "wp_lite.cuh"\n'
            "__device__ uint64_t get_hw_timer() {\n"
            "    return (uint64_t)clock64();\n"
            "}\n"
            "__device__ void ws_mbar_try_wait_parity(uint64_t* m) {\n"
            "    if (*m) return;\n"
            "}\n"
            "__global__ __launch_bounds__(128) void my_ws_kernel(float* out, int work) {\n"
            "    const int cta = blockIdx.x;\n"
            "    FI_KERNEL_SCOPE(0);\n"
            "    FI_B(4u, 0u);\n"
            "    for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;\n"
            "    FI_E(4u, 0u);\n"
            "}\n"
        )
        cpp_curr = cpp_prev.replace("__launch_bounds__(128)", "__launch_bounds__(128, 2)").replace("work / 4", "work / 8")
        cpp_inc = DIRECT_AGENT.incremental_inherit_probes(cpp_curr, cpp_prev, cpp_inst)
        self.assertIsNotNone(cpp_inc)
        patched_cpp = cpp_inc[0]
        self.assertIn("__launch_bounds__(128, 2)", patched_cpp)
        self.assertIn("work / 8", patched_cpp)
        self.assertIn("FI_KERNEL_SCOPE(0);", patched_cpp)
        self.assertIn("FI_E(4u, 0u);\n}", patched_cpp)
        # Top helpers must remain 100% untouched
        helper_prefix = patched_cpp.split("my_ws_kernel")[0]
        self.assertNotIn("FI_KERNEL_SCOPE", helper_prefix)
        self.assertNotIn("FI_E", helper_prefix)
        self.assertEqual(patched_cpp.count("{"), patched_cpp.count("}"))

        # Direct conflict on a `replace` hunk must return None cleanly instead of duplicating lines
        conflict_inst = cpp_prev.replace(
            "for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;",
            "for (int i = 0; i < work / 4; ++i) { FI_B(4u, i); out[cta] += 1.0f; FI_E(4u, i); }",
        )
        self.assertIsNone(DIRECT_AGENT.incremental_inherit_probes(cpp_curr, cpp_prev, conflict_inst))

    def test_verify_py_direct_source_verification(self):
        with tempfile.TemporaryDirectory() as td:
            src_f = Path(td) / "kern.cu"
            inst_f = Path(td) / "inst_kern.cu"
            src_code = "#include <cuda_runtime.h>\n__global__ void k() { int x = 1; }\n"
            inst_code = '#include <cuda_runtime.h>\n#include "wp_lite.cuh"\n__global__ void k() { FI_KERNEL_SCOPE(0); FI_B(4u, 0u); int x = 1; FI_E(4u, 0u); }\n'
            src_f.write_text(src_code, encoding="utf-8")
            inst_f.write_text(inst_code, encoding="utf-8")

            # Extract probes directly from source without any probes.json file
            extracted = VERIFY_PY.extract_probe_spec_from_inst_code(src_code, inst_code, "cuda_cpp")
            self.assertEqual(len(extracted["probes"]), 1)
            self.assertEqual(extracted["probes"][0]["semantic_id"], "compute")

            # Run verify.py main in offline mode without requiring --probes
            # On Windows without cl.exe, nvcc fails compilation gracefully and prints standard terminal error log
            rc = VERIFY_PY.main(["--source", str(src_f), "--inst-source", str(inst_f), "--card", "h20", "--offline-only"])
            self.assertIn(rc, (0, 1))

    def test_direct_probe_prompt_macro_signatures(self):
        # 1. CUDA C++
        kb_cuda = {"language": "cuda_cpp", "card": "h20", "arch": "sm_90a"}
        p_cuda = DIRECT_AGENT.build_direct_search_replace_prompt("void k(){}", kb_cuda)
        txt_cuda = p_cuda[1]["content"]
        self.assertIn("FI_KERNEL_SCOPE(role);", txt_cuda)
        self.assertNotIn("FI_KERNEL_SCOPE(slot, role);", txt_cuda)
        self.assertIn("FI_B(stage_id, aux);", txt_cuda)
        self.assertNotIn("FI_B(load);", txt_cuda)
        self.assertIn("FI_B(1u, 0u);", txt_cuda)

        # 2. HIP C++
        kb_hip = {"language": "hip_cpp", "card": "mi308x", "arch": "gfx942"}
        p_hip = DIRECT_AGENT.build_direct_search_replace_prompt("void k(){}", kb_hip)
        txt_hip = p_hip[1]["content"]
        self.assertIn("wp_lite_hip.h", txt_hip)
        self.assertIn("FI_KERNEL_SCOPE(role);", txt_hip)
        self.assertIn("FI_B(stage_id, aux);", txt_hip)

        # 3. Triton AMD (wave64)
        kb_triton_amd = {"language": "triton", "card": "mi308x", "arch": "gfx942"}
        p_triton_amd = DIRECT_AGENT.build_direct_search_replace_prompt("def k(): pass", kb_triton_amd)
        txt_triton_amd = p_triton_amd[1]["content"]
        self.assertIn("fi_b_16b", txt_triton_amd)
        self.assertIn("fi_e_16b", txt_triton_amd)
        self.assertNotIn('fi_b("load")', txt_triton_amd)

        # 4. Triton NV
        kb_triton_nv = {"language": "triton", "card": "h20", "arch": "sm_90a"}
        p_triton_nv = DIRECT_AGENT.build_direct_search_replace_prompt("def k(): pass", kb_triton_nv)
        txt_triton_nv = p_triton_nv[1]["content"]
        self.assertIn("n = fi_b(rec, ovf, pid, n, FI, stage_id, aux)", txt_triton_nv)
        self.assertNotIn('fi_b("load")', txt_triton_nv)

        # 5. CuTeDSL
        kb_cute = {"language": "cutedsl", "card": "b300", "arch": "sm_103a"}
        p_cute = DIRECT_AGENT.build_direct_search_replace_prompt("def k(): pass", kb_cute)
        txt_cute = p_cute[1]["content"]
        self.assertIn("ctx = fi_b(ctx, stage_id, aux)", txt_cute)
        self.assertNotIn('fi_b("load")', txt_cute)

    def test_extract_probe_spec_numeric_and_pred(self):
        sample_code = (
            '#include "wp_lite.cuh"\n'
            '__global__ void k() {\n'
            '    FI_KERNEL_SCOPE(1);\n'
            '    FI_B(1u, 0u);\n'
            '    load_op();\n'
            '    FI_E(1u, 0u);\n'
            '    FI_B_PRED(threadIdx.x == 0, 3u, 0u);\n'
            '    mma_op();\n'
            '    FI_E_PRED(threadIdx.x == 0, 3u, 0u);\n'
            '}\n'
        )
        extracted = VERIFY_PY.extract_probe_spec_from_inst_code("", sample_code, "cuda_cpp")
        probes = extracted["probes"]
        self.assertEqual(len(probes), 2)
        self.assertEqual(probes[0]["semantic_id"], "load")
        self.assertEqual(probes[0]["role"], "role_1")
        self.assertEqual(probes[1]["semantic_id"], "mma")

    def test_two_level_kb_static_guard_and_whole_kernel_semantic_audit(self):
        # 1. Verify Two-Level KB topology injection into prompt
        p2p_src = (
            "__device__ __forceinline__ void p2p_cp_async_16B(half* d, const half* s) {}\n"
            "__global__ void p2p_allgather_gemm(const half* peer_ptr, float* out) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    if (warp == 0) { /* p2p pull */ } else { /* wmma */ }\n"
            "}\n"
        )
        kb_ctx = PLACER.retrieve_route_kb("cuda_cpp", "h20", "sm_90a", p2p_src)
        msgs = DIRECT_AGENT.build_direct_search_replace_prompt(p2p_src, kb_ctx, ablation_mode="full_closed_loop")
        prompt_text = msgs[1]["content"]
        self.assertIn("Level-1 Composable Primitive Invariants", prompt_text)
        self.assertIn("multi_gpu_p2p_overlap", prompt_text)
        self.assertIn("WP_KERNEL_SCOPE(role);", prompt_text)

        # 2. Verify static preflight catches micro-helper probes and signature mutations
        bad_inst = (
            '#include "wp_lite.cuh"\n'
            "__device__ __forceinline__ void p2p_cp_async_16B(half* d, const half* s) { WP_B(1u, 0u); WP_E(1u, 0u); }\n"
            "__global__ void p2p_allgather_gemm(const half* peer_ptr, float* out, int broken_extra_arg) {\n"
            "    WP_KERNEL_SCOPE(0u);\n"
            "}\n"
        )
        issues = DIRECT_AGENT.static_preflight_validate(p2p_src, bad_inst, "cuda_cpp", "h20")
        joined = "\n".join(issues)
        self.assertIn("MICRO_HELPER_PROBE_VIOLATION", joined)
        self.assertIn("SIGNATURE_MUTATED", joined)

        # 3. Verify WP_* macro extraction in verify.py
        wp_code = (
            '#include "wp_lite.cuh"\n'
            "__global__ void k() {\n"
            "    WP_KERNEL_SCOPE(1u);\n"
            "    WP_B(9u, 0u);\n"
            "    comm_op();\n"
            "    WP_E(9u, 0u);\n"
            "}\n"
        )
        ext_wp = VERIFY_PY.extract_probe_spec_from_inst_code("", wp_code, "cuda_cpp")
        self.assertEqual(len(ext_wp["probes"]), 1)
        self.assertEqual(ext_wp["probes"][0]["semantic_id"], "communication")
        self.assertEqual(ext_wp["probes"][0]["role"], "role_1")

        # 4. Verify Whole-Kernel Incremental Semantic Audit catches non-local cross-warp role shift
        prev_src = (
            "__global__ void ws_kern(float* out, int n) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    bool is_dma = (warp == 0);\n"
            "    if (is_dma) {\n"
            "        dma_load();\n"
            "    } else {\n"
            "        math_compute(out);\n"
            "    }\n"
            "}\n"
        )
        prev_inst = (
            '#include "wp_lite.cuh"\n'
            "__global__ void ws_kern(float* out, int n) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);\n"
            "    bool is_dma = (warp == 0);\n"
            "    if (is_dma) {\n"
            "        WP_B(1u, 0u);\n"
            "        dma_load();\n"
            "        WP_E(1u, 0u);\n"
            "    } else {\n"
            "        WP_B(3u, 0u);\n"
            "        math_compute(out);\n"
            "        WP_E(3u, 0u);\n"
            "    }\n"
            "}\n"
        )
        # Non-local semantic edit: `is_dma = (warp < 2)` changes which warps enter `is_dma`,
        # making untouched `WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u)` semantically inconsistent!
        edited_src = prev_src.replace("bool is_dma = (warp == 0);", "bool is_dma = (warp < 2);")

        def fake_audit_llm(model, messages, **kwargs):
            # Auditor sees BOTH the diff and the untouched WP_KERNEL_SCOPE line in the full kernel
            self.assertIn("bool is_dma = (warp < 2);", messages[1]["content"])
            self.assertIn("WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);", messages[1]["content"])
            return {
                "ok": True,
                "elapsed_s": 2.1,
                "usage": {"prompt_tokens": 600, "completion_tokens": 45, "total_tokens": 645},
                "content": (
                    "<<<<<<< SEARCH\n"
                    "    WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);\n"
                    "=======\n"
                    "    WP_KERNEL_SCOPE((warp < 2) ? 1u : 0u);\n"
                    ">>>>>>>"
                ),
            }

        res_shift = DIRECT_AGENT.audit_incremental_with_whole_kernel_llm(
            edited_src, prev_src, prev_inst, kb_ctx, llm_caller=fake_audit_llm
        )
        self.assertTrue(res_shift["ok"])
        self.assertTrue(res_shift["semantic_shift_detected"])
        self.assertIn("WP_KERNEL_SCOPE((warp < 2) ? 1u : 0u);", res_shift["instrumented_source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
