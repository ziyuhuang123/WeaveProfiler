#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone characterization tests for `direct_probe_agent.py` (B5-G5, B5-G6)."""
from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# In-process zero-external-network socket guard (installed BEFORE loading target module)
# ---------------------------------------------------------------------------
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_GETADDRINFO = socket.getaddrinfo
_NON_LOOPBACK_ATTEMPTS: list[str] = []


def _is_loopback_host(host: object) -> bool:
    if not isinstance(host, str):
        return False
    return host.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip_literal(host: object) -> bool:
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: object) -> object:
    if isinstance(address, tuple) and len(address) >= 1 and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if isinstance(address, tuple) and len(address) >= 1 and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect_ex:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT_EX(self, address)


def _guarded_create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> socket.socket:
    host = address[0] if isinstance(address, tuple) and address else ""
    if not _is_loopback_host(host):
        _NON_LOOPBACK_ATTEMPTS.append(f"create_connection:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> list[tuple]:
    if not (_is_loopback_host(host) or _is_ip_literal(host)):
        _NON_LOOPBACK_ATTEMPTS.append(f"getaddrinfo:{host!r}:{port!r}")
        raise OSError(f"blocked by test guard: DNS lookup for {host!r}")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]
socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Single path constant for relocating this test when moving the module to `experiments/`
DPA = ROOT / "experiments/direct_probe_agent.py"


def _load_dpa_module():
    spec = importlib.util.spec_from_file_location("b5_char_direct_probe_agent", DPA)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load target module at {DPA}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DPA_MOD = _load_dpa_module()

FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

SITECUSTOMIZE_GUARD_SOURCE = """\
import ipaddress
import socket

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


def _is_loopback(h):
    return isinstance(h, str) and h.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip(h):
    if not isinstance(h, str):
        return False
    try:
        ipaddress.ip_address(h.strip())
        return True
    except ValueError:
        return False


def _g_connect(self, address):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _orig_connect(self, address)


def _g_connect_ex(self, address):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _orig_connect_ex(self, address)


def _g_create_conn(address, *args, **kwargs):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _orig_create_connection(address, *args, **kwargs)


def _g_getaddrinfo(host, port, *args, **kwargs):
    if not (_is_loopback(host) or _is_ip(host)):
        raise OSError(f"blocked by test guard: DNS lookup for {host!r}")
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.socket.connect = _g_connect
socket.socket.connect_ex = _g_connect_ex
socket.create_connection = _g_create_conn
socket.getaddrinfo = _g_getaddrinfo
"""

EXPECTED_AUDIT_KEYS = {
    "instrumented_source",
    "is_incremental",
    "ok",
    "patch",
    "semantic_shift_detected",
    "strategy",
    "telemetry",
    "used_llm_audit",
}

EXPECTED_DIRECT_INSTRUMENT_KEYS = {
    "instrumented_source",
    "is_incremental",
    "ok",
    "patch",
    "telemetry",
}


class _DpaLoopbackState:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.scripted_bodies: list[bytes] = []


class _DpaLoopbackHandler(http.server.BaseHTTPRequestHandler):
    state: _DpaLoopbackState

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        self.state.requests.append({"path": self.path, "payload": json.loads(raw)})
        body = self.state.scripted_bodies.pop(0) if self.state.scripted_bodies else b'{"error":"empty"}'
        status = 200 if body != b'{"error":"empty"}' else 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class DirectProbeAgentCharacterizationTests(unittest.TestCase):
    _ENV_SAVE_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "TMPDIR",
        "PYTHONPATH",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "WEAVEPROF_BASE_URL",
        "OPENAI_BASE_URL",
        "WEAVEPROF_API_KEY",
        "OPENAI_API_KEY",
        "WEAVEPROF_MODEL",
        "OPENAI_MODEL",
        "WEAVEPROF_SEND_REASONING_EFFORT",
        "DASHSCOPE_API_KEY",
        "BAILIAN_API_KEY",
        "FI_SAMPLE",
        "WP_SAMPLE",
        "WEAVEPROF_SAMPLE",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        _NON_LOOPBACK_ATTEMPTS.clear()
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}

        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_b5_dpa_iso_")).resolve()
        for sub in (
            "home",
            "xdg_cache",
            "xdg_config",
            "mpl",
            "sessions",
            "cache",
            "locks",
            "workdir",
            "tmp",
            "cwd",
            "netguard",
        ):
            (cls._iso_root / sub).mkdir(parents=True, exist_ok=True)

        cls._guard_dir = cls._iso_root / "netguard"
        (cls._guard_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_GUARD_SOURCE, encoding="utf-8")

        cls._http_state = _DpaLoopbackState()
        handler_cls = type("BoundDpaHandler", (_DpaLoopbackHandler,), {"state": cls._http_state})
        cls._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        cls._port = cls._server.server_address[1]
        cls._server_thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._server_thread.start()

        os.environ["HOME"] = str(cls._iso_root / "home")
        os.environ["XDG_CACHE_HOME"] = str(cls._iso_root / "xdg_cache")
        os.environ["XDG_CONFIG_HOME"] = str(cls._iso_root / "xdg_config")
        os.environ["MPLCONFIGDIR"] = str(cls._iso_root / "mpl")
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(cls._iso_root / "sessions")
        os.environ["WEAVEPROF_CACHE"] = str(cls._iso_root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._iso_root / "locks")
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._iso_root / "workdir")
        os.environ["TMPDIR"] = str(cls._iso_root / "tmp")
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["WEAVEPROF_BASE_URL"] = f"http://127.0.0.1:{cls._port}"
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{cls._port}"
        os.environ["WEAVEPROF_API_KEY"] = "test-dummy-key"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        existing_pp = [p for p in (cls._saved_env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        os.environ["PYTHONPATH"] = os.pathsep.join([str(cls._guard_dir), str(ROOT), *existing_pp])

        for unset_key in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "OPENAI_API_KEY",
            "WEAVEPROF_MODEL",
            "OPENAI_MODEL",
            "WEAVEPROF_SEND_REASONING_EFFORT",
            "DASHSCOPE_API_KEY",
            "BAILIAN_API_KEY",
            "FI_SAMPLE",
            "WP_SAMPLE",
            "WEAVEPROF_SAMPLE",
        ):
            os.environ.pop(unset_key, None)

        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")

        probe_proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import socket; socket.create_connection(('93.184.216.34', 443), timeout=1.0)",
            ],
            cwd=str(cls._iso_root / "cwd"),
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe_proc.returncode == 0 or "blocked by test guard: ('93.184.216.34', 443)" not in probe_proc.stderr:
            raise AssertionError(f"Subprocess sitecustomize guard failed: rc={probe_proc.returncode}, stderr={probe_proc.stderr!r}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()
        cls._server_thread.join(timeout=5.0)
        os.chdir(cls._saved_cwd)
        tempfile.tempdir = cls._saved_tempdir
        for k, val in cls._saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(cls._iso_root, ignore_errors=True)
        if _NON_LOOPBACK_ATTEMPTS:
            raise AssertionError(f"Unexpected non-loopback network attempts: {_NON_LOOPBACK_ATTEMPTS}")

    def _new_case_dir(self, tag: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(self._iso_root / "tmp"))).resolve()

    def test_apply_search_replace_blocks_and_incremental_inherit_probes(self) -> None:
        src = (
            "__global__ void my_gemm(float* out, int tiles) {\n"
            "    int limit = tiles * 2;\n"
            "    mma_body(out, limit);\n"
            "}\n"
        )
        # 1) Exact SEARCH/REPLACE block
        patch_exact = (
            "<<<<<<< SEARCH\n"
            "    mma_body(out, limit);\n"
            "=======\n"
            "    WP_B(3u, 0u);\n"
            "    mma_body(out, limit);\n"
            "    WP_E(3u, 0u);\n"
            ">>>>>>>"
        )
        inst = DPA_MOD.apply_search_replace_blocks(src, patch_exact)
        self.assertEqual(
            inst,
            "__global__ void my_gemm(float* out, int tiles) {\n"
            "    int limit = tiles * 2;\n"
            "    WP_B(3u, 0u);\n"
            "    mma_body(out, limit);\n"
            "    WP_E(3u, 0u);\n"
            "}\n",
        )

        # 2) Line-trimmed whitespace fallback & full-code-block fallback & short non-matching no-op
        patch_ws = (
            "<<<<<<< SEARCH\n"
            "  mma_body(out, limit);\n"
            "=======\n"
            "    FI_B(3u, 0u);\n"
            "    mma_body(out, limit);\n"
            "    FI_E(3u, 0u);\n"
            ">>>>>>>"
        )
        self.assertTrue("FI_B(3u, 0u);" in DPA_MOD.apply_search_replace_blocks(src, patch_ws))

        fenced_full = (
            "```cuda\n"
            '#include "wp_lite.cuh"\n'
            "__global__ void my_gemm(float* out, int tiles) {\n"
            "    WP_KERNEL_SCOPE(0u);\n"
            "    WP_B(3u, 0u);\n"
            "    mma_body(out, tiles);\n"
            "    WP_E(3u, 0u);\n"
            "}\n"
            "```\n"
        )
        self.assertTrue(DPA_MOD.apply_search_replace_blocks(src, fenced_full).startswith('#include "wp_lite.cuh"\n'))
        self.assertEqual(DPA_MOD.apply_search_replace_blocks(src, "no markers short"), src)

        # 3) `incremental_inherit_probes`: exact match, C++ hunk projection, conflict -> None, brace imbalance -> None
        self.assertEqual(
            DPA_MOD.incremental_inherit_probes(src, src, inst),
            (inst, "exact_match"),
        )

        cpp_prev = (
            "__device__ void helper_fn(int x) {\n"
            "    if (x) return;\n"
            "}\n"
            "__global__ __launch_bounds__(128) void ws_kern(float* out, int work) {\n"
            "    const int cta = blockIdx.x;\n"
            "    \n"
            "    for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;\n"
            "}\n"
        )
        cpp_inst = (
            '#include "wp_lite.cuh"\n'
            "__device__ void helper_fn(int x) {\n"
            "    if (x) return;\n"
            "}\n"
            "__global__ __launch_bounds__(128) void ws_kern(float* out, int work) {\n"
            "    const int cta = blockIdx.x;\n"
            "    WP_KERNEL_SCOPE(0u);\n"
            "    WP_B(4u, 0u);\n"
            "    for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;\n"
            "    WP_E(4u, 0u);\n"
            "}\n"
        )
        cpp_curr = cpp_prev.replace("__launch_bounds__(128)", "__launch_bounds__(128, 2)").replace(
            "work / 4", "work / 8"
        )
        projected = DPA_MOD.incremental_inherit_probes(cpp_curr, cpp_prev, cpp_inst)
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(projected[1], "diff_hunk_reapplied")
        self.assertTrue(projected[0].startswith('#include "wp_lite.cuh"\n'))
        self.assertTrue("__launch_bounds__(128, 2)" in projected[0])
        self.assertTrue("work / 8" in projected[0])
        self.assertTrue("WP_B(4u, 0u);" in projected[0])

        # Overlapping edit on probed line returns None
        conflict_inst = cpp_prev.replace(
            "for (int i = 0; i < work / 4; ++i) out[cta] += 1.0f;",
            "for (int i = 0; i < work / 4; ++i) { WP_B(4u, i); out[cta] += 1.0f; WP_E(4u, i); }",
        )
        self.assertIsNone(DPA_MOD.incremental_inherit_probes(cpp_curr, cpp_prev, conflict_inst))

        # Pin brace-delta conservation (`patched_brace_diff == curr_brace_diff + inst_brace_delta` at `:306-314`):
        # appending `}\n` to `prev_inst` shifts `inst_brace_delta` by -1 and `patched_brace_diff` by -1, so it still projects.
        conserved_delta_inst = cpp_inst + "}\n"
        conserved_proj = DPA_MOD.incremental_inherit_probes(cpp_curr, cpp_prev, conserved_delta_inst)
        self.assertIsNotNone(conserved_proj)
        assert conserved_proj is not None
        self.assertEqual(conserved_proj[1], "diff_hunk_reapplied")

        # Python AST validation (`:297-301`) rejects syntactically invalid projected source and returns None
        py_prev = "def k(x):\n    y = x + 1\n    return y\n"
        py_curr = "def k(x):\n    y = x + 2\n    return y\n"
        py_bad_inst = "def k(x):\n    def (\n    y = x + 1\n    return y\n"
        self.assertIsNone(DPA_MOD.incremental_inherit_probes(py_curr, py_prev, py_bad_inst))

    def test_static_preflight_validate_and_prompt_builders(self) -> None:
        base_cpp = (
            "__device__ __forceinline__ void micro_helper(float* d) { d[0] += 1.0f; }\n"
            "__global__ void p2p_allgather_gemm(const float* peer_ptr, float* out) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    if (warp == 0) { out[0] = peer_ptr[0]; } else { micro_helper(out); }\n"
            "}\n"
        )
        # 1) Identical source -> NO_PROBES_APPLIED
        self.assertEqual(
            DPA_MOD.static_preflight_validate(base_cpp, base_cpp, "cuda_cpp", "h20"),
            ["NO_PROBES_APPLIED: Instrumented code is identical to uninstrumented source; SEARCH blocks failed to match."],
        )

        # 2) C++ compound violations: brace imbalance, FiBuf redeclare, signature mutation, micro-helper, unbalanced B/E
        bad_cpp = (
            '#include "wp_lite.cuh"\n'
            "FiBuf fi_g;\n"
            "__device__ __forceinline__ void micro_helper(float* d) { WP_B(1u, 0u); d[0] += 1.0f; WP_E(1u, 0u); }\n"
            "__global__ void p2p_allgather_gemm(const float* peer_ptr, float* out, int extra_param) {\n"
            "    WP_B(3u, 0u);\n"
            "    {\n"
            "}\n"
        )
        issues_cpp = DPA_MOD.static_preflight_validate(base_cpp, bad_cpp, "cuda_cpp", "h20")
        self.assertEqual(
            [msg.split(":")[0] for msg in issues_cpp],
            [
                "CPP_BRACE_IMBALANCE",
                "GLOBAL_BUFFER_REDECLARED",
                "SIGNATURE_MUTATED",
                "MICRO_HELPER_PROBE_VIOLATION",
                "UNBALANCED_BEGIN_END_COUNT",
            ],
        )

        # 3) Python syntax error & CuTeDSL SCF scope violation
        py_orig = "@cute.kernel\ndef k(x):\n    if x:\n        cute.gemm(x, x)\n"
        self.assertEqual(
            DPA_MOD.static_preflight_validate(py_orig, "def broken(:\n", "cutedsl", "b300")[0].split(" at line")[0],
            "PYTHON_SYNTAX_ERROR",
        )
        cute_bad_scf = (
            "@cute.kernel\n"
            "def k(x):\n"
            "    ctx = wp_b(ctx, 3, 0)\n"
            "    if x:\n"
            "        ctx = wp_open(0)\n"
            "    ctx = wp_e(ctx, 3, 0)\n"
        )
        self.assertEqual(
            DPA_MOD.static_preflight_validate(py_orig, cute_bad_scf, "cutedsl", "b300"),
            ["CUTEDSL_SCF_SCOPE_VIOLATION: `ctx = wp_open(role)` must be initialized at outer function scope before `if`/`for`."],
        )

        # 4) Pin BASE bug at `direct_probe_agent.py:511-515`: `spin_lines` is a set of `(start, end, snippet)`
        # tuples while `idx_ln` is an `int`, so `idx_ln in spin_lines` is always False (`issues == []`).
        spin_orig = (
            "__global__ void spin_k(volatile int* flag, float* out) {\n"
            "    while (*flag == 0) {\n"
            "        int s = 0;\n"
            "    }\n"
            "    out[0] = 1.0f;\n"
            "}\n"
        )
        spin_inst = (
            '#include "wp_lite.cuh"\n'
            "__global__ void spin_k(volatile int* flag, float* out) {\n"
            "    WP_KERNEL_SCOPE(0u);\n"
            "    while (*flag == 0) {\n"
            "        WP_B(2u, 0u);\n"
            "        WP_E(2u, 0u);\n"
            "    }\n"
            "    out[0] = 1.0f;\n"
            "}\n"
        )
        self.assertEqual(DPA_MOD.static_preflight_validate(spin_orig, spin_inst, "cuda_cpp", "h20"), [])

        # 5) `build_direct_search_replace_prompt` across modes and languages
        kb_ctx = DPA_MOD.retrieve_route_kb("cuda_cpp", "h20", "sm_90a", base_cpp)
        bare_msgs = DPA_MOD.build_direct_search_replace_prompt(base_cpp, kb_ctx, ablation_mode="bare_direct")
        self.assertEqual([m["role"] for m in bare_msgs], ["system", "user"])
        self.assertFalse("Level-1 Composable Primitive Invariants" in bare_msgs[1]["content"])

        kb_only_msgs = DPA_MOD.build_direct_search_replace_prompt(base_cpp, kb_ctx, ablation_mode="kb_direct")
        self.assertTrue("Direct Ad-Hoc Timer Mode" in kb_only_msgs[1]["content"])
        self.assertFalse("INV-1:" in kb_only_msgs[1]["content"])

        closed_msgs = DPA_MOD.build_direct_search_replace_prompt(
            base_cpp,
            kb_ctx,
            ablation_mode="full_closed_loop",
            error_log="Overhead 22.4% > 10.0%",
        )
        self.assertTrue("CRITICAL: PREVIOUS VERIFICATION FAILED" in closed_msgs[1]["content"])
        self.assertTrue("multi_gpu_p2p_overlap" in closed_msgs[1]["content"])
        self.assertTrue("WP_KERNEL_SCOPE(role);" in closed_msgs[1]["content"])

        for route_spec, expected_marker in (
            ({"language": "hip_cpp", "card": "mi308x", "arch": "gfx942"}, "wp_lite_hip.h"),
            ({"language": "triton", "card": "mi308x", "arch": "gfx942"}, "fi_b_16b"),
            ({"language": "triton", "card": "h20", "arch": "sm_90a"}, "n = fi_b(rec, ovf, pid, n, FI, stage_id, aux)"),
            ({"language": "cutedsl", "card": "b300", "arch": "sm_103a"}, "ctx = fi_b(ctx, stage_id, aux)"),
        ):
            msgs = DPA_MOD.build_direct_search_replace_prompt("def k(): pass", route_spec)
            self.assertTrue(expected_marker in msgs[1]["content"])

    def test_audit_incremental_and_direct_instrument_self_repair_and_cli_main(self) -> None:
        case_dir = self._new_case_dir("dpa_orchestration")
        prev_src = (
            "__device__ __forceinline__ void helper_add(float* out) {\n"
            "    out[0] += 1.0f;\n"
            "}\n"
            "__global__ void ws_kern(float* out, int n) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    bool is_dma = (warp == 0);\n"
            "    if (is_dma) {\n"
            "        dma_load(out);\n"
            "    } else {\n"
            "        helper_add(out);\n"
            "    }\n"
            "}\n"
        )
        prev_inst = (
            '#include "wp_lite.cuh"\n'
            "__device__ __forceinline__ void helper_add(float* out) {\n"
            "    out[0] += 1.0f;\n"
            "}\n"
            "__global__ void ws_kern(float* out, int n) {\n"
            "    int warp = threadIdx.x >> 5;\n"
            "    WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);\n"
            "    bool is_dma = (warp == 0);\n"
            "    if (is_dma) {\n"
            "        WP_B(1u, 0u);\n"
            "        dma_load(out);\n"
            "        WP_E(1u, 0u);\n"
            "    } else {\n"
            "        WP_B(4u, 0u);\n"
            "        helper_add(out);\n"
            "        WP_E(4u, 0u);\n"
            "    }\n"
            "}\n"
        )
        kb_ctx = DPA_MOD.retrieve_route_kb("cuda_cpp", "h20", "sm_90a", prev_src)

        # 1) `audit_incremental_with_whole_kernel_llm`: exact match (0 LLM), KEEP_PROJECTED, and semantic patch
        res_exact = DPA_MOD.audit_incremental_with_whole_kernel_llm(prev_src, prev_src, prev_inst, kb_ctx)
        self.assertEqual(set(res_exact.keys()), EXPECTED_AUDIT_KEYS)
        self.assertIs(res_exact["ok"], True)
        self.assertEqual(res_exact["strategy"], "exact_source_match_zero_llm")
        self.assertIs(res_exact["used_llm_audit"], False)
        self.assertIs(res_exact["semantic_shift_detected"], False)
        self.assertEqual(res_exact["telemetry"]["total_tokens"], 0)

        edited_src = prev_src.replace("bool is_dma = (warp == 0);", "bool is_dma = (warp < 2);")

        def keep_llm(model: str, messages: list[dict], **kwargs: object) -> dict:
            return {
                "ok": True,
                "elapsed_s": 0.1,
                "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
                "content": "KEEP_PROJECTED",
            }

        res_keep = DPA_MOD.audit_incremental_with_whole_kernel_llm(
            edited_src, prev_src, prev_inst, kb_ctx, llm_caller=keep_llm
        )
        self.assertEqual(set(res_keep.keys()), EXPECTED_AUDIT_KEYS)
        self.assertIs(res_keep["ok"], True)
        self.assertEqual(res_keep["strategy"], "whole_kernel_audit_keep_projected")
        self.assertIs(res_keep["used_llm_audit"], True)
        self.assertIs(res_keep["semantic_shift_detected"], False)

        def shift_llm(model: str, messages: list[dict], **kwargs: object) -> dict:
            return {
                "ok": True,
                "elapsed_s": 0.2,
                "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
                "content": (
                    "<<<<<<< SEARCH\n"
                    "    WP_KERNEL_SCOPE((warp == 0) ? 1u : 0u);\n"
                    "=======\n"
                    "    WP_KERNEL_SCOPE((warp < 2) ? 1u : 0u);\n"
                    ">>>>>>>"
                ),
            }

        res_shift = DPA_MOD.audit_incremental_with_whole_kernel_llm(
            edited_src, prev_src, prev_inst, kb_ctx, llm_caller=shift_llm
        )
        self.assertEqual(set(res_shift.keys()), EXPECTED_AUDIT_KEYS)
        self.assertIs(res_shift["ok"], True)
        self.assertEqual(res_shift["strategy"], "whole_kernel_audit_semantic_patch")
        self.assertIs(res_shift["semantic_shift_detected"], True)
        self.assertTrue("WP_KERNEL_SCOPE((warp < 2) ? 1u : 0u);" in res_shift["instrumented_source"])

        # 2) `direct_instrument_kernel` in `full_closed_loop` with static preflight 2nd-turn self-repair
        src_p = case_dir / "ws_kern.cu"
        src_p.write_text(prev_src, encoding="utf-8")
        turn_prompts: list[str] = []

        def two_turn_repair_llm(model: str, messages: list[dict], **kwargs: object) -> dict:
            turn_prompts.append(messages[1]["content"])
            if len(turn_prompts) == 1:
                # Turn 1 intentionally puts probe inside micro-helper `helper_add` -> triggers static preflight repair
                return {
                    "ok": True,
                    "elapsed_s": 0.15,
                    "usage": {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240},
                    "content": (
                        "<<<<<<< SEARCH\n"
                        "__device__ __forceinline__ void helper_add(float* out) {\n"
                        "    out[0] += 1.0f;\n"
                        "}\n"
                        "=======\n"
                        '#include "wp_lite.cuh"\n'
                        "__device__ __forceinline__ void helper_add(float* out) {\n"
                        "    WP_B(4u, 0u);\n"
                        "    out[0] += 1.0f;\n"
                        "    WP_E(4u, 0u);\n"
                        "}\n"
                        ">>>>>>>"
                    ),
                }
            # Turn 2 moves probe out of micro-helper into `__global__ void ws_kern`
            return {
                "ok": True,
                "elapsed_s": 0.15,
                "usage": {"prompt_tokens": 220, "completion_tokens": 50, "total_tokens": 270},
                "content": (
                    "<<<<<<< SEARCH\n"
                    "__device__ __forceinline__ void helper_add(float* out) {\n"
                    "    WP_B(4u, 0u);\n"
                    "    out[0] += 1.0f;\n"
                    "    WP_E(4u, 0u);\n"
                    "}\n"
                    "=======\n"
                    "__device__ __forceinline__ void helper_add(float* out) {\n"
                    "    out[0] += 1.0f;\n"
                    "}\n"
                    ">>>>>>>\n"
                    "<<<<<<< SEARCH\n"
                    "        helper_add(out);\n"
                    "=======\n"
                    "        WP_KERNEL_SCOPE(0u);\n"
                    "        WP_B(4u, 0u);\n"
                    "        helper_add(out);\n"
                    "        WP_E(4u, 0u);\n"
                    ">>>>>>>"
                ),
            }

        res_repaired = DPA_MOD.direct_instrument_kernel(
            src_p,
            card="h20",
            ablation_mode="full_closed_loop",
            llm_caller=two_turn_repair_llm,
        )
        self.assertEqual(set(res_repaired.keys()), EXPECTED_DIRECT_INSTRUMENT_KEYS)
        self.assertEqual(len(turn_prompts), 2)
        self.assertTrue("MICRO_HELPER_PROBE_VIOLATION" in turn_prompts[1])
        self.assertIs(res_repaired["ok"], True)
        self.assertIs(res_repaired["is_incremental"], False)
        self.assertEqual(res_repaired["telemetry"]["static_preflight_issues"], [])
        self.assertTrue("/* STATIC_REPAIR */" in res_repaired["patch"])

        # 3) `main()` CLI: missing source (`rc==1`), `--no-audit-incremental` (`rc==0`, 0 tokens), and loopback LLM (`rc==0`)
        err_missing = io.StringIO()
        with contextlib.redirect_stderr(err_missing):
            rc_missing = DPA_MOD.main(["--source", str(case_dir / "nope.cu"), "--out-source", str(case_dir / "o.cu")])
        self.assertEqual(rc_missing, 1)
        self.assertTrue(err_missing.getvalue().startswith("[direct_probe_agent.py] Error: Source file not found:"))

        prev_src_f = case_dir / "prev.cu"
        prev_inst_f = case_dir / "prev_inst.cu"
        curr_src_f = case_dir / "curr.cu"
        out_inc_f = case_dir / "out_inc.cu"
        prev_src_f.write_text(prev_src, encoding="utf-8")
        prev_inst_f.write_text(prev_inst, encoding="utf-8")
        curr_src_f.write_text(edited_src, encoding="utf-8")

        out_inc_buf = io.StringIO()
        with contextlib.redirect_stdout(out_inc_buf):
            rc_inc = DPA_MOD.main(
                [
                    "--source",
                    str(curr_src_f),
                    "--out-source",
                    str(out_inc_f),
                    "--card",
                    "h20",
                    "--prev-source",
                    str(prev_src_f),
                    "--prev-inst",
                    str(prev_inst_f),
                    "--no-audit-incremental",
                ]
            )
        self.assertEqual(rc_inc, 0)
        inc_cli_doc = json.loads(out_inc_buf.getvalue())
        self.assertEqual(
            {k: inc_cli_doc[k] for k in ("ok", "is_incremental", "strategy")},
            {"ok": True, "is_incremental": True, "strategy": "diff_hunk_reapplied"},
        )
        self.assertEqual(inc_cli_doc["telemetry"]["total_tokens"], 0)
        self.assertTrue(out_inc_f.is_file())

        # Loopback HTTP server driving `DPA_MOD.main` in `open_loop` mode without patching `call_llm`
        self._http_state.scripted_bodies.append(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<<<<<<< SEARCH\n"
                                    "        dma_load(out);\n"
                                    "=======\n"
                                    "        WP_KERNEL_SCOPE(1u);\n"
                                    "        WP_B(1u, 0u);\n"
                                    "        dma_load(out);\n"
                                    "        WP_E(1u, 0u);\n"
                                    ">>>>>>>"
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 90, "completion_tokens": 35, "total_tokens": 125},
                }
            ).encode("utf-8")
        )
        out_http_f = case_dir / "out_http.cu"
        out_http_buf = io.StringIO()
        with contextlib.redirect_stdout(out_http_buf):
            rc_http = DPA_MOD.main(
                [
                    "--source",
                    str(src_p),
                    "--out-source",
                    str(out_http_f),
                    "--card",
                    "h20",
                    "--ablation-mode",
                    "open_loop",
                ]
            )
        self.assertEqual(rc_http, 0)
        http_cli_doc = json.loads(out_http_buf.getvalue())
        self.assertIs(http_cli_doc["ok"], True)
        self.assertIs(http_cli_doc["is_incremental"], False)
        self.assertEqual(http_cli_doc["telemetry"]["total_tokens"], 125)
        self.assertTrue("WP_B(1u, 0u);" in out_http_f.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
