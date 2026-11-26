#!/usr/bin/env python3
"""WeaveProfiler (`wp`) Model Context Protocol stdio server (zero external dependencies).

Speaks JSON-RPC 2.0 over stdio with automatic framing detection:
  * newline-delimited JSON messages (Claude Code / Codex CLI stdio transport), or
  * `Content-Length: <N>\\r\\n\\r\\n` LSP-style framing,
and always replies in the framing used by the client.

Launch (wired by `agent_installer.py` into `.mcp.json` / `.codex/config.toml`):
    python -B <skill_root>/weaveprof/mcp_server.py --workspace <ws> --session-dir <session_dir>

The session `.supervisor.key` is scoped to HMAC sealing and verification inside this server
process and stripped from all build/benchmark child subprocesses, so seals produced by the
profiling tools carry the session HMAC and every verified seal is registered in
`<session_dir>/session.json` for the enforcement hook.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import contextlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
SKILL_ROOT = _HERE.parent
try:
    if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == _HERE:
        sys.path.pop(0)
except OSError:
    pass
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from weaveprof.guard_hook import (  # noqa: E402
    KEY_FILE,
    read_session,
    register_seal,
    session_dir_for,
    sessions_root,
    status_report,
    supervisor_env,
)

SERVER_NAME = "weaveprofiler"
SERVER_VERSION = "1.0.0"
SUPPORTED_PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_PROTOCOL = SUPPORTED_PROTOCOLS[-1]

INSTRUCTIONS = (
    "WeaveProfiler (wp) supervised GPU-kernel optimization. 1) weaveprof_status. 2) Seal the UNMODIFIED baseline with "
    "weaveprof_megakernel_audit (fast isolated-copy audit) or weaveprof_profile: protected kernel edits are hook-blocked "
    "until an HMAC-verified PASS seal exists. 3) Optimize using weaveprof_plan_context / weaveprof_analyze / "
    "weaveprof_compare_rounds evidence; never add WP_* probes to production files. 4) Re-seal the final kernel "
    "before stopping (Stop gate)."
)

_PATH_KEYS = ("source", "source_path", "seal_path", "trace_json_path", "prev_report_path", "curr_report_path",
              "prev_causal_report_path", "run_dir", "out")


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@contextlib.contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    try:
        os.chdir(path)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.chdir(prev)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_S = {"type": "string", "minLength": 1}
TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "name": "weaveprof_status",
        "description": ("Session status: protected kernel files, live SHA-256 vs baseline and latest PASS-sealed "
                        "SHA-256, whether Edit and Stop gates are unlocked, stop blockers, incumbent best kernel_us."),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"title": "wp status", "readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "weaveprof_megakernel_audit",
        "description": ("Fast (~3-12s) isolated-copy megakernel profiler (with cached baseline builds) + 4-gate verifier "
                        "for single- or multi-file CUDA C++ / CuTeDSL / Triton megakernels (FlashAttention, FlashMLA, "
                        "FlashKDA, GDN, MoE). Never mutates the real workspace. Auto-renders a 4-panel SW+HW timeline "
                        "PNG (`<model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png`, no .perfetto.json.gz needed by default) "
                        "alongside a compact <2 KB causal JSON report and an HMAC seal.json registered in the session "
                        "(unlocks edits / Stop gate)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace": {**_S, "description": "Workspace root (defaults to the server workspace)."},
                "entry_source": {**_S, "description": "Primary protected kernel file (defaults to first protected file)."},
                "bench_cmd": {"oneOf": [_S, {"type": "array", "items": _S}], "description": "Benchmark command (locked to session bench_cmd when configured at install time)."},
                "build_cmd": {"oneOf": [_S, {"type": "array", "items": _S}], "description": "Optional parallel baseline/instrumented build command."},
                "inst_files": {"type": "object", "description": "Map of relative path -> instrumented temp-copy source text."},
                "inst_dir": _S,
                "nsys_sqlite": _S,
                "capture_nsys": {"type": "boolean"},
                "nsys_freq_hz": {"type": "integer", "minimum": 1000, "maximum": 500000, "description": "nsys --gpu-metrics-frequency in Hz (default: 200000 = 200 kHz high-frequency sampling)."},
                "nsys_gpu_device": _S,
                "nsys_capture_range": _S,
                "render_overview_png": {"type": "boolean", "description": "Auto-render 4-panel SW+HW timeline PNG locally (default: true)."},
                "write_perfetto_gz": {"type": "boolean", "description": "Opt-in: also write trace.perfetto.json.gz (default: false)."},
                "overview_title": _S,
                "overview_subtitle": _S,
                "overview_sm": {"type": "integer", "minimum": 0},
                "model": {**_S, "description": "Model name token for output PNG filename."},
                "op": {**_S, "description": "Operator name token (e.g. gemm, attn, moe) for output PNG filename."},
                "bs": {"type": "integer", "minimum": 1, "description": "Batch size for output PNG filename."},
                "seql": {"type": "integer", "minimum": 1, "description": "Sequence length for output PNG filename."},
                "kv": {"type": "integer", "minimum": 1, "description": "KV length for output PNG filename."},
                "shape": {**_S, "description": "Explicit shape token for output PNG filename."},
                "card": _S,
                "arch": _S,
                "total_sms": {"type": "integer", "minimum": 1},
                "limit": {"type": "number", "minimum": 0, "maximum": 0.5},
                "out_dir": _S,
                "gpu_id": _S,
                "use_l0_cache": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "annotations": {"title": "wp megakernel audit", "readOnlyHint": False, "openWorldHint": False},
    },
    {
        "name": "weaveprof_profile",
        "description": ("Closed-loop profile_for_agent on a kernel (isolated-copy instrumentation, ABBA overhead gate, "
                        "fidelity gates) returning a compact seal-verified report; the seal is registered in the "
                        "session."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": _S,
                "bench": {"type": "array", "items": _S},
                "card": _S,
                "model": _S,
                "api_key": {**_S, "description": "Optional LLM API key."},
                "base_url": {**_S, "description": "Optional chat completions base URL."},
                "limit": {"type": "number", "minimum": 0, "maximum": 0.5},
                "max_rounds": {"type": "integer", "minimum": 1, "maximum": 10},
                "out": _S,
                "run_dir": _S,
                "workload_model": _S,
                "op": _S,
                "bs": {"type": "integer", "minimum": 1},
                "seql": {"type": "integer", "minimum": 1},
                "kv": {"type": "integer", "minimum": 1},
                "shape": _S,
                "enforce_trace_processor": {"type": "boolean"},
                "prev_causal_report": {"type": "object"},
                "prev_causal_report_path": _S,
            },
            "required": ["source"],
            "additionalProperties": False,
        },
        "annotations": {"title": "wp profile", "readOnlyHint": False, "openWorldHint": False},
    },
    {
        "name": "weaveprof_plan_context",
        "description": ("Plan low-overhead recorder state placement (single-register default vs 16B shared-memory "
                        "+ __noinline__ fallback) and compiler flags for a kernel source."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_text": _S, "source_path": _S, "arch": _S, "gpu_name": _S,
                "block_threads": {"type": "integer", "minimum": 1, "maximum": 2048},
                "l0_regs": {"type": "integer", "minimum": 0, "maximum": 255},
                "l0_smem_bytes": {"type": "integer", "minimum": 0},
                "smem_limit_bytes": {"type": "integer", "minimum": 0},
                "max_warps_per_cta": {"type": "integer", "minimum": 1, "maximum": 64},
            },
            "additionalProperties": False,
        },
        "annotations": {"title": "wp plan context", "readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "weaveprof_analyze",
        "description": ("D3 SW/HW causality: calibrated roofline, cross-SM straggler skew, negative-ROI blockers and "
                        "ranked causal opportunities from an aligned trace."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "object"}, "trace_json_path": _S, "arch": _S, "gpu_name": _S,
                "total_sms": {"type": "integer", "minimum": 1}, "workload": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "annotations": {"title": "wp analyze", "readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "weaveprof_compare_rounds",
        "description": ("Round-to-round diff: eliminated/persisting bottlenecks, revert_previous_change, "
                        "next_round_directive."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prev_report": {"type": "object"}, "prev_report_path": _S,
                "curr_report": {"type": "object"}, "curr_report_path": _S,
                "noise_floor_pct": {"type": "number", "minimum": 0, "maximum": 50},
            },
            "additionalProperties": False,
        },
        "annotations": {"title": "wp compare rounds", "readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "weaveprof_verify_seal",
        "description": ("Fail-closed verification of a seal.json with the session supervisor HMAC key; verified "
                        "seals are registered in the session."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "seal_path": _S, "expected_source_sha": _S, "require_pass": {"type": "boolean"},
                "max_overhead_limit": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["seal_path"],
            "additionalProperties": False,
        },
        "annotations": {"title": "wp verify seal", "readOnlyHint": False, "openWorldHint": False},
    },
]


class WeaveProfServer:
    def __init__(self, workspace: Path, session_dir: Path, out_stream: Any, in_stream: Any):
        self.workspace = workspace
        self.session_dir = session_dir
        self.out = out_stream
        self.inp = in_stream
        self.mode: Optional[str] = None  # "newline" | "header"
        self.protocol = DEFAULT_PROTOCOL
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "weaveprof_status": self._tool_status,
            "weaveprof_megakernel_audit": self._tool_megakernel,
            "weaveprof_profile": self._tool_profile,
            "weaveprof_plan_context": self._tool_plan_context,
            "weaveprof_analyze": self._tool_analyze,
            "weaveprof_compare_rounds": self._tool_compare,
            "weaveprof_verify_seal": self._tool_verify_seal,
        }

    # -- session helpers ---------------------------------------------------
    def _session(self) -> Optional[Dict[str, Any]]:
        return read_session(self.session_dir)

    def _resolve_paths(self, args: Dict[str, Any], keys: Sequence[str] = _PATH_KEYS) -> None:
        for k in keys:
            v = args.get(k)
            if isinstance(v, str) and v.strip() and "\n" not in v and not v.lstrip().startswith("{"):
                p = Path(os.path.expanduser(v))
                if not p.is_absolute():
                    args[k] = str((self.workspace / p).resolve())

    def _validate_out_path(self, raw_path: Any, label: str) -> None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        p = Path(os.path.expanduser(raw_path)).resolve()
        np = os.path.normcase(str(p)).replace("\\", "/")
        home = os.path.normcase(str(Path.home().resolve())).replace("\\", "/").rstrip("/")
        ws = os.path.normcase(str(self.workspace.resolve())).replace("\\", "/").rstrip("/")
        roots = [
            os.path.normcase(str(SKILL_ROOT.resolve())).replace("\\", "/").rstrip("/"),
            os.path.normcase(str(self.session_dir.resolve())).replace("\\", "/").rstrip("/"),
            os.path.normcase(str(self.session_dir.parent.resolve())).replace("\\", "/").rstrip("/"),
            ws + "/.claude", ws + "/.codex", ws + "/.qwen",
            ws + "/.agents/hooks.json", ws + "/.agents/plugins/weaveprofiler",
            ws + "/.agents/skills/weaveprofiler", ws + "/.mcp.json",
            home + "/.claude", home + "/.codex", home + "/.qwen",
        ]
        for r in roots:
            if np == r or np.startswith(r + "/"):
                raise ValueError(f"{label}={raw_path!r} targets protected supervisor/agent configuration")
        if np == ws or np.startswith(ws + "/"):
            raise ValueError(f"{label}={raw_path!r} must be outside the workspace to avoid mutating real project files")
        name = p.name.lower()
        if name.endswith("seal.json") or name in ("verify.json", "episode_ledger.json", ".supervisor.key", "session.json"):
            raise ValueError(f"{label}={raw_path!r} targets protected evidence/session file")

    def _validate_cmd(self, cmd: Any, label: str) -> None:
        if not cmd:
            return
        text = " ".join(str(x) for x in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        low = text.replace("\\", "/").lower()
        from weaveprof.guard_hook import SECRET_NEEDLES, _CONFIG_WRITE_RE, _EVIDENCE_WRITE_RE, _INSTALLER_RE, _SAMPLING_RE
        forbidden = (
            *SECRET_NEEDLES,
            str(self.session_dir).replace("\\", "/").lower(),
            str(self.session_dir.parent).replace("\\", "/").lower(),
        )
        for needle in forbidden:
            if needle and needle in low:
                raise ValueError(f"{label} references protected supervisor state ({needle!r})")
        if _SAMPLING_RE.search(text):
            raise ValueError(f"{label} sets WP_SAMPLE/FI_SAMPLE > 1 (forbidden)")
        if _INSTALLER_RE.search(text):
            raise ValueError(f"{label} attempts to re-run the wp installer/guard-exec")
        if _EVIDENCE_WRITE_RE.search(text):
            raise ValueError(f"{label} attempts to modify sealed evidence")
        if _CONFIG_WRITE_RE.search(text):
            raise ValueError(f"{label} attempts to modify agent hook/MCP configuration")

    def _enforce_locked_bench(self, raw_cmd: Any, sess: Dict[str, Any], label: str) -> Optional[List[str]]:
        locked = sess.get("bench_cmd")
        if isinstance(raw_cmd, str):
            req = shlex.split(raw_cmd, posix=(os.name != "nt"))
        elif isinstance(raw_cmd, (list, tuple)):
            req = [str(x) for x in raw_cmd]
        else:
            req = None
        if locked:
            locked_list = [str(x) for x in locked]
            if req is not None and req != locked_list:
                raise ValueError(f"{label} is locked by the session installed at provisioning time ({locked_list!r})")
            return locked_list
        return req

    def _limit(self, args: Dict[str, Any]) -> None:
        sess = self._session() or {}
        ceiling = sess.get("max_overhead")
        if ceiling is None:
            return
        if "limit" in args and args["limit"] is not None:
            if float(args["limit"]) > float(ceiling) + 1e-12:
                raise ValueError(f"limit={args['limit']} exceeds the session overhead ceiling {ceiling} "
                                 "(locked at install time)")
        else:
            args["limit"] = float(ceiling)

    def _runs_dir(self) -> Path:
        if os.name != "nt":
            uid = os.getuid() if hasattr(os, "getuid") else 0
            base = Path(tempfile.gettempdir()) / f"weaveprof-cache-{uid}" / "agent_runs" / self.session_dir.name
        else:
            base = self.session_dir.parent.parent / "agent_runs" / self.session_dir.name
        d = base / time.strftime("%Y%m%d-%H%M%S") / os.urandom(3).hex()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _register(self, res: Any, tool: str) -> Any:
        if not isinstance(res, dict):
            return res
        seal_path = res.get("seal_path")
        if not seal_path or self._session() is None:
            return res
        kernel_us = res.get("kernel_us")
        if kernel_us is None and isinstance(res.get("metrics"), dict):
            kernel_us = res["metrics"].get("kernel_us")
        reg = register_seal(self.session_dir, seal_path, kernel_us=kernel_us, tool=tool)
        st = status_report(self.session_dir)
        res["weaveprof_session"] = {
            "registered": reg.get("registered"),
            "status": reg.get("status"),
            "covers_protected": reg.get("covers_protected"),
            "note": reg.get("note"),
            "errors": reg.get("errors"),
            "edit_unlocked": st.get("edit_unlocked"),
            "stop_unlocked": st.get("stop_unlocked"),
            "incumbent": st.get("incumbent"),
        }
        return res

    # -- tools -------------------------------------------------------------
    def _tool_status(self, args: Dict[str, Any]) -> Any:
        rep = status_report(self.session_dir)
        rep.setdefault("workspace", str(self.workspace))
        return rep

    def _tool_megakernel(self, args: Dict[str, Any]) -> Any:
        try:
            from weaveprof.megakernel import run_megakernel_audit
        except ImportError as exc:
            raise RuntimeError(f"weaveprof.megakernel is unavailable: {exc}") from exc
        from weaveprof.harness import HarnessPolicy

        args = dict(args)
        for forbidden_key in ("dump_path", "runtime_metrics", "names", "nsys_cmd", "policy"):
            if forbidden_key in args:
                raise ValueError(
                    f"`{forbidden_key}` is not allowed on MCP weaveprof_megakernel_audit "
                    "(MCP audits always execute the isolated benchmark workflow)."
                )
        sess_pre = self._session() or {}
        if sess_pre.get("png_only", True) and args.get("write_perfetto_gz"):
            raise ValueError(
                "`write_perfetto_gz=True` is blocked by the session PNG-only policy; "
                "deliver the timeline PNG instead (or re-install with `--allow-gz`)."
            )
        args.setdefault("workspace", str(self.workspace))
        sess = self._session() or {}
        if "bench" in args and "bench_cmd" not in args:
            args["bench_cmd"] = args.pop("bench")
        if "sources" in args and "entry_source" not in args:
            srcs = args.pop("sources")
            if isinstance(srcs, list) and srcs:
                args["entry_source"] = srcs[0]
        if not args.get("entry_source"):
            prot = sess.get("protected") or []
            if prot and isinstance(prot[0], dict) and prot[0].get("path"):
                args["entry_source"] = prot[0]["path"]
        self._resolve_paths(args, ("workspace", "entry_source", "inst_dir", "nsys_sqlite", "out_dir"))
        self._validate_out_path(args.get("out_dir"), "out_dir")
        locked_bench = self._enforce_locked_bench(args.get("bench_cmd"), sess, "bench_cmd")
        if not locked_bench:
            raise ValueError("`bench_cmd` is required (no default bench command recorded at install time)")
        args["bench_cmd"] = locked_bench
        self._validate_cmd(args.get("bench_cmd"), "bench_cmd")
        self._validate_cmd(args.get("build_cmd"), "build_cmd")
        try:
            params = inspect.signature(run_megakernel_audit).parameters
            accepts_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            params, accepts_kw = {}, True
        for key, val in (("card", sess.get("card")), ("arch", sess.get("arch"))):
            if key not in args and val not in (None, "", []) and (key in params or accepts_kw):
                args[key] = val
        if not accepts_kw:
            args = {k: v for k, v in args.items() if k in params}
        if "limit" in params or accepts_kw or "limit" in args:
            self._limit(args)
        ceiling = float(sess.get("max_overhead")) if sess.get("max_overhead") is not None else float(args.get("limit", 0.10))
        if "policy" in params or accepts_kw:
            args["policy"] = HarnessPolicy(max_overhead_limit=ceiling)
        ctx = supervisor_env(self.session_dir) if (self.session_dir / KEY_FILE).is_file() else contextlib.nullcontext()
        with ctx, _chdir(self.workspace):
            res = run_megakernel_audit(**args)
        if not isinstance(res, dict):
            res = {"result": res}
        return self._register(res, "weaveprof_megakernel_audit")

    def _tool_profile(self, args: Dict[str, Any]) -> Any:
        from weaveprof.aka_adapter import _handle_profile

        args = dict(args)
        args.pop("require_abba_order", None)
        self._resolve_paths(args, ("source", "run_dir", "out", "prev_causal_report_path"))
        self._validate_out_path(args.get("out"), "out")
        self._validate_out_path(args.get("run_dir"), "run_dir")
        sess = self._session() or {}
        locked_bench = self._enforce_locked_bench(args.get("bench"), sess, "bench")
        if not locked_bench:
            raise ValueError("`bench` is required (no default bench command recorded at install time)")
        args["bench"] = locked_bench
        self._validate_cmd(args.get("bench"), "bench")
        if "card" not in args and sess.get("card"):
            args["card"] = sess["card"]
        self._limit(args)
        if sess.get("max_overhead") is not None:
            args["max_overhead_limit"] = float(sess["max_overhead"])
        if sess.get("png_only", True) and args.get("write_perfetto_gz"):
            raise ValueError(
                "`write_perfetto_gz=True` is blocked by the session PNG-only policy; "
                "deliver the timeline PNG instead (or re-install with `--allow-gz`)."
            )
        if not args.get("out") and not args.get("run_dir"):
            args["run_dir"] = str(self._runs_dir())
        ctx = supervisor_env(self.session_dir) if (self.session_dir / KEY_FILE).is_file() else contextlib.nullcontext()
        with ctx, _chdir(self.workspace):
            res = _handle_profile(args)
        return self._register(res, "weaveprof_profile")

    def _tool_plan_context(self, args: Dict[str, Any]) -> Any:
        from weaveprof.aka_adapter import _handle_plan_context

        args = dict(args)
        self._resolve_paths(args, ("source_path",))
        return _handle_plan_context(args)

    def _tool_analyze(self, args: Dict[str, Any]) -> Any:
        from weaveprof.aka_adapter import _handle_analyze

        args = dict(args)
        self._resolve_paths(args, ("trace_json_path", "data"))
        return _handle_analyze(args)

    def _tool_compare(self, args: Dict[str, Any]) -> Any:
        from weaveprof.aka_adapter import _handle_compare_rounds

        args = dict(args)
        self._resolve_paths(args, ("prev_report_path", "curr_report_path", "prev_report", "curr_report"))
        return _handle_compare_rounds(args)

    def _tool_verify_seal(self, args: Dict[str, Any]) -> Any:
        from weaveprof.aka_adapter import _handle_verify_seal

        args = dict(args)
        self._resolve_paths(args, ("seal_path",))
        ctx = supervisor_env(self.session_dir) if (self.session_dir / KEY_FILE).is_file() else contextlib.nullcontext()
        with ctx:
            res = _handle_verify_seal(args)
        if isinstance(res, dict) and res.get("ok"):
            res = dict(res)
            res["seal_path"] = args["seal_path"]
            res = self._register(res, "weaveprof_verify_seal")
        return res

    # -- JSON-RPC ----------------------------------------------------------
    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requested = params.get("protocolVersion")
        self.protocol = requested if requested in SUPPORTED_PROTOCOLS else DEFAULT_PROTOCOL
        return {
            "protocolVersion": self.protocol,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "title": "WeaveProfiler (wp)", "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        }

    def _tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments")
        if args is None:
            args = {}
        if name not in self.handlers:
            raise RpcError(-32602, f"Unknown tool: {name!r}")
        if not isinstance(args, dict):
            raise RpcError(-32602, "Tool arguments must be a JSON object")
        sink = io.StringIO()
        is_error = False
        try:
            with contextlib.redirect_stdout(sink):
                res = self.handlers[name](args)
        except Exception as exc:
            is_error = True
            res = {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}",
                   "trace_tail": traceback.format_exc().strip().splitlines()[-6:]}
        stray = sink.getvalue()
        if stray:
            sys.stderr.write(f"[weaveprofiler-mcp] captured stdout from {name}:\n{stray[-4000:]}\n")
        safe = _json_safe(res)
        out: Dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(safe, ensure_ascii=False, allow_nan=False)}],
            "isError": is_error,
        }
        if self.protocol >= "2025-06-18" and isinstance(safe, dict):
            out["structuredContent"] = safe
        return out

    def dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOL_DEFS}
        if method == "tools/call":
            return self._tools_call(params)
        if method == "resources/list":
            return {"resources": []}
        if method == "resources/templates/list":
            return {"resourceTemplates": []}
        if method == "prompts/list":
            return {"prompts": []}
        if method == "logging/setLevel":
            return {}
        raise RpcError(-32601, f"Method not found: {method}")

    def handle_one(self, msg: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(msg, dict):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        method = msg.get("method")
        has_id = "id" in msg
        mid = msg.get("id")
        if method is None:
            return None  # a response to a server->client request; ignore
        if not isinstance(method, str):
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32600, "message": "Invalid Request"}}
        if method.startswith("notifications/"):
            return None
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        try:
            result = self.dispatch(method, params)
            resp: Dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "result": result}
        except RpcError as exc:
            err: Dict[str, Any] = {"code": exc.code, "message": exc.message}
            if exc.data is not None:
                err["data"] = exc.data
            resp = {"jsonrpc": "2.0", "id": mid, "error": err}
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32603, "message": f"Internal error: {type(exc).__name__}: {exc}"}}
        return resp if has_id else None

    def _read_exact(self, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = self.inp.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def read_message(self) -> Optional[bytes]:
        while True:
            line = self.inp.readline()
            if not line:
                return None
            stripped = line.strip()
            if not stripped:
                continue
            if self.mode != "newline" and stripped.lower().startswith(b"content-length:"):
                self.mode = "header"
                try:
                    length = int(stripped.split(b":", 1)[1].strip())
                except ValueError:
                    length = -1
                while True:
                    hdr = self.inp.readline()
                    if not hdr:
                        return None
                    if not hdr.strip():
                        break
                if length < 0:
                    return b""
                return self._read_exact(length)
            if self.mode is None:
                self.mode = "newline"
            return stripped

    def write_message(self, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if self.mode == "header":
            data = b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        else:
            data = body + b"\n"
        self.out.write(data)
        self.out.flush()

    def serve(self) -> int:
        while True:
            raw = self.read_message()
            if raw is None:
                return 0
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                self.write_message({"jsonrpc": "2.0", "id": None,
                                    "error": {"code": -32700, "message": f"Parse error: {exc}"}})
                continue
            if isinstance(msg, list):
                if not msg:
                    self.write_message({"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32600, "message": "Invalid Request"}})
                    continue
                replies = [r for r in (self.handle_one(m) for m in msg) if r is not None]
                if replies:
                    self.write_message(replies)
                continue
            reply = self.handle_one(msg)
            if reply is not None:
                self.write_message(reply)


def _protect_stdout() -> Any:
    """Keep a private handle on the real stdout for JSON-RPC; route fd 1 (incl. children) to stderr."""
    sys.stdout.flush()
    try:
        rpc_fd = os.dup(1)
        os.dup2(2, 1)
        return os.fdopen(rpc_fd, "wb", buffering=0)
    except OSError:
        return sys.stdout.buffer


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="WeaveProfiler (wp) MCP stdio server")
    ap.add_argument("--workspace", help="Workspace root (default: cwd or the session's workspace)")
    ap.add_argument("--session-dir", help="Explicit session directory")
    ap.add_argument("--session-id", help="Session id under the sessions root")
    ap.add_argument("--sessions-root", help="Override sessions root (else WEAVEPROF_SESSIONS_DIR / cache)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.session_dir:
        sd = Path(args.session_dir).expanduser().resolve()
    elif args.session_id:
        sd = sessions_root(args.sessions_root) / args.session_id
    else:
        sd = session_dir_for(args.workspace or os.getcwd(), args.sessions_root)
    sess = read_session(sd) or {}
    workspace = Path(args.workspace or sess.get("workspace") or os.getcwd()).expanduser().resolve()

    os.environ.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
    os.environ.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
    if not (sd / KEY_FILE).is_file():
        sys.stderr.write(f"[weaveprofiler-mcp] warning: no session key at {sd}; seals will not be registered\n")

    out = _protect_stdout()
    server = WeaveProfServer(workspace, sd, out, sys.stdin.buffer)
    try:
        return server.serve()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
