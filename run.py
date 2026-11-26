#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weaveprofiler command entry.

Every internal tool remains callable directly, but this file gives all supported
cards one stable command surface.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
CARDS = ("h20", "b300", "mi308x", "sm86")


def _cli(argv: Sequence[str]) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from weaveprof import cli as _wp_cli
    return _wp_cli.main(list(argv))


def _run_python(relative: str, argv: Sequence[str], *, cwd: Optional[Path] = None) -> int:
    cmd = [sys.executable, "-B", str(ROOT / relative), *argv]
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _parse_deadline(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        value = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("--deadline must be ISO 8601, for example 2026-11-14T18:30:00Z: {}".format(exc))
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(str(partial), str(path))


def _kill_group(proc: subprocess.Popen) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
        elif os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
        else:
            proc.kill()
        proc.wait()


def _run_observed(cmd: Sequence[str], *, cwd: Path, deadline: Optional[float],
                  heartbeat: Optional[Path], card: str, phase: str) -> int:
    if heartbeat:
        _atomic_json(heartbeat, {"schema": "weaveprofiler-heartbeat/v1", "status": "STARTING",
                                 "phase": phase, "card": card, "pid": None,
                                 "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    proc = subprocess.Popen(list(cmd), cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    thread = threading.Thread(target=reader, name="weaveprofiler-output", daemon=True)
    thread.start()
    started = time.monotonic()
    status = "RUNNING"
    rc: Optional[int] = None
    while True:
        rc = proc.poll()
        now = time.monotonic()
        if heartbeat:
            _atomic_json(heartbeat, {"schema": "weaveprofiler-heartbeat/v1", "status": status,
                                     "phase": phase, "card": card, "pid": proc.pid,
                                     "elapsed_seconds": round(now - started, 3),
                                     "deadline_epoch": deadline,
                                     "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        if rc is not None:
            break
        if deadline is not None and time.time() >= deadline:
            status = "DEADLINE_EXCEEDED"
            _kill_group(proc)
            rc = 124
            break
        time.sleep(0.5)
    thread.join(timeout=2)
    if heartbeat:
        final_status = ("PASS" if rc == 0 else
                        "DEADLINE_EXCEEDED" if status == "DEADLINE_EXCEEDED" else
                        "FAIL")
        _atomic_json(heartbeat, {"schema": "weaveprofiler-heartbeat/v1",
                                 "status": final_status,
                                 "phase": phase, "card": card, "pid": proc.pid,
                                 "exit_code": rc, "elapsed_seconds": round(time.monotonic() - started, 3),
                                 "deadline_epoch": deadline,
                                 "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    return int(rc)


def _capture(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Route one capture to rocprof-sys/rocprofv3",
        epilog="Arguments after --payload are passed verbatim to the selected capture tool.")
    parser.add_argument("--card", required=True, choices=CARDS)
    parser.add_argument("--arch")
    parser.add_argument("--platform", choices=("nv", "nvidia", "amd", "rocm"))
    parser.add_argument("--project", "--root", dest="root")
    parser.add_argument("--out", "--out-dir", dest="out_dir")
    parser.add_argument("--collector", choices=("auto", "nsys", "rocprof-sys", "rocprofv3"))
    parser.add_argument("--deadline", help="absolute ISO 8601 UTC deadline")
    parser.add_argument("--heartbeat", help="heartbeat JSON path")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    try:
        from weaveprof.pipeline.core.registry import CARD_SPECS, UnsupportedTarget, resolve_target
    except Exception as exc:
        _json_print({"ok": False, "error": "cannot import registry: {}".format(exc)})
        return 2
    if CARD_SPECS[args.card]["platform"] == "nv":
        sys.stderr.write(
            "run.py capture is not available on NVIDIA cards (h20/b300/sm86) any more.\n"
            "Use the main audit, which runs nsys with GPU metrics itself:\n"
            '  MCP tool weaveprof_megakernel_audit with "capture_nsys": true, or\n'
            "  python3 -B weaveprof/aka_adapter.py megakernel-audit < request.json\n"
            "To capture an arbitrary program with nsys only, run nsys directly (see SKILL.md).\n"
        )
        return 2
    try:
        resolve_target(card=args.card, arch=args.arch, platform=args.platform)
    except UnsupportedTarget as exc:
        _json_print(exc.as_dict())
        return 2
    root = Path(args.root).expanduser().resolve() if args.root else None
    if root is None:
        parser.error("--project/--root is required")
    if not root.is_dir():
        parser.error("--project/--root is not a directory: {}".format(root))
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is not None and (out_dir == root or root not in out_dir.parents):
        parser.error("--out/--out-dir must be inside --project/--root and must not equal it")
    try:
        deadline = _parse_deadline(args.deadline)
    except ValueError as exc:
        parser.error(str(exc))
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    if args.collector == "nsys":
        parser.error("MI308X uses rocprof-sys with rocprofv3 fallback; nsys is invalid")
    script = ROOT / "weaveprof/pipeline/capture" / "capture_rocm.py"
    tool_argv: List[str] = ["--root", str(root)]
    if out_dir is not None:
        tool_argv.extend(["--out-dir", str(out_dir)])
    if args.collector and args.collector != "auto":
        tool_argv.extend(["--collector", args.collector])
    tool_argv.extend(rest)
    heartbeat = Path(args.heartbeat).expanduser().resolve() if args.heartbeat else None
    return _run_observed([sys.executable, "-B", str(script), *tool_argv], cwd=root,
                         deadline=deadline, heartbeat=heartbeat, card=args.card,
                         phase="capture")


def _gemm_probe(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/capture/gemm_probe.py", argv)


def _auto_gemm(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/capture/auto_gemm.py", argv)


def _probe_place(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/scan/llm_probe_placer.py", argv)


def _probe_reuse(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/instrument/incremental_reuse.py", argv)


def _probe_inject(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/pipeline/instrument/rule_engine.py", argv)


def _install_agent(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/agent_installer.py", ["install", *list(argv)])


def _guard_exec(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/agent_installer.py", ["guard-exec", *list(argv)])


def _agent_status(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/agent_installer.py", ["status", *list(argv)])


def _mcp_server(argv: Sequence[str]) -> int:
    return _run_python("weaveprof/mcp_server.py", argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    command, rest = args[0], args[1:]
    handlers = {"capture": _capture,
                "gemm-probe": _gemm_probe,
                "auto-gemm": _auto_gemm,
                "probe-place": _probe_place, "probe-reuse": _probe_reuse,
                "probe-inject": _probe_inject,
                "install-agent": _install_agent, "guard-exec": _guard_exec,
                "agent-status": _agent_status, "mcp-server": _mcp_server,
                "identify": lambda a: _cli(["identify", *a]),
                "locate": lambda a: _cli(["locate", *a]),
                "profile": lambda a: _cli(["profile", *a]),
                "verify": lambda a: _cli(["verify", *a]),
                "doctor": lambda a: _cli(["doctor", *a]),
                "causal": lambda a: _cli(["causal", *a]),
                "render": lambda a: _cli(["render", *a]),
                "seal-check": lambda a: _cli(["seal-check", *a])}
    handler = handlers.get(command)
    if handler is None:
        print("unknown command {!r}; use {}".format(command, ", ".join(handlers)), file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
