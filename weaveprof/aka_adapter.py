#!/usr/bin/env python3
"""Atrex Kernel Agent (AKA) `api_version: 1` JSON stdin/stdout plugin adapter.

Invoked by `aka.resources.plugin_runtime.execution.execute_json` via:
    python -B <plugin_root>/weaveprof/aka_adapter.py <tool_name>
Reads a single JSON object from stdin and writes a single JSON object to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.dont_write_bytecode = True

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from weaveprof.api import build_agent_report, profile_for_agent
from weaveprof.causal import (
    analyze_sw_hw_causality,
    compare_optimization_rounds,
    plan_context_and_search_space,
)
from weaveprof.harness import HarnessPolicy, verify_seal
from weaveprof.megakernel import run_megakernel_audit


def _load_json_or_obj(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        p = Path(val)
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"Expected JSON object or file path, got {type(val).__name__}")


def _handle_profile(req: Dict[str, Any]) -> Dict[str, Any]:
    req_limit = float(req.get("limit", 0.10))
    policy_ceiling = float(req.get("max_overhead_limit", 0.10))
    policy = HarnessPolicy(
        max_overhead_limit=policy_ceiling,
        require_abba_order=bool(req.get("require_abba_order", False)),
        enforce_trace_processor=bool(req.get("enforce_trace_processor", False)),
    )
    prev_report = None
    if "prev_causal_report" in req:
        prev_report = _load_json_or_obj(req["prev_causal_report"])
    elif "prev_causal_report_path" in req:
        prev_report = _load_json_or_obj(req["prev_causal_report_path"])

    res = profile_for_agent(
        source=req["source"],
        bench=req["bench"],
        card=req.get("card", "auto"),
        model=req.get("model", "auto"),
        api_key=req.get("api_key"),
        base_url=req.get("base_url"),
        limit=req_limit,
        max_rounds=int(req.get("max_rounds", 3)),
        out=req.get("out"),
        run_dir=req.get("run_dir"),
        workload_model=req.get("workload_model"),
        op=req.get("op"),
        bs=req.get("bs"),
        seql=req.get("seql"),
        kv=req.get("kv"),
        shape=req.get("shape"),
        policy=policy,
    )
    if prev_report is not None and isinstance(res, dict) and "round_diff" not in res:
        res = build_agent_report(res, prev_causal_report=prev_report)
    return res


def _handle_analyze(req: Dict[str, Any]) -> Dict[str, Any]:
    if "data" in req:
        data = _load_json_or_obj(req["data"])
    elif "trace_json_path" in req:
        data = _load_json_or_obj(req["trace_json_path"])
    else:
        raise ValueError("Either 'data' or 'trace_json_path' is required")
    kwargs: Dict[str, Any] = {}
    for k in ("arch", "gpu_name", "total_sms"):
        if k in req:
            kwargs[k] = req[k]
    if "workload" in req:
        kwargs["workload"] = req["workload"]
    return analyze_sw_hw_causality(data, **kwargs)


def _handle_compare_rounds(req: Dict[str, Any]) -> Dict[str, Any]:
    prev = _load_json_or_obj(req.get("prev_report") or req.get("prev_report_path"))
    curr = _load_json_or_obj(req.get("curr_report") or req.get("curr_report_path"))
    noise_floor = float(req.get("noise_floor_pct", 1.0))
    return compare_optimization_rounds(prev, curr, noise_floor_pct=noise_floor)


def _handle_plan_context(req: Dict[str, Any]) -> Dict[str, Any]:
    if "source_text" in req:
        src = str(req["source_text"])
    elif "source_path" in req:
        src = Path(req["source_path"]).read_text(encoding="utf-8")
    else:
        raise ValueError("Either 'source_text' or 'source_path' is required")
    kwargs: Dict[str, Any] = {}
    for k in (
        "arch",
        "gpu_name",
        "block_threads",
        "l0_regs",
        "l0_smem_bytes",
        "smem_limit_bytes",
        "max_warps_per_cta",
    ):
        if k in req:
            kwargs[k] = req[k]
    return plan_context_and_search_space(src, **kwargs)


def _handle_verify_seal(req: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for k in ("expected_source_sha", "require_pass", "max_overhead_limit"):
        if k in req:
            kwargs[k] = req[k]
    return verify_seal(req["seal_path"], **kwargs)


def _handle_megakernel_audit(req: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(req)
    ws = args.pop("workspace")
    return run_megakernel_audit(ws, **args)


_DISPATCH = {
    "profile": _handle_profile,
    "megakernel-audit": _handle_megakernel_audit,
    "analyze": _handle_analyze,
    "compare-rounds": _handle_compare_rounds,
    "plan-context": _handle_plan_context,
    "verify-seal": _handle_verify_seal,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in _DISPATCH:
        sys.stderr.write(f"usage: aka_adapter.py <{'|'.join(sorted(_DISPATCH))}>\n")
        return 2
    tool = args[0]
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        if not isinstance(req, dict):
            raise ValueError("Request payload must be a JSON object")
        result = _DISPATCH[tool](req)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"weaveprofiler.{tool} error: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
