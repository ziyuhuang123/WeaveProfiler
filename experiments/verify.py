#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeaveProfiler Accurate Probe Verification (§3.3).

Deterministic verification enforcing the 4 Invariant Checks (Table II `tab:checks`):
  1. No Compiler Regression (Parallel CPU build, syntax/spin-loop screening, LCS sim, delta_spill_bytes <= 0, 0B added SMEM);
  2. Numerical Correctness (Bitwise output equality across all elements on Gaussian randn inputs);
  3. Complete Event Pairing (Matched start/end timestamps, zero buffer overflow, all warp roles present);
  4. Low Profiling Overhead (30 warm interleaved launches median overhead <= 10.0%, SW/HW trace alignment).

When any check fails, aggregates all violations and per-stage event statistics into a single
plain-text standard terminal error report on stdout/stderr and exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Enforce no bytecode cache
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.instrument import rule_engine as _re  # noqa: E402
from weaveprof.pipeline.instrument.rule_engine import run_cpu_parallel_preflight  # noqa: E402
from weaveprof.pipeline.scan.llm_probe_placer import detect_language  # noqa: E402
from weaveprof.pipeline.validate import closed_loop_verifier as _clv  # noqa: E402
from weaveprof.pipeline.validate.closed_loop_verifier import (  # noqa: E402
    _run_external_eval_cmd,
    build_feedback_json,
    decode_and_check_completeness,
    evaluate_four_gates,
)

NUMERIC_STAGE_MAP: Dict[int, str] = {
    1: "load",
    2: "wait",
    3: "mma",
    4: "compute",
    5: "epilogue",
    6: "store",
    7: "barrier",
    8: "dispatch",
    9: "communication",
    10: "compute",
}

NAME_STAGE_MAP: Dict[str, str] = {
    "load": "load",
    "wait": "wait",
    "mma": "mma",
    "compute": "compute",
    "epilogue": "epilogue",
    "store": "store",
    "barrier": "barrier",
    "dispatch": "dispatch",
    "communication": "communication",
    "sync": "barrier",
    "ev_load": "load",
    "ev_wait": "wait",
    "ev_mma": "mma",
    "ev_compute": "compute",
    "ev_gemm": "compute",
    "ev_store": "store",
    "ev_barrier": "barrier",
}


def normalize_semantic_id(raw: str) -> str:
    """Map raw event tokens (e.g. '1u', 'EV_LOAD', 'compute') to canonical stage names."""
    s = str(raw).strip().strip("'\"").lower()
    if s.endswith("u") and s[:-1].isdigit():
        s = s[:-1]
    if s.isdigit():
        return NUMERIC_STAGE_MAP.get(int(s), f"stage_{s}")
    return NAME_STAGE_MAP.get(s, s)


def format_terminal_error_report(eval_result: Mapping[str, Any], offline_only: bool = False) -> str:
    """Format failed checks and per-stage event statistics into a standard CLI error report."""
    gates = eval_result.get("gates", {})
    diag = eval_result.get("diagnostics", [])
    overhead = eval_result.get("overhead_pct")

    lines: List[str] = []
    lines.append("=" * 80)
    metrics = eval_result.get("metrics", {})
    if "assembly_change_rate" in metrics:
        lines.append(f"  assembly_change_rate = {metrics['assembly_change_rate']:.6f}")
    if "delta_spill_bytes" in metrics:
        lines.append(f"  delta_spill_bytes = {metrics['delta_spill_bytes']}")
    if "assembly_evidence_error" in metrics:
        lines.append(f"  Assembly evidence error: {metrics['assembly_evidence_error']}")

    if offline_only:
        sass_ok = bool(gates.get("sass_similarity_and_resources", False))
        if sass_ok:
            lines.append("[WeaveProfiler Preflight PASSED] Check 1: No Compiler Regression Satisfied:")
            lines.append("  [PASS] Check 1: No Compiler Regression (delta_spill_bytes <= 0, 0B added SMEM)")
            lines.append("=" * 80)
            return "\n".join(lines)
        else:
            lines.append("[WeaveProfiler Preflight FAILED] Check 1: Compiler Regression Detected:")
            lines.append("-" * 80)
            for d in diag:
                lines.append(f"  - [{d.get('code', 'VIOLATION')}]: {d.get('message')}")
            lines.append("=" * 80)
            return "\n".join(lines)

    passed = eval_result.get("all_gates_passed", False) or eval_result.get("all_passed", False)
    if passed:
        lines.append("[WeaveProfiler Verification PASSED] All 4 Invariant Checks Satisfied (Table II):")
        lines.append("  [PASS] Check 1: No Compiler Regression (delta_spill_bytes <= 0, 0B added SMEM)")
        lines.append("  [PASS] Check 2: Numerical Correctness (0 mismatches, bitwise exact randn equality)")
        lines.append("  [PASS] Check 3: Complete Event Pairing (0 overflow, strictly nested intervals, 100% paired)")
        ovh_str = f"{overhead:.2f}% <= 10.0%" if overhead is not None else "<= 10.0%"
        lines.append(f"  [PASS] Check 4: Low Profiling Overhead (Warm median overhead {ovh_str})")
        lines.append("=" * 80)
        return "\n".join(lines)

    gate_names = {
        "sass_similarity_and_resources": "Check 1: No Compiler Regression",
        "numerical_bitwise_randn": "Check 2: Numerical Correctness",
        "completeness_and_nesting": "Check 3: Complete Event Pairing",
        "warm_overhead_le_10pct": "Check 4: Low Profiling Overhead",
    }
    failed_keys = [k for k, v in gates.items() if not v]
    lines.append(f"[WeaveProfiler Verification FAILED] {len(failed_keys)} of 4 checks violated:")
    lines.append("-" * 80)

    for g_key, label in gate_names.items():
        is_ok = bool(gates.get(g_key, False))
        status = "[PASS]" if is_ok else "[FAIL]"
        lines.append(f"{status} {label}")

    lines.append("-" * 80)
    lines.append("Diagnostic Violations & Scheme A Coarsening Guidance:")
    coarsen_items = [d for d in diag if d.get("code") == "OVERHEAD_EXCEEDED_COARSEN_SCHEME_A"]
    if coarsen_items:
        c = coarsen_items[0]
        lines.append(f"  Measured 30-round median overhead: {c.get('measured_overhead_pct')}% (limit <= {c.get('limit_pct')}%)")
        lines.append("  Offending High-Frequency Inner-Loop Probes:")
        for off in c.get("offending_inner_loop_probes", []):
            lines.append(
                f"    - Probe '{off.get('probe_id')}' (stage '{off.get('semantic_id')}', lines {off.get('probe_lines')}): "
                f"fired {off.get('decoded_slice_count', 'N/A')} times per warp (loop depth {off.get('loop_nesting_depth')})."
            )
            lines.append(f"      Action (Scheme A): HOIST probe outward to enclosing loop at lines {off.get('enclosing_loop_lines')}.")
    for d in diag:
        if d.get("code") != "OVERHEAD_EXCEEDED_COARSEN_SCHEME_A":
            lines.append(f"  - [{d.get('code', 'VIOLATION')}]: {d.get('message')}")

    lines.append("=" * 80)
    return "\n".join(lines)


def extract_probe_spec_from_inst_code(source_code: str, inst_code: str, language: str) -> Dict[str, Any]:
    """Auto-extract probe positions, stage IDs, and warp roles directly from instrumented code."""
    import re
    inst_lines = inst_code.splitlines()
    open_stack: List[Tuple[str, str, int]] = []
    probes: List[Dict[str, Any]] = []
    current_role = "cta_worker"

    probe_call_re = re.compile(
        r'\b(?P<macro>(?:WP|FI)_B_PRED|(?:WP|FI)_E_PRED|(?:WP|FI)_B|(?:WP|FI)_E|(?:wp|fi)_b_16b|(?:wp|fi)_e_16b|(?:wp|fi)_b|(?:wp|fi)_e)\s*\((?P<args>[^)]*)\)',
        re.IGNORECASE,
    )
    role_pattern = re.compile(
        r'\b(?:(?:WP|FI)_KERNEL_SCOPE|(?:WP|FI)_KERNEL_SCOPE_BY|(?:WP|FI)_OPEN|(?:WP|FI)_OPEN_BY|(?:wp|fi)_open)\s*\((?P<args>[^)]*)\)',
        re.IGNORECASE,
    )

    for idx, line in enumerate(inst_lines, start=1):
        rm = role_pattern.search(line)
        if rm:
            r_args = [a.strip() for a in rm.group("args").split(",") if a.strip()]
            if r_args:
                raw_r = r_args[-1].strip("'\"")
                if raw_r.endswith("u") and raw_r[:-1].isdigit():
                    raw_r = raw_r[:-1]
                current_role = f"role_{raw_r}" if raw_r.isdigit() else raw_r

        for pm in probe_call_re.finditer(line):
            macro_upper = pm.group("macro").upper()
            if macro_upper.startswith("WP_"):
                macro_upper = "FI_" + macro_upper[3:]
            raw_args = [a.strip() for a in pm.group("args").split(",") if a.strip()]
            if not raw_args:
                continue

            is_begin = macro_upper in ("FI_B", "FI_B_PRED", "FI_B_16B")
            is_end = macro_upper in ("FI_E", "FI_E_PRED", "FI_E_16B")

            if macro_upper in ("FI_B_PRED", "FI_E_PRED"):
                stage_raw = raw_args[1] if len(raw_args) >= 2 else raw_args[0]
            elif macro_upper in ("FI_B_16B", "FI_E_16B"):
                stage_raw = raw_args[7] if len(raw_args) >= 8 else raw_args[-2]
            elif macro_upper in ("FI_B", "FI_E"):
                if len(raw_args) >= 7:
                    stage_raw = raw_args[5]
                elif len(raw_args) == 3:
                    stage_raw = raw_args[1]
                elif len(raw_args) == 2:
                    stage_raw = raw_args[0]
                else:
                    stage_raw = raw_args[0]
            else:
                stage_raw = raw_args[0]

            sem = normalize_semantic_id(stage_raw)

            if is_begin:
                open_stack.append((sem, current_role, idx))
            elif is_end:
                for k in range(len(open_stack) - 1, -1, -1):
                    if open_stack[k][0] == sem:
                        s_sem, s_role, b_idx = open_stack.pop(k)
                        probes.append({
                            "probe_id": f"{s_sem}_{len(probes)}",
                            "semantic_id": s_sem,
                            "role": s_role,
                            "mode": "sync",
                            "begin_line": b_idx,
                            "end_line": idx,
                        })
                        break

    return {
        "schema": "weaveprofiler.probe-spec/v1",
        "language": language,
        "probes": probes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WeaveProfiler Accurate Probe Verification (DAC'27 §3.3)",
    )
    parser.add_argument("--source", "-s", required=True, help="Baseline kernel source file (I_0)")
    parser.add_argument("--inst-source", "-i", required=True, help="Instrumented kernel source file (I_inst)")
    parser.add_argument("--probes", "--spec", "-p", default=None, help="Optional JSON probe specification (`probes.json`; if omitted, auto-extracted from inst-source)")
    parser.add_argument("--card", default="h20", choices=["h20", "b300", "mi308x"], help="Target GPU card")
    parser.add_argument("--arch", help="Target architecture (sm_90a, sm_100a, sm_103a, gfx942)")
    parser.add_argument("--rounds", type=int, default=30, help="Number of warm benchmark rounds (default: 30)")
    parser.add_argument("--overhead-limit", type=float, default=10.0, help="Max allowed overhead percentage (default: 10.0)")
    parser.add_argument("--eval-cmd", help="Command string template to run live GPU benchmark & dump FILITE02 buffer")
    parser.add_argument("--metrics-json", help="Direct path to runtime metrics JSON file")
    parser.add_argument("--dump", help="Path to binary FILITE02 dump file (e.g. capture.fi02.bin)")
    parser.add_argument("--names", help="Path to names JSON file mapping probe IDs to stage names")
    parser.add_argument("--out-perfetto", help="Path to write unified .perfetto trace")
    parser.add_argument("--out-report", help="Optional path to write JSON verification report")
    parser.add_argument("--feedback", help="Write the four paper gate decisions to this feedback.json path")
    parser.add_argument("--offline-only", action="store_true", help="Run CPU Check #1 only (skip GPU runtime checks)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    source_p = Path(args.source)
    inst_p = Path(args.inst_source)

    if not source_p.exists() or not inst_p.exists():
        print(f"[verify.py] Error: Missing input files: source={source_p.exists()}, inst={inst_p.exists()}", file=sys.stderr)
        return 1

    source_code = source_p.read_bytes().decode("utf-8")
    inst_code = inst_p.read_text(encoding="utf-8")
    lang = detect_language(source_p, source_code)

    if args.probes:
        probes_p = Path(args.probes)
        if not probes_p.exists():
            print(f"[verify.py] Error: Missing probes file: {probes_p}", file=sys.stderr)
            return 1
        probe_spec = json.loads(probes_p.read_text(encoding="utf-8"))
        if "roles" in probe_spec:
            probe_spec = _clv.normalize_probe_spec(probe_spec, source_code, language=lang)
    else:
        probe_spec = extract_probe_spec_from_inst_code(source_code, inst_code, lang)
    probe_spec.update(language=lang, card=args.card)
    if args.arch:
        probe_spec["arch"] = args.arch
    feedback_path = Path(args.feedback) if args.feedback else inst_p.parent / "feedback.json"

    # 1. Execute Check 1: No Compiler Regression (CPU parallel preflight)
    print(f"[verify.py] Executing Check 1 (No Compiler Regression) on CPU for {args.card}...")
    preflight_dir = inst_p.parent / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight_res = run_cpu_parallel_preflight(
        source_p,
        inst_p,
        probe_spec,
        preflight_dir,
    )

    # 2. Check 2, 3, 4: GPU Runtime, Completeness, Overhead
    runtime_res = None
    completeness_res = None

    if not args.offline_only:
        if not args.eval_cmd and not args.metrics_json:
            # Fail-closed: Never fake or synthesize runtime verification data
            diag = list(preflight_res.get("diagnostics", []))
            diag.append({
                "code": "MISSING_RUNTIME_EVALUATOR",
                "severity": "error",
                "message": (
                    "Live GPU verification requires either --eval-cmd or --metrics-json to evaluate "
                    "Numerical Correctness (Check 2), Completeness (Check 3), and Overhead (Check 4). "
                    "Pass --offline-only if only running CPU Check 1 preflight."
                ),
            })
            assembly_ok, metrics = _clv.evaluate_assembly(preflight_res, None, 0.85)
            eval_result = {
                "all_gates_passed": False,
                "gates": {
                    "sass_similarity_and_resources": assembly_ok,
                    "numerical_bitwise_randn": False,
                    "completeness_and_nesting": False,
                    "warm_overhead_le_10pct": False,
                },
                "overhead_pct": None,
                "metrics": metrics,
                "diagnostics": diag,
            }
            report_text = format_terminal_error_report(eval_result, offline_only=False)
            _clv.write_feedback(feedback_path, build_feedback_json(eval_result, probe_spec, source_code, metrics=metrics))
            print(report_text, file=sys.stderr)
            if args.out_report:
                Path(args.out_report).write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
            return 1

        names_p = Path(args.names) if args.names else None
        if names_p is None or not names_p.is_file():
            names_dir = inst_p.parent / "names"
            names_dir.mkdir(parents=True, exist_ok=True)
            names_p = names_dir / "names.json"
            names_map = _re.build_names_from_probe_spec(probe_spec)
            names_p.write_text(json.dumps(names_map, indent=2), encoding="utf-8")

        round_dir = inst_p.parent / "verify_run"
        round_dir.mkdir(parents=True, exist_ok=True)
        dump_p = Path(args.dump) if args.dump else None

        if args.eval_cmd:
            print(f"[verify.py] Executing live eval-cmd on {args.card} ({args.rounds} rounds)...")
            runtime_res, out_dump = _run_external_eval_cmd(
                args.eval_cmd,
                l0_source=source_p,
                inst_source=inst_p,
                names_json=names_p,
                round_dir=round_dir,
            )
            if out_dump and out_dump.is_file():
                dump_p = out_dump

        elif args.metrics_json:
            m_path = Path(args.metrics_json)
            if not m_path.is_file():
                print(f"[verify.py] Error: Missing metrics JSON file: {m_path}", file=sys.stderr)
                return 1
            runtime_res = json.loads(m_path.read_text(encoding="utf-8"))

        out_perfetto = Path(args.out_perfetto) if args.out_perfetto else round_dir / "trace.perfetto"
        if dump_p and dump_p.is_file():
            print(f"[verify.py] Decoding binary dump {dump_p} for Gate 3 Completeness check...")
            completeness_res = decode_and_check_completeness(
                dump_path=dump_p,
                names_path=names_p,
                probe_spec=probe_spec,
                out_trace_path=out_perfetto,
            )
        else:
            if runtime_res and "completeness" in runtime_res:
                completeness_res = runtime_res["completeness"]
            else:
                completeness_res = {
                    "ok": bool(runtime_res.get("completeness_ok", False)),
                    "overflow": int(runtime_res.get("overflow", 0)),
                    "unpaired_or_cross_nested": int(runtime_res.get("unpaired", 0)),
                    "missing_stages": runtime_res.get("missing_stages", []),
                    "event_counts": runtime_res.get("event_counts", {}),
                }

    # 3. Evaluate the 4 deterministic gates
    eval_result = evaluate_four_gates(
        source_text=source_code,
        probe_spec=probe_spec,
        preflight=preflight_res,
        runtime_metrics=runtime_res,
        completeness=completeness_res,
        overhead_limit_pct=args.overhead_limit,
    )
    _clv.write_feedback(feedback_path, build_feedback_json(eval_result, probe_spec, source_code, metrics=eval_result["metrics"]))

    # 4. Standard Batched Terminal Error Report
    report_text = format_terminal_error_report(eval_result, offline_only=args.offline_only)
    if args.offline_only:
        is_ok = bool(eval_result.get("gates", {}).get("sass_similarity_and_resources", False))
    else:
        is_ok = bool(eval_result.get("all_gates_passed", False) or eval_result.get("all_passed", False))

    if is_ok:
        print(report_text, file=sys.stdout)
    else:
        print(report_text, file=sys.stderr)

    if args.out_report:
        Path(args.out_report).write_text(json.dumps(eval_result, indent=2), encoding="utf-8")

    elapsed_s = time.perf_counter() - t0
    print(f"[verify.py] Completed in {elapsed_s:.2f}s (Result: {'PASS' if is_ok else 'FAIL'})")
    return 0 if is_ok else 1


if __name__ == "__main__":
    sys.exit(main())
