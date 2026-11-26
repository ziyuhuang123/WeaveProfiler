"""Command-line interface for WeaveProfiler (`Stage Identifier`: `identify`, `Trace Verifier`: `verify` / `profile`, `Supervisor`: `seal-check`, `Causal`: `causal`, `Env`: `doctor`)."""
import argparse
import gzip
import json
from pathlib import Path
import sys

from . import api
from .causal import analyze_sw_hw_causality
from .harness import verify_seal


def main(argv=None):
    parser = argparse.ArgumentParser(prog="weaveprof", description="WeaveProfiler Stage Identifier & Trace Verifier CLI")
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("identify", "locate", "verify", "validate", "profile"):
        is_id = action in ("identify", "locate")
        is_ver = action in ("verify", "validate")
        sub = subs.add_parser(action)
        sub.add_argument("target_pos", nargs="?", help="Optional positional source (identify/profile) or probe.json (verify)")
        sub.add_argument("--source")
        sub.add_argument("-o", "--out", required=is_id, default=None,
                         help="Output path or directory (default for profile/verify: auto <model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png)")
        sub.add_argument("--feedback")
        if not is_ver:
            sub.add_argument("--model", default="auto", help="LLM model identifier (default: auto -> qwen4-max)")
            sub.add_argument("--api-key", dest="api_key", default=None, help="Optional LLM API key")
            sub.add_argument("--base-url", dest="base_url", default=None, help="Optional chat completions base URL")
            sub.add_argument("--card", default="auto")
        if is_id:
            sub.add_argument("--offline-heuristic", action="store_true")
        else:
            sub.add_argument("--probe")
            sub.add_argument("--verify-json")
            sub.add_argument("--preview-png")
            sub.add_argument("--workload-model", dest="workload_model", default=None,
                             help="Model name token for auto-generated PNG filename")
            sub.add_argument("--op", default=None, help="Operator name token (e.g. gemm, attn, moe)")
            sub.add_argument("--bs", default=None, help="Batch size for auto-generated PNG filename")
            sub.add_argument("--seql", default=None, help="Sequence length for auto-generated PNG filename")
            sub.add_argument("--kv", default=None, help="KV cache length for auto-generated PNG filename")
            sub.add_argument("--shape", default=None, help="Explicit shape token for auto-generated PNG filename")
            sub.add_argument("--limit", type=float, default=0.10)
            sub.add_argument("--gpu", dest="gpu_id", default="auto")
            if action == "profile":
                sub.add_argument("--max-rounds", type=int, default=3)
                sub.add_argument("--llm-refine-first", action="store_true")
            sub.add_argument("bench", nargs=argparse.REMAINDER, help="-- <benchmark command> [arguments]")
    seal_sub = subs.add_parser("seal-check", help="Fail-closed verification of a sealed episode receipt (*.seal.json)")
    seal_sub.add_argument("seal", help="Path to *.seal.json")
    seal_sub.add_argument("--allow-fail", action="store_true", help="Verify integrity even if status is FAIL")
    seal_sub.add_argument("--max-limit", type=float, default=None, help="Optional hard overhead limit ceiling")

    doc_sub = subs.add_parser("doctor", help="Check local GPU/profiler/LLM/KREX-lock readiness")
    doc_sub.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi / amd-smi GPU probes")

    causal_sub = subs.add_parser("causal", help="Run SW+HW causal bottleneck & optimization-opportunity attribution")
    causal_sub.add_argument("input", help="Path to aligned Perfetto JSON[.gz] or structured SW+HW capture JSON")
    causal_sub.add_argument("--arch", default=None, help="Override GPU arch (sm_90a, sm_103a, gfx942)")
    causal_sub.add_argument("--sms", type=int, default=None, help="Override total physical SM/CU count")

    render_sub = subs.add_parser("render", help="Render a 4-panel SW+HW timeline PNG from trace.perfetto.json[.gz]")
    render_sub.add_argument("trace", help="Path to trace.perfetto.json[.gz]")
    render_sub.add_argument("-o", "--out", default=None,
                            help="Output PNG path or directory (default: auto <model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png)")
    render_sub.add_argument("--model", default=None, help="Model name token for auto-generated filename")
    render_sub.add_argument("--op", default=None, help="Operator name token (e.g. gemm, attn, moe)")
    render_sub.add_argument("--bs", default=None, help="Batch size for auto-generated filename")
    render_sub.add_argument("--seql", default=None, help="Sequence length for auto-generated filename")
    render_sub.add_argument("--kv", default=None, help="KV cache length for auto-generated filename")
    render_sub.add_argument("--shape", default=None, help="Explicit shape token for auto-generated filename")
    render_sub.add_argument("--title", default=None, help="Optional figure title")
    render_sub.add_argument("--subtitle", default=None, help="Optional figure subtitle")
    render_sub.add_argument("--sm", type=int, default=None, help="Physical SM id to feature in Panel (C)")
    render_sub.add_argument("--dpi", type=int, default=220, help="Output PNG DPI (default: 220)")
    render_sub.add_argument("--allow-no-hw", action="store_true", help="Debug-only: allow rendering without Panel (A) hardware utilization curves")

    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    if action == "seal-check":
        res = verify_seal(args["seal"], require_pass=not args["allow_fail"], max_overhead_limit=args["max_limit"])
        print(json.dumps({"ok": res["ok"], "errors": res["errors"]}, ensure_ascii=False, indent=2))
        return 0 if res["ok"] else 1
    if action == "doctor":
        rep = api.doctor(check_gpu=not args["no_gpu"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep["ready_for_offline_tests"] else 1
    if action == "causal":
        inp = Path(args["input"])
        if not inp.is_file():
            print(f"Input not found: {inp}", file=sys.stderr)
            return 1
        if inp.suffix == ".gz" or inp.name.endswith(".json.gz"):
            with gzip.open(inp, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(inp.read_text(encoding="utf-8"))
        rep = analyze_sw_hw_causality(data, arch=args["arch"], total_sms=args["sms"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0
    if action == "render":
        from .viz import render_swhw_overview_png
        out = render_swhw_overview_png(
            args["trace"], args["out"], title=args["title"], subtitle=args["subtitle"],
            focus_sm=args["sm"], dpi=args["dpi"], require_hw_curves=not args["allow_no_hw"],
            model=args["model"], op=args["op"], bs=args["bs"], seql=args["seql"],
            kv=args["kv"], shape=args["shape"],
        )
        print(str(out))
        return 0
    pos = args.pop("target_pos", None)
    if action in ("verify", "validate"):
        args["probe"] = args.get("probe") or pos
        if not args["probe"]:
            parser.error("verify requires --probe <probe.json> or positional <probe.json>")
    else:
        args["source"] = args.get("source") or pos
        if not args["source"]:
            parser.error(f"{action} requires --source <file> or positional <file>")
    if action not in ("locate", "identify"):
        if args["bench"][:1] != ["--"] or len(args["bench"]) < 2:
            parser.error("expected -- <benchmark command>")
        args["bench"] = args["bench"][1:]
    try:
        res = getattr(api, action)(**args)
        out_printed = (res.get("out_path") if isinstance(res, dict) else None) or args["out"]
        if out_printed:
            print(out_printed)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
