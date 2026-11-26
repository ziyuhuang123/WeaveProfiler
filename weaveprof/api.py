"""Stage Locator (§3.1), Trace Validator (§3.2), and Closed-Loop Profiling Interface."""
from __future__ import annotations

from contextlib import ContextDecorator
import difflib
from functools import wraps
import gzip
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .backend import VerificationError, run_benchmark, verify_round
from .causal import analyze_sw_hw_causality, compare_optimization_rounds, plan_context_and_search_space
from .contract import compact_probe_spec, normalize_probe_spec, revise_probe_spec
from .harness import (
    EpisodeSealer,
    HarnessPolicy,
    IncumbentTracker,
    LEDGER_SCHEMA,
    PolicyGuard,
    PolicyViolation,
    RoundStateMachine,
    SealedRoundRecord,
    list_lock_providers,
    probe_fingerprint,
    verify_seal,
)
from .workspace import ShadowWorkspace, cache_root, digest, implementation, read_json, safe_products, write_json


def _bind(doc, source, card="auto"):
    placer = implementation("weaveprof.pipeline.scan.llm_probe_placer")
    raw = source.read_bytes()
    sha = digest(source)
    if doc.get("source_sha256") not in (None, sha):
        raise ValueError(f"Probe SHA-256 does not match {source}")
    text = raw.decode("utf-8")
    language = placer.detect_language(source, text, doc.get("language"))
    explicit = bool(card and str(card).lower() != "auto")
    target_card = card if explicit else (doc.get("card") or "auto")
    target_arch = None if (explicit and doc.get("card") and doc.get("card") != card) else doc.get("arch")
    route = placer.resolve_route(target_card, target_arch, language)
    doc = compact_probe_spec(normalize_probe_spec(doc, text, language=language))
    doc.update({key: route[key] for key in ("language", "arch", "card", "platform")})
    doc.update(source_sha256=sha, source_path=str(source))
    return doc


def identify(source, out=None, *, feedback=None, model="auto", card="auto",
             offline_heuristic=False, llm_caller=None, api_key=None, base_url=None):
    source = Path(source).resolve(strict=True)
    cache = cache_root(source) / digest(source) / "probe.json"
    safe_products(source, out, feedback if isinstance(feedback, (str, Path)) else None)
    prior = read_json(cache) if cache.exists() else None
    if feedback and out and Path(out).is_file():
        prior = read_json(out)
    if isinstance(feedback, (str, Path)):
        feedback = read_json(feedback)
    if feedback and prior is None:
        raise ValueError("Refinement needs the original placement in the source cache or output probe")
    if prior and not feedback:
        doc = _bind(prior, source, card)
    else:
        placer = implementation("weaveprof.pipeline.scan.llm_probe_placer")
        doc = placer.identify_source(source, model=model, card=card, offline_heuristic=offline_heuristic,
                                     previous=prior, feedback=feedback, llm_caller=llm_caller,
                                     api_key=api_key, base_url=base_url)
        doc = _bind(doc, source, card)
        write_json(cache, doc)
    if out:
        write_json(out, doc)
    return doc


locate = identify


class _Session(ContextDecorator):
    def __init__(self, source, limit, out, probe, feedback, model, max_rounds, card,
                 verify_only=False, verify_json=None, preview_png=None, *,
                 policy=None, gpu_id="auto", bench_cmd=None, llm_refine_first=False,
                 llm_caller=None, api_key=None, base_url=None,
                 workload_model=None, op=None, bs=None, seql=None, kv=None, shape=None, dims=None):
        from .viz import build_timeline_png_name

        self.source = Path(source).resolve(strict=True)
        self.limit = float(limit)
        self.policy = policy if isinstance(policy, HarnessPolicy) else HarnessPolicy()
        self.guard = PolicyGuard(self.policy)
        self.gpu_id = gpu_id
        self.bench_cmd = list(bench_cmd) if bench_cmd else None
        self.llm_refine_first = bool(llm_refine_first)
        self.llm_caller = llm_caller
        self.api_key = api_key
        self.base_url = base_url

        raw_out = "" if out is None else str(out).strip()
        auto_name = (
            not raw_out
            or raw_out.lower() == "auto"
            or raw_out.endswith(("/", "\\"))
            or (Path(raw_out).exists() and Path(raw_out).is_dir())
            or (not Path(raw_out).suffix and not Path(raw_out).exists())
        )
        out_dir = (
            Path.cwd().resolve()
            if (not raw_out or raw_out.lower() == "auto")
            else (Path(raw_out).resolve() if auto_name else Path(raw_out).resolve().parent)
        )
        if auto_name:
            out_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_path = Path(feedback or (out_dir / "feedback.json")).resolve()
        self.verify_path = Path(verify_json or (out_dir / "verify.json")).resolve()
        self.ledger_path = (out_dir / "episode_ledger.json").resolve()
        self.probe_path = Path(probe).resolve() if isinstance(probe, (str, Path)) else (out_dir / "probe.json").resolve()
        if max_rounds < 1 or not math.isfinite(self.limit) or self.limit < 0:
            raise ValueError("max_rounds must be positive and limit must be finite and nonnegative")
        self.model, self.max_rounds, self.card = model, max_rounds, card
        self.workspace = ShadowWorkspace(self.source)
        self.guard.validate_session_start(self.source, self.workspace.sha, self.limit, self.bench_cmd)
        if isinstance(probe, dict):
            self.spec = _bind(probe, self.source, card)
            write_json(self.probe_path, self.spec)
        elif isinstance(probe, (str, Path)) and (verify_only or Path(probe).exists()):
            self.spec = _bind(read_json(probe), self.source, card)
        else:
            self.spec = identify(self.source, out=self.probe_path, card=card, model=model,
                                 llm_caller=self.llm_caller, api_key=self.api_key, base_url=self.base_url)

        if auto_name:
            fname = build_timeline_png_name(
                model=workload_model, op=op, bs=bs, seql=seql, kv=kv,
                shape=shape, dims=dims, source=self.source, spec=self.spec,
            )
            self.out = (out_dir / fname).resolve()
        else:
            self.out = Path(raw_out).resolve()
        self.preview_path = (
            self.out if self.out.suffix.lower() == ".png"
            else (Path(preview_png).resolve() if preview_png else self.out.with_suffix(".png"))
        )
        self.seal_path = Path(str(self.out) + ".seal.json").resolve()
        prod_paths = [self.source, self.probe_path, self.feedback_path, self.verify_path, self.out]
        if self.preview_path != self.out:
            prod_paths.append(self.preview_path)
        safe_products(*prod_paths)
        self.source_text = self.source.read_bytes().decode("utf-8")
        self.prepared = None
        self.result = None
        self.fsm = RoundStateMachine(max_rounds=self.max_rounds,
                                     escalation_on_stagnation=self.policy.escalation_on_stagnation)
        self.tracker = IncumbentTracker()

    def _prepare(self, number):
        self.guard.validate_source_immutable(self.source, self.workspace.sha)
        policy_issues = self.guard.validate_probe_spec(self.spec, self.source_text)
        if policy_issues:
            failure = dict(
                status="FAIL", failed=["assembly"], fires={}, enclosing_loop=None,
                checks={"assembly": False, "output": False, "event": False, "events": False, "overhead": False},
                metrics={"errors": {"policy": "; ".join(i["message"] for i in policy_issues)},
                         "policy_violations": policy_issues},
            )
            rdir = self.workspace.path / f"round-{number}"
            rdir.mkdir(parents=True, exist_ok=True)
            write_json(self.feedback_path, failure)
            write_json(self.verify_path, failure)
            write_json(rdir / "feedback.json", failure)
            write_json(rdir / "probe.json", self.spec)
            EpisodeSealer.seal_round(
                round_dir=rdir, round_number=number, source_path=self.source,
                source_sha256=self.workspace.sha, probe_path=rdir / "probe.json",
                instrumented_path=None, names_path=None, feedback_path=rdir / "feedback.json",
                verify_path=None, trace_path=None, failed_trace_path=None,
                result=failure, policy=self.policy, limit=self.limit,
                bench_cmd=self.bench_cmd, fsm_state="POLICY_CHECK",
            )
            return None, rdir, failure
        try:
            return self.workspace.prepare(number, self.spec), None, None
        except Exception as exc:
            failure = dict(status="FAIL", failed=["assembly"], fires={}, enclosing_loop=None,
                           checks={"assembly": False, "output": False, "event": False, "events": False, "overhead": False},
                           metrics={"errors": {"assembly": str(exc)}})
            rdir = self.workspace.path / f"round-{number}"
            rdir.mkdir(parents=True, exist_ok=True)
            write_json(self.feedback_path, failure)
            write_json(self.verify_path, failure)
            write_json(rdir / "feedback.json", failure)
            write_json(rdir / "probe.json", self.spec)
            EpisodeSealer.seal_round(
                round_dir=rdir, round_number=number, source_path=self.source,
                source_sha256=self.workspace.sha, probe_path=rdir / "probe.json",
                instrumented_path=None, names_path=None, feedback_path=rdir / "feedback.json",
                verify_path=None, trace_path=None, failed_trace_path=None,
                result=failure, policy=self.policy, limit=self.limit,
                bench_cmd=self.bench_cmd, fsm_state="PREPARE",
            )
            return None, rdir, failure

    def _write_ledger(self, final_status, reason):
        ledger = {
            "schema": LEDGER_SCHEMA,
            "status": final_status,
            "reason": reason,
            "source_path": str(self.source),
            "source_sha256": self.workspace.sha,
            "policy_digest": self.policy.digest(),
            "limit": self.limit,
            "max_rounds": self.max_rounds,
            "rounds_executed": len(self.tracker.records),
            "promoted_round": self.tracker.promoted.round_number if self.tracker.promoted else None,
            "incumbent": self.tracker.incumbent.to_dict() if self.tracker.incumbent else None,
            "rounds": [r.to_dict() for r in self.tracker.records],
            "fsm_transitions": self.fsm.transitions,
        }
        write_json(self.ledger_path, ledger)
        return ledger

    def execute(self, measure):
        directory = self.workspace.path
        escalation_tag = None
        last_reason = "unknown"
        prev_seal_sha = None
        promoted_result = None
        try:
            for number in range(1, self.max_rounds + 1):
                self.guard.validate_source_immutable(self.source, self.workspace.sha)
                self.fsm.transition("POLICY_CHECK", round=number)
                probe_fp = probe_fingerprint(self.spec, self.source_text)
                if number == 1 and self.prepared is not None:
                    prepared, fail_dir, prep_failure = self.prepared, None, None
                else:
                    prepared, fail_dir, prep_failure = self._prepare(number)

                if prep_failure is not None:
                    directory = fail_dir
                    self.result = prep_failure
                    seal_f = directory / "seal.json"
                    seal_sha = read_json(seal_f)["seal_sha256"] if seal_f.is_file() else ""
                    prev_seal_sha = seal_sha or prev_seal_sha
                    self.tracker.record(SealedRoundRecord(
                        round_number=number, probe_fp=probe_fp, status="FAIL",
                        checks={"assembly": False, "output": False, "events": False, "overhead": False},
                        failed=list(prep_failure.get("failed") or ["assembly"]),
                        overhead_ratio=None, seal_path=str(seal_f), seal_sha256=seal_sha,
                        round_dir=str(directory), escalation=escalation_tag,
                    ))
                else:
                    directory, _base, inst_src = prepared
                    self.fsm.transition("CRITICAL_REGION_MEASURE", round=number, gpu_id=self.gpu_id)
                    self.result = verify_round(
                        directory, self.spec, self.source_text, self.limit, self.out,
                        self.feedback_path, lambda: measure(prepared),
                        verify_json=self.verify_path, preview_png=self.preview_path,
                        policy=self.policy, gpu_id=self.gpu_id, round_number=number,
                        source_path=self.source, source_sha256=self.workspace.sha,
                        probe_path=directory / "probe.json", instrumented_path=inst_src,
                        bench_cmd=self.bench_cmd, fsm_state="VERIFY_AND_SEAL",
                        escalation=escalation_tag, prev_seal_sha256=prev_seal_sha,
                    )
                    checks = self.result.get("checks") or {}
                    prev_seal_sha = str(self.result.get("seal_sha256") or "") or prev_seal_sha
                    self.tracker.record(SealedRoundRecord(
                        round_number=number, probe_fp=probe_fp,
                        status=str(self.result.get("status") or "FAIL"),
                        checks={
                            "assembly": bool(checks.get("assembly")),
                            "output": bool(checks.get("output")),
                            "events": bool(checks.get("events", checks.get("event"))),
                            "overhead": bool(checks.get("overhead")),
                        },
                        failed=list(self.result.get("failed") or []),
                        overhead_ratio=(self.result.get("metrics") or {}).get("overhead_ratio"),
                        seal_path=str(self.result.get("seal_path") or (directory / "seal.json")),
                        seal_sha256=str(self.result.get("seal_sha256") or ""),
                        round_dir=str(directory), escalation=escalation_tag,
                    ))

                action, last_reason = self.fsm.decide_after_round(number, probe_fp, self.result)
                if action == "PROMOTE":
                    if self.tracker.promoted is None or self.tracker.promoted.round_number != number:
                        raise VerificationError(
                            {"status": "FAIL", "failed": ["incumbent"], "errors": ["Promoted round is not incumbent"]},
                            directory,
                        )
                    # Final Supervisor seal bound directly to the promoted `self.out` artifact
                    promoted_seal, seal_doc = EpisodeSealer.seal_round(
                        round_dir=directory, round_number=number, source_path=self.source,
                        source_sha256=self.workspace.sha,
                        probe_path=self.probe_path if self.probe_path.is_file() else (directory / "probe.json"),
                        instrumented_path=prepared[2] if prepared else None,
                        names_path=(directory / "names.json") if (directory / "names.json").is_file() else None,
                        feedback_path=self.feedback_path,
                        verify_path=self.verify_path,
                        trace_path=self.out,
                        failed_trace_path=None,
                        result=self.result,
                        policy=self.policy, limit=self.limit,
                        bench_cmd=self.bench_cmd, fsm_state="PROMOTED",
                        escalation=escalation_tag,
                        prev_seal_sha256=prev_seal_sha,
                        extra_artifacts=({"overview_png": self.preview_path} if self.preview_path.is_file() else None),
                        out_seal_path=self.seal_path,
                    )
                    check = verify_seal(
                        promoted_seal, require_pass=True,
                        expected_source_sha=self.workspace.sha,
                        expected_policy_digest=self.policy.digest(),
                        max_overhead_limit=self.policy.max_overhead_limit,
                    )
                    if not check["ok"]:
                        last_reason = "seal_verification_failed"
                        raise VerificationError(
                            {"status": "FAIL", "failed": ["seal"], "errors": check["errors"]}, directory
                        )
                    self.result["seal_path"] = str(promoted_seal)
                    self.result["seal_sha256"] = seal_doc["seal_sha256"]
                    self.result["ledger_path"] = str(self.ledger_path)
                    promoted_result = self.result
                    return promoted_result

                if action == "HALT_BLOCKED":
                    break

                if action == "ESCALATE_DETERMINISTIC":
                    escalation_tag = f"deterministic_loop_hoist:{last_reason}"
                    revised = revise_probe_spec(self.spec, self.result, self.source_text)
                else:
                    escalation_tag = None
                    placer = implementation("weaveprof.pipeline.scan.llm_probe_placer")
                    revised = placer.identify_source(
                        self.source, previous=self.spec, feedback=self.result,
                        model=self.model, card=self.card,
                        prefer_llm_on_feedback=self.llm_refine_first,
                        llm_caller=self.llm_caller,
                        api_key=self.api_key, base_url=self.base_url,
                    )
                next_fp = probe_fingerprint(revised, self.source_text)
                if next_fp == probe_fp:
                    # If LLM repeated the identical placement, try deterministic hoist escalation once before halting
                    if (self.policy.escalation_on_stagnation and self.result.get("enclosing_loop")
                            and "pairing" not in (self.result.get("failed") or [])):
                        escalation_tag = "deterministic_loop_hoist:duplicate_proposal"
                        self.fsm.transition("ESCALATE_DETERMINISTIC", round=number, reason="duplicate_proposal")
                        revised = revise_probe_spec(self.spec, self.result, self.source_text)
                        next_fp = probe_fingerprint(revised, self.source_text)
                    if next_fp == probe_fp:
                        last_reason = "no_further_refinement"
                        self.fsm.transition("HALT_BLOCKED", round=number, reason=last_reason)
                        break
                self.spec = _bind(revised, self.source, self.card)
                write_json(cache_root(self.source) / self.workspace.sha / "probe.json", self.spec)
                if self.probe_path:
                    write_json(self.probe_path, self.spec)

            raise VerificationError(self.result, directory)
        finally:
            self._write_ledger("PASS" if promoted_result is not None else "FAIL", last_reason)

    def __enter__(self):
        from .replay import capture
        prepared, fail_dir, prep_failure = self._prepare(1)
        if prep_failure is not None:
            raise VerificationError(prep_failure, fail_dir)
        self.prepared = prepared
        self.capture = capture(self.source, self.workspace.path, self.spec, self.prepared)
        self.capture.__enter__()
        return self

    def __exit__(self, kind, value, tb):
        self.capture.__exit__(kind, value, tb)
        if kind is None:
            from .replay import measure
            self.execute(lambda prepared: measure(self.workspace, prepared))
        return False


def profile(source, *, limit=0.10, out=None, probe=None, feedback=None,
            verify_json=None, preview_png=None, model="auto", max_rounds=3, card="auto",
            bench=None, policy=None, gpu_id="auto", llm_refine_first=False, llm_caller=None,
            api_key=None, base_url=None, workload_model=None, op=None, bs=None, seql=None,
            kv=None, shape=None, dims=None):
    session = _Session(source, limit, out, probe, feedback, model, max_rounds, card,
                       verify_only=False, verify_json=verify_json, preview_png=preview_png,
                       policy=policy, gpu_id=gpu_id, bench_cmd=bench,
                       llm_refine_first=llm_refine_first, llm_caller=llm_caller,
                       api_key=api_key, base_url=base_url,
                       workload_model=workload_model, op=op, bs=bs, seql=seql,
                       kv=kv, shape=shape, dims=dims)
    if bench is None:
        return session
    res = session.execute(lambda prepared: run_benchmark(
        session.workspace, prepared, bench, session.spec, timeout_s=session.policy.command_timeout_s))
    if isinstance(res, dict):
        res.setdefault("out_path", str(session.out))
    return res


def verify(probe, *, source=None, limit=0.10, feedback=None, verify_json=None,
           preview_png=None, out=None, bench, policy=None, gpu_id="auto",
           workload_model=None, op=None, bs=None, seql=None, kv=None, shape=None, dims=None):
    doc = read_json(probe) if isinstance(probe, (str, Path)) else probe
    if not source:
        source = doc.get("source_path") or doc.get("source_file")
        if not source:
            raise ValueError("This placement has no source_path. Supply --source")
        if not Path(source).is_absolute() and isinstance(probe, (str, Path)):
            source = Path(probe).resolve().parent / source
    session = _Session(source, limit, out, probe, feedback, None, 1, doc.get("card", "auto"),
                       verify_only=True, verify_json=verify_json, preview_png=preview_png,
                       policy=policy, gpu_id=gpu_id, bench_cmd=bench,
                       workload_model=workload_model, op=op, bs=bs, seql=seql,
                       kv=kv, shape=shape, dims=dims)
    res = session.execute(lambda prepared: run_benchmark(
        session.workspace, prepared, bench, session.spec, timeout_s=session.policy.command_timeout_s))
    if isinstance(res, dict):
        res.setdefault("out_path", str(session.out))
    return res


validate = verify


def wrap(function=None, *, source=None, **options):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with profile(source or inspect.getsourcefile(fn), **options):
                return fn(*args, **kwargs)
        return wrapped
    return decorate if function is None else decorate(function)


# ---------------------------------------------------------------------------
# Compact Agent-Facing Report & Atrex Closed-Loop Entrypoint
# ---------------------------------------------------------------------------

AGENT_REPORT_SCHEMA = "weaveprof.agent-report/v1"


def _load_json_maybe_gz(path):
    p = Path(path)
    if not p.is_file():
        return None
    if p.suffix == ".gz" or p.name.endswith(".json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return read_json(p)


def build_agent_report(result, *, aligned_trace=None, causal_input=None, arch=None, max_opportunities=5,
                       prev_causal_report=None):
    """Build a compact, token-efficient agent JSON report from a verified profiling result.

    Designed for coding/optimization agents (e.g. Atrex): returns verified seal status,
    overhead %, fidelity gates, ranked causal bottlenecks, and refinement verdicts without
    dumping multi-MB raw Perfetto traces into the LLM context window.

    When `prev_causal_report` is supplied (the previous round's `analyze_sw_hw_causality()`
    output, or the previous round's agent report, which carries the same diff-relevant keys),
    `round_diff` carries the D3 round-to-round diff including `next_round_directive`.
    """
    res = dict(result or {})
    metrics = dict(res.get("metrics") or {})
    status = str(res.get("status") or "FAIL")

    seal_path = res.get("seal_path")
    seal_verified = False
    seal_errors = []
    if seal_path and Path(seal_path).is_file():
        chk = verify_seal(seal_path, require_pass=(status == "PASS"))
        seal_verified = bool(chk.get("ok"))
        seal_errors = list(chk.get("errors") or [])

    ovh_ratio = metrics.get("overhead_ratio")
    overhead_pct = round(float(ovh_ratio) * 100.0, 3) if ovh_ratio is not None else None

    # Load aligned trace for causal attribution if available.
    trace_doc = causal_input if causal_input is not None else aligned_trace
    if trace_doc is None and metrics.get("perfetto_json_gz"):
        try:
            trace_doc = _load_json_maybe_gz(metrics["perfetto_json_gz"])
        except Exception:
            trace_doc = None

    causal = None
    if isinstance(trace_doc, dict) and (
        "traceEvents" in trace_doc or "hw_series" in trace_doc or "hw_both" in trace_doc or "hw_k2" in trace_doc
    ):
        causal = analyze_sw_hw_causality(trace_doc, arch=arch)

    stage_breakdown = []
    if causal and causal.get("stage_validity"):
        for sv in causal["stage_validity"][:12]:
            stage_breakdown.append({
                "stage": sv["stage"],
                "median_us": sv["median_us"],
                "share_pct": sv["share_pct"],
                "hw_regime": sv["hw_regime"],
                "refinement_verdict": sv["refinement_verdict"],
            })
    elif res.get("timeline_summary") and (res["timeline_summary"].get("hotspots") or []):
        for hs in res["timeline_summary"]["hotspots"][:10]:
            stage_breakdown.append({
                "stage": hs.get("name") or hs.get("stage"),
                "fires": hs.get("fires"),
                "avg_us": hs.get("avg_us"),
                "total_us": hs.get("total_us"),
            })
    elif res.get("fires"):
        for k, v in list((res.get("fires") or {}).items())[:12]:
            stage_breakdown.append({"stage": str(k), "fires": int(v)})

    pg = metrics.get("product_gate") or {}
    fidelity_gates = {
        "checks": dict(res.get("checks") or {}),
        "compiler_spill_gate": metrics.get("compiler_spill_gate"),
        "stub_gate": pg.get("stub_gate"),
        "coverage_gate": pg.get("coverage_gate"),
        "trace_processor": metrics.get("trace_processor"),
        "abba": metrics.get("abba"),
    }

    opportunities = (causal.get("causal_opportunities") or [])[:max_opportunities] if causal else []
    bottleneck_hint = (
        causal.get("bottleneck_hint")
        if causal
        else (
            f"Profiling failed gates: {res.get('failed')}"
            if status != "PASS"
            else "All profiling gates passed; supply aligned SW+HW trace for causal attribution."
        )
    )

    round_diff = None
    if causal and isinstance(prev_causal_report, dict):
        round_diff = compare_optimization_rounds(prev_causal_report, causal)

    return {
        "schema": AGENT_REPORT_SCHEMA,
        "status": status,
        "seal_verified": seal_verified,
        "seal_errors": seal_errors,
        "seal_path": seal_path,
        "seal_sha256": res.get("seal_sha256"),
        "ledger_path": res.get("ledger_path"),
        "overhead_pct": overhead_pct,
        "kernel_us": causal.get("kernel_us") if causal else None,
        "global_hw_regime": causal.get("global_hw_regime") if causal else None,
        "verdict": causal.get("verdict") if causal else None,
        "caliber": causal.get("caliber") if causal else None,
        "hw_summary": causal.get("hw_summary") if causal else None,
        "phases": (causal.get("phases") or [])[:12] if causal else [],
        "stage_breakdown": stage_breakdown,
        "structural_blockers": causal.get("structural_blockers") if causal else [],
        "overlap_recommendations": causal.get("overlap_recommendations") if causal else [],
        "counterfactual_upper_bound_us": causal.get("counterfactual_upper_bound_us") if causal else None,
        "causal_opportunities": opportunities,
        "bottleneck_hint": bottleneck_hint,
        "refinement_advice": causal.get("refinement_summary") if causal else None,
        "round_diff": round_diff,
        "fidelity_gates": fidelity_gates,
        "trace_path": str(Path(seal_path).name[:-len(".seal.json")]) if (seal_path and str(seal_path).endswith(".seal.json")) else None,
        "perfetto_json_gz": metrics.get("perfetto_json_gz"),
        "preview_png": metrics.get("preview_png"),
        "failed": list(res.get("failed") or []),
    }


def profile_for_agent(
    source,
    *,
    bench,
    limit=0.10,
    out=None,
    run_dir=None,
    probe=None,
    feedback=None,
    verify_json=None,
    preview_png=None,
    model="auto",
    max_rounds=3,
    card="auto",
    policy=None,
    gpu_id="auto",
    llm_refine_first=False,
    llm_caller=None,
    api_key=None,
    base_url=None,
    arch=None,
    workload_model=None,
    op=None,
    bs=None,
    seql=None,
    kv=None,
    shape=None,
    dims=None,
):
    """Run closed-loop profiling and return a compact, seal-verified report for an agent."""
    if run_dir and (not out or str(out).strip().lower() == "auto"):
        run_path = Path(run_dir).resolve()
        run_path.mkdir(parents=True, exist_ok=True)
        out = str(run_path)
    try:
        res = profile(
            source,
            limit=limit,
            out=out,
            probe=probe,
            feedback=feedback,
            verify_json=verify_json,
            preview_png=preview_png,
            model=model,
            max_rounds=max_rounds,
            card=card,
            bench=bench,
            policy=policy,
            gpu_id=gpu_id,
            llm_refine_first=llm_refine_first,
            llm_caller=llm_caller,
            api_key=api_key,
            base_url=base_url,
            workload_model=workload_model,
            op=op,
            bs=bs,
            seql=seql,
            kv=kv,
            shape=shape,
            dims=dims,
        )
        report = build_agent_report(res, arch=arch)
        actual_out = (res.get("out_path") if isinstance(res, dict) else None) or out
        if actual_out and Path(actual_out).is_file():
            report["trace_path"] = str(Path(actual_out).resolve())
        elif report.get("preview_png") and Path(report["preview_png"]).is_file():
            report["trace_path"] = str(Path(report["preview_png"]).resolve())
        return report
    except VerificationError as exc:
        report = build_agent_report(exc.feedback, arch=arch)
        report["round_dir"] = str(exc.directory)
        return report


def locate_or_remap_0token(prior_spec, old_source_text, new_source_text, *, new_source_path=None):
    """0-Token incremental probe reuse across optimization-agent edits (Atrex Route A).

    Uses `difflib.SequenceMatcher` line alignment + exact anchor-snippet verification to
    shift existing probe line numbers when a coding/optimization agent edits a kernel.
    Returns `(remapped_spec, stats)` where `stats['reused_zero_token']` is True iff every
    probe anchor survived cleanly (otherwise caller falls back to LLM re-identification).
    """
    import hashlib
    old_lines = old_source_text.splitlines()
    new_lines = new_source_text.splitlines()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    line_map = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                line_map[i1 + offset + 1] = j1 + offset + 1

    def _remap_line(old_ln):
        old_i = int(old_ln)
        if old_i in line_map:
            return line_map[old_i]
        # Fallback: check exact stripped-line match within a narrow local window around mapped neighbor.
        if 1 <= old_i <= len(old_lines):
            needle = old_lines[old_i - 1].strip()
            if needle:
                # Estimate approximate target line from nearest mapped predecessor.
                approx = old_i
                for back in range(1, 25):
                    if (old_i - back) in line_map:
                        approx = line_map[old_i - back] + back
                        break
                lo = max(1, approx - 8)
                hi = min(len(new_lines), approx + 8)
                hits = [ln for ln in range(lo, hi + 1) if new_lines[ln - 1].strip() == needle]
                if len(hits) == 1:
                    return hits[0]
        return None

    remapped = json.loads(json.dumps(prior_spec))
    failed_probes = []
    shifted = 0
    for probe in remapped.get("probes") or []:
        pid = probe.get("id") or probe.get("probe_id") or "?"
        ok_p = True
        for end_key, flat_key in (("begin", "begin_line"), ("end", "end_line")):
            if isinstance(probe.get(end_key), dict) and probe[end_key].get("line") is not None:
                old_ln = int(probe[end_key]["line"])
                new_ln = _remap_line(old_ln)
                if new_ln is None:
                    ok_p = False
                else:
                    if new_ln != old_ln:
                        shifted += 1
                    probe[end_key]["line"] = new_ln
                    if flat_key in probe:
                        probe[flat_key] = new_ln
            elif probe.get(flat_key) is not None:
                old_ln = int(probe[flat_key])
                new_ln = _remap_line(old_ln)
                if new_ln is None:
                    ok_p = False
                else:
                    if new_ln != old_ln:
                        shifted += 1
                    probe[flat_key] = new_ln
        if not ok_p:
            failed_probes.append(pid)

    new_sha = hashlib.sha256(new_source_text.encode("utf-8")).hexdigest()
    remapped["source_sha256"] = new_sha
    if new_source_path is not None:
        remapped["source_path"] = str(Path(new_source_path).resolve())

    stats = {
        "reused_zero_token": len(failed_probes) == 0 and bool(remapped.get("probes")),
        "probes_total": len(remapped.get("probes") or []),
        "endpoints_shifted": shifted,
        "failed_probes": failed_probes,
        "new_source_sha256": new_sha,
    }
    return (remapped if stats["reused_zero_token"] else None), stats


def doctor(*, check_gpu=True):
    """Inspect local profiling environment, tools, GPU visibility, KREX lock, and LLM keys."""
    def _which_any(*names):
        for name in names:
            p = shutil.which(name)
            if p:
                return p
        return None

    def _run_probe(cmd, timeout=5.0):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            return {"ok": proc.returncode == 0, "rc": proc.returncode, "out": (proc.stdout or proc.stderr or "").strip()[:300]}
        except OSError as exc:
            return {"ok": False, "rc": -1, "out": str(exc)}
        except subprocess.TimeoutExpired:
            return {"ok": False, "rc": -2, "out": "timeout"}

    from .megakernel import find_nsys_binary
    tools = {
        "nvcc": _which_any("nvcc"),
        "hipcc": _which_any("hipcc"),
        "cuobjdump": _which_any("cuobjdump"),
        "llvm_objdump": _which_any("llvm-objdump", "amd-llvm-objdump"),
        "nsys": find_nsys_binary(),
        "rocprof_sys": _which_any("rocprof-sys-run", "rocprof-sys"),
        "rocprofv3": _which_any("rocprofv3"),
        "trace_processor_shell": _which_any("trace_processor_shell", "trace_processor"),
    }
    if not tools["trace_processor_shell"]:
        for cand in (
            Path.home() / ".cache" / "weaveprof" / "trace_processor_shell",
            Path.home() / ".local" / "bin" / "trace_processor_shell",
        ):
            if cand.is_file():
                tools["trace_processor_shell"] = str(cand)
                break

    gpus = {"checked": bool(check_gpu), "nvidia": None, "amd": None}
    if check_gpu:
        if _which_any("nvidia-smi"):
            gpus["nvidia"] = _run_probe(
                ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader"]
            )
        if _which_any("amd-smi", "rocm-smi"):
            cmd = ["amd-smi", "list"] if _which_any("amd-smi") else ["rocm-smi", "--showproductname"]
            gpus["amd"] = _run_probe(cmd)

    # Check LLM endpoint & key availability without leaking secret values.
    placer = implementation("weaveprof.pipeline.scan.llm_probe_placer")
    llm_cfg = placer.resolve_llm_config()

    lock_dir = Path(
        os.environ.get("WEAVEPROF_LOCK_DIR")
        or (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "weaveprof" / "krex_locks")
    ).resolve()
    lock_writable = False
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        probe_f = lock_dir / f".doctor_{os.getpid()}"
        probe_f.write_text("ok", encoding="utf-8")
        probe_f.unlink()
        lock_writable = True
    except OSError:
        lock_writable = False

    return {
        "schema": "weaveprof.doctor/v1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tools": tools,
        "gpus": gpus,
        "llm": {
            "endpoint_mode": llm_cfg["endpoint_mode"],
            "endpoint_url": llm_cfg["endpoint_url"],
            "default_model": llm_cfg["model"],
            "openai_base_url_set": bool((os.environ.get("OPENAI_BASE_URL") or "").strip()),
            "api_key_detected": llm_cfg["api_key"] is not None,
            "api_key_source": llm_cfg["api_key_source"],
        },
        "krex_lock": {
            "active_provider": os.environ.get("WEAVEPROF_LOCK_PROVIDER") or "local_file",
            "registered_providers": list_lock_providers(),
            "lock_dir": str(lock_dir),
            "lock_dir_writable": lock_writable,
        },
        "ready_for_offline_tests": lock_writable and sys.version_info >= (3, 10),
        "ready_for_gpu_capture": bool(
            lock_writable and ((gpus["nvidia"] and gpus["nvidia"]["ok"]) or (gpus["amd"] and gpus["amd"]["ok"]))
        ),
    }
