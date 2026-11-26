"""Sparse Attention & Persistent Megakernel Audit Engine (WeaveProfiler `wp`).

Fast, fail-closed audit of persistent megakernels (SparseMLA / TopK sparse attention, GDN, MoE)
from WeaveProfiler recorder dumps (``WPLITE02`` / ``FILITE02``):

* ``WpSoController``           - ctypes wrapper over the ``-DWP_DEFINE_HOST_API`` exports
                                 (``wp_init`` / ``wp_reset`` / ``wp_dump`` / ``wp_free``) of a
                                 multi-file PyTorch / ctypes ``.so`` megakernel.
* ``analyze_megakernel_skew``  - per-SM busy distribution, straggler vs early-finishing /
                                 barrier-starved SM partition, warp-role balance, stage breakdown
                                 and concrete megakernel optimization hints.
* ``extract_nsys_sqlite_fast`` - direct ``sqlite3`` reader of the warm kernel window plus in-window
                                 ``GPU_METRICS`` curves (no multi-process exporters).
* ``run_megakernel_audit``     - Workflow A (shadow L0/L_inst build + KREX-locked bench) or
                                 Workflow B (pre-captured dump + runtime metrics), the 4 fidelity
                                 gates, skew + SW/HW causal analysis, sealed episode, <2 KB report.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ctypes
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .causal import analyze_sw_hw_causality, compare_optimization_rounds, get_arch_profile
from .harness import (
    EpisodeSealer,
    HarnessPolicy,
    PolicyGuard,
    PolicyViolation,
    gpu_critical_region,
    verify_compiler_spill_gate,
    verify_seal,
    verify_semantic_body_and_stubs,
    verify_stage_granularity,
    verify_sw_hw_coverage,
    verify_warp_completeness,
)
from .workspace import ROOT, digest, read_json, write_json

MEGAKERNEL_SCHEMA = "weaveprof.megakernel-audit/v1"
SKEW_SCHEMA = "weaveprof.megakernel-skew/v1"
NSYS_FAST_SCHEMA = "weaveprof.nsys-sqlite-fast/v1"
REPORT_MAX_BYTES = 2048

SHADOW_IGNORE: Tuple[str, ...] = (
    ".git", "__pycache__", "*.pyc", "*.bin", "*.sqlite", "*.nsys-rep", "*.so",
    "*.perfetto*", "*.tar*", "*.whl",
    ".weaveprof_cache", ".pytest_cache", ".mypy_cache", ".cache", ".venv", "venv",
    ".claude", ".codex", ".qwen", ".agents", "node_modules", "third_party",
)
DUMP_CANDIDATES: Tuple[str, ...] = ("capture.wp02.bin", "capture.fi02.bin")
_WAIT_RE = re.compile(r"(?:wait|barrier|grid_sync|sync_wait|stall|idle|spin|poll)", re.I)
_INSTRUMENTED_SRC_RE = re.compile(
    r"(?:\b(?:WP|FI)_(?:OPEN|OPEN_BY|KERNEL_SCOPE|CLOSE)\s*\(|[\"<](?:wp|fi)_lite(?:_hip)?\.(?:cuh|h)[\">]"
    r"|\b(?:wp|fi)_open(?:_by)?\s*\()"
)
_TAG_ROLE_SWITCH = 0x04000000
_TAG_OPEN = 0x40000000
_HW_METRIC_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("tensor_active", re.compile(r"tensor\s*(?:pipe\s*)?active|tensor.*throughput", re.I)),
    ("sm_active", re.compile(r"\bsms?\s*active", re.I)),
    ("warp_issue", re.compile(r"\bsm\s*issue|\bissue\s*(?:active|slot)", re.I)),
    ("dram_read", re.compile(r"dram\s*read", re.I)),
    ("dram_write", re.compile(r"dram\s*write", re.I)),
    ("nvlink_rx_req", re.compile(r"nvlink\s*rx\s*requests?\s*user\s*data", re.I)),
    ("nvlink_rx_rsp", re.compile(r"nvlink\s*rx\s*responses?\s*user\s*data", re.I)),
    ("nvlink_tx_req", re.compile(r"nvlink\s*tx\s*requests?\s*user\s*data", re.I)),
    ("nvlink_tx_rsp", re.compile(r"nvlink\s*tx\s*responses?\s*user\s*data", re.I)),
    ("nvlink_user", re.compile(r"nvlink(?!.*protocol).*user\s*data", re.I)),
)

_WP_DECODE: Any = None


# ---------------------------------------------------------------------------
# Decoder loading
# ---------------------------------------------------------------------------

def load_wp_decode() -> Any:
    """Import ``weaveprof/pipeline/validate/tools/wp_decode.py`` once (the package path starts with a digit)."""
    global _WP_DECODE
    if _WP_DECODE is not None:
        return _WP_DECODE
    path = (ROOT / "weaveprof/pipeline/validate" / "tools" / "wp_decode.py").resolve()
    mod = sys.modules.get("wp_decode")
    if mod is None or Path(getattr(mod, "__file__", "") or "").resolve() != path:
        spec = importlib.util.spec_from_file_location("wp_decode", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load decoder {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["wp_decode"] = mod
        spec.loader.exec_module(mod)
    _WP_DECODE = mod
    return mod


# ---------------------------------------------------------------------------
# ctypes host controller for -DWP_DEFINE_HOST_API shared objects
# ---------------------------------------------------------------------------

class WpSoController:
    """ctypes controller for ``.so`` / ``.dll`` megakernels built with ``-DWP_DEFINE_HOST_API``.

    Exports (wp_lite.cuh): ``int wp_init(u32 n_ctas, u32 warps_per_cta, u32 cap)``,
    ``int wp_reset(u64 stream)``, ``int wp_dump(const char* path, const char* meta_json)``,
    ``void wp_free()``.  ``wp_dump`` returns the overflow bits (>= 0) or a negative error.
    """

    _SIGNATURES = (
        ("wp_init", ctypes.c_int, (ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)),
        ("wp_reset", ctypes.c_int, (ctypes.c_uint64,)),
        ("wp_dump", ctypes.c_int, (ctypes.c_char_p, ctypes.c_char_p)),
        ("wp_free", None, ()),
    )
    _DUMP_ERRORS = {-1: "not initialized or cudaDeviceSynchronize failed", -2: "device->host copy failed",
                    -3: "fopen failed (check dump directory permissions)", -4: "invalid buffer"}

    def __init__(self, so_path: str | os.PathLike | None, *, lib: Any = None):
        self.so_path = Path(so_path).resolve() if so_path else None
        if lib is None:
            if self.so_path is None or not self.so_path.is_file():
                raise FileNotFoundError(f"WeaveProfiler host-API library not found: {so_path}")
            lib = ctypes.CDLL(str(self.so_path))
        self._lib = lib
        for name, restype, argtypes in self._SIGNATURES:
            fn = getattr(lib, name, None)
            if fn is None:
                raise AttributeError(f"{so_path} does not export {name}; rebuild with -DWP_DEFINE_HOST_API")
            try:
                fn.restype = restype
                fn.argtypes = list(argtypes)
            except (AttributeError, TypeError):
                pass  # plain-Python fakes in tests
        self.geometry: Optional[Dict[str, int]] = None

    @property
    def initialized(self) -> bool:
        return self.geometry is not None

    @staticmethod
    def _stream_ptr(stream: Any) -> int:
        if stream is None:
            return 0
        if isinstance(stream, int):
            return int(stream)
        for attr in ("cuda_stream", "handle", "ptr", "value"):
            val = getattr(stream, attr, None)
            if isinstance(val, int):
                return int(val)
        raise TypeError(f"cannot derive a CUDA stream pointer from {type(stream).__name__}")

    def init(self, n_ctas: int, warps_per_cta: int, cap: int) -> "WpSoController":
        if min(int(n_ctas), int(warps_per_cta), int(cap)) <= 0:
            raise ValueError("n_ctas, warps_per_cta and cap must be positive")
        rc = int(self._lib.wp_init(int(n_ctas), int(warps_per_cta), int(cap)))
        if rc != 0:
            raise RuntimeError(f"wp_init({n_ctas}, {warps_per_cta}, {cap}) failed: cudaError={rc}")
        self.geometry = {"n_ctas": int(n_ctas), "warps_per_cta": int(warps_per_cta), "cap": int(cap)}
        return self

    def reset(self, stream: Any = 0) -> None:
        rc = int(self._lib.wp_reset(self._stream_ptr(stream)))
        if rc != 0:
            raise RuntimeError("wp_reset failed: " + ("wp_init not called" if rc == -1 else f"cudaError={rc}"))

    def dump(self, path: str | os.PathLike, meta: Mapping[str, Any] | None = None) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(dict(meta or {}), ensure_ascii=False, separators=(",", ":"))
        rc = int(self._lib.wp_dump(str(target).encode("utf-8"), meta_json.encode("utf-8")))
        if rc < 0:
            raise RuntimeError(f"wp_dump({target}) failed ({rc}): {self._DUMP_ERRORS.get(rc, 'unknown error')}")
        return rc

    def free(self) -> None:
        self._lib.wp_free()
        self.geometry = None

    def __enter__(self) -> "WpSoController":
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.initialized:
            self.free()


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _pct(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))])


def _r(x: Optional[float], nd: int = 3) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _union_len(intervals: Iterable[Tuple[float, float]], lo: float, hi: float) -> float:
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _dist(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"min_us": None, "median_us": None, "p90_us": None, "max_us": None, "mean_us": None,
                "spread_us": None, "imbalance_pct": None}
    mx, mean = max(vals), statistics.fmean(vals)
    return {
        "min_us": _r(min(vals)),
        "median_us": _r(statistics.median(vals)),
        "p90_us": _r(_pct(vals, 0.90)),
        "max_us": _r(mx),
        "mean_us": _r(mean),
        "spread_us": _r(mx - min(vals)),
        "imbalance_pct": _r(100.0 * (mx - mean) / mx, 2) if mx > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Megakernel skew analysis
# ---------------------------------------------------------------------------

def _trace_x_events(trace: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not trace:
        return []
    return [ev for ev in (trace.get("traceEvents") or [])
            if ev.get("ph") == "X" and ev.get("cat") != "info" and ev.get("dur") is not None]


def _is_wait(name: str, args: Mapping[str, Any]) -> bool:
    text = " ".join(str(x) for x in (name, args.get("semantic_id", ""), args.get("original_label", "")))
    return bool(_WAIT_RE.search(text))


def analyze_megakernel_skew(
    summary: Mapping[str, Any],
    trace: Mapping[str, Any] | None = None,
    *,
    total_sms: Optional[int] = None,
    idle_threshold: float = 0.15,
    spread_threshold: float = 0.20,
    role_skew_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Diagnose Sparse Attention / persistent megakernel load skew from a decoded ``wp`` trace.

    ``summary`` is the ``wp_decode.decode`` summary (needs ``lane_bounds_ns``, ``cta_sm``, ``wpc``);
    ``trace`` (optional) lets barrier / grid_sync wait spans be subtracted from SM busy time.
    All durations are microseconds.
    """
    sm_ = dict(summary or {})
    wpc = max(1, int(sm_.get("wpc") or 1))
    cta_sm = {int(k): int(v) for k, v in (sm_.get("cta_sm") or {}).items()}
    t0_ns = sm_.get("observation_start_ns")
    kernel_us = float(sm_.get("span_us") or 0.0)
    role_names: Dict[int, str] = {}
    if trace:
        for k, v in ((trace.get("otherData") or {}).get("roles") or {}).items():
            try:
                role_names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

    lanes: Dict[int, Dict[str, Any]] = {}
    for key, bounds in (sm_.get("lane_bounds_ns") or {}).items():
        slot = int(key)
        b, e = int(bounds[0]), int(bounds[1])
        lanes[slot] = {"start_ns": b, "end_ns": e, "role": int(bounds[2]) if len(bounds) > 2 else 0,
                       "cta": slot // wpc}
    if t0_ns is None:
        t0_ns = min((ln["start_ns"] for ln in lanes.values()), default=0)

    x_events = _trace_x_events(trace)
    wait_by_slot: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    stage_durs: Dict[str, List[float]] = defaultdict(list)
    for ev in x_events:
        args = ev.get("args") or {}
        name = str(ev.get("name") or args.get("semantic_id") or "stage")
        ts, dur = float(ev.get("ts") or 0.0), float(ev.get("dur") or 0.0)
        stage_durs[name].append(dur)
        if args.get("wave_slot") is not None and _is_wait(name, args):
            wait_by_slot[int(args["wave_slot"])].append((ts, ts + dur))

    # Per-lane lifetime, barrier-wait and work time (us relative to the dump's first record).
    per_sm: Dict[int, Dict[str, Any]] = {}
    for slot, ln in lanes.items():
        a = (ln["start_ns"] - t0_ns) / 1000.0
        b = (ln["end_ns"] - t0_ns) / 1000.0
        life = max(0.0, b - a)
        wait = _union_len(wait_by_slot.get(slot, ()), a, b)
        ln.update(start_us=a, end_us=b, life_us=life, wait_us=wait, work_us=max(0.0, life - wait))
        sm_id = cta_sm.get(ln["cta"], -1)
        rec = per_sm.setdefault(sm_id, {"sm": sm_id, "start_us": a, "end_us": b, "busy_us": 0.0,
                                         "wait_us": 0.0, "ctas": set(), "lanes": 0})
        rec["start_us"] = min(rec["start_us"], a)
        rec["end_us"] = max(rec["end_us"], b)
        rec["busy_us"] = max(rec["busy_us"], ln["work_us"])
        rec["wait_us"] = max(rec["wait_us"], wait)
        rec["ctas"].add(ln["cta"])
        rec["lanes"] += 1

    if not per_sm and x_events:  # legacy summaries without lane bounds: per-SM envelope from spans
        for ev in x_events:
            args = ev.get("args") or {}
            sm_id = int(args.get("sm", -1) if args.get("sm") is not None else -1)
            ts, te = float(ev.get("ts") or 0.0), float(ev.get("ts") or 0.0) + float(ev.get("dur") or 0.0)
            rec = per_sm.setdefault(sm_id, {"sm": sm_id, "start_us": ts, "end_us": te, "busy_us": 0.0,
                                             "wait_us": 0.0, "ctas": set(), "lanes": 0})
            rec["start_us"], rec["end_us"] = min(rec["start_us"], ts), max(rec["end_us"], te)
            rec["busy_us"] = rec["end_us"] - rec["start_us"]
            if args.get("cta") is not None:
                rec["ctas"].add(int(args["cta"]))
    if kernel_us <= 0 and per_sm:
        kernel_us = max(r["end_us"] for r in per_sm.values()) - min(r["start_us"] for r in per_sm.values())

    unknown_sm = per_sm.pop(-1, None)
    sm_ids = sorted(per_sm)
    busy = [per_sm[s]["busy_us"] for s in sm_ids]
    observed = len(sm_ids)
    total = max(int(total_sms or 0), observed)
    unused = max(0, total - observed)
    dist = _dist(busy)
    max_busy = max(busy) if busy else 0.0

    # Partition: natural break (largest gap >= 10% of max) else tail thresholds.
    split_mode, cut = "none", None
    if len(busy) >= 2 and max_busy > 0:
        srt = sorted(busy)
        gap, idx = max((srt[i + 1] - srt[i], i) for i in range(len(srt) - 1))
        if gap / max_busy >= 0.10:
            split_mode, cut = "bimodal_gap", (srt[idx] + srt[idx + 1]) / 2.0
        else:
            split_mode = "tail_threshold"
    if split_mode == "bimodal_gap":
        stragglers = [s for s in sm_ids if per_sm[s]["busy_us"] >= cut]
        early = [s for s in sm_ids if per_sm[s]["busy_us"] < cut]
    elif split_mode == "tail_threshold":
        stragglers = [s for s in sm_ids if per_sm[s]["busy_us"] >= 0.95 * max_busy]
        early = [s for s in sm_ids if per_sm[s]["busy_us"] <= (1.0 - idle_threshold) * max_busy]
    else:
        stragglers, early = list(sm_ids), []
    starved = [s for s in early if per_sm[s]["wait_us"] >= 0.10 * max(1e-9, per_sm[s]["end_us"] - per_sm[s]["start_us"])]

    def _group(ids: Sequence[int]) -> Dict[str, Any]:
        vals = [per_sm[s]["busy_us"] for s in ids]
        return {
            "count": len(ids),
            "busy_us_median": _r(statistics.median(vals)) if vals else None,
            "busy_us_min": _r(min(vals)) if vals else None,
            "busy_us_max": _r(max(vals)) if vals else None,
            "sm_ids": list(ids[:24]),
        }

    strag_g, early_g = _group(stragglers), _group(early)
    early_g["barrier_starved"] = len(starved)
    early_g["idle_tail_us_median"] = (_r(statistics.median([max_busy - per_sm[s]["busy_us"] for s in early]))
                                      if early else None)
    idle_sm_fraction = round((len(early) + unused) / total, 4) if total else 0.0
    sm_spread_ratio = round((max_busy - min(busy)) / max_busy, 4) if busy and max_busy > 0 else 0.0

    # Warp-role balance.
    role_cta_end: Dict[int, Dict[int, float]] = defaultdict(dict)
    role_work: Dict[int, List[float]] = defaultdict(list)
    for ln in lanes.values():
        r = ln["role"]
        role_cta_end[r][ln["cta"]] = max(role_cta_end[r].get(ln["cta"], 0.0), ln.get("end_us", 0.0))
        role_work[r].append(ln.get("work_us", 0.0))
    role_stats: Dict[str, Dict[str, Any]] = {}
    role_end_medians: List[float] = []
    max_role_tail = 0.0
    for r in sorted(role_cta_end):
        ends = list(role_cta_end[r].values())
        med = statistics.median(ends)
        tail = max(ends) - med
        role_end_medians.append(med)
        max_role_tail = max(max_role_tail, tail)
        role_stats[role_names.get(r, f"role{r}")] = {
            "lanes": len(role_work[r]),
            "ctas": len(ends),
            "end_median_us": _r(med),
            "end_max_us": _r(max(ends)),
            "tail_us": _r(tail),
            "work_median_us": _r(statistics.median(role_work[r])),
        }
    role_skew_ratio = (round((max(role_end_medians) - min(role_end_medians)) / max(role_end_medians), 4)
                       if len(role_end_medians) >= 2 and max(role_end_medians) > 0 else 0.0)
    role_switches = int(sm_.get("role_switches") or 0)

    # Stage breakdown (decoder summary events, falling back to trace spans).
    events = sm_.get("events") or {}
    stage_rows: List[Dict[str, Any]] = []
    if events:
        for name, st in events.items():
            stage_rows.append({"stage": name, "fires": int(st.get("n") or 0), "total_us": float(st.get("total_us") or 0.0),
                               "median_us": float(st.get("p50_us") or 0.0), "max_us": float(st.get("max_us") or 0.0)})
    else:
        for name, durs in stage_durs.items():
            stage_rows.append({"stage": name, "fires": len(durs), "total_us": sum(durs),
                               "median_us": statistics.median(durs), "max_us": max(durs)})
    total_stage = sum(r["total_us"] for r in stage_rows) or 1.0
    stages = []
    for row in sorted(stage_rows, key=lambda r: -r["total_us"]):
        stages.append({
            "stage": row["stage"],
            "share_pct": round(100.0 * row["total_us"] / total_stage, 2),
            "median_us": _r(row["median_us"]),
            "max_us": _r(row["max_us"]),
            "fires": row["fires"],
            "is_wait": bool(_WAIT_RE.search(row["stage"])),
        })
    wait_share = round(sum(s["share_pct"] for s in stages if s["is_wait"]) / 100.0, 4)

    # Concrete megakernel optimization hints.
    hints: List[Dict[str, Any]] = []
    mean_busy = statistics.fmean(busy) if busy else 0.0
    if busy and (idle_sm_fraction >= idle_threshold or sm_spread_ratio >= spread_threshold):
        hints.append({
            "code": "SPLIT_KV_OR_DYNAMIC_TILE_QUEUE",
            "severity": "high",
            "evidence": (f"{strag_g['count']} SMs @ {strag_g['busy_us_median']:.0f}us vs {early_g['count']} SMs @ "
                         f"{(early_g['busy_us_median'] or 0.0):.0f}us; idle_sm_fraction={idle_sm_fraction:.2f}, "
                         f"sm_spread_ratio={sm_spread_ratio:.2f}"),
            "action": ("Split long KV/TopK rows across CTAs (split-KV + combine) or replace static head/tile modulo "
                       "assignment with an atomic dynamic tile queue so early SMs steal work before grid_sync."),
            "upper_bound_gain_us": _r(max_busy - mean_busy),
        })
    if role_skew_ratio >= role_skew_threshold or (kernel_us > 0 and max_role_tail / kernel_us >= 0.20 and len(role_stats) >= 2):
        hints.append({
            "code": "WARP_GROUP_REBALANCE",
            "severity": "medium",
            "evidence": f"role_skew_ratio={role_skew_ratio:.2f}, max role CTA tail={max_role_tail:.1f}us",
            "action": ("Rebalance producer/consumer warp groups (setmaxnreg split, stage count, or move work from the "
                       "late role) so all warp roles finish together."),
        })
    if wait_share >= 0.20 or (observed and len(starved) / observed >= 0.10):
        hints.append({
            "code": "GRID_SYNC_WAIT_DOMINANT",
            "severity": "medium",
            "evidence": f"wait/barrier share={100.0 * wait_share:.1f}%, barrier-starved SMs={len(starved)}",
            "action": "Overlap the barrier with next-phase prefetch or replace grid_sync with per-tile ready flags.",
        })
    if total and unused / total >= 0.10:
        hints.append({
            "code": "FILL_IDLE_SMS",
            "severity": "medium",
            "evidence": f"{unused}/{total} SMs never received a CTA",
            "action": "Launch one persistent CTA per SM (grid = #SMs) and distribute tiles across all of them.",
        })
    top_compute = next((s for s in stages if not s["is_wait"]), None)
    if top_compute and top_compute["share_pct"] >= 50.0:
        hints.append({
            "code": "OPTIMIZE_DOMINANT_STAGE",
            "severity": "low",
            "evidence": f"'{top_compute['stage']}' = {top_compute['share_pct']:.1f}% of instrumented stage time",
            "action": f"Drill into '{top_compute['stage']}' with sub-stage probes before tuning other phases.",
        })

    return {
        "schema": SKEW_SCHEMA,
        "kernel_us": _r(kernel_us),
        "observed_sms": observed,
        "total_sms": total,
        "unused_sms": unused,
        "unknown_sm_lanes": int(unknown_sm["lanes"]) if unknown_sm else 0,
        "sm_busy": dist,
        "busy_basis": "lane_lifetime_minus_wait" if wait_by_slot else ("lane_lifetime" if lanes else "span_envelope"),
        "split_mode": split_mode,
        "split_cut_us": _r(cut),
        "stragglers": strag_g,
        "early_finishing": early_g,
        "idle_sm_fraction": idle_sm_fraction,
        "sm_spread_ratio": sm_spread_ratio,
        "counterfactual_balanced_us": _r(mean_busy),
        "role_stats": role_stats,
        "role_skew_ratio": role_skew_ratio,
        "max_role_tail_us": _r(max_role_tail),
        "dynamic_role_switches": role_switches,
        "stages": stages,
        "wait_share": wait_share,
        "hints": hints,
        "per_sm": [
            {"sm": s, "busy_us": _r(per_sm[s]["busy_us"]), "life_us": _r(per_sm[s]["end_us"] - per_sm[s]["start_us"]),
             "wait_us": _r(per_sm[s]["wait_us"]), "ctas": len(per_sm[s]["ctas"])}
            for s in sm_ids
        ],
    }


# ---------------------------------------------------------------------------
# Fast nsys SQLite reader
# ---------------------------------------------------------------------------

def _sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _pick_table(tables: Sequence[str], name: str) -> Optional[str]:
    if name in tables:
        return name
    return next((t for t in sorted(tables) if t.startswith(name)), None)


def _nsys_gpu_identity(conn: sqlite3.Connection, tables: Sequence[str], device_id: int) -> Tuple[Optional[str], Optional[str]]:
    """Marketing name and chip of the GPU that ran the selected kernel (nsys ``TARGET_INFO_GPU``).

    ``TARGET_INFO_GPU.id`` matches ``CUPTI_ACTIVITY_KIND_KERNEL.deviceId``. Returns ``(None, None)`` when the
    table or columns are missing (older reports, synthetic fixtures).
    """
    info = "TARGET_INFO_GPU"  # exact name: a prefix match would hit TARGET_INFO_GPU_METRICS
    if info not in tables:
        return None, None
    try:
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{info}")')}
        if "name" not in cols:
            return None, None
        chip_sel = "chipName" if "chipName" in cols else "NULL"
        row = None
        if "id" in cols:
            row = conn.execute(f'SELECT name, {chip_sel} FROM "{info}" WHERE id = ? LIMIT 1', (int(device_id),)).fetchone()
        if row is None:
            names = conn.execute(f'SELECT DISTINCT name, {chip_sel} FROM "{info}"').fetchall()
            row = names[0] if len(names) == 1 else None  # ambiguous mixed-GPU host: do not guess
        if row is None:
            return None, None
        return (str(row[0]) if row[0] else None), (str(row[1]) if row[1] else None)
    except sqlite3.Error:
        return None, None


# nsys normalizes "Tensor Active [Throughput %]" to the full GH100 die Tensor rate; H20 has 1/4 of it per SM,
# so a saturated H20 reads ~25%. The factor and the raw ceiling come from causal.ARCH_PROFILES (single source).
_H20_NAME_RE = re.compile(r"(?i)\bH20\b")


def resolve_tensor_active_scale(
    hw: Mapping[str, Any],
    *,
    arch: Optional[str] = None,
    card: Optional[str] = None,
) -> Dict[str, Any]:
    """Decide the Tensor Active de-normalization factor for one nsys capture.

    Same rule as ``causal.calibrate_hw_series`` (factor ``tensor_active_scale``, applied only while the raw
    in-window maximum stays at or below ``tensor_raw_uncalibrated_ceiling``), but the card is identified from the
    GPU name nsys recorded for the profiled device whenever it is available. ``arch`` alone is never enough:
    ``sm_90`` / ``sm_90a`` is also H100 / H200, whose Tensor Active is already relative to its own peak.
    Without a recorded GPU name, only an explicit ``card`` naming H20 enables the correction.
    """
    curve = (hw.get("hw_series") or {}).get("tensor_active") or {}
    dur = float((hw.get("kernel") or {}).get("duration_us") or 0.0)
    pts = list(zip(curve.get("t_us") or [], curve.get("val") or []))
    in_win = [float(v) for t, v in pts if dur <= 0.0 or 0.0 <= float(t) <= dur] or [float(v) for _, v in pts]
    gpu_name = hw.get("gpu_name")
    card_s = str(card or "").strip()
    if gpu_name:
        source = f"nsys TARGET_INFO_GPU name {gpu_name!r}"
        is_h20 = bool(_H20_NAME_RE.search(str(gpu_name)))
    else:
        source = f"card={card_s or 'auto'!r} (no GPU name in the nsys report)"
        is_h20 = bool(_H20_NAME_RE.search(card_s))
    prof = get_arch_profile("sm_90a") if is_h20 else get_arch_profile(None, None)
    factor = float(prof.get("tensor_active_scale") or 1.0)
    ceiling = float(prof.get("tensor_raw_uncalibrated_ceiling") or 101.0)
    info: Dict[str, Any] = {
        "scale": 1.0,
        "applied": False,
        "card_factor": factor,
        "source": source,
        "raw_mean_pct": round(statistics.fmean(in_win), 3) if in_win else None,
        "raw_max_pct": max(in_win) if in_win else None,
        "raw_ceiling_pct": ceiling,
    }
    if not in_win:
        info["reason"] = "no Tensor Active samples"
    elif factor <= 1.0:
        info["reason"] = "card is not die-normalized (no correction needed)"
    elif max(in_win) > ceiling:
        info["reason"] = f"raw max {max(in_win):g}% exceeds the {ceiling:g}% die-normalization ceiling; left raw"
    else:
        info.update(scale=factor, applied=True,
                    reason=("nsys normalizes Tensor Active to the full GH100 die; H20 Tensor throughput per SM is "
                            f"1/{factor:g} of it, so values are multiplied by {factor:g} (causal.ARCH_PROFILES)"))
    return info


def _tensor_plateau_suspect(hw: Mapping[str, Any], scale_info: Mapping[str, Any]) -> Optional[str]:
    """Advisory: an uncorrected Tensor Active pinned near 100/2^k while SMs Active reaches 100 (die normalization)."""
    if scale_info.get("applied"):
        return None
    series = hw.get("hw_series") or {}
    dur = float((hw.get("kernel") or {}).get("duration_us") or 0.0)

    def _in_win(key: str) -> List[float]:
        c = series.get(key) or {}
        return [float(v) for t, v in zip(c.get("t_us") or [], c.get("val") or []) if dur <= 0.0 or 0.0 <= float(t) <= dur]

    tv, sv = _in_win("tensor_active"), _in_win("sm_active")
    if len(tv) < 10 or not sv or max(sv) < 99.0:
        return None
    vmax = max(tv)
    for plateau in (25.0, 50.0):
        if abs(vmax - plateau) <= 1.5 and sum(1 for v in tv if v >= vmax - 1.0) >= 0.2 * len(tv):
            return (f"TENSOR_ACTIVE_PLATEAU_UNCORRECTED (advisory): Tensor Active pinned at ~{plateau:g}% while SMs "
                    f"Active reaches 100% and no die-normalization factor was applied ({scale_info.get('source')})")
    return None


def extract_nsys_sqlite_fast(
    sqlite_path: Path | str,
    kernel_substr: str = "",
    *,
    device: Optional[int] = None,
    pad_samples: int = 0,
) -> Dict[str, Any]:
    """Extract the warm kernel window and in-window ``GPU_METRICS`` curves straight from nsys SQLite.

    Returns ``{"kernel": {start_ns, end_ns, duration_us, ...}, "hw_series": {tensor_active, sm_active,
    warp_issue, dram_read, dram_write}, "in_window_samples", ...}``; ``hw_series`` uses ``t_us``
    relative to kernel start, ready for ``analyze_sw_hw_causality``.
    """
    t0 = time.perf_counter()
    path = Path(sqlite_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"nsys sqlite not found: {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        tables = _sqlite_tables(conn)
        ktable = _pick_table(tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if ktable is None:
            raise ValueError(f"{path.name}: no CUPTI_ACTIVITY_KIND_KERNEL table")
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{ktable}")')}
        name_col = next((c for c in ("demangledName", "shortName", "mangledName") if c in cols), None)
        dev_sql = " AND k.deviceId = ?" if (device is not None and "deviceId" in cols) else ""
        params: List[Any] = []
        if name_col and "StringIds" in tables:
            sql = (f'SELECT k.start, k.end, {"k.deviceId" if "deviceId" in cols else "0"}, s.value '
                   f'FROM "{ktable}" k LEFT JOIN StringIds s ON s.id = k."{name_col}" WHERE COALESCE(s.value, \'\') LIKE ?')
            params.append(f"%{kernel_substr}%")
        else:
            sql = f'SELECT k.start, k.end, {"k.deviceId" if "deviceId" in cols else "0"}, NULL FROM "{ktable}" k WHERE 1=1'
        if dev_sql:
            sql += dev_sql
            params.append(int(device))
        sql += " ORDER BY k.start"
        launches = [(int(a), int(b), int(d or 0), n) for a, b, d, n in conn.execute(sql, params) if b is not None and b > a]
        if not launches:
            raise ValueError(f"{path.name}: no kernel launch matches {kernel_substr!r}")
        if not kernel_substr.strip() and any(l[3] for l in launches):
            by_kname: Dict[str, List[Tuple[int, int, int, Any]]] = defaultdict(list)
            for l in launches:
                by_kname[str(l[3] or "")].append(l)
            best_kname = max(by_kname, key=lambda k: statistics.median([x[1] - x[0] for x in by_kname[k]]))
            launches = by_kname[best_kname]

        gcols = {r[1] for r in conn.execute('PRAGMA table_info("GPU_METRICS")')} if "GPU_METRICS" in tables else set()
        has_type_id = "typeId" in gcols
        # 1. Parse tiny TARGET_INFO_GPU_METRICS first (<1 ms) to identify which typeId(s) carry GPU HW metrics.
        wanted_by_type: Dict[Optional[int], Dict[int, Tuple[str, str]]] = {}
        info = _pick_table(tables, "TARGET_INFO_GPU_METRICS")
        if info is not None and "GPU_METRICS" in tables:
            icols = {r[1] for r in conn.execute(f'PRAGMA table_info("{info}")')}
            q_info = f'SELECT {"typeId" if "typeId" in icols else "NULL"}, metricId, metricName FROM "{info}"'
            for tid_raw, mid, mname in conn.execute(q_info):
                tid_key: Optional[int] = int(tid_raw) if (has_type_id and tid_raw is not None) else None
                mname_s = str(mname)
                bucket = wanted_by_type.setdefault(tid_key, {})
                for key, rx in _HW_METRIC_PATTERNS:
                    if rx.search(mname_s) and key not in {k for k, _ in bucket.values()}:
                        bucket[int(mid)] = (key, mname_s)
                        break
            wanted_by_type = {k: v for k, v in wanted_by_type.items() if v}

        gm_min_rid = gm_max_rid = 0
        if "GPU_METRICS" in tables:
            rr = conn.execute("SELECT MIN(rowid), MAX(rowid) FROM GPU_METRICS").fetchone()
            if rr and rr[0] is not None and rr[1] is not None:
                gm_min_rid, gm_max_rid = int(rr[0]), int(rr[1])

        def _row_at(rid: int) -> Optional[Tuple[Optional[int], int]]:
            if has_type_id:
                row = conn.execute("SELECT typeId, timestamp FROM GPU_METRICS WHERE rowid >= ? LIMIT 1", (rid,)).fetchone()
                return (int(row[0]), int(row[1])) if row else None
            row = conn.execute("SELECT timestamp FROM GPU_METRICS WHERE rowid >= ? LIMIT 1", (rid,)).fetchone()
            return (None, int(row[0])) if row else None

        def _find_type_segments(target_tid: Optional[int]) -> List[Tuple[int, int, bool, Optional[int], Optional[int]]]:
            """Return [(rid_lo, rid_hi, is_monotonic, t_min, t_max)] for target_tid in O(log N)."""
            if gm_max_rid < gm_min_rid or gm_min_rid <= 0:
                return []
            span = gm_max_rid - gm_min_rid + 1
            if span <= 50_000 or target_tid is None:
                pts = [_row_at(gm_min_rid + (span - 1) * k // 4) for k in range(5)]
                mono = (all(p is not None and (target_tid is None or p[0] == target_tid) for p in pts)
                        and all(pts[i][1] <= pts[i + 1][1] for i in range(4)))
                return [(gm_min_rid, gm_max_rid, mono, pts[0][1] if mono else None, pts[-1][1] if mono else None)]
            n_probes = 128
            probes: List[Tuple[int, Optional[int], int]] = []
            for i in range(n_probes + 1):
                rid = gm_min_rid + (gm_max_rid - gm_min_rid) * i // n_probes
                r = _row_at(rid)
                if r is not None:
                    probes.append((rid, r[0], r[1]))
            runs: List[Tuple[int, int]] = []
            i = 0
            while i < len(probes):
                if probes[i][1] != target_tid:
                    i += 1
                    continue
                j = i
                while j + 1 < len(probes) and probes[j + 1][1] == target_tid:
                    j += 1
                # Refine left edge between probes[i-1] (different tid) and probes[i] (target_tid)
                lo_edge = probes[i][0]
                if i > 0:
                    a, b = probes[i - 1][0], probes[i][0]
                    while a < b:
                        mid = (a + b) // 2
                        r = _row_at(mid)
                        if r is not None and r[0] == target_tid:
                            b = mid
                        else:
                            a = mid + 1
                    lo_edge = a
                # Refine right edge between probes[j] (target_tid) and probes[j+1] (different tid)
                hi_edge = probes[j][0]
                if j + 1 < len(probes):
                    a, b = probes[j][0], probes[j + 1][0]
                    while a < b:
                        mid = (a + b + 1) // 2
                        r = _row_at(mid)
                        if r is not None and r[0] == target_tid:
                            a = mid
                        else:
                            b = mid - 1
                    hi_edge = a
                runs.append((lo_edge, hi_edge))
                i = j + 1
            if not runs:
                return [(gm_min_rid, gm_max_rid, False, None, None)]
            segs: List[Tuple[int, int, bool, Optional[int], Optional[int]]] = []
            for r_lo, r_hi in runs:
                pts = [_row_at(r_lo + (r_hi - r_lo) * k // 4) for k in range(5)]
                mono = (all(p is not None and p[0] == target_tid for p in pts)
                        and all(pts[k][1] <= pts[k + 1][1] for k in range(4)))
                segs.append((r_lo, r_hi, mono, pts[0][1] if mono else None, pts[-1][1] if mono else None))
            return segs

        def _bsearch_ts_in_seg(seg_lo: int, seg_hi: int, t_lo: int, t_hi: int) -> Tuple[int, int]:
            lo, hi = seg_lo, seg_hi
            while lo < hi:
                mid = (lo + hi) // 2
                row = conn.execute("SELECT timestamp FROM GPU_METRICS WHERE rowid >= ? LIMIT 1", (mid,)).fetchone()
                if row is None or row[0] < t_lo:
                    lo = mid + 1
                else:
                    hi = mid
            rid_lo = max(seg_lo, lo - 1024)
            lo, hi = rid_lo, seg_hi
            while lo < hi:
                mid = (lo + hi + 1) // 2
                row = conn.execute("SELECT timestamp FROM GPU_METRICS WHERE rowid <= ? ORDER BY rowid DESC LIMIT 1", (mid,)).fetchone()
                if row is None or row[0] > t_hi:
                    hi = mid - 1
                else:
                    lo = mid
            rid_hi = min(seg_hi, lo + 1024)
            return rid_lo, rid_hi

        segs_by_type: Dict[Optional[int], List[Tuple[int, int, bool, Optional[int], Optional[int]]]] = {
            tid: _find_type_segments(tid) for tid in wanted_by_type
        }
        pmu_bounds = [(t0_s, t1_s) for segs in segs_by_type.values() for _, _, mono, t0_s, t1_s in segs if mono and t0_s is not None and t1_s is not None]

        # Prefer launches overlapping the cudaProfilerApi GPU_METRICS capture window when present.
        candidates = launches[1:] if len(launches) > 1 else launches
        if pmu_bounds:
            in_pmu = [l for l in candidates if any(l[1] >= t0_s and l[0] <= t1_s for t0_s, t1_s in pmu_bounds)]
            if not in_pmu:
                in_pmu = [l for l in launches if any(l[1] >= t0_s and l[0] <= t1_s for t0_s, t1_s in pmu_bounds)]
            if in_pmu:
                candidates = in_pmu
        durs = sorted(b - a for a, b, _, _ in candidates)
        med = durs[(len(durs) - 1) // 2]
        start_ns, end_ns, dev_id, kname = max((l for l in candidates if l[1] - l[0] == med), key=lambda l: l[0])
        gpu_name, gpu_chip = _nsys_gpu_identity(conn, tables, dev_id)

        hw_series: Dict[str, Dict[str, Any]] = {}
        interval_us: Optional[float] = None
        in_window = 0
        if "GPU_METRICS" in tables and wanted_by_type and gm_max_rid >= gm_min_rid > 0:
            pad = int(pad_samples) * 5000
            best_score: Tuple[int, float, int] = (-1, -1.0, -1)
            for type_id, wanted in wanted_by_type.items():
                cand_series: Dict[str, Dict[str, Any]] = {}
                marks = ",".join("?" for _ in wanted)
                for seg_lo, seg_hi, mono, t0_s, t1_s in segs_by_type.get(type_id, ()):
                    if mono and t0_s is not None and t1_s is not None:
                        if end_ns + pad < t0_s or start_ns - pad > t1_s:
                            continue
                        rid_lo, rid_hi = _bsearch_ts_in_seg(seg_lo, seg_hi, start_ns - pad, end_ns + pad)
                    else:
                        rid_lo, rid_hi = seg_lo, seg_hi
                    if rid_hi < rid_lo:
                        continue
                    q = (f"SELECT metricId, timestamp, value FROM GPU_METRICS WHERE rowid BETWEEN ? AND ? "
                         f"AND metricId IN ({marks}) AND timestamp BETWEEN ? AND ?"
                         + (" AND typeId = ?" if (has_type_id and type_id is not None) else "")
                         + " ORDER BY timestamp")
                    args: List[Any] = [rid_lo, rid_hi, *wanted, start_ns - pad, end_ns + pad] + (
                        [type_id] if (has_type_id and type_id is not None) else []
                    )
                    for mid, ts, val in conn.execute(q, args):
                        key, mname = wanted[int(mid)]
                        curve = cand_series.setdefault(key, {"t_us": [], "val": [], "metric_name": mname})
                        curve["t_us"].append(round((int(ts) - start_ns) / 1000.0, 3))
                        curve["val"].append(float(val))
                if not cand_series:
                    continue
                cand_in_win = 0
                cand_interval: Optional[float] = None
                activity_sum = 0.0
                for key, curve in cand_series.items():
                    curve["in_win_count"] = sum(1 for t in curve["t_us"] if 0.0 <= t <= (end_ns - start_ns) / 1000.0)
                    cand_in_win = max(cand_in_win, curve["in_win_count"])
                    if key in ("sm_active", "tensor_active") and curve["val"]:
                        activity_sum += statistics.fmean(curve["val"])
                    if len(curve["t_us"]) >= 2:
                        steps = [b - a for a, b in zip(curve["t_us"], curve["t_us"][1:]) if b > a]
                        if steps:
                            step = statistics.median(steps)
                            cand_interval = step if cand_interval is None else min(cand_interval, step)
                score = (1 if cand_in_win > 0 else 0, activity_sum, cand_in_win)
                if score > best_score:
                    best_score = score
                    hw_series = cand_series
                    in_window = cand_in_win
                    interval_us = cand_interval
    finally:
        conn.close()
    return {
        "schema": NSYS_FAST_SCHEMA,
        "sqlite": str(path),
        "kernel": {
            "name": kname,
            "device": dev_id,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_us": round((end_ns - start_ns) / 1000.0, 3),
            "launches": len(launches),
            "warm_selection": "median_of_warm_launches" if len(launches) > 1 else "single_launch",
        },
        "gpu_name": gpu_name,
        "gpu_chip": gpu_chip,
        "hw_series": hw_series,
        "in_window_samples": in_window,
        "sample_interval_us": _r(interval_us),
        "sample_freq_hz": round(1e6 / interval_us, 1) if interval_us else None,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


# ---------------------------------------------------------------------------
# Fidelity gates
# ---------------------------------------------------------------------------

def _load_mapping(value: Mapping[str, Any] | str | Path | None, what: str) -> Tuple[Dict[str, Any], Optional[Path]]:
    if value is None:
        return {}, None
    if isinstance(value, Mapping):
        return dict(value), None
    p = Path(value).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{what} not found: {p}")
    doc = read_json(p)
    if not isinstance(doc, dict):
        raise ValueError(f"{what} must be a JSON object: {p}")
    return doc, p


def _gate_output(rm: Mapping[str, Any], base: Path, *, require_files: bool = False) -> Dict[str, Any]:
    pairs = rm.get("output_files") or rm.get("outputs")
    if isinstance(pairs, Mapping):
        pairs = [[pairs.get("baseline"), pairs.get("instrumented")]]
    if isinstance(pairs, list) and pairs:
        mismatched: List[str] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(pair):
                return {"ok": False, "code": "OUTPUT_EVIDENCE_MALFORMED", "evidence": "output_files"}
            a, b = (Path(x) if Path(x).is_absolute() else base / x for x in pair)
            if (not a.is_file() or not b.is_file() or a.resolve() == b.resolve()
                    or a.stat().st_size == 0 or digest(a) != digest(b)):
                mismatched.append(b.name)
        return {"ok": not mismatched, "evidence": "output_files", "mismatched": mismatched[:4]}
    if require_files:
        return {"ok": False, "code": "OUTPUT_FILES_REQUIRED",
                "evidence": "policy requires real baseline/instrumented output_files byte comparison"}
    if rm.get("baseline_output_sha256") and rm.get("instrumented_output_sha256"):
        ok = str(rm["baseline_output_sha256"]) == str(rm["instrumented_output_sha256"])
        return {"ok": ok, "evidence": "output_sha256"}
    if "output_match" in rm:
        return {"ok": rm["output_match"] is True, "evidence": "runtime_metrics.output_match"}
    return {"ok": False, "code": "OUTPUT_EVIDENCE_MISSING",
            "evidence": "runtime_metrics has no output_files / output_sha256 / output_match"}


def _gate_compiler(rm: Mapping[str, Any], pol: HarnessPolicy) -> Dict[str, Any]:
    l0, inst = rm.get("l0_stats"), rm.get("inst_stats")
    if not isinstance(l0, Mapping) or not isinstance(inst, Mapping):
        return {"ok": False, "violations": [{"code": "COMPILER_STATS_MISSING",
                                             "message": "runtime_metrics needs l0_stats and inst_stats"}],
                "register_count_blind_spot": False, "remediation": [], "metrics": {}}
    return verify_compiler_spill_gate(
        l0, inst,
        max_delta_spill_bytes=pol.max_delta_spill_bytes,
        max_delta_stack_bytes=pol.max_delta_stack_bytes,
        max_delta_stl=pol.max_delta_stl,
        max_delta_ldl=pol.max_delta_ldl,
        has_setmaxnreg=bool(rm.get("has_setmaxnreg")),
        smem_free_bytes=rm.get("smem_free_bytes"),
    )


def _gate_events(summary: Mapping[str, Any], trace: Mapping[str, Any], rm: Mapping[str, Any], pol: HarnessPolicy) -> Dict[str, Any]:
    stage_ns: Dict[str, List[float]] = defaultdict(list)
    sass_body = rm.get("sass_body_insns")
    for ev in _trace_x_events(trace):
        args = ev.get("args") or {}
        name = str(ev.get("name") or "stage")
        if sass_body is None and _is_wait(name, args):
            continue
        key = str(args.get("original_label") or name)
        stage_ns[key].append(float(ev.get("dur") or 0.0) * 1000.0)
    stub = verify_semantic_body_and_stubs(stage_ns, sass_body_insns=sass_body)
    gran_gate = (
        {"ok": True, "code": None, "skipped": True}
        if pol.allow_single_stage_collapse
        else verify_stage_granularity(trace)
    )
    warp_gate = (
        {"ok": True, "code": None, "skipped": True}
        if pol.allow_partial_warps
        else verify_warp_completeness(summary, trace, expected_warps=rm.get("expected_warps_per_cta"))
    )
    overflow = int(summary.get("overflow") or 0)
    pairing = int(sum(int(v) for v in (summary.get("errors") or {}).values()))
    spans = int(summary.get("spans") or 0)
    return {
        "ok": (
            bool(stub["ok"]) and bool(gran_gate["ok"]) and bool(warp_gate["ok"])
            and overflow == 0 and pairing == 0 and spans > 0
        ),
        "stub_gate": stub,
        "granularity_gate": gran_gate,
        "warp_gate": warp_gate,
        "overflow": overflow,
        "pairing_errors": pairing,
        "spans": spans,
        "decoder_errors": dict(list((summary.get("errors") or {}).items())[:6]),
    }


def _gate_overhead(guard: PolicyGuard, rm: Mapping[str, Any], limit: float) -> Dict[str, Any]:
    proto_ok, abba, errors = guard.validate_runtime_protocol(rm, limit)
    ratio = abba.get("median_overhead_ratio") if abba else None
    lim = float(limit)
    ok = bool(proto_ok and ratio is not None and math.isfinite(float(ratio)) and abs(float(ratio)) <= lim + 1e-12)
    if ratio is None and proto_ok:
        errors = list(errors) + ["no valid paired baseline_ns/instrumented_ns samples"]
    elif ratio is not None and float(ratio) > lim + 1e-12:
        errors = list(errors) + [f"warm overhead {100.0 * float(ratio):.2f}% > limit +{100.0 * lim:.2f}%"]
    elif ratio is not None and float(ratio) < -(lim + 1e-12):
        errors = list(errors) + [
            f"NEGATIVE_OVERHEAD_PERTURBATION: warm ratio {100.0 * float(ratio):.2f}% < -{100.0 * lim:.2f}% "
            "(instrumentation altered compiler unrolling/scheduling relative to the unmodified baseline)"
        ]
    return {"ok": ok, "overhead_ratio": ratio, "limit": lim, "abba": abba, "errors": list(errors)}


# ---------------------------------------------------------------------------
# Workflow A: shadow L0 / L_inst build + KREX-locked bench
# ---------------------------------------------------------------------------

def _cache_base() -> Path:
    configured = os.environ.get("WEAVEPROF_CACHE")
    if configured:
        return Path(configured).resolve()
    # Keep ephemeral shadow copies / L0 build caches on local /tmp (high-inode FS) rather than
    # network-mounted $HOME (e.g. JuiceFS with tight inode quotas).
    if os.name != "nt":
        uid = os.getuid() if hasattr(os, "getuid") else 0
        return (Path(tempfile.gettempdir()) / f"weaveprof-cache-{uid}").resolve()
    return (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "weaveprof").resolve()


def _ignore_factory(extra_abs: Sequence[Path]):
    excluded = {p.resolve() for p in extra_abs}

    def _ignore(directory: str, names: List[str]) -> List[str]:
        out = []
        dpath = Path(directory).resolve()
        for n in names:
            if any(fnmatch.fnmatch(n, pat) for pat in SHADOW_IGNORE) or (dpath / n).resolve() in excluded:
                out.append(n)
        return out

    return _ignore


def _tree_snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
    snap: Dict[str, Tuple[int, int]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            snap[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return snap


def _tree_digest(root: Path, salt: Any) -> str:
    h = hashlib.sha256(json.dumps(salt, sort_keys=True, default=str).encode("utf-8"))
    for rel in sorted(_tree_snapshot(root)):
        h.update(rel.encode("utf-8") + b"\0" + digest(root / rel).encode("ascii") + b"\n")
    return h.hexdigest()


def _argv(cmd: Sequence[str] | str, subs: Mapping[str, str]) -> List[str]:
    tokens = shlex.split(cmd, posix=(os.name != "nt")) if isinstance(cmd, str) else [str(t) for t in cmd]
    out: List[str] = []
    for tok in tokens:
        for key, val in subs.items():
            tok = tok.replace("{" + key + "}", val)
        out.append(tok)
    return out


def _shadow_env(l0: Path, inst: Path, out: Path, l0_cache: Optional[Path]) -> Dict[str, str]:
    libs = ROOT / "recorders"
    env = dict(os.environ)
    env.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
    env.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
    env.update(
        WEAVEPROF_ROOT=str(ROOT), WEAVEPROF_L0_DIR=str(l0), WEAVEPROF_INST_DIR=str(inst),
        WEAVEPROF_OUT_DIR=str(out), PYTHONDONTWRITEBYTECODE="1",
    )
    if os.name != "nt":
        jit_tmp = _cache_base() / "jit"
        env.setdefault("TRITON_CACHE_DIR", str(jit_tmp / "triton"))
        env.setdefault("TORCH_EXTENSIONS_DIR", str(jit_tmp / "torch_ext"))
    if l0_cache is not None:
        env["WEAVEPROF_L0_CACHE_DIR"] = str(l0_cache)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(libs / "triton"), str(libs / "cutedsl"),
                                         *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))])
    env["CPATH"] = os.pathsep.join([str(libs / "cuda_cpp"), str(libs / "hip_cpp"),
                                    *filter(None, env.get("CPATH", "").split(os.pathsep))])
    return env


def _run_shadow_workflow(
    ws: Path,
    *,
    bench_cmd: Sequence[str] | str,
    build_cmd: Sequence[str] | str | None,
    inst_files: Mapping[str, str] | None,
    inst_dir: Path | None,
    out: Path,
    gpu_id: str,
    use_l0_cache: bool,
    policy: HarnessPolicy,
    timeout_s: float,
) -> Dict[str, Any]:
    shadow = out / "shadow"
    if shadow.exists():
        shutil.rmtree(shadow)
    l0, inst = shadow / "L0", shadow / "L_inst"
    ignore = _ignore_factory([out] if out.resolve().is_relative_to(ws) else [])
    shutil.copytree(ws, l0, ignore=ignore)
    shutil.copytree(ws, inst, ignore=ignore)
    applied: List[str] = []
    for rel, text in (inst_files or {}).items():
        target = (inst / rel).resolve()
        if not target.is_relative_to(inst.resolve()):
            raise PolicyViolation("INST_FILE_ESCAPE", f"inst_files path escapes the shadow tree: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(str(text).encode("utf-8"))
        applied.append(Path(rel).as_posix())
    if inst_dir is not None:
        for src in sorted(Path(inst_dir).rglob("*")):
            if src.is_file() and not any(fnmatch.fnmatch(src.name, p) for p in SHADOW_IGNORE):
                rel = src.relative_to(inst_dir)
                (inst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, inst / rel)
                applied.append(rel.as_posix())
    bench_out = out / "bench"
    bench_out.mkdir(parents=True, exist_ok=True)
    l0_info: Dict[str, Any] = {"enabled": bool(use_l0_cache), "hit": False}
    l0_cache_dir: Optional[Path] = None
    if use_l0_cache:
        key = _tree_digest(l0, {"build_cmd": build_cmd if build_cmd is None or isinstance(build_cmd, str) else list(build_cmd)})
        l0_cache_dir = _cache_base() / "mk_l0_cache" / key
        l0_info["key"] = key[:16]
    env_l0 = _shadow_env(l0, inst, bench_out, l0_cache_dir)
    env_l0.update(WP_LEVEL="0", FI_LEVEL="0")
    env_inst = _shadow_env(l0, inst, bench_out, l0_cache_dir)
    builds: Dict[str, Any] = {}
    if build_cmd:
        manifest = l0_cache_dir / "products.json" if l0_cache_dir else None
        procs: Dict[str, Tuple[subprocess.Popen, Any]] = {}
        before = None
        if manifest is not None and manifest.is_file():
            try:
                mdoc = read_json(manifest)
                cached_files = mdoc.get("files") or []
                cached_shas = mdoc.get("sha256") or {}
                valid_cache = bool(cached_files) and isinstance(cached_shas, Mapping)
                if valid_cache:
                    for rel in cached_files:
                        src = l0_cache_dir / "products" / rel
                        if not src.is_file() or cached_shas.get(rel) != digest(src):
                            valid_cache = False
                            break
                if valid_cache:
                    for rel in cached_files:
                        src = l0_cache_dir / "products" / rel
                        (l0 / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, l0 / rel)
                    l0_info["hit"] = True
                else:
                    before = _tree_snapshot(l0)
            except (OSError, ValueError):
                before = _tree_snapshot(l0)
        else:
            before = _tree_snapshot(l0)
        for tag, tree, env in (("L0", l0, env_l0), ("L_inst", inst, env_inst)):
            if tag == "L0" and l0_info["hit"]:
                continue
            log = open(out / f"build_{tag}.log", "wb")
            subs = {"l0": str(l0), "inst": str(inst), "out": str(bench_out), "tree": str(tree)}
            procs[tag] = (subprocess.Popen(_argv(build_cmd, subs), cwd=str(tree), env=env, stdout=log,
                                           stderr=subprocess.STDOUT), log)
        for tag, (proc, log) in procs.items():  # parallel off-region CPU builds
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -9
            log.close()
            builds[tag] = rc
        if any(rc != 0 for rc in builds.values()):
            raise RuntimeError(f"pre-build failed: {builds} (see {out}/build_*.log)")
        if before is not None and l0_cache_dir is not None:
            after = _tree_snapshot(l0)
            products = sorted(rel for rel, sig in after.items() if before.get(rel) != sig)
            shas: Dict[str, str] = {}
            for rel in products:
                dst = l0_cache_dir / "products" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(l0 / rel, dst)
                shas[rel] = digest(dst)
            write_json(l0_cache_dir / "products.json", {"files": products, "sha256": shas, "created_at": time.time()})
            l0_info["stored_products"] = len(products)
    subs = {"l0": str(l0), "inst": str(inst), "out": str(bench_out), "tree": str(inst)}
    argv = _argv(bench_cmd, subs)
    with gpu_critical_region(gpu_id, timeout_s=policy.region_lock_timeout_s, label="megakernel_audit") as lease:
        with open(out / "bench.log", "wb") as log:
            try:
                proc = subprocess.run(argv, cwd=str(inst), env=env_inst, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=timeout_s)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = -9
    region_lock = lease.to_dict()
    if rc != 0:
        raise RuntimeError(f"bench_cmd exited {rc} (see {out / 'bench.log'})")

    def _find(names: Sequence[str], globs: Sequence[str] = ()) -> Optional[Path]:
        for root in (bench_out, inst):
            for n in names:
                if (root / n).is_file():
                    return root / n
            for g in globs:
                hits = sorted(root.glob(g))
                if hits:
                    return hits[0]
        return None

    return {
        "shadow": str(shadow),
        "l0_dir": l0,
        "inst_dir": inst,
        "l0_cache_dir": l0_cache_dir,
        "applied_inst_files": applied,
        "dump_path": _find(DUMP_CANDIDATES, ("*.wp02.bin", "*.fi02.bin")),
        "runtime_path": _find(("runtime.json",)),
        "names_path": _find(("names.json",)),
        "region_lock": region_lock,
        "l0_cache": l0_info,
        "builds": builds,
    }


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------

def _resolve_total_sms(total_sms: Optional[int], card: str, arch: Optional[str], summary_arch: Optional[str]) -> Optional[int]:
    if total_sms:
        return int(total_sms)
    gpu = "" if str(card or "auto").lower() == "auto" else str(card)
    a = arch or summary_arch
    if gpu or a:
        prof = get_arch_profile(a, gpu)
        if prof.get("default_sms"):
            return int(prof["default_sms"])
    return None


def _per_sm_w0(summary: Mapping[str, Any], trace: Mapping[str, Any], cap_spans: int = 512) -> List[Dict[str, Any]]:
    wpc = max(1, int(summary.get("wpc") or 1))
    cta_sm = {int(k): int(v) for k, v in (summary.get("cta_sm") or {}).items()}
    by_slot: Dict[int, List[List[Any]]] = defaultdict(list)
    for ev in _trace_x_events(trace):
        args = ev.get("args") or {}
        if args.get("wave_slot") is None:
            continue
        ts = float(ev.get("ts") or 0.0)
        by_slot[int(args["wave_slot"])].append([str(ev.get("name") or "stage"), ts, ts + float(ev.get("dur") or 0.0)])
    chosen: Dict[int, int] = {}
    for slot in sorted(by_slot):
        chosen.setdefault(slot // wpc, slot)
    return [{"cta": cta, "smid": cta_sm.get(cta, -1), "spans": sorted(by_slot[slot], key=lambda s: s[1])[:cap_spans]}
            for cta, slot in sorted(chosen.items())]


def _count_role_switches(d: Mapping[str, Any]) -> int:
    n = 0
    for rs in (d.get("slots") or {}).values():
        for _t, tag, _aux in rs:
            if tag & _TAG_ROLE_SWITCH and not tag & _TAG_OPEN:
                n += 1
    return n


_HW_PID = 1
_SW_PID_SHIFT = 100
_HW_METRIC_ORDER = (
    "sm_active", "warp_issue", "tensor_active", "dram_read", "dram_write",
    "nvlink_rx_req", "nvlink_rx_rsp", "nvlink_tx_req", "nvlink_tx_rsp", "nvlink_user",
)


def _hw_counter_events(hw: Mapping[str, Any], tensor_scale: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build topmost Rendering Contract R6 hardware timeline events (pid=1, sort_index=-1000).

    ``tensor_scale`` is the result of :func:`resolve_tensor_active_scale`; when it is applied, the Tensor Active
    counter is written corrected (e.g. x4 on H20) and the raw statistics go into the nsys kernel-window args.
    """
    pid = _HW_PID
    series = hw.get("hw_series") or {}
    kernel = hw.get("kernel") or {}
    proc_title = (
        "00 · GPU Hardware Utilization [0–100%] (nsys)"
        if series else
        "00 · GPU Hardware Kernel Window (nsys)"
    )
    tev: List[Dict[str, Any]] = [
        {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": proc_title}},
        {"ph": "M", "pid": pid, "name": "process_sort_index", "args": {"sort_index": -1000}},
    ]
    if kernel and kernel.get("duration_us") is not None:
        tev.append({"ph": "M", "pid": pid, "tid": 1, "name": "thread_name", "args": {"name": "nsys Kernel Window"}})
        tev.append({"ph": "M", "pid": pid, "tid": 1, "name": "thread_sort_index", "args": {"sort_index": 1}})
        k_args = {k: v for k, v in kernel.items()}
        if hw.get("sample_freq_hz") is not None:
            k_args["sample_freq_hz"] = hw["sample_freq_hz"]
        if hw.get("in_window_samples") is not None:
            k_args["in_window_samples"] = hw["in_window_samples"]
        if hw.get("gpu_name"):
            k_args["gpu_name"] = hw["gpu_name"]
        if tensor_scale:
            # Raw nsys statistics stay auditable next to the corrected curve.
            k_args["tensor_active_scale"] = float(tensor_scale.get("scale") or 1.0)
            k_args["tensor_active_raw_mean_pct"] = tensor_scale.get("raw_mean_pct")
            k_args["tensor_active_raw_max_pct"] = tensor_scale.get("raw_max_pct")
            k_args["tensor_active_scale_reason"] = tensor_scale.get("reason")
        tev.append({
            "ph": "X", "pid": pid, "tid": 1, "ts": 0.0, "dur": float(kernel["duration_us"]),
            "name": str(kernel.get("name") or "kernel")[:120], "cat": "nsys_kernel", "args": k_args,
        })
    t_scale = float(tensor_scale.get("scale") or 1.0) if tensor_scale and tensor_scale.get("applied") else 1.0
    ordered_keys = [k for k in _HW_METRIC_ORDER if k in series] + [k for k in series if k not in _HW_METRIC_ORDER]
    for key in ordered_keys:
        curve = series[key]
        label = str(curve.get("metric_name") or key)
        scale = 1.0
        if key == "tensor_active" and t_scale != 1.0:
            # Only the corrected curve is emitted: viz merges every "tensor" counter with a per-timestamp max and
            # labels the lane with the first name it sees, so emitting raw + corrected would mislabel the lane.
            # "<role> [<text>]" makes viz._pretty_hw_curve_label use the bracket text as the legend.
            scale = t_scale
            label = f"tensor_active [Tensor Active x{t_scale:g} (H20 die-normalization corrected)]"
        for t, v in zip(curve.get("t_us") or [], curve.get("val") or []):
            tf = float(t)
            if tf < 0.0:
                continue  # Perfetto rejects negative timestamps
            tev.append({"ph": "C", "pid": pid, "ts": tf, "name": label, "args": {"pct": min(100.0, float(v) * scale)}})
    return tev


def _merge_hw_trace(
    trace: Mapping[str, Any],
    hw: Mapping[str, Any],
    *,
    summary: Optional[Mapping[str, Any]] = None,
    names: Optional[Mapping[str, Any]] = None,
    tensor_scale: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Shift software PIDs by +100 and prepend topmost nsys hardware timeline events."""
    trace_doc = dict(trace)
    other = dict(trace_doc.get("otherData") or {})
    if summary:
        wpc = int(summary.get("wpc") or 0)
        if wpc > 0:
            other["wpc"] = wpc
        role_names = {str(k): str(v) for k, v in ((names or {}).get("roles") or {}).items()}
        cta_sm = summary.get("cta_sm") or {}
        lb = summary.get("lane_bounds_ns") or {}
        if isinstance(lb, Mapping) and wpc > 0:
            warp_roles: Dict[str, str] = {}
            sm_warps: Dict[str, Dict[str, str]] = {}
            for slot_k, bounds in lb.items():
                try:
                    slot_i = int(slot_k)
                    cta_i, w_i = divmod(slot_i, wpc)
                    rid = str(bounds[2]) if isinstance(bounds, (list, tuple)) and len(bounds) >= 3 else "0"
                    rname = role_names.get(rid, f"role{rid}")
                    warp_roles.setdefault(str(w_i), rname)
                    sm_v = cta_sm.get(str(cta_i))
                    if sm_v is not None:
                        sm_warps.setdefault(str(int(sm_v)), {}).setdefault(str(w_i), rname)
                except (ValueError, TypeError):
                    continue
            if warp_roles:
                other["warp_roles"] = warp_roles
            if sm_warps:
                other["sm_warps"] = sm_warps
    if not (hw and (hw.get("hw_series") or hw.get("kernel"))):
        trace_doc["otherData"] = other
        return trace_doc
    sw_events: List[Dict[str, Any]] = []
    for raw_ev in (trace.get("traceEvents") or []):
        ev = dict(raw_ev)
        if isinstance(ev.get("pid"), int):
            ev["pid"] = int(ev["pid"]) + _SW_PID_SHIFT
            if ev.get("ph") == "M" and ev.get("name") == "process_sort_index":
                args_m = dict(ev.get("args") or {})
                if isinstance(args_m.get("sort_index"), (int, float)):
                    args_m["sort_index"] = int(args_m["sort_index"]) + _SW_PID_SHIFT
                ev["args"] = args_m
        sw_events.append(ev)
    trace_doc["traceEvents"] = _hw_counter_events(hw, tensor_scale) + sw_events
    other["nsys_hw"] = {k: v for k, v in hw.items() if k != "hw_series"}
    if tensor_scale:
        other["nsys_hw"]["tensor_active_scale"] = dict(tensor_scale)
    trace_doc["otherData"] = other
    return trace_doc


def find_nsys_binary() -> Optional[str]:
    """Locate `nsys` across PATH, standard Linux paths, and Windows Program Files installations."""
    found = shutil.which("nsys")
    if found:
        return found
    if os.name == "nt":
        nv_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation"
        if nv_root.is_dir():
            cands = sorted(
                list(nv_root.glob("Nsight Systems*/target-windows-x64/nsys.exe"))
                + list(nv_root.glob("Nsight Compute*/host/target-windows-x64/nsys.exe")),
                reverse=True,
            )
            for c in cands:
                if c.is_file():
                    return str(c)
    else:
        for p in ("/usr/local/bin/nsys", "/usr/local/cuda/bin/nsys"):
            if Path(p).is_file():
                return p
        opt_nv = Path("/opt/nvidia")
        if opt_nv.is_dir():
            for c in sorted(opt_nv.glob("nsight-systems*/*/bin/nsys"), reverse=True):
                if c.is_file():
                    return str(c)
    return None


def _capture_nsys_sqlite(
    *,
    l0: Path,
    inst: Path,
    out: Path,
    l0_cache_dir: Optional[Path],
    cmd: Sequence[str] | str,
    gpu_id: str,
    nsys_gpu_device: Optional[str],
    nsys_freq_hz: int,
    nsys_capture_range: str,
    policy: HarnessPolicy,
    timeout_s: float,
) -> Dict[str, Any]:
    """Run nsys profile + export to SQLite in a clean post-ABBA capture pass."""
    nsys_bin = find_nsys_binary()
    if not nsys_bin:
        raise RuntimeError("nsys binary not found on PATH, /usr/local/bin/nsys, or Windows NVIDIA Program Files")
    nsys_out = out / "nsys_run"
    nsys_out.mkdir(parents=True, exist_ok=True)
    rep_base = out / "nsys_cap"
    sqlite_path = out / "nsys_cap.sqlite"
    dev = nsys_gpu_device
    if not dev:
        if gpu_id not in ("", "auto"):
            dev = str(gpu_id)
        else:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            dev = cvd if (cvd and "," not in cvd) else "cuda-visible"
    env_nsys = _shadow_env(l0, inst, nsys_out, l0_cache_dir)
    env_nsys["WEAVEPROF_NSYS_CAPTURE"] = "1"
    subs = {"l0": str(l0), "inst": str(inst), "out": str(nsys_out), "tree": str(inst)}
    child_argv = _argv(cmd, subs)
    prof_argv = [
        nsys_bin, "profile", "--force-overwrite=true", "--stats=false", "--trace=cuda",
    ]
    if nsys_capture_range and nsys_capture_range.lower() != "none":
        prof_argv += [f"--capture-range={nsys_capture_range}", "--capture-range-end=stop"]
    prof_argv += [
        f"--gpu-metrics-devices={dev}",
        f"--gpu-metrics-frequency={int(nsys_freq_hz)}",
        "-o", str(rep_base),
        *child_argv,
    ]

    def _run_prof(argv_run: List[str], env_run: Dict[str, str], log_path: Path) -> int:
        with open(log_path, "wb") as log:
            try:
                proc = subprocess.run(argv_run, cwd=str(inst), env=env_run, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=timeout_s)
                return proc.returncode
            except subprocess.TimeoutExpired:
                return -9

    log_path = out / "nsys_profile.log"
    with gpu_critical_region(gpu_id, timeout_s=policy.region_lock_timeout_s, label="megakernel_nsys"):
        rc = _run_prof(prof_argv, env_nsys, log_path)
    rep_file = Path(str(rep_base) + ".nsys-rep")
    if rc != 0 or not rep_file.is_file():
        log_txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if "ERR_NVGPUCTRPERM" in log_txt or "Insufficient Privilege" in log_txt or "Insufficient privilege" in log_txt:
            hint = (
                "Windows: open NVIDIA Control Panel -> Developer -> Allow access to the GPU performance counters to all users (or run elevated)"
                if os.name == "nt"
                else "Linux: configure NVreg_RestrictProfilingToAdminUsers=0 or CAP_SYS_ADMIN (or run under sudo -n -E /usr/local/bin/nsys)"
            )
            raise RuntimeError(
                f"nsys profile exited {rc} due to GPU performance counter permissions ({hint}; "
                f"automatic privilege escalation is disabled; see {log_path})"
            )
        raise RuntimeError(f"nsys profile exited {rc} (see {log_path})")
    exp_log = out / "nsys_export.log"
    with open(exp_log, "wb") as log:
        exp = subprocess.run(
            [nsys_bin, "export", "--type=sqlite", "--force-overwrite=true", "--output", str(sqlite_path), str(rep_file)],
            cwd=str(out), stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s,
        )
    if exp.returncode != 0 or not sqlite_path.is_file():
        raise RuntimeError(f"nsys export exited {exp.returncode} (see {exp_log})")
    same_run_dump: Optional[Path] = None
    for cand in DUMP_CANDIDATES:
        if (nsys_out / cand).is_file():
            same_run_dump = nsys_out / cand
            break
    return {
        "sqlite_path": sqlite_path,
        "rep_path": rep_file,
        "same_run_dump": same_run_dump,
        "gpu_device": dev,
        "freq_hz": int(nsys_freq_hz),
    }


def _compact_opportunities(causal: Mapping[str, Any], n: int = 3) -> List[Dict[str, Any]]:
    out = []
    for opp in list(causal.get("causal_opportunities") or [])[:n]:
        out.append({
            "category": opp.get("category"),
            "title": str(opp.get("title") or "")[:96],
            "upper_bound_pct": opp.get("counterfactual_speedup_upper_bound_pct"),
        })
    return out


def _compact_skew(skew: Mapping[str, Any]) -> Dict[str, Any]:
    if not skew:
        return {}
    dist = skew.get("sm_busy") or {}
    return {
        "sm_busy_us": {k.replace("_us", ""): dist.get(k) for k in ("min_us", "median_us", "p90_us", "max_us", "spread_us")},
        "imbalance_pct": dist.get("imbalance_pct"),
        "stragglers": [skew["stragglers"]["count"], skew["stragglers"]["busy_us_median"]],
        "early": [skew["early_finishing"]["count"], skew["early_finishing"]["busy_us_median"]],
        "barrier_starved": skew["early_finishing"].get("barrier_starved"),
        "idle_sm_fraction": skew.get("idle_sm_fraction"),
        "sm_spread_ratio": skew.get("sm_spread_ratio"),
        "role_skew_ratio": skew.get("role_skew_ratio"),
        "hints": [h["code"] for h in skew.get("hints") or []],
    }


def _compact_hw_quartiles(hw: Mapping[str, Any]) -> Dict[str, List[float]]:
    """4-quartile hardware utilization trajectory [Q1, Q2, Q3, Q4] for text-only LLMs.

    Tensor Active uses the same correction as the overview PNG (``hw["tensor_active_scale"]``).
    """
    series = hw.get("hw_series") or {}
    dur = float((hw.get("kernel") or {}).get("duration_us") or 0.0)
    if not series or dur <= 0.0:
        return {}
    ts_info = hw.get("tensor_active_scale") or {}
    t_scale = float(ts_info.get("scale") or 1.0) if ts_info.get("applied") else 1.0
    out: Dict[str, List[float]] = {}
    for key in ("tensor_active", "sm_active", "warp_issue", "dram_read", "dram_write"):
        curve = series.get(key)
        if not curve:
            continue
        scale = t_scale if key == "tensor_active" else 1.0
        bins: List[List[float]] = [[], [], [], []]
        for t, v in zip(curve.get("t_us") or [], curve.get("val") or []):
            tf = float(t)
            if 0.0 <= tf <= dur:
                idx = min(3, max(0, int(4.0 * tf / dur)))
                bins[idx].append(min(100.0, float(v) * scale))
        if any(bins):
            out[key] = [round(statistics.fmean(b), 1) if b else 0.0 for b in bins]
    return out


def _compact_stages(summary: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Compact {stage: [n, mean_us, share_pct]} table for text-only LLMs."""
    events = summary.get("events") or {}
    tot = sum(float(v.get("total_us") or 0.0) for v in events.values())
    if tot <= 0.0:
        return {}
    ranked = sorted(events.items(), key=lambda kv: -float(kv[1].get("total_us") or 0.0))[:6]
    return {
        k: [int(v.get("n") or 0), round(float(v.get("mean_us") or 0.0), 2),
            round(100.0 * float(v.get("total_us") or 0.0) / tot, 1)]
        for k, v in ranked
    }


def _fit_report(report: Dict[str, Any]) -> Dict[str, Any]:
    def size() -> int:
        return len(json.dumps(report, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    trims = (
        lambda: report.__setitem__("causal_opportunities", report["causal_opportunities"][:2]),
        lambda: report.__setitem__("structural_blockers", report["structural_blockers"][:3]),
        lambda: report.pop("hw_quartiles", None),
        lambda: report.pop("stage_summary", None),
        lambda: report.__setitem__("causal_opportunities", [
            {"category": o.get("category"), "upper_bound_pct": o.get("upper_bound_pct")} for o in report["causal_opportunities"]]),
        lambda: report.__setitem__("structural_blockers", [b[:60] for b in report["structural_blockers"]][:2]),
        lambda: report["megakernel_skew"].pop("sm_busy_us", None) if report.get("megakernel_skew") else None,
        lambda: report.pop("round_diff", None),
        lambda: report.pop("artifacts", None),
    )
    for trim in trims:
        if size() <= REPORT_MAX_BYTES - 48:
            break
        trim()
    report["report_bytes"] = size() + 24
    return report


def run_megakernel_audit(
    workspace: str | Path,
    *,
    bench_cmd: Sequence[str] | str | None = None,
    entry_source: str | Path | None = None,
    inst_files: Mapping[str, str] | None = None,
    inst_dir: str | Path | None = None,
    dump_path: str | Path | None = None,
    names: Mapping[str, Any] | str | Path | None = None,
    runtime_metrics: Mapping[str, Any] | str | Path | None = None,
    nsys_sqlite: str | Path | None = None,
    capture_nsys: bool = False,
    nsys_cmd: Sequence[str] | str | None = None,
    nsys_freq_hz: int = 200000,
    nsys_gpu_device: str | None = None,
    nsys_capture_range: str = "cudaProfilerApi",
    render_overview_png: bool = True,
    write_perfetto_gz: bool = False,
    overview_title: str | None = None,
    overview_subtitle: str | None = None,
    overview_sm: int | None = None,
    model: str | None = None,
    op: str | None = None,
    bs: int | None = None,
    seql: int | None = None,
    kv: int | None = None,
    shape: str | None = None,
    dims: Mapping[str, Any] | None = None,
    card: str = "auto",
    arch: str | None = None,
    total_sms: int | None = None,
    limit: float = 0.10,
    out_dir: str | Path | None = None,
    gpu_id: str = "auto",
    use_l0_cache: bool = True,
    prev_causal_report: Mapping[str, Any] | str | Path | None = None,
    build_cmd: Sequence[str] | str | None = None,
    policy: HarnessPolicy | None = None,
    round_number: int | None = None,
) -> Dict[str, Any]:
    """Audit a Sparse Attention / persistent megakernel and seal the episode.

    Workflow A (``bench_cmd`` without ``dump_path``): copy ``workspace`` into shadow ``L0`` / ``L_inst``
    trees, apply ``inst_files`` / ``inst_dir`` to ``L_inst``, optionally pre-build both trees in parallel
    (``build_cmd``; L0 products cached by content SHA-256), run ``bench_cmd`` once inside the KREX GPU
    critical region and pick up ``runtime.json`` + ``capture.wp02.bin``/``capture.fi02.bin`` (+ ``names.json``).

    Workflow B (``dump_path`` given): audit a pre-captured dump against ``runtime_metrics``.

    Both run the 4 fidelity gates (compiler spill, output equality, semantic stub + overflow + pairing,
    warm ABBA overhead <= ``limit``), ``analyze_megakernel_skew`` + ``analyze_sw_hw_causality`` (with
    ``extract_nsys_sqlite_fast`` curves when ``nsys_sqlite`` is given), write ``feedback.json``,
    ``verify.json`` and ``trace.perfetto.json.gz``, seal with ``EpisodeSealer`` + ``verify_seal`` and
    return a compact (<2 KB) agent report.
    """
    t_start = time.perf_counter()
    pol = policy or HarnessPolicy()
    guard = PolicyGuard(pol)
    ws = Path(workspace).resolve()
    out = Path(out_dir).resolve() if out_dir else (_cache_base() / "mk_audit" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}")
    out.mkdir(parents=True, exist_ok=True)
    blockers: List[str] = []
    hard_fail = False
    mode = "shadow_build_run" if (bench_cmd and dump_path is None) else "precaptured"
    bench_list = (_argv(bench_cmd, {"l0": "{l0}", "inst": "{inst}", "out": "{out}", "tree": "{tree}"})
                  if bench_cmd else None)

    entry: Optional[Path] = None
    if entry_source is not None:
        entry = Path(entry_source)
        entry = (entry if entry.is_absolute() else ws / entry).resolve()
    source_sha = digest(entry) if entry is not None and entry.is_file() else None

    def _blocked(code: str, message: str) -> Dict[str, Any]:
        rep = {
            "schema": MEGAKERNEL_SCHEMA, "status": "BLOCKED", "mode": mode, "seal_verified": False,
            "seal_path": None, "seal_sha256": None,
            "wall_clock_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            "overhead_pct": None, "kernel_us": None, "megakernel_skew": {}, "causal_opportunities": [],
            "structural_blockers": [f"{code}: {message}"[:200]], "round_diff": None, "fidelity_gates": {},
            "out_dir": str(out),
        }
        return _fit_report(rep)

    if not ws.is_dir():
        return _blocked("WORKSPACE_MISSING", str(ws))
    if mode == "shadow_build_run" and out.is_relative_to(ws):
        return _blocked("OUT_DIR_INSIDE_WORKSPACE", f"out_dir must be outside workspace ({out})")
    if entry is None or source_sha is None:
        return _blocked("ENTRY_SOURCE_MISSING", f"entry_source must name the clean production source ({entry})")
    try:
        guard.validate_session_start(entry, source_sha, float(limit), bench_cmd=bench_list)
    except PolicyViolation as exc:
        return _blocked(exc.code, str(exc))

    entry_text = entry.read_text(encoding="utf-8", errors="replace")
    if _INSTRUMENTED_SRC_RE.search(entry_text):
        blockers.append("PRODUCTION_SOURCE_INSTRUMENTED: entry_source contains recorder macros; audit the clean source")
        hard_fail = True

    region_lock: Optional[Dict[str, Any]] = None
    instrumented_path: Optional[Path] = None
    shadow_info: Dict[str, Any] = {}
    rm_path: Optional[Path] = None
    if mode == "shadow_build_run":
        try:
            shadow_info = _run_shadow_workflow(
                ws, bench_cmd=bench_cmd, build_cmd=build_cmd, inst_files=inst_files,
                inst_dir=Path(inst_dir).resolve() if inst_dir else None, out=out, gpu_id=gpu_id,
                use_l0_cache=use_l0_cache, policy=pol, timeout_s=pol.command_timeout_s,
            )
        except PolicyViolation as exc:
            return _blocked(exc.code, str(exc))
        except (RuntimeError, OSError, TimeoutError, ValueError) as exc:
            return _blocked("SHADOW_RUN_FAILED", f"{type(exc).__name__}: {exc}")
        region_lock = shadow_info["region_lock"]
        dump_path = shadow_info["dump_path"]
        if runtime_metrics is None:
            runtime_metrics = shadow_info["runtime_path"]
        if names is None:
            names = shadow_info["names_path"]
        rel = entry.relative_to(ws) if entry.is_relative_to(ws) else None
        if rel is not None and (shadow_info["inst_dir"] / rel).is_file():
            instrumented_path = shadow_info["inst_dir"] / rel

    try:
        rm, rm_path = _load_mapping(runtime_metrics, "runtime_metrics")
    except (OSError, ValueError) as exc:
        return _blocked("RUNTIME_METRICS_INVALID", str(exc))
    if rm_path is None and rm:
        rm_path = out / "runtime_metrics.json"
        write_json(rm_path, rm)
    rm_base = rm_path.parent if rm_path else ws
    if rm.get("source_sha256") and str(rm["source_sha256"]) != source_sha:
        blockers.append("SOURCE_MUTATED: runtime_metrics.source_sha256 != live entry_source sha256")
        hard_fail = True
    if instrumented_path is None and rm.get("instrumented_source"):
        cand = Path(rm["instrumented_source"])
        cand = cand if cand.is_absolute() else rm_base / cand
        instrumented_path = cand if cand.is_file() else None
    if region_lock is None:
        region_lock = dict(rm["region_lock"]) if isinstance(rm.get("region_lock"), Mapping) else {
            "mode": "precaptured", "measured_by": str(rm.get("measured_by") or "external_harness"),
            "gpu_id": gpu_id,
        }

    if dump_path is None or not Path(dump_path).is_file():
        return _blocked("DUMP_MISSING", f"no WPLITE02/FILITE02 dump ({dump_path})")
    dump = Path(dump_path).resolve()
    try:
        names_doc, names_src = _load_mapping(names, "names")
    except (OSError, ValueError) as exc:
        return _blocked("NAMES_INVALID", str(exc))
    names_path = out / "names.json"
    write_json(names_path, names_doc)

    # Decode.
    wp = load_wp_decode()
    trace: Dict[str, Any] = {"traceEvents": [], "displayTimeUnit": "ns"}
    summary: Dict[str, Any] = {}
    try:
        d = wp.read_dump(str(dump))
        trace, summary = wp.decode(d, names_doc)
        summary["role_switches"] = _count_role_switches(d)
        summary["source_dump_sha256"] = digest(dump)
    except SystemExit as exc:
        blockers.append(f"DECODE_FAILED: {exc}"[:200])
        hard_fail = True
    except Exception as exc:  # fail closed
        blockers.append(f"DECODE_FAILED: {type(exc).__name__}: {exc}"[:200])
        hard_fail = True

    # Fidelity gates (evaluated on clean non-nsys ABBA run first).
    g_compiler = _gate_compiler(rm, pol)
    g_output = _gate_output(rm, rm_base, require_files=pol.require_output_files)
    g_events = _gate_events(summary, trace, rm, pol) if summary else {
        "ok": False, "stub_gate": {"ok": False, "stubs": [], "valid_stages": [], "stub_count": 0},
        "overflow": None, "pairing_errors": None, "spans": 0, "decoder_errors": {}}
    g_overhead = _gate_overhead(guard, rm, float(limit))
    checks = {"assembly": bool(g_compiler["ok"]), "output": bool(g_output["ok"]),
              "events": bool(g_events["ok"]), "overhead": bool(g_overhead["ok"])}

    # Auto-capture nsys hardware timeline on passing shadow runs (keeps ABBA overhead unbiased
    # and avoids wasting nsys capture time on failed retry rounds).
    if (mode == "shadow_build_run" and capture_nsys and nsys_sqlite is None
            and all(checks.values()) and not hard_fail):
        try:
            nsys_cap = _capture_nsys_sqlite(
                l0=shadow_info["l0_dir"],
                inst=shadow_info["inst_dir"],
                out=out,
                l0_cache_dir=shadow_info.get("l0_cache_dir"),
                cmd=nsys_cmd or bench_cmd,
                gpu_id=gpu_id,
                nsys_gpu_device=nsys_gpu_device,
                nsys_freq_hz=int(nsys_freq_hz),
                nsys_capture_range=str(nsys_capture_range or "cudaProfilerApi"),
                policy=pol,
                timeout_s=pol.command_timeout_s,
            )
            nsys_sqlite = nsys_cap["sqlite_path"]
            shadow_info["nsys_capture"] = {
                k: (str(v) if isinstance(v, Path) else v) for k, v in nsys_cap.items()
            }
            if nsys_cap.get("same_run_dump") is not None:
                try:
                    d_sr = wp.read_dump(str(nsys_cap["same_run_dump"]))
                    tr_sr, sm_sr = wp.decode(d_sr, names_doc)
                    if int(sm_sr.get("overflow") or 0) == 0 and not sm_sr.get("errors") and int(sm_sr.get("spans") or 0) > 0:
                        sm_sr["role_switches"] = _count_role_switches(d_sr)
                        sm_sr["source_dump_sha256"] = digest(nsys_cap["same_run_dump"])
                        trace, summary = tr_sr, sm_sr
                        dump = Path(nsys_cap["same_run_dump"]).resolve()
                except Exception as exc:
                    blockers.append(f"NSYS_SAME_RUN_DUMP_FALLBACK: {exc}"[:160])
        except (RuntimeError, OSError, TimeoutError, ValueError, PolicyViolation) as exc:
            blockers.append(f"NSYS_CAPTURE_FAILED: {exc}"[:200])
            hard_fail = True

    # Reclaim duplicated shadow workspace trees immediately after bench/nsys (prevents inode bloat).
    shadow_root_str = shadow_info.get("shadow")
    if shadow_root_str and os.environ.get("WEAVEPROF_KEEP_SHADOW") != "1":
        shadow_root = Path(shadow_root_str).resolve()
        try:
            if instrumented_path is not None and instrumented_path.is_file() and instrumented_path.resolve().is_relative_to(shadow_root):
                kept_inst = out / f"instrumented_{instrumented_path.name}"
                shutil.copy2(instrumented_path, kept_inst)
                instrumented_path = kept_inst
            if dump is not None and dump.is_file() and dump.resolve().is_relative_to(shadow_root):
                kept_dump = out / dump.name
                shutil.copy2(dump, kept_dump)
                dump = kept_dump
            if rm_path is not None and rm_path.is_file() and rm_path.resolve().is_relative_to(shadow_root):
                kept_rm = out / rm_path.name
                shutil.copy2(rm_path, kept_rm)
                rm_path = kept_rm
            if names_src is not None and names_src.is_file() and names_src.resolve().is_relative_to(shadow_root):
                names_src = names_path
            shutil.rmtree(shadow_root, ignore_errors=True)
        except OSError:
            pass

    # Optional hardware curves.
    hw: Dict[str, Any] = {}
    hw_path: Optional[Path] = None
    if nsys_sqlite:
        try:
            hw = extract_nsys_sqlite_fast(nsys_sqlite, kernel_substr=str(rm.get("kernel_name") or ""))
            # nsys_hw.json keeps the raw curves; the decision travels with them so every consumer
            # (overview PNG, hw_quartiles, causal bundle) applies the same factor.
            hw["tensor_active_scale"] = resolve_tensor_active_scale(
                hw, arch=arch or (summary.get("arch") if summary else None), card=card)
            plateau = _tensor_plateau_suspect(hw, hw["tensor_active_scale"])
            if plateau:
                blockers.append(plateau[:200])
            hw_path = out / "nsys_hw.json"
            write_json(hw_path, hw)
        except (OSError, ValueError, sqlite3.Error) as exc:
            blockers.append(f"NSYS_SQLITE_UNREADABLE: {exc}"[:200])

    coverage = None
    if hw and summary:
        coverage = verify_sw_hw_coverage(float(summary.get("span_us") or 0.0), float(hw["kernel"]["duration_us"]),
                                         in_window_samples=hw.get("in_window_samples") if hw.get("hw_series") else None)
        if not coverage["ok"]:
            blockers.extend(f"{v['code']} (advisory)" for v in coverage["violations"][:2])

    # Analysis.
    summary_arch = summary.get("arch") if summary else None
    sms = _resolve_total_sms(total_sms, card, arch, summary_arch)
    skew: Dict[str, Any] = {}
    causal: Dict[str, Any] = {}
    if summary:
        skew = analyze_megakernel_skew(summary, trace, total_sms=sms)
        bundle: Dict[str, Any] = {
            "target": entry.name,
            "arch": arch or summary_arch,
            "gpu": hw.get("gpu_name") or ("" if str(card).lower() == "auto" else card),
            "sms": sms or skew.get("total_sms"),
            "sw_span_us": float(summary.get("span_us") or 0.0),
            "per_sm_w0": _per_sm_w0(summary, trace),
            "hw_series": hw.get("hw_series") or {},
        }
        bundle["phase_names"] = {s[0]: s[0] for rec in bundle["per_sm_w0"] for s in rec["spans"]}
        if hw:
            bundle["nsys_kernel_us"] = hw["kernel"]["duration_us"]
        try:
            causal = analyze_sw_hw_causality(bundle, arch=arch or summary_arch, total_sms=sms)
        except Exception as exc:  # analysis is advisory; gates stay authoritative
            blockers.append(f"CAUSAL_ANALYSIS_FAILED: {type(exc).__name__}: {exc}"[:200])
    for sb in (causal.get("structural_blockers") or [])[:3]:
        code = sb.get("code") or sb.get("category") or sb.get("title") if isinstance(sb, Mapping) else str(sb)
        if code:
            blockers.append(str(code)[:120])

    round_diff = None
    if prev_causal_report is not None and causal:
        try:
            prev, _ = _load_mapping(prev_causal_report, "prev_causal_report")
            prev_causal = prev.get("causal") if isinstance(prev.get("causal"), Mapping) else prev
            rd = compare_optimization_rounds(prev_causal, causal)
            round_diff = {
                "latency_delta_pct": rd.get("latency_delta_pct"),
                "speedup_ratio": rd.get("speedup_ratio"),
                "improved": rd.get("improved"),
                "regressed": rd.get("regressed"),
                "next": (rd.get("next_round_directive") or {}).get("action"),
            }
            if round_number is None and isinstance(prev.get("round"), int):
                round_number = int(prev["round"]) + 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blockers.append(f"ROUND_DIFF_FAILED: {exc}"[:200])

    failed = [k for k, v in checks.items() if not v]
    status = "PASS" if (not failed and not hard_fail) else "FAIL"

    # Immutability re-check right before sealing.
    try:
        guard.validate_source_immutable(entry, source_sha)
    except PolicyViolation as exc:
        blockers.append(f"{exc.code}: production source changed during the audit")
        status, hard_fail = "FAIL", True

    gates = {
        "compiler_spill_gate": g_compiler,
        "output_gate": g_output,
        "event_gate": g_events,
        "overhead_gate": g_overhead,
        "sw_hw_coverage": coverage,
    }
    feedback_path = out / "feedback.json"
    verify_path = out / "verify.json"
    trace_path = out / "trace.perfetto.json.gz"
    trace_doc = _merge_hw_trace(trace, hw, summary=summary, names=names_doc,
                                tensor_scale=hw.get("tensor_active_scale") if hw else None)
    wrote_gz = False
    if write_perfetto_gz:
        wp.write_json(trace_doc, str(trace_path))
        wrote_gz = trace_path.is_file()
    overview_png_path: Optional[Path] = None
    if render_overview_png:
        try:
            from .viz import build_timeline_png_name, render_swhw_overview_png
            png_name = build_timeline_png_name(
                trace_doc,
                model=model,
                op=op,
                bs=bs,
                seql=seql,
                kv=kv,
                shape=shape,
                dims=dims,
                title=overview_title,
                subtitle=overview_subtitle,
                source=entry,
                runtime=rm if isinstance(rm, Mapping) else None,
            )
            overview_png_path = render_swhw_overview_png(
                trace_doc, out / png_name,
                title=overview_title, subtitle=overview_subtitle, focus_sm=overview_sm,
            )
        except ImportError:
            overview_png_path = None
            blockers.append("OVERVIEW_PNG_SKIPPED: matplotlib unavailable")
        except Exception as exc:
            blockers.append(f"OVERVIEW_PNG_FAILED: {type(exc).__name__}: {exc}"[:160])
        if status == "PASS":
            valid_png = False
            if overview_png_path and overview_png_path.is_file():
                try:
                    raw_hdr = overview_png_path.read_bytes()[:8]
                    valid_png = (raw_hdr == b"\x89PNG\r\n\x1a\n" and overview_png_path.stat().st_size > 1024)
                except OSError:
                    valid_png = False
            if not valid_png:
                status = "FAIL"
                hard_fail = True
                blockers.append("MISSING_OVERVIEW_PNG: render_overview_png=True requires a valid non-empty timeline PNG")

    feedback_doc = {
        "schema": MEGAKERNEL_SCHEMA, "status": status, "mode": mode, "megakernel_skew": skew,
        "causal": causal, "fidelity_gates": gates, "structural_blockers": blockers, "round_diff": round_diff,
        "decode_summary": {k: summary.get(k) for k in ("span_us", "records", "spans", "lanes", "gran", "wpc", "n_ctas",
                                                        "overflow", "errors", "events", "roles", "magic", "arch")},
        "nsys": {k: v for k, v in hw.items() if k != "hw_series"} if hw else None,
        "l0_cache": shadow_info.get("l0_cache"),
        "round": round_number or 1,
    }
    verify_doc = {
        "schema": MEGAKERNEL_SCHEMA, "status": status, "checks": checks, "failed": failed, "limit": float(limit),
        "overhead_ratio": g_overhead.get("overhead_ratio"), "source_sha256": source_sha,
        "fidelity_gates": gates, "region_lock": region_lock,
    }
    write_json(feedback_path, feedback_doc)
    write_json(verify_path, verify_doc)
    result = {"status": status, "checks": checks, "failed": failed + (["structural"] if hard_fail else []),
              "metrics": {"overhead_ratio": g_overhead.get("overhead_ratio"), "region_lock": region_lock}}
    primary_visual = (
        trace_path if wrote_gz
        else (overview_png_path if overview_png_path and overview_png_path.is_file() else dump)
    )
    seal_path, seal_sha, seal_verified, seal_errors = None, None, False, []
    try:
        seal_path, seal_doc = EpisodeSealer.seal_round(
            round_dir=out, round_number=int(round_number or 1), source_path=entry, source_sha256=source_sha,
            probe_path=None, instrumented_path=instrumented_path, names_path=names_path,
            feedback_path=feedback_path, verify_path=verify_path,
            trace_path=primary_visual if status == "PASS" else None,
            failed_trace_path=None if status == "PASS" else primary_visual,
            result=result, policy=pol, limit=float(limit), bench_cmd=bench_list,
            fsm_state="VERIFY_AND_SEAL", region_lock=region_lock,
            extra_artifacts={"dump": dump, "runtime_metrics": rm_path, "nsys_hw": hw_path,
                             "names_source": names_src, "overview_png": overview_png_path},
            mode=mode,
        )
        seal_sha = seal_doc.get("seal_sha256")
        ver = verify_seal(seal_path, require_pass=(status == "PASS"), expected_source_sha=source_sha,
                          expected_policy_digest=pol.digest(), max_overhead_limit=float(limit))
        seal_verified, seal_errors = bool(ver["ok"]), list(ver["errors"])
    except Exception as exc:  # sealing failure is fatal for PASS
        seal_errors = [f"{type(exc).__name__}: {exc}"]
    if status == "PASS" and not seal_verified:
        status = "FAIL"
        blockers.append(("SEAL_VERIFY_FAILED: " + "; ".join(seal_errors))[:200])
        feedback_doc["status"] = status
        verify_doc["status"] = status
        write_json(feedback_path, feedback_doc)
        write_json(verify_path, verify_doc)

    neg_pert = any("NEGATIVE_OVERHEAD_PERTURBATION" in str(e) for e in (g_overhead.get("errors") or []))
    if neg_pert:
        blockers.append("NEGATIVE_OVERHEAD_PERTURBATION")
    gran_gate = g_events.get("granularity_gate") or {"ok": True}
    if not gran_gate.get("ok", True):
        blockers.append(f"SINGLE_STAGE_LOOP_COLLAPSE: {gran_gate.get('dominant_stage')} ({gran_gate.get('dominant_share_pct')}%)")
    warp_gate = g_events.get("warp_gate") or {"ok": True}
    if not warp_gate.get("ok", True):
        blockers.append(
            f"WARP_COVERAGE_INCOMPLETE: {len(warp_gate.get('observed_warps') or [])}/"
            f"{warp_gate.get('expected_warps')} warps"
        )
    gate_view = {
        "compiler_spill": {"ok": checks["assembly"], "codes": [v.get("code") for v in g_compiler.get("violations") or []][:4],
                           "reg_blind_spot": bool(g_compiler.get("register_count_blind_spot"))},
        "output": {"ok": checks["output"], **({"code": g_output["code"]} if g_output.get("code") else {})},
        "stub": {"ok": bool(g_events["stub_gate"]["ok"]),
                 "stubs": [s["stage"] for s in g_events["stub_gate"].get("stubs") or []][:4]},
        "granularity": {"ok": bool(gran_gate.get("ok", True)),
                        **({"code": gran_gate.get("code")} if gran_gate.get("code") else {})},
        "warps": {"ok": bool(warp_gate.get("ok", True)),
                  **({"code": warp_gate.get("code")} if warp_gate.get("code") else {})},
        "overflow": g_events.get("overflow"),
        "pairing_errors": g_events.get("pairing_errors"),
        "overhead": {"ok": checks["overhead"], "ratio": _r(g_overhead.get("overhead_ratio"), 5),
                     **({"code": "NEGATIVE_OVERHEAD_PERTURBATION"} if neg_pert else {})},
    }
    ratio = g_overhead.get("overhead_ratio")
    report: Dict[str, Any] = {
        "schema": MEGAKERNEL_SCHEMA,
        "status": status,
        "mode": mode,
        "seal_verified": seal_verified,
        "seal_path": str(seal_path) if seal_path else None,
        "seal_sha256": seal_sha,
        "wall_clock_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        "overhead_pct": round(100.0 * float(ratio), 3) if ratio is not None else None,
        "kernel_us": (hw.get("kernel") or {}).get("duration_us") or (skew.get("kernel_us") if skew else None),
        "megakernel_skew": _compact_skew(skew),
        "stage_summary": _compact_stages(summary) if summary else {},
        "hw_quartiles": _compact_hw_quartiles(hw) if hw else {},
        "tensor_active_scale": (hw.get("tensor_active_scale") or {}).get("scale") if hw else None,
        "causal_opportunities": _compact_opportunities(causal),
        "structural_blockers": [str(b)[:140] for b in blockers][:6],
        "round_diff": round_diff,
        "fidelity_gates": gate_view,
        "artifacts": {
            "out_dir": str(out),
            **({"timeline_png": overview_png_path.name} if overview_png_path and overview_png_path.is_file() else {}),
            "files": ["feedback.json", "verify.json"]
            + ([overview_png_path.name] if overview_png_path and overview_png_path.is_file() else [])
            + (["trace.perfetto.json.gz"] if wrote_gz else []),
        },
    }
    if status != "PASS" and seal_verified:
        report["seal_integrity_only"] = True
    return _fit_report(report)


__all__ = [
    "MEGAKERNEL_SCHEMA", "SKEW_SCHEMA", "NSYS_FAST_SCHEMA", "WpSoController", "analyze_megakernel_skew",
    "extract_nsys_sqlite_fast", "run_megakernel_audit", "load_wp_decode", "resolve_tensor_active_scale",
]
