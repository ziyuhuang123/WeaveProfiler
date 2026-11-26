"""Publication-style single-run 4-panel SW+HW overview renderer for WeaveProfiler traces.

Renders any WeaveProfiler ``trace.perfetto.json[.gz]`` (or in-memory trace dict) across
NVIDIA (H20 / B300 / sm_86, SM + warp, nsys) and AMD (MI308X, CU + wave, rocprofiler-sdk / rocprof)
into a single high-DPI PNG with one locked microsecond time axis across Panels (A)-(C) plus an
automatic deterministic Panel (D) single-CTA micro-zoom on dense timelines:
  (A) Same-run real hardware utilization curves (Compute / HBM / Intra-Node NVLink or xGMI /
      optional 5th Inter-Node RDMA NIC curve or explicit unavailable HCA footnote / Active SM or CU)
  (B) All physical SMs or CUs dense multi-stage / dual-fabric waterfall
  (C) Single SM / CU per-warp / per-wave timeline (upper main execution track + 0/1/2 on-demand
      thin async engine depth strips for MEM/MMA with fixed-pixel issue pins)
  (D) Automatic deterministic single-CTA micro-zoom (triggered when full-kernel spacing < 12 px)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime as _dt
import gzip
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_MODEL_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])((?:qwen|deepseek|llama|mistral|mixtral|gemma|glm|kimi|gpt|claude|internlm|yi|baichuan|minimax|phi|nemotron)"
    r"[a-z0-9.-]{0,24})(?![a-z0-9])",
    re.I,
)
_OP_ALIASES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?:flash[_\-\s]?attention|attention|(?:^|[_\-\s./])(?:flash[_\-\s]?attn\d*|flash[_\-\s]?mla|flash[_\-\s]?kda|fa[23]|mla|kda|sdpa|mqa|gdn|attn)(?:$|[_\-\s./0-9]))", re.I), "attn"),
    (re.compile(r"(?:megamoe|fused[_\-\s]?moe|(?:^|[_\-\s./])moe(?:$|[_\-\s./0-9]))", re.I), "moe"),
    (re.compile(r"(?:gemm|matmul|(?:^|[_\-\s./])(?:wgmma|wmma|mfma|scaled[_\-]?mm|dot)(?:$|[_\-\s./0-9]))", re.I), "gemm"),
    (re.compile(r"(?:convolution|(?:^|[_\-\s./])(?:conv[123]d|causal[_\-]?conv|conv)(?:$|[_\-\s./0-9]))", re.I), "conv"),
    (re.compile(r"(?:rmsnorm|layernorm|groupnorm|(?:^|[_\-\s./])norm(?:$|[_\-\s./0-9]))", re.I), "norm"),
    (re.compile(r"softmax", re.I), "softmax"),
    (re.compile(r"(?:allreduce|allgather|alltoall|(?:^|[_\-\s./])(?:scatter|combine|reduce)(?:$|[_\-\s./0-9]))", re.I), "comm"),
)


def _sanitize_fname_token(raw: Any, fallback: str, *, max_len: int = 32, keep_dots: bool = False) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return fallback
    pat = r"[^a-z0-9._-]+" if keep_dots else r"[^a-z0-9_-]+"
    s = re.sub(pat, "-", s)
    s = re.sub(r"[-_]{2,}", "-", s).strip("._-")
    if not s:
        return fallback
    return s[:max_len].rstrip("._-") or fallback


def _extract_model_name(
    meta: Mapping[str, Any],
    other: Mapping[str, Any],
    workload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    spec: Mapping[str, Any],
    texts: Sequence[str],
    explicit: Optional[str],
) -> str:
    for cand in (
        explicit,
        os.environ.get("WEAVEPROF_WORKLOAD_MODEL"),
        meta.get("model"),
        meta.get("model_name"),
        meta.get("workload_model"),
        other.get("model"),
        other.get("model_name"),
        workload.get("model"),
        workload.get("model_name"),
        runtime.get("workload_model"),
        runtime.get("model_name"),
        runtime.get("model"),
        spec.get("workload_model"),
        spec.get("model_name"),
    ):
        if cand is not None and str(cand).strip() and str(cand).strip().lower() not in ("auto", "none", "unknown"):
            return _sanitize_fname_token(cand, "model", max_len=32, keep_dots=True)
    for txt in texts:
        if not txt:
            continue
        m = _MODEL_TOKEN_RE.search(txt)
        if m:
            return _sanitize_fname_token(m.group(1), "model", max_len=32, keep_dots=True)
    return "model"


def _extract_op_name(
    meta: Mapping[str, Any],
    other: Mapping[str, Any],
    workload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    spec: Mapping[str, Any],
    texts: Sequence[str],
    source: Optional[str | Path],
    explicit: Optional[str],
) -> str:
    if explicit is not None and str(explicit).strip():
        return _sanitize_fname_token(explicit, "kernel", max_len=24)
    for cand in (
        meta.get("op"),
        meta.get("operator"),
        meta.get("operator_family"),
        other.get("op"),
        other.get("operator"),
        workload.get("op"),
        workload.get("operator"),
        runtime.get("op"),
        runtime.get("operator"),
        runtime.get("kernel_class"),
        spec.get("operator_family"),
    ):
        if cand is not None and str(cand).strip():
            raw = str(cand).strip()
            for rx, norm in _OP_ALIASES:
                if rx.search(raw):
                    return norm
            return _sanitize_fname_token(raw, "kernel", max_len=24)
    for txt in texts:
        if not txt:
            continue
        for rx, norm in _OP_ALIASES:
            if rx.search(txt):
                return norm
    if source:
        stem = Path(str(source)).stem
        if stem:
            return _sanitize_fname_token(stem, "kernel", max_len=24)
    return "kernel"


def _extract_shape_token(
    meta: Mapping[str, Any],
    other: Mapping[str, Any],
    workload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    spec: Mapping[str, Any],
    texts: Sequence[str],
    *,
    bs: Optional[Any] = None,
    seql: Optional[Any] = None,
    kv: Optional[Any] = None,
    shape: Optional[str] = None,
    dims: Optional[Mapping[str, Any]] = None,
) -> str:
    if shape is not None and str(shape).strip():
        return _sanitize_fname_token(str(shape).replace("×", "x"), "shape", max_len=48)

    merged: Dict[str, Any] = {}
    for src_map in (spec, runtime, other, meta, workload, dims or {}):
        if isinstance(src_map, Mapping):
            for k, v in src_map.items():
                if v is not None and str(v).strip() != "":
                    merged[str(k).lower()] = v
    if bs is not None:
        merged["bs"] = bs
    if seql is not None:
        merged["seql"] = seql
    if kv is not None:
        merged["kv"] = kv

    def _pick(*keys: str) -> Optional[str]:
        for k in keys:
            v = merged.get(k)
            if v is not None and str(v).strip() != "":
                s = str(v).strip()
                if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[kKmM]?", s):
                    return s.lower()
        return None

    v_bs = _pick("bs", "batch", "batch_size", "b")
    v_sq = _pick("seql", "seq_len", "seqlen", "sq", "s", "t", "tokens", "n_tokens")
    v_kv = _pick("kv", "kv_len", "kvlen", "kv_seq_len")
    v_h = _pick("h", "heads", "num_heads")
    v_d = _pick("d", "head_dim", "headdim")
    v_m = _pick("m")
    v_n = _pick("n")
    v_k = _pick("k")
    v_e = _pick("e", "experts", "num_experts")
    v_ep = _pick("ep")

    blob = " ".join(t for t in texts if t)
    if blob:
        if v_bs is None:
            m = re.search(r"(?:\bbs|\bbatch(?:_size)?|\bB)\s*[=:]?\s*(\d+)\b", blob)
            if m:
                v_bs = m.group(1)
        if v_sq is None:
            m = re.search(r"(?:\bseql|\bseq_?len|\bsq|\bS|\bT)\s*[=:]?\s*(\d+)\b", blob)
            if m:
                v_sq = m.group(1)
        if v_kv is None:
            m = re.search(r"(?:\bkv(?:_?len)?|\bKV)\s*[=:]?\s*(\d+)\b", blob)
            if m:
                v_kv = m.group(1)
        if v_h is None:
            m = re.search(r"(?:\bH|\bheads)\s*=\s*(\d+)\b", blob)
            if m:
                v_h = m.group(1)
        if v_d is None:
            m = re.search(r"(?:\bD|\bhead_dim)\s*=\s*(\d+)\b", blob)
            if m:
                v_d = m.group(1)
        if v_e is None:
            m = re.search(r"\bE\s*=\s*(\d+)\b", blob)
            if m:
                v_e = m.group(1)
        if v_ep is None:
            m = re.search(r"\bEP\s*=\s*(\d+)\b", blob, re.I)
            if m:
                v_ep = m.group(1)
        if v_m is None and v_n is None and v_k is None:
            m_mnk = re.search(r"\b(\d{2,6})\s*[x×]\s*(\d{2,6})\s*[x×]\s*(\d{2,6})\b", blob)
            if m_mnk:
                v_m, v_n, v_k = m_mnk.group(1), m_mnk.group(2), m_mnk.group(3)
            else:
                m_cube = re.search(r"\b(\d{2,6})\s*[³\^3]\b", blob)
                if m_cube:
                    v_m = v_n = v_k = m_cube.group(1)

    parts: List[str] = []
    if v_bs is not None:
        parts.append(f"bs{v_bs}")
    if v_sq is not None:
        parts.append(f"seql{v_sq}")
    if v_kv is not None:
        parts.append(f"kv{v_kv}")
    if v_h is not None:
        parts.append(f"h{v_h}")
    if v_d is not None:
        parts.append(f"d{v_d}")
    if v_m is not None or v_n is not None or v_k is not None:
        if v_m is not None:
            parts.append(f"m{v_m}")
        if v_n is not None:
            parts.append(f"n{v_n}")
        if v_k is not None:
            parts.append(f"k{v_k}")
    if v_e is not None:
        parts.append(f"e{v_e}")
    if v_ep is not None:
        parts.append(f"ep{v_ep}")

    if parts:
        return "_".join(parts)

    # Fallback to observed launch geometry (CTAs / warps) if workload dims were absent
    al = other.get("alignment") if isinstance(other.get("alignment"), Mapping) else {}
    ctas = merged.get("grid_ctas") or merged.get("n_ctas") or al.get("n_ctas")
    wpc = merged.get("warps_per_cta") or merged.get("wpc") or al.get("warps_per_cta")
    g_parts: List[str] = []
    if ctas is not None:
        g_parts.append(f"grid{int(ctas)}")
    if wpc is not None:
        g_parts.append(f"w{int(wpc)}")
    return "_".join(g_parts) if g_parts else "shape"


def build_timeline_png_name(
    trace: Optional[Mapping[str, Any]] = None,
    *,
    model: Optional[str] = None,
    op: Optional[str] = None,
    bs: Optional[Any] = None,
    seql: Optional[Any] = None,
    kv: Optional[Any] = None,
    shape: Optional[str] = None,
    dims: Optional[Mapping[str, Any]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    source: Optional[str | Path] = None,
    spec: Optional[Mapping[str, Any]] = None,
    runtime: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[str | _dt.datetime] = None,
) -> str:
    """Build a descriptive timeline PNG filename: ``<model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png``.

    Encodes model name, operator (e.g. ``gemm``, ``attn``, ``moe``), workload dimensions
    (``bs``, ``seql``, ``kv``, ``m/n/k``, etc.), and date-time stamp so repeated runs never
    collide on a generic ``overview.png``.
    """
    other = trace.get("otherData") if (isinstance(trace, Mapping) and isinstance(trace.get("otherData"), Mapping)) else {}
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    workload = other.get("workload") if isinstance(other.get("workload"), Mapping) else {}
    rt = runtime if isinstance(runtime, Mapping) else {}
    sp = spec if isinstance(spec, Mapping) else {}

    kernel_hint = (
        meta.get("kernel")
        or rt.get("kernel_name")
        or sp.get("kernel_name")
        or (other.get("nsys_hw") or {}).get("kernel_name")
        or ""
    )
    texts = [
        str(title or ""),
        str(subtitle or ""),
        str(kernel_hint or ""),
        str(source or ""),
    ]

    model_tok = _extract_model_name(meta, other, workload, rt, sp, texts, model)
    op_tok = _extract_op_name(meta, other, workload, rt, sp, texts, source, op)
    shape_tok = _extract_shape_token(
        meta, other, workload, rt, sp, texts,
        bs=bs, seql=seql, kv=kv, shape=shape, dims=dims,
    )
    if isinstance(timestamp, _dt.datetime):
        ts_tok = timestamp.strftime("%Y%m%d_%H%M%S")
    elif isinstance(timestamp, str) and timestamp.strip():
        ts_tok = _sanitize_fname_token(timestamp, _dt.datetime.now().strftime("%Y%m%d_%H%M%S"), max_len=24)
    else:
        ts_tok = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{model_tok}_{op_tok}_{shape_tok}_{ts_tok}.png"


def resolve_timeline_png_path(
    out_png: Optional[str | Path],
    trace: Optional[Mapping[str, Any]] = None,
    *,
    default_dir: Optional[str | Path] = None,
    **name_kwargs: Any,
) -> Path:
    """Resolve an output PNG path; auto-generate ``<model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png``
    when ``out_png`` is omitted, ``"auto"``, or points to a directory."""
    raw = "" if out_png is None else str(out_png).strip()
    if not raw or raw.lower() == "auto":
        base = Path(default_dir) if default_dir is not None else Path.cwd()
        return base / build_timeline_png_name(trace, **name_kwargs)
    p = Path(out_png)
    if raw.endswith(("/", "\\")) or (p.exists() and p.is_dir()) or (not p.suffix and not p.exists()):
        return p / build_timeline_png_name(trace, **name_kwargs)
    return p

STAGE_COLORS: Dict[str, Tuple[str, str]] = {
    "load": ("#2E7D32", "TMA / Tile Load"),
    "mma": ("#1565C0", "Tensor Core MMA / WGMMA"),
    "softmax": ("#00ACC1", "Online Softmax"),
    "compute": ("#42A5F5", "Compute / ALU"),
    "epilogue": ("#FB8C00", "Epilogue"),
    "store": ("#D84315", "Store"),
    "wait": ("#F5A623", "Exposed Wait / Sync"),
    "barrier": ("#607D8B", "Route / Barrier Sync"),
    "communication": ("#8E24AA", "Communication"),
    "reduction": ("#C2185B", "Combine / Reduce"),
    "dispatch": ("#00838F", "Dispatch / Schedule"),
    "gemm": ("#1E88E5", "GEMM / MMA"),
    "attention": ("#1976D2", "Attention"),
    "moe": ("#6A1B9A", "MoE"),
}
# Globally disjoint sub-palette shades used when multiple distinct named stages share one semantic
# family (e.g., `mma_qk` vs `mma_pv`, `gmm1` vs `gmm2`, `dispatch` vs `combine`), so distinct code
# segments stay visually distinguishable with zero cross-family color collisions.
_SEMANTIC_SHADES: Dict[str, Tuple[str, ...]] = {
    "mma": ("#1565C0", "#0D47A1", "#283593", "#0277BD", "#3F51B5", "#01579B"),
    "gemm": ("#1E88E5", "#1976D2", "#2196F3", "#039BE5"),
    "load": ("#2E7D32", "#43A047", "#558B2F", "#1B5E20", "#689F38", "#33691E"),
    "compute": ("#42A5F5", "#29B6F6", "#5C6BC0", "#64B5F6", "#7986CB"),
    "softmax": ("#00ACC1", "#26C6DA", "#0097A7", "#4DD0E1"),
    "epilogue": ("#FB8C00", "#FFA726", "#F57C00", "#FFB74D"),
    "store": ("#D84315", "#E64A19", "#BF360C", "#FF7043"),
    "communication": ("#8E24AA", "#AB47BC", "#7B1FA2", "#9C27B0"),
    "reduction": ("#C2185B", "#D81B60", "#AD1457", "#E91E63"),
    "dispatch": ("#00838F", "#00897B", "#00695C", "#26A69A"),
    "wait": ("#F5A623", "#FFB300", "#E68A00"),
    "barrier": ("#607D8B", "#78909C", "#455A64"),
}
_OVERFLOW_SHADES: Tuple[str, ...] = (
    "#8D6E63", "#5D4037", "#AFB42B", "#827717", "#FBC02D", "#F9A825", "#EC407A", "#7E57C2",
)
_DEFAULT_COLOR = ("#37474F", "Other Stage")
_STAGE_DRAW_PRIORITY = {
    "barrier": 1, "load": 2, "dispatch": 3, "compute": 4,
    "softmax": 5, "mma": 6, "gemm": 6, "wait": 7, "communication": 7, "reduction": 7,
    "epilogue": 8, "store": 8,
}
_ASYNC_ENGINE_STYLE: Dict[str, Dict[str, Any]] = {
    "mem": {
        "short": "mem",
        # High-contrast 5-stop sequential ramp (Lime -> Aqua-Mint -> Vivid Emerald -> Forest -> Deep Pine)
        "depth_colors": ("#D9F99D", "#5EEAD4", "#10B981", "#047857", "#022C22"),
        "rail_bg": "#F1F5F9",
        "rail_ec": "#CBD5E1",
        "pin_color": "#047857",                              # crisp dark emerald ▼ issue pin
        "pin_label": "Mem Async Issue (v)",
        "depth_label": "Mem In-Flight Depth",
        "panel_b_color": "#0E8561",
    },
    "mma": {
        "short": "mma",
        # High-contrast sequential azure/cobalt ramp
        "depth_colors": ("#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A8A"),
        "rail_bg": "#F1F5F9",
        "rail_ec": "#CBD5E1",
        "pin_color": "#1D4ED8",                              # vivid cobalt blue ▼ issue pin
        "pin_label": "MMA Async Issue (v)",
        "depth_label": "MMA In-Flight Depth",
        "panel_b_color": "#1B4DB1",
    },
}


def _depth_bins_spec(eng: str, scale_max: int) -> List[Tuple[int, int, str, str]]:
    """Return explicit ``(lo_depth, hi_depth, hex_color, short_label)`` bins for ``depth >= 1``.

    Adapts to ``scale_max`` so every active depth step (including paired A+B ``cp.async`` steps
    ``2, 4, 6, 8`` on deep-pipelined Triton GEMM and exact integer steps ``1, 2, 3`` on CDNA3 / TMA)
    gets a perceptually distinct color AND an exact depth label in the legend.
    """
    smax = max(1, int(scale_max))
    if eng == "mma":
        if smax <= 1:
            return [(1, 1, "#2563EB", "d=1")]
        if smax == 2:
            return [(1, 1, "#93C5FD", "d=1"), (2, 2, "#1D4ED8", "d=2")]
        if smax == 3:
            return [(1, 1, "#BFDBFE", "d=1"), (2, 2, "#3B82F6", "d=2"), (3, 3, "#1E3A8A", "d=3")]
        return [
            (1, 1, "#DBEAFE", "d=1"),
            (2, 2, "#60A5FA", "d=2"),
            (3, 3, "#2563EB", "d=3"),
            (4, smax, "#1E3A8A", "d=4" if smax == 4 else f"d=4–{smax}"),
        ]
    # MEM engine (`cp.async` / `TMA` / `vmcnt`)
    if smax <= 1:
        return [(1, 1, "#10B981", "d=1")]
    if smax == 2:
        return [(1, 1, "#86EFAC", "d=1"), (2, 2, "#047857", "d=2")]
    if smax == 3:
        return [(1, 1, "#BEF264", "d=1"), (2, 2, "#10B981", "d=2"), (3, 3, "#064E3B", "d=3")]
    if smax == 4:
        return [
            (1, 1, "#D9F99D", "d=1"),
            (2, 2, "#5EEAD4", "d=2"),
            (3, 3, "#10B981", "d=3"),
            (4, 4, "#064E3B", "d=4"),
        ]
    if smax in (5, 6):
        return [
            (1, 1, "#D9F99D", "d=1"),
            (2, 2, "#6EE7B7", "d=2"),
            (3, 3, "#10B981", "d=3"),
            (4, 4, "#047857", "d=4"),
            (5, smax, "#022C22", "d=5" if smax == 5 else f"d=5–{smax}"),
        ]
    # smax >= 7 (e.g. Triton num_stages=5 -> 8 cp.async groups stepping 2, 4, 6, 8)
    return [
        (1, 2, "#D9F99D", "d=1–2"),
        (3, 4, "#5EEAD4", "d=3–4"),
        (5, 6, "#10B981", "d=5–6"),
        (7, smax, "#064E3B", f"d=7–{smax}"),
    ]


def _depth_to_bucket_color(eng: str, depth: int, scale_max: int) -> Tuple[int, str]:
    bins = _depth_bins_spec(eng, scale_max)
    d_i = max(1, int(depth))
    for idx, (lo, hi, col, _lbl) in enumerate(bins):
        if lo <= d_i <= hi:
            return idx, col
    return len(bins) - 1, bins[-1][2]

# Dual-color Inter-Node vs Intra-Node Fabric Palette (NVIDIA IB/RDMA vs NVLink, AMD RDMA vs xGMI).
# Warm crimson / amber is strictly reserved for Inter-Node RDMA / IB; cool teal / royal blue for
# Intra-Node xGMI / NVLink; emerald / forest green for GEMM1 / GEMM2; slate for Route / Sync;
# light violet for on-GPU Local Stage prep. Min RGB channel separation >= 60 across all pairs.
FABRIC_COLORS_BY_ROLE: Dict[str, str] = {
    "inter_dispatch": "#D62728",  # crimson red
    "inter_combine": "#F28E2B",   # warm amber / orange
    "intra_dispatch": "#009FB7",  # vivid teal / cyan
    "intra_combine": "#2563EB",   # royal blue
    "gemm1": "#16A34A",           # emerald green
    "gemm2": "#166534",           # deep forest green
    "route_sync": "#94A3B8",      # cool slate gray
    "local_prep": "#8B5CF6",      # light violet
}

_FABRIC_WAIT_DROP = frozenset({
    "gemm1_wait_payload", "gemm2_wait_stage1", "post_plan_wait",
    "scatter_ready_wait", "dispatch_wait_plan", "home_ready_wait",
    "gemm1_wait_route", "route_wait_meta",
    "internode_rdma_rxwatch", "rdma_wqe_cqe_service", "legacy_gda_service",
    "disp_idle_wait",
})
_GENERIC_LABELS = frozenset(STAGE_COLORS.keys()) | {"other", "stage", "unknown", ""}
_INTER_SLICE_RE = re.compile(
    r"(?:^|[_\-\s])(?:inter[_\-\s]?node|internode|rdma|roce|infiniband|ibgda|remote[_\-\s]?node)",
    re.I,
)
_INTRA_SLICE_RE = re.compile(
    r"(?:^|[_\-\s])(?:intra[_\-\s]?node|intranode|xgmi|nvlink|home[_\-]?combine|scatter[_\-]?xgmi|dispatch[_\-]?xgmi|p2p)",
    re.I,
)
_COMBINE_RE = re.compile(r"(?:combine|reduce|return|scatter|push|store|epilogue)", re.I)
_DISPATCH_RE = re.compile(r"(?:dispatch|pull|alltoall|a2a|send|recv|load)", re.I)
_GEMM1_RE = re.compile(r"(?:gemm1|gmm1|fc1|swiglu|gate_up|l1_gemm|mma_qk|qk_mma)", re.I)
_GEMM2_RE = re.compile(r"(?:gemm2|gmm2|fc2|down_proj|l2_gemm|mma_pv|pv_mma)", re.I)
_LOCAL_PREP_RE = re.compile(r"(?:local[_\-\s]?stage|local[_\-\s]?prep|quant[_\-\s]?prep)", re.I)
_ROUTE_SYNC_RE = re.compile(r"(?:route|ballot|plan|sync|barrier|meta|cleanup)", re.I)


def _short_kernel_name(raw: str, max_len: int = 72) -> str:
    s = str(raw or "").strip()
    if not s:
        return "WeaveProfiler Kernel"
    s = s.split("(")[0].strip()
    if s.startswith("void "):
        s = s[5:].strip()
    s = re.sub(r"^_(?:fi|wp)_(?:inst|l0)_", "", s)
    if "<" in s and len(s) > max_len:
        idx = s.find("<")
        prefix = s[:idx]
        suffix = s[s.rfind(">") + 1:] if ">" in s else ""
        s = f"{prefix}<...>{suffix}"
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def _pack_concurrent_ctas(spans: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, int], int]:
    """Pack overlapping CTAs on one SM/CU into the minimal number of non-overlapping resident slots."""
    cta_win: Dict[int, List[float]] = {}
    for sp in spans:
        c = int(sp["cta"])
        w = cta_win.get(c)
        if w is None:
            cta_win[c] = [float(sp["ts"]), float(sp["end"])]
        else:
            if sp["ts"] < w[0]:
                w[0] = float(sp["ts"])
            if sp["end"] > w[1]:
                w[1] = float(sp["end"])
    ordered = sorted(cta_win.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    slot_ends: List[float] = []
    cta_to_slot: Dict[int, int] = {}
    for cta, (t0, t1) in ordered:
        chosen = None
        for idx, end_t in enumerate(slot_ends):
            if t0 >= end_t - 1e-6:
                chosen = idx
                slot_ends[idx] = t1
                break
        if chosen is None:
            chosen = len(slot_ends)
            slot_ends.append(t1)
        cta_to_slot[cta] = chosen
    return cta_to_slot, max(1, len(slot_ends))


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _load_trace(trace_or_path: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(trace_or_path, Mapping):
        return dict(trace_or_path)
    p = Path(trace_or_path)
    raw = p.read_bytes()
    if p.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _classify_hw_metric(name: str) -> Optional[str]:
    low = name.lower()
    if "rdma_util" in low or "inter-node rdma" in low or "infiniband" in low or "roce" in low:
        return "inter_rdma"
    if "tensor" in low or "compute_util" in low or "matrix core" in low or "mfma" in low:
        return "tensor_active"
    if "sms active" in low or "sm active" in low or "sm_active" in low or "active mi308x cu" in low or ("active" in low and "cu" in low):
        return "sm_active"
    if "sm issue" in low or "warp issue" in low or "issue" in low:
        return "sm_issue"
    if "memory_util" in low or "memory util" in low or ("hbm" in low and ("read+write" in low or ("read" not in low and "write" not in low))):
        return "dram_bw_direct"
    if "dram read" in low or ("hbm" in low and "read" in low):
        return "dram_read"
    if "dram write" in low or ("hbm" in low and "write" in low):
        return "dram_write"
    if "communication_rx_util" in low or ("xgmi" in low and "rx" in low):
        return "nvlink_rx_total"
    if "communication_tx_util" in low or ("xgmi" in low and "tx" in low):
        return "nvlink_tx_total"
    if "communication_util" in low or "xgmi" in low:
        return "intra_comm_direct"
    if "nvlink" in low:
        if "protocol" in low:
            return None
        if "rx" in low and "request" in low:
            return "nvlink_rx_req"
        if "rx" in low and "response" in low:
            return "nvlink_rx_rsp"
        if "tx" in low and "request" in low:
            return "nvlink_tx_req"
        if "tx" in low and "response" in low:
            return "nvlink_tx_rsp"
        if "rx" in low:
            return "nvlink_rx_total"
        if "tx" in low:
            return "nvlink_tx_total"
        return "nvlink_user"
    return None


def _pretty_hw_curve_label(raw_name: str, role: str, *, is_amd: bool, multi_node: bool) -> Tuple[str, str]:
    """Turn a raw Perfetto counter track name into a publication-grade legend label + short chip title."""
    m = re.match(r"^[a-z0-9_]+\s*\[(.+)\]\s*$", str(raw_name or "").strip(), re.I)
    bracket = m.group(1).strip() if m else ""
    if role == "tensor_active":
        if bracket:
            b_clean = bracket
            if is_amd:
                b_clean = re.sub(
                    r"%\s*of\s*(?:606|398)(?:\.\d+)?\s*T(?:\s*(?:MI308X|SKU))*",
                    "Pipe Active %",
                    b_clean,
                    flags=re.I,
                )
                b_clean = re.sub(
                    r"\s*\(398(?:\.\d+)?\s*T\s*(?:MI308X\s*)?SKU\)",
                    "",
                    b_clean,
                    flags=re.I,
                ).strip()
            return f"Compute — {b_clean}", "Compute"
        return ("Compute — Matrix Core / MFMA Pipe Active %" if is_amd else "Tensor Active / MMA Pipe %"), "Compute"
    if role == "dram_bw":
        if bracket:
            return f"HBM — {bracket}", "HBM BW"
        return "HBM / DRAM Bandwidth (R+W) %", "HBM BW"
    if role == "nvlink_tx":
        if is_amd:
            prefix = "Intra-Node xGMI TX" if multi_node else "xGMI TX"
            if bracket:
                b_clean = re.sub(r"^(?:Intra-Node\s+)?xGMI\s*TX\s*(?:\(Out\))?\s*(?:—|-)?\s*", "", bracket, flags=re.I).strip()
                return (f"{prefix} (Out) — {b_clean}" if b_clean else f"{prefix} (Out) %"), prefix
            return f"{prefix} (Out) %", prefix
        prefix = "Intra-Node NVLink TX" if multi_node else "NVLink TX"
        return f"{prefix} (Egress User Data) %", prefix
    if role == "nvlink_rx":
        if is_amd:
            prefix = "Intra-Node xGMI RX" if multi_node else "xGMI RX"
            if bracket:
                b_clean = re.sub(r"^(?:Intra-Node\s+)?xGMI\s*RX\s*(?:\(In\))?\s*(?:—|-)?\s*", "", bracket, flags=re.I).strip()
                return (f"{prefix} (In) — {b_clean}" if b_clean else f"{prefix} (In) %"), prefix
            return f"{prefix} (In) %", prefix
        prefix = "Intra-Node NVLink RX" if multi_node else "NVLink RX"
        return f"{prefix} (Ingress User Data) %", prefix
    if role == "nvlink_bw":
        fabric_nm = "xGMI" if is_amd else "NVLink"
        prefix = f"Intra-Node {fabric_nm}" if multi_node else fabric_nm
        if bracket:
            b_clean = re.sub(r"^(?:Intra-Node\s+)?(?:xGMI|NVLink\d*)\s*", "", bracket, flags=re.I).strip()
            return (f"{prefix} — {b_clean}" if b_clean else prefix), prefix
        return f"{prefix} — P2P / User Data %", prefix
    if role == "inter_rdma":
        if bracket:
            b_clean = re.sub(r"^Inter-Node\s+RDMA\s*(?:\(NIC\)|NIC)?\s*", "", bracket, flags=re.I).strip()
            return (f"Inter-Node RDMA (NIC) — {b_clean}" if b_clean else "Inter-Node RDMA (NIC) %"), "Inter-Node RDMA"
        return "Inter-Node RDMA (NIC) %", "Inter-Node RDMA"
    if role == "sm_active":
        unit_pl = "CUs" if is_amd else "SMs"
        if bracket:
            m_of = re.search(r"%\s*of\s*\d+", bracket, re.I)
            if m_of:
                return f"Active — {unit_pl} {m_of.group(0)}", f"Active {unit_pl}"
            return f"Active — {bracket}", f"Active {unit_pl}"
        return f"Active {unit_pl} %", f"Active {unit_pl}"
    if role == "sm_issue":
        return "SM Issue %", "SM Issue"
    return str(raw_name), role


def _classify_fabric_role(sp: Mapping[str, Any], *, is_amd: bool) -> Optional[str]:
    """Classify a span into a dual-fabric display role when rendering multi-node traces."""
    raw = str(sp.get("label") or sp.get("raw_name") or sp.get("sem") or "").strip()
    low = raw.lower()
    fab = str(sp.get("fabric") or "").strip().lower()
    sem = str(sp.get("sem") or "").lower()

    if fab.startswith("inter_node") or _INTER_SLICE_RE.search(raw):
        if _COMBINE_RE.search(low) or sem in ("reduction", "store", "epilogue"):
            return "inter_combine"
        return "inter_dispatch"
    if _LOCAL_PREP_RE.search(low):
        return "local_prep"
    if fab.startswith("intra_node") or _INTRA_SLICE_RE.search(raw):
        if _COMBINE_RE.search(low) or sem in ("reduction", "store", "epilogue"):
            return "intra_combine"
        return "intra_dispatch"
    if _GEMM1_RE.search(low):
        return "gemm1"
    if _GEMM2_RE.search(low):
        return "gemm2"
    if _ROUTE_SYNC_RE.search(low) or sem in ("barrier", "wait"):
        return "route_sync"
    if sem in ("mma", "gemm"):
        return "gemm1"
    if sem == "reduction":
        return "intra_combine"
    if sem in ("communication", "dispatch"):
        return "intra_dispatch"
    if sem == "load":
        return "local_prep"
    return None


def _fabric_role_label(role: str, *, is_amd: bool, use_ib: bool = False) -> str:
    intra_nm = "xGMI" if is_amd else "NVLink"
    inter_nm = "IB/RDMA" if use_ib else "RDMA"
    return {
        "inter_dispatch": f"Inter-Node {inter_nm} Dispatch",
        "inter_combine": f"Inter-Node {inter_nm} Combine",
        "intra_dispatch": f"Intra-Node {intra_nm} Dispatch",
        "intra_combine": f"Intra-Node {intra_nm} Combine",
        "gemm1": "GEMM1 (FC1+SwiGLU)",
        "gemm2": "GEMM2 (FC2)",
        "route_sync": "Route / Sync",
        "local_prep": "Local Stage (prep)",
    }[role]


def _humanize_stage_label(raw_label: str, sem: str) -> str:
    """Preserve meaningful domain/stage names instead of collapsing them to bare `load` / `mma`."""
    lbl = str(raw_label or "").strip()
    low = lbl.lower()
    if not lbl or low in _GENERIC_LABELS:
        return STAGE_COLORS.get(sem, _DEFAULT_COLOR)[1]
    # Already a richly formatted human display label (contains spaces, parens, arrows, or dashes)
    if any(ch in lbl for ch in (" ", "(", "/", "—", "->")):
        return lbl
    # Known rich compound stage tokens
    pretty_known = {
        "compute_gemm1_swiglu": "GEMM1 (FC1+SwiGLU)",
        "gemm1_tile_swiglu": "GEMM1 (FC1+SwiGLU)",
        "compute_gemm2_mfma": "GEMM2 (FC2)",
        "gemm2_tile_mfma": "GEMM2 (FC2)",
        "intranode_localstage": "Local Stage (prep)",
        "intranode_xgmi_pull": "Intra-Node xGMI Dispatch",
        "intranode_xgmi_scatterpush": "Intra-Node xGMI Combine",
        "intranode_homecombine": "Intra-Node xGMI Combine",
        "internode_rdma_dispatch": "Inter-Node RDMA Dispatch",
        "internode_rdma_return": "Inter-Node RDMA Combine",
        "internode_partial_reduce": "Inter-Node RDMA Combine",
        "route_plan_local": "Route / Sync",
        "route_plan_remote": "Route / Sync",
    }
    if low in pretty_known:
        return pretty_known[low]
    return f"{lbl} ({sem})" if low != sem else STAGE_COLORS.get(sem, _DEFAULT_COLOR)[1]


def _detect_platform(other: Mapping[str, Any], events: Sequence[Mapping[str, Any]], hw_labels: Mapping[str, str]) -> bool:
    """Return True when the trace belongs to an AMD GPU (CU / wave / rocprof), else False (NVIDIA SM / warp)."""
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    plat = str(other.get("platform") or meta.get("platform") or "").lower()
    if plat == "amd":
        return True
    if plat in ("nv", "nvidia"):
        return False
    card = str(other.get("card") or meta.get("card") or "").lower()
    arch = str(other.get("arch") or meta.get("arch") or "").lower()
    if "mi308" in card or "mi300" in card or arch.startswith("gfx"):
        return True
    if any("xgmi" in v.lower() or "mi308" in v.lower() or "matrix core" in v.lower() for v in hw_labels.values()):
        return True
    for ev in events[:120]:
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            nm = str((ev.get("args") or {}).get("name") or "")
            if re.match(r"^\s*CU\s*\d+", nm):
                return True
    return False


def _parse_trace(trace: Mapping[str, Any]) -> Dict[str, Any]:
    events = trace.get("traceEvents") or []
    other = dict(trace.get("otherData") or {})
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    hw_curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    hw_labels: Dict[str, str] = {}
    nsys_kernel: Optional[Dict[str, Any]] = None
    info_dur_us: Optional[float] = None
    sw_spans: List[Dict[str, Any]] = []
    sw_marks: List[Dict[str, Any]] = []
    sm_set: set[int] = set()
    stages_seen: Dict[str, float] = defaultdict(float)
    meta_warps_by_sm: Dict[int, Dict[int, str]] = defaultdict(dict)
    proc_names: Dict[int, str] = {}
    for ev in events:
        ph = ev.get("ph")
        if ph == "M" and ev.get("name") == "process_name":
            pid = ev.get("pid")
            if isinstance(pid, int):
                proc_names[pid] = str((ev.get("args") or {}).get("name") or "")

    def _pid_to_unit(pid_val: Any) -> Optional[int]:
        if not isinstance(pid_val, int):
            return None
        if pid_val in (1, 2, 90000):
            return None
        pname = proc_names.get(pid_val, "")
        m_unit = re.match(r"^\s*(?:SM|CU)\s*(\d+)\b", pname, re.I)
        if m_unit:
            return int(m_unit.group(1))
        if pid_val >= 100000:
            return pid_val - 100000
        if pid_val >= 100:
            return pid_val - 100
        return pid_val

    has_high_unit_pids = any(
        ev.get("ph") == "X" and isinstance(ev.get("pid"), int) and int(ev["pid"]) >= 100
        and int(ev["pid"]) != 90000
        and str(ev.get("cat") or "") not in ("nsys_kernel", "info", "hardware", "residency")
        for ev in events
    )

    for ev in events:
        ph = ev.get("ph")
        if ph == "M" and ev.get("name") == "process_name":
            continue
        elif ph == "M" and ev.get("name") == "thread_name":
            pid = ev.get("pid")
            tid = ev.get("tid")
            tname = str((ev.get("args") or {}).get("name") or "")
            if isinstance(pid, int) and pid not in (1, 90000) and (not has_high_unit_pids or pid >= 100):
                sm = _pid_to_unit(pid)
                if sm is None:
                    continue
                m = re.search(r"\b(?:[wW]|warp\s*|wave64\s*|wave\s*)(\d+)\b", tname, re.I)
                if m:
                    w_id = int(m.group(1))
                    m_role_bracket = re.search(r"\[(Compute|Comm|Dispatch|Scatter|Math|Epilogue|DMA|Producer|Consumer)[^\]]*\]", tname, re.I)
                    m_role_paren = re.search(r"\(([A-Za-z0-9_/\-\s]+)\)", tname)
                    if m_role_bracket:
                        role_part = m_role_bracket.group(1).strip()
                    elif m_role_paren:
                        role_part = m_role_paren.group(1).strip()
                    elif "·" in tname:
                        cand_r = tname.split("·")[-1].strip()
                        if re.match(r"^(?:slot\s*\d+|实例\s*\d+|instance\s*\d+)$", cand_r, re.I) or any(ord(ch) > 127 for ch in cand_r):
                            role_part = f"w{w_id:02d}"
                        else:
                            role_part = cand_r
                    elif "/" in tname:
                        role_part = tname.split("/")[-1].strip()
                    else:
                        role_part = f"w{w_id:02d}"
                    if re.fullmatch(r"[wW]\d+", role_part) or any(ord(ch) > 127 for ch in role_part):
                        role_part = f"w{w_id:02d}"
                    meta_warps_by_sm[sm].setdefault(w_id, role_part)
                    sm_set.add(sm)
        elif ph == "C":
            cname = str(ev.get("name") or "")
            key = _classify_hw_metric(cname)
            if key is None:
                continue
            ts = float(ev.get("ts") or 0.0)
            args = ev.get("args") or {}
            val = args.get("pct", args.get("value", 0.0))
            hw_curves[key].append((ts, float(val)))
            hw_labels.setdefault(key, cname)
        elif ph == "X":
            cat = str(ev.get("cat") or "")
            pid = ev.get("pid")
            args = ev.get("args") or {}
            kind = str(args.get("kind") or "").lower()
            if cat in ("nsys_kernel", "hardware") or kind == "dispatch" or pid == 90000:
                if nsys_kernel is None or cat == "nsys_kernel":
                    nsys_kernel = dict(ev)
                continue
            if cat == "info":
                d_info = float(ev.get("dur") or 0.0)
                if d_info > 0.0 and info_dur_us is None:
                    info_dur_us = d_info
                continue
            if cat == "residency" or kind == "residency":
                r_sm = args.get("sm")
                if r_sm is None:
                    r_sm = _pid_to_unit(pid)
                r_w = args.get("warp", args.get("wave"))
                if r_sm is not None and r_w is not None:
                    r_sm_i, r_w_i = int(r_sm), int(r_w)
                    r_role = str(args.get("role") or f"w{r_w_i:02d}")
                    meta_warps_by_sm[r_sm_i][r_w_i] = r_role
                    sm_set.add(r_sm_i)
                continue
            # Skip non-unit macro summary tracks (e.g. pid=1 hardware header, pid=2 macro stage summary)
            if has_high_unit_pids and isinstance(pid, int) and pid < 100:
                continue
            ts = float(ev.get("ts") or 0.0)
            dur = float(ev.get("dur") or 0.0)
            if dur <= 0.0 or not math.isfinite(dur):
                continue
            raw_name = str(ev.get("name") or "").strip()
            disp_label = str(
                args.get("display_label")
                or args.get("original_label")
                or raw_name
                or args.get("semantic_id")
                or "compute"
            ).strip()
            sem_raw = str(args.get("semantic_id") or cat or raw_name or "compute").lower()
            if sem_raw not in STAGE_COLORS:
                from .contract import resolve_semantic_id
                sem = resolve_semantic_id(disp_label or raw_name or sem_raw)
            else:
                sem = sem_raw
            sm = args.get("sm")
            if sm is None and args.get("cu") is not None and isinstance(pid, int) and pid >= 100:
                sm = _pid_to_unit(pid)
            elif sm is None:
                u = _pid_to_unit(pid)
                sm = u if u is not None else 0
            else:
                sm = int(sm)
            if args.get("warp") is not None:
                warp = int(args["warp"])
            elif args.get("wave") is not None:
                warp = int(args["wave"])
            elif isinstance(ev.get("tid"), int) and has_high_unit_pids and int(ev["tid"]) >= 1:
                warp = int(ev["tid"]) - 1 if (sm in meta_warps_by_sm and 0 in meta_warps_by_sm[sm]) else int(ev["tid"])
            else:
                warp = 0
            cta = int(args.get("cta", 0))
            role = str(args.get("role") or meta_warps_by_sm.get(sm, {}).get(warp) or f"w{warp:02d}")
            fabric = str(args.get("fabric") or "").strip().lower()
            async_engine = str(args.get("async_engine") or "").strip().lower()
            if async_engine not in ("mem", "mma"):
                async_engine = ""
            sm_set.add(sm)
            stages_seen[sem] += dur
            sw_spans.append({
                "ts": ts, "dur": dur, "end": ts + dur, "sem": sem,
                "label": disp_label, "raw_name": raw_name, "fabric": fabric,
                "async_engine": async_engine,
                "sm": sm, "warp": warp, "cta": cta, "role": role,
            })
        elif ph == "i":
            pid = ev.get("pid")
            args = ev.get("args") or {}
            if has_high_unit_pids and isinstance(pid, int) and pid < 100:
                continue
            ts = float(ev.get("ts") or 0.0)
            if not math.isfinite(ts):
                continue
            raw_name = str(ev.get("name") or "").strip()
            disp_label = str(
                args.get("display_label")
                or args.get("original_label")
                or raw_name
                or args.get("semantic_id")
                or "load"
            ).strip()
            sem_raw = str(args.get("semantic_id") or ev.get("cat") or raw_name or "load").lower()
            if sem_raw not in STAGE_COLORS:
                from .contract import resolve_semantic_id
                sem = resolve_semantic_id(disp_label or raw_name or sem_raw)
            else:
                sem = sem_raw
            sm = args.get("sm")
            if sm is None:
                u = _pid_to_unit(pid)
                sm = u if u is not None else 0
            else:
                sm = int(sm)
            if args.get("warp") is not None:
                warp = int(args["warp"])
            elif args.get("wave") is not None:
                warp = int(args["wave"])
            else:
                warp = 0
            cta = int(args.get("cta", 0))
            role = str(args.get("role") or meta_warps_by_sm.get(sm, {}).get(warp) or f"w{warp:02d}")
            async_engine = str(args.get("async_engine") or "").strip().lower()
            if async_engine not in ("mem", "mma"):
                async_engine = "mma" if sem in ("mma", "gemm") else ("mem" if sem in ("load", "dispatch", "store", "communication") else "mem")
            sm_set.add(sm)
            sw_marks.append({
                "ts": ts, "sem": sem, "label": disp_label, "raw_name": raw_name,
                "async_engine": async_engine, "sm": sm, "warp": warp, "cta": cta, "role": role,
            })

    for k in list(hw_curves.keys()):
        by_ts: Dict[float, float] = {}
        for t, v in hw_curves[k]:
            prev = by_ts.get(t)
            by_ts[t] = v if prev is None else max(prev, v)
        hw_curves[k] = sorted(by_ts.items(), key=lambda x: x[0])

    # Combine DRAM read + write (shared half-duplex HBM bus) or use direct memory_util curve
    if "dram_bw_direct" in hw_curves:
        hw_curves["dram_bw"] = list(hw_curves["dram_bw_direct"])
        hw_labels["dram_bw"] = hw_labels.get("dram_bw_direct", "memory_util")
    elif "dram_read" in hw_curves or "dram_write" in hw_curves:
        by_t: Dict[float, float] = defaultdict(float)
        for k in ("dram_read", "dram_write"):
            for t, v in hw_curves.get(k, ()):
                by_t[round(t, 3)] += v
        hw_curves["dram_bw"] = [(t, min(100.0, v)) for t, v in sorted(by_t.items())]

    # NVLink TX (egress) and NVLink RX (ingress) are TWO independent full-duplex physical links:
    # sum Request + Response WITHIN each direction only; NEVER sum or merge TX with RX!
    if "nvlink_tx_total" in hw_curves:
        hw_curves["nvlink_tx"] = list(hw_curves["nvlink_tx_total"])
        hw_labels["nvlink_tx"] = hw_labels.get("nvlink_tx_total", "NVLink TX User Data Total")
    elif "nvlink_tx_req" in hw_curves or "nvlink_tx_rsp" in hw_curves:
        tx_t: Dict[float, float] = defaultdict(float)
        for k in ("nvlink_tx_req", "nvlink_tx_rsp"):
            for t, v in hw_curves.get(k, ()):
                tx_t[round(t, 3)] += v
        if tx_t:
            hw_curves["nvlink_tx"] = [(t, min(100.0, v)) for t, v in sorted(tx_t.items())]
            hw_labels["nvlink_tx"] = "NVLink TX User Data (Req+Rsp)"

    if "nvlink_rx_total" in hw_curves:
        hw_curves["nvlink_rx"] = list(hw_curves["nvlink_rx_total"])
        hw_labels["nvlink_rx"] = hw_labels.get("nvlink_rx_total", "NVLink RX User Data Total")
    elif "nvlink_rx_req" in hw_curves or "nvlink_rx_rsp" in hw_curves:
        rx_t: Dict[float, float] = defaultdict(float)
        for k in ("nvlink_rx_req", "nvlink_rx_rsp"):
            for t, v in hw_curves.get(k, ()):
                rx_t[round(t, 3)] += v
        if rx_t:
            hw_curves["nvlink_rx"] = [(t, min(100.0, v)) for t, v in sorted(rx_t.items())]
            hw_labels["nvlink_rx"] = "NVLink RX User Data (Req+Rsp)"

    # Single-series intra-node fabric curve (e.g. AMD xGMI P2P or unsplit generic counter)
    if "intra_comm_direct" in hw_curves:
        hw_curves["nvlink_bw"] = list(hw_curves["intra_comm_direct"])
        hw_labels["nvlink_bw"] = hw_labels.get("intra_comm_direct", "communication_util")
    elif "nvlink_user" in hw_curves and "nvlink_tx" not in hw_curves and "nvlink_rx" not in hw_curves:
        hw_curves["nvlink_bw"] = list(hw_curves["nvlink_user"])

    is_amd = _detect_platform(other, events, hw_labels)
    has_inter_slices = any(
        sp["fabric"].startswith("inter_node") or bool(_INTER_SLICE_RE.search(sp["label"]))
        for sp in sw_spans
    )
    has_intra_slices = any(
        sp["fabric"].startswith("intra_node") or bool(_INTRA_SLICE_RE.search(sp["label"]))
        for sp in sw_spans
    )
    ep_val = int(meta.get("ep") or other.get("ep") or 0)
    multi_node = bool(ep_val > 8 or meta.get("multi_node") or "inter_rdma" in hw_curves or has_inter_slices)
    dual_fabric = bool(not other.get("unified_stage_mode") and has_inter_slices and (has_intra_slices or multi_node))
    use_ib = any(
        sp["fabric"] == "inter_node_ib" or bool(re.search(r"\b(?:infiniband|ibgda|ib_)\b", sp["label"], re.I))
        for sp in sw_spans
    )

    if dual_fabric:
        # Filter pure polling/background wait slices in multi-node fabric timelines so active fabric bars stay crisp.
        sw_spans = [sp for sp in sw_spans if sp["label"].lower() not in _FABRIC_WAIT_DROP]

    # Drop enclosing macro-envelope spans (e.g. outer `gemm`/`attention` wrapper) when inner
    # per-stage spans exist, so the envelope does not paint a solid bar over finer stages.
    _MACRO_ENVELOPES = frozenset({"gemm", "attention", "moe", "megakernel"})
    if any(sp["sem"] not in _MACRO_ENVELOPES for sp in sw_spans) and any(sp["sem"] in _MACRO_ENVELOPES for sp in sw_spans):
        sw_spans = [sp for sp in sw_spans if sp["sem"] not in _MACRO_ENVELOPES]
        stages_seen = defaultdict(float)
        for sp in sw_spans:
            stages_seen[sp["sem"]] += sp["dur"]

    # Optional CTA-level cross-warp producer->consumer TMA in-flight depth strip:
    # When fi_lite_decode computed deduplicated [t_issue, min_w t_wait_exit] segments per CTA,
    # surface them on the producer warp's async MEM rail (only when the producer has no native
    # async MEM spans of its own) so 1-to-N TMA depth renders cleanly without fanout double-counting.
    cw_ctas = other.get("cross_warp_inflight_ctas")
    if isinstance(cw_ctas, Mapping) and cw_ctas:
        existing_prod_mem = {(sp["sm"], sp["warp"], sp["cta"]) for sp in sw_spans if sp.get("async_engine") == "mem"}
        cw_shift_us = 0.0
        al_info = other.get("alignment") or {}
        if al_info.get("offset_ns") is not None and other.get("observation_start_ns") is not None:
            # Put segments onto the exact same absolute clock frame as aligned traceEvents before t_origin shift
            cw_shift_us = (int(other["observation_start_ns"]) + int(al_info["offset_ns"])) / 1000.0
        for _key, entry in cw_ctas.items():
            if not isinstance(entry, Mapping):
                continue
            sm_c = entry.get("sm")
            pw = entry.get("producer_warp")
            cta_c = entry.get("cta")
            segs_us = entry.get("segments_us") or []
            if sm_c is None or pw is None or cta_c is None or not segs_us:
                continue
            sm_i, pw_i, cta_i = int(sm_c), int(pw), int(cta_c)
            if (sm_i, pw_i, cta_i) in existing_prod_mem:
                continue
            role_p = str(meta_warps_by_sm.get(sm_i, {}).get(pw_i) or "tma_producer")
            sm_set.add(sm_i)
            for s0, s1, dp in segs_us:
                ts_a = max(0.0, float(s0) + cw_shift_us)
                en_a = max(ts_a, float(s1) + cw_shift_us)
                if en_a <= ts_a:
                    continue
                for _k_dp in range(max(1, int(dp))):
                    sw_spans.append({
                        "ts": ts_a, "dur": en_a - ts_a, "end": en_a, "sem": "load",
                        "label": "Cross-Warp TMA In-Flight", "raw_name": "load", "fabric": "",
                        "async_engine": "mem", "sm": sm_i, "warp": pw_i, "cta": cta_i, "role": role_p,
                    })

    # Normalize all timestamps relative to kernel start (t_origin) so aligned traces with
    # absolute nsys/rocprof timestamps (e.g. 3.4e6 us) plot cleanly on [0, kernel_dur_us].
    t_candidates: List[float] = []
    if nsys_kernel is not None and nsys_kernel.get("ts") is not None:
        k_ts = float(nsys_kernel["ts"])
        if math.isfinite(k_ts):
            t_candidates.append(k_ts)
    if sw_spans:
        min_sw = min(sp["ts"] for sp in sw_spans)
        if math.isfinite(min_sw):
            t_candidates.append(min_sw)
    if sw_marks:
        min_mk = min(mk["ts"] for mk in sw_marks)
        if math.isfinite(min_mk):
            t_candidates.append(min_mk)
    t_origin = min(t_candidates) if t_candidates else 0.0
    if t_origin > 1.0 or t_origin < -1.0:
        for sp in sw_spans:
            sp["ts"] = max(0.0, sp["ts"] - t_origin)
            sp["end"] = max(sp["ts"], sp["end"] - t_origin)
        for mk in sw_marks:
            mk["ts"] = max(0.0, mk["ts"] - t_origin)
        for k in list(hw_curves.keys()):
            hw_curves[k] = [(t - t_origin, v) for t, v in hw_curves[k]]
        if nsys_kernel is not None and nsys_kernel.get("ts") is not None:
            nsys_kernel["ts"] = max(0.0, float(nsys_kernel["ts"]) - t_origin)

    # Auto-promote sub-microsecond non-blocking `load` issue pulses (e.g. Triton `tl.load` -> `cp.async`
    # taking ~40 ns inside a K-loop) into crisp fixed-pixel `mem` async issue pins so they are never
    # drowned out to sub-pixel hairlines or allowed to overwrite MMA bars.
    if not dual_fabric and sw_spans:
        comp_durs = sorted(sp["dur"] for sp in sw_spans if sp["sem"] in ("mma", "gemm", "compute") and not sp["async_engine"])
        med_comp = comp_durs[len(comp_durs) // 2] if comp_durs else 0.0
        if med_comp >= 0.8:
            load_durs_by_lbl: Dict[str, List[float]] = defaultdict(list)
            for sp in sw_spans:
                if sp["sem"] == "load" and not sp["async_engine"]:
                    load_durs_by_lbl[sp["label"]].append(sp["dur"])
            promote_load_lbls = set()
            for lbl_k, dl in load_durs_by_lbl.items():
                dl.sort()
                med_l = dl[len(dl) // 2]
                if med_l <= 0.30 and med_l <= 0.12 * med_comp:
                    promote_load_lbls.add(lbl_k)
            if promote_load_lbls:
                kept_spans = []
                for sp in sw_spans:
                    if sp["sem"] == "load" and not sp["async_engine"] and sp["label"] in promote_load_lbls:
                        sw_marks.append({
                            "ts": sp["ts"], "sem": "load", "label": sp["label"], "raw_name": sp["raw_name"],
                            "async_engine": "mem", "sm": sp["sm"], "warp": sp["warp"],
                            "cta": sp["cta"], "role": sp["role"],
                        })
                    else:
                        kept_spans.append(sp)
                sw_spans = kept_spans
                stages_seen = defaultdict(float)
                for sp in sw_spans:
                    if not sp["async_engine"]:
                        stages_seen[sp["sem"]] += sp["dur"]

    async_spans = [sp for sp in sw_spans if sp["async_engine"] in ("mem", "mma")]
    main_spans = [sp for sp in sw_spans if not sp["async_engine"]]
    stages_seen = defaultdict(float)
    for sp in main_spans:
        stages_seen[sp["sem"]] += sp["dur"]

    sw_end = max(
        [s["end"] for s in sw_spans] + [m["ts"] for m in sw_marks],
        default=0.0,
    )
    k_dur = (
        float(nsys_kernel["dur"]) if (nsys_kernel and nsys_kernel.get("dur"))
        else (info_dur_us if info_dur_us else sw_end)
    )
    t_max = max(sw_end, k_dur, 1.0)

    return {
        "hw_curves": dict(hw_curves),
        "hw_labels": hw_labels,
        "nsys_kernel": nsys_kernel,
        "sw_spans": sw_spans,
        "main_spans": main_spans,
        "async_spans": async_spans,
        "sw_marks": sw_marks,
        "sms": sorted(sm_set),
        "meta_warps_by_sm": {k: dict(v) for k, v in meta_warps_by_sm.items()},
        "stages_seen": dict(stages_seen),
        "t_max": t_max,
        "k_dur": k_dur,
        "sw_end": sw_end,
        "other": other,
        "is_amd": is_amd,
        "multi_node": multi_node,
        "dual_fabric": dual_fabric,
        "use_ib": use_ib,
        "ep": ep_val,
    }


def _compute_panel_c_role_labels(
    rep_warps: Sequence[Tuple[int, str]],
    rep_spans: Sequence[Mapping[str, Any]],
    all_spans: Sequence[Mapping[str, Any]],
    *,
    dual_fabric: bool,
    is_amd: bool,
) -> List[str]:
    """Build clean per-warp / per-wave Y-axis labels, detecting Streaming Reduce on comm lanes."""
    if not rep_warps:
        return ["W00"]
    if not dual_fabric:
        return [f"W{w:02d} · {r}" for w, r in rep_warps]

    # Detect streaming Reduce on comm lanes using per-(unit, lane) comm identification so
    # pure-comm CTAs (where W00..W03 are also comm lanes) never pollute `tail_comb_start`.
    comm_tracks: set[Tuple[int, int]] = set()
    for sp in all_spans:
        frole = sp.get("_frole")
        rl = str(sp.get("role") or "").lower()
        if frole in ("inter_dispatch", "intra_dispatch") or any(k in rl for k in ("comm", "dispatch", "scatter")):
            comm_tracks.add((int(sp["sm"]), int(sp["warp"])))
    rep_sm_id = int(rep_spans[0]["sm"]) if rep_spans else 0
    for w, r in rep_warps:
        if any(k in r.lower() for k in ("comm", "dispatch", "scatter")):
            comm_tracks.add((rep_sm_id, int(w)))
    non_comm_comb = [
        float(sp["ts"]) for sp in all_spans
        if (int(sp["sm"]), int(sp["warp"])) not in comm_tracks
        and ("combine" in sp["label"].lower() or "reduce" in sp["label"].lower())
    ]
    tail_comb_start = min(non_comm_comb) if non_comm_comb else float("inf")
    streaming_warps = set()
    for sp in rep_spans:
        if (int(sp["sm"]), int(sp["warp"])) in comm_tracks and float(sp["ts"]) < tail_comb_start:
            low = sp["label"].lower()
            if "combine" in low or "reduce" in low or "return" in low:
                streaming_warps.add(int(sp["warp"]))
    out = []
    for w, r in rep_warps:
        rl = r.lower()
        if "compute" in rl or "math" in rl:
            out.append(f"W{w:02d} (Compute / Combine)")
        elif "comm" in rl or "dispatch" in rl or "scatter" in rl:
            if is_amd:
                suffix = "Comm / Dispatch & Async Scatter"
            else:
                suffix = "Comm / Scatter -> Streaming Reduce" if w in streaming_warps else "Comm / Scatter"
            out.append(f"W{w:02d} ({suffix})")
        else:
            out.append(f"W{w:02d} ({r})")
    return out


def _compute_depth_segments(intervals: Sequence[Tuple[float, float]]) -> Tuple[List[Tuple[float, float, int]], float, int]:
    """Sweep-line exact in-flight concurrency depth over ``[(ts, end), ...]``.

    Returns ``(segments, avg_depth, max_depth)`` where each segment is ``(t0, t1, depth)`` for ``depth >= 1``
    and ``avg_depth`` is time-weighted over the active (depth >= 1) union duration.
    """
    events: List[Tuple[float, int]] = []
    for ts, end in intervals:
        if end > ts + 1e-6:
            events.append((float(ts), 1))
            events.append((float(end), -1))
    if not events:
        return [], 0.0, 0
    # Process completions (-1) before simultaneous launches (+1) to avoid false depth spikes at boundaries.
    events.sort(key=lambda kv: (kv[0], kv[1]))
    segs: List[Tuple[float, float, int]] = []
    cur = 0
    max_d = 0
    area = 0.0
    active_dur = 0.0
    prev_t = events[0][0]
    for t, delta in events:
        dt = t - prev_t
        if dt > 1e-6 and cur > 0:
            if segs and segs[-1][2] == cur and abs(segs[-1][1] - prev_t) <= 1e-5:
                segs[-1] = (segs[-1][0], t, cur)
            else:
                segs.append((prev_t, t, cur))
            area += dt * cur
            active_dur += dt
        cur = max(0, cur + delta)
        if cur > max_d:
            max_d = cur
        prev_t = t
    avg_d = (area / active_dur) if active_dur > 1e-6 else 0.0
    return segs, avg_d, max_d


def _pick_micro_zoom_window(
    rep_main: Sequence[Mapping[str, Any]],
    rep_async: Sequence[Mapping[str, Any]],
    rep_marks: Sequence[Mapping[str, Any]],
    t_max: float,
    *,
    zoom_cta: Optional[int] = None,
    zoom_window: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float, Optional[int], float, str]]:
    """Deterministic micro-zoom selector (Panel D).

    1. Explicit overrides (`zoom_window` / `zoom_cta`) always win.
    2. Trigger rule: compute median inter-event gap on representative SM; activate only when
       `delta_px = (median_gap_us / t_max) * 2400 < 12.0` (dense full-kernel view).
    3. Multi-wave kernels: pick the complete CTA with highest `wait_ratio = wait_us / dur_us`
       (tie-break: closest to middle wave, then smallest CTA id); if zero exposed wait, pick
       the median-duration (p50) steady-state CTA.
    4. Width cap: show full CTA when `8 <= n_iters <= 48`; cap to a deterministic 32-iteration
       window (`[peak-12, peak+20]` around the worst wait, or middle 32 iters) when `n_iters > 48`;
       include adjacent CTA when `n_iters < 8`.
    5. Single-wave / persistent kernels: sliding-window scan maximizing exposed wait (or steady
       50% midpoint) at the 32-iteration target width.
    """
    if zoom_window is not None:
        z0, z1 = float(zoom_window[0]), float(zoom_window[1])
        if z1 > z0:
            return max(0.0, z0), min(t_max * 1.015, z1), zoom_cta, 0.0, "user-specified window"

    cta_bounds: Dict[int, List[float]] = {}
    cta_warps: Dict[int, set[int]] = defaultdict(set)
    cta_wait: Dict[int, float] = defaultdict(float)
    cta_wait_spans: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    cta_issues: Dict[int, List[float]] = defaultdict(list)
    all_issue_ts_by_warp: Dict[int, List[float]] = defaultdict(list)
    all_main_ts_by_warp: Dict[int, List[float]] = defaultdict(list)

    for sp in rep_main:
        c = int(sp["cta"])
        cta_warps[c].add(int(sp["warp"]))
        t0, t1 = float(sp["ts"]), float(sp["end"])
        b = cta_bounds.get(c)
        cta_bounds[c] = [t0, t1] if b is None else [min(b[0], t0), max(b[1], t1)]
        all_main_ts_by_warp[int(sp["warp"])].append(t0)
        if sp["sem"] == "wait":
            cta_wait[c] += (t1 - t0)
            cta_wait_spans[c].append((t0, t1))
    for sp in rep_async:
        c = int(sp["cta"])
        cta_warps[c].add(int(sp["warp"]))
        t0, t1 = float(sp["ts"]), float(sp["end"])
        b = cta_bounds.get(c)
        cta_bounds[c] = [t0, t1] if b is None else [min(b[0], t0), max(b[1], t1)]
        cta_issues[c].append(t0)
        all_issue_ts_by_warp[int(sp["warp"])].append(t0)
    for mk in rep_marks:
        c = int(mk["cta"])
        cta_warps[c].add(int(mk["warp"]))
        t0 = float(mk["ts"])
        b = cta_bounds.get(c)
        cta_bounds[c] = [t0, t0] if b is None else [min(b[0], t0), max(b[1], t0)]
        cta_issues[c].append(t0)
        all_issue_ts_by_warp[int(mk["warp"])].append(t0)

    def _dedup_sorted(ts_list: Sequence[float]) -> List[float]:
        out: List[float] = []
        for t in sorted(ts_list):
            if not out or (t - out[-1]) > 0.015:
                out.append(t)
        return out

    # Step 1: Deterministic density trigger (`delta_px < 28 px` on ~2400 px plot width)
    # Check both async issue spacing AND main-track stage spacing so multi-stage loops
    # (e.g. FlashMLA QK/Softmax/PV) trigger micro-zoom whenever per-stage bars get crowded.
    def _med_gap_px(ts_by_warp: Mapping[int, Sequence[float]]) -> float:
        gs: List[float] = []
        for _w, ts_l in ts_by_warp.items():
            ds = _dedup_sorted(ts_l)
            for i in range(1, len(ds)):
                g = ds[i] - ds[i - 1]
                if g > 1e-4:
                    gs.append(g)
        if not gs:
            return 2400.0
        gs.sort()
        return (gs[len(gs) // 2] / max(t_max, 1e-6)) * 2400.0

    async_gaps: List[float] = []
    for _w, ts_l in all_issue_ts_by_warp.items():
        ds = _dedup_sorted(ts_l)
        for i in range(1, len(ds)):
            g = ds[i] - ds[i - 1]
            if g > 1e-4:
                async_gaps.append(g)
    async_gaps.sort()
    med_gap = async_gaps[len(async_gaps) // 2] if async_gaps else t_max
    delta_px = min(_med_gap_px(all_issue_ts_by_warp), _med_gap_px(all_main_ts_by_warp))
    if zoom_cta is None and delta_px >= 28.0:
        return None

    # Deduplicate per-CTA issue timestamps across one representative warp to count true iterations
    rep_w_first = min(all_issue_ts_by_warp.keys()) if all_issue_ts_by_warp else 0
    cta_warp_iters: Dict[int, List[float]] = defaultdict(list)
    for sp in rep_async:
        if int(sp["warp"]) == rep_w_first:
            cta_warp_iters[int(sp["cta"])].append(float(sp["ts"]))
    for mk in rep_marks:
        if int(mk["warp"]) == rep_w_first:
            cta_warp_iters[int(mk["cta"])].append(float(mk["ts"]))
    cta_iter_ts = {c: _dedup_sorted(v) for c, v in cta_warp_iters.items()}

    def _cap_to_32_iters(c_id: int, t0: float, t1: float, reason: str) -> Tuple[float, float, Optional[int], float, str]:
        its = cta_iter_ts.get(c_id) or _dedup_sorted(cta_issues.get(c_id, ()))
        n_it = len(its)
        w_us = cta_wait.get(c_id, 0.0)
        if n_it > 108:
            # Cap strictly to 32 iterations centered on worst wait bubble when CTA has >108 issues
            # (Note: 32-K-tile GEMM with 2 cp.async + 1 wgmma per K-tile emits 96 issues <= 108, preserving full prologue & drain!)
            ws = cta_wait_spans.get(c_id, [])
            if ws:
                peak_t = max(ws, key=lambda x: x[1] - x[0])[0]
                peak_idx = min(range(n_it), key=lambda i: abs(its[i] - peak_t))
                lo_i = max(0, min(n_it - 32, peak_idx - 12))
            else:
                lo_i = max(0, (n_it - 32) // 2)
            hi_i = min(n_it - 1, lo_i + 31)
            z0, z1 = its[lo_i], (its[hi_i] + med_gap * 1.5)
            pad = max(0.25, (z1 - z0) * 0.04)
            return max(t0, z0 - pad), min(t1 + pad, z1 + pad), c_id, w_us, f"{reason}, 32-iter cap"
        pad = max(0.25, (t1 - t0) * 0.04)
        return max(0.0, t0 - pad), min(t_max * 1.015, t1 + pad), c_id, w_us, reason

    # Explicit CTA override
    if zoom_cta is not None and zoom_cta in cta_bounds:
        t0, t1 = cta_bounds[zoom_cta]
        return _cap_to_32_iters(zoom_cta, t0, t1, f"CTA {zoom_cta}")

    # Step 2: Multi-wave CTA selection vs single-wave persistent kernel
    # Require >= 6 per-warp events inside a CTA for single-CTA micro-zoom so sparse/outer-only
    # traces (e.g. K0-only peeling with 1-2 events per CTA) never zoom into an empty white canvas.
    MIN_SINGLE_CTA_EVENTS = 6
    cands = []
    sparse_cands = []
    for c, (t0, t1) in cta_bounds.items():
        dur = t1 - t0
        n_it = len(cta_iter_ts.get(c, ()))
        n_w = max(1, len(cta_warps.get(c) or ()))
        if dur >= 0.4 and n_it >= 1:
            w_ratio = min(1.0, cta_wait.get(c, 0.0) / max(dur * n_w, 1e-6))
            entry = (c, t0, t1, dur, cta_wait.get(c, 0.0), w_ratio, n_it)
            if n_it >= MIN_SINGLE_CTA_EVENTS:
                cands.append(entry)
            else:
                sparse_cands.append(entry)

    if len(cands) >= 2:
        cands.sort(key=lambda x: x[1])
        mid_t = cands[len(cands) // 2][1]
        best_wait_ratio = max(x[5] for x in cands)
        if best_wait_ratio > 0.005:
            # Highest exposed wait ratio -> tie-break: closest to middle wave, then smallest CTA id
            chosen = max(cands, key=lambda x: (round(x[5], 4), -abs(x[1] - mid_t), -x[0]))
            reason = f"peak wait_ratio {chosen[5]*100:.1f}%"
        else:
            # Steady-state median-duration (p50) CTA
            by_dur = sorted(cands, key=lambda x: (x[3], x[0]))
            chosen = by_dur[len(by_dur) // 2]
            reason = "steady-state p50 CTA"
        c_id, t0, t1, dur, _w, _wr, n_it = chosen
        # If short CTA (< 8 iters), extend to next adjacent CTA and disable single-CTA filter so both CTAs render
        keep_cta: Optional[int] = c_id
        if n_it < 8:
            pos = next(i for i, x in enumerate(cands) if x[0] == c_id)
            if pos + 1 < len(cands):
                t1 = max(t1, cands[pos + 1][2])
                reason += " + next CTA"
                keep_cta = None
        if (t1 - t0) > 0.70 * t_max and n_it <= 48:
            return None
        z0, z1, zc, zw, zr = _cap_to_32_iters(c_id, t0, t1, reason)
        return z0, z1, (zc if keep_cta is not None else None), zw, zr

    if len(cands) == 1:
        # Single-wave persistent kernel / megakernel: cap to 32-iter sliding window if dense (> 48 iters)
        c_id, t0, t1, _dur, _w, _wr, n_it = cands[0]
        if n_it > 48:
            return _cap_to_32_iters(c_id, t0, t1, "persistent-kernel")

    # Fallback for dense multi-CTA traces whose individual CTAs have < 6 events (e.g. outer-only probes):
    # zoom into a consecutive multi-CTA wave block containing >= 12 events instead of an empty single CTA!
    if len(sparse_cands) >= 6:
        sparse_cands.sort(key=lambda x: x[1])
        total_ev = sum(x[6] for x in sparse_cands)
        if total_ev >= 12:
            mid_idx = len(sparse_cands) // 2
            lo_idx = mid_idx
            hi_idx = mid_idx
            acc_ev = sparse_cands[mid_idx][6]
            while acc_ev < 16 and (lo_idx > 0 or hi_idx < len(sparse_cands) - 1) and (hi_idx - lo_idx + 1) < 16:
                if hi_idx < len(sparse_cands) - 1:
                    hi_idx += 1
                    acc_ev += sparse_cands[hi_idx][6]
                if acc_ev < 16 and lo_idx > 0:
                    lo_idx -= 1
                    acc_ev += sparse_cands[lo_idx][6]
            z0 = sparse_cands[lo_idx][1]
            z1 = sparse_cands[hi_idx][2]
            if (z1 - z0) <= 0.65 * t_max and acc_ev >= 12:
                pad = max(0.25, (z1 - z0) * 0.04)
                n_ctas_win = hi_idx - lo_idx + 1
                return (
                    max(0.0, z0 - pad), min(t_max * 1.015, z1 + pad),
                    None, 0.0, f"multi-CTA wave zoom ({n_ctas_win} CTAs, {acc_ev} events)",
                )
    return None


def _render_warp_panel(
    ax: Any,
    *,
    rep_main: Sequence[Mapping[str, Any]],
    rep_async: Sequence[Mapping[str, Any]],
    rep_marks: Sequence[Mapping[str, Any]],
    n_rep_rows: int,
    use_cta_slots: bool,
    rep_cta_slot: Mapping[int, int],
    warp_to_y: Mapping[int, int],
    span_color_fn: Any,
    dual_fabric: bool,
    x_lim: Tuple[float, float],
    k_dur: float,
    is_zoom: bool = False,
    show_right_stats: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Render per-warp/CTA main execution bars + on-demand thin async engine depth strips & ▼ issue pins."""
    from matplotlib.collections import LineCollection, PolyCollection
    import matplotlib.patches as mpatches

    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="x", linestyle="-", linewidth=0.45, color="#ECEFF1", zorder=0)
    x_lo, x_hi = float(x_lim[0]), float(x_lim[1])
    x_span = max(x_hi - x_lo, 1e-6)
    min_vis_us = x_span * (2.0 / 2800.0)

    def _row_of(ev: Mapping[str, Any]) -> Optional[int]:
        return rep_cta_slot.get(int(ev["cta"])) if use_cta_slots else warp_to_y.get(int(ev["warp"]))

    # Filter to visible window [x_lo, x_hi] for zoom speed & accuracy
    vis_main = [sp for sp in rep_main if float(sp["end"]) >= x_lo and float(sp["ts"]) <= x_hi]
    vis_async = [sp for sp in rep_async if float(sp["end"]) >= x_lo and float(sp["ts"]) <= x_hi]
    vis_marks = [mk for mk in rep_marks if x_lo <= float(mk["ts"]) <= x_hi]

    # Discover active async engines & whether each row has main spans
    row_has_main: Dict[int, bool] = defaultdict(bool)
    row_engines: Dict[int, set[str]] = defaultdict(set)
    for sp in vis_main:
        r = _row_of(sp)
        if r is not None:
            row_has_main[r] = True
    for sp in vis_async:
        r = _row_of(sp)
        if r is not None and sp["async_engine"] in _ASYNC_ENGINE_STYLE:
            row_engines[r].add(sp["async_engine"])
    for mk in vis_marks:
        r = _row_of(mk)
        if r is not None and mk["async_engine"] in _ASYNC_ENGINE_STYLE:
            row_engines[r].add(mk["async_engine"])

    # Compute exact vertical geometry per row (note: inverted Y-axis -> smaller y is visual TOP!)
    # Returns (main_top, main_bot), {engine: (sub_top, sub_bot)}
    row_geom: Dict[int, Tuple[Tuple[float, float], Dict[str, Tuple[float, float]]]] = {}
    for r in range(n_rep_rows):
        engs = [e for e in ("mem", "mma") if e in row_engines.get(r, ())]
        if not engs or dual_fabric:
            row_geom[r] = ((r - 0.36, r + 0.36), {})
        elif not row_has_main.get(r, False) and len(engs) == 1:
            # Pure-async Producer warp (e.g. TMA Producer): fill full row height with depth strip + ▼ pins
            row_geom[r] = ((r - 0.36, r + 0.36), {engs[0]: (r - 0.36, r + 0.36)})
        elif len(engs) == 1:
            # Clean two-row split: upper main track (Softmax/ALU/Wait/Store) + lower async depth row (MEM or MMA)
            row_geom[r] = ((r - 0.39, r + 0.03), {engs[0]: (r + 0.09, r + 0.39)})
        else:
            row_geom[r] = ((r - 0.40, r - 0.04), {"mem": (r + 0.01, r + 0.18), "mma": (r + 0.23, r + 0.40)})

    # 1) Draw Main Execution Track spans
    verts_by_color: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
    durs_by_color: Dict[str, List[float]] = defaultdict(list)
    sem_by_color: Dict[str, str] = {}
    is_priority_color: Dict[str, bool] = {}
    wait_verts: List[List[Tuple[float, float]]] = []
    row_wait_us: Dict[int, float] = defaultdict(float)
    row_main_us: Dict[int, float] = defaultdict(float)
    row_all_ivs: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    for sp in vis_main:
        r = _row_of(sp)
        if r is None:
            continue
        x0 = max(x_lo, float(sp["ts"]))
        x1 = min(x_hi, float(sp["end"]))
        is_crit = dual_fabric and sp.get("_frole") in ("inter_dispatch", "inter_combine", "intra_combine")
        if is_crit and (x1 - x0) < min_vis_us:
            x1 = x0 + min_vis_us
        if (x1 - x0) <= 0.0:
            continue
        row_main_us[r] += (x1 - x0)
        row_all_ivs[r].append((x0, x1))
        y_top, y_bot = row_geom[r][0]
        poly = [(x0, y_top), (x1, y_top), (x1, y_bot), (x0, y_bot)]
        if sp["sem"] == "wait" and not dual_fabric:
            row_wait_us[r] += (x1 - x0)
            wait_verts.append(poly)
            continue
        col_hex = span_color_fn(sp)
        verts_by_color[col_hex].append(poly)
        durs_by_color[col_hex].append(float(sp["dur"]))
        sem_by_color.setdefault(col_hex, sp["sem"])
        if is_crit:
            is_priority_color[col_hex] = True

    ordered_color_layers = sorted(
        verts_by_color.items(),
        key=lambda kv: (
            2 if is_priority_color.get(kv[0]) else (0 if sem_by_color.get(kv[0]) == "barrier" else 1),
            -sorted(durs_by_color[kv[0]])[len(durs_by_color[kv[0]]) // 2],
        ),
    )
    dense_main = len(vis_main) > 240
    for layer_idx, (col_hex, verts) in enumerate(ordered_color_layers):
        no_edge = is_priority_color.get(col_hex) or dense_main
        pc = PolyCollection(
            verts, facecolors=col_hex, edgecolors="none" if no_edge else "#FFFFFF",
            linewidths=0.0 if no_edge else 0.16,
            antialiased=not no_edge,
            zorder=3 + layer_idx * 0.05,
        )
        ax.add_collection(pc)

    # High-visibility Amber Gold Exposed Wait layer (always drawn above MMA, with crisp /// hatch when zoomed!)
    if wait_verts:
        dense_wait = len(wait_verts) > 120 and not is_zoom
        pc_w = PolyCollection(
            wait_verts,
            facecolors="#F5A623",
            edgecolors="#B87A00" if not dense_wait else "none",
            linewidths=0.55 if is_zoom else (0.30 if not dense_wait else 0.0),
            hatch="///" if (is_zoom or len(wait_verts) <= 48) else None,
            zorder=4.8,
        )
        ax.add_collection(pc_w)

    # 2) Draw Thin Async Engine Sub-Tracks (In-Flight Depth Strips + Vivid Fixed-Pixel ▼ Issue Pins)
    row_stats: Dict[int, Dict[str, Any]] = {}
    if not dual_fabric and any(row_engines.values()):
        # Collect async spans (with CTA id) & issue timestamps per (row, engine)
        async_by_re: Dict[Tuple[int, str], List[Tuple[float, float, int]]] = defaultdict(list)
        issues_by_re: Dict[Tuple[int, str], List[float]] = defaultdict(list)
        for sp in vis_async:
            r = _row_of(sp)
            eng = sp["async_engine"]
            if r is not None and eng in _ASYNC_ENGINE_STYLE:
                ax0, ax1 = max(x_lo, float(sp["ts"])), min(x_hi, float(sp["end"]))
                if ax1 > ax0:
                    row_all_ivs[r].append((ax0, ax1))
                async_by_re[(r, eng)].append((float(sp["ts"]), float(sp["end"]), int(sp["cta"])))
                if x_lo <= float(sp["ts"]) <= x_hi:
                    issues_by_re[(r, eng)].append(float(sp["ts"]))
        for mk in vis_marks:
            r = _row_of(mk)
            eng = mk["async_engine"]
            if r is not None and eng in _ASYNC_ENGINE_STYLE:
                issues_by_re[(r, eng)].append(float(mk["ts"]))

        # Deduplicate close issue timestamps (e.g. both explicit mark + async span start at same op)
        dedup_issues: Dict[Tuple[int, str], List[float]] = {}
        for key, ts_list in issues_by_re.items():
            ts_list.sort()
            out_t: List[float] = []
            for t in ts_list:
                if not out_t or (t - out_t[-1]) > 0.015:
                    out_t.append(t)
            dedup_issues[key] = out_t

        # Batched collections per engine
        rail_polys: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
        depth_polys: Dict[Tuple[str, int], List[List[Tuple[float, float]]]] = defaultdict(list)
        pin_stems: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)
        pin_pts_x: Dict[str, List[float]] = defaultdict(list)
        pin_pts_y: Dict[str, List[float]] = defaultdict(list)
        tick_segs: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(list)

        # Adaptive LOD pixel threshold (~11 px between full ▼ triangles in macro view)
        min_pin_gap_us = x_span * (11.0 / 2200.0)
        min_tick_gap_us = x_span * (4.5 / 2200.0)

        # Pre-pass max per-CTA depth per engine so all rows share one consistent 5-shade scale
        eng_scale_max: Dict[str, int] = {}
        for eng in ("mem", "mma"):
            gmax = 0
            for r in range(n_rep_rows):
                ivs_cta = async_by_re.get((r, eng), [])
                by_c: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
                for ts_a, en_a, cta_a in ivs_cta:
                    by_c[cta_a].append((ts_a, en_a))
                for c_ivs in by_c.values():
                    _, _, cm = _compute_depth_segments(c_ivs)
                    if cm > gmax:
                        gmax = cm
            eng_scale_max[eng] = gmax

        for r in range(n_rep_rows):
            sub_map = row_geom[r][1]
            if not sub_map:
                continue
            stat_parts: List[str] = []
            for eng in ("mem", "mma"):
                if eng not in sub_map:
                    continue
                y_top, y_bot = sub_map[eng]
                is_pure_prod = not row_has_main.get(r, False)
                rail_x1 = min(x_hi, max(k_dur, x_lo))
                rail_polys[eng].append([(x_lo, y_top), (rail_x1, y_top), (rail_x1, y_bot), (x_lo, y_bot)])

                # Exact per-CTA in-flight depth (so concurrent resident CTAs on one SM don't double-count stage depth!)
                ivs_cta = async_by_re.get((r, eng), [])
                by_cta_ivs: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
                for ts_a, en_a, cta_a in ivs_cta:
                    by_cta_ivs[cta_a].append((ts_a, en_a))
                cta_avgs: List[float] = []
                max_d = 0
                if len(by_cta_ivs) <= 1:
                    only_ivs = [(a, b) for a, b, _ in ivs_cta]
                    d_segs, avg_d, max_d = _compute_depth_segments(only_ivs)
                else:
                    # Pointwise max per-CTA depth across resident CTAs (disjoint x-intervals) and mean per-CTA depth for stats
                    pts_set = set()
                    cta_seg_lists: List[List[Tuple[float, float, int]]] = []
                    for _c_id, c_ivs in by_cta_ivs.items():
                        c_segs, c_avg, c_max = _compute_depth_segments(c_ivs)
                        if c_max > 0:
                            cta_avgs.append(c_avg)
                            max_d = max(max_d, c_max)
                            cta_seg_lists.append(c_segs)
                            for a_s, b_s, _ in c_segs:
                                pts_set.add(a_s)
                                pts_set.add(b_s)
                    pts_sorted = sorted(pts_set)
                    merged_max: List[Tuple[float, float, int]] = []
                    for idx_p in range(len(pts_sorted) - 1):
                        pa, pb = pts_sorted[idx_p], pts_sorted[idx_p + 1]
                        if pb <= pa:
                            continue
                        pm = 0.5 * (pa + pb)
                        md = 0
                        for cs_l in cta_seg_lists:
                            for sa, sb, sd in cs_l:
                                if sa <= pm <= sb and sd > md:
                                    md = sd
                        if md > 0:
                            if merged_max and merged_max[-1][2] == md and abs(merged_max[-1][1] - pa) < 1e-9:
                                merged_max[-1] = (merged_max[-1][0], pb, md)
                            else:
                                merged_max.append((pa, pb, md))
                    d_segs = merged_max
                    avg_d = (sum(cta_avgs) / len(cta_avgs)) if cta_avgs else 0.0

                smax = eng_scale_max.get(eng, max_d)
                sub_h = y_bot - y_top
                min_lbl_px = 18.0 if sub_h >= 0.22 else 22.0
                for s0, s1, d in d_segs:
                    cx0, cx1 = max(x_lo, s0), min(x_hi, s1)
                    if cx1 > cx0:
                        bucket, _bcol = _depth_to_bucket_color(eng, d, smax)
                        depth_polys[(eng, bucket)].append([(cx0, y_top), (cx1, y_top), (cx1, y_bot), (cx0, y_bot)])
                        seg_px = ((cx1 - cx0) / x_span) * 2200.0
                        need_px = min_lbl_px + (7.0 if d >= 10 else 0.0)
                        if sub_h >= 0.20 and (is_zoom or len(by_cta_ivs) <= 1) and seg_px >= need_px:
                            rr_c, gg_c, bb_c = _hex_to_rgb(_bcol)
                            lum = 0.299 * rr_c + 0.587 * gg_c + 0.114 * bb_c
                            t_col = "#FFFFFF" if lum < 145.0 else "#0F172A"
                            fs_d = 6.7 if sub_h >= 0.24 else 5.8
                            ax.text(
                                0.5 * (cx0 + cx1),
                                0.5 * (y_top + y_bot),
                                str(d),
                                ha="center",
                                va="center",
                                fontsize=fs_d,
                                fontweight="bold",
                                color=t_col,
                                zorder=4.9,
                                clip_on=True,
                            )

                # Prominent Fixed-Pixel ▼ Async Issue Pins (with clean Adaptive LOD so emerald depth strip stays crisp!)
                ts_all = dedup_issues.get((r, eng), [])
                if ts_all:
                    gaps = sorted(ts_all[i] - ts_all[i - 1] for i in range(1, len(ts_all)))
                    med_gap = gaps[len(gaps) // 2] if gaps else x_span
                    eff_pin_gap_us = (x_span * (10.5 / 2200.0)) if is_zoom else min_pin_gap_us
                    dense_pins = (len(ts_all) > (120 if is_zoom else 40)) or (med_gap < eff_pin_gap_us)
                    stem_top = y_top if is_pure_prod else (y_top - 0.05)
                    pin_y = y_top + (0.01 if is_pure_prod else -0.01)
                    tick_bot = y_top + (y_bot - y_top) * 0.45 if max_d >= 1 else y_bot
                    last_pin_t = -1e18
                    last_tick_t = -1e18
                    for idx_t, t_val in enumerate(ts_all):
                        is_wave_head = (idx_t == 0) or ((t_val - ts_all[idx_t - 1]) > max(2.2 * med_gap, eff_pin_gap_us))
                        can_place_pin = (t_val - last_pin_t) >= eff_pin_gap_us * (0.95 if is_zoom else 0.9)
                        if can_place_pin and ((not dense_pins) or is_zoom or is_wave_head):
                            pin_stems[eng].append([(t_val, stem_top), (t_val, y_bot)])
                            pin_pts_x[eng].append(t_val)
                            pin_pts_y[eng].append(pin_y)
                            last_pin_t = t_val
                            last_tick_t = t_val
                        elif (t_val - last_tick_t) >= min_tick_gap_us:
                            tick_segs[eng].append([(t_val, y_top - 0.01), (t_val, tick_bot)])
                            last_tick_t = t_val

                n_iss = len(ts_all)
                compact = len(sub_map) >= 2
                eng_short = "M" if eng == "mem" else "W"
                if max_d >= 1:
                    if compact:
                        d_txt = f"d={avg_d:.1f}/m{max_d}" if max_d >= 4 else f"d={avg_d:.1f}"
                        stat_parts.append(f"{n_iss}{eng_short} {d_txt}")
                    else:
                        d_txt = f"d={avg_d:.1f} (max {max_d})" if max_d >= 3 else f"d={avg_d:.1f}"
                        stat_parts.append(f"{n_iss} {eng} · {d_txt}")
                elif n_iss > 0:
                    stat_parts.append(f"{n_iss}{eng_short}" if compact else f"{n_iss} {eng} issues")

            if row_wait_us.get(r, 0.0) > 0.0:
                # Union wall-clock active duration on row r inside [x_lo, x_hi] (main + async),
                # so dual-async warps (cp.async + wgmma) report true % of wall time stalled in wait!
                ivs = sorted(row_all_ivs.get(r, ()))
                union_us = 0.0
                if ivs:
                    cur_s, cur_e = ivs[0]
                    for s_i, e_i in ivs[1:]:
                        if s_i <= cur_e:
                            cur_e = max(cur_e, e_i)
                        else:
                            union_us += (cur_e - cur_s)
                            cur_s, cur_e = s_i, e_i
                    union_us += (cur_e - cur_s)
                denom = max(union_us, row_main_us.get(r, 0.0), 1e-6)
                wpct = min(100.0, 100.0 * row_wait_us[r] / denom)
                stat_parts.append(f"wait {wpct:.1f}%")
            if stat_parts:
                row_stats[r] = {"text": " · ".join(stat_parts)}

        # Add batched async artists
        for eng in ("mem", "mma"):
            st = _ASYNC_ENGINE_STYLE[eng]
            if rail_polys.get(eng):
                ax.add_collection(PolyCollection(
                    rail_polys[eng], facecolors=st["rail_bg"], edgecolors=st["rail_ec"],
                    linewidths=0.45, zorder=2.2,
                ))
            bins_eng = _depth_bins_spec(eng, eng_scale_max.get(eng, 1))
            for bucket, (_lo_d, _hi_d, dcol, _dlbl) in enumerate(bins_eng):
                polys = depth_polys.get((eng, bucket))
                if polys:
                    ax.add_collection(PolyCollection(
                        polys, facecolors=dcol, edgecolors="none", linewidths=0.0, antialiased=False,
                        zorder=2.6 + bucket * 0.05,
                    ))
            if tick_segs.get(eng):
                ax.add_collection(LineCollection(
                    tick_segs[eng], colors=st["pin_color"], linewidths=0.9, alpha=0.88, zorder=5.2,
                ))
            if pin_stems.get(eng):
                ax.add_collection(LineCollection(
                    pin_stems[eng], colors=st["pin_color"], linewidths=1.75 if is_zoom else 1.4, zorder=5.6,
                ))
            if pin_pts_x.get(eng):
                ax.scatter(
                    pin_pts_x[eng], pin_pts_y[eng],
                    marker="v", s=38 if is_zoom else 24,
                    color=st["pin_color"], edgecolors="#FFFFFF", linewidths=0.55,
                    zorder=6.2, clip_on=True,
                )

        if show_right_stats and row_stats:
            trans = ax.get_yaxis_transform()
            _rfs = 8.2 if n_rep_rows >= 18 else (8.8 if n_rep_rows >= 14 else 9.4)
            for r, info in row_stats.items():
                ax.text(
                    1.008, r, info["text"], transform=trans,
                    va="center", ha="left", fontsize=_rfs, fontweight="bold", color="#334155",
                    clip_on=False,
                )

    if x_lo <= k_dur <= x_hi:
        ax.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75, zorder=6.5)
    ax.set_ylim(n_rep_rows - 0.48, -0.55)
    ax.set_yticks(range(n_rep_rows))
    ax.set_xlim(x_lo, x_hi)
    return row_stats


# Cap measured left-margin rescue so <=60-char Panel C role labels fit at 10.2 pt (~0.3217 on pingpong)
# while keeping >=75.0% of the default Panel C plot width (678.7 px vs 905.5 px at 100 dpi, right_m=0.754).
LEFT_M_MAX = 0.35
LEFT_RESCUE_MAX_EXTRA_DRAWS = 2


def _truncate_text_to_px(probe: Any, rnd: Any, text: str, min_keep: int, max_width_px: float) -> str:
    probe.set_text(text)
    if probe.get_window_extent(rnd).width <= max_width_px or min_keep <= 0 or min_keep >= len(text) or max_width_px <= 0.0:
        return text
    probe.set_text(text[:min_keep] + "\u2026")
    if probe.get_window_extent(rnd).width > max_width_px:
        return text
    lo, hi = min_keep, len(text) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = text[:mid].rstrip()
        probe.set_text((cand if len(cand) >= min_keep else text[:min_keep]) + "\u2026")
        if probe.get_window_extent(rnd).width <= max_width_px:
            lo = mid
        else:
            hi = mid - 1
    cand = text[:lo].rstrip()
    return (cand if len(cand) >= min_keep else text[:min_keep]) + "\u2026"


def render_swhw_overview_png(
    trace_or_path: Mapping[str, Any] | str | Path,
    out_png: Optional[str | Path] = None,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    focus_sm: Optional[int] = None,
    zoom_cta: Optional[int] = None,
    zoom_window: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    require_hw_curves: bool = True,
    subplot_lr: Optional[Tuple[float, float]] = None,
    model: Optional[str] = None,
    op: Optional[str] = None,
    bs: Optional[Any] = None,
    seql: Optional[Any] = None,
    kv: Optional[Any] = None,
    shape: Optional[str] = None,
    dims: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[str | _dt.datetime] = None,
) -> Path:
    """Render a publication-grade SW+HW overview PNG (with on-demand async depth strips, ▼ issue pins & micro-zoom)."""
    render_swhw_overview_png.last_left_rescue_px = 0.0
    render_swhw_overview_png.last_extra_draws = 0
    render_swhw_overview_png.last_panel_c_title_truncated = False
    import matplotlib
    matplotlib.use("Agg")
    if sys.version_info >= (3, 14):
        from matplotlib.path import Path as _MplPath
        def _safe_path_deepcopy(self, memo):
            p = _MplPath(
                self.vertices.copy(),
                self.codes.copy() if self.codes is not None else None,
            )
            p._interpolation_steps = getattr(self, "_interpolation_steps", 1)
            p._simplify_threshold = getattr(self, "_simplify_threshold", 0.0)
            p._should_simplify = getattr(self, "_should_simplify", False)
            return p
        _MplPath.__deepcopy__ = _safe_path_deepcopy
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    import numpy as np

    trace = _load_trace(trace_or_path)
    parsed = _parse_trace(trace)
    hw_curves = parsed["hw_curves"]
    hw_labels = parsed["hw_labels"]
    sw_spans = parsed["sw_spans"]
    main_spans = parsed["main_spans"]
    async_spans = parsed["async_spans"]
    sw_marks = parsed["sw_marks"]
    sms = parsed["sms"]
    t_max = parsed["t_max"]
    k_dur = parsed["k_dur"]
    nsys_k = parsed["nsys_kernel"]
    other = parsed["other"]
    meta = other.get("meta") if isinstance(other.get("meta"), Mapping) else {}
    nsys_meta = (other.get("nsys_hw") or {})
    is_amd = bool(parsed["is_amd"])
    multi_node = bool(parsed["multi_node"])
    dual_fabric = bool(parsed["dual_fabric"])
    use_ib = bool(parsed["use_ib"])
    unit_nm = "CU" if is_amd else "SM"
    lane_nm = "wave" if is_amd else "warp"

    # Pick representative SM/CU for Panel (C)
    sm_warp_role: Dict[int, Dict[int, str]] = defaultdict(dict)
    for sm_id, wmap in (parsed.get("meta_warps_by_sm") or {}).items():
        for w_id, r_lbl in wmap.items():
            sm_warp_role[int(sm_id)][int(w_id)] = str(r_lbl)
    other_sm_warps = other.get("sm_warps") or {}
    if isinstance(other_sm_warps, Mapping):
        for sm_k, wmap in other_sm_warps.items():
            if isinstance(wmap, Mapping):
                for w_k, r_lbl in wmap.items():
                    try:
                        sm_warp_role[int(sm_k)][int(w_k)] = str(r_lbl)
                    except (ValueError, TypeError):
                        pass
    for sp in sw_spans:
        sm_warp_role[sp["sm"]].setdefault(sp["warp"], sp["role"])
    for mk in sw_marks:
        sm_warp_role[mk["sm"]].setdefault(mk["warp"], mk["role"])
    wpc_meta = int(
        other.get("wpc")
        or meta.get("wpc")
        or (other.get("summary") or {}).get("wpc")
        or 0
    )
    global_warp_roles = other.get("warp_roles") or {}
    if wpc_meta > 1:
        for sm_id in list(sm_warp_role.keys()):
            for w_id in range(wpc_meta):
                sm_warp_role[sm_id].setdefault(w_id, str(global_warp_roles.get(str(w_id), "idle")))
    sm_roles: Dict[int, List[Tuple[int, str]]] = {
        sm_id: sorted(wmap.items()) for sm_id, wmap in sm_warp_role.items()
    }
    if focus_sm is not None and focus_sm in sm_roles:
        rep_sm = int(focus_sm)
    elif sms:
        # Prefer an SM with the highest wave/CTA count and highest work/event count so
        # partial-wave tail-idle SMs (both multi-CTA and persistent-kernel tail-idle SM 0)
        # don't hide active waves or prevent Panel (D) micro-zoom.
        sm_ctas: Dict[int, set[int]] = defaultdict(set)
        sm_ev_cnt: Dict[int, int] = defaultdict(int)
        for sp in sw_spans:
            s_i = int(sp["sm"])
            sm_ctas[s_i].add(int(sp["cta"]))
            sm_ev_cnt[s_i] += 1
        rep_sm = max(sms, key=lambda s: (len(sm_ctas.get(s, ())), len(sm_roles.get(s, ())), sm_ev_cnt.get(s, 0), -s))
    else:
        rep_sm = 0

    rep_warps = list(sm_roles.get(rep_sm, ()))
    rep_spans_raw = [sp for sp in sw_spans if sp["sm"] == rep_sm]
    rep_main_raw = [sp for sp in main_spans if sp["sm"] == rep_sm]
    rep_async_raw = [sp for sp in async_spans if sp["sm"] == rep_sm]
    rep_marks_raw = [mk for mk in sw_marks if mk["sm"] == rep_sm]
    active_async_engines = sorted(
        {sp["async_engine"] for sp in async_spans if sp["async_engine"] in _ASYNC_ENGINE_STYLE}
        | {mk["async_engine"] for mk in sw_marks if mk["async_engine"] in _ASYNC_ENGINE_STYLE}
    )
    has_async_depth = {
        eng: any(sp["async_engine"] == eng for sp in async_spans)
        for eng in active_async_engines
    }
    has_async_rep = (not dual_fabric) and bool(rep_async_raw or rep_marks_raw)

    use_cta_slots = False
    rep_cta_slot: Dict[int, int] = {}
    rep_slot_count = 1
    if len(rep_warps) == 1 and rep_spans_raw:
        rep_cta_slot, rep_slot_count = _pack_concurrent_ctas(rep_spans_raw)
        if rep_slot_count > 1:
            use_cta_slots = True

    n_sms = max(1, len(sms))
    n_rep_rows = rep_slot_count if use_cta_slots else max(1, len(rep_warps))

    zoom_win = (
        _pick_micro_zoom_window(
            rep_main_raw, rep_async_raw, rep_marks_raw, t_max,
            zoom_cta=zoom_cta, zoom_window=zoom_window,
        )
        if (has_async_rep or zoom_cta is not None or zoom_window is not None) else None
    )
    has_zoom = zoom_win is not None

    has_hw = any(hw_curves.get(k) for k in ("tensor_active", "sm_active", "sm_issue", "dram_bw", "nvlink_tx", "nvlink_rx", "nvlink_bw", "inter_rdma"))
    if require_hw_curves and not has_hw:
        raise ValueError(
            "MISSING_HARDWARE_UTILIZATION_CURVES: Panel (A) requires same-run hardware profiler "
            "utilization curves (200 kHz nsys on NVIDIA or rocprof-sys on AMD). Refusing to emit "
            "an incomplete SW-only PNG without Panel (A). Capture real GPU metrics (e.g., sudo nsys "
            "on Linux, or enable NVIDIA Control Panel Developer GPU Performance Counters on Windows) "
            "or pass require_hw_curves=False / --allow-no-hw explicitly for offline debugging only."
        )
    row_h_unit = 0.50 if (has_async_rep and n_rep_rows <= 6) else (0.28 if n_rep_rows >= 8 else 0.40)
    panel_c_h = max(2.25, min(5.2, row_h_unit * n_rep_rows + 1.05))
    panel_d_h = max(2.15, min(4.85, row_h_unit * n_rep_rows + 0.95)) if has_zoom else 0.0
    panel_b_h = max(2.9, min(4.2, 0.036 * n_sms + 1.45))
    panel_a_h = 2.45 if has_hw else 0.0
    footer_hw_h = 1.18 if has_hw else 0.0
    fig_h = 2.85 + panel_a_h + panel_b_h + panel_c_h + panel_d_h + footer_hw_h
    fig_w = 16.8 if has_async_rep else 16.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#B0BEC5",
        "axes.linewidth": 0.8,
        "xtick.color": "#37474F",
        "ytick.color": "#37474F",
        "text.color": "#111111",
    })

    heights: List[float] = []
    if has_hw:
        heights.append(panel_a_h)
    heights.extend([panel_b_h, panel_c_h])
    if has_zoom:
        heights.append(panel_d_h)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.49 if dual_fabric else (0.39 if has_zoom else 0.34))
    idx_g = 0
    ax_hw = fig.add_subplot(gs[idx_g]) if has_hw else None
    if has_hw:
        idx_g += 1
    ax_sm = fig.add_subplot(gs[idx_g], sharex=ax_hw)
    idx_g += 1
    ax_wp = fig.add_subplot(gs[idx_g], sharex=ax_sm)
    idx_g += 1
    ax_zoom = fig.add_subplot(gs[idx_g]) if has_zoom else None
    if ax_hw is not None:
        ax_hw.tick_params(labelbottom=False)
    ax_sm.tick_params(labelbottom=False)

    # ---- Title & subtitle ----
    kname = ""
    if nsys_k and nsys_k.get("name"):
        kname = _short_kernel_name(str(nsys_k["name"]))
    if not kname:
        kname = _short_kernel_name(str(meta.get("kernel") or meta.get("model") or "WeaveProfiler Kernel"))
    if not title:
        ep_tag = f" · EP{parsed['ep']}" if parsed["ep"] else ""
        plat_tag = "AMD MI308X" if is_amd else (
            str(meta.get("card") or other.get("card") or meta.get("arch") or other.get("arch") or "").upper() or "GPU"
        )
        title = f"{kname} ({plat_tag}{ep_tag}) — {k_dur:,.1f} µs · {len(sms)} Physical {unit_nm}s"
    if not subtitle:
        freq_khz = None
        if nsys_meta.get("sample_freq_hz"):
            freq_khz = float(nsys_meta["sample_freq_hz"]) / 1000.0
        elif nsys_k and (nsys_k.get("args") or {}).get("sample_freq_hz"):
            freq_khz = float(nsys_k["args"]["sample_freq_hz"]) / 1000.0
        hw_tool_lbl = (
            "rocprofiler-sdk PMC" if is_amd
            else (
                f"{freq_khz:.1f} kHz nsys GPU metrics" if (has_hw and freq_khz)
                else ("nsys GPU metrics" if has_hw else ("nsys window (SW-only)" if nsys_k else "software-only"))
            )
        )
        arch = str(meta.get("arch") or other.get("arch") or "")
        arch_str = f"{arch} · " if arch else ""
        magic = str(meta.get("magic") or "WPLITE02")
        lane_desc = (
            f"up to {rep_slot_count} concurrent CTAs/{unit_nm}"
            if use_cta_slots
            else f"{len(rep_warps) or 1} recorded {lane_nm} lane{'s' if len(rep_warps) != 1 else ''}/{unit_nm}"
        )
        async_note = ""
        if active_async_engines and not dual_fabric:
            eng_desc = "+".join(e.upper() for e in active_async_engines)
            async_note = f" · async {eng_desc} depth+pins"
        subtitle = (
            f"{arch_str}{len(sms)} physical {unit_nm}s × {lane_desc} · "
            f"{len(sw_spans) + len(sw_marks):,} software events{async_note} · {hw_tool_lbl} + {magic}"
        )

    title_fs = 14.8 if len(str(title)) > 98 else 16.8
    sub_fs = 10.4 if len(str(subtitle)) > 118 else 11.4
    fig.text(0.5, 0.989, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color="#0F172A")
    fig.text(0.5, 0.963 if has_zoom else 0.957, subtitle, ha="center", va="top", fontsize=sub_fs, color="#334155")

    # ---- Stage / Dual-Fabric Color Map & Legends ----
    panel_b_fabric_handles: List[Patch] = []
    panel_c_handles: List[Patch] = []
    top_legend_handles: List[Patch] = []

    if dual_fabric:
        # Multi-node dual-color fabric mode: Inter-Node RDMA/IB (warm red/amber) vs Intra-Node xGMI/NVLink (cool teal/blue)
        role_order = (
            "inter_dispatch", "inter_combine", "intra_dispatch", "intra_combine",
            "gemm1", "gemm2", "route_sync", "local_prep",
        )
        seen_roles: set[str] = set()
        for sp in sw_spans:
            frole = _classify_fabric_role(sp, is_amd=is_amd)
            sp["_frole"] = frole
            if frole:
                seen_roles.add(frole)
        # Panel B dedicated 4-swatch Inter-Node vs Intra-Node legend strip
        for frole in ("inter_dispatch", "inter_combine", "intra_dispatch", "intra_combine"):
            col = FABRIC_COLORS_BY_ROLE[frole]
            lbl = _fabric_role_label(frole, is_amd=is_amd, use_ib=use_ib)
            panel_b_fabric_handles.append(Patch(facecolor=col, edgecolor="none", label=lbl))
        # Panel C complete legend (4 fabrics + compute/route/prep stages present)
        for frole in role_order:
            if frole in ("inter_dispatch", "inter_combine", "intra_dispatch", "intra_combine") or frole in seen_roles:
                col = FABRIC_COLORS_BY_ROLE[frole]
                lbl = _fabric_role_label(frole, is_amd=is_amd, use_ib=use_ib)
                panel_c_handles.append(Patch(facecolor=col, edgecolor="none", label=lbl))

        def _span_color(sp: Mapping[str, Any]) -> str:
            fr = sp.get("_frole")
            if fr and fr in FABRIC_COLORS_BY_ROLE:
                return FABRIC_COLORS_BY_ROLE[fr]
            return STAGE_COLORS.get(sp["sem"], _DEFAULT_COLOR)[0]
    else:
        # Single-node / general kernel mode: preserve rich human-readable stage labels for EVERY stage!
        labels_by_sem: Dict[str, List[str]] = defaultdict(list)
        for sp in main_spans:
            lbl = str(sp.get("label") or sp["sem"]).strip()
            if lbl and lbl not in labels_by_sem[sp["sem"]]:
                labels_by_sem[sp["sem"]].append(lbl)
        stage_color_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
        stage_order = ("load", "mma", "gemm", "softmax", "compute", "wait", "epilogue", "store", "barrier", "communication", "reduction", "dispatch")
        ordered_sems = [s for s in stage_order if s in parsed["stages_seen"]] + sorted(
            s for s in parsed["stages_seen"] if s not in stage_order
        )
        seen_leg = set()
        used_colors = set()
        overflow_idx = 0
        for sem in ordered_sems:
            base_col, _base_desc = STAGE_COLORS.get(sem, _DEFAULT_COLOR)
            lbls = sorted(labels_by_sem.get(sem) or [sem], key=lambda x: (x.lower() == sem, x.lower()))
            shades = list(_SEMANTIC_SHADES.get(sem, (base_col,)))
            if len(lbls) >= 2:
                for idx_l, lbl in enumerate(lbls):
                    col = next((c for c in shades[idx_l:] if c not in used_colors), None)
                    if col is None:
                        col = _OVERFLOW_SHADES[overflow_idx % len(_OVERFLOW_SHADES)]
                        overflow_idx += 1
                    used_colors.add(col)
                    desc = _humanize_stage_label(lbl, sem)
                    stage_color_map[(sem, lbl)] = (col, desc)
                    if desc not in seen_leg:
                        seen_leg.add(desc)
                        top_legend_handles.append(Patch(
                            facecolor=col,
                            edgecolor="#B87A00" if sem == "wait" else "none",
                            hatch="///" if sem == "wait" else None,
                            label=desc,
                        ))
            else:
                only_lbl = lbls[0]
                used_colors.add(base_col)
                desc = _humanize_stage_label(only_lbl, sem)
                stage_color_map[(sem, only_lbl)] = (base_col, desc)
                if desc not in seen_leg:
                    seen_leg.add(desc)
                    top_legend_handles.append(Patch(
                        facecolor=base_col,
                        edgecolor="#B87A00" if sem == "wait" else "none",
                        hatch="///" if sem == "wait" else None,
                        label=desc,
                    ))

        # Build dedicated, crystal-clear per-depth color key (`async_depth_handles`) so every shade has its exact `d=` label!
        async_depth_handles: List[Any] = []
        rep_eng_max: Dict[str, int] = {}
        for eng in active_async_engines:
            by_wc: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
            for sp in rep_async_raw:
                if sp["async_engine"] == eng:
                    by_wc[(int(sp["warp"]), int(sp["cta"]))].append((float(sp["ts"]), float(sp["end"])))
            rep_eng_max[eng] = max((_compute_depth_segments(ivs)[2] for ivs in by_wc.values()), default=0)

        for eng in active_async_engines:
            st = _ASYNC_ENGINE_STYLE[eng]
            eng_u = eng.upper()
            async_depth_handles.append(Line2D(
                [0], [0], marker="v", color=st["pin_color"], markerfacecolor=st["pin_color"],
                markeredgecolor="#FFFFFF", markeredgewidth=0.6, markersize=7.2, lw=1.4,
                label=f"{eng_u} Issue",
            ))
            if has_async_depth.get(eng):
                smax = max(1, rep_eng_max.get(eng, 1))
                async_depth_handles.append(Patch(
                    facecolor=st["rail_bg"], edgecolor="#94A3B8", linewidth=0.75,
                    label=f"{eng_u} d=0",
                ))
                for _lo_d, _hi_d, dcol, dlbl in _depth_bins_spec(eng, smax):
                    async_depth_handles.append(Patch(
                        facecolor=dcol, edgecolor="#334155", linewidth=0.6,
                        label=f"{eng_u} {dlbl}",
                    ))

        def _span_color(sp: Mapping[str, Any]) -> str:
            if sp.get("async_engine") in _ASYNC_ENGINE_STYLE:
                return _ASYNC_ENGINE_STYLE[sp["async_engine"]]["panel_b_color"]
            lbl = str(sp.get("label") or sp["sem"]).strip()
            return stage_color_map.get((sp["sem"], lbl), STAGE_COLORS.get(sp["sem"], _DEFAULT_COLOR))[0]

        has_second_async_leg = bool(async_depth_handles)
        n_top_rows = 0
        n_async_rows = 0
        n_leg_cols = 5
        leg_y = (0.941 if has_second_async_leg else 0.933) if has_zoom else (0.931 if has_hw else 0.912)
        if top_legend_handles:
            max_top_lbl = max((len(str(h.get_label() or "")) for h in top_legend_handles), default=0)
            n_leg_cols = min(len(top_legend_handles), 4 if max_top_lbl >= 30 else (5 if max_top_lbl >= 22 else 6))
            n_top_rows = (len(top_legend_handles) + n_leg_cols - 1) // n_leg_cols
            fig.legend(
                handles=top_legend_handles, loc="upper center", bbox_to_anchor=(0.5, leg_y),
                ncol=n_leg_cols, frameon=False,
                fontsize=10.2 if max_top_lbl >= 32 else (10.6 if n_leg_cols >= 5 else 11.0),
                handlelength=1.40, columnspacing=1.08,
            )
        if has_second_async_leg:
            n_acol = min(len(async_depth_handles), 9)
            n_async_rows = (len(async_depth_handles) + n_acol - 1) // n_acol
            aleg_y = leg_y - (0.0195 * max(1, n_top_rows) + 0.011)
            fig.legend(
                handles=async_depth_handles, loc="upper center", bbox_to_anchor=(0.5, aleg_y),
                ncol=n_acol, frameon=True, facecolor="#F8FAFC", edgecolor="#94A3B8",
                fontsize=9.8 if n_acol >= 9 else (10.2 if n_acol >= 8 else 10.8),
                handlelength=1.35, handletextpad=0.42, columnspacing=0.90, borderpad=0.36,
            )

    x_pad = max(1.0, t_max * 0.015)
    x_lim = (0.0, t_max + x_pad)

    # ---- Panel (A): real same-run hardware utilization curves ----
    hw_summary_items: List[Tuple[str, str, float, float]] = []
    if ax_hw is not None:
        ax_hw.set_facecolor("#FFFFFF")
        ax_hw.grid(True, axis="y", linestyle="-", linewidth=0.5, color="#ECEFF1", zorder=0)
        ax_hw.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#F5F7F8", zorder=0)
        ax_hw.axhline(100.0, color="#CFD8DC", linestyle="--", linewidth=0.8, zorder=1)
        ax_hw.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)

        comm_pts = hw_curves.get("nvlink_bw") or []
        nvl_tx_pts = hw_curves.get("nvlink_tx") or []
        nvl_rx_pts = hw_curves.get("nvlink_rx") or []
        has_active_nvl_tx = bool(nvl_tx_pts and max((p[1] for p in nvl_tx_pts if 0.0 <= p[0] <= k_dur), default=0.0) >= 0.5)
        has_active_nvl_rx = bool(nvl_rx_pts and max((p[1] for p in nvl_rx_pts if 0.0 <= p[0] <= k_dur), default=0.0) >= 0.5)
        has_active_comm = bool(comm_pts and max((p[1] for p in comm_pts if 0.0 <= p[0] <= k_dur), default=0.0) >= 0.5)
        rdma_pts = hw_curves.get("inter_rdma") or []
        has_active_rdma = bool(rdma_pts and max((p[1] for p in rdma_pts if 0.0 <= p[0] <= k_dur), default=0.0) > 0.0)

        curve_order_keys = ["tensor_active", "dram_bw"]
        # NVLink TX (egress) and NVLink RX (ingress) are independent full-duplex paths: plot separately, NEVER merged.
        if has_active_nvl_tx:
            curve_order_keys.append("nvlink_tx")
        if has_active_nvl_rx:
            curve_order_keys.append("nvlink_rx")
        if not (has_active_nvl_tx or has_active_nvl_rx):
            if has_active_comm:
                curve_order_keys.append("nvlink_bw")
            elif hw_curves.get("sm_issue"):
                curve_order_keys.append("sm_issue")
        if has_active_rdma:
            curve_order_keys.append("inter_rdma")
        if hw_curves.get("sm_active"):
            curve_order_keys.append("sm_active")

        style_map = {
            "tensor_active": ("#1565C0", "-", 1.9),
            "dram_bw": ("#E65100", "-", 1.6),
            "nvlink_tx": ("#00897B", "-", 1.6),
            "nvlink_rx": ("#2E7D32", "-.", 1.55),
            "nvlink_bw": ("#00897B", "-", 1.6),
            "inter_rdma": ("#7B1FA2", "-", 1.7),
            "sm_issue": ("#00897B", "-", 1.5),
            "sm_active": ("#37474F", "--", 1.35),
        }
        chips: List[Tuple[str, str]] = []
        hw_max_val = 100.0
        for key in curve_order_keys:
            pts = hw_curves.get(key)
            if not pts:
                continue
            xs = [p[0] for p in pts if 0.0 <= p[0] <= t_max + x_pad]
            ys = [p[1] for p in pts if 0.0 <= p[0] <= t_max + x_pad]
            if not xs:
                continue
            color, ls, lw = style_map[key]
            label, short = _pretty_hw_curve_label(hw_labels.get(key, ""), key, is_amd=is_amd, multi_node=multi_node)
            ax_hw.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=lw, zorder=4)
            in_w = [p[1] for p in pts if 0.0 <= p[0] <= k_dur] or ys
            mean_v = sum(in_w) / len(in_w)
            peak_v = max(in_w)
            hw_max_val = max(hw_max_val, peak_v)
            chips.append((f"{short}  peak {peak_v:.1f}% · mean {mean_v:.1f}%", color))
            hw_summary_items.append((short, color, peak_v, mean_v))

        y_top = 108.0 if hw_max_val <= 100.0 else max(108.0, hw_max_val * 1.08)
        ax_hw.set_ylim(-4, y_top)
        ax_hw.set_yticks([0, 25, 50, 75, 100])
        ax_hw.tick_params(axis="y", labelsize=10.4)
        ax_hw.set_ylabel("Utilization (%)", fontsize=11.6, fontweight="bold", color="#1E293B")
        hw_title_src = "same-run SW+HW capture" if is_amd else "real nsys GPU metric curves"
        ax_hw.set_title(f"(A) Aligned Hardware Utilization Curves ({hw_title_src})",
                        fontsize=12.8, fontweight="bold", color="#0F172A", pad=50)
        ax_hw.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.35),
            ncol=min(max(2, len(curve_order_keys)), 3),
            frameon=False, fontsize=10.8, handlelength=2.1, columnspacing=1.35,
        )

        if multi_node and not has_active_rdma:
            ax_hw.text(
                0.012, 0.78,
                "Inter-Node RDMA NIC counters unavailable on this HCA (omitted, not zero-filled)",
                transform=ax_hw.transAxes, ha="left", va="center",
                fontsize=10.2, fontstyle="italic", color="#8B1E1E", zorder=6,
            )

    # ---- Panel (B): All physical SMs / CUs dense rasterized waterfall ----
    ax_sm.set_facecolor("#FFFFFF")
    img_w = 2400
    sm_to_row = {sm: idx for idx, sm in enumerate(sms)}
    all_warps = sorted({w for wmap in sm_warp_role.values() for w in wmap} | {sp["warp"] for sp in sw_spans})
    warp_to_sub = {w: idx for idx, w in enumerate(all_warps)}
    sm_cta_slots: Dict[Tuple[int, int], int] = {}
    if len(all_warps) == 1 and use_cta_slots:
        spans_by_sm: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for sp in sw_spans:
            spans_by_sm[sp["sm"]].append(sp)
        max_slots = 1
        for sm_id, sps in spans_by_sm.items():
            c2s, sc = _pack_concurrent_ctas(sps)
            max_slots = max(max_slots, sc)
            for c, sl in c2s.items():
                sm_cta_slots[(sm_id, c)] = sl
        n_sub = max_slots
    else:
        n_sub = max(1, len(all_warps))
    img = np.full((n_sms * n_sub, img_w, 3), 255, dtype=np.uint8)
    if dual_fabric:
        panel_b_spans = sw_spans
    else:
        # Include async MMA (e.g. Hopper wgmma / Blackwell tcgen05) and background MEM spans on
        # Panel (B) so chip-wide compute utilization renders in Tensor Core Blue instead of being
        # hidden behind short Exposed Wait bars!
        panel_b_spans = list(main_spans) + list(async_spans)
    has_compute_b = any(sp["sem"] in ("mma", "gemm", "compute") for sp in panel_b_spans)
    sorted_spans = sorted(
        panel_b_spans,
        key=lambda s: (
            0 if s.get("async_engine") == "mem" else (
                1 if s["sem"] == "barrier" else (
                    2 if s.get("async_engine") == "mma" else (
                        _STAGE_DRAW_PRIORITY.get(s["sem"], 4) + 2
                    )
                )
            ),
            float(s["dur"]),
        ),
    )
    scale_x = (img_w - 1) / max(t_max + x_pad, 1e-6)
    for sp in sorted_spans:
        row = sm_to_row.get(sp["sm"])
        if row is None:
            continue
        # Suppress sub-pixel (<0.45 px) wait ticks on Panel (B) when compute spans are present, so
        # rapid ~0.08 us sync ticks don't expand to 1-pixel orange walls over healthy Tensor Core work
        # (Panel C & Panel D still display every individual wait tick verbatim).
        if (not dual_fabric) and has_compute_b and sp["sem"] == "wait" and (float(sp["dur"]) * scale_x) < 0.45:
            continue
        if sm_cta_slots:
            sub = sm_cta_slots.get((sp["sm"], sp["cta"]), 0)
        else:
            sub = warp_to_sub.get(sp["warp"], 0)
        x0 = max(0, min(img_w - 1, int(sp["ts"] * scale_x)))
        x1 = max(x0 + 1, min(img_w, int(math.ceil(sp["end"] * scale_x))))
        if dual_fabric and sp.get("_frole") in ("inter_dispatch", "inter_combine", "intra_combine"):
            x1 = max(x1, min(img_w, x0 + 2))
        col_hex = _span_color(sp)
        if (not dual_fabric) and sp.get("async_engine") == "mem":
            col_hex = "#80CBC4"  # soft teal-green background for async mem when layered under MMA
        img[row * n_sub + sub, x0:x1] = _hex_to_rgb(col_hex)

    ax_sm.imshow(img, aspect="auto", interpolation="nearest", extent=[0.0, t_max + x_pad, n_sms - 0.5, -0.5])
    ax_sm.axvline(k_dur, color="#263238", linestyle="--", linewidth=1.0, alpha=0.75)
    step = max(1, (n_sms // 6 + 4) // 5 * 5) if n_sms >= 15 else max(1, n_sms // 4)
    yt = list(range(0, n_sms, step))
    if n_sms > 1 and ((n_sms - 1) - yt[-1]) >= max(2, step // 2):
        yt.append(n_sms - 1)
    dense_cu_idx = bool(is_amd and len(sms) == 80 and sms and sms[-1] >= 80)
    ax_sm.set_yticks(yt)
    ax_sm.set_yticklabels(
        ([f"{unit_nm}{i}" for i in yt] if dense_cu_idx else [f"{unit_nm}{sms[i]}" for i in yt])
        if sms else [f"{unit_nm}0"],
        fontsize=10.2,
    )
    rank_tag = f" (rank {meta['rank']})" if meta.get("rank") is not None else f" ({len(sms)} {unit_nm}s)"
    ax_sm.set_ylabel(f"Physical {unit_nm}{rank_tag}", fontsize=11.6, fontweight="bold", color="#1E293B")
    if dual_fabric and panel_b_fabric_handles:
        b_mode_lbl = "Multi-CTA" if sm_cta_slots else f"Persistent 1-CTA/{unit_nm}"
        ax_sm.set_title(
            f"(B) {b_mode_lbl} Execution ({unit_nm}0–{unit_nm}{len(sms)-1})",
            fontsize=11.6, fontweight="bold", color="#0F172A", loc="left", pad=21,
        )
        ax_sm.legend(
            handles=panel_b_fabric_handles, loc="lower right", bbox_to_anchor=(1.0, 1.002),
            ncol=4, frameon=False, fontsize=9.0, handlelength=1.25, columnspacing=0.95,
        )
    else:
        u_hi = (len(sms) - 1) if dense_cu_idx else (sms[-1] if sms else 0)
        u_lo = 0 if dense_cu_idx else (sms[0] if sms else 0)
        ax_sm.set_title(
            f"(B) All {len(sms)} physical {unit_nm}s ({unit_nm}{u_lo}–{unit_nm}{u_hi}) — full per-tile stage waterfall"
            if sms else f"(B) Physical {unit_nm} timeline",
            fontsize=12.8, fontweight="bold", color="#0F172A", pad=11,
        )

    # ---- Panel (C): Representative SM/CU warp/wave role breakdown (+ thin async sub-tracks) ----
    warp_to_y = {w: idx for idx, (w, _r) in enumerate(rep_warps)}
    rep_main_sorted = [sp for sp in sorted_spans if sp["sm"] == rep_sm and not sp.get("async_engine")]
    rep_spans_for_lbl = [sp for sp in sorted_spans if sp["sm"] == rep_sm]
    _render_warp_panel(
        ax_wp,
        rep_main=rep_main_sorted if not dual_fabric else rep_spans_for_lbl,
        rep_async=[] if dual_fabric else rep_async_raw,
        rep_marks=[] if dual_fabric else rep_marks_raw,
        n_rep_rows=n_rep_rows,
        use_cta_slots=use_cta_slots,
        rep_cta_slot=rep_cta_slot,
        warp_to_y=warp_to_y,
        span_color_fn=_span_color,
        dual_fabric=dual_fabric,
        x_lim=x_lim,
        k_dur=k_dur,
        is_zoom=False,
        show_right_stats=has_async_rep,
    )

    yt_fs = 8.2 if n_rep_rows >= 18 else (9.0 if n_rep_rows >= 14 else 10.2)
    ax_wp.tick_params(axis="x", labelsize=10.4)
    if use_cta_slots:
        role_lbl = rep_warps[0][1] if rep_warps else "cta"
        c_yticks = [f"Slot {i} · {role_lbl}" for i in range(n_rep_rows)]
        ax_wp.set_yticklabels(c_yticks, fontsize=yt_fs)
        ax_wp.set_ylabel(f"{unit_nm} {rep_sm} CTA slots", fontsize=11.6, fontweight="bold", color="#1E293B")
        ax_wp.set_title(
            f"(C) {unit_nm} {rep_sm} — {n_rep_rows} concurrent CTA slots ({len(rep_cta_slot)} CTAs total on {unit_nm} {rep_sm})",
            fontsize=12.6, fontweight="bold", color="#0F172A", pad=8,
        )
    else:
        c_yticks = _compute_panel_c_role_labels(rep_warps, rep_spans_for_lbl, sw_spans, dual_fabric=dual_fabric, is_amd=is_amd)
        ax_wp.set_yticklabels(c_yticks, fontsize=yt_fs)
        ax_wp.set_ylabel(f"{unit_nm} {rep_sm} {lane_nm}", fontsize=11.6, fontweight="bold", color="#1E293B")
        if dual_fabric and panel_c_handles:
            n_c_cols = 4 if len(panel_c_handles) > 5 else len(panel_c_handles)
            n_c_rows = (len(panel_c_handles) + n_c_cols - 1) // max(1, n_c_cols)
            ax_wp.set_title(
                f"(C) Representative {unit_nm} {rep_sm} — Complete Pipeline by Real {lane_nm.title()} Role",
                fontsize=12.4, fontweight="bold", color="#0F172A", pad=46 if n_c_rows >= 2 else 32,
            )
            ax_wp.legend(
                handles=panel_c_handles, loc="upper center",
                bbox_to_anchor=(0.5, 1.255 if n_c_rows >= 2 else 1.155),
                ncol=n_c_cols, frameon=False, fontsize=9.4, handlelength=1.30, columnspacing=1.05,
            )
        else:
            role_names = ", ".join(dict.fromkeys(r for _, r in rep_warps))
            async_tag = " · main track + async depth strip & issue pins (v)" if has_async_rep else ""
            c_ttl = f"(C) {unit_nm} {rep_sm} — all {len(rep_warps)} {lane_nm}s ({role_names}){async_tag}"
            if len(c_ttl) > 155 and has_async_rep:
                c_ttl = f"(C) {unit_nm} {rep_sm} — {len(rep_warps)} {lane_nm}s ({role_names}) · async depth & pins (v)"
            ax_wp.set_title(
                c_ttl,
                fontsize=9.6 if len(c_ttl) > 140 else (10.6 if len(c_ttl) > 115 else (11.6 if len(c_ttl) > 88 else 12.6)),
                fontweight="bold", color="#0F172A", pad=8,
            )
    if has_zoom and zoom_win is not None and ax_zoom is not None:
        z0, z1, z_cta, z_wait, z_reason = zoom_win
        if ax_hw is not None:
            ax_hw.axvspan(z0, z1, facecolor="#FCE4EC", edgecolor="#D81B60", alpha=0.22, lw=1.2, ls="--", zorder=1.5)
        ax_sm.axvspan(z0, z1, facecolor="none", edgecolor="#D81B60", alpha=0.85, lw=1.2, ls="--", zorder=5.0)
        ax_wp.add_patch(Rectangle(
            (z0, -0.46), z1 - z0, n_rep_rows - 0.08,
            facecolor="none", edgecolor="#D81B60", linewidth=1.6, linestyle="--", zorder=7.0,
        ))
        ax_wp.set_xlabel("Full-kernel locked time (µs) — dashed magenta window zoomed in Panel (D)", fontsize=10.8, color="#334155")
        # Isolate the single featured CTA in Panel (D) when a single CTA is selected, so concurrent
        # resident CTAs on the same SM never interleave their pins/stages in micro-zoom view!
        if z_cta is not None and (zoom_cta is not None or not dual_fabric):
            base_spans = rep_spans_for_lbl if dual_fabric else rep_main_sorted
            z_main = [sp for sp in base_spans if int(sp["cta"]) == z_cta] or base_spans
            z_async = [sp for sp in rep_async_raw if int(sp["cta"]) == z_cta]
            z_marks = [mk for mk in rep_marks_raw if int(mk["cta"]) == z_cta]
        else:
            z_main = rep_spans_for_lbl if dual_fabric else rep_main_sorted
            z_async = [] if dual_fabric else rep_async_raw
            z_marks = [] if dual_fabric else rep_marks_raw
        _render_warp_panel(
            ax_zoom,
            rep_main=z_main,
            rep_async=z_async,
            rep_marks=z_marks,
            n_rep_rows=n_rep_rows,
            use_cta_slots=use_cta_slots,
            rep_cta_slot=rep_cta_slot,
            warp_to_y=warp_to_y,
            span_color_fn=_span_color,
            dual_fabric=dual_fabric,
            x_lim=(z0, z1),
            k_dur=k_dur,
            is_zoom=True,
            show_right_stats=has_async_rep,
        )
        ax_zoom.tick_params(axis="x", labelsize=10.4)
        ax_zoom.set_yticklabels(c_yticks, fontsize=yt_fs)
        ax_zoom.set_ylabel(f"{unit_nm} {rep_sm} zoom", fontsize=11.6, fontweight="bold", color="#1E293B")
        cta_lbl = f"CTA {z_cta} on {unit_nm} {rep_sm}" if z_cta is not None else f"{unit_nm} {rep_sm}"
        zoom_detail_tail = (
            "compute & multi-role comm overlap"
            if not has_async_rep
            else "every async issue pin (v) & stage depth"
        )
        zoom_xlabel_tail = (
            "multi-role compute & comm stage overlap detail"
            if not has_async_rep
            else "isolated single-CTA async pipeline detail"
        )
        d_ttl = f"(D) Deterministic Micro-Zoom: {cta_lbl} ({z0:.1f}–{z1:.1f} µs · {z_reason}) — {zoom_detail_tail}"
        ax_zoom.set_title(
            d_ttl,
            fontsize=10.8 if len(d_ttl) > 115 else (11.6 if len(d_ttl) > 90 else 12.4),
            fontweight="bold", color="#880E4F", pad=8,
        )
        ax_zoom.set_xlabel(f"Zoomed time (µs) — {zoom_xlabel_tail}", fontsize=11.2, fontweight="bold", color="#1E293B")
    else:
        ax_wp.set_xlabel("Time (µs) — identical locked scale across all panels", fontsize=11.2, fontweight="bold", color="#1E293B")

    yt_labels = [t.get_text() for t in ax_wp.get_yticklabels()]
    max_lbl = max((len(s) for s in yt_labels), default=10)
    left_m = max(0.114, min(0.215, 0.058 + 0.0053 * max_lbl))
    right_m = (0.754 if len(active_async_engines) >= 2 else 0.786) if has_async_rep else 0.984
    if subplot_lr is not None:
        left_m, right_m = float(subplot_lr[0]), float(subplot_lr[1])
    _max_tl = max((len(str(h.get_label() or "")) for h in top_legend_handles), default=0) if top_legend_handles else 0
    _ncol_t = min(len(top_legend_handles), 4 if _max_tl >= 30 else (5 if _max_tl >= 22 else 6)) if top_legend_handles else 1
    _ntop = max(1, (len(top_legend_handles) + _ncol_t - 1) // _ncol_t) if (not dual_fabric and top_legend_handles) else 0
    _nasy = max(1, (len(async_depth_handles) + 8) // 9) if (not dual_fabric and bool(active_async_engines)) else 0
    extra_leg = (0.0215 * max(0, _ntop - 1)) + ((0.034 + 0.0205 * max(0, _nasy - 1)) if _nasy > 0 else 0.0)
    if has_zoom:
        if dual_fabric:
            top_m = 0.858 if has_hw else 0.875
        else:
            top_m = min(0.842, 1.0 - 2.65 / fig_h) - extra_leg if has_hw else min(0.855, 1.0 - 2.20 / fig_h) - extra_leg
    else:
        top_m = 0.840 if (has_hw and dual_fabric) else (min(0.790, 1.0 - 2.65 / fig_h) - extra_leg if has_hw else 0.818)
    show_hw_box = bool(hw_summary_items)
    two_row_hw = show_hw_box and len(hw_summary_items) >= 3
    if show_hw_box:
        if two_row_hw:
            bottom_m = 0.118 if has_zoom else 0.142
        else:
            bottom_m = 0.094 if has_zoom else 0.112
    else:
        bottom_m = 0.058 if has_zoom else 0.068
    fig.subplots_adjust(left=left_m, right=right_m, top=top_m, bottom=bottom_m)

    # ---- Bottom Hardware Metrics Summary Box (Large-Font Peak & Mean Readout) ----
    if show_hw_box:
        from matplotlib.patches import FancyBboxPatch
        box_x0 = 0.028
        box_x1 = 0.980
        box_w = max(0.20, box_x1 - box_x0)
        box_y0 = 0.010
        n_hw_it = len(hw_summary_items)
        hdr_txt = f"Hardware Summary (0–{k_dur:,.1f} µs):"
        if two_row_hw:
            box_h = 0.066 if has_zoom else 0.080
            fig.patches.append(FancyBboxPatch(
                (box_x0, box_y0), box_w, box_h,
                boxstyle="round,pad=0.004,rounding_size=0.007",
                transform=fig.transFigure,
                facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.45, zorder=10,
            ))
            y_r1 = box_y0 + 0.72 * box_h
            y_r2 = box_y0 + 0.27 * box_h
            sw_h = box_h * 0.29
            sw_w = 0.0102
            fs_hw = 12.4
            # Row 1: Header on left + first 1 or 2 hardware metrics
            fig.text(
                box_x0 + 0.012, y_r1, hdr_txt,
                ha="left", va="center", fontsize=13.0, fontweight="bold", color="#0F172A", zorder=11,
            )
            n_r1 = 1 if n_hw_it == 3 else 2
            r1_items = hw_summary_items[:n_r1]
            r1_x_starts = [0.420] if n_r1 == 1 else [0.365, 0.678]
            for idx_r1, (m_short, m_col, m_peak, m_mean) in enumerate(r1_items):
                ix = r1_x_starts[idx_r1]
                fig.patches.append(Rectangle(
                    (ix, y_r1 - 0.5 * sw_h), sw_w, sw_h,
                    transform=fig.transFigure, facecolor=m_col, edgecolor="#0F172A", linewidth=0.7, zorder=11,
                ))
                fig.text(
                    ix + sw_w + 0.006, y_r1,
                    f"{m_short}: peak {m_peak:.1f}%, mean {m_mean:.1f}%",
                    ha="left", va="center", fontsize=fs_hw, fontweight="bold", color="#1E293B", zorder=11,
                )
            # Row 2: Remaining 2 or 3 hardware metrics evenly distributed across box width
            r2_items = hw_summary_items[n_r1:]
            n_r2 = max(1, len(r2_items))
            r2_x0 = box_x0 + 0.012
            r2_avail = (box_x1 - 0.012) - r2_x0
            r2_slot = r2_avail / n_r2
            for idx_r2, (m_short, m_col, m_peak, m_mean) in enumerate(r2_items):
                ix = r2_x0 + idx_r2 * r2_slot
                fig.patches.append(Rectangle(
                    (ix, y_r2 - 0.5 * sw_h), sw_w, sw_h,
                    transform=fig.transFigure, facecolor=m_col, edgecolor="#0F172A", linewidth=0.7, zorder=11,
                ))
                fig.text(
                    ix + sw_w + 0.006, y_r2,
                    f"{m_short}: peak {m_peak:.1f}%, mean {m_mean:.1f}%",
                    ha="left", va="center", fontsize=fs_hw, fontweight="bold", color="#1E293B", zorder=11,
                )
        else:
            box_h = 0.042 if has_zoom else 0.052
            box_yc = box_y0 + 0.5 * box_h
            fig.patches.append(FancyBboxPatch(
                (box_x0, box_y0), box_w, box_h,
                boxstyle="round,pad=0.004,rounding_size=0.007",
                transform=fig.transFigure,
                facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.45, zorder=10,
            ))
            hdr_x = box_x0 + 0.012
            fig.text(
                hdr_x, box_yc, hdr_txt,
                ha="left", va="center", fontsize=12.6, fontweight="bold", color="#0F172A", zorder=11,
            )
            items_x0 = hdr_x + 0.305
            avail_w = max(0.10, (box_x1 - 0.010) - items_x0)
            slot_w = avail_w / max(1, n_hw_it)
            sw_h = box_h * 0.44
            sw_w = 0.0102
            for idx_h, (m_short, m_col, m_peak, m_mean) in enumerate(hw_summary_items):
                ix = items_x0 + idx_h * slot_w
                fig.patches.append(Rectangle(
                    (ix, box_yc - 0.5 * sw_h), sw_w, sw_h,
                    transform=fig.transFigure, facecolor=m_col, edgecolor="#0F172A", linewidth=0.7, zorder=11,
                ))
                fig.text(
                    ix + sw_w + 0.006, box_yc,
                    f"{m_short}: peak {m_peak:.1f}%, mean {m_mean:.1f}%",
                    ha="left", va="center", fontsize=12.0, fontweight="bold", color="#1E293B", zorder=11,
                )

    # Programmatic collision and canvas-bounds check over all visible text artists
    from matplotlib.text import Text as _MplText
    fig.canvas.draw()
    _rnd = fig.canvas.get_renderer()
    _cw, _ch = fig.canvas.get_width_height()
    if subplot_lr is None:
        # (1) Widen left_m up to LEFT_M_MAX only when a y-axis title starts left of x0 = 2.0 px
        _yl_x0s = [_a.yaxis.label.get_window_extent(_rnd).x0 for _a in fig.axes
                   if _a.yaxis.label.get_visible() and str(_a.yaxis.label.get_text() or "").strip()]
        _min_x0 = min(_yl_x0s) if _yl_x0s else 2.0
        if _min_x0 < 2.0:
            _new_left_m = min(LEFT_M_MAX, left_m + (3.0 - _min_x0) / fig.bbox.width)
            if _new_left_m > left_m:
                render_swhw_overview_png.last_left_rescue_px = float((_new_left_m - left_m) * fig.bbox.width)
                left_m = _new_left_m
                fig.subplots_adjust(left=left_m)
                fig.canvas.draw()
                _rnd = fig.canvas.get_renderer()
                render_swhw_overview_png.last_extra_draws += 1
            _yl_x0s = [_a.yaxis.label.get_window_extent(_rnd).x0 for _a in fig.axes
                       if _a.yaxis.label.get_visible() and str(_a.yaxis.label.get_text() or "").strip()]
            _min_x0 = min(_yl_x0s) if _yl_x0s else 2.0
        # (2) Truncate Panel C/D tick labels at LEFT_M_MAX & (3) Panel C title if out of canvas bounds
        _ct_txt = str(ax_wp.get_title() or "")
        _ct_bb = ax_wp.title.get_window_extent(_rnd) if _ct_txt.strip() else None
        _ct_clip = bool(_ct_bb is not None and (_ct_bb.x0 < 2.0 or _ct_bb.x1 > _cw - 2.0))
        if (_min_x0 < 2.0 or _ct_clip) and render_swhw_overview_png.last_extra_draws < LEFT_RESCUE_MAX_EXTRA_DRAWS:
            _need_redraw = False
            if _min_x0 < 2.0:
                _lane_axes = [ax_wp] + ([ax_zoom] if (has_zoom and zoom_win is not None and ax_zoom is not None) else [])
                _budgets = [
                    max((_t.get_window_extent(_rnd).width for _t in _a.get_yticklabels() if _t.get_text()), default=0.0)
                    - (3.0 - _a.yaxis.label.get_window_extent(_rnd).x0)
                    for _a in _lane_axes
                    if _a.yaxis.label.get_visible() and str(_a.yaxis.label.get_text() or "").strip()
                    and _a.yaxis.label.get_window_extent(_rnd).x0 < 2.0
                ]
                _probe_tk = next((_t for _t in ax_wp.get_yticklabels() if _t.get_text()), None)
                if _budgets and _probe_tk is not None:
                    _max_tw = min(_budgets)
                    _probe = _MplText(0.0, 0.0, "", fontproperties=_probe_tk.get_fontproperties())
                    _probe.set_figure(fig)
                    _new_yticks = [
                        _truncate_text_to_px(_probe, _rnd, _lbl, _m.end() if _m else 0, _max_tw)
                        for _lbl in c_yticks
                        for _m in (re.match(r"^(?:W\d+|Slot\s+\d+)(?:\s*·\s*|\s+\()", _lbl),)
                    ]
                    if _new_yticks != list(c_yticks):
                        c_yticks = _new_yticks
                        for _a in _lane_axes:
                            _a.set_yticklabels(c_yticks, fontsize=yt_fs)
                        _need_redraw = True
            if _ct_clip:
                _ct_pfx = (f"(C) Representative {unit_nm} {rep_sm} \u2014 " if _ct_txt.startswith("(C) Representative ")
                           else f"(C) {unit_nm} {rep_sm} \u2014 ")
                _cx = 0.5 * (ax_wp.bbox.x0 + ax_wp.bbox.x1)
                _max_ct_w = 2.0 * min(_cx - 3.0, (_cw - 3.0) - _cx)
                _p_ct = _MplText(0.0, 0.0, "", fontproperties=ax_wp.title.get_fontproperties())
                _p_ct.set_figure(fig)
                _new_ct = _truncate_text_to_px(_p_ct, _rnd, _ct_txt, len(_ct_pfx) if _ct_txt.startswith(_ct_pfx) else 0, _max_ct_w)
                if _new_ct != _ct_txt:
                    ax_wp.title.set_text(_new_ct)
                    render_swhw_overview_png.last_panel_c_title_truncated = True
                    _need_redraw = True
            if _need_redraw:
                fig.canvas.draw()
                _rnd = fig.canvas.get_renderer()
                render_swhw_overview_png.last_extra_draws += 1
    _oob_ticks = set()
    for _ax_i in fig.axes:
        _xl0, _xl1 = sorted(_ax_i.get_xlim())
        _yl0, _yl1 = sorted(_ax_i.get_ylim())
        for _tk in _ax_i.xaxis.get_major_ticks() + _ax_i.xaxis.get_minor_ticks():
            _loc = _tk.get_loc()
            if _loc < _xl0 - 1e-6 or _loc > _xl1 + 1e-6:
                _oob_ticks.add(id(_tk.label1))
                _oob_ticks.add(id(_tk.label2))
        for _tk in _ax_i.yaxis.get_major_ticks() + _ax_i.yaxis.get_minor_ticks():
            _loc = _tk.get_loc()
            if _loc < _yl0 - 1e-6 or _loc > _yl1 + 1e-6:
                _oob_ticks.add(id(_tk.label1))
                _oob_ticks.add(id(_tk.label2))
    _vis_texts = []
    _seen_ids = set()
    for _t in fig.findobj(_MplText):
        if id(_t) in _seen_ids or id(_t) in _oob_ticks:
            continue
        _seen_ids.add(id(_t))
        if _t.get_visible():
            _s = str(_t.get_text() or "").strip()
            if _s:
                _bb_t = _t.get_window_extent(_rnd)
                if _bb_t.x0 < 2.0 or _bb_t.x1 > _cw - 2.0 or _bb_t.y0 < 2.0 or _bb_t.y1 > _ch - 2.0:
                    raise RuntimeError(f"TEXT_CLIPPED in viz.py: {_s!r} {_bb_t} outside canvas {_cw}x{_ch}")
                _vis_texts.append((_s, _bb_t))
    for _i in range(len(_vis_texts)):
        _sa, _ba = _vis_texts[_i]
        for _j in range(_i + 1, len(_vis_texts)):
            _sb, _bb = _vis_texts[_j]
            if (min(_ba.x1, _bb.x1) - max(_ba.x0, _bb.x0) > 1.5 and
                    min(_ba.y1, _bb.y1) - max(_ba.y0, _bb.y0) > 1.5):
                raise RuntimeError(f"TEXT_OVERLAP in viz.py: {_sa!r} {_ba} overlaps {_sb!r} {_bb}")

    default_dir = Path(trace_or_path).resolve().parent if isinstance(trace_or_path, (str, Path)) else None
    out_path = resolve_timeline_png_path(
        out_png,
        trace,
        default_dir=default_dir,
        model=model,
        op=op,
        bs=bs,
        seql=seql,
        kv=kv,
        shape=shape,
        dims=dims,
        title=title,
        subtitle=subtitle,
        source=trace_or_path if isinstance(trace_or_path, (str, Path)) else None,
        timestamp=timestamp,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=int(dpi), facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Render a WeaveProfiler Perfetto trace into a 4-panel SW+HW timeline PNG.")
    ap.add_argument("trace", help="Path to trace.perfetto.json[.gz]")
    ap.add_argument("-o", "--out", default=None, help="Output PNG path or directory (default: auto <model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png)")
    ap.add_argument("--model", default=None, help="Model name token for auto-generated filename")
    ap.add_argument("--op", default=None, help="Operator name token for auto-generated filename (e.g. gemm, attn, moe)")
    ap.add_argument("--bs", default=None, help="Batch size for auto-generated filename")
    ap.add_argument("--seql", default=None, help="Sequence length for auto-generated filename")
    ap.add_argument("--kv", default=None, help="KV cache length for auto-generated filename")
    ap.add_argument("--shape", default=None, help="Explicit shape token (overrides bs/seql/dims in filename)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--sm", type=int, default=None, help="Physical SM/CU id to feature in Panel (C)")
    ap.add_argument("--zoom-cta", type=int, default=None, help="Override Panel (D) micro-zoom CTA id")
    ap.add_argument("--zoom-window", default=None, help="Override Panel (D) micro-zoom window 't0,t1' in us")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--allow-no-hw", action="store_true", help="Debug-only escape hatch: allow rendering without Panel (A) hardware utilization curves")
    args = ap.parse_args(argv)
    zw = None
    if args.zoom_window:
        parts = [float(x.strip()) for x in str(args.zoom_window).split(",")]
        if len(parts) == 2:
            zw = (parts[0], parts[1])
    out = render_swhw_overview_png(
        args.trace, args.out, title=args.title, subtitle=args.subtitle,
        focus_sm=args.sm, zoom_cta=args.zoom_cta, zoom_window=zw, dpi=args.dpi,
        require_hw_curves=not args.allow_no_hw,
        model=args.model, op=args.op, bs=args.bs, seql=args.seql, kv=args.kv, shape=args.shape,
    )
    print(f"RENDERED {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
