#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3.1 Semantic Probe Placement — Direct Source SEARCH/REPLACE LLM Agent.

Guided by the multi-language operator phase KB, the LLM agent directly inserts
`FI_OPEN / FI_B / FI_E / FI_CLOSE` (or `fi_b / fi_e`) probe macros into the target
kernel source via compact SEARCH/REPLACE edit blocks (~15–30 lines).

Features:
  - Zero `probes.json` indirection: directly modifies kernel source in place or to an output file.
  - Supports the 3 evaluated DAC'27 models at `reasoning_effort="max"`:
      `qwen4-max`, `deepseek-v4.1-flash-1017`, `kimi-k3.1-pro`, `glm-5.3`
  - Supports all 4 Table VI ablation modes:
      1. `bare_direct`      : Bare LLM writes ad-hoc timers and buffering from scratch.
      2. `kb_direct`        : LLM + §3.1 multi-language operator phase KB.
      3. `open_loop`        : §3.1 KB + §3.2 drop-in zero-signature global sidecar probe library.
      4. `full_closed_loop` : + §3.3 `verify.py` batched plain-text terminal error log feedback
                              driving Scheme A inner-loop probe coarsening / hoisting.
  - Turnkey Incremental Reuse (`--prev-source`, `--prev-inst`):
      Diff-aware AST context matching inherits unchanged probe placements in <0.05s
      with 0 LLM tokens across hyperparameter/tuning edits, invoking delta-LLM only on
      modified diff hunks.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.pipeline.scan.llm_probe_placer import (  # noqa: E402
    ABLATION_MODES,
    SUPPORTED_MODELS,
    call_llm,
    detect_language,
    detect_spin_loop_lines,
    extract_code_block,
    resolve_route,
    retrieve_route_kb,
)


def apply_search_replace_blocks(source_text: str, patch_text: str) -> str:
    """Apply compact SEARCH/REPLACE edit blocks to source_text.

    Format:
      <<<<<<< SEARCH
      ... original lines ...
      =======
      ... replacement lines ...
      >>>>>>>
    """
    pattern = re.compile(
        r"<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>>",
        re.DOTALL,
    )
    matches = list(pattern.finditer(patch_text))
    if not matches:
        # Check if the model emitted a full replacement code block instead of SEARCH/REPLACE
        extracted = extract_code_block(patch_text)
        if len(extracted.splitlines()) > 5:
            # If it looks like a full kernel, return it directly
            return extracted
        return source_text

    result = source_text
    for m in matches:
        search_block = m.group(1)
        replace_block = m.group(2)
        if search_block in result:
            result = result.replace(search_block, replace_block, 1)
        else:
            # Try line-trimmed matching to accommodate minor whitespace variance
            s_lines = [l.strip() for l in search_block.splitlines() if l.strip()]
            r_lines = result.splitlines()
            found_idx = -1
            for i in range(len(r_lines) - len(s_lines) + 1):
                window = [r_lines[i + k].strip() for k in range(len(s_lines))]
                if window == s_lines:
                    found_idx = i
                    break
            if found_idx >= 0:
                head = "\n".join(r_lines[:found_idx])
                tail = "\n".join(r_lines[found_idx + len(s_lines):])
                result = (head + "\n" if head else "") + replace_block + ("\n" + tail if tail else "")
    return result


_TRIVIAL_ANCHOR_TOKENS = frozenset({
    "", "{", "}", "};", ")", ");", "} else {", "else {", "else",
    "do {", "do", "try:", "else:", "finally:", "pass", "break;", "continue;", "return;",
})


def _is_nontrivial_anchor(lines: Sequence[str]) -> bool:
    if not lines:
        return False
    for raw in lines:
        s = raw.strip()
        if not s or s in _TRIVIAL_ANCHOR_TOKENS:
            continue
        if s.startswith(("//", "#", "/*")):
            if len(s) >= 24:
                return True
            continue
        if len(s) >= 4:
            return True
    return False


def _find_unique_line_window(haystack: Sequence[str], needle: Sequence[str]) -> Optional[int]:
    """Return start index if `needle` appears as an exact contiguous line slice in `haystack` ONCE."""
    m = len(needle)
    n = len(haystack)
    if m == 0 or m > n:
        return None
    first = needle[0]
    needle_list = list(needle)
    hit = -1
    for idx in range(n - m + 1):
        if haystack[idx] == first and list(haystack[idx: idx + m]) == needle_list:
            if hit >= 0:
                return None
            hit = idx
    return hit if hit >= 0 else None


def _strip_c_comments_and_strings(code: str) -> str:
    return re.sub(
        r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        " ",
        code,
        flags=re.DOTALL | re.MULTILINE,
    )


def _is_pure_header_preamble(lines: Sequence[str]) -> bool:
    if not lines:
        return False
    has_header = False
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith(("//", "#", "/*")):
            if s.startswith(("#include", "#define FI_", "#ifndef FI_", "#define WP_", "#ifndef WP_", "#endif")):
                has_header = True
            continue
        if s.startswith(("import ", "from ")):
            has_header = True
            continue
        return False
    return has_header


def incremental_inherit_probes(
    current_source: str,
    prev_source: str,
    prev_inst_source: str,
) -> Optional[Tuple[str, str]]:
    """Fast incremental probe inheritance across local kernel tuning edits.

    Uses 3-way non-overlapping diff mapping + unique multi-line context anchors
    (rejecting trivial single-line `}` / empty anchors, properly replacing
    `s_prev[i1:i2]` on `tag == 'replace'`, returning `None` on overlapping edits,
    and verifying brace/paren balance + Python `ast.parse` before returning).

    Returns:
      `(instrumented_source, 'diff_hunk_reapplied')` if all probes cleanly map (0 LLM tokens).
      `None` if changes overlap probe regions requiring delta-LLM.
    """
    if current_source == prev_source:
        return (prev_inst_source, "exact_match")

    s_curr = current_source.splitlines()
    s_prev = prev_source.splitlines()
    s_inst = prev_inst_source.splitlines()

    inst_ops = [
        op for op in difflib.SequenceMatcher(None, s_prev, s_inst, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    if not inst_ops:
        return None

    edit_ops = difflib.SequenceMatcher(None, s_prev, s_curr, autojunk=False).get_opcodes()
    equal_spans = [
        (ei1, ei2, ej1, ej2)
        for etag, ei1, ei2, ej1, ej2 in edit_ops
        if etag == "equal" and ei2 > ei1
    ]

    try:
        resolved_edits: List[Tuple[int, int, List[str]]] = []
        for tag, i1, i2, j1, j2 in inst_ops:
            rep_lines = s_inst[j1:j2] if tag in ("insert", "replace") else []

            covering = [
                (ei1, ei2, ej1, ej2)
                for (ei1, ei2, ej1, ej2) in equal_spans
                if ei1 <= i1 and i2 <= ei2
            ]

            if not covering:
                # Special case: pure #include / import header inserted at line 0 when line 0 was edited
                if tag == "insert" and i1 == 0 and i2 == 0 and _is_pure_header_preamble(rep_lines):
                    resolved_edits.append((0, 0, list(rep_lines)))
                    continue
                return None

            if len(covering) > 1:
                # User inserted new lines at exact boundary i1 == i2 where probe was also inserted -> ambiguous
                return None

            ei1, ei2, ej1, ej2 = covering[0]
            span_len = ei2 - ei1
            radii = (1, 2, 3, 4, 6, 8, 12, 16, 24, 40, span_len)
            matched_coords: Optional[Tuple[int, int]] = None

            for pad in radii:
                w1 = max(ei1, i1 - pad)
                w2 = min(ei2, i2 + pad)
                if w2 <= w1:
                    continue
                anchor_slice = s_prev[w1:w2]
                if not _is_nontrivial_anchor(anchor_slice):
                    if w1 == ei1 and w2 == ei2:
                        break
                    continue
                c_w1 = _find_unique_line_window(s_curr, anchor_slice)
                expected_c_w1 = ej1 + (w1 - ei1)
                if c_w1 is not None and c_w1 == expected_c_w1:
                    matched_coords = (c_w1 + (i1 - w1), c_w1 + (i2 - w1))
                    break
                if w1 == ei1 and w2 == ei2:
                    break

            if matched_coords is None:
                if tag == "insert" and i1 == 0 and i2 == 0 and _is_pure_header_preamble(rep_lines):
                    matched_coords = (ej1 if ei1 == 0 else 0, ej1 if ei1 == 0 else 0)
                else:
                    # Short equal span (e.g. closing `}` right after a tuned loop line):
                    # accept global LCS mapping iff local ±6-line neighborhood similarity >= 0.65
                    cand_c1 = ej1 + (i1 - ei1)
                    cand_c2 = ej1 + (i2 - ei1)
                    if s_prev[i1:i2] == s_curr[cand_c1:cand_c2]:
                        prev_nb = "\n".join(s_prev[max(0, i1 - 6): min(len(s_prev), i2 + 6)])
                        curr_nb = "\n".join(s_curr[max(0, cand_c1 - 6): min(len(s_curr), cand_c2 + 6)])
                        if difflib.SequenceMatcher(None, prev_nb, curr_nb, autojunk=False).ratio() >= 0.65:
                            matched_coords = (cand_c1, cand_c2)
                if matched_coords is None:
                    return None

            resolved_edits.append((matched_coords[0], matched_coords[1], list(rep_lines)))

        if not resolved_edits:
            return None

        out_lines: List[str] = []
        cursor = 0
        for c_i1, c_i2, rep_lines in resolved_edits:
            if c_i1 < cursor or c_i2 < c_i1 or c_i2 > len(s_curr):
                return None
            out_lines.extend(s_curr[cursor:c_i1])
            out_lines.extend(rep_lines)
            cursor = c_i2
        out_lines.extend(s_curr[cursor:])

        trailing_nl = "\n" if (current_source.endswith("\n") or prev_inst_source.endswith("\n")) else ""
        patched = "\n".join(out_lines) + trailing_nl
        if patched == current_source:
            return None

        # Python AST validity check when source is Python
        is_python = False
        try:
            ast.parse(prev_source)
            ast.parse(current_source)
            is_python = True
        except SyntaxError:
            is_python = False

        if is_python:
            try:
                ast.parse(patched)
            except SyntaxError:
                return None

        # Raw & comment-stripped brace/paren conservation checks
        inst_brace_delta = (
            (prev_inst_source.count("{") - prev_inst_source.count("}"))
            - (prev_source.count("{") - prev_source.count("}"))
        )
        curr_brace_diff = current_source.count("{") - current_source.count("}")
        patched_brace_diff = patched.count("{") - patched.count("}")
        if patched_brace_diff != curr_brace_diff + inst_brace_delta:
            return None
        if inst_brace_delta == 0 and abs(patched_brace_diff) != abs(curr_brace_diff):
            return None

        if not is_python:
            c_prev = _strip_c_comments_and_strings(prev_source)
            c_inst = _strip_c_comments_and_strings(prev_inst_source)
            c_curr = _strip_c_comments_and_strings(current_source)
            c_pat = _strip_c_comments_and_strings(patched)
            for open_ch, close_ch in (("{", "}"), ("(", ")"), ("[", "]")):
                d_inst = (c_inst.count(open_ch) - c_inst.count(close_ch)) - (
                    c_prev.count(open_ch) - c_prev.count(close_ch)
                )
                d_curr = c_curr.count(open_ch) - c_curr.count(close_ch)
                d_pat = c_pat.count(open_ch) - c_pat.count(close_ch)
                if d_pat != d_curr + d_inst:
                    return None

        return (patched, "diff_hunk_reapplied")
    except Exception:
        return None


_COMPOSABLE_KB_CACHE: Optional[Dict[str, Any]] = None


def load_composable_two_level_kb() -> Dict[str, Any]:
    """Load the Composable Two-Level Operator & Primitive Knowledge Base (§3.1)."""
    global _COMPOSABLE_KB_CACHE
    if _COMPOSABLE_KB_CACHE is not None:
        return _COMPOSABLE_KB_CACHE
    kb_path = ROOT / "configs" / "composable_two_level_kb.json"
    if kb_path.is_file():
        _COMPOSABLE_KB_CACHE = json.loads(kb_path.read_text(encoding="utf-8"))
    else:
        _COMPOSABLE_KB_CACHE = {}
    return _COMPOSABLE_KB_CACHE


def select_matching_topologies(source_text: str, max_k: int = 3) -> List[Dict[str, Any]]:
    """Rank and select the most relevant Level-2 CTA pipeline topologies for `source_text`."""
    kb = load_composable_two_level_kb()
    topologies = kb.get("level_2_cta_topologies") or []
    lower_src = source_text.lower()
    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, topo in enumerate(topologies):
        score = 0
        for cue in topo.get("cues") or []:
            if str(cue).lower() in lower_src:
                score += 10
        tid = str(topo.get("topology_id") or "")
        if "p2p" in tid and ("peer" in lower_src or "p2p" in lower_src or "rem" in lower_src):
            score += 20
        if "ring" in tid and "ring" in lower_src:
            score += 20
        if "moe" in tid and ("moe" in lower_src or "expert" in lower_src):
            score += 18
        if "flash_attention" in tid and ("hq" in lower_src or "hkv" in lower_src or "attn" in lower_src):
            score += 18
        if "softmax" in tid and "softmax" in lower_src and "qk" not in lower_src:
            score += 18
        if "megakernel" in tid and ("n_chain" in lower_src or "megamoe" in lower_src or "fc1" in lower_src):
            score += 18
        if "warp_specialized_gemm" in tid and ("warp_specialized" in lower_src or "producer" in lower_src or "3wg" in lower_src):
            score += 16
        scored.append((score, -idx, topo))
    scored.sort(reverse=True)
    chosen = [t for sc, _, t in scored[:max_k] if sc > 0]
    if not chosen and topologies:
        chosen = [topologies[0]]
    return chosen


def format_two_level_kb_context(
    source_text: str,
    route_kb: Mapping[str, Any],
    lang: str,
    card: str,
    ablation_mode: str,
) -> str:
    """Format Level-1 Primitive Rules + Level-2 CTA Topologies + Route Exemplars into prompt text."""
    kb = load_composable_two_level_kb()
    parts: List[str] = []

    l1 = kb.get("level_1_primitives") or {}
    invs = l1.get("global_invariants") or []
    if invs:
        parts.append("### Level-1 Composable Primitive Invariants")
        for inv in invs:
            if ablation_mode == "kb_direct" and "INV-1" in inv:
                continue
            parts.append(f"- {inv}")

    if ablation_mode != "kb_direct":
        l_contracts = (l1.get("language_contracts") or {}).get(lang) or {}
        if lang == "triton":
            sub = l_contracts.get("amd_contract" if card == "mi308x" else "nvidia_contract") or {}
            pitfalls = sub.get("pitfalls") or []
        else:
            pitfalls = l_contracts.get("pitfalls") or []
        if pitfalls:
            parts.append(f"\n### {lang.upper()} ({card.upper()}) Architecture & Compiler Pitfalls")
            for pf in pitfalls:
                parts.append(f"- {pf}")

    matched_topos = select_matching_topologies(source_text, max_k=2)
    if matched_topos:
        parts.append("\n### Level-2 Matched CTA Pipeline Topologies (§3.1)")
        for topo in matched_topos:
            parts.append(f"- **Topology `{topo.get('topology_id')}`** ({', '.join(topo.get('applies_to') or [])}):")
            parts.append(f"  - Warp Roles: {topo.get('warp_roles')}")
            parts.append(f"  - Compliant Strategy: {topo.get('compliant_placement_strategy')}")
            if ablation_mode != "kb_direct" and topo.get("cuda_hip_example") and lang in ("cuda_cpp", "hip_cpp"):
                parts.append(f"  - Reference Pattern:\n```cpp\n{topo['cuda_hip_example']}\n```")

    # Inject route-specific operator stages & language exemplars from retrieve_route_kb if available
    op_summaries = route_kb.get("op_summaries") or route_kb.get("operators") or []
    if op_summaries:
        parts.append("\n### Route-Specific Operator Stage Signatures (`configs/operators/operators.json`)")
        for op in op_summaries[:2]:
            parts.append(f"- Operator `{op.get('name')}` (`{op.get('operator_semantic_id')}`):")
            for st in (op.get("stages") or [])[:5]:
                parts.append(
                    f"  * Stage `{st.get('stage')}` ({st.get('semantic_ids')}): "
                    f"role={st.get('typical_role')} | rule={st.get('placement_rule')}"
                )

    lang_ex = route_kb.get("lang_exemplars") or []
    if lang_ex and ablation_mode != "kb_direct":
        parts.append("\n### Language Boundary Rules (`patterns.json`)")
        for ex in lang_ex[:4]:
            notes = "; ".join(str(x) for x in (ex.get("language_notes") or [])[:2])
            parts.append(f"- `{ex.get('semantic_id')}`: boundaries={ex.get('event_boundaries')} | {notes}")

    return "\n".join(parts)


def static_preflight_validate(
    source_text: str,
    inst_text: str,
    lang: str,
    card: str,
) -> List[str]:
    """Fast (<5ms) deterministic static preflight checks catching structural/scope/spin-loop errors."""
    issues: List[str] = []
    if inst_text.strip() == source_text.strip():
        issues.append("NO_PROBES_APPLIED: Instrumented code is identical to uninstrumented source; SEARCH blocks failed to match.")
        return issues

    if lang in ("triton", "cutedsl"):
        try:
            ast.parse(inst_text)
        except SyntaxError as exc:
            issues.append(f"PYTHON_SYNTAX_ERROR at line {exc.lineno}: {exc.msg}")
            return issues
    else:
        c_orig = _strip_c_comments_and_strings(source_text)
        c_inst = _strip_c_comments_and_strings(inst_text)
        if (c_inst.count("{") - c_inst.count("}")) != (c_orig.count("{") - c_orig.count("}")):
            issues.append("CPP_BRACE_IMBALANCE: Unbalanced `{` and `}` braces introduced by edit.")
        if re.search(r'\b(?:FiBuf|WpBuf)\s+(?:fi_g|wp_g)\b', c_inst):
            issues.append("GLOBAL_BUFFER_REDECLARED: Do NOT declare `FiBuf fi_g` or `WpBuf wp_g`; the header already defines it.")

        # Check __global__ signature preservation
        orig_globals = re.findall(r'__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\(([^)]*)\)', c_orig)
        inst_globals = re.findall(r'__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\(([^)]*)\)', c_inst)
        if orig_globals and inst_globals and len(orig_globals) == len(inst_globals):
            for (n0, p0), (n1, p1) in zip(orig_globals, inst_globals):
                if n0 == n1 and re.sub(r'\s+', '', p0) != re.sub(r'\s+', '', p1):
                    issues.append(f"SIGNATURE_MUTATED: `__global__ void {n0}(...)` parameter list was modified; keep signatures 100% unchanged.")

        # Check no probes inside __device__ __forceinline__ micro-helpers
        for m_dev in re.finditer(r'__device__\s+(?:__forceinline__\s+)?\w[\w\s\*&<>]*\s+(\w+)\s*\([^)]*\)\s*\{', inst_text):
            start_b = m_dev.end() - 1
            depth = 0
            end_b = len(inst_text)
            for k_pos in range(start_b, len(inst_text)):
                if inst_text[k_pos] == "{":
                    depth += 1
                elif inst_text[k_pos] == "}":
                    depth -= 1
                    if depth == 0:
                        end_b = k_pos
                        break
            body_dev = inst_text[start_b:end_b]
            if len(body_dev.splitlines()) <= 20 and re.search(r'\b(?:WP|FI)_[BE]\b', body_dev):
                issues.append(
                    f"MICRO_HELPER_PROBE_VIOLATION: Probe placed inside short `__device__` helper `{m_dev.group(1)}()`. "
                    "Place probes at outer stage boundaries in `__global__` kernel instead."
                )

    # Check begin/end count balance
    b_cnt = len(re.findall(r'\b(?:WP_B|FI_B|WP_B_PRED|FI_B_PRED|wp_b|fi_b|wp_b_16b|fi_b_16b)\s*\(', inst_text))
    e_cnt = len(re.findall(r'\b(?:WP_E|FI_E|WP_E_PRED|FI_E_PRED|wp_e|fi_e|wp_e_16b|fi_e_16b)\s*\(', inst_text))
    if b_cnt == 0:
        issues.append("ZERO_BEGIN_PROBES: No `WP_B`/`FI_B`/`wp_b`/`fi_b` begin probes found.")
    elif b_cnt != e_cnt:
        issues.append(f"UNBALANCED_BEGIN_END_COUNT: Found {b_cnt} begin probes vs {e_cnt} end probes; every begin must have a matching end.")

    # Check spin-loop violations
    spin_lines = set(detect_spin_loop_lines(inst_text))
    if spin_lines:
        for idx_ln, ln_txt in enumerate(inst_text.splitlines(), start=1):
            if idx_ln in spin_lines and re.search(r'\b(?:WP_B|FI_B|WP_E|FI_E|wp_b|fi_b|wp_e|fi_e)\b', ln_txt):
                issues.append(f"SPIN_LOOP_PROBE_VIOLATION at line {idx_ln}: Probe placed inside polling spin loop; move outside loop.")
                break

    # Check CuTeDSL SCF scope rule
    if lang == "cutedsl":
        lines_py = inst_text.splitlines()
        open_indents = [
            len(ln) - len(ln.lstrip())
            for ln in lines_py
            if re.search(r'\bctx\s*=\s*(?:wp_open|fi_open)\s*\(', ln)
        ]
        use_indents = [
            len(ln) - len(ln.lstrip())
            for ln in lines_py
            if re.search(r'\bctx\s*=\s*(?:wp_[be]|fi_[be])\s*\(', ln)
        ]
        if open_indents and use_indents and min(open_indents) > min(use_indents):
            issues.append("CUTEDSL_SCF_SCOPE_VIOLATION: `ctx = wp_open(role)` must be initialized at outer function scope before `if`/`for`.")

    return issues


def build_direct_search_replace_prompt(
    source_text: str,
    route_kb: Mapping[str, Any],
    *,
    ablation_mode: str = "full_closed_loop",
    error_log: Optional[str] = None,
    diagnostics: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Build the direct source instrumentation prompt backed by the Two-Level Composable KB."""
    route_info = route_kb.get("route", {})
    lang = str(route_kb.get("language") or route_info.get("language", "cuda_cpp"))
    card = str(route_kb.get("card") or route_info.get("card", "h20"))
    arch = str(route_kb.get("arch") or route_info.get("arch", "sm_90a"))

    if ablation_mode == "bare_direct":
        sys_msg = (
            "You are an expert GPU developer. You must instrument the provided GPU kernel "
            "to profile intra-kernel stage durations (load, compute, store). "
            "Write your own timing variables, recording buffers, and instrumentation logic from scratch. "
            "Output your edits using compact SEARCH/REPLACE blocks."
        )
        usr_msg = f"Kernel Source:\n```\n{source_text}\n```"
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]

    kb_block = format_two_level_kb_context(source_text, route_kb, lang, card, ablation_mode)

    if ablation_mode == "kb_direct":
        probe_guide = """
### Direct Ad-Hoc Timer Mode (KB-Only Ablation — Without WeaveProfiler Sidecar Library)
- Apply the Two-Level Knowledge Base stage & boundary rules above (outer-stage granularity, spin-loop exclusion, async completion pairing),
  using standard native GPU clock intrinsics (`clock64()` / `s_memrealtime` / inline PTX) without `#include "wp_lite.cuh"` or `#include "wp_lite.cuh"`,
  and keep outer kernel parameter signatures intact so existing callers still compile.
"""
    elif lang in ("cuda_cpp", "hip_cpp"):
        recorder_hdr = "wp_lite.cuh" if lang == "cuda_cpp" else "wp_lite_hip.h"
        wp_hdr = "wp_lite.cuh" if lang == "cuda_cpp" else "wp_lite_hip.h"
        probe_guide = f"""
### WeaveProfiler Probe Primitives (`{wp_hdr}` / `{recorder_hdr}`)
- Include `#include "{recorder_hdr}"` (or `#include "{wp_hdr}"`) at the top of the file.
- Out-of-band Global Buffer Binding (Zero Call-Chain Signature Edits):
  Global device struct `wp_g` (`fi_g`) is already defined inside the header and bound via host `MemcpyToSymbol`.
  NEVER redeclare `FiBuf fi_g;` / `WpBuf wp_g;`, and NEVER add buffer parameters to `__global__` or `__device__` signatures!
- At kernel entry (strictly AFTER early bounds-check `return;`):
  `WP_KERNEL_SCOPE(role);`  // (or `FI_KERNEL_SCOPE(role);` — 1 argument: warp role integer enum, e.g. `0u` worker, `1u` DMA/producer)
- Around outer synchronous stages or coarsened outer loops (Scheme A):
  `WP_B(stage_id, aux);`    // (or `FI_B(stage_id, aux);` — stage_id: 1u=load, 2u=wait, 3u=mma, 4u=compute, 6u=store, 7u=barrier, 8u=dispatch, 9u=communication; aux: 0u)
  ... stage or enclosing outer loop ...
  `WP_E(stage_id, aux);`    // (or `FI_E(stage_id, aux);`)
- For decoupled asynchronous operations (when NOT inside a high-frequency inner loop):
  `FI_B(1u, 0u);` // or `WP_B(1u, 0u);` at launch
  tma_copy(...);
  ...
  mbarrier_wait(...);
  `FI_E(1u, 0u);` // or `WP_E(1u, 0u);` at completion
- For conditional/predicated probes:
  `WP_B_PRED(pred, stage_id, aux);` / `WP_E_PRED(pred, stage_id, aux);` (or `FI_B_PRED` / `FI_E_PRED`)
- IMPORTANT: NEVER place wait probes inside a polling `while (*flag == 0)` spin loop, and NEVER place unconditional probes inside hot `for (kt ...)` / `#pragma unroll` inner loops!
"""
    elif lang == "triton":
        if card == "mi308x":
            probe_guide = """
### WeaveProfiler Probe Primitives (AMD Triton wave64)
- Transparent launcher proxy `_WpTritonProxy` (`_FiTritonProxy`) binds sidecar buffers without changing Python caller signatures!
- At kernel entry:
  `slot, selected, n = fi_open(role)`  # (or `wp_open(role)`)
- Around outer canonical stages (strictly OUTSIDE pipelined `for k` loops):
  `n = fi_b_16b(time_ptr, tag_ptr, aux_ptr, ovf_ptr, slot, selected, n, stage_id, aux)`  # (or `wp_b_16b`)
  ... outer stage or loop ...
  `n = fi_e_16b(time_ptr, tag_ptr, aux_ptr, ovf_ptr, slot, selected, n, stage_id, aux)`  # (or `wp_e_16b`)
  (stage_id integer literal: 1=load, 2=wait, 3=mma, 4=compute, 6=store, 7=barrier; aux: 0)
"""
        else:
            probe_guide = """
### WeaveProfiler Probe Primitives (NVIDIA Triton)
- Transparent launcher proxy `_WpTritonProxy` (`_FiTritonProxy`) binds sidecar buffers without changing `@triton.jit` caller signatures!
- At kernel entry (after computing `pid`):
  `rec, ovf, pid, n, FI = fi_open(pid, 0)`  # (or `wp_open(pid, 0)`)
- Around outer canonical stages (strictly OUTSIDE hot `for k` / `for start_n` inner loops, reassigning `n` each time):
  `n = fi_b(rec, ovf, pid, n, FI, stage_id, aux)`  # (or `wp_b(...)` with integer literals e.g. `stage_id=3, aux=0`)
  ... outer stage or enclosing loop ...
  `n = fi_e(rec, ovf, pid, n, FI, stage_id, aux)`  # (or `wp_e(...)`)
  (stage_id integer literals ONLY: 1=load, 2=wait, 3=mma, 4=compute, 6=store, 7=barrier; aux=0)
"""
    else:  # cutedsl
        probe_guide = """
### WeaveProfiler Probe Primitives (CuTeDSL)
- Transparent `_WpCuteProxy` / `_FiCuteProxy` binds sidecar tensors without changing external caller signatures!
- At top-level `@cute.kernel` entry (MUST be initialized UNCONDITIONALLY before any `if` or `for`!):
  `ctx = fi_open(0)`  # (or `wp_open(0)`)
- Around outer canonical stages (wrap enclosing `for k_tile` loop from OUTSIDE, passing integer literals):
  `ctx = fi_b(ctx, stage_id, aux)`  # (or `wp_b(ctx, 3, 0)`)
  ... outer stage or enclosing loop ...
  `ctx = fi_e(ctx, stage_id, aux)`  # (or `wp_e(ctx, 3, 0)`)
  (stage_id integer literals ONLY: 1=load, 2=wait, 3=mma, 4=compute, 6=store, 7=barrier; aux=0)
"""

    rules_summary = """
### Canonical Stage Classes & Outer-Boundary Law (Scheme A)
1. `load` (1): Prologue / outer HBM->SMEM or Q-tile data transfer.
2. `wait` (2): Producer-consumer or cross-CTA synchronization wait (strictly OUTSIDE polling `while` spin loops).
3. `mma` (3) / `compute` (4): Enclosing mainloop (`for kt`, `for k`, `for kv_blk`, `for tile`) or vector/reduction pass.
4. `barrier` (7): Explicit CTA/cluster barrier outside hot inner loops.
5. `store` (6): Enclosing epilogue accumulator writeback block (strictly OUTSIDE `for mi / for ni / for elem` loops).
6. `dispatch` (8) / `communication` (9): Expert tile scheduling or outer multi-GPU P2P transfer loop.
"""

    error_section = ""
    if error_log:
        error_section = f"""
### CRITICAL: PREVIOUS VERIFICATION FAILED — FIX ALL ISSUES IN ONE BATCH
```
{error_log}
```
**Mandatory Batch Repair Actions**:
- If Overhead > 10% or instrumented spill bytes exceed baseline spill bytes: IMMEDIATELY hoist ALL `WP_B`/`WP_E`/`FI_B`/`FI_E`/`fi_b`/`fi_e` calls OUT of `for (kt...)`, `for (t...)`, `for k in range(...)`, and `#pragma unroll` loops so each outer phase executes ONCE per CTA/role!
- If Compile Error: check exact header/macro syntax, remove any `FiBuf fi_g;` redeclaration, keep `__global__` signatures untouched, and ensure `ctx = fi_open(0)` precedes all `if`/`for` in CuTeDSL.
- If Pairing/Decode Error: ensure every `B` has a matching `E` with identical `stage_id` on the same execution path, and place `WP_KERNEL_SCOPE` after early `return;`.
"""

    usr_prompt = f"""Target Card: {card} ({arch}) | Language: {lang}
{rules_summary}
{kb_block}
{probe_guide}
{error_section}

Target Kernel Source:
```
{source_text}
```

Format your edits ONLY as one or more compact SEARCH/REPLACE blocks:
<<<<<<< SEARCH
exact lines from target kernel source
=======
exact lines with WeaveProfiler probes inserted
>>>>>>>
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert GPU performance engineer specialized in WeaveProfiler kernel instrumentation. "
                "Guided by the Two-Level Composable Knowledge Base, insert lightweight probe macros directly into kernel source. "
                "Strictly obey Outer-Stage Granularity (Scheme A): hoist probes outside hot inner loops (>8 iters), "
                "keep wait probes outside spin loops, never probe micro `__device__` helpers, and edit ONLY the kernel body "
                "via compact SEARCH/REPLACE blocks without modifying function signatures."
            ),
        },
        {"role": "user", "content": usr_prompt},
    ]
    return messages


def audit_incremental_with_whole_kernel_llm(
    current_source: str,
    prev_source: str,
    prev_inst_source: str,
    route_kb: Mapping[str, Any],
    *,
    model: str = "qwen4-max",
    reasoning_effort: str = "max",
    llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """2-Stage Diff-Preprojected Whole-Kernel Semantic Audit (§3.2).

    Why blind syntactic diff inheritance or hunk-only LLM fails:
      Editing even 1 line (e.g., changing `warp == 0` -> `warp < 2`, changing `mbarrier` stage count,
      fusing activation into epilogue, or changing loop unroll/trip count) can alter the semantics,
      role assignment, or overhead of UNTOUCHED lines hundreds of lines away in partner warps!

    How this solves it cleanly and fast:
      1. If `current_source == prev_source` (external shape/template define sweep), reuse `prev_inst_source` in 0s.
      2. Otherwise, Stage 1 (`incremental_inherit_probes`) mechanically projects `prev_inst_source` onto
         `current_source` in <5ms as a candidate draft (`projected_draft`) and computes unified `git diff`.
      3. Stage 2 invokes the LLM on the **FULL `current_source` + `unified_diff` + `projected_draft`**,
         so the LLM inspects the ENTIRE kernel (all warps, barriers, loops) to verify global semantic consistency:
         - If all projected probes remain semantically valid across the whole kernel, the LLM outputs
           `KEEP_PROJECTED` (~1 token, fast single turn).
         - If local OR distant partner-warp probes need updating due to direct or indirect semantic shifts,
           the LLM emits compact `SEARCH/REPLACE` blocks against `projected_draft` (or `current_source`).
    """
    t0 = time.perf_counter()
    if current_source == prev_source:
        return {
            "ok": True,
            "is_incremental": True,
            "strategy": "exact_source_match_zero_llm",
            "used_llm_audit": False,
            "semantic_shift_detected": False,
            "instrumented_source": prev_inst_source,
            "patch": "/* Exact source match */",
            "telemetry": {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "elapsed_s": round(time.perf_counter() - t0, 5),
                "llm_elapsed_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    caller = llm_caller or call_llm
    inc_draft = incremental_inherit_probes(current_source, prev_source, prev_inst_source)
    projected_inst = inc_draft[0] if inc_draft is not None else None

    diff_lines = list(
        difflib.unified_diff(
            prev_source.splitlines(),
            current_source.splitlines(),
            fromfile="prev_kernel",
            tofile="edited_kernel",
            lineterm="",
            n=4,
        )
    )
    unified_diff_text = "\n".join(diff_lines[:240])

    route_info = route_kb.get("route", {})
    lang = str(route_kb.get("language") or route_info.get("language", "cuda_cpp"))
    card = str(route_kb.get("card") or route_info.get("card", "h20"))
    arch = str(route_kb.get("arch") or route_info.get("arch", "sm_90a"))
    kb_block = format_two_level_kb_context(current_source, route_kb, lang, card, "full_closed_loop")

    base_for_patch = projected_inst if projected_inst is not None else current_source
    draft_section = (
        f"### Mechanical Diff-Projected Candidate (`projected_draft` — ALL lines visible)\n```\n{projected_inst}\n```"
        if projected_inst is not None
        else f"### Previous Verified Instrumented Kernel (`prev_inst` — diff overlapped probe anchors)\n```\n{prev_inst_source}\n```\n\n### New Uninstrumented Kernel (`edited_kernel` — apply SEARCH/REPLACE to this)\n```\n{current_source}\n```"
    )

    sys_msg = (
        "You are WeaveProfiler's Whole-Kernel Incremental Semantic Auditor (§3.2).\n"
        "A GPU kernel was edited during optimization. Mechanical diff projection cannot tell on its own whether "
        "a local edit indirectly altered warp-role predicates, cross-warp barrier handoffs, fused stages, or loop trip counts "
        "elsewhere in the kernel.\n"
        "Inspect the FULL kernel alongside the `git diff`:\n"
        "1. ONLY if `projected_draft` is provided AND every single probe across the ENTIRE kernel (both modified regions "
        "and untouched partner warps/epilogues) remains 100% semantically accurate, properly paired, role-aligned, "
        "and outside hot inner/spin loops, reply with the exact single token: KEEP_PROJECTED\n"
        "2. If ANY local OR distant untouched probe's role, stage meaning, barrier pairing, or loop-overhead safety changed "
        "(or if `projected_draft` is absent / missing a newly added stage), output ONLY compact `SEARCH/REPLACE` blocks "
        f"against `{'projected_draft' if projected_inst is not None else 'edited_kernel'}` to make the probes 100% accurate!"
    )

    usr_msg = f"""Target Card: {card} ({arch}) | Language: {lang}

{kb_block}

### Unified Git Diff (`prev_kernel` -> `edited_kernel`)
```diff
{unified_diff_text}
```

{draft_section}
"""

    t_llm0 = time.perf_counter()
    llm_res = caller(
        model,
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
        reasoning_effort=reasoning_effort,
    )
    t_llm = time.perf_counter() - t_llm0
    raw_reply = (llm_res.get("content") or "").strip()

    if projected_inst is not None and "KEEP_PROJECTED" in raw_reply and "<<<<<<< SEARCH" not in raw_reply:
        final_inst = projected_inst
        strategy = "whole_kernel_audit_keep_projected"
        semantic_shift = False
    else:
        patched = apply_search_replace_blocks(base_for_patch, raw_reply)
        if patched != base_for_patch:
            final_inst = patched
            strategy = "whole_kernel_audit_semantic_patch"
            semantic_shift = True
        elif projected_inst is not None:
            final_inst = projected_inst
            strategy = "whole_kernel_audit_fallback_projected"
            semantic_shift = False
        else:
            final_inst = patched
            strategy = "whole_kernel_audit_full_regen"
            semantic_shift = True

    total_elapsed = time.perf_counter() - t0
    return {
        "ok": final_inst != current_source,
        "is_incremental": True,
        "strategy": strategy,
        "used_llm_audit": True,
        "semantic_shift_detected": semantic_shift,
        "instrumented_source": final_inst,
        "patch": raw_reply,
        "telemetry": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "elapsed_s": round(total_elapsed, 4),
            "llm_elapsed_s": round(t_llm, 4),
            **llm_res.get("usage", {}),
        },
    }


def direct_instrument_kernel(
    source_path: Path,
    *,
    card: str = "h20",
    arch: Optional[str] = None,
    language: Optional[str] = None,
    model: str = "qwen4-max",
    reasoning_effort: str = "max",
    ablation_mode: str = "full_closed_loop",
    error_log: Optional[str] = None,
    prev_source_path: Optional[Path] = None,
    prev_inst_path: Optional[Path] = None,
    audit_incremental: bool = True,
    llm_caller: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute direct source kernel instrumentation via Two-Level KB + Static Guard or Audited Incremental Reuse."""
    source_text = source_path.read_text(encoding="utf-8")
    det_lang = detect_language(source_path, source_text, explicit=language)
    route = resolve_route(card, arch, det_lang)
    caller = llm_caller or call_llm
    kb_ctx = retrieve_route_kb(route["language"], route["card"], route["arch"], source_text)

    # 1. Incremental path: Diff-Preprojected Whole-Kernel Semantic Audit (default) or fast mechanical projection
    if prev_source_path and prev_inst_path and prev_source_path.is_file() and prev_inst_path.is_file():
        prev_src = prev_source_path.read_text(encoding="utf-8")
        prev_inst = prev_inst_path.read_text(encoding="utf-8")
        if audit_incremental:
            return audit_incremental_with_whole_kernel_llm(
                source_text,
                prev_src,
                prev_inst,
                kb_ctx,
                model=model,
                reasoning_effort=reasoning_effort,
                llm_caller=caller,
            )
        t_inc0 = time.perf_counter()
        inc_res = incremental_inherit_probes(source_text, prev_src, prev_inst)
        if inc_res is not None:
            inst_code, strategy = inc_res
            t_inc = time.perf_counter() - t_inc0
            return {
                "ok": True,
                "is_incremental": True,
                "strategy": strategy,
                "instrumented_source": inst_code,
                "patch": "/* Mechanical diff probe projection */",
                "telemetry": {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "ablation_mode": ablation_mode,
                    "elapsed_s": round(t_inc, 4),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

    # 2. Build direct SEARCH/REPLACE prompt with full Two-Level Composable KB
    messages = build_direct_search_replace_prompt(
        source_text,
        kb_ctx,
        ablation_mode=ablation_mode,
        error_log=error_log,
    )

    t0 = time.perf_counter()
    llm_res = caller(model, messages, reasoning_effort=reasoning_effort)
    elapsed_s = time.perf_counter() - t0

    raw_content = llm_res.get("content", "")
    inst_source = apply_search_replace_blocks(source_text, raw_content)

    # 3. Static CPU preflight self-repair in full_closed_loop before spending GPU compile/benchmark time
    static_issues: List[str] = []
    if ablation_mode == "full_closed_loop":
        static_issues = static_preflight_validate(source_text, inst_source, route["language"], route["card"])
        if static_issues:
            repair_log = "[Static Preflight Check FAILED]\n" + "\n".join(f"- {x}" for x in static_issues)
            if error_log:
                repair_log = error_log + "\n\n" + repair_log
            base_src = inst_source if inst_source != source_text else source_text
            repair_msgs = build_direct_search_replace_prompt(
                base_src,
                kb_ctx,
                ablation_mode="full_closed_loop",
                error_log=repair_log,
            )
            t1 = time.perf_counter()
            llm_res2 = caller(model, repair_msgs, reasoning_effort=reasoning_effort)
            elapsed_s += (time.perf_counter() - t1)
            repaired = apply_search_replace_blocks(base_src, llm_res2.get("content", ""))
            if repaired != base_src:
                inst_source = repaired
                raw_content = raw_content + "\n/* STATIC_REPAIR */\n" + llm_res2.get("content", "")
                static_issues = static_preflight_validate(source_text, inst_source, route["language"], route["card"])

    telemetry = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "ablation_mode": ablation_mode,
        "elapsed_s": round(elapsed_s, 4),
        "static_preflight_issues": static_issues,
        **llm_res.get("usage", {}),
    }

    return {
        "ok": inst_source != source_text,
        "is_incremental": False,
        "instrumented_source": inst_source,
        "patch": raw_content,
        "telemetry": telemetry,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WeaveProfiler Direct Kernel Source Instrumentation Agent (DAC'27 §3.1)",
    )
    parser.add_argument("--source", "-s", required=True, help="Path to uninstrumented kernel source file")
    parser.add_argument("--out-source", "-o", required=True, help="Path to write instrumented kernel source file")
    parser.add_argument("--card", default="h20", choices=["h20", "b300", "mi308x"], help="Target GPU card")
    parser.add_argument("--arch", help="Target architecture (sm_90a, sm_100a, sm_103a, gfx942)")
    parser.add_argument("--language", help="Optional explicit language (cuda_cpp, hip_cpp, triton, cutedsl)")
    parser.add_argument("--model", default="qwen4-max", choices=SUPPORTED_MODELS, help="Evaluated LLM")
    parser.add_argument("--reasoning-effort", default="max", help="Reasoning effort (default: max)")
    parser.add_argument("--ablation-mode", default="full_closed_loop", choices=ABLATION_MODES, help="Ablation mode")
    parser.add_argument("--error-log", help="Path to plain-text terminal error log from failed verify.py run")
    parser.add_argument("--prev-source", help="Previous uninstrumented kernel for incremental reuse")
    parser.add_argument("--prev-inst", help="Previous verified instrumented kernel for incremental reuse")
    parser.add_argument("--no-audit-incremental", action="store_true", help="Skip whole-kernel LLM semantic audit on incremental diffs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_p = Path(args.source)
    if not source_p.is_file():
        print(f"[direct_probe_agent.py] Error: Source file not found: {source_p}", file=sys.stderr)
        return 1

    err_text = None
    if args.error_log:
        err_p = Path(args.error_log)
        if err_p.is_file():
            err_text = err_p.read_text(encoding="utf-8")

    prev_src_p = Path(args.prev_source) if args.prev_source else None
    prev_inst_p = Path(args.prev_inst) if args.prev_inst else None

    res = direct_instrument_kernel(
        source_p,
        card=args.card,
        arch=args.arch,
        language=args.language,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        ablation_mode=args.ablation_mode,
        error_log=err_text,
        prev_source_path=prev_src_p,
        prev_inst_path=prev_inst_p,
        audit_incremental=not args.no_audit_incremental,
    )

    out_p = Path(args.out_source)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(res["instrumented_source"].replace("\r\n", "\n"), encoding="utf-8")

    print(json.dumps({
        "ok": res["ok"],
        "is_incremental": res["is_incremental"],
        "strategy": res.get("strategy"),
        "out_source": str(out_p),
        "telemetry": res["telemetry"],
    }, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
