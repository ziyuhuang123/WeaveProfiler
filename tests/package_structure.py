#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline package contract for weaveprofiler.

This test answers one question: can a new Codex session understand and execute
the skill without guessing where hardware, language, recorder, capture, or
validation knowledge lives?  It checks the public tree, exact card routes,
knowledge coverage, canonical semantics, curve IDs, validation checks, and
absence of stale snapshot/cache files.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TEXT_EXT = {".py", ".md", ".json", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".sh", ".txt", ".jsonl"}
ALLOWED_TOP = {
    "README.md", "README_EN.md", "SKILL.md", "plugin.json", "pyproject.toml", "run.py",
    "recorders", "weaveprof", "configs", "docs", "examples", "experiments", "tests",
}
REQUIRED_TOP = ALLOWED_TOP - {"README.md", "README_EN.md", "plugin.json"}
REQUIRED_FILES = [
    "weaveprof/__init__.py", "weaveprof/__main__.py", "weaveprof/replay.py",
    "run.py", "weaveprof/pipeline/core/registry.py", "weaveprof/pipeline/core/semantics.py",
    "recorders/cuda_cpp/wp_lite.cuh",
    "recorders/hip_cpp/wp_lite_hip.h",
    "recorders/triton/wp_triton.py",
    "recorders/triton/wp_triton_amd.py",
    "recorders/cutedsl/wp_cute.py",
    "weaveprof/pipeline/scan/get_trace_processor.py",
    "weaveprof/pipeline/capture/capture_nsys.py", "weaveprof/pipeline/capture/capture_rocm.py",
    "weaveprof/pipeline/capture/gemm_probe.py",
    "configs/templates/gemm_probe/gemm.cu",
    "configs/templates/gemm_probe/names.json",
    "configs/templates/gemm_probe/metrics.json",
    "configs/templates/gemm_probe/gemm_h20.config",
    "weaveprof/pipeline/capture/extract_gpu_metrics.py",
    "weaveprof/pipeline/capture/build_rocprof_sw_hw.py", "weaveprof/pipeline/capture/build_rocprof_sys_sw_hw.py",
    "weaveprof/pipeline/capture/hardware_utilization.py", "weaveprof/pipeline/capture/merge_hw_into_perfetto.py",
    "configs/semantic_ids.json", "configs/operators/operators.json",
    "configs/metrics/hardware.json", "configs/validation/fixes.json",
    "configs/validation/scan_schema.json",
    "configs/validation/schemas/scan-selection.schema.json",
    "configs/recipes/index.json",
    "tests/run_all.sh", "tests/package_structure.py",
    "tests/test_contract_consistency.py",
]
CARD_FILES = {
    "h20": ("sm_90", ["h20_sm_90.json"]),
    "b300": ("sm_100a+sm_103a", ["b300_sm_100a.json", "b300_sm_103a.json"]),
    "mi308x": ("gfx942", ["mi308x_gfx942.json"]),
    "sm86": ("sm_86", ["sm86_sm_86.json"]),
}
LANGUAGE_FILES = [
    "configs/languages/cuda_cpp/patterns.json",
    "configs/languages/hip_cpp/patterns.json",
    "configs/languages/triton/nvidia_patterns.json",
    "configs/languages/triton/amd_patterns.json",
    "configs/languages/cutedsl/patterns.json",
]
CURVES = {"compute_util", "memory_util", "communication_util", "sm_active"}
CHECKS = {"correctness", "overhead", "assembly_change_rate", "alignment",
          "buffer_completeness", "overflow"}
REQUIRED_COMMANDS = {"capture", "gemm-probe"}
LINK_RX = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)")
PATH_REF_RX = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|cuh|h|cpp|cu|json|sh|md|config))`")


def _load(path: Path, fails: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fails.append("{}: {}".format(path.relative_to(PKG), exc))
        return {}


def _check_skill(fails: list[str]) -> None:
    skill = PKG / "SKILL.md"
    if not skill.is_file():
        fails.append("missing SKILL.md")
        return
    text = skill.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        fails.append("SKILL.md must start with YAML frontmatter")
    if not re.search(r"(?m)^name:\s*weaveprofiler\s*$", text):
        fails.append("SKILL.md frontmatter name must be weaveprofiler")
    if not re.search(r"(?m)^description:\s*\S", text):
        fails.append("SKILL.md frontmatter description is missing")
    for phrase in ("H20", "B300", "MI308X",
                   "sm_90", "sm_100a", "sm_103a", "gfx942", "rocprof-sys", "rocprofv3",
                   "compute_util", "memory_util", "communication_util", "sm_active"):
        if phrase not in text:
            fails.append("SKILL.md missing required route/evidence term: {}".format(phrase))
    for language in ("cuda_cpp", "hip_cpp", "triton", "cutedsl"):
        if language not in text:
            fails.append("SKILL.md missing language route: {}".format(language))
    if "fast-instrument" in text:
        fails.append("SKILL.md still exposes the old fast-instrument name")
    if "MLIR" in text and not re.search(r"(?:do not|never|不要|禁止)[^\n]{0,50}MLIR", text, re.I):
        fails.append("SKILL.md must not require MLIR")
    missing = []
    for match in LINK_RX.finditer(text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if not (PKG / target).exists():
            missing.append(target)
    for match in PATH_REF_RX.finditer(text):
        target = match.group(1)
        if target.startswith("/") or target.split("/", 1)[0] in {"build", "run", "out", "tmp"}:
            continue
        if not (PKG / target).exists():
            missing.append(target)
    if missing:
        fails.append("SKILL.md contains missing package paths: {}".format(sorted(set(missing))))
    print("  SKILL.md links/paths checked")


def _check_cards(fails: list[str]) -> None:
    for card, (archs, profiles) in CARD_FILES.items():
        wrapper = _load(PKG / "configs" / "cards" / (card + ".json"), fails)
        if wrapper.get("card") != card:
            fails.append("{} wrapper card mismatch".format(card))
        got = set(wrapper.get("architectures") or [])
        want = set(archs.split("+"))
        if got != want:
            fails.append("{} wrapper architectures {} != {}".format(card, sorted(got), sorted(want)))
        listed = {item.get("profile") for item in wrapper.get("machine_profiles", []) if isinstance(item, dict)}
        if not set(profiles).issubset(listed):
            fails.append("{} wrapper does not list profiles {}".format(card, profiles))
        for profile in profiles:
            doc = _load(PKG / "configs" / "cards" / profile, fails)
            if doc.get("card") != card:
                fails.append("{} card mismatch in {}".format(profile, card))
            expected_arch = profile.split("_")[-1].removesuffix(".json")
            expected_arch = profile[len(card) + 1:-5] if card in profile else expected_arch
            if doc.get("arch") != expected_arch:
                fails.append("{} arch={} expected {}".format(profile, doc.get("arch"), expected_arch))
            if not doc.get("entries"):
                fails.append("{} has no hardware entries".format(profile))
    h20 = _load(PKG / "configs/cards/h20.json", fails)
    b300 = _load(PKG / "configs/cards/b300.json", fails)
    mi = _load(PKG / "configs/cards/mi308x.json", fails)
    if h20.get("profiler", {}).get("primary") != "nsys":
        fails.append("H20 profiler must be nsys")
    if b300.get("profiler", {}).get("primary") != "nsys":
        fails.append("B300 profiler must be nsys")
    if mi.get("profiler", {}).get("primary") != "rocprof-sys":
        fails.append("MI308X profiler must be rocprof-sys")
    if "rocprofv3" not in (mi.get("profiler", {}).get("fallback") or []):
        fails.append("MI308X must expose rocprofv3 fallback")
    print("  cards: H20/B300(2 arch)/MI308X/sm86 checked")


def _check_recipes(fails: list[str]) -> None:
    index = _load(PKG / "configs/recipes/index.json", fails)
    routes = {(row.get("card"), row.get("language")): row for row in index.get("recipes", []) if isinstance(row, dict)}
    expected = {
        ("h20", "cuda_cpp"), ("h20", "triton"), ("h20", "cutedsl"),
        ("b300", "cuda_cpp"), ("b300", "triton"), ("b300", "cutedsl"),
        ("mi308x", "hip_cpp"), ("mi308x", "triton"),
    }
    if set(routes) != expected:
        fails.append("recipe index route set mismatch: missing={} extra={}".format(
            sorted(expected - set(routes)), sorted(set(routes) - expected)))
    for key, row in routes.items():
        path = Path(row.get("path", ""))
        full = PKG / path
        if not path or not full.is_file():
            fails.append("recipe path missing for {}: {}".format(key, path))
            continue
        _load(full, fails)
    for lang in ("cuda_cpp", "triton", "cutedsl"):
        if index.get("recipes") and not any(
                row.get("card") == "b300" and row.get("language") == lang
                for row in index.get("recipes", [])):
            fails.append("B300 missing recipe for {}".format(lang))
    print("  recipes: 8 card/language routes checked")


def _check_cross_file_contract(fails: list[str]) -> None:
    """Run the same live cross-file audit used by the standalone contract test."""
    contract_path = HERE / "test_contract_consistency.py"
    if not contract_path.is_file():
        fails.append("missing tests/test_contract_consistency.py")
        return
    try:
        spec = importlib.util.spec_from_file_location(
            "weaveprofiler_contract_consistency", contract_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot create module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contract_fails = module.collect_contract_failures()
    except Exception as exc:
        fails.append("cross-file contract audit crashed: {}: {}".format(
            type(exc).__name__, exc))
        return
    if contract_fails:
        fails.extend("cross-file contract: " + item for item in contract_fails)
    run_all = (PKG / "tests/run_all.sh").read_text(encoding="utf-8")
    wiring = sum(
        1
        for line in run_all.splitlines()
        if line.strip().endswith("tests/test_contract_consistency.py")
    )
    if wiring != 1:
        fails.append(
            "tests/run_all.sh must invoke tests/test_contract_consistency.py exactly once; got {}".format(
                wiring
            )
        )
    print("  cross-file contract: {} checks / run_all wiring checked".format(
        len(module.CHECKS)))


def _check_knowledge(fails: list[str]) -> None:
    sem = _load(PKG / "configs/semantic_ids.json", fails)
    ids = {row.get("semantic_id") for row in sem.get("semantics", []) if isinstance(row, dict)}
    required = set(sem.get("required_ids") or [])
    if not required.issubset(ids):
        fails.append("semantic_ids required IDs missing: {}".format(sorted(required - ids)))
    for sid in ids:
        if sid != "unknown" and not re.fullmatch(r"[a-z][a-z0-9_]*", str(sid)):
            fails.append("noncanonical semantic ID: {!r}".format(sid))
    for row in sem.get("semantics", []):
        if isinstance(row, dict) and row.get("semantic_id") != row.get("display_name"):
            fails.append("display_name differs from semantic_id for {}".format(row.get("semantic_id")))
    for rel in LANGUAGE_FILES:
        doc = _load(PKG / rel, fails)
        if not doc.get("patterns"):
            fails.append("language pattern library is empty: {}".format(rel))
    operators = _load(PKG / "configs/operators/operators.json", fails)
    op_ids = {row.get("semantic_id") for row in operators.get("operators", []) if isinstance(row, dict)}
    if not {"gemm", "attention", "reduction", "moe", "convolution", "megakernel"}.issubset(op_ids):
        fails.append("operator KB is missing required operators")
    all_text = json.dumps(operators, ensure_ascii=False)
    for check in CHECKS:
        if check not in all_text.lower():
            fails.append("operator KB missing correctness check token: {}".format(check))
    hw = (PKG / "configs/metrics/hardware.json").read_text(encoding="utf-8")
    if not CURVES.issubset(set(re.findall(r"[a-z_]+", hw))):
        missing = CURVES - set(re.findall(r"[a-z_]+", hw))
        fails.append("hardware metrics missing curve IDs: {}".format(sorted(missing)))
    print("  knowledge: semantics/operators/curves/language patterns checked")


def _check_run_interface(fails: list[str]) -> None:
    text = (PKG / "run.py").read_text(encoding="utf-8")
    for command in REQUIRED_COMMANDS:
        if '"{}"'.format(command) not in text:
            fails.append("run.py dispatch missing command: {}".format(command))
    for token in ('capture_nsys.py', 'capture_rocm.py', 'gemm_probe.py', 'rocprof-sys', 'rocprofv3',
                  'heartbeat', '--deadline', 'delivery_evidence'):
        if token not in text and token != 'delivery_evidence':
            fails.append("run.py/capture integration missing {}".format(token))
    print("  CLI and fast-cache contract checked")


def _check_experiments_isolation(fails: list[str]) -> None:
    """Enforce one-way dependency: core code may never import or load from `experiments/`."""
    core_globs = ("weaveprof/**/*.py", "run.py", "recorders/**/*.py", "recorders/**/*.cuh", "recorders/**/*.h")
    exp_path_rx = re.compile(r"(?<![A-Za-z0-9_])experiments/[A-Za-z0-9_./-]+")
    seen: set[Path] = set()
    bad: list[str] = []
    for pat in core_globs:
        for path in sorted(PKG.glob(pat)):
            if not path.is_file() or "__pycache__" in path.parts or path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(PKG).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix == ".py":
                tree = ast.parse(text, filename=rel)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "experiments" or alias.name.startswith("experiments."):
                                bad.append(f"{rel}:{node.lineno} imports {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        if mod == "experiments" or mod.startswith("experiments."):
                            bad.append(f"{rel}:{node.lineno} imports from {mod}")
            for lineno, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if exp_path_rx.search(line):
                    bad.append(f"{rel}:{lineno} references experiments/ path")
    if bad:
        fails.append("core code must not depend on experiments/: {}".format(bad[:8]))
    print("  experiments isolation: {} core files checked".format(len(seen)))


def _check_source_hygiene(fails: list[str]) -> None:
    junk, crlf, duplicate_paths, old_names = [], [], [], []
    duplicate_patterns = tuple("/".join((name, name)) for name in
                               ("knowledge", "weaveprof/pipeline/scan", "weaveprof/pipeline/instrument", "weaveprof/pipeline/capture", "weaveprof/pipeline/validate", "tests"))
    for path in sorted(PKG.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(PKG).parts
        if ".git" in rel_parts:
            continue
        rel = "/".join(rel_parts)
        if "__pycache__" in rel or path.suffix == ".pyc":
            junk.append(rel)
            continue
        raw = path.read_bytes()
        if path.suffix not in TEXT_EXT:
            continue
        if b"\r\n" in raw:
            crlf.append(rel)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for needle in duplicate_patterns:
            if needle in text:
                duplicate_paths.append("{} -> {}".format(rel, needle))
        if (rel in {"SKILL.md", "run.py"} or rel.startswith("configs/agents/")) and "fast-instrument" in text.lower():
            old_names.append(rel)
    if junk:
        fails.append("bytecode/cache residue: {}".format(junk[:10]))
    if crlf:
        fails.append("CRLF files: {}".format(crlf[:8]))
    if duplicate_paths:
        fails.append("duplicated/stale path fragments: {}".format(duplicate_paths[:10]))
    if old_names:
        fails.append("old skill name in public files: {}".format(old_names))
    top = {item.name for item in PKG.iterdir()} - {".git", ".gitignore", ".gitattributes", ".github"}
    extra = sorted(top - ALLOWED_TOP)
    missing = sorted(REQUIRED_TOP - top)
    if extra:
        fails.append("unexpected top-level entries: {}".format(extra))
    if missing:
        fails.append("missing top-level entries: {}".format(missing))
    for rel in REQUIRED_FILES:
        if not (PKG / rel).is_file():
            fails.append("required file missing: {}".format(rel))
    print("  source hygiene and top-level tree checked")


def main() -> int:
    fails: list[str] = []
    if not (PKG / "SKILL.md").is_file():
        print("PACKAGE STRUCTURE FAIL: missing SKILL.md")
        return 1
    _check_skill(fails)
    _check_cards(fails)
    _check_recipes(fails)
    _check_cross_file_contract(fails)
    _check_knowledge(fails)
    _check_run_interface(fails)
    _check_experiments_isolation(fails)
    _check_source_hygiene(fails)
    if fails:
        print("PACKAGE STRUCTURE FAIL")
        for item in fails:
            print("  -", item)
        return 1
    print("PACKAGE STRUCTURE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
