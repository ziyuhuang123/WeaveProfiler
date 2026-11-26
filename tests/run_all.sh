#!/usr/bin/env bash
# weaveprofiler offline regression suite.
#
# Defaults:
#   PYTHONDONTWRITEBYTECODE=1 and Python -B for every direct test process.
#   No GPU, profiler, or network requirement.
#
# Compatibility:
#   bash tests/run_all.sh --python /path/to/python3.10
#   bash tests/run_all.sh -p /path/to/python3.10
#   PYTHON=/path/to/python3.10 bash tests/run_all.sh
#
# Cache contract:
#   A snapshot of every package __pycache__ directory and file is taken before
#   the first test.  Any directory or file added during the run is a suite
#   failure, even when every test process otherwise exits zero.
#
# Notes:
#   The tests require Python >= 3.10 because several offline contracts use
#   PEP 604 type syntax (for example ``Path | None``).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
PKG="$PWD"

export PYTHONDONTWRITEBYTECODE=1

KEEP=""
ONLY=""
PY_OPT="${PYTHON:-}"
WANT_NUMPY=1

usage() {
  cat <<'USAGE'
用法: bash tests/run_all.sh [选项]

选项:
  --python PATH, -p PATH  指定 Python 解释器（必须 >= 3.10）
  --keep-dir DIR          保留日志和临时产物到 DIR
  --only KEYWORD          只运行标签中包含 KEYWORD 的测试
  --skip-numpy            跳过需要 numpy 的可选比较测试
  -h, --help              显示本帮助
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --python|-p)
      [ $# -ge 2 ] || { echo "错误：$1 缺少解释器路径" >&2; exit 2; }
      PY_OPT="$2"
      shift 2
      ;;
    --keep-dir)
      [ $# -ge 2 ] || { echo "错误：--keep-dir 缺少目录" >&2; exit 2; }
      KEEP="$2"
      shift 2
      ;;
    --only)
      [ $# -ge 2 ] || { echo "错误：--only 缺少关键字" >&2; exit 2; }
      ONLY="$2"
      shift 2
      ;;
    --skip-numpy)
      WANT_NUMPY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Establish the run directory and cache snapshot before probing or running any
# Python process, so interpreter discovery cannot slip residue past the audit.
if [ -n "$KEEP" ]; then
  TMP="$KEEP"
  mkdir -p "$TMP"
else
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/weaveprofiler_offline.XXXXXX")" || exit 2
fi
LOGS="$TMP/logs"
mkdir -p "$LOGS"

# Snapshot both cache directories and files.  A new empty directory is residue too.
snapshot_cache() {
  {
    find "$PKG" -type d -name __pycache__ -printf 'D %P\n'
    find "$PKG" -type f -path '*/__pycache__/*' -printf 'F %P\n'
  } | LC_ALL=C sort
}
snapshot_cache >"$TMP/cache_before.txt"
CACHE_BASELINE_COUNT="$(wc -l <"$TMP/cache_before.txt" | tr -d ' ')"
SUITE_T0=$(date +%s%3N)

py_ver_ok() {
  ( cd "$TMP" && "$1" -I -B -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' ) \
    >/dev/null 2>&1
}

has_mod() {
  ( cd "$TMP" && "$1" -I -B -c "import $2" ) >/dev/null 2>&1
}

is_runnable() {
  command -v "$1" >/dev/null 2>&1 || [ -x "$1" ]
}

cands=()
if [ -n "$PY_OPT" ]; then
  if ! is_runnable "$PY_OPT"; then
    echo "指定的解释器不存在或不可执行：$PY_OPT" >&2
    exit 2
  fi
  cands=("$PY_OPT")
else
  for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 \
                   python3 /opt/conda/envs/*/bin/python \
                   /usr/local/python/*/bin/python3 /usr/local/bin/python3; do
    [ -n "$candidate" ] || continue
    is_runnable "$candidate" || continue
    cands+=("$candidate")
  done
fi

PY=""
for candidate in "${cands[@]}"; do
  if py_ver_ok "$candidate"; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "找不到 Python >= 3.10 的解释器。" >&2
  echo "用 --python /path/to/python3.10 或 PYTHON=/path/to/python3.10 指定。" >&2
  exit 2
fi

if [ -n "$PY_OPT" ] && ! py_ver_ok "$PY_OPT"; then
  echo "指定解释器 ${PY_OPT} 不满足 Python >= 3.10：" >&2
  "$PY_OPT" -V >&2 || true
  exit 2
fi

PY_NUM="$PY"
if [ "$WANT_NUMPY" -eq 0 ]; then
  PY_NUM=""
elif ! has_mod "$PY" numpy; then
  PY_NUM=""
  for candidate in "${cands[@]}"; do
    if py_ver_ok "$candidate" && has_mod "$candidate" numpy; then
      PY_NUM="$candidate"
      break
    fi
  done
fi

# Some tests launch ``sys.executable -I``.  Isolated mode ignores
# PYTHONDONTWRITEBYTECODE, so a plain environment guard is insufficient.  Route
# every Python subprocess through a tiny launcher that always supplies -B and
# preserves ``sys.executable`` as the launcher path.  This keeps the cache audit
# strict without weakening the tests themselves.
resolve_python_path() {
  local selected path dir
  selected="$1"
  if [ -x "$selected" ]; then
    path="$selected"
  else
    path="$(command -v "$selected" 2>/dev/null || true)"
  fi
  [ -n "$path" ] || path="$selected"
  # Preserve a virtualenv's wrapper path.  Resolving its symlink with
  # readlink -f lands on the base interpreter and loses site-packages
  # (notably numpy), making the launcher silently weaker than the selected
  # interpreter.
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    *)
      dir="$(cd "$(dirname "$path")" && pwd)" || return 1
      printf '%s/%s\n' "$dir" "$(basename "$path")"
      ;;
  esac
}

make_python_launcher() {
  local root real cfg pyver site launcher
  root="$1"
  real="$2"
  launcher="$root/bin/python-launcher"
  mkdir -p "$root/bin"
  cfg="$(dirname "$(dirname "$real")")/pyvenv.cfg"
  if [ -f "$cfg" ]; then
    # A shell wrapper changes argv[0], so CPython cannot discover the venv by
    # itself.  Mirror the tiny venv layout (cfg + site-packages symlink) so
    # sys.executable stays the wrapper while the wrapper still sees the
    # selected environment's packages.
    pyver="$("$real" -B -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
    site="$("$real" -B -c 'import site; print(site.getsitepackages()[0])')"
    cp "$cfg" "$root/pyvenv.cfg"
    mkdir -p "$root/lib/$pyver"
    ln -sfn "$site" "$root/lib/$pyver/site-packages"
  fi
  {
    printf '#!/bin/bash\n'
    printf 'exec -a "$0" %q -B "$@"\n' "$real"
  } >"$launcher"
  chmod +x "$launcher"
  printf '%s\n' "$launcher"
}

PY_REAL="$(resolve_python_path "$PY")"
PY_LAUNCH="$(make_python_launcher "$TMP/python" "$PY_REAL")"

PY_NUM_REAL=""
PY_NUM_LAUNCH=""
if [ -n "$PY_NUM" ]; then
  PY_NUM_REAL="$(resolve_python_path "$PY_NUM")"
  PY_NUM_LAUNCH="$(make_python_launcher "$TMP/python-numpy" "$PY_NUM_REAL")"
fi

# Also cover child commands that invoke ``python``/``python3`` by name instead
# of using sys.executable.
mkdir -p "$TMP/python-bin"
ln -sfn "$PY_LAUNCH" "$TMP/python-bin/python"
ln -sfn "$PY_LAUNCH" "$TMP/python-bin/python3"
export PATH="$TMP/python-bin:$PATH"

echo "weaveprofiler offline suite"
echo "  包目录: $PKG"
echo "  解释器: $PY_REAL ($("$PY_LAUNCH" -c 'import sys; print(sys.version.split()[0])'))"
echo "  字节码启动器: $PY_LAUNCH"
if [ "$WANT_NUMPY" -eq 1 ]; then
  if [ -n "$PY_NUM" ]; then
    [ "$PY_NUM_REAL" = "$PY_REAL" ] || echo "  numpy 测试解释器: $PY_NUM_REAL"
  else
    echo "  numpy 不可用：可选比较测试将记 SKIP"
  fi
else
  echo "  --skip-numpy：可选比较测试将记 SKIP"
fi
echo "  日志目录: $LOGS"

echo "  __pycache__ 基线: ${CACHE_BASELINE_COUNT} 项"

NAMES=()
CMDS=()
RCS=()
MS=()
SELECTED_TESTS=0

run_item() {
  local label="$1"
  shift

  if [ -n "$ONLY" ] && [[ "$label" != *"$ONLY"* ]]; then
    return 0
  fi
  SELECTED_TESTS=$((SELECTED_TESTS + 1))

  local display=""
  local arg quoted
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    display+=" ${quoted}"
  done
  display="${display# }"

  NAMES+=("$label")
  CMDS+=("$display")

  if [ $# -eq 0 ]; then
    RCS+=("SKIP")
    MS+=(0)
    printf '[SKIP] %s\n' "$label"
    return 0
  fi

  local safe log t0 t1 rc
  safe="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
  log="$LOGS/$(printf '%03d' "${#NAMES[@]}")_${safe}.log"
  t0=$(date +%s%3N)
  ( cd "$PKG" && "$@" ) >"$log" 2>&1
  rc=$?
  t1=$(date +%s%3N)
  RCS+=("$rc")
  MS+=("$((t1 - t0))")

  if [ "$rc" -eq 0 ]; then
    printf '[PASS] %s (%.2fs)\n' "$label" "$(awk "BEGIN {print (${t1}-${t0})/1000}")"
  else
    printf '[FAIL] %s (rc=%d, %.2fs)\n' "$label" "$rc" \
      "$(awk "BEGIN {print (${t1}-${t0})/1000}")"
    printf '       command: %s\n' "$display"
    printf '       log: %s\n' "$log"
    tail -n 30 "$log" | sed 's/^/       | /' >&2
  fi
  return 0
}

skip_item() {
  local label="$1"
  if [ -n "$ONLY" ] && [[ "$label" != *"$ONLY"* ]]; then
    return 0
  fi
  SELECTED_TESTS=$((SELECTED_TESTS + 1))
  NAMES+=("$label")
  CMDS+=("")
  RCS+=("SKIP")
  MS+=(0)
  printf '[SKIP] %s\n' "$label"
}

PY_RUN=("$PY_LAUNCH" -I -B)

# Knowledge base and canonical semantics.
run_item "KB tests/validate_kb.py" "${PY_RUN[@]}" tests/validate_kb.py configs
run_item "KB tests/semantic_kb_test.py" "${PY_RUN[@]}" tests/semantic_kb_test.py
run_item "KB tests/test_code_patterns.py" "${PY_RUN[@]}" tests/test_code_patterns.py
run_item "CONTRACT tests/test_contract_consistency.py" \
  "${PY_RUN[@]}" tests/test_contract_consistency.py
run_item "ROUTE tests/test_registry_language.py" "${PY_RUN[@]}" tests/test_registry_language.py

# Scan and deterministic instrumentation.
run_item "INSTRUMENT tests/test_card_schema.py" "${PY_RUN[@]}" tests/test_card_schema.py
run_item "PROBE tests/test_probe_pipeline.py" "${PY_RUN[@]}" tests/test_probe_pipeline.py
run_item "EXPERIMENTS tests/experiments/test_direct_probe_agent.py" \
  "${PY_RUN[@]}" tests/experiments/test_direct_probe_agent.py
run_item "CLI tests/test_run_cli.py" "${PY_RUN[@]}" tests/test_run_cli.py
run_item "CAPTURE tests/test_capture_rocm_paths_heartbeat.py" \
  "${PY_RUN[@]}" tests/test_capture_rocm_paths_heartbeat.py

# Capture, correctness, and delivery contracts.
run_item "VALIDATE tests/test_correctness_contract.py" \
  "${PY_RUN[@]}" tests/test_correctness_contract.py
run_item "CAPTURE tests/test_hardware_utilization.py" \
  "${PY_RUN[@]}" tests/test_hardware_utilization.py
run_item "CAPTURE tests/test_build_rocprof_sw_hw.py" \
  "${PY_RUN[@]}" tests/test_build_rocprof_sw_hw.py
run_item "CAPTURE tests/test_build_rocprof_sys_sw_hw.py" \
  "${PY_RUN[@]}" tests/test_build_rocprof_sys_sw_hw.py
run_item "CAPTURE tests/test_rocprof_sys_adapter.py" \
  "${PY_RUN[@]}" tests/test_rocprof_sys_adapter.py
run_item "DECODE tests/decode_roundtrip.py" \
  "${PY_RUN[@]}" tests/decode_roundtrip.py
run_item "RECORDER tests/frontend_api_check.py" \
  "${PY_RUN[@]}" tests/frontend_api_check.py
run_item "RECORDER tests/test_hip_recorder.py --compile-only" \
  "${PY_RUN[@]}" tests/test_hip_recorder.py --compile-only

# Harness is hermetic and builds its fixture in a temporary directory.
run_item "HARNESS tests/test_deterministic_harness.py" \
  "${PY_RUN[@]}" tests/test_deterministic_harness.py
run_item "HARNESS tests/test_fidelity_gates_and_causal.py" \
  "${PY_RUN[@]}" tests/test_fidelity_gates_and_causal.py
run_item "HARNESS tests/test_mcp_hook_installer.py" \
  "${PY_RUN[@]}" tests/test_mcp_hook_installer.py
run_item "HARNESS tests/test_wp_megakernel.py" \
  "${PY_RUN[@]}" tests/test_wp_megakernel.py

# Package contract intentionally runs last, after every test has had a chance
# to leave source/cache residue.
run_item "PACKAGE tests/package_structure.py" \
  "${PY_RUN[@]}" tests/package_structure.py

# Cache audit is an independent test result, so a cache leak cannot be hidden by
# a successful package test or a skipped --only selection.
snapshot_cache >"$TMP/cache_after.txt"
comm -13 "$TMP/cache_before.txt" "$TMP/cache_after.txt" >"$TMP/cache_added.txt" || true

if [ -s "$TMP/cache_added.txt" ]; then
  NAMES+=("CACHE no new __pycache__")
  CMDS+=("find \"$PKG\" -type d -name __pycache__ -o -type f -path '*/__pycache__/*'")
  RCS+=("1")
  MS+=(0)
  printf '[FAIL] CACHE no new __pycache__\n' >&2
  printf '       New cache entries:\n' >&2
  sed 's/^/       | /' "$TMP/cache_added.txt" >&2
else
  NAMES+=("CACHE no new __pycache__")
  CMDS+=("")
  RCS+=("0")
  MS+=(0)
  printf '[PASS] CACHE no new __pycache__\n'
fi

SUITE_T1=$(date +%s%3N)
SUITE_MS=$((SUITE_T1 - SUITE_T0))

if [ "$SELECTED_TESTS" -eq 0 ]; then
  echo "没有测试匹配 --only '$ONLY'。" >&2
  exit 2
fi

npass=0
nfail=0
nskip=0
sum_ms=0
printf '\n%-72s%-12s%10s\n' "TEST" "RESULT" "TIME(s)"
for i in "${!NAMES[@]}"; do
  if [ "${RCS[$i]}" = "SKIP" ]; then
    result="SKIP"
    nskip=$((nskip + 1))
  elif [ "${RCS[$i]}" -eq 0 ]; then
    result="PASS"
    npass=$((npass + 1))
    sum_ms=$((sum_ms + ${MS[$i]}))
  else
    result="FAIL(${RCS[$i]})"
    nfail=$((nfail + 1))
    sum_ms=$((sum_ms + ${MS[$i]}))
  fi
  printf '%-72s%-12s%10s\n' "${NAMES[$i]}" "$result" \
    "$(awk "BEGIN {printf \"%.2f\", ${MS[$i]}/1000}")"
done

if [ "$nfail" -ne 0 ]; then
  echo
  echo "失败命令（可直接复现）："
  for i in "${!NAMES[@]}"; do
    if [ "${RCS[$i]}" != "SKIP" ] && [ "${RCS[$i]}" -ne 0 ]; then
      echo "  - ${NAMES[$i]}"
      echo "    ${CMDS[$i]}"
    fi
  done
fi

echo
printf 'TOTAL: %d PASS, %d FAIL, %d SKIP | suite wall %.2fs, test sum %.2fs\n' \
  "$npass" "$nfail" "$nskip" \
  "$(awk "BEGIN {print ${SUITE_MS}/1000}")" \
  "$(awk "BEGIN {print ${sum_ms}/1000}")"

# Keep failure evidence.  Delete the temporary directory only on a clean pass.
if [ -z "$KEEP" ] && [ "$nfail" -eq 0 ]; then
  "$PY" -I -B - "$TMP" <<'PYCLEAN'
import shutil
import sys
shutil.rmtree(sys.argv[1], ignore_errors=True)
PYCLEAN
else
  echo "日志保留在: $TMP"
fi

if [ "$nfail" -ne 0 ]; then
  echo "OFFLINE SUITE FAIL"
  exit 1
fi
echo "OFFLINE SUITE PASS"
