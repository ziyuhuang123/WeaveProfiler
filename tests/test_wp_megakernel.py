#!/usr/bin/env python3
"""Hermetic tests for the first-class `wp` recorder frontends and the megakernel audit engine.

Run:  python -B -m unittest tests/test_wp_megakernel.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof import megakernel as mk  # noqa: E402

WPD = mk.load_wp_decode()
KB = {s["semantic_id"]: int(s["event_id"]) for s in WPD._fi.load_kb()["semantics"] if s.get("event_id") is not None}
E_LOAD, E_MMA, E_EPI, E_BAR = KB["load"], KB["mma"], KB["epilogue"], KB["barrier"]
OPEN, CLOSE, END = WPD.OPEN, WPD.CLOSE, WPD.END
CUDA_DIR = ROOT / "recorders" / "cuda_cpp"
HIP_DIR = ROOT / "recorders" / "hip_cpp"
CUTE_DIR = ROOT / "recorders" / "cutedsl"
TRITON_DIR = ROOT / "recorders" / "triton"
N_SMS, N_LONG, N_SHORT, LONG_US, SHORT_US = 78, 50, 28, 374, 191
BASE_NS = 5_000_000

NAMES = {
    "events": {str(E_LOAD): "kv_topk_load", str(E_MMA): "sparse_mla_mma", str(E_EPI): "epilogue",
               str(E_BAR): "grid_sync"},
    "semantics": {str(E_LOAD): "load", str(E_MMA): "mma", str(E_EPI): "epilogue", str(E_BAR): "barrier"},
    "roles": {"0": "persistent_cta"},
}
CLEAN_SOURCE = (
    "// SparseMLA persistent decode megakernel (clean production source)\n"
    "__global__ void sparse_mla_decode(const float* q, const float* kv, float* o) {\n"
    "    for (int tile = blockIdx.x; tile < 1024; tile += gridDim.x) { /* topk gather + mma */ }\n"
    "}\n"
)
INSTRUMENTED_SOURCE = CLEAN_SOURCE.replace("{\n    for", "{\n    WP_OPEN(0);\n    for").replace(
    "\n}\n", "\n    WP_CLOSE();\n}\n").replace("// SparseMLA", '#include "wp_lite.cuh"\n// SparseMLA')


def _sm_of(cta: int) -> int:
    return (cta * 7) % N_SMS  # TopK/head modulo placement: CTA -> physical SM permutation


def sparse_mla_slots(*, stub: bool = False, barrier: bool = False):
    """78 persistent CTAs (gran=1): CTAs 0..49 run 374 us, CTAs 50..77 finish at 191 us."""
    slots = {}
    for cta in range(N_SMS):
        long = cta < N_LONG
        work_end = BASE_NS + (LONG_US if long else SHORT_US) * 1000
        recs = [(BASE_NS, OPEN | 0, _sm_of(cta)),
                (BASE_NS + 1_000, E_LOAD, cta), (BASE_NS + 40_000, END | E_LOAD, cta),
                (BASE_NS + 40_000, E_MMA, cta), (work_end - 2_000, END | E_MMA, cta)]
        if stub:  # empty 64 ns probe: recorder self-latency only
            recs += [(work_end - 1_500, E_EPI, cta), (work_end - 1_436, END | E_EPI, cta)]
        close = work_end
        if barrier and not long:  # early SM waits at grid_sync for the stragglers
            close = BASE_NS + LONG_US * 1000
            recs += [(work_end, E_BAR, 0), (close, END | E_BAR, 0)]
        recs.append((close, CLOSE, 0))
        slots[cta] = recs
    return slots


def write_dump(path: Path, *, magic: bytes = b"WPLITE02", **kw) -> Path:
    blob = WPD.build_dump_bytes(sparse_mla_slots(**kw), n_slots=N_SMS, cap=16, wpc=1, magic=magic,
                                meta={"arch": "sm_90", "gran": 1, "kernel": "sparse_mla_decode"})
    path.write_bytes(blob)
    return path


def runtime_metrics(**over):
    base = [374_000.0, 374_400.0, 373_900.0, 374_200.0, 374_100.0, 373_800.0]
    rm = {
        "warmup": 3,
        "timing_protocol": "ABBA",
        "baseline_ns": base,
        "instrumented_ns": [b * 1.019 for b in base],
        "output_match": True,
        "l0_stats": {"regs": 72, "spill_stores": 16, "spill_loads": 16, "stack_bytes": 16, "stl": 2, "ldl": 2},
        "inst_stats": {"regs": 72, "spill_stores": 16, "spill_loads": 16, "stack_bytes": 16, "stl": 3, "ldl": 3},
        "kernel_name": "sparse_mla",
    }
    rm.update(over)
    return rm


def make_nsys_sqlite(path: Path, kernel_us: float = LONG_US) -> Path:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER,"
        " correlationId INTEGER, demangledName INTEGER, shortName INTEGER);"
        "CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);"
        "CREATE TABLE GPU_METRICS (typeId INTEGER, timestamp INTEGER, metricId INTEGER, value INTEGER);"
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "void sparse_mla_decode<128>(...)"), (2, "rmsnorm")])
    t = 1_000_000_000
    launches = [(t, t + 900_000, 1), (t + 2_000_000, t + 2_000_000 + 50_000, 2)]
    for i in range(3):  # warm launches
        s = t + 3_000_000 + i * 1_000_000
        launches.append((s, s + int(kernel_us * 1000) + (i - 1) * 400, 1))
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?)",
                     [(a, b, k, n, n) for k, (a, b, n) in enumerate(launches)])
    metrics = [(0, "SMs Active [Throughput %]"), (1, "SM Issue [Throughput %]"),
               (2, "Tensor Active [Throughput %]"), (3, "DRAM Read Bandwidth [Throughput %]"),
               (4, "DRAM Write Bandwidth [Throughput %]"), (5, "PCIe RX Throughput [Throughput %]")]
    conn.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, ?, ?)", metrics)
    rows = []
    for ts in range(t, t + 7_000_000, 5_000):  # 200 kHz
        for mid, _ in metrics:
            rows.append((7, ts, mid, 20 + 10 * mid))
    conn.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev
    return mod


_ENV_KEYS = ("WEAVEPROF_SUPERVISOR_HMAC_KEY", "WEAVEPROF_LOCK_DIR", "WEAVEPROF_CACHE", "WEAVEPROF_LOCK_PROVIDER")
_SAVED_ENV = {}
_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="wp-mk-env-")
    for k in _ENV_KEYS:
        _SAVED_ENV[k] = os.environ.get(k)
    os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "wp-megakernel-test-key"
    os.environ["WEAVEPROF_LOCK_DIR"] = str(Path(_TMP) / "locks")
    os.environ["WEAVEPROF_CACHE"] = str(Path(_TMP) / "cache")
    os.environ.pop("WEAVEPROF_LOCK_PROVIDER", None)


def tearDownModule():
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(_TMP, ignore_errors=True)


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="wp-mk-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class WpDecodeTest(_TmpCase):
    def test_wplite02_and_filite02_decode_identically(self):
        results = {}
        for magic in (b"WPLITE02", b"FILITE02"):
            p = write_dump(self.tmp / f"{magic.decode()}.bin", magic=magic)
            trace, summary = WPD.decode_file(p, NAMES)
            self.assertEqual(summary["magic"], magic.decode())
            self.assertEqual(summary["errors"], {})
            self.assertEqual(summary["overflow"], 0)
            self.assertEqual(summary["n_ctas"], N_SMS)
            self.assertEqual(summary["events"]["mma"]["n"], N_SMS)
            results[magic] = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertEqual(results[b"WPLITE02"], results[b"FILITE02"])

    def test_bad_magic_rejected_and_selftest(self):
        bad = self.tmp / "bad.bin"
        bad.write_bytes(b"XXLITE02" + bytes(24))
        with self.assertRaises(SystemExit):
            WPD.read_dump(str(bad))
        legacy = _load_module("fi_lite_decode_magic_probe", ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_lite_decode.py")
        good = write_dump(self.tmp / "wp.bin")
        self.assertEqual(legacy.read_dump(str(good))["magic"], "WPLITE02")
        with mock.patch("sys.stdout", new=open(os.devnull, "w", encoding="utf-8")) as sink:
            self.assertTrue(WPD.selftest())
            sink.close()

    def test_canonical_api_surface(self):
        for name in ("read_dump", "decode", "write_json", "main", "selftest"):
            self.assertTrue(callable(getattr(WPD, name)), name)


class HeaderContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wp = (CUDA_DIR / "wp_lite.cuh").read_text(encoding="utf-8")
        cls.hip = (HIP_DIR / "wp_lite_hip.h").read_text(encoding="utf-8")

    def test_wp_build_switches_map_before_defaults(self):
        for wp_name, fi_name in (("WP_LEVEL", "FI_LEVEL"), ("WP_TIMER", "FI_TIMER"), ("WP_GRAN", "FI_GRAN"),
                                 ("WP_GROUPS", "FI_GROUPS")):
            mapping = f"#define {fi_name} {wp_name}"
            self.assertIn(mapping, self.wp)
            self.assertLess(self.wp.index(mapping), self.wp.index(f"#ifndef {fi_name}\n#define {fi_name}"))
        for name in ("WP_CTX_SMEM", "WP_CTX_REG", "WP_CTX_REG_NOINLINE", "WP_DEFINE_HOST_API"):
            self.assertIn(name, self.wp)
        self.assertIn("FiBuf wp_active_buf()", self.wp)
        for wp_name, fi_name in (("WP_LEVEL", "FI_LEVEL"), ("WP_TIMER", "FI_TIMER"), ("WP_GRAN", "FI_GRAN")):
            self.assertIn(f"#define {fi_name} {wp_name}", self.hip)
        self.assertLess(self.hip.index("#define FI_TIMER WP_TIMER"), self.hip.index("#define FI_TIMER 1"))

    def test_full_wp_alias_surface(self):
        macros = ("WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLK", "WP_TAG_CLOSE", "WP_TAG_ROLE_SWITCH",
                  "WP_CTX_PARAM", "WP_CTX_ARG", "WP_BG", "WP_EG", "WP_MG", "WP_BG_PRED", "WP_EG_PRED", "WP_MG_PRED",
                  "WP_TRANS_PRED", "WP_OPEN", "WP_CLOSE", "WP_B", "WP_E", "WP_M")
        for m in macros:
            self.assertRegex(self.wp, rf"#define {m}\b", m)
        for fn in ("wp_alloc", "wp_attach", "wp_reset", "wp_dump", "wp_free", "wp_host_init", "wp_host_dump"):
            self.assertRegex(self.wp, rf"static inline \w+ {fn}\(", fn)
            self.assertRegex(self.hip, rf"static inline \w+ {fn}\(", fn)
        for m in ("WP_CTX_PARAM", "WP_CTX_ARG"):
            self.assertRegex(self.hip, rf"#define {m}\b")
        for export in ("wp_init", "wp_reset", "wp_dump", "wp_free"):
            self.assertRegex(self.wp, rf'extern "C" \w+ {export}\(')

    def test_sync_cursor_handles_every_context_mode(self):
        body = self.wp
        self.assertNotRegex(body, r"#define WP_SYNC_CURSOR_(?:OUT|IN)\(\).*fi_ctx_\.p")
        sync_body = body[body.index("wp_sync_cursor_out_smem"):]
        i_smem = sync_body.index("wp_sync_cursor_out_smem(WP_SMEM_SLOT())")
        i_reg = sync_body.index("#elif defined(WP_CTX_REG)")
        i_def = sync_body.index("#else", i_reg)
        self.assertLess(i_smem, i_reg)
        self.assertIn("wp_sync_cursor_out_reg(fi_ctx_)", sync_body[i_reg:i_def])
        self.assertIn("WP_REG_SLOT_SHIFT", sync_body[i_reg:i_def])
        self.assertIn("wp_sync_cursor_out_ctx(fi_ctx_)", sync_body[i_def:])
        self.assertEqual(len(re.findall(r"#define WP_SYNC_CURSOR_OUT\(\)", body)), 4)
        self.assertEqual(len(re.findall(r"#define WP_SYNC_CURSOR_IN\(\)", body)), 4)
        self.assertEqual(len(re.findall(r"static inline \w+ wp_host_init\(", body)), 1)
        # WP_CTX_SMEM callees bind the shared context under the kernel's name (WP_SMEM_SLOT resolves in both).
        self.assertIn("#define WP_SMEM_PARAM_DECL WpSmemCtx* fi_sctx_", self.wp)
        self.assertIn("#define WP_SMEM_PARAM_DECL WpSmemCtx& fi_sctx_", self.wp)
        self.assertIn("#define FI_CTX_ARG , fi_sctx_", self.wp)


class FrontendFacadeTest(unittest.TestCase):
    def _env(self, **values):
        keys = ("WP_LEVEL", "WP_GRAN", "WP_TIMER", "FI_LEVEL", "FI_GRAN", "FI_TIMER")
        saved = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update({k: str(v) for k, v in values.items()})

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)

    def _patch_modules(self, mapping):
        sentinel = object()
        saved = {k: sys.modules.get(k, sentinel) for k in mapping}
        for k, v in mapping.items():
            sys.modules[k] = v

        def restore():
            for k, old in saved.items():
                if old is sentinel:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old
        self.addCleanup(restore)

    def test_wp_cute_maps_env_and_degrades_without_cutlass(self):
        self._env(WP_LEVEL=1, WP_GRAN=1)
        self._patch_modules({"cutlass": None})
        mod = _load_module("wp_cute_under_test", CUTE_DIR / "wp_cute.py")
        self.assertEqual((os.environ["FI_LEVEL"], os.environ["FI_GRAN"]), ("1", "1"))
        self.assertEqual((mod.WP_LEVEL, mod.WP_GRAN), (1, 1))
        self.assertFalse(mod.wp_available())
        with self.assertRaises(ImportError):
            mod.wp_open  # noqa: B018
        with tempfile.TemporaryDirectory() as td:
            p = mod.write_names(os.path.join(td, "names.json"), {1: "load"}, roles={0: "producer"})
            self.assertEqual(json.loads(Path(p).read_text(encoding="utf-8"))["events"], {"1": "load"})
        for name in ("open_perfetto", "write_names"):
            self.assertTrue(callable(getattr(mod, name)))

    def test_wp_cute_exports_device_api_when_cutlass_present(self):
        self._env()
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.Boolean = bool
        fake_cutlass.Int32 = int
        fake_cutlass.Int64 = int
        fake_cutlass.Constexpr = type("Constexpr", (), {"__class_getitem__": classmethod(lambda cls, _i: cls)})
        fake_cutlass.const_expr = lambda v: v
        fake_cute = types.ModuleType("cutlass.cute")
        fake_cute.Tensor = object
        fake_cute.jit = lambda fn: fn
        fake_cute.arch = types.SimpleNamespace()
        fake_cutlass.cute = fake_cute
        fake_mlir = types.ModuleType("cutlass._mlir")
        fake_dialects = types.ModuleType("cutlass._mlir.dialects")
        fake_nvvm = types.ModuleType("cutlass._mlir.dialects.nvvm")
        fake_dialects.nvvm = fake_nvvm
        fake_mlir.dialects = fake_dialects
        fake_dsl = types.ModuleType("cutlass.cutlass_dsl")
        fake_dsl.T = types.SimpleNamespace(i64=lambda: "i64", i32=lambda: "i32")
        fake_dsl.dsl_user_op = lambda fn: fn
        fake_mods = {
            "cutlass": fake_cutlass, "cutlass.cute": fake_cute,
            "cutlass._mlir": fake_mlir, "cutlass._mlir.dialects": fake_dialects,
            "cutlass._mlir.dialects.nvvm": fake_nvvm, "cutlass.cutlass_dsl": fake_dsl,
        }
        self._patch_modules(fake_mods)
        mod = _load_module("wp_cute_fake_under_test", CUTE_DIR / "wp_cute.py")
        self.assertTrue(mod.wp_available())
        for wp_name, fi_name in (("WpBuffer", "FiBuffer"), ("WpCtx", "FiCtx"), ("wp_open", "fi_open"),
                                 ("wp_open_by", "fi_open_by"), ("WP_B", "fi_b"), ("WP_CLOSE", "fi_close"),
                                 ("wp_wrap", "_FiCuteProxy"), ("wp_get_attached", "fi_get_attached")):
            self.assertIs(getattr(mod, wp_name), getattr(mod, fi_name))

    def test_wp_triton_facades(self):
        self._env(WP_LEVEL=1, WP_TIMER=1)
        self._patch_modules({"triton": None, "triton.language": None})
        nv = _load_module("wp_triton_under_test", TRITON_DIR / "wp_triton.py")
        amd = _load_module("wp_triton_amd_under_test", TRITON_DIR / "wp_triton_amd.py")
        self.assertEqual(os.environ["FI_TIMER"], "1")
        cfg = nv.wp_cfg(cap=64, warps=4, n_slots=8, threads=128)
        self.assertEqual((cfg.level, cfg.timer, cfg.gran), (1, 1, 0))
        with self.assertRaises(ImportError):
            nv.wp_open  # noqa: B018
        with self.assertRaises(NotImplementedError):
            amd.wp_open_by()
        with self.assertRaises(ImportError):
            amd.wp_b_16b  # noqa: B018
        with tempfile.TemporaryDirectory() as td:
            doc = json.loads(Path(amd.write_names(os.path.join(td, "n.json"), {2: "wait"})).read_text("utf-8"))
            self.assertEqual(doc["group"], "cu")


class SkewAnalysisTest(_TmpCase):
    def _decoded(self, **kw):
        return WPD.decode_file(write_dump(self.tmp / "d.bin", **kw), NAMES)

    def test_sparse_mla_78sm_straggler_partition(self):
        trace, summary = self._decoded()
        skew = mk.analyze_megakernel_skew(summary, trace, total_sms=78)
        self.assertEqual(skew["observed_sms"], 78)
        self.assertEqual(skew["split_mode"], "bimodal_gap")
        self.assertEqual(skew["stragglers"]["count"], 50)
        self.assertEqual(skew["stragglers"]["busy_us_median"], 374.0)
        self.assertEqual(skew["early_finishing"]["count"], 28)
        self.assertEqual(skew["early_finishing"]["busy_us_median"], 191.0)
        self.assertEqual(skew["early_finishing"]["sm_ids"], sorted(_sm_of(c) for c in range(N_LONG, N_SMS))[:24])
        dist = skew["sm_busy"]
        self.assertEqual((dist["min_us"], dist["median_us"], dist["max_us"], dist["spread_us"]), (191.0, 374.0, 374.0, 183.0))
        self.assertAlmostEqual(skew["idle_sm_fraction"], round(28 / 78, 4))
        self.assertAlmostEqual(skew["sm_spread_ratio"], round(183 / 374, 4))
        codes = [h["code"] for h in skew["hints"]]
        self.assertIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", codes)
        self.assertIn("50 SMs @ 374us vs 28 SMs @ 191us", skew["hints"][0]["evidence"])
        self.assertEqual({s["stage"] for s in skew["stages"]}, {"load", "mma"})
        self.assertIn("persistent_cta", skew["role_stats"])

    def test_barrier_starved_sms_subtract_grid_sync_wait(self):
        trace, summary = self._decoded(barrier=True)
        skew = mk.analyze_megakernel_skew(summary, trace, total_sms=78)
        self.assertEqual(skew["busy_basis"], "lane_lifetime_minus_wait")
        self.assertEqual((skew["stragglers"]["count"], skew["early_finishing"]["count"]), (50, 28))
        self.assertEqual(skew["early_finishing"]["busy_us_median"], 191.0)
        self.assertEqual(skew["early_finishing"]["barrier_starved"], 28)
        self.assertIn("GRID_SYNC_WAIT_DOMINANT", [h["code"] for h in skew["hints"]])

    def test_balanced_kernel_has_no_split_hint(self):
        slots = {c: [(BASE_NS, OPEN, c), (BASE_NS + 1000, E_MMA, 0), (BASE_NS + 300_000 + c * 10, END | E_MMA, 0),
                     (BASE_NS + 301_000 + c * 10, CLOSE, 0)] for c in range(N_SMS)}
        p = self.tmp / "bal.bin"
        p.write_bytes(WPD.build_dump_bytes(slots, n_slots=N_SMS, cap=8, wpc=1, meta={"arch": "sm_90", "gran": 1}))
        trace, summary = WPD.decode_file(p, NAMES)
        skew = mk.analyze_megakernel_skew(summary, trace, total_sms=78)
        self.assertNotIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", [h["code"] for h in skew["hints"]])
        self.assertLess(skew["sm_spread_ratio"], 0.01)


class NsysSqliteTest(_TmpCase):
    def test_fast_extract_warm_kernel_and_metrics(self):
        db = make_nsys_sqlite(self.tmp / "cap.sqlite")
        hw = mk.extract_nsys_sqlite_fast(db, "sparse_mla")
        k = hw["kernel"]
        self.assertEqual(k["launches"], 4)
        self.assertEqual(k["duration_us"], 374.0)
        self.assertIn("sparse_mla_decode", k["name"])
        self.assertEqual(set(hw["hw_series"]), {"tensor_active", "sm_active", "warp_issue", "dram_read", "dram_write"})
        self.assertEqual(hw["hw_series"]["tensor_active"]["val"][0], 40.0)
        self.assertGreaterEqual(hw["in_window_samples"], 74)
        self.assertEqual(hw["sample_freq_hz"], 200000.0)
        self.assertLess(hw["elapsed_ms"], 150.0)
        with self.assertRaises(ValueError):
            mk.extract_nsys_sqlite_fast(db, "no_such_kernel")

    def test_fast_extract_multi_typeid_restarted_timestamps(self):
        # Mirrors real nsys SQLite where a secondary typeId block (e.g., PCIe/NVLink) is appended with restarted timestamps.
        db = self.tmp / "multi_tid.sqlite"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER,"
            " correlationId INTEGER, demangledName INTEGER, shortName INTEGER);"
            "CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);"
            "CREATE TABLE GPU_METRICS (typeId INTEGER, timestamp INTEGER, metricId INTEGER, value INTEGER);"
        )
        conn.execute("INSERT INTO StringIds VALUES (1, 'kernel_sparse_mla_decode')")
        # Kernel runs near the end of block 1 (t = 9_500_000..9_874_000 ns)
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (9500000, 9874000, 0, 1, 1, 1)")
        conn.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)", [
            (7, 3, "SMs Active [Throughput %]"),
            (7, 5, "Tensor Active [Throughput %]"),
            (99, 0, "PCIe RX Throughput [Throughput %]"),
        ])
        # Block 1 (>50k rows to exercise O(log N) segment binary search): typeId=7, ts=0..10_000_000
        rows = [(7, ts, mid, 65) for ts in range(0, 10_000_000, 400) for mid in (3, 5)]
        # Block 2 (appended with restarted timestamps 0..10_000_000): typeId=99
        rows.extend((99, ts, 0, 10) for ts in range(0, 10_000_000, 2_000))
        conn.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()
        hw = mk.extract_nsys_sqlite_fast(db, "sparse_mla")
        self.assertGreaterEqual(hw["in_window_samples"], 900)
        self.assertEqual(set(hw["hw_series"]), {"sm_active", "tensor_active"})
        self.assertLess(hw["elapsed_ms"], 200.0)

    def test_fast_extract_reports_profiled_gpu_name(self):
        db = make_nsys_sqlite(self.tmp / "named.sqlite")
        self.assertIsNone(mk.extract_nsys_sqlite_fast(db, "sparse_mla")["gpu_name"])  # no TARGET_INFO_GPU table
        conn = sqlite3.connect(str(db))
        conn.executescript(
            "CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT, chipName TEXT);"
            "INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H20', 'GH100');"
            "INSERT INTO TARGET_INFO_GPU VALUES (1, 'NVIDIA H100 80GB HBM3', 'GH100');"
        )
        conn.commit()
        conn.close()
        hw = mk.extract_nsys_sqlite_fast(db, "sparse_mla")  # fixture kernels run on deviceId 0
        self.assertEqual((hw["gpu_name"], hw["gpu_chip"]), ("NVIDIA H20", "GH100"))


def _tensor_hw(vals, *, gpu_name=None, sm_vals=None, dur_us=None):
    """Synthetic extract_nsys_sqlite_fast() result: one sample every 5 us."""
    t_us = [5.0 * i for i in range(len(vals))]
    hw = {
        "kernel": {"name": "gemm", "device": 0, "duration_us": dur_us if dur_us is not None else t_us[-1]},
        "gpu_name": gpu_name,
        "hw_series": {
            "tensor_active": {"t_us": t_us, "val": [float(v) for v in vals], "metric_name": "Tensor Active [Throughput %]"},
            "sm_active": {"t_us": t_us, "val": [float(v) for v in (sm_vals or [100.0] * len(vals))],
                          "metric_name": "SMs Active [Throughput %]"},
        },
    }
    return hw


class TensorActiveScaleTest(unittest.TestCase):
    """nsys Tensor Active on H20 is normalized to the full GH100 die (H20 = 1/4 per SM): the overview PNG,
    hw_quartiles and the causal analysis must all use the same x4 correction (docs/l20d_achievable_peak_rebase.md)."""

    H20_RAW = [20, 25, 25, 24, 25, 25, 25, 25, 22, 25, 25, 25]

    def test_h20_by_recorded_gpu_name_is_corrected_x4(self):
        info = mk.resolve_tensor_active_scale(_tensor_hw(self.H20_RAW, gpu_name="NVIDIA H20"))
        self.assertTrue(info["applied"])
        self.assertEqual(info["scale"], 4.0)
        self.assertEqual(info["raw_max_pct"], 25.0)
        self.assertIn("NVIDIA H20", info["source"])

    def test_full_gh100_parts_are_never_scaled(self):
        for name in ("NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA H200 NVL"):
            with self.subTest(name=name):
                info = mk.resolve_tensor_active_scale(_tensor_hw([10, 12, 15], gpu_name=name), arch="sm_90a", card="h20")
                self.assertFalse(info["applied"], name)
                self.assertEqual(info["scale"], 1.0)

    def test_raw_above_ceiling_is_left_raw(self):
        info = mk.resolve_tensor_active_scale(_tensor_hw([30, 45, 60], gpu_name="NVIDIA H20"))
        self.assertFalse(info["applied"])
        self.assertIn("ceiling", info["reason"])

    def test_without_gpu_name_only_explicit_h20_card_enables_correction(self):
        self.assertTrue(mk.resolve_tensor_active_scale(_tensor_hw(self.H20_RAW), card="h20")["applied"])
        # sm_90a alone is also H100/H200: never infer H20 from the architecture.
        self.assertFalse(mk.resolve_tensor_active_scale(_tensor_hw(self.H20_RAW), arch="sm_90a", card="auto")["applied"])

    def test_overview_trace_and_viz_use_corrected_curve(self):
        from weaveprof import viz
        hw = _tensor_hw(self.H20_RAW, gpu_name="NVIDIA H20")
        hw["tensor_active_scale"] = mk.resolve_tensor_active_scale(hw)
        ev = mk._hw_counter_events(hw, hw["tensor_active_scale"])
        tensor = [e for e in ev if e.get("ph") == "C" and "tensor" in e["name"].lower()]
        self.assertEqual([e["args"]["pct"] for e in tensor], [min(100.0, 4.0 * v) for v in self.H20_RAW])
        self.assertEqual({e["name"] for e in tensor}, {"tensor_active [Tensor Active x4 (H20 die-normalization corrected)]"})
        window = next(e for e in ev if e.get("cat") == "nsys_kernel")
        self.assertEqual(window["args"]["tensor_active_scale"], 4.0)
        self.assertEqual(window["args"]["tensor_active_raw_max_pct"], 25.0)
        self.assertEqual(window["args"]["gpu_name"], "NVIDIA H20")
        doc = mk._merge_hw_trace({"traceEvents": []}, hw, tensor_scale=hw["tensor_active_scale"])
        self.assertTrue(doc["otherData"]["nsys_hw"]["tensor_active_scale"]["applied"])
        parsed = viz._parse_trace(doc)
        self.assertEqual(max(v for _, v in parsed["hw_curves"]["tensor_active"]), 100.0)
        self.assertIn("x4", parsed["hw_labels"]["tensor_active"])
        # Other curves are untouched.
        self.assertEqual(max(v for _, v in parsed["hw_curves"]["sm_active"]), 100.0)

    def test_uncorrected_cards_keep_raw_values_and_name(self):
        hw = _tensor_hw([30, 45, 60], gpu_name="NVIDIA H100 80GB HBM3")
        hw["tensor_active_scale"] = mk.resolve_tensor_active_scale(hw)
        ev = mk._hw_counter_events(hw, hw["tensor_active_scale"])
        tensor = [e for e in ev if e.get("ph") == "C" and "tensor" in e["name"].lower()]
        self.assertEqual([e["args"]["pct"] for e in tensor], [30.0, 45.0, 60.0])
        self.assertEqual({e["name"] for e in tensor}, {"Tensor Active [Throughput %]"})

    def test_quartiles_follow_the_same_factor(self):
        hw = _tensor_hw(self.H20_RAW, gpu_name="NVIDIA H20")
        raw_q = mk._compact_hw_quartiles(hw)["tensor_active"]
        hw["tensor_active_scale"] = mk.resolve_tensor_active_scale(hw)
        cor_q = mk._compact_hw_quartiles(hw)["tensor_active"]
        for raw, cor in zip(raw_q, cor_q):
            self.assertAlmostEqual(cor, min(100.0, 4.0 * raw), delta=0.25)  # both sides rounded to 0.1

    def test_plateau_without_correction_is_flagged_advisory(self):
        hw = _tensor_hw(self.H20_RAW)  # no GPU name, card unknown -> not corrected
        info = mk.resolve_tensor_active_scale(hw, arch="sm_90a", card="auto")
        msg = mk._tensor_plateau_suspect(hw, info)
        self.assertIsNotNone(msg)
        self.assertIn("TENSOR_ACTIVE_PLATEAU_UNCORRECTED (advisory)", msg)
        self.assertIsNone(mk._tensor_plateau_suspect(hw, mk.resolve_tensor_active_scale(hw, card="h20")))
        busy = _tensor_hw([20, 40, 60, 80, 95, 60, 40, 30, 20, 50, 70, 90])  # real full-range curve
        self.assertIsNone(mk._tensor_plateau_suspect(busy, mk.resolve_tensor_active_scale(busy)))


class WpSoControllerTest(unittest.TestCase):
    def test_ctypes_plumbing_with_fake_library(self):
        calls = []

        class Fn:
            def __init__(self, name, rc=0):
                self.name, self.rc = name, rc

            def __call__(self, *args):
                calls.append((self.name, args))
                return self.rc

        lib = types.SimpleNamespace(wp_init=Fn("wp_init"), wp_reset=Fn("wp_reset"), wp_dump=Fn("wp_dump", 0),
                                    wp_free=Fn("wp_free", None))
        with tempfile.TemporaryDirectory() as td:
            with mk.WpSoController(None, lib=lib) as ctl:
                ctl.init(78, 1, 64)
                ctl.reset(types.SimpleNamespace(cuda_stream=1234))
                self.assertEqual(ctl.dump(Path(td) / "c.wp02.bin", {"arch": "sm_90"}), 0)
            self.assertEqual([c[0] for c in calls], ["wp_init", "wp_reset", "wp_dump", "wp_free"])
            self.assertEqual(calls[1][1], (1234,))
            lib.wp_dump = Fn("wp_dump", -3)
            ctl = mk.WpSoController(None, lib=lib).init(1, 1, 8)
            with self.assertRaisesRegex(RuntimeError, "fopen"):
                ctl.dump(Path(td) / "x.bin")
        with self.assertRaises(AttributeError):
            mk.WpSoController(None, lib=types.SimpleNamespace(wp_init=Fn("wp_init")))


class MegakernelAuditTest(_TmpCase):
    def _workspace(self):
        ws = Path(tempfile.mkdtemp(prefix="ws-", dir=self.tmp))
        (ws / "sparse_mla.cu").write_bytes(CLEAN_SOURCE.encode("utf-8"))
        return ws

    def _audit(self, name, *, dump_kw=None, rm=None, nsys=False, **kw):
        ws = self._workspace()
        dump = write_dump(self.tmp / f"{name}.wp02.bin", **(dump_kw or {}))
        kw.setdefault("render_overview_png", False)
        return mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
            runtime_metrics=rm if rm is not None else runtime_metrics(),
            nsys_sqlite=make_nsys_sqlite(self.tmp / f"{name}.sqlite") if nsys else None,
            card="h20", limit=0.10, out_dir=self.tmp / f"out-{name}", **kw)

    def test_a_clean_78sm_pass_sealed_fast(self):
        mk.load_wp_decode()
        t0 = time.perf_counter()
        rep = self._audit("pass", nsys=True)
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(rep["status"], "PASS", rep)
        self.assertTrue(rep["seal_verified"], rep)
        self.assertLess(rep["wall_clock_ms"], 200.0, rep)
        self.assertLess(cpu_ms, 400.0)
        self.assertLess(len(json.dumps(rep, separators=(",", ":")).encode("utf-8")), 2048)
        self.assertEqual(rep["kernel_us"], 374.0)
        self.assertAlmostEqual(rep["overhead_pct"], 1.9, places=3)
        self.assertEqual(rep["megakernel_skew"]["stragglers"], [50, 374.0])
        self.assertEqual(rep["megakernel_skew"]["early"], [28, 191.0])
        self.assertIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", rep["megakernel_skew"]["hints"])
        self.assertTrue(all(rep["fidelity_gates"][g]["ok"] for g in ("compiler_spill", "output", "stub", "overhead")))
        self.assertEqual((rep["fidelity_gates"]["overflow"], rep["fidelity_gates"]["pairing_errors"]), (0, 0))
        self.assertFalse(any("FAILED" in b for b in rep["structural_blockers"]), rep["structural_blockers"])
        out = self.tmp / "out-pass"
        for f in ("feedback.json", "verify.json", "seal.json"):
            self.assertTrue((out / f).is_file(), f)
        self.assertFalse((out / "trace.perfetto.json.gz").exists())
        fb = json.loads((out / "feedback.json").read_text(encoding="utf-8"))
        self.assertEqual(fb["causal"]["schema"], "weaveprof.causal-report/v1")
        self.assertEqual(fb["causal"]["observed_sms"], 78)
        self.assertGreater(fb["causal"]["hw_summary"]["tensor_active_cal_mean_pct"], 0.0)  # nsys curves merged
        self.assertEqual(fb["nsys"]["kernel"]["duration_us"], 374.0)
        seal = json.loads((out / "seal.json").read_text(encoding="utf-8"))
        self.assertEqual(seal["seal_sha256"], rep["seal_sha256"])
        self.assertEqual(seal["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertIsNotNone(seal["artifacts"]["dump"])
        verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        self.assertEqual(verify["status"], "PASS")
        # Verify default PNG-first delivery (no .gz by default) + opt-in R6 nsys top-pinned trace.gz
        rep_png = self._audit("pass-png", nsys=True, render_overview_png=True, write_perfetto_gz=True)
        self.assertEqual(rep_png["status"], "PASS", rep_png)
        out_png = self.tmp / "out-pass-png"
        import gzip as _gzip
        tr = json.loads(_gzip.decompress((out_png / "trace.perfetto.json.gz").read_bytes()).decode("utf-8"))
        tevs = tr["traceEvents"]
        self.assertEqual(tevs[0]["pid"], 1)
        self.assertEqual(tevs[0]["name"], "process_name")
        self.assertEqual(tevs[1]["pid"], 1)
        self.assertEqual(tevs[1]["name"], "process_sort_index")
        self.assertEqual(tevs[1]["args"]["sort_index"], -1000)
        hw_counters = [e for e in tevs if e.get("ph") == "C"]
        self.assertGreater(len(hw_counters), 50)
        self.assertTrue(all(e["pid"] == 1 and float(e["ts"]) >= 0.0 for e in hw_counters))
        hw_kernels = [e for e in tevs if e.get("cat") == "nsys_kernel"]
        self.assertEqual(len(hw_kernels), 1)
        self.assertEqual((hw_kernels[0]["pid"], hw_kernels[0]["ts"], hw_kernels[0]["dur"]), (1, 0.0, 374.0))
        sw_pids = {e["pid"] for e in tevs if isinstance(e.get("pid"), int) and e["pid"] != 1}
        self.assertTrue(sw_pids and all(p >= 100 for p in sw_pids), sw_pids)
        try:
            import matplotlib  # noqa: F401
            pngs = sorted(out_png.glob("*.png"))
            self.assertEqual(len(pngs), 1, (pngs, rep_png))
            png = pngs[0]
            self.assertRegex(png.name, r"^[a-z0-9._-]+_[a-z0-9_-]+_[a-z0-9_-]+_\d{8}_\d{6}\.png$")
            raw_png = png.read_bytes()
            self.assertTrue(raw_png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(raw_png), 20_000)
            self.assertIn(png.name, rep_png["artifacts"]["files"])
            self.assertEqual(rep_png["artifacts"].get("timeline_png"), png.name)
        except ImportError:
            pass
        # Round-to-round diff against the sealed causal report.
        rep2 = self._audit("pass-r2", prev_causal_report=fb)
        self.assertEqual(rep2["status"], "PASS")
        self.assertIsNotNone(rep2["round_diff"])
        self.assertTrue(rep2["causal_opportunities"])

    def test_h20_tensor_active_x4_end_to_end(self):
        """Workflow B on an H20 capture: overview trace, hw_quartiles and feedback all carry the x4 correction."""
        db = make_nsys_sqlite(self.tmp / "h20.sqlite")
        conn = sqlite3.connect(str(db))
        conn.executescript(
            "UPDATE GPU_METRICS SET value = 25 WHERE metricId = 2;"  # Tensor Active pinned at the H20 ceiling
            "CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT, chipName TEXT);"
            "INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H20', 'GH100');"
        )
        conn.commit()
        conn.close()
        ws = self._workspace()
        dump = write_dump(self.tmp / "h20.wp02.bin")
        rep = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES, runtime_metrics=runtime_metrics(),
            nsys_sqlite=db, card="auto", limit=0.10, out_dir=self.tmp / "out-h20",
            render_overview_png=False, write_perfetto_gz=True)
        self.assertEqual(rep["status"], "PASS", rep)
        self.assertEqual(rep["tensor_active_scale"], 4.0)
        if "hw_quartiles" in rep:
            self.assertEqual(rep["hw_quartiles"]["tensor_active"], [100.0, 100.0, 100.0, 100.0])
        self.assertFalse(any("PLATEAU" in b for b in rep["structural_blockers"]), rep["structural_blockers"])
        out = self.tmp / "out-h20"
        nsys_hw = json.loads((out / "nsys_hw.json").read_text(encoding="utf-8"))
        self.assertEqual(nsys_hw["gpu_name"], "NVIDIA H20")
        self.assertTrue(nsys_hw["tensor_active_scale"]["applied"])
        self.assertEqual(nsys_hw["hw_series"]["tensor_active"]["val"][0], 25.0)  # raw curve kept on disk
        import gzip as _gzip
        tr = json.loads(_gzip.decompress((out / "trace.perfetto.json.gz").read_bytes()).decode("utf-8"))
        tensor = [e for e in tr["traceEvents"] if e.get("ph") == "C" and "tensor" in e["name"].lower()]
        self.assertTrue(tensor)
        self.assertEqual({e["args"]["pct"] for e in tensor}, {100.0})
        self.assertEqual(tr["otherData"]["nsys_hw"]["tensor_active_scale"]["scale"], 4.0)

    def test_revise_probe_spec_preserves_dominant_semantic_stage(self):
        from weaveprof.contract import revise_probe_spec
        src = (
            "__global__ void k(float* a, float* b) {\n"
            "  for (int i = 0; i < 64; ++i) {\n"
            "    float v = a[i];\n"
            "    b[i] = v * v;\n"
            "  }\n"
            "}\n"
        )
        doc = {
            "roles": {"w": "all"},
            "probes": [
                {"id": "load", "begin": {"role": "w", "line": 3}, "end": {"role": "w", "line": 3}},
                {"id": "mma", "begin": {"role": "w", "line": 2}, "end": {"role": "w", "line": 5}},
            ],
        }
        fb = {"status": "FAIL", "fires": {"load": 64, "mma": 1}, "enclosing_loop": [2, 5]}
        revised = revise_probe_spec(doc, fb, src)
        self.assertEqual([p["id"] for p in revised["probes"]], ["mma"])
        # Verify weak-LLM tolerance: inline comments & trailing ';' in role expressions + numeric stage suffixes
        from weaveprof.contract import normalize_probe_spec
        norm = normalize_probe_spec(
            {
                "roles": {"dma": "warp_group_idx == 0 /* producer */;"},
                "probes": [
                    {"id": "dma_issue", "begin": {"role": "dma", "line": 3}, "end": {"role": "dma", "line": 3}},
                    {"id": "softmax0", "begin": {"role": "dma", "line": 4}, "end": {"role": "dma", "line": 4}},
                ],
            },
            src,
        )
        self.assertEqual(norm["warp_roles"][0]["predicate"], "warp_group_idx == 0")
        self.assertEqual([p["semantic_id"] for p in norm["probes"]], ["load", "softmax"])
        # Verify weak-LLM half-open touching boundary repair (a.end == b.begin -> trim a.end by 1)
        norm_ho = normalize_probe_spec(
            {
                "roles": {"cta": "all"},
                "probes": [
                    {"id": "load_w", "begin": {"role": "cta", "line": 2}, "end": {"role": "cta", "line": 4}},
                    {"id": "mma_wh", "begin": {"role": "cta", "line": 4}, "end": {"role": "cta", "line": 5}},
                ],
            },
            src,
        )
        self.assertEqual((norm_ho["probes"][0]["begin_line"], norm_ho["probes"][0]["end_line"]), (2, 3))
        self.assertEqual((norm_ho["probes"][1]["begin_line"], norm_ho["probes"][1]["end_line"]), (4, 5))

    def test_single_stage_loop_collapse_fails_gate(self):
        # 99.5% of active time collapsed into 1 single compute slab/lane MUST fail the event gate.
        e_comp = KB["compute"]
        slots = {
            c: [
                (BASE_NS, OPEN | 0, _sm_of(c)),
                (BASE_NS + 1_000, E_LOAD, c), (BASE_NS + 3_000, END | E_LOAD, c),
                (BASE_NS + 3_000, e_comp, c), (BASE_NS + 650_000, END | e_comp, c),
                (BASE_NS + 651_000, CLOSE, 0),
            ]
            for c in range(N_SMS)
        }
        dump = self.tmp / "collapsed.wp02.bin"
        dump.write_bytes(WPD.build_dump_bytes(slots, n_slots=N_SMS, cap=16, wpc=1,
                                              meta={"arch": "sm_90", "gran": 1, "kernel": "collapsed"}))
        names_c = {
            "events": {str(E_LOAD): "load", str(e_comp): "compute"},
            "semantics": {str(E_LOAD): "load", str(e_comp): "compute"},
            "roles": {"0": "persistent_cta"},
        }
        ws = self._workspace()
        rep = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=names_c,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-collapse",
        )
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["granularity"]["ok"])
        self.assertEqual(rep["fidelity_gates"]["granularity"]["code"], "SINGLE_STAGE_LOOP_COLLAPSE")
        self.assertTrue(any("SINGLE_STAGE_LOOP_COLLAPSE" in b for b in rep["structural_blockers"]), rep)

    def test_partial_warp_filtering_fails_gate(self):
        # Recording only 3 representative warps (0, 4, 8) out of wpc=12 MUST fail WARP_COVERAGE_INCOMPLETE.
        wpc = 12
        n_ctas = 4
        slots = {}
        for cta in range(n_ctas):
            for w in (0, 4, 8):
                s = cta * wpc + w
                slots[s] = [
                    (BASE_NS, OPEN | (0 if w == 0 else 1), cta),
                    (BASE_NS + 1_000, E_LOAD, cta), (BASE_NS + 15_000, END | E_LOAD, cta),
                    (BASE_NS + 15_000, E_MMA, cta), (BASE_NS + 60_000, END | E_MMA, cta),
                    (BASE_NS + 61_000, CLOSE, 0),
                ]
        dump = self.tmp / "partial_warps.wp02.bin"
        dump.write_bytes(WPD.build_dump_bytes(slots, n_slots=n_ctas * wpc, cap=16, wpc=wpc,
                                              meta={"arch": "sm_90", "gran": 0, "kernel": "partial_warps"}))
        ws = self._workspace()
        rep = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-partial-warps",
            render_overview_png=False,
        )
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["warps"]["ok"])
        self.assertEqual(rep["fidelity_gates"]["warps"]["code"], "WARP_COVERAGE_INCOMPLETE")
        self.assertTrue(any("WARP_COVERAGE_INCOMPLETE" in b for b in rep["structural_blockers"]), rep)

        # When ALL 12 warps open/close (even if warps 1..3 are idle OPEN+CLOSE-only producer warps), warp gate PASSES.
        for cta in range(n_ctas):
            for w in range(wpc):
                s = cta * wpc + w
                if s not in slots:
                    slots[s] = [(BASE_NS, OPEN | 0, cta), (BASE_NS + 500, CLOSE, 0)]
        dump_full = self.tmp / "all12_warps.wp02.bin"
        dump_full.write_bytes(WPD.build_dump_bytes(slots, n_slots=n_ctas * wpc, cap=16, wpc=wpc,
                                                   meta={"arch": "sm_90", "gran": 0, "kernel": "all12_warps"}))
        rep_ok = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump_full, names=NAMES,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-all12-warps",
            render_overview_png=False,
        )
        self.assertEqual(rep_ok["status"], "PASS", rep_ok)
        self.assertTrue(rep_ok["fidelity_gates"]["warps"]["ok"])

        # Regression: 9-warp persistent megakernel (~1,200 us) with 3 outer macro-blocks (25% / 5% / 70%)
        # and a resident hollow producer warp (w8 open 0..1200 us with only 1 tiny 5 us dot) MUST fail closed.
        wpc9 = 9
        coarse_names = {
            "events": {"4": "csa_attn", "5": "norm_router", "6": "moe_fused", "1": "dummy_init"},
            "semantics": {"4": "attention", "5": "compute", "6": "moe", "1": "load"},
            "roles": {"0": "compute_wg", "1": "producer_dma"},
        }
        slots9 = {}
        end_ns = BASE_NS + 1_200_000
        for cta in range(n_ctas):
            for w in range(8):
                slots9[cta * wpc9 + w] = [
                    (BASE_NS, OPEN | 0, cta),
                    (BASE_NS + 1_000, 4, 0), (BASE_NS + 300_000, END | 4, 0),
                    (BASE_NS + 300_000, 5, 0), (BASE_NS + 360_000, END | 5, 0),
                    (BASE_NS + 360_000, 6, 0), (end_ns - 1_000, END | 6, 0),
                    (end_ns, CLOSE, 0),
                ]
            slots9[cta * wpc9 + 8] = [
                (BASE_NS, OPEN | 1, cta),
                (BASE_NS + 1_000, 1, 0), (BASE_NS + 6_000, END | 1, 0),
                (end_ns, CLOSE, 0),
            ]
        dump_lazy = self.tmp / "lazy_9w.wp02.bin"
        dump_lazy.write_bytes(WPD.build_dump_bytes(slots9, n_slots=n_ctas * wpc9, cap=16, wpc=wpc9,
                                                   meta={"arch": "sm_90", "gran": 0, "kernel": "csa_moe_mk"}))
        rep_lazy = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump_lazy, names=coarse_names,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-lazy-9w",
            render_overview_png=False,
        )
        self.assertEqual(rep_lazy["status"], "FAIL", rep_lazy)
        self.assertFalse(rep_lazy["fidelity_gates"]["granularity"]["ok"])
        self.assertEqual(rep_lazy["fidelity_gates"]["granularity"]["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertFalse(rep_lazy["fidelity_gates"]["warps"]["ok"])
        self.assertEqual(rep_lazy["fidelity_gates"]["warps"]["code"], "HOLLOW_WARP_LANE")

    def test_nested_wrapper_collapse_fails_gate_and_scheduling_multiset_passes(self):
        # Reproduce FlashInfer FA2 defect: outer `attention` + `mma` macro-wrappers enclose inner
        # `load` (0.63 us) and `store` (1.42 us) so naive summed share was 49.75%. MUST fail closed!
        e_attn = KB["attention"]
        e_store = KB["store"]
        slots = {}
        for cta in range(16):
            slots[cta] = [
                (BASE_NS, OPEN | 0, cta),
                (BASE_NS + 100, e_attn, cta),
                (BASE_NS + 200, E_MMA, cta),
                (BASE_NS + 500, E_LOAD, cta), (BASE_NS + 1_130, END | E_LOAD, cta),
                (BASE_NS + 162_000, e_store, cta), (BASE_NS + 163_420, END | e_store, cta),
                (BASE_NS + 164_000, END | E_MMA, cta),
                (BASE_NS + 164_100, END | e_attn, cta),
                (BASE_NS + 164_500, CLOSE, 0),
            ]
        dump = self.tmp / "nested_fa2.wp02.bin"
        dump.write_bytes(WPD.build_dump_bytes(slots, n_slots=16, cap=16, wpc=1,
                                              meta={"arch": "sm_86", "gran": 1, "kernel": "SinglePrefill"}))
        names_fa = {
            "events": {str(e_attn): "attention", str(E_MMA): "mma", str(E_LOAD): "load", str(e_store): "store"},
            "semantics": {str(e_attn): "attention", str(E_MMA): "mma", str(E_LOAD): "load", str(e_store): "store"},
            "roles": {"0": "cta"},
        }
        ws = self._workspace()
        rep = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=names_fa,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10,
            out_dir=self.tmp / "out-nested-fa2", render_overview_png=False,
        )
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["granularity"]["ok"])
        self.assertEqual(rep["fidelity_gates"]["granularity"]["code"], "NESTED_WRAPPER_COLLAPSE")

        # Scheduling-aware assembly check in backend.assess: 100% identical core opcodes (204 == 204)
        # with software-pipeline reordering (global LCS ~0.92) MUST pass without forcing -O1!
        from weaveprof.backend import assess
        asm_dir = self.tmp / "sched_asm"
        asm_dir.mkdir()
        base_core = ["HMMA", "FFMA", "LDGSTS", "LDS", "STS", "MUFU"] * 34  # 204 core opcodes
        inst_core = list(base_core)
        for k in range(0, 36, 2):  # swap 18 adjacent pairs -> multiset 1.00, LCS ~0.91 (< 0.95)
            inst_core[k], inst_core[k + 1] = inst_core[k + 1], inst_core[k]
        (asm_dir / "l0.sass").write_text(
            "Function : flash_prefill\n" + "".join(f"/*{16 * i:04x}*/ {op} R0, R1, R2;\n" for i, op in enumerate(base_core)),
            encoding="utf-8",
        )
        (asm_dir / "inst.sass").write_text(
            "Function : flash_prefill\n" + "".join(f"/*{16 * i:04x}*/ {op} R0, R1, R2;\n" for i, op in enumerate(inst_core)),
            encoding="utf-8",
        )
        (asm_dir / "l0.bin").write_bytes(b"\x01\x02\x03\x04" * 8)
        (asm_dir / "inst.bin").write_bytes(b"\x01\x02\x03\x04" * 8)
        write_dump(asm_dir / "capture.fi02.bin")
        (asm_dir / "names.json").write_text(json.dumps(NAMES), encoding="utf-8")
        rt_asm = {
            "kernel_name": "flash_prefill",
            "baseline_sass": "l0.sass", "instrumented_sass": "inst.sass",
            "baseline_spill_bytes": 0, "instrumented_spill_bytes": 0, "extra_smem_bytes": 0,
            "baseline_output": "l0.bin", "instrumented_output": "inst.bin",
            "warmup": 3, "baseline_ns": [100_000.0] * 6, "instrumented_ns": [101_000.0] * 6,
        }
        checks_asm, metrics_asm, errors_asm, _, _, _ = assess(
            asm_dir, {"probes": [{"id": "kv_topk_load"}, {"id": "sparse_mla_mma"}]}, 0.10, runtime=rt_asm,
        )
        self.assertTrue(checks_asm["assembly"], (errors_asm, metrics_asm))
        self.assertGreaterEqual(metrics_asm["core_multiset_ratio"], 0.999)
        self.assertLess(metrics_asm["core_retention_ratio"], 0.95)

    def test_multinode_fabric_dualcolor_and_rich_stage_labels(self):
        from weaveprof.contract import resolve_fabric_scope, validate_stage_label_quality
        from weaveprof.pipeline.instrument.rule_engine import build_names_from_probe_spec
        from weaveprof.viz import _humanize_stage_label, _parse_trace, render_swhw_overview_png

        # 1. Cross-node vs intra-node fabric classification for both AMD & NVIDIA
        self.assertEqual(resolve_fabric_scope("InterNode_RDMA_Dispatch", platform="amd"), "inter_node_rdma")
        self.assertEqual(resolve_fabric_scope("InterNode_IB_Dispatch", platform="nv"), "inter_node_ib")
        self.assertEqual(resolve_fabric_scope("IntraNode_xGMI_Pull", platform="amd"), "intra_node_xgmi")
        self.assertEqual(resolve_fabric_scope("IntraNode_NVLink_Combine", platform="nv"), "intra_node_nvlink")
        self.assertEqual(resolve_fabric_scope("IntraNode_LocalStage", platform="amd"), "on_chip")
        self.assertEqual(resolve_fabric_scope("Compute_GEMM1_SwiGLU", platform="amd"), "on_chip")

        # 2. Label quality linter flags lazy 1-word generic labels, accepts descriptive labels
        warns = validate_stage_label_quality([
            {"id": "load"},
            {"id": "mma_qk_score", "label": "QK^T Score MMA"},
        ])
        self.assertEqual([w["probe_id"] for w in warns], ["load"])
        self.assertEqual(_humanize_stage_label("KV Tile Load (TMA)", "load"), "KV Tile Load (TMA)")
        self.assertEqual(_humanize_stage_label("produce_kv", "load"), "produce_kv (load)")
        self.assertEqual(_humanize_stage_label("load", "load"), "TMA / Tile Load")

        # 3. Rich `label` and `fabric` survive probe.json -> names.json -> fi_lite_decode -> viz.py
        spec = {
            "platform": "amd", "card": "mi308x", "arch": "gfx942",
            "roles": {"compute": "warps 0-3", "comm": "warps 4-7"},
            "probes": [
                {"id": "inter_node_rdma_dispatch", "label": "Inter-Node RDMA Dispatch", "fabric": "inter_node_rdma",
                 "begin": {"role": "comm", "line": 10}, "end": {"role": "comm", "line": 12}},
                {"id": "intra_node_xgmi_pull", "label": "Intra-Node xGMI Dispatch", "fabric": "intra_node_xgmi",
                 "begin": {"role": "comm", "line": 14}, "end": {"role": "comm", "line": 16}},
                {"id": "mma_gemm1_swiglu", "label": "GEMM1 (FC1+SwiGLU)",
                 "begin": {"role": "compute", "line": 20}, "end": {"role": "compute", "line": 25}},
            ],
        }
        names_doc = build_names_from_probe_spec(spec)
        self.assertIn("labels", names_doc)
        self.assertIn("fabrics", names_doc)
        self.assertIn("Inter-Node RDMA Dispatch", names_doc["labels"].values())
        self.assertIn("inter_node_rdma", names_doc["fabrics"].values())

        # 4. Synthetic 2-node AMD MI308X trace with 4 HW curves + 5th RDMA NIC curve renders cleanly
        synth = {
            "otherData": {"platform": "amd", "arch": "gfx942", "meta": {"card": "mi308x", "arch": "gfx942", "ep": 16, "rank": 0}},
            "traceEvents": [
                {"ph": "M", "pid": 100, "tid": 1, "name": "thread_name", "args": {"name": "CU00 - Wave 0 (W00) [Compute]"}},
                {"ph": "M", "pid": 100, "tid": 5, "name": "thread_name", "args": {"name": "CU00 - Wave 4 (W04) [Comm]"}},
                {"ph": "C", "pid": 1, "name": "compute_util [FP8 Matrix Core % of 606.2T]", "ts": 10.0, "args": {"value": 42.0}},
                {"ph": "C", "pid": 1, "name": "compute_util [FP8 Matrix Core % of 606.2T]", "ts": 90.0, "args": {"value": 45.0}},
                {"ph": "C", "pid": 1, "name": "memory_util [HBM3 Bandwidth % of 3684GB/s]", "ts": 10.0, "args": {"value": 24.0}},
                {"ph": "C", "pid": 1, "name": "memory_util [HBM3 Bandwidth % of 3684GB/s]", "ts": 90.0, "args": {"value": 26.0}},
                {"ph": "C", "pid": 1, "name": "communication_util [xGMI P2P % of 287.2GB/s]", "ts": 10.0, "args": {"value": 31.0}},
                {"ph": "C", "pid": 1, "name": "communication_util [xGMI P2P % of 287.2GB/s]", "ts": 90.0, "args": {"value": 15.0}},
                {"ph": "C", "pid": 1, "name": "rdma_util [Inter-Node RDMA NIC % of 100GB/s]", "ts": 10.0, "args": {"value": 18.0}},
                {"ph": "C", "pid": 1, "name": "rdma_util [Inter-Node RDMA NIC % of 100GB/s]", "ts": 90.0, "args": {"value": 22.0}},
                {"ph": "C", "pid": 1, "name": "sm_active [Active MI308X CUs % of 80]", "ts": 10.0, "args": {"value": 100.0}},
                {"ph": "C", "pid": 1, "name": "sm_active [Active MI308X CUs % of 80]", "ts": 90.0, "args": {"value": 100.0}},
                {"ph": "X", "pid": 100, "tid": 1, "ts": 5.0, "dur": 40.0, "name": "Compute_GEMM1_SwiGLU", "args": {"cu": 0, "wave": 0}},
                {"ph": "X", "pid": 100, "tid": 1, "ts": 45.0, "dur": 40.0, "name": "Compute_GEMM2_MFMA", "args": {"cu": 0, "wave": 0}},
                {"ph": "X", "pid": 100, "tid": 5, "ts": 5.0, "dur": 25.0, "name": "InterNode_RDMA_Dispatch", "args": {"cu": 0, "wave": 4}},
                {"ph": "X", "pid": 100, "tid": 5, "ts": 30.0, "dur": 25.0, "name": "IntraNode_xGMI_Pull", "args": {"cu": 0, "wave": 4}},
                {"ph": "X", "pid": 100, "tid": 5, "ts": 55.0, "dur": 20.0, "name": "InterNode_Partial_Reduce", "args": {"cu": 0, "wave": 4}},
                {"ph": "X", "pid": 100, "tid": 5, "ts": 75.0, "dur": 15.0, "name": "IntraNode_HomeCombine", "args": {"cu": 0, "wave": 4}},
            ],
        }
        parsed = _parse_trace(synth)
        self.assertTrue(parsed["is_amd"])
        self.assertTrue(parsed["multi_node"])
        self.assertTrue(parsed["dual_fabric"])
        self.assertIn("inter_rdma", parsed["hw_curves"])
        try:
            import matplotlib  # noqa: F401
            png = render_swhw_overview_png(synth, self.tmp / "mi308x_ep16_dual.png", dpi=110)
            self.assertTrue(png.is_file() and png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        except ImportError:
            pass

        # 5. NVLink TX (egress) and RX (ingress) MUST remain two independent full-duplex curves
        # (never merged/summed together), while HBM DRAM Read + Write share one bus and sum.
        nv_duplex = {
            "otherData": {"platform": "nv", "arch": "sm_90a", "meta": {"card": "h20", "arch": "sm_90a"}},
            "traceEvents": [
                {"ph": "C", "pid": 1, "name": "Tensor Active [Throughput %]", "ts": 10.0, "args": {"pct": 52.0}},
                {"ph": "C", "pid": 1, "name": "DRAM Read Bandwidth [Throughput %]", "ts": 10.0, "args": {"pct": 40.0}},
                {"ph": "C", "pid": 1, "name": "DRAM Write Bandwidth [Throughput %]", "ts": 10.0, "args": {"pct": 25.0}},
                {"ph": "C", "pid": 1, "name": "NVLink TX Requests User Data [Throughput %]", "ts": 10.0, "args": {"pct": 30.0}},
                {"ph": "C", "pid": 1, "name": "NVLink TX Responses User Data [Throughput %]", "ts": 10.0, "args": {"pct": 12.0}},
                {"ph": "C", "pid": 1, "name": "NVLink RX Requests User Data [Throughput %]", "ts": 10.0, "args": {"pct": 15.0}},
                {"ph": "C", "pid": 1, "name": "NVLink RX Responses User Data [Throughput %]", "ts": 10.0, "args": {"pct": 5.0}},
                {"ph": "C", "pid": 1, "name": "SMs Active [Throughput %]", "ts": 10.0, "args": {"pct": 95.0}},
                {"ph": "X", "pid": 100, "tid": 1, "ts": 0.0, "dur": 40.0, "name": "GEMM1 (FC1+SwiGLU)", "args": {"sm": 0, "warp": 0}},
            ],
        }
        p_nv = _parse_trace(nv_duplex)
        self.assertEqual(p_nv["hw_curves"]["dram_bw"], [(10.0, 65.0)])
        self.assertEqual(p_nv["hw_curves"]["nvlink_tx"], [(10.0, 42.0)])
        self.assertEqual(p_nv["hw_curves"]["nvlink_rx"], [(10.0, 20.0)])
        self.assertNotIn("nvlink_bw", p_nv["hw_curves"])
        try:
            import matplotlib  # noqa: F401
            png_nv = render_swhw_overview_png(nv_duplex, self.tmp / "h20_nvlink_duplex.png", dpi=110)
            self.assertTrue(png_nv.is_file() and png_nv.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        except ImportError:
            pass

    def test_b_empty_64ns_stub_probe_fails(self):
        rep = self._audit("stub", dump_kw={"stub": True})
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["stub"]["ok"])
        self.assertIn("epilogue", rep["fidelity_gates"]["stub"]["stubs"])
        self.assertTrue(rep["fidelity_gates"]["compiler_spill"]["ok"])
        seal = json.loads(Path(rep["seal_path"]).read_text(encoding="utf-8"))
        self.assertEqual(seal["status"], "FAIL")
        self.assertFalse(seal["checks"]["events"])

    def test_c_hidden_register_spill_fails(self):
        rm = runtime_metrics(inst_stats={"regs": 72, "spill_stores": 760, "spill_loads": 760, "stack_bytes": 760,
                                         "stl": 134, "ldl": 120})
        rep = self._audit("spill", rm=rm)
        self.assertEqual(rep["status"], "FAIL", rep)
        gate = rep["fidelity_gates"]["compiler_spill"]
        self.assertFalse(gate["ok"])
        self.assertTrue(gate["reg_blind_spot"])
        self.assertIn("SASS_STL_EXPLOSION", gate["codes"])
        self.assertTrue(rep["fidelity_gates"]["stub"]["ok"])

    def test_policy_and_evidence_fail_closed(self):
        ws = self._workspace()
        dump = write_dump(self.tmp / "l.wp02.bin")
        blocked = mk.run_megakernel_audit(ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
                                          runtime_metrics=runtime_metrics(), limit=0.25, out_dir=self.tmp / "o1")
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn("LIMIT_RELAXATION_BLOCKED", blocked["structural_blockers"][0])
        no_out = runtime_metrics()
        no_out.pop("output_match")
        rep = self._audit("noout", rm=no_out)
        self.assertEqual(rep["status"], "FAIL")
        self.assertEqual(rep["fidelity_gates"]["output"].get("code"), "OUTPUT_EVIDENCE_MISSING")
        slow = runtime_metrics(instrumented_ns=[b * 1.25 for b in runtime_metrics()["baseline_ns"]])
        self.assertEqual(self._audit("slow", rm=slow)["status"], "FAIL")
        fast_pert = runtime_metrics(instrumented_ns=[b * 0.55 for b in runtime_metrics()["baseline_ns"]])
        neg = self._audit("neg_pert", rm=fast_pert)
        self.assertEqual(neg["status"], "FAIL")
        self.assertFalse(neg["fidelity_gates"]["overhead"]["ok"])
        self.assertTrue(any("NEGATIVE_OVERHEAD_PERTURBATION" in b for b in neg["structural_blockers"]), neg)
        ws2 = self.tmp / "ws2"
        ws2.mkdir()
        (ws2 / "sparse_mla.cu").write_text(INSTRUMENTED_SOURCE, encoding="utf-8", newline="\n")
        dirty = mk.run_megakernel_audit(ws2, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
                                        runtime_metrics=runtime_metrics(), out_dir=self.tmp / "o2")
        self.assertEqual(dirty["status"], "FAIL")
        self.assertTrue(any("PRODUCTION_SOURCE_INSTRUMENTED" in b for b in dirty["structural_blockers"]))

    def test_workflow_a_shadow_build_run_with_l0_cache(self):
        ws = self._workspace()
        fixtures = self.tmp / "fixtures"
        fixtures.mkdir()
        write_dump(fixtures / "capture.wp02.bin")
        (fixtures / "runtime.json").write_text(json.dumps(runtime_metrics()), encoding="utf-8", newline="\n")
        (fixtures / "names.json").write_text(json.dumps(NAMES), encoding="utf-8", newline="\n")
        (ws / "bench.py").write_text(
            "import os, shutil, sys\n"
            "inst = open('sparse_mla.cu', encoding='utf-8').read()\n"
            "l0 = open(os.path.join(os.environ['WEAVEPROF_L0_DIR'], 'sparse_mla.cu'), encoding='utf-8').read()\n"
            "assert 'WP_OPEN' in inst and 'WP_OPEN' not in l0, 'shadow trees not isolated'\n"
            "assert os.path.isfile(os.path.join(os.environ['WEAVEPROF_L0_DIR'], 'built.txt')), 'L0 build missing'\n"
            "for f in ('capture.wp02.bin', 'runtime.json', 'names.json'):\n"
            "    shutil.copy(os.path.join(sys.argv[1], f), os.path.join(os.environ['WEAVEPROF_OUT_DIR'], f))\n",
            encoding="utf-8", newline="\n")
        build = [sys.executable, "-B", "-c", "open('built.txt', 'w').write('ok')"]
        sq = make_nsys_sqlite(fixtures / "capture.sqlite")
        kw = dict(entry_source="sparse_mla.cu", bench_cmd=[sys.executable, "-B", "bench.py", str(fixtures)],
                  build_cmd=build, inst_files={"sparse_mla.cu": INSTRUMENTED_SOURCE}, card="h20", nsys_sqlite=sq)
        # Fail-closed check: without nsys_sqlite, rendering overview PNG MUST block PASS
        no_hw_rep = mk.run_megakernel_audit(
            ws, out_dir=self.tmp / "wa_nohw",
            entry_source="sparse_mla.cu", bench_cmd=[sys.executable, "-B", "bench.py", str(fixtures)],
            build_cmd=build, inst_files={"sparse_mla.cu": INSTRUMENTED_SOURCE}, card="h20",
        )
        self.assertEqual(no_hw_rep["status"], "FAIL")
        self.assertTrue(any("MISSING_HARDWARE_UTILIZATION_CURVES" in b for b in no_hw_rep["structural_blockers"]), no_hw_rep)
        rep = mk.run_megakernel_audit(ws, out_dir=self.tmp / "wa1", **kw)
        self.assertEqual(rep["status"], "PASS", rep)
        self.assertEqual(rep["mode"], "shadow_build_run")
        self.assertTrue(rep["seal_verified"])
        self.assertEqual((ws / "sparse_mla.cu").read_text(encoding="utf-8"), CLEAN_SOURCE)
        fb1 = json.loads((self.tmp / "wa1" / "feedback.json").read_text(encoding="utf-8"))
        self.assertTrue(fb1["l0_cache"]["hit"])  # warmed by wa_nohw build above
        rep2 = mk.run_megakernel_audit(ws, out_dir=self.tmp / "wa2", **kw)
        self.assertEqual(rep2["status"], "PASS", rep2)
        fb2 = json.loads((self.tmp / "wa2" / "feedback.json").read_text(encoding="utf-8"))
        self.assertTrue(fb2["l0_cache"]["hit"])
        seal = json.loads(Path(rep["seal_path"]).read_text(encoding="utf-8"))
        self.assertEqual(seal["region_lock"]["label"], "megakernel_audit")
        self.assertIsNotNone(seal["artifacts"]["instrumented_source"])

    def test_async_engine_depth_pins_4panel_and_triton_ptx_pass(self):
        from weaveprof.pipeline.instrument.triton_ptx_async import build_triton_ptx_async_names, instrument_triton_ptx_async
        from weaveprof.viz import _compute_depth_segments, _parse_trace, _pick_micro_zoom_window, render_swhw_overview_png
        h20_names = build_triton_ptx_async_names({"wgmma_commits": 1})
        self.assertEqual(h20_names["async_engines"], {"1": "mem", "3": "mma"})
        self.assertIn("3", h20_names["slotted_events"])

        # 1. Exact sweep-line depth math (0->1->2->3->2->1->0)
        segs, avg_d, max_d = _compute_depth_segments([(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)])
        self.assertEqual(max_d, 3)
        self.assertGreater(avg_d, 1.5)
        self.assertEqual([(round(a, 1), round(b, 1), d) for a, b, d in segs],
                         [(0.0, 1.0, 1), (1.0, 2.0, 2), (2.0, 3.0, 3), (3.0, 4.0, 2), (4.0, 5.0, 1)])

        # 2. Triton PTX async pass handles cp.async + wait_group + mma.sync + wgmma + cursor rebinding
        fake_ptx = "\n".join([
            ".visible .entry k_test() {",
            "\t.reg .b32 %r<8>;",
            "\t.reg .b64 %rd<4>;",
            "\t// begin inline asm",
            "\tmov.u32 pid, %r1;",
            "\tmov.u32 n, %r2;",
            "\tsetp.lt.u32 p_bounds, warp, 4;",
            "\tsetp.lt.u32 p_cap, slot, 64;",
            "\tadd.u64 addr, %rd1, off;",
            "\tsetp.lt.u32 p_cap, n, 256;",
            "\t@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, 1073741824, sm};",
            "\t@p_store st.global.cs.v4.u32 [addr+16], {tlo1, thi1, 268435456, 0};",
            "\t@p_ovf atom.global.or.b32 ovfold, [%rd2], 1;",
            "\t// end inline asm",
            "\tcp.async.commit_group;",
            "\tcp.async.wait_group 0;",
            "\tbar.sync 0;",
            "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {}, {}, {}, {};",
            "\twgmma.commit_group.sync.aligned;",
            "\twgmma.wait_group.sync.aligned 0;",
            "\t// begin inline asm",
            "\tmov.u32 n, %r3;",
            "\t@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, 134217728, 0};",
            "\t@p_store st.global.cs.v4.u32 [addr+16], {tlo1, thi1, 268435456, 1};",
            "\t// end inline asm",
            "\tret;",
            "}",
        ])
        patched, st = instrument_triton_ptx_async(fake_ptx)
        self.assertEqual(st["patched"], 1)
        self.assertEqual(st["cp_async_commits"], 1)
        self.assertEqual(st["cp_async_waits"], 1)
        self.assertEqual(st["wgmma_commits"], 1)
        self.assertEqual(st["wgmma_waits"], 1)
        self.assertEqual(st["mma_clusters"], 1)
        self.assertIn("mov.u32 n, %wp_cur;", patched)
        self.assertIn("$L__WP_DRAIN_1:", patched)
        self.assertIn("$L__WP_WDRAIN_2:", patched)

        # 3. Deterministic 4-panel viz with HW curves + dual MEM & MMA async engines + micro-zoom
        evs = [
            {"ph": "C", "pid": 90000, "name": "Tensor Active [Throughput %]", "ts": 0.0, "args": {"value": 72.0}},
            {"ph": "C", "pid": 90000, "name": "Tensor Active [Throughput %]", "ts": 200.0, "args": {"value": 68.0}},
            {"ph": "C", "pid": 90000, "name": "DRAM Read Bandwidth [Throughput %]", "ts": 0.0, "args": {"value": 40.0}},
            {"ph": "C", "pid": 90000, "name": "DRAM Write Bandwidth [Throughput %]", "ts": 0.0, "args": {"value": 20.0}},
            {"ph": "C", "pid": 90000, "name": "DRAM Read Bandwidth [Throughput %]", "ts": 200.0, "args": {"value": 35.0}},
            {"ph": "C", "pid": 90000, "name": "DRAM Write Bandwidth [Throughput %]", "ts": 200.0, "args": {"value": 15.0}},
            {"ph": "X", "pid": 90000, "tid": 0, "ts": 0.0, "dur": 200.0, "name": "dual_async_gemm", "cat": "hardware", "args": {"kind": "dispatch"}},
        ]
        # 5 dense CTA waves on SM 0 (wave 2 has peak exposed wait -> deterministic winner!)
        for cta in range(5):
            base = cta * 40.0
            for k in range(20):
                t = base + k * 1.8
                evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t, "dur": 3.2, "name": "load",
                            "args": {"sm": 0, "warp": 0, "cta": cta, "role": "worker", "semantic_id": "load", "async_engine": "mem"}})
                evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t + 0.2, "dur": 2.4, "name": "mma",
                            "args": {"sm": 0, "warp": 0, "cta": cta, "role": "worker", "semantic_id": "mma", "async_engine": "mma"}})
                w_dur = 0.7 if cta == 2 else 0.1
                evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t + 0.1, "dur": w_dur, "name": "wait",
                            "args": {"sm": 0, "warp": 0, "cta": cta, "role": "worker", "semantic_id": "wait"}})
                evs.append({"ph": "X", "pid": 100000, "tid": 1, "ts": t + 0.1 + w_dur, "dur": 0.8, "name": "compute",
                            "args": {"sm": 0, "warp": 0, "cta": cta, "role": "worker", "semantic_id": "compute"}})
        synth_4p = {"otherData": {"platform": "nv", "arch": "sm_90", "meta": {"card": "h20", "arch": "sm_90"}}, "traceEvents": evs}
        try:
            import matplotlib  # noqa: F401
            out_png = render_swhw_overview_png(synth_4p, self.tmp / "dual_async_4panel.png", dpi=110)
            self.assertTrue(out_png.is_file() and out_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            # Fail-closed gate: rendering without same-run hardware curves MUST raise by default
            no_hw = {"otherData": synth_4p["otherData"], "traceEvents": [e for e in evs if e.get("ph") != "C"]}
            with self.assertRaises(ValueError) as cm:
                render_swhw_overview_png(no_hw, self.tmp / "should_fail_no_hw.png", dpi=110)
            self.assertIn("MISSING_HARDWARE_UTILIZATION_CURVES", str(cm.exception))
            # HBM Read+Write synthesized curve MUST NOT double-count raw DRAM Read + DRAM Write
            dup_evs = list(evs) + [
                {"ph": "C", "pid": 90000, "tid": 1, "ts": 10.0, "name": "DRAM Read Bandwidth [Throughput %]", "args": {"value": 5.0}},
                {"ph": "C", "pid": 90000, "tid": 1, "ts": 10.0, "name": "DRAM Write Bandwidth [Throughput %]", "args": {"value": 3.0}},
                {"ph": "C", "pid": 90000, "tid": 1, "ts": 10.0, "name": "Memory Util — HBM Read+Write (%) [NSYS PMU]", "args": {"value": 8.0}},
            ]
            parsed_dup = _parse_trace({"otherData": synth_4p["otherData"], "traceEvents": dup_evs})
            self.assertEqual(parsed_dup["hw_curves"]["dram_bw"][0][1], 8.0)
        except ImportError:
            pass


class HygieneTest(unittest.TestCase):
    OWNED = (
        "recorders/cuda_cpp/wp_lite.cuh",
        "recorders/hip_cpp/wp_lite_hip.h", "recorders/cutedsl/wp_cute.py",
        "recorders/triton/wp_triton.py", "recorders/triton/wp_triton_amd.py",
        "weaveprof/pipeline/validate/tools/wp_decode.py", "weaveprof/pipeline/validate/tools/fi_lite_decode.py", "weaveprof/megakernel.py",
        "weaveprof/viz.py", "weaveprof/fast_search.py", "tests/test_wp_megakernel.py",
    )

    def test_lf_only_and_bytecode_guard(self):
        for rel in self.OWNED:
            raw = (ROOT / rel).read_bytes()
            self.assertNotIn(b"\r\n", raw, rel)
            if rel.endswith(".py"):
                self.assertIn(b"sys.dont_write_bytecode = True", raw, rel)

    def test_viz_origin_shift_hw_pid_filter_sm86_and_cutedsl_order(self):
        # 1. viz.py normalizes absolute nsys timestamps (e.g. 3.4e6 us) to 0-based, drops pid=90000 & residency
        from weaveprof import viz
        aligned_trace = {
            "traceEvents": [
                {"ph": "M", "pid": 90000, "tid": 0, "name": "process_name", "args": {"name": "Hardware & Activity Curves"}},
                {"ph": "X", "pid": 90000, "tid": 0, "ts": 3408170.5, "dur": 192.5, "name": "gemm_k", "cat": "hardware", "args": {"kind": "dispatch"}},
                {"ph": "M", "pid": 100000, "tid": 0, "name": "process_name", "args": {"name": "SM 000"}},
                {"ph": "M", "pid": 100000, "tid": 1, "name": "thread_name", "args": {"name": "warp 00 · slot 0"}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 3408171.0, "dur": 190.0, "name": "Warp residency", "cat": "residency", "args": {"sm": 0, "warp": 0, "role": "math", "kind": "residency"}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 3408171.5, "dur": 188.0, "name": "gemm", "cat": "gemm", "args": {"sm": 0, "warp": 0, "semantic_id": "gemm", "role": "math"}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 3408172.0, "dur": 12.0, "name": "load", "cat": "load", "args": {"sm": 0, "warp": 0, "semantic_id": "load", "display_label": "A/B Tile Load", "role": "math"}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 3408184.0, "dur": 160.0, "name": "mma", "cat": "mma", "args": {"sm": 0, "warp": 0, "semantic_id": "mma", "display_label": "HMMA K-Tile", "role": "math"}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 3408344.0, "dur": 14.0, "name": "store", "cat": "store", "args": {"sm": 0, "warp": 0, "semantic_id": "store", "display_label": "C Epilogue Store", "role": "math"}},
            ],
            "otherData": {"meta": {"arch": "sm_86", "card": "sm86"}},
        }
        parsed = viz._parse_trace(aligned_trace)
        self.assertEqual(parsed["sms"], [0], f"ghost SM leaked: {parsed['sms']}")
        self.assertLess(parsed["t_max"], 250.0, f"timestamps not shifted from 3.4e6: {parsed['t_max']}")
        self.assertNotIn("gemm", parsed["stages_seen"], "outer gemm macro-envelope should be omitted when inner stages exist")
        self.assertEqual(set(parsed["stages_seen"]), {"load", "mma", "store"})

        # 2. sm86 card/arch routing & registry resolution
        from weaveprof.pipeline.scan import llm_probe_placer as placer
        from weaveprof.pipeline.core import registry
        from weaveprof.causal import get_arch_profile
        r1 = placer.resolve_route("sm86", None, "triton")
        r2 = placer.resolve_route("rtx3050", None, "cuda_cpp")
        self.assertEqual((r1["card"], r1["arch"], r1["default_timer"]), ("sm86", "sm_86", 1))
        self.assertEqual((r2["card"], r2["arch"]), ("sm86", "sm_86"))
        t = registry.resolve_target(card="sm86")
        self.assertEqual((t["card"], t["arch"], t["platform"]), ("sm86", "sm_86", "nv"))
        self.assertEqual(get_arch_profile("sm_86")["card"], "sm86")

        # 3. CuTeDSL @cute.jit defined BEFORE @cute.kernel is lowered properly
        from weaveprof.pipeline.instrument import rule_engine
        cute_src = (
            "import cutlass\nimport cutlass.cute as cute\n\n"
            "class Demo:\n"
            "    @cute.jit\n"
            "    def __call__(self, a, b, c):\n"
            "        self.kernel(a, b, c).launch(grid=[1,1,1], block=[128,1,1])\n\n"
            "    @cute.kernel\n"
            "    def kernel(self, a, b, c):\n"
            "        x = a[0]\n"
            "        y = x + b[0]\n"
            "        c[0] = y\n"
        )
        spec = {
            "language": "cutedsl", "card": "b300", "arch": "sm_103a", "operator_family": "gemm",
            "warp_roles": [{"role_id": 0, "name": "math", "predicate": "True"}],
            "probes": [
                {"id": "tile_load", "semantic_id": "load", "begin_line": 11, "end_line": 11, "role_id": 0},
                {"id": "tile_mma", "semantic_id": "mma", "begin_line": 12, "end_line": 12, "role_id": 0},
            ],
        }
        res = rule_engine.inject_probes(cute_src, spec)
        self.assertTrue(res["ok"], res.get("diagnostics"))
        self.assertIn("_fi_inst___call__(self, a, b, c, _fi_rec=None, _fi_ovf=None)", res["instrumented_source"])
        self.assertIn("self._fi_inst_kernel(a, b, c, _fi_rec=_fi_rec, _fi_ovf=_fi_ovf)", res["instrumented_source"])

    def test_cross_warp_tma_inflight_depth_earliest_consumer_dedup(self):
        # 1-to-4 Producer (W00, role=0) -> Consumers (W01..W04, role=1) with 3-stage overlapping TMA pipeline:
        #   slot 0 issued @ 1000 ns -> consumers arrive @ 4000, 4500, 5000, 5500 ns => min = 4000 ns (dur = 3000 ns)
        #   slot 1 issued @ 2000 ns -> consumers arrive @ 5000, 5800, 6000, 6200 ns => min = 5000 ns (dur = 3000 ns)
        #   slot 2 issued @ 3000 ns -> consumers arrive @ 6000, 6500, 7000, 7200 ns => min = 6000 ns (dur = 3000 ns)
        # Overlap over [1000..6000] (5000 ns):
        #   [1000,2000): depth 1 (1000 ns)
        #   [2000,3000): depth 2 (1000 ns)
        #   [3000,4000): depth 3 (1000 ns)  <-- peak depth = 3 (NOT 12 from 1-to-4 fanout!)
        #   [4000,5000): depth 2 (1000 ns)
        #   [5000,6000): depth 1 (1000 ns)
        # Exact time-weighted avg depth = (1+2+3+2+1)/5 = 1.800; residence_p50_ns = 3000 (unaffected by slow W04!).
        E_TMA = 11
        wpc = 5
        t0 = 1_000_000
        slots = {
            0: [(t0, OPEN | 0, 0), (t0 + 1000, E_TMA, 0), (t0 + 2000, E_TMA, 1), (t0 + 3000, E_TMA, 2), (t0 + 8000, CLOSE, 0)],
            1: [(t0, OPEN | 1, 0), (t0 + 4000, END | E_TMA, 0), (t0 + 5000, END | E_TMA, 1), (t0 + 6000, END | E_TMA, 2), (t0 + 8000, CLOSE, 0)],
            2: [(t0, OPEN | 1, 0), (t0 + 4500, END | E_TMA, 0), (t0 + 5800, END | E_TMA, 1), (t0 + 6500, END | E_TMA, 2), (t0 + 8000, CLOSE, 0)],
            3: [(t0, OPEN | 1, 0), (t0 + 5000, END | E_TMA, 0), (t0 + 6000, END | E_TMA, 1), (t0 + 7000, END | E_TMA, 2), (t0 + 8000, CLOSE, 0)],
            4: [(t0, OPEN | 1, 0), (t0 + 5500, END | E_TMA, 0), (t0 + 6200, END | E_TMA, 1), (t0 + 7200, END | E_TMA, 2), (t0 + 8000, CLOSE, 0)],
        }
        blob = WPD.build_dump_bytes(slots, n_slots=5, cap=16, wpc=wpc, meta={"arch": "sm_90a", "gran": 0, "kernel": "fa3_tma_test"})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tma_cw.wp02.bin"
            p.write_bytes(blob)
            d = WPD.read_dump(str(p))
        names = {
            "events": {str(E_TMA): "tma_kv_load"},
            "semantics": {str(E_TMA): "load"},
            "roles": {"0": "tma_producer", "1": "math_consumer"},
            "cross_warp": {str(E_TMA): {"id": "tma_kv_load", "begin_role": 0, "end_role": 1}},
            "group": "sm", "pair": "stack",
        }
        tr, summ = WPD.decode(d, names)
        self.assertEqual(summ["errors"], {})
        self.assertEqual(summ["cross_warp_spans"], 12)  # 3 transfers * 4 consumer warps
        cwi = summ["cross_warp_inflight"]
        self.assertEqual(cwi["ctas_measured"], 1)
        self.assertEqual(cwi["n_transfers"], 3)  # deduplicated from 12 consumer spans -> 3 true TMA transfers
        self.assertEqual(cwi["max_d_tma"], 3)
        self.assertAlmostEqual(cwi["avg_d_tma"], 1.8, places=3)
        self.assertEqual(cwi["residence_p50_ns"], 3000)
        self.assertEqual(cwi["residence_mean_ns"], 3000.0)
        self.assertIn("0", tr["otherData"]["cross_warp_inflight_ctas"])
        segs = tr["otherData"]["cross_warp_inflight_ctas"]["0"]["segments_us"]
        self.assertEqual([dp for _, _, dp in segs], [1, 2, 3, 2, 1])
        # Verify viz._parse_trace surfaces deduplicated CTA-level depth onto producer warp W00 (peak depth=3)
        from weaveprof import viz
        parsed_cw = viz._parse_trace(tr)
        prod_ivs = [(sp["ts"], sp["end"]) for sp in parsed_cw["sw_spans"] if sp["warp"] == 0 and sp["async_engine"] == "mem"]
        _, p_avg, p_max = viz._compute_depth_segments(prod_ivs)
        self.assertEqual(p_max, 3)
        self.assertAlmostEqual(p_avg, 1.8, places=2)
        # Also verify multi-event CTA-wide union (e.g. FA3 TMA K + TMA V concurrent in flight -> combined max_d=4, per_event max=2)
        E_K, E_V = 11, 12
        slots_kv = {
            0: [(t0, OPEN | 0, 0),
                (t0 + 1000, E_K, 0), (t0 + 1100, E_V, 0), (t0 + 2000, E_K, 1), (t0 + 2100, E_V, 1),
                (t0 + 8000, CLOSE, 0)],
            1: [(t0, OPEN | 1, 0),
                (t0 + 4000, END | E_K, 0), (t0 + 4500, END | E_V, 0), (t0 + 5000, END | E_K, 1), (t0 + 5500, END | E_V, 1),
                (t0 + 8000, CLOSE, 0)],
            2: [(t0, OPEN | 1, 0),
                (t0 + 4200, END | E_K, 0), (t0 + 4700, END | E_V, 0), (t0 + 5200, END | E_K, 1), (t0 + 5700, END | E_V, 1),
                (t0 + 8000, CLOSE, 0)],
        }
        blob_kv = WPD.build_dump_bytes(slots_kv, n_slots=3, cap=16, wpc=3, meta={"arch": "sm_90a", "gran": 0, "kernel": "fa3_kv"})
        with tempfile.TemporaryDirectory() as td:
            p2 = Path(td) / "kv.wp02.bin"
            p2.write_bytes(blob_kv)
            d2 = WPD.read_dump(str(p2))
        names_kv = {
            "events": {"11": "tma_k", "12": "tma_v"}, "semantics": {"11": "load", "12": "load"},
            "roles": {"0": "tma_producer", "1": "math_consumer"},
            "cross_warp": {
                "11": {"id": "tma_k", "begin_role": 0, "end_role": 1},
                "12": {"id": "tma_v", "begin_role": 0, "end_role": 1},
            },
            "group": "sm", "pair": "stack",
        }
        _, summ_kv = WPD.decode(d2, names_kv)
        self.assertEqual(summ_kv["errors"], {})
        self.assertEqual(summ_kv["cross_warp_inflight"]["max_d_tma"], 4)
        self.assertEqual(summ_kv["cross_warp_inflight"]["per_event"]["11"]["max_depth"], 2)
        self.assertEqual(summ_kv["cross_warp_inflight"]["per_event"]["12"]["max_depth"], 2)

    def test_fi_merge_perfetto_applies_h20_tensor_active_x4(self):
        import importlib.util
        from weaveprof import viz
        fmp_path = ROOT / "weaveprof/pipeline/validate/tools/fi_merge_perfetto.py"
        spec = importlib.util.spec_from_file_location("fi_merge_perfetto_t", fmp_path)
        fmp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fmp)
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            db = make_nsys_sqlite(td_p / "capture.sqlite")
            conn = sqlite3.connect(str(db))
            conn.executescript(
                "UPDATE GPU_METRICS SET value = 24 WHERE metricId = 2;"
                "CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT, chipName TEXT);"
                "INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H20', 'GH100');"
                "INSERT INTO TARGET_INFO_GPU VALUES (1, 'NVIDIA H100 80GB HBM3', 'GH100');"
            )
            conn.commit()
            conn.close()
            win = fmp.hardware_window(td_p, {"kernel_name": "sparse_mla", "capture_instance": -1})
            self.assertEqual(len(win), 4)
            self.assertEqual(getattr(win, "gpu_name", None), "NVIDIA H20")
            tr = {"traceEvents": [], "otherData": {}}
            summ = {"observation_start_ns": BASE_NS, "observation_end_ns": BASE_NS + 600_000, "first_open_ns": BASE_NS, "wpc": 4}
            al_h20 = fmp.align(tr, summ, win)
            self.assertTrue(al_h20["otherData"]["alignment"]["tensor_active_scale"]["applied"])
            self.assertEqual(al_h20["otherData"]["alignment"]["tensor_active_scale"]["scale"], 4.0)
            p_h20 = viz._parse_trace(al_h20)
            self.assertEqual(max(v for _, v in p_h20["hw_curves"]["tensor_active"]), 96.0)
            self.assertIn("x4", p_h20["hw_labels"]["tensor_active"])
            # Full GH100 (H100) must NOT be scaled even if card="h20" is passed
            al_h100 = fmp.align(tr, summ, tuple(win), runtime={"gpu_name": "NVIDIA H100 80GB HBM3", "card": "h20"})
            self.assertFalse(al_h100["otherData"]["alignment"]["tensor_active_scale"]["applied"])
            p_h100 = viz._parse_trace(al_h100)
            self.assertEqual(max(v for _, v in p_h100["hw_curves"]["tensor_active"]), 24.0)

    def test_mi308x_gfx942_profile_and_fullduplex_xgmi_viz(self):
        from weaveprof import causal, viz
        prof = causal.get_arch_profile("gfx942", "AMD Instinct MI308X")
        self.assertEqual(prof["card"], "mi308x")
        self.assertEqual(prof["tensor_active_scale"], 1.0)
        self.assertAlmostEqual(prof["nominal_tflops_fp8"], 398.7, places=1)
        self.assertAlmostEqual(prof["sustained_clock_ceiling_tflops_fp8"], 461.3, places=1)
        self.assertAlmostEqual(prof["empirical_tflops_fp8"], 443.4, places=1)
        self.assertAlmostEqual(prof["pipe_to_sku_fp8_rebase"], 1.0, places=3)
        self.assertAlmostEqual(prof["unthrottled_cdna3_tflops_fp8"], 606.2, places=1)
        self.assertAlmostEqual(prof["empirical_hbm_tbps"], 3.684, places=3)

        tr = {
            "traceEvents": [
                {"ph": "C", "pid": 900, "tid": 1, "ts": 100.0, "name": "compute_util [FP8 Matrix Core Pipe Active %]", "args": {"value": 62.2}},
                {"ph": "C", "pid": 900, "tid": 1, "ts": 200.0, "name": "compute_util [FP8 Matrix Core Pipe Active %]", "args": {"value": 58.1}},
                {"ph": "C", "pid": 900, "tid": 1, "ts": 100.0, "name": "communication_rx_util [xGMI RX (In) — % of 287.2GB/s]", "args": {"value": 16.5}},
                {"ph": "C", "pid": 900, "tid": 1, "ts": 200.0, "name": "communication_rx_util [xGMI RX (In) — % of 287.2GB/s]", "args": {"value": 0.0}},
                {"ph": "C", "pid": 900, "tid": 1, "ts": 100.0, "name": "communication_tx_util [xGMI TX (Out) — % of 287.2GB/s]", "args": {"value": 0.0}},
                {"ph": "C", "pid": 900, "tid": 1, "ts": 200.0, "name": "communication_tx_util [xGMI TX (Out) — % of 287.2GB/s]", "args": {"value": 76.4}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 100.0, "dur": 100.0, "name": "Inter-Node RDMA Dispatch", "args": {"sm": 0, "cta": 0, "warp": 0, "wave": 0, "fabric": "inter_node_rdma"}},
                {"ph": "X", "pid": 100000, "tid": 1, "ts": 200.0, "dur": 100.0, "name": "Intra-Node xGMI Combine", "args": {"sm": 0, "cta": 0, "warp": 0, "wave": 0, "fabric": "intra_node_xgmi"}},
            ],
            "otherData": {"meta": {"card": "mi308x", "arch": "gfx942", "world_size": 16}},
        }
        parsed = viz._parse_trace(tr)
        self.assertAlmostEqual(max(v for _, v in parsed["hw_curves"]["tensor_active"]), 62.2, places=1)
        tc_pretty, _ = viz._pretty_hw_curve_label(parsed["hw_labels"]["tensor_active"], "tensor_active", is_amd=True, multi_node=True)
        self.assertIn("Pipe Active %", tc_pretty)
        self.assertNotIn("398.7T", tc_pretty)
        tc_leg, _ = viz._pretty_hw_curve_label("compute_util [FP8 Matrix Core % of 606.2T]", "tensor_active", is_amd=True, multi_node=True)
        self.assertNotIn("606.2T", tc_leg)
        tc_leg398, _ = viz._pretty_hw_curve_label("compute_util [FP8 Matrix Core % of 398.7T]", "tensor_active", is_amd=True, multi_node=True)
        self.assertNotIn("398.7T", tc_leg398)
        # Keep xGMI RX (In) and xGMI TX (Out) as separate full-duplex curves!
        self.assertIn("nvlink_rx", parsed["hw_curves"])
        self.assertIn("nvlink_tx", parsed["hw_curves"])
        self.assertAlmostEqual(max(v for _, v in parsed["hw_curves"]["nvlink_rx"]), 16.5, places=1)
        self.assertAlmostEqual(max(v for _, v in parsed["hw_curves"]["nvlink_tx"]), 76.4, places=1)
        rx_pretty, _ = viz._pretty_hw_curve_label(parsed["hw_labels"]["nvlink_rx"], "nvlink_rx", is_amd=True, multi_node=True)
        tx_pretty, _ = viz._pretty_hw_curve_label(parsed["hw_labels"]["nvlink_tx"], "nvlink_tx", is_amd=True, multi_node=True)
        self.assertIn("Intra-Node xGMI RX (In)", rx_pretty)
        self.assertIn("Intra-Node xGMI TX (Out)", tx_pretty)


class NsysCaptureFailureAuditTest(_TmpCase):
    """Regression suite for M2 (M2a/M2b): requested nsys capture failure and post-PNG/seal disk JSON sync."""

    def setUp(self):
        super().setUp()
        import socket

        home = self.tmp / "home"
        (home / ".cache").mkdir(parents=True, exist_ok=True)
        (home / ".config").mkdir(parents=True, exist_ok=True)
        (home / "mpl").mkdir(parents=True, exist_ok=True)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self.fake_nsys_cfg = self.tmp / "fake_nsys_cfg.json"
        self.fake_nsys_log = self.tmp / "fake_nsys_calls.log"
        self.fake_nsys = bin_dir / "nsys"
        self.fake_nsys.write_text(
            f"#!{sys.executable} -B\n"
            "import json, os, pathlib, shutil, sys\n"
            "cfg = json.loads(pathlib.Path(os.environ['WEAVEPROF_TEST_NSYS_CFG']).read_text(encoding='utf-8'))\n"
            "sub = sys.argv[1] if len(sys.argv) > 1 else ''\n"
            "with open(cfg['log_path'], 'a', encoding='utf-8') as f:\n"
            "    f.write(sub + '\\n')\n"
            "if sub == 'profile':\n"
            "    if cfg.get('profile_write_rep', False):\n"
            "        i = sys.argv.index('-o')\n"
            "        pathlib.Path(sys.argv[i + 1] + '.nsys-rep').write_bytes(b'FAKE_NSYS_REP\\n')\n"
            "    sys.stdout.write(cfg.get('profile_msg', 'fake nsys profile\\n'))\n"
            "    sys.exit(int(cfg.get('profile_rc', 0)))\n"
            "if sub == 'export':\n"
            "    if cfg.get('export_write_sqlite', False):\n"
            "        i = sys.argv.index('--output')\n"
            "        shutil.copyfile(cfg['sqlite_template'], sys.argv[i + 1])\n"
            "    sys.stdout.write(cfg.get('export_msg', 'fake nsys export\\n'))\n"
            "    sys.exit(int(cfg.get('export_rc', 0)))\n"
            "sys.exit(2)\n",
            encoding="utf-8",
            newline="\n",
        )
        self.fake_nsys.chmod(0o755)

        env_overrides = {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "MPLCONFIGDIR": str(home / "mpl"),
            "MPLBACKEND": "Agg",
            "WEAVEPROF_TEST_NSYS_CFG": str(self.fake_nsys_cfg),
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        }
        saved = {k: os.environ.get(k) for k in env_overrides}

        def _restore_env():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore_env)
        os.environ.update(env_overrides)

        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex

        def _guard_addr(addr):
            host = addr[0] if isinstance(addr, tuple) and addr else ""
            if isinstance(host, bytes):
                host = host.decode("ascii", errors="ignore")
            if host and host not in ("127.0.0.1", "::1", "localhost"):
                raise AssertionError(f"unexpected external socket connection to {addr!r}")

        def _guarded_connect(sock, addr):
            _guard_addr(addr)
            return real_connect(sock, addr)

        def _guarded_connect_ex(sock, addr):
            _guard_addr(addr)
            return real_connect_ex(sock, addr)

        p1 = mock.patch.object(socket.socket, "connect", _guarded_connect)
        p2 = mock.patch.object(socket.socket, "connect_ex", _guarded_connect_ex)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

        self.ws = self.tmp / "ws"
        self.ws.mkdir()
        (self.ws / "sparse_mla.cu").write_text(CLEAN_SOURCE, encoding="utf-8", newline="\n")
        self.fixtures = self.tmp / "fixtures"
        self.fixtures.mkdir()
        self.dump_file = write_dump(self.fixtures / "capture.wp02.bin")
        (self.fixtures / "runtime.json").write_text(json.dumps(runtime_metrics()), encoding="utf-8", newline="\n")
        (self.fixtures / "names.json").write_text(json.dumps(NAMES), encoding="utf-8", newline="\n")
        (self.ws / "bench.py").write_text(
            "import os, shutil, sys\n"
            "out = os.environ['WEAVEPROF_OUT_DIR']\n"
            "for f in ('capture.wp02.bin', 'runtime.json', 'names.json'):\n"
            "    shutil.copy(os.path.join(sys.argv[1], f), os.path.join(out, f))\n",
            encoding="utf-8",
            newline="\n",
        )
        self.valid_sqlite = make_nsys_sqlite(self.fixtures / "valid.sqlite")
        self.no_metrics_sqlite = make_nsys_sqlite(self.fixtures / "no_metrics.sqlite")
        conn = sqlite3.connect(str(self.no_metrics_sqlite))
        conn.execute("DELETE FROM GPU_METRICS")
        conn.commit()
        conn.close()
        self.build_cmd = [sys.executable, "-B", "-c", "open('built.txt', 'w').write('ok')"]
        self.bench_cmd = [sys.executable, "-B", "bench.py", str(self.fixtures)]
        self.assertEqual(Path(mk.find_nsys_binary()).resolve(), self.fake_nsys.resolve())

    def _configure_nsys(self, *, profile_rc=0, profile_write_rep=True, export_rc=0,
                        export_write_sqlite=True, sqlite_template=None, msg="fake nsys\n"):
        if self.fake_nsys_log.exists():
            self.fake_nsys_log.unlink()
        cfg = {
            "log_path": str(self.fake_nsys_log),
            "profile_rc": profile_rc,
            "profile_write_rep": profile_write_rep,
            "profile_msg": msg,
            "export_rc": export_rc,
            "export_write_sqlite": export_write_sqlite,
            "export_msg": msg,
            "sqlite_template": str(sqlite_template or self.valid_sqlite),
        }
        self.fake_nsys_cfg.write_text(json.dumps(cfg), encoding="utf-8", newline="\n")

    def _nsys_calls(self):
        if not self.fake_nsys_log.is_file():
            return []
        return [ln for ln in self.fake_nsys_log.read_text(encoding="utf-8").splitlines() if ln]

    def _run_shadow(self, name: str, *, capture_nsys: bool, render_overview_png: bool, nsys_sqlite=None):
        out = self.tmp / f"out-{name}"
        rep = mk.run_megakernel_audit(
            self.ws,
            entry_source="sparse_mla.cu",
            bench_cmd=self.bench_cmd,
            build_cmd=self.build_cmd,
            inst_files={"sparse_mla.cu": INSTRUMENTED_SOURCE},
            card="h20",
            limit=0.10,
            capture_nsys=capture_nsys,
            nsys_sqlite=nsys_sqlite,
            render_overview_png=render_overview_png,
            out_dir=out,
        )
        for fname in ("verify.json", "feedback.json", "seal.json"):
            self.assertTrue((out / fname).is_file(), fname)
        verify_doc = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        feedback_doc = json.loads((out / "feedback.json").read_text(encoding="utf-8"))
        seal_doc = json.loads((out / "seal.json").read_text(encoding="utf-8"))
        return out, rep, verify_doc, feedback_doc, seal_doc

    def test_nsys_capture_failure_with_overview_png_enabled(self):
        # Case 1 (M2a / A6-profile): capture_nsys=True, render_overview_png=True, nsys profile fails.
        self._configure_nsys(profile_rc=1, profile_write_rep=False, msg="profile boom\n")
        out1, rep1, ver1, fb1, seal1 = self._run_shadow(
            "m2a-prof-fail", capture_nsys=True, render_overview_png=True
        )
        self.assertEqual(self._nsys_calls(), ["profile"])
        self.assertEqual(
            (ver1["status"], fb1["status"], rep1["status"], seal1["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys profile exited 1") for b in fb1["structural_blockers"]))
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys profile exited 1") for b in rep1["structural_blockers"]))
        self.assertTrue(any("MISSING_HARDWARE_UTILIZATION_CURVES" in b for b in fb1["structural_blockers"]))
        self.assertTrue(any("MISSING_HARDWARE_UTILIZATION_CURVES" in b for b in rep1["structural_blockers"]))
        self.assertFalse(any(b.startswith("MISSING_OVERVIEW_PNG") for b in fb1["structural_blockers"]))
        self.assertFalse(any(b.startswith("MISSING_OVERVIEW_PNG") for b in rep1["structural_blockers"]))
        self.assertEqual(seal1["failed"], ["structural"])
        self.assertEqual(ver1["failed"], [])
        self.assertTrue(rep1["seal_verified"])
        self.assertTrue(rep1["seal_integrity_only"])
        self.assertTrue(mk.verify_seal(out1 / "seal.json", require_pass=False)["ok"])
        self.assertFalse(mk.verify_seal(out1 / "seal.json", require_pass=True)["ok"])

        # Case 2 (M2a / A6-export): nsys profile writes .nsys-rep, nsys export fails without .sqlite.
        self._configure_nsys(profile_rc=0, profile_write_rep=True, export_rc=2, export_write_sqlite=False)
        out2, rep2, ver2, fb2, seal2 = self._run_shadow(
            "m2a-exp-fail", capture_nsys=True, render_overview_png=True
        )
        self.assertEqual(self._nsys_calls(), ["profile", "export"])
        self.assertEqual(
            (ver2["status"], fb2["status"], rep2["status"], seal2["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys export exited 2") for b in fb2["structural_blockers"]))
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys export exited 2") for b in rep2["structural_blockers"]))
        self.assertFalse(any(b.startswith("MISSING_OVERVIEW_PNG") for b in fb2["structural_blockers"]))
        self.assertTrue(mk.verify_seal(out2 / "seal.json", require_pass=False)["ok"])

    def test_nsys_capture_failure_with_overview_png_disabled(self):
        # Case 1 (M2b / A5-profile): capture_nsys=True, render_overview_png=False, nsys profile exits 1.
        self._configure_nsys(profile_rc=1, profile_write_rep=False, msg="profile boom\n")
        out1, rep1, ver1, fb1, seal1 = self._run_shadow(
            "m2b-prof-fail", capture_nsys=True, render_overview_png=False
        )
        self.assertEqual(self._nsys_calls(), ["profile"])
        self.assertEqual(
            (ver1["status"], fb1["status"], rep1["status"], seal1["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys profile exited 1") for b in fb1["structural_blockers"]))
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys profile exited 1") for b in rep1["structural_blockers"]))
        self.assertEqual(seal1["failed"], ["structural"])
        self.assertIsNone(seal1["artifacts"]["trace"])
        self.assertIsNotNone(seal1["artifacts"]["failed_trace"])
        self.assertTrue(rep1["seal_verified"])
        self.assertTrue(rep1["seal_integrity_only"])
        self.assertTrue(mk.verify_seal(out1 / "seal.json", require_pass=False)["ok"])
        self.assertFalse(mk.verify_seal(out1 / "seal.json", require_pass=True)["ok"])

        # Case 2 (M2b / A5-missing-rep): nsys profile exits 0 without producing .nsys-rep.
        self._configure_nsys(profile_rc=0, profile_write_rep=False)
        out2, rep2, ver2, fb2, seal2 = self._run_shadow(
            "m2b-missing-rep", capture_nsys=True, render_overview_png=False
        )
        self.assertEqual(self._nsys_calls(), ["profile"])
        self.assertEqual(
            (ver2["status"], fb2["status"], rep2["status"], seal2["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys profile exited 0") for b in fb2["structural_blockers"]))
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys profile exited 0") for b in rep2["structural_blockers"]))

        # Case 3 (M2b / A5-export): nsys profile succeeds, nsys export exits 2 without .sqlite.
        self._configure_nsys(profile_rc=0, profile_write_rep=True, export_rc=2, export_write_sqlite=False)
        out3, rep3, ver3, fb3, seal3 = self._run_shadow(
            "m2b-exp-fail", capture_nsys=True, render_overview_png=False
        )
        self.assertEqual(self._nsys_calls(), ["profile", "export"])
        self.assertEqual(
            (ver3["status"], fb3["status"], rep3["status"], seal3["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys export exited 2") for b in fb3["structural_blockers"]))
        self.assertTrue(any(b.startswith("NSYS_CAPTURE_FAILED: nsys export exited 2") for b in rep3["structural_blockers"]))
        self.assertTrue(mk.verify_seal(out3 / "seal.json", require_pass=False)["ok"])

    def test_nsys_capture_failure_control_capture_succeeds(self):
        # Row A1: capture_nsys=True, capture succeeds, render_overview_png=True -> 4x PASS.
        self._configure_nsys(profile_rc=0, profile_write_rep=True, export_rc=0,
                             export_write_sqlite=True, sqlite_template=self.valid_sqlite)
        out_a1, rep_a1, ver_a1, fb_a1, seal_a1 = self._run_shadow(
            "ctrl-a1", capture_nsys=True, render_overview_png=True
        )
        self.assertEqual(self._nsys_calls(), ["profile", "export"])
        self.assertEqual(
            (ver_a1["status"], fb_a1["status"], rep_a1["status"], seal_a1["status"]),
            ("PASS", "PASS", "PASS", "PASS"),
        )
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in fb_a1["structural_blockers"]))
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in rep_a1["structural_blockers"]))
        a1_pngs = list(out_a1.glob("*.png"))
        self.assertEqual(len(a1_pngs), 1, a1_pngs)
        self.assertRegex(a1_pngs[0].name, r"_\d{8}_\d{6}\.png$")
        self.assertTrue(rep_a1["seal_verified"])
        self.assertNotIn("seal_integrity_only", rep_a1)
        self.assertTrue(mk.verify_seal(out_a1 / "seal.json", require_pass=True)["ok"])

        # Row A2: capture_nsys=True, capture succeeds, render_overview_png=False -> 4x PASS.
        self._configure_nsys(profile_rc=0, profile_write_rep=True, export_rc=0,
                             export_write_sqlite=True, sqlite_template=self.valid_sqlite)
        out_a2, rep_a2, ver_a2, fb_a2, seal_a2 = self._run_shadow(
            "ctrl-a2", capture_nsys=True, render_overview_png=False
        )
        self.assertEqual(self._nsys_calls(), ["profile", "export"])
        self.assertEqual(
            (ver_a2["status"], fb_a2["status"], rep_a2["status"], seal_a2["status"]),
            ("PASS", "PASS", "PASS", "PASS"),
        )
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in fb_a2["structural_blockers"]))
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in rep_a2["structural_blockers"]))
        self.assertEqual(list(out_a2.glob("*.png")), [])
        self.assertTrue(mk.verify_seal(out_a2 / "seal.json", require_pass=True)["ok"])

        # Row A3: capture_nsys=True, capture succeeds (valid kernel window, 0 GPU_METRICS rows),
        # render_overview_png=True -> PNG fails, 4x FAIL, disk JSONs match return.
        self._configure_nsys(profile_rc=0, profile_write_rep=True, export_rc=0,
                             export_write_sqlite=True, sqlite_template=self.no_metrics_sqlite)
        out_a3, rep_a3, ver_a3, fb_a3, seal_a3 = self._run_shadow(
            "ctrl-a3", capture_nsys=True, render_overview_png=True
        )
        self.assertEqual(self._nsys_calls(), ["profile", "export"])
        self.assertEqual(
            (ver_a3["status"], fb_a3["status"], rep_a3["status"], seal_a3["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in fb_a3["structural_blockers"]))
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in rep_a3["structural_blockers"]))
        self.assertTrue(any("OVERVIEW_PNG_FAILED" in b for b in fb_a3["structural_blockers"]))
        self.assertTrue(any("OVERVIEW_PNG_FAILED" in b for b in rep_a3["structural_blockers"]))
        self.assertTrue(any(b.startswith("MISSING_OVERVIEW_PNG") for b in fb_a3["structural_blockers"]))
        self.assertTrue(any(b.startswith("MISSING_OVERVIEW_PNG") for b in rep_a3["structural_blockers"]))
        self.assertTrue(mk.verify_seal(out_a3 / "seal.json", require_pass=False)["ok"])

    def test_nsys_capture_failure_not_requested_and_png_failure_consistency(self):
        # Keep fake nsys poisoned (rc=99) on PATH to prove capture_nsys=False never invokes nsys.
        self._configure_nsys(profile_rc=99, profile_write_rep=False, export_rc=99, export_write_sqlite=False)

        # Row B1: capture_nsys=False with pre-existing nsys_sqlite, render_overview_png=True -> 4x PASS.
        out_b1, rep_b1, ver_b1, fb_b1, seal_b1 = self._run_shadow(
            "b1-png-ok", capture_nsys=False, render_overview_png=True, nsys_sqlite=self.valid_sqlite
        )
        self.assertEqual(self._nsys_calls(), [])
        self.assertEqual(
            (ver_b1["status"], fb_b1["status"], rep_b1["status"], seal_b1["status"]),
            ("PASS", "PASS", "PASS", "PASS"),
        )
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in fb_b1["structural_blockers"]))
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in rep_b1["structural_blockers"]))
        b1_pngs = list(out_b1.glob("*.png"))
        self.assertEqual(len(b1_pngs), 1, b1_pngs)
        self.assertRegex(b1_pngs[0].name, r"_\d{8}_\d{6}\.png$")
        self.assertTrue(mk.verify_seal(out_b1 / "seal.json", require_pass=True)["ok"])

        # Row B2: capture_nsys=False, nsys_sqlite=None, render_overview_png=False -> 4x PASS.
        out_b2, rep_b2, ver_b2, fb_b2, seal_b2 = self._run_shadow(
            "b2-png-off", capture_nsys=False, render_overview_png=False, nsys_sqlite=None
        )
        self.assertEqual(self._nsys_calls(), [])
        self.assertEqual(
            (ver_b2["status"], fb_b2["status"], rep_b2["status"], seal_b2["status"]),
            ("PASS", "PASS", "PASS", "PASS"),
        )
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in fb_b2["structural_blockers"]))
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in rep_b2["structural_blockers"]))
        self.assertTrue(mk.verify_seal(out_b2 / "seal.json", require_pass=True)["ok"])

        # Row B3: capture_nsys=False, nsys_sqlite=None, render_overview_png=True -> PNG fails, 4x FAIL.
        out_b3, rep_b3, ver_b3, fb_b3, seal_b3 = self._run_shadow(
            "b3-png-fail", capture_nsys=False, render_overview_png=True, nsys_sqlite=None
        )
        self.assertEqual(self._nsys_calls(), [])
        self.assertEqual(
            (ver_b3["status"], fb_b3["status"], rep_b3["status"], seal_b3["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in fb_b3["structural_blockers"]))
        self.assertFalse(any("NSYS_CAPTURE_FAILED" in b for b in rep_b3["structural_blockers"]))
        self.assertTrue(any("OVERVIEW_PNG_FAILED" in b for b in fb_b3["structural_blockers"]))
        self.assertTrue(any("OVERVIEW_PNG_FAILED" in b for b in rep_b3["structural_blockers"]))
        self.assertTrue(any(b.startswith("MISSING_OVERVIEW_PNG") for b in fb_b3["structural_blockers"]))
        self.assertTrue(any(b.startswith("MISSING_OVERVIEW_PNG") for b in rep_b3["structural_blockers"]))
        self.assertEqual(seal_b3["failed"], ["structural"])
        self.assertTrue(mk.verify_seal(out_b3 / "seal.json", require_pass=False)["ok"])

        # Precaptured (Workflow B) PNG failure also syncs verify.json and feedback.json to FAIL.
        out_pre = self.tmp / "out-precaptured-png-fail"
        rep_pre = mk.run_megakernel_audit(
            self.ws,
            entry_source="sparse_mla.cu",
            dump_path=self.dump_file,
            names=NAMES,
            runtime_metrics=runtime_metrics(),
            nsys_sqlite=None,
            card="h20",
            limit=0.10,
            render_overview_png=True,
            out_dir=out_pre,
        )
        ver_pre = json.loads((out_pre / "verify.json").read_text(encoding="utf-8"))
        fb_pre = json.loads((out_pre / "feedback.json").read_text(encoding="utf-8"))
        seal_pre = json.loads((out_pre / "seal.json").read_text(encoding="utf-8"))
        self.assertEqual(self._nsys_calls(), [])
        self.assertEqual(
            (ver_pre["status"], fb_pre["status"], rep_pre["status"], seal_pre["status"]),
            ("FAIL", "FAIL", "FAIL", "FAIL"),
        )
        self.assertTrue(any("OVERVIEW_PNG_FAILED" in b for b in fb_pre["structural_blockers"]))
        self.assertTrue(any(b.startswith("MISSING_OVERVIEW_PNG") for b in fb_pre["structural_blockers"]))
        self.assertTrue(mk.verify_seal(out_pre / "seal.json", require_pass=False)["ok"])

    def test_nsys_capture_failure_seal_verify_failed_rewrites_disk_jsons(self):
        # Sole exception E1: seal_round writes a PASS seal, then artifact corruption makes real verify_seal fail;
        # feedback.json and verify.json are rewritten to FAIL (with SEAL_VERIFY_FAILED), while seal.json stays PASS
        # and fails real verify_seal under both require_pass=True and require_pass=False.
        real_seal_round = mk.EpisodeSealer.seal_round

        def _tampering_seal_round(**kwargs):
            seal_path, seal_doc = real_seal_round(**kwargs)
            names_on_disk = Path(kwargs["names_path"])
            names_on_disk.write_bytes(names_on_disk.read_bytes() + b"\n")
            return seal_path, seal_doc

        out_e1 = self.tmp / "out-seal-verify-fail"
        with mock.patch.object(mk.EpisodeSealer, "seal_round", side_effect=_tampering_seal_round):
            rep_e1 = mk.run_megakernel_audit(
                self.ws,
                entry_source="sparse_mla.cu",
                dump_path=self.dump_file,
                names=NAMES,
                runtime_metrics=runtime_metrics(),
                nsys_sqlite=self.valid_sqlite,
                card="h20",
                limit=0.10,
                render_overview_png=False,
                out_dir=out_e1,
            )
        ver_e1 = json.loads((out_e1 / "verify.json").read_text(encoding="utf-8"))
        fb_e1 = json.loads((out_e1 / "feedback.json").read_text(encoding="utf-8"))
        seal_e1 = json.loads((out_e1 / "seal.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (ver_e1["status"], fb_e1["status"], rep_e1["status"], seal_e1["status"]),
            ("FAIL", "FAIL", "FAIL", "PASS"),
        )
        self.assertFalse(rep_e1["seal_verified"])
        self.assertTrue(any(b.startswith("SEAL_VERIFY_FAILED:") for b in fb_e1["structural_blockers"]))
        self.assertTrue(any(b.startswith("SEAL_VERIFY_FAILED:") for b in rep_e1["structural_blockers"]))
        self.assertFalse(mk.verify_seal(out_e1 / "seal.json", require_pass=True)["ok"])
        self.assertFalse(mk.verify_seal(out_e1 / "seal.json", require_pass=False)["ok"])


class TimelinePngNamingTests(unittest.TestCase):
    """Verify descriptive `<model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png` naming across entry points."""

    def test_explicit_model_op_bs_seql_and_timestamp(self):
        from weaveprof.viz import build_timeline_png_name

        name = build_timeline_png_name(
            model="Qwen3.5-397B",
            op="attn",
            bs=5,
            seql=4096,
            kv=2048,
            timestamp="20261126_123000",
        )
        self.assertEqual(name, "qwen3.5-397b_attn_bs5_seql4096_kv2048_20261126_123000.png")

    def test_gemm_extracted_from_trace_meta_and_title(self):
        from weaveprof.viz import build_timeline_png_name

        trace = {
            "otherData": {
                "meta": {"model": "DeepSeek-V3", "kernel": "triton_gemm_s5", "M": 4096, "N": 4096, "K": 2048},
            }
        }
        name = build_timeline_png_name(trace, timestamp="20261126_123001")
        self.assertEqual(name, "deepseek-v3_gemm_m4096_n4096_k2048_20261126_123001.png")

    def test_attention_parsed_from_title_when_meta_sparse(self):
        from weaveprof.viz import build_timeline_png_name

        name = build_timeline_png_name(
            title="Qwen3.5-397B FlashAttention-3 B=5, H=16, S=4096, D=128",
            timestamp="20261126_123002",
        )
        self.assertEqual(name, "qwen3.5-397b_attn_bs5_seql4096_h16_d128_20261126_123002.png")

    def test_resolve_path_handles_none_dir_and_explicit_file(self):
        from weaveprof.viz import resolve_timeline_png_path

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            p_auto = resolve_timeline_png_path(
                None, default_dir=d, model="Llama3-70B", op="gemm", shape="m1024_n1024_k512",
                timestamp="20261126_123003",
            )
            self.assertEqual(p_auto, d / "llama3-70b_gemm_m1024_n1024_k512_20261126_123003.png")
            p_dir = resolve_timeline_png_path(
                d, model="Llama3-70B", op="attn", bs=2, seql=1024, timestamp="20261126_123004",
            )
            self.assertEqual(p_dir, d / "llama3-70b_attn_bs2_seql1024_20261126_123004.png")
            p_explicit = resolve_timeline_png_path(d / "custom_chart.png", model="X", op="gemm")
            self.assertEqual(p_explicit, d / "custom_chart.png")


if __name__ == "__main__":
    unittest.main()
