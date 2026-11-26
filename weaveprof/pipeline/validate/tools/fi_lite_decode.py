# -*- coding: utf-8 -*-
"""weaveprofiler 解码器：FILITE02 转储 + canonical 事件语义 -> Perfetto trace + 摘要。

NVIDIA（sm_90 / sm_100 / sm_103）与 AMD（gfx942 等 CDNA）的转储都用这一个解码器：格式同为 FILITE02，
16 字节一条记录 (t, tag, aux)，差别只在 meta JSON 与 aux 的含义（下面写明），二进制布局一致。

用法：
  python3 -I fi_lite_decode.py <dump.bin> --names names.json -o out.perfetto.json[.gz] [--summary s.json]
                              [--allow-errors] [--gran auto|warp|sm|cu] [--group auto|role|sm|cu|cta]
                              [--no-sm-bar] [--timer-hz HZ]
  python3 -I fi_lite_decode.py --selftest

计时器（写在转储头的 timer 字段与 meta 里）：
  NVIDIA FI_TIMER=0：%globaltimer，记录值已经是 ns。
  NVIDIA FI_TIMER=1：clock64，每泳道 OPEN/CLOSE 各记一对 (%globaltimer, clock64) 锚点，解码线性换算；
                     只有关锚点时用 meta.clock_mhz。
  AMD    FI_TIMER=0：s_memrealtime（固定频率墙钟，全卡同一时基），记录值是 tick，按 meta.timer_hz 换算成 ns
                     （大整数运算，不丢精度）。MI308X 实测 99.9969 MHz，即 1 tick = 10.0003 ns。
                     meta 里没写 timer_hz 时用 --timer-hz 补，否则解码器直接报错而不是猜。
  AMD    FI_TIMER=1：readcyclecounter（s_memtime，CU 本地周期计数器），锚点仍是 s_memrealtime，
                     换算方式同 NVIDIA 的 clock64。MI308X 实测 s_memtime 1847.0942 MHz。

记录粒度（FI_GRAN，写在 meta JSON 的 gran 字段里；老转储没有该字段＝warp/wave 粒度）：
  gran=0 warp/wave 粒度：一条泳道＝一个 warp（NVIDIA）或一个 wave64（AMD，lane 0 记录），
      归组照 names.json 的 group（role/sm/cta；AMD 上 sm 等价于 cu）。
  gran=1 SM/CU 粒度：一条泳道＝一个 CTA（只有 0 号线程记录），解码器按物理单元归组：
      NVIDIA 默认 group=sm_resident：每个 SM 一个进程（一条时间条），同一 SM 上时间重叠的多个 CTA
      自动拆成并列子条（贪心着色，子条数＝该 SM 的最大并发驻留 CTA 数），先后驻留的 CTA 共用同一条；
      另加一条 cat=info 的“SM 占用”整条，一眼看出各 SM 的负载不均。--no-sm-bar 可去掉它。
      AMD 默认 group=cu：每个物理 CU 一个进程，同一 CU 上时间重叠的区间自动分条带。
      --group role|sm|cu|cta 可强制换归组。
  OPEN 记录的 aux 一律是物理位置，**与 gran 无关**（gran=0 也照记，只是解码器不按它归组；
      想把 warp/wave 泳道映射回物理 SM/CU，唯一来源就是这个 aux）：
      NVIDIA＝%smid（SASS 里 S2R SR_VIRTUALSMID，取值范围 0..SM 数-1，不是 0 也不是角色号）；
      AMD＝HW_ID/XCC_ID 编码（bits[3:0]=CU, [5:4]=SIMD, [9:8]=SE, [15:12]=XCC，
      物理 CU 号 = (XCC<<6)|(SE<<4)|CU）。写校验时别假设 aux 从 0 开始递增（fi_bugs_b3-sk-qwen B20）。

names.json 格式（events 是源码标签，semantics 是权威 canonical ID）：
  {"events": {"1": "等 conv 就绪", "2": "GDN 块计算"},
   "semantics": {"1": "wait", "2": "compute"},       # 推荐且交付时必需；ID 只来自 configs/semantic_ids.json
   # 也可写 {"semantic_id": "mma", "detail": "QK"}; display_name 不允许覆盖
   "roles":  {"0": "GDN 链", "1": "辅助 SM"},
   "aux":    {"1": "chunk", "2": "chunk"},          # 可选：aux 的含义，会进 args 和标签
   "label_aux": true,                               # 可选：标签后缀 "#aux"
   "group": "role",                                 # role=按角色分进程（默认），sm/cu=按物理 SM/CU，cta=每个 CTA 一个进程
   "pair": "stack",                                 # 可选：区间配对方式，见下
   "instances": {"3": 5}}                           # 可选：事件 3 有 5 个并发实例，实例号 = aux % 5

区间配对：
  pair=stack（默认）：同一泳道内区间必须严格嵌套，E 配最内层 B（同一 warp 顺序执行的情形）。
  pair=aux：E 配同泳道里最近一个 (事件号, aux) 相同的未配对 B。用于同一泳道同时在途多个实例
            （例如 5 个 prep 块的 TMA 同时发出，各自在 mbarrier 等待返回处记 E）。配好的区间若互相重叠，
            会自动拆到子泳道："CTA003 w00 · 实例2"，子泳道号 = instances 里给的 aux % n，没给就按区间着色贪心分配。
  两种模式下，结构错误（多余 E、未配对 B、负时长、交叉嵌套）都计数，默认有错不写文件。

trace 的 args 里 sm 是物理单元号（NVIDIA＝SM，AMD＝CU）；AMD 另给 xcc/se/cu/simd 四个字段。
退出码：0=干净；2=有结构错误（溢出/未配对/交叉嵌套/负时长），加 --allow-errors 仍写文件但退出码 2。
"""
import argparse, gzip, hashlib, json, os, statistics, struct, sys
sys.dont_write_bytecode = True
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    from weaveprof.pipeline.core.semantics import SemanticKBError, load_kb, resolve_names
except ImportError as _semantic_import_error:  # pragma: no cover - installation error
    raise SystemExit(f"cannot import canonical semantic resolver from {_ROOT}: {_semantic_import_error}")
try:
    import msgspec
except ImportError:  # stdlib fallback keeps the skill portable
    msgspec = None

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK，中文输出不崩
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

END, OPEN, MARK, CLK, CLOSE, ROLE_SWITCH, IDM = (
    0x80000000,
    0x40000000,
    0x20000000,
    0x10000000,
    0x08000000,
    0x04000000,
    0x0FFF,
)
INIT = OPEN


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks_to_ns(t, hz):
    """HIP s_memrealtime tick -> 整数 ns。大整数运算，避免 1e9 乘法在 float64 丢精度。"""
    return (int(t) * 1_000_000_000 + int(hz) // 2) // int(hz)


def _is_amd(meta):
    """AMD 转储判定，按证据强弱依次看：

    1) meta.backend 写明 nvidia_ptx / ptx -> NVIDIA（wp_triton.py 会写；它同时也会写 timer_hz=1e9，
       所以“有 timer_hz 就是 AMD”这条老规则在 NVIDIA Triton 转储上是错的，fi_bugs_b3-sk-qwen B17）；
    2) meta.arch 以 gfx 开头（HIP 头文件写入，如 "gfx942"）-> AMD；
    3) meta.arch 是 sm_* -> NVIDIA；
    4) 老转储既没 backend 也没 arch：此时若 meta 里有 timer_hz 按 AMD 处理（只有 HIP 头文件与
       AMD Triton 前端需要写换算频率）；
    拿不准时用 --platform nvidia|amd 显式指定。"""
    backend = str(meta.get("backend", "")).lower()
    if backend.startswith("nvidia") or backend == "ptx":
        return False
    if backend.startswith("amd") or backend in ("hip", "amdgcn"):
        return True
    arch = str(meta.get("arch", "")).lower()
    if arch.startswith("gfx"):
        return True
    if arch.startswith("sm_") or arch.startswith("sm"):
        return False
    if not arch and meta.get("timer_hz"):
        return True
    return False


def _cu_of(aux):
    """gfx942 HW_ID 编码 -> (物理CU号, xcc, se, cu, simd)。CU 号 = (XCC<<6)|(SE<<4)|CU。"""
    cu, simd, se, xcc = aux & 15, (aux >> 4) & 3, (aux >> 8) & 3, (aux >> 12) & 15
    return (xcc << 6) | (se << 4) | cu, xcc, se, cu, simd


_REC_STRUCT = struct.Struct("<QII")


def read_dump(path, timer_hz=None, platform=None):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] not in (b"WPLITE02", b"FILITE02"):
        raise SystemExit(f"{path}: 不是 weaveprofiler 转储（magic 应为 WPLITE02 或 FILITE02）")
    n_slots, cap, wpc, ovf, timer, ml = struct.unpack_from("<6I", buf, 8)
    off = 32
    meta = json.loads(buf[off:off + ml].decode("utf-8")) if ml else {}
    off += ml
    cnt = list(struct.unpack_from(f"<{n_slots}I", buf, off))
    off += 4 * n_slots
    slots = {}
    mv = memoryview(buf)
    unpack_iter = _REC_STRUCT.iter_unpack
    for s in range(n_slots):
        n = min(cnt[s], cap)
        if n:
            end_off = off + 16 * n
            if end_off > len(buf):
                raise SystemExit(f"{path}: 文件长度与计数不符（{end_off} vs {len(buf)}），转储损坏")
            slots[s] = list(unpack_iter(mv[off:end_off]))
            off = end_off
    if off != len(buf):
        raise SystemExit(f"{path}: 文件长度与计数不符（{off} vs {len(buf)}），转储损坏")
    d = dict(n_slots=n_slots, cap=cap, wpc=wpc, overflow=ovf, timer=timer, meta=meta, cnt=cnt, slots=slots,
             magic=buf[:8].decode("ascii"))
    # 平台先判，再决定要不要按 timer_hz 换算：NVIDIA 的 %globaltimer 记录值本来就是 ns，
    # 拿 timer_hz 去换算只有"恰好写 1e9"才不出错（fi_bugs_b3-sk-qwen B17）。
    if platform in ("nvidia", "amd"):
        d["amd"] = platform == "amd"
        d["platform_source"] = "--platform"
    else:
        d["amd"] = _is_amd(meta)
        d["platform_source"] = ("meta.backend" if meta.get("backend") else
                                ("meta.arch" if meta.get("arch") else
                                 ("meta.timer_hz" if meta.get("timer_hz") else "默认 nvidia")))
    # HIP：meta.timer_hz 是 s_memrealtime 的固定频率；FI_TIMER=0 时记录值是 tick，统一换算成 ns。
    # 主机侧没写 timer_hz 时可用 --timer-hz 补（会记明来源，不静默猜频率）。
    if timer_hz:
        meta["timer_hz"] = int(timer_hz)
        meta["timer_hz_source"] = "cli --timer-hz"
    if timer == 0 and meta.get("timer_hz") and not d["amd"]:
        # NVIDIA 转储里出现 timer_hz：只有 wp_triton（NVIDIA 后端）会写，且固定 1e9（tick 就是 ns）。
        # 不当换算系数用，如实记一条说明；万一哪天不是 1e9，也要让人一眼看出来。
        hz = int(meta["timer_hz"])
        meta["timer_hz_ignored"] = ("NVIDIA 转储（platform=nvidia，backend=%s）：%%globaltimer 记录值已是 ns，"
                                    "timer_hz=%d 不当换算系数用" % (meta.get("backend"), hz)
                                    if hz == 1_000_000_000 else
                                    "NVIDIA 转储里 timer_hz=%d 不是 1e9：%%globaltimer 本该是 ns，"
                                    "这个系数被忽略，请核对前端 meta 写法" % hz)
    elif timer == 0 and meta.get("timer_hz"):
        hz = int(meta["timer_hz"])
        if hz <= 0:
            raise SystemExit(f"{path}: meta.timer_hz 必须为正数")
        d["slots"] = {sl: [(_ticks_to_ns(t, hz), tag, aux) for t, tag, aux in rs] for sl, rs in d["slots"].items()}
    d["timer_hz"] = int(meta.get("timer_hz", 0) or 0)
    if not d["amd"]:
        d["timer_hz_applied"] = False
    d["clock_fit"] = to_globaltimer(d) if timer == 1 else {}
    # 粒度只允许写在 meta JSON 里（转储格式固定 FILITE02，不能加头字段）。老转储没有 gran 就是 warp 粒度。
    try:
        g = int(meta.get("gran", 0))
    except (TypeError, ValueError):
        g = 0
    d["gran"] = 1 if g == 1 else 0
    return d


def to_globaltimer(d):
    """FI_TIMER=1：每条泳道用 OPEN/CLOSE 两个 (globaltimer, clock64) 锚点把 clock64 线性换算成 globaltimer ns。
    只有开锚点时用 meta.clock_mhz（没有就报错）。返回每泳道的有效频率，供摘要报告。"""
    fit = {}
    hz = int(d.get("timer_hz", 0) or 0)
    to_ns = (lambda t: _ticks_to_ns(t, hz)) if hz > 0 else (lambda t: int(t))
    for s, rs in d["slots"].items():
        anc = {}
        for j, (t, tag, aux) in enumerate(rs):
            if tag & CLK and j > 0:
                anc[aux] = (rs[j - 1][0], t)  # (墙钟, 周期计数器)：NVIDIA=(%globaltimer, clock64)，AMD=(s_memrealtime, s_memtime)
        if 0 not in anc:
            raise SystemExit(f"slot {s}: FI_TIMER=1 但缺开锚点")
        g0, k0 = to_ns(anc[0][0]), anc[0][1]
        if 1 in anc and anc[1][1] > k0 and to_ns(anc[1][0]) > g0:
            ns_per_clk = (to_ns(anc[1][0]) - g0) / (anc[1][1] - k0)
        elif d["meta"].get("clock_mhz"):
            ns_per_clk = 1000.0 / float(d["meta"]["clock_mhz"])
        else:
            raise SystemExit(f"slot {s}: 缺关锚点（早退路径没调 FI_CLOSE？）且 meta 没给 clock_mhz/timer_hz")
        fit[s] = round(1000.0 / ns_per_clk, 1)  # MHz
        out = []
        for t, tag, aux in rs:
            if tag & (OPEN | CLOSE):
                out.append((to_ns(t), tag, aux))
            elif tag & CLK:
                continue
            else:
                out.append((g0 + int(round((t - k0) * ns_per_clk)), tag, aux))  # 整数相加：g0≈1.8e18，float64 会丢约 256 ns
        d["slots"][s] = out
    return fit


def pair_spans(rs, mode, errs, slotted_ids=frozenset()):
    """把一条泳道的记录配成区间。返回 spans=[(id, t_b, t_e, aux_b, aux_e)], marks=[(id, t, aux)]。
    支持混合模式：带动态 `slot` 表达式的同 Warp 多槽位异步阶段（如 Hopper `wgmma` 多在飞组、`cp.async` 多级软流水，
    槽位数任意动态）按 `(event_id, aux_slot)` 执行 FIFO 队列配对并检查同槽位未释放覆写；无槽位同步阶段走严格栈式 LIFO 配对。"""
    from collections import deque
    spans, marks = [], []
    if mode == "stack":
        stack = []
        slot_fifos = defaultdict(deque)
        for t, tag, aux in rs:
            if tag & (OPEN | CLOSE | CLK):
                continue
            i = tag & IDM
            if tag & (MARK | ROLE_SWITCH):
                marks.append((i, t, aux))
            elif i in slotted_ids:
                q = slot_fifos[(i, aux)]
                if tag & END:
                    if not q:
                        errs["多余的结束事件（没有同 (事件,slot) 的开始）"] += 1
                        continue
                    bt = q.popleft()
                    if t < bt:
                        errs["负时长"] += 1
                    spans.append((i, bt, t, aux, aux))
                else:
                    if q:
                        errs["slot_reentry_overwrite"] += 1
                    q.append(t)
            elif tag & END:
                if not stack:
                    errs["多余的结束事件"] += 1
                    continue
                if stack[-1][0] != i:
                    errs["交叉嵌套（结束的不是最内层；若是多实例并发请声明 slot）"] += 1
                    k = next((j for j in range(len(stack) - 1, -1, -1) if stack[j][0] == i), None)
                    if k is None:
                        continue
                    errs["未配对的开始事件"] += len(stack) - 1 - k
                    del stack[k + 1:]
                bi, bt, baux = stack.pop()
                if t < bt:
                    errs["负时长"] += 1
                spans.append((i, bt, t, baux, aux))
            else:
                stack.append((i, t, aux))
        errs["未配对的开始事件"] += len(stack) + sum(len(q) for q in slot_fifos.values())
        if slot_fifos:
            spans.sort(key=lambda x: (x[1], -x[2]))
    else:  # aux：按 (id, aux) FIFO 配对（支持不定数动态槽位）
        open_by = defaultdict(deque)
        for t, tag, aux in rs:
            if tag & (OPEN | CLOSE | CLK):
                continue
            i = tag & IDM
            if tag & (MARK | ROLE_SWITCH):
                marks.append((i, t, aux))
            elif tag & END:
                lst = open_by.get((i, aux))
                if not lst:
                    errs["多余的结束事件（没有同 (事件,aux) 的开始）"] += 1
                    continue
                bt = lst.popleft()
                if t < bt:
                    errs["负时长"] += 1
                spans.append((i, bt, t, aux, aux))
            else:
                if open_by[(i, aux)]:
                    errs["slot_reentry_overwrite"] += 1
                open_by[(i, aux)].append(t)
        errs["未配对的开始事件"] += sum(len(v) for v in open_by.values())
        spans.sort(key=lambda x: (x[1], -x[2]))
    return spans, marks


def _sweep_depth_ns(intervals_ns):
    """Exact sweep-line concurrency depth over integer-ns ``[(start_ns, end_ns), ...]``.

    Processes completions (-1) before simultaneous launches (+1) at equal timestamps so
    back-to-back transfers do not produce spurious depth spikes.
    Returns ``(segments_ns, avg_depth, max_depth)`` where each segment is ``(t0_ns, t1_ns, depth)``
    for ``depth >= 1`` and ``avg_depth`` is time-weighted over active (depth >= 1) union duration.
    """
    events = []
    for s_ns, e_ns in intervals_ns:
        if e_ns > s_ns:
            events.append((int(s_ns), 1))
            events.append((int(e_ns), -1))
    if not events:
        return [], 0.0, 0
    events.sort(key=lambda kv: (kv[0], kv[1]))
    segs = []
    cur = 0
    max_d = 0
    area = 0.0
    active_ns = 0
    prev_t = events[0][0]
    for t, delta in events:
        dt = t - prev_t
        if dt > 0 and cur > 0:
            segs.append((prev_t, t, cur))
            area += float(dt) * cur
            active_ns += dt
        cur = max(0, cur + delta)
        if cur > max_d:
            max_d = cur
        prev_t = t
    avg_d = (area / active_ns) if active_ns > 0 else 0.0
    return segs, avg_d, max_d


def _match_handoffs_with_depth(d, cross, lane_roles, errs):
    """Match FIFO generations per CTA, event and pipeline slot across producer->consumer roles,
    AND compute deduplicated CTA-level cross-warp in-flight depth (using earliest consumer arrival
    ``min_w t_wait_exit(w, g)`` as the conservative transfer-completion upper bound to avoid 1-to-N
    fanout double-counting or slow-consumer inflation)."""
    producers = defaultdict(lambda: defaultdict(list))
    consumers = defaultdict(lambda: defaultdict(list))
    for lane, records in d["slots"].items():
        for t, tag, aux in records:
            event = tag & IDM
            if event not in cross or tag & (OPEN | CLOSE | CLK | MARK | ROLE_SWITCH):
                continue
            ends = bool(tag & END)
            role = int(cross[event]["end_role" if ends else "begin_role"])
            if lane_roles.get(lane) != role:
                errs["wrong_role"] += 1
                continue
            key = lane // d["wpc"], event, aux
            if ends:
                consumers[key][lane].append(t)
            else:
                producers[key][lane].append(t)

    by_lane = defaultdict(list)
    # Deduplicated CTA-level in-flight intervals: (cta, event) -> [(t_issue_ns, min_consumer_exit_ns, aux), ...]
    cta_ev_intervals = defaultdict(list)
    cta_ev_prod_lane = {}
    for key in producers.keys() | consumers.keys():
        cta, event, aux = key
        prod_lanes = producers[key]
        if prod_lanes:
            lengths = [len(ts) for ts in prod_lanes.values()]
            if min(lengths) != max(lengths):
                errs["unpaired_begin"] += max(lengths) - min(lengths)
            ordered_lists = [sorted(ts) for ts in prod_lanes.values()]
            starts = [min(col) for col in zip(*ordered_lists)]
            cta_ev_prod_lane.setdefault((cta, event), min(prod_lanes.keys()))
        else:
            starts = []
        max_ends = 0
        consumer_sorted = []
        for lane, ends in consumers[key].items():
            ends = sorted(ends)
            consumer_sorted.append(ends)
            max_ends = max(max_ends, len(ends))
            errs["extra_end"] += max(0, len(ends) - len(starts))
            for start, end in zip(starts, ends):
                if end < start:
                    errs["negative_dur"] += 1
                else:
                    by_lane[lane].append((event, start, end, aux, aux))
        # Deduplicated 1-to-N earliest-consumer completion bound per (CTA, event, slot, generation)
        for gen_idx, t_issue in enumerate(starts):
            arrivals = [ends[gen_idx] for ends in consumer_sorted if gen_idx < len(ends)]
            if not arrivals:
                continue
            t_done = min(arrivals)
            if t_done >= t_issue:
                cta_ev_intervals[(cta, event)].append((t_issue, t_done, aux))
        errs["unpaired_begin"] += max(0, len(starts) - max_ends)
    return by_lane, cta_ev_intervals, cta_ev_prod_lane


def _match_handoffs(d, cross, lane_roles, errs):
    """Backward-compatible wrapper returning only per-consumer-lane handoff spans."""
    by_lane, _, _ = _match_handoffs_with_depth(d, cross, lane_roles, errs)
    return by_lane


def assign_sublanes(spans, inst_of, insts, allow_nesting=True):
    """并发实例分子泳道。spans=[(id,tb,te,auxb,auxe)]。返回每个 span 的子泳道号（0 表示主泳道）。
    规则：完好栈嵌套区间（如外层 gemm 包裹内层 load/wait/mma/store，满足 tb >= parent_tb 且 te <= parent_te）
    在 Perfetto 中天然上下叠放，全部留在同一子泳道（0 号），绝不拆成假并列实例；
    仅当出现部分交叉重叠（tb < top_end < te）、pair=aux 并发模式或显式指定 instances 时才拆分子泳道。"""
    n = len(spans)
    sub = [0] * n
    if n == 0:
        return sub
    order = sorted(range(n), key=lambda k: (spans[k][1], -spans[k][2], k))
    has_explicit_inst = any(inst_of(spans[k][0], spans[k][3]) is not None for k in range(n))
    if not allow_nesting:
        free_at = []
        for k in order:
            if inst_of(spans[k][0], spans[k][3]) is not None:
                continue
            tb, te = spans[k][1], spans[k][2]
            chosen_j = next((j for j, fa in enumerate(free_at) if fa <= tb), None)
            if chosen_j is None:
                free_at.append(te)
                chosen_j = len(free_at) - 1
            else:
                free_at[chosen_j] = te
            sub[k] = chosen_j
        base = len(free_at)
        offs, o = {}, 0
        for i in sorted(insts):
            offs[i] = o
            o += insts[i]
        for k in range(n):
            e = inst_of(spans[k][0], spans[k][3])
            if e is not None:
                sub[k] = base + offs[spans[k][0]] + e
        return sub
    test_stack = []
    partial_overlap = False
    for k in order:
        tb, te = spans[k][1], spans[k][2]
        while test_stack and test_stack[-1] <= tb:
            test_stack.pop()
        if test_stack and te > test_stack[-1]:
            partial_overlap = True
            break
        test_stack.append(te)
    if not partial_overlap and not has_explicit_inst:
        return sub
    stacks = []
    for k in order:
        if inst_of(spans[k][0], spans[k][3]) is not None:
            continue
        tb, te = spans[k][1], spans[k][2]
        chosen_j = None
        for j, st in enumerate(stacks):
            while st and st[-1] <= tb:
                st.pop()
            if not st or te <= st[-1]:
                st.append(te)
                chosen_j = j
                break
        if chosen_j is None:
            stacks.append([te])
            chosen_j = len(stacks) - 1
        sub[k] = chosen_j
    base = len(stacks)
    offs, o = {}, 0
    for i in sorted(insts):
        offs[i] = o
        o += insts[i]
    for k in range(n):
        e = inst_of(spans[k][0], spans[k][3])
        if e is not None:
            sub[k] = base + offs[spans[k][0]] + e
    return sub


def sm_grouping(d, cta_sm, errs):
    """FI_GRAN=1：把每 CTA 一条的泳道按 OPEN 记录的 %smid 归到物理 SM，并给同一 SM 上的 CTA 分子条。

    子条号用贪心区间着色：按 CTA 生命周期（第一条记录→最后一条记录）的开始时刻排序，每个 CTA 拿
    "最早空出来的那条子条"。区间图上贪心是最优的，所以子条数 == 该 SM 的最大并发驻留 CTA 数；
    先后驻留（不重叠）的 CTA 会共用 0 号子条，于是"每个 SM 一条时间条"，只有真并发时才并列拆开。
    返回 (lane_sub, sm_ctas, sm_conc, sm_span)。"""
    life = {s: (rs[0][0], rs[-1][0]) for s, rs in d["slots"].items()}
    by_sm = defaultdict(list)
    for s in sorted(life):
        by_sm[cta_sm.get(s, -1)].append(s)
    if -1 in by_sm:
        errs["SM 粒度泳道缺 OPEN（拿不到 %smid，已归入 SM 未知）"] += len(by_sm[-1])
    lane_sub, sm_ctas, sm_conc, sm_span = {}, {}, {}, {}
    for sm, ss in by_sm.items():
        free_at = []  # 每条子条被占到什么时刻
        for s in sorted(ss, key=lambda x: (life[x][0], life[x][1], x)):
            tb, te = life[s]
            j = next((j for j, fa in enumerate(free_at) if fa <= tb), None)
            if j is None:
                free_at.append(te)
                j = len(free_at) - 1
            else:
                free_at[j] = te
            lane_sub[s] = j
        sm_ctas[sm] = ss
        sm_conc[sm] = len(free_at)
        sm_span[sm] = (min(life[s][0] for s in ss), max(life[s][1] for s in ss))
    return lane_sub, sm_ctas, sm_conc, sm_span


def decode(d, names, group_override=None, sm_bar=True, semantic_kb=None,
           require_semantics=False, cta_cu_map=None, cta_hw_map=None):
    try:
        kb = dict(semantic_kb or load_kb())
        semantic_contract = resolve_names(names, kb)
    except SemanticKBError as exc:
        raise SystemExit(f"canonical semantic contract invalid: {exc}") from exc
    ev_name = {int(k): v for k, v in names.get("events", {}).items()}
    ev_label = {int(k): str(v) for k, v in (names.get("labels") or {}).items() if str(v).strip()}
    ev_fabric = {int(k): str(v) for k, v in (names.get("fabrics") or {}).items() if str(v).strip()}
    ev_async_engine = {
        int(k): str(v).strip().lower()
        for k, v in (names.get("async_engines") or {}).items()
        if str(v).strip().lower() in ("mem", "mma")
    }
    ev_sem = {int(k): v for k, v in semantic_contract["events"].items()}
    role_name = {int(k): v for k, v in names.get("roles", {}).items()}
    aux_name = {int(k): v for k, v in names.get("aux", {}).items()}
    insts = {int(k): int(v) for k, v in names.get("instances", {}).items()}
    label_aux = bool(names.get("label_aux", True))
    pair = names.get("pair", "stack")
    if pair not in ("stack", "aux"):
        raise SystemExit(f"names.json 的 pair 只能是 stack 或 aux，收到 {pair!r}")
    gran = int(d.get("gran", 0))
    amd = bool(d.get("amd", _is_amd(d["meta"])))
    unit = "CU" if amd else "SM"          # 物理单元名：NVIDIA=SM，AMD=CU
    # 归组：gran=1 时 NVIDIA 默认 sm_resident（每 SM 一条时间条 + 并发子条 + 占用整条），
    # AMD 默认 cu（每物理 CU 一个进程，重叠区间分条带）；warp/wave 粒度照 names.json 的 group。
    # --group 可强制覆盖（role/sm/cu/cta）。
    if group_override in ("role", "sm", "cu", "cta", "sm_resident"):
        group = group_override
    elif gran == 1:
        group = "cu" if amd else "sm_resident"
    else:
        group = names.get("group", "role")
    if group == "cu" and not amd:
        print("提示：group=cu 但 meta.arch 不是 gfx*，按 group=sm 处理（aux 原样当 SM 号）", file=sys.stderr)
        group = "sm"
    if group == "sm" and amd:
        print("提示：AMD 转储上 group=sm 按物理 CU 归组（aux 是 HW_ID 编码，不能直接当 SM 号）", file=sys.stderr)
        group = "cu"
    sm_group = group == "sm_resident"
    cu_group = group == "cu"
    wpc = d["wpc"]
    errs = defaultdict(int)
    if d["overflow"] & ~3:
        errs["unknown_overflow_bits"] += 1
    if d["overflow"] & 1:
        errs["泳道写满（cap 给小了）"] += 1
    if d["overflow"] & 2:
        errs["slot 越界（warps_per_cta 或 CTA 数给小了）"] += 1
    errs["泳道计数超过 cap"] += sum(1 for c in d["cnt"] if c > d["cap"])
    if gran == 1 and wpc != 1:
        errs[f"{unit} 粒度转储的 warps_per_cta 应为 1（fi_alloc 与 -DFI_GRAN 不是同一档？）"] += 1
    if amd and not d.get("timer_hz") and d["timer"] == 0:
        errs["AMD 转储缺 meta.timer_hz（s_memrealtime tick 无法换算 ns，时间轴不可信）"] += 1
    all_t = [r[0] for rs in d["slots"].values() for r in rs]
    if not all_t:
        raise SystemExit("转储里没有任何记录：FI_LEVEL 是否 >=1、fi_attach 是否在含 kernel 的同一 .cu 里调用、launch 前是否 fi_reset")
    t0 = min(all_t)
    us = lambda t: (t - t0) / 1000.0
    cta_role, cta_sm, slot_role, cta_hw = {}, {}, {}, {}  # cta_hw：AMD 的 cta -> (xcc, se, cu, simd)
    for s, rs in d["slots"].items():
        for t, tag, aux in rs:
            if tag & OPEN:
                slot_role[s] = tag & 0xFFFF  # 角色按泳道记（warp/wave 粒度下同一 CTA 内可以特化）
                cta_role.setdefault(s // wpc, tag & 0xFFFF)  # 只用于 group=role 时决定进程归属（取 CTA 里号最小的泳道）
                if amd:
                    cu_id, xcc, se, cu, simd = _cu_of(aux)
                    cta_sm.setdefault(s // wpc, cu_id)
                    cta_hw.setdefault(s // wpc, (xcc, se, cu, simd))
                else:
                    cta_sm.setdefault(s // wpc, aux)
    # Phase-rich traces may contain more than one OPEN per slot.  Taking the
    # first aux (setdefault above) then makes group=cu PID/sm assignment depend
    # on record presentation.  When supplied, the OPEN-anchor map is the sole
    # authority for all CTA placement.
    if cta_cu_map is not None:
        authoritative = {int(k): int(v) for k, v in cta_cu_map.items()}
        cta_ids = {int(s) // wpc for s in d["slots"]}
        missing = sorted(cta_ids - set(authoritative))
        if missing:
            raise SystemExit(f"CTA->CU map is missing CTAs: {missing[:16]}")
        cta_sm = {cta: authoritative[cta] for cta in sorted(cta_ids)}
        if cta_hw_map:
            for cta in sorted(cta_ids):
                hw = (cta_hw_map.get(str(cta)) or cta_hw_map.get(cta))
                if isinstance(hw, dict):
                    cta_hw[cta] = (int(hw["xcc"]), int(hw["se"]),
                                   int(hw["cu_lane"]), int(hw["simd"]))
    lane_sub, sm_ctas, sm_conc, sm_span = ({}, {}, {}, {})
    if sm_group or (cu_group and gran == 1):
        lane_sub, sm_ctas, sm_conc, sm_span = sm_grouping(d, cta_sm, errs)
    inst_of = lambda i, aux: (aux % insts[i]) if i in insts and insts[i] > 0 else None
    cross = {int(i): desc for i, desc in names.get("cross_warp", {}).items()}
    slotted_ids = {int(i) for i in (names.get("slotted_events") or {}) if int(i) not in cross}
    if cross:
        handoffs, cross_cta_intervals, cross_cta_prod_lane = _match_handoffs_with_depth(d, cross, slot_role, errs)
    else:
        handoffs, cross_cta_intervals, cross_cta_prod_lane = {}, {}, {}
    probe_ids = {int(i): name for i, name in names.get("probe_ids", {}).items()}
    probe_ids.update({i: desc["id"] for i, desc in cross.items()})
    fires = dict.fromkeys(probe_ids.values(), 0)
    first_open = min((t for rs in d["slots"].values() for t, tag, _ in rs if tag & OPEN), default=None)
    spans, marks = [], []  # spans: (slot, sub, id, s_us, e_us, aux_b, aux_e)
    deltas = []
    n_sub = defaultdict(int)  # 每泳道用到的子泳道数
    for s, rs in sorted(d["slots"].items()):
        prev = None
        lane_fires = defaultdict(int)
        for t, tag, aux in rs:
            if not tag & (OPEN | CLOSE | CLK | MARK | END) and (tag & IDM) in probe_ids:
                lane_fires[probe_ids[tag & IDM]] += 1
            if prev is not None:
                if t < prev:
                    errs["同一泳道时间倒退"] += 1
                elif t > prev:
                    deltas.append(t - prev)
            prev = t
        for probe, count in lane_fires.items():
            fires[probe] = max(fires[probe], count)
        if probe_ids and (not rs[0][1] & OPEN or not rs[-1][1] & CLOSE):
            errs["missing_anchor"] += 1
        local = [r for r in rs if r[1] & (OPEN | CLOSE | CLK | MARK) or (r[1] & IDM) not in cross] if cross else rs
        sp, mk = pair_spans(local, pair, errs, slotted_ids=slotted_ids)
        sp.extend(handoffs.get(s, ()))
        sub = assign_sublanes(sp, inst_of, insts, allow_nesting=(pair != "aux"))
        for (i, bt, et, ba, ea), j in zip(sp, sub):
            spans.append((s, j, i, us(bt), us(et), ba, ea))
            n_sub[s] = max(n_sub[s], j + 1)
        marks.extend((s, 0, i, us(t), aux) for i, t, aux in mk)
    # group=cu：每个物理 CU 一个进程，同一 CU 上（可能来自多个 CTA 的）重叠区间按贪心着色分条带。
    cu_pid_of = lambda c: cta_sm.get(c, 65535)
    cu_span_tid = cu_mark_tid = None
    if cu_group:
        if gran == 1:
            # CTA 粒度：按 CTA 真实驻留生命周期 [OPEN, CLOSE] 着色（1-based tid 避免 Perfetto tid=0->pid 碰撞），
            # 同一 CTA 的嵌套区间（gemm 外框与内部 load/mma/store）保持在同一驻留槽内，不再翻倍拆成假条带。
            cu_span_tid = [1 + lane_sub.get(s, 0) for (s, *_rest) in spans]
            cu_mark_tid = [1 + lane_sub.get(s, 0) for (s, *_rest) in marks]
        else:
            by_pid = defaultdict(list)
            for idx, (s, j, i, a, b, ba, ea) in enumerate(spans):
                by_pid[cu_pid_of(s // wpc)].append(idx)
            cu_span_tid = [1] * len(spans)
            cu_last_end = {}
            for p, idxs in by_pid.items():
                free_at = []
                for idx in sorted(idxs, key=lambda k: (spans[k][3], -spans[k][4])):
                    tb, te_ = spans[idx][3], spans[idx][4]
                    jj = next((jj for jj, fa in enumerate(free_at) if fa <= tb), None)
                    if jj is None:
                        free_at.append(te_)
                        jj = len(free_at) - 1
                    else:
                        free_at[jj] = te_
                    cu_span_tid[idx] = 1 + jj
                cu_last_end[p] = free_at
            cu_mark_tid = []
            for (s, j, i, a, aux) in marks:
                ends = cu_last_end.get(cu_pid_of(s // wpc), [])
                cu_mark_tid.append(1 + next((jj for jj, fa in enumerate(ends) if fa <= a), 0))
    # 统计
    by_name = defaultdict(list)
    for s, j, i, a, b, *_ in spans:
        by_name[i].append(b - a)
    stat = {}
    by_label = defaultdict(list)
    events_by_id = {}
    for i, v in sorted(by_name.items()):
        v = sorted(v)
        row = ev_sem.get(i) or {}
        label = row.get("display_name", "unknown")
        by_label[label].extend(v)
        events_by_id[str(i)] = dict(
            semantic_id=row.get("semantic_id", "unknown"),
            canonical_label=label,
            original_label=ev_name.get(i, f"未命名事件{i}"),
            semantic_source=row.get("semantic_source", "unknown"),
            n=len(v), total_us=round(sum(v), 3), mean_us=round(sum(v) / len(v), 3),
            p50_us=round(v[len(v) // 2], 3), max_us=round(v[-1], 3))
    for label, v in sorted(by_label.items()):
        v.sort()
        stat[label] = dict(n=len(v), total_us=round(sum(v), 3), mean_us=round(sum(v) / len(v), 3),
                           p50_us=round(v[len(v) // 2], 3), max_us=round(v[-1], 3))
    # 每个角色：该角色在每个 CTA 里所有泳道的最晚结束时刻，再在 CTA 之间看分布
    role_cta_end = defaultdict(lambda: defaultdict(float))
    for s, rs in d["slots"].items():
        r = slot_role.get(s, -1)
        c = s // wpc
        role_cta_end[r][c] = max(role_cta_end[r][c], us(rs[-1][0]))
    roles = {r: list(v.values()) for r, v in role_cta_end.items()}
    role_stat = {}
    for r, v in sorted(roles.items()):
        v.sort()
        med = statistics.median(v)
        role_stat[role_name.get(r, f"角色{r}" if r >= 0 else "未 FI_OPEN")] = dict(
            ctas=len(v), end_min_us=round(v[0], 3), end_median_us=round(med, 3), end_max_us=round(v[-1], 3),
            tail_us=round(v[-1] - med, 3))
    missing = sorted(set(by_name) - set(ev_name))
    if missing:
        errs["names.json 缺事件名"] += len(missing)
    unknown_semantics = semantic_contract["unknown_event_ids"]
    ambiguous_semantics = semantic_contract["ambiguous_event_ids"]
    if require_semantics and unknown_semantics:
        errs["事件没有唯一 canonical semantic_id"] += len(unknown_semantics)
    res = min(deltas) if deltas else None
    lanes_split = sum(1 for v in n_sub.values() if v > 1)
    observation_span_us = round(max(all_t) / 1000.0 - t0 / 1000.0, 3)
    semantic_start_us = min((sp[3] for sp in spans), default=None)
    semantic_end_us = max((sp[4] for sp in spans), default=None)
    semantic_span_us = (round(semantic_end_us - semantic_start_us, 3)
                        if semantic_start_us is not None else None)
    summary = dict(span_us=observation_span_us, semantic_start_us=semantic_start_us,
                   semantic_end_us=semantic_end_us, semantic_span_us=semantic_span_us,
                   observation_start_ns=int(t0), observation_end_ns=int(max(all_t)),
                   records=len(all_t), spans=len(spans), marks=len(marks),
                   timer_min_step_ns=res, errors={k: v for k, v in errs.items() if v}, events=stat, roles=role_stat,
                   unnamed_event_ids=missing, meta=d["meta"],
                   # 计时器名按平台判（不是按有没有 timer_hz）：NVIDIA 是 %globaltimer / clock64，
                   # AMD 是 s_memrealtime / readcyclecounter（fi_bugs_b3-sk-qwen B17）
                   timer=("clock64+锚点" if d["timer"] == 1 and not amd else
                          ("readcyclecounter+锚点" if d["timer"] == 1 else
                           ("s_memrealtime" if amd else "globaltimer"))),
                   arch=d["meta"].get("arch"), platform=("amd" if amd else "nvidia"),
                   platform_source=d.get("platform_source"), timer_hz=d.get("timer_hz") or None,
                   pair=pair, lanes=len(d["slots"]), lanes_with_sublanes=lanes_split,
                   max_sublanes=(max(n_sub.values()) if n_sub else 0),
                   clock_mhz_per_lane=(dict(min=min(d["clock_fit"].values()), max=max(d["clock_fit"].values()))
                                       if d["clock_fit"] else None),
                   semantic_schema=semantic_contract["schema"],
                   semantic_kb_sha256=semantic_contract["kb_sha256"],
                   semantic_events=semantic_contract["events"],
                   events_by_id=events_by_id,
                   semantic_unknown_event_ids=unknown_semantics,
                   semantic_ambiguous_event_ids=ambiguous_semantics,
                   semantic_errors=semantic_contract["errors"],
                   semantics_ok=bool(semantic_contract["ok"]))
    # gran/lane_kind/group 始终写出（老转储也会写 gran=0）；sm_grouping 只在 NVIDIA SM 粒度归组时出现。
    summary["gran"] = gran
    summary["first_open_ns"] = first_open
    summary["fires"] = fires
    summary["overflow"] = d["overflow"]
    summary["wpc"] = wpc
    summary["n_slots"] = d["n_slots"]
    summary["n_ctas"] = d["n_slots"] // max(1, wpc)
    summary["cta_sm"] = {str(k): int(v) for k, v in sorted(cta_sm.items())}
    summary["cta_hw"] = {str(k): list(v) for k, v in sorted(cta_hw.items())}
    summary["cross_warp_spans"] = sum(len(v) for v in handoffs.values())
    # Deduplicated CTA-level cross-warp producer->consumer in-flight depth (using min_w t_wait_exit per generation)
    cross_inflight_summary = {}
    cross_inflight_cta_segs = {}
    if cross_cta_intervals:
        by_ev_ctas = defaultdict(list)
        by_cta_all = defaultdict(list)
        by_cta_prod_lane = {}
        all_durs_ns = []
        for (cta_id, ev_id), ivs in sorted(cross_cta_intervals.items()):
            _, ev_avg_d, ev_max_d = _sweep_depth_ns([(a, b) for a, b, _ in ivs])
            durs = [int(b - a) for a, b, _ in ivs]
            all_durs_ns.extend(durs)
            by_ev_ctas[ev_id].append((cta_id, ev_max_d, ev_avg_d, durs))
            by_cta_all[cta_id].extend((a, b) for a, b, _ in ivs)
            prod_lane = cross_cta_prod_lane.get((cta_id, ev_id))
            if prod_lane is not None:
                prev = by_cta_prod_lane.get(cta_id)
                by_cta_prod_lane[cta_id] = prod_lane if prev is None else min(prev, prod_lane)
        # Sweep CTA-wide union of all cross-warp TMA/async transfers (e.g., K + V simultaneously in flight on W00)
        cta_max_d_list = []
        cta_avg_d_list = []
        for cta_id, ivs_all in sorted(by_cta_all.items()):
            segs_ns, avg_d, max_d = _sweep_depth_ns(ivs_all)
            cta_max_d_list.append(max_d)
            if avg_d > 0.0:
                cta_avg_d_list.append(avg_d)
            prod_lane = by_cta_prod_lane.get(cta_id)
            cross_inflight_cta_segs[str(cta_id)] = {
                "cta": cta_id, "sm": cta_sm.get(cta_id),
                "producer_warp": (prod_lane % wpc) if prod_lane is not None else None,
                "max_depth": max_d, "avg_depth": round(avg_d, 3), "n_transfers": len(ivs_all),
                "segments_us": [[round(us(s0), 3), round(us(s1), 3), dp] for s0, s1, dp in segs_ns],
            }
        per_event = {}
        for ev_id, rows in sorted(by_ev_ctas.items()):
            ev_durs = sorted(x for _, _, _, dl in rows for x in dl)
            ev_maxs = sorted(r[1] for r in rows)
            ev_avgs = [r[2] for r in rows if r[2] > 0.0]
            per_event[str(ev_id)] = {
                "event_id": ev_id,
                "label": ev_label.get(ev_id) or ev_name.get(ev_id, f"event_{ev_id}"),
                "ctas_measured": len(rows),
                "n_transfers": len(ev_durs),
                "max_depth": ev_maxs[-1] if ev_maxs else 0,
                "p50_max_depth": ev_maxs[len(ev_maxs) // 2] if ev_maxs else 0,
                "avg_depth": round(sum(ev_avgs) / len(ev_avgs), 3) if ev_avgs else 0.0,
                "residence_p50_ns": ev_durs[len(ev_durs) // 2] if ev_durs else 0,
                "residence_mean_ns": round(sum(ev_durs) / len(ev_durs), 1) if ev_durs else 0.0,
                "residence_p90_ns": ev_durs[min(len(ev_durs) - 1, int(len(ev_durs) * 0.9))] if ev_durs else 0,
            }
        all_durs_ns.sort()
        cta_max_d_list.sort()
        cross_inflight_summary = {
            "ctas_measured": len(by_cta_all),
            "n_transfers": len(all_durs_ns),
            "max_d_tma": cta_max_d_list[-1] if cta_max_d_list else 0,
            "p50_max_d_tma": cta_max_d_list[len(cta_max_d_list) // 2] if cta_max_d_list else 0,
            "avg_d_tma": round(sum(cta_avg_d_list) / len(cta_avg_d_list), 3) if cta_avg_d_list else 0.0,
            "residence_p50_ns": all_durs_ns[len(all_durs_ns) // 2] if all_durs_ns else 0,
            "residence_mean_ns": round(sum(all_durs_ns) / len(all_durs_ns), 1) if all_durs_ns else 0.0,
            "residence_p90_ns": all_durs_ns[min(len(all_durs_ns) - 1, int(len(all_durs_ns) * 0.9))] if all_durs_ns else 0,
            "per_event": per_event,
        }
    summary["cross_warp_inflight"] = cross_inflight_summary
    summary["lane_bounds_ns"] = {
        str(s): [int(rs[0][0]), int(rs[-1][0]), int(slot_role.get(s, 0))]
        for s, rs in sorted(d["slots"].items()) if rs
    }
    summary["lane_kind"] = "cta" if gran == 1 else ("wave" if amd else "warp")
    summary["group"] = group
    if cu_group:
        cus = sorted({v for v in cta_sm.values() if v != 65535})
        summary["cu"] = dict(count=len(cus), max_bands=(max(cu_span_tid) if cu_span_tid else 0),
                             ids=cus[:64])
    if sm_group:
        per_sm = sorted(len(v) for v in sm_ctas.values())
        conc = sorted(sm_conc.values())
        busy = sorted(round((us(b) - us(a)), 3) for a, b in sm_span.values())
        summary["sm_grouping"] = dict(
            sms_used=len(sm_ctas), ctas=len(lane_sub),
            ctas_per_sm=dict(min=per_sm[0], median=statistics.median(per_sm), max=per_sm[-1]),
            max_concurrent_resident=max(conc), sms_with_concurrent_resident=sum(1 for c in conc if c > 1),
            sm_busy_us=dict(min=busy[0], median=statistics.median(busy), max=busy[-1]),
            sm_busy_spread_us=round(busy[-1] - busy[0], 3),
            sub_bars_total=sum(conc), sm_bar=bool(sm_bar),
            unknown_sm_ctas=len(sm_ctas.get(-1, [])))
    # Chrome trace。tid 编码：warp/wave 粒度 = slot * 64 + 子泳道号；SM 粒度 = 子条号 * 64 + 子泳道号
    # （同一 SM 进程内唯一）；group=cu 时 tid = CU 内条带号（着色已在统计段前完成）。
    tev = []
    pid_of = (lambda c: cta_sm.get(c, -1)) if sm_group else \
        (lambda c: {"role": cta_role.get(c, 65535), "sm": cta_sm.get(c, 65535),
                    "cu": cta_sm.get(c, 65535)}.get(group, c))
    # Perfetto treats tid=0 as "unspecified" and resolves it to pid.  Keep the
    # SM-resident software tracks at non-zero, low tids so the main bar cannot
    # collide with another lane whose encoded tid equals the SM id.
    tid_of = ((lambda s, j: 1 + lane_sub.get(s, 0) * 64 + j) if sm_group
              else (lambda s, j: 1 + s * 64 + j))
    sm_cta_count = defaultdict(int)
    for _cta_k, _sm_v in cta_sm.items():
        sm_cta_count[_sm_v] += 1
    active_role_names = [role_name[r] for r in sorted({slot_role.get(s, -1) for s in d["slots"]}) if r in role_name]
    seen_p, seen_t = set(), set()
    # SM 粒度下每条 (SM, 子条) 轨道的名字：主条＝该 SM 的时间条；并列子条＝同 SM 上并发驻留的其它 CTA
    track_name = {}
    if sm_group or (cu_group and gran == 1):
        for sm, ss in sm_ctas.items():
            by_col = defaultdict(list)
            for s in ss:
                by_col[lane_sub[s]].append(s)
            for col, mem in by_col.items():
                mem.sort(key=lambda s: d["slots"][s][0][0])
                head = f"{unit}{sm}" if sm >= 0 else f"{unit} 未知"
                kind = (f"CTA驻留槽{col + 1}" if cu_group else ("主条" if col == 0 else f"并列子条{col}"))
                if len(mem) == 1:
                    rn = role_name.get(slot_role.get(mem[0], -1))
                    track_name[(sm, col)] = f"{head} {kind} · CTA{mem[0]:03d}" + (f" ({rn})" if rn else "")
                else:
                    lst = "、".join(f"CTA{m:03d}" for m in mem[:3]) + ("…" if len(mem) > 3 else "")
                    track_name[(sm, col)] = f"{head} {kind} · {len(mem)} 个 CTA 先后驻留（{lst}）"

    def ensure_process(p, c=None):
        if p in seen_p:
            return p
        seen_p.add(p)
        if sm_group:
            pn = (f"{unit}{p}（{len(sm_ctas[p])} 个 CTA，最多同时 {sm_conc[p]} 个）" if p >= 0
                  else f"{unit} 未知（{len(sm_ctas.get(-1, []))} 个 CTA 缺 OPEN）")
        elif group == "role":
            pn = role_name.get(p, f"角色{p}")
        elif group == "sm":
            n_c = sm_cta_count.get(p, 0)
            role_tag = f" · {' / '.join(active_role_names)}" if len(active_role_names) >= 2 else ""
            pn = f"{unit}{p}" + (f"（{n_c} 个 CTA{role_tag}）" if n_c > 0 else "")
        elif cu_group:
            hw = cta_hw.get(c) if c is not None else None
            loc = f"XCC{hw[0]} SE{hw[1]} CU{hw[2]}" if hw else ""
            if gran == 1 and p in sm_ctas:
                extra_s = f"{len(sm_ctas[p])} 个 CTA，最多同时驻留 {sm_conc[p]} 个 CTA"
                pn = f"CU{p}（{loc + ' · ' if loc else ''}{extra_s}）"
            else:
                pn = f"CU{p}" + (f"（{loc}）" if loc else "")
        else:
            pn = f"CTA{p} {role_name.get(cta_role.get(p, -1), '')}"
        tev.append(dict(ph="M", pid=p, name="process_name", args=dict(name=pn)))
        tev.append(dict(ph="M", pid=p, name="process_sort_index", args=dict(sort_index=p)))
        return p

    def ensure_thread(s, j, tid=None):
        c, w = divmod(s, wpc)
        p = ensure_process(slot_role.get(s, 65535) if cross and group == "role" else pid_of(c), c)
        tid = tid_of(s, j) if tid is None else tid
        if (p, tid) not in seen_t:
            seen_t.add((p, tid))
            if sm_group:
                nm = track_name.get((p, lane_sub.get(s, 0)), f"{unit}{p}") + (f" · 实例{j}" if n_sub.get(s, 1) > 1 else "")
            elif cu_group and gran == 1:
                nm = track_name.get((p, lane_sub.get(s, 0)), f"CU{p} CTA驻留槽{tid}")
            elif cu_group:
                nm = f"Wave驻留槽{tid}"
            else:
                sm = cta_sm.get(c)
                rn = role_name.get(slot_role.get(s, -1))
                nm = f"CTA{c:03d} w{w:02d}" + (f" ({rn})" if rn else "") + (f" {unit}{sm}" if sm is not None else "") + (f" · 实例{j}" if n_sub.get(s, 1) > 1 else "")
            tev.append(dict(ph="M", pid=p, tid=tid, name="thread_name", args=dict(name=nm)))
            tev.append(dict(ph="M", pid=p, tid=tid, name="thread_sort_index", args=dict(sort_index=tid)))
        return p, tid

    def canonical_semantic(i):
        row = ev_sem.get(i) or {}
        return row.get("semantic_id", "unknown"), row.get("display_name", "unknown"), row

    def lab(i, aux):
        sid, n, _ = canonical_semantic(i)
        # Canonical semantic visibility never drifts.  Aux is an attribute of
        # the event, not part of its display identity.
        if sid != "unknown":
            return sid
        return f"{n} #{aux}" if label_aux and i in aux_name else n

    def async_engine_of(i, is_mark=False, has_multi_sub=False):
        if i in ev_async_engine:
            return ev_async_engine[i]
        if i in slotted_ids or i in cross or is_mark or (pair == "aux" and has_multi_sub):
            sid, _, _ = canonical_semantic(i)
            if sid in ("mma", "gemm"):
                return "mma"
            if sid in ("load", "dispatch", "communication", "store"):
                return "mem"
        return None

    span_order = sorted(range(len(spans)), key=lambda k: (round(spans[k][3], 3), -round(spans[k][4], 3), k))
    for k in span_order:
        s, j, i, a, b, ba, ea = spans[k]
        c, w = divmod(s, wpc)
        p, tid = ensure_thread(s, j, cu_span_tid[k] if cu_span_tid is not None else None)
        args = dict(cta=c, warp=w, sm=cta_sm.get(c), role=role_name.get(slot_role.get(s, -1), slot_role.get(s, -1)), aux=ba)
        if i in cross:
            rid = int(cross[i]["begin_role"])
            args.update(handoff_from=role_name.get(rid, str(rid)), slot=ba)
        if amd:
            hw = cta_hw.get(c)
            if hw:
                args.update(xcc=hw[0], se=hw[1], cu=hw[2], simd=hw[3])
        sid, canonical_label, semantic_row = canonical_semantic(i)
        args["semantic_id"] = sid
        args["display_name"] = sid
        args["semantic_source"] = semantic_row.get("semantic_source", "unknown")
        # The FILITE02 lane index is the unique physical warp/wave slot.  On
        # AMD wave granularity, ``warp`` already contains (simd << 4) | wave;
        # retain both the stable slot and its two decoded components.
        args["wave_slot"] = s
        args["wave_key"] = w
        if amd:
            args["slot_simd"] = (w >> 4) & 0x3
            args["wave_id"] = w & 0xF
        args["event_id"] = i
        args["original_label"] = semantic_row.get("input_label", ev_name.get(i, ""))
        if i in ev_label:
            args["display_label"] = ev_label[i]
        if i in ev_fabric:
            args["fabric"] = ev_fabric[i]
        eng = async_engine_of(i, is_mark=False, has_multi_sub=(n_sub.get(s, 1) > 1))
        if eng:
            args["async_engine"] = eng
        if semantic_row.get("detail"):
            args["semantic_detail"] = semantic_row["detail"]
        if i in aux_name:
            args[aux_name[i]] = ba
        if ea != ba:
            args["aux_end"] = ea
        if n_sub.get(s, 1) > 1:
            args["instance"] = j
        if sm_group or (cu_group and gran == 1):
            args["resident_bar"] = lane_sub.get(s, 0)
        ts_r = round(a, 3)
        end_r = round(b, 3)
        dur_r = round(max(0.0, end_r - ts_r), 3)
        tev.append(dict(ph="X", pid=p, tid=tid, ts=ts_r, dur=dur_r, name=lab(i, ba),
                        cat=canonical_label, args=args))
    for k, (s, j, i, a, aux) in enumerate(marks):
        c, w = divmod(s, wpc)
        p, tid = ensure_thread(s, j, cu_mark_tid[k] if cu_mark_tid is not None else None)
        sid, canonical_label, semantic_row = canonical_semantic(i)
        m_args = dict(
            cta=c, warp=w, sm=cta_sm.get(c),
            role=role_name.get(slot_role.get(s, -1), slot_role.get(s, -1)),
            aux=aux, semantic_id=sid, display_name=sid, event_id=i,
            original_label=semantic_row.get("input_label", ev_name.get(i, "")),
        )
        if i in ev_label:
            m_args["display_label"] = ev_label[i]
        eng = async_engine_of(i, is_mark=True)
        if eng:
            m_args["async_engine"] = eng
        if amd:
            hw = cta_hw.get(c)
            if hw:
                m_args.update(xcc=hw[0], se=hw[1], cu=hw[2], simd=hw[3])
        tev.append(dict(ph="i", s="t", pid=p, tid=tid, ts=round(a, 3), name=lab(i, aux),
                        cat=canonical_label, args=m_args))
    # SM 粒度：每个 SM 再加一条"SM 占用"整条（cat=info，fi_overview 的覆盖率自检会跳过这一类），
    # 一眼看出各 SM 的驻留起止与负载不均；FI_LEVEL=1 的转储（没有区间）也因此在 trace 里非空。
    if sm_group and sm_bar:
        # Keep the overview track below Perfetto's 32-bit tid ceiling and
        # above every SM-resident lane.  The old 1<<40 value was truncated to
        # its low bits on import and collided with ordinary sub-lanes.
        BAR = 1 << 20
        for sm in sorted(sm_span):
            tb, te = sm_span[sm]
            ensure_process(sm)
            tid = BAR + (sm if sm >= 0 else 0)
            nm = f"{unit}{sm} 占用整条" if sm >= 0 else f"{unit} 未知 占用整条"
            tev.append(dict(ph="M", pid=sm, tid=tid, name="thread_name", args=dict(name=nm)))
            tev.append(dict(ph="M", pid=sm, tid=tid, name="thread_sort_index", args=dict(sort_index=-1)))
            tev.append(dict(ph="X", pid=sm, tid=tid, ts=round(us(tb), 3), dur=round(us(te) - us(tb), 3),
                            name=f"{unit} 占用", cat="info",
                            args=dict(sm=sm, ctas=len(sm_ctas[sm]), max_concurrent_resident=sm_conc[sm],
                                      first_us=round(us(tb), 3), last_us=round(us(te), 3))))
    trace = dict(traceEvents=tev, displayTimeUnit="ns",
                 otherData=dict(tool="weaveprofiler",
                                time_zero="本次转储最早一条记录的设备计时器（已换算 ns）",
                                observation_start_ns=summary["observation_start_ns"],
                                observation_end_ns=summary["observation_end_ns"],
                                observation_start_us=0.0,
                                observation_end_us=summary["span_us"],
                                observation_span_us=summary["span_us"],
                                semantic_span_us=summary["semantic_span_us"],
                                errors=summary["errors"], meta=d["meta"], roles=role_name, events=ev_name,
                                semantics=semantic_contract,
                                semantic_ids={str(k): v.get("semantic_id") for k, v in ev_sem.items()},
                                gran=gran, group=group, platform=summary["platform"], arch=summary["arch"],
                                cta_to_cu=({str(k): int(v) for k, v in sorted(cta_sm.items())}
                                           if cta_cu_map is not None else None),
                                cta_cu_source=(("FILITE02 OPEN aux via --cta-cu-map")
                                               if cta_cu_map is not None else None),
                                cross_warp_inflight=cross_inflight_summary or None,
                                cross_warp_inflight_ctas=cross_inflight_cta_segs or None))
    return trace, summary


def write_json(obj, path):
    # GEMM traces can exceed 300 MB before compression.  Level 9 costs ~12 s
    # on the H20 probe and saves little; level 3 is ~1 s and remains valid gzip.
    # msgspec is optional, but avoids another ~4 s of Python JSON encoding.
    data = (msgspec.json.encode(obj) if msgspec is not None else
            json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if path.endswith(".gz"):
        with gzip.open(path, "wb", compresslevel=3) as f:
            f.write(data)
    else:
        with open(path, "wb") as f:
            f.write(data)


def print_summary(sm):
    print(f"时长 {sm['span_us']} us，记录 {sm['records']} 条，区间 {sm['spans']} 个，计时器 {sm['timer']}，最小步长 {sm['timer_min_step_ns']} ns，"
          f"泳道 {sm['lanes']} 条（其中 {sm['lanes_with_sublanes']} 条拆了子泳道，最多 {sm['max_sublanes']} 个）")
    if sm.get("platform") == "amd":
        print(f"平台 AMD（arch={sm.get('arch')}，计时器 {sm['timer']}"
              + (f"，meta.timer_hz={sm['meta'].get('timer_hz')}" if sm.get("meta", {}).get("timer_hz") else "") + "）")
    if sm.get("gran") == 1:
        g = sm.get("sm_grouping") or {}
        if g:
            print(f"粒度 SM/CU（FI_GRAN=1）：泳道＝CTA，用到 {g['sms_used']} 个物理单元、{g['ctas']} 个 CTA；"
                  f"每 SM 的 CTA 数 min/中位/max = {g['ctas_per_sm']['min']}/{g['ctas_per_sm']['median']}/{g['ctas_per_sm']['max']}；"
                  f"同一单元最大并发驻留 {g['max_concurrent_resident']} 个 CTA（{g['sms_with_concurrent_resident']} 个有并发），"
                  f"共 {g['sub_bars_total']} 条时间条")
            print(f"SM 占用整条 us：min {g['sm_busy_us']['min']} / 中位 {g['sm_busy_us']['median']} / max {g['sm_busy_us']['max']}，"
                  f"最忙与最闲相差 {g['sm_busy_spread_us']} us（负载不均的直接指标）")
    print("错误：" + (json.dumps(sm["errors"], ensure_ascii=False) if sm["errors"] else "无"))
    print("事件              次数    合计us     均值us    中位us    最大us")
    for k, v in sm["events"].items():
        print(f"{k:<16}{v['n']:>6}{v['total_us']:>11.1f}{v['mean_us']:>10.2f}{v['p50_us']:>10.2f}{v['max_us']:>10.2f}")
    cu = sm.get("cu") or {}
    if cu:
        print(f"CU 归组：用到 {cu['count']} 个物理 CU，同一 CU 上最多 {cu['max_bands']} 条并列条带")
    print(("角色（按 CTA）" if sm.get("gran") == 1 else ("角色（按 wave）" if sm.get("platform") == "amd" else "角色（按 warp）")) +
          "     CTA数   最早结束us   中位结束us   最晚结束us   拖尾us(最晚-中位)")
    for k, v in sm["roles"].items():
        print(f"{k:<14}{v['ctas']:>6}{v['end_min_us']:>13.1f}{v['end_median_us']:>13.1f}{v['end_max_us']:>13.1f}{v['tail_us']:>14.1f}")
    if sm["timer"] == "globaltimer" and sm["timer_min_step_ns"] and sm["timer_min_step_ns"] >= 500:
        print(f"提示：globaltimer 实测步长 {sm['timer_min_step_ns']} ns 偏粗（消费级 GPU 常见为 1024 ns），细粒度事件请改编 -DFI_TIMER=1 用 clock64")


def _load_blob(slots, n_slots, cap, wpc, timer, meta):
    import os, tempfile
    cnt = [len(slots.get(s, [])) for s in range(n_slots)]
    blob = b"FILITE02" + struct.pack("<6I", n_slots, cap, wpc, 0, timer, len(meta)) + meta + struct.pack(f"<{n_slots}I", *cnt)
    for s in range(n_slots):
        for r in slots.get(s, []):
            blob += struct.pack("<QII", *r)
    fd, p = tempfile.mkstemp(suffix=".bin")
    os.write(fd, blob)
    os.close(fd)
    try:
        return read_dump(p)
    finally:
        os.remove(p)


def selftest():
    """CPU 自检：手工构造转储（不是测量数据），验证解码、配对、错误计数、clock64 换算、aux 配对与子泳道。"""
    # 事件 ID 一律从语义 KB 现取：configs/semantic_ids.json 的 policy 声明
    # event_id_is_authoritative，解码按 KB 的 event_id -> semantic_id 定名，names 里
    # 的标签只参与 alias 匹配。硬编码 ID 会与权威表错位（KB: load=1 wait=2 mma=3
    # barrier=7），使自检必然以 KeyError 失败，且 KB 改动时无法被自检发现。
    _ev = {s["semantic_id"]: int(s["event_id"]) for s in load_kb()["semantics"]
           if s.get("event_id") is not None}
    E_LOAD, E_MMA, E_BAR, E_COMM = _ev["load"], _ev["mma"], _ev["barrier"], _ev["communication"]
    wpc, cap, n_slots = 2, 16, 4
    meta = json.dumps({"selftest": True}).encode()
    slots = {0: [(1000, INIT | 0, 5), (2000, E_LOAD, 0), (5000, END | E_LOAD, 0),
                 (6000, E_MMA, 7), (9000, END | E_MMA, 7)],
             1: [(1500, INIT | 0, 5), (3000, MARK | E_BAR, 1)],
             2: [(1200, INIT | 1, 9), (4000, E_LOAD, 3), (4500, E_MMA, 3),
                 (4800, END | E_LOAD, 3)]}  # slot2：交叉嵌套 + 未配对
    names = {"events": {str(E_LOAD): "load", str(E_MMA): "mma", str(E_BAR): "barrier"},
             "roles": {"0": "r0", "1": "r1"}, "aux": {"2": "chunk"}}
    tr, sm = decode(_load_blob(slots, n_slots, cap, wpc, 0, meta), names)
    X = [e for e in tr["traceEvents"] if e["ph"] == "X"]
    assert sm["events"]["load"]["n"] == 2 and sm["events"]["mma"]["n"] == 1, sm["events"]
    assert abs(sm["events"]["load"]["max_us"] - 3.0) < 1e-9 and abs(sm["events"]["mma"]["total_us"] - 3.0) < 1e-9
    assert set(sm["errors"]) == {"交叉嵌套（结束的不是最内层；若是多实例并发请声明 slot）", "未配对的开始事件"}, sm["errors"]
    assert sm["errors"]["未配对的开始事件"] == 1
    assert sm["roles"]["r0"]["ctas"] == 1 and sm["roles"]["r1"]["ctas"] == 1
    assert sm["timer_min_step_ns"] == 300 and len(X) == 3
    mma_x = next(e for e in X if e["args"]["semantic_id"] == "mma")
    assert mma_x["name"] == "mma" and mma_x["args"]["display_name"] == "mma"
    assert mma_x["args"]["aux"] == 7 and "wave_slot" in mma_x["args"]
    assert sm["lanes_with_sublanes"] == 0
    # clock64 模式：开锚点 (g=10000, k=0)，关锚点 (g=12000, k=4000) => 0.5 ns/周期（2000 MHz）；周期 1000..3000 => 10500..11500 ns
    ck = {0: [(10000, OPEN | 0, 1), (0, CLK, 0), (1000, E_LOAD, 0), (3000, END | E_LOAD, 0),
              (12000, CLOSE, 0), (4000, CLK, 1)]}
    tr2, sm2 = decode(_load_blob(ck, n_slots, cap, wpc, 1, meta), names)
    x = [e for e in tr2["traceEvents"] if e["ph"] == "X"][0]
    assert abs(x["ts"] - 0.5) < 1e-9 and abs(x["dur"] - 1.0) < 1e-9, x
    assert sm2["clock_mhz_per_lane"] == {"min": 2000.0, "max": 2000.0} and not sm2["errors"], sm2
    # pair=aux：同一 warp 三个 prep 实例同时发出（B 1,2,3），完成顺序 2,1,3；还有一个嵌套在外的"块计算"
    px = {0: [(1000, INIT | 0, 5), (1000, E_COMM, 0), (2000, E_LOAD, 1), (2000, E_LOAD, 2),
              (2000, E_LOAD, 3), (5000, END | E_LOAD, 2), (6000, END | E_LOAD, 1),
              (7000, END | E_LOAD, 3), (8000, END | E_COMM, 0), (9000, E_LOAD, 4),
              (9500, END | E_LOAD, 4)]}
    names_aux = {"events": {str(E_LOAD): "prep 装载", str(E_COMM): "块计算"}, "roles": {"0": "链"},
                 "pair": "aux", "instances": {str(E_LOAD): 3}}
    tr3, sm3 = decode(_load_blob(px, n_slots, cap, wpc, 0, meta), names_aux)
    assert not sm3["errors"], sm3["errors"]
    assert sm3["events"]["load"]["n"] == 4 and abs(sm3["events"]["load"]["total_us"] - (3 + 4 + 5 + 0.5)) < 1e-9, sm3["events"]
    X3 = [e for e in tr3["traceEvents"] if e["ph"] == "X"]
    inst = {(e["args"]["aux"]): e["args"].get("instance") for e in X3 if e["args"]["semantic_id"] == "load"}
    assert inst == {1: 2, 2: 3, 3: 1, 4: 2}, inst  # 块计算贪心占 0 号；prep 子泳道 = 1 + aux % 3
    assert sm3["lanes_with_sublanes"] == 1 and sm3["max_sublanes"] == 4, (sm3["lanes_with_sublanes"], sm3["max_sublanes"])
    tn = {e["tid"]: e["args"]["name"] for e in tr3["traceEvents"] if e["ph"] == "M" and e["name"] == "thread_name"}
    assert any("实例2" in v for v in tn.values()), tn
    # 没给 instances 时贪心分配：3 个重叠区间应占 3 个子泳道
    names_aux2 = {"events": {str(E_LOAD): "prep 装载", str(E_COMM): "块计算"}, "roles": {"0": "链"},
                  "pair": "aux"}
    tr4, sm4 = decode(_load_blob(px, n_slots, cap, wpc, 0, meta), names_aux2)
    assert sm4["max_sublanes"] == 4 and not sm4["errors"], sm4  # 块计算 + 3 个重叠 prep，贪心共 4 条
    # pair=aux 的错误：没有同 aux 的开始
    bad = {0: [(1000, INIT | 0, 5), (2000, E_LOAD, 1), (3000, END | E_LOAD, 2)]}
    _, sm5 = decode(_load_blob(bad, n_slots, cap, wpc, 0, meta), names_aux2)
    assert sm5["errors"] == {"多余的结束事件（没有同 (事件,aux) 的开始）": 1, "未配对的开始事件": 1}, sm5["errors"]
    # ---- HIP / AMD 用例（与上面 NVIDIA 用例同一套断言口径）----
    # s_memrealtime：meta.timer_hz=100 MHz => 1 tick = 10 ns；aux 是 HW_ID 编码
    hmeta = json.dumps({"timer_hz": 100000000, "arch": "gfx942", "gran": 0}).encode()
    hslots = {0: [(100, OPEN | 0, (2 << 12) | (1 << 8) | 5), (200, E_LOAD, 0),
                  (500, END | E_LOAD, 0), (600, CLOSE, 0)]}
    tr6, sm6 = decode(_load_blob(hslots, 1, 8, 1, 0, hmeta), names)
    x6 = [e for e in tr6["traceEvents"] if e["ph"] == "X"][0]
    assert abs(x6["dur"] - 3.0) < 1e-9 and abs(x6["ts"] - 1.0) < 1e-9, x6  # (500-200)*10ns=3us
    assert sm6["timer"] == "s_memrealtime" and sm6["arch"] == "gfx942" and sm6["platform"] == "amd", sm6
    assert sm6["lane_kind"] == "wave" and sm6["gran"] == 0, sm6
    # FI_TIMER=1（readcyclecounter）+ timer_hz：锚点是 s_memrealtime tick，要先换算再拟合
    ameta = json.dumps({"timer_hz": 1000000000, "arch": "gfx942"}).encode()
    aslots = {0: [(10000, OPEN | 0, 1), (0, CLK, 0), (1000, E_LOAD, 0), (3000, END | E_LOAD, 0),
                  (12000, CLOSE, 0), (4000, CLK, 1)]}
    tr6b, sm6b = decode(_load_blob(aslots, 1, 8, 1, 1, ameta), names)
    x6b = [e for e in tr6b["traceEvents"] if e["ph"] == "X"][0]
    assert abs(x6b["ts"] - 0.5) < 1e-9 and abs(x6b["dur"] - 1.0) < 1e-9, x6b
    assert sm6b["clock_mhz_per_lane"] == {"min": 2000.0, "max": 2000.0}, sm6b["clock_mhz_per_lane"]
    # group=cu：两个 CTA 落在同一物理 CU（aux: XCC=2,SE=1,CU=5），区间重叠 => 2 个条带；另一个 CU 只有 1 个
    a1 = (2 << 12) | (1 << 8) | 5      # CU id = 2*64+1*16+5 = 149
    a2 = (2 << 12) | (1 << 8) | 6      # CU id = 150
    gmeta = json.dumps({"timer_hz": 1000000000, "arch": "gfx942", "gran": 1}).encode()
    gslots = {0: [(1000, OPEN | 0, a1), (2000, E_LOAD, 0), (6000, END | E_LOAD, 0), (7000, CLOSE, 0)],
              1: [(3000, OPEN | 0, a1), (4000, E_LOAD, 0), (5000, END | E_LOAD, 0), (6500, CLOSE, 0)],
              2: [(1000, OPEN | 0, a2), (2000, E_MMA, 0), (3000, END | E_MMA, 0), (4000, CLOSE, 0)]}
    gnames = {"events": {str(E_LOAD): "装载", str(E_MMA): "计算"}, "roles": {"0": "链"}}  # 不给 group：AMD gran=1 默认 cu
    tr7, sm7 = decode(_load_blob(gslots, 3, 8, 1, 0, gmeta), gnames)
    assert sm7["group"] == "cu" and sm7["cu"]["count"] == 2 and sm7["cu"]["max_bands"] == 2, sm7.get("cu")
    X7 = [e for e in tr7["traceEvents"] if e["ph"] == "X"]
    # tid 是 1-based：Perfetto 把 tid=0 当"未指定"并解析成 pid，会把该泳道合并到 pid
    # 主轨道上、与其上区间互相重叠，触发 slice_spill_overlapping_complete_event。
    assert {(e["pid"], e["tid"]) for e in X7} == {(149, 1), (149, 2), (150, 1)}, [(e["pid"], e["tid"]) for e in X7]
    pn7 = {e["pid"]: e["args"]["name"] for e in tr7["traceEvents"] if e["ph"] == "M" and e["name"] == "process_name"}
    assert pn7[149].startswith("CU149") and "XCC2" in pn7[149] and "SE1" in pn7[149], pn7
    args7 = [e["args"] for e in X7][0]
    assert args7["xcc"] == 2 and args7["se"] == 1 and args7["cu"] in (5, 6), args7
    # NVIDIA Triton 前端的转储：backend=nvidia_ptx、arch=sm_103a，但 meta 里也写了 timer_hz=1e9。
    # 标签必须是 globaltimer，且 tick 不能被"换算"（fi_bugs_b3-sk-qwen B17）
    assert _is_amd({"backend": "nvidia_ptx", "arch": "sm_103a", "timer_hz": 1000000000}) is False
    assert _is_amd({"arch": "sm_90"}) is False and _is_amd({"arch": "gfx942"}) is True
    assert _is_amd({"timer_hz": 99996900}) is True            # 老 AMD 转储：只有 timer_hz
    assert _is_amd({"backend": "amd_amdgcn"}) is True
    tri_meta = json.dumps({"fi_version": "fast-instrument-triton", "language": "Triton", "gran": 0,
                           "timer": 0, "level": 2, "arch": "sm_103a", "backend": "nvidia_ptx",
                           "timer_hz": 1000000000, "kernel": "mm_persistent_fi"}).encode()
    tri_slots = {0: [(1000, OPEN | 0, 3), (1032, 1, 0), (2032, 0x80000000 | 1, 0), (2064, 0x08000000, 0)]}
    d9 = _load_blob(tri_slots, 1, 8, 1, 0, tri_meta)
    assert d9["amd"] is False and d9["platform_source"] == "meta.backend", (d9["amd"], d9["platform_source"])
    _, sm9 = decode(d9, {"events": {"1": "计算"}, "roles": {"0": "链"}})
    assert sm9["timer"] == "globaltimer" and sm9["platform"] == "nvidia", (sm9["timer"], sm9["platform"])
    assert sm9["timer_min_step_ns"] == 32, sm9["timer_min_step_ns"]       # 1000/1032/2032/2064 未被换算
    assert "timer_hz_ignored" in sm9["meta"], sm9["meta"].get("timer_hz_ignored")
    assert sm9["meta"]["timer_hz_ignored"].startswith("NVIDIA 转储"), sm9["meta"]["timer_hz_ignored"]

    # 老转储（无 timer_hz/arch/gran 字段）仍按 NVIDIA globaltimer + role 归组解
    oldmeta = json.dumps({"T": 14631, "B": 1}).encode()
    oldslots = {0: [(1000, OPEN | 0, 5), (2000, E_LOAD, 0), (5000, END | E_LOAD, 0)]}
    tr8, sm8 = decode(_load_blob(oldslots, 1, 8, 1, 0, oldmeta), names)
    assert sm8["timer"] == "globaltimer" and sm8["gran"] == 0 and sm8["group"] == "role", (sm8["timer"], sm8["gran"], sm8["group"])
    assert sm8["platform"] == "nvidia" and sm8["lane_kind"] == "warp", sm8
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--names")
    ap.add_argument("-o", "--out", help="输出路径或目录：推荐直接传目录或 .png（传目录/auto 时自动按 <model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png 命名）；写 .gz 需显式加 --write-gz")
    ap.add_argument("--png", help="直接渲染 4-panel SW+HW 时间线 PNG（传 'auto' 或目录路径时自动命名）")
    ap.add_argument("--model", default=None, help="模型名（用于自动生成 PNG 文件名）")
    ap.add_argument("--op", default=None, help="算子名，如 gemm / attn / moe（用于自动生成 PNG 文件名）")
    ap.add_argument("--bs", default=None, help="batch size（用于自动生成 PNG 文件名）")
    ap.add_argument("--seql", default=None, help="sequence length（用于自动生成 PNG 文件名）")
    ap.add_argument("--kv", default=None, help="KV 长度（用于自动生成 PNG 文件名）")
    ap.add_argument("--shape", default=None, help="显式尺寸标签（覆盖 bs/seql 自动推导）")
    ap.add_argument("--title", default=None, help="PNG 标题（可选）")
    ap.add_argument("--sm", type=int, default=None, help="Panel (C) 展示的物理 SM/CU 编号")
    ap.add_argument("--dpi", type=int, default=220, help="PNG DPI（默认 220）")
    ap.add_argument("--allow-no-hw", action="store_true",
                    help="仅离线调试使用：允许在缺少同次硬件利用率曲线（nsys / rocprof-sys）时渲染无 (A) 栏 PNG；默认严格拒绝")
    ap.add_argument("--write-gz", action="store_true",
                    help="显式允许写出调试用 .perfetto.json.gz（默认 PNG-only 策略禁止写出 .gz）")
    ap.add_argument("--summary")
    ap.add_argument("--allow-errors", action="store_true")
    ap.add_argument("--gran", choices=["auto", "warp", "sm", "cu"], default="auto",
                    help="auto=按转储 meta 的 gran 字段（老转储没有＝warp/wave）；sm/cu 都强制成 gran=1（每 CTA 一条泳道）")
    ap.add_argument("--group", choices=["auto", "role", "sm", "cu", "cta"], default="auto",
                    help="进程归组：auto=NVIDIA SM 粒度按物理 SM、AMD 按物理 CU、warp/wave 粒度照 names.json 的 group")
    ap.add_argument("--timer-hz", type=int,
                    help="AMD s_memrealtime 的频率（Hz）；转储 meta 里没写 timer_hz 时用它换算 tick->ns")
    ap.add_argument("--platform", choices=["auto", "nvidia", "amd"], default="auto",
                    help="auto=按 meta.arch（gfx* 即 AMD；缺 arch 但有 timer_hz 也按 AMD）；可强制指定")
    ap.add_argument("--no-sm-bar", action="store_true", help="SM 粒度时不加\"SM 占用\"整条")
    ap.add_argument("--kb", help="canonical semantic KB JSON；默认使用包内 configs/semantic_ids.json")
    ap.add_argument("--require-semantics", action="store_true",
                    help="任何事件缺少唯一 canonical semantic_id 时按结构错误 fail-closed")
    ap.add_argument("--cta-cu-map",
                    help="权威 CTA->CU JSON（weave-fi-cta-cu-map/1）；覆盖 group=cu 的 sm/pid 归组")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.dump and a.names and (a.out or a.png)):
        ap.error("需要 dump、--names，以及 -o <out_dir|out.png> 或 --png <auto|out.png>")
    png_out = a.png
    trace_out = a.out
    if trace_out and (
        trace_out.lower().endswith(".png")
        or trace_out.strip().lower() == "auto"
        or os.path.isdir(trace_out)
        or trace_out.endswith(("/", "\\"))
    ):
        png_out = trace_out
        trace_out = None
    allow_gz = bool(a.write_gz or os.environ.get("WEAVEPROF_ALLOW_GZ") == "1")
    if trace_out:
        if trace_out.lower().endswith(".gz") or not trace_out.lower().endswith(".json"):
            if not allow_gz:
                raise SystemExit(
                    f"WeaveProfiler 默认仅交付 PNG，禁止生成 {trace_out}；"
                    "请改为 `-o <out_dir|timeline.png>` 或 `--png auto`（若确需调试用 gzip，请显式加 --write-gz）。"
                )
        elif not allow_gz and not png_out:
            raise SystemExit(
                f"WeaveProfiler 默认仅交付 PNG，不再自动改写为 {trace_out}.gz；"
                "请改为 `-o <out_dir|timeline.png>` 或 `--png auto`（若确需调试用 JSON/gzip，请显式加 --write-gz）。"
            )
    names = json.load(open(a.names, encoding="utf-8"))
    cta_cu_map = None
    cta_hw_map = None
    cta_cu_map_sha256 = None
    if a.cta_cu_map:
        map_doc = json.load(open(a.cta_cu_map, encoding="utf-8"))
        if map_doc.get("status") != "OK":
            raise SystemExit(f"CTA->CU map status is {map_doc.get('status')!r}")
        cta_cu_map = map_doc.get("cta_to_cu") or {}
        cta_hw_map = map_doc.get("cta_hw") or {}
        cta_cu_map_sha256 = _file_sha256(a.cta_cu_map)
    d = read_dump(a.dump, timer_hz=a.timer_hz, platform=(None if a.platform == "auto" else a.platform))
    source_dump_sha256 = _file_sha256(a.dump)
    names_sha256 = _file_sha256(a.names)
    decoder_sha256 = _file_sha256(os.path.abspath(__file__))
    if a.gran != "auto":  # 转储 meta 没写 gran（老转储）或需要强制换口径时用
        d["gran"] = 1 if a.gran in ("sm", "cu") else 0
    kb = load_kb(a.kb)
    tr, sm = decode(d, names, group_override=(None if a.group == "auto" else a.group),
                    sm_bar=not a.no_sm_bar, semantic_kb=kb,
                    require_semantics=a.require_semantics,
                    cta_cu_map=cta_cu_map, cta_hw_map=cta_hw_map)
    print_summary(sm)
    # Byte-level input lineage lets the aligner safely reuse this decode instead
    # of decoding the same multi-million-record dump a second time.
    sm["source_dump_sha256"] = source_dump_sha256
    sm["source_dump_path"] = os.path.abspath(a.dump)
    sm["names_sha256"] = names_sha256
    sm["decoder_sha256"] = decoder_sha256
    if a.cta_cu_map:
        sm["cta_cu_map_path"] = os.path.abspath(a.cta_cu_map)
        sm["cta_cu_map_sha256"] = cta_cu_map_sha256
        sm["cta_cu_source"] = "FILITE02 OPEN aux"
    if sm["errors"] and not a.allow_errors:
        print("有结构错误，未写文件；确认后加 --allow-errors 强制输出", file=sys.stderr)
        sys.exit(2)
    if png_out:
        from weaveprof.viz import render_swhw_overview_png
        rendered = render_swhw_overview_png(
            tr, png_out, title=a.title, focus_sm=a.sm, dpi=a.dpi,
            require_hw_curves=not a.allow_no_hw,
            model=a.model, op=a.op, bs=a.bs, seql=a.seql, kv=a.kv, shape=a.shape,
        )
        print("写出 PNG", rendered)
    if trace_out:
        if allow_gz and not trace_out.endswith(".gz") and trace_out.endswith(".perfetto.json"):
            trace_out += ".gz"
        write_json(tr, trace_out)
        print("写出", trace_out)
    if a.summary or trace_out:
        sum_path = a.summary or (trace_out.replace(".gz", "").replace(".json", "") + ".summary.json")
        write_json(sm, sum_path)
    sys.exit(2 if sm["errors"] else 0)


if __name__ == "__main__":
    main()
