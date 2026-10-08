# -*- coding: utf-8 -*-
"""P1 / M2 —— 磁带重放器（replay engine）。

消费 M1 捕获器（``_poc/_capture.py``）产出的磁带 JSON，按 **DAG 拓扑**把
``dec.forward_br.chain_br`` 段的算子序列**重放到真实 ``BatchRunner``** 上，
以 commit 边界分批提交。

设计要点（与 ``_diag/a26q_tape_impl.md`` §2/§3/§6 对齐）
-------------------------------------------------------
* **S0 装载**：读 JSON，校验 ``schema == "rvc.tape.v1"``。
* **S1 建 DAG**：条目字段 ``a`` / ``b`` / ``c``（或 ``in_bids``）若指向
  先前某条的 ``out_bid`` ⇒ 依赖边。**0 = 空槽**（ffi 语义），不是依赖。
* **S2 分批**：按 ``commits[].op_ix`` 切批，**逐批 flush**（镜像真实
  ``_forward_br`` 的 commit 边界，见蓝图规则 R3）。
* **S3 分派**：``name → BatchRunner 方法`` 表；用捕获的 ``out_bid`` 作为
  引擎缓冲语义的**替身**（同磁带形状重算 ⇒ 真实方法产出真实 buffer）。
* **S4 校验**：重放结束检查所有 ``in_bids`` 均已定义，否则抛
  ``TapeShapeError`` 语义的错误清单，**绝不裸崩**。

两种模式
--------
* ``--dry-run``（**默认**）：纯规划 —— 建 DAG + 拓扑校验 + 分批，
  **零 GPU**（不 import torch/vulkan，不建 context）。
* 不给 ``--dry-run``：真机重放，需要 GPU。

CLI::

    python _poc/_replay.py <tape.json> [--dry-run] [--out <out.json>]

作者：P1 重放工程师（a26aj / M2）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

# --- 路径引导（与 _capture.py 同一套路） --------------------------------
# ★ 只插入 _ROOT（项目根）——**绝不**插入 _diag：_diag\_poc 有历史残留
#   （旧版 _capture/_replay），会让 ``import _poc`` 解析到旧代码导致
#   ImportError（a26aw RS 实测踩坑）。_poc 包必须解析到项目根下的真身。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPLAY_SCHEMA = "rvc.tape.v1"
REPLAY_SURFACE = "dec.forward_br.chain_br"

# 空槽哨兵：ffi.zig 语义，buffer id 0 表示"无"（无 bias / 无该输入）。
NULL_BID = 0

# =====================================================================
# S3 分派表：op name → (BatchRunner 方法名, ps 语义槽名)
# ---------------------------------------------------------------------
# ps 布局严格按 `_diag/_check_alias.py:OP_TABLE` 的 note 字段（引擎权威）。
# =====================================================================

#: dec.forward_br.chain_br 段实际出现的 op（M1 磁带实测 5 类）
_FORWARD_TABLE = {
    # conv1d:  c=bias(0=无) p9=out
    #          p0=B p1=C_in p2=L p3=C_out p4=K p5=stride p6=pad_l p7=pad_r p8=dil
    "conv1d": dict(method="conv1d", kind="conv1d",
                   ps=("B", "C_in", "L", "C_out", "K",
                       "stride", "pad_l", "pad_r", "dil", "out_bid")),
    # conv_t1d: c=bias(0=无) p9=out
    #           p0=B p1=C_in p2=L p3=C_out p4=K p5=stride p6=padding p7=output_padding p8=dil
    # ★ R11 修正（a26ar M2 返工）：磁带 op5 走的是 **非分段** `conv_transpose1d`
    #   路径（`_capture.py` 经 `BatchRunner._record` 捕获 op=5，ps 恰 10 项，
    #   无 seg_len/l_seg/in_off/lo_off/l_out_full）。原先误映射到 16 参
    #   `conv_t1d_seg` ⇒ "missing 6 required positional args" 4 条。
    #   权威依据：`_diag/_check_alias.py:OP_TABLE[5]` 的 note，与
    #   `runtime/vulkan_ops.py:conv_transpose1d` 形参逐一对应。
    "conv_t1d": dict(method="conv_transpose1d", kind="conv_t1d",
                     ps=("B", "C_in", "L", "C_out", "K",
                         "stride", "padding", "output_padding", "dil",
                         "out_bid")),
    # add_inplace: p0=n；a 就地 += b
    "add_inplace": dict(method="add_inplace", kind="elementwise",
                        ps=("n",)),
    # leaky_relu: p0=n p1=slope(f32 位模式★ 绝不 remap)
    "leaky_relu": dict(method="leaky_relu", kind="elementwise",
                       ps=("n", "slope_bits")),
    # copy: ★特殊 —— a=dst b=src c恒0 p0=n
    "copy": dict(method="copy", kind="elementwise",
                 ps=("n",)),
}

#: 位置模式位（**绝不**当作 buffer id 去 remap）—— INV-4 白名单
BITMODE_SLOTS = {("leaky_relu", "slope_bits")}

#: 就地 op（输出写回 a 槽）
INPLACE_OPS = {"add_inplace", "leaky_relu"}


# =====================================================================
# 错误类型
# =====================================================================
class TapeShapeError(ValueError):
    """磁带结构/形状违反 —— 与 ``_diag/_check_alias.py`` 同名同义。"""


# =====================================================================
# 规范化算子视图
# =====================================================================
class Op:
    """磁带中一条算子记录（规范化视图）。"""

    __slots__ = ("ix", "name", "op_id", "a", "b", "c", "ps", "out_bid",
                 "in_bids", "inplace", "commit_ix", "out_shape", "shape",
                 "carrier", "kind", "raw")

    def __init__(self, ix, raw):
        self.ix = ix
        self.name = raw.get("name")
        self.op_id = raw.get("op_id")
        self.a = raw.get("a")
        self.b = raw.get("b")
        self.c = raw.get("c")
        self.ps = list(raw.get("ps") or [])
        self.out_bid = raw.get("out_bid")
        self.in_bids = list(raw.get("in_bids") or [])
        self.inplace = bool(raw.get("inplace"))
        self.commit_ix = raw.get("commit_ix")
        self.out_shape = raw.get("out_shape")
        self.shape = raw.get("shape")
        self.carrier = raw.get("carrier")
        self.kind = raw.get("kind")
        self.raw = raw            # ★ RS Phase1：原始 dict（含 out_num 快照元数据）

    # -- 依赖抽取（INV：0 是空槽，不是依赖） -------------------------
    def deps(self):
        """返回该 op 的**上游 buffer 依赖**列表（已剔空槽）。"""
        out = []
        for bid in self.in_bids:
            if bid is None:
                continue
            bid = int(bid)
            if bid == NULL_BID:
                continue
            out.append(bid)
        return out

    def produces(self):
        """该 op 产出的 buffer id（None ⇒ 本 op 不产出）。"""
        return None if self.out_bid is None else int(self.out_bid)

    def n_elems(self):
        sh = self.out_shape or []
        n = 1
        for x in sh:
            n *= int(x)
        return n

    def ps_map(self):
        """把 ps 列表按分派表的槽名解释为 dict（越界/缺失 ⇒ None）。"""
        spec = _FORWARD_TABLE.get(self.name)
        if spec is None:
            return {}
        out = {}
        for i, slot in enumerate(spec["ps"]):
            out[slot] = self.ps[i] if i < len(self.ps) else None
        return out

    def __repr__(self):
        return "<Op#%d %s a=%s b=%s c=%s out=%s>" % (
            self.ix, self.name, self.a, self.b, self.c, self.out_bid)


# =====================================================================
# S0 —— 装载
# =====================================================================
def load_tape(path):
    """读磁带 JSON 并做**最小**结构校验。返回 (tape_dict, [Op, ...])。"""
    if not os.path.isfile(path):
        raise TapeShapeError("磁带文件不存在: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        tape = json.load(fh)

    if not isinstance(tape, dict):
        raise TapeShapeError("磁带顶层应为 object，实为 %s" % type(tape).__name__)
    schema = tape.get("schema")
    if schema != REPLAY_SCHEMA:
        raise TapeShapeError("schema 不匹配: 期望 %r 实得 %r"
                             % (REPLAY_SCHEMA, schema))

    raw_entries = tape.get("entries")
    if not isinstance(raw_entries, list):
        raise TapeShapeError("entries 应为 list，实为 %s"
                             % type(raw_entries).__name__)
    if not raw_entries:
        raise TapeShapeError("entries 为空 —— 无可重放内容")

    ops = []
    bad = []
    for i, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            bad.append("entries[%d] 应为 object，实为 %s" % (i, type(raw).__name__))
            continue
        if not raw.get("name"):
            bad.append("entries[%d] 缺 name 字段" % i)
            continue
        ops.append(Op(i, raw))
    if bad:
        raise TapeShapeError("磁带条目结构非法（%d 条）:\n  - %s"
                             % (len(bad), "\n  - ".join(bad[:20])))

    return tape, ops


def tape_commits(tape, n_ops):
    """抽取 commit 边界（**op_ix 是"该 commit 覆盖到第几条（不含）"**）。

    返回排序去重后的 op_ix 列表；末位强制补齐 ``n_ops``（保证完整覆盖）。
    """
    raw = tape.get("commits") or []
    cuts = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        ix = c.get("op_ix")
        if ix is None:
            continue
        ix = int(ix)
        if ix <= 0 or ix > n_ops:
            raise TapeShapeError("commit.op_ix 越界: %d（n_ops=%d）" % (ix, n_ops))
        cuts.append(ix)
    cuts = sorted(set(cuts))
    if not cuts or cuts[-1] != n_ops:
        cuts.append(n_ops)
    return cuts


# =====================================================================
# S1 —— 建 DAG
# =====================================================================
def build_dag(ops):
    """按磁带顺序建依赖图。

    返回 ``(edges, defined_at, producers, externals, problems)``：

    * ``edges[i]``      —— op i 的上游 op 索引集合（拓扑序保证 < i）
    * ``defined_at``    —— ``bid -> 首次产出它的 op 索引``
    * ``producers``     —— ``bid -> [产出它的 op 索引]``（可多写，见 in-place）
    * ``externals``     —— 从未被任何 op 产出、却被读取的 bid（权重/输入常驻槽）
    * ``inplace_self``  —— 自读自写的就地槽（``bid -> [op 索引]``），合法
    * ``forward_refs``  —— **真·前向引用**（读取全部早于首次产出）⇒ DAG 破损
    * ``problems``      —— 结构问题清单（供 dry-run 报告）

    就地语义（关键）
    ----------------
    ``add_inplace`` / ``leaky_relu`` 的 ``out_bid == a``：op **读 a 槽并就地
    写回 a 槽**。因此该槽的"读取者"里**包含产生者自身**。这不是前向引用，
    而是**就地更新**（tape 里 225/308 条如此）。判定必须区分：

    * 读取者索引 == 首次产出索引 ⇒ 就地自更新，合法；
    * 读取者索引 **<**  首次产出索引 ⇒ 真·前向引用，**错误**。

    另注：**首次产出不等于"该槽诞生"**。前向段首个 op 的 ``a``/``b``/``c``
    常指向捕获前即存在的常驻槽（权重 / 模型输入），这些槽永不出现在任何
    ``out_bid`` 里（实测 175 个），故统一归入 ``externals``，合法。
    """
    edges = []
    defined_at = OrderedDict()
    producers = OrderedDict()
    readers = OrderedDict()
    first_use = OrderedDict()
    inplace_self = OrderedDict()
    problems = []

    for op in ops:
        ups = []
        for bid in op.deps():
            first_use.setdefault(bid, op.ix)
            if bid in defined_at:
                owner = defined_at[bid]
                # 自引用（就地 op：out_bid == a）不算边
                if owner != op.ix:
                    ups.append(owner)
            readers.setdefault(bid, []).append(op.ix)

        edges.append(ups)

        ob = op.produces()
        if ob is not None:
            producers.setdefault(ob, []).append(op.ix)
            if ob not in defined_at:
                defined_at[ob] = op.ix
            # 就地自更新：本 op 读了并且写了同一个槽
            if op.ix in readers.get(ob, ()) or ob in op.deps():
                inplace_self.setdefault(ob, []).append(op.ix)
        else:
            # 前向段实测 out_bid 均非 None；缺 out 只作告警
            problems.append("op#%d %s 无 out_bid（前向段应均有输出）"
                            % (op.ix, op.name))

    # ---- 复扫：三分归类 ----
    # 1) externals      : 全磁带从未被产出（权重/输入常驻槽）—— 合法
    # 2) inplace_self   : 被产出过，且存在"读取者==产出者"的就地更新 —— 合法
    # 3) forward_refs   : 存在"读取者 < 首次产出" —— **非法**
    forward_refs = OrderedDict()
    externals = OrderedDict()
    for bid, uses in first_use.items():
        if bid not in producers:
            externals[bid] = uses
            continue
        def_ix = defined_at[bid]
        early = [u for u in readers.get(bid, ()) if u < def_ix]
        if early:
            forward_refs[bid] = dict(first_use=uses, defined_at=def_ix,
                                     early_readers=early)

    return dict(edges=edges, defined_at=defined_at, producers=producers,
                readers=readers, externals=externals,
                inplace_self=inplace_self, first_use=first_use,
                forward_refs=forward_refs, problems=problems)


def dag_stats(ops, dag):
    """计算 DAG 指标：链深、拓扑序合法性、入度分布。"""
    edges = dag["edges"]
    depth = [0] * len(ops)
    outdeg = [0] * len(ops)
    for i, ups in enumerate(edges):
        depth[i] = (max((depth[u] for u in ups), default=-1) + 1)
        for u in ups:
            outdeg[u] += 1

    max_chain = max(depth) + 1 if depth else 0
    topo_ok = all(u < i for i, ups in enumerate(edges) for u in ups)
    in_deg = [len(u) for u in edges]
    roots = [i for i, d in enumerate(in_deg) if d == 0]
    leaves = [i for i, d in enumerate(outdeg) if d == 0]
    return dict(depth=depth, max_chain=max_chain, topo_ok=topo_ok,
                in_deg=in_deg, out_deg=outdeg, roots=roots, leaves=leaves)


# =====================================================================
# S2 —— 分批（commit 边界）
# =====================================================================
def make_batches(ops, cuts):
    """按 commit 边界把 op 列表切批。返回 ``[(lo, hi), ...]``。"""
    batches = []
    lo = 0
    for hi in cuts:
        if hi < lo:
            raise TapeShapeError("commit 边界非单调: %d < %d" % (hi, lo))
        batches.append((lo, hi))
        lo = hi
    if lo != len(ops):
        batches.append((lo, len(ops)))
    return batches


def cross_batch_edges(edges, batches):
    """统计跨批依赖边（前向段应尽量为 0；非 0 仅是告警，因引擎有全局屏障）。"""
    owner = {}
    for bi, (lo, hi) in enumerate(batches):
        for i in range(lo, hi):
            owner[i] = bi
    cross = []
    for i, ups in enumerate(edges):
        for u in ups:
            if owner.get(u) != owner.get(i):
                cross.append((u, i, owner.get(u), owner.get(i)))
    return cross


# =====================================================================
# 显存峰值估算（静态；真机需实测）
# =====================================================================
def peak_estimate(ops):
    """静态估算 buffer 峰值。

    口径一（**保守上界**，等价"不模拟池 / 每逻辑输出独享"）：
        所有 op 输出元素数**累加** —— 上界，绝不低估。
    口径二（**去重**）：按 ``out_bid`` 去重后累加 —— 各槽只算一次。
    口径三（**活跃峰值**）：按最后一次使用量做区间扫描 —— 近似真实峰值。
    """
    total_bytes = 0
    per_bid = OrderedDict()

    for op in ops:
        n = op.n_elems()
        total_bytes += n * 4
        ob = op.produces()
        if ob is not None:
            per_bid[ob] = max(per_bid.get(ob, 0), n * 4)

    # 活跃峰值：每个 bid 从首次产出到最后一次被读，做扫描线
    last_use = OrderedDict()
    for op in ops:
        for bid in op.deps():
            last_use[bid] = op.ix
        ob = op.produces()
        if ob is not None and ob not in last_use:
            last_use[ob] = op.ix

    first_def = OrderedDict()
    for op in ops:
        ob = op.produces()
        if ob is not None and ob not in first_def:
            first_def[ob] = op.ix

    events = []
    for bid, lo in first_def.items():
        hi = last_use.get(bid, lo)
        nb = per_bid.get(bid, 0)
        if nb:
            events.append((lo, nb, +1))
            events.append((hi, nb, -1))
    events.sort(key=lambda e: (e[0], -e[2]))

    live = 0
    peak = 0
    for _ix, nb, delta in events:
        live += delta * nb
        peak = max(peak, live)

    return dict(
        alloc_total_bytes=total_bytes,
        unique_slots=len(per_bid),
        unique_bytes=sum(per_bid.values()),
        live_peak_bytes=peak,
    )


# =====================================================================
# S3 —— 真机重放执行（需要 GPU；本任务不运行）
# =====================================================================
class NameToMethod:
    """`name -> BatchRunner 方法` 表 + 参数装配。

    重放语义：**同磁带形状/拓扑的可重放执行** —— 用捕获的 ``out_bid`` 作为
    引擎 buffer-id 语义的**替身**，真实调用引擎方法，产出真实 buffer。
    目标**不是**逐位复现原 buffer id（那需要引擎侧 slot 池的完整复刻，
    属 T-5/V3 范畴）。
    """

    def __init__(self, br):
        self.br = br
        self.unknown = []       # 未知 op 名（用于报告）
        self.calls = 0

    def _bind(self, op, h):
        """把磁带 bid 解析为重放期的实际对象。"""
        a = h.get(op.a)
        b = h.get(op.b) if op.b else None
        c = h.get(op.c) if op.c else None
        return a, b, c

    def dispatch(self, op, h):
        """执行一条 op。返回若产出则写回 ``h[out_bid]``。"""
        spec = _FORWARD_TABLE.get(op.name)
        if spec is None:
            # 前向段未实测到的 op：显式记录，不静默跳过
            self.unknown.append((op.ix, op.name, op.op_id))
            return None

        m = spec["method"]
        a, b, c = self._bind(op, h)
        self.calls += 1

        if a is None and m in ("copy", "leaky_relu", "add_inplace"):
            # ★ 就地/拷贝 op 的 a 槽缺失且**首次出现即本 op 产出**（磁带只记录
            #   了它被就地更新，未记录其段外来源）—— 三类：
            #   * copy       : a=dst 预分配输出目标（既有补丁）
            #   * leaky_relu : a=就地输入，out_bid==a，自更新；519 实测即此
            #                  （磁带尾部：段外传入的上一层 buffer）
            #   * add_inplace: 同 leaky_relu（就地 +=）
            #   必须在 :465 防线之前补占位，否则就地链全断。
            #   （a26ar 真机四连修：copy 补丁+leaky 519 段外输入。）
            sh = None
            if b is not None:
                sh = getattr(b, "shape", None) or getattr(b, "out_shape", None)
            if sh is None:
                sh = op.out_shape
            if sh:
                import numpy as _np                                 # noqa: PLC0415
                a = _np.zeros(tuple(int(x) for x in sh), dtype=_np.float32)
                h[op.a] = a

        if a is None:
            # ★ R10 防线：外部槽供给缺失时必须显式报错，而不是让引擎收到
            #   None ⇒ dtype=object 的 TypeError（a26ap 实测 304 条）。
            raise TapeShapeError(
                "op#%d %s: 输入槽 a=%r 未在 handles 中解析"
                "（外部槽注入缺失？）" % (op.ix, op.name, op.a))

        if m == "conv1d":
            pm = op.ps_map()
            out = self.br.conv1d(a, b, c,
                                 stride=int(pm["stride"]),
                                 padding=(int(pm["pad_l"]), int(pm["pad_r"])),
                                 dilation=int(pm["dil"]))
        elif m == "conv_transpose1d":
            # ★ R11 修正：非分段真路径，7 参（x, w, b, stride, padding,
            #   output_padding, dilation）—— 与磁带 op5 的 10 项 ps 一一对应。
            pm = op.ps_map()
            out = self.br.conv_transpose1d(a, b, c,
                                           stride=int(pm["stride"]),
                                           padding=int(pm["padding"]),
                                           output_padding=int(pm["output_padding"]),
                                           dilation=int(pm["dil"]))
        elif m == "add_inplace":
            out = self.br.add_inplace(a, b)
        elif m == "leaky_relu":
            pm = op.ps_map()
            out = self.br.leaky_relu(a, _bits_to_f32(pm["slope_bits"]))
        elif m == "copy":
            # copy 语义特殊：磁带 a=dst b=src（dst 是**预分配输出槽**，
            # 引擎 batchAddCopy(dst, src, n) 显示指定目标）。
            # ★ 重放修复：dst 槽 a 若未在 handles 中（既非外部槽、也非任何
            #   op 产出——它是 copy 的目标槽），按 b 的 shape 先建 zeros 占位，
            #   使 a 有值、后续依赖 a 的就地链（leaky_relu/conv_t1d 等）可解析。
            if a is None and b is not None:
                import numpy as _np                                 # noqa: PLC0415
                bsh = getattr(b, "shape", None) or getattr(b, "out_shape", None)
                if bsh:
                    a = _np.zeros(tuple(int(x) for x in bsh), dtype=_np.float32)
                    h[op.a] = a
            out = self.br.copy(b)
        else:                                    # pragma: no cover
            raise TapeShapeError("分派表方法未实现: %s" % m)

        ob = op.produces()
        if ob is not None and out is not None:
            h[ob] = out
        return out


def _bits_to_f32(bits):
    """f32 位模式 -> python float（INV-4：位模式位绝不 remap，仅回转）。"""
    import struct
    if bits is None:
        return 0.1
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


# =====================================================================
# R10 —— 外部槽形状推断（**零 GPU**，纯磁带推导）
# =====================================================================
# 背景（a26ap M3-B D1）
# -------------------
# 磁带**只记录被产出槽**（``out_bid``）的形状，**不记录 175 个外部常驻槽**
# （权重 / 偏置 / 前向段首层输入）的任何形状。重放时若不给这些槽喂
# **同形状**的 handle，``NameToMethod._bind`` 的 ``h.get(op.a)`` 恒 None，
# 引擎收到 ``None`` ⇒ ``dtype=object`` 的 TypeError（实测 304 条）。
#
# 本阶段协议（a26al R3 / a26ap §6.2）
# ----------------------------------
# **只验形状/拓扑/有限性，不验数值** —— 外部槽用**同形状 float32 zeros
# 占位**即可，不违背协议。**数值位级验证属 RS 阶段**（需先补 ``out_*``
# 数值快照，见 a26q §4.2 欠账）。
#
# 推断口径（**全部来自磁带自有信息**，不引入外部常量）
# -------------------------------------------------
# 外部槽 B 的读取者 op 已知其几何参数（``ps``）与输出形状（``out_shape``）：
#
# * ``conv1d``   : ps=[B,C_in,L,C_out,K,stride,pad_l,pad_r,dil,out]
#   - 槽位 ``a``（激活输入）→ ``out_shape``（重算所得，非快照）
#   - 槽位 ``b``（权重）    → ``(C_in, C_out, K)``
#   - 槽位 ``c``（偏置）    → ``(C_out,)``；``c == 0`` 时无偏置（空槽）
# * ``conv_t1d`` : ps=[B,C_in,L,C_out,K,stride,padding,output_padding,dil,out]
#   - 同 conv1d 口径（引擎侧 w 布局同为 ``[C_in,C_out,K]``）
# * ``add_inplace`` / ``leaky_relu`` / ``copy`` 的 y/other 输入
#   → 与同 op 的 ``a``（就地目的槽）同形状
#
# 推断失败一律**显式登记**（``unresolved``），绝不静默给 0 维/标量 ——
# 那会让引擎在形状校验处报错，掩盖真实原因。


def _ext_role(op, bid):
    """判定外部槽 ``bid`` 在 ``op`` 中的角色：``'a'`` / ``'b'`` / ``'c'``。"""
    for slot in ("a", "b", "c"):
        if getattr(op, slot) == bid:
            return slot
    return None


def _infer_ext_shape(op, slot):
    """按读取者 op 推断外部槽形状。返回 ``(shape, rule)`` 或 ``(None, 原因)``。

    全部推导仅依赖磁带的 ``ps`` / ``out_shape`` —— **零外部常量**。
    """
    ps = op.ps or []
    osh = list(op.out_shape or ())

    if op.name in ("conv1d", "conv_t1d"):
        # ps: [B, C_in, L, C_out, K, ...]  —— 索引见 OP_TABLE note
        if len(ps) < 5:
            return None, "ps 长度 %d < 5，无法取 C_in/C_out/K" % len(ps)
        C_in, C_out, K = int(ps[1]), int(ps[3]), int(ps[4])
        if slot == "a":
            # 激活输入：与输出**同空间维**但**通道为 C_in** —— (B, C_in, L)
            # （a26ar 真机实测校正：误用 out_shape 会把 C_out 当激活通道，
            #   导致 conv 通道不匹配 x=(1,C_out,L) w=(C_out,C_in,K)）
            if len(ps) < 3:
                return None, "ps 长度 %d < 3，无法取 B/C_in/L" % len(ps)
            return (int(ps[0]), C_in, int(ps[2])), "conv:a→(B,C_in,L)"
        if slot == "b":
            # ★ 权重布局**按引擎实现**而非按直觉（a26ar 真机实测校正）：
            #   * `BatchRunner.conv1d`            : w = [C_out, C_in, K]
            #     （`runtime/vulkan_ops.py:2480` docstring + `C_out, C_in2, K
            #      = w.shape`；实机 "bias 长度 512 != C_out 192" 亦证）
            #   * `BatchRunner.conv_transpose1d`  : w = [C_in, C_out, K]
            #     （PyTorch conv_transpose1d 布局；docstring 明示）
            # 磁带 ps 的 C_in/C_out 语义**两者一致**（p1=C_in, p3=C_out），
            # 仅权重排布不同。
            if op.name == "conv1d":
                return (C_out, C_in, K), "conv1d:b→(C_out,C_in,K)"
            return (C_in, C_out, K), "conv_t1d:b→(C_in,C_out,K)"
        if slot == "c":
            # 偏置长度 == C_out（两种 conv 一致；实机校验 `b_arr.shape[0]
            # != C_out` 即此）
            return (C_out,), "conv:c→(C_out,)"
        return None, "未知槽位 %r" % (slot,)

    if op.name == "add_inplace":
        # a 就地 += b；b 与 a 同形
        if not osh:
            return None, "无 out_shape"
        return tuple(osh), "add_inplace:b→out_shape"

    if op.name == "copy":
        # a=dst b=src；src 与 dst 同形
        if not osh:
            return None, "无 out_shape"
        return tuple(osh), "copy:b→out_shape"

    if op.name == "leaky_relu":
        if not osh:
            return None, "无 out_shape"
        return tuple(osh), "leaky_relu→out_shape"

    return None, "op %s 无外部槽推断规则" % op.name


def infer_external_shapes(ops, dag=None):
    """推断全部外部常驻槽的形状（**零 GPU**）。

    返回 ``(shapes, unresolved, rules)``：

    * ``shapes``     —— ``bid -> tuple(shape)``（可推断者）
    * ``unresolved`` —— ``[(bid, op_ix, op_name, slot, 原因), ...]``
    * ``rules``      —— ``bid -> 推断规则名``（证据留档）

    同一槽被多个 op 读取时取**首个可推断者**；若多个规则给出的形状冲突，
    记入 ``unresolved``（不静默取一）。
    """
    if dag is None:
        dag = build_dag(ops)
    ext = set(dag["externals"])

    shapes = OrderedDict()
    rules = OrderedDict()
    unresolved = []

    for op in ops:
        for slot in ("a", "b", "c"):
            bid = getattr(op, slot)
            if not bid:
                continue
            bid = int(bid)
            if bid not in ext:
                continue
            sh, rule = _infer_ext_shape(op, slot)
            if sh is None:
                # 仅当该槽仍未解析且尚无其它记录时才登记（去噪）
                if bid not in shapes:
                    unresolved.append((bid, op.ix, op.name, slot, rule))
                continue
            if bid in shapes:
                if tuple(shapes[bid]) != tuple(sh):
                    unresolved.append(
                        (bid, op.ix, op.name, slot,
                         "形状冲突: 已有 %s，本次 %s（%s）"
                         % (tuple(shapes[bid]), tuple(sh), rule)))
                continue
            shapes[bid] = tuple(sh)
            rules[bid] = rule

    # 去重 unresolved（同 bid 只留首条）
    seen = set()
    dedup = []
    for row in unresolved:
        if row[0] in seen and row[0] in shapes:
            continue
        seen.add(row[0])
        dedup.append(row)
    unresolved = [r for r in dedup if r[0] not in shapes]

    return shapes, unresolved, rules


def inject_externals(br, handles, ext_shapes, zeros_fn=None):
    """把外部槽以**同形状 float32 zeros 占位**注入 ``handles``。

    本阶段协议：**只验形状/拓扑/有限性**，故 zeros 占位合法。
    引擎侧 ``BatchRunner._resolve_input`` 接受 numpy 数组并按形状上传
    （``runtime/vulkan_ops.py:2309`` ``_as_f32`` + ``upload``），因此
    ``np.zeros(shape, np.float32)`` 会成为一个**真实 GPU buffer**，
    后续 op 拿到的是有效 BatchTensor 形状来源（非 ``None``）。

    返回 ``(n_injected, failures)``。
    """
    if zeros_fn is None:
        import numpy as _np                                 # noqa: PLC0415
        zeros_fn = lambda s: _np.zeros(s, dtype=_np.float32)  # noqa: E731

    n = 0
    failures = []
    for bid, sh in ext_shapes.items():
        try:
            if not sh:
                failures.append((bid, "空形状，跳过"))
                continue
            handles[bid] = zeros_fn(tuple(int(x) for x in sh))
            n += 1
        except Exception as exc:                            # noqa: BLE001
            failures.append((bid, "%s: %s" % (type(exc).__name__, exc)))
    return n, failures


def inject_externals_inw(br, handles, ext_shapes, in_w):
    """★ RS Phase2（a26ay 方向 A）：外部槽**优先注入真实权重快照**。

    ``in_w`` = ``{int bid: np.ndarray}``（``_poc._capture.load_inw`` 产出）。
    命中：直接给该 bid 的真实数值；形状与磁带推断不符 → 记 failure 并
    回退 zeros（防错位静默传播）。缺失：回退同形状 zeros（优雅降级，与
    ``inject_externals`` 语义一致）。

    返回 ``(n_injected, n_real, failures)``。
    """
    if not in_w:
        n0, f0 = inject_externals(br, handles, ext_shapes)
        return n0, 0, f0
    import numpy as _np                                     # noqa: PLC0415
    n = 0
    n_real = 0
    failures = []
    for bid, sh in ext_shapes.items():
        try:
            sh = tuple(int(x) for x in sh)
            arr = in_w.get(int(bid))
            if arr is None:
                arr = _np.zeros(sh, dtype=_np.float32)
            elif tuple(arr.shape) != sh:
                failures.append((bid, "形状不符: %s vs 磁带推断 %s，回退 zeros"
                                 % (tuple(arr.shape), sh)))
                arr = _np.zeros(sh, dtype=_np.float32)
            else:
                n_real += 1
            handles[bid] = arr
            n += 1
        except Exception as exc:                            # noqa: BLE001
            failures.append((bid, "%s: %s" % (type(exc).__name__, exc)))
    return n, n_real, failures


def replay_real(tape_path, ops=None, verbose=True, in_w=None):
    """真机重放：装载磁带 -> 建 context -> 逐批 flush。

    ``in_w``：``{int bid: np.ndarray}`` 真实外部槽数值（RS Phase2 a26ay
    方向 A；默认 None = 全 zeros 占位，Phase1 行为完全不变）。

    ★ 路径约定（a26ay 真机修复）：``in_w`` 字典由 ``load_inw`` 产出，
    而 ``load_inw(p)`` 内部会 ``p + ".inw.npz"``；因此 CLI 传参必须先
    归一化（见 ``_normalize_inw_arg``），否则 ``--in-w x.inw.npz`` 会
    被拼成 ``x.inw.inw.npz`` → 静默空 dict → 回退 zeros（A/B 无差异）。
    **本任务不执行本函数**（GPU 窗口归父代理）。真机协议见
    ``_diag/a26aj_m2_replay.md`` §4。
    """
    import numpy as np                                  # noqa: F401,PLC0415
    from runtime.vulkan_ops import BatchRunner, get_context   # noqa: PLC0415

    if ops is None:
        tape, ops = load_tape(tape_path)
    else:
        with open(tape_path, "r", encoding="utf-8") as fh:
            tape = json.load(fh)

    ctx = get_context()
    cuts = tape_commits(tape, len(ops))
    batches = make_batches(ops, cuts)
    dag = build_dag(ops)

    br = BatchRunner(ctx)
    disp = NameToMethod(br)
    handles = {}            # bid -> BatchTensor / 常驻对象
    produced = {}           # bid -> 形状清单（校验用）
    errs = []
    t0 = time.time()

    # ---- ★ RS Phase1：out_num 数值对照状态 ----
    import numpy as _np                                     # noqa: PLC0415
    from _poc._capture import load_out_num                  # noqa: PLC0415
    rs = []                      # 数值对照记录（见循环内）
    rs_by_ix = {}                # op.ix -> {snap: 缓存快照}（只加载一次）

    def _as_flat(obj):
        """把 BatchTensor / ndarray 展平为 f32 1-D（失败 None）。"""
        try:
            if isinstance(obj, _np.ndarray):
                return _np.asarray(obj, dtype=_np.float32).reshape(-1)
            if hasattr(obj, "numpy"):
                return _np.asarray(obj.numpy(), dtype=_np.float32).reshape(-1)
            return None
        except Exception:                                   # noqa: BLE001
            return None

    # ---- ★ R10 修复：注入 175 个外部常驻槽（权重/偏置/首层输入） ----------
    # 磁带不记录其形状 ⇒ 按读取者 op 的 ps/out_shape 推导（零 GPU 纯磁带）。
    # 本阶段用**同形状 float32 zeros 占位**（只验形状/拓扑；数值属 RS 阶段）。
    ext_shapes, ext_unresolved, ext_rules = infer_external_shapes(ops, dag)
    if in_w:
        # ★ RS Phase2：a26ay 方向 A —— 外部槽注入真实权重快照（by bid）
        n_ext_injected, _n_real, ext_failures = inject_externals_inw(
            br, handles, ext_shapes, in_w)
    else:
        n_ext_injected, ext_failures = inject_externals(br, handles, ext_shapes)
    if verbose:
        print("[R10] 外部槽注入: %d/%d 推断成功, 已注入 %d"
              % (len(ext_shapes), len(dag["externals"]), n_ext_injected))
        if ext_unresolved:
            print("[R10] 未解析外部槽: %d 条（首 5）" % len(ext_unresolved))
            for row in ext_unresolved[:5]:
                print("      bid=%s op#%s %s.%s: %s" % row)
        if ext_failures:
            print("[R10] 占位创建失败: %d 条（首 5）%s"
                  % (len(ext_failures), ext_failures[:5]))

    try:
        for bi, (lo, hi) in enumerate(batches):
            for op in ops[lo:hi]:
                try:
                    out = disp.dispatch(op, handles)
                except Exception as exc:                # noqa: BLE001
                    errs.append("seg%d op#%d %s: %s: %s"
                                % (bi, op.ix, op.name, type(exc).__name__, exc))
                    continue
                if out is not None:
                    produced[op.out_bid] = list(getattr(out, "shape", ()) or ())
                    # ★ 中间产出必须回写 handles，供后续 op 的 a/b/c 引用
                    handles[op.out_bid] = out
                    # ★ RS Phase1：out_num 数值对照（快照 vs 重放本步产出）
                    #   Phase1 范围：仅「机制忠实」验证——快照存在则加载并与
                    #   本步重放数值做 maxdiff 记录（不判 PASS/FAIL，供报告）。
                    #   完整位级 PASS 判定需真实权重外部槽（后续阶段）。
                    if isinstance(op.raw, dict) and op.raw.get("out_num"):
                        try:
                            _snap = load_out_num(tape_path, op.ix, op.raw)
                        except Exception:                # noqa: BLE001
                            _snap = None
                        if _snap is not None:
                            _got = _as_flat(out)
                            if _got is not None:
                                n = min(_snap.size, _got.size)
                                _md = None
                                if n > 0:
                                    _md = float(_np.max(_np.abs(
                                        _got.reshape(-1)[:n]
                                        - _snap.reshape(-1)[:n])))
                                _sh_ok = (
                                    tuple(getattr(out, "shape", ()) or ())
                                    == tuple(op.raw.get("out_shape") or ()))
                                rs.append(dict(
                                        ix=op.ix, name=op.name,
                                        elem_snap=int(_snap.size),
                                        elem_got=int(_got.size),
                                        maxdiff=_md,
                                        shape_ok=bool(_sh_ok),
                                    ))
            # 批次边界：flush（镜像真实 _forward_br 的 commit 位置）
            br.commit()
    finally:
        try:
            br.release()
        except Exception:                              # noqa: BLE001
            pass

    # ---- S4 收尾校验：所有 in_bids 必须已定义 ----
    defined = set(handles) | set(dag["externals"])
    undefined = []
    for op in ops:
        for bid in op.deps():
            if bid not in defined:
                undefined.append((op.ix, op.name, bid))

    # ---- 有限性抽检（B 组口径：形状/拓扑/有限性；**不验数值**） ----
    nonfinite = []
    try:
        import numpy as _np                                  # noqa: PLC0415
        for bid, obj in list(handles.items()):
            if isinstance(obj, _np.ndarray):
                if not bool(_np.isfinite(obj).all()):
                    nonfinite.append(bid)
    except Exception:                                        # noqa: BLE001
        pass

    return dict(
        n_ops=len(ops),
        n_batches=len(batches),
        n_calls=disp.calls,
        unknown_ops=disp.unknown,
        errors=errs,
        undefined_after_replay=undefined,
        produced_shapes=produced,
        elapsed=time.time() - t0,
        # ---- ★ R10/R11 证据段（a26ar M2 返工） ----
        n_externals_declared=len(dag["externals"]),
        n_externals_inferred=len(ext_shapes),
        n_externals_injected=n_ext_injected,
        externals_unresolved=ext_unresolved,
        externals_placeholder_failures=ext_failures,
        externals_rules=dict(ext_rules),
        externals_placeholder=dict(
            kind="zeros",
            dtype="float32",
            scope="shape/topology only",
            note=("本阶段外部槽用同形状 zeros 占位，**只验形状/拓扑/有限性**；"
                  "数值位级验证属 RS 阶段（需先补 out_* 数值快照）"),
        ),
        produced_shapes_count=len(produced),
        nonfinite_slots=nonfinite,
        # ---- ★ RS Phase1：out_num 数值对照记录（快照 vs 重放） ----
        rs_numsnap=rs,
        rs_numsnap_count=len(rs),
    )


# =====================================================================
# 依赖校验（dry-run 与真机共用口径）
# =====================================================================
def validate_deps(ops, dag, st):
    """校验全部 ``in_bids`` 可解析。返回 ``(ok, errors)``。

    口径（**与引擎 ffi 语义一致**）：

    * ``0``           —— 空槽（无 bias / 无该输入），**不算依赖**，跳过；
    * 被先前 op 产出 —— 合法依赖；
    * 从未被产出     —— 常驻槽（权重 / 模型输入），合法但计入 ``externals``；
    * 读取早于产出   —— **真·前向引用 ⇒ 错误**（DAG 破损）。
    """
    errors = []
    forward = dag["forward_refs"]
    for bid, info in forward.items():
        errors.append("前向引用: bid=%s 首次产出于 op%d，却被更早的 op%s 读取"
                      % (bid, info["defined_at"], info["early_readers"]))
    if not st["topo_ok"]:
        errors.append("拓扑序非法：存在 u >= i 的依赖边（非 DAG）")
    for msg in dag["problems"]:
        errors.append(msg)
    return (not errors), errors


# =====================================================================
# dry-run 编排（**零 GPU**）
# =====================================================================
def dry_run(tape_path, verbose=True):
    """纯规划：DAG 拓扑 + 依赖校验 + 分批；不 import GPU 任何东西。"""
    tape, ops = load_tape(tape_path)
    cuts = tape_commits(tape, len(ops))
    dag = build_dag(ops)
    st = dag_stats(ops, dag)
    batches = make_batches(ops, cuts)
    cross = cross_batch_edges(dag["edges"], batches)
    peak = peak_estimate(ops)

    deps_ok, dep_errs = validate_deps(ops, dag, st)
    unknown = sorted({op.name for op in ops if op.name not in _FORWARD_TABLE})

    plan = []
    for op in ops:
        spec = _FORWARD_TABLE.get(op.name)
        plan.append(dict(
            ix=op.ix,
            name=op.name,
            op_id=op.op_id,
            inplace=op.inplace,
            out_bid=op.out_bid,
            out_shape=op.out_shape,
            deps=op.deps(),
            deps_resolved=len(dag["edges"][op.ix]),
            method=(spec or {}).get("method"),
            commit_ix=op.commit_ix,
        ))

    return dict(
        tape=os.path.basename(tape_path),
        schema=tape.get("schema"),
        surface=tape.get("surface"),
        mode="dry-run",
        n_ops=len(ops),
        n_commits=len(cuts),
        commit_cuts=cuts,
        batch_sizes=[hi - lo for lo, hi in batches],
        deps_ok=bool(deps_ok),
        dep_errors=dep_errs,
        max_chain=int(st["max_chain"]),
        topo_ok=bool(st["topo_ok"]),
        n_edges=int(sum(len(u) for u in dag["edges"])),
        n_unique_out_bids=len(dag["producers"]),
        n_externals=len(dag["externals"]),
        n_inplace_self=len(dag["inplace_self"]),
        externals_sample=[dict(bid=b, first_use=u)
                          for b, u in list(dag["externals"].items())[:32]],
        n_cross_batch_edges=len(cross),
        cross_batch_sample=cross[:32],
        n_roots=len(st["roots"]),
        n_leaves=len(st["leaves"]),
        unknown_ops=unknown,
        buffers_peak_est=peak,
        plan=plan,
    )


def render_dry(rep, verbose=True):
    """把 dry-run 报告渲染成人读文本。"""
    L = []
    L.append("=" * 68)
    L.append("M2 磁带重放器 · dry-run 规划报告（零 GPU）")
    L.append("=" * 68)
    L.append("磁带        : %s" % rep["tape"])
    L.append("schema      : %s" % rep["schema"])
    L.append("surface     : %s" % rep["surface"])
    L.append("")
    L.append("[DAG 拓扑]")
    L.append("  n_ops              : %d" % rep["n_ops"])
    L.append("  n_edges            : %d" % rep["n_edges"])
    L.append("  max_chain          : %d" % rep["max_chain"])
    L.append("  topo_ok            : %s" % rep["topo_ok"])
    L.append("  n_unique_out_bids  : %d" % rep["n_unique_out_bids"])
    L.append("  n_externals(常驻槽): %d" % rep["n_externals"])
    L.append("  n_inplace_self     : %d（就地自更新槽，合法）" % rep["n_inplace_self"])
    L.append("  roots / leaves     : %d / %d" % (rep["n_roots"], rep["n_leaves"]))
    L.append("")
    L.append("[分批 / commit]")
    L.append("  n_commits          : %d" % rep["n_commits"])
    L.append("  commit_cuts        : %s" % rep["commit_cuts"])
    L.append("  batch_sizes        : %s" % rep["batch_sizes"])
    L.append("  跨批依赖边         : %d" % rep["n_cross_batch_edges"])
    L.append("")
    L.append("[依赖校验]")
    L.append("  deps_ok            : %s" % rep["deps_ok"])
    for e in rep["dep_errors"][:10]:
        L.append("    ! %s" % e)
    L.append("")
    L.append("[显存静态估算]")
    pk = rep["buffers_peak_est"]
    L.append("  全量分配累加       : %.2f MB（保守上界）"
             % (pk["alloc_total_bytes"] / 1048576.0))
    L.append("  去重槽合计         : %.2f MB（%d 槽）"
             % (pk["unique_bytes"] / 1048576.0, pk["unique_slots"]))
    L.append("  活跃峰值（扫描线） : %.2f MB <- 关键指标"
             % (pk["live_peak_bytes"] / 1048576.0))
    L.append("")
    L.append("[分派覆盖]")
    L.append("  表内 op 数         : %d / %d"
             % (rep["n_ops"], rep["n_ops"]))
    if rep["unknown_ops"]:
        L.append("  表外 op            : %s" % rep["unknown_ops"])
    else:
        L.append("  表外 op            : 无（全部命中分派表）")
    L.append("  计划条目数         : %d" % len(rep["plan"]))
    L.append("")
    if verbose:
        L.append("[plan 前 12 条]")
        for p in rep["plan"][:12]:
            L.append("  #%-4d %-12s out=%-5s shape=%-14s deps=%-16s m=%s"
                     % (p["ix"], p["name"], p["out_bid"],
                        str(p["out_shape"]), str(p["deps"]), p["method"]))
        L.append("")
    L.append("=" * 68)
    L.append("dry-run 结论: %s" % ("PASS" if rep["deps_ok"] else "FAIL"))
    L.append("=" * 68)
    return "\n".join(L)


# =====================================================================
# 自测（**零 GPU**）
# =====================================================================
def selftest(tape_path=None, verbose=True):
    """干跑自测：断言 DAG/分批/分派三面。**

    断言（对应任务书口径）：
      A1 ``deps_ok == True``
      A2 ``max_chain > 0``
      A3 ``len(plan) == 308``
      A4 ``n_commits == 6``
      A5 全部 op 命中分派表（前向段）
      A6 每个 op 的依赖要么解析到先前 op，要么是常驻槽
    """
    if tape_path is None:
        tape_path = os.path.join(_ROOT, "_diag", "cap_step00000001_m1cap.json")

    checks = []

    def chk(aid, title, ok, detail=""):
        checks.append((aid, title, bool(ok), detail))
        return ok

    try:
        rep = dry_run(tape_path, verbose=False)
    except TapeShapeError as exc:
        chk("A0", "磁带装载", False, str(exc))
        return dict(ok=False, checks=checks, report=None)
    except Exception as exc:                          # noqa: BLE001
        chk("A0", "磁带装载", False, "%s: %s" % (type(exc).__name__, exc))
        return dict(ok=False, checks=checks, report=None)

    chk("A0", "磁带装载", True, rep["schema"])
    chk("A1", "deps_ok == True", rep["deps_ok"] is True,
        "" if rep["deps_ok"] else "; ".join(rep["dep_errors"][:3]))
    chk("A2", "max_chain > 0", rep["max_chain"] > 0, "max_chain=%d" % rep["max_chain"])
    chk("A3", "len(plan) == 308", len(rep["plan"]) == 308,
        "实得 %d" % len(rep["plan"]))
    chk("A4", "n_commits == 6", rep["n_commits"] == 6,
        "实得 %d" % rep["n_commits"])
    chk("A5", "全部 op 命中分派表", not rep["unknown_ops"],
        "表外: %s" % rep["unknown_ops"])
    chk("A6", "拓扑序合法", rep["topo_ok"] is True, "")

    # A7：commit_ix 与 batch 划分一致性（不强求等于，仅报告）
    cix = sorted({p["commit_ix"] for p in rep["plan"] if p["commit_ix"] is not None})
    chk("A7", "commit_ix 值域合理", len(cix) <= rep["n_commits"],
        "distinct commit_ix=%s" % cix)

    # A8：峰值估算为正
    chk("A8", "峰值估算 > 0", rep["buffers_peak_est"]["live_peak_bytes"] > 0,
        "%.2f MB" % (rep["buffers_peak_est"]["live_peak_bytes"] / 1048576.0))

    # A9：每条 op 均有 out_bid（前向段实测特征）
    no_out = [p["ix"] for p in rep["plan"] if p["out_bid"] is None]
    chk("A9", "前向段每条 op 均有 out_bid", not no_out, "缺 out 的 op: %s" % no_out)

    ok = all(c[2] for c in checks)
    return dict(ok=ok, checks=checks, report=rep)


def render_selftest(res):
    L = []
    L.append("=" * 68)
    L.append("M2 重放器 · 零 GPU 自测（selftest）")
    L.append("=" * 68)
    for aid, title, ok, detail in res["checks"]:
        L.append("[%s] %-4s %s%s" % ("PASS" if ok else "FAIL", aid, title,
                                     ("  // " + detail) if detail else ""))
    L.append("-" * 68)
    L.append("总计: %d/%d PASS -> %s"
             % (sum(1 for c in res["checks"] if c[2]), len(res["checks"]),
                "PASS" if res["ok"] else "FAIL"))
    L.append("=" * 68)
    return "\n".join(L)


# =====================================================================
# 零 GPU 形状装配自测（a26ar M2 返工新增）
# =====================================================================
def selftest_shape(tape_path=None, verbose=True):
    """零 GPU 自测：外部槽推断 + R10/R11 装配口径（**不 import GPU**）。**

    断言：
      S1 推断出的外部槽数 == ``dag["externals"]`` 数（175）
      S2 无未解析外部槽
      S3 全部推断形状均为正整数元组（rank ≥ 1）
      S4 conv1d 权重槽形状 == ``(ps[1], ps[3], ps[4])``
      S5 conv1d 偏置槽形状 == ``(ps[3],)``
      S6 conv_t1d 的 4 条 op 其 ps 长度 == 10（**非** 16）⇒ 证伪「磁带是分段路径」
      S7 conv_t1d oL 公式自洽（``(L-1)*s-2p+d(K-1)+op+1 == out_shape[2]``）
      S8 zeros 占位注入全部成功且形状逐一匹配
      S9 模拟引擎 `_bind`：每条 op 的 a/b/c 均可在 handles 中解析（0 个 None）
    """
    if tape_path is None:
        tape_path = os.path.join(_ROOT, "_diag", "cap_step00000001_m3a.json")

    checks = []

    def chk(aid, title, ok, detail=""):
        checks.append((aid, title, bool(ok), detail))
        return ok

    try:
        tape, ops = load_tape(tape_path)
    except Exception as exc:                                # noqa: BLE001
        chk("S0", "磁带装载", False, "%s: %s" % (type(exc).__name__, exc))
        return dict(ok=False, checks=checks, report=None)

    dag = build_dag(ops)
    ext = dag["externals"]
    shapes, unresolved, rules = infer_external_shapes(ops, dag)

    chk("S1", "推断外部槽数 == externals 数",
        len(shapes) == len(ext),
        "inferred=%d externals=%d" % (len(shapes), len(ext)))
    chk("S2", "无未解析外部槽", not unresolved,
        "未解析 %d 条: %s" % (len(unresolved), unresolved[:3]))
    chk("S3", "全部形状为正整数元组且 rank>=1",
        all(isinstance(s, tuple) and len(s) >= 1
            and all(isinstance(d, int) and d > 0 for d in s)
            for s in shapes.values()),
        "样例: %s" % list(shapes.items())[:3])

    # S4/S5：抽一条 conv1d 权重/偏置核对
    conv_w = conv_b = None
    for op in ops:
        if op.name == "conv1d":
            for slot in ("b", "c"):
                bid = getattr(op, slot)
                if bid and int(bid) in shapes:
                    if slot == "b" and conv_w is None:
                        conv_w = (int(op.ps[1]), int(op.ps[3]), int(op.ps[4]),
                                  shapes[int(bid)])
                    if slot == "c" and conv_b is None:
                        conv_b = (int(op.ps[3]), shapes[int(bid)])
            if conv_w and conv_b:
                break
    chk("S4", "conv1d 权重槽 == (C_in,C_out,K)",
        conv_w is not None and conv_w[3] == conv_w[:3],
        "实得 %s" % (conv_w,))
    chk("S5", "conv1d 偏置槽 == (C_out,)",
        conv_b is not None and conv_b[1] == (conv_b[0],),
        "实得 %s" % (conv_b,))

    # S6/S7：conv_t1d 是真·非分段路径
    ct = [op for op in ops if op.name == "conv_t1d"]
    chk("S6", "conv_t1d ps 长度 == 10（非分段路径）",
        bool(ct) and all(len(op.ps) == 10 for op in ct),
        "n=%d ps_len=%s" % (len(ct), [len(op.ps) for op in ct]))

    formulas = []
    for op in ct:
        B, C_in, L, C_out, K, s, p, opad, d, _o = [int(x) for x in op.ps[:10]]
        oL = (L - 1) * s - 2 * p + d * (K - 1) + opad + 1
        formulas.append((op.ix, oL, int(op.out_shape[2])))
    chk("S7", "conv_t1d oL 公式自洽",
        bool(formulas) and all(a == b for _i, a, b in formulas),
        "%s" % formulas)

    # S8：zeros 占位注入
    injected, fails = inject_externals(None, {}, shapes)
    chk("S8", "zeros 占位注入全部成功",
        injected == len(shapes) and not fails,
        "injected=%d/%d fails=%s" % (injected, len(shapes), fails[:3]))

    # S9：模拟引擎 _bind —— 每条 op 的 a/b/c 均可解析
    handles = {}
    inject_externals(None, handles, shapes)
    missing = []
    for op in ops:
        for slot in ("a", "b", "c"):
            bid = getattr(op, slot)
            if not bid:
                continue                        # 0 = 空槽（ffi 语义），合法
            bid = int(bid)
            if bid in handles:
                continue
            if bid in dag["producers"]:
                continue                        # 由先前 op 产出，重放期填入
            missing.append((op.ix, op.name, slot, bid))
    chk("S9", "外部槽消费点 100% 可解析（0 个 None）",
        not missing, "缺 %d: %s" % (len(missing), missing[:5]))

    rep = dict(
        tape=os.path.basename(tape_path),
        n_externals=len(ext),
        n_inferred=len(shapes),
        unresolved=unresolved,
        rules_sample=dict(list(rules.items())[:12]),
        conv_t1d_ps_len=[len(op.ps) for op in ct],
        conv_t1d_ol_formulas=formulas,
        injected=injected,
        n_ops=len(ops),
        placeholder_note="zeros float32 占位仅用于形状档位；数值验证属 RS 阶段",
    )

    ok = all(c[2] for c in checks)
    return dict(ok=ok, checks=checks, report=rep)


def render_selftest_shape(res):
    L = []
    L.append("=" * 68)
    L.append("M2 重放器 · 零 GPU 形状装配自测（selftest-shape / a26ar）")
    L.append("=" * 68)
    for aid, title, ok, detail in res["checks"]:
        L.append("[%s] %-4s %s%s" % ("PASS" if ok else "FAIL", aid, title,
                                     ("  // " + detail) if detail else ""))
    rep = res.get("report") or {}
    if rep:
        L.append("-" * 68)
        L.append("外部槽: %d 声明 / %d 推断 / %d 未解析"
                 % (rep["n_externals"], rep["n_inferred"],
                    len(rep["unresolved"])))
        L.append("注入: %d 个 zeros 占位" % rep["injected"])
        L.append("conv_t1d ps 长度: %s（10 ⇒ 非分段路径，R11 证伪）"
                 % rep["conv_t1d_ps_len"])
        L.append("注: %s" % rep["placeholder_note"])
    L.append("-" * 68)
    L.append("总计: %d/%d PASS -> %s"
             % (sum(1 for c in res["checks"] if c[2]), len(res["checks"]),
                "PASS" if res["ok"] else "FAIL"))
    L.append("=" * 68)
    return "\n".join(L)


# =====================================================================
# CLI
# =====================================================================
def _normalize_inw_arg(p):
    """★ a26ay 真机修复：``--in-w`` 参数归一化为 ``load_inw`` 期望的基路径。

    ``load_inw(x)`` 内部做 ``inw_path_for(x) = x + ".inw.npz"``，所以：

    * 传 ``....inw.npz``      → 去掉后缀，返回裸基路径（存在则原样返回）
    * 传 ``....json``         → 原样（load_inw 会拼出 sidecar）
    * 传裸基路径（无扩展名）  → 原样

    兼容三种写法，避免 ``x.inw.inw.npz`` 静默空读。
    """
    import os as _os                                     # noqa: PLC0415
    s = str(p)
    if s.endswith(".inw.npz"):
        base = s[: -len(".inw.npz")]
        # 优先按裸基路径（load_inw 会拼回 .inw.npz）
        if _os.path.exists(base + ".inw.npz"):
            return base
        return s
    return s


def build_parser():
    ap = argparse.ArgumentParser(
        prog="_replay.py",
        description="RVC-Vulkan P1/M2 磁带重放器（消费 M1 磁带 JSON，按 DAG 分批重放）",
    )
    ap.add_argument("tape", help="M1 磁带 JSON 路径")
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="纯规划模式（DAG 拓扑 + 依赖校验），零 GPU；**默认行为**")
    ap.add_argument("--real", action="store_true", default=False,
                    help="真机重放（需 GPU；显式开启）")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="零 GPU 自测（断言 deps_ok / max_chain / 308 ops / 6 commits）")
    ap.add_argument("--selftest-shape", action="store_true", default=False,
                    help="零 GPU 形状装配自测（外部槽推断 + R10/R11 装配，a26ar）")
    ap.add_argument("--out", default=None, help="把结果 JSON 写到该路径")
    ap.add_argument("--in-w", dest="in_w", default=None,
                    help="RS Phase2（a26ay 方向 A）：真实外部槽 *.inw.npz sidecar "
                         "路径；提供后重放按 bid 注入真实权重快照（默认 zeros）")
    ap.add_argument("--quiet", action="store_true", default=False, help="只输出结论行")
    return ap


def _emit(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    # --selftest-shape：零 GPU 形状装配自测（a26ar）
    if args.selftest_shape:
        res = selftest_shape(args.tape)
        if not args.quiet:
            print(render_selftest_shape(res))
        else:
            print("SELFTEST-SHAPE %s" % ("PASS" if res["ok"] else "FAIL"))
        payload = dict(mode="selftest-shape", ok=res["ok"],
                       checks=[dict(id=a, title=t, ok=o, detail=d)
                               for a, t, o, d in res["checks"]],
                       report=res.get("report"))
        if args.out:
            _write_json(args.out, payload)
        return 0 if res["ok"] else 1

    # --selftest：优先
    if args.selftest:
        res = selftest(args.tape)
        if not args.quiet:
            print(render_selftest(res))
        else:
            print("SELFTEST %s" % ("PASS" if res["ok"] else "FAIL"))
        payload = dict(mode="selftest", ok=res["ok"],
                       checks=[dict(id=a, title=t, ok=o, detail=d)
                               for a, t, o, d in res["checks"]])
        if res["report"]:
            payload["report"] = res["report"]
        if args.out:
            _write_json(args.out, payload)
        return 0 if res["ok"] else 1

    # 真机重放：显式 --real 且不带 --dry-run
    if args.real and not args.dry_run:
        in_w = None
        if args.in_w:
            try:
                from _poc._capture import load_inw   # noqa: PLC0415,PLC2701
                _p = _normalize_inw_arg(args.in_w)
                in_w = load_inw(_p)
                if not in_w:
                    sys.stderr.write(
                        "[--in-w] 警告: %s 未读到任何 w_* 槽，将回退 zeros 占位\n"
                        % (args.in_w,))
            except Exception as exc:                  # noqa: BLE001
                sys.stderr.write("[--in-w] 加载失败: %r\n" % (exc,))
                return 2
        res = replay_real(args.tape, verbose=not args.quiet, in_w=in_w)
        if not args.quiet:
            skip = {"produced_shapes", "externals_rules"}
            print(json.dumps({k: v for k, v in res.items() if k not in skip},
                             ensure_ascii=False, indent=2))
        if args.out:
            _write_json(args.out, res)
        bad = (bool(res["errors"])
               or bool(res["undefined_after_replay"])
               or bool(res.get("externals_unresolved")))
        return 1 if bad else 0

    # 默认：dry-run
    rep = dry_run(args.tape, verbose=not args.quiet)
    if not args.quiet:
        print(render_dry(rep, verbose=not args.quiet))
    else:
        print("DRY-RUN deps_ok=%s ops=%d commits=%d max_chain=%d"
              % (rep["deps_ok"], rep["n_ops"], rep["n_commits"], rep["max_chain"]))
    if args.out:
        _write_json(args.out, rep)
    return 0 if rep["deps_ok"] else 1


def _write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def main(argv=None):
    try:
        return _emit(argv)
    except TapeShapeError as exc:
        sys.stderr.write("[TapeShapeError] %s\n" % exc)
        return 2
    except KeyboardInterrupt:                         # pragma: no cover
        sys.stderr.write("interrupted\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
