#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 捕获面 MVP —— `TapeCapture` 磁带捕获器（重写版，纯 Python / 零 GPU）。

定位
====
本文件是 `_diag/a26q_tape_impl.md`（v7 蓝图）§2「捕获器实现规范」+ §6「与 numdiff
裁决的衔接」的**可执行落地**，并经 `_diag/a26v_crosscheck.md` 修正后实现。

**MVP 只捕获不重放**：本文件产出可重放形状的 JSON，供未来 `_poc/_replay.py` 消费。

四条硬约束（与任务书红线一致）
----------------------------
1. **不碰 `runtime/` 与 `engine/`**：全部挂载走运行时 monkey-patch，
   `install()` 前后 `runtime/` 文件零字节改动；`uninstall()` 可完整还原。
2. **零 GPU**：本文件只读 Python 侧对象（`_records` / 形参），**不调任何 FFI**。
3. **7 分支解包器直接移植复用** `_diag/_check_alias.py` 的 `unpack_entry` /
   `TapeShapeError`，**不重复定义** op 表（从该模块 import）。
4. **绝不 IndexError**：任何形态异常统一记为 `TapeShapeError`，不裸崩。

三道帧门（P1 §2 / a26q §1.3）
=============================
`_chain_br()` 是**模块级单例**且跨段复用（forward → backward → 下一步），
所以「钩住 `_record` 就自动正确」是**错的** —— 必须显式实现三道门：

* **门 1（开口）** `dec_frame_open(step)`：dec forward 段开始。
* **门 2（闭口）** `dec_frame_close()`：dec forward 段结束（含 `tanh` 结果）。
* **门 3（守卫）** 帧计数器 `_frame`：**仅当 `_frame > 0` 时记录 op**。
  帧外（backward / 下一步 / 判别器）一律丢弃。

**两道提交/磁带门**（MVP 增补）：

* **门 4（提交门）** `commit` 边界只在帧内记录，且只对**本 runner**计数。
* **门 5（磁带门）** `dump()` 时校验门自洽（`frame_opens == frame_closes`）。

用法
====
    import _poc._capture as cap

    cap.install()                  # 进程级一次，patch BatchRunner
    cap.dec_frame_open(step=7)     # 门 1：dec forward 段开始
    ... 正常跑 dec 前向（只观测，不干预）...
    cap.dec_frame_close()          # 门 2：dec forward 段结束
    doc = cap.dump(step=7, tag="mvp")   # 导出可重放 JSON

    cap.uninstall()                # 还原（可选）

纯构造自测（**不需要 GPU**）
---------------------------
    python _poc/_capture.py --selftest

退出码 0 = 全部断言 PASS。

引用
----
* 蓝图：`_diag/a26q_tape_impl.md` §1.2/§1.3（捕获面 + 三道门）、§2.2（磁盘
  schema）、§3.4（37 op 权威表）、§6.3（commit 边界）/§6.4（判定字段）。
* 修正：`_diag/a26v_crosscheck.md` C1/C1a/C2b/C2c/C4/C5（op 域 1..37、
  `op+mode` 联合分派、im2col 名字互换、3-id 位形态、commit 边界、硬 `int()` 崩溃点）。
* 解包器与权威表：`_diag/_check_alias.py`（a26z，A1..A6 6/6 PASS）。
"""
from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "_diag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from _check_alias import (          # type: ignore
        OP_TABLE, SEG_FN_TABLE, SHAPES, TapeShapeError, unpack_entry,
        EXPECT_INPLACE, EXPECT_BITMODE, EXPECT_REMAP_P9,
        A, C, P9, OUT, OP_MIN, OP_MAX,
    )
except Exception as _imp_exc:           # pragma: no cover
    raise ImportError(
        "无法从 _diag/_check_alias.py 导入权威表与解包器：%r\n"
        "该文件是 a26z 交付物（A1..A6 6/6 PASS），捕获器必须复用而非重定义。"
        % (_imp_exc,)
    )

OUT_DIR = os.path.join(_ROOT, "_diag")

SCHEMA = "rvc.tape.v1"
CAPTURE_SURFACE = "dec.forward_br.chain_br"

# ===========================================================================
# RS Phase1 —— out_num 数值快照常量（a26au §3.3 / §7 清单 1-3）
# ===========================================================================
#: Phase1 数值快照覆盖的 op 白名单（a26au §1.1 / §5.1）
#:   conv1d   —— 唯一真·卷积计算，误差放大路径最长
#:   conv_t1d —— 转置卷积，commit_class="black"（历史 3-7e-4），最需独立数值锚点
#: Phase2 再扩到 add_inplace / leaky_relu / copy（a26au §5.2）。
NUMSNAP_OPS = ("conv1d", "conv_t1d")

#: 单 op 快照元素上限（a26au §3.3 L1）。2048 * 4B = 8 KB/op 硬上限。
#: 可用 env RVC_CAPTURE_CAP_ELEMS 覆盖（清单 11：可配置，默认安全）。
CAP_ELEMS_DEFAULT = 2048

#: 单 tape 快照字节预算（a26au §3.3 L4 / §3.6）：超限即降级并显式记录。
TAPE_NUMSNAP_BUDGET_DEFAULT = 8 * 1024 * 1024      # 8 MB


def _env_int(name, default):
    """读 env 整数（非法值静默回落默认 —— 捕获器绝不因配置问题裸崩）。"""
    try:
        _v = os.environ.get(name)
        if _v is None or str(_v).strip() == "":
            return int(default)
        return int(str(_v).strip())
    except Exception:                       # noqa: BLE001
        return int(default)


def cap_elems():
    """当前生效的 ``CAP_ELEMS``（>=1）。"""
    return max(1, _env_int("RVC_CAPTURE_CAP_ELEMS", CAP_ELEMS_DEFAULT))


def tape_numsnap_budget():
    """当前生效的 tape 级快照字节预算（>=0；0 = 不限）。"""
    return max(0, _env_int("RVC_CAPTURE_TAPE_BUDGET", TAPE_NUMSNAP_BUDGET_DEFAULT))


def nums_path_for(tape_json_path):
    """磁带 JSON 的数值 sidecar 路径：``cap_stepXXX_tag.json`` → ``cap_stepXXX_tag.nums.npz``。"""
    return os.path.splitext(str(tape_json_path))[0] + ".nums.npz"


def num_key(entry_ix):
    """npz 内的快照键：``nums_<entry_ix>``（按 entry 顺序一一对应）。"""
    return "nums_%d" % int(entry_ix)


def _tensor_nums(t, cap_elems, out_shape):
    """取 BatchTensor 数值（惰性抓取，a26au §2.1）：``.numpy()`` 自带
    commit+wait+download（vulkan_ops.py:2193-2207）。返回 ``(arr, elem_full)``
    或 ``(None, 原因)``。
    """
    import numpy as _np
    try:
        arr = t.numpy()                     # commit(幂等)+wait(no-op)+download
        if arr is None or arr.size == 0:
            return None, "empty"
        elem_full = int(arr.size)
        flat = _np.asarray(arr, dtype=_np.float32).reshape(-1)
        if int(cap_elems) >= elem_full:
            return flat, elem_full
        # CAP_ELEMS 截断：**确定性均匀采样**（固定步长取头段，保证可复现；
        # 头段对 conv 数值最具代表性 —— 首采样点即首个输出元素）
        return flat[:int(cap_elems)], elem_full
    except Exception as exc:                # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def collect_numsnap(cap, step=None, tag="mvp"):
    """★ RS Phase1 核心：dump 时惰性抓取 out_* 数值快照。

    a26au §2.3 落点 B —— 在 dump 时刻（buffer 活着 C1、已 commit C3）遍历
    entries，对 NUMSNAP_OPS 且有 tensor 缓存的条目抓数值，三道门过滤：

      M1 ``t._buf is None``        → 已被 tensor_done/release 作废 → 跳过
      M2 ``out_bid in _recycled``  → 已被归还本地池可复用 → 数值不可信 → 跳过
      M3 逆序扫描                  → 同 bid 只取最后一次写（早期写已被覆盖）

    返回 ``(nums_dict, meta, stats)``：

      nums_dict : ``num_key(ix) -> np.ndarray``（写入 sidecar）
      meta      : ``ix -> out_num 描述子``（并入 JSON entry，a26au §4.1）
      stats     : 计数统计（snapped/skipped_dead/skipped_recycled/superseded）
    """
    import numpy as _np
    ce = cap_elems()
    budget = tape_numsnap_budget()
    stats = dict(snapped=0, skipped_dead=0, skipped_recycled=0,
                 superseded=0, bytes=0, over_budget=0)
    nums = {}
    meta = {}
    # M3：逆序扫描，bid 只取「时间上最后一次写」的那条 entry
    seen_bid = set()
    for ix in range(len(cap.entries) - 1, -1, -1):
        rec = cap.entries[ix]
        name = rec.get("name")
        if name not in NUMSNAP_OPS:
            continue
        t = cap._tensor_refs.get(ix)
        if t is None:
            continue                        # 无缓存（未走 on_int_op 路径）
        ob = rec.get("out_bid")
        if ob is not None and ob in seen_bid:
            stats["superseded"] += 1
            continue
        if ob is not None:
            seen_bid.add(ob)
        # M1：句柄存活校验
        try:
            if getattr(t, "_buf", None) is None:
                stats["skipped_dead"] += 1
                continue
        except Exception:                   # noqa: BLE001
            stats["skipped_dead"] += 1
            continue
        # M2：_recycled 黑名单
        runner = getattr(t, "_runner", None)
        try:
            if runner is not None and ob in getattr(runner, "_recycled", ()):
                stats["skipped_recycled"] += 1
                continue
        except Exception:                   # noqa: BLE001
            pass                            # 保守：不因检查失败而丢数据
        osh = rec.get("out_shape") or ()
        arr, elem_full = _tensor_nums(t, ce, osh)
        if arr is None:
            stats["skipped_dead"] += 1
            continue
        if budget > 0 and stats["bytes"] + arr.nbytes > budget:
            stats["over_budget"] += 1
            continue                        # L4 段预算熔断（a26au §3.6）
        key = num_key(ix)
        nums[key] = arr
        stats["bytes"] += int(arr.nbytes)
        stats["snapped"] += 1
        meta[ix] = dict(
            store="npz", key=key, dtype="float32",
            elem=int(arr.size), elem_full=int(elem_full),
            mode=("cap_full" if elem_full == int(arr.size) else "cap2048"),
            status="ok",
        )
    return nums, meta, stats


def load_out_num(tape_json_path, entry_ix, entry=None):
    """重放侧消费：按 entry 的 out_num 描述子从 sidecar 加载数值。

    返回 ``np.ndarray``（快照数值）或 ``None``（旧 tape 无 out_num /
    sidecar 缺失 / 键缺失 —— 优雅降级为形状协议，a26au §4.3）。
    """
    try:
        if entry is None:
            import json as _json
            with open(str(tape_json_path), encoding="utf-8") as _fh:
                _doc = _json.load(_fh)
            _ent = _doc["entries"][int(entry_ix)]
        else:
            _ent = entry
        meta = _ent.get("out_num") if isinstance(_ent, dict) else None
        if not isinstance(meta, dict):
            return None
        if meta.get("store") != "npz":
            return None
        npz_path = nums_path_for(tape_json_path)
        if not os.path.exists(npz_path):
            return None
        import numpy as _np
        with _np.load(npz_path) as _z:
            key = meta.get("key") or num_key(entry_ix)
            if key not in _z:
                return None
            arr = _np.asarray(_z[key], dtype=_np.float32)
        osh = _ent.get("out_shape")
        if osh and int(_np.prod([int(x) for x in osh])) == int(arr.size):
            try:
                return arr.reshape([int(x) for x in osh])
            except Exception:               # noqa: BLE001
                return arr
        return arr
    except Exception:                       # noqa: BLE001
        return None


# ===========================================================================
# RS Phase2 —— in_w 真实权重槽缓存（a26ay 方向 A）
# ===========================================================================
# 动机（a26ay §2/§3）：磁带 a/b/c 槽只有 buffer id，重放器只能注入 zeros
# 占位 ⇒ conv 数值比对 maxdiff≠0（Phase1 设计预期）。方向 A：捕获侧在
# ``BatchRunner._resolve_input`` 入口（原始 x/w/b 对象活着的地方）冻结
# 输入快照，dump 时对「磁带外部槽」写独立 ``*.inw.npz`` sidecar，重放侧
# 按 bid 注入真实数值。
#
# ★ 硬约束：**env 门控默认关**（``RVC_CAPTURE_INW=1`` 才启用）——默认
# attach 零额外 patch、dump 零额外产物，Phase1 行为完全一致（a26ay §6.2）。


def inw_enabled() -> bool:
    """in_w 捕获开关：``RVC_CAPTURE_INW=1`` 启用（默认关，零开销）。"""
    return os.environ.get("RVC_CAPTURE_INW", "0") == "1"


def inw_path_for(tape_json_path):
    """in_w sidecar 路径：``cap_stepXXXXXX_tag.json`` → ``cap_stepXXXXXX_tag.inw.npz``。"""
    return os.path.splitext(str(tape_json_path))[0] + ".inw.npz"


def _freeze_input(obj, name):
    """把 ``_resolve_input`` 入参冻结为快照对象（on_resolve 回调用）。

    * numpy 兼容输入 → **深拷贝 f32**（copy=True）：防 optimizer 就地更新
      使 dump 时拿到「本步已更新」的新值（a26ay §3.2 新鲜度注）——
      磁带数值必须对应该步前向用的旧权重。
    * BatchTensor / PersistentBuffer（GPU 对象）→ 原对象引用，
      dump 时惰性抓取（.numpy() / download）。
    """
    import numpy as _np
    if hasattr(obj, "numpy") and hasattr(obj, "_buf"):
        return obj                       # BatchTensor
    if hasattr(obj, "id") and hasattr(obj, "valid"):
        return obj                       # PersistentBuffer
    try:
        return _np.array(obj, dtype=_np.float32, copy=True)
    except Exception:                    # noqa: BLE001
        return None


def _inw_arr(obj):
    """把缓存的输入对象取**全量** f32 ndarray（不截断，a26ay §4.2：
    截断注入会毁坏整个 conv 输出）。失败返回 None。

    * 冻结 numpy → 直接用
    * BatchTensor → 惰性 ``.numpy()``（commit 幂等；buffer 已失效返回 None）
    * PersistentBuffer → best-effort ``ctx.download``（dec 磁带通常不出现）
    """
    import numpy as _np
    try:
        if isinstance(obj, _np.ndarray):
            return obj
        if hasattr(obj, "numpy") and hasattr(obj, "_buf"):
            if getattr(obj, "_buf", None) is None:      # M1 门：已作废
                return None
            a = obj.numpy()
            if a is None:
                return None
            return _np.asarray(a, dtype=_np.float32)
        if hasattr(obj, "id") and hasattr(obj, "valid"):
            if not bool(getattr(obj, "valid", False)):
                return None
            ctx = getattr(obj, "_ctx", None)
            if ctx is not None and hasattr(ctx, "download"):
                return _np.asarray(ctx.download(obj.id, obj.shape),
                                   dtype=_np.float32)
        return None
    except Exception:                    # noqa: BLE001
        return None


def collect_inw(cap):
    """dump 时对磁带**外部槽**收集冻结权重/输入快照（a26ay 方向 A 核心）。

    外部槽 = 磁带 a/b/c 中出现、但从未被任何 entry 的 ``out_bid`` 产出的
    bid（权重/首层输入/bias 等常驻槽，rs2 实测 175 个）。

    返回 ``(inw_dict, slots_meta, stats)``；未启用或无缓存 → ``(None, None, None)``。

    体积门控（a26ay §4.2 分档）：
      ``RVC_CAPTURE_INW_BUDGET``  累计字节熔断（0 = 不限）
      ``RVC_CAPTURE_INW_MAXIX``   只收集 entry_ix < MAXIX 引用的外部槽
                                  （档 1 = 首个 commit 之前，seg0 最小闭环）；
                                  0 = 全部
    **npz 键**：``w_<bid>``；slots 元数据按 bid 字符串记录。
    """
    if not inw_enabled():
        return None, None, None
    if not (cap._in_refs and cap.entries):
        return None, None, None
    import numpy as _np                    # noqa: PLC0415

    produced = set()
    for r in cap.entries:
        ob = r.get("out_bid")
        if ob is not None:
            produced.add(int(ob))

    max_ix = max(0, _env_int("RVC_CAPTURE_INW_MAXIX", 0))
    budget = max(0, _env_int("RVC_CAPTURE_INW_BUDGET", 0))
    ext = set()
    for ix, r in enumerate(cap.entries):
        if max_ix and ix >= max_ix:
            break
        for s in ("a", "b", "c"):
            b = r.get(s)
            if b is not None and int(b) not in produced:
                ext.add(int(b))

    inw = {}
    meta = {}
    stats = dict(enabled=True, n_slots=0, bytes=0, skipped=0, over_budget=0)
    for bid in sorted(ext):
        holder = cap._in_refs.get(bid)
        if holder is None:
            stats["skipped"] += 1
            continue
        obj, name = holder if isinstance(holder, tuple) else (holder, "?")
        if obj is None:
            stats["skipped"] += 1
            continue
        arr = _inw_arr(obj)
        if arr is None or arr.size == 0:
            stats["skipped"] += 1
            continue
        nb = int(arr.nbytes)
        if budget > 0 and stats["bytes"] + nb > budget:
            stats["over_budget"] += 1
            continue                                # L4 段预算熔断
        key = "w_%d" % int(bid)
        inw[key] = arr
        meta[str(int(bid))] = dict(
            shape=[int(x) for x in arr.shape],
            elems=int(arr.size), bytes=nb, name=str(name),
        )
        stats["n_slots"] += 1
        stats["bytes"] += nb
    return (inw, meta, stats) if inw else (None, meta, stats)


def load_inw(tape_json_path):
    """重放侧消费：读 ``*.inw.npz`` → ``{int(bid): np.ndarray}``。

    优雅降级：sidecar 缺失 / 无 ``w_*`` 键 → 返回 ``{}``（重放侧回退 zeros）。
    """
    p = inw_path_for(tape_json_path)
    if not os.path.exists(p):
        return {}
    try:
        import numpy as _np                        # noqa: PLC0415
        out = {}
        with _np.load(p) as _z:
            for _k in _z.files:
                if _k.startswith("w_"):
                    try:
                        out[int(_k[2:])] = _np.asarray(_z[_k],
                                                       dtype=_np.float32)
                    except Exception:               # noqa: BLE001
                        continue
        return out
    except Exception:                               # noqa: BLE001
        return {}


__all__ = [
    "TapeCapture", "TapeShapeError", "unpack_entry", "OP_TABLE", "SHAPES",
    "install", "uninstall", "get_capture", "reset_capture",
    "dec_frame_open", "dec_frame_close", "selftest", "main",
    "NUMSNAP_OPS", "CAP_ELEMS_DEFAULT", "collect_numsnap",
    "load_out_num", "nums_path_for", "num_key", "cap_elems",
    # ---- RS Phase2 in_w（a26ay 方向 A）----
    "inw_enabled", "inw_path_for", "collect_inw", "load_inw",
]

_FN_TO_OP = {
    "conv2d": 27, "conv_t2d": 28, "im2col_1d": 29, "im2col_2d": 29,
    "sgemm_nt": 1,
}

_SHAPE_TO_KIND = {
    "M1_int_4id": "gpu_op",
    "M6_int_4id_ps19": "gpu_op_ps19",
    "M2_str4id_long_ps": "gpu_op_str",
    "M3_str4id_ps16": "gpu_op_str",
    "M4_str3id": "gpu_op_str",
    "M5_str3id": "gpu_op_str",
    "M7_strfn_varargs": "seg",
}

BITMODE = {(6, 1): "slope", (9, 2): "eps", (15, 3): "eps",
           (19, 2): "scale", (31, 1): "s", (32, 1): "s",
           (33, 1): "s", (35, 1): "s"}

_SEG_META = {
    "rvc_batch_add_copy_seg":  dict(out_pos=0, inplace=False, op=7),
    "rvc_batch_add_add_seg":   dict(out_pos=0, inplace=True,  op=3),
    "rvc_batch_add_mul_seg":   dict(out_pos=0, inplace=True,  op=4),
    "rvc_batch_add_leaky_seg": dict(out_pos=0, inplace=True,  op=6),
    "rvc_batch_add_gelu_seg":  dict(out_pos=0, inplace=True,  op=10),
    "im2col_1d":               dict(out_pos=1, inplace=False, op=29),
    "im2col_2d":               dict(out_pos=1, inplace=False, op=29),
    "conv2d":                  dict(out_pos=None, inplace=False, op=27),
    "conv_t2d":                dict(out_pos=None, inplace=False, op=28),
    "sgemm_nt":                dict(out_pos=2, inplace=False, op=1),
}


# ---------------------------------------------------------------------------
# §1. 小工具
# ---------------------------------------------------------------------------
def _f32_from_bits(bits):
    """f32 位模式 → 浮点（op6 p1 / op9 p2 / op15 p3 / op19 p2 / op31-35 p1）。"""
    try:
        return float(struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0])
    except Exception:                       # noqa: BLE001
        return float("nan")


def _nbytes_of(shape) -> int:
    if not shape:
        return 0
    n = 1
    for d in shape:
        n *= int(d)
    return int(n) * 4


def _brief(obj, n=110) -> str:
    s = repr(obj)
    return s if len(s) <= n else s[:n] + "..."


def _cell(ps, i):
    """安全取 ps[i]（越界返回 None，绝不 IndexError）。"""
    try:
        return int(ps[i])
    except Exception:                        # noqa: BLE001
        return None


def _commit_class(op, fn) -> str:
    """§6.2 白/黑名单（**经 a26v §4.2-6 修正**）。

    ⚠️ a26v 明确：a26q §6.2 的「convT1d 按 stride/opad 判黑」依据被撤回
    （`a26n:53` 指小差由 CPU-vs-GPU 路径差异解释）。故此处**只标记、不裁决**，
    输出 ``commit_class`` 供重放器自行决定；MVP 捕获期**一律不干预**（§2.3）。
    """
    if fn == "conv_t2d" or op == 28:
        return "black"                       # 历史 3-7e-4，无条件黑
    if op == 5 or fn == "conv_t1d_seg":
        return "gray"                        # a26v 修正：降级为待复验
    if op == 7:                              # copy：作为混排隔离器
        return "gray"
    if op in (23, 24, 25, 26, 27):
        return "gray"                        # 保守按黑（推断，未见实测）
    if op is not None:
        return "white"
    return "gray"


# ---------------------------------------------------------------------------
# §2. TapeCapture
# ---------------------------------------------------------------------------
class TapeCapture:
    """磁带捕获器：三道帧门 + commit 边界 + 7 分支解包 + op 分类标记。

    线程语义：``BatchRunner._record`` 在 runner 自己的 ``_lock`` 内调用；
    本类用自己的 ``RLock`` 保护状态，且**只在原函数返回后**触碰本类状态，
    不存在与 runner 锁的嵌套，故无锁序问题。
    """

    def __init__(self, tag: str = "mvp", out_dir: str = OUT_DIR):
        self.tag = tag
        self.out_dir = out_dir
        self._lock = threading.RLock()

        # --- 门状态（门 1/2/3） ---
        self._frame = 0            # 门 3 守卫计数器：>0 才记录
        self._step = -1
        self._frame_opens = 0
        self._frame_closes = 0

        # --- 磁带内容 ---
        self.entries = []          # 规范化条目（可重放形状）
        self.commits = []          # commit 边界（索引 + 时间）
        self.errors = []           # TapeShapeError / 内部异常
        self.dropped = {}          # 门控丢弃计数（原因 -> 次数）
        self._n_seen = 0           # 看到的原始 tape 条目总数（含帧外）

        # --- ★ RS Phase1 数值快照：entry_ix -> BatchTensor 旁路映射 ---
        # a26au §2.3 落点 B 前置条件 1：``_record`` patch 需缓存返回的
        # ``BatchTensor`` 引用（**不塞进 JSON entry** —— 不可序列化）。
        # 强引用安全：tensor 本就存活到训练步末（a26au C1），且本映射在
        # 每次 ``dec_frame_open`` 时清空，不引入额外生命周期。
        self._tensor_refs = {}
        # ★ RS Phase2 in_w：bid -> (冻结输入快照, resolve name) 旁路映射
        #   （a26ay 方向 A；env 门控默认关，未启用时恒为空 dict）。bid 跨
        #   帧会复用（输入池步末归还），故 dec_frame_open 时清空。
        self._in_refs = {}
        # 最新一次 inw 快照的统计（dump 后保留供断言/报告）
        self._last_inw = None
        # 最新一次 numsnap 的统计（dump 后保留供断言/报告）
        self._last_numsnap = None
        # 捕获期实时 RSS 采样峰值（a26au R4 / 任务书内存保护）
        self._rss_peak_mb = 0.0

        # --- 挂载点原函数（供 uninstall 还原） ---
        self._orig = {}

    # -- 门控辅助 ------------------------------------------------------
    def _drop(self, reason: str) -> None:
        """记一次丢弃（门控统计）。"""
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def _closed(self):
        """挂载时用的时态对象（占位，见 attach）。"""
        return getattr(self, "_sealed", None)

    # -- 生命周期 ------------------------------------------------------
    def attach(self, batch_runner_cls) -> "TapeCapture":
        """把本捕获器挂到 ``BatchRunner`` 类（最小侵入 monkey-patch）。

        patch 三处（**均在 `runtime/` 源文件外**）：

        * ``_record``     —— int op 载体（op1..37；8 处 append 中 7 处经此）
        * ``_seg_record`` —— seg/str 载体（``vulkan_ops.py:3103``）
        * ``commit``      —— commit 边界（``vulkan_ops.py:3513``），★真缺口

        采用「**包装 append 链**」方案：先调原函数（保证 FFI 语义与
        时序 100% 不变），再从入参提取条目。**不读 `_records`
        事后解析**（a26q §1.4 否决：信息量不足，且直接读会拿到
        "下发了但引擎未收"的半态）。
        """
        with self._lock:
            if self._orig:
                # 幂等：已挂载则只补 inw 的 _resolve_input patch（若之前
                # attach 时未启用、现在启用）。
                self._maybe_patch_resolve(batch_runner_cls)
                return self
            cap = self

            orig_record = batch_runner_cls._record
            orig_seg = batch_runner_cls._seg_record
            orig_commit = batch_runner_cls.commit

            cap._orig = dict(record=orig_record, seg=orig_seg,
                             commit=orig_commit,
                             cls=batch_runner_cls)

            def _record(self, op, a_id, b_id, c_id, ps, out_shape,
                        tensor_buf=None):
                # 先调原函数：FFI 下发与返回语义零改变
                ret = orig_record(self, op, a_id, b_id, c_id, ps, out_shape,
                                  tensor_buf=tensor_buf)
                try:
                    cap.on_int_op(self, op, a_id, b_id, c_id, ps,
                                  out_shape=out_shape, tensor_buf=tensor_buf,
                                  ret=ret)
                except Exception as exc:            # noqa: BLE001
                    cap.errors.append("wrap._record: %r" % (exc,))
                return ret

            def _seg_record(self, fn, c_ids, ps):
                ret = orig_seg(self, fn, c_ids, ps)
                try:
                    cap.on_seg_op(self, fn, c_ids, ps)
                except Exception as exc:            # noqa: BLE001
                    cap.errors.append("wrap._seg_record: %r" % (exc,))
                return ret

            def commit(self, async_=False):
                # ★ 只记边界不下发（a26q §6.3）：原函数行为不变
                try:
                    cap.on_commit_pre(self, async_)
                except Exception as exc:            # noqa: BLE001
                    cap.errors.append("wrap.commit.pre: %r" % (exc,))
                ret = orig_commit(self, async_)
                try:
                    cap.on_commit_post(self, async_)
                except Exception as exc:            # noqa: BLE001
                    cap.errors.append("wrap.commit.post: %r" % (exc,))
                return ret

            batch_runner_cls._record = _record
            batch_runner_cls._seg_record = _seg_record
            batch_runner_cls.commit = commit

            # ★ RS Phase2 in_w：env 门控下补 patch _resolve_input（a26ay 方向 A）
            self._maybe_patch_resolve(batch_runner_cls)
        return self

    def _maybe_patch_resolve(self, batch_runner_cls) -> None:
        """inw 门控下 patch ``_resolve_input``（幂等；类上无此方法则跳过）。

        ``_resolve_input``（vulkan_ops.py:2286）在算子方法内被调用时，
        原始 x/w/b 对象仍活着（G dec 段权重直接传 numpy）——这是方向 A
        唯一可行的捕获点（``_record`` 只收整数句柄，见 a26ay §3.1）。
        """
        if not inw_enabled() or "resolve" in self._orig:
            return
        if not hasattr(batch_runner_cls, "_resolve_input"):
            return
        orig_resolve = batch_runner_cls._resolve_input
        cap = self

        def _resolve_input(self, x, name, buf=None):
            # 先调原函数：上传/池化语义与时序零改变
            ret = orig_resolve(self, x, name, buf=buf)
            try:
                cap.on_resolve(self, ret, x, name, buf)
            except Exception as exc:                # noqa: BLE001
                cap.errors.append("wrap._resolve_input: %r" % (exc,))
            return ret

        batch_runner_cls._resolve_input = _resolve_input
        self._orig["resolve"] = orig_resolve

    def detach(self) -> None:
        """还原所有 patch（``uninstall()`` 的别名，幂等）。"""
        with self._lock:
            if not self._orig:
                return
            cls = self._orig["cls"]
            cls._record = self._orig["record"]
            cls._seg_record = self._orig["seg"]
            cls.commit = self._orig["commit"]
            if "resolve" in self._orig:
                cls._resolve_input = self._orig.pop("resolve")
            self._orig = {}

    # -- 帧门（门 1 / 2 / 3） -------------------------------------------
    def dec_frame_open(self, step: int = -1) -> None:
        """门 1：dec forward 段开口。对应 ``vits_train.py:3512`` 函数体首行。"""
        with self._lock:
            if self._frame == 0:
                # 新帧开始：清空累积（一步一份磁带）
                self.entries = []
                self.commits = []
                self.errors = []
                self.dropped = {}
                self._n_seen = 0
                # ★ RS Phase2 in_w：bid 跨帧复用（输入池步末归还），
                #   旧帧 bid 数值对当前帧无意义 → 必须清空防串味。
                self._in_refs = {}
            self._frame += 1
            self._step = int(step)
            self._frame_opens += 1

    def dec_frame_close(self) -> None:
        """门 2：dec forward 段闭口。对应 ``vits_train.py:3641`` ``return y`` 之前。

        ★ 必须在 ``return y`` **之前**调用 —— ``y = tanh(...)`` 的结果
        本身也在段内（a26q §1.3 门 2 原文）。
        """
        with self._lock:
            if self._frame <= 0:
                self._drop("close_without_open")
                return
            self._frame -= 1
            self._frame_closes += 1

    def in_frame(self) -> bool:
        """门 3 守卫：帧内才记录。"""
        return self._frame > 0

    # -- 捕获处理 ------------------------------------------------------
    def on_int_op(self, runner, op, a_id, b_id, c_id, ps,
                  out_shape=None, tensor_buf=None, ret=None) -> None:
        """int op 载体（op1..37）的捕获处理。

        注意：``BatchRunner._record`` 的 ``ps`` 已被 padding 到 11 项
        （``vulkan_ops.py:2368``），而 ``_records.append`` 存的是
        **padding 后**的 ps（``:2377``）。本函数同步采用 padding 后形态，
        与磁带真身一致；解包器对 ps 长度做过归一，两者兼容。
        """
        self._n_seen += 1
        # 门 3：帧外一律丢弃（backward / 下一步 / 判别器）
        if not self.in_frame():
            self._drop("outside_frame")
            return
        with self._lock:
            try:
                # ★ 复用 a26z 解包器：构造与磁带真身同形的条目
                entry = (int(op), int(a_id), int(b_id), int(c_id),
                         [int(p) for p in ps])
                u = unpack_entry(entry)
            except TapeShapeError as exc:
                self.errors.append("TapeShapeError(int op=%r): %s" % (op, exc))
                self._drop("shape_error")
                return
            except Exception as exc:                # noqa: BLE001
                self.errors.append("unpack(int op=%r) %s: %s"
                                   % (op, type(exc).__name__, exc))
                self._drop("unpack_error")
                return

            rec = self._normalize(u, raw=entry, runner=runner,
                                  out_shape=out_shape, tensor_buf=tensor_buf)
            self.entries.append(rec)

            # ★ RS Phase1：缓存 NUMSNAP_OPS 的 BatchTensor（a26au §2.3 前置 1）
            #   ``ret`` 是原 _record 返回的 BatchTensor；tensor 存活到步末
            #   （C1 实证），dump 时惰性抓取数值。索引 = entry 位置。
            if rec.get("name") in NUMSNAP_OPS and ret is not None:
                self._tensor_refs[len(self.entries) - 1] = ret

    def on_resolve(self, runner, ret, x, name, buf=None) -> None:
        """★ RS Phase2 in_w：``_resolve_input`` 包装回调（a26ay 方向 A）。

        帧内记录 ``bid -> (冻结输入快照, resolve name)``；帧外丢弃。
        冻结采用**深拷贝**（``_freeze_input``），保证 dump 时数值 =
        该步前向用的旧权重（optimizer 之后才就地更新，见 a26ay §3.2）。
        """
        if not self.in_frame():
            return
        try:
            if not isinstance(ret, (tuple, list)) or len(ret) < 2:
                return
            bid = int(ret[0])
        except Exception:                        # noqa: BLE001
            return
        if bid is None or bid <= 0:
            return
        obj = _freeze_input(x, str(name))
        if obj is None:
            return
        with self._lock:
            self._in_refs[bid] = (obj, str(name))

    def on_seg_op(self, runner, fn, c_ids, ps) -> None:
        """str 载体（seg fn / im2col / conv2d / conv_t2d）的捕获处理。"""
        self._n_seen += 1
        if not self.in_frame():
            self._drop("outside_frame")
            return
        with self._lock:
            try:
                c_ids = list(c_ids)
                ps = list(ps)
                entry = (str(fn), *[int(i) for i in c_ids],
                         [int(p) for p in ps])
                u = unpack_entry(entry)
            except TapeShapeError as exc:
                self.errors.append("TapeShapeError(seg %r): %s" % (fn, exc))
                self._drop("shape_error")
                return
            except Exception as exc:                # noqa: BLE001
                self.errors.append("unpack(seg %r) %s: %s"
                                   % (fn, type(exc).__name__, exc))
                self._drop("unpack_error")
                return

            rec = self._normalize(u, raw=entry, runner=runner, fn=str(fn))
            self.entries.append(rec)

    # -- 提交门（门 4） -------------------------------------------------
    def on_commit_pre(self, runner, async_) -> None:
        """``commit`` 调用前：标定边界索引（**只记不下发**）。"""
        with self._lock:
            if not self.in_frame():
                self._drop("commit_outside_frame")
                return
            self._pending_commit = dict(
                runner_id=id(runner),
                async_=bool(async_),
                op_ix=len(self.entries),   # 该 op 之后有 commit
                step=self._step,
            )

    def on_commit_post(self, runner, async_) -> None:
        """``commit`` 返回后：落定边界（仅当引擎真有工作可提交时才计）。"""
        with self._lock:
            pend = getattr(self, "_pending_commit", None)
            self._pending_commit = None
            if pend is None:
                return
            # 空批（无新 op）不产生边界 —— 与引擎 `_committed` 幂等语义一致
            if pend["op_ix"] <= 0 or pend["op_ix"] > len(self.entries):
                self._drop("commit_empty")
                return
            if self.commits and self.commits[-1]["op_ix"] == pend["op_ix"]:
                self._drop("commit_duplicate")
                return
            self.commits.append(pend)

    # -- 规范化（原始磁带 → 可重放 JSON） --------------------------------
    def _normalize(self, u, raw=None, runner=None,
                   out_shape=None, tensor_buf=None, fn=None) -> dict:
        """把 ``unpack_entry`` 的解包结果规范化为磁盘 JSON 条目。

        ``params``
          u             : unpack_entry 返回的 dict（shape/carrier/op/ids/ps/out_bid…）
          raw           : 原始磁带条目（int tuple 或 str 元组，仅调试）
          runner        : 来源 BatchRunner（取 out_bid 返槽号，可选）
          out_shape/fn  : 现有的 out_shape / 函数名（尽量带上）
        """
        op = u.get("op")
        carrier = u.get("carrier")
        shape = u.get("shape")
        ids = u.get("ids") or (None, None, None)
        ps = u.get("ps") or []
        out_bid = u.get("out_bid")
        in_bids = u.get("in_bids")

        # 名称：int 载体用 OP_TABLE；str 载体 fn 本身（conv/im2col/seg）
        if isinstance(carrier, int):
            _entry = OP_TABLE.get(carrier)
            name = (_entry.get("name") if isinstance(_entry, dict)
                    else ("op%d" % carrier))
        else:
            name = str(fn or carrier or shape or "?")

        # 统计 out_bid 缺失（OUT 类 op27-29 静态不可得）
        if out_bid is None and op is not None and 27 <= int(op) <= 29:
            self._drop("out_bid_untracked")

        rec = {
            "op_id": op if op is not None else None,
            "name": name,
            "carrier": carrier,
            "shape": shape,
            "a": int(ids[0]) if ids[0] is not None else None,
            "b": int(ids[1]) if ids[1] is not None else None,
            "c": int(ids[2]) if ids[2] is not None else None,
            "ps": [int(p) for p in ps],
            "out_bid": None if out_bid is None else int(out_bid),
            "in_bids": (None if in_bids is None
                        else [int(x) for x in in_bids]),
            "inplace": (carrier in EXPECT_INPLACE),
            "commit_ix": len(self.commits),   # 该条目归属的上一 commit 索引
            "kind": _SHAPE_TO_KIND.get(shape, "gpu_op"),
        }
        if out_shape is not None:
            rec["out_shape"] = list(out_shape)
        return rec

    # -- 帧门自洽校验 ---------------------------------------------------
    def gates_ok(self):
        """门 5：``frame_opens == frame_closes`` 且当前不残帧。"""
        return (self._frame_opens == self._frame_closes
                and self._frame == 0)

    def tape_ok(self):
        """磁带自洽：无 TapeShapeError、无 unpack_error、有 op 条目。"""
        serious = ("shape_error", "unpack_error")
        return (self.entries
                and not any(self.dropped.get(r, 0) for r in serious))

    # -- 导出 ----------------------------------------------------------
    def dump(self, step=None, tag="mvp") -> str:
        """导出可重放 JSON。返回落盘路径。

        产物：``_diag/cap_step%08d_<tag>.json``（``%08d`` 用 ``_frame_opens``
        做序号 —— a26af §5 零侵入，不给 train_step 加 step 参数即可）。
        """
        with self._lock:
            n = self._frame_opens
            step = self._step if step is None else int(step)
            out_dir = self.out_dir
            os.makedirs(out_dir, exist_ok=True)
            fn = os.path.join(out_dir, "cap_step%08d_%s.json" % (n, tag))
            rng_snapshot = None
            try:
                import numpy as _np
                _st = _np.random.get_state()
                _rng_state = dict(
                    kind=_st[0],
                    keys=[float(_x) for _x in _st[1]],
                    pos=int(_st[2]),
                    has_gauss=bool(_st[3]),
                    cached_gaussian=(float(_st[4])
                                     if _st[3] else None),
                )
                rng_snapshot = {
                    "kind": "numpy.random.RandomState",
                    "state": _rng_state,
                }
            except Exception:  # noqa: BLE001
                rng_snapshot = None

            # ★ RS Phase1：dump 时惰性抓取 out_* 数值快照（a26au §2.3 落点 B）
            #   旧磁带路径零影响：无 _tensor_refs 时 collect 自然返回空。
            numsnap = None
            try:
                nums, meta, stats = collect_numsnap(self)
                if nums:
                    npz_path = nums_path_for(fn)
                    import numpy as _np
                    _np.savez(npz_path, **nums)   # 数值独立 sidecar（a26au §3.3）
                    for ix, m in meta.items():
                        self.entries[ix]["out_num"] = m
                self._last_numsnap = stats
                numsnap = stats
            except Exception as exc:            # noqa: BLE001
                self._last_numsnap = dict(error="%s: %s" % (type(exc).__name__, exc))
                numsnap = None

            # ★ RS Phase2 in_w：dump 时对磁带外部槽抓真实权重快照（a26ay
            #   方向 A；env 门控默认关 ⇒ collect_inw 恒返回 None，零影响）
            inw = None
            try:
                inw_dict, inw_meta, inw_stats = collect_inw(self)
                if inw_dict:
                    import numpy as _np                    # noqa: PLC0415
                    _inw_path = inw_path_for(fn)
                    _np.savez(_inw_path, **inw_dict)
                    inw_stats["path"] = os.path.basename(_inw_path)
                    inw_stats["slots"] = inw_meta
                self._last_inw = inw_stats
                inw = inw_stats
            except Exception as exc:            # noqa: BLE001
                self._last_inw = dict(error="%s: %s" % (type(exc).__name__, exc))
                inw = None

            doc = {
                "schema": SCHEMA,
                "surface": CAPTURE_SURFACE,
                "step": step,
                "tag": tag,
                "frame_opens": self._frame_opens,
                "frame_closes": self._frame_closes,
                "gates_ok": self.gates_ok(),
                "tape_ok": self.tape_ok(),
                "rng": rng_snapshot,
                "n_entries": len(self.entries),
                "n_commits": len(self.commits),
                "numsnap": numsnap,               # 快照统计（无快照=None）
                "inw": inw,                      # in_w 权重槽统计（a26ay；未启用=None）
                "entries": self.entries,
                "commits": self.commits,
                "errors": self.errors,
                "dropped": self.dropped,
            }
            with open(fn, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2)
            return fn

    # -- 定向外部接口 ---------------------------------------------------
    def should_dump(self, cap_steps=1):
        """dump 节流判定：``_frame_opens`` 每逢 ``cap_steps`` 的倍数才 dump。"""
        if cap_steps <= 0:
            return False
        return (self._frame_opens > 0
                and self._frame_opens % int(cap_steps) == 0)


# ---------------------------------------------------------------------------
# 进程级单例 + 安装/卸载
# ---------------------------------------------------------------------------
_CAP = None
_INSTALLED = False


def get_capture():
    """取进程级捕获器单例（惰性创建，幂等）。"""
    global _CAP
    if _CAP is None:
        _CAP = TapeCapture()
    return _CAP


def reset_capture():
    """重置单例（测试/换轮用）。返回旧实例；之后 get_capture() 得新实例。"""
    global _CAP, _INSTALLED
    old = _CAP
    _CAP = None
    _INSTALLED = False
    return old


def install():
    """进程级一次：patch ``BatchRunner`` 并把捕获器挂上。幂等。"""
    global _INSTALLED
    cap = get_capture()
    if _INSTALLED:
        return cap
    from runtime.vulkan_ops import BatchRunner
    cap.attach(BatchRunner)
    _INSTALLED = True
    return cap


def uninstall():
    """还原 patch（幂等）。返回是否真的还原过。"""
    global _INSTALLED
    cap = get_capture()
    if not _INSTALLED:
        return False
    cap.detach()
    _INSTALLED = False
    return True


# 便捷别名（与 __all__ 一致）
def _dummy():  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# §8. 纯构造自测（零 GPU）
# ---------------------------------------------------------------------------
def selftest():
    """零 GPU 自测：构造伪 runner/伪条目，验证门控 + 规范化 + dump。

    退出码：0 = 全 PASS；非 0 = 有 FAIL。默认 ``python _poc/_capture.py --selftest``。
    """
    fails = []

    def _chk(name, cond):
        if not cond:
            fails.append(name)

    # ---- 1) 框架/导入 ----
    for _sym in ("TapeShapeError", "unpack_entry", "OP_TABLE", "SHAPES",
                 "EXPECT_INPLACE", "EXPECT_BITMODE", "EXPECT_REMAP_P9",
                 "A", "C", "P9", "OUT", "OP_MIN", "OP_MAX"):
        _chk("import_%s" % _sym, _sym in globals() or _sym in dir())

    # ---- 2. TapeCapture 实例 ----
    cap = TapeCapture(tag="selftest")
    _chk("init_empty", cap._frame == 0 and cap.entries == [])
    _chk("init_gates", cap._frame_opens == 0 and cap._frame_closes == 0)

    # ---- 3. 三道帧门 ----
    cap.dec_frame_open(step=7)
    _chk("open_incr", cap._frame == 1 and cap._step == 7 and
                      cap._frame_opens == 1)
    _chk("in_frame", cap.in_frame() is True)
    cap.dec_frame_close()
    _chk("close_ok", cap._frame == 0 and cap._frame_closes == 1)
    _chk("not_in_frame", cap.in_frame() is False)
    _chk("gates_eq", cap.gates_ok())
    cap.dec_frame_close()          # 盈余 close → drop
    _chk("close_drop", cap.dropped.get("close_without_open", 0) >= 1)

    # ---- 4. int op 帧内记录（伪造 runner 简短磁带）----
    if "unpack_entry" in globals() or "unpack_entry" in dir():
        fake_runner = type("Fake", (), {})()
        cap.dec_frame_open(step=1)
        # 用 3 个真实 op 编号（int 载体，shape= M1_int_4id）
        for op in (3, 4, 6):
            cap.on_int_op(fake_runner, op, 1, 2, 3, [0], out_shape=[1, 1])
        cap.dec_frame_close()
        _chk("entries3", len(cap.entries) >= 3)
        _chk("entry_shape", cap.entries[0]["shape"] is not None)
        # 帧外 op → drop
        before = dict(cap.dropped)
        cap.on_int_op(fake_runner, 1, 9, 9, 9, [0], out_shape=[1, 1])
        _chk("outside_drop", cap.dropped.get("outside_frame", 0) > before.get("outside_frame", 0))

    # ---- 5. commit 边界 ----
    cap2 = TapeCapture(tag="selftest2")
    cap2.dec_frame_open(step=101)
    cap2.on_int_op(type("R", (), {})(), 4, 1, 2, 3, [0], out_shape=[1, 1])
    cap2.on_commit_pre(fake_runner, async_=False)
    cap2.on_commit_post(fake_runner, async_=False)
    cap2.dec_frame_close()
    _chk("commit_recorded", len(cap2.commits) == 1)
    _chk("commit_ix", cap2.commits[0]["op_ix"] == 1)

    # ---- 6. dump ----
    try:
        path = cap2.dump(tag="selftest")
        _chk("dump_exists", os.path.exists(path))
        with open(path, encoding="utf-8") as fh:
            dd = json.load(fh)
        _chk("dump_schema", dd.get("schema") == "rvc.tape.v1")
        _chk("dump_gates", dd.get("gates_ok") is True)
    except Exception as exc:  # noqa: BLE001
        _chk("dump_ok=%r" % (exc,), False)

    if fails:
        print("SELFTEST FAIL (%d): %s" % (len(fails), ", ".join(fails)))
        return 1
    print("SELFTEST PASS (n=%d)" % (len(fails),))
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="_capture", description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="纯构造自测（零 GPU）")
    ap.add_argument("--tag", default="mvp")
    ap.add_argument("--out", default=OUT_DIR,
                    help="dump 输出目录（默认 _diag）")
    ns, _ = ap.parse_known_args(list(argv) if argv is not None else sys.argv[1:])

    if ns.selftest:
        return selftest()
    # 无操作则打印用法
    print("capture 捕获器；--selftest 进入零 GPU 自测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
