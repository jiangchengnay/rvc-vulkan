# -*- coding: utf-8 -*-
"""M4 阶段 0 埋点（a26ah 设计落地）：真提交 vs 幂等早退计数 + b4_outer wait 测量。

**红线**：本模块只在 env ``RVC_TRAIN_M4_PROBE=1`` 时才产生实际计数动作；
默认（env 未设）全链路走 ``if _A26AH_ON:`` 短路分支，**零行为影响**。
本模块不读队列长度（不改 engine / 不加 FFI），只读 Python 侧既有属性。

设计文档：``_diag/a26ah_m4_instrument.md`` §2（埋点 A/B/C/D）。
实测驱动：``_diag/m4_probe_run.py``。

落点清单（全部 env 门控）：
    A  runtime/models/vits_train.py::_commit_chain_br        真提交/早退三段计数
    A' runtime/models/vits_train.py + runtime/graph_runner.py 阶段 D/G 标定
    B  runtime/vulkan_ops.py::BatchRunner.commit             b4_outer vs _elapsed
    C  runtime/graph_runner.py::bwd_run / bwd_run_async      wall 六段（同步/异步分列）
    D  runtime/train/train.py::train_step 尾                 步末快照 + 可选落盘

不变式自检（埋点自身正确性，违反则先怀疑计数器而非源码）：
    a_calls == a_early + a_released + a_no_br
    a_dirty == a_calls - a_early
"""
from __future__ import annotations

import json
import os
import time

# ---------------------------------------------------------------------------
# 门控 + 状态
# ---------------------------------------------------------------------------
# 冻结式开关：模块首次 import 时读一次 env（避免热点路径反复 os.environ.get）；
# _on() 仍每次读 env——驱动脚本在 import 之后才设 env 的场景需要它，
# 该 get 在 CPython 层是 dict 查询（~1e-7 s），相对 commit（ms 级）可忽略。
_ON_ = os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1"

PHASE_D = "D"
PHASE_G = "G"

_EMPTY_PHASE = {"calls": 0, "dirty": 0, "early": 0, "released": 0, "no_br": 0}


def _on() -> bool:
    """埋点总开关（env RVC_TRAIN_M4_PROBE=1；默认关）。"""
    return _ON_ or os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1"


def _phase_of_step() -> str:
    """当前阶段标定（D=判别器步 / G=生成器步）。

    调用侧（graph_runner）无法静态判断自己在 D 还是 G——故用**显式打标**：
    train.py 在 D 步 / G 步边界调用 ``set_phase()``。未打标时归 "?"。
    """
    return _S.get("phase", "?")


def set_phase(p: str) -> None:
    """阶段边界打标（train.py 在 D/G 步切换处调用；埋点关闭时为 no-op）。"""
    if _on():
        _S["phase"] = p


# ---------------------------------------------------------------------------
# 状态字典（单一聚合；步末快照后清空采样 list，保留累计计数）
# ---------------------------------------------------------------------------
_S: dict = {
    "phase": "?",
    # ---- 埋点 A：_commit_chain_br 入口 ----
    "a_calls": 0,        # A1 被调总次数
    "a_dirty": 0,        # A2 调用时 _CHAIN_DIRTY 为真
    "a_early": 0,        # A3 幂等早退（0 GPU 工作）
    "a_released": 0,     # A5 真提交次数（commit 实际被调）
    "a_no_br": 0,        # A4 DIRTY 真但 br 空/已释放
    "a_commit_ms": [],   # 每次真提交的 commit 全量（含 DLL 内 wait）
    "a_phase": {},       # {phase: {calls, dirty, early, released, no_br}}
    # ---- 埋点 B：commit 内部 ----
    "b_calls": 0,
    "b0_committed": 0,      # S0 runner 级幂等返回
    "b1_upq": 0,            # 上传队列长度累加
    "b1_upq_empty": 0,      # 空队列调用次数
    "b2_flush_ms": [],      # S1 上传 flush
    "b3_submit_ms": [],     # S2 submit（本模块自测；引擎另有 _elapsed）
    "b3_elapsed_ms": [],    # S3 引擎既有 _elapsed（交叉校验）
    "b4_outer_ms": [],      # S2..返回 的外层 wall
    "b_outer_wait_ms": [],  # b4_outer - b3_submit ≈ wait（同步；异步≈0）
    "b_async": 0,           # async_=True 的 commit 次数
    # ---- 埋点 C：bwd_run / bwd_run_async wall 分段 ----
    "c_run_sync_n": 0,
    "c_pre_sync_ms": [],     # C1 前置（类型检查/转换）
    "c_up_sync_ms": [],      # C2 set_input 上传总量
    "c_commit_sync_ms": [],  # C3 commit 全量 ★ C1 收益对象
    "c_run_sync_ms": [],     # C4 g.run 全程（阻塞）★ 与 async 语义不同
    "c_total_sync_ms": [],   # C5 总（与既有 _last_bwd_ms 分列）
    "c_run_async_n": 0,
    "c_pre_async_ms": [],
    "c_up_async_ms": [],
    "c_commit_async_ms": [],
    "c_run_async_ms": [],    # C4 g.async_run（FFI 入队，应亚 ms）
    "c_total_async_ms": [],
    # ---- 埋点 D：步级 ----
    "d_steps": [],          # [{"step": n, "wall": s, "loss_total": x, "prof": {...}}]
    "d_loss": [],           # loss_total 序列（逐位相等自检）
}


def snapshot() -> dict:
    """返回当前计数摘要（不改状态；供 [WALL] 行内打印与步末输出）。"""
    a = {k: _S[k] for k in ("a_calls", "a_dirty", "a_early", "a_released",
                            "a_no_br")}
    a["early_pct"] = (100.0 * _S["a_early"] / _S["a_calls"]) if _S["a_calls"] else 0.0
    return {
        "A": a,
        "B": {"calls": _S["b_calls"], "b0": _S["b0_committed"],
              "upq_empty": _S["b1_upq_empty"], "async": _S["b_async"]},
        "C": {"sync_n": _S["c_run_sync_n"], "async_n": _S["c_run_async_n"]},
        "phase": _S.get("phase", "?"),
    }


def _median(xs: list) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _p95(xs: list) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[i]


def _stat(xs: list) -> dict:
    if not xs:
        return {"n": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0,
                "max": 0.0}
    n = len(xs)
    return {"n": n, "sum": float(sum(xs)), "mean": float(sum(xs)) / n,
            "median": _median(xs), "p95": _p95(xs), "max": max(xs)}


# ---------------------------------------------------------------------------
# 埋点 A —— _commit_chain_br 入口（vits_train.py）
# ---------------------------------------------------------------------------
def a_entry(dirty: bool) -> None:
    """A1/A2：入口计数（在 ``if not _CHAIN_DIRTY`` 之前调用）。"""
    if not _on():
        return
    _S["a_calls"] += 1
    _S["a_dirty"] += int(bool(dirty))
    p = _phase_of_step()
    d = _S["a_phase"].setdefault(p, dict(_EMPTY_PHASE))
    d["calls"] += 1
    d["dirty"] += int(bool(dirty))


def a_early() -> None:
    """A3：幂等早退（``_CHAIN_DIRTY`` 为假，0 GPU 工作）。"""
    if not _on():
        return
    _S["a_early"] += 1
    _S["a_phase"].setdefault(_phase_of_step(), dict(_EMPTY_PHASE))["early"] += 1


def a_released() -> None:
    """A5：真提交计数（commit 调用前）。"""
    if not _on():
        return
    _S["a_released"] += 1
    _S["a_phase"].setdefault(_phase_of_step(), dict(_EMPTY_PHASE))["released"] += 1


def a_no_br() -> None:
    """A4：DIRTY 真但链 br 为空/已释放（不产生 GPU 工作）。"""
    if not _on():
        return
    _S["a_no_br"] += 1
    _S["a_phase"].setdefault(_phase_of_step(), dict(_EMPTY_PHASE))["no_br"] += 1


def a_commit_ms(ms: float) -> None:
    """A6：真提交的 commit 全量耗时（含 DLL 内同步 wait）。"""
    if _on():
        _S["a_commit_ms"].append(ms)


# ---------------------------------------------------------------------------
# 埋点 B —— BatchRunner.commit（vulkan_ops.py）
# ---------------------------------------------------------------------------
def b_pre(committed: bool, upq_len: int, async_: bool) -> None:
    """B0/B1：runner 级幂等返回前的只读探针。

    ``committed=True`` 时本次 commit 走 S0 立即返回（第二层幂等）。
    """
    if not _on():
        return
    _S["b_calls"] += 1
    if committed:
        _S["b0_committed"] += 1
    else:
        _S["b1_upq"] += int(upq_len)
        _S["b1_upq_empty"] += int(upq_len == 0)
        _S["b_async"] += int(bool(async_))


def b_flush(ms: float) -> None:
    if _on():
        _S["b2_flush_ms"].append(ms)


def b_submit(ms: float, elapsed_ms: float) -> None:
    """S2 submit 自测 + 引擎既有 ``_elapsed`` 交叉校验列。"""
    if _on():
        _S["b3_submit_ms"].append(ms)
        _S["b3_elapsed_ms"].append(elapsed_ms)


def b_outer(ms: float) -> None:
    """b4_outer：S2..commit 返回的外层 wall。

    与 ``b3_submit_ms[-1]`` 的差值 ≈ ``wait``（同步 commit 的 fence-wait 在
    DLL 内部，``_elapsed`` 测不到 —— 这正是埋点 B 存在的唯一理由）。
    """
    if not _on():
        return
    _S["b4_outer_ms"].append(ms)
    if _S["b3_submit_ms"]:
        _S["b_outer_wait_ms"].append(ms - _S["b3_submit_ms"][-1])


# ---------------------------------------------------------------------------
# 埋点 C —— graph_runner.bwd_run / bwd_run_async（wall 分段）
# ---------------------------------------------------------------------------
def c_sync(pre_ms: float, up_ms: float, commit_ms: float, run_ms: float,
           total_ms: float) -> None:
    """同步路径（``g.run`` 阻塞）六段。C4 语义 = 等待整图执行完。"""
    if not _on():
        return
    _S["c_run_sync_n"] += 1
    _S["c_pre_sync_ms"].append(pre_ms)
    _S["c_up_sync_ms"].append(up_ms)
    _S["c_commit_sync_ms"].append(commit_ms)
    _S["c_run_sync_ms"].append(run_ms)
    _S["c_total_sync_ms"].append(total_ms)


def c_async(pre_ms: float, up_ms: float, commit_ms: float, run_ms: float,
            total_ms: float) -> None:
    """异步路径（``g.async_run`` 入队）六段。C4 语义 = FFI 入队（应亚 ms）。

    与 ``c_sync`` **字段名必须分开**（a26ah §2.3）：两者同名会把"等待整图
    执行"与"入队"混为一谈。
    """
    if not _on():
        return
    _S["c_run_async_n"] += 1
    _S["c_pre_async_ms"].append(pre_ms)
    _S["c_up_async_ms"].append(up_ms)
    _S["c_commit_async_ms"].append(commit_ms)
    _S["c_run_async_ms"].append(run_ms)
    _S["c_total_async_ms"].append(total_ms)


# ---------------------------------------------------------------------------
# 埋点 D —— 步级快照 / 落盘 / 汇总
# ---------------------------------------------------------------------------
def step_snapshot(step: int, wall_s: float, loss_total, prof: dict | None = None,
                  dump: bool = True) -> dict:
    """步末快照：记录本步 wall / loss_total / 埋点分片，并清空采样 list。

    只在埋点开启时记录；``dump=True`` 时同时打印一行 ``[M4]`` 到 stderr。
    """
    if not _on():
        return {}
    rec = {
        "step": int(step),
        "wall": float(wall_s),
        "loss_total": None if loss_total is None else float(loss_total),
        "phase": _S.get("a_phase", {}),
        "a": {k: _S[k] for k in ("a_calls", "a_dirty", "a_early", "a_released",
                                 "a_no_br")},
        "a_commit": _stat(_S["a_commit_ms"]),
        "b": {
            "calls": _S["b_calls"], "b0": _S["b0_committed"],
            "upq": _S["b1_upq"], "upq_empty": _S["b1_upq_empty"],
            "async": _S["b_async"],
            "flush": _stat(_S["b2_flush_ms"]),
            "submit": _stat(_S["b3_submit_ms"]),
            "elapsed": _stat(_S["b3_elapsed_ms"]),
            "outer": _stat(_S["b4_outer_ms"]),
            "wait": _stat(_S["b_outer_wait_ms"]),
        },
        "c": {
            "sync_n": _S["c_run_sync_n"], "async_n": _S["c_run_async_n"],
            "pre_sync": _stat(_S["c_pre_sync_ms"]),
            "up_sync": _stat(_S["c_up_sync_ms"]),
            "commit_sync": _stat(_S["c_commit_sync_ms"]),
            "run_sync": _stat(_S["c_run_sync_ms"]),
            "total_sync": _stat(_S["c_total_sync_ms"]),
            "pre_async": _stat(_S["c_pre_async_ms"]),
            "up_async": _stat(_S["c_up_async_ms"]),
            "commit_async": _stat(_S["c_commit_async_ms"]),
            "run_async": _stat(_S["c_run_async_ms"]),
            "total_async": _stat(_S["c_total_async_ms"]),
        },
        "prof": prof or {},
    }
    _S["d_steps"].append(rec)
    if loss_total is not None:
        _S["d_loss"].append(float(loss_total))
    # 采样 list 步末清空（仿 _DBG_DISC_BWD_MS.clear()，a26ah R2）；累计计数保留
    for k in ("a_commit_ms", "b2_flush_ms", "b3_submit_ms", "b3_elapsed_ms",
              "b4_outer_ms", "b_outer_wait_ms", "c_pre_sync_ms",
              "c_up_sync_ms", "c_commit_sync_ms", "c_run_sync_ms",
              "c_total_sync_ms", "c_pre_async_ms", "c_up_async_ms",
              "c_commit_async_ms", "c_run_async_ms", "c_total_async_ms"):
        _S[k].clear()
    _S["a_phase"] = {}
    if dump:
        inv = _S["a_calls"] == (_S["a_early"] + _S["a_released"] + _S["a_no_br"])
        print(
            f"[M4] step={step} wall={wall_s:.3f}s calls={_S['a_calls']} "
            f"early={_S['a_early']} released={_S['a_released']} "
            f"no_br={_S['a_no_br']} syncC={rec['c']['sync_n']} "
            f"asyncC={rec['c']['async_n']} inv={'OK' if inv else 'BAD'}",
            file=sys.stderr if False else __import__("sys").stderr, flush=True)
    return rec


def invariant_ok() -> tuple:
    """全局不变式自检：``(ok, detail)``。

    ``a_calls == a_early + a_released + a_no_br`` 且 ``a_dirty == a_calls - a_early``。
    """
    ok1 = _S["a_calls"] == (_S["a_early"] + _S["a_released"] + _S["a_no_br"])
    ok2 = _S["a_dirty"] == (_S["a_calls"] - _S["a_early"])
    detail = (f"calls={_S['a_calls']} early={_S['a_early']} "
              f"released={_S['a_released']} no_br={_S['a_no_br']} "
              f"dirty={_S['a_dirty']}")
    return (ok1 and ok2), detail


def _verdict(A: dict, C: dict, wall: dict) -> tuple:
    """判定三选一（a26ah §3.4）。返回 ``(code, text)``。"""
    n_real = A.get("a_released", 0)
    c3 = C.get("commit_sync", {}).get("sum", 0.0) + C.get("commit_async", {}).get("sum", 0.0)
    c5 = C.get("total_sync", {}).get("sum", 0.0) + C.get("total_async", {}).get("sum", 0.0)
    run = C.get("run_sync", {}).get("sum", 0.0)
    steps = max(1, int(wall.get("n_steps", 1)))
    per_step = n_real / steps
    c3_share = (c3 / c5 * 100.0) if c5 else 0.0
    run_share = (run / c5 * 100.0) if c5 else 0.0
    wait = _S.get("b_outer_wait_ms", [])
    wait_med = _median(wait)
    # 边界情形基数重算：预期收益 = 真提交/步 × 每次 wait 均 ms（a26ah §3.4）
    _gain = per_step * (wait_med / 1000.0)
    if per_step >= 30 and c3_share >= 50.0:
        return ("①", f"真提交/步 {per_step:.1f} ≥30 且 C3 占 C5 {c3_share:.1f}% ≥50% "
                      f"⇒ C1 直接命中，进阶段 1")
    if run_share >= 50.0:
        return ("③", f"同步 g.run 占 C5 {run_share:.1f}% ≥50% ⇒ O1 归因错误，"
                      f"转查图执行器；M4 降优先级")
    if per_step <= 6 or c3_share < 25.0:
        _why = []
        if per_step <= 6:
            _why.append(f"真提交/步 {per_step:.1f} ≤6")
        if c3_share < 25.0:
            _why.append(f"C3 占 C5 {c3_share:.1f}% <25%")
        return ("②", f"{' 或 '.join(_why)} ⇒ 基数失真，先查剩余墙钟归属"
                      f"（C2 上传/C4 run/表5 段差），暂缓 C1")
    _tail = (f"（wait 均值 {wait_med:.2f}ms，按 N 重算预期收益 "
             f"≈{_gain:.3f}s/步）" if wait else "")
    if wait and _gain < 0.15:
        return ("②", f"边界情形：真提交/步 {per_step:.1f} ∈ (6,30)、C3 占 C5 "
                      f"{c3_share:.1f}%，但预期收益 {_gain:.3f}s/步 <0.15s/步 "
                      f"⇒ 不足以覆盖 C1 的 wait 时序风险，判②")
    return ("②", f"边界情形：真提交/步 {per_step:.1f} ∈ (6,30) 且 C3 占 C5 "
                  f"{c3_share:.1f}% ⇒ 按 N 重算基数" + _tail)


def summarize(pair_loss: list | None = None) -> dict:
    """全局汇总（驱动脚本收尾调用）：表 1-5 全部数字 + 判定三选一 + 不变式。"""
    a_stat = {k: _S[k] for k in ("a_calls", "a_dirty", "a_early", "a_released",
                                 "a_no_br")}
    a_stat["early_pct"] = (100.0 * _S["a_early"] / _S["a_calls"]) if _S["a_calls"] else 0.0
    # 按阶段汇总（d_steps 内已按步记录）
    ph: dict = {}
    for rec in _S["d_steps"]:
        for p, v in (rec.get("phase") or {}).items():
            d = ph.setdefault(p, dict(_EMPTY_PHASE))
            for k in _EMPTY_PHASE:
                d[k] += int(v.get(k, 0))
    walls = [r["wall"] for r in _S["d_steps"]]
    losses = [r["loss_total"] for r in _S["d_steps"] if r["loss_total"] is not None]
    c_all = {k: [] for k in ()}
    del c_all
    c_sum = {
        "sync_n": _S["c_run_sync_n"], "async_n": _S["c_run_async_n"],
        "pre_sync": _stat(_S["c_pre_sync_ms"]),
        "up_sync": _stat(_S["c_up_sync_ms"]),
        "commit_sync": _stat(_S["c_commit_sync_ms"]),
        "run_sync": _stat(_S["c_run_sync_ms"]),
        "total_sync": _stat(_S["c_total_sync_ms"]),
        "pre_async": _stat(_S["c_pre_async_ms"]),
        "up_async": _stat(_S["c_up_async_ms"]),
        "commit_async": _stat(_S["c_commit_async_ms"]),
        "run_async": _stat(_S["c_run_async_ms"]),
        "total_async": _stat(_S["c_total_async_ms"]),
    }
    # 注意：步末已 clear，这里的 _stat 为空；真正数字在 d_steps 明细里。
    c_from_steps: dict = {}
    for rec in _S["d_steps"]:
        for key, v in (rec.get("c") or {}).items():
            if not isinstance(v, dict):
                continue
            d = c_from_steps.setdefault(key, {"n": 0, "sum": 0.0})
            d["n"] += int(v.get("n", 0))
            d["sum"] += float(v.get("sum", 0.0))
    b_from_steps: dict = {}
    for rec in _S["d_steps"]:
        for key, v in (rec.get("b") or {}).items():
            if not isinstance(v, dict):
                continue
            d = b_from_steps.setdefault(key, {"n": 0, "sum": 0.0})
            d["n"] += int(v.get("n", 0))
            d["sum"] += float(v.get("sum", 0.0))
    code, text = _verdict(a_stat, c_from_steps,
                          {"n_steps": len(_S["d_steps"])})
    ok, detail = invariant_ok()
    out = {
        "probe_env": os.environ.get("RVC_TRAIN_M4_PROBE", "0"),
        "graph_async": os.environ.get("RVC_TRAIN_GRAPH_ASYNC", "0"),
        "bwd_async": os.environ.get("RVC_TRAIN_BWD_ASYNC", "0"),
        "graph_bwd": os.environ.get("RVC_TRAIN_GRAPH_BWD", "0"),
        "batch_size": os.environ.get("RVC_TRAIN_BATCH_SIZE", "?"),
        "table1_A": {**a_stat, "by_phase": ph},
        "table2_B": {
            "calls": _S["b_calls"], "b0_committed": _S["b0_committed"],
            "upq_sum": _S["b1_upq"], "upq_empty": _S["b1_upq_empty"],
            "upq_empty_pct": (100.0 * _S["b1_upq_empty"] / _S["b_calls"])
            if _S["b_calls"] else 0.0,
            "async": _S["b_async"],
            "seg_sum": b_from_steps,
        },
        "table3_C": {"n_steps": len(_S["d_steps"]), "seg_sum": c_from_steps,
                     "live_stat": c_sum},
        "table4_D": {
            "n_steps": len(_S["d_steps"]),
            "wall_sum": sum(walls), "wall_median": _median(walls[2:]) if len(walls) > 2 else _median(walls),
            "walls": walls,
            "loss_total": losses,
            "per_step": _S["d_steps"],
        },
        "invariant": {"ok": ok, "detail": detail},
        "verdict": {"code": code, "text": text},
    }
    if pair_loss is not None:
        out["table4_D"]["loss_pair_equal"] = (losses == list(pair_loss))
    return out


def dump_json(path: str | None = None, pair_loss: list | None = None) -> str:
    """把汇总写入 JSON（``RVC_TRAIN_M4_PROBE_FILE``，默认 ``_diag/m4_probe.json``）。

    同时打印一份精简的表 1/表 2/表 3 到 stdout，便于直接读日志。
    """
    data = summarize(pair_loss=pair_loss)
    if not path:
        path = os.environ.get("RVC_TRAIN_M4_PROBE_FILE", "")
    if not path:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "_diag", "m4_probe.json")
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"[M4] 汇总已写入 {path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[M4] 汇总写盘失败：{exc}", flush=True)
    t1 = data["table1_A"]
    print(f"[M4-T1] calls={t1['a_calls']} dirty={t1['a_dirty']} "
          f"early={t1['a_early']} ({t1['early_pct']:.1f}%) "
          f"真提交={t1['a_released']} no_br={t1['a_no_br']} | "
          f"by_phase={t1['by_phase']}", flush=True)
    print(f"[M4-T3] {data['table3_C']['seg_sum']}", flush=True)
    print(f"[M4] 不变式 {'OK' if data['invariant']['ok'] else 'BAD'}: "
          f"{data['invariant']['detail']}", flush=True)
    print(f"[M4] 判定 {data['verdict']['code']}: {data['verdict']['text']}",
          flush=True)
    return path
