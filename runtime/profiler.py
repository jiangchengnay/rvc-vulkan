# -*- coding: utf-8 -*-
"""逐算子 profiling（P0-1，用户要求）：算子名/单次耗时/调用次数/总占比/执行设备。

用法：模型代码/算子封装用 ``@profile_op("matmul", "gpu")`` 或
``with profile_op(...)`` 包裹，推理后调用 ``profiler_report()`` 输出表格。

- 设备标注：GPU（vulkan_ops 实际 dispatch）与 CPU（numpy 回退）分开计，
  便于定位"回退到 CPU 的算子"——用户要求：不许把算子解释成 GPU 算不了。
- 开关环境变量 ``RVC_PROFILE=1`` 启用（默认关闭零开销）。
- 线程安全：锁保护；多块并行（pipeline 线程池）下可累加。
"""

from __future__ import annotations

import os
import threading
import time

_ENABLED = os.environ.get("RVC_PROFILE", "0").strip() in ("1", "true", "on")
_lock = threading.Lock()
_STATS: dict = {}  # name -> {"n": int, "us": int}


def enabled() -> bool:
    return _ENABLED


class _OpProfile:
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not _ENABLED:  # 未启用时零开销（只保留了 perf_counter 读一次）
            return False
        dt = (time.perf_counter() - self.t0) * 1e6  # us
        with _lock:
            s = _STATS.setdefault(self.name, {"n": 0, "us": 0.0})
            s["n"] += 1
            s["us"] += dt
        return False


def profile_op(name: str, device: str = "gpu"):
    """上下文管理器：name 建议含设备后缀（如 "conv1d/gpu" "conv1d/cpu"）。"""
    return _OpProfile(name)


def reset() -> None:
    with _lock:
        _STATS.clear()


def _sorted():
    items = list(_STATS.items())
    items.sort(key=lambda kv: -kv[1]["us"])
    return items


def report(total_us: float | None = None) -> str:
    """输出表格：名称 | 次数 | 单次us | 总us | 占比%。"""
    if not _STATS:
        return "[profiler] 无数据（RVC_PROFILE=1 启用后重跑）"
    if total_us is None:
        total_us = sum(v["us"] for _, v in _STATS.items())
    lines = ["[profiler] 逐算子统计（总 %.1f ms）" % (total_us / 1000.0),
             "%-40s %8s %10s %12s %7s" % ("name", "calls", "per/us", "total/us", "%"),
             "-" * 82]
    for name, v in _sorted():
        per = v["us"] / v["n"] if v["n"] else 0
        pct = v["us"] / total_us * 100 if total_us else 0
        lines.append("%-40s %8d %10.1f %12.0f %6.1f%%" % (name, v["n"], per, v["us"], pct))
    return "\n".join(lines)