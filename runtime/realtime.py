# -*- coding: utf-8 -*-
"""实时变声（占位模块）——当前不可用。

⚠️ 实时推理功能当前**完全不可用**，已归档废弃，后续将重写。

历史实现被废弃的原因（详见 docs/05-实时子系统.md）：
  - 补丁堆叠：20+ 轮修复从未重新设计；
  - 验证方法不可信：结构指标无法区分"正确输出"与"固定噪声"；
  - 根因：输出重采样无抗混叠（np.interp）；
  - 范式缺陷：f0 整窗重算、无 SOLA、return_length 无余量。

本模块仅作为占位，任何调用都会抛出明确异常，提示使用离线变声
（runtime.cli / runtime.api）替代。重写完成前请勿依赖实时功能。
"""

from __future__ import annotations

__all__ = ["Realtime", "realtime_available"]


def realtime_available() -> bool:
    """实时推理当前不可用，恒返回 False。"""
    return False


class Realtime:
    """实时变声占位类（当前不可用）。"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "实时变声当前不可用（已归档废弃，待重写）。"
            "请使用离线变声：python -m runtime.cli / python -m runtime.api。"
            "详见 docs/05-实时子系统.md。"
        )
