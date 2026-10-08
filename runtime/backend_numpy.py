# -*- coding: utf-8 -*-
"""numpy 后端 adapter（R5；**最小接口草案，待双轨评审、非定稿**）。

轻量包装：与 ``VulkanBackend`` 共用同一 ``VCBackendBase``（复用
``runtime.vc.VC`` 的公开流程），仅声明为 numpy 轨 profile（
``backend_id="numpy"``、``policy="cpu"``）。

注意：本 adapter 是**后端级**声明。实际算子执行仍由算子级
``runtime.backend``（``RVC_BACKEND``）决定——要强制纯 CPU 请在启动时设
``RVC_BACKEND=numpy``；未强制时，本 adapter 的推理同样可能走 Vulkan 算子
（与现有 CLI/API 的 numpy 分支行为一致，不改变任何默认语义）。
"""

from __future__ import annotations

from typing import Optional

from runtime.backend_api import Backend
from runtime.backend_vulkan import VCBackendBase

__all__ = ["NumpyBackend"]


class NumpyBackend(VCBackendBase):
    """numpy 轨后端 adapter：policy 常量 ``"cpu"``（R6 接口位）。"""

    backend_id = "numpy"

    def policy(self, query: Optional[object] = None) -> str:
        """划分策略：numpy 轨恒为 ``"cpu"``（接口位预留）。"""
        return "cpu"
