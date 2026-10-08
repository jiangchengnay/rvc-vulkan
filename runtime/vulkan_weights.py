# -*- coding: utf-8 -*-
"""Vulkan 模型权重常驻管理器（P1 BufferPool 模式）。

把模型权重一次性 ``persistent_upload`` 到 GPU 并记录（key -> PersistentBuffer），
推理时算子函数通过 ``buf_w=...`` / ``buf_b=...`` 等参数复用常驻 buffer，避免
每次调用重复 upload/download 权重。参考 Valkyr 引擎 ``buffer.zig`` 的 static
缓冲模式：weights 常驻 device 内存，动态数据（输入/输出）仍逐次往返。

后端为 numpy（``get_backend() != "vulkan"``）时**全部方法为空操作**：
``register`` 不做事、``get`` 恒返回 None —— 调用方照旧走 numpy 路径，保证
默认行为与未接入常驻时完全一致。

典型用法::

    from runtime.vulkan_weights import _weights
    _weights.register("dec.conv_pre.weight", w)   # 上传并记录（numpy 后端 no-op）
    pb = _weights.get("dec.conv_pre.weight")       # PersistentBuffer | None
    ...
    _weights.free_all()                            # 释放全部（换模型/退出时）

线程说明：注册与推理默认在同一线程（单请求流水线）；并发场景下调用方需自行
保证 ``register`` / ``free_all`` 不与推理交错（见遗留风险）。
"""

from __future__ import annotations

from typing import Dict

__all__ = ["GPUWeights", "_weights"]


class GPUWeights:
    """模型权重常驻管理器（模块级单例 ``_weights``）。

    属性:
        enabled: 当前后端是否为 vulkan（numpy 后端时全部方法为空操作）。
    """

    def __init__(self):
        self._enabled = False
        self._bufs: Dict[str, object] = {}
        try:
            from runtime import backend  # noqa: PLC0415  # 惰性避免初始化环

            if backend.get_backend() == "vulkan":
                self._enabled = True
        except Exception:  # noqa: BLE001  # 探测失败按禁用处理（保守回退）
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """当前后端是否为 vulkan（numpy 后端恒为 False）。"""
        return self._enabled

    def register(self, key: str, arr) -> None:
        """上传 ``arr`` 到 GPU 常驻并记录到 ``key``（numpy 后端 no-op）。

        同一 key 重复注册时：先释放旧的常驻 buffer 再上传新的（支持换权重）。
        """
        if not self._enabled:
            return
        from runtime import vulkan_ops  # noqa: PLC0415

        old = self._bufs.pop(key, None)
        if old is not None:
            old.free()
        self._bufs[key] = vulkan_ops.get_context().persistent_upload(arr)

    def get(self, key: str):
        """返回 ``key`` 对应的常驻 buffer（PersistentBuffer）；未注册 / 已释放 /
        numpy 后端时返回 None。
        """
        if not self._enabled:
            return None
        pb = self._bufs.get(key)
        if pb is None or not pb.valid:
            return None
        return pb

    def free_all(self) -> None:
        """释放全部常驻 buffer 并清空记录（幂等；多模型切换 / 进程退出时调用）。

        释放后再次 ``infer`` 不崩溃：模型侧 ``get`` 返回 None，自动回退到
        普通（非常驻）路径。
        """
        for pb in self._bufs.values():
            pb.free()
        self._bufs.clear()

    def __len__(self) -> int:
        """当前登记的常驻 buffer 数量（numpy 后端为 0）。"""
        return len(self._bufs) if self._enabled else 0


_weights = GPUWeights()
"""模块级单例：模型代码 ``from runtime.vulkan_weights import _weights`` 直接使用。"""
