# -*- coding: utf-8 -*-
"""Vulkan 后端 adapter（R5；**最小接口草案，待双轨评审、非定稿**）。

``VulkanBackend`` 把现有 ``runtime.vc.VC`` 高层封装包一层后端级接口：
``load(profile)`` 一次建 VC 实例（引擎/模型/索引上下文）→
``process(segment)`` 多次整段推理 → ``unload()`` 释放。

设计要点：

- **零侵入**：绝不修改 ``runtime/vc.py`` 与 ``runtime/pipeline.py`` 内部代码，
  只复用其公开接口（``VC(config)`` / ``vc.get_vc(model)`` /
  ``vc.vc_single(...)``），因此两路径（现状直调 vc_single vs 经本 adapter）
  行为逐位一致。
- **模型路径约定**：profile.ctx 需提供 ``model``（.pth 路径或模型名），
  以及可选 ``sid``（说话人 int，默认 0）、``index``/``index_rate``、
  ``f0_method``、``resample_sr``、``rms_mix_rate``、``protect``、
  ``slice_length``、``retrieval_mode``、``brute_mix``、``f0_up_key``。
  每段可变参数（如 pitch/f0_method）放在 ``Segment.meta``，覆盖 ctx 同名键。
- **临时 wav 中转**：process 把 Segment(audio, sr) 写为临时 wav
  （subtype='FLOAT' 32-bit IEEE float，写入/读回**逐位无损**，配合
  ``load_audio`` 在 ``source_sr==sr`` 时不重采样的行为，保证与直调
  vc_single（读原文件）路径输入完全一致）→ 调内部 VC.vc_single → 返回
  ``(sr, int16 ndarray)``。临时文件用完即删；临时目录在 ``unload`` 清理。
- **错误处理**：后端失败一律抛异常（ValueError/RuntimeError，含明确原因），
  不静默降级；``rtf_last`` 每次 process 后更新（音频秒数/墙钟秒数）。
- ``policy(query=None)`` 返回常量 ``"gpu"``（R6：Vulkan 常量 {gpu} 接口位）。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Optional

import numpy as np

from runtime.backend_api import Backend, Profile, Segment

__all__ = ["VCBackendBase", "VulkanBackend"]


class VCBackendBase(Backend):
    """基于 ``runtime.vc.VC`` 的后端公共实现（Vulkan / numpy 轨共用）。

    子类只需覆盖 ``backend_id`` 与 ``policy()``。
    """

    #: 后端标识（子类覆盖）
    backend_id: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._vc = None          # runtime.vc.VC 实例
        self._tmp_dir = None     # 临时 wav 目录（unload 清理）
        self._ctx: dict = {}     # profile.ctx + ctx_cfg 合并结果

    # -- 生命周期 ----------------------------------------------------------
    def load(self, profile: Profile, ctx_cfg: Optional[dict] = None) -> dict:
        """构造内部 VC 实例并加载模型（profile.ctx 需提供 ``model``）。

        profile.ctx 支持键：``model``（.pth 路径或模型名，必需）、``sid``
        （说话人 int，默认 0）、``index`` / ``index_rate`` / ``f0_method`` /
        ``resample_sr`` / ``rms_mix_rate`` / ``protect`` / ``slice_length`` /
        ``retrieval_mode`` / ``brute_mix`` / ``f0_up_key``（可选，process 时
        生效；``Segment.meta`` 可覆盖）。

        Returns:
            与 ``VC.get_vc`` 相同的模型信息 dict（success/path/n_spk/…）。
        """
        self.profile = profile
        self._ctx = dict(profile.ctx or {})
        if ctx_cfg:
            self._ctx.update({k: v for k, v in ctx_cfg.items() if v is not None})

        model = self._ctx.get("model")
        if not model:
            raise ValueError(
                "profile.ctx 缺少必需的 'model' 键（.pth 路径或模型名）；"
                "示例：profile.with_ctx(model='gan1_e50_s4100.pth', sid=0, "
                "index_rate=0.0, f0_method='pm')"
            )

        from runtime.native_config import Config
        from runtime.vc import VC

        vc = VC(Config())
        info = vc.get_vc(model)
        if not info["success"]:
            raise RuntimeError("模型加载失败: %s" % info.get("error"))
        self._vc = vc
        self._tmp_dir = tempfile.mkdtemp(prefix="rvc_backend_%s_" % self.backend_id)
        self._loaded = True
        return info

    def process(self, segment: Segment):
        """整段音频切片推理：写临时 wav（FLOAT 无损）→ vc_single → (sr, int16)。

        Args:
            segment: Segment(audio float32 mono [T], sr>0, meta 可选)。

        Returns:
            ``(sr_out, audio_int16)``；失败抛 ValueError/RuntimeError（明确原因）。

        Raises:
            RuntimeError: 未 load / 引擎或推理失败。
            ValueError: 输入非法（非 mono、sr<=0、空音频）。
        """
        if not self._loaded or self._vc is None:
            raise RuntimeError(
                "后端未加载：请先调用 load(profile)（backend_id=%r）" % self.backend_id
            )
        audio = np.asarray(segment.audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Segment.audio 必须是一维 float32 mono（实际 %d 维）" % audio.ndim)
        if int(segment.sr) <= 0:
            raise ValueError("Segment.sr 必须 > 0（实际 %r）" % segment.sr)
        if audio.size == 0:
            raise ValueError("Segment.audio 为空：无法推理空音频段")

        # 每段参数：profile.ctx 为基础，Segment.meta 覆盖
        p = dict(self._ctx)
        if segment.meta:
            p.update({k: v for k, v in segment.meta.items() if v is not None})
        sid = int(p.get("sid", 0))
        f0_up_key = int(p.get("f0_up_key", p.get("pitch", 0)))
        f0_method = str(p.get("f0_method", "pm"))
        file_index = p.get("index") or p.get("index_path") or p.get("file_index") or None
        index_rate = float(p.get("index_rate", 0.0))
        resample_sr = int(p.get("resample_sr", 0))
        rms_mix_rate = float(p.get("rms_mix_rate", 1.0))
        protect = float(p.get("protect", 0.33))
        slice_length = float(p.get("slice_length", 0))
        retrieval_mode = str(p.get("retrieval_mode", "ivf"))
        brute_mix = float(p.get("brute_mix", 0.0))

        # 临时 wav：FLOAT 32-bit IEEE float 无损往返，保证与直调路径输入逐位一致
        from runtime.dsp.audio_io import write_audio

        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", dir=self._tmp_dir)
        os.close(fd)
        audio_secs = audio.size / float(segment.sr)
        try:
            write_audio(tmp_wav, audio, int(segment.sr), subtype="FLOAT")
            t0 = time.perf_counter()
            status, result = self._vc.vc_single(
                sid,
                tmp_wav,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                resample_sr,
                rms_mix_rate,
                protect,
                slice_length=slice_length,
                retrieval_mode=retrieval_mode,
                brute_mix=brute_mix,
            )
            dt = time.perf_counter() - t0
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

        if not result or result[0] is None or result[1] is None:
            raise RuntimeError("推理失败（backend_id=%r）: %s" % (self.backend_id, status))
        sr_out, audio_int16 = result
        self.rtf_last = audio_secs / dt if dt > 0 else None
        return (int(sr_out), np.asarray(audio_int16))

    def unload(self) -> None:
        """释放：置空 VC 引用、删除临时目录（幂等）。"""
        self._vc = None
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
        self._loaded = False
        self.rtf_last = None


class VulkanBackend(VCBackendBase):
    """Vulkan 轨后端 adapter：policy 常量 ``"gpu"``（R6 接口位）。"""

    backend_id = "vulkan"

    def policy(self, query: Optional[object] = None) -> str:
        """划分策略：Vulkan 轨恒为 ``"gpu"``（全 GPU；接口位预留）。"""
        return "gpu"

    def capabilities(self) -> dict:
        """能力声明（R5 v0.2 建议稿接口位，只读声明现状、不改变执行语义）。

        后端抽象接口位；**待评审冻结**。
        以下为**引擎层现状**的自报（vulkan_ops 语义未动）：

        - ``pool``: True —— vulkan_ops 输出池复用已启用（vulkan_ops.py:336-506，
          ``RVC_OUT_NO_POOL`` 可关，168）；
        - ``persistent``: True —— 常驻权重在 ``RVC_BACKEND=vulkan`` 时启用
          （vulkan_weights.py:45 enabled 绑定）；
        - ``batch``/``async``: False —— 本 adapter 的 ``process`` 为逐段整段
          推理（backend_vulkan.py:97-172），未接线 BatchRunner 录制式批量 /
          真异步；对应接口位已留（Backend.process_batch/process_async 默认
          退化实现）；
        - ``f16``: False —— FP16 能力探测缺失（P-BE-011，探测位未实现）。
        """
        return {
            "batch": False,
            "pool": True,
            "persistent": True,
            "async": False,
            "f16": False,
            "reason": "pool/persistent 为引擎层现状自报；batch/async/f16 接口位"
                      "已留，实现待双轨评审冻结（未决项 1）后接线",
        }
