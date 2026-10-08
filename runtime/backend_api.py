# -*- coding: utf-8 -*-
"""后端级抽象层 Backend 接口（R5，**最小接口草案，待双轨评审、非定稿**）。

本模块与 ``runtime/backend.py`` **完全不同**：

- ``runtime/backend.py`` 是**算子级**分派器（matmul/add/mul/... 的 numpy/vulkan
  注册表，``RVC_BACKEND`` 环境变量选择）；本模块是**后端级**契约（一次推理
  会话 = 建引擎/索引 -> 多次整段推理 -> 释放）。
- **不修改 / 不混入** ``runtime/backend.py``；需要设备信息时允许调用方
  直接 import 它（``runtime.backend.get_backend()`` / ``device_info()``）。

本模块是"双后端共存"（Vulkan 轨 + OpenCL 轨）契约的一部分，字段与语义对齐
交接文档 §4.4 / §5 R5-R6（M4 审阅意见的最小接口草案）。

接口约定（全部标注"最小接口草案，待双轨评审、非定稿"）：

1. ``Segment``：**整段音频切片** = dataclass ``(audio: np.ndarray float32 mono,
   sr: int, meta: dict|None)``。内部走 pipeline，不暴露特征级段。
2. ``Profile``：命名打包 ``(检索器选择, 划分策略, 性能底线)``，字段对齐
   §4.4 schema（index_kind/index_path、policy、rtf_baseline、rtf_hard_cap、
   degrade_hint），另加 ``profile_name``；``ctx`` 为后端私有参数扩展位。
3. ``Backend``：``load(profile, ctx_cfg=None)`` 一次（建引擎/索引）→
   ``process(segment)`` 多次（**无状态**，每次独立整段推理）→ ``unload()``
   释放。后端失败**必须抛异常**（``ValueError``/``RuntimeError`` 明确原因），
   前端捕获后走降级提示；**后端不静默降级**。
4. ``policy(query=None) -> str``：划分策略决策接口（R6 接口位）；Vulkan 实现
   返回常量 ``"gpu"``。
5. 字段名列表（跨轨互通位）：``backend_id / profile / policy / rtf_last``。

生命周期约定：
    load 一次（建引擎/索引）→ process 多次（无状态，每次返回 (sr, int16)）→
    unload 释放。``rtf_last`` 在每次 ``process`` 后更新（实时因子 = 音频秒数 /
    墙钟秒数），供上层日志 / 选择器决策使用；未 process 过为 ``None``。

错误处理约定：
    - 后端失败 → 抛 ``ValueError``（参数/输入非法）或 ``RuntimeError``（引擎、
      模型、推理失败），异常消息含明确原因；
    - 前端捕获异常后走降级提示（如切 numpy 轨 / 提示用户）；后端自身
      **不静默降级**（不吞异常、不返回空音频）；
    - ``load`` 失败同样抛异常（模型路径不存在、索引解析失败等）。

开关约定（零行为变更优先 + 加开关可回退）：
    ``RVC_BACKEND_API=1`` 时 api.py / cli.py / realtime_gui.py 走本层；
    未设置时各入口行为与现在完全一致。
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "BACKEND_API_ENV",
    "backend_api_enabled",
    "Profile",
    "profile_vulkan",
    "profile_numpy",
    "Segment",
    "Backend",
    "SegmentBuffer",
    "BACKENDS",
    "register_backend",
    "get_backend_instance",
    "available_backend_names",
    "default_backend_id",
]

#: 入口开关环境变量名；**默认关**（未设置时 api/cli/realtime_gui 行为不变）。
BACKEND_API_ENV = "RVC_BACKEND_API"


def backend_api_enabled() -> bool:
    """返回后端抽象层入口开关是否开启（``RVC_BACKEND_API=1``）。"""
    v = os.environ.get(BACKEND_API_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Profile：命名打包 (检索器选择, 划分策略, 性能底线)
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    """后端 profile（对齐交接文档 §4.4 schema；**最小接口草案，待双轨评审**）。

    Attributes:
        backend_id: 后端标识（``"vulkan"`` / ``"numpy"`` / OpenCL 轨自定名）；
            与注册表 ``BACKENDS`` 的键对应。
        policy: 划分策略（``"gpu"`` / ``"cpu"`` / ``"overlap"``）；R6 接口位。
        index_kind: 检索/索引类型（``"brute"`` / ``"ivf"`` / ``"faiss"`` /
            ``"lean"`` 或 ``None``=不检索）；与 §4.1 检索统一接口联动。
        index_path: 索引文件路径（.npz/.ivf.npz/.index/.lidx；空=None 不检索）。
        rtf_baseline: 性能基线实时因子（RTF = 处理耗时/音频时长；数字待双轨
            校准后冻结，此处为占位建议值）。
        rtf_hard_cap: 硬底线 RTF（超此值前端应考虑降级；数字待冻结）。
        degrade_hint: 降级建议文本（超底线/后端失败时给用户的可读提示）。
        profile_name: 档位名（如 ``"profile-vulkan"`` / ``"profile-numpy"``）。
        ctx: 后端私有参数字典（模型路径/sid/索引率/…），双轨各自解释；
            跨轨互通字段以 Attribute 为准，ctx 不承诺互通。
    """

    backend_id: str
    policy: str
    index_kind: Optional[str] = None
    index_path: Optional[str] = None
    rtf_baseline: float = 0.0
    rtf_hard_cap: float = 0.0
    degrade_hint: str = ""
    profile_name: str = ""
    ctx: dict = field(default_factory=dict)

    def with_ctx(self, **kw: Any) -> "Profile":
        """返回 ctx 合并 ``kw`` 后的新 profile（不修改自身）。"""
        merged = dict(self.ctx)
        merged.update({k: v for k, v in kw.items() if v is not None})
        return Profile(
            backend_id=self.backend_id,
            policy=self.policy,
            index_kind=self.index_kind,
            index_path=self.index_path,
            rtf_baseline=self.rtf_baseline,
            rtf_hard_cap=self.rtf_hard_cap,
            degrade_hint=self.degrade_hint,
            profile_name=self.profile_name,
            ctx=merged,
        )


def profile_vulkan() -> Profile:
    """Vulkan 轨默认 profile（§4.4 profile-vulkan；RTF 数字为占位，待双轨冻结）。

    ctx 缺省为空；调用方按需补 ``model`` / ``sid`` / ``index`` / ``index_rate``
    等键（``with_ctx`` 或直接改 ``ctx``）。
    """
    return Profile(
        backend_id="vulkan",
        policy="gpu",
        index_kind=None,
        index_path=None,
        rtf_baseline=2.0,   # §4.4：profile-vulkan 建议 RTF<2（占位，待校准）
        rtf_hard_cap=5.0,   # §4.4：硬底线 RTF<5（占位，待校准）
        degrade_hint="Vulkan 后端推理失败或超硬底线 RTF=%.1f；建议回退 numpy 轨"
                     "或减小音频长度/关闭索引检索后重试。" % 5.0,
        profile_name="profile-vulkan",
    )


def profile_numpy() -> Profile:
    """numpy 轨默认 profile（轻量；RTF 数字为占位，待双轨冻结）。"""
    return Profile(
        backend_id="numpy",
        policy="cpu",
        index_kind=None,
        index_path=None,
        rtf_baseline=3.3,   # §7.1 核显纯串行 RTF ~3.3+ 量级（占位）
        rtf_hard_cap=10.0,  # 占位
        degrade_hint="numpy 后端推理失败或超硬底线 RTF=%.1f；建议降低音频长度/"
                     "关闭索引检索后重试。" % 10.0,
        profile_name="profile-numpy",
    )


# ---------------------------------------------------------------------------
# Segment：整段音频切片
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """一次 ``process`` 的输入：**整段音频切片**（不暴露特征级段）。

    Attributes:
        audio: float32 mono 音频 ``[T]``。
        sr: 采样率（Hz，>0）。
        meta: 每段可变参数（覆盖 profile.ctx 的同名键），如
            ``{"f0_up_key": 0, "f0_method": "pm"}``；None=全部用 profile.ctx。
    """

    audio: np.ndarray
    sr: int
    meta: Optional[dict] = None


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class Backend(ABC):
    """后端级抽象基类（R5；**最小接口草案，待双轨评审、非定稿**）。

    生命周期：``load`` 一次（建引擎/索引）→ ``process`` 多次（无状态）→
    ``unload`` 释放。

    字段名列表（跨轨互通位，见交接文档 §5 R5）：``backend_id / profile /
    policy / rtf_last``。

    四类能力接口位（后端抽象建议稿，
    待双轨评审冻结、未决项 1）：BatchRunner 录制式批量 / 输出池复用 /
    GPU 常驻权重 / async 提交——本 ABC 提供**默认退化实现**的占位方法
    （``capabilities()`` / ``process_batch()`` / ``process_async()``），
    子类按能力覆盖；默认实现不改变任何现有行为。错误处理沿本模块顶部
    约定：**后端失败必须抛异常，不静默降级**（P-BE-005 原则的接口侧声明）。
    """

    #: 后端标识（子类覆盖；与 BACKENDS 注册键一致）
    backend_id: str = ""

    def __init__(self) -> None:
        #: 最近一次 process 的实时因子（RTF = 音频秒数/墙钟秒数）；未 process 过为 None
        self.rtf_last: Optional[float] = None
        #: 当前生效的 Profile（load 成功后设置）
        self.profile: Optional[Profile] = None
        #: 是否已 load
        self._loaded = False

    # -- 生命周期 ----------------------------------------------------------
    @abstractmethod
    def load(self, profile: Profile, ctx_cfg: Optional[dict] = None):
        """建立引擎/索引等资源（一次）。

        Args:
            profile: 命名打包（backend_id/policy/index_kind/index_path/…
                + ctx 私有参数）。
            ctx_cfg: 与 profile.ctx 合并的附加参数（None 忽略；同名键覆盖
                profile.ctx）。

        Raises:
            ValueError: profile 缺少本后端必需的参数（如模型路径）。
            RuntimeError: 引擎/模型/索引加载失败。
        """

    @abstractmethod
    def process(self, segment: Segment):
        """对一段整段音频切片做一次无状态推理。

        Args:
            segment: 整段音频（float32 mono + sr + 可选 meta）。

        Returns:
            ``(sr_out: int, audio_int16: np.ndarray[int16])``。

        Raises:
            ValueError: 输入非法（非 mono / sr<=0 / 空音频）。
            RuntimeError: 后端推理失败（异常消息含明确原因）。
            后端不静默降级；失败一律抛出，由前端捕获走降级提示。
        """

    @abstractmethod
    def unload(self) -> None:
        """释放引擎/索引等资源（临时目录清理、引用置空；幂等）。"""

    # -- 策略接口（R6 接口位预留，§4.3 划分策略是数据不是代码） -------------
    def policy(self, query: Optional[Any] = None) -> str:
        """划分策略决策：``"gpu"`` / ``"cpu"`` / ``"overlap"``。

        默认返回当前 profile.policy（未 load 时返回空串）；子类可覆盖为常量
        （Vulkan 实现返回 ``"gpu"``）。``query`` 为可选的查询上下文
        （后续选型/负载感知用，当前版本不消费）。
        """
        return self.profile.policy if self.profile is not None else ""

    # -- 四类能力接口位（R5 v0.2 建议稿，默认退化实现，零行为变更） ----------
    # 后端抽象接口位；**待评审冻结**。
    # 全部为默认实现（非 abstractmethod）：不新增抽象方法，现有子类
    # （VulkanBackend/NumpyBackend）无需改动即可通过；子类按能力覆盖。
    def capabilities(self) -> dict:
        """能力探测位（关联 P-BE-005 可观测 / P-BE-011 探测）。

        返回本后端的四类能力声明 + 探测结果占位，供前端可用性展示
        （如 ``GET /backends`` 的 reason/能力字段，M7 §7.5 缺口 1/3）与
        选择器决策使用。键约定（草案）：:

            {
              "batch": bool,        # process_batch 是否原生批量提交
              "pool": bool,         # 输出池复用是否可用
              "persistent": bool,   # GPU 常驻权重是否可用
              "async": bool,        # process_async 是否真异步
              "f16": bool,          # FP16 matmul 能力（P-BE-011 探测位）
              "reason": str,        # 不可用项的可读原因（空=全部可用）
            }

        默认实现：全部 False + reason 说明未实现；子类 load 成功后按实际
        覆盖。跨轨互通：本 dict 为后端自报，不承诺字段互通（对齐
        Profile.ctx 原则），仅 ``batch/pool/persistent/async/f16`` 布尔
        为建议互通位。
        """
        return {
            "batch": False,
            "pool": False,
            "persistent": False,
            "async": False,
            "f16": False,
            "reason": "backend_id=%r 未声明四类能力（默认退化实现）" % self.backend_id,
        }

    def process_batch(self, segments: list) -> list:
        """批量推理接口位（关联 P-BE-002 多批次 / P-BE-003 slice_batch）。

        Args:
            segments: ``list[Segment]``，与 ``process`` 输入语义一致。

        Returns:
            ``list[(sr_out, int16 ndarray)]``，顺序与输入一致；任一失败抛
            异常（后端不静默降级）。

        默认实现：逐段调用 ``process``（行为等价串行，零变更）；子类可覆盖
        为后端原生批量提交（如 vulkan BatchRunner 录制式批量，对齐 §3.2
        算子接口契约）。批量语义/批次 committed 状态契约（P-BE-002 的
        接口侧声明）以本方法 docstring 为准，实现细节留待双轨评审后冻结。
        """
        return [self.process(seg) for seg in segments]

    def process_async(self, segment: Segment, callback=None) -> None:
        """异步推理接口位（关联 P-BE-012 批次锁无超时）。

        Args:
            segment: 与 ``process`` 输入一致。
            callback: 完成回调 ``callable(sr_out, audio_int16)``；None 时
                结果丢弃（仅触发副作用）。

        默认实现：同步退化——内部调 ``process`` 后回调（阻塞语义，行为
        等价现有 process）；子类可覆盖为真异步（提交后立即返回、完成后
        回调）。真异步必须保证回调在**本后端内部线程/事件循环**触发，
        不承诺线程安全；fence 等待超时等细节（R-BE-012）留待实现阶段。
        """
        sr_out, audio_int16 = self.process(segment)
        if callback is not None:
            callback(sr_out, audio_int16)

    # -- 注册表 ------------------------------------------------------------
    # （见模块级 BACKENDS / register_backend / get_backend_instance）


# ---------------------------------------------------------------------------
# 注册表：backend_id -> 工厂函数（callable 无参 -> Backend 实例）
# ---------------------------------------------------------------------------

#: 后端注册表：``backend_id -> 工厂函数``（无参 callable 返回 Backend 实例）。
BACKENDS: dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """注册一个后端工厂（backend_id -> 无参 callable 返回 Backend 实例）。

    Args:
        name: 后端标识（小写规范化；如 ``"vulkan"`` / ``"numpy"``）。
        factory: 无参工厂函数，返回 ``Backend`` 实例。

    Raises:
        ValueError: name 非法或已注册。
    """
    name = name.strip().lower()
    if not name:
        raise ValueError("后端名不能为空")
    if name in BACKENDS:
        raise ValueError("后端已注册: %r" % name)
    if not callable(factory):
        raise ValueError("factory 必须是可调用对象")
    BACKENDS[name] = factory


def get_backend_instance(name: str) -> Backend:
    """按 backend_id 构造一个后端实例（工厂惰性 import，不缓存；调用方自管生命周期）。

    Raises:
        ValueError: 未注册的 backend_id（附带可用列表）。
    """
    name = name.strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            "未知后端 %r（已注册: %s）" % (name, ", ".join(sorted(BACKENDS)) or "(空)")
        )
    return BACKENDS[name]()


def available_backend_names() -> list:
    """返回已注册的 backend_id 列表（含内置 vulkan/numpy）。"""
    return sorted(BACKENDS)


def default_backend_id() -> str:
    """按算子级探测结果给出后端级默认 id：vulkan 可用 → "vulkan"，否则 "numpy"。

    允许调用方用 ``RVC_BACKEND_API_BACKEND`` 环境变量强制覆盖
    （如 ``"numpy"``）；未设置时映射 ``runtime.backend.get_backend()``。
    """
    forced = os.environ.get("RVC_BACKEND_API_BACKEND", "").strip().lower()
    if forced:
        return forced if forced in BACKENDS else "vulkan" if "vulkan" in BACKENDS else "numpy"
    try:
        from runtime.backend import get_backend as _op_get_backend  # 算子级探测
        op = _op_get_backend()
    except Exception:  # noqa: BLE001  # 探测失败按 numpy 兜底
        op = "numpy"
    return op if op in BACKENDS else "numpy"


# 内置注册：vulkan 与 numpy（工厂惰性 import adapter，避免导入环）。
def _factory_vulkan() -> Backend:
    from runtime.backend_vulkan import VulkanBackend  # noqa: PLC0415

    return VulkanBackend()


def _factory_numpy() -> Backend:
    from runtime.backend_numpy import NumpyBackend  # noqa: PLC0415

    return NumpyBackend()


register_backend("vulkan", _factory_vulkan)
register_backend("numpy", _factory_numpy)


# ---------------------------------------------------------------------------
# SegmentBuffer：实时 GUI 的分段推理辅助（R5 最小接入；完整实时链路属 R9）
# ---------------------------------------------------------------------------


class SegmentBuffer:
    """把连续输入块攒成整段、交 ``Backend.process`` 处理，输出按块弹出的适配器。

    **用途**：realtime_gui 在 ``RVC_BACKEND_API=1`` 时，把块级输入流切成
    ``segment_seconds`` 秒的整段切片喂给 Backend（R5 最小接入演示；延迟未
    优化，SOLA/crossfade/降噪等实时后处理链的完整设计属 R9 专项）。

    行为：
    - ``push(block16k)`` 追加一段 16k float32 输入块；
    - 输入缓冲长度达到 ``segment_seconds*16000`` 时触发一次
      ``backend.process(Segment(...))``，输出 (out_sr, int16) 转 float32 后
      存入输出缓冲；
    - 每次 push 返回**本块对应长度的输出块**（长度 = block 长度 ×
      out_sr/16000；不足时返回 ``None`` 表示还需等待）。
    """

    def __init__(
        self,
        backend: "Backend",
        segment_seconds: float = 2.0,
        out_sr: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.segment_seconds = float(segment_seconds)
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._out_sr = None

    def push(self, block16k: np.ndarray) -> Optional[np.ndarray]:
        """追加一个 16k float32 输入块；返回对应的输出块（float32，out_sr），
        不足时返回 None。"""
        block = np.asarray(block16k, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None
        self._in_buf = np.concatenate([self._in_buf, block])
        out = None
        seg_len = int(round(self.segment_seconds * 16000))
        while self._in_buf.size >= seg_len:
            seg = self._in_buf[:seg_len]
            self._in_buf = self._in_buf[seg_len:]
            t0 = time.perf_counter()
            sr_out, audio_int16 = self.backend.process(Segment(seg, 16000))
            self._out_sr = sr_out
            audio_f = np.asarray(audio_int16, dtype=np.float32) / 32768.0
            self._out_buf = np.concatenate([self._out_buf, audio_f])
            if self.backend.rtf_last is not None:
                print(
                    "[backend_api.SegmentBuffer] 段 %.1fs 处理 %.1fs（rtf_last=%.2f）"
                    % (self.segment_seconds, time.perf_counter() - t0,
                       self.backend.rtf_last)
                )
        if self._out_sr is not None:
            take = int(round(block.size * self._out_sr / 16000))
            if self._out_buf.size >= take:
                out = self._out_buf[:take]
                self._out_buf = self._out_buf[take:]
        return out

    def flush(self) -> Optional[np.ndarray]:
        """把剩余输入缓冲也处理掉（结束流时调用）；返回剩余输出（可为空）。"""
        if self._in_buf.size and self.backend is not None:
            sr_out, audio_int16 = self.backend.process(Segment(self._in_buf, 16000))
            self._out_sr = sr_out
            self._out_buf = np.concatenate(
                [self._out_buf, np.asarray(audio_int16, dtype=np.float32) / 32768.0]
            )
            self._in_buf = np.zeros(0, dtype=np.float32)
        if self._out_buf.size:
            out = self._out_buf
            self._out_buf = np.zeros(0, dtype=np.float32)
            return out
        return None
