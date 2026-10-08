# -*- coding: utf-8 -*-
"""HuBERT 语音编码器 —— numpy 推理实现 + transformer attention Vulkan GPU 化
（RVC 去 CUDA 化移植 T21，P1 深化）。

对齐 HuggingFace ``transformers==4.49.0`` 的 ``HubertModel`` 完整推理语义，
权重从 ``assets/hubert_base/pytorch_model.bin``（float16 存储）加载，加载后
统一 ``astype(np.float32)`` 参与计算。**本模块不 import torch**。

结构规格（逐条核对 modeling_hubert.py 4.49 源码与 assets/hubert_base/config.json）:
    输入 x[1,T] float32 16kHz。**B5（2026-09-23）：输入为 raw 音频，不做整段
    LayerNorm**——官方 RVC infer/vc/pipeline.py 直接喂原始音频（hubert_base/
    preprocessor_config.json do_normalize: false）；此前对该输入做强制的整段
    z-score LayerNorm（_normalize_for_hubert），导致 hubert 特征相对官方体系
    偏差 ~38%，是"咬字不清"根因之一，已移除。

    1. 特征提取 conv 栈（7 层，全部 valid 卷积、无 padding、无 bias）:
       conv0: Conv1d(1→512, k=10, s=5) → GroupNorm(512 组, eps=1e-5) → GELU
       conv1..6: Conv1d(512→512, k/s = 3/2 ×4 + 2/2 ×2)，每层后 GELU
         —— 注意：transformers 的 HubertNoLayerNormConvLayer 对每层都施加
         GELU（含 conv5/6），与旧版 fairseq 省略最后两层的实现不同，此处
         以 transformers 4.49 为准。
       帧数递推 L = floor((L-k)/s)+1：T=16000 → 3199→1599→799→399→199→99→49
       输出 [1,512,L] → transpose → [1,L,512]
    2. feature_projection: LayerNorm(512, eps=1e-5) → Linear(512→768) → [1,L,768]
    3. encoder 入口: h = h + pos_conv_embed(h)；再 LayerNorm(768, eps=1e-5)
       pos_conv_embed = weight_norm Conv1d(768→768, k=128, pad=64, stride=1,
       groups=16)：
         weight_norm(dim=2) 还原 W = weight_g * weight_v / ‖weight_v‖₂，
         范数沿除 dim=2（核长维）之外的所有维求 L2，即对每个核位置 k 求
         ‖weight_v[..., k]‖₂（weight_v 形状 [768,48,128]，weight_g [1,1,128]，
         均为 float16，归一化在 float32 下进行）。
         分组卷积 groups=16：输入通道 768 分 16 组、每组 48 通道，组间参数
         不共享（输出通道同样分组），逐组 conv1d 后 concat。
         conv 输出 [1,768,L+1] → 裁剪右 1 列 → [1,768,L] → GELU → 加回 h
    4. 12 层 post-LN transformer 层（每层）:
       a) 自注意力（无相对位置偏置）：q/k/v = h@Wᵀ+b（各 Linear 768→768），
          q 投影后乘 1/√head_dim（transformers 在 q 上缩放而非 scores 上）；
          reshape (1,L,12,64)→transpose(1,2)；scores=q@kᵀ；softmax(dim=-1)；
          out=attn@v；transpose 还原；out_proj Linear
       b) 残差：h = h + attn_out
       c) LayerNorm(768, eps=1e-5)
       d) FFN：h = gelu(h@Wiᵀ+bi)@Woᵀ+bo（intermediate 3072）
       e) 残差：h = h + ffn；final LayerNorm(768, eps=1e-5)

     P1-3（上一提交）：12 层 transformer 的 attention 主计算（QKV 投影 /
     scores / 加权和 / out_proj）在常驻权重命中时走 Vulkan matmul
     （BatchRunner 合并录制、一次 submit），否则回退 numpy（逐位一致）；
     softmax / LayerNorm / 残差加 / GELU 保持 numpy（本平台 GPU 单算子
     固定开销远高于 numpy，见 _attn_opt / _layer_norm_np）；FFN 保持
     numpy 原路径。

     P1-4（本提交，数据驱动，见 profile_hubert.py）：profile 与单算子拆测
     显示本平台（AMD Radeon Pro VII + 朴素 tiled matmul shader）瓶颈并非
     "numpy 算子多"本身，而是三类病态成本：
       (a) gelu_erf 的 float64 逐元素计算 —— conv 栈 7 次 + FFN 12 次共约
           2 万个 float64 临时数组遍历，是 2s 音频最大单点（~1.2s）；
           改为全程 float32（A&S 同公式，与旧版最大差 ~5e-7）；
       (b) FFN 的 wi/wo 线性层经 ``nn_ops.linear`` 走 GPU 单算子且权重
           非持久 —— 每次调用重复 transpose+upload [768,3072] 权重并付
           单 dispatch 固定开销（~6-13ms）→ 改为 wi/wo 权重常驻 +
           BatchRunner 各一次 commit（全 T 实测优于 numpy BLAS）；
       (c) attention 逐 head 的 scores/加权和（K=64 小 matmul，12 dispatch
           一次 commit）GPU 实测 139ms（T=499）而 numpy einsum 仅 11ms；
           QKV/out_proj（K=768 单次大 matmul，常驻）GPU 5-26ms 与 numpy
           同量级/更优 → attention 改为混合路径：QKV/out_proj GPU +
           scores/softmax/加权和 numpy（``_attn_hybrid``），替代原
           ``_attn_gpu`` 的全 GPU 12 头方案（其 12 dispatch + 12MB×12
           层下载是 T≥256 时最慢的一环，实测被 numpy 反超 3-12x）。
     conv 栈复核（目标 4）：7 层 Conv1d 常驻命中正常（hubert.conv.{i}
     .weight 均已注册，bias key 历史 bug 不复存在）；10s 输入 GPU
     0.78s vs numpy 0.81s 持平略优，维持 GPU。
     环境变量 ``RVC_HUBERT_ATTN=numpy`` 可强制 attention 走 numpy 路径。

     P2（本提交）：多块批量 encode（``encode_batch``）——长音频切块后
     不再逐块串行 encode，而是所有块一次 GPU encode。引擎 attn_qk/attn_sv
     仅支持单 batch（[T,C] 输入，无 B 维），故采用**无 padding 的按块
     展开**：各块按自身长度独立录算子（conv 栈/pos_conv 按块、attention
     按块独立 attn_qk），12 层 × N 块全部录进**同一个** BatchRunner、
     每层一次**异步** commit（受引擎 recorder max_sets=384 上限约束，
     单 runner ≤19 块，见 ``_BATCH_MAX_BLOCKS``；块数超限按组分批、
     组间下载/上传传递），中间张量 GPU 驻留、只下载 need 层 → 与逐块
     路径同 kernel 同参数（maxdiff=0 逐位一致）。feature_projection 的
     bias 由 numpy 加改为 GPU bias_add（差 ≤1 ulp ≪ 1e-4）。
     性能实测：瓶颈在引擎每 dispatch ~7ms 固定成本（dispatch 间全局
     memory barrier + 调度，与 commit 次数/同步异步无关，见
     .tmp_work/p2_bottleneck.py），总 dispatch 数不变时 batch 仅减少
     commit 次数（~0.2s），无法达到 4-8x；达成需 engine 侧支持 dispatch
     无 barrier 组 / fused 批量算子（engine 改造另派）。
     环境变量 ``RVC_HUBERT_BATCH_N``（pipeline 侧）控制单批块数上限。
    5. 输出:
       version=1：取第 9 层（1-based，即 layers[8]）输出 hidden_states[9]
                  → final_proj Linear(768→256) → [1,L,256]
       version=2：12 层全部输出（last_hidden_state）→ [1,L,768]

    所有 GELU 采用精确 erf 定义（与 torch.nn.functional.gelu 默认一致）；
    runtime.nn.gelu 为 tanh 近似，精度达不到 <1e-3 对照要求，故本模块自带
    erf 实现（Abramowitz-Stegun 7.1.26，最大误差 ~1.5e-7）。

API:
    HubertEncoder(model_dir)      加载 hubert_base 权重
    encoder.encode(x, version)    x[1,T] float32 → [1,L,768]（v2）/ [1,L,256]（v1）
    encoder.encode_batch(xs, v)   list of [1,T_i] → list of [1,L_i,D]（P2 批量）
    load_hubert_model(model_dir)  模块级单例缓存（webui 多次调用复用）
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import numpy as np

from torch_compat import load_pth  # 纯 Python 读 .pth，不依赖 torch

from .. import nn as nn_ops
from ..vulkan_weights import _weights  # P1：Vulkan 权重常驻管理器（numpy 后端 no-op）

__all__ = ["HubertEncoder", "load_hubert_model"]

_F32 = np.float32

# attention 路径控制：默认走 ``_attn_hybrid``（GPU QKV/out_proj + numpy
# scores/softmax/加权和，全 T 实测均优于纯 numpy 或旧全 GPU 方案）。
# ``RVC_HUBERT_ATTN=numpy`` 强制纯 numpy；``gpu`` 强制混合路径（默认值）。
_ATTN_NUMPY_FORCED = os.environ.get("RVC_HUBERT_ATTN", "").strip().lower() == "numpy"
_ATTN_GPU_FORCED = os.environ.get("RVC_HUBERT_ATTN", "").strip().lower() == "gpu"

# P1-5 整层批量提交总开关：``RVC_HUBERT_BATCH=0`` 关闭（回退 P1-4 混合路径）；
# 默认开启，且仅 vulkan 后端（权重已注册常驻）时实际生效。
_BATCH_ENABLED_FLAG = os.environ.get("RVC_HUBERT_BATCH", "1").strip().lower() != "0"

# 分阶段 perf profile 开关：RVC_HUBERT_PROFILE=1 时 HuBERT 记录各阶段耗时到
# ``encoder._prof``（dict: 阶段→累计秒）与 ``encoder._prof_layers``（每层明细）。
# 默认关闭，关闭时各插桩点为一次布尔判断，零开销。
_PROFILE_ENABLED = os.environ.get("RVC_HUBERT_PROFILE", "").strip().lower() \
    in ("1", "true", "yes")

# P2：多块批量 encode 的每组块数上限。引擎 batch recorder 的 descriptor
# pool 为 384 sets（engine/recorder.zig init(384,1536)），单 batch 最多录
# 384 个 dispatch；12 层 transformer 每层每块 20 个算子，故单 runner
# （跨层驻留）最多容纳 floor(384/20)=19 块。超过时按组切分，组间
# 下载/上传传递（功能正确，仅固定开销重复）。150s（16 块）/ 45s（5 块）
# 均单组一次驻留。
_BATCH_MAX_BLOCKS = 19


# ---------------------------------------------------------------------------
# 精确 GELU（erf 版，对齐 torch.nn.functional.gelu 默认行为）
# ---------------------------------------------------------------------------

# Abramowitz-Stegun 7.1.26 系数（erf 近似，最大误差 1.5e-7；float32 下
# 另加 ~1e-7 舍入，与 float64 版最大差 ~5e-7，远低于全部对照阈值）
_A1, _A2, _A3, _A4, _A5 = (0.254829592, -0.284496736, 1.421413741,
                           -1.453152027, 1.061405429)
_P = 0.3275911
# float32 常量：避免 np.float32 数组与 python float 混算时（NumPy 1.x）
# 提升为 float64，把不必要的双精度逐元素计算全部留在 float32（P1-4：
# 实测 float64 erf 是 conv 栈/FFN 的最大单点耗时，f32 化后 ~1.5-2.2x）。
_SQRT2_INV = np.float32(0.7071067811865476)
_C1 = np.float32(_A1); _C2 = np.float32(_A2); _C3 = np.float32(_A3)
_C4 = np.float32(_A4); _C5 = np.float32(_A5); _CP = np.float32(_P)
_F32_ONE = np.float32(1.0)
_F32_HALF = np.float32(0.5)


def _erf(x: np.ndarray) -> np.ndarray:
    """向量化 erf（对 x≥0 的 A&S 近似，负值用奇函数性质），全程 float32。"""
    x32 = np.asarray(x, dtype=_F32)
    ax = np.abs(x32)
    t = _F32_ONE / (_F32_ONE + _CP * ax)
    y = _F32_ONE - (((((_C5 * t + _C4) * t) + _C3) * t + _C2) * t + _C1) \
        * t * np.exp(-ax * ax)
    return np.sign(x32) * y


# ---------------------------------------------------------------------------
# 可选 numba 快速逐元素内核（P1-4）：融合单遍 + prange 多线程。
# 已安装时 gelu/softmax 提速 ~2.5-9x（conv 栈 10s 的 16M 元素 erf 从
# ~1.0s 降到 ~0.11s）；未安装自动回退 numpy，行为/数值不变。内核与 numpy
# 版同一 A&S 公式，max|Δ| ~5e-7（softmax ~1.5e-8），远低于全部对照阈值。
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange  # type: ignore[import-not-found]  # noqa: PLC0415

    _NUMBA_OK = True
except Exception:  # pragma: no cover - 无 numba 的环境回退 numpy
    _NUMBA_OK = False

if _NUMBA_OK:

    @njit(parallel=True, fastmath=False, cache=True)
    def _gelu_kernel(x):  # x: contiguous float32 1D → 同形状 float32
        out = np.empty_like(x)
        n = x.size
        c = 0.7071067811865476
        a1 = 0.254829592; a2 = -0.284496736; a3 = 1.421413741
        a4 = -1.453152027; a5 = 1.061405429; p = 0.3275911
        for i in prange(n):
            u = x[i] * c
            ax = np.abs(u)
            t = 1.0 / (1.0 + p * ax)
            poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
            erfv = 1.0 - poly * np.exp(-ax * ax)
            if u < 0.0:
                erfv = -erfv
            out[i] = 0.5 * x[i] * (1.0 + erfv)
        return out

    @njit(parallel=True, fastmath=False, cache=True)
    def _softmax_kernel(x2d):  # x2d: contiguous float32 [rows, cols]，原地
        rows, cols = x2d.shape
        for r in prange(rows):
            m = x2d[r, 0]
            for c in range(1, cols):
                v = x2d[r, c]
                if v > m:
                    m = v
            s = 0.0
            for c in range(cols):
                e = np.exp(x2d[r, c] - m)
                x2d[r, c] = e
                s += e
            for c in range(cols):
                x2d[r, c] = x2d[r, c] / s


def gelu_erf(x: np.ndarray) -> np.ndarray:
    """精确 GELU：0.5x(1 + erf(x/√2))，对齐 torch 默认（非 tanh 近似）。

    P1-4：优先 numba 融合多线程内核（~9x，误差 ~5e-7）；否则 numpy A&S
    float32（原 float64 实现逐元素多走 ~6 个双精度临时数组，是 conv 栈 +
    FFN 的最大 CPU 单点耗时）。与旧 float64 版最大差 ~5e-7，torch 对照
    （<1e-3）与 GPU/numpy 一致性（<1e-4）均不受影响。
    """
    x32 = np.asarray(x, dtype=_F32)
    if _NUMBA_OK:
        flat = np.ascontiguousarray(x32).reshape(-1)
        return _gelu_kernel(flat).reshape(x32.shape)
    return (_F32_HALF * x32 * (_F32_ONE + _erf(x32 * _SQRT2_INV))).astype(_F32)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """attention 专用 softmax（最后一维，exp(x-max) 数值稳定）。

    与 ``nn_ops.softmax`` 的 numpy 分支同公式；不走 ``nn_ops.softmax``
    是为了避免 vulkan 后端把它分派到 GPU 单算子（scores [12,T,T] 上传+
    下载 ~24MB/层，固定开销 ~60ms 起，实测远慢于 numpy）。numba 可用时
    用融合多线程内核（~2.5x），否则 numpy 稳定版。
    """
    x32 = np.asarray(x, dtype=_F32)
    if _NUMBA_OK and x32.ndim >= 2:
        xr = np.ascontiguousarray(x32).reshape(-1, x32.shape[-1])
        _softmax_kernel(xr)
        return xr.reshape(x32.shape)
    m = x32.max(axis=-1, keepdims=True)
    e = np.exp(x32 - m)
    return e / e.sum(axis=-1, keepdims=True)


def _layer_norm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
    """LayerNorm 纯 numpy（biased 方差），与 ``nn.layer_norm`` 的 numpy
    分支逐位一致 —— 避免 vulkan 后端下经 ``backend.layer_norm`` 分派到 GPU
    单算子（实测固定开销 ~6ms/次，12 层 × 2 远超 numpy 的 <1ms）。"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # ddof=0，与 PyTorch biased 一致
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * gamma + beta


def _conv1d_batch_or_seg(br, x, w, b, key: str, stride: int):
    """T1.1 Step5：hubert conv 栈 conv1d 录制，输出超 GRID 时 GPU 内分段。

    ``conv1d_seg_multi`` 按输出列自动分段写父 buffer（x 保持 BatchTensor
    流转，零 numpy 往返），与整段 ``br.conv1d`` 同 kernel 同参数逐位一致
    （Step2 已 maxdiff=0 验证）；不超限走原 ``br.conv1d``（零回归）。
    hubert conv 栈为 valid 卷积（padding=0/dilation=1），b 恒为 None。
    """
    from runtime import vulkan_ops as _vo  # noqa: PLC0415

    C_out = w.shape[0]
    oL = (x.shape[2] - w.shape[2]) // stride + 1  # pad=0, dil=1
    if 1 * C_out * oL > _vo._GRID_POINTS_MAX:
        return br.conv1d_seg_multi(
            x, w, b, stride=stride, padding=0,
            out_shape=(1, C_out, oL),
            buf_w=_weights.get(key),
        )
    return br.conv1d(
        x, w, None, stride=stride, padding=0,
        buf_w=_weights.get(key),
    )


# ---------------------------------------------------------------------------
# 分组卷积（runtime/nn 未提供，groups>1 时逐组 conv1d 后 concat）
# ---------------------------------------------------------------------------

def _grouped_conv1d(x, w, b, groups, stride=1, padding=0):
    """1D 分组卷积，对齐 ``torch.nn.functional.conv1d(..., groups>1)`` 语义。

    参数:
        x: ``[B, C, T]``
        w: ``[O, C//groups, K]``（PyTorch 分组卷积权重形状，组间不共享）
        b: ``[O]``
        groups: 分组数，要求 C % groups == 0 且 (C//groups) == w.shape[1]
    返回 ``[B, O, oL]``。
    """
    x = np.asarray(x, dtype=_F32)
    w = np.asarray(w, dtype=_F32)
    B, C, T = x.shape
    O, Ci, K = w.shape
    if C % groups != 0:
        raise ValueError(f"grouped_conv1d: C={C} 不能被 groups={groups} 整除")
    cpg = Ci
    opg = O // groups
    if C // groups != Ci:
        raise ValueError(
            f"grouped_conv1d: 输入通道 C/groups={C // groups} 与权重 {Ci} 不匹配"
        )
    if isinstance(padding, (tuple, list)):
        pad_l, pad_r = int(padding[0]), int(padding[1])
    else:
        pad_l = pad_r = int(padding)
    oL = (T + pad_l + pad_r - K) // int(stride) + 1
    parts = []
    for g in range(groups):
        part = nn_ops.conv1d(
            x[:, g * cpg:(g + 1) * cpg, :],
            w[g * opg:(g + 1) * opg, :, :],
            b[g * opg:(g + 1) * opg] if b is not None else None,
            stride=stride, padding=padding,
        )  # [B, opg, oL]
        parts.append(part)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# HuBERT 编码器
# ---------------------------------------------------------------------------

# state dict 键名前缀（transformers 4.49 的 bin 顶层无 "model." 前缀；
# 旧版权重可能带，加载时做容错剥离）
_PREFIX = "model."


def _strip_prefix(key: str) -> str:
    if key.startswith(_PREFIX):
        return key[len(_PREFIX):]
    return key


class HubertEncoder:
    """HuBERT 语音编码器（纯 numpy 推理）。

    从 ``model_dir/pytorch_model.bin`` 加载 float16 权重并转为 float32，
    常驻内存供多次 ``encode`` 调用复用。
    """

    def __init__(self, model_dir: str):
        self.model_dir = str(model_dir)
        self._sd, self._layers = self._load_and_prepare(self.model_dir)
        self._register_persistent_weights()
        self._prof: Optional[Dict[str, float]] = {} if _PROFILE_ENABLED else None
        self._prof_layers: Optional[list] = [] if _PROFILE_ENABLED else None

    # -- perf 插桩（RVC_HUBERT_PROFILE=1 时启用，默认零开销）---------------

    @contextmanager
    def _pt(self, key: str) -> Iterator[None]:
        """按阶段累计墙钟耗时（仅 ``_prof`` 启用时记录）。"""
        if self._prof is None:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._prof[key] = self._prof.get(key, 0.0) + \
                (time.perf_counter() - t0)

    # -- 权重常驻（P1 BufferPool，可选接入）--------------------------------

    def _register_persistent_weights(self) -> None:
        """把 feature_projection + conv 栈 + 12 层 transformer attention 权重
        常驻 GPU。

        收益：feature_projection Linear 512→768（约 1.5 MB）、conv 栈 7 层
        Conv1d、以及 12 层 transformer 的 attention（QKV 拼接 + out_proj）
        线性层权重每次 encode 都参与计算，常驻后跳过每次 upload。
        ``matmul`` 的右操作数按 ``w.T`` 上传（线性层 ``x @ w.T`` 语义）；
        QKV 三个投影拼接为单个 ``[768, 2304]`` 矩阵（``x @ [Wq;Wk;Wv]ᵀ``
        语义，一次 matmul 出 q/k/v，数值与分别 matmul 逐位一致），并把
        拼接结果与 out_proj 的 ``w.T`` 连续数组回存到 ``layer`` 字典供
        GPU 路径直接复用（避免每次调用重复 transpose）。
        numpy 后端（GPUWeights 禁用）下为 no-op。
        """
        if not _weights.enabled:
            return
        gw, gb = self._layers["proj_ln"]
        _weights.register("hubert.proj_ln.gamma", gw)
        _weights.register("hubert.proj_ln.beta", gb)
        w, b = self._layers["proj"]
        _weights.register("hubert.proj.weight",
                          np.ascontiguousarray(w.T, dtype=np.float32))
        _weights.register("hubert.proj.bias", b)
        # P1-2：conv 栈 7 层 Conv1d（valid、无 bias），键 hubert.conv.{i}.weight
        for i, (cw, _cb) in enumerate(self._layers["conv"]):
            _weights.register(f"hubert.conv.{i}.weight", cw)
        # P1-5：conv 栈 GroupNorm（512 组，逐通道归一）+ encoder 入口 LN 常驻
        _weights.register("hubert.conv.gn.gamma", self._layers["conv_gamma"])
        _weights.register("hubert.conv.gn.beta", self._layers["conv_beta"])
        enc_gw, enc_gb = self._layers["enc_ln"]
        _weights.register("hubert.enc_ln.gamma", enc_gw)
        _weights.register("hubert.enc_ln.beta", enc_gb)
        # P1-5：pos_conv（weight_norm Conv1d 768→768 k=128 pad=64 groups=16）
        # 按 16 组切分常驻（整层批量提交时 16 个 conv1d 一次 commit）。
        pos_w, pos_b = self._layers["pos_conv"]
        for g in range(16):
            _weights.register(f"hubert.pos_conv.g{g}.weight",
                              np.ascontiguousarray(pos_w[g * 48:(g + 1) * 48],
                                                   dtype=np.float32))
            _weights.register(f"hubert.pos_conv.g{g}.bias",
                              np.ascontiguousarray(pos_b[g * 48:(g + 1) * 48],
                                                   dtype=np.float32))
        # P1-3：12 层 transformer —— attention 权重常驻（QKV 拼接 + out_proj）
        # P1-4：FFN 的 intermediate/output 权重也常驻（wi/wo 按 w.T 上传，
        # 线性层语义 x @ w.T；全 T 实测 GPU 常驻 matmul 优于 numpy BLAS
        # T=99:17ms/n层 vs 55ms+，T=499:58ms vs 75ms+，T=999:87ms vs 106ms）。
        # P1-5：整层批量提交需要 q/k/v 分离权重 + 全部 bias + LN gamma/beta；
        # q 的 1/sqrt(head_dim) 缩放折入权重与 bias（q = h@(s*Wq)ᵀ + s*bq，
        # 与 numpy 版 (h@Wqᵀ+bq)*s 差 ~1e-7 舍入，远低于全部对照阈值）。
        scale = 64.0 ** -0.5
        for i, layer in enumerate(self._layers["enc_layers"]):
            attn = layer["attn"]
            wq, bq = attn["q"]
            wk, bk = attn["k"]
            wv, bv = attn["v"]
            attn["_q_T"] = np.ascontiguousarray(wq.T * np.float32(scale),
                                                dtype=np.float32)  # [768,768]
            attn["_qb_s"] = np.ascontiguousarray(bq * np.float32(scale),
                                                 dtype=np.float32)  # [768]
            attn["_k_T"] = np.ascontiguousarray(wk.T, dtype=np.float32)
            attn["_kb"] = np.ascontiguousarray(bk, dtype=np.float32)
            attn["_v_T"] = np.ascontiguousarray(wv.T, dtype=np.float32)
            attn["_vb"] = np.ascontiguousarray(bv, dtype=np.float32)
            attn["_qkv_T"] = np.ascontiguousarray(
                np.concatenate([wq.T, wk.T, wv.T], axis=1), dtype=np.float32
            )  # [768, 2304]（P1-4 混合路径仍用）
            attn["_o_T"] = np.ascontiguousarray(attn["o"][0].T, dtype=np.float32)
            attn["_ob"] = np.ascontiguousarray(attn["o"][1], dtype=np.float32)
            _weights.register(f"hubert.enc.{i}.attn.q.weight", attn["_q_T"])
            _weights.register(f"hubert.enc.{i}.attn.q.bias", attn["_qb_s"])
            _weights.register(f"hubert.enc.{i}.attn.k.weight", attn["_k_T"])
            _weights.register(f"hubert.enc.{i}.attn.k.bias", attn["_kb"])
            _weights.register(f"hubert.enc.{i}.attn.v.weight", attn["_v_T"])
            _weights.register(f"hubert.enc.{i}.attn.v.bias", attn["_vb"])
            _weights.register(f"hubert.enc.{i}.attn.o.weight", attn["_o_T"])
            _weights.register(f"hubert.enc.{i}.attn.o.bias", attn["_ob"])
            _weights.register(f"hubert.enc.{i}.attn.qkv.weight", attn["_qkv_T"])
            ln1_gw, ln1_gb = layer["ln1"]
            _weights.register(f"hubert.enc.{i}.ln1.gamma",
                              np.ascontiguousarray(ln1_gw, dtype=np.float32))
            _weights.register(f"hubert.enc.{i}.ln1.beta",
                              np.ascontiguousarray(ln1_gb, dtype=np.float32))
            ff = layer["ff"]
            ff["_wi_T"] = np.ascontiguousarray(ff["wi"][0].T, dtype=np.float32)  # [768, 3072]
            ff["_wo_T"] = np.ascontiguousarray(ff["wo"][0].T, dtype=np.float32)  # [3072, 768]
            ff["_wib"] = np.ascontiguousarray(ff["wi"][1], dtype=np.float32)
            ff["_wob"] = np.ascontiguousarray(ff["wo"][1], dtype=np.float32)
            _weights.register(f"hubert.enc.{i}.ff.wi.weight", ff["_wi_T"])
            _weights.register(f"hubert.enc.{i}.ff.wi.bias", ff["_wib"])
            _weights.register(f"hubert.enc.{i}.ff.wo.weight", ff["_wo_T"])
            _weights.register(f"hubert.enc.{i}.ff.wo.bias", ff["_wob"])
            ln2_gw, ln2_gb = layer["ln2"]
            _weights.register(f"hubert.enc.{i}.ln2.gamma",
                              np.ascontiguousarray(ln2_gw, dtype=np.float32))
            _weights.register(f"hubert.enc.{i}.ln2.beta",
                              np.ascontiguousarray(ln2_gb, dtype=np.float32))

    # -- 权重加载 ----------------------------------------------------------

    @property
    def _batch_ok(self) -> bool:
        """整层批量提交（P1-5）可用性：开关开启 + vulkan 后端 + 关键权重已常驻。

        任一不满足则回退 P1-4 混合路径（对 numpy 后端与旧行为完全一致）。
        """
        return (
            _BATCH_ENABLED_FLAG
            and _weights.enabled
            and _weights.get("hubert.enc.0.attn.q.weight") is not None
            and _weights.get("hubert.enc.0.ln1.gamma") is not None
        )

    @staticmethod
    def _load_and_prepare(model_dir: str):
        """加载 bin 权重、统一转 float32，并把 pos_conv 的 weight_norm 解包。"""
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(
                f"HuBERT 权重不存在: {bin_path}（model_dir 应为 assets/hubert_base）"
            )
        raw = load_pth(bin_path)  # dict[str, TensorArray(float16)]
        sd: Dict[str, np.ndarray] = {}
        for k, v in raw.items():
            sd[_strip_prefix(k)] = np.asarray(v).astype(_F32)

        # weight_norm(dim=2) 解包：W = original0 * original1/‖original1‖₂
        # original0: [1,1,128]（weight_g），original1: [768,48,128]（weight_v）
        try:
            o0 = sd["encoder.pos_conv_embed.conv.parametrizations.weight.original0"]
            o1 = sd["encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
        except KeyError:
            # 兼容旧式 weight_g/weight_v 键名
            o0 = sd["encoder.pos_conv_embed.conv.weight_g"]
            o1 = sd["encoder.pos_conv_embed.conv.weight_v"]
        norm = np.sqrt(np.sum(o1.astype(np.float64) ** 2, axis=(0, 1), keepdims=True))
        pos_w = (o0 * (o1 / norm.astype(_F32))).astype(_F32)
        sd["_pos_conv_weight"] = pos_w

        layers = {
            "proj_ln": (sd["feature_projection.layer_norm.weight"],
                        sd["feature_projection.layer_norm.bias"]),
            "proj": (sd["feature_projection.projection.weight"],
                     sd["feature_projection.projection.bias"]),
            "enc_ln": (sd["encoder.layer_norm.weight"],
                       sd["encoder.layer_norm.bias"]),
            "pos_conv": (pos_w, sd["encoder.pos_conv_embed.conv.bias"]),
            "final_proj": (sd["final_proj.weight"], sd["final_proj.bias"]),
        }
        # conv 栈
        layers["conv"] = []
        for i in range(7):
            w = sd[f"feature_extractor.conv_layers.{i}.conv.weight"]
            layers["conv"].append((w, None))
        layers["conv_gamma"] = sd["feature_extractor.conv_layers.0.layer_norm.weight"]
        layers["conv_beta"] = sd["feature_extractor.conv_layers.0.layer_norm.bias"]
        # 12 个 transformer 层
        layers["enc_layers"] = []
        for i in range(12):
            p = f"encoder.layers.{i}."
            attn = {
                "q": (sd[p + "attention.q_proj.weight"], sd[p + "attention.q_proj.bias"]),
                "k": (sd[p + "attention.k_proj.weight"], sd[p + "attention.k_proj.bias"]),
                "v": (sd[p + "attention.v_proj.weight"], sd[p + "attention.v_proj.bias"]),
                "o": (sd[p + "attention.out_proj.weight"], sd[p + "attention.out_proj.bias"]),
            }
            ln1 = (sd[p + "layer_norm.weight"], sd[p + "layer_norm.bias"])
            ff = {
                "wi": (sd[p + "feed_forward.intermediate_dense.weight"],
                       sd[p + "feed_forward.intermediate_dense.bias"]),
                "wo": (sd[p + "feed_forward.output_dense.weight"],
                       sd[p + "feed_forward.output_dense.bias"]),
            }
            ln2 = (sd[p + "final_layer_norm.weight"], sd[p + "final_layer_norm.bias"])
            layers["enc_layers"].append({"attn": attn, "ln1": ln1, "ff": ff, "ln2": ln2})
        return sd, layers

    # -- 前向 --------------------------------------------------------------

    def encode(self, x: np.ndarray, version: int = 2) -> np.ndarray:
        """对整段音频做 HuBERT 编码。

        参数:
            x: ``[1, T]`` float32 16kHz（调用方已做整段 LayerNorm eps=1e-5 无仿射）
            version: 1 → ``[1, L, 256]``（第 9 层 hidden 经 final_proj）；
                     2 → ``[1, L, 768]``（last_hidden_state）
        返回 float32 ndarray。
        """
        if version not in (1, 2):
            raise ValueError(f"version 仅支持 1/2，收到 {version}")
        x = np.asarray(x, dtype=_F32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"输入应为 [1, T]，收到 {x.shape}")

        hidden_states = self._feature_extractor(x)
        hidden_states = self._feature_projection(hidden_states)
        if version == 1:
            hs = self._encoder(hidden_states, need={9})
            out = hs[9]  # 第 9 层（1-based）输出
            w, b = self._layers["final_proj"]
            return nn_ops.linear(out, w, b)  # [1, L, 256]
        hs = self._encoder(hidden_states, need={-1})
        return hs[-1]  # last_hidden_state，[1, L, 768]

    # -- P2：多块批量 encode ----------------------------------------------

    def encode_batch(self, xs, version: int = 2):
        """批量编码多块音频（P2，多块一次 GPU encode）。

        参数:
            xs: list of ``[1, T_i]`` float32（各块独立、长度**可不等**，
                调用方已做整段 LayerNorm，同 ``encode`` 约定）。
            version: 1 → 各块 ``[1, L_i, 256]``；2 → ``[1, L_i, 768]``。
        返回 list of ndarray（每块与 ``encode`` 输出语义一致）。

        实现（详见模块 docstring 的 P2 节）：vulkan 后端下 conv 栈 /
        feature_projection / pos_conv 各阶段按块展开录进一个 BatchRunner
        一次 commit；12 层 transformer 全部录进**同一个** runner、每层
        commit 一次（受引擎 384-dispatch 上限约束，见 ``_BATCH_MAX_BLOCKS``），
        中间张量全程 GPU 驻留、只下载需要层。各块按自身长度独立录算子，
        **无需 padding / mask** → 与逐块路径同 kernel 同参数（逐位一致，
        仅 feature_projection 的 bias 从 numpy 加改为 GPU bias_add，差
        ≤1 ulp ≪ 1e-4）。numpy 后端 / 块数超限 / 引擎异常 → 逐块回退
        （行为与逐块串行完全一致）。
        """
        if version not in (1, 2):
            raise ValueError(f"version 仅支持 1/2，收到 {version}")
        if isinstance(xs, np.ndarray):
            xs = [xs]
        xlist = []
        for x in xs:
            x = np.asarray(x, dtype=_F32)
            if x.ndim == 1:
                x = x[None, :]
            if x.ndim != 2 or x.shape[0] != 1:
                raise ValueError(f"batch 输入应为 [1, T]，收到 {x.shape}")
            xlist.append(x)
        out: list = []
        for g in range(0, len(xlist), _BATCH_MAX_BLOCKS):
            out.extend(self._encode_batch_group(xlist[g:g + _BATCH_MAX_BLOCKS],
                                                version))
        return out

    def _encode_batch_group(self, xlist, version: int) -> list:
        """单组（≤19 块）批量 encode；任一阶段异常 → 该阶段回退（保数值）。

        conv 栈单独 try：10s 长块的 conv 第 1 层输出（512×L 帧）可超引擎
        GPU grid 上限（_GRID_POINTS_MAX，P1-5 单块已有此回退），此时只
        回退 conv 阶段到逐块 ``_feature_extractor``（行为与现状一致），
        后续 feature_projection / encoder 仍批量。其余阶段异常 → 整体
        逐块 encode。
        """
        if not self._batch_ok:
            return [self.encode(x, version) for x in xlist]
        try:
            hs = self._feature_extractor_batch_multi(xlist)
        except (RuntimeError, ValueError):
            hs = [self._feature_extractor(x) for x in xlist]
        try:
            hs = self._feature_projection_batch_multi(hs)     # list [1,L_i,768]
            if version == 1:
                outs = self._encoder_batch_multi(hs, need={9})
                w, b = self._layers["final_proj"]
                return [nn_ops.linear(o, w, b) for o in outs[9]]
            outs = self._encoder_batch_multi(hs, need={-1})
            return outs[-1]  # list [1,L_i,768]
        except (RuntimeError, ValueError):
            return [self.encode(x, version) for x in xlist]

    # -- 子模块 ------------------------------------------------------------

    def _feature_extractor(self, x: np.ndarray) -> np.ndarray:
        """conv 栈：x [1,T] → [1, L, 512]（transpose 后）。

        P1-2：7 层 Conv1d 权重常驻（hubert.conv.{i}.weight），f32 输入命中
        常驻 buffer 时走 ``vulkan_ops.conv1d``（跳过每次 upload），否则回退
        ``nn_ops.conv1d``（numpy 后端 / 未注册时默认行为不变）。GroupNorm
        （第 0 层后，512 组）无 GPU kernel，保持 numpy。

        P1-5：vulkan 后端时优先走 ``_feature_extractor_batch`` —— 7 层
        conv1d + GroupNorm（batch group_norm op）+ 7 次 GELU 录进 **一次**
        batch commit，中间张量全程 GPU 驻留、只下载最终 [1,512,L]。
        """
        with self._pt("conv_stack"):
            if self._batch_ok:
                try:
                    return self._feature_extractor_batch(x)
                except RuntimeError:
                    pass  # 超限/引擎异常 → 回退下方逐次路径
            h = x[:, None, :]  # [1,1,T]
            convs = self._layers["conv"]
            for i, (w, b) in enumerate(convs):
                stride = 5 if i == 0 else 2
                pbw = _weights.get(f"hubert.conv.{i}.weight")
                if pbw is not None and h.dtype == _F32:
                    from runtime import vulkan_ops  # noqa: PLC0415

                    h = vulkan_ops.conv1d(h, w, None, stride=stride, padding=0,
                                          buf_w=pbw)
                else:
                    h = nn_ops.conv1d(h, w, None, stride=stride, padding=0)
                if i == 0:
                    h = nn_ops.group_norm(
                        h,
                        self._layers["conv_gamma"],
                        self._layers["conv_beta"],
                        num_groups=512, eps=1e-5,
                    )
                h = gelu_erf(h)
            return h.transpose(0, 2, 1)  # [1, L, 512]

    def _feature_extractor_batch(self, x: np.ndarray) -> np.ndarray:
        """conv 栈整层批量提交（P1-5）：7×conv1d + GroupNorm + 7×GELU 一次
        ``rvc_batch_commit``，中间张量全部为 BatchTensor（GPU 驻留）。

        GroupNorm(512 组) 即逐通道对帧维归一 —— batch group_norm op15
        （rows=512, Cpg=1, S=帧数）；GELU 用 op10（erf 版，与 ``gelu_erf``
        同公式，差 ~1e-6）。conv1d/gelu 输出超 GPU grid 上限时 **GPU 内
        分段**（``_conv1d_batch_or_seg`` → conv1d_seg_multi；``gelu_seg_multi``），
        超限链不再回退逐次 numpy 往返（T1.1 Step5）；不超限走原路径（零回归）。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        convs = self._layers["conv"]
        br = BatchRunner(get_context())
        try:
            with self._pt("conv_stack.batch"):
                h = x[:, None, :]  # [1,1,T]
                for i, (w, b) in enumerate(convs):
                    stride = 5 if i == 0 else 2
                    h = _conv1d_batch_or_seg(
                        br, h, w, b, f"hubert.conv.{i}.weight", stride,
                    )
                    if i == 0:
                        h = br.group_norm(
                            h, self._layers["conv_gamma"],
                            self._layers["conv_beta"], num_groups=512,
                            eps=1e-5,
                            buf_gamma=_weights.get("hubert.conv.gn.gamma"),
                            buf_beta=_weights.get("hubert.conv.gn.beta"),
                        )
                    h = br.gelu_seg_multi(h)
                br.commit()
                out = h.numpy()  # [1, 512, L]
            return out.transpose(0, 2, 1)  # [1, L, 512]
        finally:
            br.release()

    def _feature_extractor_batch_multi(self, xs: list) -> list:
        """conv 栈多块批量（P2）：B×(7 conv1d + GroupNorm + 7 GELU) 录进
        一个 BatchRunner 一次 commit。

        各块按自身长度独立录制（B×15 ≤ 380 个 dispatch，受 _BATCH_MAX_BLOCKS
        约束不超引擎 384 上限），与逐块 ``_feature_extractor_batch`` 同
        kernel 同参数 → 输出逐位一致。conv1d/gelu 超限层 GPU 内分段
        （T1.1 Step5，同 ``_feature_extractor_batch``）。GroupNorm(op15)
        引擎要求 B=1，故每块单独录制（512 组 = 逐通道对帧维归一）。
        返回 list of ``[1, L_i, 512]``。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        convs = self._layers["conv"]
        br = BatchRunner(get_context())
        try:
            with self._pt("conv_stack.batch"):
                hs = []
                for x in xs:
                    h = x[:, None, :]  # [1,1,T_i]
                    for i, (w, b) in enumerate(convs):
                        stride = 5 if i == 0 else 2
                        h = _conv1d_batch_or_seg(
                            br, h, w, b, f"hubert.conv.{i}.weight", stride,
                        )
                        if i == 0:
                            h = br.group_norm(
                                h, self._layers["conv_gamma"],
                                self._layers["conv_beta"], num_groups=512,
                                eps=1e-5,
                                buf_gamma=_weights.get("hubert.conv.gn.gamma"),
                                buf_beta=_weights.get("hubert.conv.gn.beta"),
                            )
                        h = br.gelu_seg_multi(h)
                    hs.append(h)  # [1, 512, L_i]
                br.commit()
                return [h.numpy().transpose(0, 2, 1) for h in hs]  # [1,L_i,512]
        finally:
            br.release()

    def _feature_projection(self, hidden_states: np.ndarray) -> np.ndarray:
        """LayerNorm(512) → Linear(512→768)。"""
        with self._pt("feat_proj"):
            gw, gb = self._layers["proj_ln"]
            pbg = _weights.get("hubert.proj_ln.gamma")
            pbb = _weights.get("hubert.proj_ln.beta")
            if pbg is not None and pbb is not None and hidden_states.dtype == _F32:
                from runtime import vulkan_ops  # noqa: PLC0415

                hidden_states = vulkan_ops.layer_norm(
                    hidden_states, gw, gb, eps=1e-5, buf_gamma=pbg, buf_beta=pbb
                )
            else:
                hidden_states = nn_ops.layer_norm(hidden_states, gw, gb, eps=1e-5)
            w, b = self._layers["proj"]
            pbw = _weights.get("hubert.proj.weight")
            if pbw is not None and hidden_states.dtype == _F32:
                from runtime import vulkan_ops  # noqa: PLC0415

                D = hidden_states.shape[-1]
                x2 = hidden_states.reshape(-1, D)  # [L, 512]
                out = vulkan_ops.matmul(x2, w.T, buf_b=pbw)  # [L, 768]
                out = out.reshape(hidden_states.shape[:-1] + (w.shape[0],))
            else:
                # 注意 linear 的 bias 置 None：统一在下方加，避免与上方重复加 bias
                out = nn_ops.linear(hidden_states, w, None)
            if b is not None:
                out = out + np.asarray(b, dtype=out.dtype).reshape(1, -1)
            return out

    def _feature_projection_batch_multi(self, hs_list: list) -> list:
        """feature_projection 多块批量（P2）：B×(LN + proj matmul +
        bias_add) 录进一个 BatchRunner 一次 commit。

        与逐块路径同 kernel 同输入（LN/matmul 逐位一致）；bias 由 numpy
        加改为 GPU ``bias_add``（op11 同公式，差 ≤1 ulp ≪ 1e-4 对照要求）。
        返回 list of ``[1, L_i, 768]``。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        gw, gb = self._layers["proj_ln"]
        w, b = self._layers["proj"]
        br = BatchRunner(get_context())
        try:
            with self._pt("feat_proj.batch"):
                outs = []
                for h in hs_list:
                    t = br.layer_norm(
                        h, gw, gb, eps=1e-5,
                        buf_gamma=_weights.get("hubert.proj_ln.gamma"),
                        buf_beta=_weights.get("hubert.proj_ln.beta"),
                    )  # [1, L_i, 512]
                    D = t.shape[-1]
                    t = br.matmul(t.reshape(-1, D), w.T, buf_b=(
                        _weights.get("hubert.proj.weight")
                        or np.ascontiguousarray(w.T, dtype=np.float32)))
                    t = br.bias_add(t, b)  # [L_i, 768]
                    outs.append(t)
                br.commit()
                return [t.numpy().reshape(1, -1, 768) for t in outs]
        finally:
            br.release()

    def _pos_conv_embed(self, hidden_states: np.ndarray) -> np.ndarray:
        """位置卷积编码：weight_norm Conv1d(768,768,k128,pad64,groups16) + GELU。

        hidden_states: ``[1, L, 768]`` → 返回 ``[1, L, 768]``（L 不变）。

        P1-5：vulkan 后端时优先走 ``_pos_conv_embed_batch``（16 组 conv1d
        一次 batch commit + 常驻组权重，替代 16 次单算子调用）。
        """
        with self._pt("pos_conv"):
            if self._batch_ok:
                try:
                    return self._pos_conv_embed_batch(hidden_states)
                except RuntimeError:
                    pass  # 超限/异常 → 回退逐组路径
            h = hidden_states.transpose(0, 2, 1)  # [1, 768, L]
            w, b = self._layers["pos_conv"]
            h = _grouped_conv1d(h, w, b, groups=16, stride=1, padding=64)
            h = h[:, :, :hidden_states.shape[1]]  # 裁剪右 1 列（SamePadLayer）
            h = gelu_erf(h)
            return h.transpose(0, 2, 1)

    def _pos_conv_embed_batch(self, hidden_states: np.ndarray) -> np.ndarray:
        """pos_conv 批量提交（P1-5）：16 个分组 conv1d 一次 commit。

        每组 ``[1,48,L]→[1,48,L+1]``（k=128, pad=64, stride=1），权重按组
        常驻（hubert.pos_conv.g{i}.weight/bias）；下载后 numpy 裁剪右 1 列 +
        GELU + 转置 —— 与逐组路径逐位一致（同一 conv1d kernel）。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        h = hidden_states.transpose(0, 2, 1)  # [1, 768, L]
        w, b = self._layers["pos_conv"]
        L = hidden_states.shape[1]
        br = BatchRunner(get_context())
        try:
            with self._pt("pos_conv.batch"):
                outs = []
                for g in range(16):
                    t = br.conv1d(
                        h[:, g * 48:(g + 1) * 48, :],
                        w[g * 48:(g + 1) * 48], b[g * 48:(g + 1) * 48],
                        stride=1, padding=64,
                        buf_w=_weights.get(f"hubert.pos_conv.g{g}.weight"),
                        buf_b=_weights.get(f"hubert.pos_conv.g{g}.bias"),
                    )
                    outs.append(t)
                br.commit()
                parts = [t.numpy() for t in outs]  # 16×[1,48,L+1]
            out = np.concatenate(parts, axis=1)[:, :, :L]  # 裁剪右 1 列
            out = gelu_erf(out)
            return out.transpose(0, 2, 1)
        finally:
            br.release()

    def _pos_conv_embed_batch_multi(self, hs_list: list) -> list:
        """pos_conv 多块批量（P2）：B×16 个分组 conv1d 录进一个 runner
        一次 commit（16B ≤ 304 个 dispatch，B≤19）。每组权重常驻；下载后
        numpy 裁剪右 1 列 + GELU + 转置 —— 与逐组路径逐位一致。
        返回 list of ``[1, L_i, 768]``。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        w, b = self._layers["pos_conv"]
        br = BatchRunner(get_context())
        try:
            with self._pt("pos_conv.batch"):
                groups = []
                for h in hs_list:
                    ht = h.transpose(0, 2, 1)  # [1, 768, L_i]
                    parts = []
                    for g in range(16):
                        parts.append(br.conv1d(
                            ht[:, g * 48:(g + 1) * 48, :],
                            w[g * 48:(g + 1) * 48], b[g * 48:(g + 1) * 48],
                            stride=1, padding=64,
                            buf_w=_weights.get(f"hubert.pos_conv.g{g}.weight"),
                            buf_b=_weights.get(f"hubert.pos_conv.g{g}.bias"),
                        ))
                    groups.append(parts)
                br.commit()
                out = []
                for parts, h in zip(groups, hs_list):
                    L = h.shape[1]
                    cat = np.concatenate([t.numpy() for t in parts],
                                         axis=1)[:, :, :L]
                    out.append(gelu_erf(cat).transpose(0, 2, 1))
                return out
        finally:
            br.release()

    def _encoder(self, hidden_states: np.ndarray, need=None):
        """Encoder 入口 + 12 层 post-LN transformer，返回全部层输出。

        返回 ``[13, 1, L, 768]``（np.stack）：索引 0 为 encoder 入口
        （pos_conv + LN 后），索引 k（1..12）为第 k 层输出 —— 与
        transformers output_hidden_states 完全一致。

        P1-5：``need`` 为可选索引集合（如 ``{9}`` / ``{-1}``）。vulkan
        批量路径时只下载需要的层输出（GPU 驻留其余张量）；返回
        ``{idx: array}`` 字典（支持 ``hs[9]`` / ``hs[-1]``）。None 时
        下载全部 13 层并返回 np.stack（与旧行为一致，供自测对照）。
        """
        pos = None
        if self._batch_ok:
            try:
                pos = self._pos_conv_embed_batch(hidden_states)
            except (RuntimeError, ValueError):
                pos = None
        if pos is None:
            with self._pt("pos_conv"):
                pos = self._pos_conv_embed(hidden_states)
        h = hidden_states + pos

        gw, gb = self._layers["enc_ln"]
        if self._batch_ok and not _ATTN_NUMPY_FORCED:
            out = self._encoder_gpu(h, gw, gb, need)
            if out is not None:
                if need is None:
                    return np.stack(out, axis=0)  # [13,1,L,768]
                return out  # {idx: array}
        h = _layer_norm_np(h, gw, gb, eps=1e-5)

        outs = [h]
        scale = 64.0 ** -0.5  # head_dim = 768 // 12 = 64
        for i, layer in enumerate(self._layers["enc_layers"]):
            h = self._encoder_layer(h, layer, i, scale)
            outs.append(h)
        stacked = np.stack(outs, axis=0)
        if need is None:
            return stacked
        return {int(k): stacked[k] for k in need}

    def _encoder_gpu(self, h: np.ndarray, gw, gb, need):
        """12 层 transformer **整层一次批量提交**（P1-5 核心）。

        入口 LayerNorm + 12 层 × 20 个算子全部录进一个 BatchRunner，一次
        ``rvc_batch_commit``：中间张量（qkv / scores [12,T,T] / attnW /
        ctx / wi 中间层…）全部驻留 GPU 显存，不落一次内存；最终按
        ``need`` 只下载需要的层输出。依赖顺序由 recorder 的 dispatch 间
        barrier 保证（录制顺序 = 执行顺序）。

        任何算子超 GPU grid 上限 / 引擎异常 → 返回 None，调用方回退
        P1-4 混合路径（数值上混合路径与批量路径只差 matmul 舍入 ~1e-6）。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        br = BatchRunner(get_context())
        try:
            with self._pt("enc.batch"):
                t = br.layer_norm(
                    h, gw, gb, eps=1e-5,
                    buf_gamma=_weights.get("hubert.enc_ln.gamma"),
                    buf_beta=_weights.get("hubert.enc_ln.beta"),
                )
                tensors = [t]
                for i, layer in enumerate(self._layers["enc_layers"]):
                    t = self._encoder_layer_batch(br, t, layer, i)
                    tensors.append(t)
                br.commit()
                if need is None:
                    return [tt.numpy() for tt in tensors]
                return {int(k): tensors[k].numpy() for k in need}
        except (RuntimeError, ValueError):
            try:
                br.discard()
            except RuntimeError:
                pass
            return None
        finally:
            try:
                br.release()
            except RuntimeError:
                pass

    def _encoder_batch_multi(self, hs_list: list, need=None):
        """Encoder 多块批量（P2）：pos_conv batch → h=hidden+pos（numpy）
        → 12 层 transformer **一个 runner**（每层 commit 一次，中间张量
        GPU 驻留）→ 只下载 need 层。

        返回 ``{idx: [每块 array [1,L_i,768]]}``（idx 0..12，0=入口 LN，
        k≥1 为第 k 层输出，与 transformers output_hidden_states 一致）。
        ``need=None`` 返回全部 13 层（自测对照用）。任一阶段异常回退
        逐块 ``_encoder``（数值与逐块路径一致）。
        """
        try:
            pos_list = self._pos_conv_embed_batch_multi(hs_list)
        except (RuntimeError, ValueError):
            pos_list = None
        if pos_list is None:
            pos_list = [self._pos_conv_embed(h) for h in hs_list]
        h_list = [h + p for h, p in zip(hs_list, pos_list)]
        gw, gb = self._layers["enc_ln"]
        if self._batch_ok and not _ATTN_NUMPY_FORCED:
            out = self._encoder_gpu_multi(h_list, gw, gb, need)
            if out is not None:
                return out
        # 回退：逐块 _encoder（numpy 后端 / 引擎异常）
        per = [self._encoder(h, need) for h in h_list]
        if need is None:
            return {k: [blk[k] for blk in per] for k in range(13)}
        return {int(k): [blk[int(k)] for blk in per] for k in need}

    def _encoder_gpu_multi(self, h_list: list, gw, gb, need):
        """12 层 × N 块整层批量提交（P2 核心）。

        与单块 ``_encoder_gpu`` 同一套算子序列，但 N 块的 12 层全部录进
        **同一个** BatchRunner：每层 commit 一次（N×20 ≤ 380 个 dispatch，
        受引擎 recorder max_sets=384 约束，见 ``_BATCH_MAX_BLOCKS``），
        层间中间张量（每块 [1,T_i,768]）保持 GPU 驻留，仅按 ``need``
        下载需要的层输出。同 kernel 同参数 → 每块与逐块路径逐位一致。
        任何算子超限 / 引擎异常 → 返回 None，调用方逐块回退。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        br = BatchRunner(get_context())
        try:
            with self._pt("enc.batch"):
                cur = [
                    br.layer_norm(
                        h, gw, gb, eps=1e-5,
                        buf_gamma=_weights.get("hubert.enc_ln.gamma"),
                        buf_beta=_weights.get("hubert.enc_ln.beta"),
                    ) for h in h_list
                ]  # 每块 [1, T_i, 768]（入口 LN 输出 = tensors[k=0]）
                layers_out = [[t] for t in cur]  # [block][k]
                for i, layer in enumerate(self._layers["enc_layers"]):
                    cur = self._encoder_layer_batch_multi(br, cur, layer, i)
                    # 每层一次提交；提交后**立即 wait**（P1-9）：解锁 BatchRunner
                    # 冻结的本地池 —— 本层 tensor_done 回收的中间 buffer（q/k/v/
                    # scores/attnW/ctx/...）在 GPU 消费完毕后，下一层可直接复用，
                    # 使 12 层 × 16 块的中间输出分配从"每层 16×(11+3) 次"降到
                    # "首层一次、其后全命中"。wait 开销 ~0.3ms/层（12 层 ~4ms）
                    # 相对每层 ~80ms 的 GPU 计算可忽略（recorder MAX_INFLIGHT
                    # 轮转仍保证在途帧不被覆盖）。
                    br.commit(async_=True)
                    br.wait()
                    for b, t in enumerate(cur):
                        layers_out[b].append(t)
                if need is None:
                    return {k: [lo[k].numpy() for lo in layers_out]
                            for k in range(13)}
                return {int(k): [lo[k].numpy() for lo in layers_out]
                        for k in need}
        except (RuntimeError, ValueError):
            try:
                br.discard()
            except RuntimeError:
                pass
            return None
        finally:
            try:
                br.release()
            except RuntimeError:
                pass

    def _encoder_layer_batch_multi(self, br, hs: list, layer: dict,
                                   idx: int) -> list:
        """单层 × N 块算子**录制**（P2，与单块 ``_encoder_layer_batch``
        逐位同 kernel 同参数：每块独立 T_i，无 padding / mask）。

        返回 list of 层输出 BatchTensor ``[1, T_i, 768]``（LN2 输出，
        下一层输入 / need 下载用）。

        P1-9（输出 buffer 复用）：每块每个中间张量在其**最后一个消费算子
        录制后**立即 ``tensor_done`` 归还 runner 本地池（粗桶），供同层
        后续块 / 同数量级输出复用——scores [12,T,T]（16 块 × 12 层约
        190MB/层）与 attn_w 是最大回收点。本地池顺序安全（录制序 = GPU
        执行序 + recorder 依赖 barrier）；跨 commit 复用由 BatchRunner 的
        frozen-pool 机制保证（commit 后 wait 才解锁，见 vulkan_ops）。
        """
        attn = layer["attn"]
        ff = layer["ff"]
        H_, D = 12, 64
        i = idx
        outs = []
        for h in hs:
            C = h.shape[-1]
            T = int(np.prod(h.shape[:-1])) if len(h.shape) > 2 else h.shape[0]
            h2 = h.reshape(T, C) if tuple(h.shape) != (T, C) else h
            # 残差 add_inplace 写回输入 buffer（上一层输出 tensor），
            # 先 copy 出工作副本保留本层输入原值（同单块版）。
            h2 = br.copy(h2)
            q = br.matmul(h2, attn["_q_T"],
                          buf_b=_weights.get(f"hubert.enc.{i}.attn.q.weight"))
            q = br.bias_add(q, attn["_qb_s"],
                            buf_bias=_weights.get(f"hubert.enc.{i}.attn.q.bias"))
            k = br.matmul(h2, attn["_k_T"],
                          buf_b=_weights.get(f"hubert.enc.{i}.attn.k.weight"))
            k = br.bias_add(k, attn["_kb"],
                            buf_bias=_weights.get(f"hubert.enc.{i}.attn.k.bias"))
            v = br.matmul(h2, attn["_v_T"],
                          buf_b=_weights.get(f"hubert.enc.{i}.attn.v.weight"))
            v = br.bias_add(v, attn["_vb"],
                            buf_bias=_weights.get(f"hubert.enc.{i}.attn.v.bias"))
            scores = br.attn_qk(q, k, H_, T, D, C)
            br.tensor_done(q)  # q/k 已被 attn_qk 消费
            br.tensor_done(k)
            attn_w = br.softmax(scores, axis_len=T)
            br.tensor_done(scores)  # scores 已被 softmax 消费
            ctx = br.attn_sv(attn_w, v, H_, T, D, C)
            br.tensor_done(attn_w)  # attnW 已被 attn_sv 消费
            br.tensor_done(v)  # v 已被 attn_sv 消费
            o = br.matmul(ctx, attn["_o_T"],
                          buf_b=_weights.get(f"hubert.enc.{i}.attn.o.weight"))
            br.tensor_done(ctx)  # ctx 已被 out_proj 消费
            o = br.bias_add(o, attn["_ob"],
                            buf_bias=_weights.get(f"hubert.enc.{i}.attn.o.bias"))
            br.add_inplace(h2, o)
            br.tensor_done(o)  # o 已被 add_inplace 消费
            h = br.layer_norm(
                h2, *layer["ln1"], eps=1e-5,
                buf_gamma=_weights.get(f"hubert.enc.{i}.ln1.gamma"),
                buf_beta=_weights.get(f"hubert.enc.{i}.ln1.beta"),
            )
            br.tensor_done(h2)  # LN1 已读尽 h2（add_inplace 写回后的工作副本）
            it = br.matmul(h, ff["_wi_T"],
                           buf_b=_weights.get(f"hubert.enc.{i}.ff.wi.weight"))
            it = br.bias_add(it, ff["_wib"],
                             buf_bias=_weights.get(f"hubert.enc.{i}.ff.wi.bias"))
            it = br.gelu(it)
            ot = br.matmul(it, ff["_wo_T"],
                           buf_b=_weights.get(f"hubert.enc.{i}.ff.wo.weight"))
            br.tensor_done(it)  # gelu 后 it 已被 wo 消费
            ot = br.bias_add(ot, ff["_wob"],
                             buf_bias=_weights.get(f"hubert.enc.{i}.ff.wo.bias"))
            br.add_inplace(h, ot)
            br.tensor_done(ot)  # ot 已被 add_inplace 消费
            out = br.layer_norm(
                h, *layer["ln2"], eps=1e-5,
                buf_gamma=_weights.get(f"hubert.enc.{i}.ln2.gamma"),
                buf_beta=_weights.get(f"hubert.enc.{i}.ln2.beta"),
            )
            br.tensor_done(h)  # LN2 已读尽 h（add_inplace 写回后的 FFN 载体）
            outs.append(out.reshape(1, T, C) if out.shape != (1, T, C) else out)
        return outs

    def _encoder_layer_batch(self, br, h, layer: dict, idx: int) -> "BatchTensor":
        """单层 transformer 全部算子的**录制**（不提交）。

        算子序列（20 个 dispatch，与 P1-4 混合路径逐位同 kernel 同布局）::

            q = h@(s·Wq)ᵀ + s·bq;  k = h@Wkᵀ+bk;  v = h@Wvᵀ+bv   （3 matmul + 3 bias）
            scores = attn_qk(q,k)  -> [12,T,T]                    （融合 12 头）
            attnW  = softmax(scores, 最后一维=T)
            ctx    = attn_sv(attnW, v) -> [T,768]                 （头合并回交错）
            o      = ctx@Woᵀ+bo;  h += o;  h = LN1(h)             （2+1+1）
            i      = h@Wiᵀ+bi;  i = gelu(i);  o = i@Woᵀ+bo
            h      = h + o;  h = LN2(h)                           （3+2+1）

        q 的 1/√64 缩放折入权重/bias（``_q_T``/``_qb_s``，见注册处）；scale
        折叠与 numpy 版 (h@Wqᵀ+bq)*s 只差 ~1e-7 舍入。返回 LN2 输出的
        BatchTensor（下一层 / 最终下载用）。
        """
        attn = layer["attn"]
        ff = layer["ff"]
        C = h.shape[-1]
        T = int(np.prod(h.shape[:-1])) if len(h.shape) > 2 else h.shape[0]
        h2 = h.reshape(T, C) if tuple(h.shape) != (T, C) else h
        # 关键：残差 add_inplace 会写回**输入 buffer**（= 上一层输出 tensor）。
        # 若直接就地累加，最后统一下载时所有中间层输出都已被下一层污染
        # （v1 取第 9 层输出时实测 max|Δ|=3.9）。先 copy 出工作副本，
        # 保留本层输入（上一层输出）原值；1 个 copy dispatch 开销可忽略。
        h2 = br.copy(h2)
        H_, D = 12, 64
        i = idx

        q = br.matmul(h2, attn["_q_T"],
                      buf_b=_weights.get(f"hubert.enc.{i}.attn.q.weight"))
        q = br.bias_add(q, attn["_qb_s"],
                        buf_bias=_weights.get(f"hubert.enc.{i}.attn.q.bias"))
        k = br.matmul(h2, attn["_k_T"],
                      buf_b=_weights.get(f"hubert.enc.{i}.attn.k.weight"))
        k = br.bias_add(k, attn["_kb"],
                        buf_bias=_weights.get(f"hubert.enc.{i}.attn.k.bias"))
        v = br.matmul(h2, attn["_v_T"],
                      buf_b=_weights.get(f"hubert.enc.{i}.attn.v.weight"))
        v = br.bias_add(v, attn["_vb"],
                        buf_bias=_weights.get(f"hubert.enc.{i}.attn.v.bias"))
        scores = br.attn_qk(q, k, H_, T, D, C)
        br.tensor_done(q)
        br.tensor_done(k)
        attn_w = br.softmax(scores, axis_len=T)
        br.tensor_done(scores)
        ctx = br.attn_sv(attn_w, v, H_, T, D, C)
        br.tensor_done(attn_w)
        br.tensor_done(v)

        o = br.matmul(ctx, attn["_o_T"],
                      buf_b=_weights.get(f"hubert.enc.{i}.attn.o.weight"))
        br.tensor_done(ctx)
        o = br.bias_add(o, attn["_ob"],
                        buf_bias=_weights.get(f"hubert.enc.{i}.attn.o.bias"))
        br.add_inplace(h2, o)
        br.tensor_done(o)
        h = br.layer_norm(
            h2, *layer["ln1"], eps=1e-5,
            buf_gamma=_weights.get(f"hubert.enc.{i}.ln1.gamma"),
            buf_beta=_weights.get(f"hubert.enc.{i}.ln1.beta"),
        )
        br.tensor_done(h2)

        it = br.matmul(h, ff["_wi_T"],
                       buf_b=_weights.get(f"hubert.enc.{i}.ff.wi.weight"))
        it = br.bias_add(it, ff["_wib"],
                         buf_bias=_weights.get(f"hubert.enc.{i}.ff.wi.bias"))
        it = br.gelu(it)
        ot = br.matmul(it, ff["_wo_T"],
                       buf_b=_weights.get(f"hubert.enc.{i}.ff.wo.weight"))
        br.tensor_done(it)
        ot = br.bias_add(ot, ff["_wob"],
                         buf_bias=_weights.get(f"hubert.enc.{i}.ff.wo.bias"))
        br.add_inplace(h, ot)
        br.tensor_done(ot)
        out = br.layer_norm(
            h, *layer["ln2"], eps=1e-5,
            buf_gamma=_weights.get(f"hubert.enc.{i}.ln2.gamma"),
            buf_beta=_weights.get(f"hubert.enc.{i}.ln2.beta"),
        )
        br.tensor_done(h)
        # 与 numpy 路径的层输出形状 [1, T, C] 对齐（同一 buffer 的视图）
        return out.reshape(1, T, C) if out.shape != (1, T, C) else out

    # -- transformer 层：attention / FFN GPU 化（P1-3）------------------------

    def _encoder_layer(self, h: np.ndarray, layer: dict, idx: int,
                       scale: float) -> np.ndarray:
        """单层 post-LN transformer 层（P1-4 混合 GPU 路径）。

        attention 的 QKV/out_proj 在常驻权重命中时走 ``vulkan_ops``
        matmul（各一次 dispatch），scores/softmax/加权和保持 numpy（逐
        head 小 matmul GPU 实测慢 3-12x，见 ``_attn_hybrid`` 说明）；
        FFN 的 wi/wo 全部走常驻 GPU matmul。LayerNorm / 残差加保持 numpy
        （引擎无 batch 录制，单算子固定开销 ~6-13ms 远超 numpy 的 <1ms，
        数值与 backend 分派的 numpy 分支一致；GELU 引擎无算子，保持
        numpy 的 float32 erf 精确版）。
        """
        # a) 自注意力
        with self._pt("attn"):
            attn_out = self._attn_opt(h, layer, idx, scale)

        # b) 残差 + c) post-attn LayerNorm
        with self._pt("ln1"):
            h = h + attn_out
            h = _layer_norm_np(h, *layer["ln1"], eps=1e-5)

        # d) FFN（intermediate + gelu + output）+ e) 残差 + final LayerNorm
        with self._pt("ffn"):
            h = h + self._ffn_opt(h, layer, idx)
        with self._pt("ln2"):
            h = _layer_norm_np(h, *layer["ln2"], eps=1e-5)
        return h

    def _attn_opt(self, h: np.ndarray, layer: dict, idx: int,
                  scale: float) -> np.ndarray:
        """attention 主计算入口（P1-4 数据驱动的混合 GPU/numpy 路径）。

        命中条件：qkv/o 两组常驻权重均已注册（vulkan 后端）且输入 float32。
        命中走 ``_attn_hybrid``：K=768 的 QKV/out_proj 各一次 GPU matmul
        （常驻权重，单 dispatch）＋ scores/softmax/加权和 保持 numpy（逐
        head 的 K=64 小 matmul 12 个 dispatch 在 GPU 实测 139ms（T=499）
        而 numpy einsum 仅 11ms，GPU 必输——故中间三件套留 numpy）。
        ``RVC_HUBERT_ATTN=numpy`` 时强制 ``_attn_numpy``（纯 numpy）；
        权重未注册 / 其他 dtype / GPU 异常回退 numpy。数值：QKV 拼接与
        分别 matmul 逐位一致，中间件与 numpy 版逐位一致 → GPU 化只引入
        matmul 舍入差（~1e-6）。
        """
        if _ATTN_NUMPY_FORCED:
            return self._attn_numpy(h, layer, scale)
        qkv_pb = _weights.get(f"hubert.enc.{idx}.attn.qkv.weight")
        o_pb = _weights.get(f"hubert.enc.{idx}.attn.o.weight")
        if qkv_pb is not None and o_pb is not None and h.dtype == _F32:
            try:
                return self._attn_hybrid(h, layer, idx, scale)
            except RuntimeError:
                pass  # GPU 路径异常（引擎/DimensionsTooLarge）→ 回退 numpy
        return self._attn_numpy(h, layer, scale)

    def _attn_numpy(self, h: np.ndarray, layer: dict, scale: float) -> np.ndarray:
        """attention 纯 numpy 路径（P1-4：scores/加权和改用 np.matmul 批式
        BLAS —— ``np.einsum`` 对 ctx 形如 ``bhts,bhsd->bhtd`` 的收缩会选到
        极慢的 C 内层路径（T=499 实测 74-124ms/层），而 ``np.matmul`` 走
        batched gemm 仅 4ms/层，数值一致）。"""
        attn = layer["attn"]
        with self._pt("attn.qkv"):
            q = nn_ops.linear(h, *attn["q"]) * scale  # transformers 在 q 上缩放
            k = nn_ops.linear(h, *attn["k"])
            v = nn_ops.linear(h, *attn["v"])
        B, T, C = q.shape
        H_, D = 12, 64
        qh = q.reshape(B, T, H_, D).transpose(0, 2, 1, 3)  # [B,H,T,D]
        kh = k.reshape(B, T, H_, D).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, H_, D).transpose(0, 2, 1, 3)
        with self._pt("attn.scores"):
            scores = np.matmul(qh, np.ascontiguousarray(kh.swapaxes(-1, -2)))
        with self._pt("attn.softmax"):
            attn_w = _softmax_np(scores)
        with self._pt("attn.ctx"):
            ctx = np.matmul(attn_w, np.ascontiguousarray(vh))
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, C)
        with self._pt("attn.out"):
            return nn_ops.linear(ctx, *attn["o"])

    def _attn_hybrid(self, h: np.ndarray, layer: dict, idx: int,
                     scale: float) -> np.ndarray:
        """attention 混合路径（P1-4）：QKV/out_proj 常驻 GPU + numpy 中间件。

        数据（profile_hubert + 单算子拆测，AMD Radeon Pro VII）：
          - 引擎每个 dispatch 固定开销 ~6-13ms、下载 ~5ms/MB、上传 ~1.4ms/次；
          - scores 12 头 K=64 小 matmul（12 dispatch 一次 commit）T=499 实测
            139ms，numpy einsum 仅 11ms；softmax/加权和同理 —— GPU 必输；
          - QKV（[T,768]@[768,2304]）与 out_proj（[T,768]@[768,768]）是
            K=768 的单次大 matmul，常驻权重命中时 T=99 实测 12/5.6ms、
            T=499 26/12ms，与 numpy（15/15、17/6.5ms）同量级或更优。
        故：QKV/out_proj 走 GPU（一次 commit 各一），中间三件套（scores
        einsum / softmax / 加权和 einsum）保持 numpy —— 输出与纯 numpy 版
        只差两个 matmul 的舍入（~1e-6，<1e-4 对照要求内）。
        """
        from runtime.vulkan_ops import BatchRunner, get_context  # noqa: PLC0415

        attn = layer["attn"]
        B, T, C = h.shape
        H_, D = 12, 64
        x2 = np.ascontiguousarray(h.reshape(-1, C))  # [T, 768]

        # QKV 投影：x @ [Wq;Wk;Wv]ᵀ → [T, 2304]（一次 matmul，常驻 b）
        with self._pt("attn.qkv"):
            br = BatchRunner(get_context())
            try:
                qkv = br.matmul(
                    x2, attn["_qkv_T"],
                    buf_b=_weights.get(f"hubert.enc.{idx}.attn.qkv.weight"))
                br.commit()
                qkv = qkv.numpy().reshape(T, 3 * C)
            finally:
                br.release()
        # bias/缩放/头拆分（视图，无拷贝；与 numpy 版 (x@wᵀ+b)*scale 同序）
        with self._pt("attn.split"):
            qa = (qkv[:, :C] + attn["q"][1]) * scale
            ka = qkv[:, C:2 * C] + attn["k"][1]
            va = qkv[:, 2 * C:] + attn["v"][1]
            q = qa.reshape(B, T, H_, D).transpose(0, 2, 1, 3)  # [B,H,T,D]
            k = ka.reshape(B, T, H_, D).transpose(0, 2, 1, 3)
            v = va.reshape(B, T, H_, D).transpose(0, 2, 1, 3)
        with self._pt("attn.scores"):
            scores = np.matmul(q, np.ascontiguousarray(k.swapaxes(-1, -2)))
        with self._pt("attn.softmax"):
            attn_w = _softmax_np(scores)
        with self._pt("attn.ctx"):
            ctx = np.matmul(attn_w, np.ascontiguousarray(v))
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, C)

        # out_proj（一次 matmul，常驻 b）
        with self._pt("attn.out"):
            br = BatchRunner(get_context())
            try:
                o = br.matmul(np.ascontiguousarray(ctx.reshape(-1, C)), attn["_o_T"],
                              buf_b=_weights.get(f"hubert.enc.{idx}.attn.o.weight"))
                br.commit()
                attn_out = o.numpy().reshape(B, T, C) + attn["o"][1]
            finally:
                br.release()
        return attn_out

    def _ffn_opt(self, h: np.ndarray, layer: dict, idx: int) -> np.ndarray:
        """FFN：intermediate/output 线性层常驻 GPU matmul（各一次 commit），
        GELU 保持 numpy erf 精确版（float32，引擎无 GPU 算子）。

        数据（P1-4）：wi/wo 常驻（``_wi_T``/``_wo_T``，跳过每次 transpose
        + upload）后，GPU 全 T 优于 numpy BLAS：T=99 17ms vs 22+71ms、
        T=499 58ms vs 26+15ms（+gelu）、T=999 87ms vs 106ms（含 gelu 时
        差距更大，因 numpy wo 在 [T,3072] 大 K 缩减上效率极差）。bias 加
        法与 ``nn_ops.linear`` 同序（先 matmul 后加 bias），数值一致。
        """
        wi_pb = _weights.get(f"hubert.enc.{idx}.ff.wi.weight")
        wo_pb = _weights.get(f"hubert.enc.{idx}.ff.wo.weight")
        if wi_pb is not None and wo_pb is not None and h.dtype == _F32:
            from runtime.vulkan_ops import (  # noqa: PLC0415
                BatchRunner, get_context, _GRID_POINTS_MAX,
            )

            ff = layer["ff"]
            T = h.shape[1]
            if T * ff["wi"][0].shape[0] <= _GRID_POINTS_MAX \
                    and T * ff["wo"][0].shape[0] <= _GRID_POINTS_MAX:
                x2 = np.ascontiguousarray(h.reshape(-1, h.shape[-1]))  # [T, 768]
                with self._pt("ffn.wi"):
                    br = BatchRunner(get_context())
                    try:
                        it = br.matmul(x2, ff["_wi_T"], buf_b=wi_pb)  # [T, 3072]
                        br.commit()
                        inter = it.numpy()
                    finally:
                        br.release()
                    inter = inter + np.asarray(ff["wi"][1], dtype=_F32).reshape(1, -1)
                with self._pt("ffn.gelu"):
                    g = gelu_erf(inter)
                with self._pt("ffn.wo"):
                    br = BatchRunner(get_context())
                    try:
                        ot = br.matmul(np.ascontiguousarray(g), ff["_wo_T"],
                                       buf_b=wo_pb)  # [T, 768]
                        br.commit()
                        out = ot.numpy()
                    finally:
                        br.release()
                    out = out + np.asarray(ff["wo"][1], dtype=_F32).reshape(1, -1)
                return out.reshape(h.shape)
        with self._pt("ffn.wi"):
            wi_out = nn_ops.linear(h, *layer["ff"]["wi"])
        with self._pt("ffn.gelu"):
            g = gelu_erf(wi_out)
        with self._pt("ffn.wo"):
            out = nn_ops.linear(g, *layer["ff"]["wo"])
        return out


# ---------------------------------------------------------------------------
# 模块级单例缓存
# ---------------------------------------------------------------------------

_cache: Dict[str, HubertEncoder] = {}


def load_hubert_model(model_dir: str) -> HubertEncoder:
    """懒加载 + 缓存：同一 model_dir 只加载一次（webui 多次调用复用）。"""
    key = os.path.abspath(model_dir)
    if key not in _cache:
        _cache[key] = HubertEncoder(model_dir)
    return _cache[key]