# -*- coding: utf-8 -*-
"""纯 numpy 的 RVC VITS 变体合成器推理（SynthesizerTrnMs256NSFsid / Ms768NSFsid）。

对齐 ``infer/module/models.py``（SynthesizerTrn + GeneratorNSF + TextEncoder +
ResidualCouplingBlock）的推理语义，是 "RVC 去 CUDA 化移植" 的变声核心。

数据流（B=1，phone 长度 P）::

    g      = emb_g(sid)[0]                          -> [256] -> [1,256,1]
    phone  = linear(phone) + embedding(pitch)       -> [1,P,192] x sqrt(192) lrelu
    x_mask = 全 1 的 [1,1,P]（RVC 推理无 pad）
    x      = 6 层 T5 式相对位置注意力 Encoder       -> [1,192,P]
    m,logs = proj(x) * x_mask（split 通道）         -> [1,192,P] x2
    z_p    = (m + exp(logs) * randn * 0.66666) * x_mask
    z      = flow 逆变换（8 个 flow：RC+Flip 反向）  -> [1,192,P]
    o      = GeneratorNSF(z * x_mask, nsff0, g)     -> [1,1,480P]（tanh）

部分合成（T52 优化，skip_head/return_length 语义，对齐原版 infer）::

    head   = skip_head（帧），length = return_length（帧）
    flow_head = max(head - 24 - _DEC_PAD, 0)   # TextEncoder 全量（注意跨全窗口），
        m/logs 取 [flow_head:]           # flow 输入裁剪（原版只提前 24 帧预热
        flow 输出第 24 帧起与全量一致）  # _FLOW_PAD=24 即 flow WN 感受野
    zs = max(head - _DEC_PAD, 0)，ze = min(P, head + length + _DEC_PAD)
    只对 z[:, :, zs:ze] 做 GeneratorNSF（含左右 _DEC_PAD 边界上下文，保证
    dec 输出目标段与"全量合成后裁剪"逐位一致，见 _DEC_PAD 说明）
    SineGen 相位预推进：段内第 j 帧相位偏移 = 全量 fmod-cumsum 对应值
    （_phase_offsets），不需要先合成 head 之前的帧
    nsff0 / 噪声 ns 仅取 [zs:ze] / [480zs:480ze] 切片 -> [1,1,480*length]

所有权重以 ``dict[str, ndarray]`` 形式从 checkpoint 加载（键名与 RVC 一致），
weight_norm 层在加载时还原为普通权重。**本模块禁止 import torch**。
"""

from __future__ import annotations

import math
import os
import threading
from typing import Optional

import numpy as np

from .. import nn  # runtime.nn 算子库
from ..vulkan_weights import _weights  # P1：Vulkan 权重常驻管理器（numpy 后端 no-op）

# P1-3：dec 主循环 batch 化总开关。环境变量 ``RVC_VITS_DEC_BATCH=0`` 可显式
# 关闭（用于与逐次路径做 A/B 对比 / 性能基准）；默认开启，且仅在 vulkan
# 后端（dec 权重已注册常驻）时实际生效 —— numpy 后端自动走原逐次路径。
_DEC_BATCH_FLAG = os.environ.get("RVC_VITS_DEC_BATCH", "1") != "0"

# P1-9：推理随机项固定种子开关。``RVC_FIXED_SEED=<int>`` 时，未显式传
# ``seed`` 的 ``infer`` 调用使用该固定种子（z_p 噪声 + SineGen 噪声确定），
# 用于验证"同进程多次推理逐位一致"（输出池复用不引入非确定性）；未设置
# 时行为与原来完全一致（每次新建随机 RandomState）。
_FIXED_SEED = os.environ.get("RVC_FIXED_SEED", "").strip()

# P1-6：dec 整段 GPU 驻留总开关（在 _DEC_BATCH_FLAG 之上叠加）。
# ``RVC_VITS_DEC_RESIDENT=0`` 关闭（回退到逐级 batch 提交的 P1-3 路径）。
# 开启时把 conv_pre + cond + 4 级 ups/ResBlock/平均 + 尾部 lrelu/conv_post
# 录进**一个** BatchRunner、一次 commit：级间 BatchTensor 流转（免 12+ 次
# 中间下载），平均用 persistent(1/3) buffer + mul_inplace，仅最后下载一次。
_DEC_RESIDENT_FLAG = os.environ.get("RVC_VITS_DEC_RESIDENT", "1") != "0"

# P1-6：dec 整段 GPU 驻留的**异步提交**开关（在 _DEC_RESIDENT 之上）。
# ``RVC_VITS_DEC_ASYNC=0`` 关闭（同步 commit+等待）；默认开启：commit 用
# ``async_=True``（不等待 GPU），下载由 ``BatchTensor.numpy()`` 内部自动
# wait —— 单批次语义与同步完全一致，但为 pipeline 多块并行等场景演示了
# engine 侧 async 提交（MAX_INFLIGHT=4 帧轮转）的正确接入路径
# （提交后不等待 → 其他独立批次可先提交 → 统一 wait）。
_DEC_ASYNC_FLAG = os.environ.get("RVC_VITS_DEC_ASYNC", "1") != "0"

# P1-5：enc_p（TextEncoder）逐层批量提交总开关。``RVC_VITS_ENCP_BATCH=0``
# 关闭；默认开启，仅 vulkan 后端（enc_p 权重常驻）时生效。把每层的
# q/k/v 三个 conv1d 与 FFN 的 conv_1→relu→conv_2→残差各合入一次
# ``rvc_batch_commit``（每层 2 次提交，替代 5 次单算子调用 + 4 次中间
# 张量往返）；attention 中间件（相对位置 scores/softmax/ctx einsum）与
# LN 的 [1,P,h]↔[1,h,P] 转置保持 numpy（P 通常 ≤300，GPU 单算子必输）。
_ENC_P_BATCH_FLAG = os.environ.get("RVC_VITS_ENCP_BATCH", "1") != "0"

# D5：enc_p TextEncoder 注意力 GPU 化开关与最小 P 阈值。
# ``RVC_VITS_ENCP_ATTN_GPU=0`` 强制 numpy 注意力（对照/回归）；P 低于
# ``RVC_VITS_ENCP_ATTN_GPU_MIN_P``（默认 512）时 GPU 固定开销（上传/下载/
# commit 同步）不敌 numpy BLAS，走 numpy 分支（与 D5 前逐位一致）。
_ENC_ATTN_GPU_FLAG = os.environ.get("RVC_VITS_ENCP_ATTN_GPU", "1") != "0"
_ENC_ATTN_GPU_MIN_P = int(os.environ.get("RVC_VITS_ENCP_ATTN_GPU_MIN_P", "512"))

# T16：enc 全 GPU 流转（transpose kernel 打通 [1,h,P]↔[P,C]/[P,h] 布局，
# LN 用 op9 layer_norm）——6 层内零 BatchTensor.numpy() 下载。0 回退 D5 路径。
_ENC_FULLGPU_FLAG = os.environ.get("RVC_VITS_ENCP_FULLGPU", "1") != "0"

# 说话清晰度混合参数（2026-09-25 用户要求）：B5 修复=给 enc 注意力输出补上
# conv_o 投影（缺失正是口齿不清根因）。mix∈[0,1] 控制 conv_o 投影强度：
#   mix=1（默认）→ 完整 conv_o（B5 修复后，清晰，零回归）；
#   mix=0 → 完全复现旧版"缺 conv_o"的口齿不清（可复现诊断）；
#   中间值 → 线性混合过渡。实现：conv_o 是 (192,192,1) 逐点卷积，推理期
#   合成权重 W' = mix*W + (1-mix)*I、b' = mix*b（精确等价部分投影），
#   保持 T16 全 GPU 流转不破坏；env RVC_VITS_CLARITY_MIX 或 API 设置。
_ENC_CLARITY_MIX = float(os.environ.get("RVC_VITS_CLARITY_MIX", "1.0"))
_ENC_CLARITY_MIX = max(0.0, min(1.0, _ENC_CLARITY_MIX))


def set_clarity_mix(mix: float) -> None:
    """运行时设置说话清晰度混合（0~1）。供 WebUI/API 调用，默认 1.0。"""
    global _ENC_CLARITY_MIX
    _ENC_CLARITY_MIX = max(0.0, min(1.0, float(mix)))


def get_clarity_mix() -> float:
    """当前清晰度混合值（默认 1.0=完整 conv_o 修复）。"""
    return _ENC_CLARITY_MIX

# D6：flow（ResidualCouplingBlock 逆变换）GPU 化总开关。``RVC_VITS_FLOW_GPU=0``
# 关闭（回退 numpy 门控路径，零回归对照）；默认开启，仅 vulkan 后端（flow
# 权重常驻）且 x_mask 全 1（RVC 推理恒全 1）时生效。把每 RC 块的
# pre/cond/in/res_skip/post 全部 conv1d + 融合 gating（tanh*sigmoid，新
# op18）录进**一个** BatchRunner 一次 commit：层间 BatchTensor 流转
# （免 ~70 次独立 dispatch 同步）；m 仅下载一次做 x1 减法（host 小数组）。
# GPU tanh/sigmoid 与 libm ~1ulp 差 → 有据浮点差异（D5 先例，非逐位）。
_FLOW_GPU_FLAG = os.environ.get("RVC_VITS_FLOW_GPU", "1") != "0"

# P2：dec 多块批量的单组块数上限。多块批量把 N 块解耦成"每块独立录制、
# 同 runner 一次 commit"（无 padding 按块展开，同 hubert 批量模式）；受
# 引擎 recorder 单批 dispatch 上限（384，hubert _BATCH_MAX_BLOCKS 同源）
# 约束：dec 每块每级 = lrelu 1 + ups(conv_t+noise conv+add) 3 + ResBlock
# 3×(copy+lrelu+conv1+lrelu+conv2+add) 54 + 平均(add×2+mul) 3 = 61 dispatch，
# 61×6=366 ≤ 384（留 18 余量）。块数超限按组循环（组间 BatchTensor 流转）。
_DEC_BATCH_MAX_BLOCKS = 6

# 48k 模型的固定上采样总倍数（hop=100 帧 -> 480 采样点/帧）
_UPP_48K = 480

# 部分合成（skip_head/return_length）的边界上下文帧数。
#
# 数值一致性要求：skip 版 dec 只合成目标段，但 conv_pre(k=7,p=3) / 转置卷积 /
# ResBlock / noise_convs 都有有限感受野，目标段波形若要与"全量合成后裁剪"逐位
# 一致，dec 输入 z 段必须额外包含左右各 _DEC_PAD 帧（帧域）。该值必须 ≥ dec 帧域
# 总半窗（conv_pre 3 帧 + 转置卷积链 + ResBlock 采样域感受野折算帧域，估算 ~9 帧），
# 取 16 留足余量；右端自动截断到 P（与全量版在序列末尾的补零行为一致）。
_DEC_PAD = 16
# flow（ResidualCouplingBlock 逆变换）的 WN 感受野：4 层 RC × 每层 WN(k=5,d=1,
# n_layers=3) 半窗 6 帧 = 24 帧。flow 输入从 max(head-24,0) 起算时，输出从第
# 24 帧（相对）起与全量逐位一致；再提前 _DEC_PAD 帧以覆盖 dec 的左上下文需求。
_FLOW_PAD = 24

__all__ = ["SynthesizerTrn", "load_synthesizer", "VitsConfig"]


def _load_ckpt(path):
    """按需导入 torch_compat（项目根需在 sys.path）。"""
    import torch_compat  # noqa: PLC0415
    return torch_compat.load_pth(path)


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
class VitsConfig:
    """VITS 结构配置：优先从 checkpoint 的 ``config`` 数组解析，否则从权重形状推断。

    字段与 RVC 的 SynthesizerTrnMs256NSFsid 构造参数一一对应：
    spec_channels / segment_size 仅为记录（推理不用）；推理真正用到的是
    phone_dim、inter/hidden/filter、n_heads、n_layers、upsample_*、n_spk、
    gin_channels、sr 与 upp。

    config 数组布局（RVC 推理 checkpoint）：``[spec, segment, inter, hidden,
    filter, n_heads, n_layers, kernel, p_dropout, resblock, resblock_kernels,
    resblock_dilations, upsample_rates, upsample_initial, upsample_kernels,
    n_spk(spk_embed_dim), gin, sr]``。
    """

    def __init__(self, w: dict, config=None, sr: int | None = None):
        self.w = w
        cfg = list(config) if config is not None else None
        # ---- TextEncoder（一律由权重形状确定，最可靠）----
        self.phone_dim = int(w["enc_p.emb_phone.weight"].shape[1])  # 256(v1)/768(v2)
        self.hidden = int(w["enc_p.emb_phone.weight"].shape[0])     # 192
        self.filter = int(w["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[0])  # 768
        n_layers = 0
        while f"enc_p.encoder.attn_layers.{n_layers}.conv_q.weight" in w:
            n_layers += 1
        self.n_layers = n_layers                                # 6
        # k_channels 由 emb_rel_k 的最后一维给出（96），n_heads = hidden // k_channels
        kc = w["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[2]
        self.k_channels = int(kc)
        self.n_heads = self.hidden // self.k_channels           # 2
        self.spec_channels = int(cfg[0]) if cfg else 1025
        self.segment_size = int(cfg[1]) if cfg else 32
        # ---- GeneratorNSF ----
        self.upsample_initial = int(w["dec.conv_pre.weight"].shape[0])  # 512
        if cfg and len(cfg) > 12:
            # 完整 config 数组优先（训练 dict 无 weight_v 键，无法形状推断）
            self.upsample_rates = [int(r) for r in cfg[12]]
            self.upsample_initial = int(cfg[13])
            self.sr = int(sr) if sr else (int(cfg[17]) if len(cfg) > 17 else 48000)
            if len(cfg) > 14 and cfg[14] is not None and cfg[14] != "":
                self.upsample_kernels = [int(k) for k in cfg[14]]
            else:
                kernels = []
                i = 0
                while f"dec.ups.{i}.weight_v" in w:
                    kernels.append(int(w[f"dec.ups.{i}.weight_v"].shape[2]))
                    i += 1
                self.upsample_kernels = kernels
        else:
            self.upsample_rates = self._infer_rates(w)
            # 无 config：48k 为默认（40k/32k 纯权重无法区分，需显式 sr 覆盖）
            self.sr = int(sr) if sr else 48000
            if sr:
                # 显式 sr 覆盖：重推第一级上采样率，使总上采样倍数 = sr//100
                # （无 config 反推按 48k(480) 打底，40k/32k 模型会被推出
                # 480/400=1.2×/1.5× 偏大的 upp → 输出时长/音调错乱；
                # 后级 rates 由 noise_convs 结构决定，仅第一级可调。
                # 48k=480/40k=400/32k=320 均整除，RVC 标准结构保证。）
                rest = math.prod(self.upsample_rates[1:])
                self.upsample_rates[0] = int(self.sr // 100) // rest
            self.upsample_kernels = []
            i = 0
            while f"dec.ups.{i}.weight_v" in w:
                self.upsample_kernels.append(
                    int(w[f"dec.ups.{i}.weight_v"].shape[2]))
                i += 1
        self.n_ups = len(self.upsample_rates)
        if self.n_ups == 0:
            i = 0
            while f"dec.ups.{i}.weight_v" in w:
                self.n_ups += 1
                i += 1
        # 总上采样倍数 = prod(rates)：48k=480 / 40k=400 / 32k=320
        self.upp = int(math.prod(self.upsample_rates))
        # noise_strides：noise_convs 的 stride（= 后级 rates 之积；最后一层 1）
        self.noise_strides = []
        for i in range(self.n_ups):
            k = int(w[f"dec.noise_convs.{i}.weight"].shape[2])
            self.noise_strides.append(k // 2 if i < self.n_ups - 1 else 1)
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.resblock = "1"
        # ---- 说话人 ----
        self.n_spk = int(w["emb_g.weight"].shape[0])
        self.gin_channels = int(w["emb_g.weight"].shape[1])  # 256

    @staticmethod
    def _infer_rates(w: dict):
        """无 config 时从 noise_convs 的 kernel 反推 upsample_rates。

        stride_f0[i] = prod(rates[i+1:])，且 noise_convs[i].kernel = stride*2
        （最后一层 kernel=1 -> stride=1）；第一级 rate = upp / stride_f0[0]，
        其中 upp 对 48k 模型固定为 480。
        """
        n_ups = 0
        while f"dec.ups.{n_ups}.weight_v" in w:
            n_ups += 1
        strides = []
        for i in range(n_ups):
            k = int(w[f"dec.noise_convs.{i}.weight"].shape[2])
            strides.append(k // 2 if i < n_ups - 1 else 1)
        rates = [strides[i] // strides[i + 1] for i in range(n_ups - 1)]
        rates.insert(0, _UPP_48K // strides[0])
        return rates

    def __repr__(self):
        return (
            f"VitsConfig(phone={self.phone_dim}, hidden={self.hidden}, "
            f"filter={self.filter}, heads={self.n_heads}, layers={self.n_layers}, "
            f"n_ups={self.n_ups}, rates={self.upsample_rates}, "
            f"kernels={self.upsample_kernels}, n_spk={self.n_spk}, "
            f"gin={self.gin_channels}, sr={self.sr})"
        )


# ---------------------------------------------------------------------------
# weight_norm 还原
# ---------------------------------------------------------------------------
def _conv1d_opt(x, w, b, key: str, stride: int = 1, padding=0, dilation: int = 1):
    """dec 卷积入口（P1 BufferPool 常驻权重优化）。

    若 ``_weights`` 已注册 ``key``（含 ``key + ".bias"``，当 ``b`` 非 None）且
    输入/权重为 float32 **或 float16**（P1-3：RVC 权重多为 half；half→f32 为
    无损转换，GPU shader 内 bias 加法）：走 ``vulkan_ops.conv1d`` 并复用常驻
    buffer（跳过每次 upload）。这使 dec 逐次路径与 batch 化路径走同一 GPU
    kernel（位级一致）；numpy 后端（权重未注册）仍走 ``nn.conv1d`` 原路径。
    """
    pb_w = _weights.get(key)
    # bias 常驻 key 是 "<层名>.bias"（注册时与 weight 分开登记，见
    # register_persistent_weights）；key 以 ".weight" 结尾时去掉该后缀再拼。
    bias_key = (key[:-7] if key.endswith(".weight") else key) + ".bias"
    pb_b = _weights.get(bias_key) if b is not None else None
    if pb_w is not None and (b is None or pb_b is not None):
        xa = np.asarray(x)
        wa = np.asarray(w)
        if xa.dtype in (np.float32, np.float16) and wa.dtype in (np.float32, np.float16):
            from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

            return vulkan_ops.conv1d(
                xa, wa, b, stride, padding, dilation, buf_w=pb_w, buf_b=pb_b
            )
    return nn.conv1d(x, w, b, stride, padding, dilation)


def _conv_t1d_opt(
    x, w, b, key: str, stride: int = 1, padding: int = 0,
    output_padding: int = 0, dilation: int = 1,
):
    """dec 转置卷积入口（P1-2：dec.ups 常驻权重 + GPU）。

    若 ``_weights`` 已注册 ``key``（含 ``key + ".bias"``，当 ``b`` 非 None）且
    输入/权重为 float32 **或 float16**（P1-3）：走 ``vulkan_ops.conv_transpose1d``
    并复用常驻 buffer。否则**完全等价于**原 ``nn.conv_transpose1d`` 路径 ——
    numpy 后端 / 权重未注册时默认行为不变。
    """
    pb_w = _weights.get(key)
    bias_key = (key[:-7] if key.endswith(".weight") else key) + ".bias"
    pb_b = _weights.get(bias_key) if b is not None else None
    if pb_w is not None and (b is None or pb_b is not None):
        xa = np.asarray(x)
        wa = np.asarray(w)
        if xa.dtype in (np.float32, np.float16) and wa.dtype in (np.float32, np.float16):
            from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

            return vulkan_ops.conv_transpose1d(
                xa, wa, b, stride, padding, output_padding, dilation,
                buf_w=pb_w, buf_b=pb_b,
            )
    return nn.conv_transpose1d(x, w, b, stride, padding, output_padding, dilation)


def _conv1d_batch_or_seg(br, x, w, b, key: str,
                         stride: int = 1, padding=0, dilation: int = 1):
    """T1.1：BatchRunner 录制 conv1d，输出超 GRID 时 GPU 内分段（conv1d_seg_multi）。

    dec 系超限级专用（x 保持 BatchTensor 流转，零 numpy 往返）：输出
    ``[1, C_out, oL]`` 超限时按输出列自动分段写父 buffer；不超限走原
    ``br.conv1d``（逐位同路径，零回归）。padding 仅支持 int（dec 场景全单值）。
    """
    from runtime import vulkan_ops as _vo  # noqa: PLC0415  # 惰性避免包初始化环

    C_out, _, K = w.shape
    pad = int(padding)
    oL = (x.shape[2] + pad + pad - dilation * (K - 1) - 1) // stride + 1
    bias_key = (key[:-7] if key.endswith(".weight") else key) + ".bias"
    pb_b = _weights.get(bias_key) if b is not None else None
    if 1 * C_out * oL > _vo._GRID_POINTS_MAX:
        return br.conv1d_seg_multi(
            x, w, b, stride=stride, padding=pad, dilation=dilation,
            out_shape=(1, C_out, oL),
            buf_w=_weights.get(key), buf_b=pb_b,
        )
    return br.conv1d(
        x, w, b, stride=stride, padding=pad, dilation=dilation,
        buf_w=_weights.get(key), buf_b=pb_b,
    )


def _conv_t1d_batch_or_seg(br, x, w, b, key: str, stride: int, padding: int,
                           out_c: int):
    """T1.1：BatchRunner 录制 conv_t1d（dec.ups），输出超 GRID 时 GPU 内分段。

    ``out_c`` 为输出通道（weight 的 C_out 与 ups 各级一致）；不超限走原
    ``br.conv_transpose1d``（零回归）。output_padding=0/dilation=1（dec ups）。
    """
    from runtime import vulkan_ops as _vo  # noqa: PLC0415  # 惰性避免包初始化环

    C_in, C_out, K = w.shape
    oL = (x.shape[2] - 1) * stride - 2 * padding + (K - 1) + 1
    bias_key = (key[:-7] if key.endswith(".weight") else key) + ".bias"
    pb_b = _weights.get(bias_key) if b is not None else None
    if 1 * out_c * oL > _vo._GRID_POINTS_MAX:
        return br.conv_t1d_seg_multi(
            x, w, b, stride=stride, padding=padding,
            out_shape=(1, C_out, oL),
            buf_w=_weights.get(key), buf_b=pb_b,
        )
    return br.conv_transpose1d(
        x, w, b, stride=stride, padding=padding,
        buf_w=_weights.get(key), buf_b=pb_b,
    )


def _linear_opt(x, w, b, key: str):
    """线性层入口（perf(P1): enc_p/flow 权重常驻 GPU 化）。

    与 ``_conv1d_opt`` 同理：``_weights`` 已注册 ``key``（含 ``key + ".bias"``，
    当 ``b`` 非 None）且输入/权重为 float32/float16 时，走
    ``runtime.vulkan_ops.matmul`` 并复用常驻 buffer（``x @ w.T``，bias 由
    调用方按 ``nn.linear`` 同序后加）；否则完全等价于 ``nn.linear``——
    numpy 后端 / 权重未注册时默认行为逐位不变。
    """
    pb_w = _weights.get(key)
    bias_key = (key[:-7] if key.endswith(".weight") else key) + ".bias"
    pb_b = _weights.get(bias_key) if b is not None else None
    if pb_w is not None and (b is None or pb_b is not None):
        xa = np.asarray(x)
        wa = np.asarray(w)
        if xa.dtype in (np.float32, np.float16) and wa.dtype in (np.float32, np.float16):
            from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

            D = xa.shape[-1]
            x2 = xa.reshape(-1, D)
            out = vulkan_ops.matmul(x2, wa.T, buf_b=pb_w)
            out = out.reshape(xa.shape[:-1] + (wa.shape[0],))
            if b is not None:
                out = out + np.asarray(b, dtype=out.dtype).reshape(1, -1)
            return out
    return nn.linear(x, w, b)


def _deweight_norm(w_v: np.ndarray, w_g: np.ndarray) -> np.ndarray:
    """还原 ``torch.nn.utils.weight_norm`` 参数化为普通权重。

    PyTorch 语义：``W = w_v * (w_g / ||w_v||_2)``，其中范数沿 weight_norm 的
    ``dim`` 维。RVC 权重（本仓库 checkpoint）的 ``weight_g`` 形状为
    ``[D0, 1, 1]``（每个 D0 通道一个标量），与 torch 默认 ``dim=0`` 的
    ``weight_g [1, ...]`` 形状不同；本函数按 checkpoint 实际形状自适应：
    ``||w_v||`` 沿除第 0 维外的所有维（即逐第 0 维通道），``w_g`` 逐通道标量广播。

    注：torch 标准 dim=0 语义是 norm 沿 axis=0、g 沿 [1,in,k] 广播；两种解释
    在 ``w_g`` 为逐通道标量时仅当权重形状特殊才等价，已用 torch 数值对照验证
    （与上游 RVC 语义一致）。
    """
    w_v = np.asarray(w_v)
    w_g = np.asarray(w_g)
    dtype = w_v.dtype if w_v.dtype in (np.float32, np.float64) else np.float32
    w_v = w_v.astype(dtype, copy=False)
    w_g = w_g.astype(dtype, copy=False)
    if w_g.ndim == 1:
        w_g = w_g.reshape(-1, 1, 1)
    norm = np.linalg.norm(w_v.reshape(w_v.shape[0], -1), ord=2, axis=1)
    # 注意：np.linalg.norm 恒返回 float64（strong 标量），必须转回 dtype，
    # 否则还原权重变 float64 -> conv1d/conv_t1d 的 f32 检查失败 -> 全部回退
    # numpy，GPU 算子永远不命中（P1-2 修复的关键点）。
    norm = np.maximum(norm, 1e-12).astype(dtype, copy=False)
    return (w_v * (w_g.reshape(-1, 1, 1) / norm.reshape(-1, 1, 1))).astype(
        dtype, copy=False)


# ---------------------------------------------------------------------------
# 工具函数（相对位置注意力）
# ---------------------------------------------------------------------------
def _get_relative_embeddings(emb_rel: np.ndarray, length: int, window_size: int):
    """对齐 attentions.MultiHeadAttention._get_relative_embeddings。

    emb_rel: ``[1, 2*window_size+1, k]``；返回 ``[1, 2*length-1, k]``：
    length > window_size+1 时两端各补 (length-window_size-1) 行零后全取；
    否则从中间窗口切片。
    """
    pad_length = max(length - (window_size + 1), 0)
    slice_start = max((window_size + 1) - length, 0)
    slice_end = slice_start + 2 * length - 1
    if pad_length > 0:
        emb = np.pad(emb_rel, ((0, 0), (pad_length, pad_length), (0, 0)))
    else:
        emb = emb_rel
    return emb[:, slice_start:slice_end, :]


def _relative_to_absolute(x):
    """对齐 _relative_position_to_absolute_position：``[b,h,l,2l-1] -> [b,h,l,l]``。"""
    b, h, length, _ = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)))  # 列尾 pad 1
    xf = xp.reshape(b, h, length * (2 * length))
    xf = np.pad(xf, ((0, 0), (0, 0), (0, length - 1)))  # 尾接 length-1 个零
    xf = xf.reshape(b, h, length + 1, 2 * length - 1)
    return xf[:, :, :length, length - 1:]


def _absolute_to_relative(x):
    """对齐 _absolute_position_to_relative_position：``[b,h,l,l] -> [b,h,l,2l-1]``。"""
    b, h, length, _ = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, length - 1)))
    xf = xp.reshape(b, h, length * length + length * (length - 1))
    xf = np.pad(xf, ((0, 0), (0, 0), (length, 0)))  # 头部 pad length 个零
    xf = xf.reshape(b, h, length, 2 * length)
    return xf[:, :, :, 1:]


# ---------------------------------------------------------------------------
# 各子模块（全部为纯函数式推理，权重从 checkpoint dict 取）
# ---------------------------------------------------------------------------
class _TextEncoder:
    """TextEncoder：emb_phone + emb_pitch -> 6 层 T5 相对注意力 Encoder -> proj。"""

    def __init__(self, w: dict, cfg: VitsConfig):
        self.w = w
        self.cfg = cfg
        self.hidden = cfg.hidden
        self.n_layers = cfg.n_layers
        self.kc = cfg.k_channels
        self.heads = cfg.n_heads

    def embed(self, phone, pitch, phone_pb=None):
        """phone: [1,P,D] f32；pitch: [1,P] i64 -> [1,P,hidden]（未乘 sqrt）。

        phone_pb: 可选 GPU 常驻 buffer（``PersistentBuffer``，形状 ``[P, D]``，
        即 ``phone[0]``）。非 None 时第一层 ``x = phone @ emb_phone.T`` 整个
        在 GPU 上完成（``BatchRunner.matmul(buf_a=phone_pb, buf_b=常驻)``），
        **不再上传 feats**——pipeline 已把 hubert 特征以 GPU buffer 形式传入
        （perf(P1) 中间张量显存驻留）。numpy 后端 / 未传时走原路径逐位不变。
        """
        if phone_pb is not None:
            from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

            pb = _weights.get("vits.enc_p.emb_phone.weight")
            if pb is not None and phone_pb.valid:
                ctx = vulkan_ops.get_context()
                br = vulkan_ops.BatchRunner(ctx)
                try:
                    x = br.matmul(
                        phone_pb,
                        self.w["enc_p.emb_phone.weight"],
                        buf_a=phone_pb,
                        buf_b=pb,
                    )  # [P, 192]（emb_phone [192,D] -> w.T [D,192]）
                    br.commit()
                    x = x.numpy().reshape(1, -1, self.hidden)
                finally:
                    br.release()
                x = x + np.asarray(self.w["enc_p.emb_phone.bias"],
                                   dtype=np.float32).reshape(1, 1, -1)
            else:
                x = _linear_opt(phone, self.w["enc_p.emb_phone.weight"],
                                self.w["enc_p.emb_phone.bias"],
                                "vits.enc_p.emb_phone.weight")
        else:
            x = _linear_opt(phone, self.w["enc_p.emb_phone.weight"],
                            self.w["enc_p.emb_phone.bias"],
                            "vits.enc_p.emb_phone.weight")
        if pitch is not None:
            x = x + nn.embedding(pitch, self.w["enc_p.emb_pitch.weight"])
        return x

    def _can_batch(self) -> bool:
        """enc_p 逐层批量（P1-5）可用性：开关 + vulkan 后端 + enc_p 权重常驻。"""
        return (
            _ENC_P_BATCH_FLAG
            and _weights.enabled
            and _weights.get("vits.enc_p.encoder.attn_layers.0.conv_q.weight") is not None
            and _weights.get("vits.enc_p.encoder.norm_layers_1.0.gamma") is not None
        )

    @staticmethod
    def _ln_np(a, gamma, beta, eps: float = 1e-5):
        """纯 numpy LayerNorm（与 nn.layer_norm 的 numpy 分支同公式）。

        批量路径专用：避免 backend 把 [1,P,h] 分派到 GPU 单算子（P 小时
        固定开销 ~10ms 远超 numpy 的 <1ms）。
        """
        mean = a.mean(-1, keepdims=True)
        var = a.var(-1, keepdims=True)
        return (a - mean) / np.sqrt(var + eps) * gamma + beta

    # ── D5：enc TextEncoder 注意力 GPU 化 ─────────────────────────────
    # 6 层 T5 相对位置注意力的 P² 级 [H,P,P]/[H,P,2P-1] matmul 是 D1 后
    # dec CPU 侧最大头（9.88s/60s 的 ~59%）。引擎 attn_qk/attn_sv kernel
    # （head-interleaved [T,C] 布局）经 D5 扩共享容量至 D≤128 后兼容
    # enc_p（hidden=192, heads=2, kc=96, D=96）。GPU 化边界：
    #   - attn_qk：scores 主项 q·kᵀ（GPU）＋ 相对位置 logits（GPU matmul）
    #   - softmax / rel↔abs 两个对角搬移（np.pad+reshape）：numpy（引擎无算子）
    #   - attn_sv：ctx 主项 p_attn·v（GPU）＋ rel_v 项（GPU matmul）
    # conv_o / LN / FFN 保持原路径。数值差异来源＝GPU matmul 累加序 vs
    # OpenBLAS（scores ~1e-6 级 → 波形 ~1e-5 级，验收放宽为有据差异）。
    def _attn_numpy_layer(self, q, k, v, i: int, P: int, attn_mask):
        """第 i 层自注意力纯 numpy（与 D5 前 _encode_batch 逐位一致）。

        q/k/v: [1,h,P]（conv_q/k/v 输出）；返回 attn_out [1,h,P]（无 conv_o）。
        P 小于 GPU 阈值 / 开关关闭 / GPU 异常时的回退分支（对照用）。
        """
        w = self.w
        heads, kc = self.heads, self.kc
        qh = q.reshape(1, heads, kc, P).transpose(0, 1, 3, 2)
        kh = k.reshape(1, heads, kc, P).transpose(0, 1, 3, 2)
        vh = v.reshape(1, heads, kc, P).transpose(0, 1, 3, 2)
        qs = qh / math.sqrt(kc)
        scores = np.matmul(qs, np.ascontiguousarray(kh.swapaxes(-1, -2)))
        emb_k = w[f"enc_p.encoder.attn_layers.{i}.emb_rel_k"]
        used_k = _get_relative_embeddings(emb_k, P, window_size=10)
        rel_logits = np.matmul(qs, np.ascontiguousarray(used_k[0].T))
        scores = scores + _relative_to_absolute(rel_logits)
        if attn_mask is not None:  # D1c-1：mask 全 1 时跳过（恒等）
            scores = np.where(attn_mask != 0, scores, -1e4)
        m_ = scores.max(-1, keepdims=True)
        e_ = np.exp(scores - m_)
        p_attn = e_ / e_.sum(-1, keepdims=True)  # 纯 numpy softmax
        out = np.matmul(p_attn, vh)
        emb_v = w[f"enc_p.encoder.attn_layers.{i}.emb_rel_v"]
        used_v = _get_relative_embeddings(emb_v, P, window_size=10)
        rel_w = _absolute_to_relative(p_attn)
        # rel_w [1,h,P,2P-1] @ used_v[0] [2P-1,kc]（m 维配对，无需转置）
        out = out + np.matmul(rel_w, np.ascontiguousarray(used_v[0]))
        return out.transpose(0, 1, 3, 2).reshape(1, self.hidden, P)

    def _attn_gpu_layer(self, q, k, v, i: int, P: int, attn_mask):
        """第 i 层自注意力 GPU 化（D5 终版 + D5b：GPU softmax 直喂）。

        相对位置融合进 kernel。主路径（``attn_mask is None``，即 RVC 推理
        x_mask 恒全 1，D1c-1 守卫）为**单 runner 三算子链**：``banded_attn_qk``
        （scores = q·(k + used_k[s-t+P-1])）→ GPU softmax（op8，对最后一维
        P 逐行归一，语义= numpy softmax(axis=-1)）→ ``banded_attn_sv``
        （ctx = p_attn·(v + used_v[s-t+P-1])）——p_attn 全程留 GPU 直喂
        banded_sv，消除每层 [H,P,P] scores 下载 + p_attn 上传两次 ~156MB
        往返（D5 §8.4 搬运 5268/4768MB 的 P² 固有项）。

        数值差异来源＝GPU matmul 累加序（同 D5，rel ~1e-6 → 波形 ~1e-5
        级，有据差异）＋ GPU softmax exp/求和序 vs numpy（p_attn rel
        ~1e-7 级，见 D5b 对照）。mask 非 None（padding 等）回退 D5 原
        路径：runner1 banded_qk → numpy softmax（其中含 mask where）→
        runner2 banded_sv（零回归）。
        """
        from runtime import vulkan_ops  # noqa: PLC0415

        w = self.w
        heads, kc = self.heads, self.kc
        if heads * kc != self.hidden:  # 防御：kc 非整除时保持 numpy
            return self._attn_numpy_layer(q, k, v, i, P, attn_mask)
        C = heads * kc
        scale = 1.0 / math.sqrt(kc)
        ctx = vulkan_ops.get_context()
        # head-interleaved [P, C]（行 t，列 h*kc+d —— attn kernel 期望
        # 布局）：q [1,h,P] -> [1,H,kc,P] -> [1,P,H,kc] -> [P,C]（t 主序）
        qs_i = q.reshape(1, heads, kc, P).transpose(0, 3, 1, 2).reshape(P, C) * scale
        k_i = k.reshape(1, heads, kc, P).transpose(0, 3, 1, 2).reshape(P, C)
        v_i = v.reshape(1, heads, kc, P).transpose(0, 3, 1, 2).reshape(P, C)
        emb_k = w[f"enc_p.encoder.attn_layers.{i}.emb_rel_k"]
        emb_v = w[f"enc_p.encoder.attn_layers.{i}.emb_rel_v"]
        used_k = _get_relative_embeddings(emb_k, P, window_size=10)[0]  # [2P-1,kc]
        used_v = _get_relative_embeddings(emb_v, P, window_size=10)[0]
        try:
            if attn_mask is None:
                # D5b 主路径：单 runner 三算子链，p_attn 留 GPU 直喂。
                # softmax 对 [H,P,P] 用 axis_len=P（reshape [H*P, P] 逐行
                # 归一，每行一个 (h,t) 的 s 向量 == numpy softmax(axis=-1)）。
                br = vulkan_ops.BatchRunner(ctx)
                try:
                    sq = br.banded_attn_qk(qs_i, k_i, used_k, heads, P, kc, C)  # [H,P,P]
                    pa = br.softmax(sq, axis_len=P)  # GPU softmax，[H,P,P]
                    br.tensor_done(sq)  # scores 已被 softmax 消费
                    cw = br.banded_attn_sv(pa, v_i, used_v, heads, P, kc, C)  # [P,C]
                    br.tensor_done(pa)  # p_attn 已被 banded_sv 消费
                    br.commit()
                    out = cw.numpy().reshape(P, heads, kc).transpose(1, 0, 2)[None]
                finally:
                    br.release()
                return out.transpose(0, 1, 3, 2).reshape(1, self.hidden, P)
            # mask 非 None：D5 原路径（numpy softmax 含 mask where）
            br = vulkan_ops.BatchRunner(ctx)
            try:
                sq = br.banded_attn_qk(qs_i, k_i, used_k, heads, P, kc, C)  # [H,P,P]
                br.commit()
                scores = sq.numpy().reshape(1, heads, P, P)
            finally:
                br.release()
            scores = np.where(attn_mask != 0, scores, -1e4)
            m_ = scores.max(-1, keepdims=True)
            e_ = np.exp(scores - m_)
            p_attn = e_ / e_.sum(-1, keepdims=True)  # 纯 numpy softmax
            br2 = vulkan_ops.BatchRunner(ctx)
            try:
                cw = br2.banded_attn_sv(p_attn[0], v_i, used_v, heads, P, kc, C)  # [P,C]
                br2.commit()
                out = cw.numpy().reshape(P, heads, kc).transpose(1, 0, 2)[None]  # [1,H,P,kc]
            finally:
                br2.release()
            return out.transpose(0, 1, 3, 2).reshape(1, self.hidden, P)
        except (RuntimeError, ValueError):
            # GPU 异常（引擎/超限）→ 回退 numpy 分支（数值与 D5 前一致）
            return self._attn_numpy_layer(q, k, v, i, P, attn_mask)

    def _encode_batch_fullgpu(self, x, x_mask):
        """T16：enc 全 GPU 流转——6 层内**零** ``BatchTensor.numpy()`` 下载。

        单 ``BatchRunner`` 录制全部 6 层（qkv 卷积 / transpose / banded
        注意力 / LN / FFN），一次 ``commit``、仅最后下载一次。布局打通：
        conv_q/k/v 输出 ``[1,h,P]`` → **transpose kernel（op19）** →
        ``[P,C]`` head-interleaved（q 折叠 ``1/sqrt(kc)`` 缩放）直喂
        ``banded_attn_qk``；``banded_attn_sv`` 输出 ``[P,C]`` →
        transpose 回 ``[1,h,P]`` → conv_o → 残差 add → LN（op9
        layer_norm，``[1,h,P]``↔``[P,h]`` 双向 transpose）→ FFN → LN2，
        x 全程 BatchTensor GPU 流转。mask 全 1（调用方已保证，D1c-1
        同款守卫）⇒ ``x*x_mask ≡ x``（IEEE x*1.0 逐位不变），省略 mask mul。

        数值差异来源（有据，~1e-6 级）：op9 layer_norm 用单遍
        ``E[x²]-E[x]²`` 累加序 vs numpy ``_ln_np`` 双遍 ``(x-mean)²``
        求和序 → LN 输出相对差 ~1e-6（D5 GPU matmul/softmax 同类先例）。
        其余算子（transpose/copy/conv1d/add/banded_qk/sv/softmax）与
        改前同 kernel 同 push → 逐位一致。

        异常（引擎/超限）由调用方捕获回退原 ``_encode_batch`` 路径。
        """
        from runtime import vulkan_ops  # noqa: PLC0415

        w = self.w
        h = self.hidden
        kc = self.kc
        heads = self.heads
        # 调用方已做前导：x = x*x_mask -> transpose(0,2,1)，此处 x 为 [1,h,P]。
        P = x.shape[-1]
        scale = 1.0 / math.sqrt(kc)
        eps = 1e-5
        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            # 入口：x [1,h,P] numpy -> copy 上传（布局已是 [1,h,P]，无需转置）
            xbt = br.copy(x)
            for i in range(self.n_layers):
                p = f"enc_p.encoder.attn_layers.{i}"
                # ── q/k/v 卷积 + transpose（一次提交）──
                tq = br.conv1d(
                    xbt, w[f"{p}.conv_q.weight"], w[f"{p}.conv_q.bias"],
                    buf_w=_weights.get(f"vits.{p}.conv_q.weight"),
                    buf_b=_weights.get(f"vits.{p}.conv_q.bias"),
                )
                tk = br.conv1d(
                    xbt, w[f"{p}.conv_k.weight"], w[f"{p}.conv_k.bias"],
                    buf_w=_weights.get(f"vits.{p}.conv_k.weight"),
                    buf_b=_weights.get(f"vits.{p}.conv_k.bias"),
                )
                tv = br.conv1d(
                    xbt, w[f"{p}.conv_v.weight"], w[f"{p}.conv_v.bias"],
                    buf_w=_weights.get(f"vits.{p}.conv_v.weight"),
                    buf_b=_weights.get(f"vits.{p}.conv_v.bias"),
                )
                qs = br.transpose(tq, h, P, scale=scale, out_shape=(P, h))
                k_i = br.transpose(tk, h, P, out_shape=(P, h))
                v_i = br.transpose(tv, h, P, out_shape=(P, h))
                br.tensor_done(tq)
                br.tensor_done(tk)
                br.tensor_done(tv)
                # ── banded 注意力（D5b 同款三算子链，p_attn 留 GPU）──
                used_k = _get_relative_embeddings(w[f"{p}.emb_rel_k"], P,
                                                  window_size=10)[0]
                used_v = _get_relative_embeddings(w[f"{p}.emb_rel_v"], P,
                                                  window_size=10)[0]
                sq = br.banded_attn_qk(qs, k_i, used_k, heads, P, kc, h)
                br.tensor_done(qs)
                br.tensor_done(k_i)
                pa = br.softmax(sq, axis_len=P)
                br.tensor_done(sq)
                cw = br.banded_attn_sv(pa, v_i, used_v, heads, P, kc, h)
                br.tensor_done(pa)
                br.tensor_done(v_i)
                # ── conv_o（转回 [1,h,P]）→ 残差 ──
                attnT = br.transpose(cw, P, h, out_shape=(1, h, P))
                br.tensor_done(cw)
                mix = _ENC_CLARITY_MIX
                if mix >= 1.0:
                    # 默认：完整 conv_o 投影（B5 修复，清晰，零回归——常驻权重）
                    ao = br.conv1d(
                        attnT, w[f"{p}.conv_o.weight"], w[f"{p}.conv_o.bias"],
                        buf_w=_weights.get(f"vits.{p}.conv_o.weight"),
                        buf_b=_weights.get(f"vits.{p}.conv_o.bias"),
                    )
                else:
                    # 说话清晰度混合（mix<1）：conv_o 是 (192,192,1) 逐点卷积，
                    # 合成部分投影权重 W' = mix*W + (1-mix)*I、b' = mix*b——
                    # mix=0 精确等价"缺 conv_o"（复现旧版口齿不清），中间值
                    # 线性过渡。数值：mix*conv_o(attnT) + (1-mix)*attnT。
                    Wo = np.asarray(w[f"{p}.conv_o.weight"], dtype=np.float32)
                    bo = np.asarray(w[f"{p}.conv_o.bias"], dtype=np.float32)
                    eye = np.eye(h, dtype=np.float32)[:, :, None]  # [h,h,1]
                    Wm = (mix * Wo + (1.0 - mix) * eye).astype(np.float32)
                    bm = (mix * bo).astype(np.float32)
                    ao = br.conv1d(attnT, Wm, bm)
                br.tensor_done(attnT)
                br.add_inplace(ao, xbt)  # 残差 x + attn_out
                br.tensor_done(xbt)
                # ── LN1（GPU）：[1,h,P] -> [P,h] -> layer_norm -> 回 ──
                ln1 = br.transpose(ao, h, P, out_shape=(P, h))
                br.tensor_done(ao)
                ln1 = br.layer_norm(
                    ln1, w[f"enc_p.encoder.norm_layers_1.{i}.gamma"],
                    w[f"enc_p.encoder.norm_layers_1.{i}.beta"], eps=eps,
                    buf_gamma=_weights.get(
                        f"vits.enc_p.encoder.norm_layers_1.{i}.gamma"),
                    buf_beta=_weights.get(
                        f"vits.enc_p.encoder.norm_layers_1.{i}.beta"),
                )
                xbt = br.transpose(ln1, P, h, out_shape=(1, h, P))
                br.tensor_done(ln1)
                # ── FFN：conv_1 → relu → conv_2 → 残差 ──
                fp = f"enc_p.encoder.ffn_layers.{i}"
                t = br.conv1d(
                    xbt, w[f"{fp}.conv_1.weight"], w[f"{fp}.conv_1.bias"],
                    padding=1,
                    buf_w=_weights.get(f"vits.{fp}.conv_1.weight"),
                    buf_b=_weights.get(f"vits.{fp}.conv_1.bias"),
                )
                t = br.relu(t)
                t = br.conv1d(
                    t, w[f"{fp}.conv_2.weight"], w[f"{fp}.conv_2.bias"],
                    padding=1,
                    buf_w=_weights.get(f"vits.{fp}.conv_2.weight"),
                    buf_b=_weights.get(f"vits.{fp}.conv_2.bias"),
                )
                br.add_inplace(t, xbt)
                br.tensor_done(xbt)
                # ── LN2（GPU）──
                ln2 = br.transpose(t, h, P, out_shape=(P, h))
                br.tensor_done(t)
                ln2 = br.layer_norm(
                    ln2, w[f"enc_p.encoder.norm_layers_2.{i}.gamma"],
                    w[f"enc_p.encoder.norm_layers_2.{i}.beta"], eps=eps,
                    buf_gamma=_weights.get(
                        f"vits.enc_p.encoder.norm_layers_2.{i}.gamma"),
                    buf_beta=_weights.get(
                        f"vits.enc_p.encoder.norm_layers_2.{i}.beta"),
                )
                xbt = br.transpose(ln2, P, h, out_shape=(1, h, P))
                br.tensor_done(ln2)
            # 尾：x [1,h,P] -> transpose -> [1,P,h] 一次下载
            out = br.transpose(xbt, h, P, out_shape=(1, P, h))
            br.tensor_done(xbt)
            br.commit()
            return out.numpy()
        finally:
            br.release()

    def _encode_batch(self, x, x_mask):
        """P1-5 逐层批量提交：每层 q/k/v 三个 conv1d 与 FFN（conv_1→relu→
        conv_2→残差 add_inplace）各合入一次 ``rvc_batch_commit``。

        与 ``encode`` 数值一致：batch conv1d/relu/add 与逐次调用同一
        kernel（位级一致），attention 中间件（相对位置 scores / softmax /
        ctx einsum）与 LN 的 [1,P,h]↔[1,h,P] 转置保持 numpy。P 超 GPU
        grid 上限时抛 ``RuntimeError("vulkan_batch_too_large: ...")``
        由 ``encode`` 捕获回退逐次路径。
        """
        from runtime import vulkan_ops  # noqa: PLC0415

        w = self.w
        h = self.hidden
        kc = self.kc
        heads = self.heads
        # D1c-1：RVC 推理 x_mask 恒为全 1（generate_mask）→ attn_mask 全非零、
        # softmax 前的 ``np.where(mask!=0, scores, -1e4)`` 恒为恒等（不替换任何
        # 元素）——全 1 时跳过该 [H,P,P] 级广播扫描（6 层 × 数千万元素，60s 档
        # 实测 ~0.3-0.5s）。跳过结果与保留逐位一致（mask 全非零 ⇒ where 逐元素
        # 取 scores 原值）。非全 1（padding 推理等）走原路径零回归。
        mask_all_one = bool(np.all(x_mask != 0.0))
        attn_mask = None if mask_all_one else (x_mask[..., None] * x_mask[..., None, :])
        x = x * x_mask.transpose(0, 2, 1)  # [1,P,h] * [1,P,1]
        x = x.transpose(0, 2, 1)  # [1,h,P]
        P = x.shape[-1]
        ctx = vulkan_ops.get_context()
        # T16：全 GPU 流转（transpose kernel 打通布局 + GPU LN），6 层内零
        # numpy() 下载。前置：开关 + mask 全 1（softmax 无 mask 语义）+ P 达
        # attn GPU 阈值 + heads*kc==hidden（banded kernel 前置）。异常
        # （引擎/超限）回退原 D5 路径（x 已为 [1,h,P]，循环直接消费，零回归）。
        if (
            _ENC_FULLGPU_FLAG
            and mask_all_one
            and P >= _ENC_ATTN_GPU_MIN_P
            and heads * kc == h
        ):
            try:
                return self._encode_batch_fullgpu(x, x_mask)
            except (RuntimeError, ValueError):
                pass
        for i in range(self.n_layers):
            # ── 自注意力：conv_q/k/v 一次 batch 提交 ──
            br = vulkan_ops.BatchRunner(ctx)
            try:
                tq = br.conv1d(
                    x, w[f"enc_p.encoder.attn_layers.{i}.conv_q.weight"],
                    w[f"enc_p.encoder.attn_layers.{i}.conv_q.bias"],
                    buf_w=_weights.get(
                        f"vits.enc_p.encoder.attn_layers.{i}.conv_q.weight"),
                    buf_b=_weights.get(
                        f"vits.enc_p.encoder.attn_layers.{i}.conv_q.bias"),
                )
                tk = br.conv1d(
                    x, w[f"enc_p.encoder.attn_layers.{i}.conv_k.weight"],
                    w[f"enc_p.encoder.attn_layers.{i}.conv_k.bias"],
                    buf_w=_weights.get(
                        f"vits.enc_p.encoder.attn_layers.{i}.conv_k.weight"),
                    buf_b=_weights.get(
                        f"vits.enc_p.encoder.attn_layers.{i}.conv_k.bias"),
                )
                tv = br.conv1d(
                    x, w[f"enc_p.encoder.attn_layers.{i}.conv_v.weight"],
                    w[f"enc_p.encoder.attn_layers.{i}.conv_v.bias"],
                    buf_w=_weights.get(
                        f"vits.enc_p.encoder.attn_layers.{i}.conv_v.weight"),
                    buf_b=_weights.get(
                        f"vits.enc_p.encoder.attn_layers.{i}.conv_v.bias"),
                )
                br.commit()
                q, k, v = tq.numpy(), tk.numpy(), tv.numpy()
            finally:
                br.release()

            # ── attention 中间件（D5：P≥阈值走 GPU attn_qk/attn_sv +
            # rel 项 matmul，softmax/对角搬移 numpy；P<阈值或开关关闭走
            # numpy 分支——与 D5 前逐位一致）──
            if _ENC_ATTN_GPU_FLAG and P >= _ENC_ATTN_GPU_MIN_P:
                attn_out = self._attn_gpu_layer(q, k, v, i, P, attn_mask)
            else:
                attn_out = self._attn_numpy_layer(q, k, v, i, P, attn_mask)
            # B5 修复（2026-09-23）：批量路径缺失 conv_o 输出投影（与下方
            # numpy 分支 679-684 行及官方 MultiHeadAttention.forward 一致）。
            # 此前 6 层累积缺失 → enc_p.encoder 输出 h 偏差 1.33（maxdiff，
            # 相对 ~17x）→ 传播到 dec 输出 → 咬字不清。补上后 h vs 官方
            # torch = 1.7e-6（逐位一致，子代理闭环验证）。
            attn_out = _conv1d_opt(
                attn_out,
                w[f"enc_p.encoder.attn_layers.{i}.conv_o.weight"],
                w[f"enc_p.encoder.attn_layers.{i}.conv_o.bias"],
                f"vits.enc_p.encoder.attn_layers.{i}.conv_o.weight",
            )

            # ── LN1（numpy，含 [1,P,h]↔[1,h,P] 转置）──
            x = self._ln_np(
                (x + attn_out).transpose(0, 2, 1),
                w[f"enc_p.encoder.norm_layers_1.{i}.gamma"],
                w[f"enc_p.encoder.norm_layers_1.{i}.beta"],
            ).transpose(0, 2, 1)

            # ── FFN：conv_1 → relu → conv_2 → 残差 add_inplace 一次 batch ──
            br2 = vulkan_ops.BatchRunner(ctx)
            try:
                t = br2.conv1d(
                    x, w[f"enc_p.encoder.ffn_layers.{i}.conv_1.weight"],
                    w[f"enc_p.encoder.ffn_layers.{i}.conv_1.bias"],
                    padding=1,
                    buf_w=_weights.get(
                        f"vits.enc_p.encoder.ffn_layers.{i}.conv_1.weight"),
                    buf_b=_weights.get(
                        f"vits.enc_p.encoder.ffn_layers.{i}.conv_1.bias"),
                )
                t = br2.relu(t)
                t = br2.conv1d(
                    t, w[f"enc_p.encoder.ffn_layers.{i}.conv_2.weight"],
                    w[f"enc_p.encoder.ffn_layers.{i}.conv_2.bias"],
                    padding=1,
                    buf_w=_weights.get(
                        f"vits.enc_p.encoder.ffn_layers.{i}.conv_2.weight"),
                    buf_b=_weights.get(
                        f"vits.enc_p.encoder.ffn_layers.{i}.conv_2.bias"),
                )
                br2.add_inplace(t, x)  # 残差 y = conv_2(...) + x
                br2.commit()
                x = t.numpy()  # [1,h,P]
            finally:
                br2.release()

            # ── LN2（numpy）──
            x = self._ln_np(
                x.transpose(0, 2, 1),
                w[f"enc_p.encoder.norm_layers_2.{i}.gamma"],
                w[f"enc_p.encoder.norm_layers_2.{i}.beta"],
            ).transpose(0, 2, 1)
        x = x * x_mask
        return x.transpose(0, 2, 1)  # [1,P,h]

    def encode(self, x, x_mask):
        """x: [1,P,hidden]；x_mask: [1,1,P] -> [1,P,hidden] 编码结果。

        P1-5：vulkan 后端 + 权重常驻时优先走 ``_encode_batch``（逐层批量
        提交）；超限/异常回退下方原逐次路径（数值一致）。
        """
        if self._can_batch():
            try:
                return self._encode_batch(x, x_mask)
            except (RuntimeError, ValueError):
                pass
        h = self.hidden
        w = self.w
        # D1c-1：mask 全 1 → 跳过 [H,P,P] 级 where（与 _encode_batch 同款守卫）。
        mask_all_one = bool(np.all(x_mask != 0.0))
        attn_mask = None if mask_all_one else (x_mask[..., None] * x_mask[..., None, :])
        x = x * x_mask.transpose(0, 2, 1)  # [1,P,h] * [1,P,1]
        # 切到 [1,h,P]
        x = x.transpose(0, 2, 1)
        for i in range(self.n_layers):
            # ---- 自注意力（相对位置）----
            q = _conv1d_opt(x, w[f"enc_p.encoder.attn_layers.{i}.conv_q.weight"],
                            w[f"enc_p.encoder.attn_layers.{i}.conv_q.bias"],
                            f"vits.enc_p.encoder.attn_layers.{i}.conv_q.weight")
            k = _conv1d_opt(x, w[f"enc_p.encoder.attn_layers.{i}.conv_k.weight"],
                            w[f"enc_p.encoder.attn_layers.{i}.conv_k.bias"],
                            f"vits.enc_p.encoder.attn_layers.{i}.conv_k.weight")
            v = _conv1d_opt(x, w[f"enc_p.encoder.attn_layers.{i}.conv_v.weight"],
                            w[f"enc_p.encoder.attn_layers.{i}.conv_v.bias"],
                            f"vits.enc_p.encoder.attn_layers.{i}.conv_v.weight")
            P = x.shape[-1]
            # [1,h,P] -> [1,heads,kc,P] -> [1,heads,P,kc]
            qh = q.reshape(1, self.heads, self.kc, P).transpose(0, 1, 3, 2)
            kh = k.reshape(1, self.heads, self.kc, P).transpose(0, 1, 3, 2)
            vh = v.reshape(1, self.heads, self.kc, P).transpose(0, 1, 3, 2)

            qs = qh / math.sqrt(self.kc)
            # T2.1：einsum → np.matmul（BLAS bmm，同 _encode_batch）
            scores = np.matmul(qs, np.ascontiguousarray(kh.swapaxes(-1, -2)))  # [1,h,P,P]

            emb_k = w[f"enc_p.encoder.attn_layers.{i}.emb_rel_k"]  # [1,21,kc]
            used_k = _get_relative_embeddings(emb_k, P, window_size=10)  # [1,2P-1,kc]
            rel_logits = np.matmul(qs, np.ascontiguousarray(used_k[0].T))
            scores = scores + _relative_to_absolute(rel_logits)
            if attn_mask is not None:  # D1c-1：mask 全 1 时跳过（恒等）
                scores = np.where(attn_mask != 0, scores, -1e4)
            p_attn = nn.softmax(scores, axis=-1)  # [1,h,P,P]

            out = np.matmul(p_attn, vh)
            emb_v = w[f"enc_p.encoder.attn_layers.{i}.emb_rel_v"]
            used_v = _get_relative_embeddings(emb_v, P, window_size=10)
            rel_w = _absolute_to_relative(p_attn)  # [1,h,P,2P-1]
            out = out + np.matmul(rel_w, np.ascontiguousarray(used_v[0]))
            # [1,h,P,kc] -> [1,h,P]
            attn_out = out.transpose(0, 1, 3, 2).reshape(1, h, P)
            attn_out = _conv1d_opt(
                attn_out,
                w[f"enc_p.encoder.attn_layers.{i}.conv_o.weight"],
                w[f"enc_p.encoder.attn_layers.{i}.conv_o.bias"],
                f"vits.enc_p.encoder.attn_layers.{i}.conv_o.weight",
            )
            x = nn.layer_norm(
                (x + attn_out).transpose(0, 2, 1),
                w[f"enc_p.encoder.norm_layers_1.{i}.gamma"],
                w[f"enc_p.encoder.norm_layers_1.{i}.beta"],
            ).transpose(0, 2, 1)

            # ---- FFN（kernel=3，_same_padding: pad_l=1, pad_r=1）----
            y = _conv1d_opt(
                x, w[f"enc_p.encoder.ffn_layers.{i}.conv_1.weight"],
                w[f"enc_p.encoder.ffn_layers.{i}.conv_1.bias"],
                f"vits.enc_p.encoder.ffn_layers.{i}.conv_1.weight",
                padding=1,
            )
            y = nn.relu(y)
            y = _conv1d_opt(
                y, w[f"enc_p.encoder.ffn_layers.{i}.conv_2.weight"],
                w[f"enc_p.encoder.ffn_layers.{i}.conv_2.bias"],
                f"vits.enc_p.encoder.ffn_layers.{i}.conv_2.weight",
                padding=1,
            )
            x = nn.layer_norm(
                (x + y).transpose(0, 2, 1),
                w[f"enc_p.encoder.norm_layers_2.{i}.gamma"],
                w[f"enc_p.encoder.norm_layers_2.{i}.beta"],
            ).transpose(0, 2, 1)
        x = x * x_mask
        return x.transpose(0, 2, 1)  # [1,P,h]

    def stats(self, x):
        """proj + split：x [1,P,h] -> (m, logs) [1,inter,P] x2（inter=hidden=192）。"""
        w = self.w
        x = x.transpose(0, 2, 1)  # [1,h,P]
        stats = _conv1d_opt(x, w["enc_p.proj.weight"], w["enc_p.proj.bias"],
                            "vits.enc_p.proj.weight")
        out = self.cfg.hidden  # out_channels = inter_channels = hidden_channels
        m, logs = stats[:, :out, :], stats[:, out:, :]
        return m, logs


class _FlowBlock:
    """ResidualCouplingBlock 逆变换（reverse=True）。

    flows 按逆序执行：RC(0),Flip,RC(2),Flip,RC(4),Flip,RC(6),Flip 反向
    即：Flip,RC(6),Flip,RC(4),Flip,RC(2),Flip,RC(0)。
    """

    def __init__(self, w: dict, cfg: VitsConfig):
        self.w = w
        self.cfg = cfg
        self.rc_ids = [0, 2, 4, 6]
        self._wn_cache: dict = {}  # deweight_norm 还原缓存（vulkan 后端注册时填充）

    @staticmethod
    def _dwn(w, idx, name):
        """deweight_norm 带缓存：``_wn_cache[base + '.weight']``。"""
        c = w._wn_cache
        key = f"flow.flows.{idx}.{name}"
        wn = c.get(key + ".weight")
        if wn is None:
            wn = _deweight_norm(w.w[key + ".weight_v"], w.w[key + ".weight_g"])
            c[key + ".weight"] = wn
        return wn

    def _rc_reverse(self, z, x_mask, g, idx):
        w = self.w
        half = self.cfg.hidden // 2  # 96
        x0, x1 = z[:, :half, :], z[:, half:, :]
        h = _conv1d_opt(x0, w[f"flow.flows.{idx}.pre.weight"],
                        w[f"flow.flows.{idx}.pre.bias"],
                        f"vits.flow.flows.{idx}.pre.weight") * x_mask
        h = self._wn(h, x_mask, g, idx)
        m = _conv1d_opt(h, w[f"flow.flows.{idx}.post.weight"],
                        w[f"flow.flows.{idx}.post.bias"],
                        f"vits.flow.flows.{idx}.post.weight") * x_mask
        x1 = (x1 - m) * x_mask
        return np.concatenate([x0, x1], axis=1)

    def _wn(self, x, x_mask, g, idx):
        """WN 门控卷积网络（res_skip 逐层累加，numpy 回退路径，D6 保留）。

        注意：ResidualCouplingBlock 的 dilation_rate=1，in_layers 膨胀系数为
        1**i = 1（每层 padding=(5*1-1)/2=2），不是 1,3,5。
        """
        w = self.w
        n_layers = 3
        hidden = self.cfg.hidden
        if g is not None:
            g = _conv1d_opt(
                g,
                self._dwn(self, idx, "enc.cond_layer"),
                w[f"flow.flows.{idx}.enc.cond_layer.bias"],
                f"vits.flow.flows.{idx}.enc.cond_layer.weight",
            )
        output = np.zeros_like(x)
        for i in range(n_layers):
            x_in = _conv1d_opt(
                x,
                self._dwn(self, idx, f"enc.in_layers.{i}"),
                w[f"flow.flows.{idx}.enc.in_layers.{i}.bias"],
                f"vits.flow.flows.{idx}.enc.in_layers.{i}.weight",
                dilation=1,
                padding=2,
            )
            if g is not None:
                off = i * 2 * hidden
                g_l = g[:, off:off + 2 * hidden, :]
            else:
                g_l = np.zeros_like(x_in)
            acts = nn.tanh(x_in + g_l)[:, :hidden, :] * nn.sigmoid(
                (x_in + g_l)[:, hidden:, :])
            res_skip = _conv1d_opt(
                acts,
                self._dwn(self, idx, f"enc.res_skip_layers.{i}"),
                w[f"flow.flows.{idx}.enc.res_skip_layers.{i}.bias"],
                f"vits.flow.flows.{idx}.enc.res_skip_layers.{i}.weight",
            )
            if i < n_layers - 1:
                res = res_skip[:, :hidden, :]
                x = (x + res) * x_mask
                output = output + res_skip[:, hidden:, :]
            else:
                output = output + res_skip
        return output * x_mask

    # ------------------------------------------------------------ D6
    # flow GPU 批量（RVC_VITS_FLOW_GPU=1 默认开）：每个 RC 块的
    # pre/cond/in/res_skip/post 全部 conv1d + 融合 gating（op18）录进
    # **一个** BatchRunner 一次 commit，层间 BatchTensor 流转；仅 m 下载
    # 一次（host 做 x1 减法与 concat，96 通道小数组）。x_mask 全 1 时
    # 跳过全部 * x_mask（D1c-1 已证 RVC 推理恒全 1）。
    # ---------------------------------------------------------------
    @staticmethod
    def _flow_gpu_ok(x_mask, g) -> bool:
        if not _FLOW_GPU_FLAG or not _weights.enabled:
            return False
        if g is None:  # flow 无 cond 路径 GPU 化未覆盖（gan1 恒有 g）→ 回退 numpy
            return False
        if _weights.get("vits.flow.flows.0.pre.weight") is None:
            return False
        m = np.asarray(x_mask)
        return bool(np.all(m != 0))  # 全 1 才走 GPU（否则回退 numpy 零回归）

    def _rc_reverse_gpu(self, z, x_mask, g, idx):
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        w = self.w
        half = self.cfg.hidden // 2  # 96
        hidden = self.cfg.hidden     # 192
        L = z.shape[2]
        x0 = z[:, :half, :]
        x1 = z[:, half:, :]
        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            # pre conv1d（常驻权重；pre/post 为普通权重，非 weight_norm）
            h = br.conv1d(
                x0,
                w[f"flow.flows.{idx}.pre.weight"],
                w[f"flow.flows.{idx}.pre.bias"],
                buf_w=_weights.get(f"vits.flow.flows.{idx}.pre.weight"),
                buf_b=_weights.get(f"vits.flow.flows.{idx}.pre.bias"),
            )
            h = self._wn_gpu(br, h, x_mask, g, idx)
            m = br.conv1d(
                h,
                w[f"flow.flows.{idx}.post.weight"],
                w[f"flow.flows.{idx}.post.bias"],
                buf_w=_weights.get(f"vits.flow.flows.{idx}.post.weight"),
                buf_b=_weights.get(f"vits.flow.flows.{idx}.post.bias"),
            )
            br.commit()
            m_np = m.numpy()  # [1, 96, L]
        finally:
            br.release()
        x1_new = (x1 - m_np)  # x_mask 全 1 → 跳过 * x_mask
        return np.concatenate([x0, x1_new], axis=1)

    def _wn_gpu(self, br, x_bt, x_mask, g, idx):
        """WN 3 层 GPU 批量版（D6）：返回 output BatchTensor（x_mask 全 1）。

        x_bt: [1, hidden, L] BatchTensor。gating kernel（op18）一次 dispatch
        完成 tanh(x_in+g_l)[:, :h] * sigmoid(x_in+g_l)[:, h:]；res_skip 拆半
        （res/skip）用 copy_off 偏移复制 + add_inplace 累加。
        """
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        w = self.w
        hidden = self.cfg.hidden  # 192
        L = x_bt.shape[2]
        n_layers = 3
        # cond g 段（[1, 3*2H, 1]；g 为 [1, gin, 1] → conv1d K=1）
        g_bt = None
        if g is not None:
            g_bt = br.conv1d(
                g,
                self._dwn(self, idx, "enc.cond_layer"),
                w[f"flow.flows.{idx}.enc.cond_layer.bias"],
                buf_w=_weights.get(f"vits.flow.flows.{idx}.enc.cond_layer.weight"),
                buf_b=_weights.get(f"vits.flow.flows.{idx}.enc.cond_layer.bias"),
            )
        # output 零 buffer（[1, hidden, L]）
        out_bt = br.copy(np.zeros((1, hidden, L), dtype=np.float32))
        for i in range(n_layers):
            x_in = br.conv1d(
                x_bt,
                self._dwn(self, idx, f"enc.in_layers.{i}"),
                w[f"flow.flows.{idx}.enc.in_layers.{i}.bias"],
                padding=2,
                buf_w=_weights.get(f"vits.flow.flows.{idx}.enc.in_layers.{i}.weight"),
                buf_b=_weights.get(f"vits.flow.flows.{idx}.enc.in_layers.{i}.bias"),
            )  # [1, 2*hidden, L]
            if g_bt is None:
                # 无 cond：_flow_gpu_ok 已拒绝（gan1 恒有 g）——防御性报错
                raise RuntimeError("_wn_gpu 需要 g 非 None（GPU 路径未覆盖无 cond flow）")
            acts = br.gating(x_in, g_bt, lg=1, h=hidden, off=i * 2 * hidden)
            rs = br.conv1d(
                acts,
                self._dwn(self, idx, f"enc.res_skip_layers.{i}"),
                w[f"flow.flows.{idx}.enc.res_skip_layers.{i}.bias"],
                buf_w=_weights.get(f"vits.flow.flows.{idx}.enc.res_skip_layers.{i}.weight"),
                buf_b=_weights.get(f"vits.flow.flows.{idx}.enc.res_skip_layers.{i}.bias"),
            )  # [1, 2*hidden, L]（i<2）或 [1, hidden, L]（i=2）
            if i < n_layers - 1:
                n_h = hidden * L
                res = br.copy_off(rs, n_h, 0, out_shape=(1, hidden, L))     # res_skip[:, :hidden]
                skip = br.copy_off(rs, n_h, n_h, out_shape=(1, hidden, L))   # res_skip[:, hidden:]
                x_bt = br.add_inplace(x_bt, res)        # x = (x + res)（x_mask 全 1）
                out_bt = br.add_inplace(out_bt, skip)   # output += res_skip[:, hidden:]
            else:
                out_bt = br.add_inplace(out_bt, rs)     # output += res_skip
        return out_bt  # * x_mask 全 1 → 跳过

    def reverse(self, z, x_mask, g):
        """逆序执行 8 个 flow：Flip,RC(6),Flip,RC(4),Flip,RC(2),Flip,RC(0)。

        D6：x_mask 全 1 且 flow 权重常驻时走 GPU 批量路径（_rc_reverse_gpu），
        否则回退原 numpy 逐次路径（零回归）。
        """
        gpu = self._flow_gpu_ok(x_mask, g)
        for idx in reversed(self.rc_ids):  # 6,4,2,0
            z = np.flip(z, axis=1)  # Flip（flows 奇索引）
            if gpu:
                z = self._rc_reverse_gpu(z, x_mask, g, idx)
            else:
                z = self._rc_reverse(z, x_mask, g, idx)
        return z


class _ResBlock1:
    """ResBlock1：3 组 (convs1 膨胀卷积, convs2) 残差，无 x_mask（Generator 调用）。

    注意：convs1 与 convs2 共用同一个 kernel_size（= resblock_kernel_sizes[j]
    ∈ [3,7,11]）；convs1 的 dilation 为 [1,3,5]，convs2 的 dilation=1。
    """

    def __init__(self, w: dict, base: str, channels: int, kernel_size: int):
        self.w = w
        self.base = base  # 如 dec.resblocks.0
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self._wn: dict = {}  # deweight_norm 缓存（常驻模式），由 GeneratorNSF 注入

    def forward(self, x):
        w = self.w
        k = self.kernel_size
        dilations = [1, 3, 5]  # resblock_dilation_sizes
        for j in range(3):
            # convs1（dilation=[1,3,5]，padding=get_padding(k,d)）
            d = dilations[j]
            c1 = nn.leaky_relu(x, 0.1)
            base = f"{self.base}.convs1.{j}"
            wn = self._wn.get(base + ".weight")
            if wn is None:
                wn = _deweight_norm(w[base + ".weight_v"], w[base + ".weight_g"])
            c1 = _conv1d_opt(
                c1, wn, w[base + ".bias"], base + ".weight",
                dilation=d, padding=(k * d - d) // 2,
            )
            c1 = nn.leaky_relu(c1, 0.1)
            base2 = f"{self.base}.convs2.{j}"
            wn2 = self._wn.get(base2 + ".weight")
            if wn2 is None:
                wn2 = _deweight_norm(w[base2 + ".weight_v"], w[base2 + ".weight_g"])
            c2 = _conv1d_opt(
                c1, wn2, w[base2 + ".bias"], base2 + ".weight",
                padding=(k - 1) // 2,
            )
            x = c2 + x
        return x


class _DecCollector:
    """T3.2：dec 异步提交的**延迟收集句柄**（块间 CPU/GPU 流水线用）。

    ``_dec_resident_forward(defer_download=True)`` 在 commit(async) 后不等待、
    不下载，把 runner/结果 tensor 的所有权转移给本对象并立即返回；调用方
    在 GPU 执行期间做其它 CPU 工作（如下一块的检索/插值准备），随后调用
    ``collect()`` 统一：wait fence → 下载 → tanh → release runner。

    线程语义：与 BatchRunner 一致，**本对象由创建线程独占**（pipeline 软件
    流水为单线程顺序调用，不引入新线程）；持有引擎批次生命周期锁直到
    ``collect()``/``abort()``，期间其它线程的 GPU 提交会被锁串行化（安全）。
    """

    __slots__ = ("_dec", "_br", "_x", "_done")

    def __init__(self, dec, br, x):
        self._dec = dec
        self._br = br
        self._x = x
        self._done = False

    def collect(self):
        """等待 fence + 下载 + tanh，释放 runner。返回 ``[1,1,480P]`` f32 波形。

        幂等：重复调用抛 ``RuntimeError``（结果已取走，buffer 已释放）。
        """
        if self._done:
            raise RuntimeError("_DecCollector 已 collect/abort")
        self._done = True
        br, x = self._br, self._x
        self._br = self._x = None
        try:
            return nn.tanh(x.numpy())  # numpy() 内 wait()（rvc_batch_wait）
        finally:
            try:
                br.release()  # async_pending 在途时 release 内部会先 wait
            except Exception:  # noqa: BLE001  # 释放失败不掩盖结果
                pass

    def abort(self):
        """放弃结果（异常路径兜底）：等待在途提交完成并释放 runner/锁。"""
        if self._done:
            return
        self._done = True
        br = self._br
        self._br = self._x = None
        try:
            br.release()  # release 对 async_pending 自动 wait，防 buffer 提前释放
        except Exception:  # noqa: BLE001
            pass


class _ImmediateResult:
    """无 async 路径（numpy 后端 / dec 非驻留 / 超限回退）的立即结果包装。

    统一 ``collect()`` 接口：pipeline 侧只认 ``hasattr(x, "collect")``。
    """

    __slots__ = ("_arr", "_done")

    def __init__(self, arr):
        self._arr = arr
        self._done = False

    def collect(self):
        if self._done:
            raise RuntimeError("_ImmediateResult 已 collect")
        self._done = True
        return self._arr

    def abort(self):
        self._done = True
        self._arr = None


class _GeneratorNSF:
    """GeneratorNSF：SineGen 激励 + 4 级转置卷积 + ResBlock1 + conv_post。"""

    def __init__(self, w: dict, cfg: VitsConfig):
        self.w = w
        self.cfg = cfg
        self.n_ups = cfg.n_ups
        self.upsample_initial = cfg.upsample_initial
        self.num_kernels = 3
        self._wn: dict = {}  # deweight_norm 缓存（常驻模式；numpy 后端为空）
        self.resblocks = []
        for i in range(self.n_ups):
            ch = cfg.upsample_initial // (2 ** (i + 1))
            for j in range(self.num_kernels):
                rb = _ResBlock1(
                    w, f"dec.resblocks.{i * 3 + j}", ch,
                    cfg.resblock_kernel_sizes[j])
                rb._wn = self._wn
                self.resblocks.append(rb)
        self.register_persistent_weights()
        # P1-3：dec 主循环 batch 化开关（仅 vulkan 后端 + dec 权重已注册常驻时
        # 启用；numpy 后端下 _weights.enabled=False，自动走原逐次路径）。
        self._dec_batch = bool(
            _DEC_BATCH_FLAG
            and _weights.enabled
            and _weights.get("dec.ups.0.weight") is not None
        )
        # P1-6：dec 整段 GPU 驻留（单 runner 单 commit）——在 batch 化之上。
        self._dec_resident = bool(
            _DEC_RESIDENT_FLAG
            and self._dec_batch
        )
        # P1-6：驻留路径是否用 async 提交（BatchTensor.numpy() 自动 wait）。
        self._dec_async = bool(_DEC_ASYNC_FLAG and self._dec_resident)
        self._thirds_cache: dict = {}  # (ch, L) -> PersistentBuffer(全 1/num_kernels)
        self._thirds_lock = threading.Lock()

    def register_persistent_weights(self) -> None:
        """把 dec 全部卷积权重常驻 GPU（P1 BufferPool；P1-2 补 dec.ups；幂等可重入）。

        覆盖 conv_pre / cond / noise_convs / resblocks(convs1+convs2) /
        conv_post / ups(4 级 ConvTranspose1d) 的 weight+bias：weight_norm 层先
        还原为普通权重并缓存（推理时不再重复 deweight_norm），再一次性上传常驻。
        numpy 后端（GPUWeights 禁用）下为 no-op，且不缓存（保持与原来完全一致）。
        """
        w = self.w
        if not _weights.enabled:
            self._wn = {}
            return
        pairs = [
            ("dec.conv_pre.weight", w["dec.conv_pre.weight"]),
            ("dec.conv_pre.bias", w["dec.conv_pre.bias"]),
            ("dec.cond.weight", w["dec.cond.weight"]),
            ("dec.cond.bias", w["dec.cond.bias"]),
            ("dec.conv_post.weight", w["dec.conv_post.weight"]),
        ]
        for i in range(self.n_ups):
            pairs.append((f"dec.noise_convs.{i}.weight",
                          w[f"dec.noise_convs.{i}.weight"]))
            pairs.append((f"dec.noise_convs.{i}.bias",
                          w[f"dec.noise_convs.{i}.bias"]))
            # P1-2：dec.ups.{i}（ConvTranspose1d weight_norm）—— 先还原再常驻
            up_base = f"dec.ups.{i}"
            wn = _deweight_norm(w[up_base + ".weight_v"], w[up_base + ".weight_g"])
            self._wn[up_base + ".weight"] = wn
            pairs.append((up_base + ".weight", wn))
            pairs.append((up_base + ".bias", w[up_base + ".bias"]))
        for rb in self.resblocks:
            for j in range(3):
                for layer in ("convs1", "convs2"):
                    base = f"{rb.base}.{layer}.{j}"
                    wn = _deweight_norm(w[base + ".weight_v"],
                                        w[base + ".weight_g"])
                    self._wn[base + ".weight"] = wn
                    pairs.append((base + ".weight", wn))
                    pairs.append((base + ".bias", w[base + ".bias"]))
        for key, arr in pairs:
            _weights.register(key, arr)

    def _sine_gen(self, f0, ns, seg_offsets=None):
        """对齐 SineGen.forward（harmonic_num=0）+ SourceModuleHnNSF。

        f0: [1,P] f32，ns: [1,P*480,1] 预生成高斯噪声 -> har_source [1,1,480P]。

        seg_offsets: [1,L,1] 或 None。None 时与现行为完全一致（f0 从第 0 帧起
        逐帧累计相位，第 0 帧偏移 0）；非 None 时 f0 为"目标段"（相对窗口起点），
        seg_offsets[j] 是段内第 j 帧（绝对帧 zs+j）的**总**相位偏移（= 全量视角
        fmod-cumsum 对应值，由调用方按全局帧索引预计算，含前帧累计与第 0 帧
        偏移），逐帧加到基相位上，保证部分合成与全量合成逐位一致。
        """
        w = self.w
        upp = self.cfg.upp
        sr = self.cfg.sr
        f0r = f0[:, None].transpose(0, 2, 1)  # [1,P,1]
        rad = f0r / sr * np.arange(1, upp + 1, dtype=np.float32)[None, None, :]
        if seg_offsets is None:
            rad2 = np.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
            rad_acc = np.fmod(np.cumsum(rad2, axis=1), 1.0)
            rad = rad.copy()
            rad[:, 1:, :] += rad_acc[:, :-1, :]
        else:
            # 逐位一致：seg_offsets[j] 已含该帧（含第 0 帧）的全部前帧相位累计。
            rad = rad + np.asarray(seg_offsets, dtype=np.float32)
        rad = rad.reshape(1, -1, 1)  # [1,P*480,1]
        sine = np.sin(2 * np.pi * rad) * 0.1
        uv = (f0r > 0).astype(np.float32)
        uv = np.repeat(uv, upp, axis=1)  # nearest 上采样 upp 倍
        noise_amp = uv * 0.003 + (1 - uv) * 0.1 / 3
        noise = noise_amp * ns
        sine = sine * uv + noise
        l_lin = nn.linear(sine, w["dec.m_source.l_linear.weight"],
                          w["dec.m_source.l_linear.bias"])
        return nn.tanh(l_lin).transpose(0, 2, 1)  # [1,1,480P]

    def _dec_conv_pre_batch(self, z, g):
        """dec 入口 batch（P1-3）：conv_pre 一次提交。

        conv_pre(z) -> x。g 非 None 时 cond(g) 叠加走 ``_conv1d_opt``（逐次
        同路径：cond 输入 [1,256,1] 元素 < ``_THRESHOLD``，逐次本就回退 numpy，
        保持位级一致；cond 的 ``[1,C,1]`` 输出靠 numpy 广播加到 x 上）。
        仅当 ``self._dec_batch``（vulkan 后端 + 常驻权重）时调用。
        """
        w = self.w
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            x = br.conv1d(
                z, w["dec.conv_pre.weight"], w["dec.conv_pre.bias"],
                padding=3,
                buf_w=_weights.get("dec.conv_pre.weight"),
                buf_b=_weights.get("dec.conv_pre.bias"),
            )
            br.commit()
            out = x.numpy()
            if g is not None:
                out = out + _conv1d_opt(g, w["dec.cond.weight"],
                                        w["dec.cond.bias"], "dec.cond.weight")
            return out
        finally:
            br.release()

    def _dec_ups_batch(self, i, x, har, up_base: str, wn):
        """dec 第 i 级上采样 batch（P1-3）：conv_t1d(ups) + conv1d(noise) +
        add_inplace 一次提交，等价于逐次的 ``x = ups(x) + noise_conv(har)``。

        x / har 为 float32 numpy；ups 与 noise_convs 走常驻权重 buffer。
        返回 numpy ``[1, C_out, oL]``。
        """
        w = self.w
        cfg = self.cfg
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            up = br.conv_transpose1d(
                x, wn, w[up_base + ".bias"],
                stride=cfg.upsample_rates[i],
                padding=(cfg.upsample_kernels[i] - cfg.upsample_rates[i]) // 2,
                buf_w=_weights.get(up_base + ".weight"),
                buf_b=_weights.get(up_base + ".bias"),
            )
            src = br.conv1d(
                har,
                w[f"dec.noise_convs.{i}.weight"],
                w[f"dec.noise_convs.{i}.bias"],
                stride=cfg.noise_strides[i],
                padding=cfg.noise_strides[i] // 2,
                buf_w=_weights.get(f"dec.noise_convs.{i}.weight"),
                buf_b=_weights.get(f"dec.noise_convs.{i}.bias"),
            )
            br.add_inplace(up, src)
            br.commit()
            return up.numpy()
        except RuntimeError as e:
            # P1-7：大块（长音频）时本级 conv_t1d/conv1d 输出超 GPU 上限
            # （vulkan_batch_too_large）。降级为本级单次路径（_conv_t1d_opt/
            # _conv1d_opt 内部自动时间维切分 GPU），而不是抛错让整段 dec
            # 回退到逐次（那会失去 noise/ResBlock 的批量收益，每块慢 ~25s）。
            if "vulkan_batch_too_large" not in str(e):
                raise
            # T1.1（阻点3）：先在 GPU 内尝试分段（conv_t1d_seg + conv1d_seg
            # + add_seg，ups/noise 各自按输出列自动分段写父 buffer，零 numpy
            # 往返）；任何失败才降级 _dec_ups_single numpy 逐次兜底。
            try:
                return self._dec_ups_seg(i, x, har, up_base, wn)
            except Exception:  # noqa: BLE001  # 显存/引擎异常 → numpy 兜底
                return self._dec_ups_single(i, x, har, up_base, wn)
        finally:
            br.release()

    def _dec_ups_seg(self, i, x, har, up_base: str, wn):
        """T1.1（阻点3）：dec 第 i 级上采样的 GPU 内分段路径。

        ups conv_t1d 与 noise conv1d 均按输出列自动分段（超限级 2..N 段）
        写回父 buffer，add_inplace 就地累加（超限时 flat 分段），x/har 为
        numpy（各自只上传一次）。数值与整段 batch 路径同 kernel 同 push
        （逐位一致）。抛错（显存/引擎）由调用方降级 numpy 兜底。
        """
        w = self.w
        cfg = self.cfg
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            up = _conv_t1d_batch_or_seg(
                br, x, wn, w[up_base + ".bias"], up_base + ".weight",
                stride=cfg.upsample_rates[i],
                padding=(cfg.upsample_kernels[i] - cfg.upsample_rates[i]) // 2,
                out_c=int(wn.shape[1]),
            )
            src = _conv1d_batch_or_seg(
                br, har, w[f"dec.noise_convs.{i}.weight"],
                w[f"dec.noise_convs.{i}.bias"],
                f"dec.noise_convs.{i}.weight",
                stride=cfg.noise_strides[i],
                padding=cfg.noise_strides[i] // 2,
            )
            br.add_inplace_seg_multi(up, src)
            br.commit()
            return up.numpy()
        finally:
            br.release()

    def _dec_ups_single(self, i, x, har, up_base: str, wn):
        """dec 第 i 级上采样：逐次调用（ups conv_t1d + noise conv1d + 相加）。

        供非 batch 路径与 batch 超限回退共用（与 _dec_ups_batch 数值一致）。
        """
        w = self.w
        cfg = self.cfg
        x = _conv_t1d_opt(
            x, wn, w[up_base + ".bias"], up_base + ".weight",
            stride=cfg.upsample_rates[i],
            padding=(cfg.upsample_kernels[i] - cfg.upsample_rates[i]) // 2,
        )
        x_source = _conv1d_opt(
            har,
            w[f"dec.noise_convs.{i}.weight"],
            w[f"dec.noise_convs.{i}.bias"],
            f"dec.noise_convs.{i}.weight",
            stride=cfg.noise_strides[i],
            padding=cfg.noise_strides[i] // 2,
        )
        return x + x_source

    def _get_thirds(self, shape) -> object:
        """返回形状为 ``shape`` 的常驻 GPU buffer（全 ``1/num_kernels``）。

        perf(P1-6)：dec GPU 驻留路径的 ResBlock 输出平均（``x = Σout /
        num_kernels``）在 GPU 上用 ``mul_inplace(acc, thirds)`` 完成 ——
        mul_inplace 的 op4 shader 是逐元素乘（无广播，b 须同尺寸），故按
        ``(ch, L)`` 缓存一个 persistent buffer（进程内惰性创建一次，显存
        占用 ≈ 每级输出尺寸，P=198 时合计 ~10MB）。数值：``x · (1/3)`` 与
        numpy ``x / 3`` 差 ≤1 ulp（<1e-4 容差内）。
        """
        key = tuple(int(s) for s in shape)
        with self._thirds_lock:
            pb = self._thirds_cache.get(key)
            if pb is None or not getattr(pb, "valid", False):
                from runtime import vulkan_ops  # noqa: PLC0415

                arr = np.full(
                    key, 1.0 / self.num_kernels, dtype=np.float32)
                pb = vulkan_ops.get_context().persistent_upload(arr)
                self._thirds_cache[key] = pb
            return pb

    def _dec_resident_forward(self, z, g, har, defer_download: bool = False):
        """P1-6：dec 整段 GPU 驻留 —— conv_pre + cond + 4 级 ups/ResBlock +
        平均 + 尾部 lrelu/conv_post 全部录进**一个** BatchRunner、一次 commit。

        与逐级 batch（P1-3）相比，级间不再 numpy 下载/上传：
          - conv_pre 的 cond 输出融合进 bias（``b + cond[:,:,0]``，GPU conv
            的 bias 加法本就在 shader 内逐通道进行，与 numpy 事后广播加差 ≤1ulp）；
          - 每级 ups（conv_t1d + noise conv1d + add）输出 BatchTensor 直传
            ResBlock，ResBlock 3 组链全 GPU 流转；3 块输出就地累加进第一块
            的 buffer，再 ``mul_inplace(·, 1/3)`` 完成平均；
          - 尾部 ``lrelu(0.01)`` + ``conv_post`` 也在 GPU，只下载一次
            ``[1,1,L]`` 结果后 numpy ``tanh``（conv_post 无 bias）。

        ``defer_download=True``（T3.2）：commit 后**不等待不下载**，把
        runner/结果的所有权转移给 ``_DecCollector`` 立即返回；调用方在 GPU
        执行 dec 期间穿插其它 CPU 工作，稍后 ``collect()`` 统一 wait+下载。
        （即使 ``_dec_async=False`` 同步 commit 也返回 collector——collect
        仅是下载，语义统一；非 defer 时行为与原来完全一致。）

        超限（``vulkan_batch_too_large``，长音频）由 forward 捕获后回退
        逐级 batch/逐次路径。数值与逐级 batch 位级一致（同一 kernel），
        与逐次路径差 ≤1e-4。
        """
        w = self.w
        cfg = self.cfg
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            # ── conv_pre（cond 融合进 bias，g 为 None 时退化为纯 bias）──
            cb = w["dec.conv_pre.bias"]
            if g is not None:
                cond = _conv1d_opt(g, w["dec.cond.weight"],
                                   w["dec.cond.bias"], "dec.cond.weight")
                cb = cb + cond[0, :, 0]
            x = br.conv1d(
                z, w["dec.conv_pre.weight"], cb,
                padding=3,
                buf_w=_weights.get("dec.conv_pre.weight"),
            )
            # har 只上传一次（noise conv 每级复用同一 BatchTensor）
            har_t = br.copy(har)
            for i in range(self.n_ups):
                # 级入口 x（conv_pre 输出或上一级平均结果）被 conv_t 消费后即死
                x = br.leaky_relu_seg_multi(x, 0.1)
                up_base = f"dec.ups.{i}"
                wn = self._wn.get(up_base + ".weight")
                if wn is None:
                    wn = _deweight_norm(w[up_base + ".weight_v"],
                                        w[up_base + ".weight_g"])
                # T1.1：超限级 GPU 内分段（x 保持 BatchTensor 流转，零 numpy 往返）
                rate_i = cfg.upsample_rates[i]
                kern_i = cfg.upsample_kernels[i]
                pad_i = (kern_i - rate_i) // 2
                oL_ups = (x.shape[2] - 1) * rate_i - 2 * pad_i + (kern_i - 1) + 1
                up_pts = 1 * cfg.upsample_initial * oL_ups
                if up_pts > vulkan_ops._GRID_POINTS_MAX:
                    # 分段：父 buffer 整段分配，各段 conv_t1d_seg 绝对寻址写回
                    up = br.conv_t1d_seg_multi(
                        x, wn, w[up_base + ".bias"], stride=rate_i, padding=pad_i,
                        out_shape=(1, cfg.upsample_initial, oL_ups),
                        buf_w=_weights.get(up_base + ".weight"),
                        buf_b=_weights.get(up_base + ".bias"),
                    )
                else:
                    up = br.conv_transpose1d(
                        x, wn, w[up_base + ".bias"],
                        stride=rate_i,
                        padding=pad_i,
                        buf_w=_weights.get(up_base + ".weight"),
                        buf_b=_weights.get(up_base + ".bias"),
                    )
                br.tensor_done(x)  # 级入口载体已消费（不再使用）
                # T1.1 阻点1：noise conv1d 超限级（12s 起 128×139800=17.89M 点
                # > GRID）加 seg 分支——否则 resident 首次尝试就在此抛错整段
                # 回退逐级 batch（丢失级间 BatchTensor 流转收益）。
                src = _conv1d_batch_or_seg(
                    br, har_t, w[f"dec.noise_convs.{i}.weight"],
                    w[f"dec.noise_convs.{i}.bias"],
                    f"dec.noise_convs.{i}.weight",
                    stride=cfg.noise_strides[i],
                    padding=cfg.noise_strides[i] // 2,
                )
                br.add_inplace_seg_multi(up, src)
                br.tensor_done(src)  # noise 输出已被 add_inplace 消费
                x = up  # x = ups(x) + noise_conv(har)（up 就地成为本级的 x 载体）
                # ── num_kernels 个 ResBlock（同 dilation 语义）────────────
                # 每个输出 buffer 在其**最后一个消费算子录制后**立即
                # ``tensor_done`` 归还 runner 本地池，供后续同尺寸输出复用
                # （队列录制顺序保证"复用后的写入 op 排在旧读者之后"）。
                # 保留：acc（块 1 最终输出的累加器，mul_inplace 后即下一级 x）。
                acc = None
                for rb in self.resblocks[i * self.num_kernels:
                                         (i + 1) * self.num_kernels]:
                    # 每块独立输入副本（leaky_relu 就地覆写，块间不共享）
                    # T1.1：超限级 copy/lrelu/add/mul 走 GPU 内分段（同 kernel
                    # 同 push，逐位一致），保持 x 全程 BatchTensor 流转。
                    cur0 = br.copy_seg_multi(x)
                    cur = cur0
                    for j in range(3):
                        orig = br.copy_seg_multi(cur)  # 残差：组输入原值
                        c1 = br.leaky_relu_seg_multi(cur, 0.1)
                        d = [1, 3, 5][j]
                        k = rb.kernel_size
                        base1 = f"{rb.base}.convs1.{j}"
                        wn1 = rb._wn.get(base1 + ".weight")
                        if wn1 is None:
                            wn1 = _deweight_norm(w[base1 + ".weight_v"],
                                                 w[base1 + ".weight_g"])
                        oL_rb = (c1.shape[2] + (k * d - d) - d * (k - 1) - 1) // 1 + 1
                        if 1 * c1.shape[1] * oL_rb > vulkan_ops._GRID_POINTS_MAX:
                            c1 = br.conv1d_seg_multi(
                                c1, wn1, w[base1 + ".bias"],
                                stride=1, padding=(k * d - d) // 2, dilation=d,
                                out_shape=(1, c1.shape[1], oL_rb),
                                buf_w=_weights.get(base1 + ".weight"),
                                buf_b=_weights.get(base1 + ".bias"),
                            )
                        else:
                            c1 = br.conv1d(
                                c1, wn1, w[base1 + ".bias"],
                                dilation=d, padding=(k * d - d) // 2,
                                buf_w=_weights.get(base1 + ".weight"),
                                buf_b=_weights.get(base1 + ".bias"),
                            )
                        if j == 0:
                            br.tensor_done(cur0)  # 块输入副本已被 conv1 消费
                        c1 = br.leaky_relu_seg_multi(c1, 0.1)
                        base2 = f"{rb.base}.convs2.{j}"
                        wn2 = rb._wn.get(base2 + ".weight")
                        if wn2 is None:
                            wn2 = _deweight_norm(w[base2 + ".weight_v"],
                                                 w[base2 + ".weight_g"])
                        oL_rb2 = (c1.shape[2] + (k - 1) - 1 * (k - 1) - 1) // 1 + 1
                        if 1 * c1.shape[1] * oL_rb2 > vulkan_ops._GRID_POINTS_MAX:
                            c2 = br.conv1d_seg_multi(
                                c1, wn2, w[base2 + ".bias"],
                                stride=1, padding=(k - 1) // 2,
                                out_shape=(1, c1.shape[1], oL_rb2),
                                buf_w=_weights.get(base2 + ".weight"),
                                buf_b=_weights.get(base2 + ".bias"),
                            )
                        else:
                            c2 = br.conv1d(
                                c1, wn2, w[base2 + ".bias"],
                                padding=(k - 1) // 2,
                                buf_w=_weights.get(base2 + ".weight"),
                                buf_b=_weights.get(base2 + ".bias"),
                            )
                        br.tensor_done(c1)  # conv 链中间结果已被 conv2 消费
                        br.add_inplace_seg_multi(c2, orig)
                        br.tensor_done(orig)  # 残差副本已被 add_inplace 消费
                        cur = c2
                    # 平均：就地把块输出累进 acc（加法顺序与 numpy 一致
                    # ((o1+o2)+o3)）；块 1 的最终 c2 即 acc 载体，不回收；
                    # 块 2+ 的输出被 acc 消费后立即回收。
                    if acc is None:
                        acc = cur
                    else:
                        br.add_inplace_seg_multi(acc, cur)
                        br.tensor_done(cur)
                # 平均：acc 乘 1/3（mul_inplace 就地）后即成为下一级的载体 x；
                # acc 的 buffer 仍在使用，不回收（由下上级的 conv_t 消费）。
                x = br.mul_inplace_seg_multi(acc, self._get_thirds(acc.shape))
            # ── 尾部：lrelu(0.01) + conv_post（无 bias）+ 一次下载 ──────
            x = br.leaky_relu_seg_multi(x, 0.01)
            carrier = x
            x = br.conv1d(
                x, w["dec.conv_post.weight"], None,
                padding=3,
                buf_w=_weights.get("dec.conv_post.weight"),
            )
            br.tensor_done(carrier)  # conv_post 已消费最后一级载体
            br.commit(async_=self._dec_async)  # 异步提交；numpy() 内自动 wait
            if defer_download:
                # T3.2：所有权转移给 collector（不等待不下载），调用方稍后
                # collect() 统一 wait+下载+tanh；runner 由 collector 释放。
                return _DecCollector(self, br, x)
            return nn.tanh(x.numpy())
        finally:
            if not defer_download:
                br.release()

    # ------------------------------------------------------------------
    # P2：dec 多块批量（所有片段一起送入声码器）
    # ------------------------------------------------------------------
    def decode_batch(self, z_list, g, f0_list, ns_list, off_list):
        """多块批量 dec：N 块独立正反向合并 GPU 推理（P2，用户清单第 10 项）。

        参数（各块独立、长度可不等）:
            z_list:   list of ``[1,192,n_seg_i]``（dec 输入，已乘 x_mask）
            g:        ``[1,256,1]`` 或 None（各块同一 sid，共用说话人嵌入）
            f0_list:  list of ``[1,n_seg_i]`` f32
            ns_list:  list of ``[1,n_seg_i*480,1]``（SineGen 噪声）
            off_list: list of ``[1,n_seg_i,1]``（相位预推进偏移）或 None
        返回 list of ``[1,1,480*n_seg_i]``（tanh 波形），与逐块
        ``forward(z, f0, g, ns, seg_offsets=off)`` 语义一致。

        numpy 后端（``_dec_batch=False``）直接逐块 ``forward``（行为与现状
        完全一致）；vulkan 后端走 ``_dec_batch_forward``（无 padding 按块
        展开、同 kernel 同参数 → 逐位一致），任何引擎异常（含超限）
        → 逐块回退（保数值、弃批量收益）。
        """
        N = len(z_list)
        if N == 0:
            return []
        if not self._dec_batch:
            return [
                self.forward(z, f0, g, ns, seg_offsets=off)
                for z, f0, ns, off in zip(z_list, f0_list, ns_list, off_list)
            ]
        try:
            return self._dec_batch_forward(z_list, g, f0_list, ns_list, off_list)
        except Exception:  # noqa: BLE001  # 引擎异常/超限 → 逐块回退（保数值）
            return [
                self.forward(z, f0, g, ns, seg_offsets=off)
                for z, f0, ns, off in zip(z_list, f0_list, ns_list, off_list)
            ]

    def _dec_batch_forward(self, z_list, g, f0_list, ns_list, off_list):
        """P2：N 块 dec 整段 GPU 批量 —— conv_pre + cond + 4 级 ups/ResBlock/
        平均 + 尾部 lrelu/conv_post 全部录进**一个** BatchRunner，无 padding
        按块展开（同 hubert 批量模式：各块独立算子序列、同 kernel 同参数）。

        与单块 ``_dec_resident_forward`` 的差异只在"块"维度：
          - conv_pre：N 块各一 conv1d（cond 融合进 bias）+ N 块 har copy，
            一次 commit；
          - 每级：每块 61 dispatch（lrelu + conv_t1d + noise conv1d + add +
            3 ResBlock × 18 + 平均），按 ``_DEC_BATCH_MAX_BLOCKS``(6) 块分组，
            每组一次 async commit + ``wait()`` —— wait 解锁本地池（组内
            tensor_done 回收的 buffer 跨 commit 复用安全，同 hubert P1-9
            策略）；级输出 x[b] 是流转 BatchTensor（不 done、不回收，
            跨组/跨级安全：recorder 按提交序单队列执行）；
          - 尾部：lrelu + conv_post 一次 commit，逐块 numpy() → numpy tanh。

        超限（``vulkan_batch_too_large``，单块尺寸与逐块相同）由
        ``decode_batch`` 捕获后逐块回退（块内现有时间维切分不受影响）。
        数值：每块与 ``_dec_resident_forward`` 逐块同 kernel 同参数（算子
        序列、输入切片、常驻权重全同）→ 输出逐位一致。
        """
        w = self.w
        cfg = self.cfg
        N = len(z_list)
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            # ── conv_pre（cond 融合进 bias，与单块 resident 同策略）──
            cb = w["dec.conv_pre.bias"]
            if g is not None:
                cond = _conv1d_opt(g, w["dec.cond.weight"],
                                   w["dec.cond.bias"], "dec.cond.weight")
                cb = cb + cond[0, :, 0]
            har_ts = [None] * N
            xs = [None] * N
            for b in range(N):
                har_ts[b] = br.copy(
                    self._sine_gen(f0_list[b], ns_list[b],
                                   seg_offsets=off_list[b] if off_list else None)
                )
                xs[b] = br.conv1d(
                    z_list[b], w["dec.conv_pre.weight"], cb,
                    padding=3,
                    buf_w=_weights.get("dec.conv_pre.weight"),
                )
            br.commit(async_=self._dec_async)
            br.wait()  # conv_pre 输出将跨 commit 消费；统一 wait 语义
            for i in range(self.n_ups):
                up_base = f"dec.ups.{i}"
                wn = self._wn.get(up_base + ".weight")
                if wn is None:
                    wn = _deweight_norm(w[up_base + ".weight_v"],
                                        w[up_base + ".weight_g"])
                for g0 in range(0, N, _DEC_BATCH_MAX_BLOCKS):
                    g1 = min(N, g0 + _DEC_BATCH_MAX_BLOCKS)
                    for b in range(g0, g1):
                        # ── 级入口 lrelu + ups + noise + add（同单块）──
                        x = br.leaky_relu(xs[b], 0.1)
                        rate_i = cfg.upsample_rates[i]
                        kern_i = cfg.upsample_kernels[i]
                        pad_i = (kern_i - rate_i) // 2
                        oL_ups = (x.shape[2] - 1) * rate_i - 2 * pad_i + (kern_i - 1) + 1
                        up_pts = 1 * cfg.upsample_initial * oL_ups
                        if up_pts > vulkan_ops._GRID_POINTS_MAX:
                            # T1.1：超限级 GPU 内分段（块内 x 保持 BatchTensor）
                            up = br.conv_t1d_seg_multi(
                                x, wn, w[up_base + ".bias"], stride=rate_i, padding=pad_i,
                                out_shape=(1, cfg.upsample_initial, oL_ups),
                                buf_w=_weights.get(up_base + ".weight"),
                                buf_b=_weights.get(up_base + ".bias"),
                            )
                        else:
                            up = br.conv_transpose1d(
                                x, wn, w[up_base + ".bias"],
                                stride=rate_i,
                                padding=pad_i,
                                buf_w=_weights.get(up_base + ".weight"),
                                buf_b=_weights.get(up_base + ".bias"),
                            )
                        br.tensor_done(x)  # 级入口载体已消费
                        src = br.conv1d(
                            har_ts[b],
                            w[f"dec.noise_convs.{i}.weight"],
                            w[f"dec.noise_convs.{i}.bias"],
                            stride=cfg.noise_strides[i],
                            padding=cfg.noise_strides[i] // 2,
                            buf_w=_weights.get(f"dec.noise_convs.{i}.weight"),
                            buf_b=_weights.get(f"dec.noise_convs.{i}.bias"),
                        )
                        br.add_inplace(up, src)
                        br.tensor_done(src)
                        x = up
                        # ── num_kernels 个 ResBlock + 平均（同单块）──
                        acc = None
                        for rb in self.resblocks[i * self.num_kernels:
                                                 (i + 1) * self.num_kernels]:
                            cur0 = br.copy(x)
                            cur = cur0
                            for j in range(3):
                                orig = br.copy(cur)  # 残差：组输入原值
                                c1 = br.leaky_relu(cur, 0.1)
                                d = [1, 3, 5][j]
                                k = rb.kernel_size
                                base1 = f"{rb.base}.convs1.{j}"
                                wn1 = rb._wn.get(base1 + ".weight")
                                if wn1 is None:
                                    wn1 = _deweight_norm(
                                        w[base1 + ".weight_v"],
                                        w[base1 + ".weight_g"])
                                oL_rb = (c1.shape[2] + (k * d - d) - d * (k - 1) - 1) // 1 + 1
                                if 1 * c1.shape[1] * oL_rb > vulkan_ops._GRID_POINTS_MAX:
                                    c1 = br.conv1d_seg_multi(
                                        c1, wn1, w[base1 + ".bias"],
                                        stride=1, padding=(k * d - d) // 2, dilation=d,
                                        out_shape=(1, c1.shape[1], oL_rb),
                                        buf_w=_weights.get(base1 + ".weight"),
                                        buf_b=_weights.get(base1 + ".bias"),
                                    )
                                else:
                                    c1 = br.conv1d(
                                        c1, wn1, w[base1 + ".bias"],
                                        dilation=d, padding=(k * d - d) // 2,
                                        buf_w=_weights.get(base1 + ".weight"),
                                        buf_b=_weights.get(base1 + ".bias"),
                                    )
                                if j == 0:
                                    br.tensor_done(cur0)
                                c1 = br.leaky_relu(c1, 0.1)
                                base2 = f"{rb.base}.convs2.{j}"
                                wn2 = rb._wn.get(base2 + ".weight")
                                if wn2 is None:
                                    wn2 = _deweight_norm(
                                        w[base2 + ".weight_v"],
                                        w[base2 + ".weight_g"])
                                oL_rb2 = (c1.shape[2] + (k - 1) - 1 * (k - 1) - 1) // 1 + 1
                                if 1 * c1.shape[1] * oL_rb2 > vulkan_ops._GRID_POINTS_MAX:
                                    c2 = br.conv1d_seg_multi(
                                        c1, wn2, w[base2 + ".bias"],
                                        stride=1, padding=(k - 1) // 2,
                                        out_shape=(1, c1.shape[1], oL_rb2),
                                        buf_w=_weights.get(base2 + ".weight"),
                                        buf_b=_weights.get(base2 + ".bias"),
                                    )
                                else:
                                    c2 = br.conv1d(
                                        c1, wn2, w[base2 + ".bias"],
                                        padding=(k - 1) // 2,
                                        buf_w=_weights.get(base2 + ".weight"),
                                    buf_b=_weights.get(base2 + ".bias"),
                                )
                                br.tensor_done(c1)
                                br.add_inplace(c2, orig)
                                br.tensor_done(orig)
                                cur = c2
                            if acc is None:
                                acc = cur
                            else:
                                br.add_inplace(acc, cur)
                                br.tensor_done(cur)
                        xs[b] = br.mul_inplace(acc, self._get_thirds(acc.shape))
                    br.commit(async_=self._dec_async)
                    br.wait()  # 组间 wait：解锁本地池供下组复用（同 hubert）
            # ── 尾部：lrelu(0.01) + conv_post + 一次下载/块 ──────
            carriers = [None] * N
            outs = [None] * N
            for b in range(N):
                carriers[b] = br.leaky_relu(xs[b], 0.01)
                outs[b] = br.conv1d(
                    carriers[b], w["dec.conv_post.weight"], None,
                    padding=3,
                    buf_w=_weights.get("dec.conv_post.weight"),
                )
                br.tensor_done(carriers[b])
            br.commit(async_=self._dec_async)
            return [nn.tanh(o.numpy()) for o in outs]
        finally:
            br.release()

    def _resblocks_batch(self, i, x_np, rbs):
        """同一级 ups 后的 ``num_kernels`` 个 ResBlock 合入一次 batch 提交（P1-3 深化）。

        ResBlock 结构（每块 3 组，dilation=1/3/5）：:

            orig = copy(x)          # 残差：保留原 x（lrelu 是就地算子，会覆写）
            c1   = lrelu(x, 0.1)
            c1   = conv1(convs1[j], dil=d)          # padding=(k*d-d)//2
            c1   = lrelu(c1, 0.1)
            c2   = conv2(convs2[j], dil=1)          # padding=(k-1)//2
            x    = c2 + orig                        # add_inplace

        每 ResBlock 3 组 × 6 算子 = 18 个 dispatch，3 个 ResBlock = 54 个
        dispatch 一次 ``rvc_batch_commit``（recorder max_sets=64，留余量）。
        常驻权重走 ``buf_w``/``buf_b``（与逐次 ``_conv1d_opt`` 同一 kernel，
        位级一致）；``x_np`` 仅上传一次、3 块共享。

        超限（conv1d/leaky/copy 输出点 > ``_GRID_POINTS_MAX``）抛
        ``RuntimeError("vulkan_batch_too_large: ...")`` 由 dec 循环捕获后
        回退逐次。返回 3 个输出 numpy 数组（与 ``rb.forward(x)`` 一致）。
        """
        w = self.w
        from runtime import vulkan_ops  # noqa: PLC0415  # 惰性避免包初始化环

        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            outs = []
            # T2.4：x_np 只上传一次为 BatchTensor（GPU 驻留），各 ResBlock
            # 的工作副本用 GPU 内 copy —— 原实现每个 rb 的 j=0 都上传同一
            # x_np（copy+leaky 各一次，12s 108 次 1.75GB ascontiguous 拷贝，
            # 0.46s）。copy 逐位无损，数值不变。
            # T1.1（阻点2）：超限级 copy/leaky/conv1d/add 走 GPU 内分段
            # （同 kernel 同 push，逐位一致），不再因 copy 17894400 点超限
            # 触发 forward 的"时间维切段 + numpy 拼接"（后者段首丢真实输入
            # 的 0 填充缺陷，见 T1.1 Step4 实测）。
            x_t = (br.copy_seg_multi(x_np) if isinstance(x_np, np.ndarray)
                   else x_np)
            for rb in rbs:
                cur = br.copy_seg_multi(x_t)  # 本 rb 工作副本（x_t 原值保留给下一 rb）
                for j in range(3):
                    # 残差保留：copy 原 cur（cur 是 BatchTensor，GPU 内 copy）
                    orig = br.copy_seg_multi(cur)
                    c1 = br.leaky_relu_seg_multi(cur, 0.1)
                    d = [1, 3, 5][j]
                    k = rb.kernel_size
                    base1 = f"{rb.base}.convs1.{j}"
                    wn1 = rb._wn.get(base1 + ".weight")
                    if wn1 is None:
                        wn1 = _deweight_norm(w[base1 + ".weight_v"],
                                             w[base1 + ".weight_g"])
                    c1 = _conv1d_batch_or_seg(
                        br, c1, wn1, w[base1 + ".bias"], base1 + ".weight",
                        dilation=d, padding=(k * d - d) // 2,
                    )
                    c1 = br.leaky_relu_seg_multi(c1, 0.1)
                    base2 = f"{rb.base}.convs2.{j}"
                    wn2 = rb._wn.get(base2 + ".weight")
                    if wn2 is None:
                        wn2 = _deweight_norm(w[base2 + ".weight_v"],
                                             w[base2 + ".weight_g"])
                    c2 = _conv1d_batch_or_seg(
                        br, c1, wn2, w[base2 + ".bias"], base2 + ".weight",
                        padding=(k - 1) // 2,
                    )
                    br.add_inplace_seg_multi(c2, orig)
                    cur = c2
                outs.append(cur)
            br.commit()
            return [o.numpy() for o in outs]
        finally:
            br.release()

    def forward(self, z, f0, g, ns, seg_offsets=None, defer_download: bool = False):
        """z: [1,192,P]；f0: [1,P]；g: [1,256,1]；ns: [1,P*480,1] -> [1,1,480P]。

        seg_offsets 透传给 _sine_gen（部分合成时的相位预推进，见其 docstring）。
        defer_download（T3.2）：resident 主路径下把下载延迟为 _DecCollector；
        回退/逐次路径无 async，直接返回波形（由调用方统一包装 collect 接口）。
        """
        w = self.w
        cfg = self.cfg
        har = self._sine_gen(f0, ns, seg_offsets=seg_offsets)
        if self._dec_batch and self._dec_resident:
            # P1-6：整段单 runner 单 commit（conv_pre+cond+ups+ResBlock+平均
            # +尾部全 GPU 驻留，级间免下载）。超限回退走下方逐级 batch 链。
            try:
                return self._dec_resident_forward(z, g, har,
                                                  defer_download=defer_download)
            except RuntimeError as e:
                if "vulkan_batch_too_large" not in str(e):
                    raise
        if self._dec_batch:
            try:
                x = self._dec_conv_pre_batch(z, g)
            except RuntimeError as e:
                if "vulkan_batch_too_large" not in str(e):
                    raise
                # 引擎 grid 上限防护：conv_pre 输出超限时本段回退逐次（
                # 不关闭全局 batch——ups/ResBlock 可能仍可用）。
                x = _conv1d_opt(z, w["dec.conv_pre.weight"],
                                w["dec.conv_pre.bias"],
                                "dec.conv_pre.weight", padding=3)
                if g is not None:
                    x = x + _conv1d_opt(g, w["dec.cond.weight"],
                                        w["dec.cond.bias"],
                                        "dec.cond.weight")
        else:
            x = _conv1d_opt(z, w["dec.conv_pre.weight"], w["dec.conv_pre.bias"],
                            "dec.conv_pre.weight", padding=3)
            if g is not None:
                x = x + _conv1d_opt(g, w["dec.cond.weight"], w["dec.cond.bias"],
                                    "dec.cond.weight")
        for i in range(self.n_ups):
            x = nn.leaky_relu(x, 0.1)
            up_base = f"dec.ups.{i}"
            wn = self._wn.get(up_base + ".weight")
            if wn is None:
                wn = _deweight_norm(w[up_base + ".weight_v"],
                                    w[up_base + ".weight_g"])
            if self._dec_batch:
                try:
                    x = self._dec_ups_batch(i, x, har, up_base, wn)
                except RuntimeError as e:
                    if "vulkan_batch_too_large" not in str(e):
                        raise
                    # 引擎 grid 上限防护：本级超限仅本次降级单次切分路径
                    # （P1-7 ups 内部已降级，这里兜底），**不持久关闭 _dec_batch**
                    # —— 否则一次超限会让后续所有推理永久走逐次（历史 bug）。
                    x = self._dec_ups_single(i, x, har, up_base, wn)
            else:
                x = self._dec_ups_single(i, x, har, up_base, wn)
            xs = None
            rbs = [self.resblocks[i * self.num_kernels + j]
                   for j in range(self.num_kernels)]
            if self._dec_batch:
                try:
                    rb_outs = self._resblocks_batch(i, x, rbs)
                except RuntimeError as e:
                    if "vulkan_batch_too_large" not in str(e):
                        raise
                    # 钥匙 #90：高级别 ups 后 ResBlock 输入 oL 大 → batch 超限。
                    # 时间维切段（每段输出 ≤ GPU 上限），段内 batch、段间拼接，
                    # 而不是整段降级逐次（那是 10s 块 ~18s 的开销来源）。
                    from runtime import vulkan_ops as _vo
                    Bx, Cx, Lx = x.shape
                    max_l = max(1, (_vo._GRID_POINTS_MAX // 2) //
                                max(1, Bx * Cx))
                    segs = []
                    for s0 in range(0, Lx, max_l):
                        segs.append(self._resblocks_batch(
                            i, x[:, :, s0:s0 + max_l], rbs))
                    rb_outs = [
                        np.concatenate([seg[k] for seg in segs], axis=2)
                        for k in range(len(segs[0]))
                    ]
            else:
                rb_outs = [rb.forward(x) for rb in rbs]
            for out in rb_outs:
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = nn.leaky_relu(x, 0.01)  # 与源码一致：最后一层 F.leaky_relu 默认 slope=0.01
        x = _conv1d_opt(x, w["dec.conv_post.weight"], None,
                        "dec.conv_post.weight", padding=3)
        return nn.tanh(x)


# ---------------------------------------------------------------------------
# SynthesizerTrn 入口
# ---------------------------------------------------------------------------
class SynthesizerTrn:
    """纯 numpy 的 RVC VITS 变体合成器（Ms256NSFsid / Ms768NSFsid 通用）。

    参数:
        checkpoint_path: .pth 权重路径（torch_compat 读取）
        seed: 可选，固定推理随机项（z_p 噪声、SineGen 噪声）
    """

    def __init__(self, checkpoint_path: str, config=None, sr: int | None = None):
        cpt = _load_ckpt(checkpoint_path)
        # checkpoint 兼容多种形态：
        #   推理格式：{"weight": {...}, "config": [...]}
        #   训练底模格式（f0G48k.pth 等）：{"model": {state_dict}, ...}，无 config
        w = None
        cfg = config
        if isinstance(cpt, dict):
            if "weight" in cpt and isinstance(cpt["weight"], dict):
                w = cpt["weight"]
                if cfg is None and "config" in cpt:
                    cfg = cpt["config"]
            elif "model" in cpt and isinstance(cpt["model"], dict):
                w = cpt["model"]
        if w is None:
            w = cpt
        self._w = w
        self.cfg = VitsConfig(w, cfg, sr=sr)
        self.enc_p = _TextEncoder(w, self.cfg)
        self.flow = _FlowBlock(w, self.cfg)
        self.dec = _GeneratorNSF(w, self.cfg)
        self.register_persistent_weights()

    def _resolve_g(self, sid) -> np.ndarray:
        """返回推理用说话人条件 g [1,gin,1]（emb_g(sid)）。"""
        return nn.embedding(sid, self._w["emb_g.weight"])[0].reshape(1, -1, 1)

    def register_persistent_weights(self) -> None:
        """把 enc_p（TextEncoder）+ flow 的全部卷积/线性权重常驻 GPU。

        perf(P1)：dec 权重此前已常驻（P1-2/P1-3）；enc_p/flow 的 conv/linear
        每次推理都在 ``nn.conv1d``/``nn.linear`` 里重新 upload 权重（conv1d
        每次 2 次 upload：x + w），是 vits 上传次数的大头。这里统一注册为
        ``vits.<原键>`` 形式（``_conv1d_opt``/``_linear_opt`` 按该 key 命中），
        使 enc_p/flow 每次调用只 upload 中间张量 x、权重不再重复上传。

        numpy 后端（GPUWeights 禁用）下为 no-op，且不填充 flow 的
        deweight_norm 缓存（保持与原来完全一致）。
        """
        w = self._w
        if not _weights.enabled:
            self.flow._wn_cache = {}
            return
        # enc_p：emb_phone 线性 + pitch 嵌入表 + 6 层 attention conv_q/k/v/o +
        # FFN conv_1/conv_2 + proj（全部 weight+bias，bias 单独注册）。
        # 注意 emb_phone 按 ``w.T``（[D,192]）上传：matmul 语义是 ``x @ w.T``，
        # 常驻 buffer 直接承载右操作数（与 hubert 的 ``_qkv_T`` 同模式）。
        pairs = [
            ("vits.enc_p.emb_phone.weight",
             np.ascontiguousarray(w["enc_p.emb_phone.weight"].T,
                                  dtype=np.float32)),
            ("vits.enc_p.emb_phone.bias", w["enc_p.emb_phone.bias"]),
        ]
        for i in range(self.cfg.n_layers):
            for name in ("conv_q", "conv_k", "conv_v", "conv_o"):
                base = f"enc_p.encoder.attn_layers.{i}.{name}"
                pairs.append((f"vits.{base}.weight", w[base + ".weight"]))
                pairs.append((f"vits.{base}.bias", w[base + ".bias"]))
            for name in ("conv_1", "conv_2"):
                base = f"enc_p.encoder.ffn_layers.{i}.{name}"
                pairs.append((f"vits.{base}.weight", w[base + ".weight"]))
                pairs.append((f"vits.{base}.bias", w[base + ".bias"]))
            for norm_name in ("norm_layers_1", "norm_layers_2"):
                base = f"enc_p.encoder.{norm_name}.{i}"
                pairs.append((f"vits.{base}.gamma", w[base + ".gamma"]))
                pairs.append((f"vits.{base}.beta", w[base + ".beta"]))
        pairs.append(("vits.enc_p.proj.weight", w["enc_p.proj.weight"]))
        pairs.append(("vits.enc_p.proj.bias", w["enc_p.proj.bias"]))
        # flow：4 个 RC 块 pre/post + WN cond/in_layers/res_skip_layers。
        # weight_norm 层先还原为普通权重并缓存（推理时不再重复 deweight_norm）。
        for idx in self.flow.rc_ids:
            for name in ("pre", "post"):
                base = f"flow.flows.{idx}.{name}"
                pairs.append((f"vits.{base}.weight", w[base + ".weight"]))
                pairs.append((f"vits.{base}.bias", w[base + ".bias"]))
            cond = f"flow.flows.{idx}.enc.cond_layer"
            cwn = _deweight_norm(w[cond + ".weight_v"], w[cond + ".weight_g"])
            self.flow._wn_cache[cond + ".weight"] = cwn
            pairs.append((f"vits.{cond}.weight", cwn))
            pairs.append((f"vits.{cond}.bias", w[cond + ".bias"]))
            for i in range(3):
                for name in ("in_layers", "res_skip_layers"):
                    base = f"flow.flows.{idx}.enc.{name}.{i}"
                    wn = _deweight_norm(w[base + ".weight_v"],
                                       w[base + ".weight_g"])
                    self.flow._wn_cache[base + ".weight"] = wn
                    pairs.append((f"vits.{base}.weight", wn))
                    pairs.append((f"vits.{base}.bias", w[base + ".bias"]))
        for key, arr in pairs:
            _weights.register(key, arr)

    # ------------------------------------------------------------------
    def _make_noise(self, phone, seed):
        """按固定顺序生成随机项：z_p 噪声 [1,h,P]，随后 SineGen 噪声 [1,P*upp,1]。

        ``RVC_FIXED_SEED`` 环境变量设置时，``seed is None`` 的调用自动使用
        该固定种子（P1-9 确定性验证）；显式 ``seed`` 优先。
        """
        P = phone.shape[1]
        if seed is None and _FIXED_SEED:
            seed = int(_FIXED_SEED)
        rng = np.random.RandomState(seed) if seed is not None \
            else np.random.RandomState()
        nz = rng.standard_normal((1, self.cfg.hidden, P)).astype(np.float32)
        ns = rng.standard_normal((1, P * self.cfg.upp, 1)).astype(np.float32)
        return nz, ns

    def infer(self, phone, pitch, nsff0, sid, seed: Optional[int] = None,
              skip_head=None, return_length=None, phone_gpu=None,
              _defer: bool = False):
        """完整推理；skip_head/return_length 提供时做"部分合成"（T52 优化）。

        参数:
            phone:  [1, P, D] float32（D = phone_dim）
            pitch:  [1, P] int64（0~255）
            nsff0:  [1, P] float32（帧 f0，48000Hz 域）
            sid:    [1] int64
            seed:   固定随机项则传入 int（z_p 噪声、SineGen 噪声）
            skip_head: 可选 int 帧数。跳过前面 skip_head 帧的特征处理，只返回
                从第 skip_head 帧开始的 return_length 帧对应的波形段。
            return_length: 可选 int 帧数。只返回这帧数对应的波形段。
            phone_gpu: 可选 ``PersistentBuffer``，为 phone[0]（[P, D]）的 GPU
                常驻副本。提供时 TextEncoder 第一层 emb_phone matmul 直接在
                GPU 上消费该 buffer（不再上传 feats）——pipeline 数据流
                中间张量显存驻留（perf(P1)）。numpy 后端应传 None。
            _defer: T3.2 内部参数。True 时 dec 下载延迟（返回带 collect() 的
                对象），供 pipeline 块间 CPU/GPU 流水线使用；部分合成路径
                （skip_head 模式）忽略本参数（返回波形）。

        返回:
            [1, 1, 480*P] float32 波形（tanh 输出）；skip_head 模式返回
            [1, 1, 480*return_length]。

        部分合成语义（对齐原版 infer，但保证数值更强）:
            - TextEncoder 全量（T5 相对位置注意力跨全窗口，与原版一致）。
            - flow 从 flow_head = max(head-24,0) 帧起算（原版语义；24 帧 =
              flow 的 WN 感受野预热），flow 输出第 24 帧起与全量逐位一致。
            - 本实现再额外提前 _DEC_PAD 帧（flow_head = head-24-_DEC_PAD），
              使 dec 输入段 [zs, ze) 全部落在 flow 的一致区（zs 相对 flow_head
              恰为 24 帧）。
            - GeneratorNSF 只合成 [zs, ze)（目标段 ± _DEC_PAD 边界上下文），
              输出中间 [head, head+length) 段，与"全量合成后裁剪"逐位一致。
            - SineGen 相位预推进：由全量 nsff0 预计算 fmod-cumsum，段内第
              j 帧相位偏移直接取全量第 (zs+j) 帧的值，无需合成前面帧。
        """
        nz, ns = self._make_noise(phone, seed)
        if skip_head is None or return_length is None:
            return self.infer_controlled(phone, pitch, nsff0, sid, nz, ns,
                                         phone_gpu=phone_gpu, _defer=_defer)
        head = int(skip_head)
        length = int(return_length)
        P = phone.shape[1]
        if head < 0 or head >= P or length <= 0 or head + length > P:
            raise ValueError(
                f"skip_head/return_length 越界: P={P}, head={head}, length={length}"
            )
        return self._infer_partial(phone, pitch, nsff0, sid, nz, ns, head,
                                   length, phone_gpu=phone_gpu)

    def _phase_offsets(self, nsff0):
        """全量视角的 SineGen 相位偏移数组 [1, P, 1]（fmod-cumsum）。

        与 _GeneratorNSF._sine_gen 全量路径的逐位浮点运算完全相同，供部分
        合成时做相位预推进（段内第 j 帧偏移 = 全量第 zs+j 帧偏移）。
        """
        cfg = self.cfg
        f0r = nsff0[:, None].transpose(0, 2, 1)  # [1,P,1]
        rad = f0r / cfg.sr * np.arange(1, cfg.upp + 1, dtype=np.float32)[None, None, :]
        rad2 = np.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
        return np.fmod(np.cumsum(rad2, axis=1), 1.0)  # [1,P,1]

    def _partial_prep(self, phone, pitch, nsff0, sid, nz, ns, head, length,
                      phone_gpu=None):
        """部分合成 dec 准备（P2）：``_infer_partial`` 的前 1~5 步，不执行 dec。

        返回 dict（直接供 ``decode_batch`` 合并批量推理）：
          - z_seg [1,192,n_seg]（已乘 x_mask）
          - nsff0_seg [1,n_seg] f32
          - ns_seg [1,n_seg*upp,1]（SineGen 噪声切片）
          - off [1,n_seg,1]（相位预推进偏移）
          - g [1,256,1]（说话人嵌入）
          - s0（波形采样点偏移 = (head-zs)*upp）
          - n_out（目标段采样点数 = length*upp）
        数值与 ``_infer_partial`` feed 给 ``dec.forward`` 的输入逐位一致。
        """
        cfg = self.cfg
        w = self._w
        P = phone.shape[1]
        upp = cfg.upp
        dec_pad = _DEC_PAD

        # 1) 说话人嵌入 + 2) phone/pitch 嵌入 + 3) 全量 TextEncoder
        g = self._resolve_g(sid)  # [1,256,1]（emb_g(sid)）
        x = self.enc_p.embed(phone, pitch, phone_pb=phone_gpu)
        x = x * math.sqrt(cfg.hidden)
        x = nn.leaky_relu(x, 0.1)
        x_mask = nn.generate_mask(P)  # [1,1,P] 全 1
        h = self.enc_p.encode(x, x_mask)  # [1,P,hidden] 全量
        m, logs = self.enc_p.stats(h)
        m = m * x_mask
        logs = logs * x_mask

        # 3) flow 段（与 _infer_partial 完全一致：flow_head 起算）
        flow_head = max(head - _FLOW_PAD - dec_pad, 0)
        z_p = (m[:, :, flow_head:]
               + np.exp(logs[:, :, flow_head:]) * nz[:, :, flow_head:] * 0.66666) \
            * x_mask[:, :, flow_head:]
        z = self.flow.reverse(z_p, x_mask[:, :, flow_head:], g)  # [1,h,P-flow_head]

        # 4) dec 段切片
        zs = max(head - dec_pad, 0)
        ze = min(P, head + length + dec_pad)
        z_seg = z[:, :, zs - flow_head: ze - flow_head]
        xm_seg = x_mask[:, :, zs:ze]
        nsff0_seg = nsff0[:, zs:ze]

        # 5) SineGen 相位预推进 + 噪声/掩码切片（逐位一致）
        rad_acc = self._phase_offsets(nsff0)  # [1,P,1] 全量
        n_seg = ze - zs
        if zs == 0:
            off = np.zeros((1, n_seg, 1), dtype=np.float32)
            if n_seg > 1:
                off[:, 1:, :] = rad_acc[:, : n_seg - 1, :]
        else:
            off = rad_acc[:, zs - 1: zs - 1 + n_seg, :]
        ns_seg = ns[:, zs * upp: ze * upp, :]

        return {
            "z_seg": z_seg * xm_seg, "nsff0_seg": nsff0_seg,
            "ns_seg": ns_seg, "off": off, "g": g,
            "s0": (head - zs) * upp, "n_out": length * upp,
        }

    def _full_prep(self, phone, pitch, nsff0, sid, nz, ns, phone_gpu=None):
        """全量 dec 准备（P2）：与 ``infer_controlled`` feed 给 ``dec.forward``
        的输入完全一致（off=None → SineGen 走全量 cumsum 相位路径）。"""
        cfg = self.cfg
        w = self._w
        P = phone.shape[1]
        upp = cfg.upp

        g = self._resolve_g(sid)  # [1,256,1]（emb_g(sid)）
        x = self.enc_p.embed(phone, pitch, phone_pb=phone_gpu)
        x = x * math.sqrt(cfg.hidden)
        x = nn.leaky_relu(x, 0.1)
        x_mask = nn.generate_mask(P)  # [1,1,P] 全 1
        h = self.enc_p.encode(x, x_mask)  # [1,P,hidden]
        m, logs = self.enc_p.stats(h)
        m = m * x_mask
        logs = logs * x_mask
        z_p = (m + np.exp(logs) * nz * 0.66666) * x_mask
        z = self.flow.reverse(z_p, x_mask, g)
        return {
            "z_seg": z * x_mask, "nsff0_seg": nsff0, "ns_seg": ns,
            "off": None, "g": g, "s0": 0, "n_out": P * upp,
        }

    def prep_dec(self, phone, pitch, nsff0, sid, seed: Optional[int] = None,
                 skip_head=None, return_length=None, phone_gpu=None):
        """P2：准备单块 dec 输入（enc_p + flow + 相位预推进），不执行 dec。

        参数与 ``infer`` 相同（skip_head/return_length 为部分合成语义）。
        返回 dict 供 ``decode_batch`` 合并多块批量推理；数值与 ``infer``
        内部 feed 给 ``dec.forward`` 的输入逐位一致（同一段 numpy 计算）。
        """
        nz, ns = self._make_noise(phone, seed)
        if skip_head is None or return_length is None:
            return self._full_prep(phone, pitch, nsff0, sid, nz, ns,
                                   phone_gpu=phone_gpu)
        head = int(skip_head)
        length = int(return_length)
        P = phone.shape[1]
        if head < 0 or head >= P or length <= 0 or head + length > P:
            raise ValueError(
                f"skip_head/return_length 越界: P={P}, head={head}, length={length}"
            )
        return self._partial_prep(phone, pitch, nsff0, sid, nz, ns, head,
                                  length, phone_gpu=phone_gpu)

    def decode_batch(self, preps):
        """P2：多块 dec 批量推理（所有片段一起送入声码器）。

        preps: list of ``prep_dec`` 返回的 dict（各块独立、长度可不等）。
        返回 list of ndarray：每块**目标段**波形 ``[1,1,n_out]``
        （自 ``s0`` 采样点起），与逐块 ``infer`` 的对应段逐位一致。
        """
        if not preps:
            return []
        z_list = [p["z_seg"] for p in preps]
        f0_list = [p["nsff0_seg"] for p in preps]
        ns_list = [p["ns_seg"] for p in preps]
        off_list = [p["off"] for p in preps]
        g = preps[0]["g"]
        outs = self.dec.decode_batch(z_list, g, f0_list, ns_list, off_list)
        return [o[:, :, p["s0"]: p["s0"] + p["n_out"]]
                for o, p in zip(outs, preps)]

    def _infer_partial(self, phone, pitch, nsff0, sid, nz, ns, head, length,
                       phone_gpu=None):
        """部分合成主体（见 infer docstring 的语义说明）。"""
        prep = self._partial_prep(phone, pitch, nsff0, sid, nz, ns, head,
                                  length, phone_gpu=phone_gpu)
        o = self.dec.forward(prep["z_seg"], prep["nsff0_seg"], prep["g"],
                             prep["ns_seg"], seg_offsets=prep["off"])
        return o[:, :, prep["s0"]: prep["s0"] + prep["n_out"]]

    def infer_controlled(self, phone, pitch, nsff0, sid, nz, ns,
                         phone_gpu=None, _defer: bool = False):
        """推理主体：随机项由调用方注入（nz=[1,h,P]、ns=[1,P*upp,1]）。

        供 torch 对照使用：两侧注入同一批噪声数组实现全确定性逐元素比较。
        phone_gpu: 同 infer（TextEncoder 第一层 emb_phone matmul 消费 GPU
        常驻 buffer，跳过 feats 上传；numpy 后端应传 None）。
        _defer（T3.2）：True 时 dec 下载延迟——返回带 ``collect()`` 的对象
        （_DecCollector / _ImmediateResult），调用方稍后取波形（块间流水线）。
        """
        cfg = self.cfg
        w = self._w
        P = phone.shape[1]

        # 1) 说话人嵌入（emb_g(sid)）
        g = self._resolve_g(sid)  # [1,256,1]

        # 2) phone embedding + pitch embedding
        x = self.enc_p.embed(phone, pitch, phone_pb=phone_gpu)
        x = x * math.sqrt(cfg.hidden)
        x = nn.leaky_relu(x, 0.1)

        # 3) x_mask 全 1（RVC 推理无 pad）
        x_mask = nn.generate_mask(P)  # [1,1,P]

        # 4) Encoder
        h = self.enc_p.encode(x, x_mask)  # [1,P,hidden]

        # 5) proj -> m, logs（各 inter=192 通道）
        m, logs = self.enc_p.stats(h)
        m = m * x_mask
        logs = logs * x_mask

        # 6) z_p（随机项 nz）
        z_p = (m + np.exp(logs) * nz * 0.66666) * x_mask

        # 7) flow 逆变换
        z = self.flow.reverse(z_p, x_mask, g)

        # 8) GeneratorNSF（SineGen 随机项 ns 由 _sine_gen 内部取用）
        o = self.dec.forward(z * x_mask, nsff0, g, ns, defer_download=_defer)
        if _defer and not hasattr(o, "collect"):
            # 回退/逐次路径（无 async 提交）→ 统一 collect() 接口
            o = _ImmediateResult(o)
        return o

    # ------------------------------------------------------------------
    @property
    def weight_dict(self):
        return self._w


_load_cache: dict = {}


def load_synthesizer(path: str) -> SynthesizerTrn:
    """懒加载缓存版构造器。"""
    if path not in _load_cache:
        _load_cache[path] = SynthesizerTrn(path)
    return _load_cache[path]
