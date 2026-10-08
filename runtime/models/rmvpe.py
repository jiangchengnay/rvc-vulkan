# -*- coding: utf-8 -*-
"""RMVPE 基频（F0）提取模型 —— 纯 numpy 推理实现（f0 性能优化版）。

对齐 ``infer/rmvpe.py``（RVC 的 torch 推理）的完整语义，作为 RVC 去 CUDA 化
移植的 f0 主力模型：输入 16kHz 单声道音频，输出每帧基频（Hz）。

模型结构（E2E(4, 1, (2, 2))，逐层核对 rmvpe.py 源码）：
    1. mel 预处理（MelSpectrogram, center=True）：
       torch.stft(1024/160, hann, center reflect pad) -> |X| [513, F]
       mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128,
                                       fmin=30, fmax=8000, htk=True)
       log_mel = log(clamp(W @ |X|, min=1e-5))  -> [1, 128, F]
    2. U-Net（DeepUnet）：
       Encoder: BN + 5 × ResEncoderBlock(kernel=(2,2), n_blocks=4)
                (1→16, 16→32, 32→64, 64→128, 128→256)，每块 4 × ConvBlockRes
                + AvgPool2d(2,2) 下采样；空间 [F,128] -> [F/32,4]
       Intermediate: 4 × ResEncoderBlock(kernel=None)（256→512 + 512→512×3）
       Decoder: 5 × ResDecoderBlock（ConvTranspose2d 3×3 s2 p1 op1 + BN + ReLU，
                cat 对应 encoder 输出，4 × ConvBlockRes）512→256→128→64→32→16
    3. cnn: Conv2d(16→3, 3×3, pad1) -> [1,3,F,128] -> [1,F,384]
    4. fc: BiGRU(384,256,1,bidirectional) -> Linear(512→360) -> Sigmoid
    5. decode: 局部平均 cents -> f0 = 10 * 2^(cents/1200)，f0==10 置 0

本模块不 import torch —— 权重用 ``torch_compat.load_pth`` 读取（float16
存储，加载后 ``astype(np.float32)`` 再参与计算），算子全部来自
``runtime/nn`` 与本文件自带的 ``conv_transpose2d`` / 双向 GRU 实现。

性能优化（本机 AMD Radeon Pro VII GPU 弱，小算子独立 GPU dispatch 负优化，
UNet 约 123 次 conv2d + ~120 次 relu 每次 dispatch 固定 6-13ms）：
    - ``relu`` 强制纯 numpy（``np.maximum``，不走 ``runtime.backend.relu`` 的
      GPU 单算子分派）：2s 音频 554ms -> ~25ms、10s 1487ms -> ~480ms；
    - ``conv2d`` 按输入规模分派（``_conv2d_dispatch``）：小张量（x.size <
      阈值，约 2s 音频的全部层）走 numpy im2col+BLAS（快于 GPU 且稳定）；
      大张量（10s 的大空间层）走 vulkan backend（GPU 吞吐优势，且避免
      numpy 大矩阵 im2col 在 OpenBLAS 多线程下的病态波动）。
      环境变量 ``RVC_RMVPE_CONV=numpy|vulkan|auto`` 强制后端（默认 auto）；
    - ``BN`` 推理预计算 scale/shift（__init__ 时每层算好，推理仅乘加），
      并把 BN 与紧随的 ReLU 合并为可选 numba 融合内核
      （``RVC_RMVPE_NUMBA=0`` 关闭，未装 numba 自动回退 numpy）。
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np

from torch_compat import load_pth  # 纯 Python 读 .pth，不依赖 torch

from .. import nn as nn_ops
from ..dsp.fft import stft
from ..dsp.mel import mel_filter_bank

__all__ = ["RMVPE", "load_rmvpe"]

_SR = 16000
_N_FFT = 1024
_HOP = 160
_WIN = 1024
_N_MELS = 128
_FMIN = 30.0
_FMAX = 8000.0
_MEL_CLAMP = 1e-5

# 权重读取后统一转为 float32（ckpt 为 float16 存储）
_F32 = np.float32

# ---------------------------------------------------------------------------
# 性能开关（f0 优化）
# ---------------------------------------------------------------------------
# conv2d 后端：``numpy`` 强制纯 numpy；``vulkan`` 强制走 backend 分派；
# ``auto``（默认）按输入张量元素数分派（小 numpy / 大 vulkan）。
_CONV_MODE = os.environ.get("RVC_RMVPE_CONV", "").strip().lower()
# 分派阈值：x.size >= 此值时走 vulkan（本机 10s 音频的大空间层；2s 全部层
# 低于该值走 numpy）。实测：encL0 16->16 x.size=458K 时 numpy 89ms < vulkan
# 135ms；x.size=2.1M 时 numpy 428ms > vulkan 250ms —— 阈值取 1M。
_CONV_VULKAN_MIN_ELEMS = 1_000_000
# numba im2col 最小空间像素数（OH*OW）：小于此值走 numpy（intermediate 层
# 空间 28-128 实测 numba 慢 2-4x，enc/dec 层空间 ≥7168 numba 快 ~2x）。
_NB_MIN_SPATIAL = 1000

# P15b：convT（conv_transpose2d）的 x_up 插零 GPU 内生成（默认开）。
# ``0`` 回退 host 插零（np.zeros + 隔位赋值，逐位一致基线）。
_CONVT_GPU_INSERT = os.environ.get("RVC_RMVPE_CONVT_GPU_INSERT", "1").strip().lower() != "0"

# numba 融合开关：``0`` 关闭（回退 numpy），默认开启（未装 numba 自动回退）。
_NUMBA_ENABLED = os.environ.get("RVC_RMVPE_NUMBA", "").strip().lower() != "0"
try:  # pragma: no cover - 取决于环境是否安装 numba
    from numba import njit, prange  # type: ignore[import-not-found]  # noqa: PLC0415

    _NUMBA_OK = True
except Exception:  # noqa: BLE001  # pragma: no cover
    _NUMBA_OK = False

if _NUMBA_OK and _NUMBA_ENABLED:

    @njit(parallel=True, fastmath=False, cache=True)
    def _bn_relu_kernel(x, scale, shift):
        """y = relu(x * scale[c] + shift[c])，x: [B,C,H,W] 连续 float32。

        BN 推理的 scale/shift 已由调用方预计算，与 ReLU 一次遍历完成
        （替代 numpy 的逐元素多临时数组，2s/10s 大张量实测 1.5-2x）。
        """
        B, C, H, W = x.shape
        out = np.empty_like(x)
        for b in prange(B):
            for c in range(C):
                s = scale[c]
                t = shift[c]
                for h in range(H):
                    for w in range(W):
                        v = x[b, c, h, w] * s + t
                        out[b, c, h, w] = v if v > 0.0 else 0.0
        return out

    @njit(parallel=True, cache=True)
    def _im2col_nb(xp, KH, KW, OH, OW):
        """对已零填充输入做 im2col（C-contiguous 直接写入，跳过 numpy
        sliding_window_view -> transpose -> reshape 的多次拷贝）。

        xp: ``[B, C, Hp, Wp]``（pad 后）；返回 ``[B, OH*OW, C*KH*KW]``。
        实测 2s/10s 大空间层比 numpy im2col 快 1.5-2.5x（空间维 prange
        多线程，不占用 BLAS 线程池）。
        """
        B, C, Hp, Wp = xp.shape
        M = OH * OW
        K = C * KH * KW
        cols = np.empty((B, M, K), dtype=xp.dtype)
        for b in range(B):
            for m in prange(M):
                oh = m // OW
                ow = m % OW
                kk = 0
                for c in range(C):
                    for kh in range(KH):
                        h = oh + kh
                        for kw in range(KW):
                            w = ow + kw
                            cols[b, m, kk] = xp[b, c, h, w]
                            kk += 1
        return cols
else:  # pragma: no cover - 无 numba / 被开关关闭
    _NUMBA_OK = False


# ---------------------------------------------------------------------------
# 少量自实现算子（runtime/nn 未覆盖的部分）
# ---------------------------------------------------------------------------

# OpenBLAS 线程控制：本机 8 核 CPU 下，im2col 小矩阵乘的多线程调度开销远大于
# 并行收益（2s/10s 均出现 2-5x 抖动，单线程稳定且快）。用 threadpoolctl 在
# 推理期间临时限制 BLAS 线程为 1（numba prange 用自己的线程池不受影响）；
# 未装 threadpoolctl 时回退环境变量（需在解释器启动前设置）。
try:  # pragma: no cover - 可选依赖
    from threadpoolctl import threadpool_limits as _threadpool_limits  # type: ignore[import-not-found]

    _THREADPOOL_OK = True
except Exception:  # noqa: BLE001  # pragma: no cover
    _threadpool_limits = None
    _THREADPOOL_OK = False


def _conv2d_nb(x, w, b=None, padding=0):
    """numba im2col + BLAS matmul 的 conv2d（stride=1，kernel ≤ 5 通用）。

    x: ``[B,C,H,W]``；w: ``[O,C,KH,KW]``；padding: int 或 (ph,pw)。
    仅适用于 stride=1 的 UNet 卷积（全部 kernel3x3 pad1 / shortcut 1x1 pad0）。
    数值与 ``nn._conv2d_numpy`` 逐位一致（同 im2col 语义，实测 maxdiff=0）。
    """
    x = np.ascontiguousarray(x)
    w = np.ascontiguousarray(w)
    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        pad_h = pad_w = int(padding)
    B, C, H, W = x.shape
    O, _, KH, KW = w.shape
    if B != 1:
        # 仅支持 batch=1（UNet 推理固定 B=1）；其他回退 numpy 语义
        return nn_ops._conv2d_numpy(x, w, b, stride=1, padding=padding)
    OH = H + 2 * pad_h - KH + 1
    OW = W + 2 * pad_w - KW + 1
    if OH <= 0 or OW <= 0:
        out = np.zeros((B, O, max(OH, 0), max(OW, 0)), dtype=x.dtype)
        if b is not None:
            pass
        return out
    xp = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    cols = _im2col_nb(xp, KH, KW, OH, OW)  # [B, OH*OW, C*KH*KW]
    w2d = w.reshape(O, -1)  # [O, K]
    out = (cols[0] @ w2d.T).transpose(1, 0).reshape(B, O, OH, OW)
    if b is not None:
        out = out + np.asarray(b, dtype=out.dtype).reshape(1, -1, 1, 1)
    return out


def _conv2d_dispatch(x, w, b=None, stride=1, padding=0):
    """UNet conv2d 统一入口：按 ``RVC_RMVPE_CONV`` / auto 阈值选择后端。

    - ``numpy``：直接 ``nn._conv2d_numpy``（不经 backend 分派，省 GPU
      dispatch 固定开销）；
    - ``vulkan``：走 ``nn.conv2d``（vulkan 可用时 GPU 计算）；
    - ``auto``（默认）：x.size >= ``_CONV_VULKAN_MIN_ELEMS`` 走 vulkan
      （10s 大空间层 GPU 有吞吐优势）；否则 stride=1 且 numba 可用时走
      ``_conv2d_nb``（numba im2col + BLAS matmul，小/中层快于 numpy），
      无 numba 时回退 numpy。

    数值：三种路径同 im2col 公式，互相比对 maxdiff=0（见 _tmp_nb_im2col.py）。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    mode = _CONV_MODE
    if mode == "numpy":
        return nn_ops._conv2d_numpy(x, w, b, stride=stride, padding=padding)
    if mode in ("vulkan",):
        return nn_ops.conv2d(x, w, b, stride=stride, padding=padding)
    # auto：先按规模分派（大张量 vulkan，GPU 吞吐优势）
    if x.size >= _CONV_VULKAN_MIN_ELEMS:
        return nn_ops.conv2d(x, w, b, stride=stride, padding=padding)
    # 中小张量：numba im2col（stride=1 的 UNet 卷积，仅空间较大的层——
    # prange 并行度随 OH*OW 提升；intermediate 层 M≈28-128 用 numba 反而
    # 慢 ~2-4x，回退 numpy im2col+BLAS）。无 numba 时一律 numpy。
    if _NUMBA_OK:
        if isinstance(stride, (tuple, list)):
            sh, sw = int(stride[0]), int(stride[1])
        else:
            sh = sw = int(stride)
        if sh == 1 and sw == 1 and x.ndim == 4 and w.ndim == 4:
            H, W = x.shape[2], x.shape[3]
            if isinstance(padding, (tuple, list)):
                pad_h = int(padding[0])
            else:
                pad_h = int(padding)
            m_pix = (H + 2 * pad_h - w.shape[2] + 1) * (W + 2 * pad_h - w.shape[3] + 1)
            if m_pix >= _NB_MIN_SPATIAL:
                return _conv2d_nb(x, w, b, padding=padding)
    return nn_ops._conv2d_numpy(x, w, b, stride=stride, padding=padding)


def _relu_np(x):
    """ReLU 纯 numpy（不走 ``nn_ops.relu`` 的 GPU 单算子分派）。

    本机 AMD GPU 单算子 dispatch 固定 6-13ms：UNet 共 ~120 次 relu，
    GPU 路径 2s 累计 554ms、10s 累计 1487ms；numpy 路径分别仅 ~25ms /
    ~480ms（逐元素大数组，OpenBLAS 无瓶颈）。
    """
    return np.maximum(np.asarray(x), 0.0)


def _bn_relu_fused(x, scale, shift):
    """BN 推理 + ReLU 融合（scale/shift 已预计算）。

    numba 可用时单遍多线程内核（省 BN 后 ReLU 的二次遍历），否则 numpy
    等价式（``maximum(x*scale+shift, 0)``，与分开计算逐位一致）。
    """
    if _NUMBA_OK:
        xc = np.ascontiguousarray(x)
        if xc.ndim == 4:
            return _bn_relu_kernel(xc, np.ascontiguousarray(scale), np.ascontiguousarray(shift))
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return np.maximum(
        np.asarray(x) * scale.reshape(shape) + shift.reshape(shape), 0.0
    )


def _convt_gpu_try(x, w_flip, B, C_in, iH, iW, x_up_h, x_up_w, pad2):
    """P15b: 尝试 GPU 内生成 x_up（stride-2 插零）+ 同 kernel conv2d。

    host 只上传原图 x（[B,C_in,iH,iW]），GPU 内展开为
    [B,C_in,2iH-1,2iW-1]（隔位 0），再喂给与 host 路径**完全相同**的
    ``rvc_conv2d`` kernel → 数值与 host 插零逐位一致（maxdiff=0）。

    只在"原路径本就会走 vulkan conv2d"时生效（否则切 GPU 是负优化，且
    改变数值来源路径）：
      - ``RVC_RMVPE_CONV=numpy`` → 永不（保持纯 numpy 基线）；
      - ``auto`` → x_up.size >= _CONV_VULKAN_MIN_ELEMS（与
        ``_conv2d_dispatch`` 的 vulkan 分派阈值一致，小层仍走 numba/numpy）；
      - ``vulkan`` → x_up.size >= 1024（ctx._THRESHOLD，低于它 ctx.conv2d
        自身回退 numpy，维持原行为）。
    另外输出超 ``_GRID_POINTS_MAX//8``（_conv2d_split_gpu 阈值）时回退
    ——split 路径需要 host 侧 x_up 切片，GPU buffer 无法直接切。

    任何异常（dll/形状/显存）都回退 host，不中断推理。

    返回 y（[B,C_out,OH,OW]）或 None（回退 host 插零路径）。
    """
    x_up_size = B * C_in * x_up_h * x_up_w
    if _CONV_MODE == "numpy":
        return None
    if _CONV_MODE in ("", "auto") and x_up_size < _CONV_VULKAN_MIN_ELEMS:
        return None  # 原路径本走 numba/numpy，不切 GPU
    if x_up_size < 1024:  # ctx._THRESHOLD（强制 vulkan 模式下 ctx 自身小张量回退）
        return None
    try:
        from runtime import vulkan_ops  # noqa: PLC0415

        ctx = vulkan_ops.get_context()
        kh, kw = w_flip.shape[2], w_flip.shape[3]
        oh = (x_up_h + 2 * pad2[0] - kh) // 1 + 1
        ow = (x_up_w + 2 * pad2[1] - kw) // 1 + 1
        if oh <= 0 or ow <= 0:
            return None
        if oh * ow > vulkan_ops._GRID_POINTS_MAX // 8:
            return None  # 原路径会走 _conv2d_split_gpu（需 host x_up），回退
        x_up_id = ctx.insert_zeros_2x(x)  # 上传原图 + GPU 插零 → x_up buffer
        try:
            return ctx.conv2d_from_buf(
                x_up_id, (B, C_in, x_up_h, x_up_w), w_flip, None,
                stride=1, padding=pad2,
            )
        finally:
            ctx.free(x_up_id)
    except Exception:  # noqa: BLE001  # 任何失败回退 host，不中断推理
        return None


def conv_transpose2d(
    x,
    w,
    b=None,
    stride=(2, 2),
    padding=(1, 1),
    output_padding=(1, 1),
):
    """2D 转置卷积（对齐 ``torch.nn.functional.conv_transpose2d`` 推理语义）。

    等价变换：输入按 stride 元素间插零 -> 核空间翻转 -> 普通 conv2d
    （padding = kernel-1-padding）-> 输出尺寸按 output_padding 补零。
    复用了 ``runtime.nn.conv2d`` 的优化 im2col 路径，数值上等价于原始
    转置卷积（卷积核不翻转、输出逐点累加）。

    参数:
        x: ``[B, C_in, H, W]``
        w: ``[C_in, C_out, KH, KW]``（PyTorch ConvTranspose2d 权重形状）
        b: ``[C_out]`` 或 None
        stride / padding / output_padding: int 或 (h, w)
    输出尺寸: ``oH = (H-1)*sh - 2*ph + (KH-1) + op_h + 1``。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(stride, (tuple, list)):
        sh, sw = int(stride[0]), int(stride[1])
    else:
        sh = sw = int(stride)
    if isinstance(padding, (tuple, list)):
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if isinstance(output_padding, (tuple, list)):
        oph, opw = int(output_padding[0]), int(output_padding[1])
    else:
        oph = opw = int(output_padding)

    B, C_in, iH, iW = x.shape
    C_out = w.shape[1]
    KH, KW = w.shape[2], w.shape[3]
    oH = (iH - 1) * sh - 2 * ph + (KH - 1) + oph + 1
    oW = (iW - 1) * sw - 2 * pw + (KW - 1) + opw + 1

    # 核空间翻转（转置卷积 vs 普通卷积的关系），并把 [C_in, C_out] 转成
    # conv2d 期望的 [C_out, C_in] 输出通道在前布局
    w_flip = np.ascontiguousarray(w.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
    pad2 = (KH - 1 - ph, KW - 1 - pw)
    # P15b：GPU 内插零（host 只传原图，省 x_up 上传；数值逐位一致，硬验收
    # maxdiff=0）。仅 stride=(2,2) 且原路径本走 vulkan conv2d 时生效；
    # 其余情形走下方 host 插零原路径（逐位一致基线）。
    y = None
    if _CONVT_GPU_INSERT and sh == 2 and sw == 2:
        y = _convt_gpu_try(x, w_flip, B, C_in, iH, iW,
                           (iH - 1) * sh + 1, (iW - 1) * sw + 1, pad2)
    if y is None:
        # 输入元素间插零（stride 倍率）
        x_up = np.zeros(
            (B, C_in, (iH - 1) * sh + 1, (iW - 1) * sw + 1), dtype=x.dtype
        )
        x_up[:, :, ::sh, ::sw] = x
        y = _conv2d_dispatch(x_up, w_flip, None, stride=1, padding=pad2)
    out = np.zeros((B, C_out, oH, oW), dtype=y.dtype)
    out[:, :, : y.shape[2], : y.shape[3]] = y
    if b is not None:
        out += np.asarray(b, dtype=out.dtype).reshape(1, -1, 1, 1)
    return out


def avg_pool2d_2x2(x):
    """AvgPool2d(kernel=(2,2), stride=(2,2))（无 padding），reshape-mean 实现。"""
    B, C, H, W = x.shape
    return x.reshape(B, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))


def _bgru(x, w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r):
    """双向单层 GRU（batch_first，独立正反向权重），对齐 PyTorch ``nn.GRU``。

    参数:
        x: ``[T, in]``（batch=1 单序列）
        w_ih / w_hh / b_ih / b_hh: 正向权重（PyTorch 门顺序 r, z, n）
        w_ih_r / w_hh_r / b_ih_r / b_hh_r: 反向独立权重（ckpt 的 *_reverse 键）

    返回:
        ``[T, 2H]``，前向隐层在前、反向隐层在后（同 PyTorch 双向输出布局）。

    T18：默认走 GPU 单 kernel 路径（``_bgru_gpu``，gru.comp：双向并行、
    帧内 matvec 由 256 线程并行、每帧 1 次 workgroup barrier），失败或
    ``RVC_RMVPE_GRU_GPU=0`` 时回退 host 逐帧 numpy 循环（原实现）。
    """
    if _GRU_GPU_FLAG:
        try:
            return _bgru_gpu(x, w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r)
        except Exception:
            # GPU 路径异常（驱动/显存/形状）→ 回退 host，不中断推理
            pass
    total = w_ih.shape[0]
    H = total // 3

    def run(xs, win, whn, bin_, bhn):
        H = whn.shape[0] // 3
        # gx = xs @ win.T + bin_ **不依赖 h** → 整体一次 matmul（[T,3H]），
        # 替代逐帧小 matmul（BiGRU 主耗时之一，长音频提速明显）。
        gx_all = xs @ win.T + bin_
        h = np.zeros(H, dtype=_F32)
        outs = np.empty((xs.shape[0], H), dtype=_F32)
        # nn.sigmoid 用 np.where 双分支求值，极端输入会产生无害的溢出警告
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(xs.shape[0]):
                gx = gx_all[t]
                gh = h @ whn.T + bhn
                r = nn_ops.sigmoid(gx[:H] + gh[:H])
                z = nn_ops.sigmoid(gx[H : 2 * H] + gh[H : 2 * H])
                n = nn_ops.tanh(gx[2 * H :] + r * gh[2 * H :])
                h = (1.0 - z) * n + z * h
                outs[t] = h
        return outs

    fwd = run(x, w_ih, w_hh, b_ih, b_hh)
    rev = run(x[::-1, :], w_ih_r, w_hh_r, b_ih_r, b_hh_r)[::-1, :]
    return np.concatenate([fwd, rev], axis=-1)


# T18：BiGRU GPU 单 kernel 路径开关（默认开；=0 回退 host）
_GRU_GPU_FLAG = os.environ.get("RVC_RMVPE_GRU_GPU", "1") != "0"


def _bgru_gpu(x, w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r):
    """T18：BiGRU 单 kernel GPU 化（engine gru.comp，双向并行一 dispatch）。

    数学与 host ``_bgru`` 完全一致（PyTorch GRU 门序 r,z,n）；差异仅
    float32 累加序（GPU 逐 k 顺序累加 vs BLAS 分块），实测逐帧
    maxdiff ~2e-6（随机权重尺度），f0 级可忽略。

    流程：gx 半区拼接（fwd = x@w_ih.T+b_ih、rev = x@w_ih_r.T+b_ih_r，
    host 一次 numpy matmul，与 host 路径相同成本）→ 权重/bias 拼接 →
    单发 rvc_gru（独立 submit，避开 batch 状态机）→ 下载 [T,2H]。
    """
    import numpy as _np  # noqa: PLC0415
    from runtime import _vulkan, vulkan_ops  # noqa: PLC0415

    T = x.shape[0]
    H = w_hh.shape[0] // 3
    K = 3 * H
    gx_f = x @ w_ih.T + b_ih  # [T, 3H]
    gx_r = x @ w_ih_r.T + b_ih_r  # [T, 3H]
    gx2 = _np.ascontiguousarray(_np.concatenate([gx_f, gx_r], axis=0), dtype=_F32)
    w2 = _np.ascontiguousarray(_np.concatenate([w_hh, w_hh_r], axis=0), dtype=_F32)
    b2 = _np.ascontiguousarray(_np.concatenate([b_hh, b_hh_r], axis=0), dtype=_F32)
    ctx = vulkan_ops.get_context()
    a_id = ctx.upload(gx2)
    b_id = ctx.upload(w2)
    c_id = ctx.upload(b2)
    o_id = ctx._alloc_output(T * 2 * H)
    try:
        with ctx._lock:
            _vulkan._check(
                _vulkan.dll.rvc_gru(ctx._handle, a_id, b_id, c_id, o_id, T, H),
                "rvc_gru",
            )
        return ctx.download(o_id, (T, 2 * H))
    finally:
        ctx.free(a_id)
        ctx.free(b_id)
        ctx.free(c_id)
        ctx.free(o_id)


def _bn_apply(x, ckpt, prefix, cache=None):
    """BatchNorm2d 推理（prefix 指向 ``*.weight`` 所在层）。

    ``cache``（dict[prefix -> (scale, shift)]）存在时预计算 scale/shift
    （``scale = gamma/sqrt(rv+eps)``、``shift = beta - rm*scale``），推理只
    做乘加，省去每层重复的除法/平方根；无 cache 时保持原实现。
    """
    if cache is not None and prefix in cache:
        scale, shift = cache[prefix]
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * scale.reshape(shape) + shift.reshape(shape)
    return nn_ops.batch_norm_infer(
        x,
        ckpt[prefix + ".weight"].astype(_F32),
        ckpt[prefix + ".bias"].astype(_F32),
        ckpt[prefix + ".running_mean"].astype(_F32),
        ckpt[prefix + ".running_var"].astype(_F32),
        eps=1e-5,
    )


def _bn_scale_shift(ckpt, prefix):
    """预计算 BN 推理的 (scale, shift)（float32，形状 [C]）。"""
    gamma = ckpt[prefix + ".weight"].astype(_F32)
    beta = ckpt[prefix + ".bias"].astype(_F32)
    rm = ckpt[prefix + ".running_mean"].astype(_F32)
    rv = ckpt[prefix + ".running_var"].astype(_F32)
    scale = gamma / np.sqrt(rv + 1e-5)
    shift = beta - rm * scale
    return scale, shift


def _conv_block_res(x, ckpt, prefix, bn_cache=None):
    """ConvBlockRes（Conv2d3x3+BN+ReLU ×2 与恒等/1x1 shortcut 残差）。

    prefix 形如 ``...conv.N``。本 ckpt 的 ConvBlockRes 有参模块编号为
    ``conv.N.conv.0``（Conv2d in→out）/ ``conv.N.conv.1``（BN）/
    ``conv.N.conv.3``（Conv2d out→out）/ ``conv.N.conv.4``（BN）
    （ReLU 无语，索引 2/5 不产生权重键；与 infer/rmvpe.py 源码的
    Sequential(0,1,3,4) 编号一致，仅 PyTorch 自动命名因此错位为 0/1/3/4）。
    可选 ``conv.N.shortcut.*``（in != out 时存在）。

    优化：conv2d 走 ``_conv2d_dispatch``（auto 规模分派），BN+ReLU 融合
    （numba 单遍或 numpy 一次乘加），shortcut 残差加法保持 numpy。
    """
    w0 = ckpt[prefix + ".conv.0.weight"].astype(_F32)
    w2 = ckpt[prefix + ".conv.3.weight"].astype(_F32)
    if bn_cache is not None:
        s0, t0 = bn_cache[prefix + ".conv.1"]
        s1, t1 = bn_cache[prefix + ".conv.4"]
        y = _bn_relu_fused(_conv2d_dispatch(x, w0, None, stride=1, padding=1), s0, t0)
        y = _bn_relu_fused(_conv2d_dispatch(y, w2, None, stride=1, padding=1), s1, t1)
    else:
        y = _relu_np(_bn_apply(_conv2d_dispatch(x, w0, None, stride=1, padding=1), ckpt, prefix + ".conv.1"))
        y = _relu_np(_bn_apply(_conv2d_dispatch(y, w2, None, stride=1, padding=1), ckpt, prefix + ".conv.4"))
    if prefix + ".shortcut.weight" in ckpt:
        ws = ckpt[prefix + ".shortcut.weight"].astype(_F32)
        bs = ckpt[prefix + ".shortcut.bias"].astype(_F32) if prefix + ".shortcut.bias" in ckpt else None
        y = y + _conv2d_dispatch(x, ws, bs, stride=1, padding=0)
    else:
        y = y + x
    return y


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class RMVPE:
    """纯 numpy RMVPE 基频提取器（对齐 ``infer/rmvpe.py`` 的 torch 版本）。"""

    def __init__(self, model_path: str):
        t0 = time.perf_counter()
        raw = load_pth(model_path)
        # 全部权重转 float32（rmvpe.pt 为 float16 存储）
        self.W: Dict[str, np.ndarray] = {
            k: (v.astype(_F32) if v.dtype != _F32 else v) for k, v in raw.items()
        }
        self.mel_basis = mel_filter_bank(
            _SR, _N_FFT, _N_MELS, _FMIN, _FMAX
        ).astype(_F32)  # [128, 513]（HTK 公式，行和归一=1）
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4)).astype(_F32)
        # BN 推理 scale/shift 预计算（f0 优化：推理仅乘加，无重复除法/开方）
        # 识别特征：同时存在 weight/bias/running_mean/running_var 的层即为 BN
        self._bn_cache: Dict[str, tuple] = {}
        for k in list(self.W):
            if k.endswith(".weight"):
                prefix = k[: -len(".weight")]
                if (
                    prefix + ".bias" in self.W
                    and prefix + ".running_mean" in self.W
                    and prefix + ".running_var" in self.W
                ):
                    self._bn_cache[prefix] = _bn_scale_shift(self.W, prefix)
        self._load_time = time.perf_counter() - t0
        self._unet_batch_flag = False
        try:
            from runtime import backend  # noqa: PLC0415

            if backend.get_backend() == "vulkan":
                self._unet_batch_flag = os.environ.get(
                    "RVC_RMVPE_UNET_BATCH", "1").strip() != "0"
        except Exception:  # noqa: BLE001  # 后端不可用等：保持 host 路径
            self._unet_batch_flag = False
        if self._unet_batch_flag:
            self._register_gpu_weights()

    def _register_gpu_weights(self) -> None:
        """把 conv-BN 融合权重注册为 GPU 常驻（P0-1：batch 路径免每层权重上传）。

        键 ``rmvpe.<ckpt 前缀>.conv.0.weight`` 等（融合后 w' = w·scale、
        bias' = shift）；无 BN 的层（shortcut/cnn）注册原权重。numpy 后端 no-op。
        """
        try:
            from runtime import vulkan_weights as _vw  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return
        import numpy as _np

        for prefix in list(self._bn_cache):
            # BN 层前缀形如 "...conv.N.conv.1/4"——对应 conv 权重 "...conv.N.conv.0/3"
            if not prefix.endswith((".conv.1", ".conv.4")):
                continue
            conv_key = prefix[: -len("1")] if prefix.endswith(".conv.1") else prefix[: -len("4")]
            conv_key += "0" if prefix.endswith(".conv.1") else "3"
            wkey = conv_key + ".weight"
            if wkey not in self.W:
                continue
            w = self.W[wkey].astype(_F32)
            s, t = self._bn_cache[prefix]
            wf = w * _np.asarray(s, dtype=_F32).reshape(-1, 1, 1, 1)
            tf = _np.asarray(t, dtype=_F32).astype(_F32)
            _vw._weights.register("rmvpe." + wkey, wf)
            _vw._weights.register("rmvpe." + conv_key + ".bias", tf)
        for k, v in self.W.items():
            if k.startswith("cnn.") or k.endswith(".shortcut.weight"):
                _vw._weights.register("rmvpe." + k, v.astype(_F32))
            elif k.startswith("cnn."):
                pass
        # shortcut bias（若存在）
        for k, v in self.W.items():
            if k.endswith(".shortcut.bias"):
                _vw._weights.register("rmvpe." + k, v.astype(_F32))

    # ------------------------------------------------------------------ mel
    def _extract_mel(self, x: np.ndarray) -> np.ndarray:
        """``[T]`` -> ``[1, 128, F]`` log-mel（center=True，clamp=1e-5）。"""
        spec = stft(
            x, _N_FFT, _HOP, _WIN, window="hann", center=True, pad_mode="reflect"
        )  # [513, F] complex128
        mag = np.abs(spec).astype(_F32)  # [513, F]
        mel = self.mel_basis @ mag  # [128, F]
        log_mel = np.log(np.clip(mel, _MEL_CLAMP, None)).astype(_F32)
        return log_mel[None, ...]  # [1, 128, F]

    # ------------------------------------------------------------ U-Net 前向
    def _encoder(self, x):
        """Encoder：BN + 5 × ResEncoderBlock((2,2), 4 blocks)。

        x: ``[1, 1, F', 128]`` -> (x_latent, concat_tensors)。
        concat_tensors[i] = 第 i 层 blocks 输出（pool 前），通道 16/32/64/128/256。
        """
        ck = self.W
        bc = self._bn_cache
        x = _bn_apply(x, ck, "unet.encoder.bn", bc)
        concat = []
        n_layers = 5
        for i in range(n_layers):
            p = f"unet.encoder.layers.{i}"
            for j in range(4):  # n_blocks=4
                x = _conv_block_res(x, ck, f"{p}.conv.{j}", bc)
            concat.append(x)
            x = avg_pool2d_2x2(x)
        return x, concat

    def _intermediate(self, x):
        """Intermediate：4 × ResEncoderBlock(kernel=None, 4 blocks)。"""
        ck = self.W
        bc = self._bn_cache
        n_layers = 4
        for i in range(n_layers):
            p = f"unet.intermediate.layers.{i}"
            for j in range(4):
                x = _conv_block_res(x, ck, f"{p}.conv.{j}", bc)
        return x

    def _decoder(self, x, concat):
        """Decoder：5 × ResDecoderBlock（stride=(2,2)，cat + 4 × ConvBlockRes）。"""
        ck = self.W
        bc = self._bn_cache
        for i in range(5):
            p = f"unet.decoder.layers.{i}"
            w1 = ck[p + ".conv1.0.weight"].astype(_F32)  # [in, out, 3, 3]
            y = _bn_relu_fused(
                conv_transpose2d(x, w1, None, stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
                *bc[p + ".conv1.1"],
            )
            y = np.concatenate([y, concat[-1 - i]], axis=1)  # cat on channel
            for j in range(4):
                y = _conv_block_res(y, ck, f"{p}.conv2.{j}", bc)
            x = y
        return x

    def _decoder_batch(self, x, concat):
        """Decoder GPU batch（P0-1 第二步）：convT+cat 在 host（通道拼接无法 GPU），
        每层 4 × ConvBlockRes 用**一个** BatchRunner 一次 commit（常驻融合权重）。"""
        import numpy as _np
        from runtime import vulkan_ops  # noqa: PLC0415

        ck = self.W
        bc = self._bn_cache
        ctx = vulkan_ops.get_context()
        for i in range(5):
            p = f"unet.decoder.layers.{i}"
            w1 = ck[p + ".conv1.0.weight"].astype(_F32)
            y = _bn_relu_fused(
                conv_transpose2d(x, w1, None, stride=(2, 2), padding=(1, 1),
                                 output_padding=(1, 1)),
                *bc[p + ".conv1.1"],
            )
            y = np.concatenate([y, concat[-1 - i]], axis=1)
            br = vulkan_ops.BatchRunner(ctx)
            try:
                if y.size > vulkan_ops._GRID_POINTS_MAX:
                    # copy 超限（长段 150s 会触发：copy > GRID 上限无防护）：
                    # 该层回退 host 逐次（_conv_block_res），避免整段批量崩。
                    for j in range(4):
                        y = _conv_block_res(y, ck, f"{p}.conv2.{j}", bc)
                else:
                    ybt = br.copy(y)
                    for j in range(4):
                        ybt = self._conv_block_res_batch(br, ybt, ck,
                                                         f"{p}.conv2.{j}", bc)
                    br.commit()
                    y = ybt.numpy()
            finally:
                br.release()
            x = y
        return x

    def _unet(self, mel_padded):
        """``[1, 1, F', 128]`` -> ``[1, F', 360]``（cnn + fc 前的完整 U-Net）。"""
        ck = self.W
        x, concat = self._encoder(mel_padded)
        x = self._intermediate(x)
        x = self._decoder(x, concat)
        # cnn: Conv2d(16 -> 3, 3x3, pad 1, bias)
        wc = ck["cnn.weight"].astype(_F32)  # [3, 16, 3, 3]
        bc = ck["cnn.bias"].astype(_F32)
        x = _conv2d_dispatch(x, wc, bc, stride=1, padding=1)  # [1, 3, F', 128]
        x = x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)  # [1, F', 384]
        # fc: BiGRU -> Linear -> Sigmoid（Dropout 推理为恒等）
        w_ih, w_hh, b_ih, b_hh = (ck[k].astype(_F32) for k in (
            "fc.0.gru.weight_ih_l0", "fc.0.gru.weight_hh_l0",
            "fc.0.gru.bias_ih_l0", "fc.0.gru.bias_hh_l0",
        ))
        w_ih_r, w_hh_r, b_ih_r, b_hh_r = (ck[k].astype(_F32) for k in (
            "fc.0.gru.weight_ih_l0_reverse", "fc.0.gru.weight_hh_l0_reverse",
            "fc.0.gru.bias_ih_l0_reverse", "fc.0.gru.bias_hh_l0_reverse",
        ))
        h = _bgru(x[0], w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r)  # [F', 512]
        wf = ck["fc.1.weight"].astype(_F32)  # [360, 512]
        bf = ck["fc.1.bias"].astype(_F32)
        out = nn_ops.sigmoid(nn_ops.linear(h[None, ...], wf, bf))  # [1, F', 360]
        return out

    # ================================================================ P0-1
    # GPU batch 版 UNet（用户明确要求：小算子也必须真正走 GPU，不许回退 CPU
    # 解释成硬件特性）。encoder+intermediate 的 conv2d/BN/ReLU/残差/pool 全部
    # 录进**一个** BatchRunner 一次 commit，级间 BatchTensor 流转（免下载）；
    # BN 用预计算 scale/shift 的 mul/add（GPU），pool 用 conv2d 对角核实现，
    # 残差用入口 copy 副本。decoder（含 transpose2d 与 cat）第一步保持 host。
    # 环境变量 RVC_RMVPE_UNET_BATCH=0 关闭（回退原 _unet）。
    # ================================================================
    def _unet_enc_batch(self, mel_padded):
        """encoder+intermediate GPU batch：返回 (x_decoder, concat_list)。

        x_decoder: numpy [1, 128, F'/32, 16]（decoder 输入）
        concat_list: numpy 列表（每层 pool 前输出，供 decoder cat）。
        """
        import numpy as _np
        from runtime import vulkan_ops  # noqa: PLC0415

        ck = self.W
        bc = self._bn_cache
        ctx = vulkan_ops.get_context()
        br = vulkan_ops.BatchRunner(ctx)
        try:
            # encoder 入口 BN（仿 _encoder 的 bn 层）
            x = _np.asarray(mel_padded, dtype=_F32)
            # 入口 BN（unet.encoder.bn）：host 做一次（权重 3x3 卷积前的规范层）
            x = _bn_apply(x, ck, "unet.encoder.bn", bc)
            xbt = br.copy(x)  # -> BatchTensor
            concat_t = []
            # encoder：5 层 × 4 块
            for i in range(5):
                p = f"unet.encoder.layers.{i}"
                for j in range(4):
                    xbt = self._conv_block_res_batch(br, xbt, ck, f"{p}.conv.{j}", bc)
                concat_t.append(xbt)
                # avg_pool2d_2x2 → conv2d 对角核（per-channel 平均，stride 2）
                # 注意：xbt 同时是 concat_t[-1]（decoder cat 需下载）——**不能
                # tensor_done**（buffer 会被池回收，numpy() 读到 pool 输出）。
                ch = xbt.shape[1]
                w_pool = self._pool_kernel(ch)
                xbt = br.conv2d(xbt, w_pool, None, stride=2, padding=0)
            # intermediate：4 层 × 4 块
            for i in range(4):
                p = f"unet.intermediate.layers.{i}"
                for j in range(4):
                    xbt = self._conv_block_res_batch(br, xbt, ck, f"{p}.conv.{j}", bc)
            br.commit()
            x_out = xbt.numpy()
            concat = [t.numpy() for t in concat_t]
            return x_out, concat
        finally:
            br.release()

    @staticmethod
    def _pool_kernel(ch):
        import numpy as _np

        w = _np.zeros((ch, ch, 2, 2), dtype=_F32)
        for i in range(ch):
            w[i, i, :, :] = 0.25
        return w

    def _conv_block_res_batch(self, br, xbt, ckpt, prefix, bn_cache):
        """ConvBlockRes GPU batch 版：conv2d(常驻融合权重)→ReLU ×2 + 残差。

        conv-BN 融合（标准部署优化）：y = conv(x, w·s) + t（b 原为 None）。
        权重在 __init__ 注册为 GPU 常驻（``rmvpe.*``），此处经 ``buf_w``/
        ``buf_b`` 直接取常驻 buffer——推理零权重上传（此前 batch 2s 上传
        曾达 324MB，权重上传是大头）。
        """
        from runtime import vulkan_weights as _vw  # noqa: PLC0415

        w0f = _vw._weights.get("rmvpe." + prefix + ".conv.0.weight")
        w2f = _vw._weights.get("rmvpe." + prefix + ".conv.3.weight")
        t0a = _vw._weights.get("rmvpe." + prefix + ".conv.0.bias")
        t1a = _vw._weights.get("rmvpe." + prefix + ".conv.3.bias")
        # 残差副本（入口 x 即将被第一个 conv2d 消费）
        rb = br.copy(xbt)
        y = br.conv2d(xbt, None, t0a, stride=1, padding=1,
                      buf_w=w0f)  # conv+BN 融合（w 常驻）
        br.tensor_done(xbt)
        y = br.relu(y)
        y = br.conv2d(y, None, t1a, stride=1, padding=1, buf_w=w2f)
        y = br.relu(y)
        # 残差：shortcut 卷积或恒等（用副本）
        if prefix + ".shortcut.weight" in ckpt:
            ws = _vw._weights.get("rmvpe." + prefix + ".shortcut.weight")
            bs = _vw._weights.get("rmvpe." + prefix + ".shortcut.bias")
            y = br.add_inplace(y, br.conv2d(rb, None, bs, stride=1, padding=0,
                                            buf_w=ws))
        else:
            y = br.add_inplace(y, rb)
        br.tensor_done(rb)
        return y

    @staticmethod
    def _bn_mul_add(br, ybt, scale, shift):
        """BN 推理（GPU）：y = y * scale + shift（scale/shift 每通道 broadcast）。"""
        import numpy as _np

        B, C, H, W = ybt.shape
        sc = _np.broadcast_to(_np.asarray(scale, dtype=_F32).reshape(1, C, 1, 1),
                              (B, C, H, W)).astype(_F32, copy=True)
        sh = _np.broadcast_to(_np.asarray(shift, dtype=_F32).reshape(1, C, 1, 1),
                              (B, C, H, W)).astype(_F32, copy=True)
        br.mul_inplace(ybt, sc)
        br.add_inplace(ybt, sh)
        return ybt

    def mel2hidden(self, mel):
        """``[1, 128, F]`` -> ``[1, F, 360]``（内部把时间维补到 32 的倍数再裁回）。"""
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, n_pad)), mode="constant")
        mel = mel.transpose(0, 2, 1)[:, None, ...]  # [1, 1, F', 128]
        if self._unet_batch_flag:
            # P0-1：encoder+intermediate 一次 commit 纯 GPU；decoder 层内 batch；
            # cnn/fc host
            x_d, concat = self._unet_enc_batch(mel)
            x_d = self._decoder_batch(x_d, concat)
            hidden = self._cnn_fc(x_d)
        else:
            hidden = self._unet(mel)
        return hidden[:, :n_frames, :]

    def _cnn_fc(self, x):
        """UNet 尾部：cnn conv2d + BiGRU + Linear + Sigmoid（host，与 _unet 一致）。"""
        ck = self.W
        wc = ck["cnn.weight"].astype(_F32)
        bc = ck["cnn.bias"].astype(_F32)
        x = _conv2d_dispatch(x, wc, bc, stride=1, padding=1)
        x = x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        w_ih, w_hh, b_ih, b_hh = (ck[k].astype(_F32) for k in (
            "fc.0.gru.weight_ih_l0", "fc.0.gru.weight_hh_l0",
            "fc.0.gru.bias_ih_l0", "fc.0.gru.bias_hh_l0",
        ))
        w_ih_r, w_hh_r, b_ih_r, b_hh_r = (ck[k].astype(_F32) for k in (
            "fc.0.gru.weight_ih_l0_reverse", "fc.0.gru.weight_hh_l0_reverse",
            "fc.0.gru.bias_ih_l0_reverse", "fc.0.gru.bias_hh_l0_reverse",
        ))
        h = _bgru(x[0], w_ih, w_hh, b_ih, b_hh, w_ih_r, w_hh_r, b_ih_r, b_hh_r)
        wf = ck["fc.1.weight"].astype(_F32)
        bf = ck["fc.1.bias"].astype(_F32)
        return nn_ops.sigmoid(nn_ops.linear(h[None, ...], wf, bf))

    # -------------------------------------------------------------- decode
    def decode(self, hidden, thred=0.03):
        """``[F, 360]``（sigmoid salience）-> ``[F]`` Hz 基频。"""
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def to_local_average_cents(self, salience, thred=0.03):
        """逐帧 argmax 局部 9-bin 加权平均 cents；max<=thred 置 0（静音）。"""
        salience = np.asarray(salience, dtype=_F32)  # [F, 360]
        F = salience.shape[0]
        center = np.argmax(salience, axis=1)  # [F]
        salience_pad = np.pad(salience, ((0, 0), (4, 4)))
        center = center + 4
        starts = center - 4
        # 向量化：取 (center-4 .. center+4) 9 bin（注意 pad 后 center+4 = 原 max 位置 +4）
        rows = np.arange(F)[:, None]
        cols = (starts[:, None] + np.arange(9)[None, :])
        todo_salience = salience_pad[rows, cols]  # [F, 9]
        todo_cents = self.cents_mapping[cols]  # [F, 9]
        product_sum = np.sum(todo_salience * todo_cents, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        devided = product_sum / weight_sum
        maxx = np.max(salience_pad, axis=1)  # pad 不影响 max
        devided[maxx <= thred] = 0
        return devided

    # ------------------------------------------------------------- 顶层入口
    def infer_from_audio(self, x: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """``[T]`` float32 16kHz 音频 -> ``[F]`` float32 基频（Hz）。

        推理期间用 threadpoolctl 临时限制 BLAS 单线程（OpenBLAS 多线程对
        im2col 小矩阵乘是负优化，见模块 docstring；numba prange 不受影响）。
        """
        x = np.asarray(x, dtype=_F32)
        if x.ndim != 1:
            raise ValueError(f"infer_from_audio 需要 1D 音频，实际 {x.ndim}D")
        if _THREADPOOL_OK:
            with _threadpool_limits(limits=1, user_api="blas"):
                mel = self._extract_mel(x)  # [1, 128, F]
                hidden = self.mel2hidden(mel)  # [1, F, 360]
                f0 = self.decode(hidden[0], thred=thred)  # [F]
        else:
            mel = self._extract_mel(x)  # [1, 128, F]
            hidden = self.mel2hidden(mel)  # [1, F, 360]
            f0 = self.decode(hidden[0], thred=thred)  # [F]
        return f0


# ---------------------------------------------------------------------------
# 懒加载缓存
# ---------------------------------------------------------------------------

_CACHE: Dict[str, RMVPE] = {}


def load_rmvpe(model_path: Optional[str] = None) -> RMVPE:
    """懒加载 RMVPE（进程内缓存，默认路径 assets/rmvpe/rmvpe.pt）。"""
    if model_path is None:
        import os

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "assets", "rmvpe", "rmvpe.pt",
        )
    model_path = str(model_path)
    if model_path not in _CACHE:
        _CACHE[model_path] = RMVPE(model_path)
    return _CACHE[model_path]