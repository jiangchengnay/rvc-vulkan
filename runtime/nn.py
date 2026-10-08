# -*- coding: utf-8 -*-
"""纯 numpy 神经网络算子库（RVC 去 CUDA 化移植的模型运行时基础）。

替代 RVC 移植所需的 PyTorch 算子（``torch.nn.functional`` 子集 + ``nn`` 模块的
推理语义），供 hubert / rmvpe / vits 模型实现直接调用。**本模块禁止 import
torch**，只允许使用 numpy。

输入约定：
    - 卷积类算子: NCHW / NCT，即 ``[B, C, T]``（1D）与 ``[B, C, H, W]``（2D）；
    - 线性 / 注意力 / 序列类算子: ``[B, T, C]``；
    - 归一化: ``gamma`` / ``beta`` 形状与 PyTorch 一致（逐通道）；
    - 一切对齐 PyTorch CPU 推理语义，具体差异在函数 docstring 中注明。

权重加载自 ``torch_compat.load_pth`` 的 dict[str, numpy.ndarray]，键名与 RVC
checkpoint 完全一致，因此这里所有权重参数形状与 PyTorch 权重形状一一对应。
"""

from __future__ import annotations

import os  # T5fix: RVC_TRAIN_FWD_MATMUL 开关
import numpy as np

# T2.3：可选 numba 快速逐元素内核（同 hubert.gelu_erf 模式）。已安装时
# leaky_relu 提速 ~5-10x（dec 逐级回退路径的大数组 [1,192,oL] np.where
# 分配多个临时数组是 12s pm 的 CPU 单点 ~0.3s）；未安装自动回退 numpy，
# 行为/数值不变。内核与 numpy 版同一公式，逐位一致（每元素独立运算）。
try:
    from numba import njit, prange  # type: ignore[import-not-found]  # noqa: PLC0415

    _NUMBA_OK = True
except Exception:  # pragma: no cover - 无 numba 的环境回退 numpy
    _NUMBA_OK = False

if _NUMBA_OK:

    @njit(parallel=True, fastmath=False, cache=True)
    def _leaky_relu_kernel(x, slope):
        """x: contiguous float32 1D → 同形状 float32；逐元素 y = x if x>=0 else slope*x。"""
        out = np.empty_like(x)
        n = x.size
        for i in prange(n):
            v = x[i]
            out[i] = v if v >= 0.0 else slope * v
        return out

def traced(name):
    """占位装饰器（无 profiling 时零开销直通）。"""
    def _deco(fn):
        return fn
    return _deco


__all__ = [
    # 卷积
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    # 归一化与激活
    "layer_norm",
    "group_norm",
    "batch_norm_infer",
    "softmax",
    "log_softmax",
    "gelu",
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    # 线性与嵌入
    "linear",
    "embedding",
    # 序列
    "gru_cell",
    "gru",
    "multi_head_attention",
    # 工具
    "interpolate_linear",
    "pad_reflect",
    "generate_mask",
    "repeat_interleave",
    "clamp_min",
    "clamp_max",
]


# --------------------------------------------------------------------------
# 卷积
# --------------------------------------------------------------------------
@traced("conv1d/cpu")
def _conv1d_numpy(x, w, b=None, stride=1, padding=0, dilation=1):
    """conv1d 的纯 numpy 实现（无 backend 分派）。

    与公开 ``conv1d`` 的 numpy 语义完全一致：显式零填充 → sliding_window_view
    取连续滑窗 → einsum 矩阵乘 → 加 bias。供 backbone 分派器与后端 numpy
    回退路径复用，避免分派环。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(padding, (tuple, list)):
        pad_l, pad_r = int(padding[0]), int(padding[1])
    else:
        pad_l = pad_r = int(padding)
    stride = int(stride)
    dilation = int(dilation)

    B, C, T = x.shape
    O, _, K = w.shape
    K_dil = (K - 1) * dilation + 1  # dilation 膨胀后的有效核长

    oL = (T + pad_l + pad_r - K_dil) // stride + 1
    if oL <= 0:
        out = np.zeros((B, O, 0), dtype=x.dtype)
        if b is not None:
            pass  # 输出长度为 0，bias 无处加
        return out

    # 权重按 dilation 插零膨胀：[O, C, K] -> [O, C, K_dil]
    if dilation == 1:
        w_dil = w
    else:
        w_dil = np.zeros((O, C, K_dil), dtype=w.dtype)
        w_dil[:, :, ::dilation] = w

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))  # [B, C, T_pad]
    T_pad = x_pad.shape[-1]
    if K_dil <= T_pad:
        # 连续滑窗（步长 1），再按 stride 抽取窗口起点
        win = np.lib.stride_tricks.sliding_window_view(x_pad, K_dil, axis=-1)
        win = win[:, :, ::stride, :]  # [B, C, oL, K_dil]
        out = np.einsum("bctk,ock->bot", win, w_dil, optimize=True)
    else:
        out = np.zeros((B, O, oL), dtype=x.dtype)
    if b is not None:
        out += np.asarray(b, dtype=out.dtype).reshape(1, -1, 1)
    return out


def conv1d(
    x,
    w,
    b=None,
    stride=1,
    padding=0,
    dilation=1,
):
    """1D 卷积，对齐 ``torch.nn.functional.conv1d`` 推理语义。

    参数:
        x: ``[B, C, T]``
        w: ``[O, C, K]``
        b: ``[O]`` 或 None
        stride: int
        padding: int 或 ``(pad_l, pad_r)``（非对称填充按 PyTorch 语义：左填充生效
            于输出首端，右填充生效于末端）
        dilation: int

    输出长度: ``oL = (T + pad_l + pad_r - dilation*(K-1) - 1)//stride + 1``。

    实现：显式零填充 → ``sliding_window_view`` 取连续滑窗（权重先按 dilation
    插零膨胀为核长 ``(K-1)*dilation+1``，再按 stride 抽取窗口起点）→ einsum
    矩阵乘 → 加 bias。

    float32 输入走 ``runtime.backend.conv1d`` 分派（Vulkan 可用时卷积核心在
    GPU 计算，bias 加法保持 numpy，数值与 numpy 版一致，小张量自动回退）；
    其余 dtype 保持纯 numpy。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if x.dtype == np.float32 and w.dtype == np.float32:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        core = backend.conv1d(x, w, None, stride, padding, dilation)
        if b is not None:
            core = core + np.asarray(b, dtype=core.dtype).reshape(1, -1, 1)
        return core
    return _conv1d_numpy(x, w, b, stride, padding, dilation)


def _conv2d_im2col(x_pad, w_dil, stride_h, stride_w):
    """对已填充输入做 im2col + matmul 的 2D 卷积核心（单 batch 或整批）。"""
    B, C, Hp, Wp = x_pad.shape
    O, _, KH_dil, KW_dil = w_dil.shape
    OH = (Hp - KH_dil) // stride_h + 1
    OW = (Wp - KW_dil) // stride_w + 1
    if OH <= 0 or OW <= 0:
        return np.zeros((B, O, max(OH, 0), max(OW, 0)), dtype=x_pad.dtype)

    win = np.lib.stride_tricks.sliding_window_view(
        x_pad, (KH_dil, KW_dil), axis=(-2, -1)
    )  # [B, C, OH', OW', KH_dil, KW_dil]
    win = win[:, :, ::stride_h, ::stride_w, :, :]  # [B, C, OH, OW, KH, KW]
    # im2col: [B, OH*OW, C*KH*KW]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(B, OH * OW, -1)
    w2d = w_dil.reshape(O, -1)  # [O, C*KH*KW]
    out = cols @ w2d.T  # [B, OH*OW, O]
    return out.transpose(0, 2, 1).reshape(B, O, OH, OW)


@traced("conv2d/cpu")
def _conv2d_numpy(x, w, b=None, stride=1, padding=0, dilation=1):
    """conv2d 的纯 numpy 实现（无 backend 分派）。

    与公开 ``conv2d`` 的 numpy 语义完全一致：dilation 插零膨胀权重 -> 显式对称
    填充 -> im2col + matmul -> 加 bias；kernel >5 时按 batch 分块限制峰值内存。
    供 backend 分派器与 numpy 回退路径复用，避免分派环。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = int(stride[0]), int(stride[1])
    else:
        stride_h = stride_w = int(stride)
    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        pad_h = pad_w = int(padding)
    if isinstance(dilation, (tuple, list)):
        dil_h, dil_w = int(dilation[0]), int(dilation[1])
    else:
        dil_h = dil_w = int(dilation)

    B, C, H, W = x.shape
    O, _, KH, KW = w.shape
    KH_dil = (KH - 1) * dil_h + 1
    KW_dil = (KW - 1) * dil_w + 1

    if dil_h != 1 or dil_w != 1:
        w_dil = np.zeros((O, C, KH_dil, KW_dil), dtype=w.dtype)
        w_dil[:, :, ::dil_h, ::dil_w] = w
    else:
        w_dil = w

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    OH = (H + 2 * pad_h - KH_dil) // stride_h + 1
    OW = (W + 2 * pad_w - KW_dil) // stride_w + 1
    if OH <= 0 or OW <= 0:
        out = np.zeros((B, O, max(OH, 0), max(OW, 0)), dtype=x.dtype)
        if b is not None:
            pass
        return out

    # 展开矩阵元素预算：单 batch (OH*OW*C*KH_dil*KW_dil) 超阈值则分块
    big_kernel = KH > 5 or KW > 5
    per_img = int(OH * OW * C * KH_dil * KW_dil)
    budget = 2**26  # 约 6700 万 float32 ≈ 268 MB
    if big_kernel and per_img > budget // max(B, 1):
        parts = []
        chunk = max(1, budget // max(per_img, 1))
        for s in range(0, B, chunk):
            parts.append(_conv2d_im2col(x_pad[s : s + chunk], w_dil, stride_h, stride_w))
        out = np.concatenate(parts, axis=0)
    else:
        out = _conv2d_im2col(x_pad, w_dil, stride_h, stride_w)
    if b is not None:
        out += np.asarray(b, dtype=out.dtype).reshape(1, -1, 1, 1)
    return out


def conv2d(x, w, b=None, stride=1, padding=0, dilation=1):
    """2D 卷积，对齐 ``torch.nn.functional.conv2d`` 推理语义。

    参数:
        x: ``[B, C, H, W]``
        w: ``[O, C, KH, KW]``
        b: ``[O]`` 或 None
        stride: int 或 ``(sh, sw)``
        padding: int 或 ``(ph, pw)``（对称填充）
        dilation: int 或 ``(dh, dw)``

    kernel（膨胀前）≤5 时整批 im2col 直接展开；kernel 更大时按 batch 分块
    展开以限制峰值内存（每块最大约 5120 万 float32 元素）。

    float32 输入（x/w）且 ``dilation==1``（引擎 GPU kernel 暂不支持 dilation）
    时经 ``runtime.backend.conv2d`` 分派（Vulkan 可用时卷积核心在 GPU 计算，
    小张量自动回退 numpy，数值一致）；其余情形保持纯 numpy。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(dilation, (tuple, list)):
        dil_h, dil_w = int(dilation[0]), int(dilation[1])
    else:
        dil_h = dil_w = int(dilation)
    if x.dtype == np.float32 and w.dtype == np.float32 and dil_h == 1 and dil_w == 1:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        try:
            return backend.conv2d(x, w, b, stride, padding)
        except ValueError:
            pass  # 形状/参数不匹配 -> 回退 numpy
    return _conv2d_numpy(x, w, b, stride, padding, dilation)


@traced("conv_t1d/cpu")
def _conv_transpose1d_numpy(
    x,
    w,
    b=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
):
    """conv_transpose1d 的纯 numpy 实现（无 backend 分派）。

    与公开 ``conv_transpose1d`` 的 numpy 语义完全一致：输出逐点累加 —— 对每个
    核位置 k，输入位置 t 的能量 ``x[b, :, t] @ w[:, :, k]`` 累加到输出
    ``pos = t*stride - padding + k*dilation``。供 backend 分派器与 numpy 回退
    路径复用，避免分派环。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    stride = int(stride)
    padding = int(padding)
    output_padding = int(output_padding)
    dilation = int(dilation)

    B, C_in, T = x.shape
    C_out = w.shape[1]
    K = w.shape[2]
    oL = (T - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    if oL <= 0:
        out = np.zeros((B, C_out, 0), dtype=x.dtype)
        if b is not None:
            pass
        return out

    out = np.zeros((B, C_out, oL), dtype=result_dtype(x, w))
    for k in range(K):
        pos = np.arange(T) * stride - padding + k * dilation  # [T]
        valid = (pos >= 0) & (pos < oL)
        if not valid.any():
            continue
        tv = np.nonzero(valid)[0]
        # x[:, :, tv]: [B, C_in, T_v]   w[:, :, k]: [C_in, C_out]
        # out[b, co, pos] += sum_c x[b,c,t] * w[c,co,k]
        if os.environ.get("RVC_TRAIN_FWD_MATMUL", "1") != "0":
            # T5fix: einsum → matmul（收缩 c，f64 等价 ~1e-14；开关同 RVC_TRAIN_FWD_MATMUL）
            contrib = np.matmul(x[:, :, tv].transpose(0, 2, 1),
                                w[:, :, k]).transpose(0, 2, 1)
        else:
            contrib = np.einsum("bct,co->bot", x[:, :, tv],
                                w[:, :, k], optimize=True)
        out[:, :, pos[tv]] += contrib
    if b is not None:
        out += np.asarray(b, dtype=out.dtype).reshape(1, -1, 1)
    return out


def conv_transpose1d(
    x,
    w,
    b=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
):
    """1D 转置卷积，对齐 ``torch.nn.functional.conv_transpose1d`` 推理语义。

    参数:
        x: ``[B, C_in, T]``
        w: ``[C_in, C_out, K]``（与 PyTorch 权重形状一致：``[in, out, kernel]``）
        b: ``[C_out]`` 或 None
        stride / padding / output_padding / dilation: int

    输出长度: ``oL = (T-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1``。

    实现：输出逐点累加 —— 对每个核位置 k，输入位置 t 的能量
    ``x[b, :, t] @ w[:, :, k]`` 累加到输出 ``pos = t*stride - padding + k*dilation``。
    同一 k 内各 t 的 pos 严格递增，故可安全地直接 ``+=``（无需 np.add.at）。
    复杂度 O(K * T * C_in * C_out)，向量化到批量维。

    float32 输入（x/w）且维度匹配时经 ``runtime.backend.conv_transpose1d`` 分派
    （Vulkan 可用时 vits dec 上采样在 GPU 计算；小张量自动回退 numpy，数值一致）；
    其余 dtype / 形状不匹配保持纯 numpy。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if x.dtype == np.float32 and w.dtype == np.float32:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        try:
            return backend.conv_transpose1d(
                x, w, b, stride, padding, output_padding, dilation
            )
        except ValueError:
            pass  # 形状/参数不匹配 -> 回退 numpy（保持默认行为）
    return _conv_transpose1d_numpy(x, w, b, stride, padding, output_padding, dilation)


# --------------------------------------------------------------------------
# 归一化
# --------------------------------------------------------------------------
def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm（对最后一维），对齐 ``torch.nn.functional.layer_norm``。

    x: ``[..., C]``；gamma/beta: ``[C]``。方差为 biased（ddof=0）。

    float32 输入（含 gamma/beta）走 ``runtime.backend.layer_norm`` 分派
    （Vulkan 可用时整行归一在 GPU 计算，数值一致）；其余 dtype 保持纯 numpy。
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.dtype == np.float32 and gamma.dtype == np.float32 and beta.dtype == np.float32:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        return backend.layer_norm(x, gamma, beta, eps)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # ddof=0，与 PyTorch biased 一致
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * gamma + beta


def group_norm(x, gamma, beta, num_groups, eps=1e-5):
    """GroupNorm，对齐 ``torch.nn.functional.group_norm``。

    x: ``[B, C, ...]``；gamma/beta: ``[C]``。通道按 ``num_groups`` 分组，每组内
    对 ``C//num_groups * 全部空间维`` 求 mean/var（biased）。要求 ``C % num_groups == 0``。
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    B, C = x.shape[0], x.shape[1]
    if C % num_groups != 0:
        raise ValueError(f"group_norm: C={C} 不能被 num_groups={num_groups} 整除")
    G = int(num_groups)
    spatial = x.shape[2:]
    xr = x.reshape(B, G, C // G, *spatial)
    axes = tuple(range(2, xr.ndim))
    mean = xr.mean(axis=axes, keepdims=True)
    var = xr.var(axis=axes, keepdims=True)
    xn = (xr - mean) / np.sqrt(var + eps)
    g = gamma.reshape(1, G, C // G, *([1] * len(spatial)))
    bt = beta.reshape(1, G, C // G, *([1] * len(spatial)))
    return (xn * g + bt).reshape(x.shape)


def batch_norm_infer(x, gamma, beta, running_mean, running_var, eps=1e-5):
    """BatchNorm2d/1d 推理模式，对齐 PyTorch ``training=False`` 路径。

    x: ``[B, C, ...]``；gamma/beta/running_mean/running_var: ``[C]``。
    公式: ``y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta``。
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    rm = np.asarray(running_mean)
    rv = np.asarray(running_var)
    shape = (1, -1) + (1,) * (x.ndim - 2)
    scale = gamma / np.sqrt(rv + eps)
    return (x - rm.reshape(shape)) * scale.reshape(shape) + beta.reshape(shape)


# --------------------------------------------------------------------------
# 激活
# --------------------------------------------------------------------------
def softmax(x, axis=-1):
    """数值稳定的 softmax。

    float32 且 ``axis == -1``（默认）时走 ``runtime.backend.softmax`` 分派
    （Vulkan 可用时逐行在 GPU 计算，exp(x-max) 数值稳定，与 numpy 一致；
    小张量自动回退 numpy）；其余 axis / dtype 保持纯 numpy。
    """
    x = np.asarray(x)
    if x.dtype == np.float32 and axis == -1:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        return backend.softmax(x)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """数值稳定的 log_softmax（``log(softmax(x))``）。"""
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return x - lse


def gelu(x):
    """GELU（tanh 近似），对齐 ``torch.nn.functional.gelu(..., approximate='tanh')``。

    ``0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))``
    """
    x = np.asarray(x)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def relu(x):
    """ReLU（``maximum(x, 0)``）。

    float32 输入经 ``runtime.backend.relu`` 分派（Vulkan 可用时走 GPU，
    引擎内 in-place、对外返回新数组不动输入）；其余 dtype 保持纯 numpy。
    """
    x = np.asarray(x)
    if x.dtype == np.float32:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        return backend.relu(x)
    return np.maximum(x, 0.0)


def leaky_relu(x, negative_slope=0.1):
    """LeakyReLU，对齐 ``F.leaky_relu``（默认 negative_slope=0.01 由调用方决定）。

    T2.3：float32 且 numba 可用时走并行内核（逐位一致，仅更快）；否则
    原 numpy where（分配 3 个临时数组，dec 大数组路径慢）。
    """
    x = np.asarray(x)
    if _NUMBA_OK and x.dtype == np.float32:
        flat = np.ascontiguousarray(x).reshape(-1)
        return _leaky_relu_kernel(flat, np.float32(negative_slope)).reshape(x.shape)
    return np.where(x >= 0, x, negative_slope * x)


def sigmoid(x):
    """Sigmoid（数值稳定；x 远小于 0 时直接取 ``exp(x)/(1+exp(x))``）。"""
    x = np.asarray(x)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def tanh(x):
    """tanh。"""
    return np.tanh(np.asarray(x))


# --------------------------------------------------------------------------
# 线性与嵌入
# --------------------------------------------------------------------------
def linear(x, w, b=None):
    """线性层，对齐 ``torch.nn.functional.linear``。

    x: ``[..., D]``，w: ``[O, D]``，b: ``[O]`` 或 None。输出 ``x @ w.T + b``。

    当 x/w 均为 float32 时，矩阵乘部分经 ``runtime.backend.matmul`` 分派
    （Vulkan 可用时走 GPU，小张量自动回退 numpy，数值与 ``x @ w.T`` 一致），
    bias 加法保持 numpy；其余 dtype 保持纯 numpy 路径。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if x.dtype == np.float32 and w.dtype == np.float32:
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        D = x.shape[-1]
        x2 = x.reshape(-1, D)  # [..., D] → [n, D]（1D 也变为 [1, D]）
        out = backend.matmul(x2, w.T)  # [n, O]
        out = out.reshape(x.shape[:-1] + (w.shape[0],))
    else:
        out = x @ w.T
    if b is not None:
        out = out + np.asarray(b).reshape(1, -1)
    return out


@traced("embedding/cpu")
def _embedding_numpy(ids, table):
    """embedding 的纯 numpy 实现（无 backend 分派）。

    与公开 ``embedding`` 的 numpy 语义完全一致：支持负索引（Python 列表语义：
    ``-1`` 取最后一行；PyTorch 的 embedding 不支持负索引，此处为宽松增强）。
    供 backend 分派器与 numpy 回退路径复用，避免分派环。
    """
    ids = np.asarray(ids)
    table = np.asarray(table)
    V = table.shape[0]
    ids2 = np.where(ids < 0, ids + V, ids).astype(np.int64)
    return np.take(table, ids2, axis=0)


def embedding(ids, table):
    """查表嵌入，对齐 ``torch.nn.functional.embedding``。

    ids: int ``[B, T]``（任意形状均可），table: ``[V, E]``。返回 ``ids.shape + [E]``。
    支持负索引（Python 列表语义：``-1`` 取最后一行；PyTorch 的 embedding 不支持
    负索引，此处为宽松增强，推理输入一般不会用到）。

    table 为 float32 且 ids 为整数时经 ``runtime.backend.embedding`` 分派
    （Vulkan 可用时 gather 在 GPU 计算；小张量 / 越界 ids 自动回退 numpy，
    数值与 numpy 版一致）；其余 dtype 保持纯 numpy。
    """
    ids = np.asarray(ids)
    table = np.asarray(table)
    if table.dtype == np.float32 and np.issubdtype(ids.dtype, np.integer):
        from runtime import backend  # noqa: PLC0415  # 惰性导入避免包初始化环

        try:
            return backend.embedding(ids, table)
        except (ValueError, TypeError):
            pass  # 形状/类型不匹配 -> 回退 numpy
    return _embedding_numpy(ids, table)


# --------------------------------------------------------------------------
# 序列
# --------------------------------------------------------------------------
def _gru_step(x_t, h, w_ih, w_hh, b_ih, b_hh, hidden):
    """单时间步 GRU 前向。x_t: [B, in]，h: [B, H]。返回新 h。"""
    gx = x_t @ w_ih.T + b_ih  # [B, 3H]
    gh = h @ w_hh.T + b_hh  # [B, 3H]
    # PyTorch 门顺序：r（reset）, z（update）, n（new）
    r = sigmoid(gx[:, 0:hidden] + gh[:, 0:hidden])
    z = sigmoid(gx[:, hidden : 2 * hidden] + gh[:, hidden : 2 * hidden])
    n = tanh(gx[:, 2 * hidden :] + r * gh[:, 2 * hidden :])
    return (1.0 - z) * n + z * h


def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh):
    """单步 GRU cell，对齐 PyTorch ``GRUCell`` 推理语义。

    x: ``[B, in]``，h: ``[B, H]``（初始常为 0）；
    w_ih: ``[3H, in]``，w_hh: ``[3H, H]``，b_ih/b_hh: ``[3H]``。
    门顺序为 PyTorch 的 r (reset), z (update), n (new)，即权重行序
    ``[0:H]=r``、``[H:2H]=z``、``[2H:3H]=n``。

    公式:
        r = sigmoid(x@W_ir^T + b_ir + h@W_hr^T + b_hr)
        z = sigmoid(x@W_iz^T + b_iz + h@W_hz^T + b_hz)
        n = tanh(x@W_in^T + b_in + r * (h@W_hn^T + b_hn))
        h' = (1 - z) * n + z * h
    """
    x = np.asarray(x)
    h = np.asarray(h)
    w_ih = np.asarray(w_ih)
    w_hh = np.asarray(w_hh)
    b_ih = np.asarray(b_ih, dtype=result_dtype(x, w_ih)) if b_ih is not None else np.zeros(w_ih.shape[0], dtype=result_dtype(x, w_ih))
    b_hh = np.asarray(b_hh, dtype=result_dtype(x, w_ih)) if b_hh is not None else np.zeros(w_hh.shape[0], dtype=result_dtype(x, w_ih))
    hidden = h.shape[-1]
    return _gru_step(x, h, w_ih, w_hh, b_ih, b_hh, hidden)


def gru(x, w_ih, w_hh, b_ih, b_hh, bidirectional=False):
    """GRU 时间序列前向，对齐 PyTorch ``nn.GRU``（batch_first=np 约定 [B,T,C]）。

    x: ``[B, T, in]``；权重形状同 ``gru_cell``。输出 ``[B, T, H]``（双向
    ``[B, T, 2H]``，前向在前、反向后拼）。

    注意: 双向时 PyTorch 有独立的反向权重（``weight_ih_l0_reverse`` 等），本函数
    按任务签名复用同一组权重完成时间反向计算；若调用方需要独立反向权重，请自行
    把反向权重拼进 ``w_ih/w_hh`` 的第 2 个 batch 维度之前预处理后调用两次再拼接。
    """
    x = np.asarray(x)
    w_ih = np.asarray(w_ih)
    w_hh = np.asarray(w_hh)
    rt = result_dtype(x, w_ih)
    if b_ih is None:
        b_ih = np.zeros(w_ih.shape[0], dtype=rt)
    if b_hh is None:
        b_hh = np.zeros(w_hh.shape[0], dtype=rt)
    b_ih = np.asarray(b_ih, dtype=rt)
    b_hh = np.asarray(b_hh, dtype=rt)

    B, T, _ = x.shape
    hidden = w_hh.shape[0] // 3

    def _run(xs):  # xs: [B, T, in]，正向循环
        h = np.zeros((B, hidden), dtype=rt)
        outs = np.empty((B, T, hidden), dtype=rt)
        for t in range(T):
            h = _gru_step(xs[:, t, :], h, w_ih, w_hh, b_ih, b_hh, hidden)
            outs[:, t, :] = h
        return outs

    fwd = _run(x)
    if not bidirectional:
        return fwd
    rev = _run(x[:, ::-1, :])[:, ::-1, :]  # 时间反向计算后翻回原序
    return np.concatenate([fwd, rev], axis=-1)


def multi_head_attention(
    q,
    k,
    v,
    w_q,
    w_k,
    w_v,
    w_o,
    b_q=None,
    b_k=None,
    b_v=None,
    b_o=None,
    num_heads=12,
    mask=None,
    scale=None,
):
    """多头自注意力（基本版），对齐 hubert 的 MHA 推理语义。

    q/k/v: ``[B, T, C]``（hubert 自注意力时 T 同长）；w_*: ``[C, C]``（Linear 权重
    ``[out, in]``），b_*: ``[C]``。输出 ``[B, T, C]``。

    - 无相对位置编码（hubert 的相对位置由调用方另行处理）；
    - ``scale`` 默认 ``1/sqrt(head_dim)``，head_dim = C // num_heads；
    - ``mask`` 为 ``[B, 1, 1, Tk]``、``[1, 1, Tq, Tk]`` 或等价可广播形状，
      ``mask == 0`` 处的分数置为 -1e4（数值上等价于屏蔽后 softmax 近似为 0，避免
      使用 -inf 产生 NaN 风险）。
    """
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    B, Tq, C = q.shape
    _, Tk, _ = k.shape
    H = int(num_heads)
    head_dim = C // H

    q_p = linear(q, w_q, b_q)  # [B, Tq, C]
    k_p = linear(k, w_k, b_k)  # [B, Tk, C]
    v_p = linear(v, w_v, b_v)  # [B, Tk, C]

    qh = q_p.reshape(B, Tq, H, head_dim).transpose(0, 2, 1, 3)  # [B,H,Tq,D]
    kh = k_p.reshape(B, Tk, H, head_dim).transpose(0, 2, 1, 3)  # [B,H,Tk,D]
    vh = v_p.reshape(B, Tk, H, head_dim).transpose(0, 2, 1, 3)  # [B,H,Tk,D]

    scale = scale if scale is not None else head_dim ** -0.5
    scores = np.einsum("bhtd, bhsd -> bhts", qh, kh, optimize=True) * scale
    if mask is not None:
        m = np.asarray(mask)
        while m.ndim > scores.ndim and m.shape[0] == 1:  # 压缩多余的前导 1 维
            m = m[0]
        scores = np.where(m != 0, scores, -1e4)

    attn = softmax(scores, axis=-1)  # [B,H,Tq,Tk]
    ctx = np.einsum("bhts, bhsd -> bhtd", attn, vh, optimize=True)  # [B,H,Tq,D]
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, Tq, C)
    return linear(ctx, w_o, b_o)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def interpolate_linear(x, scale_factor=2, axis=-1):
    """沿指定轴线性插值上采样，对齐 ``F.interpolate(mode='linear', align_corners=False)``。

    x: ``[B, C, T]``（默认最后一维）。输出映射 ``src = (i + 0.5) / scale - 0.5``，
    clamp 到 ``[0, L-1]`` 后线性插值。
    """
    x = np.asarray(x)
    L_in = x.shape[axis]
    L_out = int(L_in * scale_factor)
    if L_out == L_in:
        return x.copy()

    src = (np.arange(L_out, dtype=np.float64) + 0.5) / float(scale_factor) - 0.5
    src = np.clip(src, 0.0, L_in - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, L_in - 1)
    frac = (src - lo).astype(x.dtype) if x.dtype != np.float64 else src - lo

    move = axis != -1
    if move:
        x = np.moveaxis(x, axis, -1)
    out = x[..., lo] * (1.0 - frac) + x[..., hi] * frac
    if move:
        out = np.moveaxis(out, -1, axis)
    return out


def pad_reflect(x, pad_l, pad_r, axis=-1):
    """反射填充，对齐 ``F.pad(mode='reflect')``（仅支持单轴，边不做副本）。

    与 numpy ``mode='reflect'`` 语义一致（如 ``[1,2,3]`` pad 2 → ``[3,2,1,2,3,2,1]``）。
    """
    x = np.asarray(x)
    pad_l = int(pad_l)
    pad_r = int(pad_r)
    if pad_l == 0 and pad_r == 0:
        return x.copy()
    shape = [(0, 0)] * x.ndim
    shape[axis] = (pad_l, pad_r)
    return np.pad(x, shape, mode="reflect")


def generate_mask(length, total=None):
    """生成 seq_mask，供 x_mask 使用。

    返回 ``[1, 1, T]`` float32：``total`` 为 None 时 T=length、全部为 1；
    否则 T=total、前 ``length`` 位为 1、其余为 0（配合后续 padding 到 total）。
    """
    length = int(length)
    if total is None:
        total = length
    total = int(total)
    mask = np.zeros((1, 1, total), dtype=np.float32)
    if length > 0:
        mask[..., : min(length, total)] = 1.0
    return mask


def repeat_interleave(x, n, axis=None):
    """对齐 ``torch.repeat_interleave``：axis 为 None 时先展平再重复。"""
    x = np.asarray(x)
    if axis is None:
        return np.repeat(x.ravel(), n)
    return np.repeat(x, n, axis=axis)


def clamp_min(x, min_value):
    """逐元素 clamp 下界，等价 ``torch.clamp_min``。"""
    return np.maximum(np.asarray(x), min_value)


def clamp_max(x, max_value):
    """逐元素 clamp 上界，等价 ``torch.clamp_max``。"""
    return np.minimum(np.asarray(x), max_value)


def result_dtype(*arrays):
    """按 numpy 提升规则求结果 dtype（float 输入优先 float32/float64）。"""
    dt = np.result_type(*[np.asarray(a).dtype for a in arrays])
    return dt