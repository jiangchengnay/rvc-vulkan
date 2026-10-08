# -*- coding: utf-8 -*-
"""纯 numpy 反向传播算子集（RVC 训练侧核心，T44）。

为 ``runtime/models/vits_train.py`` 的"记录式"训练前向提供最小但正确的
逐算子反向传播函数（输入张量 require_grad 语义：每个 bp 函数接收前向输入与
上游梯度，返回各输入的梯度）。

约定：
    - 所有 numpy 数组布局与 ``runtime/nn.py`` 前向完全一致：
        卷积 ``[B, C, T]`` / ``[B, C, H, W]``，线性 ``[..., D]`` 与 w ``[O, D]``，
        转置卷积 ``x [B, C_in, T]`` / ``w [C_in, C_out, K]``；
    - bp 函数命名 ``<op>_backward``；参数顺序与对应的 forward 一致，
      ``grad_out`` 是上游梯度（与前向输出同形状）；
    - 返回值为梯度元组，顺序与（可导）前向参数顺序一致；bias 为 None 时返回的
      grad_b 亦为 None；
    - 内部按 float64 精确计算，输出 cast 回输入浮点 dtype（float32 训练时
      精度亦足够，见 tests 与数值梯度对照 <1e-4）。

每个算子附 ``runtime/tests_nn_backward.py`` 的数值梯度（central difference
eps=1e-5）对照，相对误差 <1e-4。**本模块禁止 import torch**。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "linear_backward",
    "conv1d_backward",
    "conv2d_backward",
    "conv_transpose1d_backward",
    "layer_norm_backward",
    "softmax_backward",
    "relu_backward",
    "leaky_relu_backward",
    "gelu_backward",
    "sigmoid_backward",
    "tanh_backward",
    "embedding_backward",
    "interpolate_linear_backward",
    "clamp_min_backward",
    "mul_backward",
    "add_backward",
    "div_backward",
    "exp_backward",
    "log_backward",
    "sqrt_backward",
    "pow_backward",
    "neg_backward",
    "einsum_backward",
    "pad_backward",
    "slice_backward",
    "flip_backward",
    "repeat_interleave_backward",
    "mel_spectrogram_backward",
    "numerical_grad",
]


from functools import lru_cache as _lru_cache  # noqa: E402


@_lru_cache(maxsize=1024)
def einsum_optimize_path(expr, shapes):
    """缓存 ``np.einsum_path`` 结果（键=表达式+形状元组；T2 性能优化）。

    einsum_path 对 (expr, shapes) 是确定性算法，缓存后路径与首次完全一致，
    因此 ``np.einsum(..., optimize=path)`` 结果逐位等于 ``optimize=True``
    （零回归，maxdiff=0，实测验证）。消除每次 einsum 调用重算路径规划的
    开销（调研：einsum_path 184,500 次共 7.7s/3 步）。

    注：numpy2.4.6 的 einsum_path 不接受 shape 元组作操作数（报 subscripts
    数量错），且训练大张量不能分配真实数组——用 as_strided 零内存伪造数组
    （einsum_path 只读 shape/ndim，不读数据；0 字节共享 buffer 安全）。
    """
    def _dummy(shape):
        return np.lib.stride_tricks.as_strided(
            np.zeros(0), shape=shape, strides=(0,) * len(shape))

    path, _ = np.einsum_path(expr, *[_dummy(s) for s in shapes], optimize=True)
    return path


def _fdtype(*arrays):
    """浮点 dtype 提升。"""
    dts = [np.asarray(a).dtype for a in arrays]
    fd = [d for d in dts if np.issubdtype(d, np.floating)]
    return np.result_type(*fd) if fd else np.float32


def _f64(x):
    return None if x is None else np.asarray(x, dtype=np.float64)


def _parse_padding(padding):
    if isinstance(padding, (tuple, list)):
        return int(padding[0]), int(padding[1])
    p = int(padding)
    return p, p


# ---------------------------------------------------------------------------
# 线性
# ---------------------------------------------------------------------------
def linear_backward(x, w, grad_out, b=None):
    """``y = x @ w.T + b`` 的反向。返回 (grad_x, grad_w, grad_b)。

    x: ``[..., D]``，w: ``[O, D]``。
    """
    x, go = np.asarray(x), np.asarray(grad_out)
    x64, w64, go64 = _f64(x), _f64(w), _f64(go)

    gx = go64 @ w64  # [..., D]
    x2 = x64.reshape(-1, x64.shape[-1])
    g2 = go64.reshape(-1, w64.shape[0])
    gw = (x2.T @ g2).T  # [O, D]
    gb = g2.sum(axis=0) if b is not None else None

    dt = _fdtype(x, w, go)
    return (gx.astype(dt), gw.astype(dt),
            None if gb is None else gb.astype(dt))


# ---------------------------------------------------------------------------
# 卷积 1D
# ---------------------------------------------------------------------------
def conv1d_backward(x, w, grad_out, stride=1, padding=0, dilation=1, b=None):
    """``conv1d(x, w, b, stride, padding, dilation)`` 的反向。

    返回 (grad_x, grad_w, grad_b)。padding 支持 ``(pad_l, pad_r)`` 非对称。
    x: ``[B, C, T]``，w: ``[O, C, K]``。
    """
    x, go = np.asarray(x), np.asarray(grad_out)
    x64, w64, go64 = _f64(x), _f64(w), _f64(go)

    pad_l, pad_r = _parse_padding(padding)
    stride, dilation = int(stride), int(dilation)

    B, C, T = x64.shape
    O, _, K = w64.shape
    K_dil = (K - 1) * dilation + 1
    oL = go64.shape[2]

    if dilation == 1:
        w_dil = w64
    else:
        w_dil = np.zeros((O, C, K_dil), dtype=np.float64)
        w_dil[:, :, ::dilation] = w64

    x_pad = np.pad(x64, ((0, 0), (0, 0), (pad_l, pad_r)))
    T_pad = x_pad.shape[2]

    # grad_w：im2col 与 grad_out 点积 -> [O, C, K_dil]
    win = np.lib.stride_tricks.sliding_window_view(x_pad, K_dil, axis=-1)
    win = win[:, :, ::stride, :]  # [B, C, oL, K_dil]
    gw = np.einsum("bctk,bot->ock", win, go64, optimize=True)
    if dilation == 1:
        gw = gw.reshape(O, C, K)
    else:
        gw2 = np.zeros((O, C, K), dtype=np.float64)
        gw2[:, :, :] = gw[:, :, ::dilation]
        gw = gw2

    # grad_x：每个核位置 kk 的输出位置 t 落在 x_pad 的 q = t*stride + kk
    gxp = np.zeros_like(x_pad)
    for kk in range(K_dil):
        tv = np.arange(oL)
        q = tv * stride + kk
        m = q < T_pad
        if not m.any():
            continue
        contrib = np.einsum("bot,oc->bct", go64[:, :, tv[m]], w_dil[:, :, kk],
                            optimize=True)
        gxp[:, :, q[m]] += contrib
    gx = gxp[:, :, pad_l : T_pad - pad_r]

    gb = go64.sum(axis=(0, 2)) if b is not None else None
    dt = _fdtype(x, w, go)
    return (gx.astype(dt), gw.astype(dt),
            None if gb is None else gb.astype(dt))


# ---------------------------------------------------------------------------
# 卷积 2D（判别器 DiscriminatorP 用）
# ---------------------------------------------------------------------------
def _im2col2d(x_pad, KH_dil, KW_dil, sh, sw):
    win = np.lib.stride_tricks.sliding_window_view(
        x_pad, (KH_dil, KW_dil), axis=(-2, -1)
    )  # [B, C, OH', OW', KH, KW]
    win = win[:, :, ::sh, ::sw, :, :]
    return win.transpose(0, 2, 3, 1, 4, 5)  # [B, OH, OW, C, KH, KW]


def conv2d_backward(x, w, grad_out, stride=1, padding=0, dilation=1, b=None):
    """``conv2d(x, w, b, stride, padding, dilation)`` 的反向。

    返回 (grad_x, grad_w, grad_b)。stride / padding / dilation 支持
    int 或 (h, w) 元组（对称填充）。x: ``[B, C, H, W]``，w: ``[O, C, KH, KW]``。
    """
    x, go = np.asarray(x), np.asarray(grad_out)
    x64, w64, go64 = _f64(x), _f64(w), _f64(go)

    if isinstance(stride, (tuple, list)):
        sh, sw = int(stride[0]), int(stride[1])
    else:
        sh = sw = int(stride)
    if isinstance(padding, (tuple, list)):
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if isinstance(dilation, (tuple, list)):
        dh, dw = int(dilation[0]), int(dilation[1])
    else:
        dh = dw = int(dilation)

    B, C, H, W = x64.shape
    O, _, KH, KW = w64.shape
    KH_dil = (KH - 1) * dh + 1
    KW_dil = (KW - 1) * dw + 1

    if dh == 1 and dw == 1:
        w_dil = w64
    else:
        w_dil = np.zeros((O, C, KH_dil, KW_dil), dtype=np.float64)
        w_dil[:, :, ::dh, ::dw] = w64

    x_pad = np.pad(x64, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    Hp, Wp = x_pad.shape[2], x_pad.shape[3]
    OH, OW = go64.shape[2], go64.shape[3]

    cols = _im2col2d(x_pad, KH_dil, KW_dil, sh, sw)  # [B,OH,OW,C,KH,KW]
    gw = np.einsum("bhwcxy,bohw->ocxy", cols, go64, optimize=True)
    if dh == 1 and dw == 1:
        gw = gw.reshape(O, C, KH, KW)
    else:
        gw2 = np.zeros((O, C, KH, KW), dtype=np.float64)
        gw2[:, :, :, :] = gw[:, :, ::dh, ::dw]
        gw = gw2

    gxp = np.zeros_like(x_pad)
    for khh in range(KH_dil):
        for kww in range(KW_dil):
            qh = np.arange(OH) * sh + khh
            qw = np.arange(OW) * sw + kww
            mh = qh < Hp
            mw = qw < Wp
            if not (mh.any() and mw.any()):
                continue
            ghv = np.nonzero(mh)[0]
            gwv = np.nonzero(mw)[0]
            go_sel = go64[:, :, ghv][:, :, :, gwv]  # [B, O, n1, n2]
            contrib = np.einsum("bohw,oc->bchw", go_sel, w_dil[:, :, khh, kww],
                                optimize=True)  # [B, C, n1, n2]
            tgt = np.ix_(np.arange(B), np.arange(C), qh[ghv], qw[gwv])
            np.add.at(gxp, tgt, contrib)
    gx = gxp[:, :, ph : Hp - ph, pw : Wp - pw]

    gb = go64.sum(axis=(0, 2, 3)) if b is not None else None
    dt = _fdtype(x, w, go)
    return (gx.astype(dt), gw.astype(dt),
            None if gb is None else gb.astype(dt))


# ---------------------------------------------------------------------------
# 转置卷积 1D
# ---------------------------------------------------------------------------
def conv_transpose1d_backward(x, w, grad_out, stride=1, padding=0,
                              output_padding=0, dilation=1, b=None):
    """``conv_transpose1d(x, w, b, stride, padding, output_padding, dilation)``
    的反向（= 前向 conv1d 的交换）。

    返回 (grad_x, grad_w, grad_b)。x: ``[B, C_in, T]``，w: ``[C_in, C_out, K]``。
    """
    x, go = np.asarray(x), np.asarray(grad_out)
    x64, w64, go64 = _f64(x), _f64(w), _f64(go)

    stride, padding, output_padding, dilation = (
        int(stride), int(padding), int(output_padding), int(dilation))

    B, C_in, T = x64.shape
    C_out = w64.shape[1]
    K = w64.shape[2]
    oL = go64.shape[2]

    gx = np.zeros_like(x64)
    gw = np.zeros_like(w64)
    for k in range(K):
        pos = np.arange(T) * stride - padding + k * dilation  # [T]
        valid = (pos >= 0) & (pos < oL)
        if not valid.any():
            continue
        tv = np.nonzero(valid)[0]
        g = go64[:, :, pos[tv]]  # [B, C_out, T_v]
        gx[:, :, tv] += np.einsum("bot,co->bct", g, w64[:, :, k], optimize=True)
        gw[:, :, k] += np.einsum("bot,bct->co", g, x64[:, :, tv], optimize=True)

    gb = go64.sum(axis=(0, 2)) if b is not None else None
    dt = _fdtype(x, w, go)
    return (gx.astype(dt), gw.astype(dt),
            None if gb is None else gb.astype(dt))


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def layer_norm_backward(x, gamma, beta, grad_out, eps=1e-5):
    """LayerNorm（最后一维，biased 方差）反向。

    返回 (grad_x, grad_gamma, grad_beta)。x/grad_out: ``[..., C]``。
    """
    x, go = np.asarray(x), np.asarray(grad_out)
    x64 = _f64(x)
    g64 = _f64(go) * _f64(gamma)

    mean = x64.mean(axis=-1, keepdims=True)
    var = x64.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x64 - mean) * inv

    gx = inv * (
        g64
        - g64.mean(axis=-1, keepdims=True)
        - xhat * (g64 * xhat).mean(axis=-1, keepdims=True)
    )
    axes = tuple(range(g64.ndim - 1))
    gg = (_f64(go) * xhat).sum(axis=axes)
    gb = _f64(go).sum(axis=axes)

    dt = _fdtype(x, go, gamma, beta)
    return gx.astype(dt), gg.astype(dt), gb.astype(dt)


# ---------------------------------------------------------------------------
# 激活
# ---------------------------------------------------------------------------
def softmax_backward(y, grad_out):
    """输入为 softmax 输出 y。返回 grad_x。"""
    y, go = np.asarray(y), np.asarray(grad_out)
    y64, go64 = _f64(y), _f64(go)
    s = (go64 * y64).sum(axis=-1, keepdims=True)
    return (y64 * (go64 - s)).astype(_fdtype(y, go))


def relu_backward(x, grad_out):
    x, go = np.asarray(x), np.asarray(grad_out)
    return np.where(x > 0, go, np.zeros_like(go)).astype(_fdtype(x, go))


def leaky_relu_backward(x, grad_out, slope=0.1):
    x, go = np.asarray(x), np.asarray(grad_out)
    slope = float(slope)
    return np.where(x >= 0, go, slope * go).astype(_fdtype(x, go))


def gelu_backward(x, grad_out):
    """GELU tanh 近似导（``runtime.nn.gelu`` 的解析导数）。"""
    x, go = np.asarray(x), np.asarray(grad_out)
    x64, go64 = _f64(x), _f64(go)
    c = np.sqrt(2.0 / np.pi)
    t = np.tanh(c * (x64 + 0.044715 * x64 ** 3))
    d = 0.5 * (1.0 + t) + 0.5 * x64 * (1.0 - t * t) * c * (
        1.0 + 3.0 * 0.044715 * x64 * x64
    )
    return (go64 * d).astype(_fdtype(x, go))


def sigmoid_backward(y, grad_out):
    """输入为 sigmoid 输出 y。"""
    y, go = np.asarray(y), np.asarray(grad_out)
    y64, go64 = _f64(y), _f64(go)
    return (go64 * y64 * (1.0 - y64)).astype(_fdtype(y, go))


def tanh_backward(y, grad_out):
    """输入为 tanh 输出 y。"""
    y, go = np.asarray(y), np.asarray(grad_out)
    y64, go64 = _f64(y), _f64(go)
    return (go64 * (1.0 - y64 * y64)).astype(_fdtype(y, go))


# ---------------------------------------------------------------------------
# 嵌入
# ---------------------------------------------------------------------------
def embedding_backward(ids, grad_out, table_shape):
    """``embedding(ids, table)`` 对 table 的梯度（scatter add）。

    ids: int 任意形状（须非负）；grad_out 同 ids 形状 + [E]。
    table_shape: 表 ``[V, E]`` 形状（便于预先分配）。返回 grad_table。
    """
    ids = np.asarray(ids)
    go = np.asarray(grad_out)
    if ids.min() < 0:
        raise ValueError("embedding_backward 需要非负索引")
    V, E = int(table_shape[0]), int(table_shape[1])
    g = np.zeros((V, E), dtype=np.float64)
    np.add.at(g, ids.ravel().astype(np.int64), go.reshape(-1, E).astype(np.float64))
    return g.astype(_fdtype(go))


# ---------------------------------------------------------------------------
# 工具（插值 / clamp / 逐元素 / einsum）
# ---------------------------------------------------------------------------
def interpolate_linear_backward(x, grad_out, scale_factor=2, axis=-1):
    """``interpolate_linear`` 的反向（align_corners=False）。"""
    x, go = np.asarray(x), np.asarray(grad_out)
    x64, go64 = _f64(x), _f64(go)

    L_in = x.shape[axis]
    L_out = go.shape[axis]
    src = (np.arange(L_out, dtype=np.float64) + 0.5) / float(scale_factor) - 0.5
    src = np.clip(src, 0.0, L_in - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, L_in - 1)
    frac = src - lo

    move = axis != x.ndim - 1
    if move:
        x64 = np.moveaxis(x64, axis, -1)
        go64 = np.moveaxis(go64, axis, -1)
    g = go64
    gx = np.zeros_like(x64)
    idx_lo = tuple([slice(None)] * (x64.ndim - 1) + [lo])
    idx_hi = tuple([slice(None)] * (x64.ndim - 1) + [hi])
    # 注意：fancy 索引重复时 `gx[idx] += v` 不是累加，必须用 np.add.at
    np.add.at(gx, idx_lo, g * (1.0 - frac))
    np.add.at(gx, idx_hi, g * frac)
    if move:
        gx = np.moveaxis(gx, -1, axis)
    return gx.astype(_fdtype(x, go))


def clamp_min_backward(x, grad_out, min_value):
    x, go = np.asarray(x), np.asarray(grad_out)
    return np.where(x >= min_value, go, 0.0).astype(_fdtype(x, go))


def mul_backward(a, b, grad_out):
    a, b, go = np.asarray(a), np.asarray(b), np.asarray(grad_out)
    a64, b64, go64 = _f64(a), _f64(b), _f64(go)
    dt = _fdtype(a, b, go)
    return (go64 * b64).astype(dt), (go64 * a64).astype(dt)


def add_backward(a, b, grad_out):
    return np.asarray(grad_out).copy(), np.asarray(grad_out).copy()


def div_backward(a, b, grad_out):
    a, b, go = np.asarray(a), np.asarray(b), np.asarray(grad_out)
    a64, b64, go64 = _f64(a), _f64(b), _f64(go)
    dt = _fdtype(a, b, go)
    inv = 1.0 / b64
    return (go64 * inv).astype(dt), (-go64 * a64 * inv * inv).astype(dt)


def exp_backward(y, grad_out):
    """输入为 exp 输出 y。"""
    go = np.asarray(grad_out)
    return (np.asarray(go, dtype=np.float64) * _f64(y)).astype(_fdtype(go, y))


def log_backward(x, grad_out):
    x, go = np.asarray(x), np.asarray(grad_out)
    return (_f64(go) / _f64(x)).astype(_fdtype(x, go))


def sqrt_backward(x, grad_out):
    x, go = np.asarray(x), np.asarray(grad_out)
    return (0.5 * _f64(go) / np.sqrt(_f64(x))).astype(_fdtype(x, go))


def pow_backward(x, grad_out, exp):
    x, go = np.asarray(x), np.asarray(grad_out)
    g = _f64(go) * float(exp) * _f64(x) ** (float(exp) - 1.0)
    return g.astype(_fdtype(x, go))


def neg_backward(grad_out):
    return -np.asarray(grad_out)


def einsum_backward(expr, arrays, grad_out):
    """``np.einsum(expr, *arrays)`` 的反向（返回与 arrays 一一对应的梯度）。

    要求每个输入的标签互不共享（本网络内成立）。
    """
    subs, out_sub = expr.replace(" ", "").split("->")[0].split(","), expr.replace(
        " ", "").split("->")[1]
    n = len(subs)
    arr64 = [np.asarray(a, dtype=np.float64) for a in arrays]
    go64 = np.asarray(grad_out, dtype=np.float64)
    dt = _fdtype(*arrays, grad_out)
    grads = []
    for i in range(n):
        others = [(arr64[j], subs[j]) for j in range(n) if j != i]
        if others:
            expr_i = out_sub + "," + ",".join(s for _, s in others) + "->" + subs[i]
            shapes = tuple([go64.shape] + [a.shape for a, _ in others])
            g = np.einsum(expr_i, go64, *[a for a, _ in others],
                          optimize=einsum_optimize_path(expr_i, shapes))
        else:
            g = np.einsum(out_sub + "->" + subs[i], go64)
        grads.append(g.astype(dt))
    return grads


# ---------------------------------------------------------------------------
# 形状类算子（反向为线性转置映射）
# ---------------------------------------------------------------------------
def pad_backward(grad_out, pad_l, pad_r, axis=-1):
    """``np.pad(..., mode='constant')`` 的反向 = 裁剪 pad 区。"""
    go = np.asarray(grad_out)
    sl = [slice(None)] * go.ndim
    sl[axis] = slice(pad_l, go.shape[axis] - pad_r)
    return go[tuple(sl)]


def slice_backward(grad_out, dst_shape, start, stop, axis=-1):
    """``x[..., start:stop]`` 的反向：scatter 回 dst_shape 数组。"""
    go = np.asarray(grad_out)
    g = np.zeros(dst_shape, dtype=_fdtype(go))
    sl = [slice(None)] * len(dst_shape)
    sl[axis] = slice(start, stop)
    g[tuple(sl)] += go
    return g


def flip_backward(grad_out, axis=1):
    return np.flip(np.asarray(grad_out), axis=axis)


def repeat_interleave_backward(x, grad_out, n, axis=None):
    """``np.repeat(x, n, axis)`` 的反向：沿 n 块求和。"""
    x, go = np.asarray(x), np.asarray(grad_out)
    go64 = _f64(go)
    if axis is None:
        gx = go64.reshape(-1, n).sum(axis=1).reshape(x.shape)
    else:
        ax = axis if axis >= 0 else axis + go64.ndim
        shape = list(go64.shape)
        shape[ax] = x.shape[ax]
        shape.insert(ax + 1, n)
        gx = go64.reshape(shape).sum(axis=ax + 1)
    return gx.astype(_fdtype(x, go))


# ---------------------------------------------------------------------------
# mel 谱（训练监督信号）反向
# ---------------------------------------------------------------------------
def _hann_periodic(win_size):
    n = np.arange(win_size, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_size)


# rfft 转置用的 cos/sin 矩阵缓存（key: n_fft）
_FFT_COS = None
_FFT_SIN = None
_FFT_N = -1


def _mel_basis_or_load(sampling_rate, n_fft, num_mels, fmin, fmax, mel_basis):
    if mel_basis is not None:
        return np.asarray(mel_basis, dtype=np.float64)
    from .dsp.mel import mel_filter_bank  # noqa: PLC0415
    return mel_filter_bank(
        int(sampling_rate), int(n_fft), int(num_mels), fmin, fmax, htk=False
    ).astype(np.float64)


def mel_spectrogram_backward(
    y,
    grad_mel,
    n_fft,
    sampling_rate,
    hop_size,
    win_size,
    num_mels,
    fmin,
    fmax,
    center=False,
    mel_basis=None,
):
    """``mel_spectrogram_torch(y, ...)`` 的反向（训练监督 mel 损失用）。

    前向链：y -> reflect pad -> 加窗帧 -> rfft -> mag=sqrt(re²+im²+2e-7)
    -> mel 基 matmul -> log(clamp(*, 2e-6))。本函数沿该链反向，其中 rfft 的
    转置用余弦/正弦实矩阵（``re[k] = Σ_n x[n] cos(2πkn/N)``，
    ``im[k] = -Σ_n x[n] sin(2πkn/N)``）精确完成。

    返回 grad_y ``[B, T]``（与 y 同形状）。
    """
    global _FFT_COS, _FFT_SIN, _FFT_N
    y64 = np.asarray(y, dtype=np.float64)
    go64 = np.asarray(grad_mel, dtype=np.float64)  # [B, num_mels, F]
    n_fft, hop, win = int(n_fft), int(hop_size), int(win_size)
    mb64 = _mel_basis_or_load(sampling_rate, n_fft, num_mels, fmin, fmax, mel_basis)

    if _FFT_N != n_fft:
        k = np.arange(n_fft // 2 + 1, dtype=np.float64)[:, None]
        nn = np.arange(n_fft, dtype=np.float64)[None, :]
        theta = 2.0 * np.pi * k * nn / n_fft
        _FFT_COS = np.cos(theta)
        _FFT_SIN = np.sin(theta)
        _FFT_N = n_fft

    pad = (n_fft - hop) // 2
    w64 = _hann_periodic(win)
    F = go64.shape[-1]

    grad_y = np.zeros_like(y64)
    for b in range(y64.shape[0]):
        y_len = y64.shape[1]
        x = np.pad(y64[b], (pad, pad), mode="reflect")
        if center:
            x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
        n = len(x)
        n_frames = 1 + (n - n_fft) // hop
        if n_frames != F:
            raise ValueError(
                f"mel_spectrogram_backward: 前向帧数 {n_frames} != grad_mel 帧数 {F}")
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = x[idx] * w64[None, :]  # [n_frames, n_fft]
        S = np.fft.rfft(frames, n=n_fft, axis=1).T  # [n_fft//2+1, F]

        mag = np.sqrt(S.real ** 2 + S.imag ** 2 + 2e-7)  # [n_fft//2+1, F]
        mel_in = mb64 @ mag  # [num_mels, F]
        g_mel_in = go64[b] / np.maximum(mel_in, 2e-6)
        g_mel_in = np.where(mel_in < 2e-6, 0.0, g_mel_in)  # clamp 下界梯度为 0
        g_mag = mb64.T @ g_mel_in  # [n_fft//2+1, F]
        g_re = g_mag * (S.real / mag)  # [bins, F]
        g_im = g_mag * (S.imag / mag)
        # grad_x[n] = Σ_k g_re[k]·cos(2πkn/N) - g_im[k]·sin(2πkn/N)
        g_frames = g_re.T @ _FFT_COS - g_im.T @ _FFT_SIN  # [F, N]
        g_frames = g_frames * w64[None, :]

        # 梯度算在 pad 后信号 x 上，再（经 reflect-pad 的镜像映射）散回原始 y
        gx_pad = np.zeros(n, dtype=np.float64)
        np.add.at(gx_pad, idx.ravel(), g_frames.ravel())
        center_off = n_fft // 2 if center else 0
        g_mid = gx_pad[center_off : center_off + 2 * pad + y_len]  # 仅保留 reflect pad 区
        # q -> y 索引：左 pad 镜像（q<pad: m=pad-q）、中区（m=q-pad）、右 pad 镜像
        n_mid = g_mid.shape[0]
        q = np.arange(n_mid, dtype=np.int64)
        m = q - pad
        m = np.where(q < pad, pad - q, m)
        m = np.where(q >= pad + y_len, 2 * y_len - 2 - (q - pad), m)
        np.add.at(grad_y[b], m, g_mid)
    return np.asarray(grad_y, dtype=_fdtype(y, grad_mel))


# ---------------------------------------------------------------------------
# 数值梯度工具
# ---------------------------------------------------------------------------
def numerical_grad(fwd, args, kwargs=None, grad_out=None, eps=1e-5, only=None):
    """对 ``fwd(*args, **kwargs)`` 的输出求 ``sum(out * grad_out)`` 的数值梯度。

    对 args 中每个 float ndarray 参数（可由 ``only`` 限定下标）做一阶中心差分
    （eps=1e-5）。返回 ``{param_idx: numeric_grad_ndarray}``。
    """
    kwargs = kwargs or {}
    out = np.asarray(fwd(*args, **kwargs), dtype=np.float64)
    go = (np.ones_like(out) if grad_out is None
          else np.asarray(grad_out, dtype=np.float64))
    if out.shape != go.shape:
        raise ValueError(f"numerical_grad: grad_out 形状 {go.shape} != 输出 {out.shape}")

    results = {}
    arg_list = list(args)
    for i, a in enumerate(args):
        if only is not None and i not in only:
            continue
        a = np.asarray(a)
        if a.dtype not in (np.float32, np.float64, np.float16):
            continue
        a64 = a.astype(np.float64)
        num = np.zeros_like(a64)
        for j in np.ndindex(*a64.shape):
            old = a64[j]
            a64[j] = old + eps
            arg_list[i] = a64
            fp = float(np.sum(np.asarray(fwd(*arg_list, **kwargs), dtype=np.float64) * go))
            a64[j] = old - eps
            fm = float(np.sum(np.asarray(fwd(*arg_list, **kwargs), dtype=np.float64) * go))
            a64[j] = old
            num[j] = (fp - fm) / (2.0 * eps)
        results[i] = num
        arg_list[i] = a
    return results


def _self_test():
    """minimal smoke：所有基础算子自身数值梯度自检（无 torch）。"""
    import math
    rng = np.random.default_rng(3)
    ok = True

    def rel_err(got, num, atol=1e-6):
        got = np.asarray(got, dtype=np.float64).ravel()
        num = np.asarray(num, dtype=np.float64).ravel()
        denom = float(np.max(np.abs(num))) + atol
        return float(np.max(np.abs(got - num)) / denom)

    # linear
    x = rng.standard_normal((3, 8)).astype(np.float64)
    w = rng.standard_normal((5, 8)).astype(np.float64)
    b = rng.standard_normal(5).astype(np.float64)
    go = rng.standard_normal((3, 5)).astype(np.float64)
    gx, gw, gb = linear_backward(x, w, go, b)
    num = numerical_grad(
        lambda xa, wa, ba: np.sum((xa @ wa.T + ba) * go), (x, w, b))
    ok &= rel_err(gx, num[0]) < 1e-6 and rel_err(gw, num[1]) < 1e-6 and rel_err(gb, num[2]) < 1e-6

    # conv1d
    x = rng.standard_normal((2, 4, 16)).astype(np.float64)
    w = rng.standard_normal((6, 4, 3)).astype(np.float64)
    go = rng.standard_normal((2, 6, 8)).astype(np.float64)
    fwd = lambda xa, wa: nn_conv1d(xa, wa, stride=2, padding=1)
    gx, gw, gb = conv1d_backward(x, w, go, stride=2, padding=1, b=None)
    num = numerical_grad(lambda xa, wa: np.sum(nn_conv1d(xa, wa, stride=2, padding=1) * go), (x, w))
    ok &= rel_err(gx, num[0]) < 1e-6 and rel_err(gw, num[1]) < 1e-6

    # conv_transpose1d
    w = rng.standard_normal((4, 6, 3)).astype(np.float64)
    o = nn_convT(x, w, stride=2, padding=1)
    go = rng.standard_normal(o.shape).astype(np.float64)
    gx, gw, _ = conv_transpose1d_backward(x, w, go, stride=2, padding=1)
    num = numerical_grad(lambda xa, wa: np.sum(nn_convT(xa, wa, stride=2, padding=1) * go), (x, w))
    ok &= rel_err(gx, num[0]) < 1e-6 and rel_err(gw, num[1]) < 1e-6

    # layer_norm / softmax / activations
    x = rng.standard_normal((4, 3, 12)).astype(np.float64)
    g = rng.standard_normal(12).astype(np.float64)
    be = rng.standard_normal(12).astype(np.float64)
    go = rng.standard_normal((4, 3, 12)).astype(np.float64)
    gx, gg, gb = layer_norm_backward(x, g, be, go)
    fwd_ln = lambda xa, ga, ba: nn_layer_norm(xa, ga, ba)
    num = numerical_grad(lambda xa, ga, ba: np.sum(nn_layer_norm(xa, ga, ba) * go), (x, g, be))
    ok &= rel_err(gx, num[0]) < 1e-7 and rel_err(gg, num[1]) < 1e-7 and rel_err(gb, num[2]) < 1e-7

    y = nn_softmax(x)
    go = rng.standard_normal(y.shape).astype(np.float64)
    gsm = softmax_backward(y, go)
    num = numerical_grad(lambda xa: np.sum(nn_softmax(xa) * go), (x,))
    ok &= rel_err(gsm, num[0]) < 1e-7

    for fn, bwd in ((nn_relu, relu_backward), (nn_leaky, leaky_relu_backward),
                    (nn_gelu, gelu_backward)):
        y = fn(x)
        go = rng.standard_normal(x.shape).astype(np.float64)
        g = bwd(x, go) if bwd is not leaky_relu_backward else bwd(x, go, 0.1)
        num = numerical_grad(lambda xa: np.sum(fn(xa) * go), (x,))
        ok &= rel_err(g, num[0]) < 1e-7

    y = nn_sigmoid(x)
    go = rng.standard_normal(x.shape).astype(np.float64)
    ok &= rel_err(sigmoid_backward(y, go),
                  numerical_grad(lambda xa: np.sum(nn_sigmoid(xa) * go), (x,))[0]) < 1e-7

    # embedding
    ids = np.array([[0, 1, 2], [1, 2, 3]])
    table = rng.standard_normal((4, 5)).astype(np.float64)
    go = rng.standard_normal(ids.shape + (5,)).astype(np.float64)
    gt = embedding_backward(ids, go, table.shape)
    num = np.zeros_like(table)
    for idx, g2 in zip(ids.ravel(), go.reshape(-1, 5)):
        num[idx] += g2
    ok &= np.allclose(gt, num, rtol=1e-9, atol=1e-9)

    # einsum
    A = rng.standard_normal((2, 3, 4)).astype(np.float64)
    B = rng.standard_normal((2, 5, 4)).astype(np.float64)
    go = rng.standard_normal((2, 3, 5)).astype(np.float64)
    ga, gb = einsum_backward("bqd,bkd->bqk", (A, B), go)
    num = numerical_grad(lambda xa, xb: np.sum(np.einsum("bqd,bkd->bqk", xa, xb) * go), (A, B))
    ok &= rel_err(ga, num[0]) < 1e-7 and rel_err(gb, num[1]) < 1e-7

    # interpolate
    x = rng.standard_normal((2, 3, 10)).astype(np.float64)
    go = rng.standard_normal((2, 3, 20)).astype(np.float64)
    gi = interpolate_linear_backward(x, go, 2)
    num = numerical_grad(lambda xa: np.sum(nn_interp(xa, 2) * go), (x,))
    ok &= rel_err(gi, num[0]) < 1e-7

    print("nn_backward self_test:", "PASS" if ok else "FAIL")
    return ok


def nn_conv1d(x, w, stride=1, padding=0):
    e, re = np.einsum, np.asarray(w, dtype=np.float64)
    pad_l = pad_r = int(padding)
    K = w.shape[2]
    Kd = K
    x_pad = np.pad(np.asarray(x, dtype=np.float64), ((0, 0), (0, 0), (pad_l, pad_r)))
    Tp = x_pad.shape[2]
    oL = (Tp - Kd) // stride + 1
    win = np.lib.stride_tricks.sliding_window_view(x_pad, Kd, axis=-1)
    win = win[:, :, ::stride, :]
    return e("bctk,ock->bot", win, w, optimize=True)


def nn_convT(x, w, stride=1, padding=0, output_padding=0, dilation=1):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    B, Cin, T = x.shape
    Cout = w.shape[1]
    K = w.shape[2]
    oL = (T - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    out = np.zeros((B, Cout, oL), dtype=np.float64)
    for k in range(K):
        pos = np.arange(T) * stride - padding + k * dilation
        valid = (pos >= 0) & (pos < oL)
        if not valid.any():
            continue
        tv = np.nonzero(valid)[0]
        out[:, :, pos[tv]] += np.einsum("bct,co->bot", x[:, :, tv], w[:, :, k])
    return out


def nn_layer_norm(x, gamma, beta, eps=1e-5):
    x, g, b = (np.asarray(v, dtype=np.float64) for v in (x, gamma, beta))
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def nn_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def nn_relu(x):
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def nn_leaky(x, slope=0.1):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, x, slope * x)


def nn_gelu(x):
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def nn_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def nn_interp(x, scale_factor=2, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    L_in = x.shape[axis]
    L_out = int(L_in * scale_factor)
    src = np.clip((np.arange(L_out) + 0.5) / float(scale_factor) - 0.5, 0.0, L_in - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, L_in - 1)
    frac = src - lo
    move = axis != -1
    if move:
        x = np.moveaxis(x, axis, -1)
    out = x[..., lo] * (1.0 - frac) + x[..., hi] * frac
    if move:
        out = np.moveaxis(out, -1, axis)
    return out


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
