# -*- coding: utf-8 -*-
"""FCPE 基频（F0）提取模型 —— 纯 numpy 推理实现。

对齐 torchfcpe 0.0.4 的 bundled 推理模型（``spawn_bundled_infer_model``，
即 ``infer/fcpe.py`` 的 FCPEInfer 所使用的模型）：**CFNaiveMelPE**
（``torchfcpe/assets/fcpe_c_v001.pt``，配置见权重内 config_dict：
``model.type = CFNaiveMelPE``、``conv_only = True``、``hidden_dims=512``、
``n_layers=6``、``out_dims=360``、``f0_min=32.7``、``f0_max=1975.5``）。

结构（逐层核对 wheel 内 ``torchfcpe/models.py`` 与
``torchfcpe/model_conformer_naive.py`` 源码 + 权重键名）：
    1. wav2mel（``torchfcpe/mel_extractor.py`` MelModule + Wav2MelModule）：
       手动 pad 432/432（reflect）→ stft(1024/160/1024, hann periodic,
       center=False) → sqrt(|X|^2 + 1e-9) → librosa mel 基
       (sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000, htk=False,
       norm='slaney') → log(clamp(·, min=1e-5)) → [1, F, 128]
       （帧数对齐到 int(T//160)+1，不足时复制末帧）。
    2. input_stack：Conv1d(128→512, 3, pad1) + GroupNorm(4) + LeakyReLU
       + Conv1d(512→512, 3, pad1)（输入先转置为 [B, 128, F]）。
    3. net：6 × CFNEncoderLayer（conv_only=True → 无 SelfAttention）：
         x = x + ConformerConvModule(LayerNorm(x))
       ConformerConvModule：Conv1d(512→2048, 1) → GLU(dim=1)
         → DepthWiseConv1d(1024, kernel31, pad15, groups=1024) → SiLU
         → Conv1d(1024→512, 1)。
    4. norm：LayerNorm(512)；output_proj：weight_norm(Linear(512→360))，
       Sigmoid → latent [1, F, 360]。
    5. 解码（与 ``infer/fcpe.py`` / torchfcpe 的 local_argmax 完全一致）：
       max_index = argmax(latent)；local_index = arange(9) + (max_index-4)
       clamp [0, 359]；gather cent_table 与 latent 的 9 bin；
       decoded = Σ(c·s)/Σ(s)；confidence = max(latent)；
       confidence ≤ threshold 的帧 decoded 置 -inf → f0=0；
       f0 = 10 * 2^(decoded/1200)。

本模块不 import torch —— 权重用 ``torch_compat.load_pth`` 读取（float32
存储），算子全部来自 ``runtime/nn`` 与本文件自带的少量自实现
（periodic hann、depthwise conv1d、GLU、SiLU、weight_norm linear）。

权重路径：``assets/fcpe/fcpe_c_v001.pt``（可用 ``tools/download_fcpe.py``
从 PyPI wheel 获取）。
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np

from torch_compat import load_pth  # 纯 Python 读 .pth，不依赖 torch

from .. import nn as nn_ops
from ..dsp.mel import mel_filter_bank

__all__ = ["FCPE", "load_fcpe"]

# ---------------------------------------------------------------------------
# 模型常量（与 fcpe_c_v001.pt 内 config_dict 一致）
# ---------------------------------------------------------------------------
_SR = 16000
_N_MELS = 128
_N_FFT = 1024
_WIN_SIZE = 1024
_HOP = 160
_FMIN = 0.0
_FMAX = 8000.0
_MEL_CLAMP = 1e-5

_HIDDEN = 512
_N_LAYERS = 6
_N_HEADS = 8  # conv_only=True 时未使用，仅保留文档价值
_OUT_DIMS = 360
_F0_MIN = 32.7
_F0_MAX = 1975.5

_F32 = np.float32


# ---------------------------------------------------------------------------
# 少量自实现算子（runtime/nn 未覆盖的部分）
# ---------------------------------------------------------------------------

def _periodic_hann(n: int) -> np.ndarray:
    """对齐 ``torch.hann_window(n)``（periodic，x[n] = 0.5(1-cos(2πn/N))）。"""
    n = int(n)
    t = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * t / n)).astype(np.float64)


def depthwise_conv1d(x, w, b=None, stride=1, padding=0):
    """Depthwise Conv1d（``groups == in_channels == out_channels``），
    对齐 ``torch.nn.Conv1d(..., groups=C)`` 推理语义。

    参数:
        x: ``[B, C, T]``
        w: ``[C, 1, K]``
        b: ``[C]`` 或 None
    输出长度: ``oL = (T + 2*padding - K)//stride + 1``。

    实现：显式零填充 → 滑窗 → einsum（每个输出通道只与对应输入通道卷积）。
    为限制峰值内存，T 维分块（每块 4096 帧）。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    B, C, T = x.shape
    O, _, K = w.shape
    assert O == C, "depthwise conv 要求 out_channels == in_channels"
    pad = int(padding)
    stride = int(stride)
    oL = (T + 2 * pad - K) // stride + 1
    if oL <= 0:
        return np.zeros((B, O, 0), dtype=x.dtype)

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    wc = w[:, 0, :]  # [C, K]

    out = np.empty((B, O, oL), dtype=x.dtype)
    chunk = 4096
    for s in range(0, oL, chunk):
        e = min(s + chunk, oL)
        seg = x_pad[:, :, s * stride : s * stride + (e - s - 1) * stride + K]
        win = np.lib.stride_tricks.sliding_window_view(
            seg, K, axis=-1
        )  # [B, C, n, K]
        out[:, :, s:e] = np.einsum("bctk,ck->bct", win, wc, optimize=True)
    if b is not None:
        out += np.asarray(b, dtype=out.dtype).reshape(1, -1, 1)
    return out


def _glu_dim1(x):
    """``F.glu(x, dim=1)``：沿通道维对半切分，a * sigmoid(b)。"""
    x = np.asarray(x)
    B, C, T = x.shape
    a = x[:, : C // 2, :]
    b = x[:, C // 2 :, :]
    # nn.sigmoid 的 np.where 双分支会求值被丢弃的 exp，极端输入产生无害溢出警告
    with np.errstate(over="ignore", invalid="ignore"):
        return a * nn_ops.sigmoid(b)


def _silu(x):
    """``F.silu``：x * sigmoid(x)。"""
    x = np.asarray(x)
    with np.errstate(over="ignore", invalid="ignore"):
        return x * nn_ops.sigmoid(x)


def _weight_norm_linear(x, weight_g, weight_v, bias):
    """weight_norm 包裹的 Linear 前向。

    ``w = g * v / ||v||``（对输出维逐行 L2 归一后按 g 缩放），然后 ``x @ w.T``。
    """
    v = np.asarray(weight_v, dtype=_F32)  # [O, D]
    g = np.asarray(weight_g, dtype=_F32).reshape(-1)  # [O]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    w = v / np.maximum(norms, 1e-12) * g[:, None]
    return nn_ops.linear(x, w, bias)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class FCPE:
    """纯 numpy FCPE（CFNaiveMelPE）基频提取器。

    对齐 torchfcpe bundled 推理模型：输入 16kHz 单声道音频，输出每帧基频。
    """

    def __init__(self, model_path: str):
        t0 = time.perf_counter()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"FCPE 权重不存在: {model_path!r}。请运行 "
                "`python tools/download_fcpe.py` 下载官方权重 "
                "（来源：PyPI torchfcpe==0.0.4 wheel 内的 "
                "torchfcpe/assets/fcpe_c_v001.pt）后重试。"
            )
        raw = load_pth(model_path)
        if not isinstance(raw, dict) or "model" not in raw:
            raise ValueError(
                f"FCPE 权重 {model_path!r} 顶层结构异常：期望 dict 且含 "
                "'model' 状态字典（torchfcpe bundled ckpt）。"
            )
        sd: Dict[str, np.ndarray] = raw["model"]
        self.W: Dict[str, np.ndarray] = {
            k: (v.astype(_F32) if v.dtype != _F32 else v) for k, v in sd.items()
        }
        # cent_table 直接从权重加载（buffer，360 bin 中心值）
        self.cent_table = self.W["cent_table"]  # [360]

        # librosa mel 基：sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000,
        # htk=False（Slaney 刻度 + 面积归一），与 torchfcpe MelModule 一致。
        self.mel_basis = mel_filter_bank(
            _SR, _N_FFT, _N_MELS, _FMIN, _FMAX, htk=False
        ).astype(_F32)  # [128, 513]
        self.hann = _periodic_hann(_WIN_SIZE)  # 对齐 torch.hann_window(1024)

        self.out_dims = _OUT_DIMS
        self._load_time = time.perf_counter() - t0

    # ------------------------------------------------------------------ mel
    def _wav2mel(self, x: np.ndarray) -> np.ndarray:
        """``[T]`` float32 16k -> ``[1, F, 128]`` log-mel。

        对齐 torchfcpe ``MelModule`` + ``Wav2MelModule``：
        pad 432/432（reflect）→ stft(center=False) → sqrt(|·|²+1e-9)
        → mel 基 → log(clamp 1e-5) → [F, 128]，帧数对齐 int(T//160)+1。
        """
        x = np.asarray(x, dtype=np.float64)
        T = x.shape[0]
        pad_left = (_WIN_SIZE - _HOP) // 2  # 432
        pad_right = max(
            (_WIN_SIZE - _HOP + 1) // 2,  # 432
            _WIN_SIZE - T - pad_left,
        )
        if pad_right < T:
            mode = "reflect"
        else:
            mode = "constant"
        y = np.pad(x, (pad_left, pad_right), mode=mode)
        # stft(center=False)：帧数 = 1 + (len(y) - n_fft)//hop
        spec = _stft_frames(y, self.hann)  # [513, nf] complex128
        mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(_F32)
        mel = self.mel_basis @ mag  # [128, nf]
        log_mel = np.log(np.clip(mel, _MEL_CLAMP, None)).astype(_F32)
        mel_t = log_mel.T  # [nf, 128]
        n_frames = int(T // _HOP) + 1
        if n_frames > mel_t.shape[0]:
            mel_t = np.concatenate([mel_t, mel_t[-1:]], axis=0)  # 复制末帧
        elif n_frames < mel_t.shape[0]:
            mel_t = mel_t[:n_frames, :]
        return mel_t[None, ...]  # [1, F, 128]

    # ------------------------------------------------------------ 网络前向
    def _input_stack(self, x):
        """``[1, F, 128]`` -> ``[1, F, 512]``（Conv1d+GN+LeakyReLU+Conv1d）。"""
        ck = self.W
        xt = x.transpose(0, 2, 1)  # [1, 128, F]
        w0 = ck["input_stack.0.weight"]  # [512, 128, 3]
        b0 = ck["input_stack.0.bias"]
        h = nn_ops.conv1d(xt, w0, b0, stride=1, padding=1)
        h = nn_ops.group_norm(
            h,
            ck["input_stack.1.weight"],
            ck["input_stack.1.bias"],
            num_groups=4,
            eps=1e-5,
        )
        h = nn_ops.leaky_relu(h, negative_slope=0.01)
        w3 = ck["input_stack.3.weight"]  # [512, 512, 3]
        b3 = ck["input_stack.3.bias"]
        h = nn_ops.conv1d(h, w3, b3, stride=1, padding=1)
        return h.transpose(0, 2, 1)  # [1, F, 512]

    def _encoder_layer(self, x, layer_idx):
        """单层 CFNEncoderLayer（conv_only：无注意力分支）。"""
        ck = self.W
        p = f"net.encoder_layers.{layer_idx}.conformer"
        # pre-norm：conformer 内部第一个 LayerNorm
        x_n = nn_ops.layer_norm(
            x, ck[p + ".net.0.weight"], ck[p + ".net.0.bias"], eps=1e-5
        )
        xt = x_n.transpose(0, 2, 1)  # [1, 512, F]
        h = nn_ops.conv1d(
            xt, ck[p + ".net.2.weight"], ck[p + ".net.2.bias"], stride=1, padding=0
        )  # [1, 2048, F]
        h = _glu_dim1(h)  # [1, 1024, F]
        h = depthwise_conv1d(
            h,
            ck[p + ".net.4.conv.weight"],
            ck[p + ".net.4.conv.bias"],
            stride=1,
            padding=15,
        )  # [1, 1024, F]（DepthWiseConv1d kernel31 pad15 groups=1024）
        h = _silu(h)
        h = nn_ops.conv1d(
            h, ck[p + ".net.6.weight"], ck[p + ".net.6.bias"], stride=1, padding=0
        )  # [1, 512, F]
        h = h.transpose(0, 2, 1)  # [1, F, 512]
        return x + h

    def _encoder(self, x):
        """6 × CFNEncoderLayer。"""
        for i in range(_N_LAYERS):
            x = self._encoder_layer(x, i)
        return x

    def _latent(self, mel):
        """``[1, F, 128]`` -> ``[1, F, 360]``（sigmoid salience latent）。"""
        ck = self.W
        h = self._input_stack(mel)
        h = self._encoder(h)
        h = nn_ops.layer_norm(h, ck["norm.weight"], ck["norm.bias"], eps=1e-5)
        h = _weight_norm_linear(
            h, ck["output_proj.weight_g"], ck["output_proj.weight_v"],
            ck["output_proj.bias"],
        )  # [1, F, 360]
        with np.errstate(over="ignore", invalid="ignore"):
            return nn_ops.sigmoid(h)

    # -------------------------------------------------------------- decode
    def _decode(self, latent, decoder_mode="local_argmax", threshold=0.006):
        """``[1, F, 360]`` latent -> ``[F]`` Hz f0（对齐 torchfcpe 解码）。"""
        latent = np.asarray(latent, dtype=_F32)
        batch, frames, _ = latent.shape
        cents = np.broadcast_to(
            self.cent_table[None, None, :], (batch, frames, self.out_dims)
        )

        if decoder_mode == "argmax":
            confidence = latent.max(axis=-1, keepdims=True)
            decoded = (cents * latent).sum(axis=-1, keepdims=True) / latent.sum(
                axis=-1, keepdims=True
            )
        elif decoder_mode == "local_argmax":
            confidence = latent.max(axis=-1, keepdims=True)  # [B, F, 1]
            max_index = latent.argmax(axis=-1)  # [B, F]
            local_index = (
                np.arange(9, dtype=np.int64)[None, None, :] + (max_index[..., None] - 4)
            )
            local_index = np.clip(local_index, 0, self.out_dims - 1)
            ci_l = np.take_along_axis(cents, local_index, axis=-1)  # [B, F, 9]
            y_l = np.take_along_axis(latent, local_index, axis=-1)
            decoded = (ci_l * y_l).sum(axis=-1, keepdims=True) / y_l.sum(
                axis=-1, keepdims=True
            )
        else:
            raise ValueError(f"Unknown FCPE decoder mode: {decoder_mode}")

        # confidence <= threshold 的帧 decoded 置 -inf（torch 掩码语义），
        # f0 = 10 * 2^(-inf/1200) = 0
        mask = np.where(confidence <= threshold, float("-inf"), 1.0)
        decoded = decoded * mask
        f0 = 10.0 * np.power(2.0, decoded / 1200.0)
        return f0[..., 0]  # [B, F]

    # ------------------------------------------------------------- 顶层入口
    def infer(
        self,
        x: np.ndarray,
        sr: int = 16000,
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
    ) -> np.ndarray:
        """``[T]`` float32 音频（sr=16000）-> ``[F]`` float32 基频（Hz）。

        F = int(len(x)//160) + 1（与 torchfcpe ``infer`` 输出帧数一致）。
        """
        x = np.asarray(x, dtype=_F32)
        if x.ndim != 1:
            raise ValueError(f"infer 需要 1D 音频，实际 {x.ndim}D")
        if sr != _SR:
            raise NotImplementedError(
                f"FCPE 仅在 sr={_SR} 下定义，收到 sr={sr}；请先重采样到 16k。"
            )
        mel = self._wav2mel(x)  # [1, F, 128]
        latent = self._latent(mel)  # [1, F, 360]
        f0 = self._decode(latent, decoder_mode=decoder_mode, threshold=threshold)
        return f0[0].astype(_F32)  # [F]


# ---------------------------------------------------------------------------
# STFT 辅助（对齐 torch.stft(center=False, onesided=True, normalized=False)）
# ---------------------------------------------------------------------------

def _stft_frames(y: np.ndarray, win: np.ndarray) -> np.ndarray:
    """对已 pad 的 1D 信号做 center=False 的 rfft 分帧。

    y: ``[L]``；win: 窗 ``[n_fft]``。返回 ``[n_fft//2+1, n_frames]``。
    """
    n_fft = win.shape[0]
    n_frames = 1 + (y.shape[0] - n_fft) // _HOP
    idx = np.arange(n_fft)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = y[idx] * win[None, :]
    return np.fft.rfft(frames, axis=1).T


# ---------------------------------------------------------------------------
# 懒加载缓存
# ---------------------------------------------------------------------------

_CACHE: Dict[str, FCPE] = {}


def _default_model_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets", "fcpe", "fcpe_c_v001.pt",
    )


def load_fcpe(path: Optional[str] = None) -> FCPE:
    """懒加载 FCPE（进程内缓存，默认 assets/fcpe/fcpe_c_v001.pt）。"""
    if path is None:
        path = _default_model_path()
    path = str(path)
    if path not in _CACHE:
        _CACHE[path] = FCPE(path)
    return _CACHE[path]
