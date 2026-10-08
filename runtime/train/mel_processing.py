# -*- coding: utf-8 -*-
"""RVC 训练侧 mel 谱计算（纯 numpy 移植，对齐 ``train/mel_processing.py``）。

零 torch / librosa 依赖（librosa 仅允许在 tests 对照分支 import）。

与原版接口完全一致：
- ``spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False)``
- ``spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)``
- ``mel_spectrogram_torch(y, ...)``
- ``dynamic_range_compression_torch`` / ``dynamic_range_decompression_torch`` /
  ``spectral_normalize_torch`` / ``spectral_de_normalize_torch``

数值对齐要点（保证与 torch+librosa 训练管线一致，误差 <1e-4 见 tests_mel_index.py）：
1. 窗函数用 **periodic hann**（``torch.hann_window(win_size)`` 默认 periodic=True：
   w[i] = 0.5 - 0.5*cos(2πi/win_size)），与 librosa/sym 窗（np.hanning）不同；
2. 手动 reflect pad ``(n_fft - hop)//2`` 两侧，stft 用 center=False（训练侧约定）；
3. mel 基用 ``mel_filter_bank(..., htk=False)``（Slaney 刻度 + 面积归一，
   librosa 默认），matmul 后在 float32 下 log(clamp(x, 2e-6))；
4. 幅值谱 sqrt(re² + im² + 2e-7)（与原版一致）。
"""

from __future__ import annotations

import numpy as np

from ..dsp.mel import mel_filter_bank

__all__ = [
    "MAX_WAV_VALUE",
    "dynamic_range_compression_torch",
    "dynamic_range_decompression_torch",
    "spectral_normalize_torch",
    "spectral_de_normalize_torch",
    "spectrogram_torch",
    "spec_to_mel_torch",
    "mel_spectrogram_torch",
]

MAX_WAV_VALUE = 32768.0


# ---------------------------------------------------------------------------
# 动态范围压缩 / 解压（对齐原版，纯 numpy）
# ---------------------------------------------------------------------------

def dynamic_range_compression_torch(x, C: float = 1, clip_val: float = 2e-6) -> np.ndarray:
    """对数压缩：log(clamp(x, min=clip_val) * C)。"""
    return np.log(np.clip(np.asarray(x, dtype=np.float32), clip_val, None) * C)


def dynamic_range_decompression_torch(x, C: float = 1) -> np.ndarray:
    """对数解压：exp(x) / C。"""
    return np.exp(np.asarray(x, dtype=np.float32)) / C


def spectral_normalize_torch(magnitudes) -> np.ndarray:
    """谱归一化（压缩）。"""
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes) -> np.ndarray:
    """谱反归一化（解压）。"""
    return dynamic_range_decompression_torch(magnitudes)


# ---------------------------------------------------------------------------
# 模块级缓存（同原版 mel_basis / hann_window）
# ---------------------------------------------------------------------------

mel_basis: dict = {}
hann_window: dict = {}


def _hann_periodic(win_size: int) -> np.ndarray:
    """periodic hann 窗，对齐 ``torch.hann_window(win_size)``（periodic=True）。"""
    n = np.arange(win_size, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_size)).astype(np.float32)


def _reflect_pad(x: np.ndarray, pad: int) -> np.ndarray:
    """对 1D 数组两侧 reflect pad（对齐 torch.nn.functional.pad(mode='reflect')）。"""
    if pad <= 0:
        return x
    if len(x) <= pad:
        raise ValueError(
            f"reflect pad 需要 len(x) > pad（{pad}），实际 {len(x)}"
        )
    return np.pad(x, (pad, pad), mode="reflect")


def spectrogram_torch(
    y: np.ndarray,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> np.ndarray:
    """波形 -> 线性频率线性幅度谱。

    Args:
        y: [B, T] float32 波形（batch 可 >1）。
        n_fft: FFT 长度。
        sampling_rate: 采样率（仅用于参数完整性，不参与计算）。
        hop_size: 帧移。
        win_size: 窗长（periodic hann）。
        center: False 时仅手动 pad (n_fft-hop)//2 并 stft(center=False)
            （训练侧约定）；True 时额外再 center pad n_fft//2（同 torch.stft）。

    Returns:
        [B, n_fft//2+1, F] float32 幅度谱（sqrt(re²+im²+2e-7)）。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"spectrogram_torch 需要 [B, T] 输入，实际 {y.ndim}D")

    global hann_window
    key = str(win_size)
    if key not in hann_window:
        hann_window[key] = _hann_periodic(win_size)
    w = hann_window[key]

    pad = (n_fft - hop_size) // 2
    if pad < 0:
        raise ValueError(f"n_fft ({n_fft}) 必须 >= hop_size ({hop_size})")

    specs = []
    for b in range(y.shape[0]):
        x = _reflect_pad(y[b], pad)
        if center:  # 同 torch.stft(center=True)：额外再 pad n_fft//2
            x = _reflect_pad(x, n_fft // 2)
        n = len(x)
        n_frames = 1 + (n - n_fft) // hop_size
        if n_frames <= 0:
            raise ValueError(f"信号过短：n_frames={n_frames}（len={n}, n_fft={n_fft}）")
        idx = np.arange(n_fft)[None, :] + hop_size * np.arange(n_frames)[:, None]
        frames = (x[idx] * w[None, :]).astype(np.float32)
        S = np.fft.rfft(frames, n=n_fft, axis=1).T  # complex64 [n_fft//2+1, F]
        mag = np.sqrt(S.real ** 2 + S.imag ** 2 + 2e-7)
        specs.append(mag.astype(np.float32))
    return np.stack(specs, axis=0)


def spec_to_mel_torch(
    spec: np.ndarray,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """线性幅度谱 -> log-mel 谱。

    Args:
        spec: [B, n_fft//2+1, F] float32（spectrogram_torch 输出）。
        n_fft / num_mels / sampling_rate / fmin / fmax: mel 基参数
            （htk=False：Slaney 刻度 + 面积归一，与 librosa 默认一致）。

    Returns:
        [B, num_mels, F] float32 log-mel 谱。
    """
    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim != 3:
        raise ValueError(f"spec_to_mel_torch 需要 [B, bins, F] 输入，实际 {spec.ndim}D")

    global mel_basis
    key = "%d_%d_%d_%s_%s" % (sampling_rate, n_fft, num_mels, fmin, fmax)
    if key not in mel_basis:
        mel_basis[key] = mel_filter_bank(
            sampling_rate, n_fft, num_mels, fmin, fmax, htk=False
        ).astype(np.float32)

    melspec = np.matmul(mel_basis[key], spec)  # [B, num_mels, F]
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y: np.ndarray,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> np.ndarray:
    """波形 -> Mel 频率对数幅度谱（训练侧主入口）。

    Args:
        y: [B, T] float32 波形。
        n_fft / num_mels / sampling_rate / hop_size / win_size / fmin / fmax:
            mel 谱参数（fmax 可传 None 表示 sr/2）。
        center: 传给 spectrogram_torch（训练侧传 False）。

    Returns:
        [B, num_mels, F] float32 log-mel 谱。
    """
    spec = spectrogram_torch(
        y, n_fft, sampling_rate, hop_size, win_size, center
    )
    melspec = spec_to_mel_torch(
        spec, n_fft, num_mels, sampling_rate, fmin, fmax
    )
    return melspec


def _self_test() -> bool:
    """mel_processing 自测：与 librosa mel 基 + 手算 stft 对照（无 torch 分支）。"""
    print("=== mel_processing.self_test ===")
    ok = True
    rng = np.random.default_rng(7)
    sr, n_fft, hop, win, n_mels = 48000, 2048, 480, 2048, 128
    y = (rng.standard_normal((1, 24000)) * 0.2).astype(np.float32)

    spec = spectrogram_torch(y, n_fft, sr, hop, win, center=False)
    print(f"  spec shape={spec.shape} dtype={spec.dtype}")
    ok &= spec.shape == (1, n_fft // 2 + 1, 1 + (24000 + n_fft - hop) // hop)
    ok &= spec.dtype == np.float32
    ok &= bool(np.all(spec >= 0.0))

    mel = mel_spectrogram_torch(
        y, n_fft, n_mels, sr, hop, win, fmin=0.0, fmax=None, center=False
    )
    print(f"  mel shape={mel.shape} range=[{float(mel.min()):.3f}, {float(mel.max()):.3f}]")
    ok &= mel.shape == (1, n_mels, spec.shape[2])
    ok &= bool(np.all(np.isfinite(mel)))

    # 手算对照（不依赖 torch/librosa）：periodic hann + reflect pad + rfft + slaney mel
    pad = (n_fft - hop) // 2
    x = np.pad(y[0], (pad, pad), mode="reflect")
    n = len(x)
    n_frames = 1 + (n - n_fft) // hop
    w = _hann_periodic(win)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    S = np.fft.rfft((x[idx] * w[None, :]), n=n_fft, axis=1).T
    mag = np.sqrt(S.real ** 2 + S.imag ** 2 + 2e-7).astype(np.float32)
    err_spec = float(np.max(np.abs(spec[0] - mag)))
    print(f"  vs manual stft max abs err = {err_spec:.3e}")
    ok &= err_spec < 1e-5

    print(f"  mel_processing PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
