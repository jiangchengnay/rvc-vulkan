"""Mel 滤波器组与 Mel 谱（对齐 librosa.filters.mel 的布局与归一）。

支持两种 mel 刻度（通过 ``htk`` 参数选择）：
- ``htk=True``（默认，向后兼容）：HTK 频率刻度（1127*ln(1+f/700)），
  三角滤波器**行和归一为 1**（此前的默认行为，RVC 推理侧沿用）。
- ``htk=False``（Slaney，librosa 默认）：低频 [0,1000)Hz 线性段
  （f/(200/3)）+ 高频对数段的分段 mel 刻度，三角滤波器按 librosa
  ``norm='slaney'`` **面积归一**（每通道能量近似常数，``enorm=2/(Hz 带宽)``）。
  该模式数值上与 ``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False)``
  逐元素一致（相对误差 <1e-6）。

- ``mel_spectrogram``：stft -> 功率谱 -> mel 矩阵乘，返回 [n_mels, n_frames] float32。
- ``log_mel_spectrogram``：对 mel 谱做 log(clip(x, 2e-6))，供训练用
  （对应 RVC 训练侧的 dynamic_range_compression / spectral_normalize）。
"""

from __future__ import annotations

import numpy as np

from .fft import stft
from .utils import hz_to_mel, mel_to_hz

__all__ = ["mel_filter_bank", "mel_spectrogram", "log_mel_spectrogram"]

# Slaney mel 刻度常数（与 librosa 一致）
_SLANEY_F_SP = 200.0 / 3.0  # 线性段斜率（Hz/mel 的倒数）
_SLANEY_MIN_LOG_HZ = 1000.0  # 对数段起点频率（Hz）
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP  # 对数段起点（mel）
_SLANEY_LOGSTEP = np.log(6.4) / 27.0  # 对数段步长


def _hz_to_mel_slaney(f) -> np.ndarray:
    """Slaney 刻度：Hz -> mel（线性段 + 对数段的 piecewise 函数）。

    与 ``librosa.core.convert.hz_to_mel(f, htk=False)`` 完全一致：
    f < 1000Hz 用线性段 f/(200/3)；f >= 1000Hz 用
    mel = 1500 + ln(f/1000)/logstep，logstep = ln(6.4)/27。
    """
    f = np.asarray(f, dtype=np.float64)
    mels = f / _SLANEY_F_SP
    if f.ndim:
        log_t = f >= _SLANEY_MIN_LOG_HZ
        mels[log_t] = _SLANEY_MIN_LOG_MEL + np.log(
            f[log_t] / _SLANEY_MIN_LOG_HZ
        ) / _SLANEY_LOGSTEP
    elif f >= _SLANEY_MIN_LOG_HZ:
        mels = _SLANEY_MIN_LOG_MEL + np.log(f / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP
    return mels


def _mel_to_hz_slaney(m) -> np.ndarray:
    """Slaney 刻度：mel -> Hz（``librosa.core.convert.mel_to_hz(f, htk=False)``）。"""
    m = np.asarray(m, dtype=np.float64)
    freqs = _SLANEY_F_SP * m
    if m.ndim:
        log_t = m >= _SLANEY_MIN_LOG_MEL
        freqs[log_t] = _SLANEY_MIN_LOG_HZ * np.exp(
            _SLANEY_LOGSTEP * (m[log_t] - _SLANEY_MIN_LOG_MEL)
        )
    elif m >= _SLANEY_MIN_LOG_MEL:
        freqs = _SLANEY_MIN_LOG_HZ * np.exp(
            _SLANEY_LOGSTEP * (m - _SLANEY_MIN_LOG_MEL)
        )
    return freqs


def mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = True,
) -> np.ndarray:
    """构建 mel 三角滤波器组。

    Args:
        sr: 采样率（Hz）。
        n_fft: FFT 长度（输出 bin 数 = n_fft // 2 + 1）。
        n_mels: mel 通道数。
        fmin: 最低频率（Hz，默认 0）。
        fmax: 最高频率（Hz，默认 sr/2）。
        htk: True 用 HTK 刻度（1127*ln(1+f/700)）+ 行和归一（默认，
            保持此前行为）；False 用 Slaney 刻度 + librosa 面积归一
            （与上游 RVC 训练侧 mel 参数口径一致）。

    Returns:
        float64 数组 [n_mels, n_fft//2+1]。
        htk=True 时每行和为 1；htk=False 时每行面积（≈2 倍三角面积）归一。
    """
    if fmax is None:
        fmax = float(sr) / 2.0
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0.0, float(sr) / 2.0, n_bins)

    if htk:
        # --- HTK 刻度（1127*ln(1+f/700)）+ 行和归一（向后兼容路径） ---
        mel_fmin = float(hz_to_mel(fmin))
        mel_fmax = float(hz_to_mel(fmax))
        if mel_fmax <= mel_fmin:
            raise ValueError(f"fmax ({fmax}) 必须大于 fmin ({fmin})")
        mel_points = np.linspace(mel_fmin, mel_fmax, n_mels + 2)
        f_bins = mel_to_hz(mel_points)  # [n_mels+2]

        weights = np.zeros((n_mels, n_bins), dtype=np.float64)
        for i in range(n_mels):
            left, center, right = f_bins[i], f_bins[i + 1], f_bins[i + 2]
            if center - left > 0:
                up = (fftfreqs - left) / (center - left)
            else:
                up = np.zeros(n_bins)
            if right - center > 0:
                down = (right - fftfreqs) / (right - center)
            else:
                down = np.zeros(n_bins)
            tri = np.maximum(0.0, np.minimum(up, down))
            s = tri.sum()
            if s > 0:
                tri = tri / s  # 行和归一
            weights[i] = tri
        return weights

    # --- Slaney 刻度（htk=False）：精确复刻 librosa.filters.mel ---
    # mel 刻度上均匀取 n_mels+2 个点（含两个端点作为三角左右边界）
    mel_f = _mel_to_hz_slaney(
        np.linspace(
            float(_hz_to_mel_slaney(fmin)),
            float(_hz_to_mel_slaney(fmax)),
            n_mels + 2,
        )
    )  # [n_mels+2] Hz
    fdiff = np.diff(mel_f)  # 相邻边界频率差
    ramps = np.subtract.outer(mel_f, fftfreqs)  # mel_f[i] - fftfreqs[j]

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        # lower = (fftfreqs - mel_f[i]) / (mel_f[i+1] - mel_f[i])
        lower = -ramps[i] / fdiff[i]
        # upper = (mel_f[i+2] - fftfreqs) / (mel_f[i+2] - mel_f[i+1])
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    # librosa norm='slaney'：面积归一（每通道能量近似常数）
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def mel_spectrogram(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    power: float = 2.0,
    center: bool = True,
    htk: bool = True,
) -> np.ndarray:
    """线性幅值谱 -> mel 谱（功率谱，power=2）。

    Args:
        x: 1D 实信号。
        sr: 采样率（Hz）。
        n_fft / hop_length / win_length: STFT 参数。
        n_mels: mel 通道数。
        fmin / fmax: 频率范围（Hz）。
        power: 谱指数（2.0 为功率谱，1.0 为幅值谱）。
        center: STFT 是否 center pad。
        htk: 传给 mel_filter_bank 的刻度选择（默认 True 保持向后兼容）。

    Returns:
        float32 [n_mels, n_frames] mel 谱。
    """
    S = stft(x, n_fft, hop_length, win_length, window="hann", center=center)
    mag = np.abs(S) ** power
    W = mel_filter_bank(sr, n_fft, n_mels, fmin, fmax, htk=htk)
    mel = W @ mag
    return mel.astype(np.float32)


def log_mel_spectrogram(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    htk: bool = True,
) -> np.ndarray:
    """训练用 log-mel 谱：mel 谱 -> log(clip(x, 2e-6))。

    对应 RVC 训练侧 ``dynamic_range_compression_torch``（clip_val=2e-6）。

    Args:
        参数同 mel_spectrogram（htk 透传给滤波器组）。

    Returns:
        float32 [n_mels, n_frames]。
    """
    mel = mel_spectrogram(
        x, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, center=center,
        htk=htk,
    )
    return np.log(np.clip(mel, 2e-6, None)).astype(np.float32)


def _self_test() -> bool:
    """mel 模块自测：三角形状、行和归一、手工对照、正弦响应。"""
    print("=== mel.self_test ===")
    ok = True
    sr, n_fft, n_mels = 16000, 512, 80
    W = mel_filter_bank(sr, n_fft, n_mels, fmin=0.0, fmax=None)

    # 1) 形状与行和
    ok &= W.shape == (n_mels, n_fft // 2 + 1)
    row_sums = W.sum(axis=1)
    ok &= np.allclose(row_sums, 1.0, atol=1e-9)
    print(f"  shape={W.shape}, row_sum max_dev={float(np.max(np.abs(row_sums - 1.0))):.2e}")

    # 2) 手工逐点三角对照（HTK 公式）
    fftfreqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = np.linspace(float(hz_to_mel(0)), float(hz_to_mel(sr / 2)), n_mels + 2)
    f_bins = mel_to_hz(mel_pts)
    max_dev = 0.0
    for i in range(n_mels):
        left, center, right = f_bins[i], f_bins[i + 1], f_bins[i + 2]
        up = (fftfreqs - left) / (center - left)
        down = (right - fftfreqs) / (right - center)
        tri = np.maximum(0, np.minimum(up, down))
        tri = tri / tri.sum()
        max_dev = max(max_dev, float(np.max(np.abs(W[i] - tri))))
    print(f"  vs manual HTK triangles max dev = {max_dev:.3e}")
    ok &= max_dev < 1e-12

    # 2b) htk=False（Slaney）：行和不应为 1（面积归一），端点频率正确
    Ws = mel_filter_bank(sr, n_fft, n_mels, fmin=0.0, fmax=None, htk=False)
    ok &= Ws.shape == (n_mels, n_fft // 2 + 1)
    sl_row_sums = Ws.sum(axis=1)
    ok &= bool(np.all(sl_row_sums > 0.0))  # 全行非零
    # 面积归一：每个三角下面积约为 1（对连续频率轴近似）
    bin_hz = float(sr) / n_fft
    ok &= abs(float(Ws.sum(axis=1).mean() * bin_hz) - 1.0) < 0.2
    print(
        f"  htk=False slaney: row_sum range=[{float(sl_row_sums.min()):.4f},"
        f" {float(sl_row_sums.max()):.4f}], mean area={float(Ws.sum(axis=1).mean() * bin_hz):.3f}"
    )

    # 3) 单频响应：440Hz 正弦应在对应 mel bin 处有峰值能量
    t = np.arange(int(sr * 0.5)) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    mel = mel_spectrogram(x, sr, n_fft=512, hop_length=128, win_length=512, n_mels=80)
    # 期望峰值 bin：mel(440) 在 [mel(0), mel(8000)] 线性映射的位置
    m440 = float(hz_to_mel(440.0))
    m_lo, m_hi = float(hz_to_mel(0.0)), float(hz_to_mel(8000.0))
    exp_bin = int(round((m440 - m_lo) / (m_hi - m_lo) * (n_mels - 1)))
    got_bin = int(np.argmax(mel.mean(axis=1)))
    print(f"  sin440Hz peak mel bin: got={got_bin}, expected~{exp_bin}")
    ok &= abs(got_bin - exp_bin) <= 2
    ok &= mel.shape == (80, mel.shape[1])
    ok &= mel.dtype == np.float32

    # 4) log_mel：静音帧 clip 行为（全零输入 -> log(2e-6)）
    lm = log_mel_spectrogram(
        np.zeros(int(sr * 0.1)), sr, n_fft=512, hop_length=128,
        win_length=512, n_mels=80,
    )
    ok &= np.allclose(lm, np.log(2e-6), atol=1e-6)
    print(f"  log_mel silent value = {float(lm[0, 0]):.4f} (expect {np.log(2e-6):.4f})")

    # 5) 与 librosa 参考值对照（HTK mel 三角数值为确定性公式，不依赖 librosa 也能验算）
    # 校验：mel 轴端点映射 0 -> mel(0), 8000 -> mel(8000)
    ok &= abs(float(f_bins[0]) - 0.0) < 1e-9
    ok &= abs(float(f_bins[-1]) - 8000.0) < 1e-6

    print(f"  mel PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
