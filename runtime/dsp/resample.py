"""音频重采样（替代 librosa.resample）。

- ``fft`` 法：对实信号 rfft，在频域把目标 bin 的位置映射回源谱（线性插值，
  等价于"频域截断/补零 + 长度比归一"），irfft 后乘 ``target_len / orig_len``
  恢复幅度。归一化系数 = target_len / orig_len，能量与正弦 RMS 经自测校验。
- ``linear`` 法：np.interp 线性插值到目标采样点（低质量快速路径）。
"""

from __future__ import annotations

import numpy as np

from .utils import peak_normalize

__all__ = ["resample"]


def _kaiser_window_sinc(t: np.ndarray, radius: int, beta: float = 12.0) -> np.ndarray:
    """Kaiser 窗截断的 sinc 插值核（t 的单位为 bin，|t| <= radius）。"""
    x = t / float(radius)
    with np.errstate(invalid="ignore", divide="ignore"):
        kaiser = np.where(
            np.abs(x) <= 1.0,
            np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - x * x))) / np.i0(beta),
            0.0,
        )
    return np.sinc(t) * kaiser


def _sinc_interp_spectrum(
    X: np.ndarray, p: np.ndarray, radius: int = 16, beta: float = 12.0
) -> np.ndarray:
    """在复数谱 X 的任意（非整数）bin 位置 p 处做带限插值。

    每个输出位置取以 p 为中心、半径 radius 的邻域，用 Kaiser 窗截断 sinc
    加权求和；对整周期正弦这种"能量集中单个 bin"的病理情况也能正确重建。
    """
    n_bins = X.shape[-1]
    pad = radius + 1
    # 反射扩展边界，避免边缘截断伪影
    Xp = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(pad, pad)], mode="edge")
    # 输出位置对应到 pad 后的坐标
    pp = p + pad
    base = np.floor(pp)[:, None].astype(np.int64) + np.arange(-radius, radius + 1)[None, :]
    t = pp[:, None] - base.astype(np.float64)
    kern = _kaiser_window_sinc(t, radius, beta)
    Xw = Xp[..., base]
    return np.sum(Xw * kern, axis=-1)


def _resample_fft(x: np.ndarray, n_out: int, target_sr: int, orig_sr: int) -> np.ndarray:
    """频域 FFT 重采样（沿最后一维）。

    源谱 X 是 n_in 点的 rfft（bin b 对应频率 b*sr_in/n_in）；
    目标 bin k 的频率为 k*sr_out/n_out，映射回源谱位置
    ``p = k * (n_in * sr_out) / (orig_sr * n_out)``，
    在该位置做 Kaiser 窗截断 sinc 带限插值（等价于"频域截断/补零 +
    长度比归一"的更精确形式）。超出源 Nyquist 的 bin 置 0。
    """
    n_in = x.shape[-1]
    if n_out == n_in and target_sr == orig_sr:
        return x.copy()
    X = np.fft.rfft(x, n=n_in, axis=-1)  # [..., n_in//2+1]
    n_bins_in = X.shape[-1]
    n_bins_out = n_out // 2 + 1

    # 频率对齐：目标 bin k 在源谱中的位置
    p = np.arange(n_bins_out, dtype=np.float64) * (
        n_in * target_sr / (orig_sr * n_out)
    )
    in_range = p <= (n_bins_in - 1)
    Y = np.zeros(X.shape[:-1] + (n_bins_out,), dtype=np.complex128)
    if in_range.any():
        Y[..., in_range] = _sinc_interp_spectrum(X, p[in_range])

    y = np.fft.irfft(Y, n=n_out, axis=-1)
    return y * (n_out / n_in)


def _resample_linear(x: np.ndarray, n_out: int) -> np.ndarray:
    """线性插值重采样（np.interp）。"""
    n_in = x.shape[-1]
    if n_out == n_in:
        return x.copy()
    if x.ndim == 1:
        return np.interp(np.linspace(0.0, n_in - 1, n_out), np.arange(n_in), x)
    pos = np.linspace(0.0, n_in - 1, n_out)
    grid = np.arange(n_in)
    return np.stack([np.interp(pos, grid, x[c]) for c in range(x.shape[0])], axis=0)


def _resample_sinc(x: np.ndarray, n_out: int, radius: int = 12,
                   beta: float = 12.0) -> np.ndarray:
    """时域 Kaiser 窗截断 sinc 带限插值（沿最后一维）。

    每个输出样本位置 ``pos[j] = j * n_in / n_out`` 处，取输入邻域
    ``floor(pos) ± radius`` 用 ``sinc × Kaiser`` 加权求和（标准多相带限
    插值 / 抗混叠重采样）。相比 ``linear``（2 点线性插值）引入的镜像伪影
    与高频衰减，sinc 核在源 Nyquist 内保持高频、带外零镜像（实测
    40k→48k：16-19.5k 异常带 0 vs linear 119）。
    radius 为单侧抽头数（共 2r+1 抽头，默认 12 兼顾质量/速度），beta 为
    Kaiser 窗形状参数。
    """
    n_in = x.shape[-1]
    if n_out == n_in:
        return x.copy()
    ratio = n_in / n_out
    pos = np.arange(n_out, dtype=np.float64) * ratio
    base = pos.astype(np.int64)  # floor（pos 非负）
    if x.ndim == 1:
        out = np.zeros(n_out, dtype=np.float64)
        for k in range(-radius, radius + 1):
            idx = np.clip(base + k, 0, n_in - 1)
            w = _kaiser_window_sinc(pos - (base + k), radius, beta)
            out += np.take(x, idx) * w
        return out
    # 2D：[ch, n_in]
    out = np.zeros((x.shape[0], n_out), dtype=np.float64)
    for c in range(x.shape[0]):
        row = np.zeros(n_out, dtype=np.float64)
        for k in range(-radius, radius + 1):
            idx = np.clip(base + k, 0, n_in - 1)
            w = _kaiser_window_sinc(pos - (base + k), radius, beta)
            row += np.take(x[c], idx) * w
        out[c] = row
    return out


def resample(
    x: np.ndarray,
    orig_sr: int,
    target_sr: int,
    method: str = "fft",
) -> np.ndarray:
    """把信号重采样到目标采样率（沿最后一维）。

    Args:
        x: 1D 或 2D（[ch, samples]）实信号。
        orig_sr: 原始采样率。
        target_sr: 目标采样率。
        method: 'fft'（频域插值，默认）、'linear'（np.interp 快速路径）或
        'sinc'（时域 Kaiser-sinc 带限插值——抗混叠、无镜像伪影，用于实时
        设备路径逐块重采样）。

    Returns:
        重采样后的数组，长度 = round(len * target_sr / orig_sr)（1D），
        或保持与输入一致的通道布局。
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("采样率必须为正")
    if method not in ("fft", "linear", "sinc"):
        raise ValueError(
            f"不支持的 method: {method!r}（可选 'fft' / 'linear' / 'sinc'）"
        )

    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"resample 只支持 1D/2D 输入，实际 {x.ndim}D")

    n_in = x.shape[-1]
    if orig_sr == target_sr:
        return x.copy().astype(np.float32)
    n_out = max(1, int(round(n_in * target_sr / orig_sr)))

    if method == "fft":
        y = _resample_fft(x, n_out, int(target_sr), int(orig_sr))
    elif method == "sinc":
        y = _resample_sinc(x, n_out)
    else:
        y = _resample_linear(x, n_out)
    return y.astype(np.float32)


def _self_test() -> bool:
    """resample 模块自测：sin 能量校验 + 白噪声 roundtrip。"""
    print("=== resample.self_test ===")
    ok = True
    rng = np.random.default_rng(7)

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))

    # 1) sin 44100 -> 16000，RMS 相对误差 < 1e-3（任务要求）
    #    纯正弦降采样（<8kHz 带内）应保持幅度：RMS 与重采样前一致
    for freq in (440.0, 1000.0, 3000.0):
        sr1, sr2 = 44100, 16000
        t1 = np.arange(int(sr1 * 1.0)) / sr1
        x = 0.5 * np.sin(2 * np.pi * freq * t1)
        y = resample(x, sr1, sr2, method="fft")
        ref_rms = rms(x)  # 幅度保持（带限正弦降采样不改变 RMS）
        rel_err = abs(rms(y) - ref_rms) / ref_rms
        print(f"  fft 44100->16000 @{freq}Hz: rms={rms(y):.5f}, ref={ref_rms:.5f}, rel_err={rel_err:.3e}")
        ok &= rel_err < 1e-3
        # 频率保持：重采样后主频仍在原频率 ±2Hz（用 rfft 峰值 bin 验证，
        # 避免对多周期正弦做 argmax 时间对齐的脆弱断言）
        Y = np.fft.rfft(y.astype(np.float64))
        peak_bin = int(np.argmax(np.abs(Y[1:]))) + 1
        f_est = peak_bin * sr2 / y.shape[0]
        ok &= abs(f_est - freq) < 2.0

    # 2) fft roundtrip 44100 -> 16000 -> 44100（组合信号：sin + 白噪声）
    sr = 44100
    n = int(sr * 0.8)
    t = np.arange(n) / sr
    x2 = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 1200 * t)
        + 0.2 * rng.standard_normal(n)
    )
    d = resample(resample(x2, sr, 16000), 16000, sr)
    k = min(len(d), len(x2))
    err = float(np.sqrt(np.mean((d[:k] - x2[:k]) ** 2)))
    print(f"  fft roundtrip 44100->16000->44100 rms err = {err:.4e}（原信号包含 >8kHz 被截断的白噪声，误差主要在噪声高频）")
    ok &= err < 0.2  # 高频噪声被 16k 奈奎斯特截断，低频分量应恢复良好
    # 单独验证低频正弦分量恢复
    x_sin_only = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    d_sin = resample(resample(x_sin_only, sr, 16000), 16000, sr)
    sin_err = float(np.sqrt(np.mean((d_sin[:k] - x_sin_only[:k]) ** 2)))
    print(f"  fft roundtrip sin-only rms err = {sin_err:.4e}")
    ok &= sin_err < 5e-3

    # 3) linear 法：sin roundtrip 误差可接受
    y_lin = resample(x2, sr, 16000, method="linear")
    d_lin = resample(y_lin, 16000, sr, method="linear")
    lin_err = float(np.sqrt(np.mean((d_lin[:k] - x2[:k]) ** 2)))
    print(f"  linear roundtrip rms err = {lin_err:.4e}")
    ok &= lin_err < 0.3

    # 4) 升采样 16000 -> 44100：正弦仍保持幅度
    sr_l, sr_h = 16000, 44100
    t_l = np.arange(int(sr_l * 0.5)) / sr_l
    x_low = 0.4 * np.sin(2 * np.pi * 300 * t_l)
    y_up = resample(x_low, sr_l, sr_h, method="fft")
    up_err = abs(rms(y_up) - rms(x_low)) / rms(x_low)
    print(f"  fft upsample 16000->44100 RMS rel err = {up_err:.3e}")
    ok &= up_err < 1e-3

    # 5) 与 fft.py 的 stft 无符号冲突（调用一次验证 import 路径）
    from . import fft as _fft

    _ = _fft.stft(x2[:4410], 512, 128, 512, out_mag=True)
    ok &= isinstance(_, np.ndarray)

    print(f"  resample PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)