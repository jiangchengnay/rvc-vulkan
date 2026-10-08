"""基频（F0）提取与后处理（替代 parselmouth to_pitch_ac）。

- ``f0_autocorrelation``：自相关法（rfft 快速自相关），帧窗 ~45ms 取奇数，
  在 [1/f0_max, 1/f0_min] 的 lag 区间内找归一化自相关峰值，
  超过 ``voicing_threshold`` 判为 voiced，否则输出 0（unvoiced）。
- ``median_filter_pitch``：voiced 段中值滤波（unvoiced 保持 0）。
- ``interp_f0``：线性插值填充无声帧（同 RVC 原实现 np.interp 用法）。
- ``f0_to_coarse``：mel 刻度（1127*ln(1+f/700)）量化到 [1, f0_bin-1]，
  0（无声）保留为 0，对齐 RVC 的 get_f0_post / f0_to_coarse 语义。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "f0_autocorrelation",
    "median_filter_pitch",
    "interp_f0",
    "f0_to_coarse",
]


def f0_autocorrelation(
    x: np.ndarray,
    sr: int,
    hop: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    voicing_threshold: float = 0.6,
    time_step: float = 0.01,
) -> np.ndarray:
    """自相关法基频提取（类 parselmouth to_pitch_ac）。

    帧长取 ~45ms（取奇数），帧中心对齐网格 ``i * hop``（或 time_step*sr），
    逐帧：去均值 -> 加 Hann 窗 -> rfft 自相关（|rfft|^2 -> irfft）->
    在 lag ∈ [round(sr/f0_max), round(sr/f0_min)] 内找归一化自相关峰值 r/r[0]，
    超过 voicing_threshold 判 voiced，f0 = sr / lag_best；否则 0。

    Args:
        x: 1D 单声道信号。
        sr: 采样率（Hz）。
        hop: 帧移（样本数，默认 160 = 10ms @16k）。
        f0_min / f0_max: 基频搜索范围（Hz）。
        voicing_threshold: 归一化自相关清浊判定阈值。
        time_step: 帧移（秒）；非 None 时优先使用 round(sr*time_step)。

    Returns:
        float32 数组，长度 = ceil(len(x) / hop)，无声帧 = 0。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"f0_autocorrelation 需要 1D 输入，实际 {x.ndim}D")
    if time_step is not None and time_step > 0:
        hop = int(round(sr * time_step))
    hop = max(1, int(hop))

    # 窗长：45ms 取奇数（若 f0_min 很低则至少容纳 3 个周期）
    win = int(round(sr * 0.045))
    if f0_min > 0 and win < 3 * sr / f0_min:
        win = int(round(3 * sr / f0_min))
    if win % 2 == 0:
        win += 1

    n_frames = int(np.ceil(len(x) / hop))
    f0 = np.zeros(n_frames, dtype=np.float32)

    lag_min = max(1, int(round(sr / f0_max)))
    lag_max = int(round(sr / f0_min))
    if lag_max >= win:
        lag_max = win - 1
    if lag_min > lag_max:
        raise ValueError("f0_max/f0_min 区间在窗内无有效 lag")

    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    w = np.hanning(win)  # periodic 性不强求，symmetric 即可

    half = win // 2
    for i in range(n_frames):
        c = i * hop  # 帧中心
        s = c - half
        e = s + win
        if s >= 0 and e <= len(x):
            seg = x[s:e]
        else:
            lo, hi = max(s, 0), min(e, len(x))
            seg = np.pad(x[lo:hi], (max(0, -s), max(0, e - len(x))))
        if seg.size != win:
            continue
        seg = seg - seg.mean()
        if float(np.max(np.abs(seg))) < 1e-7:
            continue  # 静音帧 -> 0
        X = np.fft.rfft(seg * w, n=nfft)
        r = np.fft.irfft(X * np.conj(X), n=nfft)[:win]  # 线性自相关（nfft>=2win-1）
        r0 = r[0]
        if r0 <= 0.0:
            continue
        rn = r / r0
        seg_r = rn[lag_min : lag_max + 1]
        k = int(np.argmax(seg_r)) + lag_min
        if seg_r[k - lag_min] > voicing_threshold:
            f0[i] = sr / k
    return f0


def median_filter_pitch(f0: np.ndarray, window: int = 7) -> np.ndarray:
    """对 voiced 帧做中值滤波；unvoiced（0）保持 0。

    对每个非零帧取其邻域内非零值的中值，避免把静音误判为有音。

    Args:
        f0: f0 序列（float）。
        window: 滤波窗长（奇数）。

    Returns:
        与输入同长度的 float32 数组。
    """
    f0 = np.asarray(f0, dtype=np.float32)
    if f0.ndim != 1:
        raise ValueError("median_filter_pitch 需要 1D 输入")
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = f0.copy()
    for i in range(len(f0)):
        if f0[i] == 0.0:
            continue
        lo, hi = max(0, i - half), min(len(f0), i + half + 1)
        nz = f0[lo:hi]
        nz = nz[nz > 0]
        if nz.size:
            out[i] = float(np.median(nz))
    return out


def interp_f0(f0: np.ndarray) -> np.ndarray:
    """线性插值填充无声帧（0 值位置），与 RVC 原实现 np.interp 用法一致。

    Args:
        f0: f0 序列（float，0 表示无声）。

    Returns:
        填充后的 float32 序列（原 voiced 帧不变）。
    """
    f0 = np.asarray(f0, dtype=np.float32)
    if f0.ndim != 1:
        raise ValueError("interp_f0 需要 1D 输入")
    out = f0.copy()
    uv = f0 <= 0
    if uv.all() or not uv.any():
        return out
    idx = np.arange(len(f0))
    voiced_idx = np.where(~uv)[0]
    out[uv] = np.interp(idx[uv], voiced_idx, f0[voiced_idx])
    return out


def f0_to_coarse(
    f0: np.ndarray,
    f0_bin: int = 256,
    f0_max: float = 1100.0,
    f0_min: float = 50.0,
) -> np.ndarray:
    """f0 转 coarse 编码（mel 刻度量化，对齐 RVC get_f0_post 语义）。

    ``coarse = rint((sl - sl_min)/(sl_max - sl_min) * (f0_bin-2)) + 1``，
    其中 ``sl = 1127*ln(1 + f/700)``；结果 clamp 到 [1, f0_bin-1]，
    无声帧（f0<=0）保持 0。

    Args:
        f0: f0 序列（Hz，0 表示无声）。
        f0_bin: coarse bin 总数（默认 256）。
        f0_max / f0_min: 频率范围（Hz）。

    Returns:
        int32 数组，范围 [0, f0_bin-1]。
    """
    f0 = np.asarray(f0, dtype=np.float64)
    sl = 1127.0 * np.log1p(f0 / 700.0)
    sl_min = 1127.0 * np.log1p(f0_min / 700.0)
    sl_max = 1127.0 * np.log1p(f0_max / 700.0)
    coarse = np.rint((sl - sl_min) / (sl_max - sl_min) * (f0_bin - 2)) + 1
    coarse = np.clip(coarse, 1, f0_bin - 1)
    coarse[f0 <= 0] = 0
    return coarse.astype(np.int32)


def _self_test() -> bool:
    """f0 模块自测：220Hz 正弦 + 白噪声段。"""
    print("=== f0.self_test ===")
    ok = True
    sr = 16000
    dur = 1.0  # 前 0.5s 220Hz 正弦，后 0.5s 白噪声
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)
    x = np.concatenate(
        [
            0.5 * np.sin(2 * np.pi * 220.0 * t[: n // 2]),
            0.5 * rng.standard_normal(n // 2),
        ]
    )

    f0 = f0_autocorrelation(x, sr, hop=160)
    n_frames = len(f0)
    print(f"  f0 frames={n_frames}, voiced={int(np.sum(f0 > 0))}, unvoiced={int(np.sum(f0 == 0))}")

    # 正弦段（中间帧，避开窗跨段边缘）：f0 ∈ [215, 225]
    voiced_vals = f0[5:40]
    in_range = np.sum((voiced_vals > 215) & (voiced_vals < 225))
    print(f"  sin 段 f0 在 220±5Hz 的帧数: {in_range}/{len(voiced_vals)}")
    ok &= in_range >= len(voiced_vals) * 0.9

    # 噪声段：绝大多数应为 unvoiced
    noise_vals = f0[55:95]
    zero_frac = float(np.mean(noise_vals == 0))
    print(f"  噪声段 unvoiced 比例: {zero_frac:.2f}")
    ok &= zero_frac >= 0.8

    # median filter 不改变 0
    f0m = median_filter_pitch(f0, window=7)
    ok &= np.all((f0m == 0) == (f0 == 0))
    ok &= f0m.dtype == np.float32

    # interp 填充后无 0
    f0i = interp_f0(f0)
    ok &= not (f0i <= 0).any()
    ok &= np.allclose(f0i[f0 > 0], f0[f0 > 0], atol=1e-6)

    # f0_to_coarse：220Hz 映射在合理 bin；0 -> 0；范围 [0, 255]
    c = f0_to_coarse(np.array([0.0, 50.0, 220.0, 440.0, 1100.0, 5000.0], dtype=np.float32))
    print(f"  f0_to_coarse([0,50,220,440,1100,5000]) = {c.tolist()}")
    ok &= c[0] == 0
    ok &= c.min() >= 0 and c.max() <= 255
    ok &= 0 < c[2] < 255 and c[2] < c[3]
    # 边界：f0_max(1100) 及以上的频率 clamp 到 [1, f0_bin-1] 上界 255；f0_min 附近为 1
    ok &= c[4] == 255 and c[5] == 255
    ok &= c[1] >= 1 and c[1] <= 5  # 50Hz 接近下界 -> 小 bin
    # int32 dtype
    ok &= c.dtype == np.int32

    # 自洽：sin 220Hz 提取 + interp + coarse 全链路
    chain = f0_to_coarse(interp_f0(f0))
    ok &= chain.dtype == np.int32 and chain.shape == (n_frames,)

    print(f"  f0 PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)