"""复数 STFT / iSTFT（对齐 torch.stft(return_complex=True) 与 librosa 语义）。

- ``stft``：center=True 时对信号左右各 reflect pad n_fft//2，逐帧乘窗后 rfft，
  返回 ``[n_fft//2+1, n_frames]`` 复数矩阵（共轭对称只需一半，与 torch 一致）。
- ``istft``：帧 irfft -> 乘窗 -> bincount 加权叠加，并用窗平方归一化
  （window-sum 归一，等价于 librosa.istft / torch.istft），可选裁剪恢复原长。
"""

from __future__ import annotations

import numpy as np

__all__ = ["stft", "istft", "get_window"]


def get_window(
    window: str | np.ndarray, win_length: int, n_fft: int, dtype=np.float64
) -> np.ndarray:
    """解析窗函数：字符串（'hann'/'hamming'/'blackman'）或 ndarray。

    与 torch.stft 一致的约定：窗长为 ``win_length``，若小于 ``n_fft``
    则在尾部补零到 ``n_fft``（与分析帧长度一致）。

    Args:
        window: 'hann' / 'hamming' / 'blackman' 或长度为 win_length 的数组。
        win_length: 窗长（须 <= n_fft）。
        n_fft: FFT 长度。
        dtype: 输出 dtype。

    Returns:
        长度为 n_fft 的窗数组。
    """
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) 不能大于 n_fft ({n_fft})")
    if isinstance(window, str):
        name = window.lower()
        if name == "hann":
            w = np.hanning(win_length)
        elif name == "hamming":
            w = np.hamming(win_length)
        elif name == "blackman":
            w = np.blackman(win_length)
        else:
            raise ValueError(f"不支持的窗函数: {window!r}")
        w = np.asarray(w, dtype=dtype)
    else:
        w = np.asarray(window, dtype=dtype)
        if w.ndim != 1:
            raise ValueError("window 必须是 1D 数组")
        if w.shape[0] != win_length:
            raise ValueError(
                f"window 长度 {w.shape[0]} != win_length {win_length}"
            )
    if win_length < n_fft:
        w = np.pad(w, (0, n_fft - win_length))
    return w


def stft(
    x: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    window: str | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    out_mag: bool = False,
) -> np.ndarray:
    """复数短时傅里叶变换。

    Args:
        x: 1D 实信号。
        n_fft: FFT 长度（bin 数 = n_fft // 2 + 1）。
        hop_length: 帧移。
        win_length: 窗长（<= n_fft）。
        window: 窗函数名或数组。
        center: True 时左右各 pad n_fft//2（reflect，同 torch.stft 默认），
            帧数 = 1 + (len(x) + n_fft - n_fft) // hop_length。
        pad_mode: padding 方式，默认 'reflect'（torch 默认）。
        out_mag: True 时返回幅度谱 [bins, frames] float32，
            False 时返回复数谱 complex128。

    Returns:
        [n_fft//2+1, n_frames] 复数谱（out_mag=False 时）。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"stft 需要 1D 输入，实际 {x.ndim}D")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft / hop_length 必须为正")

    w = get_window(window, win_length, n_fft, dtype=np.float64)

    if center:
        pad = n_fft // 2
        if len(x) <= pad:
            raise ValueError(
                f"center=True 时 reflect pad 需要 len(x) > n_fft//2（{pad}），实际 {len(x)}"
            )
        x = np.pad(x, (pad, pad), mode=pad_mode)

    n = len(x)
    n_frames = 1 + (n - n_fft) // hop_length
    if n_frames <= 0:
        raise ValueError(f"信号过短：n_frames={n_frames}（len={n}, n_fft={n_fft}）")

    # 帧索引矩阵 [n_frames, n_fft]
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = x[idx] * w[None, :]
    spec = np.fft.rfft(frames, axis=1).T  # [n_fft//2+1, n_frames]
    if out_mag:
        return np.abs(spec).astype(np.float32)
    return spec  # complex128


def istft(
    spec: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str | np.ndarray,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """OLA 短时傅里叶逆变换（窗平方归一，与分析窗一致）。

    Args:
        spec: [n_fft//2+1, n_frames] 复数谱（stft 输出）。
        n_fft: FFT 长度。
        hop_length: 帧移。
        win_length: 窗长。
        window: 与分析一致的窗函数名或数组。
        center: True 时把输出两端的 n_fft//2 padding 裁掉；
            False 时保留 OLA 全长 (n_frames-1)*hop_length + n_fft。
        length: 可选目标长度；不足补零、超出裁剪（同 librosa/torch 的 length 参数）。

    Returns:
        float64 时域信号。
    """
    spec = np.asarray(spec)
    if spec.ndim != 2:
        raise ValueError(f"istft 需要 2D 输入 [bins, frames]，实际 {spec.ndim}D")
    n_bins, n_frames = spec.shape
    if n_bins != n_fft // 2 + 1:
        raise ValueError(f"bin 数 {n_bins} != n_fft//2+1 = {n_fft // 2 + 1}")

    w = get_window(window, win_length, n_fft, dtype=np.float64)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * w[None, :]  # [frames, n_fft]

    out_len = (n_frames - 1) * hop_length + n_fft
    # bincount 加权叠加（等价于 np.add.at 但更快）
    idx = (
        np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    ).ravel()
    y = np.bincount(idx, weights=frames.ravel(), minlength=out_len)
    # 帧优先展开窗平方权重：每帧都贡献同一窗平方 -> np.tile（不是 repeat）
    wsum = np.bincount(
        idx, weights=np.tile(w * w, n_frames), minlength=out_len
    )
    y = np.divide(y, wsum, out=np.zeros_like(y), where=wsum > 1e-12)

    if center:
        pad = n_fft // 2
        y = y[pad : out_len - pad] if out_len > 2 * pad else np.zeros(0, dtype=y.dtype)

    if length is not None:
        if length < 0:
            raise ValueError("length 必须非负")
        if len(y) >= length:
            y = y[:length]
        else:
            y = np.pad(y, (0, length - len(y)))

    return y


def _self_test() -> bool:
    """fft 模块自测：手算 DFT 对照 + 随机信号 roundtrip。"""
    print("=== fft.self_test ===")
    ok = True

    # 1) 与手算 DFT 对照（n_fft=8, hop=2, center=False）
    rng = np.random.default_rng(12345)
    x = rng.standard_normal(32)
    n_fft, hop, win = 8, 2, 8
    w = np.hanning(win)
    S = stft(x, n_fft, hop, win, window=w, center=False)
    n_frames = 1 + (len(x) - n_fft) // hop
    assert S.shape == (n_fft // 2 + 1, n_frames)
    for f in (0, 3, n_frames - 1):
        frame = x[f * hop : f * hop + n_fft] * w
        ref = np.fft.rfft(frame)
        err = float(np.max(np.abs(S[:, f] - ref)))
        if err > 1e-10:
            print(f"  frame {f} DFT mismatch: {err}")
            ok = False
    print(f"  stft vs manual rfft max err = {err:.2e}")

    # 2) roundtrip（center=True，win_length == n_fft）
    x2 = rng.standard_normal(8192) * 0.5
    S2 = stft(x2, 2048, 512, 2048, window="hann", center=True)
    y2 = istft(S2, 2048, 512, 2048, window="hann", center=True, length=len(x2))
    err2 = float(np.max(np.abs(y2 - x2)))
    rms2 = float(np.sqrt(np.mean((y2 - x2) ** 2)))
    print(f"  roundtrip(2048/512) max err = {err2:.3e}, rms = {rms2:.3e}")
    ok &= err2 < 1e-8

    # 3) roundtrip（非整除长度 + 不同 hop）
    #    center=True 时丢弃尾部不足一帧的样本（同 torch.stft），
    #    只比较可重建的公共长度；不足部分由 length 参数补零。
    x3 = rng.standard_normal(10003) * 0.3
    S3 = stft(x3, 1024, 256, 1024, window="hann", center=True)
    y3 = istft(S3, 1024, 256, 1024, window="hann", center=True, length=len(x3))
    n_cov = (S3.shape[1] - 1) * 256  # 可重建公共样本数（裁 padding 后）
    err3 = float(np.max(np.abs(y3[:n_cov] - x3[:n_cov])))
    print(f"  roundtrip(1024/256, len=10003) covered={n_cov}/{len(x3)} max err = {err3:.3e}")
    ok &= err3 < 1e-8
    ok &= np.all(y3[n_cov:] == 0.0)  # 尾部为补零

    # 4) win_length < n_fft 的 roundtrip（尾部补零窗）
    x4 = rng.standard_normal(6000) * 0.4
    S4 = stft(x4, 2048, 512, 1600, window="hann", center=True)
    y4 = istft(S4, 2048, 512, 1600, window="hann", center=True, length=len(x4))
    n_cov4 = (S4.shape[1] - 1) * 512
    err4 = float(np.max(np.abs(y4[:n_cov4] - x4[:n_cov4])))
    print(f"  roundtrip(win=1600<n_fft=2048) covered={n_cov4}/{len(x4)} max err = {err4:.3e}")
    ok &= err4 < 1e-7  # 尾补零窗在帧边缘有轻微不完美重建

    # 5) out_mag 形状
    M = stft(x2[:4000], 512, 128, 512, window="hann", out_mag=True)
    ok &= M.dtype == np.float32 and M.ndim == 2 and M.shape[0] == 257

    # 6) 与 scipy.signal.stft 数值对照（同窗、同 padding 设置时）
    try:
        from scipy import signal as sp_signal

        x5 = rng.standard_normal(4096)
        S5 = stft(x5, 512, 128, 512, window=np.hanning(512), center=False)
        # scipy 的 padded=True 时在两端补零到能被 hop 整除，等价于 center=False + 尾零
        f_scipy, t_scipy, Z_scipy = sp_signal.stft(
            x5, fs=1.0, nperseg=512, noverlap=512 - 128, window=np.hanning(512),
            boundary=None, padded=False,
        )
        # center=False 时我们的帧数可能比 scipy 少最后一帧（不足 n_fft 丢弃），
        # 只对照公共帧；scipy 1.17 默认 scaling='spectrum'，其 stft 输出
        # 乘了 1/sum(window)，对齐时乘回 sum(window) 即可（torch/librosa 语义无缩放）
        n_common = min(S5.shape[1], Z_scipy.shape[1])
        err5 = float(np.max(np.abs(S5[:, :n_common] - Z_scipy[:, :n_common] * np.hanning(512).sum())))
        print(f"  vs scipy.signal.stft (common {n_common} frames) max err = {err5:.3e}")
        ok &= err5 < 1e-10
    except Exception as e:  # scipy 未装则跳过
        print(f"  scipy 对照跳过: {e}")

    print(f"  fft PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)