"""公共音频 DSP 工具：mel 刻度换算、简单 VAD、峰值归一化。

本模块是 RVC 去 CUDA 化移植的音频底层的一部分，仅依赖 numpy，
替代 librosa / torchaudio / parselmouth / scipy 的音频侧工具函数。

mel 刻度统一使用 HTK 公式（与 RVC 原项目 f0_to_coarse / get_f0_post 一致）：
    hz_to_mel(f) = 1127 * ln(1 + f / 700)
    mel_to_hz(m) = 700 * (exp(m / 1127) - 1)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hz_to_mel",
    "mel_to_hz",
    "vad_simple",
    "peak_normalize",
]


def hz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
    """HTK 公式：赫兹频率转 mel 刻度。

    Args:
        f: 频率（Hz），标量或任意形状数组。

    Returns:
        对应的 mel 值（与输入同形状）。
    """
    f = np.asarray(f, dtype=np.float64)
    return 1127.0 * np.log1p(f / 700.0)


def mel_to_hz(m: float | np.ndarray) -> float | np.ndarray:
    """HTK 公式：mel 刻度转赫兹频率。

    Args:
        m: mel 值，标量或任意形状数组。

    Returns:
        对应的频率（Hz，与输入同形状）。
    """
    m = np.asarray(m, dtype=np.float64)
    return 700.0 * (np.expm1(m / 1127.0))


def vad_simple(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -42.0,
    hop: float = 0.015,
    min_silence: float = 0.4,
) -> np.ndarray:
    """基础 VAD：按帧 RMS 的 dB 值判定语音/静音（绝对阈值）。

    用于后续语音切分的前置判断，返回逐帧布尔标记（True=语音）。
    帧长为 30ms，帧移为 ``hop`` 秒；RMS 低于 ``threshold_db`` 判为静音。
    随后把持续时长小于 ``min_silence`` 秒的"短静音"合并回语音段，
    避免把字词间的短暂停顿误切成独立片段。

    Args:
        x: 1D 单声道音频（float）。
        sr: 采样率（Hz）。
        threshold_db: 静音判定阈值（dB，相对 1.0 满刻度）。
        hop: 帧移（秒）。
        min_silence: 静音段合并阈值（秒）：短于该时长的静音段视为语音。

    Returns:
        bool 数组，长度 = 帧数，True 表示该帧为语音帧。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"vad_simple 需要 1D 输入，实际 {x.ndim}D")

    hop_samples = max(1, int(round(sr * hop)))
    frame_samples = max(1, int(round(sr * 0.03)))
    n_frames = int(np.ceil(len(x) / hop_samples))

    active = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        s = i * hop_samples
        seg = x[s : s + frame_samples]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg * seg)))
        db = 20.0 * np.log10(rms + 1e-12)
        active[i] = db > threshold_db

    # 合并短于 min_silence 的静音段
    min_silence_frames = max(1, int(round(min_silence / hop)))
    if min_silence_frames > 1:
        # 找出所有连续静音段（run of False），长度 < min_silence_frames 的置 True
        runs: list[tuple[int, int]] = []
        start = None
        for i, v in enumerate(active):
            if not v and start is None:
                start = i
            elif v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, n_frames))
        for lo, hi in runs:
            if hi - lo < min_silence_frames:
                active[lo:hi] = True

    return active


def peak_normalize(x: np.ndarray, target: float = 0.99) -> np.ndarray:
    """峰值归一化：把信号峰值缩放到 ``target``。

    Args:
        x: 任意形状的音频数组。
        target: 目标峰值幅度（默认 0.99）。

    Returns:
        归一化后的数组（新数组）；全零输入原样返回。
    """
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x.copy()
    return x * (float(target) / peak)


def _self_test() -> bool:
    """utils 模块自测：mel 刻度往返、VAD 判定、峰值归一化。"""
    print("=== utils.self_test ===")
    ok = True

    # 1) mel 刻度：已知点 + 往返
    hz = np.array([0.0, 50.0, 440.0, 1000.0, 1100.0, 8000.0])
    mel = hz_to_mel(hz)
    back = mel_to_hz(mel)
    roundtrip_err = float(np.max(np.abs(back - hz)))
    print(f"  hz_to_mel/mel_to_hz roundtrip max abs err = {roundtrip_err:.3e}")
    ok &= roundtrip_err < 1e-9
    # 0 Hz -> 0 mel
    ok &= float(hz_to_mel(0.0)) == 0.0
    # 440 Hz 的 mel 参考值 = 1127*ln(1+440/700)
    ref440 = 1127.0 * np.log(1.0 + 440.0 / 700.0)
    ok &= abs(float(hz_to_mel(440.0)) - ref440) < 1e-9

    # 2) VAD：前 0.6s 正弦（> -42dB），后 0.4s 数字零
    sr = 16000
    t = np.arange(int(sr * 1.0)) / sr
    x = np.concatenate([0.3 * np.sin(2 * np.pi * 220 * t[: int(sr * 0.6)]),
                        np.zeros(int(sr * 0.4))])
    vad = vad_simple(x, sr)
    n_frames = vad.shape[0]
    # 静音帧应在末尾；语音帧应在前部（帧长为 30ms，有重叠偏移，取容差）
    last_active = int(np.max(np.where(vad)[0])) if vad.any() else -1
    first_inactive = int(np.min(np.where(~vad)[0])) if (~vad).any() else n_frames
    print(f"  vad: {n_frames} frames, last_active={last_active}, first_inactive={first_inactive}")
    ok &= last_active >= 0 and last_active < n_frames * 0.75
    ok &= first_inactive > n_frames * 0.4

    # 3) 峰值归一化
    y = np.array([0.0, -2.0, 1.0])
    yn = peak_normalize(y, target=0.5)
    ok &= abs(float(np.max(np.abs(yn))) - 0.5) < 1e-12
    ok &= np.allclose(np.zeros(3), peak_normalize(np.zeros(3)))

    print(f"  utils PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
