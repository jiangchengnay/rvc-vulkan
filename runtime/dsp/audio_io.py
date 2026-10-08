"""音频读写（替代 soundfile / librosa.load）。

- ``load_audio``：优先 soundfile（任意格式），回退标准库 wave 模块解析
  WAV（PCM16/PCM24/PCM32 与 32-bit IEEE float）；非 wav 且无 soundfile
  时抛出清晰异常提示安装 soundfile。
- ``write_audio``：优先 soundfile；回退 wave 模块写 16-bit PCM，
  float32（subtype='FLOAT'）时手动构造 IEEE float WAV。
"""

from __future__ import annotations

import os
import struct
import wave
from typing import Any

import numpy as np

from .resample import resample

try:  # pragma: no cover - 取决于运行环境
    import soundfile as _sf

    _HAS_SOUNDFILE = True
except ImportError:  # pragma: no cover
    _sf = None  # type: ignore[assignment]
    _HAS_SOUNDFILE = False

__all__ = ["load_audio", "write_audio", "has_soundfile"]


def has_soundfile() -> bool:
    """soundfile 是否可用（决定非 wav 格式支持）。"""
    return _HAS_SOUNDFILE


# ---------------------------------------------------------------------------
# WAV 解析回退（无 soundfile 时）
# ---------------------------------------------------------------------------


def _decode_pcm(raw: bytes, sampwidth: int) -> np.ndarray:
    """把 wave 模块读出的 PCM 字节解码为 float32 [-1, 1]。"""
    if sampwidth == 2:
        vals = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 3:
        u = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        vals = (u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)).astype(np.int64)
        vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
        vals = vals.astype(np.float64) / 8388608.0
    elif sampwidth == 4:
        vals = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(
            f"不支持的 PCM 位深: {sampwidth * 8}-bit（仅支持 16/24/32）"
        )
    return vals.astype(np.float32)


def _read_wav_stdlib(path: str) -> tuple[np.ndarray, int]:
    """用标准库 wave 模块解析 WAV（PCM16/24/32）。返回 [frames, ch] 或 1D。"""
    with wave.open(path, "rb") as w:
        n_ch, sampwidth, framerate, n_frames, _, _ = w.getparams()
        raw = w.readframes(n_frames)
    x = _decode_pcm(raw, sampwidth)
    if n_ch > 1:
        x = x.reshape(n_frames, n_ch)
    return x, framerate


def _read_wav_float32(path: str) -> tuple[np.ndarray, int]:
    """手动解析 32-bit IEEE float WAV（WAVE_FORMAT_IEEE_FLOAT=3）。

    wave 模块不支持该格式（会抛 unknown format），这里直接读 RIFF 块。
    """
    with open(path, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"不是有效的 RIFF/WAVE 文件: {path}")
        fmt: dict[str, Any] = {}
        data = b""
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = struct.unpack("<4sI", hdr)
            body = f.read(csize)
            if cid == b"fmt ":
                fmt["format"] = struct.unpack("<H", body[:2])[0]
                fmt["channels"] = struct.unpack("<H", body[2:4])[0]
                fmt["rate"] = struct.unpack("<I", body[4:8])[0]
                fmt["bits"] = struct.unpack("<H", body[14:16])[0]
            elif cid == b"data":
                data = body
            if csize % 2:
                f.read(1)
        if not fmt or fmt.get("format") != 3:
            raise ValueError(f"文件不是 IEEE float WAV: {path}")
        if fmt["bits"] != 32:
            raise ValueError(f"仅支持 32-bit float WAV，实际 {fmt['bits']}-bit")
    x = np.frombuffer(data, dtype="<f4").astype(np.float32)
    n_ch = int(fmt["channels"])
    if n_ch > 1:
        x = x.reshape(-1, n_ch)
    return x, int(fmt["rate"])


def _read_wav_fallback(path: str) -> tuple[np.ndarray, int]:
    """回退读取：wave 模块优先，格式异常（float32）转手动 RIFF 解析。"""
    try:
        return _read_wav_stdlib(path)
    except (wave.Error, EOFError, ValueError):
        return _read_wav_float32(path)


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def load_audio(
    path: str,
    sr: int | None = None,
    mono: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """加载音频。

    优先 soundfile（支持 mp3/flac/ogg/wav 等）；无 soundfile 时仅支持
    wav（PCM16/24/32 与 32-bit float）。之后按 mono 混音、按 sr 重采样。

    Args:
        path: 音频文件路径。
        sr: 目标采样率；None 保持原始采样率。
        mono: True 时多声道取均值混为单声道。
        dtype: 返回 dtype（默认 float32）。

    Returns:
        音频数组：mono 为 1D，否则为 [frames, ch] float 数组。
    """
    ext = os.path.splitext(path)[1].lower()
    if _HAS_SOUNDFILE:
        x, orig_sr = _sf.read(path, dtype="float32", always_2d=False)
    else:
        if ext != ".wav":
            raise RuntimeError(
                f"无法读取 {ext} 格式：未安装 soundfile。"
                f"请 `pip install soundfile`（非 wav 格式需要 libsndfile），"
                f"或将文件转换为 wav。"
            )
        x, orig_sr = _read_wav_fallback(path)

    if mono and x.ndim > 1:
        x = x.mean(axis=1)
    if sr is not None and orig_sr != sr:
        x = resample(x, orig_sr, sr)
    return np.asarray(x, dtype=dtype)


def _normalize_for_write(x: np.ndarray) -> np.ndarray:
    """把用户输入规范化为 soundfile 需要的 [n, ch] / [n] 布局。

    约定：1D 视为 mono；2D 视为 [ch, n]（第一维=声道，与 torch 习惯一致）。
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 2:
        return np.ascontiguousarray(x.T, dtype=np.float32)  # [ch, n] -> [n, ch]
    raise ValueError(f"write_audio 只支持 1D/2D，实际 {x.ndim}D")


def _write_wav_stdlib(path: str, x: np.ndarray, sr: int) -> None:
    """标准库写 16-bit PCM WAV（支持多声道交错）。x 为 [n, ch] 或 [n]。"""
    if x.ndim == 1:
        data = x[:, None]
    else:
        data = x
    n_ch = data.shape[1]
    pcm = np.clip(np.rint(data * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _write_wav_float32(path: str, x: np.ndarray, sr: int) -> None:
    """手动构造 32-bit IEEE float WAV。x 为 [n, ch] 或 [n]。"""
    if x.ndim == 1:
        x = x[:, None]
    n_ch = x.shape[1]
    data = np.asarray(x, dtype="<f4").tobytes()
    byte_rate = sr * n_ch * 4
    block_align = n_ch * 4
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 3, n_ch, sr, byte_rate, block_align, 32)
        + b"data"
        + struct.pack("<I", len(data))
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(data)


def write_audio(
    path: str,
    x: np.ndarray,
    sr: int,
    subtype: str | None = None,
) -> None:
    """写入音频。

    Args:
        path: 输出路径（.wav 或 soundfile 支持的其它格式）。
        x: 音频数组：1D mono，或 2D [ch, n]（第一维=声道）。
        sr: 采样率。
        subtype: 'PCM_16'（默认）或 'FLOAT'（float32）；soundfile 可用时
            支持更多 libsndfile subtype（如 'PCM_24'、'FLAC' 等）。
    """
    data = _normalize_for_write(x)
    if _HAS_SOUNDFILE:
        sub = subtype if subtype is not None else "PCM_16"
        _sf.write(path, data, sr, subtype=sub)
        return

    ext = os.path.splitext(path)[1].lower()
    if ext != ".wav":
        raise RuntimeError(
            f"无法写 {ext} 格式：未安装 soundfile。请 `pip install soundfile` 或改用 .wav。"
        )
    if subtype is None or subtype == "PCM_16":
        _write_wav_stdlib(path, data, sr)
    elif subtype == "FLOAT":
        _write_wav_float32(path, data, sr)
    else:
        raise RuntimeError(
            f"无 soundfile 时仅支持 subtype='PCM_16'/'FLOAT'，实际 {subtype!r}"
        )


def _self_test() -> bool:
    """audio_io 模块自测：写 wav -> 读回 -> 数值一致。"""
    print("=== audio_io.self_test ===")
    ok = True
    import tempfile

    sr = 16000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    tmp = tempfile.mkdtemp(prefix="dsp_audioio_")

    # 1) PCM_16 往返（优先走 soundfile；也会单独验证 wave 回退路径）
    p16 = os.path.join(tmp, "t16.wav")
    write_audio(p16, x, sr, subtype="PCM_16")
    y16 = load_audio(p16, sr=None, mono=True)
    err16 = float(np.max(np.abs(y16.astype(np.float64) - x.astype(np.float64))))
    print(f"  PCM_16 roundtrip max err = {err16:.3e}（量化步长 1/32768 = {1.0 / 32768:.2e}）")
    ok &= err16 <= 1.0 / 32768.0 + 1e-9
    ok &= y16.dtype == np.float32

    # 2) FLOAT 往返
    pf = os.path.join(tmp, "tf.wav")
    write_audio(pf, x, sr, subtype="FLOAT")
    yf = load_audio(pf, sr=None, mono=True)
    errf = float(np.max(np.abs(yf.astype(np.float64) - x.astype(np.float64))))
    print(f"  FLOAT roundtrip max err = {errf:.3e}")
    ok &= errf < 1e-7

    # 3) 无 soundfile 回退路径：直接用 wave 模块写、stdlib 读
    _write_wav_stdlib(p16, x, sr)
    yb = load_audio(p16, sr=None, mono=True)
    errb = float(np.max(np.abs(yb.astype(np.float64) - x.astype(np.float64))))
    print(f"  stdlib wave 回退 roundtrip max err = {errb:.3e}")
    ok &= errb <= 1.0 / 32768.0 + 1e-9

    # 4) 手动 float32 写入 + stdlib 回退读取
    _write_wav_float32(pf, x, sr)
    yb2 = load_audio(pf, sr=None, mono=True)
    errb2 = float(np.max(np.abs(yb2.astype(np.float64) - x.astype(np.float64))))
    print(f"  stdlib float32 回退 roundtrip max err = {errb2:.3e}")
    ok &= errb2 < 1e-7

    # 4.5) 24-bit PCM：用 wave 模块写 s3 文件，验证 _decode_pcm(sampwidth=3)
    p24 = os.path.join(tmp, "t24.wav")
    pcm24 = np.clip(np.rint(x.astype(np.float64) * 8388607.0), -8388608, 8388607).astype(np.int32)
    pcm24_bytes = b"".join(
        int(v).to_bytes(3, "little", signed=True) for v in pcm24
    )
    with wave.open(p24, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(pcm24_bytes)
    y24 = load_audio(p24, sr=None, mono=True)
    err24 = float(np.max(np.abs(y24.astype(np.float64) - x.astype(np.float64))))
    print(f"  PCM_24 roundtrip max err = {err24:.3e}（量化步长 1/2^23 = {1.0 / 8388608:.2e}）")
    ok &= err24 <= 1.0 / 8388608.0 + 1e-9

    # 5) 重采样参数：load_audio(sr=8000) 应降采样到 8000
    y8 = load_audio(p16, sr=8000, mono=True)
    ok &= y8.shape[0] == sr // 2
    rms8 = float(np.sqrt(np.mean(y8.astype(np.float64) ** 2)))
    print(f"  load sr=8000: len={y8.shape[0]}, rms={rms8:.4f}")
    ok &= 0.2 < rms8 < 0.5

    # 6) 多声道（soundfile 可用时）：2ch -> mono 均值
    if _HAS_SOUNDFILE:
        stereo = np.stack([x, 0.25 * x], axis=0)  # [2, n]
        ps = os.path.join(tmp, "st.wav")
        write_audio(ps, stereo, sr, subtype="PCM_16")
        m = load_audio(ps, sr=None, mono=True)
        errs = float(np.max(np.abs(m.astype(np.float64) - 0.625 * x.astype(np.float64))))
        print(f"  stereo->mono mean max err = {errs:.3e}")
        ok &= errs <= 1.0 / 32768.0 + 1e-9

    print(f"  audio_io PASS: {ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)