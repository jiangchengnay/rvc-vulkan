# -*- coding: utf-8 -*-
"""音频 IO 兼容层：提供与 RVC `infer/audio.py` 同名同语义的接口，
但实现基于 runtime/dsp（soundfile/numpy），零 torch/torchaudio/ffmpeg-python 依赖。

移植说明（原 infer/audio.py → 本模块）：
    load_audio(file, sr, force_mono=True)   → soundfile/标准库加载 + fft 重采样
    resample_audio(...)                     → runtime/dsp/resample
    wav2 / transcode_audio_file             → 依赖 av 包（可选）或外部 ffmpeg
    clean_path(...)                         → 同原版
"""

from __future__ import annotations

import os
import platform

import numpy as np

from .dsp.resample import resample as _resample

__all__ = [
    "load_audio",
    "resample_audio",
    "wav2",
    "transcode_audio_file",
    "clean_path",
    "AUDIO_LOAD_BACKEND",
]


def clean_path(path_str) -> str:
    """去除 Windows 路径首尾的空格/引号/换行（与原版一致）。"""
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


AUDIO_LOAD_BACKEND = "soundfile"  # 本实现的后端标识


def load_audio(file, sr, force_mono=True) -> np.ndarray:
    """加载 float32 音频；mono 返回 [T]，保留声道返回 [C, T]。

    与原版不同：原版 ffmpeg 解码任意格式；本版优先 soundfile（支持 wav/flac/
    ogg/opus 等），wav 可用标准库回退，其余格式抛提示。返回后统一重采样到 sr。
    """
    file = clean_path(os.fspath(file))
    try:
        import soundfile as _sf
    except ImportError:
        _sf = None

    ext = os.path.splitext(file)[1].lower()
    if _sf is not None:
        # P1-009：损坏/非法音频文件统一包装为可读异常——soundfile 原始异常
        # （LibsndfileError 等）直接上抛会在 Web 场景透出堆栈文本；包装后由
        # api 各路由的 except 转 HTTPException（worker 不崩，全局 handler 兜底）。
        try:
            x, source_sr = _sf.read(file, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "音频文件无法解码（文件损坏或格式不支持）: %s: %s"
                % (os.path.basename(file), exc)) from None
    elif ext == ".wav":
        x, source_sr = _read_wav_fallback(file)
    else:
        raise RuntimeError(
            "无法读取 %s 格式：需要 soundfile（python -m pip install soundfile，纯 CPU）"
            % (ext or "未知格式")
        )
    if force_mono and x.ndim > 1:
        x = x.mean(axis=1)
    if not force_mono and x.ndim == 1:
        x = x[None, :]
    elif force_mono:
        x = np.asarray(x).flatten()
    if source_sr != sr:
        x = _resample(x, source_sr, sr, method="fft")
    return np.asarray(x, dtype=np.float32)


def _read_wav_fallback(path) -> tuple[np.ndarray, int]:
    """无 soundfile 时的 wav 读取（PCM16/24/32 与 float32）。"""
    import wave

    try:
        with wave.open(path, "rb") as w:
            n_ch, sampwidth, framerate, n_frames, _, _ = w.getparams()
            raw = w.readframes(n_frames)
        if sampwidth == 1:
            u = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            x = (u - 128.0) / 128.0
        elif sampwidth == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            i24 = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                   | (b[:, 2].astype(np.int32) << 16))
            i24 = np.where(i24 & 0x800000, i24 - 0x1000000, i24)
            x = i24.astype(np.float32) / 8388608.0
        else:
            raise ValueError("仅支持 16/24/32-bit PCM")
        if n_ch > 1:
            x = x.reshape(n_frames, n_ch)
        return x, framerate
    except (wave.Error, EOFError, ValueError):
        raise RuntimeError("无法解析 wav 文件（非 PCM 或格式异常）: %s" % path)


def resample_audio(audio, source_sr, target_sr, force_mono=False, res_type=None):
    """重采样 channel-first 音频 [C, T]（或 [T]），返回 numpy。"""
    mono = force_mono or audio.ndim == 1
    if mono:
        x = np.asarray(audio, dtype=np.float32).flatten()
        if x.ndim == 1:
            x = x[None, :]
    else:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
    method = "linear" if res_type else "fft"
    out = np.stack([_resample(ch, source_sr, target_sr, method=method) for ch in x], axis=0)
    return out[0].flatten() if mono else out


def _wav2_av(input_path, output_path, format):  # pragma: no cover - 需 av 包
    import av

    inp = av.open(input_path, "r")
    try:
        if format == "m4a":
            format = "mp4"
        out = av.open(output_path, "w", format=format)
        try:
            if format == "ogg":
                format = "libvorbis"
            if format == "mp4":
                format = "aac"
            if not inp.streams.audio:
                raise ValueError("Input contains no audio stream")
            input_stream = inp.streams.audio[0]
            source_rate = input_stream.codec_context.sample_rate
            ostream = (
                out.add_stream(format, rate=source_rate)
                if source_rate
                else out.add_stream(format)
            )
            source_channels = input_stream.codec_context.channels
            if source_channels == 1:
                ostream.layout = "mono"
            elif source_channels == 2:
                ostream.layout = "stereo"
            for frame in inp.decode(input_stream):
                for p in ostream.encode(frame):
                    out.mux(p)
            for p in ostream.encode(None):
                out.mux(p)
        finally:
            out.close()
    finally:
        inp.close()


def wav2(i, o, format):
    """把输入音频转码为指定容器格式（m4a/ogg/mp4/wav 等）。"""
    i = clean_path(os.fspath(i))
    o = os.fspath(o)
    try:
        _wav2_av(i, o, format)
    except ImportError:
        raise RuntimeError(
            "转码 %s 需要安装 av 包：python -m pip install av（纯 CPU，无 CUDA 依赖）；"
            "或先转成 wav 再处理。" % format
        )


def transcode_audio_file(input_path, output_path, format):
    """转码并清理失败产物（与原版行为一致）。"""
    output_path = os.fspath(output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    try:
        wav2(input_path, output_path, format)
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Audio transcoding produced no output: %s" % output_path)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise