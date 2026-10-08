"""runtime.dsp —— RVC 去 CUDA 化移植的纯 numpy 音频 DSP 底层。

替代 RVC 原项目中的 librosa / torchaudio / parselmouth / scipy 音频侧依赖。
本包内禁止 import torch，仅依赖 numpy（音频读写可选 soundfile）。

模块：
    - fft: 复数 STFT / iSTFT（对齐 torch.stft / librosa 语义）
    - resample: fft / linear 重采样（替代 librosa.resample）
    - mel: mel 滤波器组与 mel 谱（HTK 刻度，slaney 三角布局）
    - f0: 自相关基频提取与后处理（替代 parselmouth）
    - audio_io: 音频读写（替代 soundfile / librosa.load）
    - utils: mel 刻度换算、简单 VAD、峰值归一化
"""

from __future__ import annotations

from .audio_io import has_soundfile, load_audio, write_audio
from .f0 import (
    f0_autocorrelation,
    f0_to_coarse,
    interp_f0,
    median_filter_pitch,
)
from .fft import istft, stft
from .mel import log_mel_spectrogram, mel_filter_bank, mel_spectrogram
from .resample import resample
from .utils import hz_to_mel, mel_to_hz, peak_normalize, vad_simple

__version__ = "0.0.1"

__all__ = [
    # 音频 IO
    "load_audio",
    "write_audio",
    "has_soundfile",
    # STFT
    "stft",
    "istft",
    # 重采样
    "resample",
    # mel
    "mel_filter_bank",
    "mel_spectrogram",
    "log_mel_spectrogram",
    # f0
    "f0_autocorrelation",
    "median_filter_pitch",
    "interp_f0",
    "f0_to_coarse",
    # utils
    "hz_to_mel",
    "mel_to_hz",
    "vad_simple",
    "peak_normalize",
]
