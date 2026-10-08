# -*- coding: utf-8 -*-
"""RVC-Vulkan 运行时包：不依赖 PyTorch/CUDA 的 RVC 变声推理实现。

模块划分：
    retrieval   特征检索层（faiss 替代，numpy 暴力 L2 + (1/d)^2 加权）
    ivf_index   自建 IVF 倒排索引
    dsp         音频 DSP（IO/重采样/STFT/mel/f0，替代 librosa/scipy 音频侧）
    nn          神经网络算子库（conv/linear/norm/attention/gru，numpy 实现）
    models      hubert / rmvpe / fcpe / vits 模型推理图
    pipeline    变声合成管线（对齐上游 infer/vc/pipeline.py 行为）
    vc          推理高层封装（VC 类 + 索引查找）
    backend     算子级后端分派（vulkan / numpy）

本包对外约定：
- 张量一律用 numpy.ndarray（float32）；
- 权重从 .pth（torch_compat.load_pth）或 .safetensors（runtime.safetensors）加载为
  dict[str, numpy.ndarray]，键名与 RVC checkpoint 完全一致；
- 所有模块禁止 import torch / transformers / faiss / librosa。

backend 相关导出（T31）：
- ``get_backend()``：探测并返回 "vulkan" / "numpy"（进程内单例）；
- ``device_info()``：可读设备信息（"numpy (CPU)" 或 "vulkan: <device>"）。
"""

__version__ = "0.0.1"


def get_backend() -> str:
    """返回当前计算后端："vulkan"（GPU，rvc_core.dll 可用）或 "numpy"。"""
    from runtime.backend import get_backend as _get_backend  # noqa: PLC0415

    return _get_backend()


def device_info() -> str:
    """返回可读设备描述，如 ``"vulkan: Radeon Pro VII"`` 或 ``"numpy (CPU)"``。"""
    from runtime.backend import device_info as _device_info  # noqa: PLC0415

    return _device_info()
