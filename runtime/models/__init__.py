# -*- coding: utf-8 -*-
"""runtime.models：RVC 模型推理图（hubert / rmvpe / fcpe / vits）。

当前可用：
    - rmvpe  RMVPE 基频（F0）提取（纯 numpy，替代 rmvpe.pt 的 torch 推理）
    - fcpe   FCPE（CFNaiveMelPE）基频提取（纯 numpy，替代 torchfcpe 推理）
    - hubert HuBERT 语音编码器（T21，纯 numpy）：v1(256d)/v2(768d) 特征
"""

from __future__ import annotations

from .fcpe import FCPE, load_fcpe
from .hubert import HubertEncoder, load_hubert_model
from .rmvpe import RMVPE, load_rmvpe

__all__ = [
    "RMVPE", "load_rmvpe",
    "FCPE", "load_fcpe",
    "HubertEncoder", "load_hubert_model",
]