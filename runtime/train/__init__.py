# -*- coding: utf-8 -*-
"""runtime.train —— RVC 训练数据准备（纯 numpy runtime 移植）。

替代原 train/preprocess.py（数据切分）与 train/dataset/extract_f0.py
（F0 特征提取）：零 torch/librosa/parselmouth 依赖，
仅依赖 numpy / scipy.signal / soundfile（可选）。

模块：
    - preprocess   PreProcess 数据切分（对齐 train/preprocess.py）
    - extract_f0   FeatureInput / extract_feature_dir（对齐 train/dataset/extract_f0.py）
"""

__version__ = "0.0.1"