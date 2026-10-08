# -*- coding: utf-8 -*-
"""运行时配置层：去 torch/cuda_graph/torch_directml 依赖的纯 Python 配置。

属性与 RVC 原版 Config 对齐（device/dtype/is_half/x_pad/x_query/x_center/x_max/
preprocess_per/json_config/python_cmd/listen_port/noparallel...），供移植后的
pipeline/cli/webui 直接使用。设备语义：
    - "cpu"：numpy 计算（默认，任何机器可用）
    - "vulkan"：Vulkan 加速（T24-T32 接入后可用；仍保留 numpy 算子作为回退）
精度固定 float32（numpy 无 fp16 加速收益，且权重已在加载时转 float32）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

__all__ = ["Config", "infer_device", "infer_dtype", "get_training_dtype"]

# 设备选择：默认 CPU；VULKAN_BACKEND 环境变量或自动探测（T32 后启用）
_VULKAN_AVAILABLE = False
try:
    import runtime._vulkan  # noqa: F401  # T32 后由 runtime._vulkan 模块提供探测

    _VULKAN_AVAILABLE = bool(getattr(runtime._vulkan, "available", False))
except Exception:
    _VULKAN_AVAILABLE = False

infer_device = os.environ.get("RVC_VULKAN_DEVICE", "vulkan" if _VULKAN_AVAILABLE else "cpu")
infer_dtype = "float32"  # numpy 统一 float32


def get_training_dtype() -> str:
    """训练 dtype：numpy 统一 float32（无 CUDA AMP）。"""
    return "float32"


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
MODEL_CONFIG_FILES = (
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
)


def _load_json_configs() -> dict:
    d = {}
    for cfg in MODEL_CONFIG_FILES:
        p = CONFIGS_DIR / cfg
        if p.is_file():
            d[cfg] = json.loads(p.read_text(encoding="utf-8"))
    return d


class Config:
    """无 torch 的运行时配置（单例）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = infer_device
        self.dtype = infer_dtype
        self.is_half = False
        self.cuda_graph = False
        self.gpu_name = None
        self.gpu_mem = None
        self.json_config = _load_json_configs()
        self.n_cpu = cpu_count() if os.environ.get("OPENBLAS_NUM_THREADS") is None else 0
        if self.n_cpu == 0:
            self.n_cpu = cpu_count()
        (self.python_cmd, self.listen_port, self.iscolab, self.noparallel,
         self.noautoopen, self.dml) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.0 if self.device == "cpu" else 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def arg_parse():
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen", action="store_true", help="Do not open in browser automatically"
        )
        parser.add_argument("--dml", action="store_true", help="(unused) kept for compat")
        # 只取已知参数：api/uvicorn 等宿主进程的其它 argv 必须被忽略，
        # 否则 argparse 会因"无法识别的参数"抛出 SystemExit 杀死宿主（实测 bug）。
        cmd_opts, _ = parser.parse_known_args()
        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865
        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
        )

    def device_config(self):
        """对齐原版：fp32 CPU 用 5G 显存配置；后续 vulkan 加速时沿用 fp32 配置。"""
        self.device = self.instead = "cpu" if self.device != "vulkan" else self.device
        self.dtype = "float32"
        self.is_half = False
        self.preprocess_per = 3.0
        x_pad, x_query, x_center, x_max = 1, 6, 38, 41  # 5G 显存/fp32 配置
        return x_pad, x_query, x_center, x_max


def get_config() -> Config:
    return Config()
