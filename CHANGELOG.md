# 变更日志

本项目遵循语义化版本（`主.次.修订`）。日期格式 YYYY-MM-DD。

## [0.0.1] - 2026-10-09

首个公开版本：RVC（变声/音色转换）的去 CUDA 移植 —— 纯 numpy + 原生 Vulkan 后端。

### 新增
- **原生 Vulkan 计算引擎** `engine/`（Zig 0.14 + GLSL compute shader，编译为
  `rvc_core.dll`）。
- **推理**（纯 numpy）：HuBERT 编码器、RMVPE / FCPE 基频提取、VITS 合成器；
  变声管线 `runtime/pipeline.py` / `runtime/vc.py`（语义对齐上游）。
- **标准训练链**：数据切分（`preprocess`）→ F0/HuBERT 特征提取 →
  索引构建（`train_index`）→ 训练循环（`train.py` + `vits_train.py` +
  `nn_backward.py`）→ 推理模型导出（`process_ckpt`）。
- **检索**：`runtime/retrieval.py`（暴力 L2 加权）、`runtime/ivf_index.py`
  （IVF 倒排）、`runtime/faiss_reader.py` / `faiss_writer.py`（faiss `.index`
  读/写，零 faiss 依赖）。
- **入口**：CLI（`runtime/cli.py`）、HTTP API + WebUI（`runtime/api.py` +
  `runtime/static/`）、Flet 桌面版（`runtime/gui_flet.py`）。
- **后端分派**：`runtime/backend.py`（vulkan / numpy，环境变量 `RVC_BACKEND`）。
- **后端性能优化**：训练图执行器（`graph_runner.py`）、显存自适应
  （`adaptive.py`）、多级最优筛选（`smart.py`）、捕获重放（`_poc/`）。
- `torch_compat.py`：纯 Python 读取 `.pth`（无需 PyTorch）。

### 说明
- 本仓库**不含模型权重**（底模/自用模型）；用 `tools/download_weights.py` 下载或自备。
- 本仓库**不含**改动原版 RVC 算法架构的实验性训练方法。
- **实时变声当前不可用**（历史实现已废弃，待重写）；`runtime/realtime.py` 为占位模块。

### 已知限制
- 仅验证 Windows x64。
- Vulkan 引擎的网格上限相关常量在特定 GPU 上标定，换卡需自行验证。
