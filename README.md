<!-- ============================================================ -->
<!-- RVC-Vulkan：去 CUDA/PyTorch 的 RVC 变声（原生 Vulkan 后端）    -->
<!-- ============================================================ -->
<div align="center">

# RVC-Vulkan

**去 CUDA / PyTorch 的 RVC 变声推理（纯 numpy + 原生 Vulkan 计算后端）**

> ⚠️ **第三方个人项目声明**：本仓库是基于
> [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
> 的**独立移植与重写**，非 RVC 官方分支，与原项目无官方从属关系。
> 原版版权归 RVC-Project 及其贡献者所有；使用前请遵守原项目及依赖的许可协议（MIT）。

</div>

---

## 这是什么

本仓库是 **RVC（变声/音色转换）的完整纯 Python 移植**：推理与训练全部基于
numpy + 自研原生 Vulkan 计算后端（`engine/`，Zig 0.14 + GLSL，编译为
`rvc_core.dll`），**无需安装 torch / torchaudio / transformers / faiss /
librosa / parselmouth**。任何带 Vulkan 驱动的 GPU（AMD/Intel，NVIDIA 理论支持）
或纯 CPU 均可运行。

核心定位：让 **AMD / Intel 显卡在 Windows 下不通过兼容层原生运行 RVC**
（推理 + 训练），由自研 Vulkan 引擎驱动。**推理与训练语义与上游 RVC 对齐**；
本仓库**不包含**改动原版算法架构的实验性训练方法，也不包含任何模型权重。

## 特性

- **零 CUDA / 零 PyTorch**：推理与训练链路均不依赖任何 NVIDIA 专有栈。
- **纯 numpy 数值实现**：HuBERT 编码器、RMVPE / FCPE 基频提取、VITS 合成器
  均为 numpy 移植，语义与上游 RVC 对齐。
- **原生 Vulkan 加速**：`engine/` 提供 conv/matmul/attention/norm 等算子
  （GLSL compute shader），`runtime/vulkan_ops.py` 算子分派；无 Vulkan 时
  自动回退 numpy 后端。
- **推理**：CLI（单文件/目录）、HTTP API + WebUI、Flet 桌面版。
- **训练**：数据切分 → F0/HuBERT 特征提取 → 索引构建 → 训练（含图执行器、
  显存自适应等后端性能优化）→ 导出推理模型。
- **检索兼容**：支持自建 `.npz` / `.ivf.npz` 索引与 faiss `.index` 只读解析。
- **实时变声**：⚠️ **当前不可用**（历史实现已废弃，待重写），详见
  [docs/05-实时子系统.md](docs/05-实时子系统.md)。

## 推理链路（对齐上游 RVC）

```
输入音频
  → 高通滤波 / 静音-长音切分 / 分块（pipeline）
  → HuBERT 特征提取（runtime/models/hubert.py）
  → faiss 检索混合（runtime/retrieval.py / ivf_index.py）
  → 2× 上采样 + pitchff 保护掩码
  → F0 提取：pm（自相关）/ rmvpe / fcpe（runtime/dsp/f0.py + models）
  → VITS 合成（runtime/models/vits.py：enc_p → flow → GeneratorNSF）
  → 后处理（RMS 包络混合 change_rms）
输出音频
```

> 本仓库包含**推理 + 标准训练**链路。**不包含**改动原版 RVC 算法架构的实验性
> 训练方法（如对抗解耦等），也不包含任何模型权重（底模/自用模型）。
> 实时变声当前不可用（见 docs/05-实时子系统.md）。

## 快速开始

### 1. 环境

- Python 3.10+（推荐 3.11 / 3.12），Windows x64。
- 依赖见 `requirements-vulkan.txt`（纯 CPU/通用，无 CUDA）。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vulkan.txt
```

### 2. 下载推理权重

```powershell
python tools\download_weights.py --inference   # hubert_base + rmvpe.pt
python tools\download_fcpe.py                   # 可选：FCPE 权重
```

权重落到 `assets/hubert_base/`、`assets/rmvpe/rmvpe.pt`、`assets/fcpe/`。
变声模型（`.pth`）放入 `assets/weights/`；检索索引放入 `assets/indices/`。

### 3. 构建 Vulkan 引擎（可选；无则走 numpy 后端）

需要 Zig 0.14 + Vulkan SDK（glslc）：

```powershell
cd engine
zig build -Doptimize=ReleaseFast
# 产物：engine\zig-out\bin\rvc_core.dll
```

### 4. 离线变声（CLI）

```powershell
python -m runtime.cli --model my_model ^
    --input in.wav --output out.wav --pitch 0 --f0-method rmvpe
```

### 5. HTTP API + WebUI

```powershell
python -m runtime.api --port 7865 --open
# 浏览器打开 http://127.0.0.1:7865
```

## 目录结构

```
engine/            原生 Vulkan 计算引擎（Zig + GLSL）
  src/             engine.zig / recorder.zig / graph.zig / vk.zig ...
  shaders/         compute shader（conv/matmul/attention/norm/gating ...）
runtime/           纯 Python 运行时（零 torch）
  models/          hubert / rmvpe / fcpe / vits 推理图 + vits_train 训练图
  dsp/             IO / 重采样 / STFT / mel / f0
  train/           训练链（切分/特征提取/索引/训练循环/权重导出）
  pipeline.py      变声合成管线（对齐上游 infer/vc/pipeline.py）
  vc.py            高层封装（VC 类 + 索引查找）
  retrieval.py     检索层（numpy 暴力 L2 + 加权）
  ivf_index.py     自建 IVF 倒排索引
  faiss_reader.py  faiss .index 只读解析
  backend*.py      后端分派（vulkan / numpy）
  nn_backward.py   训练反向算子（numpy）
  graph_runner.py  训练图执行器（Vulkan 图化）
  cli.py           CLI 入口
  api.py           FastAPI + WebUI 后端
  realtime.py      实时变声占位（当前不可用，见 docs/05）
configs/           模型结构配置（v1/v2 采样率）
tools/             权重下载脚本 + 启动器模板
torch_compat.py    纯 Python 读取 .pth（无需 torch）
requirements-vulkan.txt
```

## 依赖策略（去 CUDA 红线）

**禁止**在运行时引入：`torch / torchaudio / torchvision / transformers /
faiss / torchfcpe / librosa / praat-parselmouth`。
`.pth` 权重由 `torch_compat.py` 纯 Python 解析，faiss 索引由
`runtime/faiss_reader.py` 纯 Python 解析。

## 与上游 RVC 的关系

| 项 | 说明 |
|---|---|
| 上游项目 | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)（MIT） |
| 本仓库 | 对该项目的**独立移植与重写**（de-CUDA 版）：保留其推理与标准训练语义，重写数值实现与计算后端 |
| 从属关系 | **无**。本仓库非 RVC 官方分支，不获官方背书，issues/PR 请勿提交到上游 |
| 代码来源 | `runtime/` 为本项目自主重写（非拷贝）；`engine/` 为本项目自研 Vulkan 引擎；算法语义与上游对齐 |
| 模型兼容 | 可直接加载上游训练产出的 `.pth`（推理格式与训练底模格式均支持） |
| 训练功能 | **含标准 RVC 训练链**（切分/特征/索引/训练/导出）；不含改动算法架构的实验性训练方法 |
| 实时功能 | **不含可用实现**（已废弃，占位模块会报错，待重写） |

## 免责声明

- 本软件以 **MIT 协议**“按原样”提供，**不附带任何明示或暗示的担保**。
- 使用者须自行确保对所处理音频/声音拥有合法权利，并遵守所在司法辖区的法律。
  **利用本软件伪造他人声音、进行诈骗、侵权或其他违法行为，责任由使用者自负。**
- 本项目与上游 RVC-Project 无任何从属或背书关系；上游作者不对本仓库负责。
- 本项目为个人独立移植，未获任何硬件厂商（AMD/Intel/NVIDIA）官方认证或支持。

## 实际能力与限制（如实声明）

**已验证可用的范围**

- 离线变声：CLI 单文件/目录、HTTP API `/infer`、`/batch`、WebUI/Flet 桌面版。
- 训练：数据切分 → F0/HuBERT 特征提取 → 索引构建 → 训练 → 导出推理模型
  （WebUI 训练页 / HTTP API / CLI）。
- 计算后端：numpy（纯 CPU，任何机器可用）/ Vulkan（原生 GPU 加速）。
- F0 提取：`pm`（自相关）/ `rmvpe` / `fcpe`。
- 检索：自建 `.npz` / `.ivf.npz` 索引；faiss `.index` 只读解析。

**已知限制**

- **平台**：仅验证 Windows x64；其他平台理论可行但未验证。
- **Vulkan 引擎**：workgroup 网格上限相关常量在特定 GPU（AMD Radeon Pro VII / Vega20）
  上实测标定，换卡可能触发越界（见 `runtime/vulkan_ops.py` 的 `_GRID_POINTS_MAX`）。
  其他设备请先用 `rvc.cmd check` 探测引擎 create/destroy。
- **后端选择**：当前为进程级环境变量（`RVC_BACKEND`），UI 未提供运行时切换。
- **不含实时**：实时流式变声当前不可用（已废弃，待重写）。
- **不含实验算法**：改动原版 RVC 算法架构的实验性训练方法不在本仓库。
- **不含模型权重**：底模、自用模型、测试模型均不在本仓库（用 `tools/download_weights.py` 下载或自备）。
- **精度**：numpy 路径为 float32；与上游 torch 参照的偏差已在移植过程中对齐，但不保证逐位一致。

## 贡献

欢迎 issue / PR。请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。贡献者名单见
[AUTHORS](AUTHORS)。版本变更见 [CHANGELOG.md](CHANGELOG.md)。

## 许可

本项目以 **MIT 协议**发布，见 [LICENSE](LICENSE)。上游 RVC 及依赖库协议见
[MIT协议暨相关引用库协议](MIT协议暨相关引用库协议)。本仓库为独立移植，
版权归其贡献者所有。
