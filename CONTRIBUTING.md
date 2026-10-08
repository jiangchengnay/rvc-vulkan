# 贡献指南

感谢参与！提交前请阅读以下约定。

## 红线（必须遵守）

1. **禁止引入 CUDA / PyTorch 栈**：运行时不得出现 `torch / torchaudio /
   torchvision / transformers / faiss / torchfcpe / librosa /
   praat-parselmouth` 的 import。
2. **数值实现用 numpy**；GPU 加速走 `engine/`（Vulkan）。
3. **保持推理语义与上游 RVC 对齐**；如需偏差，必须在注释与文档中说明。

## 开发环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements-vulkan.txt
```

## 自检（提交前请本地跑通）

```powershell
# 语法 + 导入冒烟（无需权重/GPU，自动回退 numpy 后端）
python -c "import runtime.api, runtime.cli, runtime.vc, runtime.pipeline"
# DSP 自检
python -m runtime.dsp.tests_smoke
```

## 提交规范

- 提交信息使用简洁的中文或英文祈使句，可带类型前缀（feat/fix/docs/refactor）。
- 一次提交聚焦一个改动；不要混入无关格式化。
- 新增文件带头部说明（模块用途）。

## 报告问题

请在 issue 中附上：操作系统、Python 版本、计算后端（numpy/vulkan）、
复现步骤与报错日志。**请勿将本仓库的问题提交到上游 RVC 项目。**
