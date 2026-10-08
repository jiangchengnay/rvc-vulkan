# -*- coding: utf-8 -*-
"""后端分派器：可扩展的多后端注册表（numpy 纯 CPU / Vulkan GPU / 自定义扩展）。

设计（满足"保留纯 CPU 模式 + 扩展多种后端"）：
    - 内置后端：``numpy``（纯 CPU，恒可用）与 ``vulkan``（探测 rvc_core.dll +
      engine_create 成功才启用）。
    - 扩展点：``register_backend(name, module_path, label, probe)`` 注册任意新后端
      （如 OpenCL / CUDA / DirectML），模块需实现与 ``runtime.vulkan_ops`` 同名的
      算子函数（matmul/add/mul/relu/conv1d/conv_transpose1d/conv2d/embedding/
      add_inplace/mul_inplace/layer_norm/softmax/rmsnorm/...），未实现的算子自动
      回退 numpy。
    - 选择：``RVC_BACKEND`` 环境变量（"numpy"/"vulkan"/注册名/"auto"）；auto=
      探测优先 vulkan → 回退 numpy；强制名不可用时报错提示（默认回退 numpy）。
    - 回退链：算子级——后端模块缺某算子 → numpy 实现；后端级——探测失败 →
      numpy。

对外算子签名与旧版完全一致（调用方无感）；numpy 后端语义 = numpy 原生。
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np

__all__ = [
    "get_backend",
    "device_info",
    "available_backends",
    "register_backend",
    "matmul",
    "add",
    "mul",
    "relu",
    "conv1d",
    "conv_transpose1d",
    "conv2d",
    "embedding",
    "add_inplace",
    "mul_inplace",
    "layer_norm",
    "softmax",
    "rmsnorm",
    "conv1d_backward",
    "conv2d_backward",
]

_backend: str | None = None
_backend_reason: str | None = None

# 已注册后端：name -> {"module": 实现模块名, "label": 显示名}
_BACKENDS: dict = {"numpy": {"module": None, "label": "numpy (CPU)"}}


def register_backend(name: str, module_path: str, label: str = "", probe=None) -> None:
    """注册一个新后端（扩展点）。

    Args:
        name: 后端名（供 ``RVC_BACKEND`` 选择）。
        module_path: 实现模块（需提供与 vulkan_ops 同名的算子函数；缺省回退 numpy）。
        label: 显示名（``device_info`` 用）。
        probe: 可选探测函数（返回 bool）；缺省认为可用（import 成功即用）。
    """
    name = name.strip().lower()
    if not name or name in _BACKENDS:
        raise ValueError("后端名非法或已注册: %r" % name)
    _BACKENDS[name] = {"module": module_path, "label": label or name,
                       "probe": probe}


def _probe_vulkan() -> bool:
    """Vulkan 后端可用性探测：rvc_core.dll 存在且引擎可创建/销毁。"""
    try:
        from runtime import _vulkan  # noqa: PLC0415  # 惰性导入避免包初始化环

        if not _vulkan.has_dll or not _vulkan.available:
            return False
        h = _vulkan.engine_create()
        _vulkan.engine_destroy(h)
        return True
    except Exception:  # noqa: BLE001
        return False


# 注册 Vulkan 后端（扩展点）：修复"auto 探测返回 vulkan 但 _BACKENDS 未注册、
# _impl 查不到 → 全部算子静默走 numpy"的设计断链。注册后 get_backend()=="vulkan"
# 时 backend 分派经 runtime.vulkan_ops 同名算子（matmul/add/mul/relu/conv1d/
# conv_transpose1d/conv2d/embedding/add_inplace/mul_inplace/layer_norm/softmax/
# rmsnorm），未实现/签名不兼容的算子由 _impl 返回 None 自动回退 numpy。
# 模块级注册保证 _probe() 任意分支（auto / 强制名）都能在 _BACKENDS 查到。
try:
    register_backend("vulkan", "runtime.vulkan_ops", label="vulkan (GPU)",
                     probe=_probe_vulkan)
except ValueError:
    pass  # 已注册（如模块 reload）


def available_backends() -> list:
    """列出已注册且可用的后端名（探测惰性，不强制初始化）。"""
    out = ["numpy"]
    if get_backend() != "numpy":
        out.append(get_backend())
    for name in _BACKENDS:
        if name not in ("numpy",) and name != get_backend():
            out.append(name)
    return out


def _probe() -> str:
    """执行一次后端探测并返回结果（不缓存，由 ``get_backend`` 负责单例）。"""
    global _backend_reason
    forced = os.environ.get("RVC_BACKEND", "").strip().lower()
    if forced in ("", "auto"):
        forced = ""
    if forced == "numpy":
        return "numpy"
    if forced:
        if forced in _BACKENDS:
            info = _BACKENDS[forced]
            if info.get("probe") is not None:
                try:
                    if not info["probe"]():
                        _backend_reason = "后端 %s 探测未通过" % forced
                        return "numpy"
                except Exception as exc:  # noqa: BLE001
                    _backend_reason = "后端 %s 探测异常: %s" % (forced, exc)
                    return "numpy"
            try:
                importlib.import_module(info["module"])
                return forced
            except Exception as exc:  # noqa: BLE001
                _backend_reason = "后端 %s 模块加载失败: %s" % (forced, exc)
                return "numpy"
        _backend_reason = "未知后端 %r（可选: %s），回退 numpy" % (
            forced, ",".join(_BACKENDS))
        return "numpy"

    # auto：vulkan 优先，失败回退 numpy
    try:
        from runtime import _vulkan  # noqa: PLC0415  # 惰性导入避免包初始化环

        if not _vulkan.has_dll or not _vulkan.available:
            return "numpy"
        h = _vulkan.engine_create()
        _vulkan.engine_destroy(h)
        return "vulkan"
    except Exception as exc:  # noqa: BLE001
        _backend_reason = str(exc)
        return "numpy"


def get_backend() -> str:
    """返回选中后端名（单例）："numpy" / "vulkan" / 自定义注册名。"""
    global _backend
    if _backend is None:
        _backend = _probe()
        if _backend == "vulkan":
            print(
                f"[runtime.backend] 探测到 Vulkan 后端: {device_info()}",
                file=sys.stderr,
            )
        else:
            reason = _backend_reason or "默认使用 numpy 后端"
            print(
                f"[runtime.backend] 后端 = {_backend}（{reason}）",
                file=sys.stderr,
            )
    return _backend


def device_info() -> str:
    """返回可读的设备信息（含自定义后端 label）。"""
    be = get_backend()
    if be == "vulkan":
        try:
            from runtime import _vulkan  # noqa: PLC0415

            return f"vulkan: {_vulkan.device_name()}"
        except Exception as exc:  # noqa: BLE001
            return f"vulkan: (device_name 查询失败: {exc})"
    info = _BACKENDS.get(be)
    if info and info.get("label"):
        return info["label"]
    return "numpy (CPU)"


def _impl(op: str):
    """取当前后端模块中名为 ``op`` 的函数；无则返回 None（走 numpy 内置）。"""
    be = get_backend()
    info = _BACKENDS.get(be)
    if not info or not info.get("module"):
        return None
    try:
        mod = importlib.import_module(info["module"])
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, op, None)


# --------------------------------------------------------------------------
# 算子分派（签名与旧版完全一致；后端缺算子自动回退 numpy）
# --------------------------------------------------------------------------
def matmul(a: np.ndarray, b: np.ndarray, buf_a=None, buf_b=None) -> np.ndarray:
    fn = _impl("matmul")
    if fn is not None:
        return fn(a, b, buf_a=buf_a, buf_b=buf_b)
    return np.asarray(a) @ np.asarray(b)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fn = _impl("add")
    if fn is not None:
        return fn(a, b)
    return np.asarray(a) + np.asarray(b)


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fn = _impl("mul")
    if fn is not None:
        return fn(a, b)
    return np.asarray(a) * np.asarray(b)


def relu(a: np.ndarray) -> np.ndarray:
    fn = _impl("relu")
    if fn is not None:
        return fn(a)
    return np.maximum(np.asarray(a), 0.0)


def conv1d(x, w, b=None, stride=1, padding=0, dilation=1, buf_w=None, buf_b=None):
    fn = _impl("conv1d")
    if fn is not None:
        return fn(x, w, b, stride, padding, dilation, buf_w=buf_w, buf_b=buf_b)
    from runtime import nn as _nn  # noqa: PLC0415

    return _nn._conv1d_numpy(np.asarray(x), np.asarray(w), b, stride, padding, dilation)


def conv_transpose1d(x, w, b=None, stride=1, padding=0, output_padding=0, dilation=1,
                     buf_w=None, buf_b=None):
    fn = _impl("conv_transpose1d")
    if fn is not None:
        return fn(x, w, b, stride, padding, output_padding, dilation,
                  buf_w=buf_w, buf_b=buf_b)
    from runtime import nn as _nn  # noqa: PLC0415

    return _nn._conv_transpose1d_numpy(
        np.asarray(x), np.asarray(w), b, stride, padding, output_padding, dilation)


def conv2d(x, w, b=None, stride=1, padding=0, buf_w=None, buf_b=None):
    fn = _impl("conv2d")
    if fn is not None:
        return fn(x, w, b, stride, padding, buf_w=buf_w, buf_b=buf_b)
    from runtime import nn as _nn  # noqa: PLC0415

    return _nn._conv2d_numpy(np.asarray(x), np.asarray(w), b, stride, padding, 1)


def embedding(ids, table, buf_table=None):
    fn = _impl("embedding")
    if fn is not None:
        return fn(ids, table, buf_table=buf_table)
    from runtime import nn as _nn  # noqa: PLC0415

    return _nn._embedding_numpy(ids, table)


def add_inplace(a, b, buf_a=None, buf_b=None):
    fn = _impl("add_inplace")
    if fn is not None:
        return fn(a, b, buf_a=buf_a, buf_b=buf_b)
    return np.asarray(a) + np.asarray(b)


def mul_inplace(a, b, buf_a=None, buf_b=None):
    fn = _impl("mul_inplace")
    if fn is not None:
        return fn(a, b, buf_a=buf_a, buf_b=buf_b)
    return np.asarray(a) * np.asarray(b)


def layer_norm(x, gamma, beta, eps=1e-5, buf_gamma=None, buf_beta=None):
    fn = _impl("layer_norm")
    if fn is not None:
        return fn(x, gamma, beta, eps, buf_gamma=buf_gamma, buf_beta=buf_beta)
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def softmax(x, axis=-1):
    if axis == -1:
        fn = _impl("softmax")
        if fn is not None:
            return fn(x)
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def rmsnorm(x, gamma, eps=1e-5, buf_gamma=None):
    fn = _impl("rmsnorm")
    if fn is not None:
        return fn(x, gamma, eps, buf_gamma=buf_gamma)
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * gamma


# --------------------------------------------------------------------------
# 训练侧反向算子位（VK-02 / R-TRAIN-008：P-TRAIN-008 训练算子 GPU 化）
# 只新增反向接线：vulkan 后端有 ``conv1d_backward_gpu``（vulkan_ops 新增，
# 数值口径见 _diag/vk02_train_perf.md）时走 GPU，否则回退 numpy
# （runtime.nn_backward.conv1d_backward，f64 精确参考）。
# --------------------------------------------------------------------------
def conv1d_backward(x, w, grad_out, stride=1, padding=0, dilation=1, b=None,
                    br=None, buf_w=None, wtag="d"):
    fn = _impl("conv1d_backward_gpu")
    if fn is not None:
        return fn(x, w, grad_out, stride, padding, dilation, b=b, br=br,
                  buf_w=buf_w, wtag=wtag)
    from runtime import nn_backward as _nb  # noqa: PLC0415

    return _nb.conv1d_backward(x, w, grad_out, stride=stride, padding=padding,
                               dilation=dilation, b=b)


# --------------------------------------------------------------------------
# 阶段J（J6）：conv1d_groups 反向接线（DiscriminatorS bp）。vulkan 后端有
# ``conv1d_groups_backward_gpu``（vulkan_ops 新增，op24/25/26 一次调用全量
# 上传）时走 GPU；否则回退 numpy 组循环（同 conv1d_groups 原 bp）。
# --------------------------------------------------------------------------
def conv1d_groups_backward(x, w, grad_out, groups, stride=1, padding=0,
                           b=None, br=None):
    fn = _impl("conv1d_groups_backward_gpu")
    if fn is not None:
        return fn(x, w, grad_out, groups, stride, padding, b=b, br=br)
    from runtime.vulkan_ops import _cg_bwd_numpy_fallback  # noqa: PLC0415

    return _cg_bwd_numpy_fallback(x, w, grad_out, groups, stride, padding, b)


# --------------------------------------------------------------------------
# T2（R-TRAIN-008 / P-TRAIN-008 续）：conv2d 反向接线。vulkan 后端有
# ``conv2d_backward_gpu``（vulkan_ops 新增，数值口径见
# _diag/t2_conv2d_bwd_blueprint.md）时走 GPU，否则回退 numpy
# （runtime.nn_backward.conv2d_backward，f64 精确参考）。
# --------------------------------------------------------------------------
def conv2d_backward(x, w, grad_out, stride=1, padding=0, dilation=1, b=None,
                    br=None):
    fn = _impl("conv2d_backward_gpu")
    if fn is not None:
        return fn(x, w, grad_out, stride, padding, dilation, b=b, br=br)
    from runtime import nn_backward as _nb  # noqa: PLC0415

    return _nb.conv2d_backward(x, w, grad_out, stride=stride, padding=padding,
                               dilation=dilation, b=b)
