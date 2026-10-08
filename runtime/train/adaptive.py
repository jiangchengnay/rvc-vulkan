# -*- coding: utf-8 -*-
"""T4 自适应模式切换（训练显存自适应策略）。

功能
----
训练启动时检测 Vulkan 可用显存，按阈值自动选择运行模式：

- **正常模式**（显存 > 4GB）：batch=3、池开启、判别器双缓冲 + **图化门控全开**
  （T7b：注入 ``RVC_TRAIN_GRAPH=1/GRAPH_BWD=1/ENCQ=1/FLOW=1``，与 4G 分支一致，
  DEC/ASYNC 保持默认 0）。图化是确定性执行路径（与 T6 全图化基线逐位一致），
  相对无图化默认有 ~1e-3 级微小数值差异（T6 已记录）；>4G 机器走自适应 auto 时自动受益；
- **4G 模式**（显存 ≤ 4GB）：batch=1 + 输入池关闭（RVC_TRAIN_IN_POOL=0）+
  输出池关闭（RVC_OUT_NO_POOL=1）。判别器单缓冲为代码级改造
  （graph_runner/vits_train 边界，本专项其他子代理负责），暂记待办，
  4G 模式先行「配置降级」即可把显存压到 ~2.5-2.8GB（见 _diag/mem_composition_summary.md）。

显存检测链（detect_vram_gb）
---------------------------
1. ``RVC_ADAPTIVE_GB`` env：显式覆盖检测值（测试/特殊环境用）；
2. Vulkan API（``vulkan-1.dll``）：``vkEnumeratePhysicalDevices`` +
   ``vkGetPhysicalDeviceMemoryProperties``，取 ``VK_MEMORY_HEAP_DEVICE_LOCAL_BIT``
   heap 的最大 size（= 物理显存总量，跨多适配器取最大者，通用检测）；
3. DXGI（``dxgi.dll``，Windows）：``IDXGIAdapter::GetDesc`` 的
   ``DedicatedVideoMemory``（某些环境 DXGI 工厂创建失败 E_NOINTERFACE 时不可用）；
4. 均失败 → 返回 None：调用方按「正常模式 + 告警」回退（与现状一致）。

回退 / 开关
-----------
- ``RVC_ADAPTIVE=0``：完全关闭——不检测、不注入任何 env，行为与现状一致；
- ``RVC_ADAPTIVE`` 未设或 ``1``：自动模式（默认开启，本机 16GB 走正常模式，
  自动继承图化门控，T7b）；
- ``RVC_ADAPTIVE_MODE=normal|low4g``：强制指定模式（调试/CI 用）；
- ``RVC_ADAPTIVE_MONITOR=1``：训练循环内每 ``log_interval`` 步查询一次
  DXGI 当前显存预算，接近物理显存时打印告警（默认关，零开销）。

数值路径：本模块只读 env / 写 env，不改任何算子/数值逻辑。图化 env
（GRAPH/BWD/ENCQ/FLOW）切换为确定性图化执行路径——与同配置（图化）逐位一致，
相对无图化默认 ~1e-3 级差异（T6 已记录，非随机噪声）；池 env 仅省 mem_alloc
分配不改变数值（见调研 #4）。``RVC_ADAPTIVE=0`` 时完全不注入（== 现状）。
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

__all__ = [
    "detect_vram_gb",
    "query_vram_dxgi",
    "query_vram_vulkan",
    "adaptive_enabled",
    "choose_mode",
    "apply_adaptive_mode",
    "monitor_step",
    "MODE_NORMAL",
    "MODE_4G",
    "VRAM_THRESHOLD_GB",
]

MODE_NORMAL = "normal"   # >4G：batch=3 + 池开 + 判别器双缓冲 + 图化门控全开（T7b）
MODE_4G = "low4g"        # ≤4G：batch=1 + 池关 + 判别器单缓冲（待办）
VRAM_THRESHOLD_GB = 4.0

# 4G 模式注入的 env（全部为引擎/训练既有门控，见 _diag/t0_baseline.md §2）
_4G_ENV = {
    "RVC_TRAIN_BATCH_SIZE": "1",     # train.py L710 读取
    "RVC_TRAIN_IN_POOL": "0",        # vulkan_ops.py 输入池
    "RVC_OUT_NO_POOL": "1",          # vulkan_ops.py 输出池关闭（默认池开）
}

# 正常模式注入的 env（T7b：与 low4g 分支一致的图化门控；DEC/ASYNC 保持默认 0）
_NORMAL_ENV = {
    "RVC_TRAIN_GRAPH": "1",          # 判别器 fwd 图化（t0_baseline.md §2）
    "RVC_TRAIN_GRAPH_BWD": "1",      # 判别器 bwd 图化
    "RVC_TRAIN_GRAPH_ENCQ": "1",     # 编码器队列图化
    "RVC_TRAIN_GRAPH_FLOW": "1",     # flow 图化
}


# ---------------------------------------------------------------------------
# Vulkan 显存查询（vulkan-1.dll，纯 ctypes；枚举物理设备 + DEVICE_LOCAL heap）
# ---------------------------------------------------------------------------
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_API_VERSION_1_0 = (1 << 22) | (0 << 12) | 0  # 0x00400000
VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001
VK_MAX_MEMORY_TYPES = 32
VK_MAX_MEMORY_HEAPS = 16


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_void_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_void_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class _VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class _VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", _VkMemoryType * VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", _VkMemoryHeap * VK_MAX_MEMORY_HEAPS),
    ]


def query_vram_vulkan() -> dict | None:
    """Vulkan 枚举物理设备，返回 DEVICE_LOCAL heap 总量（字节）；失败 None。

    跨多适配器取 DEVICE_LOCAL heap 最大的物理设备（独立 GPU 优先）。
    """
    try:
        vk = ctypes.WinDLL("vulkan-1.dll")
        vk_create = vk.vkCreateInstance
        vk_create.restype = ctypes.c_int32
        vk_create.argtypes = [ctypes.POINTER(_VkInstanceCreateInfo),
                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vk_destroy = vk.vkDestroyInstance
        vk_destroy.restype = None
        vk_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        vk_enum = vk.vkEnumeratePhysicalDevices
        vk_enum.restype = ctypes.c_int32
        vk_enum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                            ctypes.POINTER(ctypes.c_void_p)]
        vk_memprops = vk.vkGetPhysicalDeviceMemoryProperties
        vk_memprops.restype = None
        vk_memprops.argtypes = [ctypes.c_void_p,
                                ctypes.POINTER(_VkPhysicalDeviceMemoryProperties)]

        app = _VkApplicationInfo(VK_STRUCTURE_TYPE_APPLICATION_INFO, None,
                                 None, 1, None, 1, VK_API_VERSION_1_0)
        ici = _VkInstanceCreateInfo(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                    None, 0, ctypes.addressof(app), 0, None,
                                    0, None)
        instance = ctypes.c_void_p()
        if vk_create(ctypes.byref(ici), None, ctypes.byref(instance)) != 0 \
                or not instance:
            return None
        try:
            count = ctypes.c_uint32(0)
            if vk_enum(instance, ctypes.byref(count), None) != 0:
                return None
            devices = (ctypes.c_void_p * max(1, count.value))()
            if vk_enum(instance, ctypes.byref(count), devices) != 0:
                return None
            best = 0
            for dev in devices:
                props = _VkPhysicalDeviceMemoryProperties()
                vk_memprops(dev, ctypes.byref(props))
                local = max(
                    (props.memoryHeaps[i].size
                     for i in range(min(props.memoryHeapCount,
                                        VK_MAX_MEMORY_HEAPS))
                     if props.memoryHeaps[i].flags
                     & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT),
                    default=0)
                if local > best:
                    best = local
            return {"dedicated_vram": int(best), "current_usage": 0}
        finally:
            vk_destroy(instance, None)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# DXGI 显存查询（Windows COM，纯 ctypes；部分环境工厂创建 E_NOINTERFACE）
# ---------------------------------------------------------------------------
_IID_IDXGIFACTORY1 = (ctypes.c_ubyte * 16)(
    0x78, 0xAE, 0x0A, 0x77, 0x6F, 0xF2, 0xBA, 0x4D,  # 770aae78-f26f-4dba
    0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0x54, 0x9C,  # -a829-253c83d1549c
)


class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),   # 物理显存（字节）
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_longlong),
    ]


def _release(ptr) -> None:
    """对 COM 接口指针调用 Release()（vtable 槽位 2）。"""
    if not ptr:
        return
    try:
        vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
        release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtbl[2])
        release(ptr)
    except Exception:  # noqa: BLE001  # 尽力而为
        pass


def query_vram_dxgi() -> dict | None:
    """DXGI 枚举适配器，返回物理显存/当前使用（字节）；失败返回 None。

    Returns:
        {"dedicated_vram": int, "current_usage": int}（取 DedicatedVideoMemory
        最大的适配器；current_usage 可能为 0=不可用）。
    """
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        factory = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFACTORY1),
                                     ctypes.byref(factory))
        if hr < 0 or not factory:
            return None
        vtbl = ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p))[0]
        enum_adapters = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p))(vtbl[7])  # IDXGIFactory::EnumAdapters
        best = None
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, i, ctypes.byref(adapter)) < 0 or not adapter:
                break
            avtbl = ctypes.cast(adapter, ctypes.POINTER(ctypes.c_void_p))[0]
            get_desc = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p,
                ctypes.POINTER(_DXGI_ADAPTER_DESC))(avtbl[8])  # IDXGIAdapter::GetDesc
            desc = _DXGI_ADAPTER_DESC()
            if get_desc(adapter, ctypes.byref(desc)) >= 0:
                if best is None or desc.DedicatedVideoMemory > best[0]:
                    best = (desc.DedicatedVideoMemory, int(desc.DedicatedSystemMemory))
            _release(adapter)
            i += 1
        _release(factory)
        if best is None:
            return None
        return {"dedicated_vram": best[0],
                "dedicated_system": best[1],
                "current_usage": 0}  # 物理总量口径：CurrentUsage 需 IDXGIAdapter3，非必需
    except Exception:  # noqa: BLE001
        return None


def detect_vram_gb() -> float | None:
    """返回 Vulkan 可用显存（GB）；无法检测返回 None。

    优先 ``RVC_ADAPTIVE_GB`` env（显式覆盖，便于测试 4G 路径），
    否则走 DXGI 物理显存总量。
    """
    ov = os.environ.get("RVC_ADAPTIVE_GB", "").strip()
    if ov:
        try:
            return float(ov)
        except ValueError:
            print(f"[adaptive] 警告：RVC_ADAPTIVE_GB={ov!r} 不是数字，忽略")
    for q in (query_vram_vulkan, query_vram_dxgi):
        info = q()
        if info is not None and info["dedicated_vram"]:
            return info["dedicated_vram"] / (1024 ** 3)
    return None


def adaptive_enabled() -> bool:
    """RVC_ADAPTIVE=0 显式关闭；未设或 1 = 自动开启（默认）。"""
    return os.environ.get("RVC_ADAPTIVE", "1") != "0"


def choose_mode(vram_gb: float | None) -> str:
    """按显存阈值选模式：>4G 正常；≤4G 4G 模式；None → 正常（保守回退）。"""
    if vram_gb is None:
        return MODE_NORMAL
    return MODE_NORMAL if vram_gb > VRAM_THRESHOLD_GB else MODE_4G


def _force_mode_env() -> str | None:
    m = os.environ.get("RVC_ADAPTIVE_MODE", "").strip().lower()
    if m in (MODE_NORMAL, MODE_4G):
        return m
    return None


def _inject_env(label: str, env_map: dict, verbose: bool) -> dict:
    """通用 env 注入：用户已显式设置的同名 env 不被覆盖（记录）。"""
    injected, preserved = {}, {}
    for k, v in env_map.items():
        if k in os.environ:
            preserved[k] = os.environ[k]
            if verbose:
                print(f"[adaptive] {k} 已由用户显式设置={os.environ[k]!r}，"
                      f"尊重不覆盖（{label}模式建议值={v!r}）")
        else:
            os.environ[k] = v
            injected[k] = v
    if verbose:
        if injected:
            print(f"[adaptive] {label}模式已注入 env: "
                  + ", ".join(f"{k}={v}" for k, v in injected.items()))
        if preserved:
            print(f"[adaptive] {label}模式保留用户 env: "
                  + ", ".join(f"{k}={v}" for k, v in preserved.items()))
    return {"injected": injected, "preserved": preserved}


def _inject_4g_env(verbose: bool) -> dict:
    """注入 4G 模式 env。"""
    return _inject_env(label="4G", env_map=_4G_ENV, verbose=verbose)


def _inject_normal_env(verbose: bool) -> dict:
    """注入正常模式 env（T7b：图化门控）。"""
    return _inject_env(label="正常", env_map=_NORMAL_ENV, verbose=verbose)


def apply_adaptive_mode(verbose: bool = True) -> dict:
    """训练启动时调用：检测显存 → 选模式 → 注入对应模式 env。

    返回决策 dict（供 train.py 打印 / 测试断言）：
        {"enabled", "mode", "vram_gb", "reason", "injected", "preserved"}
    正常模式注入图化门控（T7b）；4G 模式注入池/批大小门控。
    ``RVC_ADAPTIVE=0`` 时任意模式均不注入（== 现状）。
    """
    if not adaptive_enabled():
        print("[adaptive] RVC_ADAPTIVE=0 显式关闭：不检测、不注入（行为与现状一致）")
        return {"enabled": False, "mode": MODE_NORMAL, "vram_gb": None,
                "reason": "RVC_ADAPTIVE=0", "injected": {}, "preserved": {}}

    forced = _force_mode_env()
    vram_gb = detect_vram_gb()
    if forced is not None:
        mode = forced
        reason = f"RVC_ADAPTIVE_MODE={forced}（强制，忽略检测 {vram_gb!r}）"
    elif vram_gb is None:
        mode = MODE_NORMAL
        reason = "显存检测失败（DXGI 不可用），保守回退正常模式（与现状一致）"
    else:
        mode = choose_mode(vram_gb)
        reason = (f"显存 {vram_gb:.2f}GB {'>' if mode == MODE_NORMAL else '<='} "
                  f"{VRAM_THRESHOLD_GB:g}GB")

    res = {"enabled": True, "mode": mode, "vram_gb": vram_gb,
           "reason": reason, "injected": {}, "preserved": {}}
    if mode == MODE_4G:
        res.update(_inject_4g_env(verbose))
    else:
        res.update(_inject_normal_env(verbose))
        if verbose:
            print(f"[adaptive] {reason} → 正常模式：batch=3 + 池开 + 判别器双缓冲"
                  f" + 图化门控全开（T7b，DEC/ASYNC 默认 0）")
    return res


def monitor_step(step: int, log_interval: int = 10) -> None:
    """训练循环内轻量显存预算监控（RVC_ADAPTIVE_MONITOR=1 才启用）。

    每 log_interval 步查询一次 DXGI；当物理显存 ≤4GB（4G 模式机器）或
    当前预算低于 512MB 时打印告警。默认关，开启时单次查询 <1ms。
    """
    if os.environ.get("RVC_ADAPTIVE_MONITOR") != "1":
        return
    if step % log_interval != 0:
        return
    info = query_vram_dxgi()
    if info is None:
        return
    vram = info["dedicated_vram"] / (1024 ** 3)
    if vram <= VRAM_THRESHOLD_GB + 0.5:
        print(f"[adaptive] 监控：物理显存 {vram:.2f}GB ≤4G 档（step {step}），"
              f"若出现 OOM 请确认已走 4G 模式（batch=1+池关）")
