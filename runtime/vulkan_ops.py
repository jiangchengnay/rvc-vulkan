# -*- coding: utf-8 -*-
"""Vulkan GPU 算子封装（基于 ``runtime._vulkan`` 的 ctypes 绑定）。

提供 ``VulkanContext``（一个进程一个引擎句柄的惰性单例）与
``matmul / add / mul / relu_inplace / conv1d / conv_transpose1d / conv2d /
embedding / add_inplace / mul_inplace`` 等算子，输入输出均为 **float32**
numpy 数组（float64 自动降为 float32；int/complex/bool 抛 ``TypeError``）。

小张量阈值：总元素数 < ``_THRESHOLD``（默认 1024）时直接走 numpy，避免 GPU
往返开销。dtype 约定：输入输出都按 float32 处理，调用方负责 dtype 语义。

注意：原生引擎的每次 C 调用本身已被 ``Engine.mutex`` 串行化（ffi.zig），这里的
全局锁用于保证 Python 层“上传 → 计算 → 下载 → 释放”复合操作的原子性
（避免其他线程在复合操作中途 free 同一个 buffer）。

线程语义（多块并行推理 / WebUI 多请求共享模型）：
    - 单算子（``matmul/conv1d/...``）：每笔调用内部经 ``VulkanContext._lock``
      逐段加锁 + 引擎 ``Engine.mutex`` 串行化每次 C 调用，跨线程安全。
    - ``BatchRunner``：引擎的批次状态机是**引擎级共享**的（``rvc_batch_begin``
      会重置未提交批次）。因此每个 ``BatchRunner`` 从构造（begin）到
      ``release`` 全程独占 ``_BATCH_LIFECYCLE_LOCK``，保证两个并发 runner 的
      begin/add/commit 序列不交错——GPU 批次提交被串行化（与串行调用语义
      一致），CPU 侧录制/检索/插值等仍可跨线程并行。
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time

import numpy as np
import os

def traced(name):
    """GPU 算子计时装饰器（RVC_PROFILE=1 时启用，默认零开销）。"""
    from runtime import profiler
    if not profiler.enabled():
        def _noop(fn):
            return fn
        return _noop

    def _deco(fn):
        def _wrap(*a, **k):
            with profiler.profile_op(name):
                return fn(*a, **k)
        return _wrap
    return _deco


# RVC_TRAIN_MEMTRACE=1：记录每次 GPU buffer 分配（mem_alloc/upload 新建），
# 带大小与调用点 —— 定位每步泄漏 buffer 的创建路径（P0 T0.3 专用，默认关）。
import traceback as _tb  # noqa: E402

_MT_F = None
_MT_LOCK = threading.Lock()
_MT_LIVE = {}  # bid -> (nbytes, callsite)；free 时移除


def _mtrace(bid: int, nbytes: int, kind: str) -> None:
    if os.environ.get("RVC_TRAIN_MEMTRACE") != "1":
        return
    global _MT_F
    if _MT_F is None:
        _MT_F = open(os.path.join(os.environ.get("RVC_OUT_DIR", "."),
                                  "memtrace.log"), "a", encoding="utf-8")
    with _MT_LOCK:
        try:
            st = _tb.extract_stack(limit=7)
            calls = [f"{f.filename.split(chr(92))[-1]}:{f.lineno}:{f.name}"
                     for f in st[-7:-1]]
            callsite = " <- ".join(calls)
            _MT_LIVE[int(bid)] = (int(nbytes), callsite)
            _MT_F.write(f"{kind} {nbytes} | {callsite}\n")
            _MT_F.flush()
        except Exception:
            pass


def _mtfree(bid: int) -> None:
    if os.environ.get("RVC_TRAIN_MEMTRACE") != "1":
        return
    with _MT_LOCK:
        _MT_LIVE.pop(int(bid), None)


from runtime import _vulkan

__all__ = [
    "VulkanContext",
    "PersistentBuffer",
    "BatchTensor",
    "BatchRunner",
    "get_context",
    "matmul",
    "add",
    "mul",
    "relu_inplace",
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
    "_THRESHOLD",
    "stats_reset",
    "stats_snapshot",
]

_THRESHOLD = 1024
"""小张量阈值：总元素数小于该值时直接用 numpy 计算，避免 GPU 往返开销。"""

# 引擎计算 grid.x ≤ 驱动实测 maxComputeWorkGroupCount（Vulkan 规范只保证
# 最小 65535；AMD Radeon Pro VII vulkaninfo 实测 = 4294967295 = 2^32-1），
# 每 workgroup 256 线程。D2（性能攻坚）：旧上限 16.78M（65535*256）迫使
# 长音频 dec 的 38s 级算子（ups/ResBlock）分段——每段一次 Python+ctypes
# dispatch、BatchRunner 逐算子录制、数百次上传下载同步（dec GPU busy 仅
# 23%，CPU 侧 ~47s 主导）。按驱动实际上限放宽到 2^31（约 21.5 亿点）留
# 2x 余量：点数 ≤ 2^31 < 2^32 → u32 push/寻址安全；workgroup 组数
# gx = ceilDiv(点数,256) ≤ 8.4M << 2^32-1；dec 正常块（150s ups3
# B*C_out*oL=96M 点、gx=375k 组）不再触发分段，整段 dispatch。
_GRID_POINTS_MAX = 1 << 31

# 切分递归标志：**线程局部**（thread-local）。切分期间本线程跳过二次超限
# 检查（子块尺寸已受控）；并发（多块并行 / WebUI 多请求）下若用模块级
# bool，线程 A 的切分标志会让线程 B 的超限输入**跳过切分保护**而直接
# dispatch → DimensionsTooLarge 崩溃，故必须按线程隔离。
_in_split_local = threading.local()


def _in_split_state() -> bool:
    return getattr(_in_split_local, "flag", False)

# --------------------------------------------------------------------------
# 传输计数统计（perf(P1): 中间张量显存驻留的数据流审计）
#
# 记录进程内累计的 GPU 上传/下载次数与字节数（含 persistent_upload 与
# BatchRunner 内部的 upload/download），供数据流改造前后对比：
#     vulkan_ops.stats_reset()    清零
#     vulkan_ops.stats_snapshot() 返回 {"uploads":..,"downloads":..,
#                                   "upload_bytes":..,"download_bytes":..}
# 计数点只在 VulkanContext.upload/download 两处各加一条整数自增，开销可忽略。
# --------------------------------------------------------------------------
_STATS_LOCK = threading.Lock()
_STATS = {"uploads": 0, "downloads": 0, "upload_bytes": 0, "download_bytes": 0}

# P0-1（用户要求：GPU 必须真正跑起来，不许回退 CPU 解释成硬件特性）：
# 强制小 conv2d 也走 GPU（跳过 _THRESHOLD 阈值）。默认关（保持数值/性能基线），
# RVC_CONV2D_FORCE_GPU=1 开启，用于验证与 batch 化前的过渡。
_FORCE_CONV2D_GPU = __import__("os").environ.get("RVC_CONV2D_FORCE_GPU", "0").strip() in ("1", "true", "on")

# 引擎批次状态机互斥（RLock）：BatchRunner 构造（rvc_batch_begin）到
# release 全程持有，独占引擎的 begin→add→commit 序列不被其他线程的
# begin（会重置批次）打断。多块并行 / WebUI 并发共享模型时保证安全。
_BATCH_LIFECYCLE_LOCK = threading.RLock()

# --------------------------------------------------------------------------
# 输出 buffer 池（perf(P1-6): 消灭每次推理 ~1.4GB 的零数组上传）
#
# ``VulkanContext._alloc_output`` 分配的**输出** buffer 在 ``free()`` 时自动
# 归还池（按元素数分桶），下次同尺寸的 ``_alloc_output`` 直接复用——不再
# 每次 upload 一个 M*N 零数组。安全性：engine 全部 shader 对输出 buffer
# 都是**全量覆写**（conv1d/conv_t1d/copy/elementwise/softmax/LN 等均为
# gather 式、逐输出点完整计算后写入，从不读旧值；conv_t1d 的输出越界点
# 直接 return 不写、有效点全覆盖），因此池中脏内容不会泄漏到结果。
# 归还时机：逐次算子在同步下载完成后、BatchRunner 在 commit+wait 完成后
# 才 free —— GPU 必然已消费完毕，池条目可直接复用。
# --------------------------------------------------------------------------
# T1.3（性能）：池命中率实测仅 40%——dec 同尺寸（33.6MB×138 次）并发
# 持有超过每尺寸上限 8 导致大量重复 mem_alloc（312 次 × ~2.8ms = 0.87s，
# 含 vkAllocateMemory + engine gpuFill submitOneShot+wait）。参数扫描：
# 8/512=40%命中（12s 6.95s）；24/1024=12s 6.50s 但 **150s rmvpe 显存爆**
# （VkFailed -2：池累积 + rmvpe UNet 叠加超 16GB；150s pm 93.9s 能过）。
# 主防线改为**字节预算**（_OUT_POOL_MAX_BYTES=3GB，池总显存占用上限，
# 超出直接真释放不入池）——数量上限（PER_SIZE/MAX_BUFS）只作次防线。
# T17（2026-09-25）：PER_SIZE 24→96（t17_poolscan 60s 档 A/B：base
# mem_alloc 392 次/wall 10.99s → PER=96 184 次/8.77s，-2.2s/60s）。
# dec conv1d 输出同尺寸并发持有实测可达 70+（ups/ResBlock 各级），24 的
# 每尺寸上限让高频尺寸反复 mem_alloc（含 vkCreateBuffer+vkAllocateMemory
# +gpuFill 同步点）；96 覆盖 dec 波形/中间张量并发驻留的高频尺寸。字节
# 预算 3GB 保持主防线（150s rmvpe 峰值 <16GB 数据不变，仅驻留更多复用
# buffer；真释放兜底仍生效）。
_OUT_POOL_PER_SIZE = 256    # 每尺寸最多缓存几个输出 buffer（防单尺寸无限堆积）
_OUT_POOL_MAX_BUFS = 4096   # 池总条目上限（防显存无界膨胀，超出直接真释放）
# 池总字节上限（T19：3GB→4GB——60s/150s 档 dec 跨块复用被 3GB 卡住（块0
# ups 大 buffer 超预算只留一半，块1 尺寸差 2.22x GE 找不到桶 → 每块重复
# mem_alloc ~36 次×12.9ms=0.44s/60s）。提 6GB 后 60s dec -0.47s / 150s
# -0.5s 全链 maxdiff=0（两类验证均逐位一致），但 **178s 档 VkFailed**（每块
# ~255MB×28≈7GB 池需求 + 推理 live buffer 合计超 16GB HBM2）。4GB 为折中：
# 保留相邻块 GE 复用主收益，178s 显存安全（3GB 是 T18 已实证安全门，4GB 仅
# +1GB 多留块0 大桶供块1/块2 GE 命中；真释放兜底仍生效）。
_OUT_POOL_MAX_BYTES = int(8 * 1024 * 1024 * 1024)  # 池总字节上限（8GB）
# 阶段D（D2 输入池）上限：训练侧批量输入 buffer 池（输入全量覆写安全；
# 池仅服务 BatchRunner._resolve_input 池化路径，RVC_TRAIN_IN_POOL=1 启用）
_IN_POOL_MAX_BUFS = 2048     # 池总条目上限
_IN_POOL_MAX_BYTES = int(2 * 1024 * 1024 * 1024)  # 池总字节上限（2GB）
# 音质回归修复（2026-09-21）：mem_alloc 未初始化 buffer 被池复用时，若某
# shader 未全量覆写输出（边界/padding 区），旧值残留污染高频（2s 带宽
# 15.8k→20.1k 复现）。**F1（音质修复计划）：默认恢复池复用**——禁池导致
# 每次 mem_alloc 新分配读未初始化垃圾（全噪声）+ descriptor/显存耗尽
# （VkFailed）；池复用的"旧值残留"（糊+截断）比全噪声可接受（止血态，
# 真根因=F2 哨兵测试定位未覆写 shader → F3 修复）。env 可覆盖（=1 禁池）。
_OUT_NO_POOL = os.environ.get("RVC_OUT_NO_POOL", "0") != "0"
# D1a-1（性能攻坚）：池复用放宽到"尺寸 >= 需求的最近桶"——dec/enc 各层输出
# 尺寸不重复时精确命中率低（60s 档 dec 窗口实测 87 次 mem_alloc，每次含
# vkCreateBuffer+vkAllocateMemory+gpuFill 约 10.6ms）。允许把比需求大的
# buffer 复用给更小输出，大幅提高命中率、减少 mem_alloc。
# T19（2026-09-25）：1.25→2.5——60s 档块0/块1 ups 大 buffer 尺寸差 2.22x
# （块长不同），1.25 找不到跨块可复用桶（块1 每块 ~34 次重复 mem_alloc）；
# 2.5 覆盖 2.22x 差（215.5MB 桶可服务 96.9MB 需求、43.1MB→19.4MB），60s
# 档 dec 窗口 mem_alloc 72→53（预算 3GB 下），预算提 6GB 后可再降 ~20 次。
# 安全性：engine 全部 shader 对输出 buffer 全量覆写（见模块 docstring 池
# 说明），复用 buffer 只用到前 n 元素，其余陈旧区域永不读；download 只取
# n 元素；F3-B fill_zero 在复用返回前照常执行。数值逐位一致（D1a-1 已有
# GE=1.25 全链 maxdiff=0 实测，本次仅放宽桶选择范围，fill_zero 路径不变）。
_OUT_POOL_GE_RATIO = float(os.environ.get("RVC_OUT_POOL_GE_RATIO", "2.5"))
# 复用桶尺寸上限 = ceil(n * ratio)（>1 启用 >= 复用；env 可覆盖做 A/B）


def stats_reset() -> None:
    """清零传输计数（数据流对比时在推理前调用一次）。"""
    with _STATS_LOCK:
        _STATS["uploads"] = 0
        _STATS["downloads"] = 0
        _STATS["upload_bytes"] = 0
        _STATS["download_bytes"] = 0


def stats_snapshot() -> dict:
    """返回当前传输计数快照（dict，包含 4 个键）。"""
    with _STATS_LOCK:
        return dict(_STATS)


def _stats_upload(nbytes: int) -> None:
    with _STATS_LOCK:
        _STATS["uploads"] += 1
        _STATS["upload_bytes"] += nbytes


def _stats_download(nbytes: int) -> None:
    with _STATS_LOCK:
        _STATS["downloads"] += 1
        _STATS["download_bytes"] += nbytes

F32 = ctypes.c_float
F32_PTR = ctypes.POINTER(F32)


def _f32_bits(v: float) -> int:
    """把 float 的 32 位模式编码为有符号 int64（供 i64 参数传递）。

    引擎侧按 ``@truncate`` 恢复低 32 位再 ``@bitCast`` 回 f32，因此任意
    位模式（含负数 slope 的符号位）都可无损传递。
    """
    return int(np.float32(v).view(np.int32))


def _as_f32(a: np.ndarray, name: str) -> np.ndarray:
    """校验并规整为连续的 float32 数组。

    - float32：原样（确保内存连续）；
    - 其他浮点（float64）：自动降为 float32；
    - 非浮点（int/complex/bool）：抛 ``TypeError``。
    """
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(
            f"{name} 必须是浮点 numpy 数组（Vulkan 后端仅支持 float32，"
            f"float64 会自动转换），got dtype={a.dtype}"
        )
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


class PersistentBuffer:
    """持久的 GPU buffer 包装（BufferPool 常驻权重，P1 优化）。

    与 ``VulkanContext.upload`` 返回的裸 buffer id 不同，PersistentBuffer
    **不自动释放**：显式调用 ``.free()``（幂等）或在解释器退出/GC 时由
    ``__del__`` 兜底释放。常驻 buffer 可跨多次算子调用复用 —— 算子函数
    通过 ``buf_w=...`` / ``buf_b=...`` 等参数引用它时会跳过重复 upload，
    并且 **不会** 在算子的 finally 中释放它（所有权归调用方）。

    属性:
        id: 底层 GPU buffer id；``free()`` 后为 None。
        shape: 上传时的数组形状（供调试 / 形状校验）。
        nbytes: 占用字节数。
        valid: 是否仍有效（未释放）。已释放的 buffer 被算子引用时自动
            回退到普通 upload 路径（不崩溃）。
    """

    __slots__ = ("_ctx", "_buf", "shape", "nbytes")

    def __init__(self, ctx: "VulkanContext", buf: int, shape, nbytes: int):
        self._ctx = ctx
        self._buf = int(buf)
        self.shape = tuple(shape)
        self.nbytes = int(nbytes)

    @property
    def id(self) -> int | None:
        return self._buf

    @property
    def valid(self) -> bool:
        return self._buf is not None

    def free(self) -> None:
        """显式释放 GPU buffer（幂等；释放后 ``id`` 置 None）。

        解释器关闭阶段（``sys.is_finalizing()``）**只做本地置空、不再触碰引擎**：
        此时模块级引擎全局可能已被 GC 部分清理，调用 rvc_mem_free 会在 GPU
        清理路径上死锁（实测 AMD 驱动在进程退出时挂住，faulthandler 定位
        为 __del__ → free → rvc_mem_free）。
        """
        if self._buf is not None:
            buf, self._buf = self._buf, None
            if sys.is_finalizing():
                return
            try:
                self._ctx.free(buf)
            except RuntimeError:
                pass  # 进程退出阶段的清理尽力而为

    def update(self, a) -> None:
        """就地覆盖写常驻 buffer 内容（形状须同，字节数不大于原分配）。

        训练权重场景：optimizer step 后调用，把最新参数内容刷进已在 GPU
        的常驻 buffer，避免 bp 链内逐次重复上传同一批权重（J11）。
        """
        a = _as_f32(a, "PersistentBuffer.update 输入")
        if self._buf is None:
            raise RuntimeError("PersistentBuffer 已释放，无法 update")
        if a.size * 4 > self.nbytes:
            raise ValueError(
                f"PersistentBuffer.update 字节超出: {a.size * 4} > {self.nbytes}")
        with self._ctx._lock:
            _vulkan._check(
                _vulkan.dll.rvc_mem_upload_to(
                    self._ctx._handle, a.ctypes.data_as(F32_PTR), a.size,
                    int(self._buf),
                ),
                "rvc_mem_upload_to(PersistentBuffer.update)",
            )

    def __del__(self):
        try:
            self.free()
        except Exception:  # noqa: BLE001  # 解释器退出阶段不抛
            pass

    def __repr__(self):
        state = "freed" if self._buf is None else f"buf={self._buf}"
        return f"PersistentBuffer(shape={self.shape}, {state})"


# ---------------------------------------------------------------------------
# J11：训练权重常驻 GPU 缓存（判别器/生成器权重每步 refresh 覆盖写，
# 消除 bp 链内同一权重逐次重复上传——判别器 P bwd 原 192 次/步重传）。
# ---------------------------------------------------------------------------
_WPB = {}
_WPB_LOCK = threading.Lock()
_WPB_GUARD = 8192  # T0.4 容量护栏（a24b 审计 T5）：超限打 stderr 告警（不阻塞）


def _wpb_guard(key) -> None:
    """_WPB 容量护栏：超过 _WPB_GUARD 条时告警，提示存在 key 漂移泄漏。"""
    if len(_WPB) > _WPB_GUARD:
        import sys as _sg  # noqa: PLC0415
        print(f"[WPB-GUARD] _WPB 达 {len(_WPB)} 条（key={key}）——疑似 key 漂移泄漏，"
              f"请检查 wpers_get/wpers_get_wn 的调用方是否传每步新建数组",
              file=_sg.stderr, flush=True)


def wpers_get(w: np.ndarray, br, tag: str = "") -> PersistentBuffer:
    """按权重数组恒等 id 取常驻 buffer；miss 时首次上传（常驻不释放）。

    ``w`` 必须是训练参数数组（opt step 就地在同一数组上更新）——refresh
    时读其最新内容覆盖写 GPU buffer。``tag``（"d"/"g"）供 wpers_refresh
    按步分批刷新（判别器 D 步后刷、生成器 G 步后刷，避免重复上传）。
    """
    key = id(w)
    with _WPB_LOCK:
        hit = _WPB.get(key)
        if hit is not None and hit[0].valid and hit[1].shape == w.shape:
            return hit[0]
    pb = br._ctx.persistent_upload(np.asarray(w, np.float32))
    with _WPB_LOCK:
        # T0.4：条目存原始数组活引用（第4元素）——wpers_refresh 普通分支
        # 读最新内容（a24b 审计 T3：原冻结快照与 docstring 不符）。w 是
        # 训练参数数组（optimizer 就地更新），存引用不拷贝。
        _WPB[key] = (pb, np.asarray(w, np.float32), tag, w)
        _wpb_guard(key)
    return pb


def _wn_deweight(w_v, w_g) -> np.ndarray:
    """weight_norm 还原为普通权重（f32）：W = w_v * (w_g / ||w_v||_2)。

    与 ``vits_train.AutogradTape.deweight_norm`` 前向逐位一致：L2 范数沿
    除第 0 维外的所有维（逐通道），``w_g`` 为逐通道标量广播。"""
    v32 = np.asarray(w_v, np.float32)
    g32 = np.asarray(w_g, np.float32)
    axes = tuple(range(1, v32.ndim))
    norm = np.sqrt((v32 * v32).sum(axis=axes, keepdims=True))
    g_b = g32
    while g_b.ndim < v32.ndim:
        g_b = np.expand_dims(g_b, -1)
    # T0.4：clamp 常量与 tape.deweight_norm（vits_train.py:1159）统一为 1e-12，
    # 保证范数趋零时两者逐位一致（a24b 审计 T2）。
    return (v32 * (g_b / np.maximum(norm, 1e-12))).astype(np.float32)


def wpers_get_wn(w_v, w_g, br, tag: str = "") -> PersistentBuffer:
    """weight_norm 权重（v/g 持久参数）的 GPU 常驻展开 cache——T0.3 泄漏修复。

    背景：dec/enc/flow 的 weight_norm 权重每步前向 ``deweight_norm`` 会
    新建展开数组 → ``wpers_get`` 按 ``id(展开数组)`` 缓存永远 miss → 每步
    +~114 个 PersistentBuffer 永久累积（buf_count 每步 +~105 的泄漏源）。

    本函数按 ``(id(w_v), id(w_g))`` 缓存（v/g 是训练参数持久对象，id 稳定）；
    ``wpers_refresh`` 时重新展开（v/g 已被 optimizer 更新）覆盖写 buffer。
    """
    key = ("wn", id(w_v), id(w_g))
    with _WPB_LOCK:
        hit = _WPB.get(key)
        if hit is not None and hit[0].valid and hit[1].shape == np.shape(w_v):
            return hit[0]
    w = _wn_deweight(w_v, w_g)
    pb = br._ctx.persistent_upload(np.asarray(w, np.float32))
    with _WPB_LOCK:
        _WPB[key] = (pb, np.asarray(w, np.float32), tag, w_v, w_g)
        _wpb_guard(key)
    return pb


def wpers_slot_for(w) -> int | None:
    """T4-3：返回参数数组的 GPU 常驻槽 id（未注册/已失效 → None）。

    供优化器图化复用 wpers 槽作 p 更新目标（p 免下载免重传）。
    """
    key = id(w)
    with _WPB_LOCK:
        hit = _WPB.get(key)
    if hit is None:
        return None
    pb = hit[0]
    if not pb.valid or pb._ctx is None:
        return None
    bid = getattr(pb, "id", None)
    return int(bid) if bid is not None else None


def wpers_refresh(tag: str = "") -> None:
    """optimizer step 后调用：把 tag 匹配的权重最新内容刷进常驻 buffer。

    判别器权重 D 步后刷（G 步可导判别器读新值）；生成器权重 G 步后刷。
    已释放/失败项静默跳过（下次 wpers_get 自动重建）。
    J12：按 VulkanContext 分组，复用 J9 的 ``_batch_upload``（一次
    rvc_mem_upload_to_batch = 一次 staging memcpy + 一次提交 + 一次 fence），
    替代原来逐权重独立 rvc_mem_upload_to（每权重一次 submit+fence——
    D 权重 ~200 组的同步开销进了 d_step/opt_g 段）。
    T0.3：wn 项（weight_norm 展开缓存）先按最新 v/g 重新展开再上传——
    optimizer 就地更新 v/g，缓存快照过期必须重算。
    """
    with _WPB_LOCK:
        items = list(_WPB.items())
    by_ctx = {}
    for key, item in items:
        if not item[0].valid:
            continue
        if tag and item[2] != tag:
            continue
        if isinstance(key, tuple) and key[0] == "wn":
            try:
                w_now = _wn_deweight(item[3], item[4])  # 按最新 v/g 重展开
            except Exception:  # noqa: BLE001
                continue
            w = np.asarray(w_now, np.float32)
        else:
            # T0.4：普通条目用活引用（item[3]）读最新内容；兼容旧 3 元组
            # 条目（item[1] 快照）。
            w = np.asarray(item[3] if len(item) > 3 else item[1],
                           np.float32)
        by_ctx.setdefault(item[0]._ctx, []).append((int(item[0]._buf), w))
    for ctx, pairs in by_ctx.items():
        try:
            ctx._batch_upload(pairs)
        except Exception:  # noqa: BLE001  引擎失败静默（下次 get 重建）
            pass


def _resolve_buf(pb: PersistentBuffer | None) -> int | None:
    """把算子的 ``buf_*`` 参数解析为底层 buffer id。

    返回 None 表示"未提供 / 已释放 / 类型不符"，此时由调用算子回退到普通
    upload 路径。这样即使调用方持有一份已 ``free()`` 的常驻 buffer 引用，
    算子调用也不会崩溃。
    """
    if pb is None:
        return None
    bid = getattr(pb, "id", None)
    return int(bid) if bid is not None else None


class VulkanContext:
    """持有原生引擎句柄的 GPU 上下文（一个进程一个实例，惰性创建）。

    方法：
        upload(a) -> int buf
        persistent_upload(a) -> PersistentBuffer
        download(buf, shape) -> np.ndarray
        free(buf)
        matmul(a, b) -> np.ndarray
        add(a, b) / mul(a, b) / relu_inplace(a) -> np.ndarray
    """

    def __init__(self):
        if not _vulkan.has_dll:
            raise RuntimeError(f"rvc_core.dll 不可用: {_vulkan.dll_load_error}")
        self._handle = _vulkan.engine_create()
        self._closed = False
        self._lock = threading.RLock()
        # 输出 buffer 池（perf(P1-6)）：_out_registry[buf] = 元素数，登记
        # 所有由 _alloc_output 分配的 buffer；free() 时自动归还 _out_pool。
        self._out_registry: dict[int, int] = {}
        self._out_pool: dict[int, list[int]] = {}
        self._out_pool_count = 0
        # T1.3：池总字节占用（字节预算 _OUT_POOL_MAX_BYTES 主防线，
        # 防 150s 长音频/rmvpe 路径池累积显存爆）。
        self._out_pool_bytes = 0
        # 阶段D（D2 输入池）：训练侧批量输入的 buffer 池化复用
        # （rvc_mem_upload_to 覆盖写——输入全量覆写安全，无输出池的
        # fill_zero 问题）。_in_pool_n: bid -> 元素数。
        self._in_pool: dict[int, list[int]] = {}
        self._in_pool_n: dict[int, int] = {}
        self._in_pool_count = 0
        self._in_pool_bytes = 0

    # -- 生命周期 --------------------------------------------------------
    def close(self) -> None:
        """显式销毁引擎句柄（幂等）。

        解释器退出阶段（``sys.is_finalizing()``）跳过引擎销毁：与 free 同理，
        此时调用 rvc_engine_destroy 可能在 GPU 清理路径上挂起。
        """
        if self._closed:
            return
        self._closed = True
        if sys.is_finalizing():
            return
        # 清空输出 buffer 池（真释放），随后销毁引擎
        with self._lock:
            pool, self._out_pool = self._out_pool, {}
            self._out_pool_count = 0
            self._out_pool_bytes = 0
            for bucket in pool.values():
                for bid in bucket:
                    try:
                        _vulkan._check(
                            _vulkan.dll.rvc_mem_free(self._handle, int(bid)),
                            "rvc_mem_free",
                        )
                    except RuntimeError:
                        pass
            # 阶段D（D2）：清空输入 buffer 池
            pool2, self._in_pool = self._in_pool, {}
            self._in_pool_n = {}
            self._in_pool_count = 0
            self._in_pool_bytes = 0
            for bucket in pool2.values():
                for bid in bucket:
                    try:
                        _vulkan._check(
                            _vulkan.dll.rvc_mem_free(self._handle, int(bid)),
                            "rvc_mem_free",
                        )
                    except RuntimeError:
                        pass
        try:
            _vulkan.engine_destroy(self._handle)
        except RuntimeError:
            pass  # 进程退出阶段的清理尽力而为

    def mem_stats(self) -> dict:
        """T8：只读显存统计（suballocator chunk 总量/块数、free 字节/块数、
        存活 buffer 数/字节、staging_up/dn 常驻字节、direct-alloc 字节、
        最大 buffer）。探针逐段采样用。"""
        outs = [ctypes.c_int64(0) for _ in range(11)]
        ptrs = [ctypes.byref(o) for o in outs]
        with self._lock:
            _vulkan._check(
                _vulkan.dll.rvc_mem_stats(self._handle, *ptrs),
                "rvc_mem_stats")
        keys = ("sub_total", "sub_chunks", "sub_free_bytes",
                "sub_free_blocks", "buf_count", "buf_bytes",
                "staging_up", "staging_dn", "direct_bytes",
                "max_buf_bytes", "max_buf_sub")
        return {k: int(o.value) for k, o in zip(keys, outs)}

    def mem_top(self, n: int = 20) -> list:
        """T8：Top-N 存活 buffer dump（bytes, sub, memory 三元组，降序）。"""
        buf = (ctypes.c_int64 * (n * 3))()
        with self._lock:
            cnt = _vulkan._check(
                _vulkan.dll.rvc_mem_top(self._handle, n, buf),
                "rvc_mem_top")
        return [(int(buf[i * 3]), int(buf[i * 3 + 1]), int(buf[i * 3 + 2]))
                for i in range(int(cnt))]

    def clear_pool(self) -> None:
        """清空输出 buffer 池（真释放池中全部空闲 buffer，归还显存）。

        供推理流程在**单个文件推理完成/失败后**调用（vc_single finally）：
        - 池中条目均为已消费完毕、free() 归还的空闲 buffer（live buffer 不在池），
          清池不影响进行中的推理；
        - 同文件多块推理期间的池内复用不受影响（块间仍复用，仅文件级清空）；
        - 解决长音频/连续多文件推理下池驻留显存累积导致
          VK_ERROR_OUT_OF_DEVICE_MEMORY（rvc_mem_upload VkFailed）的问题。
        与 close() 的池清理同构；不销毁引擎句柄。
        """
        with self._lock:
            pool, self._out_pool = self._out_pool, {}
            self._out_pool_count = 0
            self._out_pool_bytes = 0
            for bucket in pool.values():
                for bid in bucket:
                    try:
                        _vulkan._check(
                            _vulkan.dll.rvc_mem_free(self._handle, int(bid)),
                            "rvc_mem_free",
                        )
                    except RuntimeError:
                        pass  # 引擎销毁/进程退出阶段尽力而为

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001  # 解释器退出阶段不抛
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- 底层 buffer 操作 -------------------------------------------------
    def upload(self, a: np.ndarray) -> int:
        """上传 numpy 数组到 GPU，返回 buffer id（n 元素，4n 字节）。"""
        a = _as_f32(a, "upload 输入")
        n = a.size
        if n == 0:
            raise ValueError("upload 输入不能为空数组")
        _stats_upload(n * 4)
        out = ctypes.c_int64(0)
        with self._lock:
            _vulkan._check(
                _vulkan.dll.rvc_mem_upload(
                    self._handle, a.ctypes.data_as(F32_PTR), n, ctypes.byref(out)
                ),
                "rvc_mem_upload",
            )
        _mtrace(int(out.value), n * 4, "upload")
        return int(out.value)

    # -- 阶段D（D2 输入池）：池化上传/归还（BatchRunner 训练路径专用）----
    def _pooled_reserve(self, a: np.ndarray) -> tuple[int, bool]:
        """输入池化上传的 reserve 版（J9）：只取池 id（或新建 buffer），
        返回 ``(bid, need_copy)``——need_copy=True 时调用方须攒批后
        ``_batch_upload`` 补 copy；False 表示本次 upload() 已含 copy。

        原 ``_pooled_upload`` 拆为两段：池命中路径从「立即 rvc_mem_upload_to
        （一次 submit+fence）」改为「reserve + 攒批 + 批量一次 submit」，
        消除每步数百次上传各自的固定开销（实测上传 1.94s/1357 次为主因）。
        """
        a = _as_f32(a, "pooled upload 输入")
        n = a.size
        if n == 0:
            raise ValueError("pooled upload 输入不能为空数组")
        bucket = self._in_pool.get(n)
        if bucket:
            bid = bucket.pop()
            self._in_pool_count -= 1
            self._in_pool_bytes -= n * 4
            return int(bid), True
        bid = self.upload(a)  # 池 miss：新建 buffer（含 copy，比例低）
        self._in_pool_n[int(bid)] = n
        return int(bid), False

    def _batch_upload(self, pairs, max_bytes: int = 64 << 20) -> None:
        """J9 批量上传：把 ``[(bid, f32 ndarray), ...]`` 按 ≤``max_bytes``
        分批，每批一次 ``rvc_mem_upload_to_batch``（一次 staging memcpy +
        一次 cmd 提交 + 一次 fence 等待）。调用方保证数组引用存活到本调用。
        """
        if not pairs:
            return
        n_total = len(pairs)
        sizes_c = (ctypes.c_int64 * n_total)()
        datas_c = (F32_PTR * n_total)()
        ids_c = (ctypes.c_int64 * n_total)()
        for i, (bid, arr) in enumerate(pairs):
            sizes_c[i] = int(arr.size)
            datas_c[i] = arr.ctypes.data_as(F32_PTR)
            ids_c[i] = int(bid)
        # 分批：一批的字节和 ≤ max_bytes（ctypes 数组切片会退化成 list，
        # 故以索引范围传参）
        start = 0
        byte_sum = 0
        for i in range(n_total):
            b = int(sizes_c[i]) * 4
            if i > start and byte_sum + b > max_bytes:
                self._batch_upload_send(sizes_c, datas_c, ids_c, start, i)
                start = i
                byte_sum = 0
            byte_sum += b
        self._batch_upload_send(sizes_c, datas_c, ids_c, start, n_total)

    def _batch_upload_send(self, sizes_c, datas_c, ids_c, start: int, end: int) -> None:
        n = end - start
        if n == 0:
            return
        with self._lock:
            _vulkan._check(
                _vulkan.dll.rvc_mem_upload_to_batch(
                    self._handle, n,
                    ctypes.cast(ctypes.byref(sizes_c, start * ctypes.sizeof(ctypes.c_int64)),
                                ctypes.POINTER(ctypes.c_int64)),
                    ctypes.cast(ctypes.byref(datas_c, start * ctypes.sizeof(F32_PTR)),
                                ctypes.POINTER(F32_PTR)),
                    ctypes.cast(ctypes.byref(ids_c, start * ctypes.sizeof(ctypes.c_int64)),
                                ctypes.POINTER(ctypes.c_int64)),
                ),
                "rvc_mem_upload_to_batch",
            )

    def _batch_download(self, items, max_bytes: int = 64 << 20):
        """J18 批量下载：``[(buf_id, shape), ...]`` → ``[np.ndarray(f32)...]``
        按 ≤``max_bytes`` 分批，每批一次 ``rvc_mem_download_batch``（多条
        vkCmdCopyBuffer + 一次 submit + 一次 fence，替代逐 buffer 各自
        submit+fence 的 readback 串行）。引擎失败时逐项回退单发 download。
        """
        if not items:
            return []
        shapes = [tuple(s) for _, s in items]
        out = [np.empty(int(np.prod(s)) if s else 1, dtype=np.float32)
               for s in shapes]
        batches = []
        start, acc = 0, 0
        for i, (_, s) in enumerate(items):
            b = int(np.prod(s)) * 4 if s else 4
            if i > start and acc + b > max_bytes:
                batches.append((start, i))
                start, acc = i, 0
            acc += b
        batches.append((start, len(items)))
        for lo, hi in batches:
            n = hi - lo
            bufs_c = (ctypes.c_int64 * n)()
            dsts_c = (ctypes.POINTER(ctypes.c_uint8) * n)()
            sizes_c = (ctypes.c_int64 * n)()
            for j in range(lo, hi):
                ax = out[j]
                bufs_c[j - lo] = int(items[j][0])
                dsts_c[j - lo] = ax.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                sizes_c[j - lo] = int(ax.size)
            try:
                with self._lock:
                    _vulkan._check(
                        _vulkan.dll.rvc_mem_download_batch(
                            self._handle, bufs_c, dsts_c, sizes_c, n),
                        "rvc_mem_download_batch")
            except Exception:  # noqa: BLE001 引擎不支持/失败 → 逐项回退
                for j in range(lo, hi):
                    ax = out[j]
                    with self._lock:
                        _vulkan._check(
                            _vulkan.dll.rvc_mem_download(
                                self._handle, int(items[j][0]),
                                ax.ctypes.data_as(F32_PTR), int(ax.size)),
                            "rvc_mem_download")
        return [a.reshape(s) for a, s in zip(out, shapes)]

    def _pooled_upload(self, a: np.ndarray) -> int:
        """输入池化上传：优先复用池中同尺寸 buffer（rvc_mem_upload_to 覆盖写）。

        输入全量覆写（无输出池的 fill_zero 问题）——复用安全。池空/超限回退
        普通 upload（仍登记 _in_pool_n 供 release 归池）。
        """
        a = _as_f32(a, "pooled upload 输入")
        n = a.size
        if n == 0:
            raise ValueError("pooled upload 输入不能为空数组")
        bucket = self._in_pool.get(n)
        if bucket:
            bid = bucket.pop()
            self._in_pool_count -= 1
            self._in_pool_bytes -= n * 4
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_mem_upload_to(
                        self._handle, a.ctypes.data_as(F32_PTR), n, int(bid)
                    ),
                    "rvc_mem_upload_to",
                )
            return int(bid)
        bid = self.upload(a)
        self._in_pool_n[int(bid)] = n
        return bid

    def _pooled_free(self, bid: int, n: int) -> None:
        """输入 buffer 归还池（超限回退真释放）。"""
        if self._closed or sys.is_finalizing():
            return
        bid = int(bid)
        with self._lock:
            bucket = self._in_pool.setdefault(int(n), [])
            if (self._in_pool_count < _IN_POOL_MAX_BUFS
                    and self._in_pool_bytes + n * 4 <= _IN_POOL_MAX_BYTES):
                bucket.append(bid)
                self._in_pool_count += 1
                self._in_pool_bytes += n * 4
                self._in_pool_n.pop(bid, None)
                return
            _vulkan._check(
                _vulkan.dll.rvc_mem_free(self._handle, int(bid)), "rvc_mem_free"
            )
            self._in_pool_n.pop(bid, None)

    def download(self, buf: int, shape) -> np.ndarray:
        """从 GPU 下载 n 元素到新 float32 数组并按 shape 重塑。"""
        shape = tuple(shape)
        n = int(np.prod(shape)) if shape else 1
        _stats_download(n * 4)
        if os.environ.get("RVC_TRAIN_DOWNLOAD_DBG"):
            # J11 诊断：按字节去重打印下载调用栈（量化下载大头）
            _DLG = globals().setdefault("_DL_DBG_SET", set())
            _b = n * 4
            if _b not in _DLG and len(_DLG) < 40:
                _DLG.add(_b)
                import traceback as _tb  # noqa: PLC0415
                print(f"[DL-DBG] {_b} bytes shape={shape}", file=sys.stderr)
                _tb.print_stack(limit=6, file=sys.stderr)
        dst = np.zeros(n, dtype=np.float32)
        with self._lock:
            _vulkan._check(
                _vulkan.dll.rvc_mem_download(
                    self._handle, int(buf), dst.ctypes.data_as(F32_PTR), n
                ),
                "rvc_mem_download",
            )
        return dst.reshape(shape)

    def free(self, buf: int) -> None:
        """释放 GPU buffer（幂等；引擎已关闭或解释器退出阶段直接跳过）。

        若 ``buf`` 是 ``_alloc_output`` 分配的输出 buffer（登记于
        ``_out_registry``），则**归还输出池**而非真释放（下次同尺寸
        ``_alloc_output`` 复用，perf(P1-6)）。池满/超限（条目数或
        总字节预算 ``_OUT_POOL_MAX_BYTES``）时回退真释放。
        """
        if self._closed or sys.is_finalizing():
            return
        _mtfree(int(buf))
        with self._lock:
            n = self._out_registry.pop(int(buf), None)
            if n is not None and not _OUT_NO_POOL:
                bucket = self._out_pool.setdefault(n, [])
                if (len(bucket) < _OUT_POOL_PER_SIZE
                        and self._out_pool_count < _OUT_POOL_MAX_BUFS
                        and self._out_pool_bytes + n * 4 <= _OUT_POOL_MAX_BYTES):
                    bucket.append(int(buf))
                    self._out_pool_count += 1
                    self._out_pool_bytes += n * 4
                    return
            _vulkan._check(
                _vulkan.dll.rvc_mem_free(self._handle, int(buf)), "rvc_mem_free"
            )

    def _alloc_output(self, n: int) -> int:
        """按 test_ffi.py 语义分配输出 buffer（优先复用池中同尺寸 buffer）。

        复用不重新 upload 零数组（perf(P1-6)）：engine 全部 shader 对输出
        buffer 均为全量覆写（见模块 docstring 的池安全性说明），池中脏内容
        不会影响结果。返回的 buffer 登记进 ``_out_registry``，``free()`` 时
        自动归还。

        QF(2026-09-21)：**诊断发现输出池复用引入音质回归**——mem_alloc 未初始化
        buffer 被复用时，若某 shader 未全量覆写输出（边界/padding 区），旧值残留
        污染高频（2s 带宽 15.8k->20.1k，-61.8dB->-57.5dB 复现）。默认 ``RVC_OUT_NO_POOL=1``
        不复用（每次 mem_alloc 新分配，GPU 分配无 PCIe 传输，实测性能无回归）；
        精修 shader 覆写后可设 ``RVC_OUT_NO_POOL=0`` 恢复复用。
        """
        if not _OUT_NO_POOL:
            with self._lock:
                bucket = self._out_pool.get(n)
                if bucket:
                    bid = bucket.pop()
                    self._out_pool_count -= 1
                    self._out_pool_bytes -= n * 4
                    self._out_registry[bid] = n
                    # F3-B：池复用 buffer 为旧值（污染源）——返回前 GPU 填零
                    # （engine rvc_mem_fill_zero，无 PCIe 传输，快）。
                    _vulkan._check(
                        _vulkan.dll.rvc_mem_fill_zero(self._handle, int(bid), int(n * 4)),
                        "rvc_mem_fill_zero",
                    )
                    return bid
                # D1a-1：精确命中失败 → 找满足 n <= k <= ceil(n*ratio) 的
                # 最小桶复用（减 mem_alloc 10ms 级/次）。registry 记录真实
                # 尺寸 k，free() 归回原桶；shader 全量覆写前 n 元素即可。
                if _OUT_POOL_GE_RATIO > 1:
                    cap = int(n * _OUT_POOL_GE_RATIO)
                    best = None
                    for k in self._out_pool:
                        if k >= n and k <= cap and (best is None or k < best):
                            best = k
                    if best is not None:
                        bucket = self._out_pool[best]
                        if bucket:
                            bid = bucket.pop()
                            if not bucket:
                                del self._out_pool[best]
                            self._out_pool_count -= 1
                            self._out_pool_bytes -= best * 4
                            self._out_registry[bid] = best
                            _vulkan._check(
                                _vulkan.dll.rvc_mem_fill_zero(
                                    self._handle, int(bid), int(n * 4)),
                                "rvc_mem_fill_zero",
                            )
                            return bid
        # P1 perf：输出 buffer 由 shader 全量覆写，无需零填充——用 engine 未
        # 初始化分配（免 24.5MB×25 次零数组 PCIe 上传，2s 推理 611MB→0）。
        try:
            bid = self.mem_alloc(n * 4)
        except Exception:  # noqa: BLE001  # 旧 DLL 无 rvc_mem_alloc 时回退零上传
            zeros = np.zeros(n, dtype=np.float32)
            bid = self.upload(zeros)
        with self._lock:
            self._out_registry[bid] = n
        return bid

    def mem_alloc(self, nbytes: int) -> int:
        """分配 nbytes 字节未初始化 DEVICE_LOCAL buffer（引擎 rvc_mem_alloc）。"""
        out = ctypes.c_int64(0)
        _vulkan._check(
            _vulkan.dll.rvc_mem_alloc(self._handle, int(nbytes), ctypes.byref(out)),
            "rvc_mem_alloc",
        )
        _mtrace(int(out.value), nbytes, "mem_alloc")
        return int(out.value)

    def persistent_upload(self, a: np.ndarray) -> PersistentBuffer:
        """上传 numpy 数组到 GPU 并返回**常驻** buffer 包装（不自动释放）。

        与 ``upload`` 的区别：返回的 :class:`PersistentBuffer` 由调用方持有，
        可跨多次算子调用复用（如 conv1d 的 ``buf_w``），需要显式 ``.free()``
        释放（``__del__`` 兜底）。适用于权重等"一次上传、反复参与计算"的数据。
        """
        a = _as_f32(a, "persistent_upload 输入")
        buf = self.upload(a)
        return PersistentBuffer(self, buf, a.shape, a.nbytes)

    # -- 算子 ------------------------------------------------------------
    def matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        buf_a: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 矩阵乘：c = a @ b（行主序）。

        a: ``[M, K]`` f32，b: ``[K, N]`` f32 → 返回 ``[M, N]`` f32。
        总元素 < ``_THRESHOLD`` 时直接用 numpy。

        ``buf_a`` / ``buf_b`` 为常驻 buffer（对应数组已上传到 GPU，且**不释放**），
        传入时跳过对应 upload —— 适合线性层权重（``b``）常驻。传已释放/失效的
        buffer 自动回退普通 upload。
        """
        a = _as_f32(a, "matmul a")
        b = _as_f32(b, "matmul b")
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"matmul 仅支持 2D 输入，got a.ndim={a.ndim} b.ndim={b.ndim}")
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise ValueError(f"matmul 维度不匹配: a={a.shape} b={b.shape}")
        if M * N == 0:
            return np.zeros((M, N), dtype=np.float32)
        if a.size + b.size < _THRESHOLD:
            return (a @ b).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            b_id = _resolve_buf(buf_b)
            if b_id is None:
                b_id = self.upload(b)
                to_free.append(b_id)
            c_id = self._alloc_output(M * N)
            to_free.append(c_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_matmul(
                        self._handle, a_id, b_id, c_id, M, K, N
                    ),
                    "rvc_matmul",
                )
            return self.download(c_id, (M, N))
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def add(
        self,
        a: np.ndarray,
        b: np.ndarray,
        buf_a: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 逐元素加法（广播后展平），返回与 a 同形状 f32 数组。

        ``buf_a``/``buf_b`` 为常驻 buffer（传入时跳过对应 upload，且不释放）。
        """
        a = _as_f32(a, "add a")
        b = _as_f32(b, "add b")
        if a.shape != b.shape:
            raise ValueError(f"add 形状不一致: a={a.shape} b={b.shape}")
        n = a.size
        if n == 0:
            return np.empty(a.shape, dtype=np.float32)
        if a.size < _THRESHOLD:
            return (a + b).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            b_id = _resolve_buf(buf_b)
            if b_id is None:
                b_id = self.upload(b)
                to_free.append(b_id)
            c_id = self._alloc_output(n)
            to_free.append(c_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_add(self._handle, a_id, b_id, c_id, n),
                    "rvc_add",
                )
            return self.download(c_id, a.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def mul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        buf_a: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 逐元素乘法（广播后展平），返回与 a 同形状 f32 数组。

        ``buf_a``/``buf_b`` 为常驻 buffer（传入时跳过对应 upload，且不释放）。
        """
        a = _as_f32(a, "mul a")
        b = _as_f32(b, "mul b")
        if a.shape != b.shape:
            raise ValueError(f"mul 形状不一致: a={a.shape} b={b.shape}")
        n = a.size
        if n == 0:
            return np.empty(a.shape, dtype=np.float32)
        if a.size < _THRESHOLD:
            return (a * b).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            b_id = _resolve_buf(buf_b)
            if b_id is None:
                b_id = self.upload(b)
                to_free.append(b_id)
            c_id = self._alloc_output(n)
            to_free.append(c_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_mul(self._handle, a_id, b_id, c_id, n),
                    "rvc_mul",
                )
            return self.download(c_id, a.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def relu_inplace(self, a: np.ndarray, buf_a: PersistentBuffer | None = None) -> np.ndarray:
        """GPU 原生 ReLU（引擎内 in-place；上传副本计算，不修改调用方数组）。

        返回 ``max(a, 0)`` 的 f32 新数组。``buf_a`` 为常驻 buffer（可选）。
        """
        a = _as_f32(a, "relu 输入")
        if a.size == 0:
            return np.empty(a.shape, dtype=np.float32)
        if a.size < _THRESHOLD:
            return np.maximum(a, 0.0).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_relu(self._handle, a_id, a.size),
                    "rvc_relu",
                )
            return self.download(a_id, a.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def conv1d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray | None = None,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 1D 卷积：out = conv(x, w) (+ b)，对齐 ``torch.nn.functional.conv1d``。

        x: ``[B, C_in, L]`` f32，w: ``[C_out, C_in, K]`` f32，b: ``[C_out]`` 或 None。
        padding 支持 int 或 ``(pad_l, pad_r)``；输出 ``[B, C_out, L_out]``。
        总元素 < ``_THRESHOLD`` 时直接用 numpy（与 ``runtime.nn`` 语义一致）。

        ``buf_w`` / ``buf_b`` 为常驻权重 buffer（P1 优化）：传入时跳过对应
        upload 并在 finally 中**不释放**（所有权归调用方，由 GPUWeights 统一
        管理）。bias 走 GPU shader 内加法，数值与 numpy 加 bias 差 ~1e-7。
        传已释放/失效的 buffer 自动回退普通 upload。
        """
        x = _as_f32(x, "conv1d x")
        w = _as_f32(w, "conv1d w")
        if x.ndim != 3 or w.ndim != 3:
            raise ValueError(f"conv1d 仅支持 3D 输入，got x.ndim={x.ndim} w.ndim={w.ndim}")
        B, C_in, L = x.shape
        C_out, C_in2, K = w.shape
        if C_in != C_in2:
            raise ValueError(f"conv1d 通道不匹配: x={x.shape} w={w.shape}")
        if isinstance(padding, (tuple, list)):
            pad_l, pad_r = int(padding[0]), int(padding[1])
        else:
            pad_l = pad_r = int(padding)
        stride = int(stride)
        dilation = int(dilation)
        if stride < 1 or dilation < 1 or pad_l < 0 or pad_r < 0 or K < 1:
            raise ValueError("conv1d 参数非法（stride/dilation≥1，padding≥0，K≥1）")

        oL = (L + pad_l + pad_r - dilation * (K - 1) - 1) // stride + 1
        if oL <= 0:
            return np.zeros((B, C_out, 0), dtype=np.float32)  # 与 nn.conv1d 语义一致
        if x.size < _THRESHOLD:
            from runtime import nn as _nn  # noqa: PLC0415  # 惰性避免包初始化环

            return _nn._conv1d_numpy(x, w, b, stride, padding, dilation)
        if B * C_out * oL > _GRID_POINTS_MAX and not _in_split_state():
            # 引擎 grid.x 上限防护（超长音频）：**时间维切分，逐块 GPU 计算后拼接**
            # （不回退 CPU——numpy 卷积会生成 GB 级中间数组导致内存爆+GPU 闲置）。
            return _conv1d_split_gpu(self, x, w, b, stride, padding, dilation,
                                     K, (B, C_out, oL))

        has_bias = b is not None
        b_arr = _as_f32(b, "conv1d b") if has_bias else None
        if has_bias and b_arr.shape[0] != C_out:
            raise ValueError(f"conv1d bias 长度 {b_arr.shape[0]} != C_out {C_out}")

        to_free = []
        try:
            x_id = self.upload(x)
            to_free.append(x_id)
            w_id = _resolve_buf(buf_w)
            if w_id is None:
                w_id = self.upload(w)
                to_free.append(w_id)
            if has_bias:
                b_id = _resolve_buf(buf_b)
                if b_id is None:
                    b_id = self.upload(b_arr)
                    to_free.append(b_id)
            else:
                b_id = 0
            o_id = self._alloc_output(B * C_out * oL)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_conv1d(
                        self._handle, x_id, w_id, b_id, o_id,
                        B, C_in, L, C_out, K, stride, pad_l, pad_r, dilation,
                    ),
                    "rvc_conv1d",
                )
            return self.download(o_id, (B, C_out, oL))
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def conv_transpose1d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray | None = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 1D 转置卷积：out = conv_transpose1d(x, w) (+ b)。

        对齐 ``torch.nn.functional.conv_transpose1d``：x: ``[B, C_in, L]`` f32，
        w: ``[C_in, C_out, K]``（PyTorch 布局），b: ``[C_out]`` 或 None。
        ``oL = (L-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1``。
        总元素 < ``_THRESHOLD`` 时直接用 numpy（与 ``runtime.nn`` 语义一致）。

        ``buf_w`` / ``buf_b`` 为常驻权重 buffer（P1 优化，vits dec 权重常驻
        收益最大）：传入时跳过对应 upload 且不释放。输出 ``[B, C_out, oL]``。
        """
        x = _as_f32(x, "conv_transpose1d x")
        w = _as_f32(w, "conv_transpose1d w")
        if x.ndim != 3 or w.ndim != 3:
            raise ValueError(
                f"conv_transpose1d 仅支持 3D 输入，got x.ndim={x.ndim} w.ndim={w.ndim}"
            )
        B, C_in, L = x.shape
        C_in2, C_out, K = w.shape
        if C_in != C_in2:
            raise ValueError(f"conv_transpose1d 通道不匹配: x={x.shape} w={w.shape}")
        stride = int(stride)
        padding = int(padding)
        output_padding = int(output_padding)
        dilation = int(dilation)
        if stride < 1 or padding < 0 or output_padding < 0 or dilation < 1 or K < 1:
            raise ValueError("conv_transpose1d 参数非法（stride/dilation≥1，padding/output_padding≥0）")

        oL = (L - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
        if oL <= 0:
            return np.zeros((B, C_out, 0), dtype=np.float32)
        if x.size < _THRESHOLD:
            from runtime import nn as _nn  # noqa: PLC0415  # 惰性避免包初始化环

            return _nn._conv_transpose1d_numpy(x, w, b, stride, padding, output_padding, dilation)
        if B * C_out * oL > _GRID_POINTS_MAX and not _in_split_state():
            # 引擎 grid.x 上限防护（长音频 vits dec ups 会超限）：
            # **时间维切分，逐块 GPU 计算后拼接**（不回退 CPU——numpy
            # 转置卷积会生成 GB 级中间数组导致内存爆 + GPU 闲置）。
            return _conv_t1d_split_gpu(self, x, w, b, stride, padding,
                                       output_padding, dilation, K,
                                       (B, C_out, oL))

        has_bias = b is not None
        b_arr = _as_f32(b, "conv_transpose1d b") if has_bias else None
        if has_bias and b_arr.shape[0] != C_out:
            raise ValueError(f"conv_transpose1d bias 长度 {b_arr.shape[0]} != C_out {C_out}")

        to_free = []
        try:
            x_id = self.upload(x)
            to_free.append(x_id)
            w_id = _resolve_buf(buf_w)
            if w_id is None:
                w_id = self.upload(w)
                to_free.append(w_id)
            if has_bias:
                b_id = _resolve_buf(buf_b)
                if b_id is None:
                    b_id = self.upload(b_arr)
                    to_free.append(b_id)
            else:
                b_id = 0
            o_id = self._alloc_output(B * C_out * oL)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_conv_t1d(
                        self._handle, x_id, w_id, b_id, o_id,
                        B, C_in, L, C_out, K, stride, padding, output_padding, dilation,
                    ),
                    "rvc_conv_t1d",
                )
            return self.download(o_id, (B, C_out, oL))
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def conv_transpose2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray | None = None,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        output_padding: int | tuple = 0,
        buf_w: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 2D 转置卷积：out = conv_transpose2d(x, w) (+ b)。

        对齐 ``torch.nn.functional.conv_transpose2d``：x: ``[B, C_in, OH, OW]``
        f32，w: ``[C_in, C_out, KH, KW]``（PyTorch 布局），b: ``[C_out]`` 或 None。
        stride/padding/output_padding 支持 int 或 ``(h, w)`` 元组（对称填充，
        引擎 conv_t2d 无 dilation，dilation 恒为 1）：
        ``oH = (OH-1)*sh - 2*pad_h + KH + output_pad_h``（w_out 同理）。
        总元素 < ``_THRESHOLD`` 时直接用 numpy（与 ``runtime.nn`` 语义一致）。

        引擎 ``rvc_conv_t2d`` **无 bias**：b 非 None 时 GPU 算卷积后 host 侧
        numpy 广播加 bias。``buf_w`` 为常驻权重 buffer（传入时跳过对应
        upload 且不释放）。输出 ``[B, C_out, oH, oW]``。
        """
        x = _as_f32(x, "conv_transpose2d x")
        w = _as_f32(w, "conv_transpose2d w")
        if x.ndim != 4 or w.ndim != 4:
            raise ValueError(
                f"conv_transpose2d 仅支持 4D 输入，got x.ndim={x.ndim} w.ndim={w.ndim}"
            )
        B, C_in, OH, OW = x.shape
        C_in2, C_out, KH, KW = w.shape
        if C_in != C_in2:
            raise ValueError(f"conv_transpose2d 通道不匹配: x={x.shape} w={w.shape}")
        if isinstance(stride, (tuple, list)):
            sh, sw = int(stride[0]), int(stride[1])
        else:
            sh = sw = int(stride)
        if isinstance(padding, (tuple, list)):
            ph, pw = int(padding[0]), int(padding[1])
        else:
            ph = pw = int(padding)
        if isinstance(output_padding, (tuple, list)):
            opad_h, opad_w = int(output_padding[0]), int(output_padding[1])
        else:
            opad_h = opad_w = int(output_padding)
        if sh < 1 or sw < 1 or ph < 0 or pw < 0 or opad_h < 0 or opad_w < 0 \
                or KH < 1 or KW < 1:
            raise ValueError("conv_transpose2d 参数非法（stride≥1，padding/output_padding≥0）")

        oH = (OH - 1) * sh - 2 * ph + KH + opad_h
        oW = (OW - 1) * sw - 2 * pw + KW + opad_w
        if oH <= 0 or oW <= 0:
            return np.zeros((B, C_out, max(oH, 0), max(oW, 0)), dtype=np.float32)
        if x.size < _THRESHOLD:
            return _conv_transpose2d_numpy(x, w, b, stride, padding,
                                           output_padding)
        if B * C_out * oH * oW > _GRID_POINTS_MAX and not _in_split_state():
            # 引擎 grid.x 上限防护：超大输出（判别器反向 gx 不会触发，此处
            # 仅兜底）抛错，由调用方（conv2d_backward_gpu）捕获回退 numpy。
            raise ValueError(
                f"conv_transpose2d 输出 {B * C_out * oH * oW} 超引擎 grid 上限"
            )

        has_bias = b is not None
        b_arr = _as_f32(b, "conv_transpose2d b") if has_bias else None
        if has_bias and b_arr.shape[0] != C_out:
            raise ValueError(f"conv_transpose2d bias 长度 {b_arr.shape[0]} != C_out {C_out}")

        to_free = []
        try:
            x_id = self.upload(x)
            to_free.append(x_id)
            w_id = _resolve_buf(buf_w)
            if w_id is None:
                w_id = self.upload(w)
                to_free.append(w_id)
            o_id = self._alloc_output(B * C_out * oH * oW)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_conv_t2d(
                        self._handle, x_id, w_id, o_id,
                        B, C_in, OH, OW, C_out, KH, KW, sh, sw, ph, pw,
                        opad_h, opad_w, oH, oW,
                    ),
                    "rvc_conv_t2d",
                )
            out = self.download(o_id, (B, C_out, oH, oW))
            if has_bias:
                out = out + b_arr.reshape(1, -1, 1, 1)
            return out
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def conv2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray | None = None,
        stride=1,
        padding=0,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 2D 卷积（对称填充、dilation=1）：对齐 ``torch.nn.functional.conv2d``。

        x: ``[B, C_in, H, W]`` f32，w: ``[C_out, C_in, KH, KW]`` f32，b: ``[C_out]``。
        stride/padding 支持 int 或 ``(sh, sw)`` / ``(ph, pw)``（对称 pad）。
        总元素 < ``_THRESHOLD`` 时用 numpy。``buf_w``/``buf_b`` 为常驻权重 buffer。

        注意：引擎 kernel 无 dilation（非1由 ``runtime.nn.conv2d`` 回退 numpy）。
        """
        x = _as_f32(x, "conv2d x")
        w = _as_f32(w, "conv2d w")
        if x.ndim != 4 or w.ndim != 4:
            raise ValueError(f"conv2d 仅支持 4D 输入，got x.ndim={x.ndim} w.ndim={w.ndim}")
        B, C_in, H, W = x.shape
        C_out, C_in2, KH, KW = w.shape
        if C_in != C_in2:
            raise ValueError(f"conv2d 通道不匹配: x={x.shape} w={w.shape}")
        if isinstance(stride, (tuple, list)):
            stride_h, stride_w = int(stride[0]), int(stride[1])
        else:
            stride_h = stride_w = int(stride)
        if isinstance(padding, (tuple, list)):
            pad_h, pad_w = int(padding[0]), int(padding[1])
        else:
            pad_h = pad_w = int(padding)
        if stride_h < 1 or stride_w < 1 or pad_h < 0 or pad_w < 0 or KH < 1 or KW < 1:
            raise ValueError("conv2d 参数非法（stride≥1，padding≥0，K≥1）")

        OH = (H + 2 * pad_h - KH) // stride_h + 1
        OW = (W + 2 * pad_w - KW) // stride_w + 1
        if OH <= 0 or OW <= 0:
            return np.zeros((B, C_out, max(OH, 0), max(OW, 0)), dtype=np.float32)
        if x.size < _THRESHOLD and not _FORCE_CONV2D_GPU:
            from runtime import nn as _nn  # noqa: PLC0415

            return _nn._conv2d_numpy(x, w, b, stride, padding, 1)

        has_bias = b is not None
        b_arr = _as_f32(b, "conv2d b") if has_bias else None
        if has_bias and b_arr.shape[0] != C_out:
            raise ValueError(f"conv2d bias 长度 {b_arr.shape[0]} != C_out {C_out}")
        if OH * OW > _GRID_POINTS_MAX // 8 and not _in_split_state():
            # 引擎 grid.x 上限防护（长音频 rmvpe conv2d 会超限）：
            # **时间维(W)切分，逐块 GPU 计算后拼接**（不回退 CPU）。
            # 同 BatchRunner：D2 后约束仍是"空间 OL ≤ 上限"（2^31//8≈2.68亿），
            # 组数 gx=ceilDiv(ol,tile2) 远小于驱动实测 2^32-1。
            return _conv2d_split_gpu(self, x, w, b_arr, (stride_h, stride_w),
                                     (pad_h, pad_w), (B, C_out, OH, OW))

        to_free = []
        try:
            x_id = self.upload(x)
            to_free.append(x_id)
            w_id = _resolve_buf(buf_w)
            if w_id is None:
                w_id = self.upload(w)
                to_free.append(w_id)
            if has_bias:
                b_id = _resolve_buf(buf_b)
                if b_id is None:
                    b_id = self.upload(b_arr)
                    to_free.append(b_id)
            else:
                b_id = 0
            o_id = self._alloc_output(B * C_out * OH * OW)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_conv2d(
                        self._handle, x_id, w_id, b_id, o_id,
                        B, C_in, H, W, C_out, KH, KW, pad_h, pad_w, stride_h, stride_w,
                    ),
                    "rvc_conv2d",
                )
            return self.download(o_id, (B, C_out, OH, OW))
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def insert_zeros_2x(self, x: np.ndarray) -> int:
        """P15b：GPU 内 stride-2 插零（convT 的 x_up 生成）。

        输入 ``x`` [B, C_in, H, W]（通道合并为 C=B*C_in 后按 [C,H,W] 展开），
        返回**新分配的 GPU buffer**：形状 [B*C_in, 2H-1, 2W-1]、隔位 0
        （out[c,i,j] = i,j 均偶 ? x[c,i//2,j//2] : 0）。所有权归调用方
        （须 ``free()``；本方法不下载输入也不释放输入）。

        数值与 host ``np.zeros + [::2,::2]=x`` 逐位一致（0 就是 0、原值不
        重排），后续喂给同一 ``rvc_conv2d`` kernel 输出与 host 路径完全
        相同（P15b 硬验收 maxdiff=0）。
        """
        x = _as_f32(x, "insert_zeros_2x x")
        if x.ndim == 4:
            B, C, H, W = x.shape
            C = B * C
        elif x.ndim == 3:
            C, H, W = x.shape
        else:
            raise ValueError(
                f"insert_zeros_2x 仅支持 [C,H,W] 或 [B,C,H,W]，got ndim={x.ndim}"
            )
        if C <= 0 or H <= 0 or W <= 0:
            raise ValueError("insert_zeros_2x 形状非法")
        n_out = C * (2 * H - 1) * (2 * W - 1)
        if n_out > _GRID_POINTS_MAX:
            raise ValueError(
                f"insert_zeros_2x 输出 {n_out} 元素超 _GRID_POINTS_MAX"
            )
        x_id = self.upload(x)  # 上传原图（远小于 x_up）
        o_id = self._alloc_output(n_out)
        try:
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_insert_zeros_2x(
                        self._handle, x_id, o_id, C, H, W,
                    ),
                    "rvc_insert_zeros_2x",
                )
        except Exception:  # noqa: BLE001
            try:
                self.free(o_id)
            except RuntimeError:
                pass
            self.free(x_id)
            raise
        self.free(x_id)  # 插零输出已含全部数据，输入立即可释放
        return o_id

    def conv2d_from_buf(
        self,
        x_id: int,
        x_shape,
        w: np.ndarray,
        b: np.ndarray | None = None,
        stride=1,
        padding=0,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU conv2d，输入为**已有 GPU buffer**（跳过 x 上传，P15b）。

        与 ``conv2d`` 同一 kernel / 同一数学（数值逐位一致），仅输入来源
        不同：``x_id`` 是 GPU 中 [B, C_in, H, W] 的 buffer（如
        ``insert_zeros_2x`` 的产物），``x_shape`` 给出其形状。调用方保留
        ``x_id`` 所有权（本方法不释放）。小张量 numpy 回退与 split 防护
        不适用 —— 调用方负责只在确需 GPU 时使用（本路径已由 rmvpe
        ``_convt_gpu_try`` 把关；输出超 split 阈值直接抛错回退）。
        """
        w = _as_f32(w, "conv2d w")
        B, C_in, H, W = (int(v) for v in tuple(x_shape))
        C_out, C_in2, KH, KW = w.shape
        if C_in != C_in2:
            raise ValueError(f"conv2d 通道不匹配: x={x_shape} w={w.shape}")
        if isinstance(stride, (tuple, list)):
            stride_h, stride_w = int(stride[0]), int(stride[1])
        else:
            stride_h = stride_w = int(stride)
        if isinstance(padding, (tuple, list)):
            pad_h, pad_w = int(padding[0]), int(padding[1])
        else:
            pad_h = pad_w = int(padding)
        if stride_h < 1 or stride_w < 1 or pad_h < 0 or pad_w < 0 or KH < 1 or KW < 1:
            raise ValueError("conv2d 参数非法（stride≥1，padding≥0，K≥1）")

        OH = (H + 2 * pad_h - KH) // stride_h + 1
        OW = (W + 2 * pad_w - KW) // stride_w + 1
        if OH <= 0 or OW <= 0:
            return np.zeros((B, C_out, max(OH, 0), max(OW, 0)), dtype=np.float32)
        if OH * OW > _GRID_POINTS_MAX // 8:
            raise ValueError(
                "conv2d_from_buf 输出超 split 阈值（调用方须回退 host 路径）"
            )

        has_bias = b is not None
        b_arr = _as_f32(b, "conv2d b") if has_bias else None
        if has_bias and b_arr.shape[0] != C_out:
            raise ValueError(f"conv2d bias 长度 {b_arr.shape[0]} != C_out {C_out}")

        to_free = []
        try:
            w_id = _resolve_buf(buf_w)
            if w_id is None:
                w_id = self.upload(w)
                to_free.append(w_id)
            if has_bias:
                b_id = _resolve_buf(buf_b)
                if b_id is None:
                    b_id = self.upload(b_arr)
                    to_free.append(b_id)
            else:
                b_id = 0
            o_id = self._alloc_output(B * C_out * OH * OW)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_conv2d(
                        self._handle, int(x_id), w_id, b_id, o_id,
                        B, C_in, H, W, C_out, KH, KW, pad_h, pad_w, stride_h, stride_w,
                    ),
                    "rvc_conv2d",
                )
            return self.download(o_id, (B, C_out, OH, OW))
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def embedding(
        self,
        ids: np.ndarray,
        table: np.ndarray,
        buf_table: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 查表嵌入：out = table[ids]（纯 gather）。

        ids: 任意形状的整数数组；table: ``[V, E]`` f32。返回 ``ids.shape + [E]``。
        ids 须全部落在 ``[0, V)``（越界 / 负索引 / 小张量自动回退 numpy，语义与
        ``runtime.nn.embedding`` 一致）。``buf_table`` 为常驻 buffer（嵌入表常驻）。
        """
        ids = np.asarray(ids)
        table = _as_f32(table, "embedding table")
        if table.ndim != 2:
            raise ValueError(f"embedding table 须为 2D，got ndim={table.ndim}")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(
                f"embedding ids 必须是整数数组，got dtype={ids.dtype}"
            )
        V, E = table.shape
        if ids.size == 0:
            return np.empty(ids.shape + (E,), dtype=np.float32)
        # 越界/负索引：保持 nn.embedding 语义走 numpy（负索引按 Python 语义取倒数行）
        if int(ids.min()) < 0 or int(ids.max()) >= V:
            from runtime import nn as _nn  # noqa: PLC0415

            return _nn._embedding_numpy(ids, table)
        if ids.size < _THRESHOLD:
            from runtime import nn as _nn  # noqa: PLC0415

            return _nn._embedding_numpy(ids, table)

        ids32 = np.ascontiguousarray(ids, dtype=np.int32)
        to_free = []
        try:
            # int32 字节按 f32 槽位上传（引擎把 buffer 当不透明字节处理）
            ids_id = self.upload(ids32.view(np.float32))
            to_free.append(ids_id)
            tbl_id = _resolve_buf(buf_table)
            if tbl_id is None:
                tbl_id = self.upload(table)
                to_free.append(tbl_id)
            o_id = self._alloc_output(ids.size * E)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_embed(
                        self._handle, ids_id, tbl_id, o_id, ids.size, V, E
                    ),
                    "rvc_embed",
                )
            return self.download(o_id, ids.shape + (E,))
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def add_inplace(
        self,
        a: np.ndarray,
        b: np.ndarray,
        buf_a: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 就地加法：a = a + b（引擎内写回上传副本，不改调用方数组）。

        返回与 a 同形状 f32 新数组；比 ``add`` 少一次输出分配（上传 a、b 共 2 次）。
        ``buf_a`` 为常驻 buffer 时在其上就地计算（该常驻 buffer 内容会改变）。
        """
        a = _as_f32(a, "add_inplace a")
        b = _as_f32(b, "add_inplace b")
        if a.shape != b.shape:
            raise ValueError(f"add_inplace 形状不一致: a={a.shape} b={b.shape}")
        n = a.size
        if n == 0:
            return np.empty(a.shape, dtype=np.float32)
        if a.size < _THRESHOLD:
            return (a + b).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            b_id = _resolve_buf(buf_b)
            if b_id is None:
                b_id = self.upload(b)
                to_free.append(b_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_add_inplace(self._handle, a_id, b_id, n),
                    "rvc_add_inplace",
                )
            return self.download(a_id, a.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def mul_inplace(
        self,
        a: np.ndarray,
        b: np.ndarray,
        buf_a: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 就地乘法：a = a * b（引擎内写回上传副本，不改调用方数组）。"""
        a = _as_f32(a, "mul_inplace a")
        b = _as_f32(b, "mul_inplace b")
        if a.shape != b.shape:
            raise ValueError(f"mul_inplace 形状不一致: a={a.shape} b={b.shape}")
        n = a.size
        if n == 0:
            return np.empty(a.shape, dtype=np.float32)
        if a.size < _THRESHOLD:
            return (a * b).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            b_id = _resolve_buf(buf_b)
            if b_id is None:
                b_id = self.upload(b)
                to_free.append(b_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_mul_inplace(self._handle, a_id, b_id, n),
                    "rvc_mul_inplace",
                )
            return self.download(a_id, a.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def leaky_relu_inplace(
        self,
        a: np.ndarray,
        negative_slope: float = 0.1,
        buf_a: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 原生 LeakyReLU（引擎内 in-place；上传副本计算，不改调用方数组）。

        语义对齐 ``nn.leaky_relu`` / ``F.leaky_relu``：``where(x >= 0, x, slope*x)``。
        返回 f32 新数组。``buf_a`` 为常驻 buffer（可选）。
        """
        a = _as_f32(a, "leaky_relu 输入")
        if a.size == 0:
            return np.empty(a.shape, dtype=np.float32)
        if a.size < _THRESHOLD:
            return np.where(a >= 0, a, float(negative_slope) * a).astype(np.float32)

        to_free = []
        try:
            a_id = _resolve_buf(buf_a)
            if a_id is None:
                a_id = self.upload(a)
                to_free.append(a_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_leaky_relu(
                        self._handle, a_id, a.size, _f32_bits(negative_slope)
                    ),
                    "rvc_leaky_relu",
                )
            return self.download(a_id, a.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def copy(
        self,
        src: np.ndarray,
        buf_src: PersistentBuffer | None = None,
        out_buf: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU 复制：dst = src（flat over N floats）。

        用于在就地算子覆写前保留输入（如 ResBlock 残差）。``buf_src`` 为
        常驻 buffer（可选）；``out_buf`` 指定输出常驻 buffer（可选，须已
        分配 ≥ src.size 个 f32）。
        """
        src = _as_f32(src, "copy 输入")
        if src.size == 0:
            return np.empty(src.shape, dtype=np.float32)
        if src.size < _THRESHOLD:
            return src.copy()

        to_free = []
        try:
            src_id = _resolve_buf(buf_src)
            if src_id is None:
                src_id = self.upload(src)
                to_free.append(src_id)
            dst_id = _resolve_buf(out_buf)
            if dst_id is None:
                dst_id = self._alloc_output(src.size)
                to_free.append(dst_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_copy(self._handle, dst_id, src_id, src.size),
                    "rvc_copy",
                )
            return self.download(dst_id, src.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-5,
        buf_gamma: PersistentBuffer | None = None,
        buf_beta: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU LayerNorm（对最后一维，biased 方差），x 任意 ``[..., C]``。

        内部 reshape 为 ``[rows, cols]`` 逐行归一；gamma/beta: ``[C]``。
        总元素 < ``_THRESHOLD`` 时直接用 numpy。``buf_gamma``/``buf_beta``
        为常驻 buffer（传入时跳过对应 upload，且不释放）。
        """
        x = _as_f32(x, "layer_norm x")
        gamma = _as_f32(gamma, "layer_norm gamma")
        beta = _as_f32(beta, "layer_norm beta")
        if gamma.ndim != 1 or beta.ndim != 1:
            raise ValueError(f"layer_norm gamma/beta 须为 1D，got gamma.ndim={gamma.ndim} beta.ndim={beta.ndim}")
        cols = gamma.shape[0]
        if beta.shape[0] != cols or x.shape[-1] != cols:
            raise ValueError(f"layer_norm 维度不匹配: x[-1]={x.shape[-1]} gamma={gamma.shape} beta={beta.shape}")
        if x.size == 0:
            return np.empty(x.shape, dtype=np.float32)
        rows = x.size // cols
        if x.size < _THRESHOLD:
            xr = x.reshape(rows, cols)
            mean = xr.mean(axis=-1, keepdims=True)
            var = xr.var(axis=-1, keepdims=True)
            xn = (xr - mean) / np.sqrt(var + eps)
            return (xn * gamma + beta).reshape(x.shape)

        to_free = []
        try:
            x_id = self.upload(x)
            to_free.append(x_id)
            g_id = _resolve_buf(buf_gamma)
            if g_id is None:
                g_id = self.upload(gamma)
                to_free.append(g_id)
            bt_id = _resolve_buf(buf_beta)
            if bt_id is None:
                bt_id = self.upload(beta)
                to_free.append(bt_id)
            o_id = self._alloc_output(x.size)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_layernorm(self._handle, x_id, g_id, bt_id, o_id, rows, cols, float(eps)),
                    "rvc_layernorm",
                )
            return self.download(o_id, x.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """GPU softmax（对最后一维，exp(x-max) 数值稳定），x 任意 ``[..., C]``。

        内部 reshape 为 ``[rows, cols]`` 逐行归一。总元素 < ``_THRESHOLD`` 时
        直接用 numpy。
        """
        x = _as_f32(x, "softmax x")
        if x.size == 0:
            return np.empty(x.shape, dtype=np.float32)
        cols = x.shape[-1]
        if cols == 0:
            return np.empty(x.shape, dtype=np.float32)
        rows = x.size // cols
        if x.size < _THRESHOLD:
            xr = x.reshape(rows, cols)
            m = xr.max(axis=-1, keepdims=True)
            e = np.exp(xr - m)
            return (e / e.sum(axis=-1, keepdims=True)).reshape(x.shape)

        x_id = o_id = None
        try:
            x_id = self.upload(x)
            o_id = self._alloc_output(x.size)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_softmax(self._handle, x_id, o_id, rows, cols),
                    "rvc_softmax",
                )
            return self.download(o_id, x.shape)
        finally:
            for buf in (x_id, o_id):
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass

    def rmsnorm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        eps: float = 1e-5,
        buf_gamma: PersistentBuffer | None = None,
    ) -> np.ndarray:
        """GPU RMSNorm：``out = x / sqrt(mean(x^2) + eps) * gamma``（对最后一维）。

        x 任意 ``[..., C]``，gamma: ``[C]``。总元素 < ``_THRESHOLD`` 时直接用
        numpy。``buf_gamma`` 为常驻 buffer（传入时跳过对应 upload，且不释放）。
        """
        x = _as_f32(x, "rmsnorm x")
        gamma = _as_f32(gamma, "rmsnorm gamma")
        if gamma.ndim != 1:
            raise ValueError(f"rmsnorm gamma 须为 1D，got gamma.ndim={gamma.ndim}")
        cols = gamma.shape[0]
        if x.shape[-1] != cols:
            raise ValueError(f"rmsnorm 维度不匹配: x[-1]={x.shape[-1]} gamma={gamma.shape}")
        if x.size == 0:
            return np.empty(x.shape, dtype=np.float32)
        rows = x.size // cols
        if x.size < _THRESHOLD:
            xr = x.reshape(rows, cols)
            ms = (xr * xr).mean(axis=-1, keepdims=True)
            return (xr / np.sqrt(ms + eps) * gamma).reshape(x.shape)

        to_free = []
        try:
            x_id = self.upload(x)
            to_free.append(x_id)
            g_id = _resolve_buf(buf_gamma)
            if g_id is None:
                g_id = self.upload(gamma)
                to_free.append(g_id)
            o_id = self._alloc_output(x.size)
            to_free.append(o_id)
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_rmsnorm(self._handle, x_id, g_id, o_id, rows, cols, float(eps)),
                    "rvc_rmsnorm",
                )
            return self.download(o_id, x.shape)
        finally:
            for buf in to_free:
                if buf:
                    try:
                        self.free(buf)
                    except RuntimeError:
                        pass


# --------------------------------------------------------------------------
# BatchRunner：录制式批量提交（P1，多算子单次 submit）
# --------------------------------------------------------------------------
class BatchTensor:
    """BatchRunner 里一个 GPU 中间结果的句柄（录制期不可读）。

    由 ``BatchRunner`` 的算子方法返回：commit 之前它只是"录制队列里的一笔
    记录"；``commit()`` 之后调用 ``numpy()`` 才真正下载为 float32 数组。
    同一 runner 的 tensor 可以继续作为后续算子的输入（复用 GPU buffer，
    不重复上传，且保持 batch 内数据依赖顺序）。
    """

    __slots__ = ("_runner", "_buf", "shape", "_view_off")

    def __init__(self, runner: "BatchRunner", buf: int, shape, view_off: int = 0):
        self._runner = runner
        self._buf = int(buf)
        self.shape = tuple(shape)
        self._view_off = int(view_off)  # buffer 内字节偏移（T1.1 子视图支持，默认 0）

    def __array__(self, dtype=None):
        """隐式 numpy 化（T-H7 链式）：commit（幂等）+ wait + 下载。

        numpy 反向 bp 消费 BatchTensor 梯度时自动断链下载；GPU 链式 bp
        检测 ``isinstance(go, BatchTensor)`` 优先走 GPU 录制。
        """
        self._runner.commit()
        arr = self.numpy()
        if dtype is not None and np.dtype(dtype) != np.float32:
            return arr.astype(dtype)
        return arr

    @property
    def id(self) -> int | None:
        return self._buf

    @property
    def ndim(self) -> int:
        """T-H7 链式：与 numpy 对齐（numpy bp 隐式使用 ndim 时自动可用）。"""
        return len(self.shape)

    def __getitem__(self, key):
        """T-H7 链式：numpy bp 切片消费 BatchTensor 时自动下载断链。"""
        return np.asarray(self)[key]

    # T-H8：常用 numpy 方法委托（tape fwd 前向对 BatchTensor 输入无感——
    # 自动 commit+下载再计算；ref id 由调用方 _as_ref 保持）。
    def _np(self):
        return np.asarray(self)

    def mean(self, axis=None, keepdims=False):
        return self._np().mean(axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        return self._np().sum(axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return self._np().max(axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return self._np().min(axis=axis, keepdims=keepdims)

    def var(self, axis=None, keepdims=False):
        return self._np().var(axis=axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False):
        return self._np().std(axis=axis, keepdims=keepdims)

    def astype(self, dtype, *a, **k):
        return self._np().astype(dtype, *a, **k)

    def transpose(self, *axes):
        return self._np().transpose(*axes)

    def copy(self):
        return self._np().copy()

    def flatten(self):
        return self._np().flatten()

    def __add__(self, o):
        return self._np() + o

    def __radd__(self, o):
        return o + self._np()

    def __sub__(self, o):
        return self._np() - o

    def __rsub__(self, o):
        return o - self._np()

    def __mul__(self, o):
        return self._np() * o

    def __rmul__(self, o):
        return o * self._np()

    def __truediv__(self, o):
        return self._np() / o

    def __rtruediv__(self, o):
        return o / self._np()

    @property
    def valid(self) -> bool:
        return self._buf is not None and self._runner is not None

    def reshape(self, *shape):
        """返回同一 GPU buffer 的新形状视图（不产生任何算子/拷贝）。

        只做形状簿记（BatchRunner 的算子用 shape 做校验与输出重塑）；
        -1 语义与 numpy 一致。跨 runner / 已失效时报错。
        """
        if self._buf is None:
            raise RuntimeError("BatchTensor 已失效（batch 已 release）")
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            given = [int(s) for s in shape[0]]
        else:
            given = [int(s) for s in shape]
        n = 1
        auto = None
        for i, s in enumerate(given):
            if s == -1:
                if auto is not None:
                    raise ValueError("reshape 只能有一个 -1")
                auto = i
            else:
                n *= s
        total = int(np.prod(self.shape))
        if total == 0 or total % n != 0:
            raise ValueError(f"reshape 尺寸不匹配: {self.shape} -> {tuple(given)}")
        if auto is not None:
            given[auto] = total // n
        return BatchTensor(self._runner, self._buf, tuple(given), self._view_off)

    def slice_batch(self, lo: int, hi: int) -> "BatchTensor":
        """零拷贝段视图：同一 GPU buffer 的 [lo, hi) 段（沿最后一维）。

        T1.1 子视图支持：返回带 ``_view_off`` 的视图句柄（偏移 = 原偏移 +
        ``lo * 4`` 字节），引擎侧用 ``descriptorInfoView`` 绑定子区间。
        仅对 gather 式 kernel（conv_t1d 等）安全——段边界无跨列依赖。
        纯簿记，不产生算子/拷贝；默认路径不调用（零回归）。
        """
        if self._buf is None:
            raise RuntimeError("BatchTensor 已失效（batch 已 release）")
        lo, hi = int(lo), int(hi)
        if not (0 <= lo < hi <= self.shape[-1]):
            raise ValueError(f"slice_batch 越界: [0, {self.shape[-1]}) 切 [{lo}, {hi})")
        new_shape = (*self.shape[:-1], hi - lo)
        return BatchTensor(self._runner, self._buf, new_shape, self._view_off + lo * 4)

    def numpy(self) -> np.ndarray:
        """commit 后下载为 float32 数组（按 ``shape`` 重塑）。

        未 commit 时自动先 ``commit()``（幂等：空批次 no-op；T-H7 链式下
        numpy bp / 梯度合并消费 BatchTensor 时自动断链提交当前批次）。
        commit 之前调用抛 ``RuntimeError``；``runner.release()`` 后
        调用抛 ``RuntimeError``（GPU buffer 已释放）。若 commit 是
        异步的（``async_=True``），此处会自动 ``wait()`` 后再下载。
        """
        if self._buf is None:
            raise RuntimeError("BatchTensor 已失效（batch 已 release）")
        if not self._runner._committed:
            self._runner.commit()
        self._runner.wait()
        return self._runner._ctx.download(self._buf, self.shape)

    def __repr__(self):
        state = "freed" if self._buf is None else f"buf={self._buf}"
        committed = self._runner._committed if self._runner is not None else False
        return f"BatchTensor(shape={self.shape}, {state}, committed={committed})"


class BatchRunner:
    """积累多次算子调用 → 一次 ``rvc_batch_commit``（一次 submit + 一次
    fence-wait），把逐次调用的固定开销（AMD Radeon Pro VII 实测每调用
    8-25 ms 的 mutex+submit+同步等待）摊到整个批次上。

    用法::

        br = BatchRunner(get_context())
        c = br.matmul(a, b)                 # 只录制，不提交
        o = br.conv1d(c, w, bias)           # c 是 GPU 中间结果，直接复用
        br.add_inplace(o, residual)         # 就地累加
        br.commit()                         # 一次提交全部 dispatch
        out = o.numpy()                     # 提交后取回数组
        br.release()                        # 释放本 runner 的 GPU buffer

    数值与 ``VulkanContext`` 逐次调用**完全一致**（同一 shader、同一
    push-constant 布局、同一 grid）。与逐次调用不同，BatchRunner **不做**
    ``_THRESHOLD`` 小张量回退——它是显式强制 GPU 的录制式接口。

    输入接受三种类型：
      - numpy float32 数组（自动上传，release 时释放）；
      - ``BatchTensor``（本 runner 先前的输出，复用 buffer）；
      - ``PersistentBuffer``（常驻权重，不释放，所有权归调用方）。
    """

    def __init__(self, ctx: VulkanContext):
        self._ctx = ctx
        self._records = []  # (op, a_id, b_id, c_id, p0..p9)
        self._tensors: list[BatchTensor] = []  # 输出 tensor（release 时释放）
        self._owned: set[int] = set()  # 本 runner 上传/分配的 buffer id
        # 阶段D（D2 输入池）：池化路径上传的输入 buffer（release 时归输入池
        # 而非真释放——全量覆写安全，省 alloc+free 包络）。
        self._owned_inputs: set[int] = set()
        # J9 批量上传：录制期 reserve 的 (bid, arr) 排队，commit 前一次
        # rvc_mem_upload_to_batch（消每次上传 submit+fence 固定开销）。
        self._up_q: list = []
        self._committed = False
        self._async_pending = False  # 有未 wait 的异步提交在途
        self._recycled: set[int] = set()  # tensor_done 已回收（本地池/已作废）的 id
        self._local_pool: dict[int, list[int]] = {}  # 录制期 tensor_done 归还的
        # 输出 buffer（按元素数分桶）。**只在本 runner 的 _alloc_output 复用**：
        # 队列录制顺序保证"复用后的写入 op 必然排在旧读者之后"（同 runner 内
        # GPU 顺序执行 + 每 dispatch 前全局 barrier），跨 runner/跨推理的复用
        # 由 release() 归还全局池（ctx._out_pool）承接（此时 GPU 已完成）。
        self._released = False
        self._elapsed = 0.0
        self._lock = threading.RLock()
        self._owns_batch_lock = True
        # 独占引擎批次状态机：begin → add... → commit → release 全程持锁
        # （见模块 docstring 线程语义：begin 会重置未提交批次，跨线程交错
        # 会污染引擎批次；GPU 提交被串行化，CPU 录制/检索仍可并行）。
        _BATCH_LIFECYCLE_LOCK.acquire()
        try:
            with self._lock:
                _vulkan._check(
                    _vulkan.dll.rvc_batch_begin(self._ctx._handle), "rvc_batch_begin"
                )
        except BaseException:
            self._owns_batch_lock = False
            _BATCH_LIFECYCLE_LOCK.release()
            raise

    # -- 内部 ----------------------------------------------------------
    def _ensure_begin(self) -> None:
        """commit 之后再次 add 时自动开启新批次（懒重置）。"""
        if self._committed:
            _vulkan._check(
                _vulkan.dll.rvc_batch_begin(self._ctx._handle), "rvc_batch_begin"
            )
            self._committed = False

    def _resolve_input(self, x, name: str, buf=None):
        """把输入解析为 ``(buf_id, shape, owned)``。

        ``buf``（PersistentBuffer 常驻权重）优先：提供了就直接用它，x 仅
        作形状参考。其次 ``BatchTensor``/``PersistentBuffer`` 直接用其
        buffer；否则按 numpy 上传（记入 ``_owned``）。
        """
        bid = _resolve_buf(buf)
        if bid is not None:
            return bid, tuple(getattr(buf, "shape", ())), False
        if isinstance(x, BatchTensor):
            if x._runner is not self:
                raise ValueError(f"{name}: 不能混用不同 BatchRunner 的 tensor")
            if x._buf is None:
                raise RuntimeError(f"{name}: tensor 已失效（batch 已 release）")
            return x._buf, x.shape, False
        if isinstance(x, PersistentBuffer):
            if not x.valid:
                raise RuntimeError(f"{name}: PersistentBuffer 已释放")
            return x.id, x.shape, False
        if os.environ.get("RVC_TRAIN_BR_FWD_DBG"):
            import traceback as _tb
            print(f"[resolve-dbg] {name} type={type(x)}", file=sys.stderr)
        arr = _as_f32(x, name)
        if arr.size == 0:
            raise ValueError(f"{name} 不能为空数组")
        if os.environ.get("RVC_TRAIN_UPLOAD_STATS"):
            # J10 诊断：按 name 统计 numpy 上传字节（量化 f16 可行性）
            _UPS = globals().setdefault("_UPS", {})
            _UPS[name.split(".")[0]] = _UPS.get(name.split(".")[0], 0) + arr.size * 4
            if os.environ.get("RVC_TRAIN_UPS_DBG"):
                _SEEN = globals().setdefault("_UPS_DBG_SEEN", set())
                _tag = name.split(".")[0]
                if _tag not in _SEEN and _tag in (
                        "matmul b", "conv_transpose2d w"):
                    _SEEN.add(_tag)
                    import traceback as _tb  # noqa: PLC0415
                    print(f"[UPS-DBG] {name} {arr.size * 4} bytes shape={arr.shape}",
                          file=sys.stderr)
                    _tb.print_stack(limit=7, file=sys.stderr)
        if os.environ.get("RVC_TRAIN_IN_POOL", "1") == "1":
            # 阶段D（D2 输入池）：池化上传（复用同尺寸 buffer 覆盖写）；
            # release() 时 _owned_inputs 统一归池。
            # J9：reserve + 攒批（copy 延后到 commit 前批量一次 submit）。
            b, need_copy = self._ctx._pooled_reserve(arr)
            if need_copy:
                self._up_q.append((b, arr))
            self._owned_inputs.add(b)
            return b, arr.shape, True
        b = self._ctx.upload(arr)
        self._owned.add(b)
        return b, arr.shape, True

    def _alloc_output(self, n: int) -> int:
        """分配输出 buffer：优先复用本 runner 录制期已回收（tensor_done）的同尺寸
        buffer（本地池，队列顺序保证安全），其次全局输出池，最后 upload 零数组。"""
        bucket = self._local_pool.get(n)
        if bucket:
            bid = bucket.pop()
            self._owned.add(bid)
            self._recycled.discard(bid)  # 重新投入本 batch，release 时正常处理
            return bid
        bid = self._ctx._alloc_output(n)
        self._owned.add(bid)
        return bid

    def _record(
        self,
        op: int,
        a_id,
        b_id,
        c_id,
        ps,
        out_shape,
        tensor_buf: int | None = None,
    ) -> BatchTensor | None:
        """录制一笔算子调用。

        ``a_id/b_id/c_id`` 是传给 rvc_batch_add 的三个句柄（按 op 取用）；
        ``tensor_buf`` 是 BatchTensor 绑定的**输出** buffer（默认为 c_id；
        conv1d 的 c 参数位是 bias，输出在 p9，故需显式指定）。
        """
        ps = [int(p) for p in ps] + [0] * (11 - len(ps))
        # P1-001 前置 shape 校验（统一入口）：输出点数超引擎单 dispatch
        # 上限时在录制前拒绝（抛特定错误由模型层捕获回退），不允许
        # 越界 shape 直接下发 rvc_batch_add 造成静默越界损坏。
        if out_shape is not None:
            n_out = 1
            for d in out_shape:
                n_out *= int(d)
            if n_out > _GRID_POINTS_MAX:
                raise RuntimeError(
                    "vulkan_batch_too_large: op=%s 输出 %s 共 %d 点超 GPU 上限"
                    % (op, tuple(out_shape), n_out))
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add(
                    self._ctx._handle, op, int(a_id), int(b_id), int(c_id), *ps
                ),
                f"rvc_batch_add(op={op})",
            )
            self._records.append((op, int(a_id), int(b_id), int(c_id), ps))
        if out_shape is None:
            return None
        t = BatchTensor(self, tensor_buf if tensor_buf is not None else c_id, out_shape)
        with self._lock:
            self._tensors.append(t)
        return t

    # -- 算子（只录制，不提交） ----------------------------------------
    def matmul(
        self,
        a,
        b,
        c: BatchTensor | None = None,
        buf_a: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """录制 matmul：c = a @ b（行主序）。

        a: ``[M,K]``，b: ``[K,N]``（numpy / BatchTensor / PersistentBuffer）。
        c: 可选输出 BatchTensor（复用其 buffer）；None 时自动分配。
        ``buf_a``/``buf_b``：常驻权重（与 ``VulkanContext.matmul`` 同义）。
        返回 GPU 句柄 ``BatchTensor([M,N])``。
        """
        a_id, a_shape, _ = self._resolve_input(a, "matmul a", buf_a)
        b_id, b_shape, _ = self._resolve_input(b, "matmul b", buf_b)
        if not a_shape or not b_shape:
            raise ValueError("matmul 需要 a/b 的形状信息")
        M, K = a_shape
        K2, N = b_shape
        if K != K2:
            raise ValueError(f"matmul 维度不匹配: a={a_shape} b={b_shape}")
        if c is not None:
            if not isinstance(c, BatchTensor) or c._buf is None:
                raise ValueError("matmul 的 c 必须是本 runner 的有效 BatchTensor")
            c_id = c._buf
        else:
            c_id = self._alloc_output(M * N)
        return self._record(1, a_id, b_id, c_id, (M, K, N), (M, N))

    def conv1d(
        self,
        x,
        w,
        b,
        out: BatchTensor | None = None,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """录制 conv1d：out = conv(x, w) (+ b)，对齐 ``VulkanContext.conv1d``。

        x: ``[B,C_in,L]``，w: ``[C_out,C_in,K]``，b: ``[C_out]`` 或 None。
        padding 支持 int 或 ``(pad_l, pad_r)``。返回 ``BatchTensor([B,C_out,oL])``。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv1d x")
        w_id, w_shape, _ = self._resolve_input(w, "conv1d w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv1d 需要 x/w 的形状信息")
        B, C_in, L = x_shape
        C_out, C_in2, K = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv1d 通道不匹配: x={x_shape} w={w_shape}")
        if isinstance(padding, (tuple, list)):
            pad_l, pad_r = int(padding[0]), int(padding[1])
        else:
            pad_l = pad_r = int(padding)
        stride = int(stride)
        dilation = int(dilation)
        if stride < 1 or dilation < 1 or pad_l < 0 or pad_r < 0 or K < 1:
            raise ValueError("conv1d 参数非法（stride/dilation≥1，padding≥0，K≥1）")
        oL = (L + pad_l + pad_r - dilation * (K - 1) - 1) // stride + 1
        if oL <= 0:
            raise ValueError(f"conv1d 输出长度非正 (oL={oL})")
        if B * C_out * oL > _GRID_POINTS_MAX:
            # 引擎 grid.x 上限防护（超长音频）：抛特定错误由模型层捕获后回退逐次/numpy
            raise RuntimeError(
                "vulkan_batch_too_large: conv1d 输出 %d 点超 GPU 上限"
                % (B * C_out * oL))
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv1d b", buf_b)
            if b_shape and b_shape[0] != C_out:
                raise ValueError(f"conv1d bias 长度 {b_shape[0]} != C_out {C_out}")
        else:
            b_id = 0
        if out is not None:
            if not isinstance(out, BatchTensor) or out._buf is None:
                raise ValueError("conv1d 的 out 必须是本 runner 的有效 BatchTensor")
            o_id = out._buf
        else:
            o_id = self._alloc_output(B * C_out * oL)
        # rvc_batch_add 约定：a=x, b=w, c=bias(0=无)，p0..p8 维度，p9=out。
        # BatchTensor 绑定 o_id（tensor_buf），而非 bias。
        return self._record(
            2, x_id, w_id, b_id,
            (B, C_in, L, C_out, K, stride, pad_l, pad_r, dilation, o_id),
            (B, C_out, oL),
            tensor_buf=o_id,
        )

    def conv1d_groups(
        self,
        x,
        w,
        b,
        out: BatchTensor | None = None,
        stride: int = 1,
        padding=0,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """录制分组 1D 卷积 fwd（DiscriminatorS conv1d_groups，op=23）。

        x: ``[B,C_in,L]``，w: ``[C_out, C_in_g, K]``（C_in_g = C_in/groups，
        PyTorch groups 权重格式——w 只含组内输入通道），b: ``[C_out]`` 或
        None。groups = C_in // C_in_g（由 w.shape[1] 推出）；组 g 的输出通道
        co = g*(C_out/groups)+co_g 只消费 x 的组段通道。dilation 恒 1
        （kernel 固定；判别器 S 无 dilation）。返回
        ``BatchTensor([B,C_out,oL])``。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv1d_groups x")
        w_id, w_shape, _ = self._resolve_input(w, "conv1d_groups w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv1d_groups 需要 x/w 的形状信息")
        B, C_in, L = x_shape
        C_out, C_in_g, K = w_shape
        if C_in % C_in_g != 0:
            raise ValueError(f"conv1d_groups 通道不可分: C_in={C_in} C_in_g={C_in_g}")
        if isinstance(padding, (tuple, list)):
            pad_l, pad_r = int(padding[0]), int(padding[1])
        else:
            pad_l = pad_r = int(padding)
        stride = int(stride)
        if stride < 1 or pad_l < 0 or pad_r < 0 or K < 1:
            raise ValueError("conv1d_groups 参数非法（stride≥1，padding≥0，K≥1）")
        oL = (L + pad_l + pad_r - (K - 1) - 1) // stride + 1  # dilation=1
        if oL <= 0:
            raise ValueError(f"conv1d_groups 输出长度非正 (oL={oL})")
        if B * C_out * oL > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: conv1d_groups 输出 %d 点超 GPU 上限"
                % (B * C_out * oL))
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv1d_groups b", buf_b)
            if b_shape and b_shape[0] != C_out:
                raise ValueError(f"conv1d_groups bias 长度 {b_shape[0]} != C_out {C_out}")
        else:
            b_id = 0
        if out is not None:
            if not isinstance(out, BatchTensor) or out._buf is None:
                raise ValueError("conv1d_groups 的 out 必须是本 runner 的有效 BatchTensor")
            o_id = out._buf
        else:
            o_id = self._alloc_output(B * C_out * oL)
        # rvc_batch_add op=23 约定：a=x, b=w, c=bias(0=无)，p9=out；
        # p0=B p1=C_in p2=C_in_g p3=L p4=C_out p5=K p6=stride p7=pad_l p8=pad_r。
        return self._record(
            23, x_id, w_id, b_id,
            (B, C_in, C_in_g, L, C_out, K, stride, pad_l, pad_r, o_id),
            (B, C_out, oL),
            tensor_buf=o_id,
        )

    def conv2d(
        self,
        x,
        w,
        b,
        out: BatchTensor | None = None,
        stride=1,
        padding=0,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """录制 conv2d（P0-1：rmvpe UNet 卷积 GPU 批量化，用户要求小算子也必须真走 GPU）。

        x: ``[B,C_in,H,W]``，w: ``[C_out,C_in,KH,KW]``，b: ``[C_out]`` 或 None。
        stride/padding 支持 int 或 ``(s, s)``/``(p, p)``（对称）。返回
        ``BatchTensor([B,C_out,OH,OW])``。dilation=1（引擎 kernel 不支持）。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv2d x")
        w_id, w_shape, _ = self._resolve_input(w, "conv2d w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv2d 需要 x/w 的形状信息")
        B, C_in, H, W = x_shape
        C_out, C_in2, KH, KW = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv2d 通道不匹配: x={x_shape} w={w_shape}")
        if isinstance(stride, (tuple, list)):
            sh, sw = int(stride[0]), int(stride[1])
        else:
            sh = sw = int(stride)
        if isinstance(padding, (tuple, list)):
            ph, pw = int(padding[0]), int(padding[1])
        else:
            ph = pw = int(padding)
        if sh < 1 or sw < 1 or ph < 0 or pw < 0 or KH < 1 or KW < 1:
            raise ValueError("conv2d 参数非法（stride≥1，padding≥0，K≥1）")
        OH = (H + 2 * ph - KH) // sh + 1
        OW = (W + 2 * pw - KW) // sw + 1
        if OH <= 0 or OW <= 0:
            raise ValueError(f"conv2d 输出尺寸非正 ({OH},{OW})")
        if OH * OW > _GRID_POINTS_MAX // 8:
            # 引擎 grid.x 上限防护（engine 修复后 gx=ceilDiv(OH*OW,tile)，
            # D2 后上限 ≈ 2^31//8 ≈ 2.68亿 空间点（tile≥16，gx ≤ 16.7M <<
            # 驱动 2^32-1）；总点数 B*C_out*OH*OW 不再是约束——原 total
            # 判断使整段 150s rmvpe 首层（30.7M 点）误报。
            raise RuntimeError(
                "vulkan_batch_too_large: conv2d 输出 %d 点超 GPU 上限"
                % (B * C_out * OH * OW))
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv2d b", buf_b)
            if b_shape and b_shape[0] != C_out:
                raise ValueError(f"conv2d bias 长度 {b_shape[0]} != C_out {C_out}")
        else:
            b_id = 0
        if out is not None:
            if not isinstance(out, BatchTensor) or out._buf is None:
                raise ValueError("conv2d 的 out 必须是本 runner 的有效 BatchTensor")
            o_id = out._buf
        else:
            o_id = self._alloc_output(B * C_out * OH * OW)
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add_conv2d(
                    self._ctx._handle,
                    int(x_id), int(w_id), int(b_id), int(o_id),
                    int(B), int(C_in), int(H), int(W), int(C_out),
                    int(KH), int(KW), int(ph), int(pw), int(sh), int(sw),
                ),
                "rvc_batch_add_conv2d",
            )
            self._records.append(("conv2d", int(x_id), int(w_id), int(b_id),
                                  (B, C_in, H, W, C_out, KH, KW, ph, pw, sh, sw, o_id)))
        t = BatchTensor(self, o_id, (B, C_out, OH, OW))
        with self._lock:
            self._tensors.append(t)
        return t

    def conv_transpose2d(
        self,
        x,
        w,
        b,
        out: BatchTensor | None = None,
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        buf_w: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """录制 conv_transpose2d（阶段A A4 v2：conv2d 反向 gx 路径批量提交）。

        x: ``[B,C_in,OH,OW]``，w: ``[C_in,C_out,KH,KW]``（PyTorch 布局），
        b: ``[C_out]`` 或 None；stride/padding/output_padding 支持 int 或
        (h,w) 元组；dilation 恒 1。返回 ``BatchTensor([B,C_out,H_out,W_out])``，
        ``H_out=(OH-1)*sh - 2*ph + KH + opad_h``（W 同理）。opad 须 < stride
        （引擎校验，与 ``VulkanContext.conv_transpose2d`` 一致）。
        ``buf_w``：常驻权重 buffer（J11 权重缓存，跳过重复上传）。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv_transpose2d x")
        w_id, w_shape, _ = self._resolve_input(w, "conv_transpose2d w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv_transpose2d 需要 x/w 的形状信息")
        B, C_in, OH, OW = x_shape
        C_in2, C_out, KH, KW = w_shape
        if C_in != C_in2:
            raise ValueError(
                f"conv_transpose2d 通道不匹配: x={x_shape} w={w_shape}")
        if isinstance(stride, (tuple, list)):
            sh, sw = int(stride[0]), int(stride[1])
        else:
            sh = sw = int(stride)
        if isinstance(padding, (tuple, list)):
            ph, pw = int(padding[0]), int(padding[1])
        else:
            ph = pw = int(padding)
        if isinstance(output_padding, (tuple, list)):
            opad_h, opad_w = int(output_padding[0]), int(output_padding[1])
        else:
            opad_h = opad_w = int(output_padding)
        if sh < 1 or sw < 1 or ph < 0 or pw < 0 or opad_h < 0 or opad_w < 0 \
                or KH < 1 or KW < 1 or opad_h >= sh or opad_w >= sw:
            raise ValueError("conv_transpose2d 参数非法（stride≥1，pad≥0，"
                             "opad≥0 且 < stride）")
        h_out = (OH - 1) * sh - 2 * ph + KH + opad_h
        w_out = (OW - 1) * sw - 2 * pw + KW + opad_w
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"conv_transpose2d 输出尺寸非正 ({h_out},{w_out})")
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv_transpose2d b")
            if b_shape and b_shape[0] != C_out:
                raise ValueError(
                    f"conv_transpose2d bias 长度 {b_shape[0]} != C_out {C_out}")
        else:
            b_id = 0
        if out is not None:
            if not isinstance(out, BatchTensor) or out._buf is None:
                raise ValueError(
                    "conv_transpose2d 的 out 必须是本 runner 的有效 BatchTensor")
            o_id = out._buf
        else:
            o_id = self._alloc_output(B * C_out * h_out * w_out)
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add_conv_t2d(
                    self._ctx._handle,
                    int(x_id), int(w_id), int(b_id), int(o_id),
                    int(B), int(C_in), int(OH), int(OW), int(C_out),
                    int(KH), int(KW), int(sh), int(sw), int(ph), int(pw),
                    int(opad_h), int(opad_w), int(h_out), int(w_out),
                ),
                "rvc_batch_add_conv_t2d",
            )
            self._records.append(
                ("conv_t2d", int(x_id), int(w_id), int(b_id),
                 (B, C_in, OH, OW, C_out, KH, KW, sh, sw, ph, pw,
                  opad_h, opad_w, h_out, w_out, o_id)))
        t = BatchTensor(self, o_id, (B, C_out, h_out, w_out))
        with self._lock:
            self._tensors.append(t)
        return t

    def im2col_1d(self, x, B, C, T, oL, K_dil, stride, pad_l,
                  dilation: int = 1) -> BatchTensor:
        """录制 GPU im2col（阶段E C3）：x [B,C,T] → xw [B*oL, C*K_dil]
        （row-major）。J10：支持 dilation≠1（间隔窗 gather），此前仅
        dilation=1。gather 无计算 → 与 host 视图链逐位一致。
        """
        x_id, x_shape, _ = self._resolve_input(x, "im2col_1d x")
        if not x_shape:
            raise ValueError("im2col_1d 需要 x 的形状信息")
        if tuple(x_shape) != (B, C, T):
            raise ValueError(
                f"im2col_1d x 形状 {x_shape} != ({B},{C},{T})")
        if int(dilation) != 1:
            # P1-012/P1-007：引擎 gather 对 dilation≠1 的间隔窗索引边界有差一
            # （实测 T=1,pad>=1,dil>=2 共 12 组 maxdiff≈1~2，读取错位）——
            # dilation≠1 改用 host 视图组装（sliding_window 语义，逐位一致），
            # 上传为普通输入后照常参与录制（后续 matmul 等仍在 GPU）。
            xw = _host_im2col_1d(
                np.asarray(x, np.float32), B, C, T, oL, K_dil, stride, pad_l, dilation)
            hw_id, _, _ = self._resolve_input(xw, "im2col_1d host-xw")
            t = BatchTensor(self, hw_id, (B * oL, C * K_dil))
            with self._lock:
                self._tensors.append(t)
            return t
        o_id = self._alloc_output(B * oL * C * K_dil)
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add_im2col_1d(
                    self._ctx._handle, int(x_id), int(o_id),
                    int(B), int(C), int(T), int(oL), int(K_dil),
                    int(stride), int(pad_l), int(dilation),
                ),
                "rvc_batch_add_im2col_1d",
            )
            self._records.append(
                ("im2col_1d", int(x_id), int(o_id),
                 (B, C, T, oL, K_dil, stride, pad_l, dilation, o_id)))
        t = BatchTensor(self, o_id, (B * oL, C * K_dil))
        with self._lock:
            self._tensors.append(t)
        return t

    def im2col_2d(self, x, B, C, H, W, OH, OW, KH, KW, sh, sw, ph, pw
                  ) -> BatchTensor:
        """录制 GPU im2col 2D（阶段E C3 v2）：x [B,C,H,W] →
        xw [B*OH*OW, C*KH*KW]（row-major 行=(b,oh,ow) 行内=(c,kh,kw)，
        dilation=1）。gather 无计算 → 与 host 视图链逐位一致。
        """
        x_id, x_shape, _ = self._resolve_input(x, "im2col_2d x")
        if not x_shape:
            raise ValueError("im2col_2d 需要 x 的形状信息")
        if tuple(x_shape) != (B, C, H, W):
            raise ValueError(
                f"im2col_2d x 形状 {x_shape} != ({B},{C},{H},{W})")
        o_id = self._alloc_output(B * OH * OW * C * KH * KW)
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add_im2col_2d(
                    self._ctx._handle, int(x_id), int(o_id),
                    int(B), int(C), int(H), int(W),
                    int(OH), int(OW), int(KH), int(KW),
                    int(sh), int(sw), int(ph), int(pw),
                ),
                "rvc_batch_add_im2col_2d",
            )
            self._records.append(
                ("im2col_2d", int(x_id), int(o_id),
                 (B, C, H, W, OH, OW, KH, KW, sh, sw, ph, pw, o_id)))
        t = BatchTensor(self, o_id, (B * OH * OW, C * KH * KW))
        with self._lock:
            self._tensors.append(t)
        return t

    def conv_transpose1d(
        self,
        x,
        w,
        b,
        out: BatchTensor | None = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """录制 conv_transpose1d：对齐 ``VulkanContext.conv_transpose1d``。

        x: ``[B,C_in,L]``，w: ``[C_in,C_out,K]``（PyTorch 布局），b: ``[C_out]`` 或 None。
        ``oL = (L-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1``。
        ``buf_w``/``buf_b`` 为常驻权重（与逐次同义）。返回 ``BatchTensor([B,C_out,oL])``。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv_transpose1d x")
        w_id, w_shape, _ = self._resolve_input(w, "conv_transpose1d w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv_transpose1d 需要 x/w 的形状信息")
        B, C_in, L = x_shape
        C_in2, C_out, K = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv_transpose1d 通道不匹配: x={x_shape} w={w_shape}")
        stride = int(stride)
        padding = int(padding)
        output_padding = int(output_padding)
        dilation = int(dilation)
        if stride < 1 or padding < 0 or output_padding < 0 or dilation < 1 or K < 1:
            raise ValueError(
                "conv_transpose1d 参数非法（stride/dilation≥1，padding/output_padding≥0）"
            )
        oL = (L - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
        if oL <= 0:
            raise ValueError(f"conv_transpose1d 输出长度非正 (oL={oL})")
        if B * C_out * oL > _GRID_POINTS_MAX:
            # 引擎 grid.x 上限防护（超长音频）：抛特定错误由模型层捕获后回退逐次/numpy
            raise RuntimeError(
                "vulkan_batch_too_large: conv_transpose1d 输出 %d 点超 GPU 上限"
                % (B * C_out * oL))
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv_transpose1d b", buf_b)
            if b_shape and b_shape[0] != C_out:
                raise ValueError(f"conv_transpose1d bias 长度 {b_shape[0]} != C_out {C_out}")
        else:
            b_id = 0
        if out is not None:
            if not isinstance(out, BatchTensor) or out._buf is None:
                raise ValueError("conv_transpose1d 的 out 必须是本 runner 的有效 BatchTensor")
            o_id = out._buf
        else:
            o_id = self._alloc_output(B * C_out * oL)
        # rvc_batch_add 约定（op=5）：a=x, b=w, c=bias(0=无)，p0..p8 维度，p9=out。
        return self._record(
            5, x_id, w_id, b_id,
            (B, C_in, L, C_out, K, stride, padding, output_padding, dilation, o_id),
            (B, C_out, oL),
            tensor_buf=o_id,
        )

    def conv_t1d_seg(
        self,
        x,
        w,
        b,
        out: BatchTensor,
        stride: int,
        padding: int,
        output_padding: int,
        dilation: int,
        seg_len: int,
        l_seg: int,
        in_off: int,
        lo_off: int,
        l_out_full: int,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """T1.1 分段版 conv_t1d：x/out 整 buffer 绑定 + push 绝对寻址（GPU 内段流转）。

        ``seg_len`` = 本段输出列数；``l_seg`` = 本段输入窗口长度（边界检查）；
        ``in_off`` = 输入起始列（绝对）；``lo_off`` = 输出段起始列（绝对）；
        ``l_out_full`` = 输出 buffer 行全长（行步长）。整 buffer 无子视图偏移。
        仅 gather 式 kernel（conv_t1d）安全；默认路径不调用（零回归）。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv_t1d_seg x")
        w_id, w_shape, _ = self._resolve_input(w, "conv_t1d_seg w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv_t1d_seg 需要 x/w 的形状信息")
        B, C_in, L = x_shape
        C_in2, C_out, K = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv_t1d_seg 通道不匹配: x={x_shape} w={w_shape}")
        stride = int(stride); padding = int(padding)
        output_padding = int(output_padding); dilation = int(dilation)
        if not isinstance(out, BatchTensor) or out._buf is None:
            raise ValueError("conv_t1d_seg 的 out 必须是本 runner 的有效 BatchTensor")
        o_id = out._buf
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv_t1d_seg b", buf_b)
        else:
            b_id = 0
        ps = [int(p) for p in (
            B, C_in, L, C_out, K, stride, padding, output_padding, dilation, o_id,
            seg_len, l_seg, in_off, lo_off, l_out_full,
            -1, -1, -1, -1)]  # view_offs：整 buffer 无偏移
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add_conv_t1d_seg(
                    self._ctx._handle, x_id, w_id, b_id, *ps
                ),
                "rvc_batch_add_conv_t1d_seg",
            )
            self._records.append((5, int(x_id), int(w_id), int(b_id), ps))
        return out

    def conv_t1d_seg_multi(
        self,
        x,
        w,
        b,
        stride: int,
        padding: int,
        out_shape,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
        output_padding: int = 0,
        dilation: int = 1,
    ) -> BatchTensor:
        """T1.1 分段辅助：超限 conv_t1d 按输出列自动分段，GPU 内写回整段父 buffer。

        ``out_shape`` = ``(B, C_out, oL)`` 整段形状；各段按 ``_GRID_POINTS_MAX``
        切输出列，输入窗口按 i_min/i_max 公式（见 T1.1实施草案），每段
        ``conv_t1d_seg`` 绝对寻址写父 buffer。x 保持 BatchTensor 流转（零 numpy
        往返）。返回整段 ``BatchTensor([B,C_out,oL])``。
        """
        from runtime import vulkan_ops as _vo  # noqa: PLC0415

        x_id, x_shape, _ = self._resolve_input(x, "conv_t1d_seg_multi x")
        w_id, w_shape, _ = self._resolve_input(w, "conv_t1d_seg_multi w", buf_w)
        B, C_in, L = x_shape
        C_in2, C_out, K = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv_t1d_seg_multi 通道不匹配: x={x_shape} w={w_shape}")
        stride = int(stride); padding = int(padding)
        output_padding = int(output_padding); dilation = int(dilation)
        oL = int(out_shape[2])
        o_id = self._alloc_output(B * C_out * oL)
        out_t = BatchTensor(self, o_id, (B, C_out, oL))
        s, p, op, d = stride, padding, output_padding, dilation
        # 输出列切段（上限留 10% 余量）
        limit = _GRID_POINTS_MAX * 9 // 10
        per_seg = max(1, limit // max(1, B * C_out))
        seg0 = 0
        while seg0 < oL:
            seg1 = min(oL, seg0 + per_seg)
            i_min = max(0, -(-(seg0 + p - (K - 1) * d - op) // s))
            i_max = min(L - 1, (seg1 - 1 + p - op) // s)
            i_min = max(0, min(i_min, L - 1))
            l_seg = i_max - i_min + 1
            self.conv_t1d_seg(
                x, w, b, out=out_t, stride=s, padding=p,
                output_padding=op, dilation=d,
                seg_len=seg1 - seg0, l_seg=l_seg, in_off=i_min, lo_off=seg0,
                l_out_full=oL, buf_w=buf_w, buf_b=buf_b,
            )
            seg0 = seg1
        return out_t

    def conv1d_seg(
        self,
        x,
        w,
        b,
        out: BatchTensor,
        stride: int,
        padding,
        dilation: int,
        seg_len: int,
        lo_off: int,
        l_out_full: int,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """T1.1 分段版 conv1d：x/out 整 buffer 绑定 + push 绝对寻址（GPU 内段流转）。

        ``seg_len`` = 本段输出列数；``lo_off`` = 输出段起始列（绝对）；
        ``l_out_full`` = 输出 buffer 行全长（行步长）。x 按绝对列 gather
        （load_x 用 ol+lo_off 映射输入 pos），无需输入窗口偏移。
        仅 gather 式 kernel（conv1d）安全；默认路径不调用（零回归）。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv1d_seg x")
        w_id, w_shape, _ = self._resolve_input(w, "conv1d_seg w", buf_w)
        if not x_shape or not w_shape:
            raise ValueError("conv1d_seg 需要 x/w 的形状信息")
        B, C_in, L = x_shape
        C_out, C_in2, K = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv1d_seg 通道不匹配: x={x_shape} w={w_shape}")
        if isinstance(padding, (tuple, list)):
            pad_l, pad_r = int(padding[0]), int(padding[1])
        else:
            pad_l = pad_r = int(padding)
        stride = int(stride); dilation = int(dilation)
        if not isinstance(out, BatchTensor) or out._buf is None:
            raise ValueError("conv1d_seg 的 out 必须是本 runner 的有效 BatchTensor")
        o_id = out._buf
        has_bias = b is not None
        if has_bias:
            b_id, b_shape, _ = self._resolve_input(b, "conv1d_seg b", buf_b)
        else:
            b_id = 0
        ps = [int(p) for p in (
            B, C_in, L, C_out, K, stride, pad_l, pad_r, dilation, o_id,
            seg_len, lo_off, l_out_full)]
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                _vulkan.dll.rvc_batch_add_conv1d_seg(
                    self._ctx._handle, x_id, w_id, b_id, *ps
                ),
                "rvc_batch_add_conv1d_seg",
            )
            self._records.append((2, int(x_id), int(w_id), int(b_id), ps))
        return out

    def conv1d_seg_multi(
        self,
        x,
        w,
        b,
        stride: int,
        padding,
        out_shape,
        dilation: int = 1,
        buf_w: PersistentBuffer | None = None,
        buf_b: PersistentBuffer | None = None,
    ) -> BatchTensor:
        """T1.1 分段辅助：超限 conv1d 按输出列自动分段，GPU 内写回整段父 buffer。

        ``out_shape`` = ``(B, C_out, oL)`` 整段形状；输出列切段，每段
        ``conv1d_seg`` 绝对寻址写父 buffer。x 保持 BatchTensor 流转（零 numpy
        往返）。返回整段 ``BatchTensor([B,C_out,oL])``。
        """
        x_id, x_shape, _ = self._resolve_input(x, "conv1d_seg_multi x")
        w_id, w_shape, _ = self._resolve_input(w, "conv1d_seg_multi w", buf_w)
        B, C_in, L = x_shape
        C_out, C_in2, K = w_shape
        if C_in != C_in2:
            raise ValueError(f"conv1d_seg_multi 通道不匹配: x={x_shape} w={w_shape}")
        if isinstance(padding, (tuple, list)):
            pad_l, pad_r = int(padding[0]), int(padding[1])
        else:
            pad_l = pad_r = int(padding)
        stride = int(stride); dilation = int(dilation)
        oL = int(out_shape[2])
        o_id = self._alloc_output(B * C_out * oL)
        out_t = BatchTensor(self, o_id, (B, C_out, oL))
        # 输出列切段（上限留 10% 余量）
        limit = _GRID_POINTS_MAX * 9 // 10
        per_seg = max(1, limit // max(1, B * C_out))
        seg0 = 0
        while seg0 < oL:
            seg1 = min(oL, seg0 + per_seg)
            self.conv1d_seg(
                x, w, b, out=out_t, stride=stride, padding=(pad_l, pad_r),
                dilation=dilation, seg_len=seg1 - seg0, lo_off=seg0,
                l_out_full=oL, buf_w=buf_w, buf_b=buf_b,
            )
            seg0 = seg1
        return out_t

    def add_inplace(self, a, b) -> BatchTensor:
        """录制就地加法 a = a + b（写回 a 的 buffer）。返回更新后的句柄。"""
        a_id, a_shape, _ = self._resolve_input(a, "add_inplace a")
        b_id, b_shape, _ = self._resolve_input(b, "add_inplace b")
        if a_shape and b_shape and tuple(a_shape) != tuple(b_shape):
            raise ValueError(f"add_inplace 形状不一致: a={a_shape} b={b_shape}")
        n = int(np.prod(a_shape)) if a_shape else 1
        # c 参数位传 a_id：BatchTensor 必须绑定被就地更新的 a。
        return self._record(3, a_id, b_id, a_id, (n,), a_shape)

    def mul_inplace(self, a, b) -> BatchTensor:
        """录制就地乘法 a = a * b（写回 a 的 buffer）。返回更新后的句柄。"""
        a_id, a_shape, _ = self._resolve_input(a, "mul_inplace a")
        b_id, b_shape, _ = self._resolve_input(b, "mul_inplace b")
        if a_shape and b_shape and tuple(a_shape) != tuple(b_shape):
            raise ValueError(f"mul_inplace 形状不一致: a={a_shape} b={b_shape}")
        n = int(np.prod(a_shape)) if a_shape else 1
        return self._record(4, a_id, b_id, a_id, (n,), a_shape)

    def leaky_relu(self, x, negative_slope: float = 0.1) -> BatchTensor:
        """录制就地 LeakyReLU（op=6）：x = where(x>=0, x, slope*x)。

        语义与 ``nn.leaky_relu`` / ``F.leaky_relu`` 一致（默认 slope=0.1）。
        就地写回 x 的 buffer；**调用方须注意**：若后续还需要 x 的原值
        （如残差加），先用 ``copy(x)`` 保留副本。返回更新后的句柄。
        """
        a_id, a_shape, _ = self._resolve_input(x, "leaky_relu x")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: leaky_relu %d 点超 GPU 上限" % n)
        return self._record(
            6, a_id, 0, a_id, (n, _f32_bits(negative_slope)), a_shape)

    def copy(self, x) -> BatchTensor:
        """录制复制（op=7）：dst = x（flat over N floats），返回新句柄。

        用于在就地算子覆写前保留输入（ResBlock 残差：先 copy 原 x，
        卷积链完成后 add_inplace(conv_out, x_copy)）。
        """
        a_id, a_shape, _ = self._resolve_input(x, "copy x")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: copy %d 点超 GPU 上限" % n)
        dst_id = self._alloc_output(n)
        return self._record(7, dst_id, a_id, 0, (n,), a_shape, tensor_buf=dst_id)

    # -- T1.1 elementwise 分段（copy/add/mul/lrelu 超限级 GPU 内分段） ----
    # 复用 recorder 的 view_offs 子视图机制（conv_t1d_seg/conv1d_seg 同一
    # 路径）：每段绑定 buffer 子区间（偏移×4 字节），shader 的 gid 相对段内
    # → 与整段算子同 kernel 同 push，输出逐位一致。engine 侧为纯新增
    # rvc_batch_add_*_seg（现有 op 不动，零回归）。

    def _seg_record(self, fn: str, c_ids, ps) -> None:
        """录制一笔 seg op（独立 ffi 入口，不经 op1-15 的 rvc_batch_add）。"""
        ps = [int(p) for p in ps]
        with self._lock:
            self._ensure_begin()
            _vulkan._check(
                getattr(_vulkan.dll, fn)(self._ctx._handle, *c_ids, *ps), fn
            )
            self._records.append((fn, *[int(i) for i in c_ids], ps))

    def copy_seg_multi(self, x) -> BatchTensor:
        """T1.1：GPU 内分段 copy —— 整段 dst = x（flat），返回新 BatchTensor。

        ``n ≤ _GRID_POINTS_MAX`` 退化为 ``copy``（零回归）；超限时按 flat 段
        ``copy_seg`` 写父 buffer（同 kernel 同 push → 逐位一致）。x 为
        numpy / BatchTensor；numpy 输入只上传一次（resolve 一次）。
        """
        a_id, a_shape, _ = self._resolve_input(x, "copy_seg_multi x")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n <= _GRID_POINTS_MAX:
            dst_id = self._alloc_output(n)
            return self._record(7, dst_id, a_id, 0, (n,), a_shape,
                                tensor_buf=dst_id)
        o_id = self._alloc_output(n)
        out_t = BatchTensor(self, o_id, a_shape)
        limit = _GRID_POINTS_MAX * 9 // 10
        off = 0
        while off < n:
            seg_n = min(limit, n - off)
            self._seg_record("rvc_batch_add_copy_seg",
                             (o_id, a_id), (seg_n, off, off))
            off += seg_n
        return out_t

    def add_inplace_seg_multi(self, a, b) -> BatchTensor:
        """T1.1：GPU 内分段就地加 —— a = a + b（flat，a/b 同形状）。

        ``n ≤ _GRID_POINTS_MAX`` 退化为 ``add_inplace``（零回归）。就地写回
        a 的 buffer（a 为 numpy 时上传为 GPU buffer 后就地，返回其句柄）。
        """
        a_id, a_shape, _ = self._resolve_input(a, "add_inplace_seg_multi a")
        b_id, b_shape, _ = self._resolve_input(b, "add_inplace_seg_multi b")
        if a_shape and b_shape and tuple(a_shape) != tuple(b_shape):
            raise ValueError(f"add_inplace_seg_multi 形状不一致: a={a_shape} b={b_shape}")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n <= _GRID_POINTS_MAX:
            return self._record(3, a_id, b_id, a_id, (n,), a_shape)
        limit = _GRID_POINTS_MAX * 9 // 10
        off = 0
        while off < n:
            seg_n = min(limit, n - off)
            self._seg_record("rvc_batch_add_add_seg",
                             (a_id, b_id), (seg_n, off, off))
            off += seg_n
        return BatchTensor(self, a_id, a_shape)

    def mul_inplace_seg_multi(self, a, b) -> BatchTensor:
        """T1.1：GPU 内分段就地乘 —— a = a * b（flat，a/b 同形状）。

        语义同 ``add_inplace_seg_multi``；超限分段、不超限退化 ``mul_inplace``。
        """
        a_id, a_shape, _ = self._resolve_input(a, "mul_inplace_seg_multi a")
        b_id, b_shape, _ = self._resolve_input(b, "mul_inplace_seg_multi b")
        if a_shape and b_shape and tuple(a_shape) != tuple(b_shape):
            raise ValueError(f"mul_inplace_seg_multi 形状不一致: a={a_shape} b={b_shape}")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n <= _GRID_POINTS_MAX:
            return self._record(4, a_id, b_id, a_id, (n,), a_shape)
        limit = _GRID_POINTS_MAX * 9 // 10
        off = 0
        while off < n:
            seg_n = min(limit, n - off)
            self._seg_record("rvc_batch_add_mul_seg",
                             (a_id, b_id), (seg_n, off, off))
            off += seg_n
        return BatchTensor(self, a_id, a_shape)

    def leaky_relu_seg_multi(self, x, negative_slope: float = 0.1) -> BatchTensor:
        """T1.1：GPU 内分段就地 LeakyReLU（op=6 语义，flat）。

        超限时按 flat 段 ``leaky_seg`` 就地写回 x 的 buffer（同 kernel 同
        push → 逐位一致）；不超限退化 ``leaky_relu``。
        """
        a_id, a_shape, _ = self._resolve_input(x, "leaky_relu_seg_multi x")
        n = int(np.prod(a_shape)) if a_shape else 1
        slope_bits = _f32_bits(negative_slope)
        if n <= _GRID_POINTS_MAX:
            return self._record(6, a_id, 0, a_id, (n, slope_bits), a_shape)
        limit = _GRID_POINTS_MAX * 9 // 10
        off = 0
        while off < n:
            seg_n = min(limit, n - off)
            self._seg_record("rvc_batch_add_leaky_seg",
                             (a_id,), (seg_n, off, slope_bits))
            off += seg_n
        return BatchTensor(self, a_id, a_shape)

    def gelu_seg_multi(self, x) -> BatchTensor:
        """T1.1：GPU 内分段就地 GELU（op=10 语义，exact erf，flat）。

        超限时按 flat 段 ``gelu_seg`` 就地写回 x 的 buffer（同 kernel 同
        push → 逐位一致）；不超限退化 ``gelu``。hubert conv 栈超限链用。
        """
        a_id, a_shape, _ = self._resolve_input(x, "gelu_seg_multi x")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n <= _GRID_POINTS_MAX:
            return self._record(10, a_id, 0, a_id, (n,), a_shape)
        limit = _GRID_POINTS_MAX * 9 // 10
        off = 0
        while off < n:
            seg_n = min(limit, n - off)
            self._seg_record("rvc_batch_add_gelu_seg",
                             (a_id,), (seg_n, off))
            off += seg_n
        return BatchTensor(self, a_id, a_shape)

    # -- attention 中间件批量算子（P1-5：整层 GPU 驻留） ----------------
    def softmax(self, x, axis_len=None) -> BatchTensor:
        """录制 softmax（op=8）：对最后一维逐行归一，返回同形状句柄。

        x 任意 ``[..., C]``：内部 reshape 为 ``[rows, cols]``（cols = 最后一
        维长度，可用 ``axis_len`` 显式覆盖），与 ``VulkanContext.softmax``
        同一 shader 语义（exp(x-max) 数值稳定）。
        """
        a_id, a_shape, _ = self._resolve_input(x, "softmax x")
        cols = axis_len if axis_len is not None else a_shape[-1]
        rows = int(np.prod(a_shape)) // int(cols)
        o_id = self._alloc_output(rows * cols)
        return self._record(8, a_id, 0, o_id, (rows, cols), a_shape)

    def layer_norm(self, x, gamma, beta, eps: float = 1e-5,
                   buf_gamma=None, buf_beta=None) -> BatchTensor:
        """录制 LayerNorm（op=9）：x 任意 ``[..., C]`` 对最后一维归一。

        gamma/beta: ``[C]``（常驻权重传 ``buf_gamma``/``buf_beta``）；与
        ``VulkanContext.layer_norm`` 同一 shader（biased 方差）。
        """
        a_id, a_shape, _ = self._resolve_input(x, "layer_norm x")
        g_id, g_shape, _ = self._resolve_input(gamma, "layer_norm gamma", buf_gamma)
        bt_id, bt_shape, _ = self._resolve_input(beta, "layer_norm beta", buf_beta)
        cols = g_shape[0] if g_shape else (a_shape[-1])
        if a_shape and bt_shape and bt_shape[0] != cols:
            raise ValueError(f"layer_norm beta 长度 {bt_shape[0]} != cols {cols}")
        if a_shape and a_shape[-1] != cols:
            raise ValueError(f"layer_norm 维度不匹配: x[-1]={a_shape[-1]} gamma={g_shape}")
        rows = int(np.prod(a_shape)) // int(cols)
        o_id = self._alloc_output(rows * cols)
        return self._record(
            9, a_id, g_id, bt_id,
            (rows, cols, _f32_bits(eps), 0, 0, 0, 0, 0, 0, o_id),
            a_shape, tensor_buf=o_id,
        )

    def group_norm(self, x, gamma, beta, num_groups: int, eps: float = 1e-5,
                   buf_gamma=None, buf_beta=None) -> BatchTensor:
        """录制 GroupNorm（op=15）：x ``[B, C, S...]``（B 须为 1）按组归一。

        与 ``nn.group_norm`` 同语义（每组对 ``C//G * 空间维`` 求 biased
        mean/var）；hubert conv 栈的 GroupNorm(512 组) 即 ``num_groups=C``。
        """
        a_id, a_shape, _ = self._resolve_input(x, "group_norm x")
        g_id, g_shape, _ = self._resolve_input(gamma, "group_norm gamma", buf_gamma)
        bt_id, bt_shape, _ = self._resolve_input(beta, "group_norm beta", buf_beta)
        if len(a_shape) < 2 or a_shape[0] != 1:
            raise ValueError(f"group_norm 仅支持 B=1，got shape={a_shape}")
        B, C = a_shape[0], a_shape[1]
        if C % num_groups != 0:
            raise ValueError(f"group_norm: C={C} 不能被 num_groups={num_groups} 整除")
        S = int(np.prod(a_shape[2:]))
        if g_shape and g_shape[0] != C:
            raise ValueError(f"group_norm gamma 长度 {g_shape[0]} != C {C}")
        o_id = self._alloc_output(int(np.prod(a_shape)))
        return self._record(
            15, a_id, g_id, bt_id,
            (num_groups, C // num_groups, S, _f32_bits(eps), 0, 0, 0, 0, 0, o_id),
            a_shape, tensor_buf=o_id,
        )

    def gelu(self, x) -> BatchTensor:
        """录制就地 GELU（op=10，erf 精确版，float32 A&S）：x = gelu(x)。

        语义与 ``runtime.models.hubert.gelu_erf`` / torch 默认 gelu 一致
        （0.5x(1+erf(x/√2))）。就地写回 x 的 buffer。
        """
        a_id, a_shape, _ = self._resolve_input(x, "gelu x")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n > _GRID_POINTS_MAX:
            raise RuntimeError("vulkan_batch_too_large: gelu %d 点超 GPU 上限" % n)
        return self._record(10, a_id, 0, a_id, (n,), a_shape)

    def relu(self, x) -> BatchTensor:
        """录制就地 ReLU（op=14）：x = max(x, 0)（写回 x 的 buffer）。"""
        a_id, a_shape, _ = self._resolve_input(x, "relu x")
        n = int(np.prod(a_shape)) if a_shape else 1
        if n > _GRID_POINTS_MAX:
            raise RuntimeError("vulkan_batch_too_large: relu %d 点超 GPU 上限" % n)
        return self._record(14, a_id, 0, a_id, (n,), a_shape)

    def bias_add(self, x, bias, out: BatchTensor | None = None,
                 buf_bias=None) -> BatchTensor:
        """录制行广播偏置加（op=11）：out[i,j] = x[i,j] + bias[j]。

        ``bias``: ``[cols]``（常驻传 ``buf_bias``）；x 任意 ``[rows, cols]``。
        线性层 bias 加法的 GPU 版（避免下载 matmul 结果到 numpy 再加）。
        """
        a_id, a_shape, _ = self._resolve_input(x, "bias_add x")
        b_id, b_shape, _ = self._resolve_input(bias, "bias_add bias", buf_bias)
        cols = b_shape[0] if b_shape else a_shape[-1]
        n = int(np.prod(a_shape)) if a_shape else cols
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: bias_add %d 点超 GPU 上限" % n)
        o_id = out._buf if out is not None else self._alloc_output(n)
        return self._record(11, a_id, b_id, o_id, (n, cols), a_shape)

    def gating(self, x_in, g, n: int | None = None, lg: int = 1, h: int | None = None,
               off: int = 0, out: BatchTensor | None = None,
               buf_g: PersistentBuffer | None = None) -> BatchTensor:
        """录制融合 WN 门控（op=18，D6）：c[i] = tanh(a[i]+g1) * sigmoid(a[i+n]+g2)。

        x_in: ``[1, 2H, L]``（conv1d 输出：前 H 通道 = tanh 分支、后 H = sigmoid
        分支）；g: ``[1, 3*2H, Lg]``（cond 段全量；``off`` = 本层段起始元素
        偏移，``lg`` = 1 广播或 = L）；输出 ``[1, H, L]``。``h`` 默认 = C//2。
        GPU tanh/sigmoid 与 libm ~1ulp 差（D5 先例，有据差异）。
        """
        x_id, x_shape, _ = self._resolve_input(x_in, "gating x_in")
        g_id, g_shape, _ = self._resolve_input(g, "gating g", buf_g)
        if not x_shape or len(x_shape) != 3:
            raise ValueError("gating x_in 需要 [1, 2H, L]")
        _, C, L = x_shape
        H = C // 2
        if h is None:
            h = H
        if n is None:
            n = H * L
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: gating %d 点超 GPU 上限" % n)
        o_id = out._buf if out is not None else self._alloc_output(n)
        return self._record(18, x_id, g_id, o_id, (n, lg, h, off), (1, H, L),
                            tensor_buf=o_id)

    def copy_off(self, src, n: int, src_off: int, out_shape=None) -> BatchTensor:
        """录制偏移复制（copy 子段，D6 flow 拆半）：新 buffer dst[0..n) =
        src[src_off..src_off+n)（元素偏移；与整段 copy 同 kernel 同 push）。
        ``out_shape`` 为 BatchTensor 声明的形状（默认 ``(n,)`` flat）。
        """
        a_id, a_shape, _ = self._resolve_input(src, "copy_off src")
        dst_id = self._alloc_output(n)
        self._seg_record("rvc_batch_add_copy_seg", (dst_id, a_id), (n, 0, src_off))
        return BatchTensor(self, dst_id, out_shape if out_shape is not None else (n,))

    def gating_backward(self, go, a, b, n=None, lg=1, h=None, off=0) -> BatchTensor:
        """录制 gating 反向（op=42，op18 的融合反向）。

        go: 输出梯度 [1,H,L]；a: gating 输入 [1,2H,L]（conv 输出）；b: cond
        段全量 [1,3*2H,Lg]（off/lg 同 op18）。输出 g_acts_in [1,2H,L]=dL/d(a)。
        语义与逐算子链（slice+tanh+sigmoid+mul+add bp）等价（GPU f32 ~1e-7）。
        go/a/cond 走 a/b/c，out 走 p4（BatchEntry 上限 4 buffer）。
        """
        go_id, go_shape, _ = self._resolve_input(go, "gating_bwd go")
        a_id, a_shape, _ = self._resolve_input(a, "gating_bwd a")
        b_id, b_shape, _ = self._resolve_input(b, "gating_bwd b")
        if not a_shape or len(a_shape) != 3:
            raise ValueError("gating_bwd a 需要 [1, 2H, L]")
        _, C, L = a_shape
        H = C // 2
        if h is None:
            h = H
        if n is None:
            n = H * L
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: gating_bwd %d 点超 GPU 上限" % n)
        o_id = self._alloc_output(2 * n)
        return self._record(
            42, go_id, a_id, b_id, (n, lg, h, off, o_id),
            (1, 2 * H, L), tensor_buf=o_id)

    def attn_qk(self, q, k, H: int, T: int, D: int, C: int) -> BatchTensor:
        """录制融合 attention scores（op=12）：out[H,T,T] = q_h . k_h^T。

        q/k: ``[T, C]`` **头交错**布局（head h 占据列 ``[h*D, (h+1)*D)``，
        C == H*D）。一次 dispatch 完成全部 H 个头（GPU 全程无 permute）。
        """
        q_id, q_shape, _ = self._resolve_input(q, "attn_qk q")
        k_id, k_shape, _ = self._resolve_input(k, "attn_qk k")
        if H * T * T > (1 << 40):
            raise RuntimeError("vulkan_batch_too_large: attn_qk 输出超限")
        o_id = self._alloc_output(H * T * T)
        return self._record(12, q_id, k_id, o_id, (H, T, D, C), (H, T, T))

    def attn_sv(self, w, v, H: int, T: int, D: int, C: int) -> BatchTensor:
        """录制融合 attention 加权和（op=13）：ctx[T,C] = attnW . v。

        attnW: ``[H, T, T]``（softmax 后的 scores）；v: ``[T, C]`` 头交错
        （head h 列 ``[h*D,(h+1)*D)``）。输出 ctx 为 ``[T, C]`` —— 头合并
        在 kernel 内完成，可直接喂 out_proj matmul。
        """
        w_id, w_shape, _ = self._resolve_input(w, "attn_sv w")
        v_id, v_shape, _ = self._resolve_input(v, "attn_sv v")
        o_id = self._alloc_output(T * C)
        return self._record(13, w_id, v_id, o_id, (H, T, D, C), (T, C))

    def banded_attn_qk(self, q, k, used, H: int, T: int, D: int, C: int) -> BatchTensor:
        """录制带相对位置的融合 attention scores（op=16，D5）。

        out[H,T,T] = q_h . (k_h + used[(s-t+T-1)])^T —— T5 相对位置注意力
        ``q@kᵀ + rel_to_abs(q@usedᵀ)`` 的融合形式（线性等价，无 [H,P,2P-1]
        中间往返）。q/k: ``[T, C]`` 头交错；used: ``[2T-1, D]``（每层相对
        位置嵌入，``_get_relative_embeddings`` 输出 [1,2P-1,kc] 的 [0]）。
        """
        q_id, q_shape, _ = self._resolve_input(q, "banded_attn_qk q")
        k_id, k_shape, _ = self._resolve_input(k, "banded_attn_qk k")
        u_id, u_shape, _ = self._resolve_input(used, "banded_attn_qk used")
        if H * T * T > (1 << 40):
            raise RuntimeError("vulkan_batch_too_large: banded_attn_qk 输出超限")
        o_id = self._alloc_output(H * T * T)
        # op16：a=q b=k c=used，out 句柄走 p9（同 group_norm 模式）
        return self._record(
            16, q_id, k_id, u_id,
            (H, T, D, C, 0, 0, 0, 0, 0, o_id),
            (H, T, T), tensor_buf=o_id)

    def banded_attn_sv(self, w, v, used, H: int, T: int, D: int, C: int) -> BatchTensor:
        """录制带相对位置的融合 attention 加权和（op=17，D5）。

        out[T,C] = attnW . (v + used[(s-t+T-1)]) —— ``attnW@v +
        abs_to_rel(attnW)@used_v`` 的融合（无 rel_w 往返）。attnW:
        ``[H,T,T]``；v: ``[T,C]`` 头交错；used: ``[2T-1, D]``。
        """
        w_id, w_shape, _ = self._resolve_input(w, "banded_attn_sv w")
        v_id, v_shape, _ = self._resolve_input(v, "banded_attn_sv v")
        u_id, u_shape, _ = self._resolve_input(used, "banded_attn_sv used")
        o_id = self._alloc_output(T * C)
        return self._record(
            17, w_id, v_id, u_id,
            (H, T, D, C, 0, 0, 0, 0, 0, o_id),
            (T, C), tensor_buf=o_id)

    def transpose(self, x, C: int, P: int, scale: float = 1.0,
                  out_shape=None) -> BatchTensor:
        """录制 tiled GPU 转置（op=19，T16）：out[P,C] = src[C,P] * scale。

        x 的 buffer 按 ``[C, P]`` 行主序解释（如 conv1d 输出的
        ``[1, C, P]`` 即 flat ``[C,P]``；banded_attn_sv 输出的 ``[P, C]``
        即 flat ``[P,C]``）。kernel 自逆：反向转置传 ``transpose(t, P, C)``
        （flat 首维=``C``、次维=``P``）。``scale`` 折叠进 kernel 写回
        （enc q 路径的 ``1/sqrt(kc)``，纯转置默认 1.0）。``out_shape``
        为 BatchTensor 声明的形状（默认 ``(P, C)``，如 [1,h,P] 需显式传）。
        """
        x_id, x_shape, _ = self._resolve_input(x, "transpose x")
        if C < 1 or P < 1:
            raise ValueError(f"transpose 维度非法: C={C} P={P}")
        n = int(C) * int(P)
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: transpose %d 点超 GPU 上限" % n)
        o_id = self._alloc_output(n)
        return self._record(
            19, x_id, 0, o_id,
            (C, P, _f32_bits(scale), 0, 0, 0, 0, 0, 0, 0),
            out_shape if out_shape is not None else (P, C),
            tensor_buf=o_id)

    def transpose_b(self, x, B: int, C: int, P: int) -> BatchTensor:
        """录制批量转置（op=20，T-H7）：src [B,C,P] 行主序 → out [C, B*P]。

        backward conv1d/conv2d 的 gw 需要 go_r [O, B*oL] = go [B,O,oL]
        转置（batch 轴并入内层）——2D transpose（op=19）只能交换两轴，
        故新增该 gather kernel：``dst[o, b*P+p] = src[b, o*P+p]``。
        返回 BatchTensor，形状声明为 ``(C, B*P)``。
        """
        x_id, x_shape, _ = self._resolve_input(x, "transpose_b x")
        if B < 1 or C < 1 or P < 1:
            raise ValueError(f"transpose_b 维度非法: B={B} C={C} P={P}")
        n = int(B) * int(C) * int(P)
        if n > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: transpose_b %d 点超 GPU 上限" % n)
        o_id = self._alloc_output(n)
        return self._record(
            20, x_id, 0, o_id,
            (B, C, P, 0, 0, 0, 0, 0, 0, 0),
            (C, B * P),
            tensor_buf=o_id)

    def reduce_rows(self, x, M: int, N: int) -> BatchTensor:
        """录制行归约求和（op=21，T-H7）：out[m] = Σ_n x[m,n]，x [M,N] 行主序。

        conv bias 梯度 gb = reduce_rows(go_r)（go_r [O, B*oL] → [O]），
        使 backward 链全程留 GPU（避免逐层下载 go）。返回 BatchTensor([M])。
        """
        x_id, x_shape, _ = self._resolve_input(x, "reduce_rows x")
        if M < 1 or N < 1:
            raise ValueError(f"reduce_rows 维度非法: M={M} N={N}")
        if M > _GRID_POINTS_MAX:
            raise RuntimeError(
                "vulkan_batch_too_large: reduce_rows %d 行超 GPU 上限" % M)
        o_id = self._alloc_output(M)
        return self._record(
            21, x_id, 0, o_id,
            (M, N, 0, 0, 0, 0, 0, 0, 0, 0),
            (M,),
            tensor_buf=o_id)

    def leaky_relu_backward(self, go, x, negative_slope: float = 0.1):
        """录制 LeakyReLU 反向（op=22，T-H7）：gx = go * (x>=0 ? 1 : slope)。

        链式 backward 的 numpy 断链点 GPU 化：x 为前向输入（numpy 上传，
        每层一次小上传），go 可 BatchTensor（链上）或 numpy。返回 BatchTensor。
        """
        go_id, go_shape, _ = self._resolve_input(go, "leaky_bwd go")
        x_id, x_shape, _ = self._resolve_input(x, "leaky_bwd x")
        if go_shape is None or len(go_shape) == 0:
            raise ValueError("leaky_bwd 需要 go 形状")
        n = int(np.prod(go_shape))
        if n < 1 or int(np.prod(x_shape)) != n:
            raise ValueError(f"leaky_bwd 尺寸不匹配: go={go_shape} x={x_shape}")
        if n > _GRID_POINTS_MAX:
            raise RuntimeError("vulkan_batch_too_large: leaky_bwd %d 超上限" % n)
        o_id = self._alloc_output(n)
        return self._record(
            22, go_id, x_id, o_id,
            (n, _f32_bits(negative_slope), 0, 0, 0, 0, 0, 0, 0, 0),
            go_shape,
            tensor_buf=o_id)

    def activation_forward(self, x, mode: int):
        """录制 sigmoid/tanh 前向（op=39）：out = f(x)。

        mode 0=sigmoid 1=tanh；返回 BatchTensor（与 x 同形）。
        """
        x_id, x_shape, _ = self._resolve_input(x, "activation_fwd x")
        if x_shape is None or len(x_shape) == 0:
            raise ValueError("activation_fwd 需要 x 形状")
        n = int(np.prod(x_shape))
        if n < 1:
            raise ValueError("activation_fwd 尺寸非法")
        if n > _GRID_POINTS_MAX:
            raise RuntimeError("vulkan_batch_too_large: activation_fwd %d 超上限" % n)
        o_id = self._alloc_output(n)
        return self._record(
            39, x_id, 0, o_id,
            (n, int(mode), 0, 0, 0, 0, 0, 0, 0, 0),
            x_shape,
            tensor_buf=o_id)

    def slice_forward(self, x, start, stop, axis=1):
        # op41 mode0: out = x[:, start:stop, :]，仅 3D [B,C,T] axis=1。
        x_id, x_shape, _ = self._resolve_input(x, "slice_fwd x")
        if x_shape is None or len(x_shape) != 3:
            raise ValueError("slice_fwd only 3D, got %s" % (x_shape,))
        B, C_in, T = int(x_shape[0]), int(x_shape[1]), int(x_shape[2])
        s = int(start)
        e2 = int(stop) if stop is not None else C_in
        C_out = e2 - s
        if s < 0 or C_out < 1 or e2 > C_in:
            raise ValueError("slice_fwd range bad: start=%d stop=%d C_in=%d" % (s, e2, C_in))
        o_id = self._alloc_output(B * C_out * T)
        return self._record(
            41, x_id, o_id, 0,
            (B, C_in, C_out, T, s, 0, 0, 0, 0, 0),
            (B, C_out, T),
            tensor_buf=o_id)

    def slice_backward(self, go, x_shape, start, stop):
        # op41 mode1: gx[b,c,t] = go[b, c-start, t]（其余 0）。x_shape=[B,C_in,T]。
        B, C_in, T = int(x_shape[0]), int(x_shape[1]), int(x_shape[2])
        s = int(start)
        e2 = int(stop) if stop is not None else C_in
        C_out = e2 - s
        go_id, go_shape, _ = self._resolve_input(go, "slice_bwd go")
        o_id = self._alloc_output(B * C_in * T)
        return self._record(
            41, go_id, o_id, 0,
            (B, C_in, C_out, T, s, 1, 0, 0, 0, 0),
            (B, C_in, T),
            tensor_buf=o_id)

    def mul_backward(self, go, a, b):
        """录制 mul 反向（op=40）：ga = go*b（mode0），gb = go*a（mode1）。

        返回 (ga, gb) 两个 BatchTensor。a/b 可 BatchTensor 或 numpy。
        """
        ga = None
        gb = None
        for mode, other in ((0, b), (1, a)):
            go_id, go_shape, _ = self._resolve_input(go, "mul_bwd go")
            o_id2, o_shape, _ = self._resolve_input(other, "mul_bwd other")
            if go_shape is None:
                raise ValueError("mul_bwd 需要 go 形状")
            n = int(np.prod(go_shape))
            if n < 1 or int(np.prod(o_shape)) != n:
                raise ValueError(f"mul_bwd 尺寸不匹配: go={go_shape} other={o_shape}")
            if n > _GRID_POINTS_MAX:
                raise RuntimeError("vulkan_batch_too_large: mul_bwd %d 超上限" % n)
            d_id = self._alloc_output(n)
            t = self._record(
                40, go_id, o_id2, d_id,
                (n, int(mode), 0, 0, 0, 0, 0, 0, 0, 0),
                go_shape, tensor_buf=d_id)
            if mode == 0:
                ga = t
            else:
                gb = t
        return ga, gb

    def activation_backward(self, go, out_v, mode: int):
        """录制 sigmoid/tanh 反向（op=38）：dst = go * f'(out)。

        mode 0=sigmoid 1=tanh；out_v 是前向激活值，go 可为 BatchTensor
        或 numpy。返回 BatchTensor（与 go 同形）。
        """
        go_id, go_shape, _ = self._resolve_input(go, "activation_bwd go")
        out_id, out_shape, _ = self._resolve_input(out_v, "activation_bwd out")
        if go_shape is None or len(go_shape) == 0:
            raise ValueError("activation_bwd 需要 go 形状")
        n = int(np.prod(go_shape))
        if n < 1 or int(np.prod(out_shape)) != n:
            raise ValueError(f"activation_bwd 尺寸不匹配: go={go_shape} out={out_shape}")
        if n > _GRID_POINTS_MAX:
            raise RuntimeError("vulkan_batch_too_large: activation_bwd %d 超上限" % n)
        o_id = self._alloc_output(n)
        return self._record(
            38, go_id, out_id, o_id,
            (n, int(mode), 0, 0, 0, 0, 0, 0, 0, 0),
            go_shape,
            tensor_buf=o_id)

    # -- 提交 / 生命周期 ------------------------------------------------
    def _flush_up_q(self) -> None:
        """J9：把录制期攒的批量上传队列一次提交（copy 必须在任何 batch
        提交之前完成，故 commit 前调用；幂等，空队列 no-op）。"""
        if not self._up_q:
            return
        q, self._up_q = self._up_q, []
        self._ctx._batch_upload(q)

    def commit(self, async_: bool = False) -> list[BatchTensor]:
        """一次提交全部录制的 dispatch（一次 submit + 一次 fence-wait）。

        ``async_=False``（默认）：同步等待，返回时结果即可用（旧语义）。
        ``async_=True``：**异步提交**，不等待 GPU —— 立即返回，多个
        commit 可在 GPU 上流水线执行；调用 ``wait()`` 或任一
        ``BatchTensor.numpy()`` 时再统一等待。适合若干独立批次
        连续提交后一次性取结果（核心减少每 commit 的同步等待开销）。

        返回输出句柄列表（用于逐一下载）。幂等：重复调用返回同一
        结果。耗时记于 ``self.elapsed``（异步时为 submit 耗时）。
        """
        # ---- P1 M4 埋点 B（a26ah §2）：commit() 外层 wall（b4_outer）----
        # 同步 commit 的 fence-wait 在 DLL 内部，_elapsed 只测 submit；
        # b4_outer - b3_submit 才是 wait。env RVC_TRAIN_M4_PROBE=1 门控。
        _m4b = os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1"
        if _m4b:
            from runtime import m4_probe as _mp  # noqa: PLC0415
            _t_out = time.perf_counter()
            _mp.b_pre(bool(self._committed), len(self._up_q), bool(async_))
        with self._lock:
            if self._committed:
                if _m4b:
                    _mp.b_outer((time.perf_counter() - _t_out) * 1000.0)
                return list(self._tensors)
            # J9：批量上传队列先于任何 dispatch 提交（复制必须在 GPU 使用前完成）
            self._flush_up_q()
            if _m4b:
                _mp.b_flush((time.perf_counter() - _t_out) * 1000.0)
            t0 = time.perf_counter()
            if async_:
                _vulkan._check(
                    _vulkan.dll.rvc_batch_commit_async(self._ctx._handle),
                    "rvc_batch_commit_async",
                )
                self._async_pending = True
            else:
                _vulkan._check(
                    _vulkan.dll.rvc_batch_commit(self._ctx._handle), "rvc_batch_commit"
                )
            self._elapsed = time.perf_counter() - t0
            if _m4b:
                _mp.b_submit(time.perf_counter() - t0, self._elapsed)
            self._committed = True
            _ret = list(self._tensors)
        if _m4b:
            # S4 外层差：_t_out → 函数返回，差值 = wait（同步）/ ≈0（异步）
            _mp.b_outer((time.perf_counter() - _t_out) * 1000.0)
        return _ret

    def wait(self) -> None:
        """等待本 runner 所有在途异步提交完成（同步 commit 时为 no-op）。

        幂等；``BatchTensor.numpy()`` 会自动调用，无需手动。
        """
        with self._lock:
            if not self._async_pending:
                return
            _vulkan._check(
                _vulkan.dll.rvc_batch_wait(self._ctx._handle), "rvc_batch_wait"
            )
            self._async_pending = False

    def discard(self) -> None:
        """丢弃未提交批次（``rvc_batch_discard``），不提交 GPU 工作。"""
        with self._lock:
            if self._released:
                return
            # J9：未提交的上传队列作废（copy 不再需要）
            self._up_q.clear()
            _vulkan._check(
                _vulkan.dll.rvc_batch_discard(self._ctx._handle), "rvc_batch_discard"
            )
            self._committed = False

    def tensor_done(self, t: BatchTensor) -> None:
        """提前释放 ``BatchTensor`` 的 GPU buffer（归还输出池）并作废句柄。

        perf(P1-6)：dec GPU 驻留路径的中间张量（组内 orig/c1/c2、noise
        输出等）在录制完其**最后一个消费算子**后不再需要——立即归还池，
        让后续 ``_alloc_output`` 复用，把"一次推理的输出 buffer 峰值需求"
        从整段（~96 个/尺寸）压到单级/单块（~十几个）。

        **调用方必须承诺**：该 tensor 之后不会被用作任何算子的输入，也
        不会调用 ``numpy()``。就地算子（add_inplace/leaky_relu 等）返回的
        tensor 与原输入共享 buffer：本方法对同一 buffer 幂等（每个 bid 只
        归还一次，重复调用直接跳过），但调用方应避免在共享双方的任一方
        已 done 后仍把另一方当输入用。
        """
        with self._lock:
            if self._released:
                return
            bid = t._buf
            if bid is None or bid in self._recycled:
                return
            self._recycled.add(bid)
            t._buf = None
            self._owned.discard(bid)
            # 归还本 runner 本地池（录制期复用）；release() 时一并归还全局池。
            n = int(np.prod(t.shape)) if t.shape else 1
            self._local_pool.setdefault(n, []).append(bid)

    def release(self) -> None:
        """释放本 runner 上传/分配的 GPU buffer，并作废所有 tensor（幂等）。

        commit 前调用等价于 discard + 释放；commit 后调用只释放 buffer。
        同时释放引擎批次独占锁（_BATCH_LIFECYCLE_LOCK），幂等：只有首次
        release 真正释放一次，后续调用为 no-op（包括 __del__ 兜底路径）。
        """
        try:
            with self._lock:
                if self._released:
                    return
                self._released = True
                if not self._committed:
                    # J9：未提交的上传队列作废（copy 不再需要）
                    self._up_q.clear()
                    try:
                        _vulkan._check(
                            _vulkan.dll.rvc_batch_discard(self._ctx._handle),
                            "rvc_batch_discard",
                        )
                    except RuntimeError:
                        pass
                elif self._async_pending:
                    # 有在途异步提交引用着这些 buffer：必须先 wait，
                    # 否则 rvc_mem_free 前后会被队列 drain（safe，但此处
                    # 显式等待避免释放仍在使用的缓冲）。
                    try:
                        # P1-002：检查返回码（原直接忽略，GPU 内部错误
                        # 在 release 路径上完全无感知）；release 是尽力
                        # 而为路径，失败打 stderr 不阻断资源释放。
                        _vulkan._check(
                            _vulkan.dll.rvc_batch_wait(self._ctx._handle),
                            "rvc_batch_wait(release)",
                        )
                    except RuntimeError as _we:
                        import sys as _sg  # noqa: PLC0415
                        print(f"[P1-002] release 期 rvc_batch_wait 失败: {_we}",
                              file=_sg.stderr, flush=True)
                    self._async_pending = False
                _freed: set[int] = set()
                for t in self._tensors:
                    if t._buf is not None:
                        bid, t._buf = t._buf, None
                        self._owned.discard(bid)
                        if bid in _freed or bid in self._recycled:
                            # 就地算子（add_inplace 等）的输出 tensor 与原输入
                            # tensor 绑定同一 buffer（_tensors 中会出现两次）；
                            # tensor_done 已提前归还池的 bid 同理。保证每个
                            # bid 只释放一次（含池归还，否则二次释放会让池
                            # 残留失效 id）。
                            continue
                        _freed.add(bid)
                        try:
                            self._ctx.free(bid)
                        except RuntimeError:
                            pass
                for bid in self._owned:
                    if bid in _freed or bid in self._recycled:
                        continue
                    _freed.add(bid)
                    try:
                        self._ctx.free(bid)
                    except RuntimeError:
                        pass
                self._owned.clear()
                # 阶段D（D2 输入池）：输入 buffer 归池（超限回退真释放）
                for bid in self._owned_inputs:
                    if bid in _freed or bid in self._recycled:
                        continue
                    _freed.add(bid)
                    n = self._ctx._in_pool_n.get(bid)
                    try:
                        if n is not None:
                            self._ctx._pooled_free(bid, n)
                        else:
                            self._ctx.free(bid)
                    except RuntimeError:
                        pass
                self._owned_inputs.clear()
                self._records.clear()
                # 本地池剩余（未被同 runner 复用）的输出 buffer 归还全局池；
                # 此刻 GPU 已完成本批次（同步 commit / 已 wait），可安全复用。
                # 这些 bid 在 tensor_done 时已从 _owned 移除且登记进 _recycled，
                # 上面的 _tensors/_owned 循环不会处理它们；此处直接归池即可。
                for bucket in self._local_pool.values():
                    for bid in bucket:
                        try:
                            self._ctx.free(bid)
                        except RuntimeError:
                            pass
                self._local_pool.clear()
        finally:
            if self._owns_batch_lock:
                self._owns_batch_lock = False
                _BATCH_LIFECYCLE_LOCK.release()

    @property
    def elapsed(self) -> float:
        """最近一次 commit 的墙钟耗时（秒）。"""
        return self._elapsed

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: BLE001  # 解释器退出阶段不抛
            pass


# --------------------------------------------------------------------------
# 共享 BatchRunner（阶段H：backward 逐算子 br 创建/释放/锁包络削减）
# --------------------------------------------------------------------------
# 训练 backward 每次 conv1d/conv2d 反向都新建一个局部 BatchRunner（1486 次/步
# RVC_TRAIN_BR_BWD=1 路径）——begin/锁/对象分配/release 包络固定成本高。
# RVC_TRAIN_BR_SHARED=1 时改用进程级共享 br（threading.local 单例）：
# commit 后经 _ensure_begin 懒重置自动开新批次（BatchRunner 本就可复用），
# 上传/输出走池化复用（尺寸有限集），由调用方在每步末尾显式
# release_shared_br() 归还缓冲（防跨步内存累积）。数值零影响（同内核同参数
# 逐位一致）。默认 0 保零回归。
_BR_SHARED = threading.local()


def get_shared_br(ctx: "VulkanContext") -> "BatchRunner":
    """返回当前线程的共享 BatchRunner（惰性创建；ctx 或已释放时重建）。"""
    br = getattr(_BR_SHARED, "br", None)
    if br is None or br._ctx is not ctx or br._released:
        br = _BR_SHARED.br = BatchRunner(ctx)
    return br


def release_shared_br() -> None:
    """显式释放共享 BatchRunner（训练每步末调用，防 buffer 跨步累积）。"""
    br = getattr(_BR_SHARED, "br", None)
    if br is not None and not br._released:
        br.release()
    _BR_SHARED.br = None


if os.environ.get("RVC_TRAIN_UPLOAD_STATS"):
    import atexit as _atexit  # noqa: PLC0415

    def _dump_ups_stats() -> None:
        _u = globals().get("_UPS")
        if not _u:
            return
        import collections as _c  # noqa: PLC0415
        tot = sum(_u.values())
        print("[UPS-STATS]", dict(_c.Counter(_u).most_common(15)),
              "total_MB=%.1f" % (tot / 1e6), file=sys.stderr)

    _atexit.register(_dump_ups_stats)


# --------------------------------------------------------------------------
# 进程级单例
# --------------------------------------------------------------------------
_context: VulkanContext | None = None
_context_error: str | None = None
_context_lock = threading.Lock()


def get_context() -> VulkanContext:
    """返回进程级单例 ``VulkanContext``（惰性创建，失败不缓存）。"""
    global _context, _context_error
    if _context is not None:
        return _context
    with _context_lock:
        if _context is not None:
            return _context
        if not _vulkan.has_dll:
            raise RuntimeError(f"rvc_core.dll 不可用: {_vulkan.dll_load_error}")
        try:
            _context = VulkanContext()
            _context_error = None
        except Exception as exc:  # noqa: BLE001
            _context_error = str(exc)
            raise
        return _context


# --------------------------------------------------------------------------
# 模块级便捷函数（内部使用单例上下文）
# --------------------------------------------------------------------------
@traced("matmul/gpu")
def matmul(
    a: np.ndarray,
    b: np.ndarray,
    buf_a: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 matmul（等价于 ``get_context().matmul(a, b)``）。"""
    return get_context().matmul(a, b, buf_a=buf_a, buf_b=buf_b)


def add(
    a: np.ndarray,
    b: np.ndarray,
    buf_a: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 add。"""
    return get_context().add(a, b, buf_a=buf_a, buf_b=buf_b)


@traced("mul/gpu")
def mul(
    a: np.ndarray,
    b: np.ndarray,
    buf_a: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 mul。"""
    return get_context().mul(a, b, buf_a=buf_a, buf_b=buf_b)


def relu_inplace(a: np.ndarray, buf_a: PersistentBuffer | None = None) -> np.ndarray:
    """模块级 relu_inplace。"""
    return get_context().relu_inplace(a, buf_a=buf_a)


def relu(a: np.ndarray, buf_a: PersistentBuffer | None = None) -> np.ndarray:
    """模块级 relu（``runtime.backend.relu`` 分派入口；等价 ``relu_inplace``）。

    返回 ``max(a, 0)`` 的 f32 **新数组**（引擎内就地写上传副本，不修改调用方
    数组），语义与 ``runtime.backend.relu`` / ``nn.relu`` 完全一致。backend
    分派以 ``_impl("relu")`` 同名查找，故在此补薄包装（不改 ``relu_inplace``
    既有调用点语义）。``buf_a`` 为常驻 buffer（可选）。
    """
    return get_context().relu_inplace(a, buf_a=buf_a)


@traced("conv1d/gpu")
def conv1d(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    stride: int = 1,
    padding=0,
    dilation: int = 1,
    buf_w: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 conv1d。"""
    return get_context().conv1d(
        x, w, b, stride, padding, dilation, buf_w=buf_w, buf_b=buf_b
    )


def conv_transpose1d(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    buf_w: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 conv_transpose1d。"""
    return get_context().conv_transpose1d(
        x, w, b, stride, padding, output_padding, dilation, buf_w=buf_w, buf_b=buf_b
    )


# ---------------------------------------------------------------------------
# 训练侧反向接线（VK-02 / R-TRAIN-008：P-TRAIN-008）
# 只新增反向算子，不改任何既有正向算子语义；numpy 参考为
# ``runtime.nn_backward.conv1d_backward``（f64 精确、cast 回输入 dtype）。
# 数值口径：引擎 f32 累加 vs numpy f64 参考，结果 cast 到输入浮点 dtype
# （训练为 f16）后逐位一致/≤1 f16 LSB（1LSB 有据，见 _diag/vk02_train_perf.md）。
# ---------------------------------------------------------------------------


def _im2col1d_host(x, B, C, T, oL, K_dil, stride, pad_l):
    """host im2col（conv1d backward gw 的参考/回退路径——dilation≠1 或
    BR_BWD=0；对称 pad，零填充；与 GPU im2col_1d gather 逐位一致）。"""
    x32 = np.asarray(x, np.float32)
    x_pad = np.zeros((B, C, T + 2 * pad_l), dtype=np.float32)
    x_pad[:, :, pad_l:pad_l + T] = x32
    win = np.lib.stride_tricks.sliding_window_view(x_pad, K_dil, axis=-1)
    win = win[:, :, ::stride, :]                           # [B,C,oL,K_dil]
    return np.ascontiguousarray(
        win.transpose(0, 2, 1, 3).reshape(B * oL, C * K_dil))


def _host_im2col_1d(x, B, C, T, oL, K_dil, stride, pad_l, dilation):
    """host im2col（与 runtime.nn_backward 视图链逐位一致；dilation 通用）。

    P1-007：引擎 im2col_1d 对 dilation≠1 的窗口布局与 host 视图链不一致，
    conv1d backward 的 gw 一律用本函数组装后直传 matmul（对称 padding）。
    """
    x = np.asarray(x, np.float32)
    x_pad = np.zeros((B, C, T + 2 * int(pad_l)), dtype=np.float32)
    x_pad[:, :, int(pad_l):int(pad_l) + T] = x
    win = np.lib.stride_tricks.sliding_window_view(x_pad, int(K_dil), axis=-1)
    win = win[:, :, int(stride)::int(stride), :] if False else \
        win[:, :, ::int(stride), :]                            # [B,C,oL,K_dil]
    return np.ascontiguousarray(
        win.transpose(0, 2, 1, 3).reshape(B * oL, C * K_dil))


def conv1d_backward_gpu(x, w, grad_out, stride=1, padding=0, dilation=1,
                        b=None, br=None, buf_w=None, wtag="d"):
    """GPU 版 ``conv1d_backward``：返回 ``(grad_x, grad_w, grad_b)``。

    数学表达（与 ``runtime.nn_backward.conv1d_backward`` 等价）：
        grad_x = conv_transpose1d(grad_out, w, stride, padding, 0, dilation)
        grad_w[o,c,k] = Σ_{b,t} x_pad[b,c,t*stride + k*dilation] * go[b,o,t]
                      = matmul(go^T 展平, im2col(x) 展平)
        grad_b = Σ grad_out（归约，用 numpy——引擎无归约算子）

    仅在引擎可用且形状适合时走 GPU；任何异常/不支持（如非对称 padding
    无法用引擎 conv_t1d 表达）回退 numpy（调用方负责）。

    阶段A（A4 v3）：``br`` 非 None 时为**录制模式**——两算子录进外部
    BatchRunner 不 commit，返回 ``(gx BatchTensor, gw BatchTensor, gb)``
    （dilation≠1 或引擎异常时回退本地路径/ numpy，返回 ndarray 三元组）；
    调用方 commit 后 ``.numpy()`` 下载（判别器 S conv1d_groups 组间合并）。
    """
    from runtime import nn_backward as _nb  # noqa: PLC0415

    # T-H7 链式：BatchTensor 输入须保留 GPU 引用（np.asarray 会触发 __array__
    # 下载断链）；numpy 输入照常转换。w 恒为 numpy（参数）。
    go_bt = isinstance(grad_out, BatchTensor)
    x_bt = isinstance(x, BatchTensor)
    w = np.asarray(w)
    go = grad_out if go_bt else np.asarray(grad_out)
    x = x if x_bt else np.asarray(x)
    if isinstance(padding, (tuple, list)):
        pad_l, pad_r = int(padding[0]), int(padding[1])
    else:
        pad_l = pad_r = int(padding)
    stride = int(stride)
    dilation = int(dilation)
    if pad_l != pad_r:
        # 引擎 conv_t1d 仅支持对称 padding（P-BE-001 同源限制）→ 回退 numpy
        return _nb.conv1d_backward(x, w, go, stride=stride, padding=padding,
                                   dilation=dilation, b=b)
    ctx = get_context()
    try:
        B, C, T = x.shape
        O, _, K = w.shape
        K_dil = (K - 1) * dilation + 1
        oL = go.shape[2]
        # gx 长度须与输入一致（PyTorch 语义）：用 output_padding 补齐
        # conv_t1d 输出到 T；缺失量非法时回退 numpy。
        len0 = (oL - 1) * stride - 2 * pad_l + dilation * (K - 1) + 1
        opad = int(T - len0)
        # 仅 conv_t1d 表达（stride>1 时 output_padding 补齐受限）受此约束；
        # cg_bwd（op24/25/26 三合一 kernel）mode0 逐输入元素直接算 gx，
        # stride/dilation 任意（dec/enc 下采样 conv1d bp 走 GPU 免 numpy 回退）。
        _use_cg = (br is not None
                   and os.environ.get("RVC_TRAIN_C1D_CG", "1") == "1")
        if not _use_cg and (opad < 0 or opad >= stride):
            return _nb.conv1d_backward(x, w, go, stride=stride,
                                       padding=padding, dilation=dilation,
                                       b=b)
        # gx：conv_transpose1d(grad_out, w)；gw：im2col 展平 + matmul（f32）
        # 阶段E（C3）：br 路径（外部 br 或局部 br）用 GPU im2col（gather——
        # 逐位同 host 视图链），省 host pad+sliding_window+reshape+copy 与
        # xw 上传；BR_BWD=0 保留 host 组装（零回归参考）。
        # T-H7 链式分支：go 为 BatchTensor（backward GPU 链）——全 GPU 录制：
        #   go_r = transpose_b(go) [O,B*oL]（batch 轴并入内层）
        #   gx   = conv_t1d(go)（留 GPU 传下一层）
        #   gw   = matmul(go_r, xw)（im2col gather，dilation 窗）
        #   gb   = reduce_rows(go_r)（bias 归约，免 23MB go 每层下载）
        # 全部录进外部 br、不 commit——由 tape.backward 末尾统一提交/下载。
        if go_bt:
            if br is None:
                raise RuntimeError(
                    "conv1d_backward 链式分支须外部 BatchRunner（br）")
            # J21/J24：conv1d_groups 三合一反向 kernel（op24/25/26：gx+gw+gb
            # 一个 batch 录制，dilation 通用）替代 transpose_b+convT+im2col+
            # matmul+reduce_rows 五次录制——dec/flow/enc 的 conv1d bp 大张量
            # 下省 kernel 与 ffi 时间；dilation≠1 时 gw 直接出 (O,C,K)（免
            # host 收缩）。数值单测 ≤1.3e-5（gx）。RVC_TRAIN_C1D_CG=0 关。
            w_f32 = np.asarray(w, np.float32)
            if os.environ.get("RVC_TRAIN_C1D_CG", "1") == "1":
                gx_t, gw_t, gb_t = conv1d_groups_backward_gpu(
                    x, w_f32, go, 1, stride=stride, padding=pad_l,
                    dilation=dilation, b=b, br=br, buf_w=buf_w, wtag=wtag)
                # cg_bwd（groups=1）返回形状已正确：gx (B,C,T) /
                # gw (O,C,K) / gb (O,)——均为 BatchTensor，无需 reshape。
                return (gx_t, gw_t, gb_t)
            go_r_t = br.transpose_b(go, B, O, oL)          # [O, B*oL]
            gx_t = br.conv_transpose1d(
                go, w_f32, None, stride=stride, padding=pad_l,
                output_padding=opad, dilation=dilation)
            # J10：dilation≠1 也走 GPU gather（间隔窗）
            # P1-007 修复：引擎 gather 对 dilation≠1 的窗口布局与 host 视图链
            # 不一致（实测 im2col maxdiff=4.28 → gw 全错 rel≈1.3）——dilation≠1
            # 改用 host 视图组装（与 nn_backward f64 参考逐位一致），numpy 直接
            # 传给 matmul（_resolve_input 上传），matmul 仍留 GPU。
            if dilation == 1:
                xw_t = br.im2col_1d(
                    x if x_bt else np.asarray(x, np.float32),
                    B, C, T, oL, K_dil, stride, pad_l, dilation)
            else:
                xw_t = _host_im2col_1d(
                    x if x_bt else np.asarray(x, np.float32),
                    B, C, T, oL, K_dil, stride, pad_l, dilation)
            gw_t = br.matmul(go_r_t, xw_t)                 # [O, C*K_dil]
            gw_t = gw_t.reshape(O, C, K_dil)   # 零拷贝形状视图
            if dilation != 1:
                # 引擎无稀疏收缩算子 → host 收缩回 (O,C,K)（gw 小张量断链
                # 可接受；gx 仍留 GPU 链）。非链式路径（L3561）同款逻辑。
                _gw_np = gw_t.numpy()
                gw2 = np.zeros((O, C, K), dtype=np.float32)
                gw2[:, :, :] = _gw_np[:, :, ::dilation]
                gw_t = gw2
            gb_t = br.reduce_rows(go_r_t, O, B * oL) if b is not None else None
            return (gx_t, gw_t, gb_t)
        go_f = np.asarray(go, np.float32)
        # go_r [O, B*oL] 列序必须 (b,t)（b 外层 t 快）以匹配 xw 行序——
        # [B,O,oL] 直接 reshape 在 B>1 时按展平序 [b][o][t] 错位（行 o 跨
        # b 边界）；必须 transpose(1,0,2) 后 reshape（R22 修复）。B==1 时
        # [1,O,oL] reshape 即 [o][t]（零 copy 视图，数学同 transpose——
        # 保留训练性能路径）。
        go_r = go_f.reshape(O, B * oL) if B == 1 else \
            go_f.transpose(1, 0, 2).reshape(O, B * oL)
        w_f32 = np.asarray(w, np.float32)
        if br is not None:
            dt = _fdtype_dt(x, w, go)
            # numpy go（断链点/首层 seed / flow-enc 段）接回 GPU 链：gx/gw 录
            # BatchTensor 留 GPU；gb 走 host 归约（go 已在 host，零上传成本）。
            # J24：一律走 cg_bwd（dilation 通用，三合一 kernel）替代 convT+
            # im2col+matmul——numpy go + dilation≠1 的 flow/enc dilated conv
            # bp 也省 2 次录制。
            gb = None
            if b is not None:
                gb = np.asarray(go, np.float32).sum(axis=(0, 2)).astype(dt)
            if os.environ.get("RVC_TRAIN_C1D_CG", "1") == "1":
                gx_t, gw_t, _ = conv1d_groups_backward_gpu(
                    np.asarray(x, np.float32), w_f32, go_f, 1,
                    stride=stride, padding=pad_l, dilation=dilation,
                    b=None, br=br, buf_w=buf_w, wtag=wtag)
                return (gx_t, gw_t, gb)
            if dilation == 1:
                gx_t = br.conv_transpose1d(
                    go_f, w_f32, None, stride=stride, padding=pad_l,
                    output_padding=opad, dilation=1)
                xw_t = br.im2col_1d(
                    np.asarray(x, np.float32), B, C, T, oL, K_dil,
                    stride, pad_l)
                gw_t = br.matmul(go_r, xw_t).reshape(O, C, K_dil)
                return (gx_t, gw_t, gb)
            # dilation≠1 且 cg 关：落到通用 BR_BWD 分支（含 dilation）
        if os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1":
            from runtime.vulkan_ops import BatchRunner as _BR  # noqa: PLC0415
            _use_shared = os.environ.get("RVC_TRAIN_BR_SHARED", "0") == "1"
            _br = get_shared_br(ctx) if _use_shared else _BR(ctx)
            _local_br = not _use_shared
            _gx_t = _gw_t = None
            try:
                _gx_t = _br.conv_transpose1d(
                    go_f, w_f32, None, stride=stride, padding=pad_l,
                    output_padding=opad, dilation=dilation)
                # J10：dilation≠1 也走 GPU gather（间隔窗）
                # P1-007 修复：引擎 gather dilation≠1 布局错 → host 组装直传
                # matmul（上传一次，matmul 仍 GPU）。
                if dilation == 1:
                    _xw_t = _br.im2col_1d(
                        np.asarray(x, np.float32), B, C, T, oL, K_dil,
                        stride, pad_l, dilation)
                else:
                    _xw_t = _host_im2col_1d(
                        np.asarray(x, np.float32), B, C, T, oL, K_dil,
                        stride, pad_l, dilation)
                _gw_t = _br.matmul(go_r, _xw_t)
                _br.commit()
                gx = np.asarray(_gx_t.numpy(), np.float32)
                gw = np.asarray(_gw_t.numpy(), np.float32)
            finally:
                if _local_br:
                    _br.release()
                else:
                    # 共享 br：下载后立即归还输出 buffer 到本地池（复用，
                    # 防步内 1486×3 个输出 buffer 无限累积压爆显存）
                    if _gx_t is not None:
                        _br.tensor_done(_gx_t)
                    if _gw_t is not None:
                        _br.tensor_done(_gw_t)
        else:
            x_pad = np.zeros((B, C, T + pad_l + pad_r), dtype=np.float32)
            x_pad[:, :, pad_l:pad_l + T] = np.asarray(x, np.float32)
            win = np.lib.stride_tricks.sliding_window_view(
                x_pad, K_dil, axis=-1)
            win = win[:, :, ::stride, :]                   # [B,C,oL,K_dil]
            xw = np.ascontiguousarray(
                win.transpose(0, 2, 1, 3).reshape(B * oL, C * K_dil))
            gx = ctx.conv_transpose1d(
                go_f, w_f32, None, stride, pad_l, opad, dilation)
            gw = np.asarray(ctx.matmul(go_r, xw), np.float32)
        gw = gw.reshape(O, C, K_dil)
        if dilation != 1:
            gw2 = np.zeros((O, C, K), dtype=np.float32)
            gw2[:, :, :] = gw[:, :, ::dilation]
            gw = gw2
    except Exception as e:  # noqa: BLE001  引擎失败回退 numpy（数值一致参考）
        if os.environ.get("RVC_DEBUG_BWD_FALLBACK", "0") == "1":
            import traceback as _tb  # noqa: PLC0415
            _tb.print_exc()
        return _nb.conv1d_backward(x, w, go, stride=stride, padding=padding,
                                   dilation=dilation, b=b)

    dt = _fdtype_dt(x, w, go)
    gb = None
    if b is not None:
        gb = np.asarray(go, np.float32).sum(axis=(0, 2)).astype(dt)
    return (_maybe_cast(gx, dt), _maybe_cast(gw, dt), _maybe_cast(gb, dt))


def conv1d_groups_backward_gpu(x, w, grad_out, groups, stride=1, padding=0,
                               dilation=1, b=None, br=None, buf_w=None,
                               wtag="d"):
    """GPU 分组 1D 卷积反向（DiscriminatorS conv1d_groups bp，op24/25/26）。

    x: ``[B,C_in,L]``（fwd 输入全量）；w: ``[C_out, C_in_g, K]``
    （C_in_g = C_in/groups，PyTorch groups 权重格式）；go: ``[B,C_out,oL]``
    （BatchTensor 或 numpy）。dilation 通用（J24：dec/enc dilated conv1d
    bp——gx/gw 的窗内 tap 偏移乘 dilation）、对称 padding（pad_l == pad_r）。
    返回 ``(gx, gw, gb)``：``br`` 非 None 录制模式（BatchTensor 三元组，
    不 commit，由调用方统一提交）；``br`` None 时内部局部 BatchRunner
    执行并下载，返回 numpy（f32）。
    """
    from runtime import nn_backward as _nb  # noqa: PLC0415

    go_bt = isinstance(grad_out, BatchTensor)
    x_bt = isinstance(x, BatchTensor)
    go = grad_out if go_bt else np.asarray(grad_out)
    x = x if x_bt else np.asarray(x)
    if os.environ.get("RVC_TRAIN_BWD_PROFILE"):  # TEMP-DBG
        import sys as _sy  # noqa: PLC0415
        try:
            _shp_x = x.shape
        except Exception:  # noqa: BLE001
            _shp_x = np.shape(x)
        try:
            _shp_go = go.shape
        except Exception:  # noqa: BLE001
            _shp_go = np.shape(go)
        _n = int(np.prod(_shp_x)) if hasattr(_shp_x, "__len__") else 0
        if _n > 100000:
            print(f"[C1D-DBG] x={tuple(_shp_x)} go_bt={go_bt} x_bt={x_bt} "
                  f"go={tuple(_shp_go)} br={br is not None} "
                  f"pad={padding} stride={stride} dil={dilation}",
                  file=_sy.stderr)
    w = np.asarray(w)
    if isinstance(padding, (tuple, list)):
        pad_l, pad_r = int(padding[0]), int(padding[1])
    else:
        pad_l = pad_r = int(padding)
    stride = int(stride)
    dilation = int(dilation)
    if pad_l != pad_r:
        return _cg_bwd_numpy_fallback(x, w, go, groups, stride, padding, b)
    B, C, T = x.shape
    O, Ci, K = w.shape
    if C % Ci != 0:
        raise ValueError(f"conv1d_groups_bwd 通道不可分: C={C} Ci={Ci}")
    if groups != C // Ci:
        groups = C // Ci  # groups 由权重通道比推出（同 conv1d_groups fwd）
    if O % groups != 0:
        raise ValueError(f"conv1d_groups_bwd O={O} 不可按 groups={groups} 分")
    oL = go.shape[2]
    expect = (T + pad_l + pad_r - (K - 1) * dilation - 1) // stride + 1
    if (B, O, expect) != go.shape[:3]:
        raise ValueError(
            f"conv1d_groups_bwd oL 不匹配: go={go.shape[:3]} expect=(B={B},"
            f" O={O}, oL={expect})")

    local_br = br is None
    if local_br:
        br = BatchRunner(get_context())
    _cg_t0 = None
    if os.environ.get("RVC_TRAIN_CG_DBG"):
        import time as _cgt  # noqa: PLC0415
        _cg_t0 = _cgt.perf_counter()
    try:
        x_id, _, _ = br._resolve_input(x, "cg_bwd x")
        # J11：权重常驻（每步 refresh 一次，免 bp 链重复上传）。T0.3：
        # buf_w 由调用方提供（wn 权重 = wpers_get_wn 展开 cache；普通 =
        # wpers_get），避免每步按展开数组 id 重建缓存。
        if buf_w is None:
            buf_w = wpers_get(w, br, wtag)
        w_id, _, _ = br._resolve_input(w, "cg_bwd w", buf_w)
        go_id, _, _ = br._resolve_input(go, "cg_bwd go")
        P = (B, C, Ci, T, O, K, stride, pad_l, pad_r)
        gx_id = br._alloc_output(B * C * T)
        gw_id = br._alloc_output(O * Ci * K)
        gb_id = br._alloc_output(O)
        # J24：p9=out、p10=dilation（ffi op24/25/26 新约定）
        gx_t = br._record(24, x_id, w_id, go_id, P + (gx_id, dilation),
                          (B, C, T), tensor_buf=gx_id)
        gw_t = br._record(25, x_id, w_id, go_id, P + (gw_id, dilation),
                          (O, Ci, K), tensor_buf=gw_id)
        gb_t = br._record(26, x_id, w_id, go_id, P + (gb_id, dilation),
                          (O,), tensor_buf=gb_id)
        if local_br:
            br.commit()
            gx = np.asarray(gx_t.numpy(), np.float32)
            gw = np.asarray(gw_t.numpy(), np.float32)
            gb = np.asarray(gb_t.numpy(), np.float32) if b is not None else None
            if os.environ.get("RVC_TRAIN_CG_DBG") and _cg_t0 is not None:
                import time as _cgt  # noqa: PLC0415
                print(f"[CG-DBG] x={x.shape} go={np.shape(go)} local "
                      f"t={(_cgt.perf_counter() - _cg_t0)*1000:.1f}ms",
                      file=sys.stderr)
            return (gx, gw, gb)
        if os.environ.get("RVC_TRAIN_CG_DBG") and _cg_t0 is not None:
            import time as _cgt  # noqa: PLC0415
            print(f"[CG-DBG] x={x.shape} go={np.shape(go)} chain "
                  f"t={(_cgt.perf_counter() - _cg_t0)*1000:.1f}ms",
                  file=sys.stderr)
        return (gx_t, gw_t, gb_t if b is not None else None)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("RVC_DEBUG_BWD_FALLBACK", "0") == "1":
            import traceback as _tb  # noqa: PLC0415
            _tb.print_exc()
        if local_br:
            return _cg_bwd_numpy_fallback(x, w, go, groups, stride, padding, b)
        raise
    finally:
        if local_br:
            br.release()


def _cg_bwd_numpy_fallback(x, w, go, groups, stride, padding, b):
    """conv1d_groups bp 的 numpy 回退（组循环 conv1d_backward，与
    conv1d_groups 原 bp 同构——无 GPU 依赖，任何异常安全降级）。"""
    from runtime import nn_backward as _nb  # noqa: PLC0415

    x = np.asarray(x)
    go = np.asarray(go)
    B, C, T = x.shape
    O, Ci, K = w.shape
    # P1-013：组切片边界校验——通道/输出不能整除 groups 时，C//groups
    # 切片会静默漏掉尾部通道（如 C=5,g=2 只取 0:4，通道4 丢失），
    # 必须显式报错，禁止静默越界/丢数据。
    if C % groups != 0:
        raise ValueError(
            "conv1d_groups_bwd 通道不可分: C=%d groups=%d（%d %% %d = %d）"
            % (C, groups, C, groups, C % groups))
    if O % groups != 0:
        raise ValueError(
            "conv1d_groups_bwd O 不可分: O=%d groups=%d（%d %% %d = %d）"
            % (O, groups, O, groups, O % groups))
    gx = np.zeros_like(x)
    gw = np.zeros_like(w)
    gb = np.zeros((O,), dtype=np.float32) if b is not None else None
    for g in range(groups):
        cs = g * (C // groups)
        os_ = g * (O // groups)
        bg = None if b is None else b[os_: os_ + (O // groups)]
        gx_g, gw_g, gb_g = _nb.conv1d_backward(
            x[:, cs: cs + (C // groups), :],
            w[os_: os_ + (O // groups), :, :],
            go[:, os_: os_ + (O // groups), :], stride=stride,
            padding=padding, b=bg)
        gx[:, cs: cs + (C // groups), :] += gx_g
        gw[os_: os_ + (O // groups), :, :] += gw_g
        if gb is not None:
            gb[os_: os_ + (O // groups)] += gb_g
    return (gx, gw, gb)


def _maybe_cast(a, dt):
    """dtype 相同则原样返回（ndarray.astype 同 dtype 也会强制拷贝——
    大张量每 bp 的转换是 profile 热点）；不同才转换。None 透传。"""
    if a is None:
        return None
    return a if np.asarray(a).dtype == dt else np.asarray(a).astype(dt)


def _fdtype_dt(*arrays):
    """与 nn_backward._fdtype 一致的浮点 dtype 判定。"""
    dts = [np.asarray(a).dtype for a in arrays]
    fd = [d for d in dts if np.issubdtype(d, np.floating)]
    return np.result_type(*fd) if fd else np.float32


def _conv_transpose2d_numpy(x, w, b=None, stride=1, padding=0,
                            output_padding=0):
    """numpy 参考：conv_transpose2d（对齐 ``torch.nn.functional.conv_transpose2d``）。

    x: ``[B, C_in, OH, OW]``，w: ``[C_in, C_out, KH, KW]``（PyTorch 布局），
    b: ``[C_out]`` 或 None。stride/padding/output_padding 支持 int 或
    ``(h, w)`` 元组；dilation 恒为 1。
    ``oH = (OH-1)*sh - 2*pad_h + KH + opad_h``（oW 同理）。
    内部 float64 累加（与 ``runtime.nn_backward`` 参考同口径），返回 float32。
    """
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(stride, (tuple, list)):
        sh, sw = int(stride[0]), int(stride[1])
    else:
        sh = sw = int(stride)
    if isinstance(padding, (tuple, list)):
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if isinstance(output_padding, (tuple, list)):
        opad_h, opad_w = int(output_padding[0]), int(output_padding[1])
    else:
        opad_h = opad_w = int(output_padding)
    B, C_in, OH, OW = x.shape
    C_in2, C_out, KH, KW = w.shape
    if C_in != C_in2:
        raise ValueError(
            f"_conv_transpose2d_numpy 通道不匹配: x={x.shape} w={w.shape}")
    oH = (OH - 1) * sh - 2 * ph + KH + opad_h
    oW = (OW - 1) * sw - 2 * pw + KW + opad_w
    if oH <= 0 or oW <= 0:
        return np.zeros((B, C_out, max(oH, 0), max(oW, 0)), dtype=np.float32)
    x64 = np.asarray(x, np.float64)
    w64 = np.asarray(w, np.float64)
    out = np.zeros((B, C_out, oH, oW), dtype=np.float64)
    # 逐核散射：输出 (i*sh - ph + kh, j*sw - pw + kw) 累加 x[i,j]*w[kh,kw]。
    # 同一 (kh,kw) 内输出坐标随 (i,j) 单调且互异 → 无重复索引 += 冲突；
    # 不同 (kh,kw) 的贡献由外层循环逐条累加。
    for kh in range(KH):
        oh_of_i = np.arange(OH) * sh + (kh - ph)
        mh = (oh_of_i >= 0) & (oh_of_i < oH)
        if not mh.any():
            continue
        i_idx = np.nonzero(mh)[0]
        ohs = oh_of_i[i_idx]
        for kw in range(KW):
            ow_of_j = np.arange(OW) * sw + (kw - pw)
            mw = (ow_of_j >= 0) & (ow_of_j < oW)
            if not mw.any():
                continue
            j_idx = np.nonzero(mw)[0]
            ows = ow_of_j[j_idx]
            contrib = np.einsum(
                "bcij,co->boij",
                x64[:, :, i_idx][:, :, :, j_idx],
                w64[:, :, kh, kw],
                optimize=True,
            )
            out[:, :, ohs[:, None], ows[None, :]] += contrib
    if b is not None:
        out += np.asarray(b, np.float64).reshape(1, -1, 1, 1)
    return np.asarray(out, np.float32)


# ---------------------------------------------------------------------------
# T2（R-TRAIN-008 / P-TRAIN-008 续）：conv2d 反向 GPU 化（判别器
# DiscriminatorP 的 convs/conv_post 训练路径）。numpy 参考为
# ``runtime.nn_backward.conv2d_backward``（f64 精确、cast 回输入 dtype）。
# 数值口径：引擎 f32 累加 vs numpy f64 参考，结果 cast 到输入浮点 dtype
# （训练为 f16）后 ≤1~2 f16 LSB（见 _diag/t2_conv2d_bwd_blueprint.md）。
# ---------------------------------------------------------------------------


def conv2d_backward_gpu(x, w, grad_out, stride=1, padding=0, dilation=1,
                        b=None, br=None):
    """GPU 版 ``conv2d_backward``：返回 ``(grad_x, grad_w, grad_b)``。

    数学表达（与 ``runtime.nn_backward.conv2d_backward`` 等价，dilation=1）：
        grad_x[b,c,i,j] = Σ_{o,kh,kw} go[b,o,oh,ow] * w[o,c,kh,kw]
                        = conv_transpose2d(go, w, stride, padding, opad)
          引擎 conv_t2d 的权重布局 [C_in,C_out,KH,KW] 取 C_in=O（go 通道）、
          C_out=C（x 通道）——即判别器卷积核 w[O,C,KH,KW] 原样，**无需**
          ``w.transpose(1,0,2,3)``（转置后 C_in=C≠O 恒触发 conv_transpose2d
          通道校验失败 → GPU 路径永远回退；engine/src/shaders/conv_t2d.comp
          与 engine/test_ffi.py §6d2 组2 均以 w[C_in=O, C_out=C] 原样跑通）。
        grad_w[o,c,kh,kw] = Σ_{b,oh,ow} x_pad[b,c,oh*sh+kh, ow*sw+kw]
                             * go[b,o,oh,ow]
                          = matmul(go^T 展平, im2col(x) 展平)
        grad_b = Σ grad_out（numpy 归约——引擎无归约算子）

    仅当引擎可用、参数合法（dilation=1、output_padding 补齐量
    0 ≤ opad < stride、conv_transpose2d 输出形状吻合）时走 GPU；任何
    异常/不支持回退 numpy（调用方透明）。
    """
    from runtime import nn_backward as _nb  # noqa: PLC0415

    # T-H7 链式：BatchTensor 输入保留 GPU 引用（np.asarray 会下载断链）。
    go_bt = isinstance(grad_out, BatchTensor)
    x_bt = isinstance(x, BatchTensor)
    w = np.asarray(w)
    go = grad_out if go_bt else np.asarray(grad_out)
    x = x if x_bt else np.asarray(x)
    if isinstance(stride, (tuple, list)):
        sh, sw = int(stride[0]), int(stride[1])
    else:
        sh = sw = int(stride)
    if isinstance(padding, (tuple, list)):
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if isinstance(dilation, (tuple, list)):
        dh, dw = int(dilation[0]), int(dilation[1])
    else:
        dh = dw = int(dilation)
    if sh < 1 or sw < 1 or ph < 0 or pw < 0 or dh != 1 or dw != 1:
        # 引擎 conv_t2d 无 dilation（判别器恒 1）；负 padding 无 PyTorch
        # 语义 → 均回退 numpy。
        return _nb.conv2d_backward(x, w, go, stride=stride, padding=padding,
                                   dilation=dilation, b=b)
    ctx = get_context()
    try:
        B, C, H, W = x.shape
        O, _, KH, KW = w.shape
        OH, OW = go.shape[2], go.shape[3]
        # gx 长度须与输入一致（PyTorch 语义）：用 output_padding 补齐
        # conv_t2d 输出到 H/W；补齐量非法（<0 或 ≥stride，引擎校验
        # opad < stride）时回退 numpy。
        len_h = (OH - 1) * sh - 2 * ph + KH
        len_w = (OW - 1) * sw - 2 * pw + KW
        opad_h = int(H - len_h)
        opad_w = int(W - len_w)
        if opad_h < 0 or opad_h >= sh or opad_w < 0 or opad_w >= sw:
            return _nb.conv2d_backward(x, w, go, stride=stride,
                                       padding=padding, dilation=dilation,
                                       b=b)
        # gx：conv_transpose2d(go, w)（w[C_in=O,C_out=C] 原样）；gw：im2col
        # 展平 + 引擎 matmul（f32）。展平次序必须两侧一致：go_r 列为
        # batch-major（先 b 再 oh/ow），故 xw 行也须 (b,oh,ow) 主序。
        # R16：A4 v2 曾弃用（batch conv_t2d 误判数值差）——重测证实
        # 判别器P参数下 batch convT2d 与单发逐位一致（md=0，当时 DIFF 系
        # 参数错位未修后重测）→ gx 并入 br 一次 commit（convT2d+im2col_2d
        # +matmul 三算子）；=0 走原单算子+host 组装（零回归参考）。
        # go_r [O, B*OH*OW] 列序必须 (b,oh,ow)（B>1 时 transpose——已正确）；
        # B==1 时 [1,O,OH,OW] 直接 reshape 即 [o][oh][ow]（零 copy 视图，
        # 数学同 transpose——R24 同 R23 优化）。
        # T-H7 链式分支（go BatchTensor，backward GPU 链）：全 GPU 录制——
        # go_r=transpose_b(go.reshape(B,O,OH*OW))，gx=conv_t2d(go) 留 GPU，
        # gw=matmul(go_r,im2col_2d(x))，gb=reduce_rows(go_r)。外部 br 统一
        # 由 tape.backward 尾提交。
        if go_bt:
            if br is None:
                raise RuntimeError(
                    "conv2d_backward 链式分支须外部 BatchRunner（br）")
            w_f32 = np.asarray(w, np.float32)
            # J11：判别器权重常驻 GPU（每步 refresh 一次，免 bp 链内
            # 192 次/步重复上传 conv_t2d w）
            _wpb = wpers_get(w, br, "d")
            go_r_t = br.transpose_b(go.reshape(B, O, OH * OW),
                                    B, O, OH * OW)          # [O, B*OH*OW]
            gx_t = br.conv_transpose2d(
                go, w_f32, None, stride=(sh, sw), padding=(ph, pw),
                output_padding=(opad_h, opad_w), buf_w=_wpb)
            xw_t = br.im2col_2d(
                x if x_bt else np.asarray(x, np.float32),
                B, C, H, W, OH, OW, KH, KW, sh, sw, ph, pw)
            gw_t = br.matmul(go_r_t, xw_t).reshape(O, C, KH, KW)
            gb_t = br.reduce_rows(go_r_t, O, B * OH * OW) if b is not None else None
            return (gx_t, gw_t, gb_t)
        _go32 = np.asarray(go, np.float32)
        if B == 1:
            go_r = _go32.reshape(O, B * OH * OW)
        else:
            go_r = np.ascontiguousarray(
                _go32.transpose(1, 0, 2, 3).reshape(O, B * OH * OW))
        if br is not None or os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1":
            # T-H7 链式：外部 br（go numpy 接回 GPU 链）——全录不 commit，
            # gx/gw 留 GPU，gb 走 host 归约（go 已在 host，零上传）。
            if br is not None:
                # J11：判别器权重常驻（免 bp 链重复上传）
                gx_t = br.conv_transpose2d(
                    np.asarray(go, np.float32), np.asarray(w, np.float32),
                    None, stride=(sh, sw), padding=(ph, pw),
                    output_padding=(opad_h, opad_w),
                    buf_w=wpers_get(w, br, "d"))
                _xw_t = br.im2col_2d(
                    x if x_bt else np.asarray(x, np.float32),
                    B, C, H, W, OH, OW, KH, KW, sh, sw, ph, pw)
                gw_t = br.matmul(go_r, _xw_t).reshape(O, C, KH, KW)
                gb_t = (br.reduce_rows(go_r, O, B * OH * OW)
                        if b is not None else None)
                return (gx_t, gw_t, gb_t)
            from runtime.vulkan_ops import BatchRunner as _BR6  # noqa: PLC0415
            _use_shared = os.environ.get("RVC_TRAIN_BR_SHARED", "0") == "1"
            _br = get_shared_br(ctx) if _use_shared else _BR6(ctx)
            _local_br = not _use_shared
            _gx_t = _gw_t = None
            try:
                # R16：A4 v2 曾弃用（batch conv_t2d 误判数值差）——重测证实
                # 判别器P convs 层（O>1）下 batch convT2d 与单发逐位一致
                # （md=0）；**conv_post（O=1，大 C 小输出）batch 路径有
                # 7.6e-6 差（容差内但非逐位）→ O==1 时 gx 回退单发**（保
                # 零回归）。gx(convT2d)+im2col_2d+matmul 一次 commit。
                if O > 1:
                    _gx_t = _br.conv_transpose2d(
                        np.asarray(go, np.float32), np.asarray(w, np.float32),
                        None, stride=(sh, sw), padding=(ph, pw),
                        output_padding=(opad_h, opad_w),
                        buf_w=wpers_get(w, _br, "d"))
                else:
                    gx = ctx.conv_transpose2d(
                        np.asarray(go, np.float32), np.asarray(w, np.float32),
                        None, (sh, sw), (ph, pw), (opad_h, opad_w),
                        buf_w=wpers_get(w, _br, "d"))
                _xw_t = _br.im2col_2d(
                    x if x_bt else np.asarray(x, np.float32), B, C, H, W,
                    OH, OW, KH, KW, sh, sw, ph, pw)
                _gw_t = _br.matmul(go_r, _xw_t)
                _br.commit()
                if O > 1:
                    gx = np.asarray(_gx_t.numpy(), np.float32)
                gw = np.asarray(_gw_t.numpy(), np.float32)
            finally:
                if _local_br:
                    _br.release()
                else:
                    # 共享 br：下载后立即归还输出 buffer（防步内累积压显存）
                    _br.tensor_done(_gw_t)
                    if O > 1:
                        _br.tensor_done(_gx_t)
            if gx.shape != (B, C, H, W):
                return _nb.conv2d_backward(x, w, go, stride=stride,
                                           padding=padding, dilation=dilation,
                                           b=b)
        else:
            gx = ctx.conv_transpose2d(
                np.asarray(go, np.float32), np.asarray(w, np.float32), None,
                (sh, sw), (ph, pw), (opad_h, opad_w))
            if gx.shape != (B, C, H, W):
                return _nb.conv2d_backward(x, w, go, stride=stride,
                                           padding=padding, dilation=dilation,
                                           b=b)
            x_pad = np.zeros((B, C, H + 2 * ph, W + 2 * pw),
                             dtype=np.float32)
            x_pad[:, :, ph:ph + H, pw:pw + W] = np.asarray(x, np.float32)
            win = np.lib.stride_tricks.sliding_window_view(
                x_pad, (KH, KW), axis=(-2, -1))
            win = win[:, :, ::sh, ::sw, :, :]            # [B,C,OH,OW,KH,KW]
            xw = np.ascontiguousarray(
                win.transpose(0, 2, 3, 1, 4, 5)
                .reshape(B * OH * OW, C * KH * KW))
            gw = np.asarray(ctx.matmul(go_r, xw), np.float32)
        gw = gw.reshape(O, C, KH, KW)
    except Exception:  # noqa: BLE001  引擎失败回退 numpy（数值一致参考）
        return _nb.conv2d_backward(x, w, go, stride=stride, padding=padding,
                                   dilation=dilation, b=b)

    dt = _fdtype_dt(x, w, go)
    gb = None
    if b is not None:
        gb = np.asarray(go, np.float32).sum(axis=(0, 2, 3)).astype(dt)
    return (_maybe_cast(gx, dt), _maybe_cast(gw, dt), _maybe_cast(gb, dt))


def _conv1d_split_gpu(ctx, x, w, b_arr, stride, padding, dilation, K, out_shape):
    """conv1d 时间维切分：逐块 GPU 计算后拼接（引擎 grid.x 上限防护）。

    与 ``_conv2d_split_gpu`` 同理：全局输出列 ow 覆盖输入列
    ``[ow*stride - pad_l, ow*stride - pad_l + K*dilation)``；每块输入切片
    + 手动 pad（左/右非对称）→ 引擎 padding=0（stride/dilation 不变）
    → 输出列从块起点对齐拼接。
    """
    B, C_in, L = x.shape
    C_out, _, _ = w.shape
    if isinstance(padding, (tuple, list)):
        pad_l, pad_r = int(padding[0]), int(padding[1])
    else:
        pad_l = pad_r = int(padding)
    oL = out_shape[2]
    limit = _GRID_POINTS_MAX * 9 // 10
    per_block = max(1, limit // max(1, B * C_out))
    chunk = min(oL, max(1, per_block))
    n_blocks = (oL + chunk - 1) // chunk
    cols = []
    _in_split_local.flag = True
    try:
        for k in range(n_blocks):
            ow0 = k * chunk
            ow1 = min(oL, ow0 + chunk)
            w_lo = ow0 * stride - pad_l
            # conv1d 输出列 ow 覆盖输入 [ow*s - pad_l, ow*s - pad_l + dil*(K-1)]；
            # 块最后输出列 ow1-1 的输入上界（exclusive）：
            w_hi = (ow1 - 1) * stride - pad_l + dilation * (K - 1) + 1
            src_lo = max(0, w_lo)
            src_hi = min(L, w_hi)
            pad_lo = max(0, -w_lo)
            pad_hi = max(0, w_hi - L)
            xk = x[:, :, src_lo:src_hi]
            if pad_lo or pad_hi:
                xk = np.pad(xk, ((0, 0), (0, 0), (pad_lo, pad_hi)))
            ok = ctx.conv1d(xk, w, b_arr, stride=stride, padding=0, dilation=dilation)
            cols.append(ok[:, :, :ow1 - ow0])
    finally:
        delattr(_in_split_local, "flag")
    return np.concatenate(cols, axis=2)


def _conv_t1d_split_gpu(ctx, x, w, b_arr, stride, padding, output_padding,
                        dilation, K, out_shape):
    """conv_transpose1d 时间维切分：逐块 GPU 计算后拼接（引擎 grid.x 上限防护）。

    转置卷积是"输入少输出多"：输出列 ow 由输入 il 贡献当
    ``ow = il*stride - padding + k*dilation + output_padding``（PyTorch
    conv_transpose1d 语义，单 padding；engine shader 同：
    ``num = lo + padding - kk*dil`` → ``lo = i*stride - padding + kk*dil``）。
    块输出列 ``[ow0, ow1)`` 需要的输入范围：
    ``il_lo = max(0, ceil((ow0 + p - (K-1)*d - op)/s))``，
    ``il_hi = min(L-1, floor((ow1-1 + p - op)/s))``。
    块输入切片 → 引擎 conv_transpose1d（参数不变）→ 输出列从
    ``ow_start = il_lo*stride`` 对齐全局：局部输出列 j 由局部输入 il、tap
    kk 满足 ``j = il*s - p + kk*d + op``，对应全局输入 ``i = il_lo + il``
    → 全局列 ``ow = (il_lo+il)*s - p + kk*d + op = j + il_lo*s``，故
    全局列 = 局部列 + ``il_lo*s``。取 ``[ow0-ow_start : ow1-ow_start]``
    拼接（块间输入有重叠、输出无重叠）。
    """
    B, C_in, L = x.shape
    _, C_out, _ = w.shape  # w 布局 [C_in, C_out, K]（PyTorch conv_transpose1d）；修复：误取 shape[0] 致 per_block 偏大、12s 整段不切分
    s = int(stride); p = int(padding); op = int(output_padding); d = int(dilation)
    oL = out_shape[2]
    limit = _GRID_POINTS_MAX * 9 // 10
    per_block = max(1, limit // max(1, B * C_out))
    chunk = min(oL, max(1, per_block))
    n_blocks = (oL + chunk - 1) // chunk
    cols = []
    _in_split_local.flag = True
    try:
        for k in range(n_blocks):
            ow0 = k * chunk
            ow1 = min(oL, ow0 + chunk)
            # 输入范围按 ow = i*s - p + kk*d + op 反解（注意是**单** padding；
            # 旧代码误用 +2p，漏掉块尾最后一列输入 tap，导致超限回退的
            # conv_t 输出错位，实测 max|Δ|~0.5）。
            il_lo = max(0, -(-(ow0 + p - (K - 1) * d - op) // s))  # ceil 整除
            il_hi = min(L - 1, (ow1 - 1 + p - op) // s)
            il_lo = max(0, min(il_lo, L - 1))
            xk = x[:, :, il_lo:il_hi + 1]
            ok = ctx.conv_transpose1d(xk, w, b_arr, stride=s, padding=p,
                                      output_padding=op, dilation=d)
            # 全局列 = 局部列 + il_lo*s（见 docstring 推导）
            ow_start = il_lo * s
            off = ow0 - ow_start
            cols.append(ok[:, :, off:off + (ow1 - ow0)])
    finally:
        delattr(_in_split_local, "flag")
    return np.concatenate(cols, axis=2)


def _conv2d_split_gpu(ctx, x, w, b_arr, stride, padding, out_shape):
    """conv2d 时间维(W)切分：逐块 GPU 计算后拼接（引擎 grid.x 上限防护）。

    长音频（如 150s）的 rmvpe conv2d 输出元素数超一次 dispatch 上限
    （``_GRID_POINTS_MAX``）；整体回退 CPU 极慢，故按时间维切块：
    每块输出元素 ≤ 上限 → 各块走 GPU（padding=0、pad 已手动并入块输入），
    再按列拼接，结果与整体 conv2d 完全一致。

    数学：全局输出列 ow 覆盖输入列 ``[ow*sw - pw, ow*sw - pw + KW)``。
    块 k（输出列 ``[ow0, ow1)``）取输入列 ``[w_lo, w_hi)``，其中
    ``w_lo = ow0*sw - pw``、``w_hi = (ow1-1)*sw + KW``；对输入切片左右
    手动补 ``max(0,-w_lo)`` / ``max(0, w_hi-W)`` 后交给引擎 padding=0 卷积，
    输出列从 ow0 对齐全局，取前 ``ow1-ow0`` 列拼接。
    """
    B, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    OH, OW = out_shape[2], out_shape[3]

    # 每块输出元素数上限（留 1/2 余量防边界抖动）
    limit = _GRID_POINTS_MAX * 9 // 10
    per_block = max(1, limit // max(1, B * C_out * OH))
    chunk = min(OW, max(1, per_block))
    n_blocks = (OW + chunk - 1) // chunk

    cols = []
    _in_split_local.flag = True
    try:
        for k in range(n_blocks):
            ow0 = k * chunk
            ow1 = min(OW, ow0 + chunk)
            w_lo = ow0 * sw - pw
            w_hi = (ow1 - 1) * sw + KW
            src_lo = max(0, w_lo)
            src_hi = min(W, w_hi)
            pad_l = max(0, -w_lo)
            pad_r = max(0, w_hi - W)
            xk = x[:, :, :, src_lo:src_hi]
            if pad_l or pad_r:
                xk = np.pad(xk, ((0, 0), (0, 0), (0, 0), (pad_l, pad_r)))
            ok = ctx.conv2d(xk, w, b_arr, stride=(sh, sw), padding=(ph, 0))
            cols.append(ok[:, :, :, :ow1 - ow0])
    finally:
        delattr(_in_split_local, "flag")
    return np.concatenate(cols, axis=3)


@traced("conv2d/gpu")
def conv2d(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    stride=1,
    padding=0,
    buf_w: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 conv2d（dilation=1）。"""
    return get_context().conv2d(x, w, b, stride, padding, buf_w=buf_w, buf_b=buf_b)


def embedding(
    ids: np.ndarray,
    table: np.ndarray,
    buf_table: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 embedding。"""
    return get_context().embedding(ids, table, buf_table=buf_table)


@traced("add_inplace/gpu")
def add_inplace(
    a: np.ndarray,
    b: np.ndarray,
    buf_a: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 add_inplace。"""
    return get_context().add_inplace(a, b, buf_a=buf_a, buf_b=buf_b)


def mul_inplace(
    a: np.ndarray,
    b: np.ndarray,
    buf_a: PersistentBuffer | None = None,
    buf_b: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 mul_inplace。"""
    return get_context().mul_inplace(a, b, buf_a=buf_a, buf_b=buf_b)


@traced("layer_norm/gpu")
def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
    buf_gamma: PersistentBuffer | None = None,
    buf_beta: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 layer_norm。"""
    return get_context().layer_norm(x, gamma, beta, eps, buf_gamma=buf_gamma, buf_beta=buf_beta)


def softmax(x: np.ndarray) -> np.ndarray:
    """模块级 softmax。"""
    return get_context().softmax(x)


@traced("rmsnorm/gpu")
def rmsnorm(
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-5,
    buf_gamma: PersistentBuffer | None = None,
) -> np.ndarray:
    """模块级 rmsnorm。"""
    return get_context().rmsnorm(x, gamma, eps, buf_gamma=buf_gamma)