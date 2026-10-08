# -*- coding: utf-8 -*-
"""rvc_core.dll（Vulkan 计算引擎，T31）的 ctypes 绑定。

C ABI 完整定义见 ``engine/src/ffi.zig``，调用用法参考 ``engine/test_ffi.py``：

    int64_t  rvc_engine_create(void);                  // 返回句柄或 -1
    int32_t  rvc_engine_destroy(int64_t h);
    int32_t  rvc_mem_upload(int64_t h, const float* src, int64_t n, int64_t* out_buf);
    int32_t  rvc_mem_download(int64_t h, int64_t buf, float* dst, int64_t n);
    int32_t  rvc_mem_free(int64_t h, int64_t buf);
    int32_t  rvc_matmul(int64_t h, int64_t a, int64_t b, int64_t c,
                        int64_t M, int64_t K, int64_t N);     // c = a @ b，行主序
    int32_t  rvc_add(int64_t h, int64_t a, int64_t b, int64_t c, int64_t n);
    int32_t  rvc_mul(int64_t h, int64_t a, int64_t b, int64_t c, int64_t n);
    int32_t  rvc_relu(int64_t h, int64_t a, int64_t n);       // in-place
    int32_t  rvc_conv1d(int64_t h, int64_t x, int64_t w, int64_t b, int64_t out,
                        int64_t B, int64_t C_in, int64_t L, int64_t C_out,
                        int64_t K, int64_t stride, int64_t pad_l, int64_t pad_r,
                        int64_t dil);                          // b=0 表示无 bias
    int32_t  rvc_conv_t1d(int64_t h, int64_t x, int64_t w, int64_t b, int64_t out,
                          int64_t B, int64_t C_in, int64_t L, int64_t C_out,
                          int64_t K, int64_t stride, int64_t padding,
                          int64_t output_padding, int64_t dil);   // 转置卷积；w=[C_in,C_out,K]
    int32_t  rvc_conv2d(int64_t h, int64_t x, int64_t w, int64_t b, int64_t out,
                        int64_t B, int64_t C_in, int64_t H, int64_t W, int64_t C_out,
                        int64_t KH, int64_t KW, int64_t pad_h, int64_t pad_w,
                        int64_t stride_h, int64_t stride_w);   // 对称 pad；dilation=1
    int32_t  rvc_conv_t2d(int64_t h, int64_t x, int64_t w, int64_t out,
                          int64_t B, int64_t c_in, int64_t oh, int64_t ow,
                          int64_t c_out, int64_t kh, int64_t kw, int64_t sh,
                          int64_t sw, int64_t ph, int64_t pw, int64_t opad_h,
                          int64_t opad_w, int64_t h_out, int64_t w_out);
                          // 2D 转置卷积（无 bias，b 恒为 0 位模式）；
                          // w=[c_in,c_out,kh,kw]，dilation 恒为 1
    int32_t  rvc_embed(int64_t h, int64_t ids, int64_t table, int64_t out,
                       int64_t N, int64_t TableRows, int64_t EmbDim); // ids=int32 原始字节
    int32_t  rvc_add_inplace(int64_t h, int64_t a, int64_t b, int64_t n); // a=a+b
    int32_t  rvc_mul_inplace(int64_t h, int64_t a, int64_t b, int64_t n); // a=a*b
    int32_t  rvc_leaky_relu(int64_t h, int64_t a, int64_t n, int64_t slope); // 就地；slope=f32 位模式
    int32_t  rvc_copy(int64_t h, int64_t dst, int64_t src, int64_t n);   // dst[i]=src[i]
    int32_t  rvc_layernorm(int64_t h, int64_t x, int64_t gamma, int64_t beta,
                           int64_t out, int64_t rows, int64_t cols, double eps);
    int32_t  rvc_softmax(int64_t h, int64_t x, int64_t out, int64_t rows, int64_t cols);
    int32_t  rvc_rmsnorm(int64_t h, int64_t x, int64_t gamma, int64_t out,
                         int64_t rows, int64_t cols, double eps);
    int32_t  rvc_batch_begin(int64_t h);          // 重置批次（丢弃未提交）
    int32_t  rvc_batch_add(int64_t h, int32_t op, // 录制一个算子（不提交）
                           int64_t a, int64_t b, int64_t c,
                           int64_t p0..p9);       // op: 1=matmul 2=conv1d
                                                  //     3=add_inplace 4=mul_inplace
                                                  //     5=conv_t1d（a=x b=w c=bias
                                                  //     p0..p8 维度 p9=out）
                                                  //     6=leaky_relu（a 就地，
                                                  //     p0=n p1=slope=f32 位模式）
                                                  //     7=copy（a=dst b=src p0=n）
    int32_t  rvc_batch_commit(int64_t h);         // 一次提交全部录制 dispatch + 等待
    int32_t  rvc_batch_commit_async(int64_t h);   // 异步提交（不等待，批量流水线）
    int32_t  rvc_batch_wait(int64_t h);           // 等待所有在途异步提交完成
    int32_t  rvc_batch_discard(int64_t h);        // 丢弃未提交批次（幂等）
    int32_t  rvc_device_name(char* buf, int64_t buf_size);
    int32_t  rvc_last_error(char* buf, int64_t buf_size);

DLL 查找顺序：
    1. 环境变量 ``RVC_CORE_DLL``（显式指定完整路径）；
    2. 包内：本文件上级目录（runtime/ 的父目录，即项目根）下的
       ``engine/zig-out/bin/rvc_core.dll``；
    3. 项目根（当前工作目录）下的 ``engine/zig-out/bin/rvc_core.dll``。

找不到时**模块仍可导入、不抛异常**：``has_dll=False`` 且 ``available=False``，
``dll_load_error`` 携带可读原因，调用方据此回退 numpy。

注意：``available`` 是模块级 bool（而非函数），以便 ``runtime/native_config.py`` 的
``bool(getattr(runtime._vulkan, "available", False))`` 探测逻辑正确生效（函数会被
``bool()`` 恒判为 True）。需要函数形式请用 ``is_available()``。
"""

from __future__ import annotations

import ctypes
import os

__all__ = [
    "has_dll",
    "available",
    "dll_path",
    "dll_load_error",
    "is_available",
    "engine_create",
    "engine_destroy",
    "device_name",
    "last_error",
    "get_dll",
]

F32 = ctypes.c_float
I32 = ctypes.c_int32
I64 = ctypes.c_int64
F32_PTR = ctypes.POINTER(F32)
CHAR_PTR = ctypes.POINTER(ctypes.c_char)


# --------------------------------------------------------------------------
# DLL 查找与加载（模块级单例，失败不炸）
# --------------------------------------------------------------------------
def _candidate_paths() -> list[str]:
    """按查找顺序返回候选 DLL 路径（含环境变量/包内/项目根）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)  # runtime/ 的父目录 = 项目根
    rel = os.path.join("engine", "zig-out", "bin", "rvc_core.dll")
    paths = []
    env = os.environ.get("RVC_CORE_DLL", "").strip()
    if env:
        paths.append(env)
    paths.append(os.path.join(project_root, rel))
    paths.append(os.path.join(os.getcwd(), rel))
    return paths


def _get_dll_path() -> str | None:
    """返回第一个存在的 DLL 路径，找不到返回 None。"""
    for p in _candidate_paths():
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _load_dll() -> tuple[ctypes.CDLL | None, str]:
    """加载 DLL 并完成全部函数签名声明。

    返回 ``(dll, error)``：成功时 dll 非 None、error 为空；失败时 dll 为 None、
    error 为可读原因（含 ``zig build`` 提示与 RVC_CORE_DLL 用法）。
    """
    path = _get_dll_path()
    if path is None:
        candidates = "\n  - ".join(_candidate_paths())
        return None, (
            "未找到 rvc_core.dll。查找了以下位置（均不存在）：\n"
            f"  - {candidates}\n"
            "请先在 engine/ 下构建：\n"
            "    zig build -Doptimize=ReleaseFast\n"
            "或设置环境变量 RVC_CORE_DLL 指向编译产物。"
        )
    try:
        dll = ctypes.CDLL(path)
    except OSError as exc:
        return None, f"加载 {path} 失败: {exc}"

    dll.rvc_engine_create.restype = I64
    dll.rvc_engine_create.argtypes = []
    dll.rvc_engine_destroy.restype = I32
    dll.rvc_engine_destroy.argtypes = [I64]
    dll.rvc_mem_fill_zero.restype = I32
    dll.rvc_mem_fill_zero.argtypes = [I64, I64, I64]
    dll.rvc_mem_alloc.restype = I32
    dll.rvc_mem_alloc.argtypes = [I64, I64, ctypes.POINTER(I64)]
    dll.rvc_mem_upload.restype = I32
    dll.rvc_mem_upload.argtypes = [I64, F32_PTR, I64, ctypes.POINTER(I64)]
    dll.rvc_mem_upload_to.restype = I32
    dll.rvc_mem_upload_to.argtypes = [I64, F32_PTR, I64, I64]  # h, src, n, buf
    dll.rvc_mem_upload_to_batch.restype = I32
    dll.rvc_mem_upload_to_batch.argtypes = [I64, I64, ctypes.POINTER(I64), ctypes.POINTER(F32_PTR), ctypes.POINTER(I64)]  # h, n, sizes, datas, ids
    dll.rvc_mem_download_batch.restype = I32
    dll.rvc_mem_download_batch.argtypes = [I64, ctypes.POINTER(I64), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(I64), I64]  # h, bufs, dsts, sizes, count（J18；ffi.zig L199）
    dll.rvc_mem_upload_bytes.restype = I32
    dll.rvc_mem_upload_bytes.argtypes = [I64, ctypes.c_void_p, I64, ctypes.POINTER(I64)]
    dll.rvc_mem_download.restype = I32
    dll.rvc_mem_download.argtypes = [I64, I64, F32_PTR, I64]
    dll.rvc_mem_free.restype = I32
    dll.rvc_mem_free.argtypes = [I64, I64]
    dll.rvc_matmul.restype = I32
    dll.rvc_matmul.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_matmul_f16.restype = I32
    dll.rvc_matmul_f16.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_add.restype = I32
    dll.rvc_add.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_mul.restype = I32
    dll.rvc_mul.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_relu.restype = I32
    dll.rvc_relu.argtypes = [I64, I64, I64]
    dll.rvc_conv1d.restype = I32
    dll.rvc_conv1d.argtypes = [I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_conv_t1d.restype = I32
    dll.rvc_conv_t1d.argtypes = [I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_conv2d.restype = I32
    dll.rvc_conv2d.argtypes = [I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_conv_t2d.restype = I32
    dll.rvc_conv_t2d.argtypes = [I64] * 19  # h, x, w, out, B, c_in, oh, ow, c_out, kh, kw, sh, sw, ph, pw, opad_h, opad_w, h_out, w_out
    dll.rvc_insert_zeros_2x.restype = I32
    dll.rvc_insert_zeros_2x.argtypes = [I64, I64, I64, I64, I64, I64]  # h, x, out, C, H, W
    dll.rvc_embed.restype = I32
    dll.rvc_embed.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_add_inplace.restype = I32
    dll.rvc_add_inplace.argtypes = [I64, I64, I64, I64]
    dll.rvc_mul_inplace.restype = I32
    dll.rvc_mul_inplace.argtypes = [I64, I64, I64, I64]
    dll.rvc_leaky_relu.restype = I32
    dll.rvc_leaky_relu.argtypes = [I64, I64, I64, I64]
    dll.rvc_copy.restype = I32
    dll.rvc_copy.argtypes = [I64, I64, I64, I64]
    dll.rvc_layernorm.restype = I32
    dll.rvc_layernorm.argtypes = [I64, I64, I64, I64, I64, I64, I64, ctypes.c_double]
    dll.rvc_softmax.restype = I32
    dll.rvc_softmax.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_rmsnorm.restype = I32
    dll.rvc_rmsnorm.argtypes = [I64, I64, I64, I64, I64, I64, ctypes.c_double]
    dll.rvc_batch_begin.restype = I32
    dll.rvc_batch_begin.argtypes = [I64]
    dll.rvc_batch_add.restype = I32
    dll.rvc_batch_add.argtypes = [
        I64, I32, I64, I64, I64,
        I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64,
    ]
    dll.rvc_batch_add_conv2d.restype = I32
    dll.rvc_batch_add_conv2d.argtypes = [
        I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_batch_add_conv_t2d.restype = I32
    dll.rvc_batch_add_conv_t2d.argtypes = [I64] * 20  # h,x,w,b,out,B,c_in,oh,ow,c_out,kh,kw,sh,sw,ph,pw,opad_h,opad_w,h_out,w_out
    dll.rvc_batch_add_im2col_1d.restype = I32
    dll.rvc_batch_add_im2col_1d.argtypes = [I64] * 11  # h,x,out,B,C,T,oL,K_dil,stride,pad_l,dilation
    dll.rvc_batch_add_im2col_2d.restype = I32
    dll.rvc_batch_add_im2col_2d.argtypes = [I64] * 15  # h,x,out,B,C,H,W,OH,OW,KH,KW,sh,sw,ph,pw
    dll.rvc_batch_add_gru.restype = I32
    dll.rvc_batch_add_gru.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_gru.restype = I32
    dll.rvc_gru.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_batch_add_gru_sync.restype = I32
    dll.rvc_batch_add_gru_sync.argtypes = [I64, I64, I64, I64]
    dll.rvc_batch_add_conv_t1d_seg.restype = I32
    # h, a, b, c, p0..p9 (14), l_out/l_in_seg/in_off/lo_off/l_out_full (5),
    # vx/vw/vb/vo (4) = 23
    dll.rvc_batch_add_conv_t1d_seg.argtypes = [
        I64, I64, I64, I64, I64, I64, I64, I64, I64, I64,
        I64, I64, I64, I64, I64, I64, I64, I64, I64, I64,
        I64, I64, I64,
    ]
    dll.rvc_batch_add_conv1d_seg.restype = I32
    # h, a, b, c, p0..p9 (14), seg_len/lo_off/l_out_full (3) = 17
    dll.rvc_batch_add_conv1d_seg.argtypes = [
        I64, I64, I64, I64, I64, I64, I64, I64, I64, I64,
        I64, I64, I64, I64, I64, I64, I64,
    ]
    dll.rvc_batch_add_copy_seg.restype = I32
    # T1.1 elementwise 分段：h, a, b, n, a_off, b_off（元素计数/偏移）
    dll.rvc_batch_add_copy_seg.argtypes = [I64, I64, I64, I64, I64, I64]
    dll.rvc_batch_add_add_seg.restype = I32
    dll.rvc_batch_add_add_seg.argtypes = [I64, I64, I64, I64, I64, I64]
    dll.rvc_batch_add_mul_seg.restype = I32
    dll.rvc_batch_add_mul_seg.argtypes = [I64, I64, I64, I64, I64, I64]
    dll.rvc_batch_add_leaky_seg.restype = I32
    # h, a, n, a_off, slope(f32 位模式)
    dll.rvc_batch_add_leaky_seg.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_batch_add_gelu_seg.restype = I32
    # h, a, n, a_off（T1.1 hubert conv 栈超限链）
    dll.rvc_batch_add_gelu_seg.argtypes = [I64, I64, I64, I64]
    dll.rvc_batch_commit.restype = I32
    dll.rvc_batch_commit.argtypes = [I64]
    dll.rvc_batch_commit_async.restype = I32
    dll.rvc_batch_commit_async.argtypes = [I64]
    dll.rvc_batch_wait.restype = I32
    dll.rvc_batch_wait.argtypes = [I64]
    dll.rvc_batch_discard.restype = I32
    dll.rvc_batch_discard.argtypes = [I64]
    dll.rvc_ts_stats.restype = I32
    dll.rvc_ts_stats.argtypes = [I64, ctypes.POINTER(I64), ctypes.POINTER(I64)]
    dll.rvc_ts_reset.restype = I32
    dll.rvc_ts_reset.argtypes = [I64]
    # T8: 只读显存统计（suballocator chunk/free/live-buffer/staging/direct）
    dll.rvc_mem_stats.restype = I32
    dll.rvc_mem_stats.argtypes = [I64] + [ctypes.POINTER(I64)] * 11
    # T8: Top-N 存活 buffer dump（打包三元组）
    dll.rvc_mem_top.restype = I64
    dll.rvc_mem_top.argtypes = [I64, I64, ctypes.POINTER(I64)]
    dll.rvc_device_name.restype = I32
    dll.rvc_device_name.argtypes = [CHAR_PTR, I64]
    dll.rvc_probe_extensions.restype = I32
    dll.rvc_probe_extensions.argtypes = [I64, CHAR_PTR, I64]
    dll.rvc_last_error.restype = I32
    dll.rvc_last_error.argtypes = [CHAR_PTR, I64]
    return dll, ""


dll_path: str | None = _get_dll_path()
"""已找到的 DLL 路径；找不到时为 None（此时可尝试 ``RVC_CORE_DLL``/``zig build``）。"""

dll: ctypes.CDLL | None
dll_load_error: str
dll, dll_load_error = _load_dll()

has_dll: bool = dll is not None
"""DLL 是否成功加载（注意：加载成功不代表 `rvc_engine_create` 一定能创建引擎）。"""

available: bool = has_dll
"""Vulkan 后端是否可用（``native_config`` 用 ``bool(getattr(..., "available", False))``
探测，故此处必须是 bool）。dll 加载成功即视为可用；引擎创建失败会在调用时抛错。"""


def is_available() -> bool:
    """函数形式的可用性探测（等价于读取模块级 ``available``）。"""
    return available


def get_dll() -> ctypes.CDLL:
    """返回已加载的 ctypes 模块对象；未加载时抛 ``RuntimeError``。"""
    if dll is None:
        hint = f"（{dll_load_error}）" if dll_load_error else ""
        raise RuntimeError(f"rvc_core.dll 不可用{hint}")
    return dll


# --------------------------------------------------------------------------
# 顶层辅助
# --------------------------------------------------------------------------
def last_error() -> str:
    """读取引擎最近一次错误的文本（``rvc_last_error``）。"""
    buf = ctypes.create_string_buffer(4096)
    dll.rvc_last_error(buf, len(buf))
    return buf.value.decode("utf-8", "replace") or "unknown error"


def _check(rc: int, what: str) -> int:
    """统一错误处理：rc == -1 时抛 ``RuntimeError``（携带 last_error 内容）。"""
    if rc == -1:
        raise RuntimeError(f"{what} 失败: {last_error()}")
    return rc


def engine_create() -> int:
    """创建 Vulkan 计算引擎，返回非负句柄；失败抛 ``RuntimeError``。"""
    lib = get_dll()
    h = lib.rvc_engine_create()
    if h < 0:
        raise RuntimeError(f"rvc_engine_create 失败: {last_error()}")
    return int(h)


def engine_destroy(h: int) -> int:
    """销毁引擎句柄，返回 rc（0 成功）。"""
    lib = get_dll()
    return int(lib.rvc_engine_destroy(int(h)))


def device_name() -> str:
    """查询实际 GPU 设备名。

    原生 ``rvc_device_name`` 需要已存在（最近创建的）引擎：若当前没有引擎，
    自动创建一个临时引擎查询后销毁。查询失败抛 ``RuntimeError``。
    """
    lib = get_dll()
    buf = ctypes.create_string_buffer(256)

    def _query() -> bool:
        rc = lib.rvc_device_name(buf, len(buf))
        return rc == 0

    if not _query():
        # no engine created：临时拉一个引擎查询后销毁
        h = engine_create()
        try:
            _check(int(lib.rvc_device_name(buf, len(buf))), "rvc_device_name")
        finally:
            engine_destroy(h)
    return buf.value.decode("utf-8", "replace") or "unknown device"