# ctypes self-test for rvc_core.dll (engine/zig-out/bin).
# Run from engine/:  python test_ffi.py   (or: zig build test)
#
# Covers: engine create/device_name, aligned matmul, misaligned
# (pad-path) matmul, add/mul/relu parity vs numpy, and the error path
# (use-after-free -> -1 + non-empty rvc_last_error). Prints PASS on
# full success, non-zero exit otherwise.

import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.environ.get(
    "RVC_CORE_DLL", os.path.join(HERE, "zig-out", "bin", "rvc_core.dll")
)

F32 = ctypes.c_float
I64 = ctypes.c_int64


def load_dll():
    if not os.path.exists(DLL_PATH):
        sys.exit(f"FATAL: {DLL_PATH} not found — run 'zig build -Doptimize=ReleaseFast' first")
    dll = ctypes.CDLL(DLL_PATH)
    dll.rvc_engine_create.restype = I64
    dll.rvc_engine_create.argtypes = []
    dll.rvc_engine_destroy.restype = ctypes.c_int32
    dll.rvc_engine_destroy.argtypes = [I64]
    dll.rvc_mem_upload.restype = ctypes.c_int32
    dll.rvc_mem_upload.argtypes = [I64, ctypes.POINTER(F32), I64, ctypes.POINTER(I64)]
    dll.rvc_mem_download.restype = ctypes.c_int32
    dll.rvc_mem_download.argtypes = [I64, I64, ctypes.POINTER(F32), I64]
    dll.rvc_mem_free.restype = ctypes.c_int32
    dll.rvc_mem_free.argtypes = [I64, I64]
    dll.rvc_matmul.restype = ctypes.c_int32
    dll.rvc_matmul.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_add.restype = ctypes.c_int32
    dll.rvc_add.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_mul.restype = ctypes.c_int32
    dll.rvc_mul.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_relu.restype = ctypes.c_int32
    dll.rvc_relu.argtypes = [I64, I64, I64]
    dll.rvc_conv1d.restype = ctypes.c_int32
    dll.rvc_conv1d.argtypes = [I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_conv_t1d.restype = ctypes.c_int32
    dll.rvc_conv_t1d.argtypes = [I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_conv2d.restype = ctypes.c_int32
    dll.rvc_conv2d.argtypes = [I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_conv_t2d.restype = ctypes.c_int32
    dll.rvc_conv_t2d.argtypes = [I64] * 19  # h,x,w,out,B,c_in,oh,ow,c_out,kh,kw,sh,sw,ph,pw,opad_h,opad_w,h_out,w_out
    dll.rvc_embed.restype = ctypes.c_int32
    dll.rvc_embed.argtypes = [I64, I64, I64, I64, I64, I64, I64]
    dll.rvc_add_inplace.restype = ctypes.c_int32
    dll.rvc_add_inplace.argtypes = [I64, I64, I64, I64]
    dll.rvc_mul_inplace.restype = ctypes.c_int32
    dll.rvc_mul_inplace.argtypes = [I64, I64, I64, I64]
    dll.rvc_leaky_relu.restype = ctypes.c_int32
    dll.rvc_leaky_relu.argtypes = [I64, I64, I64, I64]  # a, n, slope(f32 位模式)
    dll.rvc_copy.restype = ctypes.c_int32
    dll.rvc_copy.argtypes = [I64, I64, I64, I64]  # dst, src, n
    dll.rvc_layernorm.restype = ctypes.c_int32
    dll.rvc_layernorm.argtypes = [I64, I64, I64, I64, I64, I64, I64, ctypes.c_double]
    dll.rvc_softmax.restype = ctypes.c_int32
    dll.rvc_softmax.argtypes = [I64, I64, I64, I64, I64]
    dll.rvc_rmsnorm.restype = ctypes.c_int32
    dll.rvc_rmsnorm.argtypes = [I64, I64, I64, I64, I64, I64, ctypes.c_double]
    dll.rvc_batch_begin.restype = ctypes.c_int32
    dll.rvc_batch_begin.argtypes = [I64]
    dll.rvc_batch_add.restype = ctypes.c_int32
    dll.rvc_batch_add.argtypes = [
        I64, ctypes.c_int32, I64, I64, I64,
        I64, I64, I64, I64, I64, I64, I64, I64, I64, I64,
    ]
    dll.rvc_batch_commit.restype = ctypes.c_int32
    dll.rvc_batch_commit.argtypes = [I64]
    dll.rvc_batch_commit_async.restype = ctypes.c_int32
    dll.rvc_batch_commit_async.argtypes = [I64]
    dll.rvc_batch_wait.restype = ctypes.c_int32
    dll.rvc_batch_wait.argtypes = [I64]
    dll.rvc_batch_discard.restype = ctypes.c_int32
    dll.rvc_batch_discard.argtypes = [I64]
    dll.rvc_device_name.restype = ctypes.c_int32
    dll.rvc_device_name.argtypes = [ctypes.POINTER(ctypes.c_char), I64]
    dll.rvc_last_error.restype = ctypes.c_int32
    dll.rvc_last_error.argtypes = [ctypes.POINTER(ctypes.c_char), I64]
    return dll


def last_error(dll):
    buf = ctypes.create_string_buffer(4096)
    dll.rvc_last_error(buf, len(buf))
    return buf.value.decode("utf-8", "replace")


def to_ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(F32))


def rel_err(a, b):
    denom = max(np.linalg.norm(b), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def upload(dll, h, arr):
    out = I64(0)
    rc = dll.rvc_mem_upload(h, to_ptr(arr), arr.size, ctypes.byref(out))
    assert rc == 0, f"upload failed: {last_error(dll)}"
    return int(out.value)


def download(dll, h, buf_id, n):
    dst = np.zeros(n, dtype=np.float32)
    rc = dll.rvc_mem_download(h, buf_id, to_ptr(dst), n)
    assert rc == 0, f"download failed: {last_error(dll)}"
    return dst


def batch_add(dll, h, op, a, b, c, *ps):
    """rvc_batch_add 包装：ps 补齐到 10 个 p 参数（conv1d 的 out 在 p9）。"""
    ps = list(ps) + [0] * (10 - len(ps))
    rc = dll.rvc_batch_add(h, op, a, b, c, *ps)
    return rc


def conv1d_ref(x, w, b, stride, pad_l, pad_r, dil):
    """独立 numpy 参考（显式 pad + 五重循环，与 GPU shader 不同写法）。"""
    B, C, L = x.shape
    Co, _, K = w.shape
    K_dil = (K - 1) * dil + 1
    oL = (L + pad_l + pad_r - K_dil) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
    out = np.zeros((B, Co, oL), dtype=np.float64)
    for bb in range(B):
        for oo in range(Co):
            for tt in range(oL):
                acc = 0.0
                for cc in range(C):
                    for kk in range(K):
                        acc += xp[bb, cc, tt * stride + kk * dil] * w[oo, cc, kk]
                out[bb, oo, tt] = acc + (b[oo] if b is not None else 0.0)
    return out.astype(np.float32)


def layernorm_ref(x, gamma, beta, eps):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)  # biased, ddof=0
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32)


def softmax_ref(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def rmsnorm_ref(x, gamma, eps):
    ms = (x * x).mean(-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * gamma).astype(np.float32)


def gelu_ref(x):
    """erf 版 GELU（A&S 7.1.26, float64 参考），与 GPU float32 shader 对照。"""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x / np.sqrt(2.0))
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741)
             * t - 0.284496736) * t + 0.254829592) * t
    erfv = 1.0 - poly * np.exp(-ax * ax)
    erfv = np.where(x / np.sqrt(2.0) < 0, -erfv, erfv)
    return (0.5 * x * (1.0 + erfv)).astype(np.float32)


def attn_ref(q, k, v, D=64):
    """reference attention（头交错布局）：scores[H,T,T] -> softmax -> ctx[T,C]。"""
    q, k, v = (np.asarray(a) for a in (q, k, v))
    if q.ndim == 2:
        q, k, v = q[None], k[None], v[None]
    B, T, C = q.shape[0], q.shape[1], q.shape[2]
    H = C // D
    qh = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    kh = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    scores = np.einsum("bhtd,bhsd->bhts", qh, kh)
    m = scores.max(-1, keepdims=True)
    e = np.exp(scores - m)
    attn_w = e / e.sum(-1, keepdims=True)
    ctx = np.einsum("bhts,bhsd->bhtd", attn_w, vh)
    return attn_w[0].copy(), ctx.transpose(0, 2, 1, 3).reshape(T, C)


def conv_t1d_ref(x, w, b, stride, padding, output_padding, dil):
    """独立 numpy 参考（直接按公式逐输出点累加，与 GPU shader 不同写法）。"""
    B, Cin, L = x.shape
    Cout = w.shape[1]
    K = w.shape[2]
    oL = (L - 1) * stride - 2 * padding + dil * (K - 1) + output_padding + 1
    out = np.zeros((B, Cout, oL), dtype=np.float64)
    for bb in range(B):
        for co in range(Cout):
            for lo in range(oL):
                acc = 0.0
                for ci in range(Cin):
                    for kk in range(K):
                        num = lo + padding - kk * dil
                        if num >= 0 and num % stride == 0:
                            i = num // stride
                            if i < L:
                                acc += x[bb, ci, i] * w[ci, co, kk]
                out[bb, co, lo] = acc + (b[co] if b is not None else 0.0)
    return out.astype(np.float32)


def conv2d_ref(x, w, b, pad_h, pad_w, stride_h, stride_w):
    """独立 numpy 参考（显式 pad + 六重循环）。"""
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    out = np.zeros((B, Cout, OH, OW), dtype=np.float64)
    for bb in range(B):
        for co in range(Cout):
            for oh in range(OH):
                for ow in range(OW):
                    acc = 0.0
                    for ci in range(Cin):
                        for kh in range(KH):
                            for kw in range(KW):
                                acc += xp[bb, ci, oh * stride_h + kh, ow * stride_w + kw] * w[co, ci, kh, kw]
                    out[bb, co, oh, ow] = acc + (b[co] if b is not None else 0.0)
    return out.astype(np.float32)


def conv_t2d_ref(x, w, sh, sw, ph, pw, opad_h, opad_w):
    """独立 numpy 参考：转置 2D 卷积（逐输出点累加，与 GPU shader 不同写法）。

    x = go 形状 [B, C_in, OH, OW]，w 为 PyTorch 转置布局 [C_in, C_out, KH, KW]。
    H_out = (OH-1)*sh - 2*ph + KH + opad_h；W_out 同理。
    """
    B, Cin, OH, OW = x.shape
    Cout = w.shape[1]
    KH, KW = w.shape[2], w.shape[3]
    H_out = (OH - 1) * sh - 2 * ph + KH + opad_h
    W_out = (OW - 1) * sw - 2 * pw + KW + opad_w
    out = np.zeros((B, Cout, H_out, W_out), dtype=np.float64)
    for bb in range(B):
        for co in range(Cout):
            for ho in range(H_out):
                for wo in range(W_out):
                    acc = 0.0
                    for ci in range(Cin):
                        for kh in range(KH):
                            nh = ho + ph - kh
                            if nh >= 0 and nh % sh == 0:
                                ih = nh // sh
                                if ih < OH:
                                    for kw in range(KW):
                                        nw = wo + pw - kw
                                        if nw >= 0 and nw % sw == 0:
                                            iw = nw // sw
                                            if iw < OW:
                                                acc += x[bb, ci, ih, iw] * w[ci, co, kh, kw]
                    out[bb, co, ho, wo] = acc
    return out.astype(np.float32)


def main():
    dll = load_dll()
    timings = {}
    failures = []

    def check(cond, label):
        if cond:
            print(f"  [ok] {label}")
        else:
            failures.append(label)
            print(f"  [FAIL] {label}")

    # ── 1. create + device name ──────────────────────────────────────
    t0 = time.perf_counter()
    h = dll.rvc_engine_create()
    timings["create"] = time.perf_counter() - t0
    check(h > 0, "rvc_engine_create returns a positive handle")
    if h <= 0:
        sys.exit(f"FATAL: engine create failed: {last_error(dll)}")

    name_buf = ctypes.create_string_buffer(256)
    rc = dll.rvc_device_name(name_buf, len(name_buf))
    dev_name = name_buf.value.decode("utf-8", "replace")
    check(rc == 0 and len(dev_name) > 0, f"rvc_device_name non-empty ({dev_name!r})")

    # ── 2. aligned matmul M=32 K=48 N=64 ─────────────────────────────
    rng = np.random.default_rng(0)
    M, K, N = 32, 48, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C_ref = (A @ B).astype(np.float32)

    a_id = upload(dll, h, A.ravel())
    b_id = upload(dll, h, B.ravel())
    c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))

    t0 = time.perf_counter()
    rc = dll.rvc_matmul(h, a_id, b_id, c_id, M, K, N)
    timings["matmul_32x48x64"] = time.perf_counter() - t0
    check(rc == 0, f"matmul 32x48x64 rc=0 (err={last_error(dll) if rc else ''})")

    C = download(dll, h, c_id, M * N).reshape(M, N)
    e = rel_err(C, C_ref)
    check(e < 1e-4, f"matmul 32x48x64 rel_err={e:.2e} < 1e-4")

    # ── 3. misaligned matmul M=17 K=23 N=33 (pad path) ───────────────
    M, K, N = 17, 23, 33
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C_ref = (A @ B).astype(np.float32)

    a_id = upload(dll, h, A.ravel())
    b_id = upload(dll, h, B.ravel())
    c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))

    t0 = time.perf_counter()
    rc = dll.rvc_matmul(h, a_id, b_id, c_id, M, K, N)
    timings["matmul_17x23x33"] = time.perf_counter() - t0
    check(rc == 0, f"matmul 17x23x33 rc=0 (err={last_error(dll) if rc else ''})")

    C = download(dll, h, c_id, M * N).reshape(M, N)
    e = rel_err(C, C_ref)
    check(e < 1e-4, f"matmul 17x23x33 (pad path) rel_err={e:.2e} < 1e-4")

    # ── 3b. matmul tiling: 随机多组尺寸 vs numpy <1e-4 ─────────────
    # 覆盖三种 tile 路径（16/32/64）与 tile 边界、K 尾块、细长矩阵。
    rng2 = np.random.default_rng(7)
    tile_shapes = [
        (1, 64, 1),      # 极小
        (15, 20, 33),    # 非倍数
        (16, 16, 16),    # tile16 恰好
        (17, 16, 17),    # tile16 边界+1
        (31, 31, 31),    # tile32 边界-1
        (32, 32, 32),    # tile32 恰好
        (33, 33, 33),    # tile32 边界+1
        (63, 63, 63),    # tile64 边界-1
        (64, 64, 64),    # tile64 恰好
        (65, 65, 65),    # tile64 边界+1
        (128, 100, 128),
        (256, 17, 32),   # K 非16倍数（K=17 尾块）
        (64, 33, 64),    # K 非16倍数（K=33 尾块）
        (1, 768, 256),   # 细长
        (3, 7, 5),       # 极小非倍数
    ]
    for (M, K, N) in tile_shapes:
        A = rng2.standard_normal((M, K)).astype(np.float32)
        B = rng2.standard_normal((K, N)).astype(np.float32)
        a_id = upload(dll, h, A.ravel())
        b_id = upload(dll, h, B.ravel())
        c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))
        rc = dll.rvc_matmul(h, a_id, b_id, c_id, M, K, N)
        check(rc == 0, f"tiled matmul {M}x{K}x{N} rc=0")
        C = download(dll, h, c_id, M * N).reshape(M, N)
        e = rel_err(C, A @ B)
        check(e < 1e-4, f"tiled matmul {M}x{K}x{N} rel_err={e:.2e} < 1e-4")
        dll.rvc_mem_free(h, a_id)
        dll.rvc_mem_free(h, b_id)
        dll.rvc_mem_free(h, c_id)

    # ── 3c. matmul 性能对比（tiled shader vs 旧 16x16 基线）────────
    # 旧基线 = 替换前的固定 16x16 tile shader（本机中位数 ms），用于
    # 报告速度比；新值为当前自适应 tile（16/32/64 三路径）中位数。
    # 极小矩阵（<1024 元素）不受影响：仍走 numpy 阈值。
    old_ms = {
        (1188, 64, 1188): 0.169,
        (64, 768, 768): 0.205,
        (1, 256, 768): 0.106,
        (1, 768, 256): 0.121,
        (512, 512, 512): 0.243,
        (1024, 1024, 1024): 1.816,
    }
    print("\n  matmul perf (median of 7, tiled vs old 16x16):")
    print(f"    {'shape':<18} {'new ms':>8} {'old ms':>8} {'speedup':>8} {'new GFLOPS':>11}")
    for (M, K, N) in old_ms:
        A = rng2.standard_normal((M, K)).astype(np.float32)
        B = rng2.standard_normal((K, N)).astype(np.float32)
        a_id = upload(dll, h, A.ravel())
        b_id = upload(dll, h, B.ravel())
        c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))
        dll.rvc_matmul(h, a_id, b_id, c_id, M, K, N)  # warmup
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            dll.rvc_matmul(h, a_id, b_id, c_id, M, K, N)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[len(ts) // 2]
        flops = 2.0 * M * K * N
        sp = old_ms[(M, K, N)] / (med * 1e3)
        print(
            f"    {f'{M}x{K}x{N}':<18} {med*1e3:8.3f} {old_ms[(M,K,N)]:8.3f} "
            f"{sp:7.2f}x {flops/med/1e9:11.1f}"
        )
        dll.rvc_mem_free(h, a_id)
        dll.rvc_mem_free(h, b_id)
        dll.rvc_mem_free(h, c_id)

    # ── 4. add / mul / relu ──────────────────────────────────────────
    Nn = 12345  # deliberately non-multiple of 256
    a = rng.standard_normal(Nn).astype(np.float32)
    b = rng.standard_normal(Nn).astype(np.float32)
    a_id = upload(dll, h, a)
    b_id = upload(dll, h, b)
    c_id = upload(dll, h, np.zeros(Nn, dtype=np.float32))

    t0 = time.perf_counter()
    rc = dll.rvc_add(h, a_id, b_id, c_id, Nn)
    timings["add"] = time.perf_counter() - t0
    check(rc == 0, "rvc_add rc=0")
    e = rel_err(download(dll, h, c_id, Nn), (a + b).astype(np.float32))
    check(e < 1e-6, f"add rel_err={e:.2e} < 1e-6")

    t0 = time.perf_counter()
    rc = dll.rvc_mul(h, a_id, b_id, c_id, Nn)
    timings["mul"] = time.perf_counter() - t0
    check(rc == 0, "rvc_mul rc=0")
    e = rel_err(download(dll, h, c_id, Nn), (a * b).astype(np.float32))
    check(e < 1e-6, f"mul rel_err={e:.2e} < 1e-6")

    t0 = time.perf_counter()
    rc = dll.rvc_relu(h, a_id, Nn)
    timings["relu"] = time.perf_counter() - t0
    check(rc == 0, "rvc_relu rc=0 (in-place)")
    e = rel_err(download(dll, h, a_id, Nn), np.maximum(a, 0.0))
    check(e < 1e-6, f"relu rel_err={e:.2e} < 1e-6")

    # ── 5. error path: use-after-free ────────────────────────────────
    rc = dll.rvc_mem_free(h, a_id)
    check(rc == 0, "rvc_mem_free rc=0")
    rc = dll.rvc_matmul(h, a_id, b_id, c_id, 4, 4, 4)
    err = last_error(dll)
    check(rc == -1 and len(err) > 0, f"use-after-free -> rc=-1, last_error={err!r}")

    # ── 6. conv1d (T28) ──────────────────────────────────────────────
    # 6a. K=3 stride=1 pad=1 with bias
    B, C_in, L, C_out, K, stride, pad_l, pad_r, dil = 2, 4, 64, 8, 3, 1, 1, 1, 1
    oL = (L + pad_l + pad_r - dil * (K - 1) - 1) // stride + 1
    x = rng.standard_normal((B, C_in, L)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
    b = rng.standard_normal(C_out).astype(np.float32)
    ref = conv1d_ref(x, w, b, stride, pad_l, pad_r, dil)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    b_id = upload(dll, h, b)
    o_id = upload(dll, h, np.zeros(B * C_out * oL, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_conv1d(h, x_id, w_id, b_id, o_id, B, C_in, L, C_out, K, stride, pad_l, pad_r, dil)
    timings["conv1d_K3_s1_p1"] = time.perf_counter() - t0
    check(rc == 0, f"conv1d K=3 s=1 p=1 rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, B * C_out * oL).reshape(B, C_out, oL)
    e = rel_err(got, ref)
    check(e < 1e-4, f"conv1d K=3 s=1 p=1 (bias) rel_err={e:.2e} < 1e-4")

    # 6b. K=5 dil=2 pad=(2,2), no bias (b=0 -> engine zero-bias path)
    B, C_in, L, C_out, K, stride, pad_l, pad_r, dil = 2, 4, 32, 6, 5, 1, 2, 2, 2
    oL = (L + pad_l + pad_r - dil * (K - 1) - 1) // stride + 1
    x = rng.standard_normal((B, C_in, L)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
    ref = conv1d_ref(x, w, None, stride, pad_l, pad_r, dil)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    o_id = upload(dll, h, np.zeros(B * C_out * oL, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_conv1d(h, x_id, w_id, 0, o_id, B, C_in, L, C_out, K, stride, pad_l, pad_r, dil)
    timings["conv1d_K5_d2"] = time.perf_counter() - t0
    check(rc == 0, f"conv1d K=5 dil=2 pad=(2,2) rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, B * C_out * oL).reshape(B, C_out, oL)
    e = rel_err(got, ref)
    check(e < 1e-4, f"conv1d K=5 dil=2 pad=(2,2) (no bias, b=0) rel_err={e:.2e} < 1e-4")

    # 6b1. stride=2 + 非对称 pad (pad_l=2, pad_r=1) 组合
    B, C_in, L, C_out, K, stride, pad_l, pad_r, dil = 2, 6, 48, 5, 3, 2, 2, 1, 1
    oL = (L + pad_l + pad_r - dil * (K - 1) - 1) // stride + 1
    x = rng.standard_normal((B, C_in, L)).astype(np.float32)
    w = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
    ref = conv1d_ref(x, w, None, stride, pad_l, pad_r, dil)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    o_id = upload(dll, h, np.zeros(B * C_out * oL, dtype=np.float32))
    rc = dll.rvc_conv1d(h, x_id, w_id, 0, o_id, B, C_in, L, C_out, K, stride, pad_l, pad_r, dil)
    check(rc == 0, f"conv1d K=3 s=2 pad=(2,1) rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, B * C_out * oL).reshape(B, C_out, oL)
    e = rel_err(got, ref)
    check(e < 1e-4, f"conv1d K=3 s=2 pad=(2,1) (asymmetric, no bias) rel_err={e:.2e} < 1e-4")
    dll.rvc_mem_free(h, x_id)
    dll.rvc_mem_free(h, w_id)
    dll.rvc_mem_free(h, o_id)

    # ── 6b2. conv1d 性能对比（tiled shader vs 旧逐点扫描基线）────────
    # 旧基线 = 替换前的逐点扫描 shader（feab2fd，本机中位数 ms，同测量
    # 方式：单次 rvc_conv1d 墙钟，含提交+等待）；新值 = 自适应 TILE
    # (16/32/64) tiled shader（shared memory + register blocking + vec4，
    # 类 matmul.comp 模式）。形状覆盖 vits dec ResBlock 各级
    # （[256,882]x3 d1/d3、[128,12800]x7、[64,25600]x7、[32,51200]x11）、
    # 通用大形状、stride=2、以及小形状。
    old_conv1d_ms = {
        (1, 256, 882, 256, 3, 1, 1, 1, 1): 0.643,
        (1, 256, 882, 256, 3, 1, 3, 3, 3): 0.660,
        (1, 128, 12800, 128, 7, 1, 18, 18, 3): 4.227,
        (1, 64, 25600, 64, 7, 1, 18, 18, 5): 2.252,
        (1, 32, 51200, 32, 11, 1, 25, 25, 5): 1.746,
        (1, 512, 1000, 512, 5, 1, 2, 2, 1): 3.230,
        (1, 192, 200, 192, 3, 1, 1, 1, 1): 0.241,
        (1, 768, 200, 768, 1, 1, 0, 0, 1): 0.872,
        (1, 64, 2000, 64, 9, 2, 4, 4, 1): 0.328,
    }
    print("\n  conv1d perf (median of 7, tiled vs old pointwise):")
    print(f"    {'shape':<26} {'new ms':>8} {'old ms':>8} {'speedup':>8} {'new GFLOPS':>11}")
    for (Bb, Ci, L, Co, K, s, pl, pr, d), old_ms in old_conv1d_ms.items():
        oL = (L + pl + pr - d * (K - 1) - 1) // s + 1
        x = rng2.standard_normal((Bb, Ci, L)).astype(np.float32)
        w = rng2.standard_normal((Co, Ci, K)).astype(np.float32)
        b = rng2.standard_normal(Co).astype(np.float32)
        x_id = upload(dll, h, x.ravel())
        w_id = upload(dll, h, w.ravel())
        b_id = upload(dll, h, b)
        o_id = upload(dll, h, np.zeros(Bb * Co * oL, dtype=np.float32))
        args = (h, x_id, w_id, b_id, o_id, Bb, Ci, L, Co, K, s, pl, pr, d)
        dll.rvc_conv1d(*args)  # warmup
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            dll.rvc_conv1d(*args)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[len(ts) // 2]
        flops = 2.0 * Bb * Co * oL * Ci * K
        sp = old_ms / (med * 1e3)
        label = f"{Co}x{L} K{K} s{s} d{d}"
        print(
            f"    {label:<26} {med*1e3:8.3f} {old_ms:8.3f} "
            f"{sp:7.2f}x {flops/med/1e9:11.1f}"
        )
        dll.rvc_mem_free(h, x_id)
        dll.rvc_mem_free(h, w_id)
        dll.rvc_mem_free(h, b_id)
        dll.rvc_mem_free(h, o_id)

    # ── 6c. conv_transpose1d（T31，vits dec 瓶颈）───────────────────
    # 小维度：B=1,Cin=192,L=16,Cout=384,K=4,stride=2,pad=1,op=1（vits 第0级形状）
    B, Cin, L, Cout, K = 1, 192, 16, 384, 4
    stride, pad, op, dil = 2, 1, 1, 1
    oL = (L - 1) * stride - 2 * pad + dil * (K - 1) + op + 1
    x = rng.standard_normal((B, Cin, L)).astype(np.float32)
    w = rng.standard_normal((Cin, Cout, K)).astype(np.float32)  # PyTorch 布局 [Cin,Cout,K]
    b = rng.standard_normal(Cout).astype(np.float32)
    ref = conv_t1d_ref(x, w, b, stride, pad, op, dil)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    b_id = upload(dll, h, b)
    o_id = upload(dll, h, np.zeros(B * Cout * oL, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_conv_t1d(h, x_id, w_id, b_id, o_id, B, Cin, L, Cout, K, stride, pad, op, dil)
    timings["conv_t1d_K4_s2_p1"] = time.perf_counter() - t0
    check(rc == 0, f"conv_t1d K=4 s=2 p=1 op=1 rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, B * Cout * oL).reshape(B, Cout, oL)
    e = rel_err(got, ref)
    check(e < 1e-4, f"conv_t1d K=4 s=2 p=1 op=1 (bias) rel_err={e:.2e} < 1e-4")

    # 大核：K=16 d=2 stride=4（覆盖 dil>1 与稀疏 tap 路径），无 bias
    B, Cin, L, Cout, K = 1, 8, 12, 8, 16
    stride, pad, op, dil = 4, 3, 0, 2
    oL = (L - 1) * stride - 2 * pad + dil * (K - 1) + op + 1
    x = rng.standard_normal((B, Cin, L)).astype(np.float32)
    w = rng.standard_normal((Cin, Cout, K)).astype(np.float32)
    ref = conv_t1d_ref(x, w, None, stride, pad, op, dil)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    o_id = upload(dll, h, np.zeros(B * Cout * oL, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_conv_t1d(h, x_id, w_id, 0, o_id, B, Cin, L, Cout, K, stride, pad, op, dil)
    timings["conv_t1d_K16_d2"] = time.perf_counter() - t0
    check(rc == 0, f"conv_t1d K=16 dil=2 s=4 rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, B * Cout * oL).reshape(B, Cout, oL)
    e = rel_err(got, ref)
    check(e < 1e-4, f"conv_t1d K=16 dil=2 s=4 (no bias) rel_err={e:.2e} < 1e-4")

    # ── 6d. conv2d（对称 pad、双轴 stride）──────────────────────────
    B, Cin, H, W, Cout = 2, 3, 8, 9, 4
    KH, KW = 3, 3
    pad_h, pad_w, sh, sw = 1, 1, 2, 1
    OH, OW = (H + 2 * pad_h - KH) // sh + 1, (W + 2 * pad_w - KW) // sw + 1
    x = rng.standard_normal((B, Cin, H, W)).astype(np.float32)
    w = rng.standard_normal((Cout, Cin, KH, KW)).astype(np.float32)
    b = rng.standard_normal(Cout).astype(np.float32)
    ref = conv2d_ref(x, w, b, pad_h, pad_w, sh, sw)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    b_id = upload(dll, h, b)
    o_id = upload(dll, h, np.zeros(B * Cout * OH * OW, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_conv2d(h, x_id, w_id, b_id, o_id, B, Cin, H, W, Cout, KH, KW, pad_h, pad_w, sh, sw)
    timings["conv2d_3x3_s2x1"] = time.perf_counter() - t0
    check(rc == 0, f"conv2d KH=3 KW=3 s=(2,1) rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, B * Cout * OH * OW).reshape(B, Cout, OH, OW)
    e = rel_err(got, ref)
    check(e < 1e-4, f"conv2d 3x3 s=(2,1) p=1 (bias) rel_err={e:.2e} < 1e-4")

    # ── 6d2. conv_transpose2d（T2 gx 路径；w 为 PyTorch 转置布局 [Cin,Cout,KH,KW]）
    def _run_t2(x, w, sh, sw, ph, pw, opad_h, opad_w, label):
        B, Cin, OH, OW = x.shape
        Cout = w.shape[1]
        KH, KW = w.shape[2], w.shape[3]
        ref = conv_t2d_ref(x, w, sh, sw, ph, pw, opad_h, opad_w)
        H_out, W_out = ref.shape[2], ref.shape[3]
        x_id = upload(dll, h, x.ravel())
        w_id = upload(dll, h, w.ravel())
        o_id = upload(dll, h, np.zeros(B * Cout * H_out * W_out, dtype=np.float32))
        t0 = time.perf_counter()
        rc = dll.rvc_conv_t2d(h, x_id, w_id, o_id, B, Cin, OH, OW, Cout,
                              KH, KW, sh, sw, ph, pw, opad_h, opad_w, H_out, W_out)
        timings[f"conv_t2d_{label}"] = time.perf_counter() - t0
        check(rc == 0, f"conv_t2d {label} rc=0 (err={last_error(dll) if rc else ''})")
        got = download(dll, h, o_id, B * Cout * H_out * W_out).reshape(B, Cout, H_out, W_out)
        e = rel_err(got, ref)
        check(e < 1e-3, f"conv_t2d {label} rel_err={e:.2e} < 1e-3")
        return e

    # 组1：通用随机小形状（KH=3 KW=2 s=(2,1) p=(1,0) opad_h=1）
    x = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    w = rng.standard_normal((3, 4, 3, 2)).astype(np.float32)
    e_t2_1 = _run_t2(x, w, 2, 1, 1, 0, 1, 0, "3x2_s21_p10_op_h1")
    # 组2：KW=1 判别器风格（KH=5 KW=1 s_h=3 p_h=2 → opad_h=1；Cout=64 触发 tile64）
    x = rng.standard_normal((2, 8, 14, 14)).astype(np.float32)
    w = rng.standard_normal((8, 64, 5, 1)).astype(np.float32)
    e_t2_2 = _run_t2(x, w, 3, 1, 2, 0, 1, 0, "5x1_s31_p20_op_h1")
    # 组3：双轴 opad>0（KH=KW=3 s=(2,2) p=1 opad=(1,1)）
    x = rng.standard_normal((2, 4, 5, 5)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    e_t2_3 = _run_t2(x, w, 2, 2, 1, 1, 1, 1, "3x3_s22_p1_op_11")

    # ── 6e. embedding lookup（int32 ids 作为原始字节上传）───────────
    N, TableRows, EmbDim = 64, 100, 16
    ids = rng.integers(0, TableRows, size=N).astype(np.int32)
    table = rng.standard_normal((TableRows, EmbDim)).astype(np.float32)
    ref = table[ids]  # [N, EmbDim]
    ids_id = upload(dll, h, ids.view(np.float32))  # int32 字节作为 f32 槽上传
    tbl_id = upload(dll, h, table.ravel())
    o_id = upload(dll, h, np.zeros(N * EmbDim, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_embed(h, ids_id, tbl_id, o_id, N, TableRows, EmbDim)
    timings["embed_64x16"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_embed rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, N * EmbDim).reshape(N, EmbDim)
    e = rel_err(got, ref)
    check(e < 1e-6, f"embedding gather rel_err={e:.2e} < 1e-6")

    # ── 6f. 元素级 in-place add/mul ─────────────────────────────────
    Nn = 12345
    a = rng.standard_normal(Nn).astype(np.float32)
    bb = rng.standard_normal(Nn).astype(np.float32)
    a_id = upload(dll, h, a)
    b_id = upload(dll, h, bb)
    t0 = time.perf_counter()
    rc = dll.rvc_add_inplace(h, a_id, b_id, Nn)
    timings["add_inplace"] = time.perf_counter() - t0
    check(rc == 0, "rvc_add_inplace rc=0")
    e = rel_err(download(dll, h, a_id, Nn), (a + bb).astype(np.float32))
    check(e < 1e-6, f"add_inplace rel_err={e:.2e} < 1e-6")

    rc = dll.rvc_mul_inplace(h, a_id, b_id, Nn)
    check(rc == 0, "rvc_mul_inplace rc=0")
    e = rel_err(download(dll, h, a_id, Nn), ((a + bb) * bb).astype(np.float32))
    check(e < 1e-6, f"mul_inplace(after add) rel_err={e:.2e} < 1e-6")

    # ── 7. layernorm (T29): 256x512 vs 手算 ──────────────────────────
    rows, cols, eps = 256, 512, 1e-5
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    gamma = rng.standard_normal(cols).astype(np.float32)
    beta = rng.standard_normal(cols).astype(np.float32)
    ref = layernorm_ref(x, gamma, beta, eps)
    x_id = upload(dll, h, x.ravel())
    g_id = upload(dll, h, gamma)
    bt_id = upload(dll, h, beta)
    o_id = upload(dll, h, np.zeros(rows * cols, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_layernorm(h, x_id, g_id, bt_id, o_id, rows, cols, eps)
    timings["layernorm_256x512"] = time.perf_counter() - t0
    check(rc == 0, f"layernorm 256x512 rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, rows * cols).reshape(rows, cols)
    e = rel_err(got, ref)
    check(e < 1e-4, f"layernorm 256x512 rel_err={e:.2e} < 1e-4")

    # ── 8. softmax (T29): 大数值 1000 稳定性 ─────────────────────────
    sm_rows, sm_cols = 8, 512
    x = (rng.standard_normal((sm_rows, sm_cols)).astype(np.float32) + 1000.0)
    ref = softmax_ref(x)
    x_id = upload(dll, h, x.ravel())
    o_id = upload(dll, h, np.zeros(sm_rows * sm_cols, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_softmax(h, x_id, o_id, sm_rows, sm_cols)
    timings["softmax_8x512"] = time.perf_counter() - t0
    check(rc == 0, f"softmax 8x512 rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, sm_rows * sm_cols).reshape(sm_rows, sm_cols)
    e = rel_err(got, ref)
    check(np.isfinite(got).all(), "softmax 大数值 1000 全部有限（无 inf/nan）")
    check(e < 1e-4, f"softmax 大数值 1000 rel_err={e:.2e} < 1e-4")

    # ── 9. rmsnorm (T30): 128x256 ────────────────────────────────────
    rows, cols, eps = 128, 256, 1e-5
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    gamma = rng.standard_normal(cols).astype(np.float32)
    ref = rmsnorm_ref(x, gamma, eps)
    x_id = upload(dll, h, x.ravel())
    g_id = upload(dll, h, gamma)
    o_id = upload(dll, h, np.zeros(rows * cols, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_rmsnorm(h, x_id, g_id, o_id, rows, cols, eps)
    timings["rmsnorm_128x256"] = time.perf_counter() - t0
    check(rc == 0, f"rmsnorm 128x256 rc=0 (err={last_error(dll) if rc else ''})")
    got = download(dll, h, o_id, rows * cols).reshape(rows, cols)
    e = rel_err(got, ref)
    check(e < 1e-4, f"rmsnorm 128x256 rel_err={e:.2e} < 1e-4")

    # ── 9b. batch recorder：多算子单次提交 ─────────────────────────
    # matmul x3 + conv1d x2 + add_inplace x2 → 一次 rvc_batch_commit，
    # 每个算子与 numpy 参考一致；含 batch 内数据依赖链（barrier 验证）。
    rng = np.random.default_rng(123)
    batch_refs = []

    rc = dll.rvc_batch_begin(h)
    check(rc == 0, "rvc_batch_begin rc=0")

    # 3x matmul（不同尺寸）
    for (M, K, N) in [(32, 48, 64), (17, 23, 33), (64, 16, 8)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        a_id = upload(dll, h, A.ravel())
        b_id = upload(dll, h, B.ravel())
        c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))
        rc = batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
        check(rc == 0, f"batch_add matmul {M}x{K}x{N} rc=0")
        batch_refs.append((c_id, (A @ B).ravel(), f"matmul_{M}x{K}x{N}"))

    # 2x conv1d（一个带 bias、一个无 bias 走零 buffer 路径）
    for (Bb, Ci, L, Co, Kk, s, pl, pr, d, has_bias) in [
        (2, 4, 64, 8, 3, 1, 1, 1, 1, True),
        (2, 4, 32, 6, 5, 1, 2, 2, 2, False),
    ]:
        oL = (L + pl + pr - d * (Kk - 1) - 1) // s + 1
        x = rng.standard_normal((Bb, Ci, L)).astype(np.float32)
        w = rng.standard_normal((Co, Ci, Kk)).astype(np.float32)
        b_arr = rng.standard_normal(Co).astype(np.float32) if has_bias else None
        ref = conv1d_ref(x, w, b_arr, s, pl, pr, d)
        x_id = upload(dll, h, x.ravel())
        w_id = upload(dll, h, w.ravel())
        b_id = upload(dll, h, b_arr) if has_bias else 0
        o_id = upload(dll, h, np.zeros(Bb * Co * oL, dtype=np.float32))
        # conv1d：p0..p8 为维度，out 句柄放 p9
        rc = batch_add(dll, h, 2, x_id, w_id, b_id, Bb, Ci, L, Co, Kk, s, pl, pr, d, o_id)
        check(rc == 0, f"batch_add conv1d K={Kk} bias={int(has_bias)} rc=0")
        batch_refs.append((o_id, ref.ravel(), f"conv1d_K{Kk}_bias{int(has_bias)}"))

    # 2x add_inplace：同一 buffer 链式累加（验证同 buffer 多写依赖）
    Nn = 20000
    a4 = rng.standard_normal(Nn).astype(np.float32)
    b4 = rng.standard_normal(Nn).astype(np.float32)
    b5 = rng.standard_normal(Nn).astype(np.float32)
    a_id = upload(dll, h, a4)
    bb_id = upload(dll, h, b4)
    bb2_id = upload(dll, h, b5)
    rc = batch_add(dll, h, 3, a_id, bb_id, 0, Nn)
    check(rc == 0, "batch_add add_inplace#1 rc=0")
    rc = batch_add(dll, h, 3, a_id, bb2_id, 0, Nn)
    check(rc == 0, "batch_add add_inplace#2 rc=0")
    batch_refs.append((a_id, (a4 + b4 + b5).ravel(), "add_inplace_chain"))

    # 依赖链：matmul 输出 → add_inplace 读它（验证 batch 内 barrier）
    M, K, N = 32, 32, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    D = rng.standard_normal((M, N)).astype(np.float32)
    a_id = upload(dll, h, A.ravel())
    b_id = upload(dll, h, B.ravel())
    c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))
    d_id = upload(dll, h, D.ravel())
    rc = batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
    check(rc == 0, "batch_add dep matmul rc=0")
    rc = batch_add(dll, h, 3, c_id, d_id, 0, M * N)
    check(rc == 0, "batch_add dep add_inplace rc=0")
    batch_refs.append((c_id, (A @ B + D).ravel(), "matmul_then_add_inplace"))

    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_8ops"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit rc=0 (err={last_error(dll) if rc else ''})")

    for buf_id, ref, label in batch_refs:
        got = download(dll, h, buf_id, ref.size)
        e = rel_err(got, ref)
        check(e < 1e-4, f"batch {label} rel_err={e:.2e} < 1e-4")

    # 空批次 commit / discard 幂等
    rc = dll.rvc_batch_commit(h)
    check(rc == 0, "rvc_batch_commit on empty batch rc=0")
    rc = dll.rvc_batch_discard(h)
    check(rc == 0, "rvc_batch_discard rc=0 (idempotent)")

    # discard 中途批次后 batch 恢复正常
    rc = dll.rvc_batch_begin(h)
    rc = batch_add(dll, h, 3, a_id, bb_id, 0, 1000)
    rc = dll.rvc_batch_discard(h)
    check(rc == 0, "rvc_batch_discard mid-batch rc=0")
    rc = dll.rvc_batch_commit(h)
    check(rc == 0, "commit after discard rc=0 (batch was reset)")

    # 非法 op 报错
    rc = batch_add(dll, h, 99, 0, 0, 0)
    err = last_error(dll)
    check(rc == -1 and len(err) > 0, f"batch_add invalid op -> rc=-1 ({err!r})")

    # ── 9c. batch conv_t1d（op=5）：与单次 rvc_conv_t1d / numpy 对照 ──
    # 布局：a=x b=w c=bias，p0=B p1=C_in p2=L p3=C_out p4=K p5=stride
    # p6=padding p7=output_padding p8=dil p9=out（w 为 PyTorch [C_in,C_out,K]）。
    # dec ups 形状（B=1 Cin=192 L=16 Cout=384 K=4 s=2 p=1 op=1）与 K=16 大核。
    for (Bb, Ci, L, Co, Kk, s, pad, opad, dil, has_bias) in [
        (1, 192, 16, 384, 4, 2, 1, 1, 1, True),
        (1, 8, 16, 16, 16, 4, 3, 0, 2, False),
    ]:
        oL = (L - 1) * s - 2 * pad + dil * (Kk - 1) + opad + 1
        x = rng.standard_normal((Bb, Ci, L)).astype(np.float32)
        w = rng.standard_normal((Ci, Co, Kk)).astype(np.float32)
        b_arr = rng.standard_normal(Co).astype(np.float32) if has_bias else None
        ref = conv_t1d_ref(x, w, b_arr, s, pad, opad, dil)
        x_id = upload(dll, h, x.ravel())
        w_id = upload(dll, h, w.ravel())
        b_id = upload(dll, h, b_arr) if has_bias else 0
        o_id = upload(dll, h, np.zeros(Bb * Co * oL, dtype=np.float32))
        rc = batch_add(dll, h, 5, x_id, w_id, b_id,
                       Bb, Ci, L, Co, Kk, s, pad, opad, dil, o_id)
        check(rc == 0, f"batch_add conv_t1d K={Kk} bias={int(has_bias)} rc=0")
        batch_refs.append((o_id, ref.ravel(), f"conv_t1d_K{Kk}_bias{int(has_bias)}"))

    # 依赖链：conv_t1d 输出 → add_inplace 读它（验证 batch 内 barrier）
    Bb, Ci, L, Co, Kk, s, pad, opad, dil = 1, 4, 12, 8, 3, 2, 1, 1, 1
    oL = (L - 1) * s - 2 * pad + dil * (Kk - 1) + opad + 1
    x = rng.standard_normal((Bb, Ci, L)).astype(np.float32)
    w = rng.standard_normal((Ci, Co, Kk)).astype(np.float32)
    add_arr = rng.standard_normal(Bb * Co * oL).astype(np.float32)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    o_id = upload(dll, h, np.zeros(Bb * Co * oL, dtype=np.float32))
    add_id = upload(dll, h, add_arr)
    rc = batch_add(dll, h, 5, x_id, w_id, 0, Bb, Ci, L, Co, Kk, s, pad, opad, dil, o_id)
    check(rc == 0, "batch_add dep conv_t1d rc=0")
    rc = batch_add(dll, h, 3, o_id, add_id, 0, Bb * Co * oL)
    check(rc == 0, "batch_add dep add_inplace after conv_t1d rc=0")
    ref_dep = (conv_t1d_ref(x, w, None, s, pad, opad, dil).ravel() + add_arr).astype(np.float32)
    batch_refs.append((o_id, ref_dep, "conv_t1d_then_add_inplace"))

    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9c"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9c rc=0 (err={last_error(dll) if rc else ''})")
    for buf_id, ref, label in batch_refs:
        got = download(dll, h, buf_id, ref.size)
        e = rel_err(got, ref)
        check(e < 1e-4, f"batch {label} rel_err={e:.2e} < 1e-4")

    # 与单次 rvc_conv_t1d 对照（同输入、同 kernel、同 push 布局 -> 位级一致）
    Bb, Ci, L, Co, Kk, s, pad, opad, dil = 1, 192, 16, 384, 4, 2, 1, 1, 1
    oL = (L - 1) * s - 2 * pad + dil * (Kk - 1) + opad + 1
    x = rng.standard_normal((Bb, Ci, L)).astype(np.float32)
    w = rng.standard_normal((Ci, Co, Kk)).astype(np.float32)
    b_arr = rng.standard_normal(Co).astype(np.float32)
    x_id = upload(dll, h, x.ravel())
    w_id = upload(dll, h, w.ravel())
    b_id = upload(dll, h, b_arr)
    o1_id = upload(dll, h, np.zeros(Bb * Co * oL, dtype=np.float32))
    o2_id = upload(dll, h, np.zeros(Bb * Co * oL, dtype=np.float32))
    rc = batch_add(dll, h, 5, x_id, w_id, b_id, Bb, Ci, L, Co, Kk, s, pad, opad, dil, o1_id)
    check(rc == 0, "batch_add conv_t1d 单次对照 rc=0")
    rc = dll.rvc_batch_commit(h)
    check(rc == 0, "batch commit 单次对照 rc=0")
    rc = dll.rvc_conv_t1d(h, x_id, w_id, b_id, o2_id, Bb, Ci, L, Co, Kk, s, pad, opad, dil)
    check(rc == 0, "single rvc_conv_t1d 对照 rc=0")
    d_batch = download(dll, h, o1_id, Bb * Co * oL)
    d_single = download(dll, h, o2_id, Bb * Co * oL)
    e_single = rel_err(d_batch, d_single)
    timings["batch_vs_single_conv_t1d"] = e_single
    check(e_single == 0.0, f"batch conv_t1d 与单次调用位级一致 rel_err={e_single:.2e} == 0")

    # ── 9d. batch leaky_relu（op=6）+ copy（op=7）───────────────────
    # leaky：a 就地，p0=n，p1=slope 的 f32 位模式；与 numpy where 语义对照
    # （含负 slope 的位模式传递）；copy：dst=src（残差保留原值）。
    batch_refs = []
    Nn = 30000
    a6 = rng.standard_normal(Nn).astype(np.float32)
    a6_id = upload(dll, h, a6)
    rc = batch_add(dll, h, 6, a6_id, 0, 0, Nn, np.float32(0.1).view(np.int32).item())
    check(rc == 0, "batch_add leaky_relu slope=0.1 rc=0")
    ref6 = np.where(a6 >= 0, a6, 0.1 * a6).astype(np.float32)
    batch_refs.append((a6_id, ref6.ravel(), "leaky_relu_slope0.1"))

    # 负 slope（-0.5）：位模式符号位为 1，验证 i64 传递不被 <0 校验误拒
    a7 = rng.standard_normal(Nn).astype(np.float32)
    a7_id = upload(dll, h, a7)
    rc = batch_add(dll, h, 6, a7_id, 0, 0, Nn, np.float32(-0.5).view(np.int32).item())
    check(rc == 0, "batch_add leaky_relu slope=-0.5（负位模式）rc=0")
    ref7 = np.where(a7 >= 0, a7, -0.5 * a7).astype(np.float32)
    batch_refs.append((a7_id, ref7.ravel(), "leaky_relu_slope-0.5"))

    # 链式：copy 保留原值 → leaky 就地 → add_inplace 恢复（ResBlock 残差模式）
    a8 = rng.standard_normal(Nn).astype(np.float32)
    a8_id = upload(dll, h, a8)
    orig_id = upload(dll, h, np.zeros(Nn, dtype=np.float32))
    rc = batch_add(dll, h, 7, orig_id, a8_id, 0, Nn)
    check(rc == 0, "batch_add copy rc=0")
    rc = batch_add(dll, h, 6, a8_id, 0, 0, Nn, np.float32(0.1).view(np.int32).item())
    check(rc == 0, "batch_add leaky after copy rc=0")
    rc = batch_add(dll, h, 3, a8_id, orig_id, 0, Nn)
    check(rc == 0, "batch_add add_inplace(leaky(x), copy(x)) rc=0")
    ref8 = (np.where(a8 >= 0, a8, 0.1 * a8) + a8).astype(np.float32)
    batch_refs.append((a8_id, ref8.ravel(), "copy_then_leaky_then_add"))

    # 与单次 rvc_leaky_relu / rvc_copy 对照（同 kernel/push -> 位级一致）
    a9 = rng.standard_normal(Nn).astype(np.float32)
    a9_id = upload(dll, h, a9)
    a9b_id = upload(dll, h, a9)
    rc = batch_add(dll, h, 6, a9_id, 0, 0, Nn, np.float32(0.2).view(np.int32).item())
    check(rc == 0, "batch_add leaky 单次对照 rc=0")
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9d"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9d rc=0 (err={last_error(dll) if rc else ''})")
    rc = dll.rvc_leaky_relu(h, a9b_id, Nn, np.float32(0.2).view(np.int32).item())
    check(rc == 0, "single rvc_leaky_relu 对照 rc=0")
    d9b = download(dll, h, a9_id, Nn)
    d9s = download(dll, h, a9b_id, Nn)
    e9 = rel_err(d9b, d9s)
    timings["batch_vs_single_leaky"] = e9
    check(e9 == 0.0, f"batch leaky 与单次调用位级一致 rel_err={e9:.2e} == 0")

    for buf_id, ref, label in batch_refs:
        got = download(dll, h, buf_id, ref.size)
        e = rel_err(got, ref)
        check(e < 1e-4, f"batch {label} rel_err={e:.2e} < 1e-4")

    # ── 9e. batch attention middleware（op8-15）──────────────────────
    # softmax / layer_norm / gelu / bias_add / attn_qk / attn_sv / relu /
    # group_norm：与 numpy 参考对照 + 与单次调用位级一致。
    batch_refs = []

    # 9e-1: batch softmax（op8）——行语义 + 大数值稳定性 + 与单次位级一致
    sm_rows, sm_cols = 32, 128
    sm = (rng.standard_normal((sm_rows, sm_cols)).astype(np.float32) + 50.0)
    sm_id = upload(dll, h, sm.ravel())
    sm_o_id = upload(dll, h, np.zeros(sm_rows * sm_cols, dtype=np.float32))
    rc = batch_add(dll, h, 8, sm_id, 0, sm_o_id, sm_rows, sm_cols)
    check(rc == 0, "batch_add softmax rc=0")
    batch_refs.append((sm_o_id, softmax_ref(sm).ravel(), "softmax_32x128"))
    # 与单次 rvc_softmax 位级一致
    sm_o2_id = upload(dll, h, np.zeros(sm_rows * sm_cols, dtype=np.float32))
    rc = dll.rvc_softmax(h, sm_id, sm_o2_id, sm_rows, sm_cols)
    check(rc == 0, "single rvc_softmax 对照 rc=0")
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9e1"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9e1 rc=0 (err={last_error(dll) if rc else ''})")
    d_sm = download(dll, h, sm_o_id, sm_rows * sm_cols)
    d_sm2 = download(dll, h, sm_o2_id, sm_rows * sm_cols)
    check(np.array_equal(d_sm, d_sm2), "batch softmax 与单次调用位级一致")
    # z-split 路径（rows > 65535 -> gy=65535, gz>1）
    zr, zc = 70000, 4
    z = rng.standard_normal((zr, zc)).astype(np.float32)
    z_id = upload(dll, h, z.ravel())
    z_o_id = upload(dll, h, np.zeros(zr * zc, dtype=np.float32))
    rc = batch_add(dll, h, 8, z_id, 0, z_o_id, zr, zc)
    check(rc == 0, "batch_add softmax rows>65535 rc=0")
    rc = dll.rvc_batch_commit(h)
    check(rc == 0, "batch commit softmax z-split rc=0")
    z_got = download(dll, h, z_o_id, zr * zc).reshape(zr, zc)
    e = rel_err(z_got, softmax_ref(z))
    check(e < 1e-4, f"batch softmax z-split rel_err={e:.2e} < 1e-4")
    dll.rvc_mem_free(h, z_id)
    dll.rvc_mem_free(h, z_o_id)

    # 9e-2: batch layer_norm（op9）——eps 边界（0 与默认 1e-5）+ 位级一致
    ln_rows, ln_cols, ln_eps = 64, 256, 1e-5
    lx = rng.standard_normal((ln_rows, ln_cols)).astype(np.float32)
    lg = rng.standard_normal(ln_cols).astype(np.float32)
    lb = rng.standard_normal(ln_cols).astype(np.float32)
    lx_id = upload(dll, h, lx.ravel())
    lg_id = upload(dll, h, lg)
    lb_id = upload(dll, h, lb)
    ln_o_id = upload(dll, h, np.zeros(ln_rows * ln_cols, dtype=np.float32))
    eps_bits = np.float32(ln_eps).view(np.int32).item()
    rc = batch_add(dll, h, 9, lx_id, lg_id, lb_id,
                   ln_rows, ln_cols, eps_bits, 0, 0, 0, 0, 0, 0, ln_o_id)
    check(rc == 0, "batch_add layer_norm rc=0")
    batch_refs.append((ln_o_id, layernorm_ref(lx, lg, lb, ln_eps).ravel(),
                       "layer_norm_64x256_eps1e-5"))
    ln_o2_id = upload(dll, h, np.zeros(ln_rows * ln_cols, dtype=np.float32))
    rc = dll.rvc_layernorm(h, lx_id, lg_id, lb_id, ln_o2_id, ln_rows, ln_cols, ln_eps)
    check(rc == 0, "single rvc_layernorm 对照 rc=0")
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9e2"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9e2 rc=0 (err={last_error(dll) if rc else ''})")
    d_ln = download(dll, h, ln_o_id, ln_rows * ln_cols)
    d_ln2 = download(dll, h, ln_o2_id, ln_rows * ln_cols)
    check(np.array_equal(d_ln, d_ln2), "batch layer_norm 与单次调用位级一致")
    # eps=0 边界（方差恰好为 0 的行 -> 可除性由 var+eps 保护）
    lz = np.zeros((4, 16), dtype=np.float32)
    lz_id = upload(dll, h, lz.ravel())
    lz_o_id = upload(dll, h, np.zeros(64, dtype=np.float32))
    rc = batch_add(dll, h, 9, lz_id, lg_id, lb_id,
                   4, 16, np.float32(0.0).view(np.int32).item(),
                   0, 0, 0, 0, 0, 0, lz_o_id)
    check(rc == 0, "batch_add layer_norm eps=0 rc=0")
    rc = dll.rvc_batch_commit(h)
    check(rc == 0, "batch commit layer_norm eps=0 rc=0")
    lz_got = download(dll, h, lz_o_id, 64).reshape(4, 16)
    # eps=0 且方差为 0 的行：numpy 参考同样产生 0/0=NaN —— 两端 NaN 模式一致即可
    lz_ref = layernorm_ref(lz, lg[:16], lb[:16], 0.0)
    same_nan = ((np.isnan(lz_got) == np.isnan(lz_ref)).all()
                and np.allclose(lz_got, lz_ref, atol=1e-6, equal_nan=True))
    check(same_nan, "layer_norm eps=0 全零行与 numpy 参考 NaN 模式一致")
    dll.rvc_mem_free(h, lz_id)
    dll.rvc_mem_free(h, lz_o_id)

    # 9e-3: batch gelu（op10）+ relu（op14）——就地，与 numpy erf 参考对照
    gn2 = 20000
    ga = rng.standard_normal(gn2).astype(np.float32)
    ga_id = upload(dll, h, ga)
    rc = batch_add(dll, h, 10, ga_id, 0, 0, gn2)
    check(rc == 0, "batch_add gelu rc=0")
    gb = rng.standard_normal(gn2).astype(np.float32)
    gb_id = upload(dll, h, gb)
    rc = batch_add(dll, h, 14, gb_id, 0, 0, gn2)
    check(rc == 0, "batch_add relu rc=0")
    batch_refs.append((ga_id, gelu_ref(ga).ravel(), "gelu_20000"))
    batch_refs.append((gb_id, np.maximum(gb, 0.0).ravel(), "relu_20000"))
    # gelu 与 numpy 最大相对差（float32 A&S vs float64 参考）
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9e3"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9e3 rc=0 (err={last_error(dll) if rc else ''})")
    g_got = download(dll, h, ga_id, gn2)
    check(rel_err(g_got, gelu_ref(ga)) < 1e-4,
          f"batch gelu rel_err={rel_err(g_got, gelu_ref(ga)):.2e} < 1e-4")

    # 9e-4: batch bias_add（op11）——行广播
    ba_rows, ba_cols = 128, 64
    ba = rng.standard_normal((ba_rows, ba_cols)).astype(np.float32)
    bb_vec = rng.standard_normal(ba_cols).astype(np.float32)
    ba_id = upload(dll, h, ba.ravel())
    bbv_id = upload(dll, h, bb_vec)
    bao_id = upload(dll, h, np.zeros(ba_rows * ba_cols, dtype=np.float32))
    rc = batch_add(dll, h, 11, ba_id, bbv_id, bao_id, ba_rows * ba_cols, ba_cols)
    check(rc == 0, "batch_add bias_add rc=0")
    batch_refs.append((bao_id, (ba + bb_vec).ravel(), "bias_add_128x64"))

    # 9e-5: batch attn_qk（op12）+ softmax + attn_sv（op13）——端到端 attention 链
    attn_T, attn_H, attn_D, attn_C = 73, 4, 32, 128  # 非 16 倍数 T + 多 head
    aq = rng.standard_normal((attn_T, attn_C)).astype(np.float32)
    ak = rng.standard_normal((attn_T, attn_C)).astype(np.float32)
    av = rng.standard_normal((attn_T, attn_C)).astype(np.float32)
    aq_id = upload(dll, h, aq.ravel())
    ak_id = upload(dll, h, ak.ravel())
    av_id = upload(dll, h, av.ravel())
    scores_id = upload(dll, h, np.zeros(attn_H * attn_T * attn_T, dtype=np.float32))
    aw_id = upload(dll, h, np.zeros(attn_H * attn_T * attn_T, dtype=np.float32))
    ctx_id = upload(dll, h, np.zeros(attn_T * attn_C, dtype=np.float32))
    rc = batch_add(dll, h, 12, aq_id, ak_id, scores_id,
                   attn_H, attn_T, attn_D, attn_C)
    check(rc == 0, "batch_add attn_qk rc=0")
    rc = batch_add(dll, h, 8, scores_id, 0, aw_id, attn_H * attn_T, attn_T)
    check(rc == 0, "batch_add softmax over scores rc=0")
    rc = batch_add(dll, h, 13, aw_id, av_id, ctx_id,
                   attn_H, attn_T, attn_D, attn_C)
    check(rc == 0, "batch_add attn_sv rc=0")
    aw_ref, ctx_ref = attn_ref(aq, ak, av, D=attn_D)
    aqh = aq.reshape(1, attn_T, attn_H, attn_D).transpose(0, 2, 1, 3)
    akh = ak.reshape(1, attn_T, attn_H, attn_D).transpose(0, 2, 1, 3)
    scores_ref = np.einsum("bhtd,bhsd->bhts", aqh, akh)[0]  # [H,T,T]
    batch_refs.append((scores_id, scores_ref.ravel(), "attn_qk_73x4x32"))
    batch_refs.append((aw_id, aw_ref.ravel(), "attn_softmax_73x73"))
    batch_refs.append((ctx_id, ctx_ref.ravel(), "attn_sv_73x128"))
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9e5"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9e5 rc=0 (err={last_error(dll) if rc else ''})")

    # 9e-6: batch group_norm（op15）——与 numpy group_norm 参考对照
    gB, gC, gS, gG = 1, 512, 37, 512  # hubert conv0 形态（每通道一组）
    gx = rng.standard_normal((gB, gC, gS)).astype(np.float32)
    gg = rng.standard_normal(gC).astype(np.float32)
    gbt = rng.standard_normal(gC).astype(np.float32)
    gx_id = upload(dll, h, gx.ravel())
    gg_id = upload(dll, h, gg)
    gbt_id = upload(dll, h, gbt)
    go_id = upload(dll, h, np.zeros(gB * gC * gS, dtype=np.float32))
    geps = np.float32(1e-5).view(np.int32).item()
    rc = batch_add(dll, h, 15, gx_id, gg_id, gbt_id,
                   gG, gC // gG, gS, geps, 0, 0, 0, 0, 0, go_id)
    check(rc == 0, "batch_add group_norm rc=0")
    # numpy 参考（nn.group_norm 同公式）
    xr = gx.reshape(gB, gG, gC // gG, gS)
    mean = xr.mean(axis=(2, 3), keepdims=True)
    var = xr.var(axis=(2, 3), keepdims=True)
    xn = (xr - mean) / np.sqrt(var + 1e-5)
    ref_gn = (xn * gg.reshape(1, gG, gC // gG, 1)
              + gbt.reshape(1, gG, gC // gG, 1)).reshape(gx.shape)
    batch_refs.append((go_id, ref_gn.ravel(), "group_norm_1x512x37"))
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["batch_commit_9e6"] = time.perf_counter() - t0
    check(rc == 0, f"rvc_batch_commit 9e6 rc=0 (err={last_error(dll) if rc else ''})")

    for buf_id, ref, label in batch_refs:
        got = download(dll, h, buf_id, ref.size)
        e = rel_err(got, ref)
        check(e < 1e-4, f"batch {label} rel_err={e:.2e} < 1e-4")

    # ── 10. 性能：大张量 layernorm 512x2048（含上传/下载）──────────
    rows, cols = 512, 2048
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    gamma = rng.standard_normal(cols).astype(np.float32)
    beta = rng.standard_normal(cols).astype(np.float32)
    x_id = upload(dll, h, x.ravel())
    g_id = upload(dll, h, gamma)
    bt_id = upload(dll, h, beta)
    o_id = upload(dll, h, np.zeros(rows * cols, dtype=np.float32))
    t0 = time.perf_counter()
    rc = dll.rvc_layernorm(h, x_id, g_id, bt_id, o_id, rows, cols, 1e-5)
    timings["layernorm_512x2048"] = time.perf_counter() - t0
    check(rc == 0, "layernorm 512x2048 rc=0 (perf)")

    # ── 10b. 性能：10x matmul 逐次 vs 一次 batch ────────────────────
    M, K, N = 128, 256, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    a_id = upload(dll, h, A.ravel())
    b_id = upload(dll, h, B.ravel())
    c_ids = [upload(dll, h, np.zeros(M * N, dtype=np.float32)) for _ in range(10)]

    dll.rvc_matmul(h, a_id, b_id, c_ids[0], M, K, N)  # warmup

    t0 = time.perf_counter()
    for c_id in c_ids:
        dll.rvc_matmul(h, a_id, b_id, c_id, M, K, N)
    t_seq = time.perf_counter() - t0
    timings["10x_matmul_sequential"] = t_seq

    rc = dll.rvc_batch_begin(h)
    check(rc == 0, "perf batch_begin rc=0")
    for c_id in c_ids:
        rc = batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
        assert rc == 0, f"perf batch_add failed: {last_error(dll)}"
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    t_batch = time.perf_counter() - t0
    timings["10x_matmul_batch"] = t_batch
    check(rc == 0, "perf 10x matmul batch commit rc=0")

    per_op_seq = t_seq / 10
    per_op_batch = t_batch / 10
    speedup = per_op_seq / per_op_batch if per_op_batch > 0 else float("inf")
    print(
        f"  10x matmul 128x256x128: per-op sequential={per_op_seq*1e3:.3f} ms "
        f"batch={per_op_batch*1e3:.3f} ms speedup={speedup:.1f}x"
    )
    got = download(dll, h, c_ids[9], M * N)
    e = rel_err(got, (A @ B).ravel())
    check(e < 1e-4, f"perf batch10x matmul final rel_err={e:.2e} < 1e-4")

    # ── 10c. async commit（P1：异步提交 + 批量 wait）──────────────
    # rvc_batch_commit_async 提交不等待；多个 async 批次在队列上流水线
    # 执行（顺序保证），一次 rvc_batch_wait 收齐。语义测试：
    #   - async 提交后（未 wait）即可再次录制/提交下一批（依赖安全：
    #     同一 queue 顺序执行；recorder 帧轮转防止覆盖在途命令缓冲）
    #   - wait 后结果与 numpy 一致
    #   - 无在途时 wait 为 no-op
    M, K, N = 128, 256, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    a_id = upload(dll, h, A.ravel())
    b_id = upload(dll, h, B.ravel())
    c_ids2 = [upload(dll, h, np.zeros(M * N, dtype=np.float32)) for _ in range(4)]

    # 空批次 async 提交（no-op，rc=0）
    rc = dll.rvc_batch_begin(h)
    check(rc == 0, "async: empty batch begin rc=0")
    rc = dll.rvc_batch_commit_async(h)
    check(rc == 0, "async: empty batch commit_async rc=0 (no-op)")

    # 4 个 async 批次连续提交，最后统一 wait
    for idx, c_id in enumerate(c_ids2):
        rc = dll.rvc_batch_begin(h)
        check(rc == 0, f"async: begin batch#{idx} rc=0")
        rc = batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
        check(rc == 0, f"async: batch#{idx} add matmul rc=0")
        t0 = time.perf_counter()
        rc = dll.rvc_batch_commit_async(h)
        timings[f"async_commit_{idx}"] = time.perf_counter() - t0
        check(rc == 0, f"async: batch#{idx} commit_async rc=0")

    t0 = time.perf_counter()
    rc = dll.rvc_batch_wait(h)
    timings["async_wait_4"] = time.perf_counter() - t0
    check(rc == 0, f"async: rvc_batch_wait rc=0 (err={last_error(dll) if rc else ''})")

    for idx, c_id in enumerate(c_ids2):
        got = download(dll, h, c_id, M * N)
        e = rel_err(got, (A @ B).ravel())
        check(e < 1e-4, f"async: batch#{idx} matmul rel_err={e:.2e} < 1e-4")

    # 无在途时 wait no-op
    t0 = time.perf_counter()
    rc = dll.rvc_batch_wait(h)
    timings["async_wait_idle"] = time.perf_counter() - t0
    check(rc == 0, "async: wait with nothing in flight rc=0 (no-op)")

    # 混合：async 提交后立即再开一个同步 batch（验证帧轮转不冲突）
    rc = dll.rvc_batch_begin(h)
    rc = batch_add(dll, h, 1, a_id, b_id, c_ids2[0], M, K, N)
    check(rc == 0, "async: sync batch add after async rc=0")
    rc = dll.rvc_batch_commit_async(h)
    check(rc == 0, "async: second async commit rc=0")
    # 依赖安全：async 提交后未 wait，直接同步 commit 另一批
    rc = dll.rvc_batch_begin(h)
    rc = batch_add(dll, h, 1, a_id, b_id, c_ids2[1], M, K, N)
    check(rc == 0, "async: sync batch add#2 rc=0")
    t0 = time.perf_counter()
    rc = dll.rvc_batch_commit(h)
    timings["async_then_sync_commit"] = time.perf_counter() - t0
    check(rc == 0, "async: sync commit after async rc=0")
    rc = dll.rvc_batch_wait(h)
    check(rc == 0, "async: final wait rc=0")
    got = download(dll, h, c_ids2[1], M * N)
    e = rel_err(got, (A @ B).ravel())
    check(e < 1e-4, f"async: sync-after-async matmul rel_err={e:.2e} < 1e-4")

    for c_id in c_ids2:
        dll.rvc_mem_free(h, c_id)

    # ── 10d. 每 commit 固定开销拆测（P1）──────────────
    # 单 op commit 墙钟 - 空 GPU 执行 ≈ 固定开销；10x batch 摊薄。
    M, K, N = 1024, 1024, 1024
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    a_id = upload(dll, h, A.ravel())
    b_id = upload(dll, h, B.ravel())
    c_id = upload(dll, h, np.zeros(M * N, dtype=np.float32))
    # warmup
    rc = dll.rvc_batch_begin(h)
    batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
    dll.rvc_batch_commit(h)
    dll.rvc_batch_begin(h)
    batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
    t0 = time.perf_counter()
    dll.rvc_batch_commit(h)
    timings["commit_1op_1024cubed"] = time.perf_counter() - t0
    dll.rvc_batch_begin(h)
    for _ in range(10):
        batch_add(dll, h, 1, a_id, b_id, c_id, M, K, N)
    t0 = time.perf_counter()
    dll.rvc_batch_commit(h)
    timings["commit_10op_1024cubed"] = time.perf_counter() - t0
    dll.rvc_mem_free(h, a_id)
    dll.rvc_mem_free(h, b_id)
    dll.rvc_mem_free(h, c_id)

    # ── cleanup ──────────────────────────────────────────────────────
    rc = dll.rvc_engine_destroy(h)
    check(rc == 0, "rvc_engine_destroy rc=0")

    print("\n  timings (per-call, incl. upload+submit+wait):")
    for name, t in sorted(timings.items()):
        print(f"    {name:<18} {t * 1e3:8.2f} ms")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
