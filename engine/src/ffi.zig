//! C ABI boundary for Python ctypes.
//!
//! Every exported function returns i32: 0 = ok, -1 = error (details in
//! the thread-safe global error buffer, retrieved via rvc_last_error).
//! Handles (engine) and buffer ids are opaque i64 tokens. All engine
//! calls are serialised through Engine.mutex so concurrent ctypes
//! callers are safe.

const std = @import("std");
const engine = @import("engine.zig");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const graph = @import("graph.zig");

// ── Global error buffer (thread-safe) ────────────────────────────────
var g_err_mutex: std.Thread.Mutex = .{};
var g_err_buf: [4096]u8 = undefined;
var g_err_len: usize = 0;

// Registry of the live engine handle so rvc_device_name (which takes no
// handle) can report the device.
var g_engine_mutex: std.Thread.Mutex = .{};
var g_engine: ?i64 = null;

fn setErrorMsg(msg: []const u8) void {
    g_err_mutex.lock();
    defer g_err_mutex.unlock();
    const n = @min(msg.len, g_err_buf.len);
    @memcpy(g_err_buf[0..n], msg[0..n]);
    g_err_len = n;
}

fn setErrorName(e: anyerror) void {
    setErrorMsg(@errorName(e));
}

fn handleErr() i32 {
    setErrorMsg("invalid engine handle");
    return -1;
}

fn engineFromHandle(h: i64) ?*engine.Engine {
    if (h <= 0) return null;
    return @alignCast(@as(*engine.Engine, @ptrFromInt(@as(usize, @bitCast(h)))));
}

// ── Exports ──────────────────────────────────────────────────────────

/// Create an engine; returns the heap pointer as an i64 handle, or -1.
pub export fn rvc_engine_create() i64 {
    const eng = engine.Engine.create(std.heap.c_allocator) catch |e| {
        setErrorName(e);
        return -1;
    };
    const h: i64 = @intCast(@intFromPtr(eng));
    g_engine_mutex.lock();
    g_engine = h;
    g_engine_mutex.unlock();
    return h;
}

/// Destroy an engine and free every buffer it owns.
pub export fn rvc_engine_destroy(handle: i64) i32 {
    const eng = engineFromHandle(handle) orelse return handleErr();
    eng.mutex.lock();
    g_engine_mutex.lock();
    if (g_engine != null and g_engine.? == handle) g_engine = null;
    g_engine_mutex.unlock();
    // destroy() frees `eng`; do not touch it after.
    eng.destroy();
    return 0;
}

/// Create a buffer from `n` f32s; the new buffer id is written to
/// `out_buf`.
pub export fn rvc_mem_upload(h: i64, src: [*]const f32, n: i64, out_buf: *i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    const data = src[0..@as(usize, @intCast(n))];
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const id: u64 = eng.memUpload(data) catch |e| {
        setErrorName(e);
        return -1;
    };
    out_buf.* = @intCast(id);
    return 0;
}

/// 阶段D（D2 输入池）：把数据写进**已存在**的 buffer（覆盖写，不分配新
/// buffer）——训练侧批量输入的池化复用（省每次 upload 的 alloc+free）。
pub export fn rvc_mem_upload_to(h: i64, src: [*]const f32, n: i64, buf: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    const data = src[0..@as(usize, @intCast(n))];
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.memUploadTo(data, @intCast(buf)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Batch upload-to（J9）：n 组 (sizes[i] 个 f32, datas[i], ids[i]) 一次
/// staging 批量 copy + 一次 submit/fence。调用方按 staging 容量分批。
pub export fn rvc_mem_upload_to_batch(
    h: i64,
    n: i64,
    sizes: [*]const i64,
    datas: [*]const [*]const f32,
    ids: [*]const i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n <= 0) return 0;
    const cnt: usize = @intCast(n);
    eng.mutex.lock();
    defer eng.mutex.unlock();
    var ds = eng.allocator.alloc([]const f32, cnt) catch return -2;
    defer eng.allocator.free(ds);
    const ids_c = ids[0..cnt];
    const sizes_c = sizes[0..cnt];
    const datas_c = datas[0..cnt];
    for (0..cnt) |i| {
        if (sizes_c[i] < 0) return -1;
        ds[i] = datas_c[i][0..@as(usize, @intCast(sizes_c[i]))];
    }
    const ids2 = eng.allocator.alloc(u64, cnt) catch return -2;
    defer eng.allocator.free(ids2);
    for (ids_c, 0..) |v, i| ids2[i] = @intCast(v);
    eng.memUploadToBatch(ds, ids2) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Allocate an uninitialised device-local buffer of `nbytes` bytes
/// (P1 perf: output buffers are fully overwritten, zero-fill is wasted).
pub export fn rvc_mem_alloc(h: i64, nbytes: i64, out_buf: *i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (nbytes <= 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const id: u64 = eng.memAlloc(@intCast(nbytes)) catch |e| {
        setErrorName(e);
        return -1;
    };
    out_buf.* = @intCast(id);
    return 0;
}

/// Create a buffer from `nbytes` raw bytes (e.g. float16 payloads);
/// the new buffer id is written to `out_buf`.
pub export fn rvc_mem_upload_bytes(h: i64, src: [*]const u8, nbytes: i64, out_buf: *i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (nbytes < 0) return -1;
    const data = src[0..@as(usize, @intCast(nbytes))];
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const id: u64 = eng.memUploadBytes(data) catch |e| {
        setErrorName(e);
        return -1;
    };
    out_buf.* = @intCast(id);
    return 0;
}

/// GPU zero-fill buffer `buf` (first `nbytes` bytes).
pub export fn rvc_mem_fill_zero(h: i64, buf: i64, nbytes: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (nbytes <= 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.memFillZero(@intCast(buf), @intCast(nbytes)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Copy `n` f32s from buffer `buf` into host memory `dst`.
pub export fn rvc_mem_download(h: i64, buf: i64, dst: [*]f32, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    const out = dst[0..@as(usize, @intCast(n))];
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.memDownload(@intCast(buf), out) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// J18 批量下载：一次 submit + 一次 fence 处理 `count` 个 readback
/// （bufs[i] → dsts[i]，各 sizes[i] f32）。失败时数据不可用，调用方
/// 回退逐次 rvc_mem_download。
pub export fn rvc_mem_download_batch(h: i64, bufs: [*]const i64, dsts: [*]const [*]u8, sizes: [*]const i64, count: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (count < 0) return -1;
    const n: usize = @intCast(count);
    if (n == 0) return 0;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const items = eng.allocator.alloc(buffer.DownloadItem, n) catch return -1;
    defer eng.allocator.free(items);
    for (0..n) |i| {
        items[i] = .{
            .src = @ptrFromInt(@as(usize, @intCast(bufs[i]))),
            .dst = dsts[i],
            .len = @as(usize, @intCast(sizes[i])) * @sizeOf(f32),
        };
    }
    eng.memDownloadBatch(items) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_mem_free(h: i64, buf: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.memFree(@intCast(buf)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_matmul(h: i64, a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (m < 0 or k < 0 or n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.matmul(@intCast(a), @intCast(b), @intCast(c), @intCast(m), @intCast(k), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// FP16-input matmul (P1-4): a/b buffers hold float16 data, c is f32.
pub export fn rvc_matmul_f16(h: i64, a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (m < 0 or k < 0 or n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.matmulF16(@intCast(a), @intCast(b), @intCast(c), @intCast(m), @intCast(k), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_add(h: i64, a: i64, b: i64, c: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.add(@intCast(a), @intCast(b), @intCast(c), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_mul(h: i64, a: i64, b: i64, c: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.mul(@intCast(a), @intCast(b), @intCast(c), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_relu(h: i64, a: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.relu(@intCast(a), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_conv1d(
    h: i64,
    x: i64,
    w: i64,
    b: i64,
    out: i64,
    B: i64,
    c_in: i64,
    l: i64,
    c_out: i64,
    k: i64,
    stride: i64,
    pad_l: i64,
    pad_r: i64,
    dil: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or c_in < 0 or l < 0 or c_out < 0 or k < 0 or stride < 0 or pad_l < 0 or pad_r < 0 or dil < 0)
        return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.conv1d(
        @intCast(x),
        @intCast(w),
        @intCast(b),
        @intCast(out),
        @intCast(B),
        @intCast(c_in),
        @intCast(l),
        @intCast(c_out),
        @intCast(k),
        @intCast(stride),
        @intCast(pad_l),
        @intCast(pad_r),
        @intCast(dil),
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_conv_t1d(
    h: i64,
    x: i64,
    w: i64,
    b: i64,
    out: i64,
    B: i64,
    c_in: i64,
    l: i64,
    c_out: i64,
    k: i64,
    stride: i64,
    padding: i64,
    output_padding: i64,
    dil: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or c_in < 0 or l < 0 or c_out < 0 or k < 0 or stride < 0 or padding < 0 or output_padding < 0 or dil < 0)
        return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.convTranspose1d(
        @intCast(x),
        @intCast(w),
        @intCast(b),
        @intCast(out),
        @intCast(B),
        @intCast(c_in),
        @intCast(l),
        @intCast(c_out),
        @intCast(k),
        @intCast(stride),
        @intCast(padding),
        @intCast(output_padding),
        @intCast(dil),
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_conv2d(
    h: i64,
    x: i64,
    w: i64,
    b: i64,
    out: i64,
    B: i64,
    c_in: i64,
    h_: i64,
    w_: i64,
    c_out: i64,
    kh: i64,
    kw: i64,
    pad_h: i64,
    pad_w: i64,
    stride_h: i64,
    stride_w: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or c_in < 0 or h_ < 0 or w_ < 0 or c_out < 0 or kh < 0 or kw < 0 or pad_h < 0 or pad_w < 0 or stride_h < 0 or stride_w < 0)
        return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.conv2d(
        @intCast(x),
        @intCast(w),
        @intCast(b),
        @intCast(out),
        @intCast(B),
        @intCast(c_in),
        @intCast(h_),
        @intCast(w_),
        @intCast(c_out),
        @intCast(kh),
        @intCast(kw),
        @intCast(pad_h),
        @intCast(pad_w),
        @intCast(stride_h),
        @intCast(stride_w),
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// T2：out[B,C_out,H_out,W_out] = conv_transpose2d(x[B,C_in,OH,OW],
/// w[C_in,C_out,KH,KW])（PyTorch 转置布局；gx 路径无 bias）。
/// 单发入口，整段 H 轴（in_off=0/h_seg=OH/ho_off=0/H_out_full=H_out）；
/// H_out/W_out 由调用方按 opad 公式给出，engine 校验一致性。
pub export fn rvc_conv_t2d(
    h: i64,
    x: i64,
    w: i64,
    out: i64,
    B: i64,
    c_in: i64,
    oh: i64,
    ow: i64,
    c_out: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    ph: i64,
    pw: i64,
    opad_h: i64,
    opad_w: i64,
    h_out: i64,
    w_out: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or c_in < 0 or oh < 0 or ow < 0 or c_out < 0 or kh < 0 or kw < 0 or sh < 0 or sw < 0 or ph < 0 or pw < 0 or opad_h < 0 or opad_w < 0 or h_out < 0 or w_out < 0)
        return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.convTranspose2d(
        @intCast(x),
        @intCast(w),
        0, // b：gx 路径无 bias → 共享零 buffer
        @intCast(out),
        @intCast(B),
        @intCast(c_in),
        @intCast(oh),
        @intCast(ow),
        @intCast(c_out),
        @intCast(kh),
        @intCast(kw),
        @intCast(sh),
        @intCast(sw),
        @intCast(ph),
        @intCast(pw),
        @intCast(opad_h),
        @intCast(opad_w),
        @intCast(h_out),
        @intCast(w_out),
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// P15b: out[C, 2H-1, 2W-1] = stride-2 zero insertion of x[C, H, W]
/// （convT 的 x_up GPU 内生成；C = B*C_in 合并通道数）。
pub export fn rvc_insert_zeros_2x(h: i64, x: i64, out: i64, c: i64, h_: i64, w: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (c < 0 or h_ < 0 or w < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.insertZeros2x(
        @intCast(x),
        @intCast(out),
        @intCast(c),
        @intCast(h_),
        @intCast(w),
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// out[N, EmbDim] = table[ids[N], :]. `ids` buffer holds N int32 indices
/// (raw bytes; validated non-negative and < TableRows by the caller).
pub export fn rvc_embed(h: i64, ids: i64, table: i64, out: i64, n: i64, table_rows: i64, emb_dim: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0 or table_rows < 0 or emb_dim < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.embed(
        @intCast(ids),
        @intCast(table),
        @intCast(out),
        @intCast(n),
        @intCast(table_rows),
        @intCast(emb_dim),
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_add_inplace(h: i64, a: i64, b: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.addInplace(@intCast(a), @intCast(b), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_mul_inplace(h: i64, a: i64, b: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.mulInplace(@intCast(a), @intCast(b), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// In-place LeakyReLU: a[i] = (a[i]>=0) ? a[i] : slope*a[i] over `n`
/// floats. `slope` is the f32 negative slope; its bit pattern is passed
/// through the i64 argument (any 32-bit pattern is valid).
pub export fn rvc_leaky_relu(h: i64, a: i64, n: i64, slope: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    const slope_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(slope)))));
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.leakyRelu(@intCast(a), @intCast(n), slope_f) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Copy `n` floats: dst[i] = src[i] (dst and src must be distinct buffers).
pub export fn rvc_copy(h: i64, dst: i64, src: i64, n: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.copy(@intCast(dst), @intCast(src), @intCast(n)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_layernorm(
    h: i64,
    x: i64,
    gamma: i64,
    beta: i64,
    out: i64,
    rows: i64,
    cols: i64,
    eps: f64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (rows < 0 or cols < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.layernorm(
        @intCast(x),
        @intCast(gamma),
        @intCast(beta),
        @intCast(out),
        @intCast(rows),
        @intCast(cols),
        eps,
    ) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_softmax(h: i64, x: i64, out: i64, rows: i64, cols: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (rows < 0 or cols < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.softmax(@intCast(x), @intCast(out), @intCast(rows), @intCast(cols)) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

pub export fn rvc_rmsnorm(h: i64, x: i64, gamma: i64, out: i64, rows: i64, cols: i64, eps: f64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (rows < 0 or cols < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.rmsnorm(@intCast(x), @intCast(gamma), @intCast(out), @intCast(rows), @intCast(cols), eps) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

// ── Batch recorder (P1: multi-op single submit) ─────────────────────
//
// rvc_batch_begin / rvc_batch_add / rvc_batch_commit let callers record
// several op calls and submit them as ONE command buffer with ONE
// fence-wait, cutting the per-call fixed overhead (mutex + submit +
// sync wait, 8-25 ms on AMD Radeon Pro VII) to a single occurrence.
// Semantics of every recorded op are identical to its rvc_* twin.

pub export fn rvc_batch_add_conv_t1d_seg(
    h: i64,
    a: i64,
    b: i64,
    c: i64,
    p0: i64,
    p1: i64,
    p2: i64,
    p3: i64,
    p4: i64,
    p5: i64,
    p6: i64,
    p7: i64,
    p8: i64,
    p9: i64,
    // T1.1 分段参数
    l_out: i64,
    l_in_seg: i64,
    in_off: i64,
    lo_off: i64,
    l_out_full: i64,
    // T1.1 子视图：每 binding 字节偏移，-1 = 无偏移（全量）
    vx: i64,
    vw: i64,
    vb: i64,
    vo: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0 or l_out < 0 or l_in_seg < 0 or in_off < 0 or lo_off < 0 or l_out_full < 0) {
        setErrorMsg("invalid conv_t1d seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const offs = [_]usize{
        if (vx < 0) 0 else @intCast(vx),
        if (vw < 0) 0 else @intCast(vw),
        if (vb < 0) 0 else @intCast(vb),
        if (vo < 0) 0 else @intCast(vo),
    };
    eng.batchAddConvT1dView(
        @intCast(a),
        @intCast(b),
        @intCast(c),
        @intCast(p9),
        @intCast(p0),
        @intCast(p1),
        @intCast(p2),
        @intCast(p3),
        @intCast(p4),
        @intCast(p5),
        @intCast(p6),
        @intCast(p7),
        @intCast(p8),
        @intCast(l_out),
        @intCast(l_in_seg),
        @intCast(in_off),
        @intCast(lo_off),
        @intCast(l_out_full),
        &offs,
    ) catch return handleErr();
    return 0;
}

/// T1.1 分段版 conv1d（ResBlock 超限级）：输出段 [lo_off, lo_off+seg_len)。
/// 参数：a=x b=w c=bias(0=none)；p0..p8 维度（B,C_in,L,C_out,K,stride,pad_l,pad_r,dil），
/// p9=out；seg_len 段输出列数，lo_off 段起始绝对列，l_out_full 输出行全长。
pub export fn rvc_batch_add_conv1d_seg(
    h: i64,
    a: i64,
    b: i64,
    c: i64,
    p0: i64,
    p1: i64,
    p2: i64,
    p3: i64,
    p4: i64,
    p5: i64,
    p6: i64,
    p7: i64,
    p8: i64,
    p9: i64,
    seg_len: i64,
    lo_off: i64,
    l_out_full: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p8 < 0 or seg_len < 0 or lo_off < 0 or l_out_full < 0) {
        setErrorMsg("invalid conv1d seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchAddConv1dView(
        @intCast(a),
        @intCast(b),
        @intCast(c),
        @intCast(p9),
        @intCast(p0),
        @intCast(p1),
        @intCast(p2),
        @intCast(p3),
        @intCast(p4),
        @intCast(p5),
        @intCast(p6),
        @intCast(p7),
        @intCast(p8),
        @intCast(seg_len),
        @intCast(lo_off),
        @intCast(l_out_full),
    ) catch return handleErr();
    return 0;
}

/// T1.1 elementwise 分段版：copy 子段 —— dst[off..off+n) = src[off..off+n)
/// （n/a_off/b_off 为**元素**计数/偏移；engine 内部 ×4 字节做子视图绑定）。
/// shader 的 gid 相对段内，与整段 copy 同 kernel 同 push → 逐位一致。
pub export fn rvc_batch_add_copy_seg(
    h: i64,
    a: i64,
    b: i64,
    n: i64,
    a_off: i64,
    b_off: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0 or a_off < 0 or b_off < 0) {
        setErrorMsg("invalid copy seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchAddCopyView(
        @intCast(a),
        @intCast(b),
        @intCast(n),
        @intCast(a_off),
        @intCast(b_off),
    ) catch |e| { setErrorName(e); return -1; };
    return 0;
}

/// T1.1 elementwise 分段版：就地加子段 —— a[off..off+n) += b[off..off+n)。
pub export fn rvc_batch_add_add_seg(
    h: i64,
    a: i64,
    b: i64,
    n: i64,
    a_off: i64,
    b_off: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0 or a_off < 0 or b_off < 0) {
        setErrorMsg("invalid add seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchAddAddInplaceView(
        @intCast(a),
        @intCast(b),
        @intCast(n),
        @intCast(a_off),
        @intCast(b_off),
    ) catch |e| { setErrorName(e); return -1; };
    return 0;
}

/// T1.1 elementwise 分段版：就地乘子段 —— a[off..off+n) *= b[off..off+n)。
pub export fn rvc_batch_add_mul_seg(
    h: i64,
    a: i64,
    b: i64,
    n: i64,
    a_off: i64,
    b_off: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0 or a_off < 0 or b_off < 0) {
        setErrorMsg("invalid mul seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchAddMulInplaceView(
        @intCast(a),
        @intCast(b),
        @intCast(n),
        @intCast(a_off),
        @intCast(b_off),
    ) catch |e| { setErrorName(e); return -1; };
    return 0;
}

/// T1.1 elementwise 分段版：就地 LeakyReLU 子段 —— a[off..off+n) = lrelu(a)。
/// slope 为 f32 位模式（i64 传入，engine 内 bitCast）。
pub export fn rvc_batch_add_leaky_seg(
    h: i64,
    a: i64,
    n: i64,
    a_off: i64,
    slope: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0 or a_off < 0) {
        setErrorMsg("invalid leaky seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const slope_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(slope)))));
    eng.batchAddLeakyReluView(
        @intCast(a),
        @intCast(n),
        slope_f,
        @intCast(a_off),
    ) catch |e| { setErrorName(e); return -1; };
    return 0;
}

/// T1.1 elementwise 分段版：就地 GELU 子段 —— a[off..off+n) = gelu(a)。
/// 同整段 gelu 同 kernel 同 push（exact erf，float32 A&S），逐位一致。
pub export fn rvc_batch_add_gelu_seg(
    h: i64,
    a: i64,
    n: i64,
    a_off: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (n < 0 or a_off < 0) {
        setErrorMsg("invalid gelu seg dims");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchAddGeluView(
        @intCast(a),
        @intCast(n),
        @intCast(a_off),
    ) catch |e| { setErrorName(e); return -1; };
    return 0;
}

/// Reset the pending batch (discards uncommitted ops). 0 = ok.
pub export fn rvc_batch_begin(h: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchBegin();
    return 0;
}

/// Record one op into the current batch; nothing is submitted until
/// rvc_batch_commit. op codes and parameter layout:
///
///   1 = matmul       a,b,c = a,b,c;          p0=M p1=K p2=N
///   2 = conv1d       a=x b=w c=bias(0=none); p0=B p1=C_in p2=L p3=C_out
///                     p4=K p5=stride p6=pad_l p7=pad_r p8=dil p9=out
///   3 = add_inplace  a,b;                    p0=n
///   4 = mul_inplace  a,b;                    p0=n
///   5 = conv_t1d     a=x b=w c=bias(0=none); p0=B p1=C_in p2=L p3=C_out
///                     p4=K p5=stride p6=padding p7=output_padding
///                     p8=dil p9=out           (w = [C_in, C_out, K])
///   6 = leaky_relu   a (in-place);           p0=n p1=slope(f32 位模式)
///   7 = copy         a=dst b=src;            p0=n
///   8 = softmax      a=x c=out;              p0=rows p1=cols
///   9 = layer_norm   a=x b=gamma c=beta;     p0=rows p1=cols p2=eps(f32 位)
///                     p9=out
///  10 = gelu         a (in-place, erf);      p0=n
///  11 = bias_add     a b=bias c=out;         p0=n p1=cols
///  12 = attn_qk      a=q b=k c=out;          p0=H p1=T p2=D p3=C
///                     out[H,T,T] = q_h . k_h^T（q/k 为 [T,C] 头交错）
///  13 = attn_sv      a=attnW b=v c=out;      p0=H p1=T p2=D p3=C
///                     out[T,C] = attnW . v_h（头合并回交错布局）
///  14 = relu         a (in-place);           p0=n
///  15 = group_norm   a=x b=gamma c=beta;     p0=G p1=Cpg p2=S p3=eps(f32 位)
///                     p9=out
///  16 = banded_attn_qk  a=q b=k c=used;      p0=H p1=T p2=D p3=C p9=out
///                     out[H,T,T] = q_h . (k_h + used[(s-t+T-1)])^T
///                     （T5 相对位置折入，D5）
///  17 = banded_attn_sv  a=attnW b=v c=used;  p0=H p1=T p2=D p3=C p9=out
///                     out[T,C] = attnW . (v + used[(s-t+T-1)])（D5）
///  19 = transpose_tensor a=src c=out;        p0=C p1=P p2=scale f32 位模式
///                     out[P,C] = src[C,P] * scale（tiled 16x16，T16）
///  30 = sqrt_inplace  a (in-place);          p0=n
///  31 = div_const     a (in-place);          p0=n p1=s(f32 位模式)
///  32 = mul_const     a (in-place);          p0=n p1=s(f32 位模式)
///  33 = madd_const    a (in-place) b=向量;   p0=n p1=s(f32 位模式)
///                     a[i] += s * b[i]
///  34 = sub_inplace   a (in-place) b=向量;   p0=n      a[i] -= b[i]
///  35 = add_const     a (in-place);          p0=n p1=s(f32 位模式)
///  36 = mul_buf_scalar a (in-place) b=1元素; p0=n      a[i] *= b[0]
///  37 = rcp_inplace    a (in-place);          p0=n      a[i] = 1/a[i]
///                     （T4-2 AdamW 标量元素算子；30-37 全部就地 n 元素一维）
pub export fn rvc_batch_add(
    h: i64,
    op: i32,
    a: i64,
    b: i64,
    c: i64,
    p0: i64,
    p1: i64,
    p2: i64,
    p3: i64,
    p4: i64,
    p5: i64,
    p6: i64,
    p7: i64,
    p8: i64,
    p9: i64,
    p10: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (op < 1 or op > 42) {
        setErrorMsg("invalid batch op code (1..42: gating_bwd activation_bwd activation_fwd mul_bwd slice_bwd) matmul conv1d add_inplace mul_inplace conv_t1d leaky_relu copy softmax layer_norm gelu bias_add attn_qk attn_sv relu group_norm banded_attn_qk banded_attn_sv gating transpose transpose_b reduce_rows leaky_bwd conv1d_groups conv1d_groups_bwd_gx conv1d_groups_bwd_gw conv1d_groups_bwd_gb conv2d conv_t2d im2col_2d sqrt_inplace rcp_inplace div_const mul_const madd_const sub_inplace add_const mul_buf_scalar)");
        return -1;
    }
    eng.mutex.lock();
    defer eng.mutex.unlock();
    var p = [11]i64{ p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 };
    eng.batchAddOp(op, a, b, c, 0, &p) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Submit every recorded op as ONE command buffer + ONE fence-wait.
/// 0 = ok; -1 on failure (batch is left intact for discard/retry).
pub export fn rvc_batch_commit(h: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchCommit() catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Submit every recorded op as ONE command buffer WITHOUT waiting for
/// the GPU (async commit). The submission is parked in an in-flight
/// frame; results are ready after rvc_batch_wait. Multiple async
/// commits pipeline on the queue (executed in order), so callers can
/// submit several batches and wait once — removing the per-commit
/// submit+wait round-trip. Buffers used by an in-flight submission must
/// not be freed before rvc_batch_wait (rvc_mem_free drains the queue
/// itself, so it is always safe). 0 = ok; -1 on failure.
pub export fn rvc_batch_commit_async(h: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchCommitAsync() catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Block until every in-flight async submission has completed. No-op
/// when nothing is in flight. All GPU writes are visible to the host
/// afterwards. 0 = ok.
pub export fn rvc_batch_wait(h: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchWait() catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Discard the pending batch without submitting (idempotent). 0 = ok.
pub export fn rvc_batch_discard(h: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.batchDiscard();
    return 0;
}

// ── Training-graph executor (T2 POC) ──────────────────────────────────
// Graph handle = heap pointer as i64 (same convention as engine handle).
// The Graph owns a deep copy of the node table; Python owns the buffers
// referenced by the node's a/b/c/p9 slots (upload/allocate/free), so the
// executor performs no GPU memory management of its own.

fn graphFromHandle(h: i64) ?*graph.Graph {
    if (h <= 0) return null;
    return @alignCast(@as(*graph.Graph, @ptrFromInt(@as(usize, @bitCast(h)))));
}

/// Compile a static training graph from `nodes` (deep copy). Returns the
/// graph handle, or -1 on failure (error text in rvc_last_error).
/// `n_nodes` must be > 0. Op codes and parameter layout match
/// rvc_batch_add exactly; each node is one dispatch.
pub export fn rvc_graph_create(h: i64, nodes_ptr: [*]const graph.GraphNode, n_nodes: i64) i64 {
    const eng = engineFromHandle(h) orelse {
        setErrorMsg("invalid engine handle");
        return -1;
    };
    if (n_nodes <= 0) {
        setErrorMsg("empty graph");
        return -1;
    }
    const nodes = nodes_ptr[0..@as(usize, @intCast(n_nodes))];
    const g = graph.Graph.create(std.heap.c_allocator, eng, nodes) catch |e| {
        setErrorName(e);
        return -1;
    };
    return @intCast(@intFromPtr(g));
}

/// Run the whole graph as ONE batch (begin → N×add → commit). `async_`
/// nonzero uses commit_async (pipeline; results after rvc_graph_wait).
/// 0 = ok; -1 on failure (batch discarded on error).
pub export fn rvc_graph_run(g: i64, async_: i32) i32 {
    const gr = graphFromHandle(g) orelse return handleErr();
    gr.eng.mutex.lock();
    defer gr.eng.mutex.unlock();
    gr.run(async_ != 0) catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Block until all in-flight async graph submissions complete
/// (engine-wide wait; no-op when nothing in flight). 0 = ok.
pub export fn rvc_graph_wait(g: i64) i32 {
    const gr = graphFromHandle(g) orelse return handleErr();
    gr.eng.mutex.lock();
    defer gr.eng.mutex.unlock();
    gr.wait() catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// Discard the graph and free its node table. The engine and Python-owned
/// buffers are untouched. 0 = ok.
pub export fn rvc_graph_destroy(g: i64) i32 {
    const gr = graphFromHandle(g) orelse return handleErr();
    gr.destroy();
    return 0;
}

/// Copy the current device name into `buf` (NUL-terminated). Requires a
/// live engine (rvc_engine_create must have been called).
pub export fn rvc_device_name(buf: [*]u8, buf_size: i64) i32 {
    if (buf_size <= 0) return -1;
    const out = buf[0..@as(usize, @intCast(buf_size))];
    g_engine_mutex.lock();
    const h = g_engine;
    g_engine_mutex.unlock();
    const eng = if (h) |hh| engineFromHandle(hh) else null;
    if (eng) |e| {
        e.mutex.lock();
        defer e.mutex.unlock();
        const name = e.deviceName();
        const name_len = std.mem.len(name);
        const n = @min(name_len, out.len - 1);
        @memcpy(out[0..n], name[0..n]);
        out[n] = 0;
        return 0;
    }
    setErrorMsg("no engine created");
    return -1;
}

/// Copy the last error message into `buf` (NUL-terminated). Always
/// succeeds; empty string when no error has been recorded.
pub export fn rvc_last_error(buf: [*]u8, buf_size: i64) i32 {
    if (buf_size <= 0) return -1;
    const out = buf[0..@as(usize, @intCast(buf_size))];
    g_err_mutex.lock();
    defer g_err_mutex.unlock();
    const n = @min(g_err_len, out.len - 1);
    @memcpy(out[0..n], g_err_buf[0..n]);
    out[n] = 0;
    return 0;
}

/// RVC_TS=1 measure-only: return accumulated GPU-busy nanoseconds and
/// submit count since engine creation (busy_ns_out / submits_out).
/// Returns 0 with zeros when RVC_TS is disabled.
pub export fn rvc_ts_stats(h: i64, busy_ns_out: *i64, submits_out: *i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const st = eng.tsStats();
    busy_ns_out.* = @intCast(st.busy_ns);
    submits_out.* = @intCast(st.submits);
    return 0;
}

/// RVC_TS=1 measure-only: reset the accumulated busy/submit counters
/// (call before a timed window to get a delta).
pub export fn rvc_ts_reset(h: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    eng.tsReset();
    return 0;
}

/// T8: read-only VRAM stats — suballocator chunk total/count, free list
/// bytes/blocks, live buffer count/bytes, persistent staging sizes,
/// direct-alloc (non-suballocator) bytes, largest live buffer.
pub export fn rvc_mem_stats(
    h: i64,
    sub_total_out: *i64,
    sub_chunks_out: *i64,
    sub_free_bytes_out: *i64,
    sub_free_blocks_out: *i64,
    buf_count_out: *i64,
    buf_bytes_out: *i64,
    staging_up_out: *i64,
    staging_dn_out: *i64,
    direct_bytes_out: *i64,
    max_buf_bytes_out: *i64,
    max_buf_sub_out: *i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const s = eng.memStats();
    sub_total_out.* = @intCast(s.sub_total);
    sub_chunks_out.* = @intCast(s.sub_chunks);
    sub_free_bytes_out.* = @intCast(s.sub_free_bytes);
    sub_free_blocks_out.* = @intCast(s.sub_free_blocks);
    buf_count_out.* = @intCast(s.buf_count);
    buf_bytes_out.* = @intCast(s.buf_bytes);
    staging_up_out.* = @intCast(s.staging_up);
    staging_dn_out.* = @intCast(s.staging_dn);
    direct_bytes_out.* = @intCast(s.direct_bytes);
    max_buf_bytes_out.* = @intCast(s.max_buf_bytes);
    max_buf_sub_out.* = @intCast(s.max_buf_sub);
    return 0;
}

/// T8: dump top-N live buffers as (bytes, sub, memory) triples.
/// out must have room for n*3 i64; returns count of triples written.
pub export fn rvc_mem_top(
    h: i64,
    n: i64,
    out: [*]i64,
) i64 {
    const eng = engineFromHandle(h) orelse return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const want: usize = @intCast(@max(0, n));
    const cnt = eng.memTop(want, out[0 .. want * 3]);
    return @intCast(cnt);
}

/// Record a conv2d into the current batch (P0-1: rmvpe UNet conv2d GPU
/// batch化——用户要求小算子也必须真正走 GPU)。
pub export fn rvc_batch_add_conv2d(
    h: i64,
    x: i64,
    w: i64,
    b: i64,
    out: i64,
    B: i64,
    c_in: i64,
    h_: i64,
    w_: i64,
    c_out: i64,
    kh: i64,
    kw: i64,
    pad_h: i64,
    pad_w: i64,
    stride_h: i64,
    stride_w: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or c_in < 0 or h_ < 0 or w_ < 0 or c_out < 0 or kh < 0 or kw < 0 or
        pad_h < 0 or pad_w < 0 or stride_h < 0 or stride_w < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.batchAddConv2d(
        @intCast(x),
        @intCast(w),
        @intCast(b),
        @intCast(out),
        @intCast(B),
        @intCast(c_in),
        @intCast(h_),
        @intCast(w_),
        @intCast(c_out),
        @intCast(kh),
        @intCast(kw),
        @intCast(pad_h),
        @intCast(pad_w),
        @intCast(stride_h),
        @intCast(stride_w),
    );
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// T2 续（阶段A A4 v2）：batch conv_transpose2d（反向 gx 路径批量提交；
/// engine.batchAddConvT2d 已实现，Python 侧此前未接）。权重布局
/// [C_in, C_out, KH, KW]（PyTorch）；opad 须 < stride（引擎校验）。
pub export fn rvc_batch_add_conv_t2d(
    h: i64,
    x: i64,
    w: i64,
    b: i64,
    out: i64,
    B: i64,
    c_in: i64,
    oh: i64,
    ow: i64,
    c_out: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    ph: i64,
    pw: i64,
    opad_h: i64,
    opad_w: i64,
    h_out: i64,
    w_out: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or c_in < 0 or oh < 0 or ow < 0 or c_out < 0 or kh < 0 or kw < 0 or
        sh < 0 or sw < 0 or ph < 0 or pw < 0 or opad_h < 0 or opad_w < 0 or
        h_out < 0 or w_out < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.batchAddConvT2d(
        @intCast(x),
        @intCast(w),
        @intCast(b),
        @intCast(out),
        @intCast(B),
        @intCast(c_in),
        @intCast(oh),
        @intCast(ow),
        @intCast(c_out),
        @intCast(kh),
        @intCast(kw),
        @intCast(sh),
        @intCast(sw),
        @intCast(ph),
        @intCast(pw),
        @intCast(opad_h),
        @intCast(opad_w),
        @intCast(h_out),
        @intCast(w_out),
    );
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// 阶段E（C3）：batch im2col_1d（GPU gather；conv1d backward gw / forward）。
/// J10：新增 dilation（间隔窗 gather，此前仅 dilation=1）。
pub export fn rvc_batch_add_im2col_1d(
    h: i64,
    x: i64,
    out: i64,
    B: i64,
    C: i64,
    T: i64,
    oL: i64,
    K_dil: i64,
    stride: i64,
    pad_l: i64,
    dilation: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or C < 0 or T < 0 or oL < 0 or K_dil < 0 or stride < 0 or pad_l < 0 or dilation < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.batchAddIm2Col1d(
        @intCast(x),
        @intCast(out),
        @intCast(B),
        @intCast(C),
        @intCast(T),
        @intCast(oL),
        @intCast(K_dil),
        @intCast(stride),
        @intCast(pad_l),
        @intCast(dilation),
    );
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// 阶段E（C3 v2）：batch im2col_2d（GPU gather；conv2d backward gw / forward）。
pub export fn rvc_batch_add_im2col_2d(
    h: i64,
    x: i64,
    out: i64,
    B: i64,
    C: i64,
    H: i64,
    W: i64,
    OH: i64,
    OW: i64,
    KH: i64,
    KW: i64,
    sh: i64,
    sw: i64,
    ph: i64,
    pw: i64,
) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (B < 0 or C < 0 or H < 0 or W < 0 or OH < 0 or OW < 0 or
        KH < 0 or KW < 0 or sh < 0 or sw < 0 or ph < 0 or pw < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.batchAddIm2Col2d(
        @intCast(x), @intCast(out),
        @intCast(B), @intCast(C), @intCast(H), @intCast(W),
        @intCast(OH), @intCast(OW), @intCast(KH), @intCast(KW),
        @intCast(sh), @intCast(sw), @intCast(ph), @intCast(pw),
    );
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}
/// （fwd/rev 半区）、b=w_hh [2][3H,H]、c=b_hh [2][3H]、out=[T,2H]
/// （前 H=fwd、后 H=rev）。grid=(1,1,2) 两方向并行，kernel 内翻转。
pub export fn rvc_batch_add_gru(h: i64, a: i64, b: i64, c: i64, out: i64, t: i64, hdim: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (t < 0 or hdim < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.batchAddGru(
        @intCast(a), @intCast(b), @intCast(c), @intCast(out),
        @intCast(t), @intCast(hdim),
    );
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// T18：GRU 单 kernel 单发路径（独立 submit，避开 batch 状态机）。
pub export fn rvc_gru(h: i64, a: i64, b: i64, c: i64, out: i64, t: i64, hdim: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (t < 0 or hdim < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.gru(
        @intCast(a), @intCast(b), @intCast(c), @intCast(out),
        @intCast(t), @intCast(hdim),
    );
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// T18 诊断：纯 barrier 循环（步间同步成本）。a=out [1]，grid=(1,1,1)。
pub export fn rvc_batch_add_gru_sync(h: i64, a: i64, t: i64, hdim: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (t < 0 or hdim < 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
    const rc = eng.batchAddGruSync(@intCast(a), @intCast(t), @intCast(hdim));
    rc catch |e| {
        setErrorName(e);
        return -1;
    };
    return 0;
}

/// P0-3：探测设备特性/扩展支持（shaderFloat16、cooperative matrix、
/// shader_float16_int8），写回 "name=0|1,..."（NUL 结尾）。
pub export fn rvc_probe_extensions(h: i64, buf: [*]u8, buf_size: i64) i32 {
    const eng = engineFromHandle(h) orelse return handleErr();
    if (buf_size <= 0) return -1;
    eng.mutex.lock();
    defer eng.mutex.unlock();
        // shaderFloat16 由 VK_KHR_shader_float16_int8 扩展提供（旧 cimport 的
    // VkPhysicalDeviceFeatures 缺 shaderFloat16 字段，故用扩展等价判断）。
    var n: u32 = 0;
    _ = vk.c.vkEnumerateDeviceExtensionProperties(eng.ctx.physical_device, null, &n, null);
    var coop: u32 = 0;
    var f16i8: u32 = 0;
    if (n > 0) {
        var cap: u32 = @min(n, 256);
        var exts: [256]vk.c.VkExtensionProperties = undefined;
        _ = vk.c.vkEnumerateDeviceExtensionProperties(eng.ctx.physical_device, null, &cap, &exts);
        for (exts[0..cap]) |e| {
            const nm = std.mem.sliceTo(&e.extensionName, 0);
            if (std.mem.eql(u8, nm, "VK_KHR_cooperative_matrix")) coop = 1;
            if (std.mem.eql(u8, nm, "VK_KHR_shader_float16_int8")) f16i8 = 1;
        }
    }
    const shader16: u32 = f16i8;  // shaderFloat16 由该扩展提供
    const out = buf[0..@as(usize, @intCast(buf_size))];
    const s = std.fmt.bufPrint(out,
        "shaderFloat16={d},VK_KHR_cooperative_matrix={d},VK_KHR_shader_float16_int8={d}",
        .{ shader16, coop, f16i8 }) catch return -1;
    _ = s;
    return 0;
}
