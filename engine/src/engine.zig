//! Engine aggregate: Vulkan context + 13 compute kernels + recorder +
//! buffer registry. One `Engine` per `rvc_engine_create` handle.
//!
//! Threading contract: the FFI layer serialises every exported call
//! through `Engine.mutex`, so none of the methods here lock internally.
//! Every op submits one dispatch and waits for the fence before
//! returning — no in-flight work survives across calls.

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const pipeline = @import("pipeline.zig");
const recorder = @import("recorder.zig");
const shaders = @import("shaders");

const MatMulPush = extern struct { m: u32, k: u32, n: u32, tile: u32 };
const ElemPush = extern struct { n: u32, base_off: u32 = 0 };
/// In-place LeakyReLU push constants: element count + float slope (the
/// f32 bit pattern is passed through the u32 slot and reinterpretted by
/// the shader's `float slope` member — same 8-byte layout).
const LeakyPush = extern struct { n: u32, slope: f32 };
/// conv1d_groups 反向（DiscriminatorS bp）——3 个 dispatch 模式（mode 0=gx
/// 1=gw 2=gb），每次一个输出 buffer（BatchEntry 绑定上限 4）。字段序与
/// shaders/conv1d_groups_bwd.comp 的 12 u32 布局逐一对齐（J24 加 dilation）。
const Conv1dGroupsBwdPush = extern struct {
    b: u32,
    c_in: u32,
    c_in_g: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    pad_l: u32,
    pad_r: u32, // 校验用
    dilation: u32,
    l_out: u32,
    mode: u32,
};
/// conv1d_groups（分组 1D 卷积 fwd，DiscriminatorS）push constants —— 字段序
/// 必须与 shaders/conv1d_groups.comp 的 11 u32 布局逐一对齐。w 布局
/// [C_out, C_in_g, K]（C_in_g = C_in/groups）；组 g = co/(C_out/groups)，
/// 输入通道段 = g*C_in_g；dilation 恒 1（判别器 S 无 dilation，kernel 内固定）。
const Conv1dGroupsPush = extern struct {
    b: u32,
    c_in: u32,
    c_in_g: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    pad_l: u32,
    pad_r: u32, // 校验用；shader 对称 pad（pad_l）
    l_out: u32,
    has_bias: u32,
};
const Conv1dPush = extern struct {
    b: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    pad_l: u32,
    dil: u32,
    l_out: u32,
    has_bias: u32,
    // T1.1 分段（整段时保持 0/L_out → shader 索引不变，零回归）
    lo_off: u32 = 0,
    l_out_full: u32 = 0,
};
const ConvT1dPush = extern struct {
    b: u32,
    c_in: u32,
    l: u32,
    c_out: u32,
    k: u32,
    stride: u32,
    padding: u32,
    output_padding: u32,
    dil: u32,
    l_out: u32,
    has_bias: u32,
    // T1.1 分段（整段时保持 0/L/0/L_out → shader 索引不变，零回归）
    in_off: u32 = 0,
    l_seg: u32 = 0,
    lo_off: u32 = 0,
    l_out_full: u32 = 0,
};
const Conv2dPush = extern struct {
    b: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    kh: u32,
    kw: u32,
    pad_h: u32,
    pad_w: u32,
    stride_h: u32,
    stride_w: u32,
    oh: u32,
    ow: u32,
    has_bias: u32,
};
/// conv_t2d（转置 2D 卷积，反向 gx 路径）push constants —— 字段序必须与
/// shaders/conv_t2d.comp 的 20 u32 布局逐一对齐（16 主字段 + T2 T1.1 风格
/// H 轴分段 4 字段；整段时 in_off=0/h_seg=OH/ho_off=0/h_out_full=H_out）。
/// 输入 x=go 形状 [B, C_in, OH, OW]，权重 w 为 PyTorch 布局 [C_in, C_out,
/// KH, KW]（前向 w[O,C,KH,KW] 在 host 转置），输出 [B, C_out, H_out,
/// W_out]。has_bias 恒 0（gx 无 bias），binding 2 绑零 buffer 保持布局对齐。
const ConvT2dPush = extern struct {
    b: u32,
    c_in: u32,
    oh: u32,
    ow: u32,
    c_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    opad_h: u32,
    opad_w: u32,
    h_out: u32,
    w_out: u32,
    has_bias: u32,
    // T2 分段（整段时保持 0/OH/0/H_out → shader 索引不变，零回归）
    in_off: u32 = 0,
    h_seg: u32 = 0,
    ho_off: u32 = 0,
    h_out_full: u32 = 0,
};
const EmbedPush = extern struct { n: u32, emb_dim: u32 };
/// P15b：stride-2 插零上采样 push constants（C 为合并通道数 B*C_in）。
const InsertZeros2xPush = extern struct { c: u32, h: u32, w: u32 };
/// 阶段E（C3）：GPU im2col（conv1d backward gw / forward 组装）。
/// xw[B*oL, C*K_dil] row-major，dilation=1（调用方回退 host）。
const Im2col1dPush = extern struct {
    B: u32,
    C: u32,
    T: u32,
    oL: u32,
    K_dil: u32,
    stride: u32,
    pad_l: u32,
    dilation: u32, // J10：支持 dilation≠1（gather 间隔窗，此前仅 dilation=1）
};
/// 阶段E（C3 v2）：GPU im2col 2D（conv2d backward gw / forward 组装）。
/// xw[B*OH*OW, C*KH*KW] row-major（行=(b,oh,ow) 行内=(c,kh,kw)），dilation=1。
const Im2col2dPush = extern struct {
    B: u32,
    C: u32,
    H: u32,
    W: u32,
    OH: u32,
    OW: u32,
    KH: u32,
    KW: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
};
const NormPush = extern struct { rows: u32, cols: u32, eps: f32 };
const SoftmaxPush = extern struct { rows: u32, cols: u32 };
const BiasAddPush = extern struct { n: u32, cols: u32 };
const AttnPush = extern struct { H: u32, T: u32, D: u32, C: u32 };
const GnPush = extern struct { G: u32, Cpg: u32, S: u32, eps: f32 };
const GatingPush = extern struct { n: u32, lg: u32, h: u32, off: u32 };
/// T16：tiled GPU transpose push constants（C/P 为源矩阵 [C,P] 的维度；
/// scale 折叠到写回（enc q 路径的 1/sqrt(kc)），纯转置传 1.0）。
const TransposePush = extern struct { c: u32, p: u32, scale: f32 };
/// T-H7：backward gw 批量转置 [B,C,P] → [C,B*P]（b 并入内层；直读 gather kernel）。
const TransposeBPush = extern struct { b: u32, c: u32, p: u32 };
/// T-H7：行归约求和 out[m] = Σ_n src[m,n]（conv bias 梯度用）。
const ReducePush = extern struct { m: u32, n: u32 };
/// T-H7：LeakyReLU 反向 dst[i] = go[i] * (x[i]>=0 ? 1 : slope)（slope 位模式）。
const LeakyBwdPush = extern struct { n: u32, slope_bits: u32 };
/// activation 反向（mode 0=sigmoid 1=tanh）：dst = go * f'(out)。
const ActivationBwdPush = extern struct { n: u32, mode: u32 };
/// activation 前向（mode 0=sigmoid 1=tanh）：out = f(x)。
const ActivationFwdPush = extern struct { n: u32, mode: u32 };
/// mul 反向（mode 0=ga=go*b 1=gb=go*a）。
const MulBwdPush = extern struct { n: u32, mode: u32 };
/// axis=1 slice fwd/bwd（B, C_in, C_out, T, start, mode）。
const SliceBwdPush = extern struct { B: u32, C_in: u32, C_out: u32, T: u32, start: u32, mode: u32 };
/// gating 反向（op42）：push 与 GatingPush 同布局（n, lg, h, off）。
const GatingBwdPush = extern struct { n: u32, lg: u32, h: u32, off: u32 };
/// T18：GRU 单 kernel push constants（T 帧数、H 单向隐层；双向由 grid z
/// 选择，fwd/rev 各自独立权重 buffer，一 dispatch 并行两方向）。
const GruPush = extern struct { T: u32, H: u32 };

/// Push-constant payload of one recorded batch op. Extern union sized for
/// the largest member (ConvT2dPush = 20 u32); the per-op member sits at
/// offset 0 exactly like the one-shot path, and the recorder only copies
/// `kern.push_bytes` (matmul=16, elem=4, conv1d=40, conv_t1d=56, conv2d=56,
/// conv_t2d=80) so the bytes pushed to the GPU are bit-identical to a
/// single rvc_* call.
const BatchPush = extern union {
    matmul: MatMulPush,
    elem: ElemPush,
    conv1d: Conv1dPush,
    conv1d_groups: Conv1dGroupsPush,
    conv1d_groups_bwd: Conv1dGroupsBwdPush,
    conv_t1d: ConvT1dPush,
    conv2d: Conv2dPush,
    conv_t2d: ConvT2dPush,
    leaky: LeakyPush,
    norm: NormPush,
    softmax: SoftmaxPush,
    bias_add: BiasAddPush,
    im2col_1d: Im2col1dPush,
    im2col_2d: Im2col2dPush,
    attn: AttnPush,
    gn: GnPush,
    gating: GatingPush,
    transpose: TransposePush,
    transpose_b: TransposeBPush,
    reduce: ReducePush,
    leaky_bwd: LeakyBwdPush,
    activation_bwd: ActivationBwdPush,
    activation_fwd: ActivationFwdPush,
    mul_bwd: MulBwdPush,
    slice_bwd: SliceBwdPush,
    gating_bwd: GatingBwdPush,
    gru: GruPush,
};

/// One recorded dispatch in a batch: kernel + resolved buffer pointers +
/// push constants + grid. Buffers are resolved at record time (like the
/// one-shot path), so a buffer freed between add and commit fails the
/// commit — matching single-call semantics.
pub const BatchEntry = struct {
    kern: *const pipeline.Kernel,
    bufs: [4]*const buffer.Buffer,
    nbufs: u8,
    /// Binding index of the output (written) buffer — recorder's
    /// dependency tracker uses it to separate reads from writes.
    write_idx: u8,
    push: BatchPush,
    gx: u32,
    gy: u32,
    gz: u32,
    /// Per-binding byte offsets (T1.1 buffer subview); all-zero = whole
    /// buffer (default, zero regression). Passed to dispatchView.
    view_offs: [4]usize = .{ 0, 0, 0, 0 },
};

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// D2（性能攻坚）：驱动实测 maxComputeWorkGroupCount = 2^32-1（AMD
/// Radeon Pro VII，vulkaninfo；Vulkan 规范只保证最小 65535）。旧 guard
/// 按 65535 把点数/空间上限压在 16.78M（conv_t1d/embed）与 2.1M
/// （conv2d），迫使长音频 dec/rmvpe 的 38s 级算子分段（每段一次
/// Python+ctypes dispatch + 上传下载同步，dec GPU busy 仅 23%）。
/// D2 按驱动实际上限放宽：workgroup 组数 = ceilDiv(点数, 256 线程 或
/// tile) ≤ 点数 ≤ u32 max，恒在驱动组上限 2^32-1 内——故 guard 退化为
/// 纯 u32 溢出防护（保护后面 @intCast 与 push 常量），实际分段与否由
/// Python 侧 _GRID_POINTS_MAX=2^31（更保守，留 2x 余量）控制。
const MAX_GRID_COUNT: u32 = 0xFFFF_FFFF; // 驱动 maxComputeWorkGroupCount（每分量）

/// Pick the matmul tile edge (16/32/64) for a given shape. The shader's
/// workgroup is fixed at 256 threads with TILE/16 register blocking per
/// thread; the tile choice balances workgroup count (parallelism for
/// small matrices) against shared-memory traffic (big matrices want
/// deeper register blocking). Thresholds tuned on Radeon Pro VII:
///   wg(64) >= 128 -> 64 (big shapes: deep 4x4 blocking wins big)
///   wg(32) >= 64  -> 32 (mid shapes: 2x2 blocking)
///   else          -> 16 (small shapes: ~16x16, 1 output/thread)
fn pickTile(m: u32, n: u32) u32 {
    // Dev override: RVC_MATMUL_TILE=16|32|64 forces one path (bench).
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_MATMUL_TILE")) |v| {
        defer std.heap.page_allocator.free(v);
        if (std.fmt.parseInt(u32, v, 10)) |t| {
            if (t == 16 or t == 32 or t == 64) return t;
        } else |_| {}
    } else |_| {}
    const wg64 = ceilDiv(m, 64) * ceilDiv(n, 64);
    if (wg64 >= 128) return 64;
    const wg32 = ceilDiv(m, 32) * ceilDiv(n, 32);
    if (wg32 >= 64) return 32;
    return 16;
}

/// Pick the conv1d tile edge (16/32/64) for a given (C_out, L_out)
/// output shape, mirroring pickTile. The conv1d shader is an implicit
/// GEMM over C_in*K with a TILE x TILE (Co x Lo) output tile per
/// workgroup. Tuned on Radeon Pro VII via RVC_CONV1D_TILE:
///   TILE=64 needs enough workgroups (>=400) AND a Co dimension that
///     fills the tile (>=64), else the Co-direction waste / low
///     concurrency loses to shallower blocking (measured: [32,51200]
///     and [512,1000] are ~25% faster on TILE=32).
///   TILE=32 needs >=64 workgroups; below that TILE=16 (1 out/thread).
fn pickConvTile(c_out: u32, l_out: u32) u32 {
    // Dev override: RVC_CONV1D_TILE=16|32|64 forces one path (bench).
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_CONV1D_TILE")) |v| {
        defer std.heap.page_allocator.free(v);
        if (std.fmt.parseInt(u32, v, 10)) |t| {
            if (t == 16 or t == 32 or t == 64) return t;
        } else |_| {}
    } else |_| {}
    const wg64 = ceilDiv(c_out, 64) * ceilDiv(l_out, 64);
    if (c_out >= 64 and wg64 >= 400) return 64;
    const wg32 = ceilDiv(c_out, 32) * ceilDiv(l_out, 32);
    if (wg32 >= 64) return 32;
    return 16;
}

/// Pick the conv2d tile edge (16/32/64) for a given (C_out, OH*OW) output
/// shape, mirroring pickConvTile. conv2d is an implicit GEMM over
/// C_in*KH*KW with a TILE x TILE (Co x OL) output tile per workgroup.
/// Tuned on Radeon Pro VII: TILE=64 needs enough workgroups AND a Co
/// dimension that fills the tile; small shapes prefer 16 (1 out/thread).
fn pickConv2dTile(c_out: u32, ol: u32) u32 {
    // Dev override: RVC_CONV2D_TILE=16|32|64 forces one path (bench).
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_CONV2D_TILE")) |v| {
        defer std.heap.page_allocator.free(v);
        if (std.fmt.parseInt(u32, v, 10)) |t| {
            if (t == 16 or t == 32 or t == 64) return t;
        } else |_| {}
    } else |_| {}
    const wg64 = ceilDiv(c_out, 64) * ceilDiv(ol, 64);
    if (c_out >= 64 and wg64 >= 400) return 64;
    const wg32 = ceilDiv(c_out, 32) * ceilDiv(ol, 32);
    if (wg32 >= 64) return 32;
    return 16;
}

/// Pick the conv_t1d tile edge (16/32/64) for a given (C_out, L_out)
/// output shape, mirroring pickConvTile. The conv_t1d shader is an
/// implicit GEMM over C_in*K with a TILE x TILE (Co x Lo) output tile per
/// workgroup, reduction gk' = k*C_in + ci (k-major — see conv_t1d.comp).
/// 实测调优（Radeon Pro VII，dec 真实形状，2026-09-25）：
///   conv_t1d 的 Kdim = C_in*K 远大于 conv1d（dec 第一级 8192 vs conv1d
///   1536）——深寄存器阻塞收益大：Co≥64 时 TILE=64 全面最优（oL=5000 档
///   tile64 1728 vs tile32 989 GFLOPS，+75%；oL=50000 档 2351 vs 1950），
///   即使 wg64 只有 ~80-316（conv1d 的 wg64≥400 阈值对 conv_t1d 太保守）。
///   Co<64（dec 64→32 级 Co=32 填不满 tile64，tile64 1365 < tile32 1931）
///   用 TILE=32；小形状（wg32<64）用 16。
/// Dev override: RVC_CONV_T1D_TILE=0 forces the naive pointwise kernel
/// (conv_t1d_naive.comp — bit-identical baseline, bench/regression);
/// RVC_CONV_T1D_TILE=16|32|64 forces one tiled path.
fn pickConvT1dTile(c_out: u32, l_out: u32) u32 {
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_CONV_T1D_TILE")) |v| {
        defer std.heap.page_allocator.free(v);
        if (std.fmt.parseInt(u32, v, 10)) |t| {
            if (t == 16 or t == 32 or t == 64) return t;
        } else |_| {}
    } else |_| {}
    if (c_out >= 64) return 64;
    const wg32 = ceilDiv(c_out, 32) * ceilDiv(l_out, 32);
    if (wg32 >= 64) return 32;
    return 16;
}

/// conv_t1d dispatch tile：RVC_CONV_T1D_TILE=0 → 0（naive 回退）；
/// =16|32|64 → 强制对应 TILE；未设 → 按形状 pickConvT1dTile。
/// 返回 0 表示用 conv_t1d_naive（1D flat grid），否则对应 TILE kernel。
fn convT1dTile(c_out: u32, l_out: u32) u32 {
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_CONV_T1D_TILE")) |v| {
        defer std.heap.page_allocator.free(v);
        if (std.fmt.parseInt(u32, v, 10)) |t| {
            if (t == 0 or t == 16 or t == 32 or t == 64) return t;
        } else |_| {}
    } else |_| {}
    return pickConvT1dTile(c_out, l_out);
}

/// Pick the conv_t2d tile edge (16/32/64) for a given (C_out, H_out*W_out)
/// output shape, mirroring pickConvT1dTile / pickConv2dTile. The conv_t2d
/// shader is an implicit GEMM over C_in*KH*KW with a TILE x TILE (Co x Lo,
/// Lo = linearised H_out*W_out) output tile per workgroup; the same
/// Co/RPT register-blocking tradeoff as conv_t1d applies, so reuse its
/// thresholds (Co>=64 -> 64, else wg32>=64 -> 32, small -> 16).
/// Dev override: RVC_CONV_T2D_TILE=16|32|64 forces one path.
fn pickConvT2dTile(c_out: u32, ol: u32) u32 {
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_CONV_T2D_TILE")) |v| {
        defer std.heap.page_allocator.free(v);
        if (std.fmt.parseInt(u32, v, 10)) |t| {
            if (t == 16 or t == 32 or t == 64) return t;
        } else |_| {}
    } else |_| {}
    if (c_out >= 64) return 64;
    const wg32 = ceilDiv(c_out, 32) * ceilDiv(ol, 32);
    if (wg32 >= 64) return 32;
    return 16;
}

// J25 suballocator：DEVICE_LOCAL 大块内切块复用（训练输出 buffer 高频
// mem_alloc 省 vkAllocateMemory）。SubChunk 是一块大 device-local memory；
// SubBlock 是其中的空闲段（best-fit + 相邻合并）。
pub const SubChunk = struct { memory: vk.c.VkDeviceMemory, size: usize };
pub const SubBlock = struct { chunk: u32, offset: usize, size: usize };

pub const Engine = struct {
    ctx: vk.Context,
    kern_matmul16: pipeline.Kernel, // TILE=16 (small shapes)
    kern_matmul32: pipeline.Kernel, // TILE=32 (mid shapes)
    kern_matmul64: pipeline.Kernel, // TILE=64 (large shapes, 4x4 block)
    kern_matmul_f16_16: pipeline.Kernel, // f16 A/B TILE=16 (experiment P1-4)
    kern_matmul_f16_32: pipeline.Kernel, // f16 A/B TILE=32
    kern_matmul_f16_64: pipeline.Kernel, // f16 A/B TILE=64
    kern_conv1d16: pipeline.Kernel, // conv1d TILE=16
    kern_conv1d32: pipeline.Kernel, // conv1d TILE=32
    kern_conv1d64: pipeline.Kernel, // conv1d TILE=64 (vec4, 4x4 block)
    kern_conv1d_groups: pipeline.Kernel, // conv1d_groups naive（DiscriminatorS）
    kern_conv1d_groups_bwd: pipeline.Kernel, // conv1d_groups 反向（3-mode）
    kern_add: pipeline.Kernel,
    kern_mul: pipeline.Kernel,
    kern_relu: pipeline.Kernel,
    kern_conv_t1d16: pipeline.Kernel, // conv_t1d TILE=16 (原朴素 kernel 的 TILE 化)
    kern_conv_t1d32: pipeline.Kernel, // conv_t1d TILE=32
    kern_conv_t1d64: pipeline.Kernel, // conv_t1d TILE=64 (4x4 block)
    kern_conv_t1d_naive: pipeline.Kernel, // conv_t1d 朴素逐点（RVC_CONV_T1D_TILE=0 回退）
    kern_conv_t2d16: pipeline.Kernel, // conv_t2d TILE=16
    kern_conv_t2d32: pipeline.Kernel, // conv_t2d TILE=32
    kern_conv_t2d64: pipeline.Kernel, // conv_t2d TILE=64 (4x4 block)
    kern_im2col_1d: pipeline.Kernel, // 阶段E(C3)：GPU im2col（conv1d 组装）
    kern_im2col_2d: pipeline.Kernel, // 阶段E(C3 v2)：GPU im2col 2D（conv2d 组装）
    kern_conv2d16: pipeline.Kernel, // conv2d TILE=16
    kern_conv2d32: pipeline.Kernel, // conv2d TILE=32
    kern_conv2d64: pipeline.Kernel, // conv2d TILE=64 (vec4)
    kern_embed: pipeline.Kernel,
    kern_insert_zeros_2x: pipeline.Kernel, // P15b: stride-2 插零上采样（convT x_up GPU 内生成）
    kern_add_inplace: pipeline.Kernel,
    kern_mul_inplace: pipeline.Kernel,
    kern_leaky_relu: pipeline.Kernel,
    kern_sqrt_inplace: pipeline.Kernel, // T4-2 AdamW: a[i]=sqrt(a[i])
    kern_rcp_inplace: pipeline.Kernel,  // T4-2 AdamW: a[i]=1/a[i]
    kern_div_const: pipeline.Kernel,    // T4-2 AdamW: a[i]/=s
    kern_mul_const: pipeline.Kernel,    // T4-2 AdamW: a[i]*=s
    kern_madd_const: pipeline.Kernel,   // T4-2 AdamW: a[i]+=s*b[i]
    kern_sub_inplace: pipeline.Kernel,  // T4-2 AdamW: a[i]-=b[i]
    kern_add_const: pipeline.Kernel,    // T4-2 AdamW: a[i]+=s
    kern_mul_buf_scalar: pipeline.Kernel, // T4-2 AdamW: a[i]*=b[0]
    kern_copy: pipeline.Kernel,
    kern_layernorm: pipeline.Kernel,
    kern_softmax: pipeline.Kernel,
    kern_rmsnorm: pipeline.Kernel,
    kern_gelu: pipeline.Kernel,
    kern_bias_add: pipeline.Kernel,
    kern_attn_qk: pipeline.Kernel,
    kern_attn_sv: pipeline.Kernel,
    kern_banded_attn_qk: pipeline.Kernel,
    kern_banded_attn_sv: pipeline.Kernel,
    kern_gn: pipeline.Kernel,
    kern_gating: pipeline.Kernel,
    kern_transpose: pipeline.Kernel, // T16 tiled GPU transpose [C,P]->[P,C]
    kern_transpose_b: pipeline.Kernel, // T-H7 batch transpose [B,C,P]->[C,B*P]
    kern_reduce_rows: pipeline.Kernel, // T-H7 row-sum reduce [M,N]->[M]
    kern_leaky_bwd: pipeline.Kernel,   // T-H7 leaky_relu backward elementwise
    kern_activation_bwd: pipeline.Kernel, // sigmoid/tanh backward elementwise
    kern_activation_fwd: pipeline.Kernel, // sigmoid/tanh forward elementwise
    kern_mul_bwd: pipeline.Kernel, // mul backward elementwise
    kern_slice_bwd: pipeline.Kernel, // axis=1 slice fwd/bwd
    kern_gating_bwd: pipeline.Kernel, // gating backward（op18 的融合反向）
    kern_gru: pipeline.Kernel, // T18 GRU 单 kernel（grid z=2 双向并行）
    kern_gru_sync: pipeline.Kernel, // T18 诊断：纯 barrier 循环（步间同步成本）
    rec: recorder.Recorder,
    pipeline_cache: vk.c.VkPipelineCache,
    allocator: std.mem.Allocator,
    buffers: std.AutoHashMapUnmanaged(u64, *buffer.Buffer),
    next_buf_id: u64,
    /// Recorded-but-uncommitted batch ops (rvc_batch_*). Cleared by
    /// batchBegin / batchCommit (on success) / batchDiscard.
    batch_ops: std.ArrayListUnmanaged(BatchEntry),
    /// Lazily-created zero buffer bound as the conv1d bias when
    /// `b == 0` (no bias): the shader never reads it (has_bias=0).
    bias_zero: ?*buffer.Buffer,
    mutex: std.Thread.Mutex = .{},

    // J25 suballocator state（结构定义见模块级 SubChunk/SubBlock）：
    // 训练输出 buffer 高频 mem_alloc：每次 vkCreateBuffer+vkGetBufferMemoryRequirements
    // +vkAllocateMemory+vkBindBufferMemory+gpuFill（~0.8ms）→ 大块内切块复用，
    // 只保留 createBuffer+bind+fill（省 vkAllocateMemory 大头）。空闲列表
    // 按 (chunk, offset) 排序，插入/释放时定位+相邻合并。
    sub_chunks: std.ArrayListUnmanaged(SubChunk) = .{},
    sub_free: std.ArrayListUnmanaged(SubBlock) = .{},
    sub_total: usize = 0,

    pub fn create(allocator: std.mem.Allocator) !*Engine {
        const eng = try allocator.create(Engine);
        errdefer allocator.destroy(eng);

        var ctx = try vk.Context.init(allocator);
        errdefer ctx.deinit();

        // P0-2：VkPipelineCache 磁盘持久化（驱动缓存管线编译结果，二次启动免重编）
        const pcache = try loadPipelineCache(ctx.device);
        errdefer vk.c.vkDestroyPipelineCache(ctx.device, pcache, null);

        var km16 = try pipeline.Kernel.init(&ctx, &shaders.matmul16, 3, @sizeOf(MatMulPush), pcache);
        errdefer km16.deinit();
        var km32 = try pipeline.Kernel.init(&ctx, &shaders.matmul32, 3, @sizeOf(MatMulPush), pcache);
        errdefer km32.deinit();
        var km64 = try pipeline.Kernel.init(&ctx, &shaders.matmul64, 3, @sizeOf(MatMulPush), pcache);
        errdefer km64.deinit();
        var kmf16_16 = try pipeline.Kernel.init(&ctx, &shaders.matmul_f16_16, 3, @sizeOf(MatMulPush), pcache);
        errdefer kmf16_16.deinit();
        var kmf16_32 = try pipeline.Kernel.init(&ctx, &shaders.matmul_f16_32, 3, @sizeOf(MatMulPush), pcache);
        errdefer kmf16_32.deinit();
        var kmf16_64 = try pipeline.Kernel.init(&ctx, &shaders.matmul_f16_64, 3, @sizeOf(MatMulPush), pcache);
        errdefer kmf16_64.deinit();
        var ka = try pipeline.Kernel.init(&ctx, &shaders.add, 3, @sizeOf(ElemPush), pcache);
        errdefer ka.deinit();
        var kq = try pipeline.Kernel.init(&ctx, &shaders.mul, 3, @sizeOf(ElemPush), pcache);
        errdefer kq.deinit();
        var kr = try pipeline.Kernel.init(&ctx, &shaders.relu, 1, @sizeOf(ElemPush), pcache);
        errdefer kr.deinit();
        var kc16 = try pipeline.Kernel.init(&ctx, &shaders.conv1d16, 4, @sizeOf(Conv1dPush), pcache);
        errdefer kc16.deinit();
        var kc32 = try pipeline.Kernel.init(&ctx, &shaders.conv1d32, 4, @sizeOf(Conv1dPush), pcache);
        errdefer kc32.deinit();
        var kc64 = try pipeline.Kernel.init(&ctx, &shaders.conv1d64, 4, @sizeOf(Conv1dPush), pcache);
        errdefer kc64.deinit();
        var kcg = try pipeline.Kernel.init(&ctx, &shaders.conv1d_groups, 4, @sizeOf(Conv1dGroupsPush), pcache);
        errdefer kcg.deinit();
        var kcgb = try pipeline.Kernel.init(&ctx, &shaders.conv1d_groups_bwd, 4, @sizeOf(Conv1dGroupsBwdPush), pcache);
        errdefer kcgb.deinit();
        var kct16 = try pipeline.Kernel.init(&ctx, &shaders.conv_t1d16, 4, @sizeOf(ConvT1dPush), pcache);
        errdefer kct16.deinit();
        var kct32 = try pipeline.Kernel.init(&ctx, &shaders.conv_t1d32, 4, @sizeOf(ConvT1dPush), pcache);
        errdefer kct32.deinit();
        var kct64 = try pipeline.Kernel.init(&ctx, &shaders.conv_t1d64, 4, @sizeOf(ConvT1dPush), pcache);
        errdefer kct64.deinit();
        var kctn = try pipeline.Kernel.init(&ctx, &shaders.conv_t1d_naive, 4, @sizeOf(ConvT1dPush), pcache);
        errdefer kctn.deinit();
        var kc2_16 = try pipeline.Kernel.init(&ctx, &shaders.conv2d16, 4, @sizeOf(Conv2dPush), pcache);
        errdefer kc2_16.deinit();
        var kc2_32 = try pipeline.Kernel.init(&ctx, &shaders.conv2d32, 4, @sizeOf(Conv2dPush), pcache);
        errdefer kc2_32.deinit();
        var kc2_64 = try pipeline.Kernel.init(&ctx, &shaders.conv2d64, 4, @sizeOf(Conv2dPush), pcache);
        errdefer kc2_64.deinit();
        var kt2_16 = try pipeline.Kernel.init(&ctx, &shaders.conv_t2d16, 4, @sizeOf(ConvT2dPush), pcache);
        errdefer kt2_16.deinit();
        var kt2_32 = try pipeline.Kernel.init(&ctx, &shaders.conv_t2d32, 4, @sizeOf(ConvT2dPush), pcache);
        errdefer kt2_32.deinit();
        var kt2_64 = try pipeline.Kernel.init(&ctx, &shaders.conv_t2d64, 4, @sizeOf(ConvT2dPush), pcache);
        errdefer kt2_64.deinit();
        var kim = try pipeline.Kernel.init(&ctx, &shaders.im2col_1d, 2, @sizeOf(Im2col1dPush), pcache);
        errdefer kim.deinit();
        var kim2 = try pipeline.Kernel.init(&ctx, &shaders.im2col_2d, 2, @sizeOf(Im2col2dPush), pcache);
        errdefer kim2.deinit();
        var ke = try pipeline.Kernel.init(&ctx, &shaders.embed, 3, @sizeOf(EmbedPush), pcache);
        errdefer ke.deinit();
        var kiz = try pipeline.Kernel.init(&ctx, &shaders.insert_zeros_2x, 2, @sizeOf(InsertZeros2xPush), pcache);
        errdefer kiz.deinit();
        var kai = try pipeline.Kernel.init(&ctx, &shaders.add_inplace, 2, @sizeOf(ElemPush), pcache);
        errdefer kai.deinit();
        var kmi = try pipeline.Kernel.init(&ctx, &shaders.mul_inplace, 2, @sizeOf(ElemPush), pcache);
        errdefer kmi.deinit();
        var klr = try pipeline.Kernel.init(&ctx, &shaders.leaky_relu, 1, @sizeOf(LeakyPush), pcache);
        errdefer klr.deinit();
        // T4-2 AdamW 标量元素算子（op30-37）：就地/双 buffer，push 布局同
        // ElemPush（n）/LeakyPush（n+s）。
        var ksq = try pipeline.Kernel.init(&ctx, &shaders.sqrt_inplace, 1, @sizeOf(ElemPush), pcache);
        errdefer ksq.deinit();
        var krp = try pipeline.Kernel.init(&ctx, &shaders.rcp_inplace, 1, @sizeOf(ElemPush), pcache);
        errdefer krp.deinit();
        var kdc = try pipeline.Kernel.init(&ctx, &shaders.div_const, 1, @sizeOf(LeakyPush), pcache);
        errdefer kdc.deinit();
        var kmc = try pipeline.Kernel.init(&ctx, &shaders.mul_const, 1, @sizeOf(LeakyPush), pcache);
        errdefer kmc.deinit();
        var kmd = try pipeline.Kernel.init(&ctx, &shaders.madd_const, 2, @sizeOf(LeakyPush), pcache);
        errdefer kmd.deinit();
        var ksb = try pipeline.Kernel.init(&ctx, &shaders.sub_inplace, 2, @sizeOf(ElemPush), pcache);
        errdefer ksb.deinit();
        var kac = try pipeline.Kernel.init(&ctx, &shaders.add_const, 1, @sizeOf(LeakyPush), pcache);
        errdefer kac.deinit();
        var kmbs = try pipeline.Kernel.init(&ctx, &shaders.mul_buf_scalar, 2, @sizeOf(ElemPush), pcache);
        errdefer kmbs.deinit();
        var kcp = try pipeline.Kernel.init(&ctx, &shaders.copy, 2, @sizeOf(ElemPush), pcache);
        errdefer kcp.deinit();
        var kl = try pipeline.Kernel.init(&ctx, &shaders.layernorm, 4, @sizeOf(NormPush), pcache);
        errdefer kl.deinit();
        var ks = try pipeline.Kernel.init(&ctx, &shaders.softmax, 2, @sizeOf(SoftmaxPush), pcache);
        errdefer ks.deinit();
        var krn = try pipeline.Kernel.init(&ctx, &shaders.rmsnorm, 3, @sizeOf(NormPush), pcache);
        errdefer krn.deinit();
        var kg = try pipeline.Kernel.init(&ctx, &shaders.gelu, 1, @sizeOf(ElemPush), pcache);
        errdefer kg.deinit();
        var kba = try pipeline.Kernel.init(&ctx, &shaders.bias_add, 3, @sizeOf(BiasAddPush), pcache);
        errdefer kba.deinit();
        var kaq = try pipeline.Kernel.init(&ctx, &shaders.attn_qk, 3, @sizeOf(AttnPush), pcache);
        errdefer kaq.deinit();
        var kas = try pipeline.Kernel.init(&ctx, &shaders.attn_sv, 3, @sizeOf(AttnPush), pcache);
        errdefer kas.deinit();
        var kbq = try pipeline.Kernel.init(&ctx, &shaders.banded_attn_qk, 4, @sizeOf(AttnPush), pcache);
        errdefer kbq.deinit();
        var kbs = try pipeline.Kernel.init(&ctx, &shaders.banded_attn_sv, 4, @sizeOf(AttnPush), pcache);
        errdefer kbs.deinit();
        var kgn = try pipeline.Kernel.init(&ctx, &shaders.gn, 4, @sizeOf(GnPush), pcache);
        errdefer kgn.deinit();
        var kgt = try pipeline.Kernel.init(&ctx, &shaders.gating, 3, @sizeOf(GatingPush), pcache);
        errdefer kgt.deinit();
        var ktp = try pipeline.Kernel.init(&ctx, &shaders.transpose, 2, @sizeOf(TransposePush), pcache);
        errdefer ktp.deinit();
        var ktb = try pipeline.Kernel.init(&ctx, &shaders.transpose_b, 2, @sizeOf(TransposeBPush), pcache);
        errdefer ktb.deinit();
        var krr = try pipeline.Kernel.init(&ctx, &shaders.reduce_rows, 2, @sizeOf(ReducePush), pcache);
        errdefer krr.deinit();
        var klb = try pipeline.Kernel.init(&ctx, &shaders.leaky_bwd, 3, @sizeOf(LeakyBwdPush), pcache);
        errdefer klb.deinit();
        var kab = try pipeline.Kernel.init(&ctx, &shaders.activation_bwd, 3, @sizeOf(ActivationBwdPush), pcache);
        errdefer kab.deinit();
        var kaf = try pipeline.Kernel.init(&ctx, &shaders.activation_fwd, 2, @sizeOf(ActivationFwdPush), pcache);
        errdefer kaf.deinit();
        var kmb = try pipeline.Kernel.init(&ctx, &shaders.mul_bwd, 3, @sizeOf(MulBwdPush), pcache);
        errdefer kmb.deinit();
        var ksl = try pipeline.Kernel.init(&ctx, &shaders.slice_bwd, 2, @sizeOf(SliceBwdPush), pcache);
        errdefer ksl.deinit();
        var kgb = try pipeline.Kernel.init(&ctx, &shaders.gating_bwd, 4, @sizeOf(GatingBwdPush), pcache);
        errdefer kgb.deinit();
        var kgru = try pipeline.Kernel.init(&ctx, &shaders.gru, 4, @sizeOf(GruPush), pcache);
        errdefer kgru.deinit();
        var kgrus = try pipeline.Kernel.init(&ctx, &shaders.gru_sync, 1, @sizeOf(GruPush), pcache);
        errdefer kgrus.deinit();

        // T1.1：dec 超限级 GPU 内分段后单 batch dispatch 数暴涨——12s 全
        // seg resident 约 500 段、150s 量级约 3000 段（每超限 op 2..12 段）。
        // 384/1536（原值）会 DescriptorPoolExhausted；放宽到 8192 sets ×
        // 32768 descriptors（每 set ≤4 binding），显存 ≈ 1MB，可忽略。
        var rec = try recorder.Recorder.init(&ctx, 8192, 32768);
        errdefer rec.deinit();

        eng.* = .{
            .ctx = ctx,
            .kern_matmul16 = km16,
            .kern_matmul32 = km32,
            .kern_matmul64 = km64,
            .kern_matmul_f16_16 = kmf16_16,
            .kern_matmul_f16_32 = kmf16_32,
            .kern_matmul_f16_64 = kmf16_64,
            .kern_add = ka,
            .kern_mul = kq,
            .kern_relu = kr,
            .kern_conv1d16 = kc16,
            .kern_conv1d32 = kc32,
            .kern_conv1d64 = kc64,
            .kern_conv1d_groups = kcg,
            .kern_conv1d_groups_bwd = kcgb,
            .kern_conv_t1d16 = kct16,
            .kern_conv_t1d32 = kct32,
            .kern_conv_t1d64 = kct64,
            .kern_conv_t1d_naive = kctn,
            .kern_conv_t2d16 = kt2_16,
            .kern_conv_t2d32 = kt2_32,
            .kern_conv_t2d64 = kt2_64,
            .kern_im2col_1d = kim,
            .kern_im2col_2d = kim2,
            .kern_conv2d16 = kc2_16,
            .kern_conv2d32 = kc2_32,
            .kern_conv2d64 = kc2_64,
            .kern_embed = ke,
            .kern_insert_zeros_2x = kiz,
            .kern_add_inplace = kai,
            .kern_mul_inplace = kmi,
            .kern_leaky_relu = klr,
            .kern_sqrt_inplace = ksq,
            .kern_rcp_inplace = krp,
            .kern_div_const = kdc,
            .kern_mul_const = kmc,
            .kern_madd_const = kmd,
            .kern_sub_inplace = ksb,
            .kern_add_const = kac,
            .kern_mul_buf_scalar = kmbs,
            .kern_copy = kcp,
            .kern_layernorm = kl,
            .kern_softmax = ks,
            .kern_rmsnorm = krn,
            .kern_gelu = kg,
            .kern_bias_add = kba,
            .kern_attn_qk = kaq,
            .kern_attn_sv = kas,
            .kern_banded_attn_qk = kbq,
            .kern_banded_attn_sv = kbs,
            .kern_gn = kgn,
            .kern_gating = kgt,
            .kern_transpose = ktp,
            .kern_transpose_b = ktb,
            .kern_reduce_rows = krr,
            .kern_leaky_bwd = klb,
            .kern_activation_bwd = kab,
            .kern_activation_fwd = kaf,
            .kern_mul_bwd = kmb,
            .kern_slice_bwd = ksl,
            .kern_gating_bwd = kgb,
            .kern_gru = kgru,
            .kern_gru_sync = kgrus,
            .rec = rec,
            .pipeline_cache = pcache,
            .allocator = allocator,
            .buffers = .{},
            .next_buf_id = 1,
            .batch_ops = .{},
            .bias_zero = null,
        };
        return eng;
    }

    pub fn destroy(self: *Engine) void {
        // P0-2：持久化管线缓存（须在 kernel 销毁前取数据）
        savePipelineCache(self.ctx.device, self.pipeline_cache);
        vk.c.vkDestroyPipelineCache(self.ctx.device, self.pipeline_cache, null);
        var it = self.buffers.valueIterator();
        while (it.next()) |b| {
            b.*.deinit(self.ctx.device);
            self.allocator.destroy(b.*);
        }
        self.buffers.deinit(self.allocator);
        self.batch_ops.deinit(self.allocator);
        // J25: release suballocator chunks (all sub buffers were already
        // destroyed above — deinit only frees the VkBuffer, chunk memory here).
        for (self.sub_chunks.items) |ch| {
            vk.c.vkFreeMemory(self.ctx.device, ch.memory, null);
        }
        self.sub_chunks.deinit(self.allocator);
        self.sub_free.deinit(self.allocator);
        if (self.bias_zero) |z| {
            z.deinit(self.ctx.device);
            self.allocator.destroy(z);
        }
        self.rec.deinit();
        self.kern_transpose.deinit();
        self.kern_transpose_b.deinit();
        self.kern_reduce_rows.deinit();
        self.kern_leaky_bwd.deinit();
        self.kern_activation_bwd.deinit();
        self.kern_activation_fwd.deinit();
        self.kern_mul_bwd.deinit();
        self.kern_slice_bwd.deinit();
        self.kern_gating_bwd.deinit();
        self.kern_gru.deinit();
        self.kern_gru_sync.deinit();
        self.kern_gating.deinit();
        self.kern_gn.deinit();
        self.kern_banded_attn_sv.deinit();
        self.kern_banded_attn_qk.deinit();
        self.kern_attn_sv.deinit();
        self.kern_attn_qk.deinit();
        self.kern_bias_add.deinit();
        self.kern_gelu.deinit();
        self.kern_rmsnorm.deinit();
        self.kern_softmax.deinit();
        self.kern_layernorm.deinit();
        self.kern_copy.deinit();
        self.kern_leaky_relu.deinit();
        self.kern_mul_buf_scalar.deinit();
        self.kern_add_const.deinit();
        self.kern_sub_inplace.deinit();
        self.kern_madd_const.deinit();
        self.kern_mul_const.deinit();
        self.kern_div_const.deinit();
        self.kern_rcp_inplace.deinit();
        self.kern_sqrt_inplace.deinit();
        self.kern_mul_inplace.deinit();
        self.kern_add_inplace.deinit();
        self.kern_embed.deinit();
        self.kern_insert_zeros_2x.deinit();
        self.kern_conv2d16.deinit();
        self.kern_conv2d32.deinit();
        self.kern_conv2d64.deinit();
        self.kern_conv_t1d_naive.deinit();
        self.kern_conv_t1d64.deinit();
        self.kern_conv_t1d32.deinit();
        self.kern_conv_t1d16.deinit();
        self.kern_conv_t2d64.deinit();
        self.kern_im2col_1d.deinit();
        self.kern_im2col_2d.deinit();
        self.kern_conv_t2d32.deinit();
        self.kern_conv_t2d16.deinit();
        self.kern_conv1d64.deinit();
        self.kern_conv1d_groups.deinit();
        self.kern_conv1d_groups_bwd.deinit();
        self.kern_conv1d32.deinit();
        self.kern_conv1d16.deinit();
        self.kern_relu.deinit();
        self.kern_mul.deinit();
        self.kern_add.deinit();
        self.kern_matmul16.deinit();
        self.kern_matmul32.deinit();
        self.kern_matmul64.deinit();
        self.kern_matmul_f16_16.deinit();
        self.kern_matmul_f16_32.deinit();
        self.kern_matmul_f16_64.deinit();
        self.ctx.deinit();
        self.allocator.destroy(self);
    }

    pub fn deviceName(self: *const Engine) [*:0]const u8 {
        return self.ctx.deviceName();
    }

    fn getBuf(self: *const Engine, id: u64) !*buffer.Buffer {
        return self.buffers.get(id) orelse error.InvalidBuffer;
    }

    /// Return the shared zero buffer used as a dummy conv1d bias binding
    /// (16 zero bytes; shader never reads it when has_bias==0).
    fn zeroBuf(self: *Engine) !*buffer.Buffer {
        if (self.bias_zero) |z| return z;
        const z = try self.allocator.create(buffer.Buffer);
        errdefer self.allocator.destroy(z);
        z.* = try buffer.Buffer.initDeviceOnly(&self.ctx, 16);
        self.bias_zero = z;
        return z;
    }

    // ── Memory API ─────────────────────────────────────────────────

    /// Create a static DEVICE_LOCAL buffer holding `data` (f32 count =
    /// data.len). Returns the opaque buffer id.
    pub fn memUpload(self: *Engine, data: []const f32) !u64 {
        if (data.len == 0) return error.EmptyUpload;
        const bytes = data.len * @sizeOf(f32);
        const buf = try self.allocator.create(buffer.Buffer);
        errdefer self.allocator.destroy(buf);
        buf.* = try buffer.Buffer.initStatic(&self.ctx, bytes);
        errdefer buf.deinit(self.ctx.device);
        try buf.upload(&self.ctx, std.mem.sliceAsBytes(data));
        const id = self.next_buf_id;
        self.next_buf_id += 1;
        try self.buffers.put(self.allocator, id, buf);
        return id;
    }

    /// Overwrite an existing buffer's contents (stage-D input pooling:
    /// reuse a pooled device buffer instead of alloc+free every upload).
    /// The buffer must be big enough (else error.BufferTooSmall).
    pub fn memUploadTo(self: *Engine, data: []const f32, id: u64) !void {
        if (data.len == 0) return error.EmptyUpload;
        const buf = try self.getBuf(id);
        const bytes = data.len * @sizeOf(f32);
        if (buf.bytes < bytes) return error.BufferTooSmall;
        try buf.upload(&self.ctx, std.mem.sliceAsBytes(data));
    }

    /// Batch upload-to（J9）：多条 `memUploadTo` 合并为一次 staging 批量
    /// copy（一次 submit + 一次 fence 等待）。datas/ids 等长；每条校验
    /// buffer 容量。调用方按 staging 容量分批（Python 侧 ≤64MB/批）。
    pub fn memUploadToBatch(self: *Engine, datas: []const []const f32, ids: []const u64) !void {
        if (datas.len == 0) return;
        if (datas.len != ids.len) return error.MismatchedBatch;
        var items = try self.allocator.alloc(buffer.UploadItem, datas.len);
        defer self.allocator.free(items);
        for (datas, ids, 0..) |d, id, i| {
            if (d.len == 0) return error.EmptyUpload;
            const buf = try self.getBuf(id);
            const bytes = d.len * @sizeOf(f32);
            if (buf.bytes < bytes) return error.BufferTooSmall;
            items[i] = .{ .dst = buf.handle, .data = std.mem.sliceAsBytes(d) };
        }
        try buffer.stagingUploadBatch(&self.ctx, items);
    }

    /// RVC_SUBALLOC_CHECK=1：每次 memAlloc 后验证新 sub-buffer 不与任何
    /// 现存 sub-buffer 物理重叠（训练路径不用；仅 A/B 定位重叠 bug）。
    var suballoc_check: ?bool = null;
    fn subCheckEnabled(self: *Engine) bool {
        if (suballoc_check == null) {
            const v = std.process.getEnvVarOwned(self.allocator, "RVC_SUBALLOC_CHECK") catch null;
            if (v) |s| {
                defer self.allocator.free(s);
                suballoc_check = s.len > 0 and s[0] != '0';
            } else {
                suballoc_check = false;
            }
        }
        return suballoc_check.?;
    }

    fn subAllocCheck(self: *Engine, new_buf: *const buffer.Buffer, new_id: u64) void {
        if (!self.subCheckEnabled()) return;
        var it = self.buffers.valueIterator();
        while (it.next()) |entry| {
            const b = entry.*;
            if (!b.sub or b.memory != new_buf.memory) continue;
            const a0 = new_buf.offset;
            const a1 = new_buf.offset + new_buf.bytes;
            const b0 = b.offset;
            const b1 = b.offset + b.bytes;
            if (a0 < b1 and b0 < a1) {
                std.debug.print(
                    "SUBALLOC_OVERLAP new_id={d} [{d},{d}) bytes={d} vs existing [{d},{d})\n",
                    .{ new_id, a0, a1, new_buf.bytes, b0, b1 },
                );
            }
        }
    }

    /// Allocate an uninitialised DEVICE_LOCAL buffer of `bytes` bytes
    /// (P1 perf: output buffers are fully overwritten by every shader, so
    /// zero-filling them is pure wasted PCIe traffic). Returns the id.
    ///
    /// J25: goes through the suballocator (large DEVICE_LOCAL chunks +
    /// best-fit free list) so frequent training output allocations reuse
    /// memory instead of paying vkAllocateMemory each time; falls back to
    /// the direct per-buffer allocation when the suballocator is out of
    /// budget or vkAllocateMemory for a new chunk fails.
    pub fn memAlloc(self: *Engine, bytes: usize) !u64 {
        if (bytes == 0) return error.EmptyUpload;
        const buf = try self.allocator.create(buffer.Buffer);
        errdefer self.allocator.destroy(buf);
        if (self.subAlloc(bytes)) |sb| {
            const ch = self.sub_chunks.items[sb.chunk];
            buf.* = try buffer.Buffer.initDeviceOnlySub(
                &self.ctx, bytes, ch.memory, sb.offset,
            );
            errdefer buf.deinit(self.ctx.device);
            // initDeviceOnlySub aligned the offset (memory type alignment);
            // return the gap and the tail to the free list.
            const off = buf.offset;
            const blk_end = sb.offset + sb.size;
            if (off + buf.bytes > blk_end) {
                // FIX #2: alignment pushed the allocation past the block end
                // (small tail blocks whose start is not alignment-aligned).
                // The block cannot host this buffer — return it whole to the
                // free list and fall back to a direct allocation for this
                // request. Without this check the VkBuffer was bound outside
                // the block, physically overlapping the neighbouring block →
                // deterministic cross-buffer overwrites in run_mode #2.
                buf.deinit(self.ctx.device);
                self.subInsertFree(sb.chunk, sb.offset, sb.size);
                buf.* = try buffer.Buffer.initDeviceOnly(&self.ctx, bytes);
                errdefer buf.deinit(self.ctx.device);
            } else {
                if (off > sb.offset) {
                    self.subInsertFree(sb.chunk, sb.offset, off - sb.offset);
                }
                const used_end = off + buf.bytes;
                if (used_end < blk_end) {
                    self.subInsertFree(sb.chunk, used_end, blk_end - used_end);
                }
            }
        } else {
            // Suballocator exhausted / failed: direct allocation path.
            buf.* = try buffer.Buffer.initDeviceOnly(&self.ctx, bytes);
            errdefer buf.deinit(self.ctx.device);
        }
        const id = self.next_buf_id;
        self.next_buf_id += 1;
        if (self.subCheckEnabled() and buf.sub) self.subAllocCheck(buf, id);
        try self.buffers.put(self.allocator, id, buf);
        return id;
    }

    // ── J25 suballocator internals ───────────────────────────────────
    const SUB_CHUNK_MIN: usize = 256 * 1024 * 1024; // first chunk 256MB
    const SUB_CHUNK_MAX: usize = 2 * 1024 * 1024 * 1024; // per-chunk cap

    /// env RVC_SUBALLOC_MB（MB，0=关 suballocator 回退每次 vkAllocateMemory）
    /// ——A/B 验证用；缺省 6144MB。注意 Zig 侧 env 在 create() 时读取一次。
    var suballoc_mb: ?usize = null;
    fn suballocEnabled(self: *Engine) bool {
        if (suballoc_mb == null) {
            const v = std.process.getEnvVarOwned(self.allocator, "RVC_SUBALLOC_MB") catch null;
            if (v) |s| {
                defer self.allocator.free(s);
                suballoc_mb = std.fmt.parseInt(usize, s, 10) catch 6144;
            } else {
                suballoc_mb = 6144;
            }
        }
        return suballoc_mb.? > 0;
    }

    fn subAlloc(self: *Engine, bytes: usize) ?SubBlock {
        if (!self.suballocEnabled()) return null;
        // Best-fit over the (sorted) free list.
        var best: ?usize = null;
        for (self.sub_free.items, 0..) |blk, i| {
            if (blk.size >= bytes and
                (best == null or blk.size < self.sub_free.items[best.?].size))
            {
                best = i;
            }
        }
        if (best) |i| {
            return self.sub_free.orderedRemove(i);
        }
        // Grow a new chunk.
        const idx = self.growSubChunk(@max(bytes, 16)) catch return null;
        return .{ .chunk = idx, .offset = 0, .size = self.sub_chunks.items[idx].size };
    }

    fn growSubChunk(self: *Engine, min_bytes: usize) !u32 {
        const total_cap = suballoc_mb.? * 1024 * 1024;
        if (!self.suballocEnabled() or self.sub_total >= total_cap) return error.SubAllocLimit;
        var size = SUB_CHUNK_MIN;
        for (self.sub_chunks.items) |ch| size = @max(size, ch.size * 2);
        size = @max(size, min_bytes);
        size = @min(size, SUB_CHUNK_MAX);
        if (self.sub_total + size > total_cap) {
            size = total_cap - self.sub_total;
            if (size < @max(min_bytes, 64 * 1024 * 1024)) return error.SubAllocLimit;
        }

        var mai = std.mem.zeroes(vk.c.VkMemoryAllocateInfo);
        mai.sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = size;
        // DEVICE_LOCAL memory type (same filter as buffer.createBuffer).
        var req = std.mem.zeroes(vk.c.VkBufferCreateInfo);
        req.sType = vk.c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        req.size = @intCast(size);
        req.usage = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        var probe: vk.c.VkBuffer = null;
        try vk.check(vk.c.vkCreateBuffer(self.ctx.device, &req, null, &probe));
        defer vk.c.vkDestroyBuffer(self.ctx.device, probe, null);
        var mem_req: vk.c.VkMemoryRequirements = undefined;
        vk.c.vkGetBufferMemoryRequirements(self.ctx.device, probe, &mem_req);
        mai.memoryTypeIndex = try buffer.findMemoryType(
            self.ctx.physical_device,
            mem_req.memoryTypeBits,
            vk.c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );

        // Try the full size, then halve on failure (driver memory pressure).
        while (true) {
            var memory: vk.c.VkDeviceMemory = null;
            const rc = vk.c.vkAllocateMemory(self.ctx.device, &mai, null, &memory);
            if (rc == vk.c.VK_SUCCESS) {
                const idx: u32 = @intCast(self.sub_chunks.items.len);
                try self.sub_chunks.append(self.allocator, .{ .memory = memory, .size = size });
                self.sub_total += size;
                return idx;
            }
            if (size <= @max(min_bytes, 64 * 1024 * 1024)) break;
            size = @max(size / 2, min_bytes);
            mai.allocationSize = size;
        }
        return error.SubAllocFail;
    }

    /// Insert a free block keeping the list sorted by (chunk, offset) and
    /// merge adjacent blocks of the same chunk.
    fn subInsertFree(self: *Engine, chunk: u32, offset: usize, size: usize) void {
        if (size == 0) return;
        var i: usize = 0;
        while (i < self.sub_free.items.len) : (i += 1) {
            const b = self.sub_free.items[i];
            if (b.chunk > chunk or (b.chunk == chunk and b.offset > offset)) break;
            // Merge into the previous block (prev end == this start).
            if (b.chunk == chunk and b.offset + b.size == offset) {
                self.sub_free.items[i].size += size;
                // Chain-merge with the next block.
                if (i + 1 < self.sub_free.items.len) {
                    const n = self.sub_free.items[i + 1];
                    if (n.chunk == chunk and
                        self.sub_free.items[i].offset + self.sub_free.items[i].size == n.offset)
                    {
                        self.sub_free.items[i].size += n.size;
                        _ = self.sub_free.orderedRemove(i + 1);
                    }
                }
                return;
            }
        }
        // Merge with the previous (i-1) block.
        if (i > 0) {
            const p = self.sub_free.items[i - 1];
            if (p.chunk == chunk and p.offset + p.size == offset) {
                self.sub_free.items[i - 1].size += size;
                if (i < self.sub_free.items.len) {
                    const n = self.sub_free.items[i];
                    if (n.chunk == chunk and
                        self.sub_free.items[i - 1].offset + self.sub_free.items[i - 1].size == n.offset)
                    {
                        self.sub_free.items[i - 1].size += n.size;
                        _ = self.sub_free.orderedRemove(i);
                    }
                }
                return;
            }
        }
        self.sub_free.insert(self.allocator, i, .{ .chunk = chunk, .offset = offset, .size = size }) catch return;
        // Merge with the next block after insertion.
        if (i + 1 < self.sub_free.items.len) {
            const n = self.sub_free.items[i + 1];
            if (n.chunk == chunk and offset + size == n.offset) {
                self.sub_free.items[i].size += n.size;
                _ = self.sub_free.orderedRemove(i + 1);
            }
        }
    }

    fn subChunkIndex(self: *Engine, memory: vk.c.VkDeviceMemory) u32 {
        for (self.sub_chunks.items, 0..) |ch, i| {
            if (ch.memory == memory) return @intCast(i);
        }
        return 0;
    }

    // ── T8：只读显存统计（探针逐段采样定位 chunk 增长/泄漏源）─────
    pub const MemStats = struct {
        sub_total: u64,        // suballocator 已分配 chunk 总字节（DEVICE_LOCAL）
        sub_chunks: u32,       // chunk 块数
        sub_free_bytes: u64,   // free list 空闲总字节（chunk 内未用）
        sub_free_blocks: u32,  // free list 块数（碎片度）
        buf_count: u32,        // 存活 buffer 数
        buf_bytes: u64,        // 存活 buffer 需求总字节（含 suballocator 内的）
        staging_up: u64,       // staging_up 常驻字节（HOST_VISIBLE）
        staging_dn: u64,       // staging_dn 常驻字节（HOST_VISIBLE）
        direct_bytes: u64,     // 非 suballocator（直接 vkAllocateMemory）存活字节
        max_buf_bytes: u64,    // 单个最大存活 buffer 字节
        max_buf_sub: u32,      // 最大 buffer 是否 suballocator 内（1/0）
    };

    pub fn memStats(self: *Engine) MemStats {
        var s = MemStats{
            .sub_total = 0,
            .sub_chunks = @intCast(self.sub_chunks.items.len),
            .sub_free_bytes = 0,
            .sub_free_blocks = @intCast(self.sub_free.items.len),
            .buf_count = 0,
            .buf_bytes = 0,
            .staging_up = self.ctx.staging_up.bytes,
            .staging_dn = self.ctx.staging_dn.bytes,
            .direct_bytes = 0,
            .max_buf_bytes = 0,
            .max_buf_sub = 0,
        };
        for (self.sub_chunks.items) |ch| s.sub_total += ch.size;
        for (self.sub_free.items) |blk| s.sub_free_bytes += blk.size;
        var it = self.buffers.valueIterator();
        while (it.next()) |b| {
            s.buf_count += 1;
            s.buf_bytes += b.*.bytes;
            if (!b.*.sub) s.direct_bytes += b.*.bytes;
            if (b.*.bytes > s.max_buf_bytes) {
                s.max_buf_bytes = b.*.bytes;
                s.max_buf_sub = if (b.*.sub) 1 else 0;
            }
        }
        return s;
    }

    /// T8：Top-N 存活 buffer dump（size 降序；打包 (bytes, sub, memory) 三元组）。
    pub fn memTop(self: *Engine, n: usize, out: []i64) usize {
        const Elem = struct { bytes: u64, sub: bool, mem: u64 };
        var top: [64]Elem = undefined;
        var cnt: usize = 0;
        var it = self.buffers.valueIterator();
        while (it.next()) |b| {
            const e: Elem = .{ .bytes = b.*.bytes, .sub = b.*.sub, .mem = @as(u64, @intFromPtr(b.*.memory)) };
            // 降序插入：找到首个 < e.bytes 的位置 j，挤出末尾（保持 top 前 cnt 有效）
            var j: usize = 0;
            while (j < cnt and top[j].bytes >= e.bytes) : (j += 1) {}
            if (j >= top.len) continue; // 比 top 里全部（含末尾哨兵）都小 → 不入选
            var k = @min(cnt, top.len - 1);
            while (k > j) : (k -= 1) top[k] = top[k - 1];
            top[j] = e;
            if (cnt < top.len) cnt += 1;
        }
        const m = @min(cnt, @min(n, top.len));
        for (0..m) |i| {
            out[i * 3] = @intCast(top[i].bytes);
            out[i * 3 + 1] = if (top[i].sub) 1 else 0;
            out[i * 3 + 2] = @intCast(top[i].mem);
        }
        return m;
    }

    /// GPU zero-fill an existing buffer (F3-B: pooled output buffers must be
    /// re-initialised before reuse — stale values pollute otherwise).
    pub fn memFillZero(self: *Engine, id: u64, bytes: usize) !void {
        const buf = try self.getBuf(id);
        try buffer.gpuFill(&self.ctx, buf.handle, bytes);
    }

    /// Create a static DEVICE_LOCAL buffer holding raw bytes (`data`) —
    /// the f16 upload path for FP16 matmul inputs (half values are just
    /// 2-byte little-endian; the shader reinterprets them as float16_t).
    /// Returns the opaque buffer id.
    pub fn memUploadBytes(self: *Engine, data: []const u8) !u64 {
        if (data.len == 0) return error.EmptyUpload;
        const buf = try self.allocator.create(buffer.Buffer);
        errdefer self.allocator.destroy(buf);
        buf.* = try buffer.Buffer.initStatic(&self.ctx, data.len);
        errdefer buf.deinit(self.ctx.device);
        try buf.upload(&self.ctx, data);
        const id = self.next_buf_id;
        self.next_buf_id += 1;
        try self.buffers.put(self.allocator, id, buf);
        return id;
    }

    /// Copy `dst.len` f32s out of buffer `id` into host memory.
    pub fn memDownload(self: *Engine, id: u64, dst: []f32) !void {
        const b = try self.getBuf(id);
        const want = dst.len * @sizeOf(f32);
        if (want > b.bytes) return error.BufferTooSmall;
        try b.download(&self.ctx, std.mem.sliceAsBytes(dst));
    }

    /// J18 批量下载：一次 submit + 一次 fence 处理多缓冲 readback（对称
    /// 上传批量）。`items` 由调用方持有；本方法校验各 buffer 大小后
    /// 用真实 VkBuffer 句柄重建 items（J18 遗留 bug 修复：调用方经 ffi
    /// 传入的 DownloadItem.src 是 buffer id 的 ptrFromInt 伪装，不能直接
    /// 交给驱动），再委托 stagingDownloadBatch。返回 error 时数据不可用
    /// （调用方自行回退逐次 download）。
    pub fn memDownloadBatch(self: *Engine, items: []const buffer.DownloadItem) !void {
        var fixed = try self.allocator.alloc(buffer.DownloadItem, items.len);
        defer self.allocator.free(fixed);
        for (items, 0..) |it, i| {
            const b = try self.getBuf(@intCast(@intFromPtr(it.src)));
            if (it.len > b.bytes) return error.BufferTooSmall;
            fixed[i] = .{ .src = b.handle, .dst = it.dst, .len = it.len };
        }
        try buffer.stagingDownloadBatch(&self.ctx, fixed);
    }

    pub fn memFree(self: *Engine, id: u64) !void {
        // The buffer may still be referenced by an in-flight (async)
        // submission — drain the queue before destroying it.
        try self.rec.waitAll();
        var b = self.buffers.fetchRemove(id) orelse return error.InvalidBuffer;
        if (b.value.sub) {
            // Suballocated: return the block to the chunk's free list
            // (deinit destroys only the VkBuffer, not the chunk memory).
            self.subInsertFree(self.subChunkIndex(b.value.memory), b.value.offset, b.value.bytes);
        }
        b.value.deinit(self.ctx.device);
        self.allocator.destroy(b.value);
    }

    // ── Ops ────────────────────────────────────────────────────────

    /// C[M,N] = A[M,K] x B[K,N] (row-major). All sizes checked against
    /// the backing buffer byte counts; C must have been allocated with
    /// at least M*N floats (e.g. via memUpload).
    pub fn matmul(self: *Engine, a: u64, b: u64, c: u64, m: u32, k: u32, n: u32) !void {
        if (m == 0 or k == 0 or n == 0) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const C = try self.getBuf(c);
        // u128 arithmetic so malicious m/k/n can't overflow the check.
        if (@as(u128, m) * k > (1 << 62) or @as(u128, k) * n > (1 << 62) or @as(u128, m) * n > (1 << 62))
            return error.DimensionsTooLarge;
        if (A.bytes < m * k * 4 or B.bytes < k * n * 4 or C.bytes < m * n * 4)
            return error.BufferTooSmall;

        const tile = pickTile(m, n);
        const push = MatMulPush{ .m = m, .k = k, .n = n, .tile = tile };
        const bufs = [_]*const buffer.Buffer{ A, B, C };
        const kern = switch (tile) {
            16 => &self.kern_matmul16,
            32 => &self.kern_matmul32,
            else => &self.kern_matmul64,
        };
        try self.rec.begin();
        // shader: col uses gl_WorkGroupID.x (gx = N tiles),
        // row uses .y (gy = M tiles). TILE x TILE output tile per wg.
        try self.rec.dispatch(kern, &bufs, null, &push, ceilDiv(n, tile), ceilDiv(m, tile), 1);
        try self.rec.endAndSubmit();
    }

    /// C[M,N] = A[M,K] x B[K,N] with FP16 inputs (P1-4 experiment).
    /// A/B buffers hold float16 data (2 bytes/elem); C is f32. The
    /// shader accumulates in f32 (precision-safe). Buffer byte checks
    /// use the half sizes for A/B.
    pub fn matmulF16(self: *Engine, a: u64, b: u64, c: u64, m: u32, k: u32, n: u32) !void {
        if (m == 0 or k == 0 or n == 0) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const C = try self.getBuf(c);
        if (@as(u128, m) * k > (1 << 62) or @as(u128, k) * n > (1 << 62) or @as(u128, m) * n > (1 << 62))
            return error.DimensionsTooLarge;
        if (A.bytes < m * k * 2 or B.bytes < k * n * 2 or C.bytes < m * n * 4)
            return error.BufferTooSmall;

        const tile = pickTile(m, n);
        const push = MatMulPush{ .m = m, .k = k, .n = n, .tile = tile };
        const bufs = [_]*const buffer.Buffer{ A, B, C };
        const kern = switch (tile) {
            16 => &self.kern_matmul_f16_16,
            32 => &self.kern_matmul_f16_32,
            else => &self.kern_matmul_f16_64,
        };
        try self.rec.begin();
        try self.rec.dispatch(kern, &bufs, null, &push, ceilDiv(n, tile), ceilDiv(m, tile), 1);
        try self.rec.endAndSubmit();
    }

    fn elementwise(self: *Engine, kern: *const pipeline.Kernel, ids: []const u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        var bufs: [3]*const buffer.Buffer = undefined;
        for (ids, 0..) |id, i| {
            const b = try self.getBuf(id);
            if (b.bytes < n * 4) return error.BufferTooSmall;
            bufs[i] = b;
        }
        const push = ElemPush{ .n = n };
        try self.rec.begin();
        try self.rec.dispatch(kern, bufs[0..ids.len], null, &push, ceilDiv(n, 256), 1, 1);
        try self.rec.endAndSubmit();
    }

    /// Two-buffer in-place elementwise op: a = a <op> b (writes into `a`).
    fn elementwiseInplace(self: *Engine, kern: *const pipeline.Kernel, a: u64, b: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < n * 4 or B.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        try self.rec.begin();
        // In-place: the FIRST binding (a) is the output.
        try self.rec.dispatch(kern, &bufs, 0, &push, ceilDiv(n, 256), 1, 1);
        try self.rec.endAndSubmit();
    }

    pub fn add(self: *Engine, a: u64, b: u64, c: u64, n: u32) !void {
        try self.elementwise(&self.kern_add, &.{ a, b, c }, n);
    }

    pub fn mul(self: *Engine, a: u64, b: u64, c: u64, n: u32) !void {
        try self.elementwise(&self.kern_mul, &.{ a, b, c }, n);
    }

    pub fn relu(self: *Engine, a: u64, n: u32) !void {
        try self.elementwise(&self.kern_relu, &.{a}, n);
    }

    /// In-place elementwise: a = a + b (flat over `n` floats).
    pub fn addInplace(self: *Engine, a: u64, b: u64, n: u32) !void {
        try self.elementwiseInplace(&self.kern_add_inplace, a, b, n);
    }

    /// In-place elementwise: a = a * b (flat over `n` floats).
    pub fn mulInplace(self: *Engine, a: u64, b: u64, n: u32) !void {
        try self.elementwiseInplace(&self.kern_mul_inplace, a, b, n);
    }

    /// In-place LeakyReLU over `n` floats: a[i] = (a[i]>=0) ? a[i] : slope*a[i].
    /// Semantics match torch F.leaky_relu / runtime.nn.leaky_relu.
    pub fn leakyRelu(self: *Engine, a: u64, n: u32, slope: f32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = slope };
        const bufs = [_]*const buffer.Buffer{A};
        try self.rec.begin();
        try self.rec.dispatch(&self.kern_leaky_relu, &bufs, null, &push, ceilDiv(n, 256), 1, 1);
        try self.rec.endAndSubmit();
    }

    /// Copy `n` floats: dst[i] = src[i] (flat). Used to preserve an input
    /// before an in-place op overwrites it.
    pub fn copy(self: *Engine, dst: u64, src: u64, n: u32) !void {
        try self.elementwiseInplace(&self.kern_copy, dst, src, n);
    }

    /// out[B,C_out,L_out] = conv(x[B,C_in,L], w[C_out,C_in,K]) (+ b[C_out]).
    /// `b == 0` means no bias (a shared zero buffer is bound instead).
    /// L_out = (L + pad_l + pad_r - dil*(K-1) - 1)/stride + 1.
    pub fn conv1d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        pad_l: u32,
        pad_r: u32,
        dil: u32,
    ) !void {
        if (B == 0 or c_in == 0 or l == 0 or c_out == 0 or k == 0 or stride == 0 or dil == 0)
            return error.InvalidDimensions;
        const kd: u64 = @as(u64, dil) * (k - 1);
        const l_out: u64 = if (kd + 1 <= @as(u64, l) + pad_l + pad_r)
            (@as(u64, l) + pad_l + pad_r - kd - 1) / stride + 1
        else
            0;
        if (l_out == 0) return error.InvalidDimensions; // no valid output points (also guards vkCmdDispatch gx=0)
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(l_out)) * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = Conv1dPush{
            .b = B,
            .c_in = c_in,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .pad_l = pad_l,
            .dil = dil,
            .l_out = @intCast(l_out),
            .has_bias = if (b == 0) 0 else 1,
            .lo_off = 0,
            .l_out_full = @intCast(l_out),
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        try self.rec.begin();
        // Tiled dispatch: gx = Lo tiles, gy = Co tiles, gz = batch. The
        // shader derives tile coords from gl_WorkGroupID.{x,y,z} and never
        // multiplies a global id with a local id (lane0 workaround).
        const tile = pickConvTile(c_out, @intCast(l_out));
        const kern = switch (tile) {
            16 => &self.kern_conv1d16,
            32 => &self.kern_conv1d32,
            else => &self.kern_conv1d64,
        };
        try self.rec.dispatch(kern, &bufs, null, &push, ceilDiv(@intCast(l_out), tile), ceilDiv(c_out, tile), B);
        try self.rec.endAndSubmit();
    }

    /// out[B,C_out,L_out] = conv_transpose1d(x[B,C_in,L], w[C_in,C_out,K]) (+ b[C_out]).
    /// PyTorch weight layout [C_in, C_out, K]; `b == 0` means no bias.
    /// L_out = (L-1)*stride - 2*padding + dil*(K-1) + output_padding + 1.
    pub fn convTranspose1d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        padding: u32,
        output_padding: u32,
        dil: u32,
    ) !void {
        if (B == 0 or c_in == 0 or l == 0 or c_out == 0 or k == 0 or stride == 0 or dil == 0)
            return error.InvalidDimensions;
        const l_out: u64 = (@as(u64, l - 1) * stride + dil * (k - 1) + output_padding + 1) - 2 * @as(u64, padding);
        if (2 * @as(u64, padding) > @as(u64, l - 1) * stride + dil * (k - 1))
            return error.InvalidDimensions; // padding must fit the upsampled extent
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_in) * c_out * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(l_out)) * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = ConvT1dPush{
            .b = B,
            .c_in = c_in,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .padding = padding,
            .output_padding = output_padding,
            .dil = dil,
            .l_out = @intCast(l_out),
            .has_bias = if (b == 0) 0 else 1,
            .in_off = 0,
            .l_seg = l, // 整段：输入窗口=全长（T1.1 分段时由调用方覆盖）
            .lo_off = 0,
            .l_out_full = @intCast(l_out),
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        try self.rec.begin();
        const total: u64 = @as(u64, B) * c_out * @as(u64, @intCast(l_out));
        // D2：组数 ≤ total ≤ u32 max << 驱动上限 2^32-1 → guard 仅防 u32
        // @intCast 溢出（实际上限由 Python 侧 2^31 控制）。
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        // TILE 化（conv1d 同款）：naive 用 1D flat grid，tile 用
        // gx=Lo 瓦片、gy=Co 瓦片、gz=batch（shader 用 gl_WorkGroupID 寻址）。
        // RVC_CONV_T1D_TILE=0 → tile=0（naive）；16|32|64 → 强制；未设按形状。
        const lout32: u32 = @intCast(l_out);
        const tile: u32 = convT1dTile(c_out, lout32);
        const kern = switch (tile) {
            0 => &self.kern_conv_t1d_naive,
            16 => &self.kern_conv_t1d16,
            32 => &self.kern_conv_t1d32,
            else => &self.kern_conv_t1d64,
        };
        const gx: u32 = if (tile == 0) ceilDiv(@intCast(total), 256) else ceilDiv(lout32, tile);
        const gy: u32 = if (tile == 0) 1 else ceilDiv(c_out, tile);
        const gz: u32 = if (tile == 0) 1 else B;
        try self.rec.dispatch(kern, &bufs, null, &push, gx, gy, gz);
        try self.rec.endAndSubmit();
    }

    /// out[B,C_out,H_out,W_out] = conv_transpose2d(x[B,C_in,OH,OW],
    /// w[C_in,C_out,KH,KW]) (+ b[C_out]).
    /// PyTorch weight layout [C_in, C_out, KH, KW]（= 前向 w[O,C,KH,KW] 的
    /// (1,0,2,3) 转置）；`b == 0` means no bias（gx 路径恒 0，零 buffer 绑定）。
    /// H_out = (OH-1)*sh - 2*ph + KH + opad_h；W_out = (OW-1)*sw - 2*pw + KW + opad_w。
    /// 由调用方传入 h_out/w_out（Python 侧 opad 计算后给出），engine 校验其与
    /// 公式一致 + opad ∈ [0, stride)（蓝图 §3 门禁，否则回退 numpy）。dilation=1。
    pub fn convTranspose2d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        oh: u32,
        ow: u32,
        c_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        opad_h: u32,
        opad_w: u32,
        h_out: u32,
        w_out: u32,
    ) !void {
        if (B == 0 or c_in == 0 or oh == 0 or ow == 0 or c_out == 0 or kh == 0 or kw == 0 or sh == 0 or sw == 0)
            return error.InvalidDimensions;
        if (opad_h >= sh or opad_w >= sw)
            return error.InvalidDimensions; // output_padding must be < stride (else host falls back to numpy)
        const h_ext: u64 = @as(u64, oh - 1) * sh + (kh - 1);
        const w_ext: u64 = @as(u64, ow - 1) * sw + (kw - 1);
        if (2 * @as(u64, ph) > h_ext or 2 * @as(u64, pw) > w_ext)
            return error.InvalidDimensions; // padding must fit the upsampled extent
        if (@as(u64, h_out) != h_ext + opad_h + 1 - 2 * @as(u64, ph) or
            @as(u64, w_out) != w_ext + opad_w + 1 - 2 * @as(u64, pw))
            return error.InvalidDimensions; // caller's H_out/W_out must match the transposed-conv formula
        const l_out: u64 = @as(u64, h_out) * w_out;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * oh * ow * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_in) * c_out * @as(u64, kh) * kw * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * l_out * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = ConvT2dPush{
            .b = B,
            .c_in = c_in,
            .oh = oh,
            .ow = ow,
            .c_out = c_out,
            .kh = kh,
            .kw = kw,
            .sh = sh,
            .sw = sw,
            .ph = ph,
            .pw = pw,
            .opad_h = opad_h,
            .opad_w = opad_w,
            .h_out = h_out,
            .w_out = w_out,
            .has_bias = if (b == 0) 0 else 1,
            .in_off = 0,
            .h_seg = oh, // 整段：输入窗口=OH 行（T2 分段时由调用方覆盖）
            .ho_off = 0,
            .h_out_full = h_out,
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        try self.rec.begin();
        const total: u64 = @as(u64, B) * c_out * l_out;
        // D2：组数 ≤ total ≤ u32 max << 驱动上限 2^32-1 → guard 仅防 u32
        // @intCast 溢出（实际上限由 Python 侧 2^31 控制）。
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        // TILE 化（conv_t1d/conv2d 同款）：gx=Lo(H_out*W_out 平面) 瓦片、
        // gy=Co 瓦片、gz=batch（shader 用 gl_WorkGroupID 寻址）。
        const lout32: u32 = @intCast(l_out);
        const tile: u32 = pickConvT2dTile(c_out, lout32);
        const kern = switch (tile) {
            16 => &self.kern_conv_t2d16,
            32 => &self.kern_conv_t2d32,
            else => &self.kern_conv_t2d64,
        };
        try self.rec.dispatch(kern, &bufs, null, &push, ceilDiv(lout32, tile), ceilDiv(c_out, tile), B);
        try self.rec.endAndSubmit();
    }

    /// out[B,C_out,OH,OW] = conv2d(x[B,C_in,H,W], w[C_out,C_in,KH,KW]) (+ b[C_out]).
    /// Symmetric padding (pad_h/pad_w both sides), per-axis stride; dilation=1.
    /// `b == 0` means no bias.
    /// OH = (H + 2*pad_h - KH)/stride_h + 1;  OW = (W + 2*pad_w - KW)/stride_w + 1.
    pub fn conv2d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        h: u32,
        w_: u32,
        c_out: u32,
        kh: u32,
        kw: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
    ) !void {
        if (B == 0 or c_in == 0 or h == 0 or w_ == 0 or c_out == 0 or kh == 0 or kw == 0 or stride_h == 0 or stride_w == 0)
            return error.InvalidDimensions;
        if (kh > h + 2 * pad_h or kw > w_ + 2 * pad_w)
            return error.InvalidDimensions; // kernel must fit the padded extent
        const oh: u64 = (@as(u64, h) + 2 * pad_h - kh) / stride_h + 1;
        const ow: u64 = (@as(u64, w_) + 2 * pad_w - kw) / stride_w + 1;
        if (oh == 0 or ow == 0) return error.InvalidDimensions;
        if (oh > std.math.maxInt(u32) or ow > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * h * w_ * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in * kh * kw * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(oh)) * @as(u64, @intCast(ow)) * 4)
            return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = Conv2dPush{
            .b = B,
            .c_in = c_in,
            .h = h,
            .w = w_,
            .c_out = c_out,
            .kh = kh,
            .kw = kw,
            .pad_h = pad_h,
            .pad_w = pad_w,
            .stride_h = stride_h,
            .stride_w = stride_w,
            .oh = @intCast(oh),
            .ow = @intCast(ow),
            .has_bias = if (b == 0) 0 else 1,
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        try self.rec.begin();
        const tile2 = pickConv2dTile(c_out, @intCast(oh * ow));
        const kern2 = switch (tile2) {
            16 => &self.kern_conv2d16,
            32 => &self.kern_conv2d32,
            else => &self.kern_conv2d64,
        };
        // gx = OL（空间）瓦片数：shader 的 gl_WorkGroupID.x 按 OL tile 寻址
        // （olBase = tc*TILE），gy 按 Co tile、gz 按 batch。此前误用
        // B*c_out*OL/tile 作为 gx → 每个 Co tile 重复 ~c_out/tile 倍冗余
        // workgroup（整段 K 循环空转不写输出）：实测 c_out=512 层 dispatch
        // ~500x 冗余、GPU busy 几乎全浪费（与 conv1d 的 ceilDiv(l_out,tile)
        // 一致的正确 grid）。D2：gx 上限按驱动实测 2^32-1 放宽，guard 仅防
        // u32 @intCast 溢出（ol=oh*ow 可能超 u32；oh/ow 各自已 guard）。
        const ol: u64 = oh * ow;
        if (ol > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        try self.rec.dispatch(kern2, &bufs, null, &push, ceilDiv(@intCast(ol), tile2), ceilDiv(c_out, tile2), B);
        try self.rec.endAndSubmit();
    }

    /// out[N, EmbDim] = table[ids[N], :]. `ids` buffer holds N int32
    /// non-negative indices (uploaded as raw bytes; one int32 per f32 slot).
    pub fn embed(self: *Engine, ids: u64, table: u64, out: u64, n: u32, table_rows: u32, emb_dim: u32) !void {
        if (n == 0 or table_rows == 0 or emb_dim == 0) return error.InvalidDimensions;
        const IDS = try self.getBuf(ids);
        const T = try self.getBuf(table);
        const O = try self.getBuf(out);
        if (IDS.bytes < @as(u64, n) * 4) return error.BufferTooSmall;
        if (T.bytes < @as(u64, table_rows) * emb_dim * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, n) * emb_dim * 4) return error.BufferTooSmall;

        const push = EmbedPush{ .n = n, .emb_dim = emb_dim };
        const bufs = [_]*const buffer.Buffer{ IDS, T, O };
        try self.rec.begin();
        const total: u64 = @as(u64, n) * emb_dim;
        // D2：gx = ceilDiv(total,256) ≤ total ≤ u32 max << 驱动上限 2^32-1。
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        try self.rec.dispatch(&self.kern_embed, &bufs, null, &push, ceilDiv(@intCast(total), 256), 1, 1);
        try self.rec.endAndSubmit();
    }

    /// P15b: out[C, 2H-1, 2W-1] = stride-2 zero insertion of x[C, H, W]:
    /// out[c,i,j] = (i%2==0 && j%2==0) ? x[c,i/2,j/2] : 0.0. Bit-identical
    /// to the host `np.zeros + [::2,::2]=x` recipe, so feeding the result
    /// into the same conv2d kernel reproduces the host convT output
    /// exactly. `C` is the merged channel count (B*C_in for a conv2d input
    /// [B, C_in, H, W]).
    pub fn insertZeros2x(self: *Engine, x: u64, out: u64, c: u32, h: u32, w: u32) !void {
        if (c == 0 or h == 0 or w == 0) return error.InvalidDimensions;
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, c) * h * w * 4) return error.BufferTooSmall;
        const uh: u64 = @as(u64, h) * 2 - 1;
        const uw: u64 = @as(u64, w) * 2 - 1;
        const total: u64 = @as(u64, c) * uh * uw;
        // D2：gx = ceilDiv(total,256) ≤ total ≤ u32 max << 驱动组上限 2^32-1
        // （实际上限由 Python 侧 _GRID_POINTS_MAX=2^31 控制，此处仅防溢出）。
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        if (O.bytes < total * 4) return error.BufferTooSmall;

        const push = InsertZeros2xPush{ .c = c, .h = h, .w = w };
        const bufs = [_]*const buffer.Buffer{ X, O };
        try self.rec.begin();
        try self.rec.dispatch(&self.kern_insert_zeros_2x, &bufs, null, &push, ceilDiv(@intCast(total), 256), 1, 1);
        try self.rec.endAndSubmit();
    }

    /// Per-row LayerNorm over `cols` floats: rows workgroups, each with
    /// 256 threads; `cols` of any size handled by strided loops.
    pub fn layernorm(self: *Engine, x: u64, gamma: u64, beta: u64, out: u64, rows: u32, cols: u32, eps: f64) !void {
        if (rows == 0 or cols == 0) return error.InvalidDimensions;
        if (rows > MAX_GRID_COUNT) return error.DimensionsTooLarge; // gy：驱动上限 2^32-1（D2，rows u32 恒满足）
        const X = try self.getBuf(x);
        const G = try self.getBuf(gamma);
        const Bt = try self.getBuf(beta);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, rows) * cols * 4 or O.bytes < @as(u64, rows) * cols * 4)
            return error.BufferTooSmall;
        if (G.bytes < cols * 4 or Bt.bytes < cols * 4) return error.BufferTooSmall;

        const push = NormPush{ .rows = rows, .cols = cols, .eps = @floatCast(eps) };
        const bufs = [_]*const buffer.Buffer{ X, G, Bt, O };
        try self.rec.begin();
        try self.rec.dispatch(&self.kern_layernorm, &bufs, null, &push, 1, rows, 1);
        try self.rec.endAndSubmit();
    }

    /// Per-row softmax over `cols` floats (numerically stable: exp(x-max)).
    pub fn softmax(self: *Engine, x: u64, out: u64, rows: u32, cols: u32) !void {
        if (rows == 0 or cols == 0) return error.InvalidDimensions;
        if (rows > MAX_GRID_COUNT) return error.DimensionsTooLarge; // gy：驱动上限 2^32-1（D2）
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, rows) * cols * 4 or O.bytes < @as(u64, rows) * cols * 4)
            return error.BufferTooSmall;

        const push = SoftmaxPush{ .rows = rows, .cols = cols };
        const bufs = [_]*const buffer.Buffer{ X, O };
        try self.rec.begin();
        try self.rec.dispatch(&self.kern_softmax, &bufs, null, &push, 1, rows, 1);
        try self.rec.endAndSubmit();
    }

    /// Per-row RMSNorm over `cols` floats: out = x/sqrt(mean(x^2)+eps)*gamma.
    pub fn rmsnorm(self: *Engine, x: u64, gamma: u64, out: u64, rows: u32, cols: u32, eps: f64) !void {
        if (rows == 0 or cols == 0) return error.InvalidDimensions;
        if (rows > MAX_GRID_COUNT) return error.DimensionsTooLarge; // gy：驱动上限 2^32-1（D2）
        const X = try self.getBuf(x);
        const G = try self.getBuf(gamma);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, rows) * cols * 4 or O.bytes < @as(u64, rows) * cols * 4)
            return error.BufferTooSmall;
        if (G.bytes < cols * 4) return error.BufferTooSmall;

        const push = NormPush{ .rows = rows, .cols = cols, .eps = @floatCast(eps) };
        const bufs = [_]*const buffer.Buffer{ X, G, O };
        try self.rec.begin();
        try self.rec.dispatch(&self.kern_rmsnorm, &bufs, null, &push, 1, rows, 1);
        try self.rec.endAndSubmit();
    }

    // ── Batch recorder (multi-op single submit) ─────────────────────
    //
    // rvc_batch_* accumulate op calls as (kernel, buffers, push, grid)
    // records, then one `batchCommit` replays them all inside a single
    // command buffer: one vkQueueSubmit + one fence-wait instead of N.
    // The recorder inserts a global memory barrier between dispatches,
    // so in-batch data dependencies (matmul output read by add_inplace,
    // etc.) are preserved with the same semantics as sequential single
    // calls. All per-op validation is duplicated from the one-shot
    // methods so invalid batches fail at add-time with identical errors.

    /// Reset the pending batch (discards any uncommitted ops). Idempotent.
    pub fn batchBegin(self: *Engine) void {
        self.batch_ops.clearRetainingCapacity();
    }

    fn batchAppend(
        self: *Engine,
        kern: *const pipeline.Kernel,
        bufs: []const *const buffer.Buffer,
        write_idx: u8,
        push: BatchPush,
        gx: u32,
        gy: u32,
        gz: u32,
    ) !void {
        return self.batchAppendView(kern, bufs, write_idx, push, gx, gy, gz, null);
    }

    fn batchAppendView(
        self: *Engine,
        kern: *const pipeline.Kernel,
        bufs: []const *const buffer.Buffer,
        write_idx: u8,
        push: BatchPush,
        gx: u32,
        gy: u32,
        gz: u32,
        view_offs: ?[]const usize,
    ) !void {
        if (bufs.len > 4) return error.TooManyBindings;
        if (view_offs) |vo| {
            if (vo.len != bufs.len) return error.TooManyBindings;
        }
        // Bounded by the recorder's descriptor pool: one set per dispatch,
        // and one descriptor per buffer (max 4).
        if (self.batch_ops.items.len >= self.rec.max_sets)
            return error.DescriptorPoolExhausted;
        var entry = BatchEntry{
            .kern = kern,
            .bufs = undefined,
            .nbufs = @intCast(bufs.len),
            .write_idx = write_idx,
            .push = push,
            .gx = gx,
            .gy = gy,
            .gz = gz,
        };
        for (bufs, 0..) |b, i| entry.bufs[i] = b;
        if (view_offs) |vo| {
            for (vo, 0..) |o, i| entry.view_offs[i] = o;
        }
        try self.batch_ops.append(self.allocator, entry);
    }

    /// Record matmul (same semantics/validation as `matmul`). C[M,N] =
    /// A[M,K] x B[K,N]; returns without submitting — call `batchCommit`.
    pub fn batchAddMatmul(self: *Engine, a: u64, b: u64, c: u64, m: u32, k: u32, n: u32) !void {
        if (m == 0 or k == 0 or n == 0) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const C = try self.getBuf(c);
        if (@as(u128, m) * k > (1 << 62) or @as(u128, k) * n > (1 << 62) or @as(u128, m) * n > (1 << 62))
            return error.DimensionsTooLarge;
        if (A.bytes < m * k * 4 or B.bytes < k * n * 4 or C.bytes < m * n * 4)
            return error.BufferTooSmall;

        const tile = pickTile(m, n);
        const push = MatMulPush{ .m = m, .k = k, .n = n, .tile = tile };
        const bufs = [_]*const buffer.Buffer{ A, B, C };
        const kern = switch (tile) {
            16 => &self.kern_matmul16,
            32 => &self.kern_matmul32,
            else => &self.kern_matmul64,
        };
        try self.batchAppend(kern, &bufs, @intCast(bufs.len - 1), .{ .matmul = push }, ceilDiv(n, tile), ceilDiv(m, tile), 1);
    }

    /// Record conv1d (same semantics/validation as `conv1d`). `b == 0`
    /// means no bias — the shared zero buffer is bound at record time.
    pub fn batchAddConv1d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        pad_l: u32,
        pad_r: u32,
        dil: u32,
    ) !void {
        if (B == 0 or c_in == 0 or l == 0 or c_out == 0 or k == 0 or stride == 0 or dil == 0)
            return error.InvalidDimensions;
        const kd: u64 = @as(u64, dil) * (k - 1);
        const l_out: u64 = if (kd + 1 <= @as(u64, l) + pad_l + pad_r)
            (@as(u64, l) + pad_l + pad_r - kd - 1) / stride + 1
        else
            0;
        if (l_out == 0) return error.InvalidDimensions;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(l_out)) * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = Conv1dPush{
            .b = B,
            .c_in = c_in,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .pad_l = pad_l,
            .dil = dil,
            .l_out = @intCast(l_out),
            .has_bias = if (b == 0) 0 else 1,
            .lo_off = 0,
            .l_out_full = @intCast(l_out),
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        // Tiled dispatch (same grid layout as `conv1d`).
        const tile = pickConvTile(c_out, @intCast(l_out));
        const kern = switch (tile) {
            16 => &self.kern_conv1d16,
            32 => &self.kern_conv1d32,
            else => &self.kern_conv1d64,
        };
        try self.batchAppend(kern, &bufs, @intCast(bufs.len - 1), .{ .conv1d = push }, ceilDiv(@intCast(l_out), tile), ceilDiv(c_out, tile), B);
    }

    /// 分组 1D 卷积 fwd（DiscriminatorS conv1d_groups）。w 布局
    /// [C_out, C_in_g, K]（C_in_g = C_in/groups，PyTorch groups 权重格式）；
    /// 组 g = co/(C_out/groups)，输入通道段 = g*C_in_g。dilation 恒 1；
    /// pad_r 仅校验（shader 对称 pad_l）。naive per-thread kernel（组内
    /// 归约小：C_in_g*K ≤ 164，共享内存 tile 无收益）。
    pub fn batchAddConv1dGroups(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        c_in_g: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        pad_l: u32,
        pad_r: u32,
    ) !void {
        if (B == 0 or c_in == 0 or c_in_g == 0 or c_in_g > c_in or l == 0 or
            c_out == 0 or k == 0 or stride == 0 or c_in % c_in_g != 0 or
            c_out % (c_in / c_in_g) != 0)
            return error.InvalidDimensions;
        const kd: u64 = k - 1; // dilation = 1
        const l_out: u64 = if (kd + 1 <= @as(u64, l) + pad_l + pad_r)
            (@as(u64, l) + pad_l + pad_r - kd - 1) / stride + 1
        else
            0;
        if (l_out == 0) return error.InvalidDimensions;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in_g * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(l_out)) * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = Conv1dGroupsPush{
            .b = B,
            .c_in = c_in,
            .c_in_g = c_in_g,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .pad_l = pad_l,
            .pad_r = pad_r,
            .l_out = @intCast(l_out),
            .has_bias = if (b == 0) 0 else 1,
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        // flat grid over B*C_out*L_out, 64 threads/workgroup
        try self.batchAppend(&self.kern_conv1d_groups, &bufs, @intCast(bufs.len - 1), .{ .conv1d_groups = push }, ceilDiv(B * c_out * @as(u32, @intCast(l_out)), 64), 1, 1);
    }

    /// conv1d_groups 反向（DiscriminatorS bp）：单 dispatch 单输出——
    /// mode 0=gx 1=gw 2=gb（BatchEntry 绑定上限 4，三次调用由调用方发出）。
    /// dilation 通用（J24：dec/enc dilated conv1d bp）；对称 pad_l。
    /// 数值口径同 conv1d_backward（引擎 f32）。
    pub fn batchAddConv1dGroupsBwdMode(
        self: *Engine,
        x: u64,
        w: u64,
        go: u64,
        out: u64,
        B: u32,
        c_in: u32,
        c_in_g: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        pad_l: u32,
        pad_r: u32,
        dilation: u32,
        mode: u32,
    ) !void {
        if (B == 0 or c_in == 0 or c_in_g == 0 or c_in_g > c_in or l == 0 or
            c_out == 0 or k == 0 or stride == 0 or dilation == 0 or
            c_in % c_in_g != 0 or c_out % (c_in / c_in_g) != 0)
            return error.InvalidDimensions;
        const kd: u64 = (k - 1) * dilation;
        const l_out: u64 = if (kd + 1 <= @as(u64, l) + pad_l + pad_r)
            (@as(u64, l) + pad_l + pad_r - kd - 1) / stride + 1
        else
            0;
        if (l_out == 0) return error.InvalidDimensions;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const GO = try self.getBuf(go);
        const OUT = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in_g * k * 4) return error.BufferTooSmall;
        if (GO.bytes < @as(u64, B) * c_out * @as(u64, @intCast(l_out)) * 4) return error.BufferTooSmall;
        if (mode == 0) {
            if (OUT.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        } else if (mode == 1) {
            if (OUT.bytes < @as(u64, c_out) * c_in_g * k * 4) return error.BufferTooSmall;
        } else {
            if (OUT.bytes < c_out * 4) return error.BufferTooSmall;
        }

        const bufs = [_]*const buffer.Buffer{ X, W, GO, OUT };
        const push = Conv1dGroupsBwdPush{
            .b = B, .c_in = c_in, .c_in_g = c_in_g, .l = l,
            .c_out = c_out, .k = k, .stride = stride, .pad_l = pad_l,
            .pad_r = pad_r, .dilation = dilation, .l_out = @intCast(l_out),
            .mode = mode,
        };
        const grid = if (mode == 0)
            ceilDiv(B * c_in * @as(u32, @intCast(l)), 64)
        else if (mode == 1)
            // J12: mode1（gw）每权重元素一个 workgroup（64 lane 共享归约）
            @as(u32, @intCast(c_out)) * c_in_g *
            @as(u32, @intCast(k))
        else
            ceilDiv(c_out, 64);
        try self.batchAppend(&self.kern_conv1d_groups_bwd, &bufs, 3, .{ .conv1d_groups_bwd = push }, grid, 1, 1);
    }

    /// T1.1 分段版 conv1d：输出段 [lo_off, lo_off+l_out)，x 整 buffer 绑定 +
    /// push 绝对寻址（lo_off/l_out_full），GPU 内段流转零 numpy 往返。
    pub fn batchAddConv1dView(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        pad_l: u32,
        _pad_r: u32, // 未用（engine conv1d 仅用 pad_l；dec ResBlock 对称 pad）
        dil: u32,
        l_out: u32,
        lo_off: u32,
        l_out_full: u32,
    ) !void {
        if (B == 0 or c_in == 0 or l == 0 or c_out == 0 or k == 0 or stride == 0 or dil == 0)
            return error.InvalidDimensions;
        // engine conv1d shader 仅有 pad_l（对称 padding 语义）——分段路径要求对称
        if (_pad_r != pad_l) return error.InvalidDimensions;
        if (l_out == 0) return error.InvalidDimensions;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(lo_off + l_out)) * 4)
            return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = Conv1dPush{
            .b = B,
            .c_in = c_in,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .pad_l = pad_l,
            .dil = dil,
            .l_out = l_out,
            .has_bias = if (b == 0) 0 else 1,
            .lo_off = lo_off,
            .l_out_full = l_out_full,
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        const tile = pickConvTile(c_out, l_out);
        const kern = switch (tile) {
            16 => &self.kern_conv1d16,
            32 => &self.kern_conv1d32,
            else => &self.kern_conv1d64,
        };
        try self.batchAppend(kern, &bufs, @intCast(bufs.len - 1), .{ .conv1d = push }, ceilDiv(l_out, tile), ceilDiv(c_out, tile), B);
    }

    /// Record conv2d (same semantics/validation as `conv2d`). `b == 0` means
    /// no bias — the shared zero buffer is bound at record time.
    /// x [B,C_in,H,W], w [C_out,C_in,KH,KW], out [B,C_out,OH,OW].
    pub fn batchAddConv2d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        h: u32,
        ww: u32,
        c_out: u32,
        kh: u32,
        kw: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
    ) !void {
        if (B == 0 or c_in == 0 or h == 0 or ww == 0 or c_out == 0 or kh == 0 or kw == 0 or
            stride_h == 0 or stride_w == 0) return error.InvalidDimensions;
        const oh: u64 = (@as(u64, h) + 2 * pad_h - kh) / stride_h + 1;
        const ow: u64 = (@as(u64, ww) + 2 * pad_w - kw) / stride_w + 1;
        if (oh == 0 or ow == 0) return error.InvalidDimensions;
        if (oh > std.math.maxInt(u32) or ow > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        const oh32: u32 = @intCast(oh);
        const ow32: u32 = @intCast(ow);

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * h * ww * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_out) * c_in * kh * kw * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * oh * ow * 4) return error.BufferTooSmall;
        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = Conv2dPush{
            .b = B,
            .c_in = c_in,
            .h = h,
            .w = ww,
            .c_out = c_out,
            .kh = kh,
            .kw = kw,
            .pad_h = pad_h,
            .pad_w = pad_w,
            .stride_h = stride_h,
            .stride_w = stride_w,
            .oh = oh32,
            .ow = ow32,
            .has_bias = if (b == 0) 0 else 1,
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        const total: u64 = @as(u64, B) * c_out * oh * ow;
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        const tile2 = pickConv2dTile(c_out, @intCast(oh * ow));
        const kern2 = switch (tile2) {
            16 => &self.kern_conv2d16,
            32 => &self.kern_conv2d32,
            else => &self.kern_conv2d64,
        };
        // 同 conv2d()：gx 应为 OL 瓦片数（shader gl_WorkGroupID.x 按 OL tile
        // 寻址），此前误用 B*c_out*OL/tile 造成 ~c_out/tile 倍冗余 workgroup。
        // D2：gx 上限按驱动实测 2^32-1 放宽，guard 仅防 u32 @intCast 溢出。
        const ol: u64 = oh * ow;
        if (ol > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        try self.batchAppend(kern2, &bufs, @intCast(bufs.len - 1), .{ .conv2d = push },
            ceilDiv(@intCast(ol), tile2), ceilDiv(c_out, tile2), B);
    }

    /// Record conv_transpose1d (same semantics/validation as
    /// `convTranspose1d`). `b == 0` means no bias — the shared zero buffer
    /// is bound at record time. Weight layout [C_in, C_out, K] (PyTorch).
    pub fn batchAddConvT1d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        padding: u32,
        output_padding: u32,
        dil: u32,
    ) !void {
        if (B == 0 or c_in == 0 or l == 0 or c_out == 0 or k == 0 or stride == 0 or dil == 0)
            return error.InvalidDimensions;
        const l_out: u64 = (@as(u64, l - 1) * stride + dil * (k - 1) + output_padding + 1) - 2 * @as(u64, padding);
        if (2 * @as(u64, padding) > @as(u64, l - 1) * stride + dil * (k - 1))
            return error.InvalidDimensions; // padding must fit the upsampled extent
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * l * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_in) * c_out * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(l_out)) * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = ConvT1dPush{
            .b = B,
            .c_in = c_in,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .padding = padding,
            .output_padding = output_padding,
            .dil = dil,
            .l_out = @intCast(l_out),
            .has_bias = if (b == 0) 0 else 1,
            .in_off = 0,
            .l_seg = l, // 整段：输入窗口=全长（T1.1 分段时由调用方覆盖）
            .lo_off = 0,
            .l_out_full = @intCast(l_out),
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        const total: u64 = @as(u64, B) * c_out * @as(u64, @intCast(l_out));
        // D2：组数 ≤ total ≤ u32 max << 驱动上限 2^32-1。
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        // TILE 化：naive 1D flat grid；tile gx=Lo 瓦片、gy=Co 瓦片、gz=batch。
        const lout32: u32 = @intCast(l_out);
        const tile: u32 = convT1dTile(c_out, lout32);
        const kern = switch (tile) {
            0 => &self.kern_conv_t1d_naive,
            16 => &self.kern_conv_t1d16,
            32 => &self.kern_conv_t1d32,
            else => &self.kern_conv_t1d64,
        };
        const gx: u32 = if (tile == 0) ceilDiv(@intCast(total), 256) else ceilDiv(lout32, tile);
        const gy: u32 = if (tile == 0) 1 else ceilDiv(c_out, tile);
        const gz: u32 = if (tile == 0) 1 else B;
        try self.batchAppend(kern, &bufs, @intCast(bufs.len - 1), .{ .conv_t1d = push }, gx, gy, gz);
    }

    /// Record conv_transpose2d (same semantics/validation as
    /// `convTranspose2d`). `b == 0` means no bias — the shared zero buffer
    /// is bound at record time. Weight layout [C_in, C_out, KH, KW] (PyTorch).
    /// 注：T2 一期 GPU 反向走单发 rvc_conv_t2d（判别器每层一次调用），
    /// Python 侧 BatchRunner 未接此 batch 路径——函数保留供后续接入。
    pub fn batchAddConvT2d(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        oh: u32,
        ow: u32,
        c_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        opad_h: u32,
        opad_w: u32,
        h_out: u32,
        w_out: u32,
    ) !void {
        if (B == 0 or c_in == 0 or oh == 0 or ow == 0 or c_out == 0 or kh == 0 or kw == 0 or sh == 0 or sw == 0)
            return error.InvalidDimensions;
        if (opad_h >= sh or opad_w >= sw)
            return error.InvalidDimensions; // output_padding must be < stride
        const h_ext: u64 = @as(u64, oh - 1) * sh + (kh - 1);
        const w_ext: u64 = @as(u64, ow - 1) * sw + (kw - 1);
        if (2 * @as(u64, ph) > h_ext or 2 * @as(u64, pw) > w_ext)
            return error.InvalidDimensions; // padding must fit the upsampled extent
        if (@as(u64, h_out) != h_ext + opad_h + 1 - 2 * @as(u64, ph) or
            @as(u64, w_out) != w_ext + opad_w + 1 - 2 * @as(u64, pw))
            return error.InvalidDimensions;
        const l_out: u64 = @as(u64, h_out) * w_out;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * c_in * oh * ow * 4) return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_in) * c_out * @as(u64, kh) * kw * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * l_out * 4) return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = ConvT2dPush{
            .b = B,
            .c_in = c_in,
            .oh = oh,
            .ow = ow,
            .c_out = c_out,
            .kh = kh,
            .kw = kw,
            .sh = sh,
            .sw = sw,
            .ph = ph,
            .pw = pw,
            .opad_h = opad_h,
            .opad_w = opad_w,
            .h_out = h_out,
            .w_out = w_out,
            .has_bias = if (b == 0) 0 else 1,
            .in_off = 0,
            .h_seg = oh, // 整段：输入窗口=OH 行（T2 分段时由调用方覆盖）
            .ho_off = 0,
            .h_out_full = h_out,
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        const total: u64 = @as(u64, B) * c_out * l_out;
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        const lout32: u32 = @intCast(l_out);
        const tile: u32 = pickConvT2dTile(c_out, lout32);
        const kern = switch (tile) {
            16 => &self.kern_conv_t2d16,
            32 => &self.kern_conv_t2d32,
            else => &self.kern_conv_t2d64,
        };
        try self.batchAppend(kern, &bufs, @intCast(bufs.len - 1), .{ .conv_t2d = push }, ceilDiv(lout32, tile), ceilDiv(c_out, tile), B);
    }

    /// 阶段E（C3）：GPU im2col（conv1d backward gw / forward 组装）。
    /// xw[B*oL, C*K_dil] row-major，dilation=1（调用方对 dilation≠1 回退
    /// host）。gather 无计算 → 与 host 视图链逐位一致（零回归）。
    pub fn batchAddIm2Col1d(
        self: *Engine,
        x: u64,
        out: u64,
        B: u32,
        C: u32,
        T: u32,
        oL: u32,
        K_dil: u32,
        stride: u32,
        pad_l: u32,
        dilation: u32,
    ) !void {
        if (B == 0 or C == 0 or T == 0 or oL == 0 or K_dil == 0 or stride == 0 or dilation == 0)
            return error.InvalidDimensions;
        const total: u64 = @as(u64, B) * oL * C * K_dil;
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * C * T * 4) return error.BufferTooSmall;
        if (O.bytes < total * 4) return error.BufferTooSmall;
        const push = Im2col1dPush{
            .B = B,
            .C = C,
            .T = T,
            .oL = oL,
            .K_dil = K_dil,
            .stride = stride,
            .pad_l = pad_l,
            .dilation = dilation,
        };
        const bufs = [_]*const buffer.Buffer{ X, O };
        try self.batchAppend(&self.kern_im2col_1d, &bufs, 1, .{ .im2col_1d = push }, ceilDiv(@intCast(total), 256), 1, 1);
    }

    /// 阶段E（C3 v2）：GPU im2col 2D（conv2d backward gw / forward 组装）。
    /// xw[B*OH*OW, C*KH*KW] row-major（行=(b,oh,ow) 行内=(c,kh,kw)），
    /// dilation=1（调用方对 dilation≠1 回退 host）。gather 无计算 → 与
    /// host 视图链逐位一致（零回归）。
    pub fn batchAddIm2Col2d(
        self: *Engine,
        x: u64,
        out: u64,
        B: u32,
        C: u32,
        H: u32,
        W: u32,
        OH: u32,
        OW: u32,
        KH: u32,
        KW: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
    ) !void {
        if (B == 0 or C == 0 or H == 0 or W == 0 or OH == 0 or OW == 0 or
            KH == 0 or KW == 0 or sh == 0 or sw == 0)
            return error.InvalidDimensions;
        const total: u64 = @as(u64, B) * OH * OW * C * KH * KW;
        if (total > std.math.maxInt(u32)) return error.DimensionsTooLarge;
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, B) * C * H * W * 4) return error.BufferTooSmall;
        if (O.bytes < total * 4) return error.BufferTooSmall;
        const push = Im2col2dPush{
            .B = B, .C = C, .H = H, .W = W, .OH = OH, .OW = OW,
            .KH = KH, .KW = KW, .sh = sh, .sw = sw, .ph = ph, .pw = pw,
        };
        const bufs = [_]*const buffer.Buffer{ X, O };
        try self.batchAppend(&self.kern_im2col_2d, &bufs, 1, .{ .im2col_2d = push }, ceilDiv(@intCast(total), 256), 1, 1);
    }

    /// T1.1 分段版：conv_t1d 的子段（GPU 内，无 host 往返）。
    /// ``l_in_seg`` = 本段输入窗口长度；``in_off`` = 输入起始列；
    /// ``lo_off`` = 输出段起始列（绝对）；``view_offs`` = 每 binding 字节偏移。
    pub fn batchAddConvT1dView(
        self: *Engine,
        x: u64,
        w: u64,
        b: u64,
        out: u64,
        B: u32,
        c_in: u32,
        l: u32,
        c_out: u32,
        k: u32,
        stride: u32,
        padding: u32,
        output_padding: u32,
        dil: u32,
        l_out: u32,
        l_in_seg: u32,
        in_off: u32,
        lo_off: u32,
        l_out_full: u32,
        view_offs: ?[]const usize,
    ) !void {
        if (B == 0 or c_in == 0 or l_in_seg == 0 or c_out == 0 or k == 0 or stride == 0 or dil == 0)
            return error.InvalidDimensions;
        if (l_out > std.math.maxInt(u32)) return error.DimensionsTooLarge;

        const X = try self.getBuf(x);
        const W = try self.getBuf(w);
        const O = try self.getBuf(out);
        // x 校验按子段（in_off + l_in_seg 列）；w/out 按整段
        if (X.bytes < (@as(u64, B) * c_in * (in_off + l_in_seg) * 4) and X.bytes < @as(u64, B) * c_in * l * 4)
            return error.BufferTooSmall;
        if (W.bytes < @as(u64, c_in) * c_out * k * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, B) * c_out * @as(u64, @intCast(lo_off + l_out)) * 4)
            return error.BufferTooSmall;

        const Buf: *const buffer.Buffer = if (b == 0) try self.zeroBuf() else try self.getBuf(b);
        if (b != 0 and Buf.bytes < c_out * 4) return error.BufferTooSmall;

        const push = ConvT1dPush{
            .b = B,
            .c_in = c_in,
            .l = l,
            .c_out = c_out,
            .k = k,
            .stride = stride,
            .padding = padding,
            .output_padding = output_padding,
            .dil = dil,
            .l_out = l_out,
            .has_bias = if (b == 0) 0 else 1,
            .in_off = in_off,
            .l_seg = l_in_seg,
            .lo_off = lo_off,
            .l_out_full = l_out_full, // 输出 buffer 行全长（绝对寻址步长）
        };
        const bufs = [_]*const buffer.Buffer{ X, W, Buf, O };
        // TILE 化：naive 1D flat grid；tile gx=Lo 瓦片、gy=Co 瓦片、gz=batch
        // （段长 l_out 参与瓦片选择，分段时自动用小 tile）。
        const tile: u32 = convT1dTile(c_out, l_out);
        const kern = switch (tile) {
            0 => &self.kern_conv_t1d_naive,
            16 => &self.kern_conv_t1d16,
            32 => &self.kern_conv_t1d32,
            else => &self.kern_conv_t1d64,
        };
        const gx: u32 = if (tile == 0) @intCast((@as(u64, B) * c_out * l_out + 255) / 256) else ceilDiv(l_out, tile);
        const gy: u32 = if (tile == 0) 1 else ceilDiv(c_out, tile);
        const gz: u32 = if (tile == 0) 1 else B;
        try self.batchAppendView(kern, &bufs, @intCast(bufs.len - 1), .{ .conv_t1d = push }, gx, gy, gz, view_offs);
    }

    /// Record in-place elementwise add a = a + b (flat over n floats).
    pub fn batchAddAddInplace(self: *Engine, a: u64, b: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < n * 4 or B.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        // In-place: the FIRST binding (a) is the output.
        try self.batchAppend(&self.kern_add_inplace, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise mul a = a * b (flat over n floats).
    pub fn batchAddMulInplace(self: *Engine, a: u64, b: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < n * 4 or B.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        // In-place: the FIRST binding (a) is the output.
        try self.batchAppend(&self.kern_mul_inplace, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    // ── T4-2 AdamW 标量元素算子（op30-37）──────────────────────────────
    // 全部就地、n 元素、一维；标量经 push constant 传（div/mul/madd/add_const
    // 用 LeakyPush{n, s} 布局，sqrt/rcp/sub/mul_buf_scalar 用 ElemPush{n}）。

    /// Record in-place elementwise sqrt: a[i] = sqrt(a[i]) (flat over n floats).
    pub fn batchAddSqrtInplace(self: *Engine, a: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_sqrt_inplace, &bufs, @intCast(bufs.len - 1), .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise reciprocal: a[i] = 1.0 / a[i] (flat over n floats).
    pub fn batchAddRcpInplace(self: *Engine, a: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_rcp_inplace, &bufs, @intCast(bufs.len - 1), .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise divide by scalar: a[i] /= s.
    pub fn batchAddDivConst(self: *Engine, a: u64, n: u32, s: f32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = s };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_div_const, &bufs, @intCast(bufs.len - 1), .{ .leaky = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise multiply by scalar: a[i] *= s.
    pub fn batchAddMulConst(self: *Engine, a: u64, n: u32, s: f32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = s };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_mul_const, &bufs, @intCast(bufs.len - 1), .{ .leaky = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise fused multiply-add: a[i] += s * b[i]
    /// (a in-place output, b read-only).
    pub fn batchAddMaddConst(self: *Engine, a: u64, b: u64, n: u32, s: f32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < n * 4 or B.bytes < n * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = s };
        const bufs = [_]*const buffer.Buffer{ A, B };
        // In-place: the FIRST binding (a) is the output.
        try self.batchAppend(&self.kern_madd_const, &bufs, 0, .{ .leaky = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise subtract: a[i] -= b[i]
    /// (a in-place output, b read-only).
    pub fn batchAddSubInplace(self: *Engine, a: u64, b: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < n * 4 or B.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        // In-place: the FIRST binding (a) is the output.
        try self.batchAppend(&self.kern_sub_inplace, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise add scalar: a[i] += s.
    pub fn batchAddAddConst(self: *Engine, a: u64, n: u32, s: f32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = s };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_add_const, &bufs, @intCast(bufs.len - 1), .{ .leaky = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place elementwise multiply by scalar-from-buffer:
    /// a[i] *= b[0] (b is a 1-element scalar slot; step-varying denom).
    pub fn batchAddMulBufScalar(self: *Engine, a: u64, b: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < n * 4 or B.bytes < 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        // In-place: the FIRST binding (a) is the output.
        try self.batchAppend(&self.kern_mul_buf_scalar, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record in-place LeakyReLU over n floats (same semantics/validation
    /// as `leakyRelu`). `slope` is the f32 negative slope (e.g. 0.1).
    pub fn batchAddLeakyRelu(self: *Engine, a: u64, n: u32, slope: f32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = slope };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_leaky_relu, &bufs, @intCast(bufs.len - 1), .{ .leaky = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record copy of n floats: dst[i] = src[i] (flat). `dst`/`src` are
    /// distinct buffers (dst must be pre-allocated with >= n floats).
    pub fn batchAddCopy(self: *Engine, dst: u64, src: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const D = try self.getBuf(dst);
        const S = try self.getBuf(src);
        if (D.bytes < n * 4 or S.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ D, S };
        // In-place-style: the FIRST binding (dst) is the output.
        try self.batchAppend(&self.kern_copy, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// T1.1 elementwise 分段版：copy 的子段 —— dst[off..off+n) = src[off..off+n)
    /// （flat，元素偏移 ×4 字节）。经 descriptorInfoView 子视图绑定，shader 的
    /// gid 相对段内（同一 kernel 同一 push → 与整段 copy 逐位一致）。engine 侧
    /// grid 上限防护由 Python 侧按段切分保证（每段 ≤ _GRID_POINTS_MAX）。
    pub fn batchAddCopyView(self: *Engine, dst: u64, src: u64, n: u32, dst_off: u32, src_off: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const D = try self.getBuf(dst);
        const S = try self.getBuf(src);
        if (D.bytes < (@as(u64, dst_off) + n) * 4 or S.bytes < (@as(u64, src_off) + n) * 4)
            return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ D, S };
        const view_offs = [_]usize{ @as(usize, dst_off) * 4, @as(usize, src_off) * 4 };
        try self.batchAppendView(&self.kern_copy, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1, &view_offs);
    }

    /// T1.1 elementwise 分段版：就地加的子段 —— a[off..off+n) += b[off..off+n)。
    pub fn batchAddAddInplaceView(self: *Engine, a: u64, b: u64, n: u32, a_off: u32, b_off: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < (@as(u64, a_off) + n) * 4 or B.bytes < (@as(u64, b_off) + n) * 4)
            return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        const view_offs = [_]usize{ @as(usize, a_off) * 4, @as(usize, b_off) * 4 };
        try self.batchAppendView(&self.kern_add_inplace, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1, &view_offs);
    }

    /// T1.1 elementwise 分段版：就地乘的子段 —— a[off..off+n) *= b[off..off+n)。
    pub fn batchAddMulInplaceView(self: *Engine, a: u64, b: u64, n: u32, a_off: u32, b_off: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        if (A.bytes < (@as(u64, a_off) + n) * 4 or B.bytes < (@as(u64, b_off) + n) * 4)
            return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{ A, B };
        const view_offs = [_]usize{ @as(usize, a_off) * 4, @as(usize, b_off) * 4 };
        try self.batchAppendView(&self.kern_mul_inplace, &bufs, 0, .{ .elem = push }, ceilDiv(n, 256), 1, 1, &view_offs);
    }

    /// T1.1 elementwise 分段版：就地 LeakyReLU 的子段 —— a[off..off+n) = lrelu(a)。
    /// ``slope`` 为 f32 负斜率（同 ``batchAddLeakyRelu``）。
    pub fn batchAddLeakyReluView(self: *Engine, a: u64, n: u32, slope: f32, a_off: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < (@as(u64, a_off) + n) * 4) return error.BufferTooSmall;
        const push = LeakyPush{ .n = n, .slope = slope };
        const bufs = [_]*const buffer.Buffer{A};
        const view_offs = [_]usize{@as(usize, a_off) * 4};
        try self.batchAppendView(&self.kern_leaky_relu, &bufs, @intCast(bufs.len - 1), .{ .leaky = push }, ceilDiv(n, 256), 1, 1, &view_offs);
    }

    // ── P1-5 attention middleware batch ops ────────────────────────────
    //
    // softmax / layer_norm / gelu / bias_add / attn_qk / attn_sv / gn let
    // the hubert/vits transformer layers run fully on the GPU inside ONE
    // batch commit (BatchTensor chaining, single host round-trip at the
    // end), instead of numpy round-tripping the attention middleware. All
    // of them reuse the exact single-call kernels/push layouts where one
    // exists (softmax/layer_norm), so batch vs single results are
    // bit-identical.

    /// Record per-row softmax: out[rows, cols] (same kernel + push as
    /// `softmax`). Rows are split across gy/gz (gz carries the overflow
    /// past 65535 rows — the shader adds gid.z*65535 to the row).
    pub fn batchAddSoftmax(self: *Engine, x: u64, out: u64, rows: u32, cols: u32) !void {
        if (rows == 0 or cols == 0) return error.InvalidDimensions;
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, rows) * cols * 4 or O.bytes < @as(u64, rows) * cols * 4)
            return error.BufferTooSmall;
        const push = SoftmaxPush{ .rows = rows, .cols = cols };
        const bufs = [_]*const buffer.Buffer{ X, O };
        const gy = @min(rows, 65535);
        try self.batchAppend(&self.kern_softmax, &bufs, @intCast(bufs.len - 1), .{ .softmax = push }, 1, gy, ceilDiv(rows, 65535));
    }

    /// Record per-row LayerNorm (same kernel + push as `layernorm`).
    /// gamma/beta are [cols]; rows split across gy/gz as for softmax.
    pub fn batchAddLayerNorm(self: *Engine, x: u64, gamma: u64, beta: u64, out: u64, rows: u32, cols: u32, eps: f64) !void {
        if (rows == 0 or cols == 0) return error.InvalidDimensions;
        const X = try self.getBuf(x);
        const G = try self.getBuf(gamma);
        const Bt = try self.getBuf(beta);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, rows) * cols * 4 or O.bytes < @as(u64, rows) * cols * 4)
            return error.BufferTooSmall;
        if (G.bytes < cols * 4 or Bt.bytes < cols * 4) return error.BufferTooSmall;
        const push = NormPush{ .rows = rows, .cols = cols, .eps = @floatCast(eps) };
        const bufs = [_]*const buffer.Buffer{ X, G, Bt, O };
        const gy = @min(rows, 65535);
        try self.batchAppend(&self.kern_layernorm, &bufs, @intCast(bufs.len - 1), .{ .norm = push }, 1, gy, ceilDiv(rows, 65535));
    }

    /// Record in-place GELU (exact erf, float32 A&S) over n floats.
    pub fn batchAddGelu(self: *Engine, a: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_gelu, &bufs, @intCast(bufs.len - 1), .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// T1.1 elementwise 分段版：就地 GELU 的子段 —— a[off..off+n) = gelu(a)。
    /// 同 ``batchAddGelu`` 同 kernel 同 push，仅绑定 buffer 子视图（偏移×4 字节），
    /// shader 的 gid 相对段内 → 与整段 gelu 逐位一致（hubert conv 栈超限链用）。
    pub fn batchAddGeluView(self: *Engine, a: u64, n: u32, a_off: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < (@as(u64, a_off) + n) * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{A};
        const view_offs = [_]usize{@as(usize, a_off) * 4};
        try self.batchAppendView(&self.kern_gelu, &bufs, @intCast(bufs.len - 1), .{ .elem = push }, ceilDiv(n, 256), 1, 1, &view_offs);
    }

    /// Record in-place ReLU over n floats (a[i] = max(a[i], 0)).
    pub fn batchAddRelu(self: *Engine, a: u64, n: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        if (A.bytes < n * 4) return error.BufferTooSmall;
        const push = ElemPush{ .n = n };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_relu, &bufs, @intCast(bufs.len - 1), .{ .elem = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record row-broadcast bias add: out[i,j] = a[i,j] + b[j] (n = rows*cols).
    pub fn batchAddBiasAdd(self: *Engine, a: u64, b: u64, out: u64, n: u32, cols: u32) !void {
        if (n == 0 or cols == 0) return error.InvalidDimensions;
        if (@as(u64, n) > (1 << 40)) return error.DimensionsTooLarge;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const O = try self.getBuf(out);
        if (A.bytes < n * 4 or O.bytes < n * 4 or B.bytes < cols * 4)
            return error.BufferTooSmall;
        const push = BiasAddPush{ .n = n, .cols = cols };
        const bufs = [_]*const buffer.Buffer{ A, B, O };
        try self.batchAppend(&self.kern_bias_add, &bufs, @intCast(bufs.len - 1), .{ .bias_add = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record fused attention scores: q/k are [T, C] with head-interleaved
    /// columns (head h at [h*D, (h+1)*D)); out[H, T, T] = q_h . k_h^T per
    /// head. One dispatch for all heads (vs 12 separate matmuls).
    pub fn batchAddAttnQk(self: *Engine, q: u64, k: u64, out: u64, H: u32, T: u32, D: u32, C: u32) !void {
        if (H == 0 or T == 0 or D == 0 or C == 0) return error.InvalidDimensions;
        if (D > 128) return error.InvalidDimensions; // shared tile capacity (enc_p D=96/96 ok)
        const Q = try self.getBuf(q);
        const K = try self.getBuf(k);
        const O = try self.getBuf(out);
        if (Q.bytes < @as(u64, T) * C * 4 or K.bytes < @as(u64, T) * C * 4)
            return error.BufferTooSmall;
        if (O.bytes < @as(u64, H) * T * T * 4) return error.BufferTooSmall;
        const push = AttnPush{ .H = H, .T = T, .D = D, .C = C };
        const bufs = [_]*const buffer.Buffer{ Q, K, O };
        try self.batchAppend(&self.kern_attn_qk, &bufs, @intCast(bufs.len - 1), .{ .attn = push }, ceilDiv(T, 16), ceilDiv(T, 16), H);
    }

    /// Record fused attention context: attnW[H, T, T] (softmaxed scores) x
    /// v[T, C] head-interleaved -> ctx[T, C] (heads merged in the kernel).
    pub fn batchAddAttnSv(self: *Engine, w: u64, v: u64, out: u64, H: u32, T: u32, D: u32, C: u32) !void {
        if (H == 0 or T == 0 or D == 0 or C == 0) return error.InvalidDimensions;
        if (D > 128) return error.InvalidDimensions; // kernel column window covers D<=128
        const W = try self.getBuf(w);
        const V = try self.getBuf(v);
        const O = try self.getBuf(out);
        if (W.bytes < @as(u64, H) * T * T * 4 or V.bytes < @as(u64, T) * C * 4)
            return error.BufferTooSmall;
        if (O.bytes < @as(u64, T) * C * 4) return error.BufferTooSmall;
        const push = AttnPush{ .H = H, .T = T, .D = D, .C = C };
        const bufs = [_]*const buffer.Buffer{ W, V, O };
        try self.batchAppend(&self.kern_attn_sv, &bufs, @intCast(bufs.len - 1), .{ .attn = push }, ceilDiv(D, 16), ceilDiv(T, 16), H);
    }

    /// Record fused banded attention scores (T5 relative-position fused,
    /// D5): scores[H, T, T] = q_h . (k_h + used[(s-t+T-1)])^T, where
    /// used[2T-1, D] is the relative-position embedding. Equivalent to
    /// q@k^T + rel_to_abs(q@used^T) with position bias folded in.
    /// q/k [T, C] head-interleaved; used [2T-1, D] (D == kc <= 128).
    pub fn batchAddBandedAttnQk(self: *Engine, q: u64, k: u64, used: u64, out: u64, H: u32, T: u32, D: u32, C: u32) !void {
        if (H == 0 or T == 0 or D == 0 or C == 0) return error.InvalidDimensions;
        if (D > 128) return error.InvalidDimensions; // shared tile capacity
        const Q = try self.getBuf(q);
        const K = try self.getBuf(k);
        const U = try self.getBuf(used);
        const O = try self.getBuf(out);
        if (Q.bytes < @as(u64, T) * C * 4 or K.bytes < @as(u64, T) * C * 4)
            return error.BufferTooSmall;
        if (U.bytes < @as(u64, 2 * T - 1) * D * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, H) * T * T * 4) return error.BufferTooSmall;
        const push = AttnPush{ .H = H, .T = T, .D = D, .C = C };
        const bufs = [_]*const buffer.Buffer{ Q, K, U, O };
        try self.batchAppend(&self.kern_banded_attn_qk, &bufs, @intCast(bufs.len - 1), .{ .attn = push }, ceilDiv(T, 16), ceilDiv(T, 16), H);
    }

    /// Record fused banded attention context (D5): ctx[T, C] = attnW .
    /// (v + used[(s-t+T-1)]) with the relative-position VALUE embedding
    /// folded in — equivalent to attnW@v + abs_to_rel(attnW)@used_v.
    /// attnW [H, T, T]; v [T, C] head-interleaved; used [2T-1, D].
    pub fn batchAddBandedAttnSv(self: *Engine, w: u64, v: u64, used: u64, out: u64, H: u32, T: u32, D: u32, C: u32) !void {
        if (H == 0 or T == 0 or D == 0 or C == 0) return error.InvalidDimensions;
        if (D > 128) return error.InvalidDimensions;
        const W = try self.getBuf(w);
        const V = try self.getBuf(v);
        const U = try self.getBuf(used);
        const O = try self.getBuf(out);
        if (W.bytes < @as(u64, H) * T * T * 4 or V.bytes < @as(u64, T) * C * 4)
            return error.BufferTooSmall;
        if (U.bytes < @as(u64, 2 * T - 1) * D * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, T) * C * 4) return error.BufferTooSmall;
        const push = AttnPush{ .H = H, .T = T, .D = D, .C = C };
        const bufs = [_]*const buffer.Buffer{ W, V, U, O };
        try self.batchAppend(&self.kern_banded_attn_sv, &bufs, @intCast(bufs.len - 1), .{ .attn = push }, ceilDiv(D, 16), ceilDiv(T, 16), H);
    }

    /// Record GroupNorm: x[G, Cpg*S] (G groups, Cpg channels/group,
    /// S spatial elements) normalized per group with gamma/beta[C]
    /// indexed by channel. Matches nn.group_norm semantics (biased var).
    pub fn batchAddGroupNorm(self: *Engine, x: u64, gamma: u64, beta: u64, out: u64, G: u32, Cpg: u32, S: u32, eps: f64) !void {
        if (G == 0 or Cpg == 0 or S == 0) return error.InvalidDimensions;
        const X = try self.getBuf(x);
        const Ga = try self.getBuf(gamma);
        const Bt = try self.getBuf(beta);
        const O = try self.getBuf(out);
        const n: u64 = @as(u64, G) * Cpg * S;
        const C: u64 = @as(u64, G) * Cpg;
        if (X.bytes < n * 4 or O.bytes < n * 4) return error.BufferTooSmall;
        if (Ga.bytes < C * 4 or Bt.bytes < C * 4) return error.BufferTooSmall;
        const push = GnPush{ .G = G, .Cpg = Cpg, .S = S, .eps = @floatCast(eps) };
        const bufs = [_]*const buffer.Buffer{ X, Ga, Bt, O };
        const gy = @min(G, 65535);
        try self.batchAppend(&self.kern_gn, &bufs, @intCast(bufs.len - 1), .{ .gn = push }, 1, gy, ceilDiv(G, 65535));
    }

    /// Record fused WN gate (flow, D6): c[i] = tanh(a[i]+g1) * sigmoid(a[i+n]+g2).
    /// a: [1,2H,L] (conv1d out: first H ch = tanh branch, next H = sigmoid),
    /// b: [1, 3*2H, lg] cond segments (off = layer seg start), c: [1,H,L].
    /// n = H*L; lg = 1 (g broadcast along L) or lg = L. GPU tanh/sigmoid ~1ulp
    /// vs libm → documented float delta (D5 precedent), NOT bit-identical.
    pub fn batchAddGating(self: *Engine, a: u64, b: u64, out: u64, n: u32, lg: u32, h: u32, off: u32) !void {
        if (n == 0 or h == 0 or lg == 0) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const O = try self.getBuf(out);
        if (A.bytes < @as(u64, n) * 2 * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, n) * 4) return error.BufferTooSmall;
        if (B.bytes < (@as(u64, off) + 2 * @as(u64, h) * lg) * 4) return error.BufferTooSmall;
        const push = GatingPush{ .n = n, .lg = lg, .h = h, .off = off };
        const bufs = [_]*const buffer.Buffer{ A, B, O };
        try self.batchAppend(&self.kern_gating, &bufs, @intCast(bufs.len - 1), .{ .gating = push }, ceilDiv(n, 256), 1, 1);
    }

    /// Record tiled GPU transpose (T16, op=19): out[P,C] = src[C,P] * scale
    /// (both row-major; kernel is its own inverse — swap C/P for the reverse
    /// direction). Grid: gx = ceil(P/16) (src col tiles), gy = ceil(C/16).
    pub fn batchAddTranspose(self: *Engine, src: u64, out: u64, c: u32, p: u32, scale: f32) !void {
        if (c == 0 or p == 0) return error.InvalidDimensions;
        const S = try self.getBuf(src);
        const O = try self.getBuf(out);
        if (S.bytes < @as(u64, c) * p * 4 or O.bytes < @as(u64, c) * p * 4)
            return error.BufferTooSmall;
        const push = TransposePush{ .c = c, .p = p, .scale = scale };
        const bufs = [_]*const buffer.Buffer{ S, O };
        try self.batchAppend(
            &self.kern_transpose, &bufs, @intCast(bufs.len - 1),
            .{ .transpose = push },
            ceilDiv(p, 16), ceilDiv(c, 16), 1,
        );
    }

    /// T-H7：批量转置 [B,C,P] → [C,B*P]（b 并入内层），backward conv1d/conv2d
    /// gw 的 go_r 布局（go [B,O,oL] → [O,B*oL]）。直读 gather kernel：
    /// dst[o, b*P+p] = src[b, o*P+p]，写入沿 p 连续（coalesced）。grid
    /// gx = ceilDiv(C*B*P, 64)。
    pub fn batchAddTransposeB(self: *Engine, src: u64, out: u64, b: u32, c: u32, p: u32) !void {
        if (b == 0 or c == 0 or p == 0) return error.InvalidDimensions;
        const S = try self.getBuf(src);
        const O = try self.getBuf(out);
        if (S.bytes < @as(u64, b) * c * p * 4 or O.bytes < @as(u64, b) * c * p * 4)
            return error.BufferTooSmall;
        const push = TransposeBPush{ .b = b, .c = c, .p = p };
        const bufs = [_]*const buffer.Buffer{ S, O };
        try self.batchAppend(
            &self.kern_transpose_b, &bufs, @intCast(bufs.len - 1),
            .{ .transpose_b = push },
            ceilDiv(c * b * p, 64), 1, 1,
        );
    }

    /// T-H7：行归约求和 out[m] = Σ_n src[m,n]（conv bias 梯度 gb = reduce(go_r)）。
    /// 每行一个 64 线程 workgroup：stride 累加 + 共享内存树归约。grid gx = M。
    pub fn batchAddReduceRows(self: *Engine, src: u64, out: u64, m: u32, n: u32) !void {
        if (m == 0 or n == 0) return error.InvalidDimensions;
        const S = try self.getBuf(src);
        const O = try self.getBuf(out);
        if (S.bytes < @as(u64, m) * n * 4 or O.bytes < @as(u64, m) * 4)
            return error.BufferTooSmall;
        const push = ReducePush{ .m = m, .n = n };
        const bufs = [_]*const buffer.Buffer{ S, O };
        try self.batchAppend(
            &self.kern_reduce_rows, &bufs, @intCast(bufs.len - 1),
            .{ .reduce = push },
            m, 1, 1,
        );
    }

    /// T-H7：LeakyReLU 反向 dst[i] = go[i] * (x[i]>=0 ? 1 : slope_bits 位模式)。
    /// 三 buffer（go/x/out），每元素一线程，grid gx = ceilDiv(n,256)。
    pub fn batchAddLeakyBwd(self: *Engine, go: u64, x: u64, out: u64, n: u32, slope_bits: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        const S = try self.getBuf(go);
        const M = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (S.bytes < @as(u64, n) * 4 or M.bytes < @as(u64, n) * 4 or O.bytes < @as(u64, n) * 4)
            return error.BufferTooSmall;
        const push = LeakyBwdPush{ .n = n, .slope_bits = slope_bits };
        const bufs = [_]*const buffer.Buffer{ S, M, O };
        try self.batchAppend(
            &self.kern_leaky_bwd, &bufs, @intCast(bufs.len - 1),
            .{ .leaky_bwd = push },
            ceilDiv(n, 256), 1, 1,
        );
    }

    /// sigmoid/tanh 反向：dst[i] = go[i] * f'(out[i])。三 buffer
    /// （go/out/dst），mode 0=sigmoid 1=tanh，每元素一线程。
    pub fn batchAddActivationBwd(self: *Engine, go: u64, out_v: u64, dst: u64, n: u32, mode: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        const S = try self.getBuf(go);
        const M = try self.getBuf(out_v);
        const O = try self.getBuf(dst);
        if (S.bytes < @as(u64, n) * 4 or M.bytes < @as(u64, n) * 4 or O.bytes < @as(u64, n) * 4)
            return error.BufferTooSmall;
        const push = ActivationBwdPush{ .n = n, .mode = mode };
        const bufs = [_]*const buffer.Buffer{ S, M, O };
        try self.batchAppend(
            &self.kern_activation_bwd, &bufs, @intCast(bufs.len - 1),
            .{ .activation_bwd = push },
            ceilDiv(n, 256), 1, 1,
        );
    }

    /// sigmoid/tanh 前向：out[i] = f(x[i])。两 buffer（x/out），
    /// mode 0=sigmoid 1=tanh，每元素一线程。
    pub fn batchAddActivationFwd(self: *Engine, x: u64, out: u64, n: u32, mode: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        if (X.bytes < @as(u64, n) * 4 or O.bytes < @as(u64, n) * 4)
            return error.BufferTooSmall;
        const push = ActivationFwdPush{ .n = n, .mode = mode };
        const bufs = [_]*const buffer.Buffer{ X, O };
        try self.batchAppend(
            &self.kern_activation_fwd, &bufs, @intCast(bufs.len - 1),
            .{ .activation_fwd = push },
            ceilDiv(n, 256), 1, 1,
        );
    }

    /// mul 反向：dst[i] = go[i] * x[i]（mode 0: x=b→ga；mode 1: x=a→gb）。
    pub fn batchAddMulBwd(self: *Engine, go: u64, x: u64, dst: u64, n: u32, mode: u32) !void {
        if (n == 0) return error.InvalidDimensions;
        const S = try self.getBuf(go);
        const M = try self.getBuf(x);
        const O = try self.getBuf(dst);
        if (S.bytes < @as(u64, n) * 4 or M.bytes < @as(u64, n) * 4 or O.bytes < @as(u64, n) * 4)
            return error.BufferTooSmall;
        const push = MulBwdPush{ .n = n, .mode = mode };
        const bufs = [_]*const buffer.Buffer{ S, M, O };
        try self.batchAppend(
            &self.kern_mul_bwd, &bufs, @intCast(bufs.len - 1),
            .{ .mul_bwd = push },
            ceilDiv(n, 256), 1, 1,
        );
    }

    /// axis=1 slice fwd/bwd。mode 0: out[B,C_out,T]=x[:,start:start+C_out,:]；
    /// mode 1: gx[B,C_in,T] scatter go（其余 0）。grid = 输出元素数。
    pub fn batchAddSliceBwd(self: *Engine, x: u64, out: u64, B: u32, C_in: u32, C_out: u32, T: u32, start: u32, mode: u32) !void {
        if (B == 0 or C_in == 0 or C_out == 0 or T == 0) return error.InvalidDimensions;
        const X = try self.getBuf(x);
        const O = try self.getBuf(out);
        const n_in: u64 = if (mode == 0) @as(u64, B) * C_in * T else @as(u64, B) * C_out * T;
        const n_out: u64 = if (mode == 0) @as(u64, B) * C_out * T else @as(u64, B) * C_in * T;
        if (X.bytes < n_in * 4 or O.bytes < n_out * 4) return error.BufferTooSmall;
        const push = SliceBwdPush{ .B = B, .C_in = C_in, .C_out = C_out, .T = T, .start = start, .mode = mode };
        const bufs = [_]*const buffer.Buffer{ X, O };
        try self.batchAppend(
            &self.kern_slice_bwd, &bufs, @intCast(bufs.len - 1),
            .{ .slice_bwd = push },
            ceilDiv(@intCast(n_out), 256), 1, 1,
        );
    }

    /// T18：GRU 单 kernel（双向并行，一 dispatch）。a=gx_all [2][T,3H]
    /// （fwd 半区 = x@w_ih.T+b_ih、rev 半区 = x@w_ih_r.T+b_ih_r）、
    /// b=w_hh [2][3H,H]（fwd/rev 拼接）、c=b_hh [2][3H]，out=[T,2H]
    /// （前 H=fwd、后 H=rev，同 rmvpe._bgru 输出布局）。grid=(1,1,2)
    /// 两方向各一 workgroup，kernel 内完成反向索引翻转。
    /// gating 反向（op42）：go[c 梯度] + a[gating 输入] + cond → g[a 梯度]。
    /// go/a/cond 走 a/b/c，out 走 p4（BatchEntry 绑定上限 4）。push 与 gating 同布局。
    pub fn batchAddGatingBwd(self: *Engine, go: u64, a: u64, cond: u64, out: u64, n: u32, lg: u32, h: u32, off: u32) !void {
        if (n == 0 or h == 0 or lg == 0) return error.InvalidDimensions;
        const GO = try self.getBuf(go);
        const A = try self.getBuf(a);
        const B = try self.getBuf(cond);
        const O = try self.getBuf(out);
        if (GO.bytes < @as(u64, n) * 4) return error.BufferTooSmall;
        if (A.bytes < @as(u64, n) * 2 * 4) return error.BufferTooSmall;
        if (O.bytes < @as(u64, n) * 2 * 4) return error.BufferTooSmall;
        if (B.bytes < (@as(u64, off) + 2 * @as(u64, h) * lg) * 4) return error.BufferTooSmall;
        const push = GatingBwdPush{ .n = n, .lg = lg, .h = h, .off = off };
        const bufs = [_]*const buffer.Buffer{ GO, A, B, O };
        try self.batchAppend(&self.kern_gating_bwd, &bufs, @intCast(bufs.len - 1), .{ .gating_bwd = push }, ceilDiv(n, 256), 1, 1);
    }

    pub fn batchAddGru(self: *Engine, a: u64, b: u64, c: u64, out: u64, t: u32, h: u32) !void {
        if (t == 0 or h == 0 or h > 256) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const C = try self.getBuf(c);
        const O = try self.getBuf(out);
        const k: u64 = 3 * @as(u64, h);
        if (A.bytes < 2 * @as(u64, t) * k * 4 or B.bytes < 2 * k * @as(u64, h) * 4 or
            C.bytes < 2 * k * 4 or O.bytes < @as(u64, t) * 2 * @as(u64, h) * 4)
            return error.BufferTooSmall;
        const push = GruPush{ .T = t, .H = h };
        const bufs = [_]*const buffer.Buffer{ A, B, C, O };
        try self.batchAppend(&self.kern_gru, &bufs, @intCast(bufs.len - 1), .{ .gru = push }, 1, 1, 2);
    }

    /// T18：GRU 单 kernel 单发路径（独立 submit，避开 batch 状态机）。
    /// 参数与 batchAddGru 一致；grid=(1,1,2) 双向并行，kernel 内翻转。
    pub fn gru(self: *Engine, a: u64, b: u64, c: u64, out: u64, t: u32, h: u32) !void {
        if (t == 0 or h == 0 or h > 256) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        const B = try self.getBuf(b);
        const C = try self.getBuf(c);
        const O = try self.getBuf(out);
        const k: u64 = 3 * @as(u64, h);
        if (A.bytes < 2 * @as(u64, t) * k * 4 or B.bytes < 2 * k * @as(u64, h) * 4 or
            C.bytes < 2 * k * 4 or O.bytes < @as(u64, t) * 2 * @as(u64, h) * 4)
            return error.BufferTooSmall;
        const push = GruPush{ .T = t, .H = h };
        const bufs = [_]*const buffer.Buffer{ A, B, C, O };
        try self.rec.begin();
        try self.rec.dispatch(&self.kern_gru, &bufs, @intCast(bufs.len - 1), &push, 1, 1, 2);
        try self.rec.endAndSubmit();
    }

    /// T18 诊断：纯 barrier 循环（步间同步成本）。a=out [1]（防删循环），
    /// T/H 仅用 T。grid=(1,1,1)。
    pub fn batchAddGruSync(self: *Engine, a: u64, t: u32, h: u32) !void {        if (t == 0 or h == 0) return error.InvalidDimensions;
        const A = try self.getBuf(a);
        if (A.bytes < 4) return error.BufferTooSmall;
        const push = GruPush{ .T = t, .H = h };
        const bufs = [_]*const buffer.Buffer{A};
        try self.batchAppend(&self.kern_gru_sync, &bufs, 0, .{ .gru = push }, 1, 1, 1);
    }

    /// Unified batch-op dispatcher — the single source for `rvc_batch_add`
    /// (FFI) and the training-graph executor (`graph.zig`). `p` mirrors the
    /// FFI's p0..p10 layout (see the op-code table above rvc_batch_add in
    /// ffi.zig). `out` is used by op 27+ (conv2d: too many params for
    /// p[0..11]); ops 1-26 ignore it (out lives in p9 as before).
    /// Invalid op → error.InvalidOp.
    pub fn batchAddOp(self: *Engine, op: i32, a: i64, b: i64, c: i64, out: i64, p: *const [11]i64) !void {
        const p0 = p[0];
        const p1 = p[1];
        const p2 = p[2];
        const p3 = p[3];
        const p4 = p[4];
        const p5 = p[5];
        const p6 = p[6];
        const p7 = p[7];
        const p8 = p[8];
        const p9 = p[9];
        const p10 = p[10];
        switch (op) {
            1 => {
                if (p0 < 0 or p1 < 0 or p2 < 0) return error.InvalidDimensions;
                try self.batchAddMatmul(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1), @intCast(p2),
                );
            },
            2 => {
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0)
                    return error.InvalidDimensions;
                try self.batchAddConv1d(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8),
                );
            },
            3 => {
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddAddInplace(@intCast(a), @intCast(b), @intCast(p0));
            },
            4 => {
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddMulInplace(@intCast(a), @intCast(b), @intCast(p0));
            },
            5 => {
                // conv_t1d：p0..p8 为 9 个维度参数，out 句柄放 p9（与 conv1d 一致）。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0)
                    return error.InvalidDimensions;
                try self.batchAddConvT1d(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8),
                );
            },
            6 => {
                // leaky_relu：a 就地；p0=n（≥0），p1=slope 的 f32 位模式。
                if (p0 < 0) return error.InvalidDimensions;
                const slope_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p1)))));
                try self.batchAddLeakyRelu(@intCast(a), @intCast(p0), slope_f);
            },
            7 => {
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddCopy(@intCast(a), @intCast(b), @intCast(p0));
            },
            8 => {
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddSoftmax(@intCast(a), @intCast(c), @intCast(p0), @intCast(p1));
            },
            9 => {
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                const eps_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p2)))));
                try self.batchAddLayerNorm(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), eps_f,
                );
            },
            10 => {
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddGelu(@intCast(a), @intCast(p0));
            },
            11 => {
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddBiasAdd(@intCast(a), @intCast(b), @intCast(c), @intCast(p0), @intCast(p1));
            },
            12 => {
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0) return error.InvalidDimensions;
                try self.batchAddAttnQk(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                );
            },
            13 => {
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0) return error.InvalidDimensions;
                try self.batchAddAttnSv(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                );
            },
            14 => {
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddRelu(@intCast(a), @intCast(p0));
            },
            15 => {
                if (p0 < 0 or p1 < 0 or p2 < 0) return error.InvalidDimensions;
                const eps_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p3)))));
                try self.batchAddGroupNorm(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), eps_f,
                );
            },
            16 => {
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0) return error.InvalidDimensions;
                try self.batchAddBandedAttnQk(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                );
            },
            17 => {
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0) return error.InvalidDimensions;
                try self.batchAddBandedAttnSv(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                );
            },
            18 => {
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0) return error.InvalidDimensions;
                try self.batchAddGating(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                );
            },
            19 => {
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                const scale_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p2)))));
                try self.batchAddTranspose(@intCast(a), @intCast(c), @intCast(p0), @intCast(p1), scale_f);
            },
            20 => {
                if (p0 < 0 or p1 < 0 or p2 < 0) return error.InvalidDimensions;
                try self.batchAddTransposeB(@intCast(a), @intCast(c), @intCast(p0), @intCast(p1), @intCast(p2));
            },
            21 => {
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddReduceRows(@intCast(a), @intCast(c), @intCast(p0), @intCast(p1));
            },
            22 => {
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddLeakyBwd(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1),
                );
            },
            23 => {
                // conv1d_groups：a=x b=w c=bias(0=无) p9=out；p0=B p1=C_in
                // p2=C_in_g p3=L p4=C_out p5=K p6=stride p7=pad_l p8=pad_r。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0)
                    return error.InvalidDimensions;
                try self.batchAddConv1dGroups(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8),
                );
            },
            24, 25, 26 => {
                // conv1d_groups_bwd（mode 0=gx 1=gw 2=gb）：a=x b=w c=go
                // p9=out；p0..p8 同 op23，p10=dilation。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0 or p10 < 0)
                    return error.InvalidDimensions;
                try self.batchAddConv1dGroupsBwdMode(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p9),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8), @intCast(p10), @intCast(op - 24),
                );
            },
            27 => {
                // conv2d（专用入口 rvc_batch_add_conv2d 并入图执行器）：
                // a=x b=w c=bias(0=无) out=out；p0=B p1=C_in p2=H p3=W
                // p4=C_out p5=KH p6=KW p7=pad_h p8=pad_w p9=stride_h
                // p10=stride_w。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0 or p9 < 0 or p10 < 0)
                    return error.InvalidDimensions;
                try self.batchAddConv2d(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(out),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8), @intCast(p9), @intCast(p10),
                );
            },
            28 => {
                // conv_t2d（conv2d bwd gx 并入图执行器，T3-c 后续）：
                // a=go b=w c=opad 打包（数值，非 buffer 句柄：低 32 位
                // opad_h、高 32 位 opad_w；bwd gx 无 bias 故 c 槽闲置复用）
                // out=out；p0=B p1=c_in(=go 通道) p2=oh p3=ow
                // p4=c_out(=输出通道) p5=kh p6=kw p7=sh p8=sw p9=ph
                // p10=pw；h_out/w_out 由几何公式推导（引擎 batchAddConvT2d
                // 内部校验同式，防参数错位）。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0 or p9 < 0 or p10 < 0)
                    return error.InvalidDimensions;
                const _opad_h: u32 = @truncate(@as(u64, @bitCast(c)));
                const _opad_w: u32 = @truncate(@as(u64, @bitCast(c)) >> 32);
                const _h_out: i64 = (p2 - 1) * p7 + p5 + @as(i64, _opad_h) - 2 * p9;
                const _w_out: i64 = (p3 - 1) * p8 + p6 + @as(i64, _opad_w) - 2 * p10;
                if (_h_out < 1 or _w_out < 1) return error.InvalidDimensions;
                try self.batchAddConvT2d(
                    @intCast(a), @intCast(b), 0, @intCast(out),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8), @intCast(p9), @intCast(p10),
                    _opad_h, _opad_w, @intCast(_h_out), @intCast(_w_out),
                );
            },
            29 => {
                // im2col_2d（conv2d bwd gw 并入图执行器，T3-c 后续）：
                // a=x c=pw（数值，非 buffer 句柄——12 维超 p[11] 槽位，
                // pw 恒 0 复用闲置 c 槽）out=out；p0=B p1=C p2=H p3=W
                // p4=OH p5=OW p6=KH p7=KW p8=sh p9=sw p10=ph。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0 or p6 < 0 or p7 < 0 or p8 < 0 or p9 < 0 or p10 < 0)
                    return error.InvalidDimensions;
                try self.batchAddIm2Col2d(
                    @intCast(a), @intCast(out),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                    @intCast(p4), @intCast(p5), @intCast(p6), @intCast(p7),
                    @intCast(p8), @intCast(p9), @intCast(p10), @intCast(c),
                );
            },
            30 => {
                // sqrt_inplace：a 就地；p0=n。
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddSqrtInplace(@intCast(a), @intCast(p0));
            },
            31 => {
                // div_const：a 就地；p0=n，p1=s（f32 位模式）。
                if (p0 < 0) return error.InvalidDimensions;
                const s_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p1)))));
                try self.batchAddDivConst(@intCast(a), @intCast(p0), s_f);
            },
            32 => {
                // mul_const：a 就地；p0=n，p1=s（f32 位模式）。
                if (p0 < 0) return error.InvalidDimensions;
                const s_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p1)))));
                try self.batchAddMulConst(@intCast(a), @intCast(p0), s_f);
            },
            33 => {
                // madd_const：a 就地 + b 只读；p0=n，p1=s（f32 位模式）。
                if (p0 < 0) return error.InvalidDimensions;
                const s_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p1)))));
                try self.batchAddMaddConst(@intCast(a), @intCast(b), @intCast(p0), s_f);
            },
            34 => {
                // sub_inplace：a 就地 + b 只读；p0=n。
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddSubInplace(@intCast(a), @intCast(b), @intCast(p0));
            },
            35 => {
                // add_const：a 就地；p0=n，p1=s（f32 位模式）。
                if (p0 < 0) return error.InvalidDimensions;
                const s_f: f32 = @bitCast(@as(u32, @truncate(@as(u64, @bitCast(p1)))));
                try self.batchAddAddConst(@intCast(a), @intCast(p0), s_f);
            },
            36 => {
                // mul_buf_scalar：a 就地 + b（1 元素标量槽）只读；p0=n。
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddMulBufScalar(@intCast(a), @intCast(b), @intCast(p0));
            },
            37 => {
                // rcp_inplace：a 就地；p0=n。
                if (p0 < 0) return error.InvalidDimensions;
                try self.batchAddRcpInplace(@intCast(a), @intCast(p0));
            },
            38 => {
                // activation_bwd：a=go b=out c=dst；p0=n p1=mode(0=sigmoid 1=tanh)。
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddActivationBwd(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1),
                );
            },
            39 => {
                // activation_fwd：a=x c=out；p0=n p1=mode(0=sigmoid 1=tanh)。
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddActivationFwd(
                    @intCast(a), @intCast(c),
                    @intCast(p0), @intCast(p1),
                );
            },
            40 => {
                // mul_bwd：a=go b=x c=dst；p0=n p1=mode(0=ga 1=gb)。
                if (p0 < 0 or p1 < 0) return error.InvalidDimensions;
                try self.batchAddMulBwd(
                    @intCast(a), @intCast(b), @intCast(c),
                    @intCast(p0), @intCast(p1),
                );
            },
            41 => {
                // slice_bwd：a=x b=out；p0=B p1=C_in p2=C_out p3=T p4=start p5=mode。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or p5 < 0) return error.InvalidDimensions;
                try self.batchAddSliceBwd(
                    @intCast(a), @intCast(b),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3), @intCast(p4), @intCast(p5),
                );
            },
            42 => {
                // gating_bwd：a=go b=a(输入) c=cond；out 句柄走 p4。
                // p0=n p1=lg p2=h p3=off p4=g(a 梯度 buffer)。
                if (p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0) return error.InvalidDimensions;
                try self.batchAddGatingBwd(
                    @intCast(a), @intCast(b), @intCast(c), @intCast(p4),
                    @intCast(p0), @intCast(p1), @intCast(p2), @intCast(p3),
                );
            },
            else => return error.InvalidOp,
        }
    }

    /// Submit every recorded op as ONE command buffer + ONE fence-wait,
    /// then clear the batch. Empty batches succeed trivially. On error
    /// the batch is left intact for discard/retry.
    pub fn batchCommit(self: *Engine) !void {
        try self.batchCommitInternal(true);
    }

    /// Submit every recorded op as ONE command buffer WITHOUT waiting
    /// for the GPU, then clear the batch. The submission is parked in
    /// the recorder's in-flight frame; call `batchWait` (or any later
    /// commit) to block until it completes. The recorder keeps up to
    /// MAX_INFLIGHT frames, so callers can pipeline several independent
    /// batches before a single `batchWait` — this removes one
    /// submit+wait round-trip per batch (the core P1 bottleneck on AMD
    /// Radeon Pro VII: ~6-13 ms fixed cost per dispatch).
    ///
    /// Semantics: the queue executes submissions in order, so a later
    /// async batch reading an earlier batch's output is safe even
    /// without an explicit wait; `batchWait` is only needed before
    /// reading results on the host (download) or freeing buffers.
    pub fn batchCommitAsync(self: *Engine) !void {
        try self.batchCommitInternal(false);
    }

    fn batchCommitInternal(self: *Engine, wait: bool) !void {
        if (self.batch_ops.items.len == 0) return;
        try self.rec.begin();
        for (self.batch_ops.items) |e| {
            const all_zero = e.view_offs[0] == 0 and e.view_offs[1] == 0 and
                e.view_offs[2] == 0 and e.view_offs[3] == 0;
            if (all_zero) {
                try self.rec.dispatch(e.kern, e.bufs[0..e.nbufs], e.write_idx, &e.push, e.gx, e.gy, e.gz);
            } else {
                try self.rec.dispatchView(
                    e.kern, e.bufs[0..e.nbufs], e.write_idx, &e.push,
                    e.gx, e.gy, e.gz, e.view_offs[0..e.nbufs],
                );
            }
        }
        if (wait) {
            try self.rec.endAndSubmit();
        } else {
            try self.rec.endAndSubmitAsync();
        }
        self.batch_ops.clearRetainingCapacity();
    }

    /// Block until every in-flight (async-committed) submission has
    /// completed. No-op when nothing is in flight. All GPU writes are
    /// visible to the host afterwards.
    pub fn batchWait(self: *Engine) !void {
        try self.rec.waitAll();
    }

    /// Discard the pending batch without submitting (idempotent).
    pub fn batchDiscard(self: *Engine) void {
        self.batch_ops.clearRetainingCapacity();
    }

    /// RVC_TS=1 measure-only: accumulated GPU-busy ns + submit count.
    pub fn tsStats(self: *const Engine) recorder.Recorder.TsStats {
        return self.rec.tsStats();
    }

    /// RVC_TS=1 measure-only: reset busy/submit counters.
    pub fn tsReset(self: *Engine) void {
        self.rec.tsReset();
    }
};


/// P0-2: load (or create empty) pipeline cache from disk; caller owns.
fn loadPipelineCache(device: vk.c.VkDevice) !vk.c.VkPipelineCache {
    const path: []const u8 = "rvc_core.cache";
    var pcci = std.mem.zeroes(vk.c.VkPipelineCacheCreateInfo);
    pcci.sType = vk.c.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    var data_buf: []u8 = &[_]u8{};
    if (std.fs.cwd().readFileAlloc(std.heap.page_allocator, path, 64 * 1024 * 1024)) |data| {
        data_buf = data;
        pcci.initialDataSize = data.len;
        pcci.pInitialData = data.ptr;
    } else |_| {
        data_buf = &[_]u8{};
    }
    defer if (data_buf.len > 0) std.heap.page_allocator.free(data_buf);
    var cache: vk.c.VkPipelineCache = null;
    try vk.check(vk.c.vkCreatePipelineCache(device, &pcci, null, &cache));
    return cache;
}

/// P0-2: persist pipeline cache to disk (best-effort, ignore errors).
fn savePipelineCache(device: vk.c.VkDevice, cache: vk.c.VkPipelineCache) void {
    var size: usize = 0;
    if (vk.c.vkGetPipelineCacheData(device, cache, &size, null) != vk.c.VK_SUCCESS) return;
    if (size == 0 or size > 64 * 1024 * 1024) return;
    const buf = std.heap.page_allocator.alloc(u8, size) catch return;
    defer std.heap.page_allocator.free(buf);
    if (vk.c.vkGetPipelineCacheData(device, cache, &size, buf.ptr) != vk.c.VK_SUCCESS) return;
    for (buf[0..size]) |*b| b.* = b.*; // ensure writable slice
    const f = std.fs.cwd().createFile("rvc_core.cache", .{ .truncate = true }) catch return;
    defer f.close();
    f.writeAll(buf[0..size]) catch {};
}
