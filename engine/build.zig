const std = @import("std");

// Mirrors the active build mode for the duration of `build()`. glslc's
// -O flag: -O0 in Debug, full optimisation otherwise. File-scope, like
// valkyr's build.zig — build scripts are single-threaded.
var g_glslc_opt: []const u8 = "-O0";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    g_glslc_opt = switch (optimize) {
        .Debug => "-O0",
        .ReleaseSafe, .ReleaseFast, .ReleaseSmall => "-O",
    };

    // ── GLSL → SPIR-V ──
    // elementwise.comp is a template stamped into three kernels via
    // `-DRVC_OP=<n>` (1=add, 2=mul, 3=relu). matmul.comp and conv1d.comp
    // are templates stamped into three kernels via `-DRVC_TILE=<n>`
    // (16|32|64) — the host picks the tile per shape (see engine.zig
    // pickTile / pickConvTile). The remaining shaders are standalone.
    const matmul16_spv = compileShaderD(b, "matmul", "matmul16", &.{ "RVC_TILE=16" });
    const matmul32_spv = compileShaderD(b, "matmul", "matmul32", &.{ "RVC_TILE=32" });
    const matmul64_spv = compileShaderD(b, "matmul", "matmul64", &.{ "RVC_TILE=64" });
    const matmul_f16_16_spv = compileShaderD(b, "matmul_f16", "matmul_f16_16", &.{ "RVC_TILE=16" });
    const matmul_f16_32_spv = compileShaderD(b, "matmul_f16", "matmul_f16_32", &.{ "RVC_TILE=32" });
    const matmul_f16_64_spv = compileShaderD(b, "matmul_f16", "matmul_f16_64", &.{ "RVC_TILE=64" });
    const conv1d16_spv = compileShaderD(b, "conv1d", "conv1d16", &.{ "RVC_TILE=16" });
    const conv1d32_spv = compileShaderD(b, "conv1d", "conv1d32", &.{ "RVC_TILE=32" });
    const conv1d64_spv = compileShaderD(b, "conv1d", "conv1d64", &.{ "RVC_TILE=64" });
    const add_spv = compileShaderD(b, "elementwise", "add", &.{ "RVC_OP=1" });
    const mul_spv = compileShaderD(b, "elementwise", "mul", &.{ "RVC_OP=2" });
    const relu_spv = compileShaderD(b, "elementwise", "relu", &.{ "RVC_OP=3" });
    // conv_t1d TILE 化（conv_t1d.comp 是 conv1d.comp 同款模板，RVC_TILE=16|32|64）；
    // conv_t1d_naive.comp 为原朴素逐点 kernel（RVC_CONV_T1D_TILE=0 回退开关）。
    const conv_t1d16_spv = compileShaderD(b, "conv_t1d", "conv_t1d16", &.{ "RVC_TILE=16" });
    const conv_t1d32_spv = compileShaderD(b, "conv_t1d", "conv_t1d32", &.{ "RVC_TILE=32" });
    const conv_t1d64_spv = compileShaderD(b, "conv_t1d", "conv_t1d64", &.{ "RVC_TILE=64" });
    const conv_t1d_naive_spv = compileShaderD(b, "conv_t1d_naive", "conv_t1d_naive", &.{});
    // conv_t2d TILE 化（conv_t2d.comp 是 conv_t1d.comp 的 2D 泛化，RVC_TILE=16|32|64）。
    // 无 naive 变体：RVC_CONV_T2D_TILE=0 的回退语义（走 Python numpy）由 vulkan_ops
    // 接线层处理，引擎层只实现 TILE 16/32/64。
    const conv_t2d16_spv = compileShaderD(b, "conv_t2d", "conv_t2d16", &.{ "RVC_TILE=16" });
    const conv_t2d32_spv = compileShaderD(b, "conv_t2d", "conv_t2d32", &.{ "RVC_TILE=32" });
    const conv_t2d64_spv = compileShaderD(b, "conv_t2d", "conv_t2d64", &.{ "RVC_TILE=64" });
    const conv2d16_spv = compileShaderD(b, "conv2d", "conv2d16", &.{ "RVC_TILE=16" });
    const conv2d32_spv = compileShaderD(b, "conv2d", "conv2d32", &.{ "RVC_TILE=32" });
    const conv2d64_spv = compileShaderD(b, "conv2d", "conv2d64", &.{ "RVC_TILE=64" });
    const conv1d_groups_spv = compileShaderD(b, "conv1d_groups", "conv1d_groups", &.{});
    const conv1d_groups_bwd_spv = compileShaderD(b, "conv1d_groups_bwd", "conv1d_groups_bwd", &.{});
    const embed_spv = compileShaderD(b, "embed", "embed", &.{});
    const insert_zeros_2x_spv = compileShaderD(b, "insert_zeros_2x", "insert_zeros_2x", &.{});
    const im2col_1d_spv = compileShaderD(b, "im2col_1d", "im2col_1d", &.{});
    const im2col_2d_spv = compileShaderD(b, "im2col_2d", "im2col_2d", &.{});
    const add_inplace_spv = compileShaderD(b, "add_inplace", "add_inplace", &.{});
    const mul_inplace_spv = compileShaderD(b, "mul_inplace", "mul_inplace", &.{});
    const leaky_relu_spv = compileShaderD(b, "leaky_relu", "leaky_relu", &.{});
    // T4-2 AdamW 标量元素算子（op30-37）：
    const sqrt_inplace_spv = compileShaderD(b, "sqrt_inplace", "sqrt_inplace", &.{});
    const rcp_inplace_spv = compileShaderD(b, "rcp_inplace", "rcp_inplace", &.{});
    const div_const_spv = compileShaderD(b, "div_const", "div_const", &.{});
    const mul_const_spv = compileShaderD(b, "mul_const", "mul_const", &.{});
    const madd_const_spv = compileShaderD(b, "madd_const", "madd_const", &.{});
    const sub_inplace_spv = compileShaderD(b, "sub_inplace", "sub_inplace", &.{});
    const add_const_spv = compileShaderD(b, "add_const", "add_const", &.{});
    const mul_buf_scalar_spv = compileShaderD(b, "mul_buf_scalar", "mul_buf_scalar", &.{});
    const copy_spv = compileShaderD(b, "copy", "copy", &.{});
    const layernorm_spv = compileShaderD(b, "layernorm", "layernorm", &.{});
    const softmax_spv = compileShaderD(b, "softmax", "softmax", &.{});
    const rmsnorm_spv = compileShaderD(b, "rmsnorm", "rmsnorm", &.{});
    const gelu_spv = compileShaderD(b, "gelu", "gelu", &.{});
    const bias_add_spv = compileShaderD(b, "bias_add", "bias_add", &.{});
    const attn_qk_spv = compileShaderD(b, "attn_qk", "attn_qk", &.{});
    const attn_sv_spv = compileShaderD(b, "attn_sv", "attn_sv", &.{});
    const banded_attn_qk_spv = compileShaderD(b, "banded_attn_qk", "banded_attn_qk", &.{});
    const banded_attn_sv_spv = compileShaderD(b, "banded_attn_sv", "banded_attn_sv", &.{});
    const gn_spv = compileShaderD(b, "gn", "gn", &.{});
    const gating_spv = compileShaderD(b, "gating", "gating", &.{});
    const gating_bwd_spv = compileShaderD(b, "gating_bwd", "gating_bwd", &.{});
    const transpose_spv = compileShaderD(b, "transpose", "transpose", &.{});
    const transpose_b_spv = compileShaderD(b, "transpose_b", "transpose_b", &.{});
    const reduce_rows_spv = compileShaderD(b, "reduce_rows", "reduce_rows", &.{});
    const leaky_bwd_spv = compileShaderD(b, "leaky_bwd", "leaky_bwd", &.{});
    const activation_bwd_spv = compileShaderD(b, "activation_bwd", "activation_bwd", &.{});
    const activation_fwd_spv = compileShaderD(b, "activation_fwd", "activation_fwd", &.{});
    const mul_bwd_spv = compileShaderD(b, "mul_bwd", "mul_bwd", &.{});
    const slice_bwd_spv = compileShaderD(b, "slice_bwd", "slice_bwd", &.{});
    const gru_spv = compileShaderD(b, "gru", "gru", &.{});
    const gru_sync_spv = compileShaderD(b, "gru_sync", "gru_sync", &.{});

    // Stage compiled SPIR-V into one anonymous module so `@embedFile`
    // can reach it. SPIR-V must be 4-byte aligned for Vulkan's pCode —
    // the align(4) on each `@embedFile(...).*` materialises the bytes
    // at a u32-aligned address (same trick as valkyr's build.zig).
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(matmul16_spv, "matmul16.spv");
    _ = wf.addCopyFile(matmul32_spv, "matmul32.spv");
    _ = wf.addCopyFile(matmul64_spv, "matmul64.spv");
    _ = wf.addCopyFile(matmul_f16_16_spv, "matmul_f16_16.spv");
    _ = wf.addCopyFile(matmul_f16_32_spv, "matmul_f16_32.spv");
    _ = wf.addCopyFile(matmul_f16_64_spv, "matmul_f16_64.spv");
    _ = wf.addCopyFile(conv1d16_spv, "conv1d16.spv");
    _ = wf.addCopyFile(conv1d32_spv, "conv1d32.spv");
    _ = wf.addCopyFile(conv1d64_spv, "conv1d64.spv");
    _ = wf.addCopyFile(add_spv, "add.spv");
    _ = wf.addCopyFile(mul_spv, "mul.spv");
    _ = wf.addCopyFile(relu_spv, "relu.spv");
    _ = wf.addCopyFile(conv_t1d16_spv, "conv_t1d16.spv");
    _ = wf.addCopyFile(conv_t1d32_spv, "conv_t1d32.spv");
    _ = wf.addCopyFile(conv_t1d64_spv, "conv_t1d64.spv");
    _ = wf.addCopyFile(conv_t1d_naive_spv, "conv_t1d_naive.spv");
    _ = wf.addCopyFile(conv_t2d16_spv, "conv_t2d16.spv");
    _ = wf.addCopyFile(conv_t2d32_spv, "conv_t2d32.spv");
    _ = wf.addCopyFile(conv_t2d64_spv, "conv_t2d64.spv");
    _ = wf.addCopyFile(conv2d16_spv, "conv2d16.spv");
    _ = wf.addCopyFile(conv2d32_spv, "conv2d32.spv");
    _ = wf.addCopyFile(conv2d64_spv, "conv2d64.spv");
    _ = wf.addCopyFile(conv1d_groups_spv, "conv1d_groups.spv");
    _ = wf.addCopyFile(conv1d_groups_bwd_spv, "conv1d_groups_bwd.spv");
    _ = wf.addCopyFile(embed_spv, "embed.spv");
    _ = wf.addCopyFile(insert_zeros_2x_spv, "insert_zeros_2x.spv");
    _ = wf.addCopyFile(im2col_1d_spv, "im2col_1d.spv");
    _ = wf.addCopyFile(im2col_2d_spv, "im2col_2d.spv");
    _ = wf.addCopyFile(add_inplace_spv, "add_inplace.spv");
    _ = wf.addCopyFile(mul_inplace_spv, "mul_inplace.spv");
    _ = wf.addCopyFile(leaky_relu_spv, "leaky_relu.spv");
    _ = wf.addCopyFile(sqrt_inplace_spv, "sqrt_inplace.spv");
    _ = wf.addCopyFile(rcp_inplace_spv, "rcp_inplace.spv");
    _ = wf.addCopyFile(div_const_spv, "div_const.spv");
    _ = wf.addCopyFile(mul_const_spv, "mul_const.spv");
    _ = wf.addCopyFile(madd_const_spv, "madd_const.spv");
    _ = wf.addCopyFile(sub_inplace_spv, "sub_inplace.spv");
    _ = wf.addCopyFile(add_const_spv, "add_const.spv");
    _ = wf.addCopyFile(mul_buf_scalar_spv, "mul_buf_scalar.spv");
    _ = wf.addCopyFile(copy_spv, "copy.spv");
    _ = wf.addCopyFile(layernorm_spv, "layernorm.spv");
    _ = wf.addCopyFile(softmax_spv, "softmax.spv");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    _ = wf.addCopyFile(gelu_spv, "gelu.spv");
    _ = wf.addCopyFile(bias_add_spv, "bias_add.spv");
    _ = wf.addCopyFile(attn_qk_spv, "attn_qk.spv");
    _ = wf.addCopyFile(attn_sv_spv, "attn_sv.spv");
    _ = wf.addCopyFile(banded_attn_qk_spv, "banded_attn_qk.spv");
    _ = wf.addCopyFile(banded_attn_sv_spv, "banded_attn_sv.spv");
    _ = wf.addCopyFile(gn_spv, "gn.spv");
    _ = wf.addCopyFile(gating_spv, "gating.spv");
    _ = wf.addCopyFile(gating_bwd_spv, "gating_bwd.spv");
    _ = wf.addCopyFile(transpose_spv, "transpose.spv");
    _ = wf.addCopyFile(transpose_b_spv, "transpose_b.spv");
    _ = wf.addCopyFile(reduce_rows_spv, "reduce_rows.spv");
    _ = wf.addCopyFile(leaky_bwd_spv, "leaky_bwd.spv");
    _ = wf.addCopyFile(activation_bwd_spv, "activation_bwd.spv");
    _ = wf.addCopyFile(activation_fwd_spv, "activation_fwd.spv");
    _ = wf.addCopyFile(mul_bwd_spv, "mul_bwd.spv");
    _ = wf.addCopyFile(slice_bwd_spv, "slice_bwd.spv");
    _ = wf.addCopyFile(gru_spv, "gru.spv");
    _ = wf.addCopyFile(gru_sync_spv, "gru_sync.spv");
    const shader_mod = wf.add("shaders.zig",
        \\pub const matmul16 align(4) = @embedFile("matmul16.spv").*;
        \\pub const matmul32 align(4) = @embedFile("matmul32.spv").*;
        \\pub const matmul64 align(4) = @embedFile("matmul64.spv").*;
        \\pub const matmul_f16_16 align(4) = @embedFile("matmul_f16_16.spv").*;
        \\pub const matmul_f16_32 align(4) = @embedFile("matmul_f16_32.spv").*;
        \\pub const matmul_f16_64 align(4) = @embedFile("matmul_f16_64.spv").*;
        \\pub const conv1d16 align(4) = @embedFile("conv1d16.spv").*;
        \\pub const conv1d32 align(4) = @embedFile("conv1d32.spv").*;
        \\pub const conv1d64 align(4) = @embedFile("conv1d64.spv").*;
        \\pub const conv1d_groups align(4) = @embedFile("conv1d_groups.spv").*;
        \\pub const conv1d_groups_bwd align(4) = @embedFile("conv1d_groups_bwd.spv").*;
        \\pub const add align(4) = @embedFile("add.spv").*;
        \\pub const mul align(4) = @embedFile("mul.spv").*;
        \\pub const relu align(4) = @embedFile("relu.spv").*;
        \\pub const conv_t1d16 align(4) = @embedFile("conv_t1d16.spv").*;
        \\pub const conv_t1d32 align(4) = @embedFile("conv_t1d32.spv").*;
        \\pub const conv_t1d64 align(4) = @embedFile("conv_t1d64.spv").*;
        \\pub const conv_t1d_naive align(4) = @embedFile("conv_t1d_naive.spv").*;
        \\pub const conv_t2d16 align(4) = @embedFile("conv_t2d16.spv").*;
        \\pub const conv_t2d32 align(4) = @embedFile("conv_t2d32.spv").*;
        \\pub const conv_t2d64 align(4) = @embedFile("conv_t2d64.spv").*;
        \\pub const conv2d16 align(4) = @embedFile("conv2d16.spv").*;
        \\pub const conv2d32 align(4) = @embedFile("conv2d32.spv").*;
        \\pub const conv2d64 align(4) = @embedFile("conv2d64.spv").*;
        \\pub const embed align(4) = @embedFile("embed.spv").*;
        \\pub const insert_zeros_2x align(4) = @embedFile("insert_zeros_2x.spv").*;
        \\pub const im2col_1d align(4) = @embedFile("im2col_1d.spv").*;
        \\pub const im2col_2d align(4) = @embedFile("im2col_2d.spv").*;
        \\pub const add_inplace align(4) = @embedFile("add_inplace.spv").*;
        \\pub const mul_inplace align(4) = @embedFile("mul_inplace.spv").*;
        \\pub const leaky_relu align(4) = @embedFile("leaky_relu.spv").*;
        \\pub const sqrt_inplace align(4) = @embedFile("sqrt_inplace.spv").*;
        \\pub const rcp_inplace align(4) = @embedFile("rcp_inplace.spv").*;
        \\pub const div_const align(4) = @embedFile("div_const.spv").*;
        \\pub const mul_const align(4) = @embedFile("mul_const.spv").*;
        \\pub const madd_const align(4) = @embedFile("madd_const.spv").*;
        \\pub const sub_inplace align(4) = @embedFile("sub_inplace.spv").*;
        \\pub const add_const align(4) = @embedFile("add_const.spv").*;
        \\pub const mul_buf_scalar align(4) = @embedFile("mul_buf_scalar.spv").*;
        \\pub const copy align(4) = @embedFile("copy.spv").*;
        \\pub const layernorm align(4) = @embedFile("layernorm.spv").*;
        \\pub const softmax align(4) = @embedFile("softmax.spv").*;
        \\pub const rmsnorm align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const gelu align(4) = @embedFile("gelu.spv").*;
        \\pub const bias_add align(4) = @embedFile("bias_add.spv").*;
        \\pub const attn_qk align(4) = @embedFile("attn_qk.spv").*;
        \\pub const attn_sv align(4) = @embedFile("attn_sv.spv").*;
        \\pub const banded_attn_qk align(4) = @embedFile("banded_attn_qk.spv").*;
        \\pub const banded_attn_sv align(4) = @embedFile("banded_attn_sv.spv").*;
        \\pub const gn align(4) = @embedFile("gn.spv").*;
        \\pub const gating align(4) = @embedFile("gating.spv").*;
        \\pub const transpose align(4) = @embedFile("transpose.spv").*;
        \\pub const transpose_b align(4) = @embedFile("transpose_b.spv").*;
        \\pub const reduce_rows align(4) = @embedFile("reduce_rows.spv").*;
        \\pub const leaky_bwd align(4) = @embedFile("leaky_bwd.spv").*;
        \\pub const activation_bwd align(4) = @embedFile("activation_bwd.spv").*;
        \\pub const activation_fwd align(4) = @embedFile("activation_fwd.spv").*;
        \\pub const mul_bwd align(4) = @embedFile("mul_bwd.spv").*;
        \\pub const slice_bwd align(4) = @embedFile("slice_bwd.spv").*;
        \\pub const gating_bwd align(4) = @embedFile("gating_bwd.spv").*;
        \\pub const gru align(4) = @embedFile("gru.spv").*;
        \\pub const gru_sync align(4) = @embedFile("gru_sync.spv").*;
    );

    // ── Vulkan SDK resolution (mirrors valkyr build.zig:350-440) ──
    var vulkan_sdk_path: []const u8 = "";
    var vulkan_sdk_owned = false;
    defer if (vulkan_sdk_owned) b.allocator.free(vulkan_sdk_path);
    if (std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK")) |s| {
        vulkan_sdk_path = s;
        vulkan_sdk_owned = true;
    } else |_| {
        const candidates = [_][]const u8{
            "C:\\VulkanSDK",
            "C:\\VulkanSDK\\1.4.304.0",
            "C:\\VulkanSDK\\1.3.296.0",
        };
        for (candidates) |cand| {
            if (std.fs.cwd().access(b.fmt("{s}/Include/vulkan/vulkan.h", .{cand}), .{})) |_| {
                vulkan_sdk_path = cand;
                break;
            } else |_| {}
        }
        if (vulkan_sdk_path.len == 0) {
            std.debug.print(
                "warning: VULKAN_SDK is unset and no SDK was found in the usual " ++
                    "locations; falling back to 'C:\\VulkanSDK'.\n",
                .{},
            );
            vulkan_sdk_path = "C:\\VulkanSDK";
        }
    }

    // ── Shared library: rvc_core.dll ──
    const lib = b.addSharedLibrary(.{
        .name = "rvc_core",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.root_module.addAnonymousImport("shaders", .{
        .root_source_file = shader_mod,
    });
    lib.linkLibC();

    if (target.result.os.tag == .windows) {
        lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/Include", .{vulkan_sdk_path}) });
        lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/Lib", .{vulkan_sdk_path}) });
        lib.linkSystemLibrary("vulkan-1");
    } else {
        lib.linkSystemLibrary("vulkan");
    }

    b.installArtifact(lib);

    // ── `zig build test` — shell out to the Python self-test ──
    const test_cmd = b.addSystemCommand(&.{ "python", "test_ffi.py" });
    test_cmd.setCwd(b.path("."));
    test_cmd.step.dependOn(b.getInstallStep());
    const test_step = b.step("test", "Run test_ffi.py against the built DLL");
    test_step.dependOn(&test_cmd.step);
}

/// Compile a shader with optional `-D` defines under a separate output
/// name. `src_name` selects the .comp source; `out_name` becomes the
/// .spv filename. Depfile support so edits to #included files trigger
/// a recompile.
fn compileShaderD(b: *std.Build, src_name: []const u8, out_name: []const u8, defines: []const []const u8) std.Build.LazyPath {
    const src = b.fmt("shaders/{s}.comp", .{src_name});
    const spv = b.fmt("{s}.spv", .{out_name});
    const dep = b.fmt("{s}.d", .{out_name});
    const glslc = findGlslc(b);
    const cmd = b.addSystemCommand(&.{ glslc, "--target-env=vulkan1.3", g_glslc_opt });
    cmd.addArg("-I");
    cmd.addDirectoryArg(b.path("shaders"));
    for (defines) |def| {
        cmd.addArg(b.fmt("-D{s}", .{def}));
    }
    cmd.addArg("-MD");
    cmd.addArg("-MF");
    _ = cmd.addDepFileOutputArg(dep);
    cmd.addFileArg(b.path(src));
    cmd.addArg("-o");
    return cmd.addOutputFileArg(spv);
}

/// Locate glslc: prefer VULKAN_SDK/Bin/glslc.exe, else PATH ("glslc").
fn findGlslc(b: *std.Build) []const u8 {
    if (std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK")) |sdk| {
        defer b.allocator.free(sdk);
        if (sdk.len > 0) {
            const full = b.fmt("{s}/Bin/glslc.exe", .{sdk});
            if (std.fs.cwd().access(full, .{})) |_| {
                return full;
            } else |_| {}
        }
    } else |_| {}
    return "glslc";
}
