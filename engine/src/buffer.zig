//! Storage-buffer lifecycle helper (clipped from valkyr-engine's
//! src/gpu/buffer.zig). Two modes:
//!
//!   static      — DEVICE_LOCAL, populated from host via a transient
//!                 staging copy. Use for weights / inputs.
//!   device_only — DEVICE_LOCAL, written by the GPU (outputs).
//!
//! Both carry TRANSFER_SRC|TRANSFER_DST so uploads, downloads and
//! fill-zero all work through vkCmdCopyBuffer / vkCmdFillBuffer.

const std = @import("std");
const vk = @import("vk.zig");
const c = vk.c;

pub const Mode = enum { static, device_only };

pub const Buffer = struct {
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
    /// Allocated capacity in bytes.
    bytes: usize,
    mode: Mode,
    /// Byte offset within `memory` this buffer is bound at (suballocator
    /// path; 0 for the regular per-buffer vkAllocateMemory path).
    offset: usize = 0,
    /// True when this buffer's memory lives in a shared suballocator chunk
    /// (deinit must NOT vkFreeMemory — the chunk owns it).
    sub: bool = false,

    /// DEVICE_LOCAL storage buffer, TRANSFER_SRC|DST enabled, not
    /// zero-filled (caller uploads before use).
    pub fn initStatic(ctx: *vk.Context, capacity_bytes: usize) !Buffer {
        const total = @max(capacity_bytes, 16);
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
        return .{ .handle = raw.handle, .memory = raw.memory, .bytes = total, .mode = .static };
    }

    /// DEVICE_LOCAL storage buffer, zero-filled via vkCmdFillBuffer.
    pub fn initDeviceOnly(ctx: *vk.Context, capacity_bytes: usize) !Buffer {
        const total = @max(capacity_bytes, 16);
        const raw = try createBuffer(
            ctx,
            total,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
        errdefer {
            c.vkDestroyBuffer(ctx.device, raw.handle, null);
            c.vkFreeMemory(ctx.device, raw.memory, null);
        }
        try gpuFill(ctx, raw.handle, total);
        return .{ .handle = raw.handle, .memory = raw.memory, .bytes = total, .mode = .device_only };
    }

    /// DEVICE_LOCAL storage buffer **inside a shared suballocator chunk**
    /// (J25：训练输出 buffer 高频 mem_alloc，节省每次 vkAllocateMemory)。
    /// `memory`/`offset_hint` come from the suballocator's chunk; the buffer
    /// is created and bound at the next offset aligned to the memory type's
    /// buffer alignment. Caller (Engine.memAlloc) must record the **actual**
    /// used offset (== returned `offset`) back into its free-list bookkeeping
    /// so the waste gap stays reserved. Zero-filled via vkCmdFillBuffer like
    /// initDeviceOnly (F3-B).
    pub fn initDeviceOnlySub(
        ctx: *vk.Context,
        capacity_bytes: usize,
        memory: c.VkDeviceMemory,
        offset_hint: usize,
    ) !Buffer {
        const total = @max(capacity_bytes, 16);
        var bci = std.mem.zeroes(c.VkBufferCreateInfo);
        bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = @intCast(total);
        bci.usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            c.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;

        var handle: c.VkBuffer = null;
        try vk.check(c.vkCreateBuffer(ctx.device, &bci, null, &handle));
        errdefer c.vkDestroyBuffer(ctx.device, handle, null);

        var req: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(ctx.device, handle, &req);
        // Align to the memory type's alignment requirement; the gap stays
        // reserved (caller tracks the actual offset).
        const off = alignForward(offset_hint, @as(usize, @intCast(req.alignment)));
        try vk.check(c.vkBindBufferMemory(ctx.device, handle, memory, @intCast(off)));

        try gpuFill(ctx, handle, total);
        return .{
            .handle = handle,
            .memory = memory,
            .bytes = total,
            .mode = .device_only,
            .offset = off,
            .sub = true,
        };
    }

    fn alignForward(addr: usize, alignment: usize) usize {
        if (alignment <= 1) return addr;
        const rem = addr % alignment;
        return if (rem == 0) addr else addr + (alignment - rem);
    }

    /// Upload raw bytes into the buffer via a transient staging copy.
    /// Init- or test-time only — pays a vkQueueWaitIdle-equivalent fence
    /// wait (D1a-2: 复用 ctx 常驻 staging，不再每次 create/free）。
    pub fn upload(self: *Buffer, ctx: *vk.Context, data: []const u8) !void {
        if (data.len == 0) return;
        if (data.len > self.bytes) return error.UploadTooLarge;
        try stagingUpload(ctx, self.handle, data);
    }

    /// Download raw bytes from the buffer back to host via a transient
    /// staging copy. Init- or test-time only.
    pub fn download(self: *const Buffer, ctx: *vk.Context, dst: []u8) !void {
        const want_bytes = dst.len;
        if (want_bytes == 0) return;
        if (want_bytes > self.bytes) return error.ReadBackTooLarge;

        // T1.3（实验）：HOST_COHERENT staging 在 AMD 上读回实测仅 ~0.4GB/s
        // （upload 同路径 ~2GB/s，host memcpy ~6GB/s）——Vega 读回路径对
        // coherent 内存走同步读很慢。改用 HOST_CACHED + 读前
        // vkInvalidateMappedMemoryRanges（non-coherent 标准流程），期望
        // 读回带宽恢复到 ~2GB/s+（12s pm 下载 1.64GB：4.1s → ~0.8s）。
        // D1a-2：staging 常驻 ctx（仅增长时重建），免每次 create/free。
        try ensureStaging(ctx, .dn, want_bytes);
        const st = &ctx.staging_dn;

        try submitOneShot(ctx, struct {
            src: c.VkBuffer,
            dst: c.VkBuffer,
            size: usize,
            pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
                const region = c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = @intCast(s.size),
                };
                c.vkCmdCopyBuffer(cmd, s.src, s.dst, 1, &region);
            }
        }{ .src = self.handle, .dst = st.handle, .size = want_bytes });

        var mapped: ?*anyopaque = null;
        try vk.check(c.vkMapMemory(ctx.device, st.memory, 0, want_bytes, 0, &mapped));
        defer c.vkUnmapMemory(ctx.device, st.memory);
        // non-coherent：CPU 读前 invalidate（GPU 写对 CPU 不可见，必须刷新）。
        var range = std.mem.zeroes(c.VkMappedMemoryRange);
        range.sType = c.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = st.memory;
        range.size = c.VK_WHOLE_SIZE;
        try vk.check(c.vkInvalidateMappedMemoryRanges(ctx.device, 1, &range));
        @memcpy(dst, @as([*]u8, @ptrCast(mapped.?))[0..want_bytes]);
    }

    /// VkDescriptorBufferInfo for a descriptor write. Range covers the
    /// full allocation; shaders index by element so extra capacity is
    /// harmless.
    pub fn descriptorInfo(self: *const Buffer) c.VkDescriptorBufferInfo {
        return .{
            .buffer = self.handle,
            .offset = 0,
            .range = @intCast(self.bytes),
        };
    }

    /// VkDescriptorBufferInfo for a **sub-region** (T1.1 buffer subview).
    /// ``offset_bytes``/``len_bytes`` clamped to the allocation so callers
    /// can pass raw element offsets without extra bounds math; the storage
    /// buffer range only affects GPU access visibility (read/write within
    /// the range), capacity beyond is harmless for our shaders.
    pub fn descriptorInfoView(
        self: *const Buffer,
        offset_bytes: usize,
        len_bytes: usize,
    ) c.VkDescriptorBufferInfo {
        const off = @min(offset_bytes, self.bytes);
        const remain = self.bytes - off;
        return .{
            .buffer = self.handle,
            .offset = @intCast(off),
            .range = @intCast(@min(len_bytes, remain)),
        };
    }

    pub fn deinit(self: *Buffer, device: c.VkDevice) void {
        c.vkDestroyBuffer(device, self.handle, null);
        if (!self.sub) {
            c.vkFreeMemory(device, self.memory, null);
        }
    }
};

// ── Internals ────────────────────────────────────────────────────────

const RawBuffer = struct {
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
};

pub fn findMemoryType(pdev: c.VkPhysicalDevice, type_filter: u32, properties: u32) !u32 {
    var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
    for (0..mem_props.memoryTypeCount) |i| {
        const idx: u5 = @intCast(i);
        if ((type_filter & (@as(u32, 1) << idx)) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return @intCast(i);
        }
    }
    return error.NoSuitableMemoryType;
}

fn createBuffer(
    ctx: *const vk.Context,
    bytes: usize,
    usage: c.VkBufferUsageFlags,
    properties: c.VkMemoryPropertyFlags,
) !RawBuffer {
    var bci = std.mem.zeroes(c.VkBufferCreateInfo);
    bci.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = @intCast(bytes);
    bci.usage = usage;
    bci.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;

    var handle: c.VkBuffer = null;
    try vk.check(c.vkCreateBuffer(ctx.device, &bci, null, &handle));
    errdefer c.vkDestroyBuffer(ctx.device, handle, null);

    var req: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(ctx.device, handle, &req);

    var mai = std.mem.zeroes(c.VkMemoryAllocateInfo);
    mai.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = try findMemoryType(ctx.physical_device, req.memoryTypeBits, properties);

    var memory: c.VkDeviceMemory = null;
    try vk.check(c.vkAllocateMemory(ctx.device, &mai, null, &memory));
    errdefer c.vkFreeMemory(ctx.device, memory, null);

    try vk.check(c.vkBindBufferMemory(ctx.device, handle, memory, 0));
    return .{ .handle = handle, .memory = memory };
}

/// 常驻 staging buffer 的方向（D1a-2）。
pub const StagingDir = enum { up, dn };

/// Ensure ctx 的 staging buffer 至少有 `bytes` 字节（仅增长时重建）。
/// 上传方向 HOST_COHERENT（写快），下载方向 HOST_CACHED（读回快）。
fn ensureStaging(ctx: *vk.Context, dir: StagingDir, bytes: usize) !void {
    const st = switch (dir) {
        .up => &ctx.staging_up,
        .dn => &ctx.staging_dn,
    };
    if (st.bytes >= bytes) return;
    const new_bytes = @max(bytes, 64 * 1024); // 最小 64KB，避免频繁重建
    const usage: c.VkBufferUsageFlags = switch (dir) {
        .up => c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .dn => c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    const props: c.VkMemoryPropertyFlags = switch (dir) {
        .up => c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        .dn => c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    };
    const raw = try createBuffer(ctx, new_bytes, usage, props);
    if (st.handle != null) c.vkDestroyBuffer(ctx.device, st.handle, null);
    if (st.memory != null) c.vkFreeMemory(ctx.device, st.memory, null);
    st.* = .{ .handle = raw.handle, .memory = raw.memory, .bytes = new_bytes };
}

/// One-shot blocking staging upload: HOST_VISIBLE staging buffer,
/// memcpy, copy into `dst`, submit, wait idle, free. Init-time only.
fn stagingUpload(ctx: *vk.Context, dst: c.VkBuffer, bytes: []const u8) !void {
    if (bytes.len == 0) return;
    // 注：upload 保持 HOST_COHERENT（实测 ~2GB/s 与 CACHED+flush 持平且
    // 更简单）；只有 download 读回路径在 AMD 上对 coherent 慢（0.4GB/s），
    // 已改为 CACHED+invalidate（T1.3）。D1a-2：staging 常驻 ctx。
    try ensureStaging(ctx, .up, bytes.len);
    const st = &ctx.staging_up;

    var mapped: ?*anyopaque = null;
    try vk.check(c.vkMapMemory(ctx.device, st.memory, 0, bytes.len, 0, &mapped));
    @memcpy(@as([*]u8, @ptrCast(mapped.?))[0..bytes.len], bytes);
    c.vkUnmapMemory(ctx.device, st.memory);

    try submitOneShot(ctx, struct {
        src: c.VkBuffer,
        dst: c.VkBuffer,
        size: usize,
        pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
            const region = c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = @intCast(s.size),
            };
            c.vkCmdCopyBuffer(cmd, s.src, s.dst, 1, &region);
        }
    }{ .src = st.handle, .dst = dst, .size = bytes.len });
}

/// Batch staging upload（J9）：一次 memcpy 到 staging 各偏移 + 多条
/// vkCmdCopyBuffer 录进同一 command buffer + 一次 submit + 一次 fence 等待。
/// 训练每步数百次小上传（BR 链）各付一次 submitOneShot（submit+wait 固定
/// 开销 ~1ms）→ 合并为每批一次。staging 需 ≥ 本批总和（Python 侧按
/// staging 容量分批；本实现按需 ensureStaging 增长）。
pub fn stagingUploadBatch(ctx: *vk.Context, items: []const UploadItem) !void {
    if (items.len == 0) return;
    var total: usize = 0;
    for (items) |it| total += it.data.len;
    try ensureStaging(ctx, .up, total);
    const st = &ctx.staging_up;

    var mapped: ?*anyopaque = null;
    try vk.check(c.vkMapMemory(ctx.device, st.memory, 0, total, 0, &mapped));
    var off: usize = 0;
    for (items) |it| {
        @memcpy(@as([*]u8, @ptrCast(mapped.?))[off..][0..it.data.len], it.data);
        off += it.data.len;
    }
    c.vkUnmapMemory(ctx.device, st.memory);

    try submitOneShot(ctx, struct {
        st_handle: c.VkBuffer,
        items: []const UploadItem,
        pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
            var woff: usize = 0;
            for (s.items) |it| {
                const region = c.VkBufferCopy{
                    .srcOffset = @intCast(woff),
                    .dstOffset = 0,
                    .size = @intCast(it.data.len),
                };
                c.vkCmdCopyBuffer(cmd, s.st_handle, it.dst, 1, &region);
                woff += it.data.len;
            }
        }
    }{ .st_handle = st.handle, .items = items });
}

pub const DownloadItem = struct {
    src: c.VkBuffer,
    dst: [*]u8,
    len: usize,
};

/// Batch staging download（J18）：多条 vkCmdCopyBuffer（设备缓冲 → staging
/// 各偏移）录进同一 command buffer + 一次 submit + 一次 fence 等待，之后
/// 一次 map 逐段 memcpy 到宿主。与 stagingUploadBatch 对称；省训练每步
/// 数百次 readback 各自的 submit+fence 固定开销。staging 方向 .dn。
pub fn stagingDownloadBatch(ctx: *vk.Context, items: []const DownloadItem) !void {
    if (items.len == 0) return;
    var total: usize = 0;
    for (items) |it| total += it.len;
    try ensureStaging(ctx, .dn, total);
    const st = &ctx.staging_dn;

    try submitOneShot(ctx, struct {
        st_handle: c.VkBuffer,
        items: []const DownloadItem,
        pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
            var woff: usize = 0;
            for (s.items) |it| {
                const region = c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = @intCast(woff),
                    .size = @intCast(it.len),
                };
                c.vkCmdCopyBuffer(cmd, it.src, s.st_handle, 1, &region);
                woff += it.len;
            }
        }
    }{ .st_handle = st.handle, .items = items });

    var mapped: ?*anyopaque = null;
    try vk.check(c.vkMapMemory(ctx.device, st.memory, 0, total, 0, &mapped));
    var off: usize = 0;
    for (items) |it| {
        @memcpy(it.dst[0..it.len], @as([*]u8, @ptrCast(mapped.?))[off..][0..it.len]);
        off += it.len;
    }
    c.vkUnmapMemory(ctx.device, st.memory);
}

pub const UploadItem = struct {
    dst: c.VkBuffer,
    data: []const u8,
};

/// Zero-fill a device-local buffer via vkCmdFillBuffer.
pub fn gpuFill(ctx: *vk.Context, dst: c.VkBuffer, bytes: usize) !void {
    try submitOneShot(ctx, struct {
        dst: c.VkBuffer,
        size: usize,
        pub fn record(s: @This(), cmd: c.VkCommandBuffer) void {
            c.vkCmdFillBuffer(cmd, s.dst, 0, @intCast(s.size), 0);
        }
    }{ .dst = dst, .size = bytes });
}

/// Submit one command buffer and wait for its fence (D1a-2：常驻命令缓冲
/// + 常驻 fence 复用，免每次 vkAllocateCommandBuffers/vkFreeCommandBuffers
/// + vkQueueWaitIdle 全队列等待；本提交之后无其他排队提交，fence 等待与
/// waitIdle 语义等价）。调用方须保证串行（引擎 mutex 已保证）。
pub fn submitOneShot(ctx: *vk.Context, recorder: anytype) !void {
    if (ctx.one_shot_cmd == null) {
        var cb_ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
        cb_ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_ai.commandPool = ctx.cmd_pool;
        cb_ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;
        var cmd: c.VkCommandBuffer = null;
        try vk.check(c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &cmd));
        ctx.one_shot_cmd = cmd;
    }
    if (ctx.one_shot_fence == null) {
        var fci = std.mem.zeroes(c.VkFenceCreateInfo);
        fci.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        var fence: c.VkFence = null;
        try vk.check(c.vkCreateFence(ctx.device, &fci, null, &fence));
        ctx.one_shot_fence = fence;
    }
    const cmd = ctx.one_shot_cmd.?;
    try vk.check(c.vkResetCommandBuffer(cmd, 0));

    var begin = std.mem.zeroes(c.VkCommandBufferBeginInfo);
    begin.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    try vk.check(c.vkBeginCommandBuffer(cmd, &begin));
    recorder.record(cmd);
    try vk.check(c.vkEndCommandBuffer(cmd));

    try vk.check(c.vkResetFences(ctx.device, 1, &ctx.one_shot_fence.?));
    var submit = std.mem.zeroes(c.VkSubmitInfo);
    submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    try vk.check(c.vkQueueSubmit(ctx.queue, 1, &submit, ctx.one_shot_fence.?));
    try vk.check(c.vkWaitForFences(ctx.device, 1, &ctx.one_shot_fence.?, c.VK_TRUE, std.math.maxInt(u64)));
}
