//! Command-buffer batcher for compute dispatches.
//!
//! Simplified from valkyr-engine's src/gpu/recorder.zig. One recorder
//! lives in the engine and is reused for every op: `begin()` resets the
//! descriptor pool + command buffer, `dispatch()` allocates ONE fresh
//! descriptor set per dispatch (avoids the use-after-write hazard of
//! reusing a set updated between recorded dispatches) and records
//! bind/push/dispatch, `endAndSubmit()` submits and waits on a fence.
//!
//! Async support (P1 throughput): the recorder owns MAX_INFLIGHT frames
//! (command buffer + fence + descriptor pool each). `endAndSubmitAsync()`
//! submits without waiting and advances to the next frame; `waitAll()`
//! blocks until every in-flight frame completes. `begin()` implicitly
//! waits the frame it is about to reuse, so a caller that keeps
//! submitting async batches never overwrites in-flight command buffers
//! or descriptor pools — correctness is preserved with at most
//! MAX_INFLIGHT outstanding submissions. `endAndSubmit()` keeps the
//! original synchronous semantics (submit + immediate wait).
//!
//! P1-9 dependency-tracked barriers: instead of a global memory barrier
//! before every dispatch after the first, the recorder tracks which
//! buffers have been written (and which read) since the last barrier
//! and only inserts one when the next dispatch genuinely depends on a
//! prior write:
//!   - reads/writes of a buffer written since the last barrier => RAW/WAW;
//!   - writes of a buffer only read since the last barrier        => WAR.
//! A single barrier covers ALL pending writes/reads at once and clears
//! the tracking sets, so the barrier count drops from O(dispatches) to
//! O(true write->read edges) — a batch of N independent blocks (hubert
//! 12 layers x N blocks) now needs ~12 barriers instead of N*12*20.

const std = @import("std");
const vk = @import("vk.zig");
const buffer = @import("buffer.zig");
const pipeline = @import("pipeline.zig");
const c = vk.c;

/// Number of frames kept in flight for async submission. Each frame is a
/// full command buffer + fence + descriptor pool, so this bounds both
/// the number of outstanding submits and the peak descriptor usage.
pub const MAX_INFLIGHT: u32 = 4;

/// Capacity of the dependency-tracking sets. Bounded by the recorder's
/// dispatch cap: max_sets (384) dispatches x 4 bindings = 1536 distinct
/// buffers worst case; 2048 leaves headroom for in-place ops (read+write
/// of the same buffer are tracked in different sets).
const MAX_TRACKED: usize = 2048;

const Frame = struct {
    cmd: c.VkCommandBuffer,
    fence: c.VkFence,
    pool: c.VkDescriptorPool,
    /// Frame index (query-slot offset for RVC_TS timestamps).
    index: u32,
    /// True between submit and wait: the frame's cmd/pool must not be
    /// touched while the GPU may still be reading it.
    in_flight: bool = false,
};

/// Timestamp queries per frame (start + end). RVC_TS=1 (default off,
/// measure-only) records a TOP_OF_PIPE timestamp after begin() and a
/// BOTTOM_OF_PIPE timestamp before end, then accumulates (end-start)
/// ticks per waited frame -> GPU-busy wall estimate at submit granularity.
const TS_PER_FRAME: u32 = 2;

pub const Recorder = struct {
    // Handles copied by value so the recorder never holds a pointer into
    // an engine's construction-time stack frame.
    device: c.VkDevice,
    queue: c.VkQueue,
    cmd_pool: c.VkCommandPool,
    frames: [MAX_INFLIGHT]Frame,
    /// Index of the frame the next begin() will record into.
    cur: u32,
    n_dispatched: u32,
    max_sets: u32,
    max_descriptors: u32,
    descriptors_used: u32,
    /// Buffers written by dispatches recorded since the last barrier: a
    /// later dispatch that reads OR writes any of these needs a barrier
    /// (RAW / WAW).
    writes_pending: [MAX_TRACKED]*const buffer.Buffer,
    writes_pending_len: usize,
    /// Buffers read (but not written) since the last barrier: a later
    /// dispatch that writes any of these needs a barrier (WAR).
    reads_tracked: [MAX_TRACKED]*const buffer.Buffer,
    reads_tracked_len: usize,
    /// How many vkCmdPipelineBarrier calls were recorded since begin().
    /// Diagnostics only (RVC_BARRIER_STATS env → one stderr line at
    /// submit): lets benchmarks quantify the optimisation.
    n_barriers: u64,
    log_barriers: bool,
    /// RVC_TS=1 timestamp query pool (measure-only, default off). Null
    /// pool when disabled. Two queries per frame (start/end); busy ticks
    /// accumulated after each waited frame.
    ts_pool: c.VkQueryPool,
    ts_enabled: bool,
    ts_period: f64,
    ts_busy_ticks: u64,
    ts_submits: u64,

    pub fn init(ctx: *const vk.Context, max_sets: u32, max_descriptors: u32) !Recorder {
        // Diagnostics off by default; RVC_BARRIER_STATS=1 logs one
        // "dispatches/barriers" line per submit to stderr.
        var log_barriers = false;
        if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_BARRIER_STATS")) |v| {
            defer std.heap.page_allocator.free(v);
            log_barriers = v.len > 0 and v[0] != '0';
        } else |_| {}
        var rec = Recorder{
            .device = ctx.device,
            .queue = ctx.queue,
            .cmd_pool = ctx.cmd_pool,
            .frames = undefined,
            .cur = 0,
            .n_dispatched = 0,
            .max_sets = max_sets,
            .max_descriptors = max_descriptors,
            .descriptors_used = 0,
            .writes_pending = undefined,
            .writes_pending_len = 0,
            .reads_tracked = undefined,
            .reads_tracked_len = 0,
            .n_barriers = 0,
            .log_barriers = log_barriers,
            .ts_pool = null,
            .ts_enabled = false,
            .ts_period = 1.0,
            .ts_busy_ticks = 0,
            .ts_submits = 0,
        };
        for (&rec.frames, 0..) |*f, i| {
            var pool_size = c.VkDescriptorPoolSize{
                .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = max_descriptors,
            };
            var dpci = std.mem.zeroes(c.VkDescriptorPoolCreateInfo);
            dpci.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets = max_sets;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &pool_size;
            var pool: c.VkDescriptorPool = null;
            try vk.check(c.vkCreateDescriptorPool(ctx.device, &dpci, null, &pool));
            errdefer c.vkDestroyDescriptorPool(ctx.device, pool, null);

            var cb_ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
            cb_ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cb_ai.commandPool = ctx.cmd_pool;
            cb_ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cb_ai.commandBufferCount = 1;
            var cmd: c.VkCommandBuffer = null;
            try vk.check(c.vkAllocateCommandBuffers(ctx.device, &cb_ai, &cmd));
            errdefer c.vkFreeCommandBuffers(ctx.device, ctx.cmd_pool, 1, &cmd);

            var fci = std.mem.zeroes(c.VkFenceCreateInfo);
            fci.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            var fence: c.VkFence = null;
            try vk.check(c.vkCreateFence(ctx.device, &fci, null, &fence));
            errdefer c.vkDestroyFence(ctx.device, fence, null);

            f.* = .{ .cmd = cmd, .fence = fence, .pool = pool, .index = @intCast(i), .in_flight = false };
        }

        // RVC_TS=1 (default off): timestamp query pool for GPU-busy
        // measurement. Two queries per frame (start/end). Pure
        // measurement — zero effect on recorded dispatches; when
        // disabled the pool stays null and ts_enabled=false (no
        // vkCmdWriteTimestamp calls at all).
        if (std.process.getEnvVarOwned(std.heap.page_allocator, "RVC_TS")) |v| {
            defer std.heap.page_allocator.free(v);
            if (v.len > 0 and v[0] != '0') {
                var qpci = std.mem.zeroes(c.VkQueryPoolCreateInfo);
                qpci.sType = c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qpci.queryType = c.VK_QUERY_TYPE_TIMESTAMP;
                qpci.queryCount = MAX_INFLIGHT * TS_PER_FRAME;
                var pool: c.VkQueryPool = null;
                if (c.vkCreateQueryPool(ctx.device, &qpci, null, &pool) == c.VK_SUCCESS) {
                    rec.ts_pool = pool;
                    rec.ts_enabled = true;
                    rec.ts_period = @floatCast(ctx.props.limits.timestampPeriod);
                    std.debug.print("RVC_TS: timestamp query enabled (period {d} ns/tick)\n", .{rec.ts_period});
                } else {
                    std.debug.print("RVC_TS: vkCreateQueryPool failed, disabled\n", .{});
                }
            }
        } else |_| {}
        return rec;
    }

    pub fn deinit(self: *Recorder) void {
        self.waitAll() catch {};
        for (&self.frames) |*f| {
            c.vkDestroyFence(self.device, f.fence, null);
            c.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &f.cmd);
            c.vkDestroyDescriptorPool(self.device, f.pool, null);
        }
        if (self.ts_pool != null) {
            c.vkDestroyQueryPool(self.device, self.ts_pool, null);
        }
    }

    fn frame(self: *Recorder) *Frame {
        return &self.frames[self.cur];
    }

    /// Linear membership test on a tracking set (sets are small: bounded
    /// by dispatches-per-batch x bindings, <= 1536).
    fn contains(bufs: []const *const buffer.Buffer, b: *const buffer.Buffer) bool {
        for (bufs) |x| {
            if (x == b) return true;
        }
        return false;
    }

    /// Wait the current frame's fence if it is in flight. Called by
    /// begin() so a frame is never re-recorded while the GPU may still
    /// be executing it (async submission safety net).
    fn waitCurrent(self: *Recorder) !void {
        const f = self.frame();
        if (f.in_flight) {
            try self.waitFence(f);
            f.in_flight = false;
        }
    }

    fn waitFence(self: *Recorder, f: *Frame) !void {
        const timeout_ns: u64 = 30 * 1_000_000_000;
        const res = c.vkWaitForFences(self.device, 1, &f.fence, c.VK_TRUE, timeout_ns);
        try vk.check(res);
        // RVC_TS：读回该帧的 (end-start) 并累计 GPU busy ticks。
        if (self.ts_enabled) {
            var vals: [TS_PER_FRAME]u64 = undefined;
            const q0: u32 = f.index * TS_PER_FRAME;
            const r2 = c.vkGetQueryPoolResults(
                self.device,
                self.ts_pool,
                q0,
                TS_PER_FRAME,
                @sizeOf([TS_PER_FRAME]u64),
                &vals,
                @sizeOf(u64),
                c.VK_QUERY_RESULT_64_BIT,
            );
            if (r2 == c.VK_SUCCESS and vals[1] >= vals[0]) {
                self.ts_busy_ticks += vals[1] - vals[0];
                self.ts_submits += 1;
            }
        }
    }

    /// RVC_TS read-back: accumulated GPU-busy nanoseconds and submit
    /// count since engine start (measure-only, see RVC_TS env).
    pub const TsStats = struct { busy_ns: u64, submits: u64 };
    pub fn tsStats(self: *const Recorder) TsStats {
        const ns: u64 = @intFromFloat(@as(f64, @floatFromInt(self.ts_busy_ticks)) * self.ts_period);
        return .{ .busy_ns = ns, .submits = self.ts_submits };
    }

    /// RVC_TS reset: zero the accumulated busy/submit counters (call
    /// before a timed window to get a delta).
    pub fn tsReset(self: *Recorder) void {
        self.ts_busy_ticks = 0;
        self.ts_submits = 0;
    }

    /// Block until every in-flight frame has completed. All GPU writes
    /// are then visible to the host.
    pub fn waitAll(self: *Recorder) !void {
        for (&self.frames) |*f| {
            if (f.in_flight) {
                try self.waitFence(f);
                f.in_flight = false;
            }
        }
    }

    /// Reset descriptor pool + command buffer and begin recording.
    /// Call before the first dispatch of a new op. Implicitly waits the
    /// frame being recycled (async safety).
    pub fn begin(self: *Recorder) !void {
        try self.waitCurrent();
        const f = self.frame();
        try vk.check(c.vkResetDescriptorPool(self.device, f.pool, 0));
        try vk.check(c.vkResetCommandBuffer(f.cmd, 0));
        var bi = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        bi.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try vk.check(c.vkBeginCommandBuffer(f.cmd, &bi));
        self.n_dispatched = 0;
        self.descriptors_used = 0;
        self.writes_pending_len = 0;
        self.reads_tracked_len = 0;
        self.n_barriers = 0;
        // RVC_TS: reset the frame's two query slots, record start.
        if (self.ts_enabled) {
            const q0: u32 = f.index * TS_PER_FRAME;
            c.vkCmdResetQueryPool(f.cmd, self.ts_pool, q0, TS_PER_FRAME);
            c.vkCmdWriteTimestamp(f.cmd, c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.ts_pool, q0);
        }
    }

    /// Record one kernel dispatch with a fresh descriptor set from the
    /// current frame's pool, filled with `buffers` in binding order. A
    /// memory barrier precedes the dispatch only when it reads or writes
    /// a buffer a previous dispatch wrote since the last barrier (RAW /
    /// WAW), or writes a buffer a previous dispatch read (WAR) — a
    /// single global barrier covers all pending accesses and clears the
    /// tracking sets. `write_idx` names the output binding; `null` means
    /// the last binding (the convention for every out-of-place kernel).
    pub fn dispatch(
        self: *Recorder,
        kern: *const pipeline.Kernel,
        buffers: []const *const buffer.Buffer,
        write_idx: ?u32,
        push: ?*const anyopaque,
        gx: u32,
        gy: u32,
        gz: u32,
    ) !void {
        return self.dispatchView(kern, buffers, write_idx, push, gx, gy, gz, null);
    }

    /// dispatch + per-binding byte offsets (T1.1 buffer subview support).
    /// ``view_offs`` 每 binding 的字节偏移（binding 0 对应 buffers[0]）；
    /// ``null`` = 全 0（等价 dispatch，零回归）。仅对 gather 式 kernel
    /// （conv_t1d 等）安全——segment 边界无跨列依赖。
    pub fn dispatchView(
        self: *Recorder,
        kern: *const pipeline.Kernel,
        buffers: []const *const buffer.Buffer,
        write_idx: ?u32,
        push: ?*const anyopaque,
        gx: u32,
        gy: u32,
        gz: u32,
        view_offs: ?[]const usize,
    ) !void {
        if (buffers.len != kern.binding_count) return error.BindingCountMismatch;
        if (view_offs) |vo| {
            if (vo.len != buffers.len) return error.BindingCountMismatch;
        }
        const widx: usize = write_idx orelse (buffers.len - 1);
        if (widx >= buffers.len) return error.InvalidBindingIndex;
        const f = self.frame();

        // Guard the descriptor pool before allocating — the driver only
        // reports OUT_OF_POOL_MEMORY at submit time.
        if (self.n_dispatched >= self.max_sets)
            return error.DescriptorPoolExhausted;
        if (self.descriptors_used + buffers.len > self.max_descriptors)
            return error.DescriptorPoolExhausted;

        // ── Dependency check (P1-9) ────────────────────────────────
        var need_barrier = false;
        // RAW / WAW: any bound buffer was written since the last barrier.
        for (buffers) |b| {
            if (contains(self.writes_pending[0..self.writes_pending_len], b)) {
                need_barrier = true;
                break;
            }
        }
        // WAR: the output buffer was only read since the last barrier.
        if (!need_barrier and contains(self.reads_tracked[0..self.reads_tracked_len], buffers[widx])) {
            need_barrier = true;
        }
        // Capacity: the tracking sets would overflow — force a barrier
        // so the sets reset (a barrier is never wrong, only extra).
        if (!need_barrier) {
            const new_writes: usize = if (contains(self.writes_pending[0..self.writes_pending_len], buffers[widx])) 0 else 1;
            var new_reads: usize = 0;
            for (buffers, 0..) |b, i| {
                if (i == widx) continue;
                if (!contains(self.reads_tracked[0..self.reads_tracked_len], b)) new_reads += 1;
            }
            if (self.writes_pending_len + new_writes > MAX_TRACKED or
                self.reads_tracked_len + new_reads > MAX_TRACKED)
                need_barrier = true;
        }

        if (need_barrier) {
            var mb = std.mem.zeroes(c.VkMemoryBarrier);
            mb.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT;
            c.vkCmdPipelineBarrier(
                f.cmd,
                c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &mb,
                0,
                null,
                0,
                null,
            );
            self.n_barriers += 1;
            self.writes_pending_len = 0;
            self.reads_tracked_len = 0;
        }

        // Track this dispatch's accesses for future dependency checks.
        self.writes_pending[self.writes_pending_len] = buffers[widx];
        self.writes_pending_len += 1;
        for (buffers, 0..) |b, i| {
            if (i == widx) continue;
            if (!contains(self.reads_tracked[0..self.reads_tracked_len], b)) {
                self.reads_tracked[self.reads_tracked_len] = b;
                self.reads_tracked_len += 1;
            }
        }

        // ── Allocate + update a fresh descriptor set ────────────────
        var dsai = std.mem.zeroes(c.VkDescriptorSetAllocateInfo);
        dsai.sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = f.pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &kern.set_layout;
        var set: c.VkDescriptorSet = null;
        try vk.check(c.vkAllocateDescriptorSets(self.device, &dsai, &set));

        var infos: [16]c.VkDescriptorBufferInfo = undefined;
        var writes: [16]c.VkWriteDescriptorSet = undefined;
        for (buffers, 0..) |buf, i| {
            if (view_offs) |vo| {
                // 子视图：从 vo[i] 字节开始绑定（range 由 descriptorInfoView
                // 自动 clamp 到分配剩余）。shader 的 global_id 相对段内
                // （conv_t1d 用 push in_off/out_off 做段内寻址）。
                infos[i] = buf.descriptorInfoView(vo[i], buf.bytes - vo[i]);
            } else {
                infos[i] = buf.descriptorInfo();
            }
            writes[i] = std.mem.zeroes(c.VkWriteDescriptorSet);
            writes[i].sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = @intCast(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        c.vkUpdateDescriptorSets(self.device, @intCast(buffers.len), &writes, 0, null);

        // ── Record bind / push / dispatch ───────────────────────────
        c.vkCmdBindPipeline(f.cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, kern.pipeline);
        c.vkCmdBindDescriptorSets(
            f.cmd,
            c.VK_PIPELINE_BIND_POINT_COMPUTE,
            kern.pipeline_layout,
            0,
            1,
            &set,
            0,
            null,
        );
        if (kern.push_bytes > 0 and push != null) {
            c.vkCmdPushConstants(
                f.cmd,
                kern.pipeline_layout,
                c.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                kern.push_bytes,
                push,
            );
        }
        c.vkCmdDispatch(f.cmd, gx, gy, gz);

        self.n_dispatched += 1;
        self.descriptors_used += @intCast(buffers.len);
    }

    /// End recording, submit to the queue WITHOUT waiting, mark the
    /// frame in flight and advance to the next frame. Callers must
    /// eventually call waitAll() (or begin() will wait implicitly when
    /// the frame is recycled).
    pub fn endAndSubmitAsync(self: *Recorder) !void {
        const f = self.frame();
        const nd: u32 = self.n_dispatched;
        // RVC_TS: record the end timestamp just before vkEndCommandBuffer.
        if (self.ts_enabled) {
            const q0: u32 = f.index * TS_PER_FRAME;
            c.vkCmdWriteTimestamp(f.cmd, c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.ts_pool, q0 + 1);
        }
        try vk.check(c.vkEndCommandBuffer(f.cmd));
        try vk.check(c.vkResetFences(self.device, 1, &f.fence));

        var submit = std.mem.zeroes(c.VkSubmitInfo);
        submit.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &f.cmd;
        try vk.check(c.vkQueueSubmit(self.queue, 1, &submit, f.fence));

        f.in_flight = true;
        self.cur = (self.cur + 1) % MAX_INFLIGHT;
        self.n_dispatched = 0;
        self.descriptors_used = 0;
        // A fresh command buffer starts a fresh dependency domain: the
        // tracked sets must not leak across submissions (a barrier in
        // this buffer cannot order a previous buffer's writes anyway).
        self.writes_pending_len = 0;
        self.reads_tracked_len = 0;
        if (self.log_barriers) {
            std.debug.print("recorder: {d} dispatches, {d} barriers\n", .{ nd, self.n_barriers });
        }
        self.n_barriers = 0;
    }

    /// End recording, submit to the queue, wait for the fence. After
    /// this returns all GPU writes are visible to the host.
    pub fn endAndSubmit(self: *Recorder) !void {
        try self.endAndSubmitAsync();
        try self.waitAll();
    }
};
