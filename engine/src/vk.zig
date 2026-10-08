//! Headless Vulkan compute context (rvc_core).
//!
//! Clipped from valkyr-engine's src/gpu/vk.zig: instance, physical
//! device pick (prefer discrete), logical device with one compute-
//! capable queue, and a command pool. No surface, no swapchain.
//! Validation layers are enabled in Debug / ReleaseSafe builds when
//! the KHRONOS layer is actually installed; off in ReleaseFast.

const std = @import("std");
const builtin = @import("builtin");

pub const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub fn check(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) return;
    std.debug.print("Vulkan call failed: VkResult={d}\n", .{result});
    return error.VkFailed;
}

/// True iff `name` appears in vkEnumerateInstanceLayerProperties —
/// only enable a layer when the SDK package actually provides it,
/// otherwise vkCreateInstance fails with VK_ERROR_LAYER_NOT_PRESENT.
fn hasInstanceLayer(name: []const u8) bool {
    var count: u32 = 0;
    if (c.vkEnumerateInstanceLayerProperties(&count, null) != c.VK_SUCCESS) return false;
    if (count == 0) return false;
    var props: [32]c.VkLayerProperties = undefined;
    var got: u32 = @min(count, @as(u32, props.len));
    if (c.vkEnumerateInstanceLayerProperties(&got, &props) != c.VK_SUCCESS) return false;
    for (props[0..got]) |lp| {
        const layer_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&lp.layerName)), 0);
        if (std.mem.eql(u8, layer_name, name)) return true;
    }
    return false;
}

fn makeApiVersion(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

const enable_validation = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    .ReleaseFast, .ReleaseSmall => false,
};

/// 常驻 staging buffer 描述（D1a-2：上传/下载各一个可增长 host-visible
/// staging，免每次传输 create/free 的驱动开销）。
pub const Staging = struct {
    handle: c.VkBuffer = null,
    memory: c.VkDeviceMemory = null,
    bytes: usize = 0,
};

pub const Context = struct {
    instance: c.VkInstance,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    queue_family: u32,
    queue: c.VkQueue,
    cmd_pool: c.VkCommandPool,
    props: c.VkPhysicalDeviceProperties,

    // D1a-2（性能攻坚）：一次传输（upload/download/fill）的**常驻**资源池。
    // 原实现每次传输都 vkCreateBuffer+vkAllocateMemory（上传/下载/分配每
    // 次 ~3-10ms 驱动开销）+ vkAllocateCommandBuffers/vkFreeCommandBuffers
    // + vkQueueWaitIdle（全队列等待）。改为：上传/下载各一个可增长的
    // staging buffer、一个常驻 one-shot 命令缓冲 + fence——仅在增长时
    // 重建，等待改为只等本次提交的 fence（本提交之后无其他提交，与
    // waitIdle 等价但省去整队列查询）。所有调用都在 Engine.mutex 内
    // 串行化，无需额外互斥。
    one_shot_cmd: c.VkCommandBuffer = null,
    one_shot_fence: c.VkFence = null,
    staging_up: Staging = .{}, // HOST_VISIBLE|HOST_COHERENT, TRANSFER_SRC
    staging_dn: Staging = .{}, // HOST_VISIBLE|HOST_CACHED,   TRANSFER_DST

    pub fn init(allocator: std.mem.Allocator) !Context {
        _ = allocator; // reserved — extension/layer enumeration may need it

        const verbose = std.process.hasEnvVarConstant("RVC_VK_VERBOSE");

        // ── Instance ────────────────────────────────────────────────
        var app_info = std.mem.zeroes(c.VkApplicationInfo);
        app_info.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "rvc_core";
        app_info.applicationVersion = makeApiVersion(0, 0, 1, 0);
        app_info.pEngineName = "rvc_core";
        app_info.engineVersion = makeApiVersion(0, 0, 1, 0);
        app_info.apiVersion = makeApiVersion(0, 1, 3, 0);

        const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
        const want_validation = enable_validation and hasInstanceLayer("VK_LAYER_KHRONOS_validation");

        var ici = std.mem.zeroes(c.VkInstanceCreateInfo);
        ici.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app_info;
        if (want_validation) {
            ici.enabledLayerCount = validation_layers.len;
            ici.ppEnabledLayerNames = @ptrCast(&validation_layers);
        } else if (enable_validation) {
            std.debug.print(
                "note: VK_LAYER_KHRONOS_validation not installed; running without validation.\n",
                .{},
            );
        }

        var instance: c.VkInstance = null;
        try check(c.vkCreateInstance(&ici, null, &instance));
        errdefer c.vkDestroyInstance(instance, null);

        // ── Physical device pick ────────────────────────────────────
        var dev_count: u32 = 0;
        try check(c.vkEnumeratePhysicalDevices(instance, &dev_count, null));
        if (dev_count == 0) return error.NoVulkanDevice;
        var devs: [16]c.VkPhysicalDevice = undefined;
        const cap = @min(dev_count, devs.len);
        dev_count = cap;
        try check(c.vkEnumeratePhysicalDevices(instance, &dev_count, &devs));

        // Rank by type: discrete > integrated > virtual > cpu. Windows
        // often enumerates a software adapter ahead of the real GPU.
        const rank = struct {
            fn score(t: c.VkPhysicalDeviceType) u32 {
                return switch (t) {
                    c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 4,
                    c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 3,
                    c.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 2,
                    c.VK_PHYSICAL_DEVICE_TYPE_CPU => 1,
                    else => 0,
                };
            }
        };
        var picked: c.VkPhysicalDevice = devs[0];
        var picked_props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(picked, &picked_props);
        var best_score: u32 = rank.score(picked_props.deviceType);
        if (verbose) std.debug.print("vk: enumerated {d} physical device(s):\n", .{dev_count});
        for (devs[0..dev_count]) |pd| {
            var p: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(pd, &p);
            if (verbose) {
                const name_slice = std.mem.sliceTo(&p.deviceName, 0);
                std.debug.print("vk:   - [{s}] {s} (api {d}.{d}.{d})\n", .{
                    switch (p.deviceType) {
                        c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => "discrete",
                        c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => "integrated",
                        c.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => "virtual",
                        c.VK_PHYSICAL_DEVICE_TYPE_CPU => "cpu",
                        else => "other",
                    },
                    name_slice,
                    c.VK_VERSION_MAJOR(p.apiVersion),
                    c.VK_VERSION_MINOR(p.apiVersion),
                    c.VK_VERSION_PATCH(p.apiVersion),
                });
            }
            const s = rank.score(p.deviceType);
            if (s > best_score) {
                picked = pd;
                picked_props = p;
                best_score = s;
            }
        }
        if (verbose) {
            const picked_name = std.mem.sliceTo(&picked_props.deviceName, 0);
            std.debug.print("vk: picked {s}\n", .{picked_name});
        }

        // ── Queue family pick ───────────────────────────────────────
        // Prefer a compute-only family (async compute), else any
        // compute-capable family.
        var qf_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(picked, &qf_count, null);
        var qfs: [16]c.VkQueueFamilyProperties = undefined;
        const qf_cap = @min(qf_count, qfs.len);
        qf_count = qf_cap;
        c.vkGetPhysicalDeviceQueueFamilyProperties(picked, &qf_count, &qfs);

        var queue_family: ?u32 = null;
        for (qfs[0..qf_count], 0..) |qf, i| {
            const has_compute = (qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0;
            const has_graphics = (qf.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0;
            if (has_compute and !has_graphics) {
                queue_family = @intCast(i);
                break;
            }
        }
        if (queue_family == null) {
            for (qfs[0..qf_count], 0..) |qf, i| {
                if ((qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0) {
                    queue_family = @intCast(i);
                    break;
                }
            }
        }
        const qf_index = queue_family orelse return error.NoComputeQueue;

        // ── Logical device + queue ──────────────────────────────────
        const queue_priority: f32 = 1.0;
        var dqci = std.mem.zeroes(c.VkDeviceQueueCreateInfo);
        dqci.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqci.queueFamilyIndex = qf_index;
        dqci.queueCount = 1;
        dqci.pQueuePriorities = &queue_priority;

        var dci = std.mem.zeroes(c.VkDeviceCreateInfo);
        dci.sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &dqci;

        var device: c.VkDevice = null;
        try check(c.vkCreateDevice(picked, &dci, null, &device));
        errdefer c.vkDestroyDevice(device, null);

        var queue: c.VkQueue = null;
        c.vkGetDeviceQueue(device, qf_index, 0, &queue);

        // ── Command pool ────────────────────────────────────────────
        // RESET_COMMAND_BUFFER_BIT lets the recorder recycle its single
        // command buffer between dispatches.
        var cpci = std.mem.zeroes(c.VkCommandPoolCreateInfo);
        cpci.sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qf_index;

        var cmd_pool: c.VkCommandPool = null;
        try check(c.vkCreateCommandPool(device, &cpci, null, &cmd_pool));

        return .{
            .instance = instance,
            .physical_device = picked,
            .device = device,
            .queue_family = qf_index,
            .queue = queue,
            .cmd_pool = cmd_pool,
            .props = picked_props,
        };
    }

    pub fn deinit(self: *Context) void {
        if (self.one_shot_cmd != null) {
            c.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &self.one_shot_cmd);
        }
        if (self.one_shot_fence != null) {
            c.vkDestroyFence(self.device, self.one_shot_fence, null);
        }
        for ([2]*Staging{ &self.staging_up, &self.staging_dn }) |st| {
            if (st.handle != null) c.vkDestroyBuffer(self.device, st.handle, null);
            if (st.memory != null) c.vkFreeMemory(self.device, st.memory, null);
        }
        c.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        c.vkDestroyDevice(self.device, null);
        c.vkDestroyInstance(self.instance, null);
    }

    /// Human-readable device name (null-terminated, owned by Vulkan
    /// driver — do not free).
    pub fn deviceName(self: *const Context) [*:0]const u8 {
        return @ptrCast(&self.props.deviceName);
    }
};
