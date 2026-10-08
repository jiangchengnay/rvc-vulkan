const std = @import("std");
pub const c = @cImport({ @cInclude("vulkan/vulkan.h"); });
pub fn main() !void {
    var count: u32 = 0;
    _ = c.vkEnumerateInstanceLayerProperties(&count, null);
    std.debug.print("vulkan header OK, layers={d}\n", .{count});
}
