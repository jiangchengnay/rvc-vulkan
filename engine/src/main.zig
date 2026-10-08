//! DLL root. Referencing the FFI module pulls every `pub export fn`
//! into the link so the linker exports them from rvc_core.dll.
const ffi = @import("ffi.zig");

comptime {
    _ = ffi;
}
