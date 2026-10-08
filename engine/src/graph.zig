//! Training-graph executor（训练图下沉 Zig 层的核心，T2 POC）。
//!
//! Graph 是编译后的静态节点序列（SSA buffer 句柄编码数据流），由 Python
//! 侧录制一次（模型结构固定），之后每步 `run()` 只换输入 buffer 引用。
//! `run()` 把整图作为**一个 batch**（一次 begin → N×add → 一次
//! commit/commit_async），中间张量全程留在引擎（GPU 内流转，零 host
//! 往返）；python 按 outputs 批量下载结果。
//!
//! GraphNode 完全镜像 rvc_batch_add 的扁平参数 (op,a,b,c,p0..p10)：
//! 每个节点就是一次 dispatch 的完整描述。依赖关系天然由 buffer 句柄
//! 编码（同句柄=同数据；N 读前序 out=有依赖），引擎 recorder 的依赖
//! 跟踪屏障（P1-9）自动处理 RAW/WAR/WAW。
//!
//! 生命周期约定：**Python 管理 buffer**（上传/分配/常驻/释放），Graph
//! 只持有节点表与引擎指针，不做任何内存分配（除了节点副本）。

const std = @import("std");
const engine = @import("engine.zig");

/// 单个图节点：与 rvc_batch_add 的参数同构；`out` 为显式输出 buffer
/// 句柄（op 1-26 沿用 p9 约定时 out 可填 0；op 27+ 专用算子如 conv2d
/// 参数超过 p[11] 槽位时使用 out 字段）。
pub const GraphNode = extern struct {
    op: i32, // 1..26 与 rvc_batch_add 一致；27=conv2d（专用）
    a: i64, // 输入 buffer 句柄（引擎侧 u64 句柄，传负数无意义）
    b: i64,
    c: i64,
    out: i64, // 显式输出 buffer（conv2d 等专用算子；其余填 0）
    p: [11]i64, // p0..p10（与 rvc_batch_add 一致）
};

/// 编译后的静态训练图。
pub const Graph = struct {
    allocator: std.mem.Allocator,
    eng: *engine.Engine,
    nodes: []GraphNode, // 深度拷贝（调用方栈/临时缓冲可释放）
    n_nodes: usize,

    /// 构建图（拷贝节点表）。失败返回 error；调用方负责 release。
    pub fn create(
        allocator: std.mem.Allocator,
        eng: *engine.Engine,
        nodes: []const GraphNode,
    ) !*Graph {
        if (nodes.len == 0) return error.EmptyGraph;
        const g = try allocator.create(Graph);
        errdefer allocator.destroy(g);
        const copy = try allocator.alloc(GraphNode, nodes.len);
        errdefer allocator.free(copy);
        @memcpy(copy, nodes);
        g.* = .{
            .allocator = allocator,
            .eng = eng,
            .nodes = copy,
            .n_nodes = nodes.len,
        };
        return g;
    }

    pub fn destroy(self: *Graph) void {
        const a = self.allocator;
        a.free(self.nodes);
        a.destroy(self);
    }

    /// 整图一次提交。`async_` 为真时用 commit_async（不等待，可多图/多步
    /// 流水线后统一 wait）；为假时同步提交（等 fence）。
    /// 在任何 add 之前自动 batchBegin（幂等安全）。
    pub fn run(self: *Graph, async_: bool) !void {
        const eng = self.eng;
        eng.batchBegin();
        for (self.nodes[0..self.n_nodes]) |nd| {
            try eng.batchAddOp(nd.op, nd.a, nd.b, nd.c, nd.out, &nd.p);
        }
        if (async_) {
            try eng.batchCommitAsync();
        } else {
            try eng.batchCommit();
        }
    }

    /// 阻塞等待所有在途异步提交完成（rvc_batch_wait 语义）。同步提交后
    /// 调用是 no-op。
    pub fn wait(self: *Graph) !void {
        try self.eng.batchWait();
    }

    /// 丢弃未提交的输入配置（幂等；run 前调用安全）。
    pub fn discard(self: *Graph) void {
        self.eng.batchDiscard();
    }
};

// ---------------------------------------------------------------------------
// 便捷：从 rvc_batch_add 同款 15 参数组装 GraphNode
// （供 Python side 或测试直接构造；也可由 Python 端预先算好再批量传入）
// ---------------------------------------------------------------------------
pub fn node(
    op: i32,
    a: i64,
    b: i64,
    c: i64,
    p: [11]i64,
) GraphNode {
    return .{ .op = op, .a = a, .b = b, .c = c, .p = p };
}