"""T3-a：训练图执行器 Python 封装（纯新增，默认不启用）。

作用：把判别器 forward 链（DiscriminatorS/P）交给 Zig 图执行器
（rvc_graph_create/run/destroy）整图一次提交，替代 Python 逐算子录制
（BatchRunner 簿记开销下沉 Zig）。复用 ``get_context()`` 单例引擎句柄，
权重走 ``wpers_get`` 常驻 buffer（零重传），输入每步覆写固定 buffer。

与 tape 链的兼容（J19 硬约束）：图执行器输出的中间 buffer 包装成
``BatchTensor(_runner=chain_br)``——backward 的 ``_resolve_input`` 校验
``x._runner is self``，用全局 chain br 包装后 GPU 链式 bp 直接消费图执行器
buffer（中间张量不出引擎，零上传）。record 的 out/x 均为 BatchTensor 且
与下游引用同一对象，bp 链 id 匹配。

生命周期：每判别器缓存 2 张图（A/B 双缓冲），D 步/G 步的两次 forward
（real/fake）交替使用，避免互覆中间值；backward 均在当步内完成，故中间
槽每步覆写安全。权重句柄（wpers）每步 refresh 只覆写内容、句柄不变，
图无需重建。

开关：RVC_TRAIN_GRAPH=1（判别器 forward 图化；默认 0 走现有 BatchRunner，
零回归）。
"""
from __future__ import annotations

import ctypes

import numpy as np

from runtime import _vulkan
from runtime.vulkan_ops import BatchTensor, get_context

# GraphNode 必须与 engine/src/graph.zig 的 extern struct 布局一致：
#   op: i32, a/b/c/out: i64, p: i64[11]
class GraphNode(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_int32),
        ("a", ctypes.c_int64),
        ("b", ctypes.c_int64),
        ("c", ctypes.c_int64),
        ("out", ctypes.c_int64),
        ("p", ctypes.c_int64 * 11),
    ]


class GraphHandle:
    """rvc_graph_handle（i64 指针），engine 生命周期内有效。"""

    __slots__ = ("_v",)

    def __init__(self, v: int):
        self._v = int(v)

    @property
    def value(self) -> int:
        return self._v


class _GraphFFI:
    """rvc_graph_* 的 ctypes 绑定（首次使用惰性初始化）。"""
    _init = False

    def __init__(self, dll):
        if not _GraphFFI._init:
            dll.rvc_graph_create.argtypes = [
                ctypes.c_int64, ctypes.POINTER(GraphNode), ctypes.c_int64]
            dll.rvc_graph_create.restype = ctypes.c_int64
            dll.rvc_graph_run.argtypes = [ctypes.c_int64, ctypes.c_int32]
            dll.rvc_graph_run.restype = ctypes.c_int32
            dll.rvc_graph_wait.argtypes = [ctypes.c_int64]
            dll.rvc_graph_wait.restype = ctypes.c_int32
            dll.rvc_graph_destroy.argtypes = [ctypes.c_int64]
            dll.rvc_graph_destroy.restype = ctypes.c_int32
            _GraphFFI._init = True
        self.dll = dll


class DiscGraph:
    """判别器整链图（S 或 P）：图句柄 + 固定槽位，run 一次整链。

    结构（与 forward_br 的 pairs 语义一致）：
      - in_buf: 输入槽（每步覆写）
      - slots: 中间/输出槽 [(buf_id, n)]（conv 输出；leaky 就地共用）
      - pairs: [(cbt_buf, lbt_shape), ...]（cbt=conv 输出槽，lbt 同槽）
      - post: (post_buf, post_shape)
    run(x, runner) → 返回 (pairs, post)：全部包装为 BatchTensor（runner 为
    chain br，J19 链匹配），backward 零上传消费这些槽位。
    """

    __slots__ = ("kind", "gh", "in_buf", "pairs", "post", "post_buf",
                 "shapes", "convs_p", "grunner",
                 "bwd_gh", "bwd_go", "bwd_fm", "bwd_out", "_last_bwd_ms",
                 "fm_clean")

    def __init__(self, grunner, kind: str, nodes, in_buf, pairs, post,
                 post_buf, shapes, convs_p, bwd=None):
        self.grunner = grunner
        self.kind = kind
        self.gh = grunner.build(nodes)
        self.in_buf = in_buf
        self.pairs = pairs          # [(conv_buf, conv_n, lbt_shape), ...]
        self.post = post            # (post_buf, post_n)
        self.post_buf = post_buf
        self.shapes = shapes        # [conv_shape, ...]（与 pairs 对齐）
        self.convs_p = convs_p      # [(s, p, gr) or (s, p), ...]（record 用）
        # T3-b/T3-c 后续：backward 图（S 链 conv1d_bwd；P 链 conv2d_bwd，
        # 5 算子/层——两者均由各自 _build_*_bwd 构建后 attach）。bwd 结构
        # 见 _build_s_bwd/_build_p_bwd 返回。
        self.bwd_gh = None
        self.bwd_go = None          # go 输入槽（post 输出梯度，每步覆写）
        self.bwd_fm = None          # [fm 槽×6]（各层 leaky 输出 FM 种子）
        self.bwd_out = None         # {key: buf}（gx_in/gw_i/gb_i/gx_post/gy_i）
        self.fm_clean = None        # T6b：内容全零的 FM 槽（dirty-skip 用）
        if bwd is not None:
            self._attach_bwd(bwd)

    def _attach_bwd(self, bwd) -> None:
        """挂接已构建的 backward 图（bwd_nodes + 槽位分配）。FM 槽已在
        _build_s_bwd 初始化为零（D 步无 FM 种子时图内 add 读零值，正确）；
        G 步按层覆写。

        fm_clean: 记录当前内容仍为全零的 FM 槽（构建时预零 → 初始全
        clean）。RVC_TRAIN_SKIP_FM_ZEROS=1 时 bwd_run 对 clean 槽跳过
        零上传（零语义等价：槽已是零，上传零无意义）；槽被非零种子覆写
        后移出 clean，下一次 v is None 必须重新置零。"""
        nodes, go, fm, out = bwd
        self.bwd_go = go
        self.bwd_fm = fm
        self.bwd_out = out
        self.bwd_gh = self.grunner.build(nodes)
        self.fm_clean = set(fm.keys())  # 构建时均预置零 → 全部 clean

    def run(self, x: np.ndarray, runner=None) -> tuple:
        """覆写输入 → 整链 run → 返回 ((pairs_bt, post_bt), fmap_shapes)。"""
        self.grunner.set_input(x, self.in_buf)
        self.grunner.run(self.gh)
        if runner is None:
            from runtime.models import vits_train as _vt  # noqa: PLC0415
            runner = _vt._chain_br()
        pairs = []
        for i, (cbuf, cn, lshp) in enumerate(self.pairs):
            cbt = BatchTensor(runner, cbuf, self.shapes[i])
            lbt = BatchTensor(runner, cbuf, lshp)  # leaky 就地同槽
            pairs.append((cbt, lbt))
        pb = self.post_buf
        post_bt = BatchTensor(runner, pb, self.post[1])
        return pairs, post_bt

    def run_async(self, x: np.ndarray, runner=None) -> tuple:
        """R-T4：覆写输入 → async_run（不等待，入在途队列）→ 惰性包装。
        结果在调用方统一 wait_all() 后才可下载；引擎队列顺序执行保证
        RAW 安全。返回结构同 run()。"""
        self.grunner.set_input(x, self.in_buf)
        self.grunner.async_run(self.gh)
        if runner is None:
            from runtime.models import vits_train as _vt  # noqa: PLC0415
            runner = _vt._chain_br()
        pairs = []
        for i, (cbuf, cn, lshp) in enumerate(self.pairs):
            cbt = BatchTensor(runner, cbuf, self.shapes[i])
            lbt = BatchTensor(runner, cbuf, lshp)  # leaky 就地同槽
            pairs.append((cbt, lbt))
        pb = self.post_buf
        post_bt = BatchTensor(runner, pb, self.post[1])
        return pairs, post_bt

    def bwd_run(self, go_np, fm_list, runner=None) -> None:
        """T3-b：覆写 go/FM 槽 → 一次 run 整条 backward 链（结果留在
        bwd_out 槽，Python 侧用 bwd_take 包装成 BatchTensor）。fm_list 为
        {层 i: 种子 numpy}（缺省层上传 zeros——D 步无 FM 种子时 FM 槽必须
        置 0，避免残留 G 步值污染图内加法）。go 可为 BatchTensor 或 numpy。
        R-T4：env RVC_TRAIN_GRAPH_ASYNC=1 时转异步（bwd_run_async）。"""
        import os as _os2  # noqa: PLC0415
        if _os2.environ.get("RVC_TRAIN_GRAPH_ASYNC", "0") == "1":
            return self.bwd_run_async(go_np, fm_list, runner)
        import time as _tm  # noqa: PLC0415
        _t0 = _tm.perf_counter()
        # M4 埋点 C（a26ah §2.3）：同步路径六段 wall（C4 = g.run 阻塞全程 ★，
        # 与异步的 async_run 入队语义不同，字段名分列）。
        _m4c = _os2.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1"
        if self.bwd_gh is None:
            raise RuntimeError(f"{self.kind} 图未构建 backward 段")
        if isinstance(go_np, BatchTensor):
            go_np = go_np.numpy()
        g = self.grunner
        _t_c1 = _tm.perf_counter()
        self._bwd_upload(g, go_np, fm_list)
        # T3-b：graph.run 内部 batchBegin 会清空 engine 未提交 ops——P 链
        # 逐层 bp 已录制进 chain br（同一 engine batch 队列），必须先提交，
        # 否则 P 链梯度全部丢失（D 步 conv_post 系统性差异根因）。
        from runtime.models import vits_train as _vt  # noqa: PLC0415
        _t_c2 = _tm.perf_counter()
        _vt._commit_chain_br()
        _t_c3 = _tm.perf_counter()
        g.run(self.bwd_gh)
        _t_c4 = _tm.perf_counter()
        self._last_bwd_ms = (_t_c4 - _t0) * 1000.0  # TEMP-DBG
        if _m4c:
            from runtime import m4_probe as _mp  # noqa: PLC0415
            _mp.c_sync((_t_c1 - _t0) * 1000.0,      # C1 前置
                       (_t_c2 - _t_c1) * 1000.0,    # C2 上传
                       (_t_c3 - _t_c2) * 1000.0,    # C3 commit ★
                       (_t_c4 - _t_c3) * 1000.0,    # C4 g.run 阻塞 ★
                       (_t_c4 - _t0) * 1000.0)      # C5 总

    def bwd_run_async(self, go_np, fm_list, runner=None) -> None:
        """R-T4：bwd_run 的异步版——覆写 go/FM 槽 → async_run（不等待，
        入在途队列）；go/FM 同 bwd_run 语义（go 或 FM 项为 BatchTensor 时
        先 numpy 下载）。结果留在 bwd_out 槽；调用方在批量下载前统一
        wait_all()（引擎队列顺序执行保证链内 RAW 安全）。"""
        import os as _os2  # noqa: PLC0415
        import time as _tm  # noqa: PLC0415
        _t0 = _tm.perf_counter()
        # M4 埋点 C（a26ah §2.3）：异步路径六段（C4 = async_run 入队，应亚 ms；
        # 字段名与 c_sync 严格分列——语义不同，不可混）。
        _m4c = _os2.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1"
        if self.bwd_gh is None:
            raise RuntimeError(f"{self.kind} 图未构建 backward 段")
        if isinstance(go_np, BatchTensor):
            go_np = go_np.numpy()
        g = self.grunner
        _t_c1 = _tm.perf_counter()
        self._bwd_upload(g, go_np, fm_list)
        from runtime.models import vits_train as _vt  # noqa: PLC0415
        _t_c2 = _tm.perf_counter()
        _vt._commit_chain_br()
        _t_c3 = _tm.perf_counter()
        g.async_run(self.bwd_gh)
        _t_c4 = _tm.perf_counter()
        self._last_bwd_ms = (_t_c4 - _t0) * 1000.0  # TEMP-DBG
        if _m4c:
            from runtime import m4_probe as _mp  # noqa: PLC0415
            _mp.c_async((_t_c1 - _t0) * 1000.0,     # C1 前置
                        (_t_c2 - _t_c1) * 1000.0,   # C2 上传
                        (_t_c3 - _t_c2) * 1000.0,   # C3 commit ★
                        (_t_c4 - _t_c3) * 1000.0,   # C4 async_run 入队
                        (_t_c4 - _t0) * 1000.0)     # C5 总

    def _bwd_upload(self, g, go_np, fm_list) -> None:
        """C2b-3：disc backward 的 go/FM 槽上传。

        收集本次 backward 需要覆写的全部槽位数组（等价于原逐槽
        ``set_input`` 序列），RVC_C2B3_ON=1 时一次 ``set_input_batch``
        合并提交（1 次 submit+fence/步·判别器），否则逐槽 set_input
        （原行为，逐位等价）。fm_clean 零槽追踪两路径共用；nil FM 槽
        与 BatchTensor 种子转换语义与 bwd_run_async 原实现一致。
        """
        import os as _os2  # noqa: PLC0415
        c2b3 = _os2.environ.get("RVC_C2B3_ON", "0") == "1"
        items = []
        if isinstance(go_np, BatchTensor):
            go_np = go_np.numpy()
        items.append((self.bwd_go, np.ascontiguousarray(
            np.asarray(go_np, np.float32))))
        fm = fm_list or {}
        _skip_fm = _os2.environ.get("RVC_TRAIN_SKIP_FM_ZEROS", "0") == "1"
        for i, fb in self.bwd_fm.items():
            v = fm.get(i)
            if v is None:
                # FM 槽构建时预置零；RVC_TRAIN_SKIP_FM_ZEROS=1 时若槽内容
                # 仍为全零（fm_clean 追踪），跳过零上传（零语义等价，省
                # 一次全零 upload）；被非零种子覆写后必须重新置零。
                if _skip_fm and i in self.fm_clean:
                    continue
                items.append((fb, np.zeros(int(np.prod(self.shapes[i])),
                                           np.float32)))
                self.fm_clean.add(i)
            else:
                if isinstance(v, BatchTensor):
                    v = v.numpy()
                items.append((fb, np.ascontiguousarray(
                    np.asarray(v, np.float32))))
                self.fm_clean.discard(i)
        if c2b3:
            g.set_input_batch(items)
        else:
            for bid, arr in items:
                g.set_input(arr, bid)

    def bwd_take(self, key, shape, runner=None):
        """把 bwd_out[key] 槽包装成 BatchTensor（_runner=chain br，J19）。"""
        if runner is None:
            from runtime.models import vits_train as _vt  # noqa: PLC0415
            runner = _vt._chain_br()
        return BatchTensor(runner, self.bwd_out[key], shape)

    def destroy(self) -> None:
        g = self.grunner
        g.destroy(self.gh)
        g.free(self.in_buf)
        for (cbuf, _cn, _l) in self.pairs:
            g.free(cbuf)
        g.free(self.post_buf)
        if self.bwd_gh is not None:
            g.destroy(self.bwd_gh)
            if self.bwd_go:
                g.free(self.bwd_go)
            for _k, f in (self.bwd_fm or {}).items():
                g.free(f)
            for _k, b in (self.bwd_out or {}).items():
                g.free(b)


class DecGraph:
    """T3-c：dec 前向整链单图（conv_pre + cond add + 4×ups(leaky+convT+
    noise+add) + 12×ResBlock + leaky(0.01) + conv_post）。

    所有权重/输入均为固定槽：每训练步 ``set_input`` 覆写最新值（deweight
    每步产生新数组 → 结点若绑定 wpers 常驻句柄会随数组 id 失效，故绑固定
    槽 id、内容每步覆写；上传数 = _forward_br 同权重一次/步，无额外开销）。
    ÷3 用图内 mul_inplace（预填 1/3 常量槽），消除 _forward_br 每段 numpy
    断链下载 + 再上传。tanh 仍 Python（tape.tanh）。

    槽命名见 ``_build_dec``；``outs`` 为 record 包装点（同一 buffer 多个
    条目 = 多个 BatchTensor 对象，J19 就地覆写后 out/x 引用分离）。
    """

    __slots__ = ("kind", "grunner", "gh", "in_z", "in_cond", "in_har",
                 "wslots", "outs", "one_third", "key")

    def __init__(self, grunner, key, gh, in_z, in_cond, in_har, wslots,
                 outs, one_third):
        self.kind = "dec"
        self.grunner = grunner
        self.key = key
        self.gh = gh
        self.in_z = in_z
        self.in_cond = in_cond
        self.in_har = in_har
        self.wslots = wslots
        self.outs = outs
        self.one_third = one_third

    def destroy(self) -> None:
        g = self.grunner
        g.destroy(self.gh)
        g.free(self.in_z)
        g.free(self.in_cond)
        g.free(self.in_har)
        for _k, b in self.wslots.items():
            g.free(b)
        seen = set()
        for _k, (b, _s) in self.outs.items():
            if b not in seen:
                seen.add(b)
                g.free(b)
        for b in self.one_third:
            g.free(b)


class EncGraph:
    """T3-d：TextEncoderTrain 前向分段图集（emb + 6 层 × 6 段 + proj = 38 图）。

    引擎无 3D/4D transpose/reshape/pad/slice/embedding 图 op、matmul 仅 2D
    无 trans_b、mul/add 无广播 → 链被切成分段图，段间 Python 衔接（下载/
    视图/预乘/预填/上传）。段内算子与 BatchRunner 同 kernel（位级一致）。

    所有权重/输入为固定槽：每训练步 ``set_input`` 覆写（权重每步 deweight
    新数组 → 绑槽 id 不随数组对象变化）。``ghs`` name→GraphHandle（emb、
    ``l{i}.g1..g5d``、proj）；``slots`` 为 Python 侧覆写/预填的输入与中间
    槽；``outs`` 为 record 包装点（同一 buffer 多条目 = 多个 BatchTensor
    对象，J19 就地覆写后 out/x 引用分离）；``wslots`` 权重槽。见
    ``_build_enc`` 槽命名。
    """

    __slots__ = ("kind", "grunner", "key", "ghs", "slots", "wslots", "outs")

    def __init__(self, grunner, key, ghs, slots, wslots, outs):
        self.kind = "enc"
        self.grunner = grunner
        self.key = key
        self.ghs = ghs
        self.slots = slots
        self.wslots = wslots
        self.outs = outs

    def destroy(self) -> None:
        g = self.grunner
        for gh in self.ghs.values():
            g.destroy(gh)
        seen = set()
        for _k, b in self.wslots.items():
            if b not in seen:
                seen.add(b)
                g.free(b)
        for _k, b in self.slots.items():
            if b not in seen:
                seen.add(b)
                g.free(b)
        for _k, (b, _s) in self.outs.items():
            if b not in seen:
                seen.add(b)
                g.free(b)


class GraphRunner:
    """图执行器封装：build 一次编译，run 整图一次提交。"""

    def __init__(self, ctx=None):
        self.ctx = ctx or get_context()
        self.ffi = _GraphFFI(_vulkan.get_dll())
        self.dll = self.ffi.dll
        self._disc_cache = {}       # key → [DiscGraph_A, DiscGraph_B]
        self._dec_cache = {}        # key → DecGraph（T3-c 单图）
        self._enc_cache = {}        # key → EncGraph（T3-d 分段图集）
        self._last_async = None     # R-T3：最近 async_run 的图句柄（wait_all 用）
        self.bwd_fold = []          # T6b：判别器 bwd 折叠共享 buffer（RVC_TRAIN_DISC_SINGLE_BUF）

    # -- 基础资源 ------------------------------------------------------
    def alloc(self, nfloats: int) -> int:
        """分配 nfloats 个 f32 的 DEVICE_LOCAL buffer（零上传）。"""
        if nfloats <= 0:
            raise ValueError(f"alloc 大小非法: {nfloats}")
        return self.ctx.mem_alloc(int(nfloats) * 4)

    def free(self, buf: int) -> None:
        try:
            self.ctx.free(buf)
        except RuntimeError:
            pass

    # -- 图生命周期 ------------------------------------------------------
    def build(self, nodes) -> GraphHandle:
        """nodes: [(op, a, b, c, out, p_tuple11), ...] → 编译图（深拷贝）。"""
        arr = (GraphNode * len(nodes))()
        for i, (op, a, b, c, out, p) in enumerate(nodes):
            nd = arr[i]
            nd.op = int(op)
            nd.a = int(a)
            nd.b = int(b)
            nd.c = int(c)
            nd.out = int(out)
            for j, pj in enumerate(p):
                nd.p[j] = int(pj)
        gh = self.dll.rvc_graph_create(self.ctx._handle, arr, len(nodes))
        if gh <= 0:
            raise RuntimeError(f"graph create failed: {_vulkan.last_error()}")
        return GraphHandle(gh)

    def run(self, g: GraphHandle) -> None:
        rc = self.dll.rvc_graph_run(g.value, 0)
        if rc != 0:
            raise RuntimeError(f"graph_run: rc={rc} err={_vulkan.last_error()}")

    def async_run(self, g: GraphHandle) -> None:
        """R-T3：异步提交（rvc_graph_run async=1 → engine batchCommitAsync，
        不等待）；结果在 wait_all() 后可用。可在途叠加多个图（MAX_INFLIGHT=4）。"""
        rc = self.dll.rvc_graph_run(g.value, 1)
        if rc != 0:
            raise RuntimeError(f"graph_run(async): rc={rc} err={_vulkan.last_error()}")
        self._last_async = g

    def wait_all(self) -> None:
        """R-T3：等待全部在途异步图完成（engine-wide wait；需任意有效图句柄，
        无在途时 no-op）。"""
        g = getattr(self, "_last_async", None)
        if g is None or g.value <= 0:
            return
        rc = self.dll.rvc_graph_wait(g.value)
        if rc != 0:
            raise RuntimeError(f"graph_wait: rc={rc} err={_vulkan.last_error()}")

    def destroy(self, g: GraphHandle) -> None:
        if g is not None and g.value > 0:
            rc = self.dll.rvc_graph_destroy(g.value)
            if rc != 0:
                # P1-002：原先忽略返回码——图销毁失败（GPU 错误）静默丢失
                import sys as _sg  # noqa: PLC0415
                print(f"[P1-002] graph_destroy rc={rc} err={_vulkan.last_error()}",
                      file=_sg.stderr, flush=True)

    # -- 输入覆写 --------------------------------------------------------
    def set_input(self, a: np.ndarray, buf_id: int) -> None:
        """把输入数组覆写进固定 buffer（全量覆写安全）。"""
        a = np.ascontiguousarray(a, dtype=np.float32)
        n = a.size
        dll = _vulkan.get_dll()
        dll.rvc_mem_upload_to.argtypes = [ctypes.c_int64,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int64, ctypes.c_int64]
        dll.rvc_mem_upload_to.restype = ctypes.c_int32
        rc = dll.rvc_mem_upload_to(self.ctx._handle, a.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)), n, int(buf_id))
        if rc != 0:
            from runtime import _vulkan as _vk2  # noqa: PLC0415
            raise RuntimeError(f"set_input upload_to rc={rc} "
                               f"n={n} buf={buf_id} err={_vk2.last_error()}")

    def set_input_batch(self, items) -> None:
        """C2b-3：批量覆写上传（纯 Python 批合并，0 改 Zig）。

        ``items = [(buf_id, np_array), ...]``——逐项语义与 ``set_input``
        等价（各 buffer 覆写相互独立、无数据依赖，与提交顺序无关，字节
        级一致），但合并为一次 ``rvc_mem_upload_to_batch``（一次 staging
        memcpy + 一次 cmd 提交 + 一次 fence 等待），消除逐次 set_input
        各自的 submit+fence 固定开销（M4：5.42ms/次 × 36 次/步 = 195ms/
        步）。按 staging 容量自动分批（VulkanContext._batch_upload，
        ≤64MB/批）。数组内容在返回前同步完成上传，调用方可安全释放。
        """
        if not items:
            return
        norm = [(int(bid), np.ascontiguousarray(np.asarray(a, np.float32)))
                for bid, a in items]
        self.ctx._batch_upload(norm)

    # -- 权重常驻 --------------------------------------------------------
    def _wpers_id(self, w: np.ndarray) -> int:
        """权重常驻 buffer 句柄（wpers_get 缓存，id 即 PersistentBuffer.id）。"""
        from runtime.vulkan_ops import wpers_get  # noqa: PLC0415
        from runtime.models import vits_train as _vt  # noqa: PLC0415
        pb = wpers_get(w, _vt._chain_br(), "d")
        if pb.id is None:
            raise RuntimeError("wpers buffer 已释放")
        return int(pb.id)

    # -- 判别器图（双缓冲缓存：T3-a）------------------------------------
    def _build_s_bwd(self, meta, convs, post_w, post_b, L_in, slots, in_buf,
                     B=1):
        """S 链 backward 图（39 节点）。meta: [(c_in, ci_g, L, co, k, s, p)]。

        数据流（从后往前，直接消费 fwd 槽位，leaky 就地覆写后符号判据
        等价——slope>0 保号）：
          go_buf → post_bwd(op24/25/26, groups=1, x=slots[5]) → gx_post
          copy gx_post→go6 + addInplace FM5 → leaky6_bwd(op22, x=slots[5])
          → gy5 → conv6_bwd(op24/25/26, x=slots[4], w=convs[5]) → gx4
          copy gx4→go5 + add FM4 → leaky5_bwd → gy4 → conv5_bwd → gx3 ...
          → conv1_bwd(x=in_buf) → gxin
        FM 加在独立 copy 槽（不污染 gx 输出槽——Python 侧 bp 返回纯梯度）。
        返回 (bwd_nodes, go_buf, fm_bufs, bwd_out)。
        """
        import os as _os  # noqa: PLC0415
        LRELU_SLOPE_BITS = 0x3DCCCCCD
        nodes = []
        out = {}
        # 各层输出长度（与 fwd slots 对齐）：lo_list[i] = conv_{i+1} 输出长度
        lo_list = [(L + 2 * p - k) // s + 1 for (_, _, L, _, k, s, p) in meta]
        # --- post conv1d_bwd（groups=1：c_in=1024, c_in_g=1024）---
        # post 输入 = leaky6 输出 = conv6 输出（形状 shapes[5]，长度 lo_list[-1]）
        L6 = lo_list[-1]                     # post 输入长度
        n_post = B * 1 * L6                  # post 输出长度 = L6（k3 s1 p1）
        go_buf = self.alloc(n_post)
        fm_bufs = {}
        gx_post = self.alloc(B * 1024 * L6)         # 到 leaky6 输出的梯度
        gw_post = self.alloc(int(np.prod(np.asarray(post_w).shape)))
        gb_post = self.alloc(int(np.prod(np.asarray(post_b).shape)))
        post_w_id = self._wpers_id(post_w)
        for op, ob in ((24, gx_post), (25, gw_post), (26, gb_post)):
            nodes.append((op, slots[5], post_w_id, go_buf, 0,
                          (B, 1024, 1024, L6, 1, 3, 1, 1, 1, ob, 1)))
        # --- leaky6_bwd：go = gx_post + FM5（copy 到独立槽再加，保 gx_post 纯）---
        n6 = B * meta[-1][3] * lo_list[-1]   # conv6 输出 = shapes[5]
        # T6b①：RVC_TRAIN_DISC_SINGLE_BUF=1 → go 槽折叠。go6/go5..go1 的
        # 生命周期串行（copy→add→leaky 同段即死，下一层 conv 段之后才写
        # 下一个 go 槽），共享一个 max-size buffer 安全（引擎 BufferTooSmall
        # 校验只要求 bytes≥n*4，节点按各自 n 读写前 n 元素）。
        _fold = _os.environ.get("RVC_TRAIN_DISC_SINGLE_BUF", "0") == "1"
        if _fold:
            _go_ns = [n6] + [B * meta[i - 1][3] * lo_list[i - 1]
                             for i in range(5, 0, -1)]
            go6 = self.alloc(max(_go_ns))
            self.bwd_fold.append(go6)
        else:
            go6 = self.alloc(n6)
        fm_bufs[5] = self.alloc(n6)
        self.set_input(np.zeros(n6, np.float32), fm_bufs[5])
        nodes.append((7, go6, gx_post, 0, 0, (n6,) + (0,) * 10))     # copy(dst=go6, src=gx_post)
        nodes.append((3, go6, fm_bufs[5], 0, 0, (n6,) + (0,) * 10))   # go6 += FM5
        gy5 = self.alloc(n6)
        out["gy5"] = gy5
        nodes.append((22, go6, slots[5], gy5, 0,
                      (n6, LRELU_SLOPE_BITS) + (0,) * 9))
        # --- 5..0 层（leaky_i bwd + conv_i bwd）---
        for i in range(5, -1, -1):
            # conv_{i+1} bwd：x=slots[i]（i>0）或 in_buf（i=0），go=gy_i
            _c_in, _ci_g, _L, co, k, s, p = meta[i]
            lo = lo_list[i]
            if i > 0:
                c_in = meta[i - 1][3]            # conv_i 输入通道（=上一层 co）
                L_in_i = lo_list[i - 1]
                gx_i = self.alloc(B * c_in * L_in_i)  # 到 leaky_i 输出的梯度
                x_ref = slots[i - 1]
            else:
                c_in = 1
                L_in_i = L_in
                gx_i = self.alloc(B * 1 * L_in)  # gxin（到输入的梯度）
                x_ref = in_buf
            gy_i = gy5 if i == 5 else out[f"gy{i}"]
            w_i_id = self._wpers_id(convs[i][0])
            ci_g = int(np.asarray(convs[i][0]).shape[1])
            k_i = int(np.asarray(convs[i][0]).shape[2])
            gw_i = self.alloc(int(np.prod(np.asarray(convs[i][0]).shape)))
            gb_i = self.alloc(int(np.prod(np.asarray(convs[i][1]).shape)))
            for op, ob in ((24, gx_i), (25, gw_i), (26, gb_i)):
                nodes.append((op, x_ref, w_i_id, gy_i, 0,
                              (B, c_in, ci_g, L_in_i, co, k_i, s, p, p, ob,
                               1)))
            out[f"gx{i - 1 if i > 0 else 'in'}"] = gx_i
            out[f"gw{i}"] = gw_i
            out[f"gb{i}"] = gb_i
            if i == 0:
                break
            # leaky_i bwd：go = gx_i + FM_{i-1}（copy 到独立槽再加，保 gx_i 纯）
            # go_i/FM 槽大小 = leaky_i 输出 = conv_i 输出 = shapes[i-1]
            n_i = B * meta[i - 1][3] * lo_list[i - 1]
            go_i = go6 if _fold else self.alloc(n_i)
            fm_bufs[i - 1] = self.alloc(n_i)
            self.set_input(np.zeros(n_i, np.float32), fm_bufs[i - 1])
            nodes.append((7, go_i, gx_i, 0, 0, (n_i,) + (0,) * 10))
            nodes.append((3, go_i, fm_bufs[i - 1], 0, 0,
                          (n_i,) + (0,) * 10))
            gy_prev = self.alloc(n_i)
            out[f"gy{i - 1}"] = gy_prev
            nodes.append((22, go_i, slots[i - 1], gy_prev, 0,
                          (n_i, LRELU_SLOPE_BITS) + (0,) * 9))
        out["gx_post"] = gx_post
        out["gw_post"] = gw_post
        out["gb_post"] = gb_post
        return (nodes, go_buf, fm_bufs, out)

    def _build_s(self, convs, post_w, post_b, L_in, B=1):
        """构建一张 DiscriminatorS 链图（13 节点 + 33 节点 backward 段）。

        convs: [(w, b, s, p, gr), ...]×6；post conv1d k=3 s=1 pad=1。
        backward 段（T3-b）：post conv1d_bwd（op24/25/26 groups=1）+6×
        (leaky_bwd op22 + conv1d_groups_bwd op24/25/26) + FM 就地加（op3）。
        消费 fwd 槽位（leaky 就地覆写后符号判据等价）+ wpers 权重 + go/FM
        输入槽；输出 gx_in/gw_i/gb_i/gx_post/gy_i 槽（bwd_out 映射）。
        """
        nodes = []
        slots = []
        in_buf = self.alloc(B * 1 * L_in)   # 输入槽
        cur = in_buf
        w_bufs = [self._wpers_id(w) for (w, _, _, _, _) in convs]
        b_bufs = [self._wpers_id(b) for (_, b, _, _, _) in convs]
        pairs = []
        shapes = []
        convs_p = []
        L = L_in
        meta = []   # (c_in, ci_g, L, co, k, s, p) per conv（bwd 构建用）
        for i, ((w, b, s, p, gr), wb, bb) in enumerate(
                zip(convs, w_bufs, b_bufs)):
            co, ci_g, k = w.shape
            c_in = ci_g * gr
            Lo = (L + 2 * p - k) // s + 1
            o_buf = self.alloc(B * co * Lo)
            slots.append(o_buf)
            nodes.append((23, cur, wb, bb, 0,
                          (B, c_in, ci_g, L, co, k, s, p, p, o_buf)))
            nodes.append((6, o_buf, 0, 0, 0, (B * co * Lo, 0x3DCCCCCD)))
            pairs.append((o_buf, B * co * Lo, (B, co, Lo)))
            shapes.append((B, co, Lo))
            convs_p.append((s, p, gr))
            meta.append((c_in, ci_g, L, co, k, s, p))
            cur, L = o_buf, Lo
        n_post = B * 1 * (L + 2 * 1 - 3 + 1)
        po = self.alloc(n_post)
        nodes.append((2, cur, self._wpers_id(post_w), self._wpers_id(post_b),
                      0, (B, 1024, L, 1, 3, 1, 1, 1, 1, po)))
        dg = DiscGraph(self, "S", nodes, in_buf, pairs, (po, (B, 1, L + 2 - 3 + 1)),
                       po, shapes, convs_p)
        bwd = self._build_s_bwd(meta, convs, post_w, post_b, L_in, slots,
                                in_buf, B=B)
        dg._attach_bwd(bwd)
        return dg

    def _build_p_bwd(self, meta_p, convs, post_w, post_b, H_in, W_in, slots,
                     in_buf, B=1):
        """P 链 backward 图（T3-c 后续：判别器 P 链 backward 整链图执行器化）。

        meta_p: [(O_i, OH_i, OW_i)]×5（conv_{i+1} 输出形状，与 fwd slots 对齐）；
        post 输出 (1, OHp, OWp)（stride=1 pad=1 k=3 → OHp=OH4、OWp=OW4）。

        数据流（从后往前，与 conv2d_backward 链式分支 L4203-4221 逐位一致）：
          go_buf → [op20 transpose_b + op28 conv_t2d + op29 im2col_2d
                    + op1 matmul + op21 reduce_rows]（post，5 算子）
          copy gx_post→go5 + addInplace FM4 → leaky5_bwd(op22, x=slots[4])
          → gy5 → conv5_bwd(5 算子, x=slots[3], go=gy5) → gx4
          copy gx4→go4 + add FM3 → leaky4_bwd → gy4 → conv4_bwd → gx3 ...
          → conv1_bwd(x=in_buf) → gxin
        每层 conv2d_bwd 5 算子（与链式 br 同 engine kernel、同参数）：
          go_r=op20 transpose_b(go.reshape(B,O,OH*OW), B,O,OH*OW) [O,B*OH*OW]
          gx =op28 conv_t2d(go, w)：c 槽打包 opad（低 32 位 opad_h、高 32 位
                opad_w；bwd gx 无 bias，c 槽闲置复用为数值参数），out=gx 槽；
                p0=B p1=O(=c_in) p2=OH p3=OW p4=c_in(=c_out) p5=KH p6=KW
                p7=sh p8=sw p9=ph p10=pw；h_out/w_out 引擎几何推导
          xw =op29 im2col_2d(x)：12 维超 p[11] 槽 → c 槽复用 pw（数值，恒 0）；
                p0=B p1=C p2=H p3=W p4=OH p5=OW p6=KH p7=KW p8=sh p9=sw
                p10=ph
          gw =op1 matmul(go_r, xw) [O, C*KH*KW]
          gb =op21 reduce_rows(go_r) [O]
        opad（conv2d_backward L4181-4184 同式）：len_h=(OH-1)*sh-2*ph+KH，
        opad_h=H-len_h（P 层 stride_h=3 → (H-1)%3 ∈ {0,1,2}<sh 恒过引擎
        校验）；opad_w=W-((OW-1)*sw-2*pw+KW)=0（stride_w=1、fwd OW=W）。
        FM 加在独立 copy 槽（不污染 gx 输出槽——Python 侧 bp 返回纯梯度）。
        返回 (bwd_nodes, go_buf, fm_bufs, bwd_out)。
        """
        import os as _os  # noqa: PLC0415
        LRELU_SLOPE_BITS = 0x3DCCCCCD
        nodes = []
        out = {}
        fm_bufs = {}
        O4, OH4, OW4 = meta_p[4]          # conv5 输出（post 输入）
        OHp, OWp = OH4, OW4               # post stride=1 pad=1 k=3 → 同尺寸
        # T6b①：RVC_TRAIN_DISC_SINGLE_BUF=1 → 中间槽折叠两组。
        # 组X（go_r/go 系）：go_r_post, go5, go_r_i, go_i——生命周期串行
        # （各自 transpose/copy 写 → matmul/reduce 或 leaky 读后即死，下一
        # 段才写下一个），共享一个 max buffer。组Y（xw 系）：xw_post, xw_i
        # 同理串行。组X 与组Y 必须分离：matmul(op1) 同时读 go_r_i 与 xw_i。
        _fold = _os.environ.get("RVC_TRAIN_DISC_SINGLE_BUF", "0") == "1"
        if _fold:
            _x_sizes = [B * 1 * OHp * OWp, B * O4 * OH4 * OW4]   # go_r_post, go5
            _y_sizes = [B * OHp * OWp * O4 * 3]                  # xw_post
            for i in range(4, -1, -1):
                O_i, OH_i, OW_i = meta_p[i]
                _x_sizes.append(B * O_i * OH_i * OW_i)           # go_r_i
                _c_in_y = meta_p[i - 1][0] if i > 0 else 1
                _y_sizes.append(B * OH_i * OW_i * _c_in_y * 5)   # xw_i
                if i > 0:
                    _c_in, _H, _W = meta_p[i - 1]
                    _x_sizes.append(B * _c_in * _H * _W)         # go_i
            x_fold = self.alloc(max(_x_sizes))
            y_fold = self.alloc(max(_y_sizes))
            self.bwd_fold += [x_fold, y_fold]
        else:
            x_fold = y_fold = None
        # --- post conv2d_bwd（5 算子）---
        go_buf = self.alloc(B * 1 * OHp * OWp)          # post 输出梯度输入槽
        go_r_post = x_fold if _fold else self.alloc(B * 1 * OHp * OWp)
        nodes.append((20, go_buf, 0, go_r_post, 0,
                      (B, 1, OHp * OWp) + (0,) * 8))
        gx_post = self.alloc(B * O4 * OH4 * OW4)        # [B, O4, OH4, OW4]
        # opad：len_h=(OHp-1)*1-2*1+3=OHp → opad_h=OH4-OHp=0；opad_w=0
        nodes.append((28, go_buf, self._wpers_id(post_w), 0, gx_post,
                      (B, 1, OHp, OWp, O4, 3, 1, 1, 1, 1, 0)))
        xw_post = y_fold if _fold else self.alloc(B * OHp * OWp * O4 * 3)
        nodes.append((29, slots[4], 0, 0, xw_post,
                      (B, O4, OH4, OW4, OHp, OWp, 3, 1, 1, 1, 1)))
        gw_post = self.alloc(O4 * 3)                   # [1, O4*3]
        nodes.append((1, go_r_post, xw_post, gw_post, 0,
                      (1, B * OHp * OWp, O4 * 3)))
        gb_post = self.alloc(1)                        # [1]
        nodes.append((21, go_r_post, 0, gb_post, 0, (1, B * OHp * OWp)))
        # --- leaky5_bwd：go = gx_post + FM4（copy 独立槽再加，保 gx_post 纯）---
        n5 = B * O4 * OH4 * OW4
        go5 = x_fold if _fold else self.alloc(n5)
        fm_bufs[4] = self.alloc(n5)
        self.set_input(np.zeros(n5, np.float32), fm_bufs[4])
        nodes.append((7, go5, gx_post, 0, 0, (n5,) + (0,) * 10))
        nodes.append((3, go5, fm_bufs[4], 0, 0, (n5,) + (0,) * 10))
        gy5 = self.alloc(n5)
        out["gy4"] = gy5
        nodes.append((22, go5, slots[4], gy5, 0,
                      (n5, LRELU_SLOPE_BITS) + (0,) * 9))
        # --- 4..0 层（conv2d_bwd 5 算子 + leaky_i bwd + FM 加）---
        for i in range(4, -1, -1):
            O_i, OH_i, OW_i = meta_p[i]   # conv_{i+1} 输出（go 形状）
            if i > 0:
                c_in, H, W = meta_p[i - 1]   # conv_i 输入 = 上一层输出
                gx_i = self.alloc(B * c_in * H * W)
                x_ref = slots[i - 1]
            else:
                c_in, H, W = 1, H_in, W_in
                gx_i = self.alloc(B * 1 * H_in * W_in)   # gxin（到输入梯度）
                x_ref = in_buf
            gy_i = out[f"gy{i}"]            # conv_{i+1} 输入梯度 = leaky_{i+1} 输出
            w_i_id = self._wpers_id(convs[i][0])
            len_h = (OH_i - 1) * 3 - 2 * 2 + 5          # sh=3 ph=2 KH=5
            opad_h = int(H - len_h)
            c_pack = opad_h | (0 << 32)                # opad_w 恒 0
            nodes.append((28, gy_i, w_i_id, c_pack, gx_i,
                          (B, O_i, OH_i, OW_i, c_in, 5, 1, 3, 1, 2, 0)))
            xw_i = y_fold if _fold else self.alloc(B * OH_i * OW_i * c_in * 5)
            nodes.append((29, x_ref, 0, 0, xw_i,
                          (B, c_in, H, W, OH_i, OW_i, 5, 1, 3, 1, 2)))
            go_r_i = x_fold if _fold else self.alloc(B * O_i * OH_i * OW_i)
            nodes.append((20, gy_i, 0, go_r_i, 0,
                          (B, O_i, OH_i * OW_i) + (0,) * 8))
            gw_i = self.alloc(O_i * c_in * 5)          # [O_i, c_in*5]
            nodes.append((1, go_r_i, xw_i, gw_i, 0,
                          (O_i, B * OH_i * OW_i, c_in * 5)))
            gb_i = self.alloc(O_i)                     # [O_i]
            nodes.append((21, go_r_i, 0, gb_i, 0, (O_i, B * OH_i * OW_i)))
            out[f"gx{i - 1 if i > 0 else 'in'}"] = gx_i
            out[f"gw{i}"] = gw_i
            out[f"gb{i}"] = gb_i
            if i == 0:
                break
            # leaky_i bwd：go = gx_i + FM_{i-1}（copy + add），x=slots[i-1]
            n_i = B * c_in * H * W
            go_i = x_fold if _fold else self.alloc(n_i)
            fm_bufs[i - 1] = self.alloc(n_i)
            self.set_input(np.zeros(n_i, np.float32), fm_bufs[i - 1])
            nodes.append((7, go_i, gx_i, 0, 0, (n_i,) + (0,) * 10))
            nodes.append((3, go_i, fm_bufs[i - 1], 0, 0, (n_i,) + (0,) * 10))
            gy_prev = self.alloc(n_i)
            out[f"gy{i - 1}"] = gy_prev
            nodes.append((22, go_i, slots[i - 1], gy_prev, 0,
                          (n_i, LRELU_SLOPE_BITS) + (0,) * 9))
        out["gx_post"] = gx_post
        out["gw_post"] = gw_post
        out["gb_post"] = gb_post
        return (nodes, go_buf, fm_bufs, out)

    def _build_p(self, convs, post_w, post_b, H_in, W_in, B=1):
        """构建一张 DiscriminatorP 链图（11 节点）。convs: [(w, b)]×5，
        w=[O,C,5,1]；conv2d stride=(3,1) pad=(2,0)；post (3,1) pad=(1,0)。
        """
        nodes = []
        in_buf = self.alloc(B * 1 * H_in * W_in)
        cur = in_buf
        w_bufs = [self._wpers_id(w) for (w, _) in convs]
        b_bufs = [self._wpers_id(b) for (_, b) in convs]
        pairs = []
        shapes = []
        convs_p = []
        slots = []          # conv 输出槽（bwd 图消费：leaky x / conv x）
        meta_p = []         # (O, OH, OW) per conv（bwd 图构建用）
        h, wd, C = H_in, W_in, 1
        for i, ((w, b), wb, bb) in enumerate(zip(convs, w_bufs, b_bufs)):
            O, _, KH, KW = w.shape
            OH = (h + 2 * 2 - KH) // 3 + 1
            OW = (wd + 2 * 0 - KW) // 1 + 1
            o_buf = self.alloc(B * O * OH * OW)
            nodes.append((27, cur, wb, bb, o_buf,
                          (B, C, h, wd, O, KH, KW, 2, 0, 3, 1)))
            nodes.append((6, o_buf, 0, 0, 0, (B * O * OH * OW, 0x3DCCCCCD)))
            pairs.append((o_buf, B * O * OH * OW, (B, O, OH, OW)))
            shapes.append((B, O, OH, OW))
            convs_p.append((DISCRIMINATOR_P_S, DISCRIMINATOR_P_PAD))
            slots.append(o_buf)
            meta_p.append((O, OH, OW))
            cur, h, wd, C = o_buf, OH, OW, O
        O, _, KH, KW = post_w.shape
        OH = (h + 2 * 1 - KH) // 1 + 1
        OW = (wd + 2 * 0 - KW) // 1 + 1
        po = self.alloc(B * O * OH * OW)
        nodes.append((27, cur, self._wpers_id(post_w), self._wpers_id(post_b),
                      po, (B, C, h, wd, O, KH, KW, 1, 0, 1, 1)))
        # T3-c 后续：P 链 backward 整链图（conv2d_bwd 5 算子/层，见 _build_p_bwd）
        bwd = self._build_p_bwd(meta_p, convs, post_w, post_b, H_in, W_in,
                                slots, in_buf, B=B)
        dg = DiscGraph(self, "P", nodes, in_buf, pairs, (po, (B, O, OH, OW)),
                       po, shapes, convs_p, bwd)
        return dg

    def disc_graphs(self, key, build_fn):
        """key: 判别器标识；build_fn → DiscGraph。返回 [A, B]（双缓冲）。"""
        pair = self._disc_cache.get(key)
        if pair is None:
            pair = [build_fn(), build_fn()]
            self._disc_cache[key] = pair
        return pair

    def dec_graph(self, key, build_fn):
        """T3-c：dec 前向图缓存（单图，key 如 ("dec", B, Tf)）。

        与判别器双缓冲不同：dec 权重/输入每步覆写，无 ping-pong 需求。结构
        变化（B/Tf/权重形状）由 key 区分；旧图保留（一般仅 1-2 个组合），
        release_all 统一释放。
        """
        dg = self._dec_cache.get(key)
        if dg is None:
            dg = build_fn()
            dg.key = key
            self._dec_cache[key] = dg
        return dg

    def enc_graph(self, key, build_fn):
        """T3-d：enc 前向分段图集缓存（key=("enc", B, P)）。

        与 dec 单图不同：attn 链含大量 Python 视图衔接（无图 op 可表达），
        故为多图集（每图一次 run 提交）；权重/输入每步覆写 → 图绑固定槽 id，
        结构变化（B/P/权重形状）由 key 区分，旧图保留，release_all 统一释放。
        """
        eg = self._enc_cache.get(key)
        if eg is None:
            eg = build_fn()
            eg.key = key
            self._enc_cache[key] = eg
        return eg

    def encq_graph(self, key, build_fn):
        """T4-1：enc_q（PosteriorEncoder）前向分段图集缓存（key=("encq", B, F)）。

        与 enc_graph 同语义：多图集、固定槽每步 set_input 覆写；结构变化由
        key 区分。复用 _enc_cache（key 前缀天然隔离），release_all 统一释放。
        """
        eg = self._enc_cache.get(key)
        if eg is None:
            eg = build_fn()
            eg.key = key
            self._enc_cache[key] = eg
        return eg

    def flow_graph(self, key, build_fn):
        """T4-1：flow（ResidualCouplingBlock）前向分段图集缓存（key=("flow", B, F)）。

        同 encq_graph 语义；键名带层前缀 l{idx}.（idx=0,2,4,6）。
        """
        eg = self._enc_cache.get(key)
        if eg is None:
            eg = build_fn()
            eg.key = key
            self._enc_cache[key] = eg
        return eg

    # -- enc 前向分段图集（T3-d）-----------------------------------------
    def _build_enc(self, enc, cfg, B, P, phone_dim):
        """构建 TextEncoderTrain 前向分段图集。enc: TextEncoderTrain（只读
        权重形状）。节点严格拓扑序（与 forward_br 值依赖一致）；权重/输入
        固定槽每步 set_input 覆写。分段（段间 Python 衔接，见 forward_graph）：
          G_emb: matmul(phone2d×emb_w) + bias_add + add(pitch) + mul(√hid)
                 + leaky(0.1) → x0 [B,P,hid]           （5 节点）
          层 i G1: conv_q/k/v（3 节点）→ q/k/v [B,hid,P]
          层 i G2: 12×matmul（每 bh：scores=qs×khT、rel=qs×used_kT）
          层 i G3: 每 bh add(rel_abs)+mul(mask)+add(neg)+softmax → p_attn
          层 i G4: 每 bh matmul(p_attn×vh)+matmul(rel_w×used_v)+add → out
          层 i G5a: conv_o + copy(x1=x)+add(x1+=attn_o)
          层 i G5b: layer_norm1（in_xt→xln）
          层 i G5c: conv_1(pad1)+relu+conv_2(pad1)+copy(x3)+add(x3+=y2)
          层 i G5d: layer_norm2（in_xt2→xln2）
          G_proj: mul_inplace(x×xmask)+conv(proj) → stats [B,2hid,P]
        常量/只读共享槽：khT [kc,P]、used_kT [kc,M]、used_v [M,kc]、mask/
        neg [P,P]（每步由 attn_mask 覆写）；per-bh 槽 khT.{i}.{b}
        [kc,P]、qs/rel_abs/vh/rel_w 每步覆写；sqrt_full build 预填 √hid。
        """
        LRELU01 = 0x3DCCCCCD          # leaky slope 0.1
        EPS_BITS = 0x3727C5AC         # layer_norm eps=1e-5（f32 bits）
        nodes = {}
        ghs = {}
        wslots = {}
        slots = {}
        outs = {}

        def alloc_w(nfloats, name):
            bid = self.alloc(int(nfloats))
            wslots[name] = bid
            return bid

        def alloc_s(nfloats, name):
            bid = self.alloc(int(nfloats))
            slots[name] = bid
            return bid

        def out_entry(name, bid, shape):
            outs[name] = (bid, tuple(shape))

        hid = int(cfg.hidden)
        kc = int(cfg.k_channels)      # enc 头维度 = k_channels（emb_kc 不存在）
        heads = int(cfg.n_heads)
        n_layers = int(cfg.n_layers)
        filt = int(cfg.filter)
        M = 2 * P - 1
        n_emb = int(B * P * hid)
        n_ph = int(B * P * phone_dim)
        np_ = int(P * P)
        nkv = int(P * kc)
        nrel = int(P * M)
        nattn = int(B * hid * P)

        # -- 输入/常量/中间槽（段间 Python 衔接点）--
        ph_buf = alloc_s(n_ph, "phone")
        pe_buf = alloc_s(n_emb, "pe")
        xmask_buf = alloc_s(n_emb, "xmask")        # xmask_full [B,hid,P] 每步覆写
        sqrt_buf = alloc_s(n_emb, "sqrt_full")     # √hid 常量，build 预填一次
        self.set_input(np.full(n_emb, np.sqrt(np.float32(hid)), np.float32),
                       sqrt_buf)
        x_buf = alloc_s(n_emb, "x")                # 层输入/收尾输入 [B,hid,P]
        lin_buf = alloc_s(n_emb, "lin")            # emb linear+bias+add+mul+leaky 覆写
        attn_in_buf = alloc_s(nattn, "attn_in")
        attn_o_buf = alloc_s(nattn, "attn_o")
        x1_buf = alloc_s(nattn, "x1")
        in_xt_buf = alloc_s(n_emb, "in_xt")
        xln_buf = alloc_s(n_emb, "xln")
        in_x2_buf = alloc_s(n_emb, "in_x2")
        y1_buf = alloc_s(int(B * P * filt), "y1")
        y2_buf = alloc_s(n_emb, "y2")
        x3_buf = alloc_s(n_emb, "x3")
        in_xt2_buf = alloc_s(n_emb, "in_xt2")
        xln2_buf = alloc_s(n_emb, "xln2")
        stats_buf = alloc_s(int(B * 2 * hid * P), "stats")
        used_kT_buf = alloc_s(int(kc * M), "used_kT")
        used_v_buf = alloc_s(int(M * kc), "used_v")
        mask_buf = alloc_s(np_, "mask")            # attn_mask[0,0] 每步覆写
        neg_buf = alloc_s(np_, "neg")              # (1-mask)*-1e4 每步覆写
        qkv = {}
        qs = {}
        scores = {}
        rel = {}
        rel_abs = {}
        p_attn = {}
        vh = {}
        rel_w = {}
        o1 = {}
        o2 = {}
        khT = {}                                   # per-bh（每 bh 的 k^T 不同）
        nbh = B * heads
        for i in range(n_layers):
            qkv[i] = [alloc_s(nattn, f"q{i}"),
                      alloc_s(nattn, f"k{i}"),
                      alloc_s(nattn, f"v{i}")]
            for b in range(nbh):
                qs[f"{i}.{b}"] = alloc_s(nkv, f"qs.{i}.{b}")
                scores[f"{i}.{b}"] = alloc_s(np_, f"scores.{i}.{b}")
                rel[f"{i}.{b}"] = alloc_s(nrel, f"rel.{i}.{b}")
                rel_abs[f"{i}.{b}"] = alloc_s(np_, f"rel_abs.{i}.{b}")
                p_attn[f"{i}.{b}"] = alloc_s(np_, f"p_attn.{i}.{b}")
                vh[f"{i}.{b}"] = alloc_s(nkv, f"vh.{i}.{b}")
                rel_w[f"{i}.{b}"] = alloc_s(nrel, f"rel_w.{i}.{b}")
                o1[f"{i}.{b}"] = alloc_s(nkv, f"o1.{i}.{b}")
                o2[f"{i}.{b}"] = alloc_s(nkv, f"o2.{i}.{b}")
                khT[f"{i}.{b}"] = alloc_s(nkv, f"khT.{i}.{b}")

        # -- 权重固定槽（元素数=数组大小；内容每步覆写）--
        alloc_w(int(phone_dim * hid), "emb_w")
        alloc_w(hid, "emb_b")
        for i in range(n_layers):
            alloc_w(int(hid * hid), f"l{i}.q_w"); alloc_w(hid, f"l{i}.q_b")
            alloc_w(int(hid * hid), f"l{i}.k_w"); alloc_w(hid, f"l{i}.k_b")
            alloc_w(int(hid * hid), f"l{i}.v_w"); alloc_w(hid, f"l{i}.v_b")
            alloc_w(int(hid * hid), f"l{i}.o_w"); alloc_w(hid, f"l{i}.o_b")
            # ffn conv 权重 [O, Ci, K=3]：槽容量需含 kernel 宽度（K=3）
            alloc_w(int(hid * filt * 3), f"l{i}.c1_w"); alloc_w(filt, f"l{i}.c1_b")
            alloc_w(int(filt * hid * 3), f"l{i}.c2_w"); alloc_w(hid, f"l{i}.c2_b")
            alloc_w(hid, f"l{i}.n1_g"); alloc_w(hid, f"l{i}.n1_b")
            alloc_w(hid, f"l{i}.n2_g"); alloc_w(hid, f"l{i}.n2_b")
        alloc_w(int(2 * hid * hid), "proj_w")
        alloc_w(2 * hid, "proj_b")

        # -- G_emb：linear(phone2d×emb_w)+bias+add(pe)+mul(√hid)+leaky --
        gemb = [
            (1, ph_buf, wslots["emb_w"], lin_buf, 0,
             (B * P, phone_dim, hid) + (0,) * 8),
            (11, lin_buf, wslots["emb_b"], lin_buf, 0,
             (n_emb, hid) + (0,) * 9),
            (3, lin_buf, pe_buf, lin_buf, 0, (n_emb,) + (0,) * 10),
            (4, lin_buf, sqrt_buf, lin_buf, 0, (n_emb,) + (0,) * 10),
            (6, lin_buf, 0, 0, 0, (n_emb, LRELU01) + (0,) * 9),
        ]
        out_entry("lin", lin_buf, (B * P, hid))      # matmul+bias 后
        out_entry("lin_a", lin_buf, (B * P, hid))    # +pitch 后
        out_entry("lin_mc", lin_buf, (B * P, hid))   # ×√hid 后
        out_entry("x0", lin_buf, (B * P, hid))       # leaky 后（emb 输出，逐元素于 [B*P,hid]）
        ghs["emb"] = self.build(gemb)

        # -- 层图 --
        for i in range(n_layers):
            qb, kb_, vb = qkv[i]
            g1 = [
                (2, x_buf, wslots[f"l{i}.q_w"], wslots[f"l{i}.q_b"], 0,
                 (B, hid, P, hid, 1, 1, 0, 0, 1, qb, 0)),
                (2, x_buf, wslots[f"l{i}.k_w"], wslots[f"l{i}.k_b"], 0,
                 (B, hid, P, hid, 1, 1, 0, 0, 1, kb_, 0)),
                (2, x_buf, wslots[f"l{i}.v_w"], wslots[f"l{i}.v_b"], 0,
                 (B, hid, P, hid, 1, 1, 0, 0, 1, vb, 0)),
            ]
            out_entry(f"l{i}.q", qb, (B, hid, P))
            out_entry(f"l{i}.k", kb_, (B, hid, P))
            out_entry(f"l{i}.v", vb, (B, hid, P))
            ghs[f"l{i}.g1"] = self.build(g1)

            g2 = []
            for b in range(nbh):
                g2.append((1, qs[f"{i}.{b}"], khT[f"{i}.{b}"],
                           scores[f"{i}.{b}"], 0, (P, kc, P) + (0,) * 8))
                g2.append((1, qs[f"{i}.{b}"], used_kT_buf,
                           rel[f"{i}.{b}"], 0, (P, kc, M) + (0,) * 8))
                out_entry(f"l{i}.scores.{b}", scores[f"{i}.{b}"], (P, P))
                out_entry(f"l{i}.rel.{b}", rel[f"{i}.{b}"], (P, M))
            ghs[f"l{i}.g2"] = self.build(g2)

            g3 = []
            for b in range(nbh):
                sc = scores[f"{i}.{b}"]
                pa = p_attn[f"{i}.{b}"]
                g3.append((3, sc, rel_abs[f"{i}.{b}"], sc, 0,
                           (np_,) + (0,) * 10))
                g3.append((4, sc, mask_buf, sc, 0, (np_,) + (0,) * 10))
                g3.append((3, sc, neg_buf, sc, 0, (np_,) + (0,) * 10))
                g3.append((8, sc, 0, pa, 0, (P, P) + (0,) * 9))
                out_entry(f"l{i}.s_a.{b}", sc, (P, P))
                out_entry(f"l{i}.s_m.{b}", sc, (P, P))
                out_entry(f"l{i}.s_n.{b}", sc, (P, P))
                out_entry(f"l{i}.p_attn.{b}", pa, (P, P))
            ghs[f"l{i}.g3"] = self.build(g3)

            g4 = []
            for b in range(nbh):
                o1_id = o1[f"{i}.{b}"]
                g4.append((1, p_attn[f"{i}.{b}"], vh[f"{i}.{b}"], o1_id, 0,
                           (P, P, kc) + (0,) * 8))
                g4.append((1, rel_w[f"{i}.{b}"], used_v_buf, o2[f"{i}.{b}"],
                           0, (P, M, kc) + (0,) * 8))
                g4.append((3, o1_id, o2[f"{i}.{b}"], o1_id, 0,
                           (nkv,) + (0,) * 10))
                out_entry(f"l{i}.o1.{b}", o1_id, (P, kc))
                out_entry(f"l{i}.o2.{b}", o2[f"{i}.{b}"], (P, kc))
                out_entry(f"l{i}.out.{b}", o1_id, (P, kc))
            ghs[f"l{i}.g4"] = self.build(g4)

            g5a = [
                (2, attn_in_buf, wslots[f"l{i}.o_w"], wslots[f"l{i}.o_b"], 0,
                 (B, hid, P, hid, 1, 1, 0, 0, 1, attn_o_buf, 0)),
                (7, x1_buf, x_buf, 0, 0, (nattn,) + (0,) * 10),
                (3, x1_buf, attn_o_buf, x1_buf, 0, (nattn,) + (0,) * 10),
            ]
            out_entry(f"l{i}.attn_o", attn_o_buf, (B, hid, P))
            out_entry(f"l{i}.x1", x1_buf, (B, hid, P))
            ghs[f"l{i}.g5a"] = self.build(g5a)

            g5b = [
                (9, in_xt_buf, wslots[f"l{i}.n1_g"], wslots[f"l{i}.n1_b"], 0,
                 (B * P, hid, EPS_BITS, 0, 0, 0, 0, 0, 0, xln_buf, 0)),
            ]
            out_entry(f"l{i}.xln", xln_buf, (B, P, hid))
            ghs[f"l{i}.g5b"] = self.build(g5b)

            g5c = [
                # conv1d p: (B, c_in, l, c_out, k, stride, pad_l, pad_r, dil, out@p9)
                (2, in_x2_buf, wslots[f"l{i}.c1_w"], wslots[f"l{i}.c1_b"], 0,
                 (B, hid, P, filt, 3, 1, 1, 1, 1, y1_buf, 0)),
                (14, y1_buf, 0, 0, 0, (int(B * P * filt),) + (0,) * 10),
                (2, y1_buf, wslots[f"l{i}.c2_w"], wslots[f"l{i}.c2_b"], 0,
                 (B, filt, P, hid, 3, 1, 1, 1, 1, y2_buf, 0)),
                (7, x3_buf, in_x2_buf, 0, 0, (n_emb,) + (0,) * 10),
                (3, x3_buf, y2_buf, x3_buf, 0, (n_emb,) + (0,) * 10),
            ]
            out_entry(f"l{i}.y1", y1_buf, (B, filt, P))
            out_entry(f"l{i}.y1r", y1_buf, (B, filt, P))
            out_entry(f"l{i}.y2", y2_buf, (B, hid, P))
            out_entry(f"l{i}.x3", x3_buf, (B, hid, P))
            ghs[f"l{i}.g5c"] = self.build(g5c)

            g5d = [
                (9, in_xt2_buf, wslots[f"l{i}.n2_g"], wslots[f"l{i}.n2_b"], 0,
                 (B * P, hid, EPS_BITS, 0, 0, 0, 0, 0, 0, xln2_buf, 0)),
            ]
            out_entry(f"l{i}.xln2", xln2_buf, (B, P, hid))
            ghs[f"l{i}.g5d"] = self.build(g5d)

        # -- G_proj：二次 mask（mul_inplace）+ conv(proj) → stats --
        gpr = [
            (4, x_buf, xmask_buf, x_buf, 0, (n_emb,) + (0,) * 10),
            (2, x_buf, wslots["proj_w"], wslots["proj_b"], 0,
             (B, hid, P, 2 * hid, 1, 1, 0, 0, 1, stats_buf, 0)),
        ]
        out_entry("x_last_m", x_buf, (B, hid, P))   # 收尾 mul 后（x_buf 覆写）
        out_entry("stats", stats_buf, (B, 2 * hid, P))
        ghs["proj"] = self.build(gpr)

        return EncGraph(self, ("enc", B, P), ghs, slots, wslots, outs)

    # -- enc_q 前向分段图集（T4-1）----------------------------------------
    def _build_encq(self, encq, B, F):
        """构建 PosteriorEncoder（enc_q）前向分段图集。encq: PosteriorEncoder
        （只读权重数组）。段图（键名对齐 vits_train _forward_graph 衔接）：
          pre : conv1d(x→h) + mul_inplace(xmask) → h        （2 节点）
          cond: conv1d(g→g_cond)                             （gin≠0 时，1 节点）
          in.{i} / rs.{i}: 每层 WN 的 in / res_skip conv（各 1 节点；
                          kernel/pad 由权重 shape 推导）
          proj: conv1d(h→stats) + mul_inplace(xmask) → stats（2 节点）
        conv1d 图节点 p=(B, Ci, L, Co, K, s, pad_l, pad_r, dil, out@p9, 0)；
        mul/add 节点 p=(n,)+(0,)*10。xmask 槽按 [B,2*hid,F] 分配：pre mul
        只读前 n_hid 元素、proj mul 读全量（值同 broadcast 前缀一致）。
        """
        hid = int(encq.hidden)
        wn = encq.wn
        n_layers = wn.n_layers
        pre_w = np.asarray(encq.pre_w)
        proj_w = np.asarray(encq.proj_w)
        spec_ch = int(pre_w.shape[1])
        K_pre = int(pre_w.shape[2])
        pad_pre = (K_pre - 1) // 2
        K_proj = int(proj_w.shape[2])
        pad_proj = (K_proj - 1) // 2
        gin = 0 if wn.cond_w is None else int(wn.cond_w.shape[1])
        n_hid = int(B * hid * F)
        n_2h = int(B * 2 * hid * F)
        n_spec = int(B * spec_ch * F)

        ghs, wslots, slots, outs = {}, {}, {}, {}

        def alloc_w(n, name):
            bid = self.alloc(int(n))
            wslots[name] = bid
            return bid

        def alloc_s(n, name):
            bid = self.alloc(int(n))
            slots[name] = bid
            return bid

        def out_entry(name, bid, shape):
            outs[name] = (bid, tuple(shape))

        # -- 输入/中间槽 --
        x_buf = alloc_s(n_spec, "x")               # spec [B,spec_ch,F]
        xmask_buf = alloc_s(n_2h, "xmask")         # xmask_full [B,2hid,F]
        h_buf = alloc_s(n_hid, "h")                # pre 输出 / proj 输入
        cur_buf = alloc_s(n_hid, "cur")
        acts_buf = alloc_s(n_hid, "acts")
        xin_buf = alloc_s(n_2h, "xin")
        rs_buf = alloc_s(n_2h, "rs")
        stats_buf = alloc_s(n_2h, "stats")
        if gin:
            g_buf = alloc_s(int(B * gin), "g")     # g [B,gin,1]
            gc_buf = alloc_s(int(B * 2 * hid * n_layers), "g_cond")
        # -- 权重槽 --
        alloc_w(int(pre_w.size), "pre_w")
        alloc_w(int(encq.pre_b.size), "pre_b")
        alloc_w(int(proj_w.size), "proj_w")
        alloc_w(int(encq.proj_b.size), "proj_b")
        if gin:
            alloc_w(int(wn.cond_w.size), "cond_w")
            alloc_w(int(wn.cond_b.size), "cond_b")
        for i in range(n_layers):
            alloc_w(int(wn.in_w[i].size), f"in.{i}.w")
            alloc_w(int(wn.in_b[i].size), f"in.{i}.b")
            alloc_w(int(wn.rs_w[i].size), f"rs.{i}.w")
            alloc_w(int(wn.rs_b[i].size), f"rs.{i}.b")

        # -- pre：conv + 输出侧 mask --
        gpre = [
            (2, x_buf, wslots["pre_w"], wslots["pre_b"], 0,
             (B, spec_ch, F, hid, K_pre, 1, pad_pre, pad_pre, 1, h_buf, 0)),
            (4, h_buf, xmask_buf, h_buf, 0, (n_hid,) + (0,) * 10),
        ]
        out_entry("h0", h_buf, (B, hid, F))
        out_entry("h", h_buf, (B, hid, F))
        ghs["pre"] = self.build(gpre)
        if gin:
            gcond = [
                (2, g_buf, wslots["cond_w"], wslots["cond_b"], 0,
                 (B, gin, 1, 2 * hid * n_layers, 1, 1, 0, 0, 1, gc_buf, 0)),
            ]
            out_entry("g_cond", gc_buf, (B, 2 * hid * n_layers, 1))
            ghs["cond"] = self.build(gcond)
        # -- 每层 WN 段图 --
        for i in range(n_layers):
            Kw = int(wn.in_w[i].shape[2])
            pw = (Kw - 1) // 2
            ghs[f"in.{i}"] = self.build([
                (2, cur_buf, wslots[f"in.{i}.w"], wslots[f"in.{i}.b"], 0,
                 (B, hid, F, 2 * hid, Kw, 1, pw, pw, 1, xin_buf, 0))])
            out_entry(f"in.{i}", xin_buf, (B, 2 * hid, F))
            Kr = int(wn.rs_w[i].shape[2])
            pr = (Kr - 1) // 2
            Co_rs = int(wn.rs_w[i].shape[0])  # 末层=hid，其余=2*hid
            ghs[f"rs.{i}"] = self.build([
                (2, acts_buf, wslots[f"rs.{i}.w"], wslots[f"rs.{i}.b"], 0,
                 (B, hid, F, Co_rs, Kr, 1, pr, pr, 1, rs_buf, 0))])
            out_entry(f"rs.{i}", rs_buf, (B, Co_rs, F))
        # -- proj：conv + 输出侧 mask --
        gproj = [
            (2, h_buf, wslots["proj_w"], wslots["proj_b"], 0,
             (B, hid, F, 2 * hid, K_proj, 1, pad_proj, pad_proj, 1,
              stats_buf, 0)),
            (4, stats_buf, xmask_buf, stats_buf, 0, (n_2h,) + (0,) * 10),
        ]
        out_entry("stats0", stats_buf, (B, 2 * hid, F))
        out_entry("stats", stats_buf, (B, 2 * hid, F))
        ghs["proj"] = self.build(gproj)

        return EncGraph(self, ("encq", B, F), ghs, slots, wslots, outs)

    # -- flow 前向分段图集（T4-1）-----------------------------------------
    def _build_flow(self, block, B, F):
        """构建 ResidualCouplingBlock（flow）前向分段图集。block:
        ResidualCouplingBlockTrain（4 层 idx 0,2,4,6）。每层 8 段图：
          l{idx}.pre / l{idx}.cond（gin≠0）/ l{idx}.in.{i} / l{idx}.rs.{i}
          （i=0..2）/ l{idx}.post；段间 Python 衔接（slice/add/mul/
          concat/flip 为 tape op）。4 层共享输入/中间槽（顺序 run 无冲突，
          权重槽带层前缀）。conv 参数（K/pad）由权重 shape 推导。
        """
        layer0 = block.layers[0]
        hid = int(layer0.wn.hidden)
        half = hid // 2
        n_hid = int(B * hid * F)
        n_half = int(B * half * F)
        n_2h = int(B * 2 * hid * F)
        gin = 0 if layer0.wn.cond_w is None else int(layer0.wn.cond_w.shape[1])

        ghs, wslots, slots, outs = {}, {}, {}, {}

        def alloc_w(n, name):
            bid = self.alloc(int(n))
            wslots[name] = bid
            return bid

        def alloc_s(n, name):
            bid = self.alloc(int(n))
            slots[name] = bid
            return bid

        def out_entry(name, bid, shape):
            outs[name] = (bid, tuple(shape))

        x0_buf = alloc_s(n_half, "x0")             # x0 [B,half,F]（每层覆写）
        maskh_buf = alloc_s(n_hid, "mask_hid")     # xmask_full [B,hid,F]
        maskhf_buf = alloc_s(n_half, "mask_half")  # xmask_full [B,half,F]
        h_buf = alloc_s(n_hid, "h")
        cur_buf = alloc_s(n_hid, "cur")
        acts_buf = alloc_s(n_hid, "acts")
        xin_buf = alloc_s(n_2h, "xin")
        rs_buf = alloc_s(n_2h, "rs")
        m_buf = alloc_s(n_half, "m")
        if gin:
            g_buf = alloc_s(int(B * gin), "g")
            gc_buf = alloc_s(int(B * 2 * hid * 3), "g_cond")
        for j, layer in enumerate(block.layers):
            idx = j * 2                              # 0, 2, 4, 6
            wn = layer.wn
            pre_w = np.asarray(layer.pre_w)
            post_w = np.asarray(layer.post_w)
            K_pre = int(pre_w.shape[2])
            pad_pre = (K_pre - 1) // 2
            K_post = int(post_w.shape[2])
            pad_post = (K_post - 1) // 2
            Ci_pre = int(pre_w.shape[1])
            alloc_w(int(pre_w.size), f"l{idx}.pre_w")
            alloc_w(int(layer.pre_b.size), f"l{idx}.pre_b")
            alloc_w(int(post_w.size), f"l{idx}.post_w")
            alloc_w(int(layer.post_b.size), f"l{idx}.post_b")
            if gin:
                alloc_w(int(wn.cond_w.size), f"l{idx}.cond_w")
                alloc_w(int(wn.cond_b.size), f"l{idx}.cond_b")
            for i in range(wn.n_layers):
                alloc_w(int(wn.in_w[i].size), f"l{idx}.in.{i}.w")
                alloc_w(int(wn.in_b[i].size), f"l{idx}.in.{i}.b")
                alloc_w(int(wn.rs_w[i].size), f"l{idx}.rs.{i}.w")
                alloc_w(int(wn.rs_b[i].size), f"l{idx}.rs.{i}.b")

            ghs[f"l{idx}.pre"] = self.build([
                (2, x0_buf, wslots[f"l{idx}.pre_w"], wslots[f"l{idx}.pre_b"],
                 0, (B, Ci_pre, F, hid, K_pre, 1, pad_pre, pad_pre, 1,
                     h_buf, 0)),
                (4, h_buf, maskh_buf, h_buf, 0, (n_hid,) + (0,) * 10),
            ])
            out_entry(f"l{idx}.h0", h_buf, (B, hid, F))
            out_entry(f"l{idx}.h", h_buf, (B, hid, F))
            if gin:
                ghs[f"l{idx}.cond"] = self.build([
                    (2, g_buf, wslots[f"l{idx}.cond_w"],
                     wslots[f"l{idx}.cond_b"], 0,
                     (B, gin, 1, 2 * hid * wn.n_layers, 1, 1, 0, 0, 1,
                      gc_buf, 0))])
                out_entry(f"l{idx}.g_cond", gc_buf,
                          (B, 2 * hid * wn.n_layers, 1))
            for i in range(wn.n_layers):
                Kw = int(wn.in_w[i].shape[2])
                pw = (Kw - 1) // 2
                ghs[f"l{idx}.in.{i}"] = self.build([
                    (2, cur_buf, wslots[f"l{idx}.in.{i}.w"],
                     wslots[f"l{idx}.in.{i}.b"], 0,
                     (B, hid, F, 2 * hid, Kw, 1, pw, pw, 1, xin_buf, 0))])
                out_entry(f"l{idx}.in.{i}", xin_buf, (B, 2 * hid, F))
                Kr = int(wn.rs_w[i].shape[2])
                pr = (Kr - 1) // 2
                Co_rs = int(wn.rs_w[i].shape[0])  # 末层=hid，其余=2*hid
                ghs[f"l{idx}.rs.{i}"] = self.build([
                    (2, acts_buf, wslots[f"l{idx}.rs.{i}.w"],
                     wslots[f"l{idx}.rs.{i}.b"], 0,
                     (B, hid, F, Co_rs, Kr, 1, pr, pr, 1, rs_buf, 0))])
                out_entry(f"l{idx}.rs.{i}", rs_buf, (B, Co_rs, F))
            ghs[f"l{idx}.post"] = self.build([
                (2, h_buf, wslots[f"l{idx}.post_w"], wslots[f"l{idx}.post_b"],
                 0, (B, hid, F, half, K_post, 1, pad_post, pad_post, 1,
                     m_buf, 0)),
                (4, m_buf, maskhf_buf, m_buf, 0, (n_half,) + (0,) * 10),
            ])
            out_entry(f"l{idx}.m0", m_buf, (B, half, F))
            out_entry(f"l{idx}.m", m_buf, (B, half, F))

        return EncGraph(self, ("flow", B, F), ghs, slots, wslots, outs)

    # -- dec 生成器整链图（T3-c）-----------------------------------------
    def _build_dec(self, dec, cfg, B, Tf, har_L):
        """构建 dec 前向整链单图。dec: GeneratorNSFTrain（只读权重形状）。
        节点严格拓扑序（与 _forward_br 值依赖一致）；权重槽固定，每步
        set_input 覆写（去 wpers 缓存，因 deweight 每步新数组）。
        buffer id 由 alloc 分配（独立段约定 100+，非强制）。
        返回 DecGraph。
        """
        LRELU01 = 0x3DCCCCCD   # leaky slope 0.1
        LRELU001 = 0x3C23D70A  # leaky slope 0.01
        nodes = []
        wslots = {}
        outs = {}

        def alloc_w(nfloats, name):
            bid = self.alloc(int(nfloats))
            wslots[name] = bid
            return bid

        def out_entry(name, bid, shape):
            outs[name] = (bid, tuple(shape))

        # -- 权重固定槽（元素数=数组大小；内容每步覆写）--
        alloc_w(np.asarray(dec.conv_pre_w).size, "pre_w")
        alloc_w(np.asarray(dec.conv_pre_b).size, "pre_b")
        alloc_w(np.asarray(dec.conv_post_w).size, "post_w")
        for i in range(dec.n_ups):
            alloc_w(np.asarray(dec.ups_w[i]).size, f"up.{i}.w")
            alloc_w(np.asarray(dec.ups_b[i]).size, f"up.{i}.b")
            alloc_w(np.asarray(dec.noise_w[i]).size, f"ns.{i}.w")
            alloc_w(np.asarray(dec.noise_b[i]).size, f"ns.{i}.b")
        for r in range(len(dec.resblocks)):
            rb = dec.resblocks[r]
            for j in range(3):
                alloc_w(np.asarray(rb.c1_w[j]).size, f"rb.{r}.{j}.c1w")
                alloc_w(np.asarray(rb.c1_b[j]).size, f"rb.{r}.{j}.c1b")
                alloc_w(np.asarray(rb.c2_w[j]).size, f"rb.{r}.{j}.c2w")
                alloc_w(np.asarray(rb.c2_b[j]).size, f"rb.{r}.{j}.c2b")

        # -- 输入槽（har 通道 = noise_convs 输入通道，权重形状为准）--
        nw0 = np.asarray(dec.noise_w[0])
        har_c = int(nw0.shape[1])
        in_z = self.alloc(int(B * 192 * Tf))
        in_cond = self.alloc(int(B * 512 * Tf))
        in_har = self.alloc(int(B * har_c * har_L))

        # -- conv_pre (k7 s1 pad3 pad3) + cond add（图输入 #2 为 expand 后
        #    cond_full；record 侧仍用未广播 cond_t，镜像 _forward_br）--
        n0 = int(B * 512 * Tf)
        x0 = self.alloc(n0)
        nodes.append((2, in_z, wslots["pre_w"], wslots["pre_b"], 0,
                      (B, 192, Tf, 512, 7, 1, 3, 3, 1, x0, 0)))
        nodes.append((3, x0, in_cond, 0, 0, (n0,) + (0,) * 10))
        out_entry("x0", x0, (B, 512, Tf))     # conv_pre 输出（add 的 a）
        out_entry("x0a", x0, (B, 512, Tf))    # add 覆写后（下游 x）

        # -- ups 循环 --
        cur_in = x0
        ch_in, L_in = 512, Tf
        one_third = []
        for i in range(dec.n_ups):
            uw = np.asarray(dec.ups_w[i])
            k = int(uw.shape[2])
            stride = int(cfg.upsample_rates[i])
            pad = (k - stride) // 2
            ch_out = int(uw.shape[1])
            L_out = (L_in - 1) * stride - 2 * pad + (k - 1) + 1
            n_in = int(B * ch_in * L_in)
            n_out = int(B * ch_out * L_out)
            # copy + leaky(0.1) + convT
            lx = self.alloc(n_in)
            nodes.append((7, lx, cur_in, 0, 0, (n_in,) + (0,) * 10))
            nodes.append((6, lx, 0, 0, 0, (n_in, LRELU01) + (0,) * 9))
            out_entry(f"lx.{i}", lx, (B, ch_in, L_in))      # copy 输出
            out_entry(f"lx_lr.{i}", lx, (B, ch_in, L_in))  # leaky 覆写后
            up = self.alloc(n_out)
            nodes.append((5, lx, wslots[f"up.{i}.w"], wslots[f"up.{i}.b"], 0,
                          (B, ch_in, L_in, ch_out, k, stride, pad, 0, 1,
                           up, 0)))
            out_entry(f"up.{i}", up, (B, ch_out, L_out))    # convT 输出
            # noise conv1d
            nw = np.asarray(dec.noise_w[i])
            k_n = int(nw.shape[2])
            s_n = int(cfg.noise_strides[i])
            pad_n = s_n // 2
            L_ns = (har_L + 2 * pad_n - k_n) // s_n + 1
            if L_ns != L_out:  # pragma: no cover
                raise RuntimeError(
                    f"dec graph ups{i}: noise L {L_ns} != ups L {L_out}")
            ns = self.alloc(n_out)
            nodes.append((2, in_har, wslots[f"ns.{i}.w"],
                          wslots[f"ns.{i}.b"], 0,
                          (B, har_c, har_L, ch_out, k_n, s_n, pad_n, pad_n,
                           1, ns, 0)))
            out_entry(f"ns.{i}", ns, (B, ch_out, L_out))
            nodes.append((3, up, ns, 0, 0, (n_out,) + (0,) * 10))
            out_entry(f"up_a.{i}", up, (B, ch_out, L_out))  # add 覆写后
            # 3×ResBlock（同一 x_base=up 槽）
            rb_outs = []
            for j in range(3):
                rb = dec.resblocks[i * 3 + j]
                cur_r = self.alloc(n_out)
                nodes.append((7, cur_r, up, 0, 0, (n_out,) + (0,) * 10))
                out_entry(f"rb.{i * 3 + j}.cur", cur_r, (B, ch_out, L_out))
                cur_ref = cur_r
                for jj in range(3):
                    d = (1, 3, 5)[jj]
                    k_r = int(np.asarray(rb.c1_w[jj]).shape[2])
                    pad1 = (k_r * d - d) // 2
                    pad2 = (k_r - 1) // 2
                    lx_r = self.alloc(n_out)
                    nodes.append((7, lx_r, cur_ref, 0, 0,
                                  (n_out,) + (0,) * 10))
                    nodes.append((6, lx_r, 0, 0, 0,
                                  (n_out, LRELU01) + (0,) * 9))
                    out_entry(f"rb.{i * 3 + j}.lx.{jj}", lx_r,
                              (B, ch_out, L_out))
                    out_entry(f"rb.{i * 3 + j}.lx_lr.{jj}", lx_r,
                              (B, ch_out, L_out))
                    c1 = self.alloc(n_out)
                    nodes.append((2, lx_r,
                                  wslots[f"rb.{i * 3 + j}.{jj}.c1w"],
                                  wslots[f"rb.{i * 3 + j}.{jj}.c1b"], 0,
                                  (B, ch_out, L_out, ch_out, k_r, 1, pad1,
                                   pad1, d, c1, 0)))
                    out_entry(f"rb.{i * 3 + j}.c1.{jj}", c1,
                              (B, ch_out, L_out))
                    snap = self.alloc(n_out)
                    nodes.append((7, snap, c1, 0, 0, (n_out,) + (0,) * 10))
                    out_entry(f"rb.{i * 3 + j}.snap.{jj}", snap,
                              (B, ch_out, L_out))
                    nodes.append((6, c1, 0, 0, 0,
                                  (n_out, LRELU01) + (0,) * 9))
                    out_entry(f"rb.{i * 3 + j}.c1_lr.{jj}", c1,
                              (B, ch_out, L_out))
                    c2 = self.alloc(n_out)
                    nodes.append((2, c1,
                                  wslots[f"rb.{i * 3 + j}.{jj}.c2w"],
                                  wslots[f"rb.{i * 3 + j}.{jj}.c2b"], 0,
                                  (B, ch_out, L_out, ch_out, k_r, 1, pad2,
                                   pad2, 1, c2, 0)))
                    out_entry(f"rb.{i * 3 + j}.c2.{jj}", c2,
                              (B, ch_out, L_out))
                    nodes.append((3, c2, cur_ref, 0, 0,
                                  (n_out,) + (0,) * 10))
                    out_entry(f"rb.{i * 3 + j}.c2_a.{jj}", c2,
                              (B, ch_out, L_out))
                    cur_ref = c2
                rb_outs.append(c2)   # RB 输出 = 最后 add 覆写的 c2 槽
            # xs = rb0+rb1+rb2（copy+add 两次）
            t1 = self.alloc(n_out)
            nodes.append((7, t1, rb_outs[0], 0, 0, (n_out,) + (0,) * 10))
            out_entry(f"t1.{i}", t1, (B, ch_out, L_out))
            out_entry(f"t1a.{i}", t1, (B, ch_out, L_out))
            nodes.append((3, t1, rb_outs[1], 0, 0, (n_out,) + (0,) * 10))
            t2 = self.alloc(n_out)
            nodes.append((7, t2, t1, 0, 0, (n_out,) + (0,) * 10))
            out_entry(f"t2.{i}", t2, (B, ch_out, L_out))
            out_entry(f"t2a.{i}", t2, (B, ch_out, L_out))
            nodes.append((3, t2, rb_outs[2], 0, 0, (n_out,) + (0,) * 10))
            # ÷3：copy 到新槽再 mul_inplace（t2 原值保留供 record）
            xm = self.alloc(n_out)
            nodes.append((7, xm, t2, 0, 0, (n_out,) + (0,) * 10))
            ot = self.alloc(n_out)
            self.set_input(np.full(n_out, 1.0 / 3.0, np.float32), ot)
            one_third.append(ot)
            nodes.append((4, xm, ot, 0, 0, (n_out,) + (0,) * 10))
            out_entry(f"xm.{i}", xm, (B, ch_out, L_out))
            cur_in = xm
            ch_in, L_in = ch_out, L_out

        # -- 段2：leaky(0.01) + conv_post（无 bias）--
        n4 = int(B * ch_in * L_in)
        nodes.append((6, cur_in, 0, 0, 0, (n4, LRELU001) + (0,) * 9))
        out_entry("xm_lr", cur_in, (B, ch_in, L_in))
        kp = int(np.asarray(dec.conv_post_w).shape[2])
        y_pre = self.alloc(int(B * 1 * L_in))
        nodes.append((2, cur_in, wslots["post_w"], 0, 0,
                      (B, ch_in, L_in, 1, kp, 1, 3, 3, 1, y_pre, 0)))
        out_entry("y_pre", y_pre, (B, 1, L_in))

        gh = self.build(nodes)
        return DecGraph(self, None, gh, in_z, in_cond, in_har, wslots, outs,
                        one_third)

    def release_all(self) -> None:
        """释放全部缓存图与槽位（训练末调用；幂等）。"""
        for pair in self._disc_cache.values():
            for dg in pair:
                dg.destroy()
        self._disc_cache.clear()
        for dg in self._dec_cache.values():
            dg.destroy()
        self._dec_cache.clear()
        for eg in self._enc_cache.values():
            eg.destroy()
        self._enc_cache.clear()
        # T6b：释放判别器 bwd 折叠共享 buffer（rvc_graph_destroy 不释放
        # Python 持有的 buffer，需显式 free）。
        for _f in self.bwd_fold:
            self.free(_f)
        self.bwd_fold = []


# 供 _build_p 使用的常量（避免 import vits_train 循环）
DISCRIMINATOR_P_S = 3
DISCRIMINATOR_P_PAD = 2


# ---------------------------------------------------------------------------
# T4-2：AdamW 优化器图化（RVC_TRAIN_GRAPH_OPT=1 启用；默认 numpy 零回归）
# ---------------------------------------------------------------------------
# 每参数 16/17 节点（m/v 槽保留净值，中间计算走 t1/t2 临时槽）：
#   1  op32 mul_const    p *= 1-lr*wd
#   2  op32 mul_const    m *= b1
#   3  op33 madd_const   m += (1-b1)*g        [m 净值]
#   4  op7  copy         t1 = m
#   5  op36 mul_buf_scalar t1 *= mhat_denom   [t1 = mhat 校正]
#   6  op4  mul          g = g*g（g 槽就地平方）
#   7  op32 mul_const    v *= b2
#   8  op33 madd_const   v += (1-b2)*g²       [v 净值]
#   9  op7  copy         t2 = v
#   10 op36 mul_buf_scalar t2 *= vhat_denom   [t2 = vhat 校正]
#   11 op30 sqrt_inplace t2 = sqrt(t2)
#   12 op35 add_const    t2 += eps
#   13 op37 rcp_inplace  t2 = 1/t2
#   14 op32 mul_const    t1 *= lr
#   15 op4  mul          t1 *= t2             [t1 = lr*mhat/(sqrt(vhat)+eps)]
#   16 op34 sub_inplace  p -= t1
# 槽：p_i/m_i/v_i/g_i/t1_i/t2_i（每参数 6 个）+ denom_m/denom_v（1 元素各 1）。
# 每步：上传 denom → 逐参数 set_input(g) → run（同步）→ 下载 p/m/v。
# p 返回调用方同步回 params_dict；m/v 下载回 numpy 镜像（state() 可读）。
_ADAMW_OP_SQRT = 30
_ADAMW_OP_DIV_CONST = 31
_ADAMW_OP_MUL_CONST = 32
_ADAMW_OP_MADD_CONST = 33
_ADAMW_OP_SUB = 34
_ADAMW_OP_ADD_CONST = 35
_ADAMW_OP_MUL_BUF_SCALAR = 36
_ADAMW_OP_RCP = 37
_OP_MUL = 4
# 引擎描述符池上限 8192 dispatches（engine.zig L594）：每图最多 ~400 参数
# （400×17=6800 节点 <8192 保险）
_ADAMW_PARAMS_PER_GRAPH = 400


def _f32_bits(s: float) -> int:
    """f32 位模式（与引擎 op6 leaky 的 p1 标量传参一致）。"""
    import struct  # noqa: PLC0415
    return struct.unpack("<i", struct.pack("<f", float(s)))[0]


class AdamWGraph:
    """T4-2：AdamW 单图执行器（m/v/p 常驻 GPU，g/denom 每步上传）。

    ``step`` 后 params_dict 内各参数数组被下载的新值覆写（原地同步，
    np.copyto），与 numpy AdamW 的原地更新语义一致。
    """

    __slots__ = ("gr", "names", "shapes", "slots", "ghs", "m", "v",
                 "lr", "b1", "b2", "eps", "wd", "p_ext")

    def __init__(self, gr, names, shapes, slots, ghs, m, v,
                 lr, b1, b2, eps, wd, p_slots=None):
        self.gr = gr
        self.names = list(names)
        self.shapes = shapes            # {name: shape}（dict 保留）
        self.slots = slots              # {name: (p, m, v, g, t1, t2)}
        self.ghs = list(ghs)            # 分组图（引擎描述符池 8192 上限）
        self.m = m                    # numpy m dict（常驻态镜像，save 用）
        self.v = v
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self.wd = float(wd)
        # T4-3：外部 p 槽（wpers 权重槽）——p 更新直接落 GPU 常驻槽，
        # step 免下载（sync_gpu_to_host 在 save 时补下载）。
        self.p_ext = set(p_slots.keys()) if p_slots else set()

    def sync_gpu_to_host(self):
        """T4-3：save 前把 GPU 最新 p/m/v 下载回 numpy 镜像/params。"""
        gr = self.gr
        import numpy as np  # noqa: PLC0415
        for name in self.names:
            p_buf, m_buf, v_buf, _g, _t1, _t2 = self.slots[name]
            if name in self.p_ext:
                np.copyto(self.m[name], gr.ctx.download(m_buf,
                                                        self.m[name].shape))
                np.copyto(self.v[name], gr.ctx.download(v_buf,
                                                        self.v[name].shape))
            else:
                # 内部 p 槽：p 下载回 m/v 镜像所属的 params（调用方持有）
                out = gr.ctx.download(p_buf, self.shapes[name])
                # 无法直接拿 params——由调用方在 step 内 np.copyto；此处
                # 仅下载 m/v（与 T4-2 走法一致由 step 处理 p）
                np.copyto(self.m[name], gr.ctx.download(m_buf,
                                                        self.m[name].shape))
                np.copyto(self.v[name], gr.ctx.download(v_buf,
                                                        self.v[name].shape))

    def step(self, grads_dict, t) -> dict:
        """执行一步；返回 {name: 更新后 p ndarray}（调用方 np.copyto 回
        params_dict；外部 p 槽参数不在返回中——GPU 已最新）。
        m/v 同步下载回 numpy 镜像（state() 可读）。"""
        gr = self.gr
        import numpy as np  # noqa: PLC0415
        # 1) denom 标量槽（1 元素）上传——图内 t1 *= denom（mul_buf_scalar）
        #    numpy 侧 mhat = m/mhat_denom（除）→ 上传倒数 1/mhat_denom
        mdn = float(1.0 / (1.0 - self.b1 ** t))
        vdn = float(1.0 / (1.0 - self.b2 ** t))
        gr.set_input(np.array([mdn], dtype=np.float32), self.slots["_denom_m"][0])
        gr.set_input(np.array([vdn], dtype=np.float32), self.slots["_denom_v"][0])
        # 2) 逐参数 g 上传（g 缺失的参数以同形零填充，保证图节点输入非空）
        for name in self.names:
            g = grads_dict.get(name)
            if g is None:
                g = np.zeros(int(np.prod(self.shapes[name])),
                             dtype=np.float32)
            else:
                g = np.ascontiguousarray(g, dtype=np.float32)
            gr.set_input(g, self.slots[name][3])
        # 3) run 各分组图（同步提交，下载前已 fence）
        for gh in self.ghs:
            gr.run(gh)
        # 4) 下载 p/m/v（p 返回调用方同步回 params_dict）
        out_p = {}
        for name in self.names:
            p_buf, m_buf, v_buf, _g, _t1, _t2 = self.slots[name]
            if name not in self.p_ext:  # 外部槽 GPU 已最新，免下载
                out_p[name] = gr.ctx.download(p_buf, self.shapes[name])
            np.copyto(self.m[name], gr.ctx.download(m_buf,
                                                    self.m[name].shape))
            np.copyto(self.v[name], gr.ctx.download(v_buf,
                                                    self.v[name].shape))
        return out_p

    def sync_mv_to_host(self, m_dict, v_dict) -> None:
        """把 GPU 侧 m/v 下载同步回外部 dict（AdamW.state 用）。"""
        gr = self.gr
        for name in self.names:
            _p, _m, _v, _g, _t1, _t2 = self.slots[name]
            np.copyto(m_dict[name], gr.ctx.download(_m, m_dict[name].shape))
            np.copyto(v_dict[name], gr.ctx.download(_v, v_dict[name].shape))


def build_adamw_graph(gr, params_dict, m_dict, v_dict,
                      lr, b1, b2, eps, wd, p_slots=None):
    """构建 AdamW 单图（T4-2）。params/m/v：{name: ndarray} 同键同形。

    p_slots（可选，T4-3）：{name: gpu_buf_id}——p 更新目标用外部常驻槽
    （如 wpers 权重槽：fwd 复用 GPU p，免下载/重传）。外部槽 p 不 alloc
    不上传；m/v 每参数槽仍由本图分配（下载镜像经 sync_gpu_to_host）。
    返回 AdamWGraph。p/m/v/g/t1/t2 每参数槽 + denom 2 槽。
    """
    import numpy as np  # noqa: PLC0415
    names = list(params_dict.keys())
    shapes = {n: tuple(np.asarray(v).shape) for n, v in params_dict.items()}
    slots = {}
    nodes = []
    ghs = []
    p_slots = p_slots or {}
    # denom 2 槽（1 元素，每步上传 mhat_denom/vhat_denom）
    denom_m = gr.alloc(1)
    denom_v = gr.alloc(1)
    slots["_denom_m"] = (denom_m,)
    slots["_denom_v"] = (denom_v,)
    wd_fac = float(1.0 - lr * wd) if wd else 1.0
    for name in names:
        n = int(np.prod(shapes[name]))
        p_ext = p_slots.get(name)
        p_buf = p_ext if p_ext is not None else gr.alloc(n)
        m_buf = gr.alloc(n)
        v_buf = gr.alloc(n)
        g_buf = gr.alloc(n)
        # 初始上传 m/v/p（外部 p 槽已由 wpers 持有，仅传 m/v；p 图内就地更新）
        if p_ext is None:
            gr.set_input(np.ascontiguousarray(params_dict[name], dtype=np.float32), p_buf)
        gr.set_input(np.ascontiguousarray(m_dict[name], dtype=np.float32), m_buf)
        gr.set_input(np.ascontiguousarray(v_dict[name], dtype=np.float32), v_buf)
        t1 = gr.alloc(n)
        t2 = gr.alloc(n)
        slots[name] = (p_buf, m_buf, v_buf, g_buf, t1, t2)
        # -- 15/16 节点：m/v 槽保留 AdamW 净值；中间计算走 t1/t2 --
        # 1  p *= 1-lr*wd
        if wd:
            nodes.append((_ADAMW_OP_MUL_CONST, p_buf, 0, 0, p_buf,
                          (n, _f32_bits(wd_fac), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 2  m *= b1
        nodes.append((_ADAMW_OP_MUL_CONST, m_buf, 0, 0, m_buf,
                      (n, _f32_bits(b1), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 3  m += (1-b1)*g（m 净值 = b1*m+(1-b1)*g）
        nodes.append((_ADAMW_OP_MADD_CONST, m_buf, g_buf, 0, m_buf,
                      (n, _f32_bits(1.0 - b1), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 4  t1 = m（copy；t1 = mhat 未校正）
        nodes.append((7, t1, m_buf, 0, t1,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 5  t1 *= mhat_denom（t1 = mhat 校正）
        nodes.append((_ADAMW_OP_MUL_BUF_SCALAR, t1, denom_m, 0, t1,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 6  g = g*g（g 槽就地平方）
        nodes.append((_OP_MUL, g_buf, g_buf, 0, g_buf,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 7  v *= b2
        nodes.append((_ADAMW_OP_MUL_CONST, v_buf, 0, 0, v_buf,
                      (n, _f32_bits(b2), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 8  v += (1-b2)*g²（v 净值 = b2*v+(1-b2)*g²）
        nodes.append((_ADAMW_OP_MADD_CONST, v_buf, g_buf, 0, v_buf,
                      (n, _f32_bits(1.0 - b2), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 9  t2 = v（copy；t2 = vhat 未校正）
        nodes.append((7, t2, v_buf, 0, t2,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 10 t2 *= vhat_denom（t2 = vhat 校正）
        nodes.append((_ADAMW_OP_MUL_BUF_SCALAR, t2, denom_v, 0, t2,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 11 t2 = sqrt(t2)（sqrt(vhat)——numpy 侧 np.sqrt(vhat)）
        nodes.append((_ADAMW_OP_SQRT, t2, 0, 0, t2,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 12 t2 += eps（t2 = sqrt(vhat)+eps）
        nodes.append((_ADAMW_OP_ADD_CONST, t2, 0, 0, t2,
                      (n, _f32_bits(eps), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 13 t2 = 1/t2（rcp）
        nodes.append((_ADAMW_OP_RCP, t2, 0, 0, t2,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 14 t1 *= lr
        nodes.append((_ADAMW_OP_MUL_CONST, t1, 0, 0, t1,
                      (n, _f32_bits(lr), 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 15 t1 *= t2（t1 = lr*mhat/(sqrt(vhat)+eps)）
        nodes.append((_OP_MUL, t1, t2, 0, t1,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 16 p -= t1
        nodes.append((_ADAMW_OP_SUB, p_buf, t1, 0, p_buf,
                      (n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        # 引擎描述符池上限 8192 dispatches：每 400 参数（~6800 节点）flush 一图
        if len(nodes) >= _ADAMW_PARAMS_PER_GRAPH * 18:
            ghs.append(gr.build(nodes))
            nodes = []
    if nodes:
        ghs.append(gr.build(nodes))
    return AdamWGraph(gr, names, shapes, slots, ghs, m_dict, v_dict,
                      lr, b1, b2, eps, wd, p_slots=p_slots)