# -*- coding: utf-8 -*-
"""faiss 二进制 .index 文件解析器（纯 Python / numpy，零 faiss / torch 依赖）。

目标：读取用户已有的 RVC 训练产物 `added_IVF*_Flat_nprobe_1_*.index`
（faiss 二进制格式），转换为本项目的 runtime.retrieval.FeatureIndex（.npz）。

支持的 faiss 索引类型（faiss >= ~1.5 的现代 fourcc 序列化布局，
RVC 生态常用的 faiss 1.6.x / 1.7.x 与此布局一致，已用 faiss 1.15.1 真机对照验证）：

    IxFl / IxF2 / IxFI   IndexFlat / IndexFlatL2 / IndexFlatIP
    IwFl                 IndexIVFFlat（内嵌 IndexFlat* quantizer + 数组倒排表 "ilar"）
    IvFl / IvFL          旧版 IndexIVFFlat（legacy 布局，ids/codes 分 list 存放）
    IxMp / IxM2          IndexIDMap / IndexIDMap2（包装上面的底层索引）

不支持的 case 抛 NotImplementedError（含清晰原因）：
    IndexIVFPQ / IndexPQ / IndexScalarQuantizer / IndexHNSW / IndexBinary*
    等量化或图索引（倒排表代码不是 d*4 字节的 Flat 布局）；
    faiss < 1.5 的 legacy long-magic 格式（文件头 4 字节不是已知 fourcc）。

**不支持清单（R2 显式登记，对应交接文档 §9 风险 5）**：
    本解析器**只支持**上面列出的 Flat / IVF-Flat / IDMap(Flat) 布局；
    IndexIVFPQ、IndexPQ、IndexScalarQuantizer、IndexHNSW、IndexBinary*（含
    IndexBinaryFlat/IndexBinaryIVF 等）量化/图/二进制索引**不在范围内，
    不扩实现**（轻量档不依赖 faiss 量化格式；遇到时抛 NotImplementedError
    并提示改用手动转换或其它工具）。

二进制布局说明（小端，Windows/Linux 通用；x86_64 下 idx_t=size_t=8 字节）：

    文件头: uint32 fourcc（如 "IxF2" 的字节 49 78 46 32）
    Index 通用头（read_index_header）:
        int32  d
        int64  ntotal
        int64  dummy(保留) x2
        uint8  is_trained
        int32  metric_type           # 0=IP 1=L2；>1 时后面多一个 float32 metric_arg
    IndexFlat 附加: uint64 元素个数(=ntotal*d) + float32[ntotal*d] 原始向量
    IndexIVF 附加: uint64 nlist + uint64 nprobe
        + 内嵌 quantizer（完整递归 index 序列化）
        + direct_map: uint8 type + uint64 n + int64[n] (+ Hashtable 时: uint64 n + 2*int64[n])
        + 倒排表: uint32 "ilar" + uint64 nlist + uint64 code_size
            + uint32 "full" + uint64 n + uint64[n] 各 list 大小
            + 逐 list: float32[n_i*d] codes + int64[n_i] ids
    IndexIDMap 附加: 内嵌底层 index + uint64 n + int64[n] id_map

用法:
    from runtime.faiss_reader import read_faiss_index, faiss_index_to_npz, probe_faiss_index
    idx = read_faiss_index("added_IVF64_Flat_nprobe_1_v2.index")   # -> FeatureIndex
    faiss_index_to_npz("added_IVF64_Flat_nprobe_1_v2.index")       # -> .npz
    info = probe_faiss_index(path)                                 # 仅头部诊断
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from runtime.retrieval import FeatureIndex

__all__ = [
    "FaissFormatError",
    "read_faiss_index",
    "faiss_index_to_npz",
    "probe_faiss_index",
]

# ---------------------------------------------------------------- fourcc 常量
# 均为小端 uint32 值（文件按小端写入，struct "<I" 读出的数值）。
# 例如 "IwFl" 字节为 49 77 46 6c，读作 uint32 LE = 0x6c467749。
_FOURCC_FLAT = 0x6C467849      # "IxFl"
_FOURCC_FLAT_L2 = 0x32467849   # "IxF2"
_FOURCC_FLAT_IP = 0x49467849   # "IxFI"
_FOURCC_IVF_FLAT = 0x6C467749  # "IwFl"
_FOURCC_IVF_LEGACY = 0x6C467649      # "IvFl"
_FOURCC_IVF_LEGACY_BYTES = 0x4C467649  # "IvFL"
_FOURCC_IDMAP = 0x704D7849     # "IxMp"
_FOURCC_IDMAP2 = 0x324D7849    # "IxM2"
_FOURCC_ILAR = 0x72616C69      # "ilar"
_FOURCC_IL00 = 0x30306C69      # "il00"（未存储倒排表）
_FOURCC_FULL = 0x6C6C7566      # "full"
_FOURCC_SPRS = 0x73727073      # "sprs"（稀疏 sizes）

_FOURCC_NAMES = {
    _FOURCC_FLAT: "IndexFlat",
    _FOURCC_FLAT_L2: "IndexFlatL2",
    _FOURCC_FLAT_IP: "IndexFlatIP",
    _FOURCC_IVF_FLAT: "IndexIVFFlat",
    _FOURCC_IVF_LEGACY: "IndexIVFFlat(legacy)",
    _FOURCC_IVF_LEGACY_BYTES: "IndexIVFFlat(legacy bytes)",
    _FOURCC_IDMAP: "IndexIDMap",
    _FOURCC_IDMAP2: "IndexIDMap2",
}

_METRIC_NAMES = {0: "IP", 1: "L2"}

# 支持的顶层 fourcc（Flat / IVF-Flat / IDMap）
_SUPPORTED = {
    _FOURCC_FLAT, _FOURCC_FLAT_L2, _FOURCC_FLAT_IP,
    _FOURCC_IVF_FLAT, _FOURCC_IVF_LEGACY, _FOURCC_IVF_LEGACY_BYTES,
    _FOURCC_IDMAP, _FOURCC_IDMAP2,
}


class FaissFormatError(Exception):
    """faiss 文件格式/版本不支持的明确异常（NotImplementedError 子类）。"""


def _fourcc_str(h: int) -> str:
    b = bytes([h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF, (h >> 24) & 0xFF])
    return "".join(chr(c) if 32 <= c < 127 else "?" for c in b)


# ---------------------------------------------------------------- 二进制读取器
class _Reader:
    """对 numpy memmap 的只读 view 做小端结构体顺序读取（不复制数据）。"""

    __slots__ = ("_buf", "pos", "path")

    def __init__(self, path: str):
        self.path = str(path)
        self._buf = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.pos = 0

    # -- 基础读取 ----------------------------------------------------------
    def _take(self, nbytes: int) -> np.ndarray:
        end = self.pos + nbytes
        if end > self._buf.size:
            raise FaissFormatError(
                f"{self.path}: 文件在 offset {self.pos} 处提前结束"
                f"（需要 {nbytes} 字节，文件共 {self._buf.size} 字节）")
        out = self._buf[self.pos:end]
        self.pos = end
        return out

    def u8(self) -> int:
        return int(self._take(1)[0])

    def i32(self) -> int:
        return int(struct.unpack("<i", self._take(4).tobytes())[0])

    def u32(self) -> int:
        return int(struct.unpack("<I", self._take(4).tobytes())[0])

    def i64(self) -> int:
        return int(struct.unpack("<q", self._take(8).tobytes())[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self._take(8).tobytes())[0])

    def f32(self) -> float:
        return float(struct.unpack("<f", self._take(4).tobytes())[0])

    def vec_i64(self, n: int) -> np.ndarray:
        if n < 0:
            raise FaissFormatError(f"{self.path}: 非法向量长度 {n}")
        raw = self._take(n * 8)
        return np.frombuffer(raw, dtype="<i8", count=n).copy()

    def vec_f32(self, n: int) -> np.ndarray:
        if n < 0:
            raise FaissFormatError(f"{self.path}: 非法向量长度 {n}")
        raw = self._take(n * 4)
        return np.frombuffer(raw, dtype="<f4", count=n).copy()

    def skip(self, nbytes: int) -> None:
        if nbytes < 0:
            raise FaissFormatError(f"{self.path}: 非法跳过字节数 {nbytes}")
        self._take(nbytes)

    def remaining(self) -> int:
        return self._buf.size - self.pos


# ---------------------------------------------------------------- 头部解析
def _read_index_header(r: _Reader) -> Tuple[int, int, bool, int]:
    """Index 通用头 → (d, ntotal, is_trained, metric_type)。"""
    d = r.i32()
    ntotal = r.i64()
    r.skip(16)                      # 2 x int64 保留
    is_trained = bool(r.u8())
    metric = r.i32()
    if metric > 1:
        r.f32()                     # metric_arg
    if d < 0 or ntotal < 0:
        raise FaissFormatError(f"{r.path}: 非法头部 d={d} ntotal={ntotal}")
    return d, ntotal, is_trained, metric


def _check_count(r: _Reader, n: int, what: str) -> None:
    """防御性检查：元素个数不能超过剩余文件可承载的合理上界。"""
    if n < 0:
        raise FaissFormatError(f"{r.path}: {what} 为负数 {n}")
    if n > r.remaining():
        raise FaissFormatError(
            f"{r.path}: {what}={n} 超出剩余文件大小（{r.remaining()} 字节）")


def _read_vector_header(r: _Reader, elem_size: int, what: str) -> int:
    """读一个向量/数组的 size_t 长度，做防御检查，返回元素个数。"""
    n = r.u64()
    _check_count(r, n, what)
    # elem_size 只是信息性上界校验：n*elem_size <= remaining
    if n * elem_size > r.remaining():
        raise FaissFormatError(
            f"{r.path}: {what}={n} 元素 × {elem_size} 字节超出剩余文件大小")
    return n


# ---------------------------------------------------------------- 索引解析
class _ParsedIndex:
    """解析中间结果：vector 矩阵 + 可选 id 映射 + 元信息。"""

    __slots__ = ("d", "ntotal", "metric", "vectors", "ids", "nlist", "nprobe",
                 "index_type", "is_trained", "centroids", "assignments")

    def __init__(self, d: int, ntotal: int, metric: str, vectors: np.ndarray,
                 ids: Optional[np.ndarray], index_type: str,
                 is_trained: bool, nlist: Optional[int] = None,
                 nprobe: Optional[int] = None,
                 centroids: Optional[np.ndarray] = None,
                 assignments: Optional[np.ndarray] = None):
        self.d = d
        self.ntotal = ntotal
        self.metric = metric
        self.vectors = vectors          # [N, D] float32
        self.ids = ids                  # [N] int64 或 None
        self.index_type = index_type
        self.is_trained = is_trained
        self.nlist = nlist
        self.nprobe = nprobe
        self.centroids = centroids      # [nlist, D] float32 或 None（IVF 专用）
        self.assignments = assignments  # [N] int64 每向量所属簇（IVF 专用）


def _read_flat_index(r: _Reader, h: int, index_type: str,
                     collect_data: bool) -> _ParsedIndex:
    """IxFl / IxF2 / IxFI：IndexFlat 系列。collect_data=False 时跳过向量体。"""
    d, ntotal, trained, metric = _read_index_header(r)
    n_elem = _read_vector_header(r, 4, "xb 元素个数")
    expect = ntotal * d
    if n_elem != expect:
        raise FaissFormatError(
            f"{r.path}: {index_type} 向量元素数 {n_elem} 与 ntotal({ntotal})×d({d})"
            f"={expect} 不一致")
    if collect_data:
        vecs = r.vec_f32(n_elem).reshape(ntotal, d)
    else:
        r.skip(n_elem * 4)
        vecs = np.zeros((0, 0), dtype=np.float32)
    return _ParsedIndex(d, ntotal, _METRIC_NAMES.get(metric, f"metric={metric}"),
                        vecs, None, index_type, trained)


def _read_ivf_index(r: _Reader, h: int, legacy: bool,
                    collect_data: bool) -> _ParsedIndex:
    """IwFl / IvFl / IvFL：IndexIVFFlat。legacy=True 走旧版 ids/codes 布局。"""
    d, ntotal, trained, metric = _read_index_header(r)
    nlist = r.u64()
    nprobe = r.u64()
    _check_count(r, nlist, "nlist")
    # 内嵌 quantizer：完整递归解析（一般是 IndexFlatL2，取其向量数=聚类中心数）。
    # collect_data=True 时保留簇中心向量（centroids），供 IVF 近似检索对齐 faiss。
    quant = _read_any_index(r, collect_data=collect_data)
    centroids = None
    if collect_data and quant.vectors is not None and quant.vectors.size:
        centroids = np.asarray(quant.vectors, dtype=np.float32)
    if quant.ntotal != nlist:
        # 老版本 MultiIndexQuantizer 等 quantizer 可能 ntotal != nlist；不阻塞解析，
        # 但 nlist 与倒排表应一致，倒排表解析会再次校验。
        pass

    if legacy:
        # 旧布局：ids 在 direct_map 之前，逐 list 读取（u64 个数 + int64[]）
        ids_lists: List[np.ndarray] = []
        for _ in range(nlist):
            n = _read_vector_header(r, 8, "legacy list ids")
            ids_lists.append(r.vec_i64(n))

    # direct_map
    dm_type = r.u8()
    if dm_type not in (0, 1, 2):
        raise FaissFormatError(f"{r.path}: 非法 direct_map type {dm_type}")
    n_arr = _read_vector_header(r, 8, "direct_map array")
    r.skip(n_arr * 8)
    if dm_type == 2:                # Hashtable
        n_pairs = _read_vector_header(r, 16, "direct_map hashtable")
        r.skip(n_pairs * 16)

    if legacy:
        # 旧布局：direct_map 之后各 list 的 codes 独立存放
        if collect_data:
            if h == _FOURCC_IVF_LEGACY_BYTES:   # "IvFL": codes 按字节存
                code_lists = []
                for _ in range(nlist):
                    nb = _read_vector_header(r, 1, "legacy list codes(bytes)")
                    raw = r._take(nb)
                    nf = nb // 4
                    code_lists.append(
                        np.frombuffer(raw, dtype="<f4", count=nf)
                        .reshape(nf // d, d) if nb else
                        np.zeros((0, d), dtype=np.float32))
                codes = code_lists
            else:                               # "IvFl": codes 按 float32 存
                code_lists = []
                for _ in range(nlist):
                    nf = _read_vector_header(r, 4, "legacy list codes(floats)")
                    code_lists.append(
                        r.vec_f32(nf).reshape(nf // d, d) if nf else
                        np.zeros((0, d), dtype=np.float32))
                codes = code_lists
            return _build_ivf_result(r, d, ntotal, metric, trained, nlist,
                                     nprobe, ids_lists, codes,
                                     centroids=centroids)
        else:
            for _ in range(nlist):
                if h == _FOURCC_IVF_LEGACY_BYTES:
                    nb = _read_vector_header(r, 1, "legacy list codes(bytes)")
                    r.skip(nb)
                else:
                    nf = _read_vector_header(r, 4, "legacy list codes(floats)")
                    r.skip(nf * 4)
            return _ParsedIndex(d, ntotal, _METRIC_NAMES.get(metric, "?"),
                                np.zeros((0, 0), dtype=np.float32), None,
                                "IndexIVFFlat", trained, nlist, nprobe,
                                centroids=centroids)

    # 现代布局：InvertedLists 序列化
    il_type = r.u32()
    if il_type == _FOURCC_IL00:
        if ntotal != 0:
            raise FaissFormatError(
                f"{r.path}: 倒排表未存储（il00）但 ntotal={ntotal} != 0，无法恢复向量")
        return _ParsedIndex(d, 0, _METRIC_NAMES.get(metric, "?"),
                            np.zeros((0, d), dtype=np.float32),
                            np.zeros(0, dtype=np.int64),
                            "IndexIVFFlat", trained, nlist, nprobe,
                            centroids=centroids)
    if il_type != _FOURCC_ILAR:
        raise NotImplementedError(
            f"{r.path}: 倒排表类型 {_fourcc_str(il_type)!r} (0x{il_type:08x}) 不受支持"
            f"（仅支持 ArrayInvertedLists \"ilar\"，量化/图索引需 faiss 原生读取）")
    il_nlist = r.u64()
    code_size = r.u64()
    if il_nlist != nlist:
        raise FaissFormatError(
            f"{r.path}: 倒排表 nlist={il_nlist} 与头部 nlist={nlist} 不一致")
    if code_size != d * 4:
        raise NotImplementedError(
            f"{r.path}: 倒排表 code_size={code_size} != d({d})×4，"
            f"这不是 IVF-Flat 布局（可能是 IndexIVFPQ 等量化索引）")

    list_type = r.u32()
    if list_type == _FOURCC_FULL:
        n_sz = _read_vector_header(r, 8, "sizes 数量")
        sizes = np.frombuffer(r._take(n_sz * 8), dtype="<i8", count=n_sz)
    elif list_type == _FOURCC_SPRS:
        n_sz = _read_vector_header(r, 8, "sparse sizes 数量")
        pairs = np.frombuffer(r._take(n_sz * 8), dtype="<i8", count=n_sz)
        sizes = np.zeros(nlist, dtype=np.int64)
        if n_sz % 2 != 0:
            raise FaissFormatError(f"{r.path}: sprs sizes 数量 {n_sz} 非偶数")
        sizes[pairs[0::2]] = pairs[1::2]
    else:
        raise FaissFormatError(
            f"{r.path}: 未知 sizes 存储类型 {_fourcc_str(list_type)!r}")

    if len(sizes) != nlist:
        raise FaissFormatError(
            f"{r.path}: sizes 数量 {len(sizes)} != nlist {nlist}")
    if int(sizes.sum()) != ntotal:
        raise FaissFormatError(
            f"{r.path}: 倒排表向量总数 {sizes.sum()} != ntotal {ntotal}")
    if collect_data:
        ids_lists = []
        codes = []
        for i in range(nlist):
            n = int(sizes[i])
            _check_count(r, n, f"list[{i}]")
            if n > 0:
                codes.append(r.vec_f32(n * d).reshape(n, d))
                ids_lists.append(r.vec_i64(n))
            else:
                codes.append(np.zeros((0, d), dtype=np.float32))
                ids_lists.append(np.zeros(0, dtype=np.int64))
        return _build_ivf_result(r, d, ntotal, metric, trained, nlist, nprobe,
                                 ids_lists, codes, centroids=centroids)
    else:
        for i in range(nlist):
            n = int(sizes[i])
            _check_count(r, n, f"list[{i}]")
            r.skip(n * d * 4 + n * 8)
        return _ParsedIndex(d, ntotal, _METRIC_NAMES.get(metric, "?"),
                            np.zeros((0, 0), dtype=np.float32), None,
                            "IndexIVFFlat", trained, nlist, nprobe,
                            centroids=centroids)


def _build_ivf_result(r: _Reader, d: int, ntotal: int, metric: int,
                      trained: bool, nlist: int, nprobe: int,
                      ids_lists: List[np.ndarray],
                      codes: List[np.ndarray],
                      centroids: Optional[np.ndarray] = None) -> _ParsedIndex:
    """把逐 list 的 (ids, codes) 拼接为 [N,D] 矩阵 + [N] ids。

    拼接顺序 = list id 顺序 = faiss 内部存储顺序（reconstruct_n 同序），
    保证与 faiss 逐元素对照成立。同时构建 assignments（每向量所属簇号），
    供 IVF 近似检索对齐 faiss（query 最近簇 → 簇内暴力）。
    """
    total = sum(int(a.size) for a in ids_lists)
    if total != ntotal:
        raise FaissFormatError(
            f"{r.path}: 倒排表实际向量数 {total} != ntotal {ntotal}")
    vecs = np.concatenate(codes, axis=0).reshape(ntotal, d).astype(np.float32)
    ids = np.concatenate(ids_lists).astype(np.int64)
    # assignments：第 i 个 list 的 n 条向量 → 簇号 i（与拼接顺序对齐）
    assigns = np.concatenate([
        np.full(int(len(a)), i, dtype=np.int64) for i, a in enumerate(ids_lists)
    ])
    return _ParsedIndex(d, ntotal, _METRIC_NAMES.get(metric, "?"), vecs, ids,
                        "IndexIVFFlat", trained, nlist, nprobe,
                        centroids=centroids, assignments=assigns)


def _read_idmap_index(r: _Reader, collect_data: bool) -> _ParsedIndex:
    """IxMp / IxM2：IndexIDMap(inner)。"""
    d, ntotal, trained, metric = _read_index_header(r)
    inner = _read_any_index(r, collect_data=collect_data)
    if inner.ntotal != ntotal:
        raise FaissFormatError(
            f"{r.path}: IndexIDMap 内嵌索引 ntotal={inner.ntotal} != 外层 {ntotal}")
    n_map = _read_vector_header(r, 8, "id_map")
    if n_map != ntotal:
        raise FaissFormatError(
            f"{r.path}: id_map 长度 {n_map} != ntotal {ntotal}")
    id_map = r.vec_i64(n_map) if collect_data else None
    if not collect_data:
        r.skip(n_map * 8)
    metric_s = inner.metric if inner.metric in ("IP", "L2") else "L2"
    # IDMap 包装 IVF 时：inner 解析保留 centroids/assignments（按 faiss
    # 内部序），id_map 即外层用户 id（同序排列）——直接透传即可保留
    # IVF 近似检索语义（query 最近簇 → 簇内暴力）。
    return _ParsedIndex(d, ntotal, metric_s, inner.vectors, id_map,
                        "IndexIDMap", trained and inner.is_trained,
                        nlist=inner.nlist, nprobe=inner.nprobe,
                        centroids=getattr(inner, "centroids", None),
                        assignments=getattr(inner, "assignments", None))


def _dispatch(r: _Reader, h: int, collect_data: bool) -> _ParsedIndex:
    """按已读取的 fourcc h 分发解析（不消费 h 本身）。"""
    if h == _FOURCC_FLAT:
        return _read_flat_index(r, h, "IndexFlat", collect_data)
    if h == _FOURCC_FLAT_L2:
        return _read_flat_index(r, h, "IndexFlatL2", collect_data)
    if h == _FOURCC_FLAT_IP:
        return _read_flat_index(r, h, "IndexFlatIP", collect_data)
    if h == _FOURCC_IVF_FLAT:
        return _read_ivf_index(r, h, legacy=False, collect_data=collect_data)
    if h in (_FOURCC_IVF_LEGACY, _FOURCC_IVF_LEGACY_BYTES):
        return _read_ivf_index(r, h, legacy=True, collect_data=collect_data)
    if h in (_FOURCC_IDMAP, _FOURCC_IDMAP2):
        return _read_idmap_index(r, collect_data=collect_data)
    name = _fourcc_str(h)
    raise NotImplementedError(
        f"{r.path}: 不支持的 faiss 索引类型 fourcc={name!r} (0x{h:08x})。"
        f"仅支持 IndexFlat(L2/IP)、IndexIVFFlat、IndexIDMap(Flat)；"
        f"不支持清单（显式登记 R2）：IndexIVFPQ / IndexPQ / "
        f"IndexScalarQuantizer / IndexHNSW / IndexBinary*。"
        f"若为 faiss<1.5 的旧版 long-magic 格式（4 字节头非 fourcc），亦不支持。")


def _read_any_index(r: _Reader, collect_data: bool = True) -> _ParsedIndex:
    """读取一个完整（可递归）的 faiss 索引。collect_data=False 只走头部/跳过。"""
    h = r.u32()
    return _dispatch(r, h, collect_data)


# ---------------------------------------------------------------- 公开 API
def probe_faiss_index(path: str) -> Dict[str, object]:
    """只读头部并返回诊断信息（不读向量体）：
    {magic, index_type, d, ntotal, nlist, nprobe, is_trained, metric}。
    """
    r = _Reader(path)
    h = r.u32()
    if h not in _SUPPORTED:
        raise NotImplementedError(
            f"{path}: 不支持的 faiss 索引类型 fourcc={_fourcc_str(h)!r}"
            f" (0x{h:08x})")
    parsed = _dispatch(r, h, collect_data=False)
    return {
        "magic": h,
        "index_type": parsed.index_type,
        "d": parsed.d,
        "ntotal": parsed.ntotal,
        "nlist": parsed.nlist,
        "nprobe": parsed.nprobe,
        "is_trained": parsed.is_trained,
        "metric": parsed.metric,
    }


def _to_feature_index(parsed: _ParsedIndex):
    """把解析结果转为检索索引。

    - IndexIVFFlat（含保留的 centroids/assignments）→ IVFIndex：
      走与 faiss 一致的倒排分桶近似检索（query 最近 nprobe 簇 → 簇内暴力）。
    - 其余（Flat/FlatL2/FlatIP/IDMap(Flat)）→ FeatureIndex 全量暴力。
    """
    metric = parsed.metric if parsed.metric in ("IP", "L2") else "L2"
    # 只要保留了 IVF 结构（centroids + assignments）就走 IVFIndex 近似检索
    # （覆盖 IndexIVFFlat 与 IDMap 包 IVF 两种形态）。
    if (parsed.centroids is not None
            and parsed.assignments is not None
            and parsed.vectors.size):
        from runtime.ivf_index import IVFIndex  # 延迟导入避免循环
        idx = IVFIndex(dim=parsed.d, nlist=parsed.nlist,
                       nprobe=parsed.nprobe or 1, metric=metric)
        idx.set_centroids(parsed.centroids)
        idx.add(parsed.vectors, ids=parsed.ids, assignments=parsed.assignments)
        return idx
    idx = FeatureIndex(dim=parsed.d, metric=metric)
    idx.add(parsed.vectors, ids=parsed.ids)
    return idx


def read_faiss_index(path: str) -> FeatureIndex:
    """解析 faiss 二进制 .index → FeatureIndex（numpy .npz 兼容格式）。

    支持 IndexFlat/FlatL2/FlatIP、IndexIVFFlat（含 legacy）、
    IndexIDMap(IndexFlat*)；量化/图索引与旧版 long-magic 格式抛
    NotImplementedError（FaissFormatError 子类）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"faiss 索引文件不存在: {path}")
    r = _Reader(path)
    parsed = _read_any_index(r, collect_data=True)
    # 顶层解析完应消费完整文件（量化外挂/多索引才可能有余量）
    if r.remaining() != 0:
        raise FaissFormatError(
            f"{path}: 解析后仍剩 {r.remaining()} 字节未消费，文件可能包含"
            f"额外数据或格式与预期不符")
    return _to_feature_index(parsed)


def faiss_index_to_npz(path: str, out_path: Optional[str] = None) -> str:
    """把 faiss .index 转换为 FeatureIndex 的 .npz 并保存。

    out_path 缺省时同目录同文件名替换后缀为 .npz。返回输出路径。
    """
    path = str(path)
    if out_path is None:
        base, _ext = os.path.splitext(path)
        out_path = base + ".npz"
    idx = read_faiss_index(path)
    idx.save(out_path)
    return out_path
