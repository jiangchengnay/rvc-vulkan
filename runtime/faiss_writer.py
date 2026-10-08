# -*- coding: utf-8 -*-
"""faiss 二进制 .index 写出器（纯 Python / numpy，零 faiss / torch 依赖）。

与 ``runtime.faiss_reader.py`` **字节级对称**的写路径（VK-05）：本模块产出的
``.index`` / ``.faiss`` 文件可直接被 ``faiss_reader.read_faiss_index`` /
``retrieval.load_index`` 读回并用本仓库检索层检索，形成 写→读→检索 闭环，
补齐 02 基线"只有读路径、无写路径"缺口。

支持的索引类型（与 reader 的 ``_SUPPORTED`` 对称子集；写法均为小端 float32 / int64）：

    - IndexFlatL2 （fourcc "IxFl"）  FeatureIndex 或裸 (vectors[, ids])
    - IndexIVFFlat（fourcc "IwFl"）  IVFIndex 或 dict{centroids, assignments,
                                    vectors, ids}（现代 "ilar" 倒排表布局）
    - IndexIDMap  （fourcc "IxMp"）  上述两者的 id_map 包装。faiss 的 IndexFlat
                                    本身不存 ids（隐式 0..ntotal-1），因此
                                    **Flat 输入带自定义 ids 时自动包裹 IxMp**
                                    以保留 id 映射；IVF 的 ids 天然逐 list 存储。

**度量铁律（P-RET-001 / P-RET-002）**：
当前检索层只实现 L2（retrieval.py:111 注释"先支持 L2"）。因此写出器**只接受
L2（faiss metric_type=1）**：输入 IP（metric_type=0）或其它度量（>1，如 L1≈2）
时**显式抛 NotImplementedError**（绝不静默写成 L2，避免下游拿到错误语义的
索引造成静默检索错误）。

二进制布局（小端；x86_64 下 idx_t=size_t=8 字节；与 reader 的读取布局严格互逆）：

    文件头: uint32 fourcc
    Index 通用头: int32 d | int64 ntotal | int64 dummy x2 | uint8 is_trained
                 | int32 metric_type（>1 时另附 float32 metric_arg；写路径从不产生）
    IndexFlat 附加: uint64 元素个数(=ntotal*d) + float32[ntotal*d]（C 行序）
    IndexIVF 附加: uint64 nlist + uint64 nprobe
        + 内嵌 quantizer（完整 IndexFlatL2 序列化，fourcc "IxF2"，ntotal=nlist
          的中心矩阵——与 faiss 原生输出字节级一致）
        + direct_map: uint8 type(0=NoMap) + uint64 n(0)
        + 倒排表: uint32 "ilar" + uint64 nlist + uint64 code_size(=d*4)
            + uint32 "full" + uint64 nlist + int64[nlist] 各 list 大小
            + 逐 list: float32[n_i*d] codes + int64[n_i] ids
    IndexIDMap 附加: 内嵌底层 index + uint64 n + int64[n] id_map

一致性与内存约束：
    - 写出前校验 vectors/ids 维度、长度一致、ntotal 一致、nlist 一致、
      assignments 范围 < nlist（P-RET-006 / P-RET-005 精神：不一致数据在
      **写前**暴露，而不是写成坏文件）；
    - 数据集必须小（≤ 几千条，与 VK-05 规格一致），全文件在内存组装后
      原子写（tmp + os.replace，避免半写文件）。

用法::

    from runtime.faiss_writer import write_faiss_index
    from runtime.retrieval import FeatureIndex, export_faiss

    idx = FeatureIndex(dim=128); idx.add(vectors)   # 或 IVFIndex / 裸数组
    write_faiss_index(idx, "out.index")             # -> IndexFlatL2
    write_faiss_index({"vectors": v, "centroids": c,
                       "assignments": a}, "ivf.index")  # -> IndexIVFFlat
    write_faiss_index(idx, "out.faiss", id_map=custom_ids)  # -> IndexIDMap

    export_faiss("added_..._v2.ivf.npz")            # 便捷入口（见 retrieval）
"""

from __future__ import annotations

import io
import os
import struct
from typing import Any, Dict, Optional

import numpy as np

# 与 reader 共用的 fourcc 常量：直接引用保证字节级对称，永不漂移。
from runtime.faiss_reader import (  # noqa: WPS436 —— 同包内部常量复用
    _FOURCC_FLAT,        # "IxFl"  IndexFlat（顶层 IndexFlatL2 用）
    _FOURCC_FLAT_L2,     # "IxF2"  IndexFlatL2（IVF 内嵌 quantizer 用，faiss 原生写法）
    _FOURCC_IVF_FLAT,    # "IwFl"  IndexIVFFlat
    _FOURCC_IDMAP,       # "IxMp"  IndexIDMap
    _FOURCC_ILAR,        # "ilar"  ArrayInvertedLists
    _FOURCC_FULL,        # "full"  密集 sizes 表
)

__all__ = ["write_faiss_index"]

# 通用头保留字段（faiss 写 1<<20；reader 对 dummy 取值不校验，仅 skip）
_DUMMY_HEADER = 1 << 20
_METRIC_L2 = 1
_INT32_MAX = 0x7FFFFFFF


# ---------------------------------------------------------------- 度量铁律
def _metric_code(metric) -> int:
    """度量白名单：仅 L2（faiss metric_type=1）。其余一律 NotImplementedError。

    P-RET-001（IP=0 按 L2 检索）与 P-RET-002（metric>1 静默降级 L2）的
    写侧闭环：IP / 其它度量直接拒绝，绝不写错。
    """
    if isinstance(metric, (int, np.integer)):
        if int(metric) == _METRIC_L2:
            return _METRIC_L2
    elif isinstance(metric, str):
        if metric.strip().upper() in ("L2", "1"):
            return _METRIC_L2
    raise NotImplementedError(
        "faiss_writer 仅支持写出 L2 度量索引（faiss metric_type=1，IP=0 / "
        "L1≈2 等其它度量当前检索层未实现，"
        "P-RET-001/002）。收到 metric=%r；为避免静默写入错误度量语义，"
        "显式拒绝写出。" % (metric,))


# ---------------------------------------------------------------- 字节写出
class _BytesWriter:
    """顺序写出小端标量 / numpy 数组到 io.BytesIO（与 reader 的 _Reader 互逆）。"""

    __slots__ = ("_b",)

    def __init__(self, buf):
        self._b = buf

    def raw(self, data: bytes) -> None:
        self._b.write(bytes(data))

    def u8(self, v: int) -> None:
        self._b.write(struct.pack("<B", int(v)))

    def i32(self, v: int) -> None:
        self._b.write(struct.pack("<i", int(v)))

    def u32(self, v: int) -> None:
        self._b.write(struct.pack("<I", int(v)))

    def i64(self, v: int) -> None:
        self._b.write(struct.pack("<q", int(v)))

    def u64(self, v: int) -> None:
        self._b.write(struct.pack("<Q", int(v)))

    def vec_f32(self, arr: np.ndarray) -> None:
        self._b.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    def vec_i64(self, arr: np.ndarray) -> None:
        self._b.write(np.ascontiguousarray(arr, dtype="<i8").tobytes())


def _write_header(w: _BytesWriter, d: int, ntotal: int,
                  metric_code: int, trained: int = 1) -> None:
    """Index 通用头（与 reader._read_index_header 严格互逆）。"""
    w.i32(d)
    w.i64(ntotal)
    w.i64(_DUMMY_HEADER)
    w.i64(_DUMMY_HEADER)
    w.u8(int(trained))
    w.i32(int(metric_code))


def _serialize_flat(d: int, vectors: np.ndarray,
                    fourcc: int = _FOURCC_FLAT) -> bytes:
    """IndexFlat（默认 "IxFl"，L2）：fourcc + 通用头 + u64 n_elem + float32[n_elem]。

    fourcc 可变：顶层 IndexFlatL2 用 "IxFl"，IVF 内嵌 quantizer 用 "IxF2"
    （faiss 原生对 IndexFlatL2 quantizer 的写法，二者 reader/faiss 均可读）。
    """
    n = vectors.shape[0]
    b = io.BytesIO()
    w = _BytesWriter(b)
    w.u32(fourcc)
    _write_header(w, d, n, _METRIC_L2)
    w.u64(n * d)
    w.vec_f32(vectors)                  # C 行序 [n, d] 连续写
    return b.getvalue()


def _default_assign(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """缺省 assignments：每行归入最近 L2 中心（与 IVFIndex.add 的 _assign 同语义）。

    仅在 dict 输入给了 centroids 却未给 assignments 时使用。
    """
    v = np.asarray(vectors, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    labels = np.empty(v.shape[0], dtype=np.int64)
    v2 = np.einsum("nd,nd->n", v, v)
    c2 = np.einsum("nd,nd->n", c, c)
    chunk = 8192
    for s in range(0, v.shape[0], chunk):
        e = min(s + chunk, v.shape[0])
        dist = v2[s:e, None] + c2[None, :] - 2.0 * np.dot(v[s:e], c.T)
        np.maximum(dist, 0.0, out=dist)
        labels[s:e] = np.argmin(dist, axis=1)
    return labels


def _serialize_ivf(d: int, vectors: np.ndarray, ids: np.ndarray,
                   centroids: np.ndarray, assignments: np.ndarray,
                   nlist: int, nprobe: int) -> bytes:
    """IndexIVFFlat（"IwFl"，L2）：现代 "ilar"/"full" 倒排表布局。"""
    n = vectors.shape[0]
    b = io.BytesIO()
    w = _BytesWriter(b)
    w.u32(_FOURCC_IVF_FLAT)
    _write_header(w, d, n, _METRIC_L2)
    w.u64(nlist)
    w.u64(nprobe)
    # 内嵌 quantizer：完整 IndexFlatL2 序列化（ntotal=nlist 的中心矩阵，
    # fourcc "IxF2" 与 faiss 原生输出一致）
    w.raw(_serialize_flat(d, centroids, fourcc=_FOURCC_FLAT_L2))
    # direct_map：NoMap
    w.u8(0)
    w.u64(0)
    # 倒排表：ilar + full sizes + 逐 list codes/ids
    w.u32(_FOURCC_ILAR)
    w.u64(nlist)
    w.u64(d * 4)                        # code_size = d * float32 字节
    w.u32(_FOURCC_FULL)
    w.u64(nlist)
    sizes = np.bincount(assignments, minlength=nlist).astype(np.int64)
    w.vec_i64(sizes)
    for i in range(nlist):
        sel = assignments == i          # 保留原数组内出现顺序
        w.vec_f32(vectors[sel])
        w.vec_i64(ids[sel])
    return b.getvalue()


def _serialize_idmap(d: int, inner_bytes: bytes, id_map: np.ndarray) -> bytes:
    """IndexIDMap（"IxMp"）：外层通用头 + 内嵌 index + u64 n + int64[n] id_map。

    id_map 长度即外层 ntotal，必须等于内嵌索引 ntotal（由调用方保证）。
    """
    n = len(id_map)
    b = io.BytesIO()
    w = _BytesWriter(b)
    w.u32(_FOURCC_IDMAP)
    _write_header(w, d, n, _METRIC_L2)
    b.write(inner_bytes)
    w.u64(n)
    w.vec_i64(id_map)
    return b.getvalue()


# ---------------------------------------------------------------- 输入解析
def _resolve_index(index) -> Dict[str, Any]:
    """把输入归一化为数据字典，并判定 kind ∈ {'flat','ivf'}。"""
    from runtime.retrieval import FeatureIndex  # noqa: PLC0415 —— 延迟避免循环
    from runtime.ivf_index import IVFIndex      # noqa: PLC0415

    d: Dict[str, Any] = {}
    if isinstance(index, IVFIndex):
        d["kind"] = "ivf"
        d["vectors"] = index._vectors
        d["ids"] = index._ids
        d["centroids"] = index.centroids
        d["assignments"] = index._assignments
        d["nlist"] = index.nlist
        d["nprobe"] = index.nprobe
        d["metric"] = index.metric
        return d
    if isinstance(index, FeatureIndex):
        d["kind"] = "flat"
        d["vectors"] = index._vectors
        d["ids"] = index._ids if index._ids.size else None
        d["metric"] = index.metric
        return d
    if isinstance(index, np.ndarray):
        d["kind"] = "flat"
        d["vectors"] = index
        return d
    if isinstance(index, dict):
        d = dict(index)
        d["kind"] = ("ivf" if ("centroids" in index or "assignments" in index
                               or index.get("nlist") is not None) else "flat")
        return d
    raise TypeError(
        "write_faiss_index 的 index 参数支持 FeatureIndex / IVFIndex / "
        "np.ndarray / dict（键：vectors 必需；ids/centroids/assignments/"
        "nlist/nprobe/dim/metric 可选），实际类型 %r" % type(index).__name__)


# ---------------------------------------------------------------- 公开 API
def write_faiss_index(index, path: str, *, metric: object = "L2",
                      ids: Optional[np.ndarray] = None,
                      nprobe: Optional[int] = None,
                      id_map: Optional[np.ndarray] = None,
                      dim: Optional[int] = None) -> str:
    """写出 faiss 兼容二进制 .index/.faiss（小端 float32，与 reader 对称）。

    Args:
        index: FeatureIndex / IVFIndex / np.ndarray / dict。
            - FeatureIndex      → IndexFlatL2（ids 非隐式时自动 IndexIDMap 包裹）
            - IVFIndex          → IndexIVFFlat（centroids/assignments/ids 全保留）
            - np.ndarray [N,D]  → IndexFlatL2
            - dict：{'vectors': [N,D] float32, 'ids'?: [N] int64,
              'centroids'?: [nlist,D] float32, 'assignments'?: [N] int64,
              'nlist'?, 'nprobe'?, 'metric'?}；给 centroids 未给 assignments
              时按最近 L2 中心自动分桶。
        path: 输出路径（约定 .index / .faiss 后缀，不强制）。
        metric: 写出度量，**仅支持 L2**（接受 "L2"/1）；IP 或其它度量抛
            NotImplementedError（P-RET-001/002 铁律，不静默写错）。
        ids: 覆盖输入自带 ids（Flat 走 id_map 自动包裹；IVF 逐 list 写入）。
        nprobe: 覆盖 IVF nprobe（缺省取输入值或 1）。
        id_map: 显式 IndexIDMap 包装的 id 映射（长度须等于 ntotal）。
        dim: 显式声明向量维度，与 vectors.shape[1] 不一致时抛 ValueError。

    Returns:
        输出路径（写成功后才返回）。

    Raises:
        NotImplementedError: 非 L2 度量（IP/其它）——写侧度量铁律。
        ValueError: 输入数据不一致（维度/长度/ntotal/nlist/assignments 范围）。
        TypeError: index 参数类型不受支持。
    """
    data = _resolve_index(index)

    # 度量铁律：先于一切数据校验（P-RET-001/002）
    _metric_code(metric)                        # 参数侧
    if data.get("metric") is not None:
        _metric_code(data["metric"])            # 输入对象/字典侧

    # ---- 向量归一化与一致性校验（P-RET-006 精神）----
    if data.get("vectors") is None:
        raise ValueError("缺少 vectors 数据（裸 ndarray/dict 输入必须提供 vectors）")
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors 必须是 [N, D] 二维数组，实际 %dD" % vectors.ndim)
    n, d = vectors.shape
    if d < 1 or d > _INT32_MAX:
        raise ValueError("非法维度 d=%d（faiss 头为 int32，须 1<=d<2^31）" % d)
    if dim is not None and int(dim) != d:
        raise ValueError("dim 参数 %d 与 vectors 维度 %d 不一致" % (dim, d))
    if n < 0:
        raise ValueError("vectors 行数非法：%d" % n)

    ids_arr = data.get("ids") if ids is None else ids
    if ids_arr is None:
        ids_arr = np.arange(n, dtype=np.int64)
    else:
        ids_arr = np.asarray(ids_arr, dtype=np.int64)
        if ids_arr.ndim != 1 or len(ids_arr) != n:
            raise ValueError(
                "ids 长度 %d != vectors 行数 %d（P-RET-006 一致性校验）"
                % (len(ids_arr) if ids_arr.ndim else 0, n))

    if data["kind"] == "ivf":
        body = _serialize_ivf_from(data, vectors, ids_arr, d, nprobe)
    else:
        body = _serialize_flat(d, vectors)
        # Flat 的 ids 隐式 0..ntotal-1；带自定义 ids 时自动 IndexIDMap 包裹
        if id_map is not None:
            id_map_arr = np.asarray(id_map, dtype=np.int64)
            if id_map_arr.ndim != 1 or len(id_map_arr) != n:
                raise ValueError(
                    "id_map 长度 %d != ntotal %d" % (len(id_map_arr), n))
            body = _serialize_idmap(d, body, id_map_arr)
        elif not np.array_equal(ids_arr, np.arange(n)):
            body = _serialize_idmap(d, body, ids_arr)

    _atomic_write(path, body)
    return str(path)


def _serialize_ivf_from(data: Dict[str, Any], vectors: np.ndarray,
                        ids: np.ndarray, d: int,
                        nprobe: Optional[int]) -> bytes:
    """IVF 分支：校验字段后序列化（P-RET-006/005 一致性检查集中于此）。"""
    if data.get("centroids") is None:
        raise ValueError("IVF 写出需要 centroids（[nlist, D] 聚类中心）")
    centroids = np.asarray(data["centroids"], dtype=np.float32)
    if centroids.ndim != 2 or centroids.shape[0] < 1:
        raise ValueError("centroids 必须是 [nlist, D] 非空二维数组")
    nlist = centroids.shape[0]
    if centroids.shape[1] != d:
        raise ValueError("centroids 维度 %d != vectors 维度 %d（nlist 不一致）"
                         % (centroids.shape[1], d))
    if data.get("nlist") is not None and int(data["nlist"]) != nlist:
        raise ValueError("nlist %d 与 centroids 行数 %d 不一致"
                         % (int(data["nlist"]), nlist))

    n = vectors.shape[0]
    assignments = data.get("assignments")
    if assignments is None:
        assignments = _default_assign(vectors, centroids)
    else:
        assignments = np.asarray(assignments, dtype=np.int64)
        if assignments.ndim != 1 or len(assignments) != n:
            raise ValueError("assignments 长度 %d != vectors 行数 %d"
                             % (len(assignments), n))
    if n and (assignments.min() < 0 or assignments.max() >= nlist):
        raise ValueError(
            "assignments 越界：值域 [%d, %d]，须落在 [0, nlist=%d)"
            % (int(assignments.min()), int(assignments.max()), nlist))

    if nprobe is None:
        nprobe = data.get("nprobe", 1)
    nprobe = int(nprobe)
    if nprobe < 1:
        raise ValueError("nprobe 必须 >= 1，实际 %d" % nprobe)

    # ntotal 一致性：vectors 行数 = ids 长度 = assignments 长度，且桶合计一致
    sizes = np.bincount(assignments, minlength=nlist)
    if int(sizes.sum()) != n:
        raise ValueError("内部错误：BInc桶合计 %d != 向量数 %d" % (int(sizes.sum()), n))
    if len(ids) != n:
        raise ValueError("ids 长度 %d != vectors 行数 %d（P-RET-006 一致性校验）"
                         % (len(ids), n))

    return _serialize_ivf(d, vectors, ids, centroids, assignments, nlist, nprobe)


def _atomic_write(path: str, body: bytes) -> None:
    """临时文件 + os.replace 原子落盘（避免半写文件，对齐 P-TRAIN-005 精神）。"""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise