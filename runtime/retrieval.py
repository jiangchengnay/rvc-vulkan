# -*- coding: utf-8 -*-
"""特征检索层：纯 numpy 实现 faiss(IndexIVFFlat/IndexFlatIP) 的替代。

RVC 原实现（infer/vc/pipeline.py）：
    index = faiss.read_index(file_index)
    index_vectors = index.reconstruct_n(0, index.ntotal)
    score, ix = index.search(npy, k=8)
    w = (1/score)**2; w /= w.sum(axis=1, keepdims=True)
    npy = np.sum(index_vectors[ix] * w[..., None], axis=1)

本模块提供等价行为：
- 自建索引存为 .npz（vectors + ids），零第三方依赖；
- search 用分块矩阵乘法计算 L2 距离，避免一次性 O(N*D) 内存峰值；
- k=8、(1/d)^2 权重与 RVC 完全一致。

用法：
    idx = FeatureIndex(dim=768)
    idx.add(vectors)              # vectors: [N, D]
    idx.save("added_my_v2.npz")
    idx2 = FeatureIndex.load("added_my_v2.npz")
    scores, inds = idx2.search(query, k=8)   # query: [F, D]
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

__all__ = ["FeatureIndex", "search_l2", "weighted_blend", "load_faiss_index",
           "load_index", "export_faiss"]


def search_l2(query: np.ndarray, db: np.ndarray, k: int = 8,
              chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """暴力 L2 最近邻。

    Args:
        query: [F, D] float32
        db:    [N, D] float32
        k:     返回近邻数
        chunk: 分块大小，控制内存峰值（分块矩阵乘法，每块 O(F*chunk)）

    Returns:
        (scores [F, k] float32, indices [F, k] int64)
        距离从小到大排列；N < k 时不足位用 -1 填充、距离用 inf 填充。
    """
    query = np.asarray(query, dtype=np.float32)
    db = np.asarray(db, dtype=np.float32)
    F, D = query.shape
    N = db.shape[0]
    k = min(k, N)
    # ||q-d||^2 = ||q||^2 + ||d||^2 - 2 q·d
    q2 = np.einsum("fd,fd->f", query, query)          # [F]
    d2 = np.einsum("nd,nd->n", db, db)                # [N]
    scores = np.full((F, k), np.inf, dtype=np.float32)
    inds = np.full((F, k), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        dist = q2[:, None] + d2[None, start:end] - 2.0 * (query @ db[start:end].T)
        np.maximum(dist, 0.0, out=dist)               # 数值修剪
        # 取当前块内 top-k
        blk_k = min(k, end - start)
        blk_idx = np.argpartition(dist, blk_k - 1, axis=1)[:, :blk_k]
        blk_scores = np.take_along_axis(dist, blk_idx, axis=1)
        order = np.argsort(blk_scores, axis=1)
        blk_idx = np.take_along_axis(blk_idx, order, axis=1)
        blk_scores = np.take_along_axis(blk_scores, order, axis=1)
        blk_inds = blk_idx + start
        # 合并进全局 top-k
        merged = np.concatenate([scores, blk_scores], axis=1)
        merged_idx = np.concatenate([inds, blk_inds], axis=1)
        keep = np.argpartition(merged, k - 1, axis=1)[:, :k]
        # 对每行按 merged 值排序
        row = np.arange(F)[:, None]
        vals = merged[row, keep]
        ord2 = np.argsort(vals, axis=1)
        keep = keep[row, ord2]
        scores = merged[row, keep]
        inds = merged_idx[row, keep]
    return scores, inds


def weighted_blend(query: np.ndarray, db: np.ndarray, k: int = 8) -> np.ndarray:
    """RVC 风格检索加权混合：返回与 query 同形的检索向量。

    score 为 L2 距离（越小越近），权重 w = (1/score)^2，行归一化后
    对 db[indices] 加权平均。等价于原 pipeline 的 npy 计算。
    """
    scores, inds = search_l2(query, db, k=k)
    valid = inds >= 0
    w = np.zeros_like(scores)
    w[valid] = 1.0 / np.maximum(scores[valid], 1e-6) ** 2
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    F, D = query.shape
    out = np.zeros((F, D), dtype=np.float32)
    for f in range(F):
        if not valid[f].any():
            continue
        out[f] = w[f, valid[f]] @ db[inds[f, valid[f]]]
    return out


class FeatureIndex:
    """自建特征索引：numpy .npz 存储，提供与 faiss 兼容的最小接口。"""

    def __init__(self, dim: int = 0, metric: str = "L2",
                 ids: Optional[np.ndarray] = None):
        self.dim = dim
        self.metric = metric.upper()          # L2 或 IP（内积未实现加权差异，先支持 L2）
        self._vectors = np.zeros((0, dim), dtype=np.float32) if dim else np.zeros((0, 0), dtype=np.float32)
        self._ids = np.zeros(0, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是 [N, D] 二维数组")
        if self.dim == 0:
            self.dim = vectors.shape[1]
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        elif vectors.shape[1] != self.dim:
            raise ValueError("维度不匹配: %d vs %d" % (vectors.shape[1], self.dim))
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + len(vectors), dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) != len(vectors):
            raise ValueError("ids 长度必须等于 vectors 行数")
        self._vectors = np.concatenate([self._vectors, vectors], axis=0)
        self._ids = np.concatenate([self._ids, ids], axis=0)

    def reconstruct(self, i: int) -> np.ndarray:
        return self._vectors[i].copy()

    def reconstruct_n(self, start: int, n: int) -> np.ndarray:
        return self._vectors[start:start + n].copy()

    def search(self, query: np.ndarray, k: int = 8):
        return search_l2(np.asarray(query, dtype=np.float32), self._vectors, k=k)

    def save(self, path: str) -> None:
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self._vectors,
            ids=self._ids,
            dim=np.int64(self.dim),
            metric=np.asarray(self.metric, dtype="S8"),
        )

    @classmethod
    def load(cls, path: str) -> "FeatureIndex":
        # 预检文件头：NPZ 是 ZIP（PK..），npy 是 \x93NUMPY；其余（如 faiss .index
        # 二进制）直接给出清晰错误，而不是 np.load 的 "pickled (object) data"。
        with open(path, "rb") as f:
            head = f.read(6)
        if head.startswith(b"PK\x03\x04"):
            # ZIP 容器：还需确认内含 .npy 成员（npz 特征）；torch .pth 也是
            # ZIP（含 data.pkl）会被 np.load 报 KeyError/allow_pickle，需拦截。
            import zipfile
            try:
                with zipfile.ZipFile(path) as zf:
                    has_npy = any(n.endswith(".npy") for n in zf.namelist())
            except zipfile.BadZipFile:
                has_npy = False
            if not has_npy:
                raise ValueError(
                    "%s 不是 .npz 索引（ZIP 内无 .npy 成员——可能是 torch .pth "
                    "模型文件或其它 ZIP）。索引请用 .npz/.ivf.npz/.index。" % path
                )
        elif not head.startswith(b"\x93NUMPY"):
            raise ValueError(
                "%s 不是 .npz 索引文件（文件头 %r）。若这是 faiss .index，请用 "
                ".index 后缀路径（自动走 faiss_reader 解析），或先转换。" % (path, head[:6])
            )
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("读取索引 %s 失败（非 .npz 或已损坏）：%s" % (path, exc))
        try:
            dim = int(data["dim"])
            metric = data["metric"].item()
            metric = metric.decode() if isinstance(metric, bytes) else str(metric)
            vectors = data["vectors"].astype(np.float32)
            ids = data["ids"].astype(np.int64)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("索引 %s 缺少预期键（dim/vectors/ids）或已损坏：%s" % (path, exc))
        # P1-011 完整性/版本校验：交叉一致性不满足即报错，禁止静默解析
        # 出"能加载但检索乱序/错位"的索引。
        if vectors.ndim != 2:
            raise ValueError("索引 %s vectors 必须是 [N,D]：实际 %r" % (path, vectors.shape))
        if dim > 0 and vectors.shape[1] != dim:
            raise ValueError(
                "索引 %s dim 不一致：声明 %d，vectors 实际 %d（文件损坏/版本不兼容）"
                % (path, dim, vectors.shape[1]))
        if len(ids) != len(vectors):
            raise ValueError(
                "索引 %s ids 与 vectors 行数不一致：%d vs %d（文件损坏）"
                % (path, len(ids), len(vectors)))
        if metric.upper() not in ("L2", "IP"):
            raise ValueError("索引 %s 不支持的 metric: %r" % (path, metric))
        idx = cls(dim=vectors.shape[1], metric=metric)
        idx._vectors = vectors
        idx._ids = ids
        return idx


def load_faiss_index(path: str) -> FeatureIndex:
    """解析 faiss 二进制 .index 文件并转为 FeatureIndex（.npz 兼容）。

    支持 IndexFlat/FlatL2/FlatIP、IndexIVFFlat（含 legacy 布局）与
    IndexIDMap(IndexFlat*)；量化索引（IVFPQ 等）与 faiss<1.5 旧版
    long-magic 格式抛 NotImplementedError（见 runtime.faiss_reader）。

    转换后的 FeatureIndex 用暴力 L2 检索，检索结果与 faiss 等价
    （向量与 id 完全保留，仅内部顺序按 faiss 倒排表顺序重排）。
    """
    from runtime.faiss_reader import read_faiss_index  # 延迟导入避免循环依赖
    return read_faiss_index(path)


def load_index(path: str):
    """统一索引加载入口：按文件后缀自动路由解析器（R1，双轨共同接口契约 §4.1）。

    路由表：
        ``.npz``      → ``FeatureIndex.load``（暴力检索，完整档默认）
        ``.ivf.npz``  → ``IVFIndex.load``（倒排分桶近似检索）
        ``.index`` / ``.faiss`` → ``faiss_reader.read_faiss_index``（faiss
            二进制，内部转 FeatureIndex / IVFIndex）
        ``.lidx``     → ``NotImplementedError``（轻量原型库格式，构建器与
            loader 由 OpenCL 轨提供，格式设计 v0.2 待双轨评审）

    设计约定（零行为变更）：
        - 本入口是双轨"检索统一接口"（交接文档 §4.1）的统一加载位，供
          后端选择器 / 未来前端无感切换索引格式使用；**不接入现有推理链路**
          （pipeline 内部仍走其私有 ``_load_index_cached``，行为不变）。
        - 返回 ``FeatureIndex`` 或 ``IVFIndex`` 实例（二者接口对齐：
          search / vectors_by_rows / ntotal / save / load，调用方无感）。
        - 路径不存在 → ``FileNotFoundError``；格式损坏 → ``ValueError`` /
          ``NotImplementedError``（由调用方捕获降级）。

    Args:
        path: 索引文件路径（.npz / .ivf.npz / .index / .faiss / .lidx）。

    Returns:
        ``FeatureIndex`` 或 ``IVFIndex``。
    """
    path = str(path)
    low = path.lower()
    if low.endswith(".ivf.npz"):
        from runtime.ivf_index import IVFIndex  # 延迟导入避免循环依赖
        return IVFIndex.load(path)
    if low.endswith(".index") or low.endswith(".faiss"):
        from runtime.faiss_reader import read_faiss_index  # noqa: PLC0415
        return read_faiss_index(path)
    if low.endswith(".lidx"):
        raise NotImplementedError(
            "%s: .lidx（轻量原型库）加载器由 OpenCL 轨提供，格式设计 v0.2 "
            "待双轨评审——当前 Vulkan 轨暂不支持，请改用 .npz/.ivf.npz/.index。" % path
        )
    return FeatureIndex.load(path)


def export_faiss(path: str, out_path: Optional[str] = None,
                 *, metric: object = "L2") -> str:
    """把本仓库索引（.npz/.ivf.npz/.index/.faiss）导出为 faiss .index 文件。

    VK-05 便捷入口（最小接线）：``load_index`` 读入 → ``faiss_writer`` 写回，
    使训练侧产出的自建索引可一键导出到通用 faiss 生态（"其他后端基本可用"的
    导出侧）。不改变任何现有读路径语义。

    Args:
        path: 源索引路径（.npz / .ivf.npz / .index / .faiss，按 load_index 路由）。
        out_path: 输出路径；缺省为同目录同名替换后缀为 .index。
        metric: 写出度量，仅支持 L2（IP/其它抛 NotImplementedError，
            见 faiss_writer 度量铁律 P-RET-001/002）。

    Returns:
        输出路径。
    """
    idx = load_index(path)
    if out_path is None:
        out_path = os.path.splitext(str(path))[0] + ".index"
    from runtime.faiss_writer import write_faiss_index  # noqa: PLC0415 —— 延迟避免循环依赖
    return write_faiss_index(idx, out_path, metric=metric)
