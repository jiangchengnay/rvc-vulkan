# -*- coding: utf-8 -*-
"""倒排分桶（IVF）特征检索索引——零 faiss / torch 依赖（P2）。

背景：``runtime.retrieval.FeatureIndex`` 的 ``search`` 是**全量暴力 L2**
（O(N·D)）。当特征 >20 万条时，RVC 每 10ms 帧一次的检索会把推理拖垮。
本模块实现经典粗量化倒排索引，把查询复杂度从 O(N·D) 降到
O(nlist·D + nprobe·(N/nlist)·D)：

    IVFIndex: centroids[nlist, D] + 桶分配（每条向量归入最近中心的桶）
    查询：query 找最近 ``nprobe`` 个中心（O(nlist·D)）
          → 只在这 nprobe 个桶内做 L2 暴力 kNN
            （复用 ``runtime.retrieval.search_l2``，桶均规模 N/nlist；
            批量查询按小块把块内各行桶取并集为候选池，是逐行桶检索的
            超集，只增召回）；
          → 桶内 top-k 即最终结果（无需跨桶合并）。

向量仍**全量存储**（[N, D] f32，与暴力索引同内存）；省的是查询计算量
（每查询只扫 nprobe/nlist 比例的库）。``nprobe == nlist`` 时退化为全库
暴力检索，与 FeatureIndex **数学等价**（走同一份 search_l2）。

与 ``FeatureIndex`` 的接口对齐（pipeline / realtime 无感切换）：
    - ``search(query, k) -> (scores, inds)``：scores 升序、inds 为向量
      行号（-1 / inf 填充语义一致），RVC 的 (1/d)^2 加权逻辑无需改动；
    - ``vectors_by_rows(inds) -> [.., D]``：按行号取向量（加权混合用，
      等价于 ``FeatureIndex`` 的 ``index_vectors[ix]`` 语义）；
    - ``reconstruct / reconstruct_n / ntotal / save / load``：等价；
    - ``to_flat()``：转回 FeatureIndex（需要全量暴力时）。

.npz 存储键：``centroids / assignments / vectors / ids``（任务约定），
另附 ``dim / metric / nlist / nprobe``——其中 dim/metric/vectors/ids 与
FeatureIndex 保存格式一致，旧代码用 ``FeatureIndex.load`` 读 .ivf.npz 也能
得到全量索引（向后兼容的降级路径）。

用法::

    ivf = IVFIndex(dim=768, nlist=256, nprobe=1)
    ivf.train(vectors)              # 粗量化器中心（自带 minibatch-kmeans）
    ivf.add(vectors, ids)           # 分桶
    scores, inds = ivf.search(query, k=8)
    vecs = ivf.vectors_by_rows(inds)
    ivf.save("added_..._v2.ivf.npz")
    ivf2 = IVFIndex.load("added_..._v2.ivf.npz")
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from runtime.retrieval import FeatureIndex, search_l2

__all__ = ["IVFIndex"]

# 查询分块：一次处理多少条 query（块内各行 nprobe 桶取**并集**作为候选池，
# 是逐行桶检索的超集——只增召回；用小块把并集规模控制在 ~块数×nprobe 桶）
_QUERY_CHUNK = 8
# 分桶分配分块（控制 [chunk, nlist] float64 中间阵峰值）
_ASSIGN_CHUNK = 8192
# kmeans 强度（IVF 自带的 minibatch_kmeans）：大特征集
# （>20 万条）用更强的批量/迭代避免中心坍缩导致桶严重失衡（默认弱参数
# 在无聚类结构的随机数据上会把查询拖回接近全量暴力）。
_KMEANS_BATCH = 1024
_KMEANS_ITERS = 30


def _pairwise_l2(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """分块计算 [n, m] L2 距离平方矩阵（float64，独立实现避免循环依赖）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.shape[0], b.shape[0]
    a2 = np.einsum("nd,nd->n", a, a)
    b2 = np.einsum("md,md->m", b, b)
    out = np.empty((n, m), dtype=np.float64)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        out[:, s:e] = a2[:, None] + b2[None, s:e] - 2.0 * (a @ b[s:e].T)
    np.maximum(out, 0.0, out=out)
    return out


def _minibatch_kmeans(X: np.ndarray, n_clusters: int, batch_size: int = 256,
                      iters: int = 8, seed: int = 42) -> np.ndarray:
    """简化版 mini-batch KMeans（零 sklearn 依赖），供 IVF 粗量化器中心使用。

    与 sklearn MiniBatchKMeans 的语义差异：中心随机采样初始化（不跑
    kmeans++）；每轮随机取小批量，对命中中心用该批点均值整批重估（不做
    指数衰减）。行为合理且可复现（固定 seed），足够作为粗量化中心。

    Args:
        X: [N, D] float32 特征。
        n_clusters: 中心数（> N 时自动缩到 N）。
        batch_size: 每轮小批量大小。
        iters: 迭代轮数。
        seed: 随机种子。

    Returns:
        [k, D] float32 聚类中心，k = min(n_clusters, N)。
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"minibatch_kmeans 需要 [N, D] 输入，实际 {X.ndim}D")
    n = X.shape[0]
    k = int(min(n_clusters, n))
    if k <= 0:
        raise ValueError("n_clusters 必须为正")
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(n, k, replace=False)].astype(np.float64).copy()
    bs = min(batch_size, n)
    for _ in range(iters):
        batch = X[rng.choice(n, bs, replace=False)]  # [bs, D]
        dist = _pairwise_l2(batch, centers)
        assign = np.argmin(dist, axis=1)
        for c in np.unique(assign):
            sel = batch[assign == c]
            centers[c] = sel.mean(axis=0)
    return centers.astype(np.float32)


class IVFIndex:
    """粗量化倒排索引：centroids[nlist,D] + 桶列表。

    语义对齐 ``FeatureIndex.search_l2``（scores 升序、inds 对应行号），
    保证 pipeline 的 (1/d)^2 加权逻辑无需改动。
    """

    def __init__(self, dim: int = 0, nlist: int = 256, nprobe: int = 1,
                 metric: str = "L2"):
        """Args:
            dim: 特征维度（train() 时会按训练数据自动确定）。
            nlist: 粗量化桶数（> 训练样本数时自动缩到样本数）。
            nprobe: 默认查询桶数（search 可临时覆盖；nprobe=nlist 等价暴力）。
            metric: 仅支持 "L2"（对齐 FeatureIndex）。
        """
        nlist = int(nlist)
        nprobe = int(nprobe)
        if nlist < 1:
            raise ValueError("nlist 必须 >= 1")
        if nprobe < 1:
            raise ValueError("nprobe 必须 >= 1")
        self.dim = int(dim)
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = str(metric).upper()
        self.centroids = np.zeros((0, self.dim), dtype=np.float32) if dim \
            else np.zeros((0, 0), dtype=np.float32)
        self._vectors = np.zeros((0, self.dim), dtype=np.float32) if dim \
            else np.zeros((0, 0), dtype=np.float32)
        self._ids = np.zeros(0, dtype=np.int64)
        self._assignments = np.zeros(0, dtype=np.int64)
        self._buckets: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # 基础状态
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        return len(self._ids)

    @property
    def buckets(self) -> List[np.ndarray]:
        """每桶的行号数组（全局行号，即 _vectors 的行下标）。"""
        return self._buckets

    # ------------------------------------------------------------------
    # 训练与构建
    # ------------------------------------------------------------------
    def train(self, vectors: np.ndarray, seed: int = 42,
              batch_size: int = _KMEANS_BATCH,
              iters: int = _KMEANS_ITERS) -> "IVFIndex":
        """用 minibatch-kmeans 训练粗量化器中心（IVF 自带实现）。

        Args:
            vectors: [N, D] float32 训练特征（可传大特征集的代表中心）。
            seed: 聚类随机种子（固定可复现）。
            batch_size: kmeans 小批量大小（默认 1024，大特征集更稳）。
            iters: kmeans 迭代轮数（默认 30；更强聚类 → 桶更均衡 → 查询更快）。

        Returns:
            self（支持链式调用）。
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"train 需要 [N, D] 输入，实际 {vectors.ndim}D")
        if vectors.shape[0] == 0:
            raise ValueError("train 训练集为空")
        self.dim = vectors.shape[1]
        k = min(self.nlist, vectors.shape[0])
        if k <= 0:
            raise ValueError("训练样本数不足以训练任何桶")
        self.nlist = k
        self.centroids = _minibatch_kmeans(
            vectors, n_clusters=k, batch_size=int(batch_size),
            iters=int(iters), seed=seed,
        ).astype(np.float32)
        return self

    def set_centroids(self, centroids: np.ndarray) -> "IVFIndex":
        """直接设置已训练好的粗量化中心（供外部复用其聚类中心，
        避免对同一批数据重复聚类）。

        Args:
            centroids: [nlist, D] float32。

        Returns:
            self。
        """
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[0] == 0:
            raise ValueError("centroids 必须为 [nlist, D] 非空二维数组")
        self.centroids = centroids
        self.nlist = centroids.shape[0]
        self.dim = centroids.shape[1]
        return self

    def _assign(self, vectors: np.ndarray) -> np.ndarray:
        """把 vectors 每行分配到最近中心（桶标签 [len, ] int64）。"""
        centers = np.asarray(self.centroids, dtype=np.float64)
        v = np.asarray(vectors, dtype=np.float64)
        labels = np.empty(v.shape[0], dtype=np.int64)
        for s in range(0, v.shape[0], _ASSIGN_CHUNK):
            e = min(s + _ASSIGN_CHUNK, v.shape[0])
            labels[s:e] = np.argmin(_pairwise_l2(v[s:e], centers), axis=1)
        return labels

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None,
            assignments: Optional[np.ndarray] = None) -> None:
        """加入向量并分桶（须先 train()/set_centroids()）。

        Args:
            vectors: [N, D] float32。
            ids: [N] int64 外部 id（缺省为连续行号）。
            assignments: [N] int64 预计算桶标签（已分桶时传入，
                避免重复计算；校验长度后直接使用）。
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是 [N, D] 二维数组")
        if self.centroids.shape[0] == 0:
            raise RuntimeError("IVFIndex 未训练：请先调用 train()/set_centroids()")
        if vectors.shape[1] != self.dim:
            raise ValueError("维度不匹配: %d vs %d" % (vectors.shape[1], self.dim))
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + len(vectors), dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) != len(vectors):
            raise ValueError("ids 长度必须等于 vectors 行数")
        if assignments is None:
            new_assign = self._assign(vectors)
        else:
            new_assign = np.asarray(assignments, dtype=np.int64)
            if len(new_assign) != len(vectors):
                raise ValueError("assignments 长度必须等于 vectors 行数")
        self._vectors = np.concatenate([self._vectors, vectors], axis=0)
        self._ids = np.concatenate([self._ids, ids], axis=0)
        self._assignments = np.concatenate([self._assignments, new_assign], axis=0)
        self._rebuild_buckets()

    def _rebuild_buckets(self) -> None:
        """由 assignments 重建桶列表（O(N log N)，单次构建可接受）。"""
        nlist = self.nlist
        a = self._assignments
        if a.size == 0:
            self._buckets = [np.zeros(0, dtype=np.int64) for _ in range(nlist)]
            return
        order = np.argsort(a, kind="stable")
        counts = np.bincount(a, minlength=nlist)
        splits = np.cumsum(counts)[:-1]
        self._buckets = [arr.astype(np.int64, copy=False)
                         for arr in np.split(order, splits)]

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _nearest_centroids(self, query: np.ndarray, nprobe: int,
                           empty_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """query [F, D] → 每行最近 nprobe 个桶 id [F, nprobe]（按距升序）。

        empty_mask 为 [nlist] bool 时先把空桶距离置 inf，保证探测到的桶
        非空（退化聚类产生重复中心时，查询不会命中空桶而丢候选）。
        """
        c = self.centroids
        q2 = np.einsum("fd,fd->f", query, query)
        c2 = np.einsum("nd,nd->n", c, c)
        dist = q2[:, None] + c2[None, :] - 2.0 * (query @ c.T)
        np.maximum(dist, 0.0, out=dist)
        if empty_mask is not None:
            dist[:, empty_mask] = np.inf
        top = np.argpartition(dist, nprobe - 1, axis=1)[:, :nprobe]
        vals = np.take_along_axis(dist, top, axis=1)
        order = np.argsort(vals, axis=1)
        return np.take_along_axis(top, order, axis=1)

    def search(self, query: np.ndarray, k: int = 8,
               nprobe: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """倒排分桶近似 kNN。

        实现：query 按小块（_QUERY_CHUNK 条）处理，每块先对中心打分找出
        各行最近 nprobe 个桶，取块内各行的桶**并集**为候选池（逐行桶检索的
        超集，不会降低召回），再对候选池做一次分块 L2 kNN（复用 search_l2）
        并把局部行号映射回全局行号。

        Args:
            query: [F, D] float32。
            k: 返回近邻数（候选池不足 k 时 -1/inf 填充，与暴力语义一致）。
            nprobe: 本次查询的桶数；缺省用 self.nprobe。
                nprobe >= nlist 时退化为全库暴力（与 FeatureIndex 数学等价）。

        Returns:
            (scores [F, k] float32 升序, inds [F, k] int64 全局行号)。
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 2:
            raise ValueError(f"query 需要 [F, D] 输入，实际 {query.ndim}D")
        if nprobe is None:
            nprobe = self.nprobe
        nprobe = int(nprobe)
        if nprobe < 1:
            raise ValueError("nprobe 必须 >= 1")
        F, D = query.shape
        N = self.ntotal
        k = min(int(k), N)
        if F == 0 or k <= 0 or N == 0:
            return (np.full((F, k), np.inf, dtype=np.float32),
                    np.full((F, k), -1, dtype=np.int64))
        if D != self.dim:
            raise ValueError("维度不匹配: %d vs %d" % (D, self.dim))
        # nprobe >= nlist（或只有 1 个桶 / 未分桶）→ 全量暴力，数学等价
        if nprobe >= self.nlist or len(self._buckets) <= 1:
            return search_l2(query, self._vectors, k=k)

        buckets = self._buckets
        empty_mask = np.fromiter(
            (b.size == 0 for b in buckets), dtype=bool, count=len(buckets)
        )
        scores = np.full((F, k), np.inf, dtype=np.float32)
        inds = np.full((F, k), -1, dtype=np.int64)
        for s in range(0, F, _QUERY_CHUNK):
            e = min(s + _QUERY_CHUNK, F)
            top = self._nearest_centroids(query[s:e], nprobe, empty_mask)
            uni = np.unique(top)                                # 去重桶 id
            parts = [buckets[i] for i in uni if buckets[i].size]
            if not parts:
                continue
            pool = np.concatenate(parts)                        # 全局行号并集
            sc_block, ix_block = search_l2(
                query[s:e], self._vectors[pool], k=k
            )                                                   # [rows, m] m<=k
            if sc_block.shape[1] < k:                           # 候选不足 k → 补齐
                pad_n = k - sc_block.shape[1]
                sc_block = np.concatenate(
                    [sc_block, np.full((sc_block.shape[0], pad_n),
                                       np.inf, dtype=np.float32)], axis=1)
                ix_block = np.concatenate(
                    [ix_block, np.full((ix_block.shape[0], pad_n),
                                       -1, dtype=np.int64)], axis=1)
            inds[s:e] = np.where(ix_block >= 0,
                                 pool[np.clip(ix_block, 0, None)], -1)
            scores[s:e] = sc_block
        return scores, inds

    def vectors_by_rows(self, inds: np.ndarray) -> np.ndarray:
        """按行号取回向量（pipeline 加权混合用，等价 index_vectors[ix]）。

        Args:
            inds: int64 行号数组（任意形状；-1 或越界行号 → 该位置零向量）。

        Returns:
            [*inds.shape, D] float32。
        """
        inds = np.asarray(inds, dtype=np.int64)
        flat = inds.reshape(-1)
        out = np.zeros((flat.size, self.dim), dtype=np.float32)
        valid = (flat >= 0) & (flat < self.ntotal)
        out[valid] = self._vectors[flat[valid]]
        return out.reshape(inds.shape + (self.dim,))

    def reconstruct(self, i: int) -> np.ndarray:
        return self._vectors[i].copy()

    def reconstruct_n(self, start: int, n: int) -> np.ndarray:
        return self._vectors[start:start + n].copy()

    # ------------------------------------------------------------------
    # 持久化 / 转换
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """保存为 .ivf.npz（键：centroids/assignments/vectors/ids/dim/metric/
        nlist/nprobe；前四者加 dim/metric 与 FeatureIndex 格式兼容）。"""
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            centroids=self.centroids,
            assignments=self._assignments,
            vectors=self._vectors,
            ids=self._ids,
            dim=np.int64(self.dim),
            metric=np.asarray(self.metric, dtype="S8"),
            nlist=np.int64(self.nlist),
            nprobe=np.int64(self.nprobe),
        )

    @classmethod
    def load(cls, path: str) -> "IVFIndex":
        with open(path, "rb") as f:
            head = f.read(6)
        if head.startswith(b"PK\x03\x04"):
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
                ".index 后缀路径（自动走 faiss_reader 解析）。" % (path, head[:6])
            )
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("读取索引 %s 失败（非 .npz 或已损坏）：%s" % (path, exc))
        dim = int(data["dim"])
        metric = data["metric"].item()
        if isinstance(metric, bytes):
            metric = metric.decode()
        else:
            metric = str(metric)
        centroids = data["centroids"].astype(np.float32)
        nlist = int(data["nlist"]) if "nlist" in data else centroids.shape[0]
        nprobe = int(data["nprobe"]) if "nprobe" in data else 1
        vectors = data["vectors"].astype(np.float32)
        ids = data["ids"].astype(np.int64)
        a = data["assignments"] if "assignments" in data \
            else np.zeros(len(ids), dtype=np.int64)
        a = np.asarray(a, dtype=np.int64)
        # P1-011 完整性/版本校验：交叉一致性不满足即报错，禁止静默解析出
        # "能加载但桶错位→检索乱序"的索引。
        if vectors.ndim != 2:
            raise ValueError("%s vectors 必须是 [N,D]：实际 %r" % (path, vectors.shape))
        if dim > 0 and vectors.shape[1] != dim:
            raise ValueError("%s dim 不一致：声明 %d，vectors 实际 %d（文件损坏/版本不兼容）"
                             % (path, dim, vectors.shape[1]))
        if len(ids) != len(vectors):
            raise ValueError("%s ids 与 vectors 行数不一致：%d vs %d" % (path, len(ids), len(vectors)))
        if centroids.ndim != 2 or centroids.shape[1] != dim:
            raise ValueError("%s centroids 形状 %r 与 dim=%d 不符（文件损坏）"
                             % (path, centroids.shape, dim))
        if nlist != centroids.shape[0]:
            raise ValueError("%s nlist=%d 与 centroids 行数 %d 不一致（文件损坏）"
                             % (path, nlist, centroids.shape[0]))
        if len(a) != len(ids):
            raise ValueError("%s assignments 与 vectors 行数不一致：%d vs %d"
                             % (path, len(a), len(ids)))
        if len(a) and (int(a.min()) < 0 or int(a.max()) >= nlist):
            raise ValueError("%s assignments 越界：范围 [%d,%d) 超出 nlist=%d（文件损坏）"
                             % (path, int(a.min()), int(a.max()), nlist))
        if metric.upper() not in ("L2", "IP"):
            raise ValueError("%s 不支持的 metric: %r" % (path, metric))
        idx = cls(dim=dim, nlist=nlist, nprobe=nprobe, metric=metric)
        idx.centroids = centroids
        idx._vectors = vectors
        idx._ids = ids
        idx._assignments = a
        idx._rebuild_buckets()
        return idx

    def to_flat(self) -> FeatureIndex:
        """转回全量暴力 FeatureIndex（需要全量检索/兼容旧代码时）。"""
        idx = FeatureIndex(dim=self.dim, metric=self.metric, ids=self._ids.copy())
        idx._vectors = self._vectors.copy()
        return idx