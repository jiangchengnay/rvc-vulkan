# -*- coding: utf-8 -*-
"""RVC 训练侧特征索引构建（纯 numpy 移植，对齐 ``train/train_index.py``）。

零 torch / faiss / sklearn 依赖；scipy 不需要。

与原版（faiss IVF）的差异（IVF 简化方式）：
1. 原版用 sklearn MiniBatchKMeans(n_clusters=10000) 压缩 >20 万条特征，
   本实现用自带的 ``minibatch_kmeans``（每轮小批量整批重估中心）；
2. 原版用 faiss "IVF{n},Flat"（n_ivf 个粗量化器桶 + nprobe=1 检索），
   本实现保留 n_ivf 公式与桶划分语义（粗中心训练 + 逐向量分桶统计），
   检索层有两种模式：
     - ``mode="ivf"``（默认）：输出 ``.ivf.npz``，用
       ``runtime.ivf_index.IVFIndex`` 的**倒排分桶检索**（查询复杂度
       O(nlist·D + nprobe·N/nlist·D)，适合 >20 万条的大索引）；
     - ``mode="flat"``（及历史别名 ``auto``/``single``，旧行为）：
       输出 ``.npz``，用 ``runtime.retrieval.FeatureIndex`` 的**暴力 L2**
       检索（不丢失召回，代价是检索 O(N·D)，适合中小规模索引）；
3. 原版输出 ``added_IVF*_Flat_nprobe_*_{exp}_{version}.index``（faiss 二进制），
   本实现输出同名约定但后缀为 ``.ivf.npz``（IVFIndex）或 ``.npz``
   （FeatureIndex 自描述格式）；
4. 多说话人 manifest 的分组训练（对齐原版 train_index.py 的多说话人模式）：
   存在 ``<exp_dir>/multispeaker_manifest.json`` 且 ``mode != "single"`` 时，
   按 speaker_id 分组（manifest 条目的 ``speaker_id`` 字段优先，否则从
   ``output_key`` 的 ``_s<digits>`` 段解析），每组独立构建索引并输出
   ``..._spkidN.ivf.npz`` / ``..._spkidN.npz``（对齐原版 ``_spkidN`` 后缀），
   每组另存 ``total_fea_spkidN.npy``；manifest 缺失 / 解析失败 /
   ``mode="single"`` 时保持单索引行为（全部特征合并为一个索引，回归不变）。

用法（作为库）::

    from runtime.train.train_index import train_index
    path = train_index("workspaces/<项目>/<任务>/exp", version=2, n_cpu=1, outside_root="assets/indices")  # 默认 ivf
    paths = train_index("workspaces/<项目>/<任务>/exp", version=2, manifest_path="workspaces/<项目>/<任务>/exp/multispeaker_manifest.json")  # 多说话人 -> list[str]

命令行（与原版一致）::

    python runtime/train/train_index.py <exp_name> <version> <outside_root> <n_cpu> [mode]
    <exp_name> 须为工作区绝对路径（workspaces/<项目>/<任务>/exp）；旧 logs/<exp> 已废弃。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys

import numpy as np

try:  # 作为包的一部分被 import（推荐）
    from ..retrieval import FeatureIndex
except ImportError:  # 以脚本方式直接运行
    _ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from runtime.retrieval import FeatureIndex

__all__ = [
    "build_feature_store",
    "minibatch_kmeans",
    "train_index",
]

# 特征条数超过该值时先用 KMeans 压缩为聚类中心再训练粗量化器（对齐原版阈值）
_KMEANS_N = 200000
# 压缩目标中心数上限（原版固定 10000；这里允许缩到 N//10 以控制复杂度）
_KMEANS_CLUSTERS = 10000
# 固定随机种子（打乱 / 聚类可复现）
_SEED = 42


def _log_file(exp_dir: str) -> str:
    return os.path.join(exp_dir, "train_index.log")


def _println(msg, exp_dir: str) -> None:
    """打印到 stdout 并追加写入 ``<exp_dir>/train_index.log``。"""
    print(msg, flush=True)
    try:
        os.makedirs(exp_dir, exist_ok=True)
        with open(_log_file(exp_dir), "a", encoding="utf8") as f:
            f.write("%s\n" % msg)
    except OSError:
        pass


def _feature_dir(exp_dir: str, version: int) -> str:
    """返回 3_feature256（v1）或 3_feature768（v2）目录。"""
    sub = "3_feature256" if int(version) == 1 else "3_feature768"
    return os.path.join(exp_dir, sub)


def build_feature_store(exp_dir: str, version: int = 2, paths=None):
    """读取 ``3_feature256|768`` 下全部 .npy 特征并拼接。

    Args:
        exp_dir: 实验目录（工作区绝对路径 workspaces/<项目>/<任务>/exp）。
        version: 1 → 3_feature256（HuBERT v1，256 维）；2 → 3_feature768。
        paths: 可选，限定只读这些 .npy 路径（多说话人分组时按组传入）；
            缺省扫描 ``3_feature*`` 目录下全部 .npy。

    Returns:
        (X, stats)：
            X: [N, D] float32 拼接后的全部特征（行序与文件名排序一致）；
            stats: list[dict]，每项 {file, frames, shape} 逐文件统计。

    Raises:
        FileNotFoundError: 特征目录不存在。
        RuntimeError: 特征目录为空或无有效 .npy。
        ValueError: 特征维度不一致或不是 2D。
    """
    feature_dir = _feature_dir(exp_dir, version)
    if paths is None:
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(
                "特征目录不存在：%s（请先进行特征提取）" % feature_dir
            )
        paths = sorted(
            os.path.join(feature_dir, name)
            for name in os.listdir(feature_dir)
            if name.lower().endswith(".npy")
        )
    else:
        paths = sorted(os.path.abspath(os.fspath(p)) for p in paths)
    if not paths:
        raise RuntimeError(
            "特征目录为空：%s（请先进行特征提取）" % feature_dir
        )

    arrays = []
    stats = []
    dim = None
    for p in paths:
        a = np.load(p, allow_pickle=False)
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(
                "特征文件必须是 2D [frame, D]：%s（实际 %dD）" % (p, a.ndim)
            )
        if dim is None:
            dim = a.shape[1]
        elif a.shape[1] != dim:
            raise ValueError(
                "特征维度不一致：%s 为 %d，其余为 %d" % (p, a.shape[1], dim)
            )
        arrays.append(a)
        stats.append(
            {"file": os.path.basename(p), "frames": int(a.shape[0]),
             "shape": (int(a.shape[0]), int(a.shape[1]))}
        )
    X = np.concatenate(arrays, axis=0)
    return X, stats


def minibatch_kmeans(
    X: np.ndarray,
    n_clusters: int,
    batch_size: int = 256,
    iters: int = 8,
    seed: int = _SEED,
) -> np.ndarray:
    """简化版 mini-batch KMeans（自实现，零 sklearn 依赖）。

    与 sklearn MiniBatchKMeans 的语义差异（docstring 注明）：
    - 中心从数据中随机采样初始化（不跑 kmeans++）；
    - 每轮随机取一个小批量，对命中中心用**该批点的均值**整批重估
      （sklearn 是逐点指数衰减更新 lr=1/count；这里不做衰减）；
    - 行为合理且可复现（固定 seed），足够作为 IVF 粗量化器中心。

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
        # 分块距离 [bs, k] 取最近中心
        dist = _pairwise_l2(batch, centers)
        assign = np.argmin(dist, axis=1)
        for c in np.unique(assign):
            sel = batch[assign == c]
            centers[c] = sel.mean(axis=0)  # 整批重估
    return centers.astype(np.float32)


def _pairwise_l2(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """分块计算 [n, m] L2 距离平方矩阵（控制内存峰值）。"""
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


def _assign_buckets(X: np.ndarray, centers: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """把 X 每行分配到最近的桶中心（IVF 粗分配），返回桶标签 [N]。"""
    X = np.asarray(X, dtype=np.float64)
    labels = np.empty(X.shape[0], dtype=np.int64)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        labels[s:e] = np.argmin(_pairwise_l2(X[s:e], centers), axis=1)
    return labels


def _load_speaker_map(exp_dir: str, manifest_path=None) -> dict:
    """读取多说话人 manifest → ``{output_key: speaker_id}``。

    manifest 缺失 / JSON 损坏 / 无有效条目时返回 ``{}``（调用方回退单索引，
    与现有行为一致）。speaker_id 优先取条目 ``speaker_id`` 字段；缺失时从
    ``output_key`` 的 ``_s<digits>`` 段解析（对齐 ``tools/multispeaker.py``
    的 ``ms%04d_s%03d_<sha>`` 命名）。

    Args:
        exp_dir: 实验目录（工作区绝对路径 workspaces/<项目>/<任务>/exp）。
        manifest_path: 显式 manifest 路径；缺省读
            ``<exp_dir>/multispeaker_manifest.json``。

    Returns:
        dict: ``{output_key: int speaker_id}``（可能为空）。
    """
    path = manifest_path or os.path.join(exp_dir, "multispeaker_manifest.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return {}
    entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
    if not isinstance(entries, list) or not entries:
        return {}
    out = {}
    _sid_in_key = re.compile(r"_s(\d{1,3})(?:_|$)")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("output_key")
        if not isinstance(key, str) or not key:
            continue
        sid = entry.get("speaker_id")
        if sid is None:
            m = _sid_in_key.search(key)
            if m is None:
                continue
            sid = int(m.group(1))
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            continue
        out[key] = sid
    return out


def _feature_stem(path: str) -> str:
    """特征文件 stem：去掉 ``.npy`` 后缀，兼容 ``<key>.wav.npy`` 命名。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith(".wav"):
        stem = stem[:-4]
    return stem


def _group_feature_paths(feature_dir: str, speaker_map: dict) -> dict:
    """按 speaker_id 把特征文件分组（对齐原版 train_index.py 的分组逻辑）。

    ``output_key`` 匹配优先取完整 stem，否则去掉末尾 ``_<idx>`` 段
    （preprocess 切分产物 ``<output_key>_<idx>.npy``）。

    Args:
        feature_dir: 3_feature256|768 目录。
        speaker_map: ``{output_key: speaker_id}``（可为空）。

    Returns:
        dict: ``{speaker_id: [feature_path, ...]}``；speaker_map 为空时为空
        dict（调用方回退单索引）。
    """
    if not speaker_map:
        return {}
    paths = sorted(
        os.path.join(feature_dir, name)
        for name in os.listdir(feature_dir)
        if name.lower().endswith(".npy")
    )
    groups: dict = {}
    for p in paths:
        stem = _feature_stem(p)
        output_key = stem if stem in speaker_map else stem.rsplit("_", 1)[0]
        sid = speaker_map.get(output_key)
        if sid is not None:
            groups.setdefault(sid, []).append(p)
    return groups


def _build_single_index(
    exp_dir: str,
    version: int,
    n_cpu: int,
    mode: str,
    outside_root: str,
    seed: int,
    exp_name: str,
    _flat: bool,
    paths,
    speaker_id=None,
) -> str:
    """为**一组**特征构建一个索引（单说话人或多说话人的单个分组）。

    逻辑与原 train_index 主体一致（读取→打乱→聚类压缩→IVF→保存→外链）；
    ``speaker_id`` 非 None 时输出名带 ``_spkidN`` 后缀（对齐原版），并保存
    ``total_fea_spkidN.npy``（打乱后的全量特征，对齐原版）。

    Args:
        paths: 特征文件列表；None 表示扫描整个 ``3_feature*`` 目录（单索引）。

    Returns:
        索引文件绝对路径。
    """
    scope = ("[spk%s] " % speaker_id) if speaker_id is not None else ""
    suffix = "" if speaker_id is None else "_spkid%s" % speaker_id

    # 1) 读取并打乱特征（固定 seed 可复现）
    big_npy, stats = build_feature_store(exp_dir, version, paths=paths)
    n_total, dim = big_npy.shape
    rng = np.random.default_rng(seed)
    big_npy = big_npy[rng.permutation(n_total)]
    np.save(os.path.join(exp_dir, "total_fea%s.npy" % suffix), big_npy)
    _println(
        "[索引训练] %s特征文件数：%s | 总条数：%s | 维度：%s"
        % (scope, len(stats), n_total, dim),
        exp_dir,
    )
    for st in stats[:5]:
        _println(
            "  [索引训练] %s  %s frames=%s" % (scope, st["file"], st["frames"]),
            exp_dir,
        )
    if len(stats) > 5:
        _println(
            "  [索引训练] %s  ... 其余 %s 个文件" % (scope, len(stats) - 5),
            exp_dir,
        )

    # 2) 大特征集先压缩为聚类代表中心（对齐原版 MiniBatchKMeans(10000)）
    repr_npy = big_npy
    if n_total > _KMEANS_N:
        n_centers = int(min(_KMEANS_CLUSTERS, n_total // 10))
        _println(
            "[索引训练] %s特征超过 %s 条，聚类为 %s 个中心作为粗量化器训练集"
            % (scope, _KMEANS_N, n_centers),
            exp_dir,
        )
        repr_npy = minibatch_kmeans(
            big_npy,
            n_clusters=n_centers,
            batch_size=max(256, 256 * int(n_cpu)),
            seed=seed,
        )
        _println(
            "[索引训练] %s聚类完成，代表集形状：%s" % (scope, repr_npy.shape), exp_dir
        )

    # 3) n_ivf 公式（保留原版）＋ 粗量化器中心训练
    n_ivf = max(1, min(int(16 * np.sqrt(n_total)), n_total // 39))
    _println(
        "[索引训练] %s特征形状：%s | IVF数量：%s" % (scope, big_npy.shape, n_ivf),
        exp_dir,
    )
    bucket_centers = minibatch_kmeans(
        repr_npy,
        n_clusters=n_ivf,
        batch_size=max(256, 256 * int(n_cpu)),
        seed=seed,
    )
    _println("[索引训练] %s粗量化器（桶）训练完成：%s 个桶" % (scope, n_ivf), exp_dir)

    # 4) IVF 分桶统计（检索层按 mode 决定；桶分布仅输出）
    bucket_labels = _assign_buckets(big_npy, bucket_centers)
    sizes = np.bincount(bucket_labels, minlength=n_ivf)
    _println(
        "[索引训练] %s桶大小 min/max/mean：%s/%s/%.1f"
        % (scope, sizes.min(), sizes.max(), sizes.mean()),
        exp_dir,
    )

    # 5) 构建检索索引并写入全部特征（ivf → 倒排分桶；flat → 全量暴力）
    ids = np.arange(n_total, dtype=np.int64)
    if _flat:
        index = FeatureIndex(dim=dim, metric="L2")
        index.add(big_npy, ids=ids)
        _println(
            "[索引训练] %sFeatureIndex 写入完成：ntotal=%s" % (scope, index.ntotal),
            exp_dir,
        )
    else:
        from runtime.ivf_index import IVFIndex  # 延迟导入避免循环依赖

        index = IVFIndex(dim=dim, nlist=n_ivf, nprobe=1, metric="L2")
        index.set_centroids(bucket_centers)  # 复用 step-3 已训练的粗量化器
        index.add(big_npy, ids=ids, assignments=bucket_labels)  # 复用已分桶
        _println(
            "[索引训练] %sIVFIndex 写入完成：ntotal=%s nlist=%s nprobe=%s"
            % (scope, index.ntotal, index.nlist, index.nprobe),
            exp_dir,
        )

    # 6) 保存（工作区 exp/ + 外部目录硬链接/复制）
    suffix_ext = ".npz" if _flat else ".ivf.npz"
    added_name = "added_IVF%s_Flat_nprobe_1_%s_v%s%s%s" % (
        n_ivf, exp_name, version, suffix, suffix_ext
    )
    added_path = os.path.join(exp_dir, added_name)
    index.save(added_path)
    _println("[索引训练] %s成功构建索引：%s" % (scope, added_name), exp_dir)

    _link_outside(added_path, outside_root, exp_name, exp_dir)
    return added_path


def train_index(
    exp_dir: str,
    version: int = 2,
    n_cpu: int = 1,
    mode: str = "ivf",
    outside_root: str = "assets/indices",
    seed: int = _SEED,
    manifest_path=None,
):
    """构建特征索引（对齐原版 ``train/train_index.py`` 的训练流程）。

    Args:
        exp_dir: 实验目录（工作区绝对路径，须含 3_feature256|768/*.npy）。
        version: 1 → 256 维特征（3_feature256）；2 → 768 维（3_feature768）。
        n_cpu: 进程数（本实现为纯 numpy 单进程，保留参数以对齐原版接口，
            仅用于日志展示与 batch 规模选择）。
        mode: 'ivf'（默认）→ 倒排分桶索引，保存 ``.ivf.npz``
            （IVFIndex，近似检索）；'flat' → 全量暴力索引，保存 ``.npz``
            （FeatureIndex）；'auto' 为 'flat' 的历史别名；'single' 对齐
            原版语义：**强制单索引**（忽略 manifest，多说话人也合并）。
        outside_root: 外部索引目录（默认 assets/indices），构建完成后
            把索引硬链接/复制过去（命名 ``<exp>_added_...npz``）。
        seed: 打乱与聚类随机种子（固定可复现）。
        manifest_path: 可选，多说话人清单路径；缺省自动读
            ``<exp_dir>/multispeaker_manifest.json``（不存在则单索引）。

    Returns:
        单说话人路径：索引文件 str（<exp>/added_IVF{n}_Flat_nprobe_1_
        {exp}_v{version}.ivf.npz 或 .npz，及 total_fea.npy）；
        多说话人路径：``[str, ...]``，每个说话人一个
        ``..._v{version}_spkidN.ivf.npz/.npz``（及 total_fea_spkidN.npy）。

    Raises:
        FileNotFoundError / RuntimeError: 特征缺失或为空（见 build_feature_store）。
    """
    exp_dir = os.path.abspath(exp_dir)
    exp_name = os.path.basename(exp_dir)
    # 'auto'/'single' 为历史别名等价 'flat'；'single' 额外对齐原版语义：
    # 强制单索引（忽略 manifest，多说话人也合并）。
    _flat = mode in ("flat", "auto", "single")
    force_single = mode == "single"
    if not (_flat or mode == "ivf"):
        raise ValueError(
            f"mode 仅支持 'ivf'/'flat'（历史别名 'auto'；'single' 强制单索引），"
            f"收到 {mode!r}"
        )
    _println(
        "[索引训练] exp=%s | version=%s | n_cpu=%s | mode=%s"
        % (exp_name, version, n_cpu, mode),
        exp_dir,
    )

    # 多说话人分组（对齐原版）：manifest 存在且未强制 single 时按 speaker 分组
    speaker_map = {} if force_single else _load_speaker_map(exp_dir, manifest_path)
    if speaker_map:
        feature_dir = _feature_dir(exp_dir, version)
        groups = _group_feature_paths(feature_dir, speaker_map)
        _println(
            "[索引训练] 检测到多说话人 manifest：%d 个说话人分组"
            % len(groups),
            exp_dir,
        )
    else:
        groups = {}

    if groups:
        # 多说话人：每组独立构建索引（spk 排序稳定输出；对齐原版遍历顺序）
        result = []
        for sid in sorted(groups):
            result.append(
                _build_single_index(
                    exp_dir, version, n_cpu, mode, outside_root, seed,
                    exp_name, _flat, groups[sid], speaker_id=sid,
                )
            )
        return result
    # 单说话人 / manifest 缺失 / 无有效分组：保持现有单索引行为（回归不变）
    return _build_single_index(
        exp_dir, version, n_cpu, mode, outside_root, seed,
        exp_name, _flat, None, speaker_id=None,
    )


def _link_outside(
    added_path: str, outside_root: str, exp_name: str, exp_dir: str
) -> str:
    """把索引硬链接（Windows/类 Unix）到外部目录；失败时退化为复制。

    Returns:
        外部目标路径。
    """
    if not outside_root:
        return added_path
    try:
        os.makedirs(outside_root, exist_ok=True)
        target = os.path.abspath(
            os.path.join(outside_root, "%s_%s" % (exp_name, os.path.basename(added_path)))
        )
        source = os.path.abspath(added_path)
        # 同目录则跳过
        if os.path.commonpath([source, os.path.abspath(outside_root)]) == os.path.abspath(
            outside_root
        ):
            return added_path
        if os.path.lexists(target):
            try:
                if os.path.samefile(source, target):
                    return target
            except OSError:
                pass
            os.unlink(target)
        try:
            os.link(source, target)  # 硬链接优先
        except OSError:
            shutil.copy2(source, target)  # 退化：浅拷贝
        _println(
            "[索引训练] 已链接索引到外部目录：%s" % target, exp_dir
        )
        return target
    except Exception as exc:  # noqa: BLE001 —— 外部链接失败不影响主流程
        _println(
            "[索引训练][警告] 无法链接索引到外部目录 %s：%s" % (outside_root, exc),
            exp_dir,
        )
        return added_path


if __name__ == "__main__":
    # python runtime/train/train_index.py <exp_name> <version> <outside_root> <n_cpu> [mode]
    if len(sys.argv) < 2:
        print(
            "用法: python runtime/train/train_index.py"
            " <exp_name> <version> <outside_root> <n_cpu> [mode]"
        )
        sys.exit(1)
    _exp_name = sys.argv[1]
    _version = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    _outside = sys.argv[3] if len(sys.argv) > 3 else "assets/indices"
    _n_cpu = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    _mode = sys.argv[5] if len(sys.argv) > 5 else "ivf"
    # 工作区语义（2026-10-01）：绝对路径（workspaces/<项目>/<任务>/exp）直接使用；
    # 其他（旧实验名）→ 报错，不再映射 logs/<exp>（logs/ 已废弃）
    if not os.path.isabs(_exp_name):
        print("[索引训练][错误] 实验名 %r 不是绝对路径："
              "旧 logs/<exp> 已废弃，请传工作区绝对路径 "
              "workspaces/<项目>/<任务>/exp" % _exp_name)
        sys.exit(1)
    _exp_dir = _exp_name
    try:
        _path = train_index(
            _exp_dir, version=_version, n_cpu=_n_cpu, mode=_mode,
            outside_root=_outside,
        )
        print("[索引训练] 完成：%s" % _path)
    except Exception:
        import traceback

        print("[索引训练][失败] %s" % traceback.format_exc())
        sys.exit(1)
