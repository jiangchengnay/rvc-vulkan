# -*- coding: utf-8 -*-
"""多说话人分组索引测试（P2）：``runtime/train/train_index.py`` + ``runtime/vc.py``。

运行方式::

    python runtime/train/tests_index_group.py

测试内容（对齐原版 ``train/train_index.py`` 的多说话人模式）：
1. 含 2 个说话人的 manifest（``entries`` 带 ``speaker_id`` 字段）+ 每说话人
   若干条随机特征（含 ``<output_key>_<idx>.npy`` 切片命名）→ ``train_index``
   → 生成 2 个 ``..._spkid0/1.npz`` 索引，各 ntotal 等于该说话人特征总行数；
   ``total_fea_spkidN.npy`` 落盘。
2. ``vc.find_index_path_for_model`` 按 speaker_id 精确匹配 ``_spkidN`` 后缀
   （含默认 ``ivf`` 模式的 ``.ivf.npz``）；无专属索引时回退单说话人索引
   （原版 fallback 语义）；完全没有匹配时返回空字符串。
3. 单说话人路径回归：无 manifest / manifest 损坏 / ``mode="single"`` →
   单索引（``_spkid`` 后缀不出现、ntotal=全部特征行数、返回值是 str）。

**本模块禁止 import torch / faiss**。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from runtime.ivf_index import IVFIndex
from runtime.retrieval import FeatureIndex
from runtime.train.train_index import train_index
from runtime.vc import find_index_path_for_model


def _make_exp(tmp: str, name: str, dim: int = 8) -> str:
    exp = os.path.join(tmp, name)
    os.makedirs(os.path.join(exp, "3_feature768"), exist_ok=True)
    return exp


def _save_features(exp: str, keys_and_frames, seed: int, dim: int = 8):
    """写一批随机特征文件。keys_and_frames: [(文件名 stem, 帧数), ...]。"""
    fd = os.path.join(exp, "3_feature768")
    rng = np.random.default_rng(seed)
    for stem, frames in keys_and_frames:
        a = (rng.standard_normal((frames, dim)) * 0.5).astype(np.float32)
        np.save(os.path.join(fd, stem + ".npy"), a)


def _write_manifest(exp: str, entries):
    path = os.path.join(exp, "multispeaker_manifest.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump({"entries": entries}, f)
    return path


def _spk_of(path: str) -> int:
    """从索引文件名解析 _spkidN（兼容 .npz 与 .ivf.npz）。"""
    base = os.path.basename(path)
    stem = base[:-4] if base.endswith(".ivf.npz") else os.path.splitext(base)[0]
    tail = stem.rsplit("_spkid", 1)[1]
    if tail.endswith(".ivf"):
        tail = tail[:-4]
    return int(tail)


def check_multi_speaker(tmp: str) -> None:
    """任务核心：2 说话人 manifest + 分组索引 + ntotal + vc 匹配。"""
    print("=== 1. 多说话人分组索引（flat .npz）===")
    exp = _make_exp(tmp, "exp_ms")
    # speaker 0: 2 个原始键 + 1 个切片文件；speaker 1: 1 个原始键
    _save_features(exp, [
        ("ms0000_s000_aaaa", 40),
        ("ms0001_s000_bbbb", 50),
        ("ms0001_s000_bbbb_0", 30),   # 切片：rsplit('_',1) 后匹配 output_key
        ("ms0002_s001_cccc", 60),
    ], seed=10)
    _write_manifest(exp, [
        {"path": "a.wav", "output_key": "ms0000_s000_aaaa", "speaker_id": 0},
        {"path": "b.wav", "output_key": "ms0001_s000_bbbb", "speaker_id": 0},
        {"path": "c.wav", "output_key": "ms0002_s001_cccc", "speaker_id": 1},
    ])
    out_root = os.path.join(tmp, "out_ms")
    res = train_index(exp, version=2, n_cpu=1, mode="flat",
                      outside_root=out_root, seed=42)
    assert isinstance(res, list) and len(res) == 2, res
    by_spk = {_spk_of(p): p for p in res}
    assert set(by_spk) == {0, 1}, by_spk
    for spk, want in ((0, 40 + 50 + 30), (1, 60)):
        idx = FeatureIndex.load(by_spk[spk])
        assert idx.ntotal == want, (spk, idx.ntotal, want)
        assert f"_spkid{spk}.npz" in os.path.basename(by_spk[spk])
        # 总特征文件（对齐原版 total_fea_spkidN.npy）
        tf = os.path.join(exp, f"total_fea_spkid{spk}.npy")
        assert os.path.isfile(tf), tf
        assert np.load(tf).shape[0] == want
    # vc.find_index_path_for_model 按 speaker 匹配
    prev_out, prev_in = os.environ.get("outside_index_root"), os.environ.get("index_root")
    os.environ["outside_index_root"] = out_root
    os.environ["index_root"] = ""
    try:
        p0 = find_index_path_for_model("exp_ms.pth", 0)
        p1 = find_index_path_for_model("exp_ms.pth", 1)
        pn = find_index_path_for_model("exp_ms.pth", None)
    finally:
        if prev_out is None:
            os.environ.pop("outside_index_root", None)
        else:
            os.environ["outside_index_root"] = prev_out
        if prev_in is None:
            os.environ.pop("index_root", None)
        else:
            os.environ["index_root"] = prev_in
    # vc 返回外部链接（<exp>_<added_name>），train_index 返回 exp 内同名文件
    assert os.path.basename(p0) == "exp_ms_" + os.path.basename(by_spk[0]), (p0, by_spk[0])
    assert os.path.basename(p1) == "exp_ms_" + os.path.basename(by_spk[1]), (p1, by_spk[1])
    assert "_spkid" not in pn, pn  # 未指定说话人时不选 spkid 索引
    print(f"  spk0 ntotal={FeatureIndex.load(by_spk[0]).ntotal} "
          f"spk1 ntotal={FeatureIndex.load(by_spk[1]).ntotal} -> vc 匹配 OK")


def check_multi_speaker_ivf(tmp: str) -> None:
    """默认 ivf 模式：_spkidN.ivf.npz + vc 匹配（覆盖 .ivf.npz 的 stem）。"""
    print("=== 2. 多说话人分组索引（ivf .ivf.npz，默认模式）===")
    exp = _make_exp(tmp, "exp_msivf")
    _save_features(exp, [
        ("ms0000_s003_aaaa", 100),
        ("ms0001_s003_bbbb", 100),
        ("ms0002_s004_cccc", 80),
    ], seed=11)
    _write_manifest(exp, [
        {"path": "a.wav", "output_key": "ms0000_s003_aaaa", "speaker_id": 3},
        {"path": "b.wav", "output_key": "ms0001_s003_bbbb", "speaker_id": 3},
        {"path": "c.wav", "output_key": "ms0002_s004_cccc", "speaker_id": 4},
    ])
    out_root = os.path.join(tmp, "out_msivf")
    res = train_index(exp, version=2, n_cpu=1, mode="ivf",
                      outside_root=out_root, seed=42)
    assert isinstance(res, list) and len(res) == 2, res
    by_spk = {_spk_of(p): p for p in res}
    assert set(by_spk) == {3, 4}
    for spk, want in ((3, 200), (4, 80)):
        idx = IVFIndex.load(by_spk[spk])
        assert idx.ntotal == want, (spk, idx.ntotal, want)
        assert f"_spkid{spk}.ivf.npz" in os.path.basename(by_spk[spk])
    prev_out, prev_in = os.environ.get("outside_index_root"), os.environ.get("index_root")
    os.environ["outside_index_root"] = out_root
    os.environ["index_root"] = ""
    try:
        p3 = find_index_path_for_model("exp_msivf.pth", 3)
        p4 = find_index_path_for_model("exp_msivf.pth", 4)
        p5 = find_index_path_for_model("exp_msivf.pth", 5)  # 无专属索引 -> 空
    finally:
        if prev_out is None:
            os.environ.pop("outside_index_root", None)
        else:
            os.environ["outside_index_root"] = prev_out
        if prev_in is None:
            os.environ.pop("index_root", None)
        else:
            os.environ["index_root"] = prev_in
    assert os.path.basename(p3) == "exp_msivf_" + os.path.basename(by_spk[3]), (p3, by_spk[3])
    assert os.path.basename(p4) == "exp_msivf_" + os.path.basename(by_spk[4]), (p4, by_spk[4])
    assert p5 == "", p5
    print("  ivf 模式 _spkidN.ivf.npz 生成与 vc 匹配 OK")


def check_single_regression(tmp: str) -> None:
    """单说话人路径回归：无 manifest / 损坏 manifest / mode='single'。"""
    print("=== 3. 单说话人路径回归 ===")
    exp = _make_exp(tmp, "exp_single")
    _save_features(exp, [("feat_0", 40), ("feat_1", 50), ("feat_2", 60)], seed=12)
    out_root = os.path.join(tmp, "out_single")

    # 无 manifest -> 单索引（返回值 str）
    p = train_index(exp, version=2, n_cpu=1, mode="flat",
                    outside_root=out_root, seed=42)
    assert isinstance(p, str), p
    assert os.path.basename(p).endswith(".npz") and "_spkid" not in os.path.basename(p)
    assert FeatureIndex.load(p).ntotal == 150
    assert os.path.isfile(os.path.join(exp, "total_fea.npy"))

    # 损坏 manifest（JSON 无效）-> 回退单索引
    bad = os.path.join(exp, "multispeaker_manifest.json")
    with open(bad, "w", encoding="utf8") as f:
        f.write("{not json!")
    p2 = train_index(exp, version=2, n_cpu=1, mode="flat",
                     outside_root=out_root, seed=42)
    assert isinstance(p2, str) and FeatureIndex.load(p2).ntotal == 150

    # 合法 manifest 但 output_key 无法解析 speaker_id -> 回退单索引
    _write_manifest(exp, [
        {"path": "a.wav", "output_key": "plain_key_1"},
        {"path": "b.wav", "output_key": "other_2"},
    ])
    p3 = train_index(exp, version=2, n_cpu=1, mode="flat",
                     outside_root=out_root, seed=42)
    assert isinstance(p3, str) and FeatureIndex.load(p3).ntotal == 150

    # mode='single' 强制单索引（即使有合法 manifest 也不分组）
    _write_manifest(exp, [
        {"path": "a.wav", "output_key": "ms0000_s000_aaaa", "speaker_id": 0},
        {"path": "b.wav", "output_key": "ms0001_s001_bbbb", "speaker_id": 1},
    ])
    p4 = train_index(exp, version=2, n_cpu=1, mode="single",
                     outside_root=out_root, seed=42)
    assert isinstance(p4, str) and "_spkid" not in os.path.basename(p4)
    assert FeatureIndex.load(p4).ntotal == 150
    print("  无 manifest / 损坏 / 无法解析 / single 模式 全部回退单索引 OK")


def main() -> None:
    print("=== runtime/train/tests_index_group.py ===")
    tmp = tempfile.mkdtemp(prefix="rvctidx_", dir=_ROOT)
    try:
        check_multi_speaker(tmp)
        check_multi_speaker_ivf(tmp)
        check_single_regression(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
