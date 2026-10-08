# -*- coding: utf-8 -*-
"""runtime/train 自测：preprocess（数据切分）+ extract_f0（F0 特征提取）。

运行方式::

    python runtime/train/tests_train_prep.py

测试内容：
    1. 生成 3 个 5 秒测试 wav（440Hz 正弦+噪声 / 纯正弦 / 带 1s 静音段）；
    2. preprocess_trainset(sr=40000, per=2.0, n_p=2)：
       - 0_gt_wavs 有 wav 产物且可读回；
       - 1_16k_wavs 产物采样率 16k；
       - 音量峰值落在合理区间（归一化后峰值 <= 0.9*0.75 + 0.25*峰值，保守上界 1.35）；
    3. 多说话人 manifest：缺失时报错；有效时产出 ms* 前缀 wav；
    4. extract_feature_dir(exp_dir, f0_method="pm", n_p=2)：
       - 2a_f0 / 2b-f0nsf 生成与 1_16k_wavs 一一对应的 .npy；
       - coarse 值在 [0, 255]、非静音段 > 0；
       - 440Hz 正弦段 f0 中位数 ≈ 441±10 Hz；
    全部通过打印 PASS，否则抛 AssertionError。
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
import soundfile as sf

from runtime.dsp.audio_io import load_audio as io_load, write_audio
from runtime.train.extract_f0 import extract_feature_dir
from runtime.train.preprocess import (
    ManifestError,
    load_manifest,
    preprocess_trainset,
)

SR = 40000
DUR = 5.0
AMP = 0.5


def make_test_wavs(dirpath: str) -> list:
    """生成 3 个 5 秒测试 wav：正弦+噪声 / 纯正弦 / 前后静音+正弦。"""
    t = np.arange(int(SR * DUR)) / SR
    sine = AMP * np.sin(2 * np.pi * 440.0 * t)
    rng = np.random.default_rng(42)
    x1 = (sine + 0.08 * rng.standard_normal(t.size)).astype(np.float32)
    x2 = sine.astype(np.float32)
    x3 = np.zeros_like(x2)
    s, e = int(1.5 * SR), int(4.0 * SR)
    x3[s:e] = AMP * np.sin(2 * np.pi * 440.0 * t[s:e])
    x3 = x3.astype(np.float32)
    paths = []
    for i, x in enumerate((x1, x2, x3)):
        p = os.path.join(dirpath, "test_%d.wav" % i)
        write_audio(p, x, SR, subtype="FLOAT")
        paths.append(p)
    return paths


def _list_wavs(folder: str) -> list:
    if not os.path.isdir(folder):
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    return sorted(n for n in os.listdir(folder) if n.endswith(".wav"))


def check_preprocess(exp_dir: str) -> None:
    """数据切分产物断言。"""
    gt_dir = os.path.join(exp_dir, "0_gt_wavs")
    w16_dir = os.path.join(exp_dir, "1_16k_wavs")
    gt_wavs = _list_wavs(gt_dir)
    w16_wavs = _list_wavs(w16_dir)
    print("  0_gt_wavs=%d 1_16k_wavs=%d" % (len(gt_wavs), len(w16_wavs)))
    assert len(gt_wavs) >= 3, "0_gt_wavs 产物过少（每个输入至少 1 段）"
    assert len(w16_wavs) >= 3, "1_16k_wavs 产物过少"

    # 可读回 + 采样率 16k + 音量归一
    for name in w16_wavs:
        info = sf.info(os.path.join(w16_dir, name))
        assert info.samplerate == 16000, "%s 采样率非 16k: %d" % (
            name,
            info.samplerate,
        )
    peaks = []
    for name in gt_wavs:
        x = io_load(os.path.join(gt_dir, name), sr=None, mono=True)
        assert x.dtype == np.float32 and x.ndim == 1
        peaks.append(float(np.abs(x).max()))
    peak = max(peaks)
    print("  0_gt_wavs 峰值范围: [%.3f, %.3f]" % (min(peaks), peak))
    assert 0.2 < peak <= 1.35, "归一化后峰值越界: %s" % peak


def check_manifest_error(tmp: str) -> None:
    """manifest 缺失时 preprocess_trainset 应抛 ManifestError。"""
    exp = os.path.join(tmp, "exp_nomanifest")
    os.makedirs(exp, exist_ok=True)
    try:
        preprocess_trainset(tmp, SR, 1, exp, per=2.0, manifest_path="1")
        raise AssertionError("缺少 manifest 时未报错")
    except ManifestError as e:
        print("  manifest 缺失报错 OK: %s" % str(e)[:60])


def check_manifest_ok(inp_dir: str, tmp: str) -> None:
    """有效 manifest 应产出 ms* 前缀 wav。"""
    src = os.path.join(inp_dir, "test_0.wav")
    exp = os.path.join(tmp, "exp_ms")
    os.makedirs(exp, exist_ok=True)
    manifest = {
        "entries": [{"path": os.path.abspath(src), "output_key": "ms0001_s000_abc"}]
    }
    with open(os.path.join(exp, "multispeaker_manifest.json"), "w", encoding="utf8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    loaded = load_manifest(exp)
    assert loaded["entries"][0]["output_key"] == "ms0001_s000_abc"
    preprocess_trainset(inp_dir, SR, 2, exp, per=2.0, manifest_path="1")
    ms_gt = [n for n in _list_wavs(os.path.join(exp, "0_gt_wavs")) if n.startswith("ms")]
    ms_16 = [n for n in _list_wavs(os.path.join(exp, "1_16k_wavs")) if n.startswith("ms")]
    print("  ms 产物 0_gt=%d 1_16k=%d" % (len(ms_gt), len(ms_16)))
    assert ms_gt and ms_16, "多说话人 manifest 切分未产出 ms 文件"


def check_extract_f0(exp_dir: str) -> None:
    """f0 提取产物断言。"""
    stats = extract_feature_dir(exp_dir, "pm", n_p=2)
    print("  extract_feature_dir stats: %s" % stats)
    assert stats["failed"] == 0, "f0 提取存在失败"
    w16 = sorted(n for n in os.listdir(os.path.join(exp_dir, "1_16k_wavs")) if n.endswith(".wav"))
    f0a = sorted(os.listdir(os.path.join(exp_dir, "2a_f0")))
    f0b = sorted(os.listdir(os.path.join(exp_dir, "2b-f0nsf")))
    assert len(f0a) == len(w16) and len(f0b) == len(w16), "2a/2b 与 1_16k_wavs 数量不一致"
    for name, a_name, b_name in zip(w16, f0a, f0b):
        assert a_name == name + ".npy" and b_name == name + ".npy", (
            "npy 命名不匹配: %s %s" % (a_name, b_name)
        )
        coarse = np.load(os.path.join(exp_dir, "2a_f0", a_name))
        nsf = np.load(os.path.join(exp_dir, "2b-f0nsf", b_name))
        assert coarse.dtype == np.int32 and nsf.dtype == np.float32
        assert coarse.min() >= 0 and coarse.max() <= 255, "coarse 越界 [0,255]"
        assert (coarse > 0).all(), "非静音段存在 coarse == 0（插值后应为 1..255）"
        assert (nsf > 0).all(), "f0 序列存在 0（插值后应全 > 0）"

    # 正弦段 f0 ≈ 441±10：取任一 npy 的中位数（所有测试输入都含 440Hz 正弦）
    medians = []
    for b_name in f0b:
        nsf = np.load(os.path.join(exp_dir, "2b-f0nsf", b_name))
        pos = nsf[nsf > 0]
        medians.append(float(np.median(pos)))
    med = float(np.median(medians))
    print("  f0 中位数: %.1f Hz" % med)
    assert 431.0 <= med <= 451.0, "440Hz 正弦段 f0 偏差过大: %.1f" % med


def main() -> bool:
    tmp = tempfile.mkdtemp(prefix="rvc_train_prep_test_")
    try:
        inp_dir = os.path.join(tmp, "input")
        os.makedirs(inp_dir, exist_ok=True)
        make_test_wavs(inp_dir)

        exp_dir = os.path.join(tmp, "exp")
        print("[1] preprocess_trainset (sr=%d per=2.0 n_p=2)" % SR)
        preprocess_trainset(inp_dir, SR, 2, exp_dir, per=2.0, noparallel=False)
        check_preprocess(exp_dir)

        print("[2] manifest 缺失报错 + 有效 manifest 切分")
        check_manifest_error(tmp)
        check_manifest_ok(inp_dir, tmp)

        print("[3] extract_feature_dir (pm, n_p=2)")
        check_extract_f0(exp_dir)

        print("PASS")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)