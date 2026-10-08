# -*- coding: utf-8 -*-
"""HuBERT 训练特征提取（补全训练管线：1_16k_wavs → 3_feature256|768）。

对 ``logs/<exp>/1_16k_wavs/*.wav`` 逐个用 runtime HuBERT 编码器提取特征，
按版本写入 ``3_feature256``（v1，256 维）或 ``3_feature768``（v2，768 维），
每文件一个 ``<name>.wav.npy``（帧 × D，与 RVC DataLoader 约定一致）。

零 torch/transformers 依赖；hubert 模型经 ``runtime.models.load_hubert_model``
懒加载（与推理管线共享单例，不重复加载）；支持多进程并行。

用法::

    python -m runtime.train.extract_hubert <exp_dir> <version> [n_p]
"""

from __future__ import annotations

import multiprocessing
import os
import sys

import numpy as np

try:  # 包方式导入（推荐）
    from ..dsp.audio_io import load_audio
    from ..models import load_hubert_model
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except ImportError:  # 脚本方式
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from runtime.dsp.audio_io import load_audio  # noqa: E402
    from runtime.models import load_hubert_model  # noqa: E402

_ASSETS = os.path.join(_ROOT, "assets")


def _hubert_feature_path(exp_dir: str, name: str, version: int) -> str:
    feat_dir = "3_feature256" if version == 1 else "3_feature768"
    d = os.path.join(exp_dir, feat_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ".wav.npy")


def _extract_one(job):
    """单文件提取（子进程入口）。返回 (成功, 文件名, 帧数)。"""
    wav_path, out_path, version = job
    try:
        x = load_audio(wav_path, 16000, mono=True)          # [T] f32
        enc = load_hubert_model(os.path.join(_ASSETS, "hubert_base"))
        feats = enc.encode(x[None, :], version=version)      # [1, L, D]
        arr = feats[0].astype(np.float32)                    # [L, D]
        np.save(out_path, arr)
        return True, os.path.basename(wav_path), arr.shape[0]
    except Exception as exc:  # noqa: BLE001
        print("  [失败] %s: %s" % (wav_path, exc), file=sys.stderr)
        return False, os.path.basename(wav_path), -1


def extract_hubert_dir(exp_dir: str, version: int = 2, n_p: int = 1,
                       verbose: bool = True) -> dict:
    """对 logs/<exp>/1_16k_wavs/*.wav 提取 hubert 特征。

    Returns: {"total", "ok", "failed", "frames", "version"}
    """
    src_dir = os.path.join(exp_dir, "1_16k_wavs")
    if not os.path.isdir(src_dir):
        raise RuntimeError("1_16k_wavs 目录不存在：%s（请先运行数据预处理）" % src_dir)
    names = sorted(n for n in os.listdir(src_dir) if n.lower().endswith(".wav"))
    if not names:
        raise RuntimeError("1_16k_wavs 为空：%s" % src_dir)
    jobs = [
        (os.path.join(src_dir, n),
         _hubert_feature_path(exp_dir, n[:-4], version), version)
        for n in names
    ]
    ok = failed = total_frames = 0
    if n_p <= 1 or len(jobs) == 1:
        results = [_extract_one(j) for j in jobs]
    else:
        with multiprocessing.Pool(min(n_p, len(jobs))) as pool:
            results = pool.map(_extract_one, jobs)
    for success, name, frames in results:
        if success:
            ok += 1
            total_frames += frames
            if verbose:
                print("  [OK] %s (%d 帧)" % (name, frames))
        else:
            failed += 1
            if verbose:
                print("  [FAIL] %s" % name)
    stats = {"total": len(jobs), "ok": ok, "failed": failed,
             "frames": total_frames, "version": version}
    if verbose:
        print("hubert 特征提取完成：%s" % stats)
    return stats


def _cli():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    exp_dir = sys.argv[1]
    version = int(sys.argv[2])
    n_p = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    try:
        print(extract_hubert_dir(exp_dir, version, n_p))
        return 0
    except Exception as exc:  # noqa: BLE001
        print("提取失败：%s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_cli())