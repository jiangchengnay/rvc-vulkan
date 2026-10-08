# -*- coding: utf-8 -*-
"""RVC 训练数据切分预处理（纯 numpy runtime 移植，对齐 ``train/preprocess.py``）。

功能：
    1. 对输入目录（或多说话人 manifest）中的每个音频文件：
       加载 -> 高通滤波（scipy.signal.butter + lfilter，与 RVC 一致，
       不用 filtfilt，避免预振铃）-> Slicer 静音切分 -> 二次切段；
    2. 按 ``norm_write`` 归一化写入 ``<exp_dir>/0_gt_wavs``（原采样率）
       与 ``<exp_dir>/1_16k_wavs``（16kHz 重采样，供后续 f0/特征提取）。

零 torch / librosa / parselmouth 依赖；仅 numpy + scipy.signal
（+ soundfile 可选，经 runtime/dsp 封装）。

命令行用法（与原版一致）::

    python runtime/train/preprocess.py <inp_root> <sr> <n_p> <exp_dir> <True/False> <per> [manifest]
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import traceback

import numpy as np
from scipy import signal

try:  # 作为包的一部分被 import（推荐）
    from ..audio import load_audio
    from ..dsp.audio_io import write_audio
    from ..dsp.resample import resample
    from ..dsp.slicer import Slicer
except ImportError:  # 以脚本方式直接运行（python runtime/train/preprocess.py）
    _ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from runtime.audio import load_audio
    from runtime.dsp.audio_io import write_audio
    from runtime.dsp.resample import resample
    from runtime.dsp.slicer import Slicer

__all__ = ["PreProcess", "preprocess_trainset", "load_manifest", "ManifestError"]

# ---------------------------------------------------------------------------
# 日志（简化版，替代原版 i18n + tools.progress；不引入 i18n 依赖）
# ---------------------------------------------------------------------------

_LOG_FILE = None  # 文件句柄缓存（按进程）

_LOG_NAME = "preprocess.log"


def _ensure_log(exp_dir: str) -> None:
    """打开（追加模式）日志文件，只打开一次。"""
    global _LOG_FILE
    if _LOG_FILE is None:
        os.makedirs(exp_dir, exist_ok=True)
        _LOG_FILE = open(
            os.path.join(exp_dir, _LOG_NAME), "a", encoding="utf8"
        )


def println(msg) -> None:
    """打印到 stdout 并追加写入日志文件。"""
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write("%s\n" % msg)
        _LOG_FILE.flush()


def _should_report(index: int, total: int, max_updates: int = 12) -> bool:
    """进度上报节流（对齐 tools/progress.should_report）。"""
    if total <= 0:
        return False
    if total <= max_updates:
        return True
    import math

    interval = max(1, math.ceil(total / max_updates))
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


# ---------------------------------------------------------------------------
# 多说话人 manifest（简化版，对齐 tools/multispeaker.load_manifest 的读取侧）
# ---------------------------------------------------------------------------


class ManifestError(RuntimeError):
    """多说话人 manifest 读取出错。"""


def load_manifest(exp_dir: str) -> dict:
    """读取 ``<exp_dir>/multispeaker_manifest.json``。

    Args:
        exp_dir: 实验目录（训练日志根）。

    Returns:
        ``{"entries": [{"path": str, "output_key": str}, ...]}``。

    Raises:
        ManifestError: 文件不存在、JSON 损坏或没有有效 entries。
    """
    path = os.path.join(exp_dir, "multispeaker_manifest.json")
    if not os.path.isfile(path):
        raise ManifestError(
            "多说话人训练集清单不存在：%s"
            "（请先在多说话人辅助页面生成 multispeaker_manifest.json）" % path
        )
    with open(path, "r", encoding="utf8") as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                "多说话人训练集清单 JSON 解析失败：%s | %s" % (path, exc)
            )
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ManifestError("多说话人训练集清单没有有效音频条目：%s" % path)
    out = []
    for entry in entries:
        try:
            out.append(
                {"path": str(entry["path"]), "output_key": str(entry["output_key"])}
            )
        except (KeyError, TypeError, ValueError):
            raise ManifestError(
                "多说话人训练集清单条目缺少 path/output_key：%s | %r" % (path, entry)
            )
    return {"entries": out}


# ---------------------------------------------------------------------------
# 数据切分
# ---------------------------------------------------------------------------


class PreProcess:
    """数据切分器（参数与 RVC train/preprocess.PreProcess 完全一致）。

    Args:
        sr: 输入音频统一重采样到的采样率（Hz）。
        exp_dir: 输出实验目录（内含 0_gt_wavs / 1_16k_wavs）。
        per: 每段秒数（默认 3.7，与原版一致）。
    """

    def __init__(self, sr: int, exp_dir: str, per: float = 3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        # 带通/高通滤波：butter(5, 48, 'high')，与 RVC 一致
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
        _ensure_log(exp_dir)

    def norm_write(self, tmp_audio: np.ndarray, output_key: str, idx1: int) -> bool:
        """归一化并写入一段音频（0_gt_wavs 原采样率 + 1_16k_wavs 16kHz）。

        与原版一致：
            - 峰值非法（非有限 / <=0 / >2.5）跳过；
            - 归一化 ``(x/max*0.9*0.75) + 0.25*x``；
            - 0_gt_wavs 写 float32 原始采样率，1_16k_wavs 重采样到 16k。
        """
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
            println(
                "[数据切分][跳过] 无效或异常音频片段：%s_%s | 峰值：%s"
                % (output_key, idx1, tmp_max)
            )
            return False
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        gt_path = os.path.join(
            self.gt_wavs_dir, "%s_%s.wav" % (output_key, idx1)
        )
        write_audio(gt_path, tmp_audio.astype(np.float32), self.sr)
        audio_16k = resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000, method="fft"
        ).astype(np.float32)
        w16_path = os.path.join(
            self.wavs16k_dir, "%s_%s.wav" % (output_key, idx1)
        )
        write_audio(w16_path, audio_16k, 16000)
        return True

    def pipeline(
        self, path: str, output_key: str, progress_index: int, total: int
    ) -> bool:
        """处理单个音频文件：加载 -> 高通滤波 -> 静音切分 -> 二次切段。

        Args:
            path: 输入音频路径。
            output_key: 输出文件名前缀（单说话人为数字编号，多说话人为 ms 键）。
            progress_index / total: 进度信息。

        Returns:
            True 成功 / False 失败（失败已记日志）。
        """
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, output_key, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
            self.norm_write(tmp_audio, output_key, idx1)
            if _should_report(progress_index, total):
                println(
                    "[数据切分] 进度：%s/%s | %s"
                    % (progress_index + 1, total, os.path.basename(path))
                )
            return True
        except Exception:
            println(
                "[数据切分][失败] %s\n%s" % (path, traceback.format_exc())
            )
            return False

    def pipeline_mp(self, infos) -> None:
        """处理一批 (path, output_key, progress_index, total)（供子进程执行）。"""
        _ensure_log(self.exp_dir)
        success = 0
        failed = 0
        for path, output_key, progress_index, total in infos:
            if self.pipeline(path, output_key, progress_index, total):
                success += 1
            else:
                failed += 1
        if infos:
            println(
                "[数据切分] 子任务完成 | 成功：%s | 失败：%s" % (success, failed)
            )

    def _cleanup_ms(self) -> None:
        """清理上一轮多说话人（ms*）切片产物，避免残留旧 speaker 数据。"""
        for output_dir in (self.gt_wavs_dir, self.wavs16k_dir):
            for name in os.listdir(output_dir):
                if name.startswith("ms") and name.endswith(".wav"):
                    try:
                        os.remove(os.path.join(output_dir, name))
                    except OSError:
                        pass

    def pipeline_mp_inp_dir(self, inp_root: str, n_p: int, noparallel: bool) -> None:
        """扫描输入目录所有文件，按文件数分片并行（或串行）切分。"""
        try:
            names = sorted(
                name
                for name in os.listdir(inp_root)
                if os.path.isfile(os.path.join(inp_root, name))
            )
            total = len(names)
            infos = [
                (os.path.join(inp_root, name), str(idx), idx, total)
                for idx, name in enumerate(names)
            ]
            worker_count = max(n_p, 1)
            worker_count = min(worker_count, max(total, 1))
            println(
                "[数据切分] 待处理：%s | 进程数：%s" % (total, worker_count)
            )
            if noparallel:
                for i in range(worker_count):
                    self.pipeline_mp(infos[i::worker_count])
            else:
                ps = []
                for i in range(worker_count):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::worker_count],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(worker_count):
                    ps[i].join()
        except Exception:
            println("[数据切分][失败] %s" % traceback.format_exc())

    def pipeline_mp_manifest(
        self, manifest_entries: list, n_p: int, noparallel: bool
    ) -> None:
        """按多说话人 manifest entries 并行切分。"""
        infos = [
            (entry["path"], entry["output_key"], idx, len(manifest_entries))
            for idx, entry in enumerate(manifest_entries)
        ]
        total = len(infos)
        worker_count = max(n_p, 1)
        worker_count = min(worker_count, max(total, 1))
        println("[数据切分] 多说话人待处理：%s | 进程数：%s" % (total, worker_count))
        if noparallel:
            for i in range(worker_count):
                self.pipeline_mp(infos[i::worker_count])
            return
        ps = []
        for i in range(worker_count):
            p = multiprocessing.Process(
                target=self.pipeline_mp, args=(infos[i::worker_count],)
            )
            ps.append(p)
            p.start()
        for p in ps:
            p.join()


def preprocess_trainset(
    inp_root: str,
    sr: int,
    n_p: int | None = None,
    exp_dir: str = "",
    per: float = 3.7,
    noparallel: bool = False,
    manifest_path: str = "",
):
    """数据切分主入口（对应原版 ``train/preprocess.py`` 的命令行执行）。

    Args:
        inp_root: 输入音频目录（manifest_path 为空时使用）。
        sr: 重采样目标采样率（如 40000 / 48000）。
        n_p: 并行进程数；None 时按内存/CPU 自适应（P1-3）。
        exp_dir: 输出实验目录（logs/<exp>）。
        per: 每段秒数。
        noparallel: True 时串行处理。
        manifest_path: 非空时从 ``load_manifest(exp_dir)`` 读多说话人清单
            切分（manifest 文件固定放 <exp_dir>/multispeaker_manifest.json）。

    Raises:
        ManifestError: manifest_path 非空但清单缺失/无效。
    """
    if n_p is None:
        try:
            from runtime import memory as _mem  # noqa: PLC0415
            n_p = _mem.suggest_preprocess_workers()
            println("[数据切分] 内存/CPU 自适应并行数=%d" % n_p)
        except Exception:  # noqa: BLE001
            n_p = 2
    pp = PreProcess(sr, exp_dir, per)
    println("[数据切分] 开始")
    pp._cleanup_ms()
    if manifest_path:
        try:
            manifest = load_manifest(exp_dir)
            pp.pipeline_mp_manifest(manifest["entries"], n_p, noparallel)
        except ManifestError as error:
            println("[数据切分][失败] %s" % error)
            raise
    else:
        pp.pipeline_mp_inp_dir(inp_root, n_p, noparallel)
    println("[数据切分] 完成")


if __name__ == "__main__":
    # python runtime/train/preprocess.py <inp_root> <sr> <n_p> <exp_dir> <True/False> <per> [manifest]
    if len(sys.argv) < 7:
        println(
            "用法: python runtime/train/preprocess.py"
            " <inp_root> <sr> <n_p> <exp_dir> <noparallel:True/False> <per> [manifest]"
        )
        sys.exit(1)
    inp_root_arg = sys.argv[1]
    sr_arg = int(sys.argv[2])
    n_p_arg = int(sys.argv[3])
    exp_dir_arg = sys.argv[4]
    noparallel_arg = sys.argv[5] == "True"
    per_arg = float(sys.argv[6])
    manifest_arg = sys.argv[7] if len(sys.argv) > 7 else ""
    preprocess_trainset(
        inp_root_arg,
        sr_arg,
        n_p_arg,
        exp_dir_arg,
        per=per_arg,
        noparallel=noparallel_arg,
        manifest_path=manifest_arg,
    )