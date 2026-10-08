# -*- coding: utf-8 -*-
"""RVC 训练 F0 特征提取的纯 numpy runtime 移植（对齐 ``train/dataset/extract_f0.py``）。

FeatureInput 与原版语义对齐：
    - ``pm``：``runtime/dsp/f0.f0_autocorrelation``（对齐 parselmouth
      to_pitch_ac：time_step=10ms、voicing_threshold=0.6、f0 范围 50~1100Hz），
      结果 pad 到 ``p_len = len(x) // hop``；
    - ``rmvpe``：运行时懒加载 ``runtime.models.rmvpe.RMVPE``，调用
      ``infer_from_audio(x, thred=0.03)``（import 失败时抛出清晰提示）；
    - ``fcpe``：抛 ``NotImplementedError``（FCPE 后续实现）；
    - 后处理：uv 帧 np.interp 线性插值 -> f0 *= 2^(f0_up_key/12) ->
      f0bak 备份 -> mel 量化 coarse（clamp [1, 255]）-> 返回 (f0_coarse, f0bak)。

``extract_feature_dir`` 扫描 ``<exp_dir>/1_16k_wavs/*.wav``，多进程为每个文件
提取并落盘：
    - ``<exp_dir>/2a_f0/<name>.wav.npy``：coarse int32
    - ``<exp_dir>/2b-f0nsf/<name>.wav.npy``：连续 f0 float32

零 torch / librosa / parselmouth 依赖。

命令行用法::

    python runtime/train/extract_f0.py <exp_dir> <f0_method:pm|rmvpe> <n_p> [version]
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

import numpy as np

try:  # 作为包的一部分被 import（推荐）
    from ..audio import load_audio
    from ..dsp.f0 import f0_autocorrelation
except ImportError:  # 以脚本方式直接运行
    _ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from runtime.audio import load_audio
    from runtime.dsp.f0 import f0_autocorrelation

__all__ = ["FeatureInput", "extract_feature_dir", "compute_f0_file"]

# ---------------------------------------------------------------------------
# 日志（简化版）
# ---------------------------------------------------------------------------

_LOG_FILE = None

_LOG_NAME = "extract_f0_feature.log"


def _ensure_log(exp_dir: str) -> None:
    global _LOG_FILE
    if _LOG_FILE is None:
        os.makedirs(exp_dir, exist_ok=True)
        _LOG_FILE = open(os.path.join(exp_dir, _LOG_NAME), "a", encoding="utf8")


def printt(msg) -> None:
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write("%s\n" % msg)
        _LOG_FILE.flush()


def _should_report(index: int, total: int, max_updates: int) -> bool:
    """进度上报节流（对齐 tools/progress.should_report）。"""
    if total <= 0:
        return False
    if total <= max_updates:
        return True
    import math

    interval = max(1, math.ceil(total / max_updates))
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def _import_rmvpe():
    """懒加载 runtime.models.rmvpe（把项目根加入 sys.path 后重试一次）。

    Raises:
        RuntimeError: import 失败时给出清晰提示（不阻塞其他 f0 方法）。
    """
    try:
        return importlib.import_module("runtime.models.rmvpe")
    except ImportError:
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            return importlib.import_module("runtime.models.rmvpe")
        except Exception as exc:
            raise RuntimeError(
                "无法导入 runtime.models.rmvpe（RMVPE 模型模块不可用）。"
                "请确认 runtime/models/rmvpe.py 已移植完成、torch_compat 可导入，"
                "并确保工作目录为项目根（assets/rmvpe/rmvpe.pt 存在）。\n"
                "原始错误：%s" % exc
            ) from exc


# ---------------------------------------------------------------------------
# F0 特征
# ---------------------------------------------------------------------------


class FeatureInput:
    """F0 特征提取器（与原版 FeatureInput 语义一致）。

    Args:
        sr: 输入采样率（默认 16000，与 1_16k_wavs 对齐）。
        hop: f0 帧移（样本，默认 160 = 10ms @16k）。
        f0_bin / f0_max / f0_min: coarse 量化 bin 数与频率范围。
    """

    def __init__(
        self,
        sr: int = 16000,
        hop: int = 160,
        f0_bin: int = 256,
        f0_max: float = 1100.0,
        f0_min: float = 50.0,
    ):
        self.fs = int(sr)
        self.hop = int(hop)
        self.f0_bin = int(f0_bin)
        self.f0_max = float(f0_max)
        self.f0_min = float(f0_min)
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.model_rmvpe = None  # 懒加载缓存

    # ------------------------------------------------------------------ 提取
    def get_f0(
        self,
        x: np.ndarray,
        f0_method: str,
        f0_up_key: float = 0.0,
        inp_f0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """从音频信号提取 f0。

        Args:
            x: 1D float32/float64 音频（采样率须为 self.fs）。
            f0_method: "pm"（自相关）或 "rmvpe"（模型）；"fcpe" 未实现。
            f0_up_key: 音高位移（半音），f0 *= 2^(f0_up_key/12)。
            inp_f0: 可选外部 f0 曲线（Hz）；提供时覆盖模型提取结果，
                长度不匹配则线性插值到 p_len。

        Returns:
            (f0_coarse, f0bak)：
                - f0_coarse: int32，mel 量化到 [1, 255]；
                - f0bak: float32，位移后的连续 f0（2b-f0nsf 落盘值）。

        Raises:
            ValueError: 不支持的 f0_method，或整段音高全为 0（无意义）。
            NotImplementedError: f0_method == "fcpe"。
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("get_f0 需要 1D 音频输入，实际 %dD" % x.ndim)
        p_len = x.shape[0] // self.hop

        if f0_method == "pm":
            f0 = f0_autocorrelation(
                x,
                self.fs,
                hop=self.hop,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                voicing_threshold=0.6,
                time_step=0.01,  # 10ms，对齐 parselmouth time_step=160/16000*1000
            )
            # 对齐原版：把 parselmouth 帧数 pad 到 p_len
            f0 = np.asarray(f0, dtype=np.float64)
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "rmvpe":
            f0 = self._rmvpe_infer(x)
        elif f0_method == "fcpe":
            raise NotImplementedError(
                "FCPE 音高提取尚未实现（runtime 后续版本支持，请改用 pm/rmvpe）"
            )
        else:
            raise ValueError(
                "仅支持 pm 和 rmvpe 音高提取算法，收到：%r" % f0_method
            )

        f0 = np.asarray(f0, dtype=np.float64)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        if inp_f0 is not None:
            f0 = self._apply_inp_f0(f0, p_len, inp_f0)

        uv = f0 == 0
        if uv.all():
            raise ValueError("音高全部为0，该音频无意义")
        if uv.any():
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

        f0 = f0 * (2 ** (f0_up_key / 12.0))
        f0bak = f0.copy()
        f0_coarse = self.coarse_f0(f0)
        return f0_coarse, f0bak

    def _apply_inp_f0(
        self, f0: np.ndarray, p_len: int, inp_f0: np.ndarray
    ) -> np.ndarray:
        """用外部 f0 曲线覆盖提取结果（长度不匹配时线性插值到 p_len）。"""
        inp = np.asarray(inp_f0, dtype=np.float64).ravel()
        if inp.size == f0.size and inp.shape == f0.shape:
            return np.where(inp > 0, inp, f0)
        if inp.size == p_len:
            return np.where(inp > 0, inp, f0)
        if inp.size <= 1:
            return f0
        interp = np.interp(
            np.linspace(0.0, inp.size - 1, f0.size), np.arange(inp.size), inp
        )
        return np.where(interp > 0, interp, f0)

    def _rmvpe_infer(self, x: np.ndarray) -> np.ndarray:
        """RMVPE 模型推理（懒加载，模型对象在实例内缓存）。"""
        if self.model_rmvpe is None:
            mod = _import_rmvpe()
            if not hasattr(mod, "load_rmvpe"):
                raise RuntimeError(
                    "runtime.models.rmvpe 缺少 load_rmvpe() 入口（可能仍在移植中）"
                )
            printt("[F0提取] 正在加载 RMVPE 模型")
            self.model_rmvpe = mod.load_rmvpe()
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)

    # ------------------------------------------------------------ 后处理量化
    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """mel 刻度量化 f0（对齐原版 FeatureInput.coarse_f0，clamp [1, 255]）。"""
        f0 = np.asarray(f0, dtype=np.float64)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    # ------------------------------------------------------------- 原版兼容
    def compute_f0(self, path: str, f0_method: str) -> np.ndarray | None:
        """原版 API：按文件提取 f0（返回插值后的连续 f0；全 0 返回 None）。

        内部等价于：load_audio(path, 16000) -> get_f0 -> f0bak。
        """
        if f0_method not in ("pm", "rmvpe"):
            raise ValueError("仅支持 pm 和 rmvpe 音高提取算法")
        x = load_audio(path, self.fs)
        try:
            _, f0bak = self.get_f0(x, f0_method)
        except ValueError:
            return None
        return f0bak


# ---------------------------------------------------------------------------
# 批量提取（多进程）
# ---------------------------------------------------------------------------


def _extract_one(fi: FeatureInput, inp_path: str, opt1: str, opt2: str, f0_method: str):
    """提取并落盘单个文件（频率点经 np.save，allow_pickle=False）。"""
    x = load_audio(inp_path, fi.fs)
    f0_coarse, f0_nsf = fi.get_f0(x, f0_method)
    if f0_nsf.dtype != np.float32:
        f0_nsf = f0_nsf.astype(np.float32)
    if f0_coarse.dtype != np.int32:
        f0_coarse = f0_coarse.astype(np.int32)
    np.save(opt2, f0_nsf, allow_pickle=False)  # 2b-f0nsf 连续值
    np.save(opt1, f0_coarse, allow_pickle=False)  # 2a_f0 coarse


def _go(
    fi: FeatureInput, paths: list, f0_method: str, max_updates: int
) -> dict:
    """处理一块路径列表（供单进程与子进程共用），返回统计。"""
    success = 0
    skipped = 0
    failed = 0
    if len(paths) == 0:
        printt("[F0提取] 无待处理音频，已全部跳过")
    else:
        printt("[F0提取] 待处理：%s" % len(paths))
        for idx, (inp_path, opt1, opt2) in enumerate(paths):
            try:
                if os.path.exists(opt1) and os.path.exists(opt2):
                    skipped += 1
                    continue
                _extract_one(fi, inp_path, opt1, opt2, f0_method)
                success += 1
                if _should_report(idx, len(paths), max_updates):
                    printt(
                        "[F0提取] 进度：%s/%s | 成功：%s | 跳过：%s | %s"
                        % (
                            idx + 1,
                            len(paths),
                            success,
                            skipped,
                            os.path.basename(inp_path),
                        )
                    )
            except ValueError as ve:
                # 音高全部为 0，该音频无意义
                skipped += 1
                printt("音高全部为0，该音频无意义，跳过：%s（%s）" % (inp_path, ve))
            except Exception:
                failed += 1
                printt(
                    "[F0提取][失败] %s\n%s" % (inp_path, traceback.format_exc())
                )
        printt(
            "[F0提取] 完成 | 成功：%s | 跳过：%s | 失败：%s"
            % (success, skipped, failed)
        )
    return {"success": success, "skipped": skipped, "failed": failed}


def _worker_go(paths: list, f0_method: str, max_updates: int, exp_dir: str, queue):
    """multiprocessing 子进程入口（spawn 安全：独立 FeatureInput 实例）。"""
    _ensure_log(exp_dir)
    try:
        fi = FeatureInput()
    except Exception as exc:  # 构造失败（如 rmvpe 权重缺失）按失败上报
        printt("[F0提取][失败] FeatureInput 初始化失败：%s" % exc)
        queue.put({"success": 0, "skipped": 0, "failed": len(paths)})
        return
    try:
        stats = _go(fi, paths, f0_method, max_updates)
    except Exception as exc:  # 保险路径：异常按文件重新统计落盘结果
        printt("[F0提取][失败] 子进程 _go 异常：%s" % exc)
        uniq = {(p1, p2) for _, p1, p2 in paths}
        ok = sum(1 for p1, p2 in uniq if os.path.exists(p1) and os.path.exists(p2))
        stats = {"success": ok, "skipped": 0, "failed": len(uniq) - ok}
    queue.put(stats)


def extract_feature_dir(
    exp_dir: str, f0_method: str = "pm", n_p: int = 1, version: int = 2
) -> dict:
    """扫描 ``<exp_dir>/1_16k_wavs/*.wav`` 批量提取 f0，落盘 2a_f0 / 2b-f0nsf。

    Args:
        exp_dir: 实验目录（logs/<exp>，须已由 preprocess 生成 1_16k_wavs）。
        f0_method: "pm" 或 "rmvpe"。
        n_p: 并行进程数（>1 时多进程）。
        version: 兼容参数（RVC v1/v2 目录约定一致，均输出 2a_f0/2b-f0nsf）。

    Returns:
        {"success": int, "skipped": int, "failed": int} 汇总统计。
    """
    inp_root = os.path.join(exp_dir, "1_16k_wavs")
    opt_root1 = os.path.join(exp_dir, "2a_f0")
    opt_root2 = os.path.join(exp_dir, "2b-f0nsf")
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    if not os.path.isdir(inp_root):
        raise RuntimeError(
            "找不到 1_16k_wavs 目录：%s（请先运行 preprocess 数据切分）" % inp_root
        )

    paths = []
    for name in sorted(os.listdir(inp_root)):
        inp_path = os.path.join(inp_root, name)
        if not os.path.isfile(inp_path):
            continue
        if "spec" in inp_path:
            continue
        opt1 = os.path.join(opt_root1, name + ".npy")
        opt2 = os.path.join(opt_root2, name + ".npy")
        if os.path.exists(opt1) and os.path.exists(opt2):
            continue  # 已存在则跳过（断点续跑）
        paths.append([inp_path, opt1, opt2])

    if not paths:
        printt("[F0提取] 无待处理音频，已全部跳过")
        return {"success": 0, "skipped": 0, "failed": 0}

    worker_count = min(max(1, int(n_p)), len(paths))
    max_updates = max(1, (12 + worker_count - 1) // worker_count)

    if worker_count <= 1:
        _ensure_log(exp_dir)
        return _go(FeatureInput(), paths, f0_method, max_updates)

    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    ps = []
    for i in range(worker_count):
        p = ctx.Process(
            target=_worker_go,
            args=(paths[i::worker_count], f0_method, max_updates, exp_dir, queue),
        )
        ps.append(p)
        p.start()
    stats = {"success": 0, "skipped": 0, "failed": 0}
    for _ in ps:
        part = queue.get(timeout=3600)
        for key in stats:
            stats[key] += part.get(key, 0)
    for p in ps:
        p.join()
    return stats


def compute_f0_file(path: str, f0_method: str = "pm", sr: int = 16000) -> np.ndarray:
    """便捷函数：单个音频文件的连续 f0（Hz），全 0 时抛出 ValueError。"""
    fi = FeatureInput(sr=sr, hop=160)
    x = load_audio(path, fi.fs)
    _, f0bak = fi.get_f0(x, f0_method)
    return f0bak


if __name__ == "__main__":
    # python runtime/train/extract_f0.py <exp_dir> <f0_method> <n_p> [version]
    if len(sys.argv) < 2:
        printt(
            "用法: python runtime/train/extract_f0.py"
            " <exp_dir> <f0_method:pm|rmvpe> <n_p> [version]"
        )
        sys.exit(1)
    exp_dir_arg = sys.argv[1]
    f0_method_arg = sys.argv[2].lower() if len(sys.argv) > 2 else "pm"
    n_p_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    version_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    _ensure_log(exp_dir_arg)
    stats = extract_feature_dir(exp_dir_arg, f0_method_arg, n_p_arg, version_arg)
    if stats["failed"]:
        sys.exit(2)