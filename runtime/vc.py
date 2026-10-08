# -*- coding: utf-8 -*-
"""RVC 推理高层封装（T35）：``VC`` 类 + 索引查找工具。

对齐 ``infer/vc/modules.py`` 的 ``VC`` 类（87-427 行）语义，但**零 torch /
faiss / librosa / parselmouth 依赖**：模型权重用 ``torch_compat.load_pth``
读取，推理全部走 ``runtime``（hubert / rmvpe / vits / pipeline）。

用法::

    from runtime.native_config import Config
    from runtime.vc import VC

    vc = VC(Config())
    info = vc.get_vc("my_model")            # 模型名 -> assets/weights/my_model.pth
    status, (sr, audio_int16) = vc.vc_single(
        0, "in.wav", 0, "rmvpe", None, 0.0, 48000, 1.0, 0.33,
    )
    # audio_int16 为 numpy int16；写文件需先 /32768.0 转 float（见 runtime/cli.py）

模型 checkpoint 兼容两种形态（任务要求）:
    - 推理格式: ``{"weight": state_dict, "config": [...], "f0": 1, "version": "v2"}``
    - 训练底模: ``{"model": state_dict, "iteration": ..., "learning_rate": ...}``
      （无 ``config``：tgt_sr 从权重反推，version 从 ``emb_phone`` 维数推断）

路径约定（与 RVC 原版一致的环境变量，未设置时使用项目默认）:
    - ``weight_root``      -> assets/weights
    - ``index_root``       -> logs（自动索引扫描）
    - ``outside_index_root`` -> assets/indices（自动索引扫描）
    - hubert: assets/hubert_base；rmvpe: assets/rmvpe/rmvpe.pt
"""

from __future__ import annotations

import os
import re
import traceback
from pathlib import Path

import numpy as np

from torch_compat import load_pth  # 纯 Python 读 .pth，不依赖 torch

from runtime.audio import load_audio
from runtime.models.hubert import load_hubert_model
from runtime.models.rmvpe import load_rmvpe
from runtime.models.vits import VitsConfig, load_synthesizer
from runtime.native_config import Config
from runtime.pipeline import Pipeline

__all__ = ["VC", "find_index_path_for_model", "normalized_speaker_info"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # projects/rvc-vulkan
ASSETS = PROJECT_ROOT / "assets"

# 音频扩展名（CLI 目录扫描用）
AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus",
    ".aac", ".wma", ".mp4", ".mkv", ".webm",
}

# 索引文件名中的说话人 id 后缀模式（如 ..._spkid0.npz / ..._spkid0.ivf.npz）。
# 本仓库 ivf 模式索引以 ".ivf.npz" 结尾，stem 形如 "..._spkid0.ivf"，
# 因此允许 _spkidN 后跟可选的 ".ivf" 段。
_SPKID_RE = re.compile(r"_spkid(\d+)(?:\.ivf)?$", re.IGNORECASE)
# 训练迭代后缀（模型文件名里的 _e\d+_s\d+$ 与索引无关，去掉以匹配实验名）
_EXPERIMENT_SUFFIX_RE = re.compile(r"_e\d+_s\d+$", re.IGNORECASE)


def normalized_speaker_info(checkpoint, n_spk):
    """从 checkpoint 的 ``speaker_info`` 字段提取合法说话人列表（对齐原版）。"""
    speaker_info = []
    seen = set()
    for item in checkpoint.get("speaker_info", []):
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            speaker_id < 0
            or speaker_id >= int(n_spk)
            or not speaker_name
            or speaker_id in seen
        ):
            continue
        seen.add(speaker_id)
        speaker_info.append({"id": speaker_id, "name": speaker_name})
    speaker_info.sort(key=lambda item: item["id"])
    return speaker_info


def _weight_roots() -> list:
    """权重根目录候选（models/ 优先，assets/weights 兼容回退）。

    环境变量 ``weight_root`` 显式设置时只使用它；否则按
    ``<项目根>/models`` → ``assets/weights`` 顺序返回候选。
    """
    env = os.environ.get("weight_root")
    if env:
        return [env]
    return [str(PROJECT_ROOT / "models"), str(ASSETS / "weights")]


def _weight_root() -> str:
    """主权重根目录：models/（存在时），否则 assets/weights（向后兼容）。"""
    for root in _weight_roots():
        if os.path.isdir(root):
            return root
    return _weight_roots()[0]


def find_index_path_for_model(model_name: str, speaker_id=None) -> str:
    """在 ``outside_index_root`` / ``index_root`` 下为模型寻找匹配的索引。

    兼容三种索引文件：.npz（自建暴力）、.ivf.npz（倒排）、.index/.faiss
    （faiss 原版，05 §3.5 契约后缀）。匹配：``<实验名>_added_*``、索引名
    包含模型文件名（大小写不敏感，含去掉 ``_e\d+_s\d+`` 后的实验名）；
    排除 ``trained`` 索引；带 ``_spkidN`` 后缀的索引按说话人过滤。
    找不到返回空字符串。
    """
    model_stem = os.path.splitext(os.path.basename(str(model_name or "")))[0]
    experiment = _EXPERIMENT_SUFFIX_RE.sub("", model_stem)
    if not experiment:
        return ""
    try:
        target_spk = None if speaker_id is None else int(speaker_id)
    except (TypeError, ValueError):
        target_spk = None

    roots = []
    env_outside = os.environ.get("outside_index_root")
    if env_outside:
        roots.append(env_outside)
    else:
        roots.append(str(PROJECT_ROOT / "models" / "indices"))  # 新默认
        roots.append(str(ASSETS / "indices"))                    # 兼容回退
    roots.append(os.environ.get("index_root") or str(PROJECT_ROOT / "logs"))
    candidates = []
    for index_root in roots:
        if not index_root or not os.path.isdir(index_root):
            continue
        for root, _, files in os.walk(index_root, topdown=False):
            for name in files:
                # 兼容三种索引：.npz（自建暴力）、.ivf.npz（倒排）、.index（faiss 原版）
                if not name.lower().endswith((".npz", ".index", ".faiss")) \
                        or "trained" in name.lower():
                    continue
                index_stem = os.path.splitext(name)[0]
                lower_index = index_stem.lower()
                lower_experiment = experiment.lower()
                lower_model = model_stem.lower()
                spk_match = _SPKID_RE.search(index_stem)
                indexed_spk = int(spk_match.group(1)) if spk_match else None
                if target_spk is None and indexed_spk is not None:
                    continue
                if (
                    target_spk is not None
                    and indexed_spk is not None
                    and indexed_spk != target_spk
                ):
                    continue
                standard_match = (
                    lower_index.startswith(lower_experiment + "_added_")
                    or ("_" + lower_experiment + "_v1") in lower_index
                    or ("_" + lower_experiment + "_v2") in lower_index
                )
                # 宽松匹配：索引名包含模型名（含去 _e\d+_s\d+ 后缀后的实验名）
                exact_model_match = (
                    lower_model in lower_index
                    or (lower_experiment and lower_experiment in lower_index)
                )
                if not (standard_match or exact_model_match):
                    continue
                path = os.path.abspath(os.path.join(root, name))
                score = (
                    0 if indexed_spk == target_spk else 1,
                    0 if standard_match else 1,
                    0 if os.path.abspath(index_root) == os.path.abspath(roots[0]) else 1,
                    -os.path.getmtime(path),
                    path.lower(),
                )
                candidates.append((score, path))
    return min(candidates, default=(None, ""), key=lambda item: item[0])[1]


def _extract_weight_config(cpt: dict):
    """从 checkpoint dict 提取 ``(weight_dict, config_or_None)``。

    兼容推理格式（``weight`` 键）与训练底模（``model`` 键），
    以及顶层直接就是 state_dict 的形态。config 为 list 或 None。
    """
    if not isinstance(cpt, dict):
        raise ValueError("checkpoint 顶层不是 dict，无法识别模型格式")
    w = None
    cfg = None
    if "weight" in cpt and isinstance(cpt["weight"], dict):
        w = cpt["weight"]
        cfg = cpt.get("config")
    elif "model" in cpt and isinstance(cpt["model"], dict):
        w = cpt["model"]
        cfg = cpt.get("config")
    elif "emb_g.weight" in cpt:
        w = cpt
    else:
        raise ValueError(
            "无法识别 checkpoint 结构：缺少 'weight'/'model' 或顶层权重键"
        )
    if "emb_g.weight" not in w:
        raise ValueError("模型权重缺少 'emb_g.weight'（说话人嵌入），不是合法的 RVC 模型")
    return w, (list(cfg) if cfg is not None else None)


def _infer_version(w: dict) -> str:
    """无 ``version`` 键时从特征维数推断：768d -> v2，256d -> v1。"""
    dim = int(w["enc_p.emb_phone.weight"].shape[1])
    return "v2" if dim == 768 else "v1"


class VC:
    """RVC 推理高层封装（对齐 ``infer/vc/modules.py.VC``）。

    生命周期：``VC(config)`` -> ``get_vc(模型名)`` -> ``vc_single(...)``
    （可多次）-> ``vc_multi(...)``（批量便捷方法）。
    所有返回音频为 numpy int16（pipeline 输出），采样率见元组第一项。
    """

    def __init__(self, config):
        """config：``runtime.native_config.Config`` 实例（提供 x_pad 等切分参数）。"""
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None          # runtime.models.vits.SynthesizerTrn
        self.pipeline = None       # runtime.pipeline.Pipeline
        self.cpt = None            # torch_compat 读取的 checkpoint dict
        self.version = None        # "v1" / "v2"
        self.if_f0 = None          # 是否音高引导（本项目恒为 1）
        self.hubert_model = None   # runtime.models.hubert.HubertEncoder
        self.model_rmvpe = None    # runtime.models.rmvpe.RMVPE（懒加载）
        self.index_paths = {}      # 实验名 -> [绝对路径 .npz/.index/.faiss]（get_vc 时扫描）
        self.config = config
        self.net_g_path = None     # 当前已加载合成器的 .pth 路径

    # ------------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------------
    def _resolve_model_path(self, sid) -> str:
        """把 sid 解析为存在的 .pth 路径。

        顺序：原样（绝对/相对路径）→ 依次尝试 ``weight_root`` 候选
        （``models/`` 优先、``assets/weights`` 兼容回退）下的
        ``<sid>[.pth]``。找不到抛 ``FileNotFoundError`` 并给出清晰提示。
        """
        sid = os.fspath(sid).strip()
        if not sid:
            raise FileNotFoundError("模型名（sid）为空")
        candidates = []
        # 1) 绝对路径 / 已存在的相对路径（cwd 或项目根下）
        if os.path.isabs(sid) or os.path.isfile(sid):
            candidates.append(sid)
        # 2) 每个 weight_root 候选下的 <sid>[.pth]
        for root in _weight_roots():
            candidates.append(os.path.join(root, sid))
            if not sid.lower().endswith(".pth"):
                candidates.append(os.path.join(root, sid + ".pth"))
        for p in candidates:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            "找不到模型文件：%r（已尝试 %d 个候选路径，weight_root=%s）"
            % (sid, len(candidates), _weight_root())
        )

    @staticmethod
    def _hubert_dir() -> str:
        return str(ASSETS / "hubert_base")

    @staticmethod
    def _rmvpe_path() -> str:
        return str(ASSETS / "rmvpe" / "rmvpe.pt")

    # ------------------------------------------------------------------
    # 索引扫描（可选，T33.5 前用 .npz）
    # ------------------------------------------------------------------
    def _scan_index_paths(self) -> dict:
        """扫描 index_root / outside_index_root 下全部 .npz/.index/.faiss 索引
        （排除 trained），与 find_index_path_for_model 后缀集合一致。

        返回 ``{实验名: [绝对路径, ...]}``，供 vc_single 在 index_rate>0
        且未显式给索引时自动匹配。
        """
        result: dict = {}
        roots = []
        env_outside = os.environ.get("outside_index_root")
        if env_outside:
            roots.append(env_outside)
        else:
            roots.append(str(PROJECT_ROOT / "models" / "indices"))  # 新默认
            roots.append(str(ASSETS / "indices"))                    # 兼容回退
        roots.append(os.environ.get("index_root") or str(PROJECT_ROOT / "logs"))
        for index_root in roots:
            if not index_root or not os.path.isdir(index_root):
                continue
            for root, _, files in os.walk(index_root, topdown=False):
                for name in files:
                    if not name.lower().endswith((".npz", ".index", ".faiss")) \
                            or "trained" in name.lower():
                        continue
                    stem = os.path.splitext(name)[0]
                    # 实验名 = 索引名去掉 _added_ 及之后部分（含 _spkidN）
                    experiment = re.split(r"_added_", stem, flags=re.IGNORECASE)[0]
                    result.setdefault(experiment, []).append(
                        os.path.abspath(os.path.join(root, name))
                    )
        return result

    def _auto_index_for(self, model_name) -> str:
        """为模型自动挑选索引：精确实验名优先，其次名称包含匹配。"""
        if not self.index_paths:
            return ""
        stem = os.path.splitext(os.path.basename(str(model_name)))[0]
        experiment = _EXPERIMENT_SUFFIX_RE.sub("", stem)
        if experiment in self.index_paths:
            return self.index_paths[experiment][0]
        lower = experiment.lower()
        for name, paths in self.index_paths.items():
            if lower and lower in name.lower():
                return paths[0]
        return ""

    # ------------------------------------------------------------------
    # get_vc：加载模型
    # ------------------------------------------------------------------
    def get_vc(self, sid, *to_return_protect) -> dict:
        """加载 sid 指定的模型，准备推理上下文，返回模型信息 dict。

        参数:
            sid: 模型文件名（assets/weights/<sid>.pth，自动补 .pth）或绝对路径。
            to_return_protect: 兼容原版 gradio 签名，忽略其内容。

        返回:
            ``{"success": bool, "sid": str, "path": str, "n_spk": int,
            "tgt_sr": int, "version": str, "if_f0": int,
            "speakers": [{"id","name"}], "index_paths": dict,
            "error": str|None}``；加载失败时 success=False 且 error 含原因。
        """
        try:
            path = self._resolve_model_path(sid)
            cpt = load_pth(path)
            w, cfg_list = _extract_weight_config(cpt)
            self.cpt = cpt
            self.net_g_path = path

            # tgt_sr：推理格式取 config[-1]；底模无 config 时从权重反推
            if cfg_list:
                self.tgt_sr = int(cfg_list[-1])
            else:
                self.tgt_sr = VitsConfig(w, None).sr  # 48k 底模默认 48000

            self.n_spk = int(w["emb_g.weight"].shape[0])
            self.if_f0 = int(cpt.get("f0", 1))
            raw_version = cpt.get("version")
            self.version = (
                str(raw_version) if raw_version is not None else _infer_version(w)
            )
            if self.version not in ("v1", "v2"):
                raise ValueError(
                    "模型 version 仅支持 v1/v2，实际 %r" % (self.version,)
                )

            # 合成器懒加载缓存（vits.py 内部已兼容 weight/model 两种格式）
            # T1.2：切换模型前清理旧权重注册——同进程 f0G48k→gan1 时旧模型
            # 残留 key 会被 _weights.get 命中（形状/数值错位→dec add_inplace
            # 形状不一致崩）；free_all 后切换正常（实测 11.07s 6s pm）。
            from runtime import vulkan_weights as _vw  # noqa: PLC0415
            _vw._weights.free_all()
            self.net_g = load_synthesizer(path)
            self.pipeline = Pipeline(self.tgt_sr, self.config)

            # hubert / rmvpe 模型
            self.hubert_model = load_hubert_model(self._hubert_dir())
            # T17：rmvpe 权重在模型加载阶段预热（模块级 _CACHE 缓存，进程内
            # 只冷加载一次；pipeline.get_f0 首次调用 load_rmvpe 命中缓存，
            # 1.1s 冷加载从推理墙移出）。pm/fcpe 场景多占一次预热，权重缺失
            # 时静默跳过（get_f0 懒加载路径兜底，行为与原来一致）。
            # 注意：本方法开头 free_all() 会清掉 rmvpe.* 常驻注册表，而 _CACHE
            # 复用实例不会重新执行 __init__（_register_gpu_weights 不重跑到）——
            # 多模型切换后这里必须按注册表缺失显式重注册，否则 rmvpe batch
            # 路径 conv2d 拿不到 buf_w（None+None → dtype=object 崩溃）。
            try:
                if os.path.isfile(self._rmvpe_path()):
                    _rm = load_rmvpe(self._rmvpe_path())
                    if _vw._weights.get("rmvpe.cnn.weight") is None:
                        _rm._register_gpu_weights()
            except Exception:  # noqa: BLE001
                self.model_rmvpe = None
            self.index_paths = self._scan_index_paths()

            speakers = normalized_speaker_info(cpt, self.n_spk)
            return {
                "success": True,
                "sid": os.fspath(sid),
                "path": path,
                "n_spk": self.n_spk,
                "tgt_sr": self.tgt_sr,
                "version": self.version,
                "if_f0": self.if_f0,
                "speakers": speakers,
                "index_paths": dict(self.index_paths),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001  # 向 CLI/上层报告完整失败原因
            info = traceback.format_exc()
            return {
                "success": False,
                "sid": os.fspath(sid),
                "path": None,
                "n_spk": None,
                "tgt_sr": None,
                "version": None,
                "if_f0": None,
                "speakers": [],
                "index_paths": {},
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": info,
            }

    # ------------------------------------------------------------------
    # vc_single：单文件变声
    # ------------------------------------------------------------------
    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        format=None,
        progress_cb=None,
        slice_length=0,
        retrieval_mode="ivf",
        brute_mix=0.0,
    ):
        """单文件变声（对齐原版 vc_single 主流程；format 仅作签名兼容）。

        参数:
            sid: 说话人 ID（int；模型已由 get_vc 加载）。
            input_audio_path: 输入音频路径（任意 soundfile 可读格式）。
            f0_up_key: 移调半音数。
            f0_method: "pm"（自相关）或 "rmvpe"。
            file_index: FeatureIndex 的 .npz 路径或 None（index_rate>0 时必填
                或可自动匹配，否则抛清晰错误）。
            index_rate: 检索混合率 0~1。
            resample_sr: 输出采样率（0 或 >=16000；0 表示保持模型 tgt_sr）。
            rms_mix_rate: RMS 包络混合率。
            protect: 清辅音保护（0~0.5）。
            format: 兼容参数，不参与处理（写文件编码由 CLI 负责）。
            progress_cb: 可选 ``f(done, total, msg)`` 进度回调（每处理一块
                调用）；None 时默认写到进程级 logstream（形如 "进度: 3/12"），
                供 WebUI 实时日志使用（CLI 场景无人读取，无副作用）。
            slice_length: 切分长度（秒，>0 时覆盖 pipeline 的免切分阈值，
                用于"切分长度"滑块）。

        返回:
            ``("转换成功", (tgt_sr, audio_int16))``；失败抛异常。
        """
        if self.net_g is None or self.pipeline is None:
            raise RuntimeError("模型未加载：请先调用 get_vc(sid) 选择模型")
        if input_audio_path is None:
            raise ValueError("input_audio_path 不能为 None")

        if progress_cb is None:
            def progress_cb(done, total, msg):
                from . import logstream
                logstream.write_line("进度: %d/%d %s" % (done, total, msg))

        f0_up_key = int(f0_up_key)
        # 1) 归一化（与原版一致：load 16k -> 峰值归一到 <=0.95）
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio = audio / audio_max
        times = [0.0, 0.0, 0.0]

        # 2) 索引解析：index_rate>0 时必须有有效索引
        file_index = self._resolve_file_index(file_index, index_rate)

        # 3) rmvpe 懒加载（与 pipeline 内部共享模块缓存，只加载一次）
        if f0_method == "rmvpe" and self.model_rmvpe is None:
            self.model_rmvpe = load_rmvpe(self._rmvpe_path())

        # 4) 主流程（参数顺序与 runtime.pipeline.pipeline 一致；
        #    pipeline 返回 (tgt_sr, audio_int16) 元组，需拆包）
        #    finally：单文件推理结束（成功或失败）清空输出 buffer 池（真释放
        #    显存）——防长音频/连续多文件推理的池驻留显存累积导致
        #    VK_ERROR_OUT_OF_DEVICE_MEMORY（rvc_mem_upload VkFailed）。
        try:
            _pipeline_sr, audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                int(sid),
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                float(index_rate),
                self.if_f0,
                self.tgt_sr,
                int(resample_sr),
                float(rms_mix_rate),
                self.version,
                float(protect),
                progress_cb=progress_cb,
                slice_length=slice_length,
                retrieval_mode=retrieval_mode,
                brute_mix=brute_mix,
            )
        finally:
            from runtime import vulkan_ops as _vo  # noqa: PLC0415
            _vo.get_context().clear_pool()

        # 5) 输出采样率（与原版一致）
        if self.tgt_sr != resample_sr >= 16000:
            tgt_sr = int(resample_sr)
        else:
            tgt_sr = self.tgt_sr
        return "转换成功", (tgt_sr, audio_opt)

    def _resolve_file_index(self, file_index, index_rate) -> str:
        """把用户 file_index 规整为 pipeline 可用的路径字符串。

        - index_rate == 0：不检索，返回 ""（pipeline 内部跳过索引）。
        - 显式给路径：strip 引号/空格，trained->added 替换；**路径不存在则报错**
          （用户明确指定了文件却找不到，值得提示）。
        - 未显式给：按当前模型自动匹配（.npz/.ivf.npz/.index）；**匹配不到时
          降级返回 ""**（等效 index_rate=0，对齐原版 RVC 行为——原版无索引时
          静默跳过检索，而不是中断推理）。
        """
        if index_rate == 0:
            return ""
        raw = (
            str(file_index or "").strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if raw:
            if "trained" in os.path.basename(raw):
                raw = raw.replace("trained", "added")
            if os.path.isfile(raw):
                return raw
            raise RuntimeError(
                "指定的索引文件不存在：%r（index_rate=%s）。"
                "请检查路径，或将索引放入索引目录，或设 index_rate=0。"
                % (file_index, index_rate)
            )
        # 未显式给索引 -> 自动匹配（当前模型）；匹配不到自动降级（对齐原版）
        if self.net_g_path:
            auto = self._auto_index_for(os.path.basename(self.net_g_path))
            if auto:
                return auto
        return ""

    # ------------------------------------------------------------------
    # vc_multi：批量变声（简化版）
    # ------------------------------------------------------------------
    def vc_multi(
        self,
        sid,
        paths,
        f0_up_key=0,
        f0_method="rmvpe",
        file_index=None,
        index_rate=0.0,
        resample_sr=0,
        rms_mix_rate=1.0,
        protect=0.33,
    ):
        """批量变声：对 ``paths`` 逐文件调用 vc_single。

        返回 ``[(path, (tgt_sr, audio_int16)|None, error|None), ...]``：
        单个文件失败不中断，错误记录在对应元组中。
        """
        results = []
        for path in paths:
            try:
                _, opt = self.vc_single(
                    sid, path, f0_up_key, f0_method, file_index,
                    index_rate, resample_sr, rms_mix_rate, protect,
                )
                results.append((path, opt, None))
            except Exception as exc:  # noqa: BLE001
                results.append((path, None, "%s: %s" % (type(exc).__name__, exc)))
        return results


def model_speaker_info(model_path: str):
    """读取模型的说话人信息，返回 (speaker_count, [{"id","name"}, ...])。

    兼容推理格式（{"weight","config","speaker_info"}）与训练底模
    （{"model",...}）。供 WebUI/API 的说话人列表使用。
    """
    import torch_compat
    cpt = torch_compat.load_pth(os.fspath(model_path))
    w, cfg = _extract_weight_config(cpt)
    emb = w.get("emb_g.weight")
    if emb is None:
        raise ValueError("模型 %s 不含 emb_g.weight" % model_path)
    n_spk = int(emb.shape[0])
    speakers = []
    seen = set()
    for item in cpt.get("speaker_info", []):
        try:
            spk_id = int(item["id"])
            name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= spk_id < n_spk and name and spk_id not in seen:
            speakers.append({"id": spk_id, "name": name})
            seen.add(spk_id)
    speakers.sort(key=lambda s: s["id"])
    return n_spk, speakers


def _self_test():
    """vc 模块自测：路径解析 / 格式提取 / 索引查找冒烟。"""
    print("=== vc._self_test ===")
    ok = True

    # 1) _extract_weight_config 两种格式
    w_fake = {"emb_g.weight": np.zeros((2, 256), np.float32),
              "enc_p.emb_phone.weight": np.zeros((192, 768), np.float32)}
    w2, cfg2 = _extract_weight_config({"weight": w_fake, "config": [1, 2, 3]})
    ok &= w2 is w_fake and cfg2 == [1, 2, 3]
    w3, cfg3 = _extract_weight_config({"model": w_fake})
    ok &= w3 is w_fake and cfg3 is None
    w4, cfg4 = _extract_weight_config(w_fake)
    ok &= w4 is w_fake and cfg4 is None
    print("  _extract_weight_config 两种格式 PASS:", ok)

    # 2) version 推断
    ok &= _infer_version(w_fake) == "v2"
    w_v1 = dict(w_fake)
    w_v1["enc_p.emb_phone.weight"] = np.zeros((192, 256), np.float32)
    ok &= _infer_version(w_v1) == "v1"
    print("  _infer_version PASS:", ok)

    # 3) 索引查找（assets/indices 有 mute 索引；模型名不应匹配）
    found = find_index_path_for_model("nonexistent_model")
    ok &= found == ""
    print("  find_index_path_for_model 不匹配 PASS:", ok)
    print("  vc._self_test %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
