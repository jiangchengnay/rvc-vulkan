# -*- coding: utf-8 -*-
"""Flet 桌面 GUI（静态 Web UI 迁移版，对齐 docs/参考/08-UI迁移清单.md）。

把 runtime/static/index.html（Gradio 风格静态页）的**正经推理**交互迁移为
Flet 桌面应用：同进程直接 import runtime.vc / logstream 等后端模块，不再依赖
FastAPI/HTTP。**网页版 UI（FastAPI + index.html）保留不动，两套 UI 并存**；
实时推理 UI（realtime_gui.py）只做非网页版（Tkinter + sounddevice）。

调试接口约定（重要）：
    每个实体按钮对应一个注册在 ``ACTIONS`` 里的动作函数：:

        @register_action("infer", "开始变声", "单文件变声推理")
        def action_infer(ui=None, *, model="", speaker_id=0, ..., audio=None) -> dict:
            ...

    - 按钮点击 → 收集控件值 → ``run_action("infer", kwargs, ui=self)``；
    - 命令行直调（headless 调试）：:

        python -m runtime.gui_flet --list-actions
        python -m runtime.gui_flet --action health
        python -m runtime.gui_flet --action infer --model xxx --audio in.wav --pitch 0

用法::

    cd projects/rvc-vulkan
    python -m runtime.gui_flet                       # 打开窗口

依赖: flet>=1.0（``pip install flet``）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import flet as ft

from runtime import logstream

APP_TITLE = "RVC-Vulkan 变声器（Flet 桌面版）"
POLL_INTERVAL_S = 1.0  # 日志轮询间隔（对齐原版 1s 轮询）

# ======================================================================
# 动作注册表：每个实体按钮 → 一个可命令行直调的函数
# ======================================================================
ACTIONS: dict[str, dict] = {}


def register_action(name: str, label: str, desc: str = ""):
    """注册一个按钮动作。用法：:

        @register_action("infer", "开始变声", "单文件变声推理")
        def action_infer(ui=None, *, model="", ...): ...

    参数规约：首参 ``ui``（GuiFletApp 或 None），其余全部关键字参数，
    默认值即 UI 控件缺省值——保证 headless 与按钮两种调用完全同参。
    """

    def deco(fn: Callable) -> Callable:
        ACTIONS[name] = {"fn": fn, "label": label, "desc": desc, "name": name}
        return fn

    return deco


def run_action(name: str, kwargs: Optional[dict] = None, ui=None):
    """按名称执行动作（按钮与命令行共用入口）。"""
    reg = ACTIONS.get(name)
    if reg is None:
        raise KeyError("未知动作: %r（可用: %s）" % (name, ", ".join(ACTIONS)))
    return reg["fn"](ui, **(kwargs or {}))


# ----------------------------------------------------------------------
# 共享后端辅助（与 api.py 同逻辑，但去掉 HTTP 层）
# ----------------------------------------------------------------------
def _output_dir() -> str:
    """推理结果输出目录（对齐 api.py 的 APP_ROOT/输出）。"""
    root = Path(__file__).resolve().parents[1]
    d = root / "输出"
    d.mkdir(exist_ok=True)
    return str(d)


def _find_model(model: str) -> Optional[str]:
    from runtime.api import _list_models
    for m in _list_models():
        if m["name"] == model:
            return m["path"]
    return None


def _infer_once(model_path: str, audio_path: str, speaker_id: int = 0,
                pitch: int = 0, f0_method: str = "rmvpe", index: str = "",
                index_rate: float = 0.0, resample_sr: int = 0,
                rms_mix_rate: float = 1.0, protect: float = 0.33,
                slice_length: float = 0.0, retrieval_mode: str = "ivf",
                brute_mix: float = 0.0, clarity_mix: float = 1.0):
    """单文件变声核心（等价 api.py /infer 的默认分支）。

    返回 ``(sr, out_path)``；结果落盘到 输出/ 目录。
    """
    from runtime.api import _get_vc
    from runtime.models.vits import set_clarity_mix
    from runtime.dsp.audio_io import write_audio
    import numpy as np
    import time as _time

    vc = _get_vc(model_path)
    logstream.write_line("推理: 模型=%s spk=%s 变调=%s 索引率=%s f0=%s" %
                         (os.path.basename(model_path), speaker_id, pitch,
                          index_rate, f0_method))
    set_clarity_mix(clarity_mix)
    with logstream.capture_stdout():
        status, result = vc.vc_single(
            speaker_id, audio_path, pitch, f0_method, index or None,
            index_rate, resample_sr, rms_mix_rate, protect,
            slice_length=slice_length,
            retrieval_mode=retrieval_mode, brute_mix=brute_mix,
        )
    if not result or result[0] is None or result[1] is None:
        raise RuntimeError(status or "未知错误")
    sr, audio_int16 = result
    float_audio = audio_int16.astype(np.float32) / 32768.0
    out_name = "infer_%s.wav" % _time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_output_dir(), out_name)
    write_audio(out_path, float_audio, sr)
    logstream.write_line("推理: 完成（%d Hz）→ %s" % (sr, out_name))
    return sr, out_path


# ----------------------------------------------------------------------
# 基础动作
# ----------------------------------------------------------------------
@register_action("health", "健康状态", "查询计算后端与设备")
def action_health(ui=None):
    from runtime import backend
    dev = backend.device_info() or "?"
    out = {"backend": backend.get_backend(), "device": dev}
    if ui is None:
        print(json.dumps(out, ensure_ascii=False))
    return out


@register_action("list_models", "刷新模型列表", "列出 models/ 优先、assets 兼容的模型")
def action_list_models(ui=None):
    from runtime.api import _list_models
    models = _list_models()
    if ui is None:
        for m in models:
            print("%s\t%s" % (m["name"], m["path"]))
    return models


@register_action("list_files", "文件管理", "列出 models/ 与 models/indices/ 下的文件")
def action_list_files(ui=None):
    import os as _os
    from runtime.api import _list_models  # noqa: F401
    items = {"models": [], "indices": []}
    root = _models_root()
    if root:
        for name in sorted(_os.listdir(root)):
            p = _os.path.join(root, name)
            if _os.path.isfile(p) and name.lower().endswith((".pth", ".safetensors")):
                items["models"].append({"name": name, "kind": "model", "path": p})
        idx_root = _os.path.join(root, "indices")
        if _os.path.isdir(idx_root):
            for name in sorted(_os.listdir(idx_root)):
                p = _os.path.join(idx_root, name)
                if _os.path.isfile(p) and name.lower().endswith((".npz", ".index")):
                    items["indices"].append({"name": name, "kind": "index", "path": p})
    if ui is None:
        for it in items["models"] + items["indices"]:
            print("%s\t%s\t%s" % (it["kind"], it["name"], it["path"]))
    return items


def _models_root() -> Optional[str]:
    """models/ 目录（对齐 api.py 的 APP_ROOT/models 推导）。"""
    try:
        from runtime import api as _api
        root = os.path.join(_api.APP_ROOT, "models")
        if os.path.isdir(root):
            return root
    except Exception:  # noqa: BLE001
        pass
    here = Path(__file__).resolve().parents[1]
    root = here / "models"
    return str(root) if root.is_dir() else None


@register_action("list_speakers", "说话人列表", "查询模型说话人数量（model 必填）")
def action_list_speakers(ui=None, *, model: str = ""):
    from runtime.api import _list_models
    from runtime.vc import model_speaker_info
    found = next((m for m in _list_models() if m["name"] == model), None)
    if found is None:
        raise ValueError("模型不存在: %s" % model)
    count, speakers = model_speaker_info(found["path"])
    out = {"model": model, "speaker_count": count, "speakers": speakers}
    if ui is None:
        print(json.dumps(out, ensure_ascii=False))
    return out


@register_action("infer", "开始变声", "单文件变声推理（--model --audio 必填）")
def action_infer(ui=None, *, model: str = "", audio: str = "",
                 speaker_id: int = 0, pitch: int = 0,
                 f0_method: str = "rmvpe", index: str = "",
                 index_rate: float = 0.0, resample_sr: int = 0,
                 rms_mix_rate: float = 1.0, protect: float = 0.33,
                 slice_length: float = 0.0, retrieval_mode: str = "ivf",
                 brute_mix: float = 0.0, clarity_mix: float = 1.0):
    """单文件变声（headless 或按钮共用）。返回 {"status","sr","file","path"}。"""
    if not model:
        raise ValueError("请选择模型（--model）")
    if not audio or not os.path.isfile(audio):
        raise ValueError("输入音频不存在: %r" % (audio,))
    path = _find_model(model)
    if path is None:
        raise ValueError("模型不存在: %s" % model)
    if ui is not None:
        ui._log_ui("infer", "推理: 开始（模型=%s）" % model)
    try:
        sr, out_path = _infer_once(
            path, audio, speaker_id, pitch, f0_method, index, index_rate,
            resample_sr, rms_mix_rate, protect, slice_length,
            retrieval_mode, brute_mix, clarity_mix)
    except Exception as exc:  # noqa: BLE001
        if ui is not None:
            ui._log_ui("infer", "推理: 失败 %s" % exc)
        raise
    out = {"status": "ok", "sr": sr,
           "file": os.path.basename(out_path), "path": out_path}
    if ui is not None:
        ui._on_infer_done(out)
    elif os.name == "nt":
        print("✅ 转换成功：%s" % out_path)
    return out


@register_action("batch", "批量变声", "多文件/目录批量变声（--audio 可多次？否，用 in_dir）")
def action_batch(ui=None, *, model: str = "", in_dir: str = "",
                 out_dir: str = "", speaker_id: int = 0, pitch: int = 0,
                 f0_method: str = "rmvpe", index: str = "",
                 index_rate: float = 0.0, resample_sr: int = 0,
                 rms_mix_rate: float = 1.0, protect: float = 0.33,
                 slice_length: float = 0.0, retrieval_mode: str = "ivf",
                 brute_mix: float = 0.0, clarity_mix: float = 1.0,
                 files: Optional[list] = None):
    """批量变声（files=本地路径列表 或 in_dir=目录）。返回结果表。"""
    from runtime.api import _list_models
    from runtime.vc import AUDIO_EXTENSIONS
    if not model:
        raise ValueError("请选择模型")
    path = _find_model(model)
    if path is None:
        raise ValueError("模型不存在: %s" % model)

    inputs = []
    if files:
        for p in files:
            inputs.append((os.path.basename(p), str(p)))
    elif in_dir:
        in_dir = os.path.normpath(in_dir)
        if not os.path.isdir(in_dir):
            raise ValueError("输入目录不存在: %s" % in_dir)
        for name in sorted(os.listdir(in_dir)):
            p = os.path.join(in_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                inputs.append((name, p))
        if not inputs:
            raise ValueError("输入目录中没有支持的音频文件: %s" % in_dir)
    else:
        raise ValueError("需要 files 或 in_dir 参数")

    out_root = os.path.normpath(out_dir) if out_dir else \
        (os.path.normpath(in_dir) if in_dir else os.path.join(_output_dir(), "batch"))
    os.makedirs(out_root, exist_ok=True)

    if ui is not None:
        ui._log_ui("infer", "批量: 开始（%d 个文件，模型=%s）" % (len(inputs), model))
    logstream.write_line("批量: 开始（%d 个文件，模型=%s）" % (len(inputs), model))
    from runtime.api import _get_vc
    from runtime.models.vits import set_clarity_mix
    from runtime.dsp.audio_io import write_audio
    import numpy as np
    set_clarity_mix(clarity_mix)
    vc = _get_vc(path)
    results = []
    total = len(inputs)
    try:
        with logstream.capture_stdout():
            for i, (name, p) in enumerate(inputs, 1):
                logstream.write_line("批量: 文件 %d/%d %s" % (i, total, name))
                out_name = os.path.splitext(name)[0] + ".wav"
                out_path = os.path.join(out_root, out_name)
                try:
                    status, result = vc.vc_single(
                        speaker_id, p, pitch, f0_method, index or None,
                        index_rate, resample_sr, rms_mix_rate, protect,
                        slice_length=slice_length,
                        retrieval_mode=retrieval_mode, brute_mix=brute_mix,
                    )
                    if not result or result[0] is None or result[1] is None:
                        raise RuntimeError(status or "未知错误")
                    sr, audio_int16 = result
                    float_audio = audio_int16.astype(np.float32) / 32768.0
                    write_audio(out_path, float_audio, sr)
                    results.append({"name": name, "ok": True, "out_file": out_path,
                                    "error": None})
                except Exception as exc:  # noqa: BLE001
                    results.append({"name": name, "ok": False, "out_file": "",
                                    "error": "%s: %s" % (type(exc).__name__, exc)})
        ok_n = sum(1 for r in results if r["ok"])
        logstream.write_line("批量: 完成 %d/%d" % (ok_n, total))
    except Exception as exc:  # noqa: BLE001
        if ui is not None:
            ui._log_ui("infer", "批量: 失败 %s" % exc)
        raise
    if ui is not None:
        ui._on_batch_done(results, out_root)
    return {"status": "ok", "total": total, "results": results, "out_dir": out_root}


@register_action("batch_zip", "打包下载 zip", "把批量成功结果打包为 zip")
def action_batch_zip(ui=None, *, out_dir: str = "", zip_path: str = ""):
    import zipfile
    if not out_dir or not os.path.isdir(out_dir):
        raise ValueError("out_dir 不存在: %r" % (out_dir,))
    if not zip_path:
        zip_path = os.path.join(out_dir, "rvc_batch.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(out_dir)):
            if name.lower().endswith(".wav"):
                zf.write(os.path.join(out_dir, name), arcname=name)
    if ui is None:
        print("zip: %s" % zip_path)
    return {"status": "ok", "zip": zip_path}


# ======================================================================
# Flet 应用主体
# ======================================================================
class GuiFletApp:
    """Flet 应用主体：页面装配 + 日志轮询线程 + 按钮→动作桥接。"""

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._log_pos = 0
        self._polling = False
        self.infer_log_col: Optional[ft.Column] = None
        self.train_log_col: Optional[ft.Column] = None
        self._file_picker: Optional[ft.FilePicker] = None
        # 推理参数控件（T2）
        self.model_dd: Optional[ft.Dropdown] = None
        self.f0_dd: Optional[ft.Dropdown] = None
        self.spk_sl: Optional[ft.Slider] = None
        self.pitch_sl: Optional[ft.Slider] = None
        self.slice_sl: Optional[ft.Slider] = None
        self.audio_path: Optional[str] = None
        self.idx_dd: Optional[ft.Dropdown] = None
        self.idxr_sl: Optional[ft.Slider] = None
        self.rmode_dd: Optional[ft.Dropdown] = None
        self.brmix_sl: Optional[ft.Slider] = None
        self.prot_sl: Optional[ft.Slider] = None
        self.clmix_sl: Optional[ft.Slider] = None
        self.rms_sl: Optional[ft.Slider] = None
        self.resr_sl: Optional[ft.Slider] = None
        self.go_btn: Optional[ft.ElevatedButton] = None
        self.infer_prog: Optional[ft.ProgressBar] = None
        self.result_box: Optional[ft.Column] = None
        self.batch_files: list = []
        self.batch_in_dir: Optional[ft.TextField] = None
        self.batch_out_dir: Optional[ft.TextField] = None
        self.batch_go_btn: Optional[ft.ElevatedButton] = None
        self.batch_zip_btn: Optional[ft.ElevatedButton] = None
        self.batch_prog: Optional[ft.ProgressBar] = None
        self.batch_result: Optional[ft.Column] = None
        self._last_batch_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # 页面装配
    # ------------------------------------------------------------------
    def build(self) -> None:
        self.page.title = APP_TITLE
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 18
        self.page.spacing = 12
        self.page.scroll = ft.ScrollMode.AUTO
        self.page.width = 1080
        self.page.height = 800

        self._file_picker = ft.FilePicker(on_result=self._on_pick_audio)
        self.page.overlay.append(self._file_picker)

        self.health_text = ft.Text("检测中…", size=12, color=ft.Colors.GREY_400)
        header = ft.Row(
            controls=[
                ft.Text("🎙️ RVC-Vulkan 变声器", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                ft.Text("Flet 桌面版 · 正经推理", size=11, color=ft.Colors.GREY_500),
                self.health_text,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.tabs = ft.Tabs(
            expand=True,
            animation_duration=150,
            tabs=[
                ft.Tab(text="模型推理", icon=ft.Icons.MIC, content=self._page_infer()),
                ft.Tab(text="训练", icon=ft.Icons.SCHOOL, content=self._page_train()),
                ft.Tab(text="模型工具", icon=ft.Icons.CONSTRUCTION, content=self._page_tools()),
                ft.Tab(text="文件管理", icon=ft.Icons.FOLDER, content=self._page_files()),
                ft.Tab(text="关于", icon=ft.Icons.INFO, content=self._page_about()),
            ],
        )

        footer = ft.Text(
            "RVC-Vulkan · 第三方个人项目 · 网页版(FastAPI)与桌面版(Flet)两套 UI 并存",
            size=11, color=ft.Colors.GREY_600, italic=True,
        )
        self.page.add(header, self.tabs, footer)

    # ------------------------------------------------------------------
    # 滑块助手：滑块 + 右侧当前值徽标（对齐原版 bindSlider）
    # ------------------------------------------------------------------
    @staticmethod
    def _slider_row(label: str, min_v: float, max_v: float, value: float,
                    step: float, fmt: Callable[[float], str],
                    on_change: Optional[Callable] = None,
                    store: Optional[dict] = None,
                    store_key: str = "") -> ft.Row:
        sval = ft.Text(fmt(value), size=12, color=ft.Colors.AMBER_300,
                       font_family="monospace")

        def _on_change(e):
            sval.value = fmt(float(e.control.value))
            e.control.page.update()
            if on_change is not None:
                on_change(e)

        sl = ft.Slider(min=min_v, max=max_v, value=value, divisions=None,
                       label="{value}", on_change=_on_change)
        if store is not None and store_key:
            store[store_key] = sl
        return ft.Row([
            ft.Container(ft.Text(label, size=12), expand=True),
            sl, sval,
        ])

    # ------------------------------------------------------------------
    # 模型推理页（T2/T3）
    # ------------------------------------------------------------------
    def _page_infer(self) -> ft.Control:
        from runtime.api import _list_models
        models = _list_models()
        opts = [ft.dropdown.Option(m["name"]) for m in models] or \
               [ft.dropdown.Option("（未找到模型）")]
        self.model_dd = ft.Dropdown(options=opts, value=models[0]["name"] if models else None,
                                    on_change=self._on_model_change)

        self.f0_dd = ft.Dropdown(
            options=[ft.dropdown.Option("rmvpe", "rmvpe（推荐）"),
                     ft.dropdown.Option("pm", "pm"),
                     ft.dropdown.Option("fcpe", "fcpe")],
            value="rmvpe")

        self.idx_dd = ft.Dropdown(
            options=[ft.dropdown.Option("", "自动匹配（推荐）"),
                     ft.dropdown.Option("__none__", "不使用索引")],
            value="")
        self.rmode_dd = ft.Dropdown(
            options=[ft.dropdown.Option("ivf", "正版(IVF近似)"),
                     ft.dropdown.Option("brute", "暴力检索")],
            value="ivf")

        self.go_btn = ft.ElevatedButton("🎤 开始变声", icon=ft.Icons.MIC,
                                        on_click=self._on_go)
        self.infer_prog = ft.ProgressBar(value=0, visible=False)
        self.result_box = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)
        self.infer_log_col = ft.Column(
            [self._log_line("— 实时日志：推理/批量输出将显示在这里 —")],
            spacing=2, scroll=ft.ScrollMode.AUTO)

        # 批量
        self.batch_in_dir = ft.TextField(label="或服务器目录 in_dir（可选）",
                                         hint_text=r"E:\dataset\in", dense=True)
        self.batch_out_dir = ft.TextField(label="输出目录 out_dir（可选）",
                                          hint_text="留空=与输入同目录", dense=True)
        self.batch_go_btn = ft.ElevatedButton("⚡ 批量变声", icon=ft.Icons.PLAY_ARROW,
                                              on_click=self._on_batch_go)
        self.batch_zip_btn = ft.ElevatedButton("📦 打包下载 zip", icon=ft.Icons.ARCHIVE,
                                               on_click=self._on_batch_zip, disabled=True)
        self.batch_prog = ft.ProgressBar(value=0, visible=False)
        self.batch_result = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)

        def _fmt_pitch(v): return str(int(v))
        def _fmt_slice(v): return "自动" if v <= 0 else ("%.1fs" % v)
        def _fmt_f2(v): return "%.2f" % v
        def _fmt_resr(v): return "0（原生）" if v <= 0 else ("%.0fk" % (v / 1000))

        return ft.Column([
            ft.Text("模型推理", size=15, weight=ft.FontWeight.BOLD),
            ft.Row([
                self.model_dd, self.f0_dd,
                ft.ElevatedButton("↻ 刷新", icon=ft.Icons.REFRESH,
                                  on_click=lambda _: self._refresh_models()),
            ], spacing=8),
            ft.Text("说话人 ID / 变调（半音）/ 切分长度（秒，0=自动）", size=11,
                    color=ft.Colors.GREY_500),
            self._slider_row("说话人 ID", 0, 0, 0, 1, _fmt_pitch, self._on_slider,
                             self.__dict__, "spk_sl"),
            self._slider_row("变调", -24, 24, 0, 1, _fmt_pitch, self._on_slider,
                             self.__dict__, "pitch_sl"),
            self._slider_row("切分", 0, 60, 0, 0.5, _fmt_slice, self._on_slider,
                             self.__dict__, "slice_sl"),
            ft.Row([
                ft.ElevatedButton("📁 选择输入音频（wav/flac/ogg/opus…）",
                                  icon=ft.Icons.FOLDER_OPEN,
                                  on_click=lambda _: self._file_picker.pick_files(
                                      allow_multiple=False)),
                self.go_btn,
            ], spacing=8),
            self.infer_prog,
            self.result_box,
            ft.Text("特征索引（自动匹配优先）", size=12, color=ft.Colors.GREY_500),
            ft.Row([self.idx_dd,
                    ft.Text("索引率", size=12)], spacing=8),
            self._slider_row("索引率", 0, 1, 0.75, 0.05, _fmt_f2, self._on_slider,
                             self.__dict__, "idxr_sl"),
            ft.Row([self.rmode_dd,
                    ft.Text("检索模式 / 暴力混合 / 保护 / 清晰度 / RMS / 输出采样率",
                            size=11, color=ft.Colors.GREY_500)], spacing=8),
            self._slider_row("暴力混合", 0, 1, 0.0, 0.05, _fmt_f2, self._on_slider,
                             self.__dict__, "brmix_sl"),
            self._slider_row("保护", 0, 0.5, 0.33, 0.01, _fmt_f2, self._on_slider,
                             self.__dict__, "prot_sl"),
            self._slider_row("说话清晰度", 0, 1, 1.0, 0.05, _fmt_f2, self._on_slider,
                             self.__dict__, "clmix_sl"),
            self._slider_row("RMS 混合率", 0, 1, 1.0, 0.05, _fmt_f2, self._on_slider,
                             self.__dict__, "rms_sl"),
            self._slider_row("输出采样率", 0, 48000, 0, 1000, _fmt_resr, self._on_slider,
                             self.__dict__, "resr_sl"),
            ft.Divider(),
            ft.Text("批量变声（多文件 / 目录）", size=15, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.ElevatedButton("📁 多文件上传", icon=ft.Icons.FILE_UPLOAD,
                                  on_click=lambda _: self._file_picker.pick_files(
                                      allow_multiple=True)),
            ], spacing=8),
            self.batch_in_dir,
            self.batch_out_dir,
            ft.Row([self.batch_go_btn, self.batch_zip_btn], spacing=8),
            self.batch_prog,
            self.batch_result,
            ft.Divider(),
            ft.Text("实时日志", size=12, color=ft.Colors.GREY_500),
            ft.Container(
                content=self.infer_log_col,
                height=180, padding=10,
                bgcolor=ft.Colors.BLACK_12, border_radius=8,
            ),
        ], spacing=10, scroll=ft.ScrollMode.AUTO)

    def _on_slider(self, e):
        e.control.page.update()

    def _on_model_change(self, e):
        # 联动：加载说话人数（异步，防止卡 UI）
        self.page.run_thread(self._load_speakers, self.model_dd.value)

    def _load_speakers(self, model: str):
        try:
            info = run_action("list_speakers", {"model": model})
            self.page.run_thread(self._apply_speakers, info["speaker_count"])
        except Exception:  # noqa: BLE001
            self.page.run_thread(self._apply_speakers, 0)

    def _apply_speakers(self, count: int):
        max_v = max(0, int(count) - 1)
        self.spk_sl.max = max_v
        if self.spk_sl.value > max_v:
            self.spk_sl.value = 0
        self.page.update()

    def _refresh_models(self):
        def _do():
            try:
                models = run_action("list_models")
                self.page.run_thread(self._apply_models, models)
            except Exception as exc:  # noqa: BLE001
                self.page.run_thread(self._toast, "刷新模型失败: %s" % exc)
        self.page.run_thread(_do)

    def _apply_models(self, models: list):
        self.model_dd.options = [ft.dropdown.Option(m["name"]) for m in models]
        if models:
            self.model_dd.value = models[0]["name"]
        self.page.update()

    def _on_pick_audio(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return
        self.audio_path = e.files[0].path
        if e.files[0].name:
            self._toast("已选择：%s" % e.files[0].name)
        if len(e.files) > 1:
            self.batch_files = [f.path for f in e.files]
        elif len(e.files) == 1:
            self.batch_files = [e.files[0].path]
            self.audio_path = e.files[0].path

    def _on_go(self, e):
        if not self.audio_path:
            self._toast("请先选择音频文件")
            return
        if not self.model_dd.value:
            self._toast("请选择模型")
            return
        self.go_btn.disabled = True
        self.infer_prog.visible = True
        self.infer_prog.value = None  # 不确定进度
        self.page.update()
        kwargs = self._collect_infer_kwargs()
        self.page.run_thread(self._run_infer_thread, kwargs)

    def _collect_infer_kwargs(self) -> dict:
        return dict(
            model=self.model_dd.value, audio=self.audio_path,
            speaker_id=int(self.spk_sl.value or 0), pitch=int(self.pitch_sl.value or 0),
            f0_method=self.f0_dd.value, index=self.idx_dd.value or "",
            index_rate=float(self.idxr_sl.value or 0),
            retrieval_mode=self.rmode_dd.value, brute_mix=float(self.brmix_sl.value or 0),
            protect=float(self.prot_sl.value or 0), clarity_mix=float(self.clmix_sl.value or 1),
            rms_mix_rate=float(self.rms_sl.value or 0),
            resample_sr=int(self.resr_sl.value or 0), slice_length=float(self.slice_sl.value or 0),
        )

    def _run_infer_thread(self, kwargs: dict):
        try:
            run_action("infer", kwargs, ui=self)
        except Exception as exc:  # noqa: BLE001
            self.page.run_thread(self._show_result, "❌ 转换失败：%s" % exc, is_err=True)
        finally:
            self.page.run_thread(self._infer_done_ui)

    def _infer_done_ui(self):
        self.go_btn.disabled = False
        self.infer_prog.value = 1.0
        self.page.update()

    def _on_infer_done(self, out: dict):
        self.page.run_thread(self._show_result,
                             "✅ 转换成功（%s）— 已保存到 输出/%s" % (
                                 os.path.basename(out["path"]), out["file"]))

    def _show_result(self, msg: str, is_err: bool = False):
        color = ft.Colors.RED_400 if is_err else ft.Colors.GREEN_400
        self.result_box.controls.append(ft.Text(msg, color=color, size=13,
                                                selectable=True))
        self.page.update()

    def _on_batch_go(self, e):
        files = self.batch_files or []
        in_dir = self.batch_in_dir.value.strip()
        if not files and not in_dir:
            self._toast("请选择音频文件或填写服务器目录 in_dir")
            return
        if not self.model_dd.value:
            self._toast("请选择模型")
            return
        self.batch_go_btn.disabled = True
        self.batch_prog.visible = True
        self.page.update()
        kwargs = self._collect_infer_kwargs()
        kwargs["files"] = files
        kwargs["in_dir"] = in_dir
        kwargs["out_dir"] = self.batch_out_dir.value.strip()
        self.page.run_thread(self._run_batch_thread, kwargs)

    def _run_batch_thread(self, kwargs: dict):
        try:
            out = run_action("batch", kwargs, ui=self)
            self._last_batch_dir = out.get("out_dir")
        except Exception as exc:  # noqa: BLE001
            self.page.run_thread(self._batch_fail, "批量失败：%s" % exc)
        finally:
            self.page.run_thread(self._batch_done_ui)

    def _batch_done_ui(self):
        self.batch_go_btn.disabled = False
        self.batch_zip_btn.disabled = False
        self.batch_prog.value = 1.0
        self.page.update()

    def _batch_fail(self, msg: str):
        self._show_result(msg, is_err=True)

    def _on_batch_done(self, results: list, out_root: str):
        ok_n = sum(1 for r in results if r["ok"])
        rows = []
        for r in results:
            color = ft.Colors.GREEN_400 if r["ok"] else ft.Colors.RED_400
            rows.append(ft.Row([
                ft.Text(r["name"], expand=True, size=12),
                ft.Text("✓ 成功" if r["ok"] else "✗ 失败", color=color, size=12),
                ft.Text(os.path.basename(r["out_file"]) if r["ok"] else "",
                        color=ft.Colors.GREY_400, size=11),
            ], spacing=6))
        self.page.run_thread(self._render_batch, ok_n, len(results), rows)

    def _render_batch(self, ok_n: int, total: int, rows: list):
        self.batch_result.controls.append(
            ft.Text("✅ 批量完成 %d/%d（输出目录：%s）" % (ok_n, total, self._last_batch_dir),
                    color=ft.Colors.GREEN_400, size=13))
        for row in rows:
            self.batch_result.controls.append(row)
        self.page.update()

    def _on_batch_zip(self, e):
        if not self._last_batch_dir:
            self._toast("尚无批量结果")
            return
        try:
            run_action("batch_zip", {"out_dir": self._last_batch_dir}, ui=self)
            self._toast("zip 已生成：%s" % os.path.join(self._last_batch_dir, "rvc_batch.zip"))
        except Exception as exc:  # noqa: BLE001
            self._toast("打包失败：%s" % exc)

    # ------------------------------------------------------------------
    # 其余页签（T4 填充）
    # ------------------------------------------------------------------
    def _page_train(self) -> ft.Control:
        self.train_log_col = ft.Column(
            [self._log_line("— 切分/特征/索引的实时日志将显示在这里 —")],
            spacing=2, scroll=ft.ScrollMode.AUTO)
        return ft.Column([
            ft.Text("训练工作区（T4 迁移控件）", color=ft.Colors.GREY_500),
            ft.Container(
                content=self.train_log_col,
                height=200, padding=10,
                bgcolor=ft.Colors.BLACK_12, border_radius=8,
            ),
        ])

    def _page_tools(self) -> ft.Control:
        return ft.Column([
            ft.Text("模型融合 / 小模型提取 / 后端信息（T4 迁移控件）",
                    color=ft.Colors.GREY_500),
        ])

    def _page_files(self) -> ft.Control:
        return ft.Column([
            ft.Text("文件管理（models/ 目录，T4 迁移控件）", color=ft.Colors.GREY_500),
        ])

    def _page_about(self) -> ft.Control:
        return ft.Column([
            ft.Text("关于本项目", size=15, weight=ft.FontWeight.BOLD),
            ft.Text(
                "RVC-Vulkan 是基于 RVC-Project/Retrieval-based-Voice-Conversion-WebUI 的"
                "第三方个人移植项目：将原版深度依赖 CUDA/PyTorch 的实现完整重写为"
                "纯 numpy + 原生 Vulkan 计算后端。\n\n"
                "两套 UI 并存：网页版（FastAPI + index.html，http://127.0.0.1:8000）"
                "与桌面版（本 Flet 应用）。实时推理 UI 为非网页版（realtime_gui.py）。",
                selectable=True,
            ),
        ])

    # ------------------------------------------------------------------
    # 日志轮询（对齐原版 1s 轮询 /logs?since=N）
    # ------------------------------------------------------------------
    @staticmethod
    def _log_line(text: str) -> ft.Text:
        color = ft.Colors.AMBER_300 if "进度:" in text else \
                ft.Colors.RED_300 if ("❌" in text or "失败" in text) else ft.Colors.WHITE_60
        return ft.Text(text, size=12, font_family="monospace", color=color,
                       selectable=True)
    def _log_ui(self, which: str, line: str):
        """本地即时回显（对齐原版 logLine）。"""
        box = self.infer_log_col if which == "infer" else self.train_log_col
        if box is None:
            return
        box.controls.append(self._log_line(line))
        if len(box.controls) > 600:
            del box.controls[:-400]
        self.page.update()

    def _poll_loop(self) -> None:
        while self._polling:
            try:
                pos, lines = logstream.recent(max(0, self._log_pos))
                if lines:
                    self._log_pos = pos
                    self.page.run_thread(self._append_logs, list(lines))
            except Exception:  # noqa: BLE001
                pass
            threading.Event().wait(POLL_INTERVAL_S)

    def _append_logs(self, lines: list):
        for box in (self.infer_log_col, self.train_log_col):
            if box is None:
                continue
            for line in lines[-20:]:
                box.controls.append(self._log_line(line))
            if len(box.controls) > 600:
                del box.controls[:-400]
        self.page.update()

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._polling = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        self._refresh_health()

    def _refresh_health(self) -> None:
        try:
            info = run_action("health")
            self.health_text.value = "后端: %s" % info["device"]
        except Exception:  # noqa: BLE001
            self.health_text.value = "后端不可用"
        self.page.update()

    def _toast(self, msg: str):
        self.page.show_dialog(ft.SnackBar(ft.Text(msg, size=12)))
        self.page.update()


# ======================================================================
# 命令行入口（headless 调试 + 启动窗口）
# ======================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gui_flet",
        description="RVC-Vulkan Flet 桌面版。无参数启动窗口；"
                    "--action <名> 直接调用按钮动作（headless 调试）。")
    p.add_argument("--action", metavar="NAME", default=None,
                   help="要 headless 调用的动作名（见 --list-actions）")
    p.add_argument("--list-actions", action="store_true",
                   help="列出全部可调用动作并退出")
    for flag, dest, help_ in (
        ("--model", "model", "模型名"),
        ("--speaker-id", "speaker_id", "说话人 ID（int）"),
        ("--pitch", "pitch", "变调半音（int）"),
        ("--audio", "audio", "输入音频路径"),
        ("--f0-method", "f0_method", "F0 方法（rmvpe/pm/fcpe）"),
        ("--index", "index", "特征索引路径（空=自动）"),
        ("--index-rate", "index_rate", "索引率（float）"),
        ("--in-dir", "in_dir", "批量输入目录"),
        ("--out-dir", "out_dir", "批量输出目录"),
    ):
        p.add_argument(flag, default=None, help=help_)
    return p


def _cli_kwargs(args: argparse.Namespace) -> dict:
    kw = {}
    for key in ("model", "speaker_id", "pitch", "audio", "f0_method",
                "index", "index_rate", "in_dir", "out_dir"):
        val = getattr(args, key, None)
        if val is None:
            continue
        if key in ("speaker_id", "pitch"):
            val = int(val)
        elif key == "index_rate":
            val = float(val)
        kw[key] = val
    return kw


def main(page: ft.Page) -> None:
    app = GuiFletApp(page)
    app.build()
    app.start()


def _entry() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list_actions:
        print("可用动作（每个对应一个实体按钮）:")
        for name in sorted(ACTIONS):
            reg = ACTIONS[name]
            print("  %-16s %s — %s" % (name, reg["label"], reg["desc"]))
        return
    if args.action:
        try:
            run_action(args.action, _cli_kwargs(args))
        except KeyError as exc:
            parser.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("动作失败: %s" % exc, file=sys.stderr)
            raise SystemExit(1)
        return
    ft.run(main)


if __name__ == "__main__":
    _entry()