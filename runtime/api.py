# -*- coding: utf-8 -*-
"""RVC-Vulkan HTTP API 服务（T51）。

用法：
    python -m runtime.api --port 8000
    # 或 python -c "import uvicorn, runtime.api; uvicorn.run(runtime.api.app, port=8000)"

端点：
    GET  /health                    服务与后端状态
    GET  /models                    可用模型列表
    GET  /speakers?model=xxx        模型说话人列表
    POST /infer                     变声（multipart：audio 文件 + 表单参数）
    POST /preprocess                训练数据预处理（测试用，返回任务开始）
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from . import backend
from . import backend_api as _bapi  # R5 后端抽象层（开关 RVC_BACKEND_API=1 时启用）
from .native_config import Config
from . import logstream

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATIC_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + "static"
_THEMES_DIR = os.path.join(_STATIC_DIR, "themes")
_STATIC_HTML = os.path.join(_STATIC_DIR, "index.html")
# 单次推理输出目录（中文名，用户 2026-09-25 指定）：推理结果落盘到磁盘，
# 前端从 /output/<文件名> 拉取——即使 fetch 长连接失败，文件也已保存，
# 不再丢失结果（根治 "Failed to fetch" 丢结果问题）。
_OUTPUT_DIR = os.path.join(APP_ROOT, "输出")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="RVC-Vulkan", version="0.0.1", description="去 CUDA 化 RVC 变声服务")


# P1-005：全局未捕获业务异常兜底——各路由虽有局部 try/except，但漏网的
# 异常会由 FastAPI 默认处理器返回原始堆栈（信息泄露）。统一收敛为干净的
# 500 JSON，堆栈只进服务端日志（logstream），前端收到可读 message。
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    try:
        logstream.write_line("API 未捕获异常: %s %s -> %s" %
                             (request.method, request.url.path, exc))
    except Exception:  # noqa: BLE001  # 日志失败不影响响应
        pass
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误: %s" % type(exc).__name__},
    )


def _resolve_workspace_exp(project: str = "", task: str = "",
                           exp_dir: str = "") -> str:
    """解析训练数据目录为自研工作区任务 exp（workspaces/<项目>/<任务>/exp）。

    工作区语义（前端主路径）：project+task 必填 → 任务 exp 目录（自动建目录，
    切片/特征/索引/训练产物全部落此处，弃用原版 logs/<exp>）。
    兼容旧调用：exp_dir 为绝对路径时直接使用（脚本直调/旧前端不破坏）。
    全部为空 → 400 并给出人性化提示。
    """
    if project.strip() and task.strip():
        from .train import workspace as ws
        base = os.path.join(ws.PROJECTS_ROOT, project.strip(),
                            task.strip(), "exp")
        os.makedirs(base, exist_ok=True)
        return base
    if exp_dir.strip():
        p = os.path.abspath(exp_dir.strip())
        os.makedirs(p, exist_ok=True)
        return p
    raise HTTPException(
        400, "请先在上方「训练工作区」载入项目并挂载/创建任务，"
             "训练产物会写入工作区任务目录（workspaces/项目/任务/）")


@app.get("/output/{name}", include_in_schema=False)
def output_file(name: str):
    """输出目录静态文件（推理结果 wav）。路径穿越防护。"""
    safe = os.path.normpath(name)
    if safe.startswith("..") or os.path.isabs(safe):
        raise HTTPException(404, "非法文件名")
    fp = os.path.join(_OUTPUT_DIR, safe)
    if not os.path.isfile(fp) or not os.path.realpath(fp).startswith(
            os.path.realpath(_OUTPUT_DIR)):
        raise HTTPException(404, "文件不存在")
    return Response(content=open(fp, "rb").read(), media_type="audio/wav")


@app.get("/outputs", include_in_schema=False)
def list_outputs():
    """输出目录文件列表（按修改时间倒序，供前端展示最近结果）。"""
    if not os.path.isdir(_OUTPUT_DIR):
        return {"files": []}
    files = []
    for n in sorted(os.listdir(_OUTPUT_DIR), key=lambda f: os.path.getmtime(
            os.path.join(_OUTPUT_DIR, f)), reverse=True):
        fp = os.path.join(_OUTPUT_DIR, n)
        if os.path.isfile(fp) and n.lower().endswith(".wav"):
            files.append({"name": n, "url": "/output/" + n,
                          "size": os.path.getsize(fp)})
    return {"files": files}


@app.get("/", include_in_schema=False)
def index():
    """WebUI 首页（原版布局多 Tab 版本，零 gradio，纯 HTML+JS）。"""
    if os.path.isfile(_STATIC_HTML):
        with open(_STATIC_HTML, encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html; charset=utf-8")
    return Response(content="<h1>RVC-Vulkan</h1>", media_type="text/html")


@app.get("/themes", include_in_schema=False)
def list_themes():
    """主题包列表（预留扩展）：扫描 static/themes/ 下的主题目录 + default。"""
    themes = ["default"]
    if os.path.isdir(_THEMES_DIR):
        for name in sorted(os.listdir(_THEMES_DIR)):
            d = os.path.join(_THEMES_DIR, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "theme.css")):
                themes.append(name)
    return {"themes": themes}


@app.get("/themes/{path:path}", include_in_schema=False)
def theme_static(path: str):
    """主题静态文件（default.css 或 <主题>/theme.css 等），路径穿越防护。"""
    safe = os.path.normpath(path)
    if safe.startswith("..") or os.path.isabs(safe):
        return Response(status_code=404)
    fp = os.path.join(_THEMES_DIR, safe)
    if not os.path.isfile(fp) or not os.path.realpath(fp).startswith(
            os.path.realpath(_THEMES_DIR)):
        return Response(status_code=404)
    ctype = "text/css" if fp.endswith(".css") else "application/octet-stream"
    with open(fp, "rb") as f:
        return Response(content=f.read(), media_type=ctype)

# 模型缓存：{模型路径: (VC实例, 锁)}，最多缓存 3 个
_model_cache: dict = {}
_cache_lock = threading.Lock()


def _list_models() -> list:
    """模型列表：根目录 ``models/``（成品模型，新默认）优先，
    ``assets/weights`` / ``assets/pretrained_v2`` / ``assets/pretrained`` 兼容回退。

    models/ 只收 .pth/.safetensors（成品模型）；assets 下额外兼容 .npz
    （旧版自建索引/模型混放）。
    """
    roots = [
        (os.path.join(APP_ROOT, "models"), (".pth", ".safetensors")),
        (os.path.join(APP_ROOT, "assets", "weights"),
         (".pth", ".safetensors", ".npz")),
        (os.path.join(APP_ROOT, "assets", "pretrained_v2"),
         (".pth", ".safetensors")),
        (os.path.join(APP_ROOT, "assets", "pretrained"),
         (".pth", ".safetensors")),
    ]
    models = []
    for root, exts in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.lower().endswith(exts):
                models.append({"name": name, "root": os.path.basename(root),
                               "path": os.path.join(root, name)})
    return models


def _list_indices() -> list:
    """索引列表：根目录 ``models/indices/``（新默认）优先，``assets/indices/``
    + ``logs/`` 下全部 .npz/.ivf.npz/.index（兼容回退，排除 trained）。"""
    roots = [os.path.join(APP_ROOT, "models", "indices"),
             os.path.join(APP_ROOT, "assets", "indices"),
             os.path.join(APP_ROOT, "logs")]
    seen = set()
    items = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for cur, _, files in os.walk(root, topdown=False):
            for name in sorted(files):
                if not name.lower().endswith((".npz", ".index")):
                    continue
                if "trained" in name.lower():
                    continue
                p = os.path.abspath(os.path.join(cur, name))
                if p in seen:
                    continue
                seen.add(p)
                items.append({"name": name, "root": os.path.basename(root),
                              "path": p})
    return items


def _human_size(n: int) -> str:
    """字节数 → 人类可读（B/KB/MB/GB）。"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%.0f%s" if unit == "B" else "%.1f%s") % (n, unit)
        n /= 1024.0
    return "%d B" % n


def _get_vc(model_path: str):
    """按模型路径取缓存的 VC 实例（线程安全，最多缓存 3 个）。"""
    with _cache_lock:
        vc = _model_cache.get(model_path)
        if vc is None:
            if len(_model_cache) >= 3:
                _model_cache.pop(next(iter(_model_cache)))  # 简单 LRU 淘汰
            from .vc import VC
            vc = VC(Config())
            vc.get_vc(model_path)  # 预加载
            _model_cache[model_path] = vc
    return vc


# ---------------------------------------------------------------------------
# R5 后端抽象层（开关 RVC_BACKEND_API=1 时启用；默认分支原样，零行为变更）
# ---------------------------------------------------------------------------

# Backend 实例缓存：{模型路径: Backend}（load 一次；推理参数走 Segment.meta）
_backend_cache: dict = {}
_backend_cache_lock = threading.Lock()


def _get_backend(model_path: str, sid: int):
    """开关开启时取（或建）模型路径对应的 Backend 实例，load 一次。

    profile.ctx 固定放 model/sid；每请求推理参数（pitch/f0_method/index/
    index_rate/...）走 ``Segment.meta``，实现 process 多次无状态复用。
    """
    with _backend_cache_lock:
        backend_obj = _backend_cache.get(model_path)
        if backend_obj is None:
            bid = _bapi.default_backend_id()
            profile = (
                _bapi.profile_vulkan().with_ctx(model=model_path, sid=sid)
                if bid == "vulkan"
                else _bapi.profile_numpy().with_ctx(model=model_path, sid=sid)
            )
            backend_obj = _bapi.get_backend_instance(bid)
            backend_obj.load(profile)
            _backend_cache[model_path] = backend_obj
        return backend_obj


def _infer_via_backend(model_path: str, sid: int, audio16: object,
                       meta: dict) -> tuple:
    """开关开启时的单文件变声（Backend 路径）；返回 (sr, int16 ndarray)。"""
    backend_obj = _get_backend(model_path, sid)
    segment = _bapi.Segment(audio16, 16000, meta=meta)
    with logstream.capture_stdout():
        sr, audio_int16 = backend_obj.process(segment)
    if sr is None or audio_int16 is None:
        raise RuntimeError("变声失败: 后端返回空结果")
    return sr, audio_int16


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": backend.get_backend(),
        "device": backend.device_info(),
    }


@app.get("/index_for_model")
def index_for_model(model: str, speaker_id: int = 0):
    """返回与模型名自动匹配的索引路径（空串=未找到）。

    与推理时的自动匹配逻辑一致（find_index_path_for_model：扫描
    assets/indices + logs，排除 trained、按 added_/v1/v2/_spkidN 匹配）。
    """
    from .vc import find_index_path_for_model
    try:
        path = find_index_path_for_model(model, speaker_id)
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "speaker_id": speaker_id,
                "index": "", "error": str(exc)}
    return {"model": model, "speaker_id": speaker_id, "index": path or ""}


@app.get("/models")
def models():
    return {"models": _list_models()}


@app.get("/logs")
def logs(since: int = 0):
    """实时日志增量拉取：``GET /logs?since=N`` → ``{"pos":…, "lines":[…]}``。

    前端每秒轮询；``since`` 为上次返回的 ``pos``。日志来自进程级环形缓冲
    （最近 2000 行）：推理/批量/预处理/特征/索引/融合执行时的 print 与
    "进度: k/n" 进度行均在其中。
    """
    lines, pos = logstream.recent(max(0, since))
    return {"pos": pos, "lines": lines}


@app.get("/files")
def files():
    """models/ 目录文件管理：成品模型（models/）与索引（models/indices/），
    含大小（字节 + 人类可读）。删除仅由前端标记，不做实际删除（D 阶段再接入）。
    """
    model_items = []
    root = os.path.join(APP_ROOT, "models")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(p) and name.lower().endswith((".pth", ".safetensors")):
                sz = os.path.getsize(p)
                model_items.append({"name": name, "kind": "model",
                                    "size": sz, "size_h": _human_size(sz),
                                    "path": p})
    index_items = []
    indices_root = os.path.join(root, "indices")
    if os.path.isdir(indices_root):
        for name in sorted(os.listdir(indices_root)):
            p = os.path.join(indices_root, name)
            if os.path.isfile(p) and name.lower().endswith((".npz", ".index")):
                sz = os.path.getsize(p)
                index_items.append({"name": name, "kind": "index",
                                    "size": sz, "size_h": _human_size(sz),
                                    "path": p})
    return {"root": root, "models": model_items, "indices": index_items}


def _models_path_safe(rel: str) -> str:
    """把相对路径规范化为 models/ 内的绝对路径；穿越/越界返回 None。"""
    root = os.path.realpath(os.path.join(APP_ROOT, "models"))
    p = os.path.realpath(os.path.join(root, rel.lstrip("/\\")))
    return p if p.startswith(root + os.sep) or p == root else None


@app.post("/files/delete")
def files_delete(path: str = Form(...)):
    """真实删除 models/ 下的文件（模型/索引；仅允许 models/ 内，防穿越）。"""
    p = _models_path_safe(path)
    if p is None or not os.path.isfile(p):
        raise HTTPException(400, "路径非法或不存在: %s" % path)
    try:
        os.remove(p)
        logstream.write_line("文件管理: 删除 %s" % os.path.basename(p))
        return {"status": "ok", "deleted": p}
    except OSError as exc:
        raise HTTPException(500, "删除失败: %s" % exc)


@app.get("/files/download")
def files_download(path: str):
    """下载 models/ 下的文件（模型/索引；仅允许 models/ 内）。"""
    p = _models_path_safe(path)
    if p is None or not os.path.isfile(p):
        raise HTTPException(404, "文件不存在: %s" % path)
    return Response(content=open(p, "rb").read(),
                    media_type="application/octet-stream",
                    headers={"Content-Disposition":
                             "attachment; filename=%s" % os.path.basename(p)})


@app.get("/speakers")
def speakers(model: str):
    found = next((m for m in _list_models() if m["name"] == model), None)
    if found is None:
        raise HTTPException(404, "模型不存在: %s" % model)
    try:
        from .vc import model_speaker_info
        info = model_speaker_info(found["path"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "读取说话人失败: %s" % exc)
    return {"model": model, "speaker_count": info[0], "speakers": info[1]}


@app.post("/infer")
def infer(
    audio: UploadFile = File(...),
    model: str = Form(...),
    speaker_id: int = Form(0),
    pitch: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index: str = Form(""),
    index_rate: float = Form(0.0),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(1.0),
    protect: float = Form(0.33),
    slice_length: float = Form(10),
    retrieval_mode: str = Form("ivf"),
    brute_mix: float = Form(0.0),
    clarity_mix: float = Form(1.0, ge=0.0, le=1.0),
):
    """变声端点：上传音频 + 模型名 → wav 响应。

    ``slice_length``：切分长度（秒，<=0 自动）；默认 10s——长音频切成小块，
    hubert attention 为 O(T²)，小块（T≈1000 帧）比大块（T≈3750 帧）快一个
    量级（150s 音频 ~40min → 预计 ~6-10min）。执行期间进度进入 /logs。
    """
    if audio.content_type not in ("audio/wav", "audio/x-wav", "audio/flac",
                                  "audio/ogg", "audio/mpeg", None):
        # 不限制，soundfile 可解码 wav/flac/ogg/opus 等
        pass
    if f0_method not in ("pm", "rmvpe", "fcpe"):
        raise HTTPException(400, "f0_method 必须是 pm/rmvpe/fcpe")
    if not 0 <= index_rate <= 1:
        raise HTTPException(400, "index_rate 必须在 [0,1]")
    if not 0 <= protect <= 0.5:
        raise HTTPException(400, "protect 必须在 [0,0.5]")
    if retrieval_mode not in ("ivf", "brute"):
        raise HTTPException(400, "retrieval_mode 必须是 ivf/brute")
    brute_mix = max(0.0, min(1.0, brute_mix))  # clamp 到 [0,1]

    found = next((m for m in _list_models() if m["name"] == model), None)
    if found is None:
        raise HTTPException(404, "模型不存在: %s" % model)

    # 保存上传音频到临时文件（VC 接口按路径处理）
    suffix = os.path.splitext(audio.filename or "in.wav")[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        content = audio.file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        vc = _get_vc(found["path"])
        logstream.write_line("推理: 模型=%s spk=%s 变调=%s 索引率=%s f0=%s 清晰度=%s" %
                             (model, speaker_id, pitch, index_rate, f0_method,
                              clarity_mix))
        # 说话清晰度混合（B5 conv_o 投影强度）：mix=1 完整修复（默认）、
        # mix=0 复现旧版口齿不清、中间值线性过渡。
        from .models.vits import set_clarity_mix
        set_clarity_mix(clarity_mix)
        if _bapi.backend_api_enabled():
            # R5 开关路径：走 Backend+profile（与默认分支结果逐位一致）
            from .audio import load_audio as _load_audio
            audio16 = _load_audio(tmp_path, 16000)
            meta = {
                "sid": speaker_id, "f0_up_key": pitch, "f0_method": f0_method,
                "index": index or None, "index_rate": index_rate,
                "resample_sr": resample_sr, "rms_mix_rate": rms_mix_rate,
                "protect": protect, "slice_length": slice_length,
                "retrieval_mode": retrieval_mode, "brute_mix": brute_mix,
                "clarity_mix": clarity_mix,
            }
            sr, audio_int16 = _infer_via_backend(found["path"], speaker_id,
                                                 audio16, meta)
        else:
            with logstream.capture_stdout():
                status, result = vc.vc_single(
                    speaker_id,
                    tmp_path,
                    pitch,
                    f0_method,
                    index or None,
                    index_rate,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                    slice_length=slice_length,
                    retrieval_mode=retrieval_mode,
                    brute_mix=brute_mix,
                )
            if not result or result[0] is None or result[1] is None:
                raise HTTPException(500, "变声失败: %s" % (status or "未知错误"))
            sr, audio_int16 = result
        # int16 → float32，写 wav 到输出目录（结果落盘：即使前端 fetch
        # 连接失败，文件也已保存，不丢结果——根治 "Failed to fetch" 丢结果）
        import numpy as np
        import time as _time
        float_audio = audio_int16.astype(np.float32) / 32768.0
        out_name = "infer_%s.wav" % _time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(_OUTPUT_DIR, out_name)
        from .dsp.audio_io import write_audio
        write_audio(out_path, float_audio, sr)
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        logstream.write_line("推理: 完成（%d Hz，%d 字节）→ %s" %
                             (sr, len(wav_bytes), out_name))
        # 返回 JSON（含文件 URL），不再直接返回大 blob——响应体小、连接层
        # 更稳；前端用 /output/<name> 播放/下载（结果已落盘）。
        return {
            "status": "ok", "sr": sr, "bytes": len(wav_bytes),
            "file": out_name, "url": "/output/" + out_name,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(500, "变声失败: %s" % exc)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/batch")
def batch(
    model: str = Form(...),
    speaker_id: int = Form(0),
    pitch: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index: str = Form(""),
    index_rate: float = Form(0.0),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(1.0),
    protect: float = Form(0.33),
    slice_length: float = Form(10),
    retrieval_mode: str = Form("ivf"),
    brute_mix: float = Form(0.0),
    clarity_mix: float = Form(1.0, ge=0.0, le=1.0),
    files: list[UploadFile] = File(default=[]),
    in_dir: str = Form(""),
    out_dir: str = Form(""),
    export: str = Form(""),
):
    """批量（目录/多文件）变声。

    两种输入模式（可同时给出，files 优先）：
    - ``files``：多文件上传（web 端拖多文件），输出写临时目录；
    - ``in_dir``：服务器端目录路径，输出写 ``out_dir``（默认与 in_dir 同目录
      下的 ``<in_dir>_rvc_out``，或 in_dir 本身）。

    ``export``：
    - ``"zip"``：返回全部成功结果的 zip 字节流（application/zip）；
    - 其它/空：返回 JSON ``{"results": [{name, ok, out_file, error}]}``。

    内部逐文件 vc_single，每文件进度写入实时日志（``进度: k/n`` 为块级、
    ``批量: 文件 i/N`` 为文件级）。
    """
    if f0_method not in ("pm", "rmvpe", "fcpe"):
        raise HTTPException(400, "f0_method 必须是 pm/rmvpe/fcpe")
    if not 0 <= index_rate <= 1:
        raise HTTPException(400, "index_rate 必须在 [0,1]")
    if not 0 <= protect <= 0.5:
        raise HTTPException(400, "protect 必须在 [0,0.5]")
    if retrieval_mode not in ("ivf", "brute"):
        raise HTTPException(400, "retrieval_mode 必须是 ivf/brute")
    brute_mix = max(0.0, min(1.0, brute_mix))  # clamp 到 [0,1]
    found = next((m for m in _list_models() if m["name"] == model), None)
    if found is None:
        raise HTTPException(404, "模型不存在: %s" % model)

    from .vc import AUDIO_EXTENSIONS

    # ---- 收集输入文件（files 优先，其次 in_dir） ----
    inputs = []  # [(原名, 本地临时路径)]
    tmp_upload_dir = None
    if files:
        tmp_upload_dir = tempfile.mkdtemp(prefix="rvc_batch_upload_")
        for up in files:
            name = os.path.basename(up.filename or "input.wav")
            target = os.path.join(tmp_upload_dir, name)
            with open(target, "wb") as f:
                f.write(up.file.read())
            inputs.append((name, target))
    elif in_dir:
        in_dir = os.path.normpath(in_dir)
        if not os.path.isdir(in_dir):
            raise HTTPException(400, "输入目录不存在: %s" % in_dir)
        for name in sorted(os.listdir(in_dir)):
            p = os.path.join(in_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                inputs.append((name, p))
        if not inputs:
            raise HTTPException(400, "输入目录中没有支持的音频文件: %s" % in_dir)
    else:
        raise HTTPException(400, "需要 files 多文件上传或 in_dir 目录参数")

    # ---- 输出目录 ----
    if in_dir and out_dir:
        out_root = os.path.normpath(out_dir)
    elif in_dir:
        out_root = os.path.normpath(in_dir)
    else:
        out_root = os.path.join(tempfile.gettempdir(), "rvc_batch_out")
    os.makedirs(out_root, exist_ok=True)

    vc = _get_vc(found["path"])
    results = []
    total = len(inputs)
    logstream.write_line("批量: 开始（%d 个文件，模型=%s）" % (total, model))
    meta = {
        "sid": speaker_id, "f0_up_key": pitch, "f0_method": f0_method,
        "index": index or None, "index_rate": index_rate,
        "resample_sr": resample_sr, "rms_mix_rate": rms_mix_rate,
        "protect": protect, "slice_length": slice_length,
        "retrieval_mode": retrieval_mode, "brute_mix": brute_mix,
        "clarity_mix": clarity_mix,
    }
    from .models.vits import set_clarity_mix
    set_clarity_mix(clarity_mix)
    try:
        with logstream.capture_stdout():
            for i, (name, path) in enumerate(inputs, 1):
                logstream.write_line("批量: 文件 %d/%d %s" % (i, total, name))
                out_name = os.path.splitext(name)[0] + ".wav"
                out_path = os.path.join(out_root, out_name)
                try:
                    if _bapi.backend_api_enabled():
                        # R5 开关路径：走 Backend+profile（逐文件 Segment）
                        from .audio import load_audio as _load_audio
                        audio16 = _load_audio(path, 16000)
                        sr, audio_int16 = _infer_via_backend(
                            found["path"], speaker_id, audio16, meta)
                    else:
                        status, result = vc.vc_single(
                            speaker_id, path, pitch, f0_method, index or None,
                            index_rate, resample_sr, rms_mix_rate, protect,
                            slice_length=slice_length,
                            retrieval_mode=retrieval_mode,
                            brute_mix=brute_mix,
                        )
                        if not result or result[0] is None or result[1] is None:
                            raise RuntimeError(status or "未知错误")
                        sr, audio_int16 = result
                    import numpy as np
                    from .dsp.audio_io import write_audio
                    float_audio = audio_int16.astype(np.float32) / 32768.0
                    write_audio(out_path, float_audio, sr)
                    results.append({"name": name, "ok": True, "out_file": out_path,
                                    "error": None})
                except Exception as exc:  # noqa: BLE001
                    results.append({"name": name, "ok": False, "out_file": "",
                                    "error": "%s: %s" % (type(exc).__name__, exc)})
        ok_n = sum(1 for r in results if r["ok"])
        logstream.write_line("批量: 完成 %d/%d" % (ok_n, total))
    finally:
        if tmp_upload_dir:
            import shutil
            shutil.rmtree(tmp_upload_dir, ignore_errors=True)

    if export == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                if r["ok"] and os.path.isfile(r["out_file"]):
                    zf.write(r["out_file"], arcname=os.path.basename(r["out_file"]))
        return Response(content=buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition":
                                 'attachment; filename="rvc_batch.zip"'})
    return {"status": "ok", "total": total, "results": results}


@app.post("/preprocess")
def preprocess(
    inp_root: str = Form(...),
    sr: int = Form(48000),
    exp_dir: str = Form(""),
    project: str = Form(""),
    task: str = Form(""),
    per: float = Form(3.7),
    noparallel: bool = Form(False),
):
    """训练数据预处理（同步执行；多进程并行由 preprocess_trainset 内部处理）。

    工作区语义：project+task → workspaces/<项目>/<任务>/exp（自动建目录）。
    exp_dir：兼容旧调用（绝对路径直接用）。切片音频写入任务 exp 的 0_gt_wavs/1_16k_wavs。
    """
    try:
        import multiprocessing
        from .train.preprocess import preprocess_trainset
        n_p = max(1, multiprocessing.cpu_count() // 2)
        exp = _resolve_workspace_exp(project, task, exp_dir)
        logstream.write_line("预处理: inp=%s exp=%s sr=%s" % (inp_root, exp, sr))
        with logstream.capture_stdout():
            result = preprocess_trainset(inp_root, sr, n_p, exp, per=per,
                                         noparallel=noparallel)
        logstream.write_line("预处理: 完成 %s" % result)
        return {"status": "ok", "exp_dir": exp, "detail": result}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "预处理失败: %s" % exc)


@app.post("/merge")
def merge_models(
    model1: str = Form(...),
    model2: str = Form(...),
    alpha1: float = Form(0.5),
    name: str = Form(""),
):
    """模型融合：out = alpha1*model1 + (1-alpha1)*model2（emb_g 取 min 行数）。"""
    if not 0 <= alpha1 <= 1:
        raise HTTPException(400, "alpha1 必须在 [0,1]")
    models = {m["name"]: m["path"] for m in _list_models()}
    for tag in ("model1", "model2"):
        if models.get(locals()[tag]) is None:
            raise HTTPException(404, "模型不存在: %s" % locals()[tag])
    p1, p2 = models[model1], models[model2]
    # 成品模型默认输出到根目录 models/（P3-1 新默认；assets/weights 兼容保留）
    out_dir = os.path.join(APP_ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)
    out_name = (name.strip() or "%s_%s_mix" %
                (os.path.splitext(model1)[0], os.path.splitext(model2)[0]))
    if not out_name.lower().endswith(".pth"):
        out_name += ".pth"
    out_path = os.path.join(out_dir, out_name)
    try:
        from .train.process_ckpt import merge as merge_ckpt
        logstream.write_line("融合: %s x%s + %s x%s" %
                             (model1, alpha1, model2, round(1 - alpha1, 4)))
        with logstream.capture_stdout():
            merge_ckpt(p1, p2, alpha1, out_path)
        logstream.write_line("融合: 完成 %s" % out_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "融合失败: %s" % exc)
    _model_cache.clear()  # 新模型加入，清缓存
    return {"status": "ok", "output": out_path,
            "alpha1": alpha1, "model1": model1, "model2": model2}


@app.post("/extract_features")
def extract_features(
    exp_dir: str = Form(""),
    project: str = Form(""),
    task: str = Form(""),
    f0_method: str = Form("rmvpe"),
    version: int = Form(2),
    n_p: int = Form(1),
):
    """训练特征提取：f0（2a_f0/2b-f0nsf）+ hubert（3_feature256|768）。

    工作区语义：project+task → workspaces/<项目>/<任务>/exp（须已由 /preprocess
    生成 1_16k_wavs）；exp_dir 兼容旧调用（绝对路径）。
    """
    if f0_method not in ("pm", "rmvpe", "fcpe"):
        raise HTTPException(400, "f0_method 必须是 pm/rmvpe/fcpe")
    if version not in (1, 2):
        raise HTTPException(400, "version 必须是 1 或 2")
    exp_dir = _resolve_workspace_exp(project, task, exp_dir)
    if not os.path.isdir(os.path.join(exp_dir, "1_16k_wavs")):
        raise HTTPException(400, "训练数据目录未预处理（缺 1_16k_wavs，请先运行数据切分）：%s" % exp_dir)
    try:
        from .train.extract_f0 import extract_feature_dir
        from .train.extract_hubert import extract_hubert_dir
        logstream.write_line("特征提取: exp=%s f0=%s version=%s" %
                             (exp_dir, f0_method, version))
        with logstream.capture_stdout():
            f0_stats = extract_feature_dir(exp_dir, f0_method, max(n_p, 1), version)
            hb_stats = extract_hubert_dir(exp_dir, version, max(n_p, 1), verbose=False)
        logstream.write_line("特征提取: 完成")
        return {"status": "ok", "exp_dir": exp_dir,
                "f0": f0_stats, "hubert": hb_stats}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "特征提取失败: %s" % exc)


@app.post("/train_index")
def train_index_endpoint(
    exp_dir: str = Form(""),
    project: str = Form(""),
    task: str = Form(""),
    version: int = Form(2),
    mode: str = Form("auto"),
):
    """训练索引：把 3_feature256|768 特征构建为可检索索引（.npz / .ivf.npz）。

    工作区语义：project+task → workspaces/<项目>/<任务>/exp；exp_dir 兼容旧调用。
    """
    exp_dir = _resolve_workspace_exp(project, task, exp_dir)
    feat_dir = "3_feature256" if version == 1 else "3_feature768"
    if not os.path.isdir(os.path.join(exp_dir, feat_dir)):
        raise HTTPException(400, "特征目录不存在: %s（请先运行特征提取）" % feat_dir)
    try:
        from .train.train_index import train_index
        logstream.write_line("索引构建: exp=%s version=%s mode=%s" %
                             (exp_dir, version, mode))
        with logstream.capture_stdout():
            out = train_index(exp_dir, version, n_cpu=max(1, os.cpu_count() // 2),
                              mode=mode)
        logstream.write_line("索引构建: 完成 %s" % out)
        return {"status": "ok", "index": out}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "索引构建失败: %s" % exc)


_train_state = {"running": False, "task": None}


@app.post("/train")
def train_start(
    exp_dir: str = Form(""),
    project: str = Form(""),
    task: str = Form(""),
    steps: int = Form(0),
    epochs: int = Form(0),
    save_every: int = Form(10),
    log_interval: int = Form(10),
    smart_json: str = Form(""),
    resume: str = Form(""),
    out_dir: str = Form(""),
):
    """启动训练（后台线程，日志进 /logs 实时显示）。

    工作区语义：project+task → exp=workspaces/<项目>/<任务>/exp（含特征），
    out_dir 默认=workspaces/<项目>/<任务>/checkpoints（resume_state/G_*.npz/best_*）。
    exp_dir 兼容旧调用；smart_json: rvc-project.json 的 smart_sampling JSON 字符串。
    epochs>0：按"训练轮数"语义（每轮=遍历全部样本一遍，总步数=epochs×
    ceil(样本/batch)，轮数与数据量成反比——数据集越大每轮步数越多轮数越少）；
    epochs=0 且 steps>0：旧语义（直接用总步数）；都缺省时由 train_main
    按 6000 步预算自适应换算轮数。
    """
    global _train_state
    if _train_state["running"]:
        raise HTTPException(409, "已有训练在运行，请等待或停止后再试")
    exp_dir = _resolve_workspace_exp(project, task, exp_dir)
    feat_ok = (os.path.isdir(os.path.join(exp_dir, "3_feature256"))
               or os.path.isdir(os.path.join(exp_dir, "3_feature768")))
    if not feat_ok or not os.path.isdir(os.path.join(exp_dir, "2a_f0")):
        raise HTTPException(400, "特征目录不完整（需 3_feature256|768 与 2a_f0）："
                                "请先运行特征提取")
    if not out_dir.strip():
        if project.strip() and task.strip():
            from .train import workspace as ws
            out_dir = os.path.join(ws.PROJECTS_ROOT, project.strip(),
                                   task.strip(), "checkpoints")
        else:
            out_dir = ""

    smart = {}
    if smart_json.strip():
        try:
            import json as _json
            smart = _json.loads(smart_json)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, "smart_json 不是有效 JSON: %s" % exc)

    def _worker():
        try:
            from .train.train import train_main
            with logstream.capture_stdout():
                r = train_main(exp_dir, epochs=epochs or None,
                               steps=steps, save_every=save_every,
                               log_interval=log_interval, smart=smart or None,
                               resume=resume.strip() or None,
                               out_dir=out_dir.strip() or None)
            logstream.write_line("训练: 完成 | 智能摘要=%s"
                                 % (r.get("smart_summary") if r else None))
        except Exception as exc:  # noqa: BLE001
            logstream.write_line("训练: 失败 %s" % exc)
        finally:
            _train_state["running"] = False

    _train_state["running"] = True
    _train_state["task"] = exp_dir
    import threading as _th
    _th.Thread(target=_worker, daemon=True).start()
    logstream.write_line("训练: 开始 exp=%s 轮数=%s 步数=%s save_every=%d smart=%s"
                         % (exp_dir, epochs or "自动", steps or "自动",
                            save_every, bool(smart)))
    return {"status": "ok", "started": True, "running": True,
            "epochs": epochs, "steps": steps}


@app.post("/ckpt_savee")
def ckpt_savee(
    exp_dir: str = Form(""),
    project: str = Form(""),
    task: str = Form(""),
    sr: int = Form(48000),
    if_f0: int = Form(1),
    version: str = Form("v2"),
    name: str = Form(""),
):
    """训练产物 → 推理模型：任务 checkpoints 下最新的 G_*.npz → 任务 out/ .pth。

    工作区语义：project+task → checkpoints=workspaces/<项目>/<任务>/checkpoints、
    out=workspaces/<项目>/<任务>/out（无训练产物时返回清晰 404）。
    """
    exp_dir = _resolve_workspace_exp(project, task, exp_dir)
    ckpt_dir = os.path.join(os.path.dirname(exp_dir), "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise HTTPException(404, "checkpoints 目录不存在：%s（请先训练）"
                                % ckpt_dir)
    cands = [os.path.join(ckpt_dir, n) for n in os.listdir(ckpt_dir)
             if n.startswith("G_") and n.endswith(".npz")]
    cands.sort(key=os.path.getmtime, reverse=True)
    if not cands:
        raise HTTPException(404, "checkpoints 下没有训练产物 G_*.npz：%s"
                                "（请先训练）" % ckpt_dir)
    inp = cands[0]  # 最新的 G_*.npz
    out_dir = os.path.join(os.path.dirname(exp_dir), "out")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(inp))[0]
    out_name = (name.strip() or "%s_%s" % (base, version)).replace(".pth", "")
    try:
        from .train.process_ckpt import savee
        logstream.write_line("ckpt工具: savee %s → %s/（sr=%s if_f0=%s ver=%s）"
                             % (os.path.basename(inp), out_dir, sr, if_f0,
                                version))
        with logstream.capture_stdout():
            p = savee(inp, out_name, sr, if_f0, version, out_root=out_dir,
                      info="自研训练产物 %s → 推理（%s）" % (os.path.basename(inp),
                                                          version))
        logstream.write_line("ckpt工具: 完成 %s" % p)
        _model_cache.clear()
        return {"status": "ok", "output": p, "source": inp, "out_dir": out_dir}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "转换失败: %s" % exc)


@app.post("/ckpt_extract")
def ckpt_extract(
    model: str = Form(...),
    sr: int = Form(48000),
    if_f0: int = Form(1),
    version: str = Form("v2"),
    name: str = Form(""),
):
    """模型工具：提取小模型（从训练底模/大模型生成推理格式 .pth 到 models/）。"""
    found = next((m for m in _list_models() if m["name"] == model), None)
    if found is None:
        raise HTTPException(404, "模型不存在: %s" % model)
    out_name = (name.strip() or (os.path.splitext(model)[0] + "_small"))
    if not out_name.lower().endswith(".pth"):
        out_name += ".pth"
    out = os.path.join(APP_ROOT, "models", out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        from .train.process_ckpt import extract_small_model
        logstream.write_line("ckpt工具: 提取小模型 %s → %s" % (model, out))
        with logstream.capture_stdout():
            p = extract_small_model(found["path"], out, sr, if_f0, version,
                                    info="WebUI 提取小模型")
        logstream.write_line("ckpt工具: 完成 %s" % p)
        _model_cache.clear()
        return {"status": "ok", "output": p}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "提取失败: %s" % exc)


@app.get("/train_status")
def train_status():
    """训练运行状态（供前端轮询）。"""
    return {"running": _train_state["running"], "task": _train_state["task"]}


@app.get("/projects")
def projects_list():
    """列出训练项目（workspaces/）。"""
    from .train import workspace as ws
    return {"projects": ws.list_projects(), "root": ws.PROJECTS_ROOT}


@app.post("/projects")
def projects_create(
    name: str = Form(...),
    mode: str = Form("single"),
    sample_rate: int = Form(48000),
    speakers: str = Form(""),
):
    """新建训练项目（工作区）。speakers: 格式 "0:名字,1:名字2" 或留空=单说话人同名。"""
    from .train import workspace as ws
    spk = []
    if speakers.strip():
        for part in speakers.split(","):
            part = part.strip()
            if not part:
                continue
            sid, _, sname = part.partition(":")
            spk.append({"id": int(sid.strip()), "name": sname.strip() or ("spk%s" % sid.strip())})
    if not spk:
        spk = [{"id": 0, "name": name}]
    try:
        cfg = ws.create_project(name, speakers=spk, mode=mode, sample_rate=sample_rate)
        return {"status": "ok", "project": cfg["project"], "tasks": ws.list_tasks(name)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "创建项目失败: %s" % exc)


@app.get("/projects/{project_name}")
def projects_get(project_name: str):
    """读取项目配置 + 任务列表。"""
    from .train import workspace as ws
    try:
        cfg = ws.load_project(project_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "项目不存在: %s（%s）" % (project_name, exc))
    return {"project": cfg["project"], "training": cfg["training"],
            "smart_sampling": cfg["smart_sampling"], "tasks": ws.list_tasks(project_name)}


@app.post("/projects/{project_name}/tasks")
def projects_create_task(
    project_name: str,
    task_tag: str = Form(...),
    speaker_ids: str = Form("0"),
    tag_desc: str = Form(""),
):
    """在项目下新建训练任务文件夹（exp/checkpoints/out）。"""
    from .train import workspace as ws
    ids = [int(x.strip()) for x in speaker_ids.split(",") if x.strip()]
    try:
        t = ws.create_task(project_name, task_tag, speaker_ids=ids, tag_desc=tag_desc)
        return {"status": "ok", "task": task_tag, "exp_dir": t["exp_dir"],
                "checkpoint_dir": t["checkpoint_dir"], "out_dir": t["out_dir"]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "创建任务失败: %s" % exc)


@app.post("/projects/{project_name}/config")
def projects_save_config(project_name: str, section: str = Form(...),
                         payload: str = Form(...)):
    """保存项目配置片段（D-3：方案二等参数可保存/读取）。

    section: "smart_sampling" / "training" / 顶层字段名；payload: JSON 字符串。
    仅合并写入本项目专用 rvc-project.json，不改动原版 configs/*.json。
    """
    import json as _json
    from .train import workspace as ws
    try:
        cfg = ws.load_project(project_name)
        data = _json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "参数无效: %s" % exc)
    if section not in ("smart_sampling", "training"):
        raise HTTPException(400, "仅支持保存 smart_sampling / training 配置段")
    cfg.setdefault(section, {})
    if isinstance(data, dict):
        cfg[section].update(data)
    else:
        raise HTTPException(400, "payload 必须是 JSON 对象")
    try:
        ws.save_config(project_name, cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "保存失败: %s" % exc)
    return {"status": "ok", "section": section, "saved": cfg[section]}


if __name__ == "__main__":
    import argparse
    import threading
    import webbrowser

    parser = argparse.ArgumentParser(description="RVC-Vulkan HTTP API + WebUI")
    parser.add_argument("--port", type=int, default=7865, help="监听端口（默认 7865，同原版）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--noautoopen", action="store_true", help="禁止自动打开浏览器")
    args = parser.parse_args()

    if args.open and not args.noautoopen:
        threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:%d" % args.port)).start()
    print("RVC-Vulkan 网页后端：http://127.0.0.1:%d （Ctrl+C 停止）" % args.port)
    import uvicorn
    # timeout_keep_alive：默认 5s 太短——前端 1s 轮询 /logs 与 /infer 主请求
    # 共享连接池时，空闲连接易被服务端提前关闭，浏览器复用死连接 → 前端
    # "Failed to fetch"（后端实际推理成功，纯连接层误报）。调大到 60s 根治。
    # 另设 timeout_graceful_shutdown 正常退出兜底。
    uvicorn.run(app, host=args.host, port=args.port,
                timeout_keep_alive=60,
                timeout_graceful_shutdown=5)