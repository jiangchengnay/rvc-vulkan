# -*- coding: utf-8 -*-
"""RVC checkpoint 处理工具（T48）：``savee`` / ``extract_small_model`` / ``merge``。

对齐 ``train/process_ckpt.py`` 的三个核心功能，但**零 torch 依赖**：读取用
``torch_compat``（纯 Python），数值运算全 numpy，输出为本仓库推理端
（``runtime/vc.py`` / ``runtime/models/vits.py``）可直接加载的 .pth。

三个功能
--------
1. ``savee``：训练权重（``G_{step}.npz`` 或 .pth 底模/检查点）→ 推理 .pth。
   剔除 ``enc_q.*``（推理不需要后验编码器），普通权重按推理端
   ``runtime.models.vits._deweight_norm`` 的逆操作重新参数化为
   ``weight_v=W``、``weight_g=||W||``（逐第 0 维通道 L2 范数，形状 ``[D0,1,1]``，
   与 RVC 官方 checkpoint 的 weight_norm 参数化完全一致）。
2. ``extract_small_model``：训练底模 → 推理 .pth（与 savee 相同的转换逻辑，
   输入取 ``cpt["model"]``）。
3. ``merge``：两个模型按 ``alpha1*w1 + (1-alpha1)*w2`` 逐键融合（``emb_g``
   取 min 行数），config 取 alpha1 权重更大（融合占比更高）的那个模型，
   输出推理格式。

输出格式（决策说明）
--------------------
输出 .pth 是 **zip 容器 + ``data.pkl``（纯 pickle 的 dict-of-ndarray）**：
用标准库 ``zipfile`` + ``pickle`` 写入，``torch_compat.load_pth`` 可完整读回
（其 Unpickler 对 ``numpy.*`` 模块放行，ndarray 原样保留），因此 ``runtime/vc.py``
与 ``runtime/models/vits.py`` 无需任何改动即可加载。相对 safetensors /
torch 官方 zip 布局，这是零新增依赖、改动面最小且数值无损的方案。

config 数组
-----------
从权重形状推断（而非写死）：``upsample_kernels`` 取 ``dec.ups.<i>.weight`` 的
kernel 维，``upsample_rates`` 由 ``dec.noise_convs.<i>.weight`` 的 kernel 反推
（对齐 ``vits.VitsConfig._infer_rates``），``n_spk`` = ``emb_g.weight`` 行数，
``gin`` = 其列数；其余结构字段（spec/segment/hidden/filter/heads/layers 等）
从权重推断或按 RVC 固定布局补齐，``sr`` 由调用方参数决定。

**本模块禁止 import torch / faiss / librosa**；Python 3.10+。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import zipfile

import numpy as np

try:  # 以包方式导入
    from ..models.vits import VitsConfig, _deweight_norm  # noqa: PLC0415
except ImportError:  # 以脚本方式运行（python runtime/train/process_ckpt.py）
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from runtime.models.vits import VitsConfig, _deweight_norm  # noqa: PLC0415

__all__ = [
    "savee",
    "extract_small_model",
    "merge",
    "load_checkpoint_dict",
    "reparam_weight_norm",
    "deparam_weight_norm",
    "infer_config",
    "save_pth",
    "normalize_speaker_info",
]

# 推理端（runtime/models/vits.py _deweight_norm）逐层还原 weight_norm 的键集合。
# 训练/底模中这些层是普通权重 ``X.weight``；转推理格式时拆成
# ``X.weight_v`` + ``X.weight_g``。其余层（emb/Linear/Embedding/noise_convs 等）
# 没有 weight_norm，保持原名。
_WN_LAYER_RE = re.compile(
    r"^(?:"
    r"flow\.flows\.\d+\.enc\.(?:cond_layer|in_layers\.\d+|res_skip_layers\.\d+)"
    r"|dec\.ups\.\d+"
    r"|dec\.resblocks\.\d+\.convs[12]\.\d+"
    r")\.weight$"
)

# 训练产出 npz 中的非权重元数据键（train.py 的 train_main 额外写入 _meta）
_META_KEYS = frozenset({"_meta", "step", "iteration", "learning_rate"})


def normalize_speaker_info(speaker_info):
    """校验并规整说话人信息（对齐原版 process_ckpt.normalize_speaker_info）。"""
    result = []
    seen = set()
    for item in speaker_info or []:
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if speaker_id < 0 or speaker_id > 109 or not speaker_name \
                or speaker_id in seen:
            continue
        seen.add(speaker_id)
        result.append({"id": speaker_id, "name": speaker_name})
    return sorted(result, key=lambda item: item["id"])


def _norm_sr(sr) -> int:
    """把 '48k'/'40k'/'32k' 或 int 归一化为采样率 int。"""
    if isinstance(sr, str):
        s = sr.strip().lower()
        if s.endswith("k"):
            s = s[:-1]
        sr = int(float(s) * 1000)
    sr = int(sr)
    if sr not in (32000, 40000, 48000):
        raise ValueError(f"不支持的采样率 sr={sr}（仅支持 32000/40000/48000）")
    return sr


def load_checkpoint_dict(path) -> dict:
    """读取 .pth / .npz 权重，返回 {参数名: ndarray}（普通权重域优先）。

    - ``.npz``：``np.load`` 的键值（跳过 ``_meta`` 等元数据键）。若为
      weight_norm 参数化训练产物（含 ``weight_v``/``weight_g``，P2 起
      ``train.py`` 直接以推理格式保存），先还原为普通权重。
    - ``.pth``：``torch_compat.load_pth`` 后兼容三种形态：
      ``{"weight": ...}`` / ``{"model": ...}`` / 顶层即 state_dict。
      若为推理格式（含 ``weight_v``/``weight_g``），先还原为普通权重。
    """
    path = os.fspath(path)
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            data = {k: np.asarray(v) for k, v in z.items()
                    if k not in _META_KEYS}
        if any(k.endswith(".weight_v") for k in data):
            data = deparam_weight_norm(data)
        return data
    # .pth（或其它）：torch_compat 读取
    import torch_compat  # noqa: PLC0415  # 项目根需在 sys.path

    cpt = torch_compat.load_pth(path)
    if isinstance(cpt, dict) and "weight" in cpt \
            and isinstance(cpt["weight"], dict):
        w = cpt["weight"]
    elif isinstance(cpt, dict) and "model" in cpt \
            and isinstance(cpt["model"], dict):
        w = cpt["model"]
    elif isinstance(cpt, dict) and "emb_g.weight" in cpt:
        w = cpt
    else:
        raise ValueError(
            f"无法识别 checkpoint 结构：{path}（缺 'weight'/'model' 或顶层权重键）")
    if any(k.endswith(".weight_v") for k in w):
        w = deparam_weight_norm(w)
    return {k: np.asarray(v) for k, v in w.items()}


# ---------------------------------------------------------------------------
# weight_norm 重参数化（与 runtime/models/vits.py 还原公式互为逆操作）
# ---------------------------------------------------------------------------
def _channel_l2_norm(w: np.ndarray) -> np.ndarray:
    """逐第 0 维通道的 L2 范数（与 vits._deweight_norm 的 norm 完全一致）。"""
    w = np.asarray(w, dtype=np.float64)
    norm = np.linalg.norm(w.reshape(w.shape[0], -1), ord=2, axis=1)
    return norm


def reparam_weight_norm(w_in: dict) -> dict:
    """普通权重 dict → 推理权重 dict（weight_norm 层拆成 ``weight_v``/``weight_g``）。

    weight_norm 层（见 ``_WN_LAYER_RE``）：``W`` → ``weight_v = W``、
    ``weight_g = ||W||``（逐第 0 维通道范数，形状 ``[D0,1,1]``，对齐 RVC 官方
    checkpoint 布局）。数值验证：``_deweight_norm(W, ||W||) = W``。
    其余键原样保留。不会修改输入 dict。
    """
    out = {}
    for key, val in w_in.items():
        arr = np.asarray(val)
        if key.endswith(".weight") and _WN_LAYER_RE.match(key):
            norm = _channel_l2_norm(arr)
            g = norm.astype(arr.dtype if arr.dtype in (np.float32, np.float64)
                            else np.float32)
            out[key + "_v"] = arr
            out[key + "_g"] = g.reshape(g.shape[0], 1, 1)
        else:
            out[key] = arr
    return out


def deparam_weight_norm(w_in: dict) -> dict:
    """推理权重 dict（含 weight_v/weight_g）→ 普通权重 dict（还原 W）。

    复用 ``vits._deweight_norm``（与推理端完全一致），``weight_g`` 键合并。
    不修改输入 dict。
    """
    out = {}
    for key, val in w_in.items():
        if key.endswith(".weight_v"):
            base = key[:-9]
            g = w_in.get(base + ".weight_g")
            if g is None:
                raise KeyError(f"缺少 {base + '.weight_g'}，无法还原 weight_norm")
            out[base + ".weight"] = np.array(
                _deweight_norm(np.asarray(val), np.asarray(g)),
                copy=True, order="C")
        elif key.endswith(".weight_g"):
            continue
        else:
            out[key] = np.asarray(val)
    return out


# ---------------------------------------------------------------------------
# config 数组推断
# ---------------------------------------------------------------------------
def _count_attn_layers(w: dict) -> int:
    n = 0
    while f"enc_p.encoder.attn_layers.{n}.conv_q.weight" in w:
        n += 1
    return n


def _infer_upsample_rates(w: dict, upp: int) -> list:
    """从 noise_convs kernel 反推 upsample_rates（对齐 vits._infer_rates）。

    定义 strides[i] = prod(rates[i+1:])（i=0..n-2，strides[n-1]=1）：
    noise_convs.<i>.kernel = strides[i]*2（最后一层 kernel=1 -> stride=1）。
    则 rates[i] = strides[i-1] // strides[i]（i>=1），rates[0] = upp // strides[0]，
    其中 upp = sr//100（48k=480 / 40k=400 / 32k=320）。
    """
    n_ups = 0
    while f"dec.ups.{n_ups}.weight" in w:
        n_ups += 1
    if n_ups == 0:
        raise ValueError("权重中缺少 dec.ups.* 键，无法推断 upsample 结构")
    strides = []
    for i in range(n_ups):
        k = int(np.asarray(w[f"dec.noise_convs.{i}.weight"]).shape[2])
        strides.append(k // 2 if i < n_ups - 1 else 1)
    # strides[i] = prod(rates[i+1:])；rates[i] = strides[i-1]/strides[i] (i>=1)
    rates = [strides[i - 1] // strides[i] for i in range(1, n_ups)]
    rates.insert(0, upp // strides[0])
    return rates


def infer_config(w: dict, sr) -> list:
    """从权重 dict 推断 RVC 推理 checkpoint 的 ``config`` 数组。

    布局（对齐 RVC 推理格式 / vits.VitsConfig）：``[spec, segment, inter,
    hidden, filter, n_heads, n_layers, kernel, p_dropout, resblock,
    resblock_kernels, resblock_dilations, upsample_rates, upsample_initial,
    upsample_kernels, n_spk, gin, sr]``。

    结构字段尽量从权重形状推断（兼容 768d v2 / 256d v1），固定布局字段按
    RVC 标准补齐（segment=32、kernel=3、resblock="1"、[3,7,11] 等）。
    """
    w = {k: np.asarray(v) for k, v in w.items()}
    if "emb_g.weight" not in w:
        raise ValueError("权重缺少 'emb_g.weight'（说话人嵌入），无法推断 n_spk")
    sr = _norm_sr(sr)
    spec = 513 if sr == 32000 else 1025
    # TextEncoder（一律由权重形状确定）
    hidden = int(w["enc_p.emb_phone.weight"].shape[0])
    phone_dim = int(w["enc_p.emb_phone.weight"].shape[1])
    filter_ = int(w["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[0])
    kc = int(w["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[2])
    n_heads = hidden // kc
    n_layers = _count_attn_layers(w)
    # GeneratorNSF
    upsample_initial = int(w["dec.conv_pre.weight"].shape[0])
    n_ups = 0
    while f"dec.ups.{n_ups}.weight" in w:
        n_ups += 1
    upsample_kernels = [int(np.asarray(w[f"dec.ups.{i}.weight"]).shape[2])
                        for i in range(n_ups)]
    expected_upp = sr // 100  # 48k->480 / 40k->400 / 32k->320
    rates = _infer_upsample_rates(w, expected_upp)
    if int(np.prod(rates)) != expected_upp:
        print(
            f"[infer_config] 警告：upsample_rates 乘积 {int(np.prod(rates))} "
            f"≠ 预期 {expected_upp}（sr={sr}）；config 将按实际推断值写入")
    n_spk = int(w["emb_g.weight"].shape[0])
    gin = int(w["emb_g.weight"].shape[1])
    return [
        spec, 32, hidden, hidden, filter_, n_heads, n_layers, 3, 0, "1",
        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        rates, upsample_initial, upsample_kernels, n_spk, gin, sr,
    ]


# ---------------------------------------------------------------------------
# 输出 .pth（zip + pickle dict-of-ndarray）
# ---------------------------------------------------------------------------
def save_pth(opt: dict, path: str, dtype="float32"):
    """写推理 .pth：zip 容器 + ``data.pkl``（纯 pickle dict-of-ndarray）。

    ``torch_compat.load_pth`` 可直接读回（numpy pickle 放行 + ndarray 保留），
    ``runtime/vc.py`` / ``runtime/models/vits.py`` 无需改动即可加载。
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
        opt = dict(opt)
        opt["weight"] = {
            k: np.asarray(v, dtype=dtype) if isinstance(v, np.ndarray) else v
            for k, v in opt["weight"].items()
        }
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.pkl", pickle.dumps(opt, protocol=4))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def _build_infer_opt(w: dict, sr, if_f0, version, info=None,
                     speaker_info=None) -> dict:
    """构造推理 checkpoint dict：剔除 enc_q + 推断 config + weight_norm 重参数化。

    注意：config 推断必须在重参数化**之前**（普通权重域键 ``dec.ups.<i>.weight``
    等），因为重参数化后这些键变成 ``dec.ups.<i>.weight_v``。
    """
    w = {k: np.asarray(v) for k, v in w.items()}
    w = {k: v for k, v in w.items() if "enc_q" not in k}
    config = infer_config(w, sr)
    w = reparam_weight_norm(w)
    opt = {
        "weight": w,
        "config": config,
        "info": info if info is not None else "",
        "sr": _norm_sr(sr),
        "f0": int(if_f0) if if_f0 is not None else 1,
        "version": str(version),
    }
    spk = normalize_speaker_info(speaker_info)
    if spk:
        opt["speaker_info"] = spk
    return opt


# ---------------------------------------------------------------------------
# 三个核心功能
# ---------------------------------------------------------------------------
def savee(inp_path, name, sr, if_f0, version, out_root="assets/weights",
          info=None, speaker_info=None, dtype="float32"):
    """训练权重 → 推理 .pth（对齐原版 ``savee``）。

    Args:
        inp_path: 训练权重路径（``G_{step}.npz``（我们的格式）或 .pth
            （取 ``cpt["model"]``/``cpt["weight"]``，支持底模/训练检查点）。
        name: 输出模型名（自动补 .pth）。
        sr: 采样率（48000/40000/32000，或 "48k" 等字符串）。
        if_f0: 是否音高引导（0/1）。
        version: "v1" / "v2"。
        out_root: 输出目录（默认 assets/weights）。
        info: 可选模型信息字符串。
        speaker_info: 可选说话人列表 [{"id","name"}, ...]。
        dtype: 输出权重 dtype（默认 float32；可传 "float16" 对齐原版 .half()）。

    Returns:
        输出 .pth 绝对路径。
    """
    w = load_checkpoint_dict(inp_path)
    if "enc_p.emb_phone.weight" not in w:
        raise ValueError(f"{inp_path} 不是 RVC 生成器权重（缺 enc_p.emb_phone.weight）")
    if not any(k.startswith(("enc_p.", "flow.", "dec.")) for k in w):
        raise ValueError(f"{inp_path} 中未发现生成器权重键（enc_p/flow/dec）")
    opt = _build_infer_opt(w, sr, if_f0, version, info=info,
                           speaker_info=speaker_info)
    out_path = os.path.join(out_root, name if name.endswith(".pth")
                            else name + ".pth")
    save_pth(opt, out_path, dtype=dtype)
    return os.path.abspath(out_path)


def extract_small_model(inp, out, sr, if_f0, version, info=None,
                        speaker_info=None, dtype="float32"):
    """训练底模 → 推理 .pth（对齐原版 ``extract_small_model``）。

    Args:
        inp: 底模 .pth（取 ``cpt["model"]``；推理格式输入也兼容）。
        out: 输出路径（.pth 结尾直接写；否则视为目录写 ``<out>/<basename>.pth``）。
        sr / if_f0 / version / info / speaker_info / dtype: 同 ``savee``。

    Returns:
        输出 .pth 绝对路径。
    """
    w = load_checkpoint_dict(inp)
    opt = _build_infer_opt(w, sr, if_f0, version, info=info,
                           speaker_info=speaker_info)
    if out.lower().endswith(".pth"):
        out_path = out
    else:
        out_path = os.path.join(out, os.path.basename(inp) + ".pth")
    save_pth(opt, out_path, dtype=dtype)
    return os.path.abspath(out_path)


def merge(model1, model2, alpha1, out, sr=None, if_f0=None, version=None,
          info=None, speaker_info=None, dtype="float32"):
    """两个模型按 ``alpha1*w1 + (1-alpha1)*w2`` 逐键融合，输出推理 .pth。

    Args:
        model1 / model2: 模型路径（.pth 推理格式或底模，或 .npz 训练产出）。
        alpha1: 模型1的融合权重（0~1）。
        out: 输出路径（.pth 结尾直接写；否则视为目录写 ``<out>/<name>.pth``）。
        sr / if_f0 / version: 可选覆盖；缺省继承 alpha1 权重更大的模型的元数据
            （config 亦取该模型；都无 config 时 sr 默认 48000、version 自动推断）。
        info: 可选模型信息字符串。
        speaker_info: 可选说话人列表。
        dtype: 输出权重 dtype（默认 float32）。

    Returns:
        输出 .pth 绝对路径。
    """
    alpha1 = float(alpha1)
    if not 0.0 <= alpha1 <= 1.0:
        raise ValueError(f"alpha1 必须在 [0,1]，实际 {alpha1}")
    cpt1 = _read_cpt_with_meta(model1)
    cpt2 = _read_cpt_with_meta(model2)
    w1 = deparam_weight_norm(cpt1["weight"])
    w2 = deparam_weight_norm(cpt2["weight"])
    w1 = {k: v for k, v in w1.items() if "enc_q" not in k}
    w2 = {k: v for k, v in w2.items() if "enc_q" not in k}
    if sorted(w1.keys()) != sorted(w2.keys()):
        raise ValueError(
            "模型融合失败：两个模型的结构不一致（键集合不同）\n"
            f"仅模型1有: {sorted(set(w1) - set(w2))}\n"
            f"仅模型2有: {sorted(set(w2) - set(w1))}")

    out_w = {}
    for key in w1:
        a = np.asarray(w1[key])
        b = np.asarray(w2[key])
        if key == "emb_g.weight" and a.shape != b.shape:
            min0 = min(a.shape[0], b.shape[0])
            a, b = a[:min0], b[:min0]
        if a.shape != b.shape:
            raise ValueError(
                f"融合失败：键 {key} 形状不一致 {a.shape} vs {b.shape}")
        out_w[key] = (alpha1 * a.astype(np.float32)
                      + (1.0 - alpha1) * b.astype(np.float32))

    # ---- 元数据：取 alpha1 权重更大的那个模型；缺省时继承/推断 ----
    primary = cpt1 if alpha1 >= 0.5 else cpt2
    config = primary.get("config")
    if config is None:
        config = infer_config(out_w, sr if sr is not None else 48000)
    else:
        config = list(config)
    out_w = reparam_weight_norm(out_w)  # 普通域融合完成后统一重参数化
    tgt_sr = int(config[-1]) if config else (sr or 48000)
    ver = version or primary.get("version") or _infer_version(out_w)
    prim_f0 = primary.get("f0")
    f0 = if_f0 if if_f0 is not None else (1 if prim_f0 is None else prim_f0)
    spk = normalize_speaker_info(
        speaker_info if speaker_info is not None
        else (primary.get("speaker_info", []) or cpt2.get("speaker_info", [])))

    opt = {
        "weight": out_w,
        "config": config,
        "info": info if info is not None else primary.get("info", ""),
        "sr": tgt_sr,
        "f0": int(f0),
        "version": str(ver),
    }
    if spk:
        opt["speaker_info"] = spk
    if out.lower().endswith(".pth"):
        out_path = out
    else:
        base = os.path.splitext(os.path.basename(model1))[0]
        out_path = os.path.join(out, f"{base}_mix{alpha1}.pth")
    save_pth(opt, out_path, dtype=dtype)
    return os.path.abspath(out_path)


def _read_cpt_with_meta(path: str) -> dict:
    """读取模型并返回 {"weight": 普通权重域 dict, "config": list|None,
    "version": str|None, "f0": int|None, "info": str|None,
    "speaker_info": list}（推理格式保留元数据）。"""
    import torch_compat  # noqa: PLC0415

    path = os.fspath(path)
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            w = {k: np.asarray(v) for k, v in z.items()
                 if k not in _META_KEYS}
        meta = {}
        if "_meta" in z.files:
            try:
                meta = json.loads(z["_meta"].item())
            except (ValueError, TypeError):
                meta = {}
        return {"weight": w, "config": meta.get("config"),
                "version": meta.get("version"), "f0": None,
                "info": meta.get("info", ""), "speaker_info": []}
    cpt = torch_compat.load_pth(path)
    if isinstance(cpt, dict) and "weight" in cpt \
            and isinstance(cpt["weight"], dict):
        w, cfg, extra = cpt["weight"], cpt.get("config"), cpt
    elif isinstance(cpt, dict) and "model" in cpt \
            and isinstance(cpt["model"], dict):
        w, cfg, extra = cpt["model"], cpt.get("config"), cpt
    elif isinstance(cpt, dict) and "emb_g.weight" in cpt:
        w, cfg, extra = cpt, None, {}
    else:
        raise ValueError(f"无法识别 checkpoint 结构：{path}")
    return {
        "weight": {k: np.asarray(v) for k, v in w.items()},
        "config": list(cfg) if cfg is not None else None,
        "version": extra.get("version"),
        "f0": extra.get("f0"),
        "info": extra.get("info", ""),
        "speaker_info": extra.get("speaker_info", []),
    }


def _infer_version(w: dict) -> str:
    """无 version 时从特征维数推断：768d -> v2，256d -> v1。"""
    dim = int(np.asarray(w["enc_p.emb_phone.weight"]).shape[1])
    return "v2" if dim == 768 else "v1"


def _self_test():
    """冒烟自测：重参数化往返 / config 推断 / pickle zip 读写。"""
    print("=== process_ckpt._self_test ===")
    ok = True
    rng = np.random.RandomState(0)
    # 1) weight_norm 往返：W -> (v,g) -> W
    W = rng.randn(256, 128, 3).astype(np.float32)
    rw = reparam_weight_norm({"flow.flows.0.enc.in_layers.0.weight": W})
    back = deparam_weight_norm(rw)
    err = float(np.abs(back["flow.flows.0.enc.in_layers.0.weight"] - W).max())
    ok &= err < 1e-5
    print(f"  weight_norm 往返误差 {err:.2e} PASS={err < 1e-5}")
    # 2) 非 wn 键原样保留
    rw2 = reparam_weight_norm({"emb_g.weight": W[:, :, 0]})
    ok &= "emb_g.weight" in rw2 and "emb_g.weight_v" not in rw2
    print("  非 wn 键保留 PASS")
    # 3) save_pth 往返
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "ckpt_self_test.pth")
    opt = {"weight": {"a": np.zeros((2, 2), np.float32)},
           "config": [1], "sr": 48000, "f0": 1, "version": "v2"}
    save_pth(opt, tmp)
    import torch_compat  # noqa: PLC0415
    back2 = torch_compat.load_pth(tmp)
    ok &= isinstance(back2["weight"]["a"], np.ndarray)
    ok &= back2["config"] == [1] and back2["version"] == "v2"
    os.remove(tmp)
    print("  save_pth -> torch_compat 往返 PASS")
    print("  process_ckpt._self_test %s" % ("PASS" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(
        description="RVC checkpoint 工具（纯 numpy，零 torch 依赖）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_savee = sub.add_parser("savee", help="训练权重 -> 推理 .pth")
    p_savee.add_argument("inp", help="训练权重（G_*.npz 或 .pth）")
    p_savee.add_argument("name", help="输出模型名（自动补 .pth）")
    p_savee.add_argument("--sr", type=str, default="48k")
    p_savee.add_argument("--f0", type=int, default=1)
    p_savee.add_argument("--version", default="v2")
    p_savee.add_argument("--out", default="assets/weights")
    p_savee.add_argument("--info", default="")
    p_savee.add_argument("--dtype", default="float32")
    p_savee.set_defaults(func=lambda a: print(savee(
        a.inp, a.name, a.sr, a.f0, a.version, out_root=a.out,
        info=a.info or None, dtype=a.dtype)))

    p_ext = sub.add_parser("extract", aliases=["extract_small_model"],
                           help="底模 -> 推理 .pth")
    p_ext.add_argument("inp", help="底模 .pth")
    p_ext.add_argument("out", help="输出 .pth 路径")
    p_ext.add_argument("--sr", type=str, default="48k")
    p_ext.add_argument("--f0", type=int, default=1)
    p_ext.add_argument("--version", default="v2")
    p_ext.add_argument("--info", default="")
    p_ext.add_argument("--dtype", default="float32")
    p_ext.set_defaults(func=lambda a: print(extract_small_model(
        a.inp, a.out, a.sr, a.f0, a.version, info=a.info or None,
        dtype=a.dtype)))

    p_merge = sub.add_parser("merge", help="两模型按 alpha 融合")
    p_merge.add_argument("model1")
    p_merge.add_argument("model2")
    p_merge.add_argument("alpha1", type=float)
    p_merge.add_argument("out", help="输出 .pth 路径")
    p_merge.add_argument("--sr", type=int, default=None)
    p_merge.add_argument("--f0", type=int, default=None)
    p_merge.add_argument("--version", default=None)
    p_merge.add_argument("--info", default="")
    p_merge.add_argument("--dtype", default="float32")
    p_merge.set_defaults(func=lambda a: print(merge(
        a.model1, a.model2, a.alpha1, a.out, sr=a.sr, if_f0=a.f0,
        version=a.version, info=a.info or None, dtype=a.dtype)))

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    _cli()
