# -*- coding: utf-8 -*-
"""RVC 推理命令行入口（T37）——零 torch 依赖。

对齐 ``infer/cli.py`` 的参数与流程，但模型读取用 ``torch_compat``、
推理走 ``runtime``（hubert/rmvpe/vits/pipeline/VC）。

用法示例::

    python -m runtime.cli --model my_model \\
        --input in.wav --output out.wav --pitch 0 --f0-method rmvpe
    python -m runtime.cli --model my_model --input ./audio_dir --output ./out_dir
    python -m runtime.cli --model my_model --list-speakers

参数说明（与原版一致）:
    --model       模型名（assets/weights/<名>.pth，自动补 .pth）或 .pth 路径，必需
    --input       输入音频文件或目录，必需（--list-speakers 时除外）
    --output      输出文件或目录，必需（同上）
    --speaker-id  说话人 ID；多说话人模型默认取声明的最小 ID
    --pitch       移调半音数（默认 0）
    --f0-method   pm | rmvpe（默认 rmvpe）
    --index       .npz 索引路径；缺省时按模型名自动匹配 logs/ 与 assets/indices/
    --index-rate  检索混合率 0~1（默认 0.75；无索引时请设 0）
    --resample-sr 输出采样率（0=模型原生，或 >=16000）
    --rms-mix-rate RMS 包络混合率（默认 1.0）
    --protect     清辅音保护 0~0.5（默认 0.33）
    --format      wav | flac | mp3 | m4a（默认按输出后缀，否则 wav）
    --overwrite   覆盖已存在输出
    --recursive   输入为目录时递归扫描子目录
    --list-speakers 打印说话人列表后退出

输出格式支持: wav/flac 直接写（soundfile 或标准库回退）；mp3/m4a 等容器
格式本版未接入转码，会给出清晰提示。

返回码: 0 全部成功；1 存在失败（或参数/运行时错误）。
"""

from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

from torch_compat import load_pth  # 纯 Python 读 .pth，不依赖 torch

from runtime import backend_api as _bapi  # R5 后端抽象层（开关 RVC_BACKEND_API=1 时启用）
from runtime.dsp.audio_io import write_audio as _write_audio
from runtime.vc import (
    AUDIO_EXTENSIONS,
    PROJECT_ROOT,
    find_index_path_for_model,
    normalized_speaker_info,
)

OUTPUT_FORMATS = {"wav", "flac", "mp3", "m4a"}
# 本版直接支持的写格式（其它格式需要 av 转码，未接入）
DIRECT_FORMATS = {"wav", "flac"}


def build_parser():
    """参数解析（对齐原版 infer/cli.py，--model/--input/--output 语义一致）。"""
    parser = argparse.ArgumentParser(
        description="RVC 离线变声推理（纯 numpy，零 torch/faiss 依赖）。"
    )
    parser.add_argument("--model", required=True, help="模型文件名或 .pth 路径。")
    parser.add_argument("--input", help="输入音频文件或目录。")
    parser.add_argument("--output", help="输出音频文件或目录。")
    parser.add_argument(
        "--speaker-id",
        type=int,
        help="说话人 ID；多说话人模型默认取声明的最小 ID。",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="打印模型说话人列表并退出。",
    )
    parser.add_argument("--pitch", type=int, default=0, help="移调半音数。")
    parser.add_argument(
        "--f0-method", choices=["pm", "rmvpe", "fcpe"], default="rmvpe"
    )
    parser.add_argument(
        "--index",
        help="显式 .npz 索引路径；缺省时按模型名自动匹配。",
    )
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument(
        "--format", dest="output_format", choices=sorted(OUTPUT_FORMATS)
    )
    parser.add_argument(
        "--recursive", action="store_true", help="递归扫描输入子目录。"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_model(value):
    """把 --model 解析为存在的 .pth 绝对路径（对齐原版逻辑）。"""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if not candidate.is_file():
        candidate = (Path(os.environ.get("weight_root", PROJECT_ROOT / "assets" / "weights")) / value).resolve()
    if not candidate.is_file():
        raise FileNotFoundError("Model not found: %s" % value)
    if candidate.suffix.lower() != ".pth":
        raise ValueError("Model must be a .pth file: %s" % candidate)
    return candidate


def load_model_metadata(model_path):
    """用 torch_compat 读取模型元数据：说话人数 + 说话人列表。

    兼容推理格式（weight）与训练底模（model）两种 checkpoint 形态。
    """
    checkpoint = load_pth(str(model_path))
    if not isinstance(checkpoint, dict):
        raise ValueError("模型文件顶层不是 dict，无法读取元数据")
    weight = checkpoint.get("weight") or checkpoint.get("model") or {}
    if not isinstance(weight, dict):
        weight = {}
    embedding = weight.get("emb_g.weight")
    if embedding is None:
        raise ValueError("模型不包含 emb_g.weight（说话人嵌入），不是合法 RVC 模型")
    speaker_count = int(np.asarray(embedding).shape[0])
    speakers = normalized_speaker_info(checkpoint, speaker_count)
    return speaker_count, speakers


def select_speaker(speaker_count, speakers, requested_id):
    """确定说话人 ID（对齐原版：有声明的走声明列表，否则 0~n_spk-1）。"""
    if speakers:
        valid_ids = {item["id"] for item in speakers}
        speaker_id = speakers[0]["id"] if requested_id is None else requested_id
        if speaker_id not in valid_ids:
            raise ValueError(
                "说话人 ID %s 未被该模型声明；可用 ID：%s"
                % (speaker_id, ", ".join(str(v) for v in sorted(valid_ids)))
            )
        return speaker_id
    speaker_id = 0 if requested_id is None else requested_id
    if speaker_id < 0 or speaker_id >= speaker_count:
        raise ValueError(
            "说话人 ID 必须在 0~%s 之间（该模型共 %s 个说话人）"
            % (speaker_count - 1, speaker_count)
        )
    return speaker_id


def resolve_index(value, model_name, speaker_id, index_rate):
    """解析索引路径（对齐原版；.npz/.index + trained->added 替换 + 自动匹配）。

    与 vc_single 的降级语义一致：index_rate>0 但未找到索引时**返回 ""**
    （等效 index_rate=0，不报错、正常转换），并在调用方打印提示。
    """
    if index_rate == 0:
        return ""
    if value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if "trained" in candidate.name:
            candidate = candidate.with_name(candidate.name.replace("trained", "added"))
        index_path = str(candidate)
        if not Path(index_path).is_file():
            raise FileNotFoundError(
                "指定的索引文件不存在：%s（请检查路径）" % candidate)
    else:
        index_path = find_index_path_for_model(model_name, speaker_id)
        if not index_path:
            print("未找到与模型匹配的索引，已自动降级为不检索（等效 --index-rate 0）。")
            return ""
    return str(Path(index_path).resolve())


def collect_jobs(input_value, output_value, output_format, recursive):
    """把输入/输出组织为 ``[(in_path, out_path, fmt), ...]``（对齐原版）。"""
    input_path = Path(input_value).expanduser().resolve()
    output_path = Path(output_value).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError("输入不存在: %s" % input_path)

    if input_path.is_file():
        suffix_format = output_path.suffix.lower().lstrip(".")
        if output_path.suffix and suffix_format not in OUTPUT_FORMATS:
            raise ValueError("不支持的输出后缀: %s" % output_path.suffix)
        selected_format = output_format or suffix_format or "wav"
        if output_path.suffix:
            output_file = output_path.with_suffix(".%s" % selected_format)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / (input_path.stem + "." + selected_format)
        return [(input_path, output_file, selected_format)]

    selected_format = output_format or "wav"
    output_path.mkdir(parents=True, exist_ok=True)
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    jobs = []
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative_parent = path.relative_to(input_path).parent if recursive else Path()
        output_file = output_path / relative_parent / (path.stem + "." + selected_format)
        jobs.append((path, output_file, selected_format))
    if not jobs:
        raise ValueError("输入目录中没有支持的音频文件: %s" % input_path)
    return jobs


def write_audio(path, audio_int16, sample_rate, output_format):
    """写输出音频。

    audio_int16 为 pipeline 返回的 int16；写盘前转回 float[-1,1]（write_audio
    约定：输入一律按 float 幅度处理，不识别 int16 编码）。
    wav/flac 直接写；其它格式给出清晰提示（av 转码未接入）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format not in DIRECT_FORMATS:
        raise RuntimeError(
            "输出格式 %s 暂不支持：本版直接支持 wav/flac；"
            "mp3/m4a 等容器转码（需 av 包）未接入。请改用 --format wav（或 flac）。"
            % output_format
        )
    audio = np.asarray(audio_int16, dtype=np.float32) / 32768.0
    _write_audio(str(path), audio, sample_rate)


def create_config():
    """构造 runtime 配置（临时清空 sys.argv，避免 Config 内部的 argparse 干扰）。"""
    from runtime.native_config import Config

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        return Config()
    finally:
        sys.argv = original_argv


def main(argv=None):
    """CLI 主流程；返回退出码（0 成功 / 1 失败）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_speakers and (not args.input or not args.output):
        parser.error("--input 和 --output 必需（--list-speakers 模式除外）")
    if not 0 <= args.index_rate <= 1:
        parser.error("--index-rate 必须在 0~1 之间")
    if not 0 <= args.rms_mix_rate <= 1:
        parser.error("--rms-mix-rate 必须在 0~1 之间")
    if not 0 <= args.protect <= 0.5:
        parser.error("--protect 必须在 0~0.5 之间")
    if args.resample_sr and args.resample_sr < 16000:
        parser.error("--resample-sr 必须为 0 或 >=16000")

    model_path = resolve_model(args.model)
    os.environ["weight_root"] = str(model_path.parent)
    model_name = model_path.name
    speaker_count, speakers = load_model_metadata(model_path)
    if args.list_speakers:
        if speakers:
            for item in speakers:
                print("%s\t%s" % (item["id"], item["name"]))
        else:
            print("0-%s" % (speaker_count - 1))
        return 0

    speaker_id = select_speaker(speaker_count, speakers, args.speaker_id)
    index_path = resolve_index(args.index, model_name, speaker_id, args.index_rate)
    jobs = collect_jobs(args.input, args.output, args.output_format, args.recursive)
    existing = [str(output) for _, output, _ in jobs if output.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出文件已存在，使用 --overwrite 覆盖: %s" % existing[0]
        )

    config = create_config()
    print("当前设备：%s | 推理精度：%s" % (config.device, config.dtype))
    print("选择模型: %s" % model_name)
    print("说话人ID: %s" % speaker_id)
    print("选择索引: %s" % (index_path or "未使用"))

    if _bapi.backend_api_enabled():
        # R5 开关路径：走 Backend+profile（--model/--index 等参数映射进 profile）
        bid = _bapi.default_backend_id()
        profile = (_bapi.profile_vulkan() if bid == "vulkan"
                   else _bapi.profile_numpy())
        profile = profile.with_ctx(
            model=str(model_path), sid=speaker_id,
            index=index_path or None, index_rate=args.index_rate,
            f0_method=args.f0_method, resample_sr=args.resample_sr,
            rms_mix_rate=args.rms_mix_rate, protect=args.protect,
            f0_up_key=args.pitch,
        )
        backend_obj = _bapi.get_backend_instance(bid)
        backend_obj.load(profile)
        print("Backend: %s | policy=%s | profile=%s"
              % (backend_obj.backend_id, backend_obj.policy(),
                 profile.profile_name or profile.backend_id))
        failed = 0
        for input_path, output_path, output_format in jobs:
            from runtime.audio import load_audio as _load_audio  # noqa: PLC0415
            audio16 = _load_audio(str(input_path), 16000)
            try:
                sr, audio_int16 = backend_obj.process(
                    _bapi.Segment(audio16, 16000))
            except Exception as exc:  # noqa: BLE001  # 对齐默认分支：失败计数继续
                print("rvc-cli: 后端推理失败: %s" % exc, file=sys.stderr)
                failed += 1
                continue
            print("转换成功")
            write_audio(output_path, audio_int16, sr, output_format)
            print(str(output_path))
        return 1 if failed else 0

    from runtime.vc import VC

    vc = VC(config)
    info = vc.get_vc(model_name)
    if not info["success"]:
        print("模型加载失败: %s" % info["error"], file=sys.stderr)
        if info.get("traceback"):
            print(info["traceback"], file=sys.stderr)
        return 1

    failed = 0
    for input_path, output_path, output_format in jobs:
        status, result = vc.vc_single(
            speaker_id,
            str(input_path),
            args.pitch,
            args.f0_method,
            index_path,
            args.index_rate,
            args.resample_sr,
            args.rms_mix_rate,
            args.protect,
        )
        print(status)
        if not result or result[0] is None or result[1] is None:
            failed += 1
            continue
        write_audio(output_path, result[1], result[0], output_format)
        print(str(output_path))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print("rvc-cli: error: %s" % error, file=sys.stderr)
        raise SystemExit(1)
