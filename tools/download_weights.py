# -*- coding: utf-8 -*-
"""RVC-Vulkan 模型权重下载脚本（T58）。

从 HuggingFace（默认 hf-mirror.com 镜像）下载推理/训练所需权重到 assets/ 下：

    python tools/download_weights.py [--all | --inference | --training] [--mirror https://hf-mirror.com]

--inference：hubert_base/*、rmvpe.pt（离线推理必需）
--training ：pretrained_v2/{f0G,f0D}48k.pth、mute.zip（训练必需；默认只取 48k f0 版）
--all      ：全部

用法示例：
    python tools/download_weights.py --inference
    HF_ENDPOINT=https://hf-mirror.com python tools/download_weights.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO = "lj1995/VoiceConversionWebUI"
DEFAULT_MIRROR = "https://hf-mirror.com"

FILES_INFERENCE = [
    ("hubert_base/config.json", "assets/hubert_base/config.json"),
    ("hubert_base/preprocessor_config.json", "assets/hubert_base/preprocessor_config.json"),
    ("hubert_base/pytorch_model.bin", "assets/hubert_base/pytorch_model.bin"),
    ("rmvpe.pt", "assets/rmvpe/rmvpe.pt"),
]
FILES_TRAINING = [
    ("pretrained_v2/f0G48k.pth", "assets/pretrained_v2/f0G48k.pth"),
    ("pretrained_v2/f0D48k.pth", "assets/pretrained_v2/f0D48k.pth"),
    ("mute.zip", "assets/pretrained_v2/mute.zip"),  # 解压到 assets/pretrained_v2/mute 请手动 Expand-Archive
]
ALL = FILES_INFERENCE + FILES_TRAINING


def _download(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rvc-vulkan/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print("\r  %s  %.1f%% (%d/%d MB)"
                          % (os.path.basename(dest), done * 100.0 / total,
                             done >> 20, total >> 20), end="", flush=True)
        print()
        os.replace(tmp, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print("  FAIL %s: %s" % (url, exc))
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 RVC-Vulkan 模型权重")
    parser.add_argument("--all", action="store_true", help="下载全部")
    parser.add_argument("--inference", action="store_true", help="仅推理必需")
    parser.add_argument("--training", action="store_true", help="仅训练必需")
    parser.add_argument("--mirror", default=os.environ.get("HF_ENDPOINT", DEFAULT_MIRROR))
    args = parser.parse_args()

    if args.all:
        files = ALL
    elif args.inference:
        files = FILES_INFERENCE
    elif args.training:
        files = FILES_TRAINING
    else:
        files = ALL
    base = args.mirror.rstrip("/")
    ok = 0
    for path, dest in files:
        print("== %s -> %s" % (path, dest))
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print("  已存在，跳过")
            ok += 1
            continue
        if _download("%s/%s/resolve/main/%s" % (base, REPO, path), dest):
            ok += 1
    print("完成：%d/%d 文件" % (ok, len(files)))
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())