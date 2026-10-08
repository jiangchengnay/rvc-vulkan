# -*- coding: utf-8 -*-
"""下载并解包 torchfcpe 官方 FCPE 权重到 assets/fcpe/。

FCPE（CFNaiveMelPE）的官方权重 `fcpe_c_v001.pt` 随 PyPI 包 `torchfcpe==0.0.4`
分发（位置：torchfcpe/assets/fcpe_c_v001.pt）。本脚本：

1. 用 ``pip download torchfcpe==0.0.4 --no-deps`` 拉取 wheel（优先清华镜像，
   失败自动回退官方 PyPI）；
2. zipfile 解包 wheel，提取 ``torchfcpe/assets/fcpe_c_v001.pt``；
3. 复制到 ``assets/fcpe/fcpe_c_v001.pt``（纯文件操作，不安装 torch/torchfcpe）。

运行：``python tools/download_fcpe.py``（工作目录 projects/rvc-vulkan）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # projects/rvc-vulkan
DEFAULT_OUT = ROOT / "assets" / "fcpe" / "fcpe_c_v001.pt"

WHEEL_NAME = "torchfcpe==0.0.4"
WHEEL_MEMBER = "torchfcpe/assets/fcpe_c_v001.pt"

MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",  # 清华（默认，快）
    "https://mirrors.aliyun.com/pypi/simple",  # 阿里云备份
    "https://pypi.org/simple",  # 官方
]
HF_URL = (
    "https://huggingface.co/cnchth/fcpe/resolve/main/fcpe_c_v001.pt"
)


def _pip_available() -> bool:
    return shutil.which("pip") is not None or shutil.which("pip3") is not None


def download_wheel_zip(dst_dir: Path) -> Path:
    """pip download wheel 并返回其路径。"""
    pip = "pip" if shutil.which("pip") else "pip3"
    last_err = None
    for mirror in MIRRORS:
        print(f"  [1/3] pip download {WHEEL_NAME} -i {mirror}")
        try:
            r = subprocess.run(
                [pip, "download", WHEEL_NAME, "--no-deps", "-d", str(dst_dir),
                 "-i", mirror, "--disable-pip-version-check"],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                last_err = r.stderr.strip().splitlines()[-3:] if r.stderr else []
                print(f"    pip 失败({mirror}): {last_err}")
                continue
            wheels = [p for p in dst_dir.glob("*.whl") if "torchfcpe" in p.name]
            if not wheels:
                last_err = ["未找到 torchfcpe wheel"]
                continue
            return wheels[0]
        except FileNotFoundError:
            raise SystemExit(
                "未找到 pip；请先安装 pip（或使用含 pip 的 Python 环境）"
            ) from None
        except subprocess.TimeoutExpired:
            last_err = ["pip download 超时"]
            print(f"    pip 超时({mirror})")
            continue
    raise SystemExit(
        f"pip download 全部镜像失败: {last_err or '未知错误'}"
    )


def fetch_hf_direct(dst: Path) -> bool:
    """回退：直接从 HuggingFace 下载已知 URL（cnchth/fcpe）。"""
    import urllib.request

    print(f"  [备用] 尝试 HuggingFace 直连: {HF_URL}")
    try:
        urllib.request.urlretrieve(HF_URL, str(dst))
        if dst.stat().st_size > 1_000_000:
            print("  HF 直连成功")
            return True
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  HF 直连失败: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 FCPE 官方权重到 assets/fcpe/")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="输出 .pt 路径（默认 assets/fcpe/fcpe_c_v001.pt）")
    ap.add_argument("--keep-wheel", action="store_true",
                    help="保留下载的 wheel（默认清理）")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"  [已存在] {out}（{out.stat().st_size:,} B），跳过。"
              f"如需覆盖请删除后重跑。")
        return 0

    with tempfile.TemporaryDirectory(prefix="fcpe_dl_") as tmp:
        tmp = Path(tmp)
        if _pip_available():
            wheel = download_wheel_zip(tmp)
            print(f"    wheel: {wheel.name}（{wheel.stat().st_size // 1024 // 1024} MB）")
            with zipfile.ZipFile(wheel) as zf:
                names = zf.namelist()
                member = next((n for n in names if n.lower().endswith(
                    "fcpe_c_v001.pt")), None)
                if member is None:
                    print(f"    wheel 内未找到 fcpe_c_v001.pt，列出 fcpe 相关项：")
                    for n in names:
                        if "fcpe" in n.lower():
                            print(f"      {n}")
                    return 1
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)
                print(f"  [2/3] 已提取 {member} -> {out}")
        else:
            print("  未找到 pip，改用 HuggingFace 直连")
            out.parent.mkdir(parents=True, exist_ok=True)
            if not fetch_hf_direct(out):
                return 1

    size = out.stat().st_size
    print(f"  [3/3] 完成：{out}（{size / 1024 / 1024:.1f} MB）")

    # 快速校验：顶层结构可读
    try:
        sys.path.insert(0, str(ROOT))
        from torch_compat import load_pth  # noqa: PLC0415
        raw = load_pth(str(out))
        assert {"global_step", "model", "config_dict"} <= set(raw), "顶层键异常"
        assert raw["config_dict"]["model"]["type"] == "CFNaiveMelPE"
        print(f"  [校验] 可读权重：{len(raw['model'])} 个权重键 / "
              f"type={raw['config_dict']['model']['type']} ✓")
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] 权重读取校验失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())