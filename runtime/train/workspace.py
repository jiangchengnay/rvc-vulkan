# -*- coding: utf-8 -*-
"""训练工作区管理（D 阶段）：项目/任务目录 + 本项目专用配置。

对应设计：docs/训练工作区设计.md。约定：

    workspaces/<项目名>/rvc-project.json    项目级配置（本项目专用，不魔改原版 config）
    workspaces/<项目名>/<说话人>_<tag>/    每个训练任务文件夹
        ├── exp/          原版 logs/<exp> 语义（0_gt/1_16k/f0/特征/训练产物）
        ├── checkpoints/  G_*.pth / D_*.pth / resume_state.npz / best_* / best_segment_*
        └── out/          savee 输出的成品模型（可复制到 models/）

原版 configs/*.json 只读引用（source_config 字段），一切本项目扩展只写
rvc-project.json——绝不改写原版配置文件结构。

用法::

    from runtime.train import workspace as ws
    ws.create_project("singerA", speakers=[{"id":0,"name":"spk0"}])
    p = ws.load_project("singerA")
    t = ws.create_task("singerA", "spk0", tag="v2")
    t.exp_dir / t.checkpoint_dir / t.out_dir
"""

from __future__ import annotations

import json
import os
import shutil
import uuid

PROJECTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "workspaces",
)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------

DEFAULT_PROJECT = {
    "schema_version": SCHEMA_VERSION,
    "kind": "rvc-vulkan-project",
    "note": "本项目私有配置：扩充但不修改原版 configs/*.json 结构",
    "project": {
        "name": "",
        "created": "",
        "speakers": [],
        "mode": "single",
    },
    "paths": {
        "experiment_root": "",
        "checkpoint_root": "",
        "models_out": "models",
    },
    "training": {
        "source_config": "configs/v2/48k.json",
        "data": {"sample_rate": 48000, "preprocess_per": 3.7},
        "feature": {"f0_method": "rmvpe", "version": 2},
        "params": {"total_epochs": 200, "save_every_epochs": 10, "batch_size": 4},
        "resume": {"enabled": True, "auto_resume_latest": True},
        "best_model": {"strategy": "loss_gen_all", "save_best": True},
    },
    "smart_sampling": {
        "enabled": False,
        "memory_budget_mb": 4096,
        "epoch_segment": 10,
        "keep_segment_best": True,
        "keep_global_best": True,
        "prune_in_memory": True,
        "max_models_in_memory": 8,
        "save_interval_epochs": 1,
    },
}


def _config_path(project_name: str) -> str:
    return os.path.join(PROJECTS_ROOT, project_name, "rvc-project.json")


def create_project(name: str, speakers=None, mode: str = "single",
                   source_config: str = "configs/v2/48k.json",
                   sample_rate: int = 48000) -> dict:
    """新建项目（工作区）并写 rvc-project.json；已存在则直接加载。"""
    name = name.strip()
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("项目名只能含字母数字 _ - ：%r" % name)
    cfg = json.loads(json.dumps(DEFAULT_PROJECT))
    cfg["project"]["name"] = name
    cfg["project"]["created"] = os.environ.get("RVC_PROJECT_DATE", "2026-09-19")
    cfg["project"]["speakers"] = speakers or [{"id": 0, "name": name}]
    cfg["project"]["mode"] = mode
    cfg["training"]["source_config"] = source_config
    cfg["training"]["data"]["sample_rate"] = sample_rate
    root = os.path.join(PROJECTS_ROOT, name)
    os.makedirs(root, exist_ok=True)
    with open(_config_path(name), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def load_project(name: str) -> dict:
    """读取项目配置；缺失抛清晰错误。"""
    p = _config_path(name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            "项目不存在：%s（先 create_project 或检查 workspaces/ 下目录）" % p)
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    # 浅合并默认值（向前兼容新增字段）
    merged = json.loads(json.dumps(DEFAULT_PROJECT))
    _deep_update(merged, cfg)
    return merged


def save_config(name: str, cfg: dict) -> None:
    with open(_config_path(name), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _deep_update(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def list_projects() -> list:
    """列出 workspaces/ 下的项目（按 rvc-project.json 存在性）。"""
    if not os.path.isdir(PROJECTS_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(PROJECTS_ROOT)):
        if os.path.isfile(_config_path(name)):
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# 任务（每个说话人/多说话人合一训练文件夹）
# ---------------------------------------------------------------------------

def task_dir(project_name: str, task_tag: str) -> str:
    return os.path.join(PROJECTS_ROOT, project_name, task_tag)


def create_task(project_name: str, task_tag: str,
                speaker_ids=(0,), tag_desc: str = "") -> dict:
    """在项目下新建一个训练任务文件夹。

    task_tag: 目录名（如 "spk0" / "spk0_v2" / "multi"），须安全字符。
    目录内含 exp/ checkpoints/ out/，并把 task 信息登记到 rvc-project.json。
    """
    cfg = load_project(project_name)
    if not task_tag or not all(c.isalnum() or c in "_-" for c in task_tag):
        raise ValueError("任务名只能含字母数字 _ - ：%r" % task_tag)
    root = task_dir(project_name, task_tag)
    exp = os.path.join(root, "exp")
    ckpt = os.path.join(root, "checkpoints")
    out = os.path.join(root, "out")
    for d in (root, exp, ckpt, out):
        os.makedirs(d, exist_ok=True)
    tasks = cfg.setdefault("tasks", {})
    tasks[task_tag] = {
        "speaker_ids": list(speaker_ids),
        "tag_desc": tag_desc,
        "created": cfg["project"].get("created", "2026-09-19"),
        "exp_dir": exp,
        "checkpoint_dir": ckpt,
        "out_dir": out,
    }
    save_config(project_name, cfg)
    return tasks[task_tag]


def list_tasks(project_name: str) -> dict:
    cfg = load_project(project_name)
    return cfg.get("tasks", {})


def remove_task(project_name: str, task_tag: str, keep_files: bool = False) -> None:
    """删除任务登记（keep_files=False 时连目录一起删）。"""
    cfg = load_project(project_name)
    tasks = cfg.get("tasks", {})
    if task_tag not in tasks:
        raise KeyError("任务不存在：%s" % task_tag)
    del tasks[task_tag]
    save_config(project_name, cfg)
    if not keep_files:
        shutil.rmtree(task_dir(project_name, task_tag), ignore_errors=True)
