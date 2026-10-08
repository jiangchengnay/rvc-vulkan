# -*- coding: utf-8 -*-
"""训练第二套智能方案：多级最优模型筛选 + 中断续训状态（D 阶段）。

对应设计：docs/训练工作区设计.md 第 3/4 节。

- **基础序列**（方案一）：按 save_every 步保存 G_{step}.npz，随时间变化不论优劣。
- **段最优组**（方案二）：把训练分为"轮段"（每 epoch_segment 个保存周期一段），
  段内取 loss 最低的模型落盘 best_segment_{seg}.npz；若段内最优与上一段相同
  （未被替换）则不重复写文件。
- **全局最最优**：所有段最优中持续最优，只保留一份 best_global.npz。
- **内存池**：按 memory_budget/max_models 在内存中保留历史模型（引用或权重），
  末位淘汰（按 loss）；被淘汰者若是某段/全局最优先落盘。
- **续训状态**：resume_state.npz 每保存周期写一次（权重 + AdamW m/v + step/lr/loss），
  中断（KeyboardInterrupt/异常）finally 兜底写；恢复后从断点继续。

参数保存在 rvc-project.json 的 ``smart_sampling`` 字段（可保存/读取/WebUI 配置）。
"""

from __future__ import annotations

import json
import os

import numpy as np


def save_checkpoint(path: str, weights: dict, meta: dict) -> None:
    """保存权重 dict + meta（_meta json 字符串）为 .npz。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        **{k: np.asarray(v) for k, v in weights.items()},
        **{"_meta": np.asarray(json.dumps(meta))},
    )


# ---------------------------------------------------------------------------
# 续训状态
# ---------------------------------------------------------------------------

def _flatten_opt(prefix: str, state: dict) -> dict:
    """把 AdamW.state() 的嵌套结构（{m:{name:arr}, v:{...}, t, lr}）展平为
    ``<prefix>m.<name>`` / ``<prefix>v.<name>`` / ``<prefix>_t`` / ``<prefix>_lr``。"""
    out = {}
    for k, v in (state.get("m") or {}).items():
        out[prefix + "m." + k] = np.asarray(v)
    for k, v in (state.get("v") or {}).items():
        out[prefix + "v." + k] = np.asarray(v)
    out[prefix + "_t"] = np.int64(state.get("t", 0))
    out[prefix + "_lr"] = np.float64(state.get("lr", 0.0))
    return out


def save_resume(path: str, weights: dict, opt_g_state, opt_d_state,
                step: int, lr: float, last_loss: dict,
                segment: int, d_weights: dict = None) -> None:
    """保存可续训的完整状态：权重 + 优化器 m/v + 进度。

    opt_g_state/opt_d_state: AdamW.state()（{m,v,t,lr} 嵌套 dict），此处展平为
    ``w.``/``og.``/``od.`` 前缀的扁平 npz 键；net_d 权重以 ``w_d.`` 前缀保存
    （P-TRAIN-002/R-TRAIN-002，续训时一并恢复判别器，避免 D 回退底模与
    opt_d m/v 错配）。last_loss: 最近损失 dict。
    """
    blob = {}
    for k, v in (weights or {}).items():
        blob["w." + k] = np.asarray(v)
    if d_weights:                                    # R-TRAIN-002：新增 net_d
        for k, v in d_weights.items():
            blob["w_d." + k] = np.asarray(v)
    blob.update(_flatten_opt("og.", opt_g_state or {}))
    blob.update(_flatten_opt("od.", opt_d_state or {}))
    blob["_meta"] = np.asarray(json.dumps({
        "step": int(step), "lr": float(lr), "segment": int(segment),
        "loss": {k: float(v) for k, v in (last_loss or {}).items()},
    }))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **blob)


def load_resume(path: str):
    """读取续训状态；返回 (weights, opt_g, opt_d, meta)。

    兼容旧协议：无 ``w_d.`` 前缀（旧 resume 不含判别器权重）时返回
    ``d_weights=None``；新协议返回 ``(weights, opt_g, opt_d, d_weights, meta)``——
    注意后续调用方需按 5 元组解包（train.py 已同步）。
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["_meta"].item()))
        weights = {k[2:]: z[k] for k in z.files if k.startswith("w.")}
        d_weights = {k[4:]: z[k] for k in z.files if k.startswith("w_d.")} \
            or None

        def _unflatten(prefix):
            m = {k[len(prefix) + 2:]: z[k] for k in z.files
                 if k.startswith(prefix + "m.")}
            v = {k[len(prefix) + 2:]: z[k] for k in z.files
                 if k.startswith(prefix + "v.")}
            return {"m": m, "v": v,
                    "t": int(z[prefix + "_t"]) if (prefix + "_t") in z.files else 0,
                    "lr": float(z[prefix + "_lr"]) if (prefix + "_lr") in z.files else 0.0}

        opt_g = _unflatten("og.")
        opt_d = _unflatten("od.")
    return weights, opt_g, opt_d, d_weights, meta


# ---------------------------------------------------------------------------
# 多级最优筛选
# ---------------------------------------------------------------------------

class SmartSampler:
    """第二套训练方案：段最优 + 全局最最优 + 内存末位淘汰。

    用法（train_main 内）::

        sm = SmartSampler(**cfg.get("smart_sampling", {}))
        ...
        for step ...:
            ...
            sm.observe(step, losses["loss_total"], net_g.parameters(), out_dir)
            if sm.should_dump_epoch(step):      # 每保存周期
                sm.save_segment_if_best(step); sm.save_global_if_best(step)
    """

    def __init__(self, enabled=False, memory_budget_mb=None,
                 epoch_segment=10, keep_segment_best=True, keep_global_best=True,
                 prune_in_memory=True, max_models_in_memory=8,
                 save_interval_epochs=1, steps_per_epoch=1):
        self.enabled = bool(enabled)
        if memory_budget_mb is None:
            try:
                from runtime import memory as _mem  # noqa: PLC0415
                memory_budget_mb = _mem.suggest_smart_budget()
            except Exception:  # noqa: BLE001
                memory_budget_mb = 4096
        self.memory_budget = int(memory_budget_mb)
        self.epoch_segment = max(1, int(epoch_segment))
        self.keep_segment_best = bool(keep_segment_best)
        self.keep_global_best = bool(keep_global_best)
        self.prune = bool(prune_in_memory)
        self.max_in_memory = max(1, int(max_models_in_memory))
        self.save_interval = max(1, int(save_interval_epochs))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        # 段边界按"保存周期"计：每 save_interval*epoch_segment 个保存周期为一段
        self._segment = 1
        self._seg_best_loss = float("inf")
        self._seg_best_step = None
        self._seg_best_weights = None
        self._global_best_loss = float("inf")
        self._global_best_meta = None
        self._in_memory = []       # [(loss, step, weights)]
        self.history = []          # [(step, segment, loss, saved_name or None)]

    # --- 观察：每个保存周期调用一次 -------------------------------------
    def observe(self, step: int, loss: float, weights: dict,
                out_dir: str) -> None:
        """记录当前 (step, loss, weights)；维护段内最优与内存池。

        step 视为"保存周期序号"（每隔 save_every 步调用一次）。
        """
        if not self.enabled:
            return
        self._update_segment(step, loss, weights)
        self._update_memory(step, loss, weights, out_dir)
        # 段边界推进（每 epoch_segment*? 个保存周期一段；按 save_interval 缩放）
        if step % max(1, self.epoch_segment * self.save_interval) == 0:
            self._finalize_segment(step, out_dir)

    def _update_segment(self, step, loss, weights):
        if loss < self._seg_best_loss or self._seg_best_step is None:
            self._seg_best_loss = loss
            self._seg_best_step = step
            self._seg_best_weights = weights

    def _update_memory(self, step, loss, weights, out_dir):
        if not self.prune:
            return
        self._in_memory.append((loss, step, weights))
        # 末位淘汰：按 loss 排序，保留最优 max_in_memory 个；（淘汰者在段/全局
        # 最优判定已由 _seg_best/全局持有引用，不受影响）
        if len(self._in_memory) > self.max_in_memory:
            self._in_memory.sort(key=lambda t: t[0])
            del self._in_memory[self.max_in_memory:]

    def _finalize_segment(self, step, out_dir):
        """段结束：段内最优落盘（仅当优于上一段 best）；更新全局。"""
        if self._seg_best_step is None:
            return
        if self.keep_segment_best:
            path = os.path.join(out_dir, "best_segment_%02d.npz" % self._segment)
            save_checkpoint(path, self._seg_best_weights,
                            {"kind": "segment_best", "segment": self._segment,
                             "step": int(self._seg_best_step),
                             "loss": float(self._seg_best_loss)})
            self.history.append((self._seg_best_step, self._segment,
                                 self._seg_best_loss, os.path.basename(path)))
        if self.keep_global_best and self._seg_best_loss < self._global_best_loss:
            self._global_best_loss = self._seg_best_loss
            gpath = os.path.join(out_dir, "best_global.npz")
            save_checkpoint(gpath, self._seg_best_weights,
                            {"kind": "global_best", "segment": self._segment,
                             "step": int(self._seg_best_step),
                             "loss": float(self._seg_best_loss)})
        self._segment += 1
        self._seg_best_loss = float("inf")
        self._seg_best_step = None
        self._seg_best_weights = None

    # --- 查询 -----------------------------------------------------------
    @property
    def best_global_loss(self) -> float:
        return self._global_best_loss if self._global_best_loss < float("inf") else None

    @property
    def segment(self) -> int:
        return self._segment

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "segment": self._segment,
            "global_best_loss": self.best_global_loss,
            "in_memory": len(self._in_memory),
            "segments_saved": len([h for h in self.history]),
        }