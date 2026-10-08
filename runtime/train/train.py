# -*- coding: utf-8 -*-
"""纯 numpy 的 RVC 训练循环简化版（T46）。

包含：
    - ``AdamW``：params_dict 为 ``{name: ndarray}``（原地更新），与 RVC 训练
      hps（lr=1e-4、betas=(0.8,0.99)、eps=1e-9、weight_decay=0）一致；
    - ``train_step``：单步训练 —— D 步（LSGAN，判别器权重从底模加载并同步
      训练）+ G 步（gen loss (1-dg)²、feature loss ×2、mel loss ×45、
      KL ×1.0），反向全部走 ``runtime/nn_backward`` + 训练版模型的 tape；
      KL/mel 核对原版 losses.py / train.py 的公式与系数；
    - ``train_main``：简化训练入口 —— 读 ``workspaces/<项目>/<任务>/exp`` 的 0_gt_wavs +
      2a_f0/2b-f0nsf + 3_feature768，特征对齐（np.repeat 2 倍 → 100Hz、
      截断 ≤900 帧）、随机 segment 切片、前向/损失/反向/AdamW 更新，
      每 N 步打印 loss，保存 ``G_{step}.npz``（权重 dict，供 process_ckpt）。

损失公式核对（与 train/train.py + train/losses.py 逐一对照）：
    - discriminator_loss: r_loss = mean((1-dr)²), g_loss = mean(dg²)（LSGAN）；
    - generator_loss: mean((1-dg)²)；
    - feature_loss: Σ mean(|fmap_r - fmap_g|) * 2；
    - mel loss: L1(y_mel, y_hat_mel) * c_mel(=45)；
    - kl: kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl(=1.0)，公式为仓库
      实际实现（logs_p - logs_q - 0.5 + 0.5*(z_p-m_p)²*exp(-2logs_p)）。

**本模块禁止 import torch**。
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from runtime.models.vits_train import (
    AutogradTape,
    SynthesizerTrnTrain,
    MultiPeriodDiscriminator,
    load_g_weights,
    load_d_weights,
)
from runtime.train.mel_processing import mel_spectrogram_torch

# KL 溯源插桩开关（env RVC_TRAIN_KLTRACE，默认关 ⇒ 零开销零回归）。
# 用于判别 KL 长程上升机制：logs_p 先验方差收缩 vs (z_p-m_p)² 估计器噪声。
_KLTRACE = os.environ.get("RVC_TRAIN_KLTRACE", "") == "1"

# T4 自适应模式切换（自适应策略实现，子代理1）：启动时检测显存自动选模式。
# RVC_ADAPTIVE=0 完全关闭（行为与现状一致）；>4G 零注入走正常模式；
# ≤4G 注入 batch=1 + 池关（判别器单缓冲为待办）。详见 runtime/train/adaptive.py。
from runtime.train.adaptive import apply_adaptive_mode, monitor_step

try:  # 以包方式导入
    from ..dsp.audio_io import load_audio  # noqa: PLC0415
except ImportError:  # 以脚本运行
    from runtime.dsp.audio_io import load_audio  # noqa: PLC0415

__all__ = ["AdamW", "SamplingConfig", "train_step", "train_main",
           "load_samples", "build_pretrained"]


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------
class AdamW:
    """AdamW（decoupled weight decay）：params_dict 原地更新。

    Args:
        params_dict: {name: ndarray}（模型权重，直接可写）。
        lr / betas / eps / weight_decay: 默认与 RVC hps（48k.json train 节）。
    """

    def __init__(self, params_dict, lr=1e-4, betas=(0.8, 0.99), eps=1e-9,
                 weight_decay=0.0):
        self.params = params_dict
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.m = {k: np.zeros_like(v, dtype=np.float32) for k, v in
                  params_dict.items()}
        self.v = {k: np.zeros_like(v, dtype=np.float32) for k, v in
                  params_dict.items()}
        self.t = 0
        # T4-2：优化器图化（env RVC_TRAIN_GRAPH_OPT=1 → AdamW 单图执行；
        # 默认 numpy 零回归）。首次 step 收到 grads_dict 时懒构建——只含
        # 实际有梯度的参数（tape.grad_of 非 None 的键）；无梯度参数保持
        # numpy 语义「跳过不更新」（215/560 结构性无梯度，如 m_source/
        # emb/attn conv）。图构建需引擎已初始化（train_main 中安全）。
        self._ag = None
        self._ag_full_p = False  # T4-3：p_slots 全覆盖标记（wpers_refresh 跳过条件）
        # T6：多线程 AdamW（env RVC_TRAIN_ADAM_THREADS，默认 0 = 单线程零回归）。
        # numpy ufunc 在大数组上释放 GIL，判别器/生成器权重更新可并行
        # （每个参数独立运算，线程间无共享写 → 数值与单线程 bit-exact）。
        self._nt = int(os.environ.get("RVC_TRAIN_ADAM_THREADS", "0") or 0)
        self._pool = None

    def zero_grad(self):
        pass  # 梯度以 dict 传入，无状态累积

    def _sync_gpu(self):
        """T4-3：图化外部 p 槽时，把 GPU 最新 p 同步回 numpy params
        （save/续训前调用；m/v 由 step 每步下载，已最新）。"""
        if self._ag is None:
            return
        gr = self._ag.gr
        for name in self._ag.names:
            if name in self._ag.p_ext:
                p_buf = self._ag.slots[name][0]
                np.copyto(self.params[name],
                          gr.ctx.download(p_buf, self.params[name].shape))

    def step(self, grads_dict):
        """grads_dict: {name: ndarray}（与 params 同形状）。"""
        self.t += 1
        # 梯度裁剪 + 范数日志（防训练发散；env 门控，默认关零回归）。
        # 背景：纯基线训练在 60~100 步出现 d/g 爆炸（d=1245/g=624）后
        # dec 输出窄带化（"蚊子叫"）。全局范数裁剪可兜住尖峰。
        _clip = float(os.environ.get("RVC_TRAIN_GRAD_CLIP", "0") or 0.0)
        _glog = os.environ.get("RVC_TRAIN_GRAD_LOG", "0") == "1"
        if (_clip > 0.0 or _glog) and grads_dict:
            _sq = 0.0
            for _g in grads_dict.values():
                if _g is not None:
                    _sq += float(np.sum(np.asarray(_g, dtype=np.float64) ** 2))
            _norm = float(np.sqrt(_sq))
            if _glog:
                print("[gradnorm] t=%d |g|=%.4f" % (self.t, _norm), flush=True)
            if _clip > 0.0 and _norm > _clip:
                _sc = _clip / (_norm + 1e-9)
                grads_dict = {_k: (np.asarray(_g, dtype=np.float32) * _sc
                                   if _g is not None else None)
                              for _k, _g in grads_dict.items()}
        if self._ag is None and os.environ.get("RVC_TRAIN_GRAPH_OPT") == "1":
            # 懒构建：只用有梯度的键（numpy 语义对齐——无梯度参数跳过）
            try:
                from runtime import graph_runner as _gr  # noqa: PLC0415
                from runtime.models import vits_train as _vt  # noqa: PLC0415
                gr = _vt._graph_runner()
                sub = {k: self.params[k] for k in grads_dict
                       if k in self.params}
                if sub:
                    # T4-3: p 更新目标 = wpers GPU 常驻槽（fwd 复用，p
                    # 免下载/免 wpers_refresh 重传）。仅当**全部**参数都有
                    # 常驻槽才启用（部分覆盖时 wpers_refresh 需照常刷新
                    # 无槽参数的上传/常驻——整体 skip 会 stale，回退 T4-2）。
                    from runtime import vulkan_ops as _vo  # noqa: PLC0415
                    p_slots = {}
                    for _k in sub:
                        _bid = _vo.wpers_slot_for(sub[_k])
                        if _bid is not None:
                            p_slots[_k] = _bid
                    self._ag_full_p = len(p_slots) == len(sub)
                    if not self._ag_full_p:
                        p_slots = None
                    self._ag = _gr.build_adamw_graph(
                        gr, sub, self.m, self.v,
                        self.lr, self.b1, self.b2, self.eps, self.wd,
                        p_slots=p_slots)
            except Exception as _e:  # noqa: BLE001
                import traceback as _tb  # noqa: PLC0415
                print(f"[AdamW] 图化构建失败，回退 numpy: {_e}")
                _tb.print_exc()
                self._ag = None
        if self._ag is not None:
            # T4-2：图化路径（单图执行；p 覆写回 params_dict）
            new_p = self._ag.step(grads_dict, self.t)
            for name, arr in new_p.items():
                np.copyto(self.params[name], arr)
            return
        b1, b2 = self.b1, self.b2
        mhat_denom = 1.0 - b1 ** self.t
        vhat_denom = 1.0 - b2 ** self.t
        names = list(self.params.keys())
        if self._nt >= 2 and len(names) >= 16:
            # T6：多线程 AdamW。每个参数独立更新（线程间无共享写），
            # numpy ufunc 在大数组上释放 GIL → 真并行。数值 bit-exact。
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
                self._pool = ThreadPoolExecutor(max_workers=self._nt)
            chunks = [names[i::self._nt] for i in range(self._nt)]
            list(self._pool.map(
                lambda c: self._adam_update(c, grads_dict,
                                            b1, b2, mhat_denom, vhat_denom),
                chunks))
        else:
            self._adam_update(names, grads_dict, b1, b2, mhat_denom, vhat_denom)

    def _adam_update(self, names, grads_dict, b1, b2, mhat_denom, vhat_denom):
        """AdamW 单步参数子集更新（线程内调用；每参数独立）。"""
        for name in names:
            p = self.params[name]
            g = grads_dict.get(name)
            if g is None:
                continue
            g = np.asarray(g, dtype=np.float32)
            if p.shape != g.shape:
                raise ValueError(f"AdamW: 参数 {name} 形状 {p.shape} != 梯度 {g.shape}")
            if self.wd:
                p *= 1.0 - self.lr * self.wd
            m = self.m[name]
            v = self.v[name]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            mhat = m / mhat_denom
            vhat = v / vhat_denom
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def lr_now(self):
        return self.lr

    def state(self) -> dict:
        """优化器状态（m/v/t/lr），供续训保存。"""
        return {"m": self.m, "v": self.v, "t": int(self.t), "lr": float(self.lr)}

    def load_state(self, st: dict) -> None:
        """恢复优化器状态（键与 state() 一致）。"""
        if not st:
            return
        m, v = st.get("m"), st.get("v")
        if m:
            for k in m:
                if k in self.m and m[k].shape == self.m[k].shape:
                    self.m[k][:] = m[k]
        if v:
            for k in v:
                if k in self.v and v[k].shape == self.v[k].shape:
                    self.v[k][:] = v[k]
        self.t = int(st.get("t", self.t))
        self.lr = float(st.get("lr", self.lr))
        return self.lr


# ---------------------------------------------------------------------------
# 采样配置
# ---------------------------------------------------------------------------
class SamplingConfig:
    """训练超参与 mel 参数（默认 48k v2，与 configs/v2/48k.json 一致）。"""

    def __init__(self, sr=48000, n_fft=2048, hop=480, win=2048, n_mels=128,
                 fmin=0.0, fmax=None, segment_size=17280, c_mel=45.0,
                 c_kl=1.0, max_frames=900, version="v2",
                 # ---- T1.2 数据侧 T 锁定（方案 A，_diag/replay_t12.md）----
                 lock_frames=368, lock_mode="padcrop"):
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = fmax
        self.segment_size = int(segment_size)          # 采样点数（17280）
        self.segment_frames = self.segment_size // self.hop  # 帧数（36）
        self.c_mel = float(c_mel)
        self.c_kl = float(c_kl)
        self.max_frames = int(max_frames)
        self.version = version
        # ---- T1.2 数据侧 T 锁定（方案 A，_diag/replay_t12.md §3.2）--------
        # T0 = 目标帧数（**模型帧口径** = aligned() 的 n_min，非 feature 帧数）。
        #   0   = 关闭（完全走原路径，位级零回归的逃生门）
        #   368 = 实测 354/361 = 98.1% 样本的天然 n_min（模型帧口径）
        #   口径桥梁：feature 帧经 train.py:337 repeat(2) 得 phone 帧；
        #             n_min = min(2*feature, F=spec.shape[-1]=370)。
        #             feature=184 -> 368（phone 侧胜出）；feature=189 -> min(378,370)=370（spec 侧胜出）。
        self.lock_frames = int(lock_frames or 0)
        # "padcrop"（先裁后补，默认）| "pad"（只补）| "crop"（只裁）
        # padcrop 的理由（R9）：max_frames 只截断不补齐（train.py:340-341），
        # 纯 pad 会放过 n_min>max_frames 的长样本 => 锁定失效且 [P,P] 常量槽
        # 随 P=2*T0 平方膨胀（T0=368 => 736^2 = 541,696 元素 ≈ 2.17 MB）。
        self.lock_mode = str(lock_mode)
        if self.lock_frames > 0:
            _t0_min = self.segment_frames + 1          # = 37
            if self.lock_frames < _t0_min:
                # 否则 aligned() 对所有样本 return None => 训练静默空转
                raise ValueError(
                    f"lock_frames={self.lock_frames} < segment_frames+1="
                    f"{_t0_min}：所有样本都会被 aligned() 跳过")
            if self.lock_frames > int(self.max_frames):
                # 否则 max_frames 截断先于锁定生效 => 锁定不变量被破坏（R10）
                # 默认参数下永不触发（368 < 900）
                raise ValueError(
                    f"lock_frames={self.lock_frames} > max_frames="
                    f"{self.max_frames}：max_frames 截断会破坏锁定不变量")
        if self.lock_mode not in ("pad", "crop", "padcrop"):
            raise ValueError(f"未知 lock_mode={self.lock_mode!r}")
        # 模型结构 config 数组（对齐 VitsConfig 解析 / process_ckpt 布局）
        self.model_config = [
            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ([12, 10, 2, 2] if self.segment_size % 480 == 0
             and self.segment_size >= 17280 else [10, 6, 2, 2, 2]),
            512,
            ([24, 20, 4, 4] if self.segment_size >= 17280 else [16, 16, 4, 4, 4]),
            109, 256, self.sr,
        ]


# ---------------------------------------------------------------------------
# 数据加载（工作区 exp 目录，含 0_gt_wavs / 1_16k_wavs / 2a_f0 / 2b-f0nsf / 3_feature768）
# ---------------------------------------------------------------------------
def _stem_candidates(stem: str):
    """按 RVC 目录惯例尝试匹配（去掉 48k/40k/32k 后缀）。"""
    cands = [stem]
    for suf in ("48k", "40k", "32k"):
        if stem.endswith(suf):
            cands.append(stem[:-len(suf)])
    return cands


def load_samples(exp_dir, version="v2"):
    """扫描实验目录，返回样本记录列表 + 预计算 spec/mel 缓存。

    每条记录::
        {feature, f0, f0nsf, wav, spec (ndarray [1, 1025, F]),
         spec_mel (ndarray [1, n_mels, F])}
    """
    gt_dir = os.path.join(exp_dir, "0_gt_wavs")
    feat_dir = os.path.join(exp_dir, "3_feature768")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    if not os.path.isdir(gt_dir) or not os.path.isdir(feat_dir):
        raise FileNotFoundError(
            f"训练数据目录不完整（需要 0_gt_wavs 与 3_feature768）：{exp_dir}")

    wavs = sorted(glob.glob(os.path.join(gt_dir, "*.wav")))
    feat_files = {}
    for p in glob.glob(os.path.join(feat_dir, "*.npy")):
        b = os.path.basename(p)[:-4]      # 去 .npy
        if b.endswith(".wav"):
            b = b[:-4]                    # 兼容 <key>.wav.npy 命名（extract_hubert）
        feat_files.setdefault(b, p)        # 与 train_index._feature_stem 语义一致
    f0_files = {os.path.basename(p)[:-8]: p
                for p in glob.glob(os.path.join(f0_dir, "*.wav.npy"))}
    f0n_files = {os.path.basename(p)[:-8]: p
                 for p in glob.glob(os.path.join(f0nsf_dir, "*.wav.npy"))}

    samples = []
    for wav_path in wavs:
        base = os.path.basename(wav_path)[:-4]
        rec = None
        for cand in _stem_candidates(base):
            if cand in feat_files and cand in f0_files and cand in f0n_files:
                rec = dict(feature=feat_files[cand], f0=f0_files[cand],
                           f0nsf=f0n_files[cand], wav=wav_path, base=cand)
                break
        if rec is None:
            # 不完整样本（缺 feature/f0/f0nsf 之一）跳过，避免后续 np.load(None) 崩溃。
            # 典型来源：某切片 f0 提取失败（VkFailed）或未提取。
            if base in feat_files and base in f0_files and base in f0n_files:
                rec = dict(feature=feat_files[base], f0=f0_files[base],
                           f0nsf=f0n_files[base], wav=wav_path, base=base)
            else:
                continue
        samples.append(rec)
    if not samples:
        raise FileNotFoundError(f"未找到可用的训练样本：{exp_dir}")
    return samples


class _FullSample:
    """加载并预计算单个样本的 spec/feature/f0（含 mel 缓存）。"""

    def __init__(self, rec, cfg: SamplingConfig):
        audio = load_audio(rec["wav"], sr=cfg.sr)
        audio = np.asarray(audio, dtype=np.float32)[None, :]
        from runtime.train.mel_processing import spectrogram_torch

        spec_lin = np.asarray(spectrogram_torch(
            audio, cfg.n_fft, cfg.sr, cfg.hop, cfg.win, center=False),
            dtype=np.float32)
        mel = np.asarray(mel_spectrogram_torch(
            audio, cfg.n_fft, cfg.n_mels, cfg.sr, cfg.hop, cfg.win,
            cfg.fmin, cfg.fmax, center=False), dtype=np.float32)
        feature = np.load(rec["feature"])
        f0 = np.load(rec["f0"])
        f0nsf = np.load(rec["f0nsf"])
        self.audio = audio
        self.spec = spec_lin
        self.mel = mel
        self.feature = feature
        self.f0 = f0
        self.f0nsf = f0nsf
        self.base = str(rec.get("base") or os.path.basename(rec["wav"]))

    def aligned(self, cfg: SamplingConfig, rng=None):
        """对齐 features + 截断（对齐 train/data_utils.get_labels + collate）。

        短样本（帧数 < segment_frames+1）返回 None，由训练循环跳过并计数
        （R-TRAIN-006：原 raise 中断改为跳过告警，不打断训练）。
        """
        feat = np.asarray(self.feature, dtype=np.float32)
        phone = np.repeat(feat, 2, axis=0)
        pitch = np.asarray(self.f0).astype(np.int64)
        pitchf = np.asarray(self.f0nsf, dtype=np.float32)
        n_num = min(phone.shape[0], cfg.max_frames)
        phone = phone[:n_num]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        # ---- T1.2/R3（一票否决点）：pitch/pitchf 短于 n_num 时补零 ----------
        # 病态样本：大肥鱼1/1_45（feature=184 -> phone 帧 368，但 f0 仅 71 帧）。
        # numpy 的 arr[:368] 对长度 71 的数组**不报错**、静默返回长度 71 =>
        # pitch.shape[0]=71 而 phone.shape[0]=368 => vits_train.py:3691 的
        # x_mask 只按 phone 长度建面 [1,1,368]，随后 :3693 enc_p.forward
        # (tape, phone, pitch, x_mask) 内逐帧加法形状不匹配 => 抽到即崩。
        # 这**不是新引入的 bug，而是现状就存在**（lock_frames=0 也崩）。
        if phone.shape[0] != pitch.shape[0]:
            _short = int(phone.shape[0] - pitch.shape[0])
            print(f"[train][R3 补零] {self.base!r}：pitch {pitch.shape[0]} -> "
                  f"{phone.shape[0]}（f0 欠长 {_short} 帧，防御式补零）")
            # a26aa：np.pad 对负宽度抛 ValueError（numpy≥2.4），防御取 max(0,·)
            pitch = np.pad(pitch, (0, max(0, _short)), constant_values=0)
        if phone.shape[0] != pitchf.shape[0]:
            pitchf = np.pad(
                pitchf, (0, max(0, int(phone.shape[0] - pitchf.shape[0]))),
                constant_values=0.0)
        # 与 spec 帧数对齐（截到 min）
        F = self.spec.shape[-1]
        n_min = min(phone.shape[0], F)
        phone = phone[:n_min]
        pitch = pitch[:n_min]
        pitchf = pitchf[:n_min]
        spec = self.spec[:, :, :n_min]
        mel = self.mel[:, :, :n_min]
        audio = self.audio[:, : n_min * cfg.hop]
        if n_min < cfg.segment_frames + 1:
            print(
                f"[train][跳过短样本] {self.base!r}：帧数 {n_min} < "
                f"segment_frames+1={cfg.segment_frames + 1}"
                f"（P-TRAIN-006/R-TRAIN-006：跳过该样本，不中断训练）")
            return None

        # ---- T1.2：数据侧 T 锁定（方案 A）----------------------------------
        # 目标：让 spec/phone/pitch/pitchf/wave 的帧轴恒为 T0，从而
        # enc_p(P=2*T0) / enc_q(F=T0) / flow(F=T0) 三族图 key 跨步恒定。
        # T0==0 或 n_min==T0 时不触碰任何数组 => 位级零回归（98.1% 样本走此路径）。
        spec_lengths_real = int(n_min)
        T0 = int(getattr(cfg, "lock_frames", 0) or 0)
        mode = str(getattr(cfg, "lock_mode", "padcrop"))
        if T0 > 0 and n_min != T0:
            if n_min > T0 and mode in ("crop", "padcrop"):
                # 截断：统一取前 T0 帧（覆盖大肥鱼1 的 1 个 189 帧样本，
                # feature=189 -> phone=378 -> n_min=min(378,F=370)=370 > 368）
                phone = phone[:T0]
                pitch = pitch[:T0]
                pitchf = pitchf[:T0]
                spec = spec[:, :, :T0]
                mel = mel[:, :, :T0]
                audio = audio[:, : T0 * cfg.hop]
                n_min = T0
            elif mode in ("pad", "padcrop"):
                # 补齐：spec/mel 帧轴补零；phone/pitch/pitchf 补零；
                # wave 补 (T0-n_min)*hop 采样点。
                # ★ R8（最易漏）：audio **必须** pad——否则 train.py:556-558 的
                #   wave_r = wave[i, ids0*hop : ids0*hop+17280] 在补齐样本上
                #   被 numpy 静默截短 <17280 => 判别器图 key 漂移成
                #   ("S", id, B, 短长度) => 反噬 P1 靶标（每族 1 key）。
                pad_f = T0 - n_min
                phone = np.pad(phone, ((0, pad_f), (0, 0)))
                pitch = np.pad(pitch, (0, pad_f), constant_values=0)
                pitchf = np.pad(pitchf, (0, pad_f), constant_values=0.0)
                spec = np.pad(spec, ((0, 0), (0, 0), (0, pad_f)))
                mel = np.pad(mel, ((0, 0), (0, 0), (0, pad_f)))
                audio = np.pad(audio, ((0, 0), (0, pad_f * cfg.hop)))
                n_min = T0
            else:
                # mode == "crop" 且 n_min < T0：裁无可裁、补不允许
                # => 与 BLOCK-2 的 lock_mode 校验互为兜底
                raise ValueError(
                    f"lock_mode={mode!r} 无法处理 n_min={n_min} < T0={T0}"
                    f"（样本 {self.base!r}）")
            # ---- 后置不变量断言（T1.2 靶标的可证性载体）----
            assert phone.shape[0] == T0, f"phone {phone.shape[0]} != T0 {T0}"
            assert pitch.shape[0] == T0, f"pitch {pitch.shape[0]} != T0 {T0}"
            assert pitchf.shape[0] == T0, f"pitchf {pitchf.shape[0]} != T0 {T0}"
            assert spec.shape[-1] == T0, f"spec {spec.shape[-1]} != T0 {T0}"
            assert mel.shape[-1] == T0, f"mel {mel.shape[-1]} != T0 {T0}"
            assert audio.shape[-1] == T0 * cfg.hop, (
                f"audio {audio.shape[-1]} != T0*hop {T0 * cfg.hop}")

        return dict(phone=phone[None], pitch=pitch[None], pitchf=pitchf[None],
                    spec=spec, spec_mel=mel, wave=audio, spec_lengths=n_min,
                    spec_lengths_real=spec_lengths_real,
                    ref_mel=self.ref_mel)


# ---------------------------------------------------------------------------
# 单步训练
# ---------------------------------------------------------------------------
def _disc_loss(tape, mpd, x_real, x_fake, detach_fake=False):
    """判别器前向 + LSGAN 损失梯度（种子注入）。

    返回 (loss 标量, score 列表, fmap 列表, seeds)。x_fake detach 时复制数组。
    """
    if detach_fake:
        x_fake = np.asarray(x_fake)
        tape.mark_const(x_fake)
    tape.mark_const(x_real)
    s_r, fmap_r = mpd.forward(tape, x_real)
    s_g, fmap_g = mpd.forward(tape, x_fake)
    seeds = {}
    loss = 0.0
    # r_loss = mean((1-dr)^2)；g_loss = mean(dg^2)
    for dr0 in s_r:
        dr = np.asarray(dr0)
        n = dr.size
        loss += float(np.mean((1.0 - dr) ** 2))
        seeds[id(dr0)] = -2.0 * (1.0 - dr) / n
    for dg0 in s_g:
        dg = np.asarray(dg0)
        n = dg.size
        loss += float(np.mean(dg ** 2))
        # R-TRAIN-001（P-TRAIN-001）：fake 分支梯度种子无条件注入。
        # D 步 detach_fake=True 时 x_fake 已被 tape.mark_const，反向只流向
        # D 参数、不污染生成器；判别器从 real/fake 两分支同时获得对抗信号。
        seeds[id(dg0)] = 2.0 * dg / n
    if os.environ.get("RVC_DISC_DIAG", "0") == "1":
        try:
            _sr = ["%.2f" % float(np.abs(np.asarray(x)).mean()) for x in s_r]
            _sg = ["%.2f" % float(np.abs(np.asarray(x)).mean()) for x in s_g]
            print("[discdiag] s_r|abs|=%s s_g|abs|=%s" % (_sr, _sg), flush=True)
        except Exception:
            pass
    return loss, s_r, s_g, fmap_r, fmap_g, seeds


def _gen_disc_loss(tape, s_g, fmap_r, fmap_g):
    """G 步：gen loss (1-dg)² + feature loss 的梯度种子。"""
    seeds = {}
    # J20：fmap/score 全为 BatchTensor——一次性批量下载（原逐层
    # np.asarray = 每次 readback 各自 submit+fence）。按节点 id 去重
    # （s_g 的 score 与 fmap 的 post 是同一节点）。
    from runtime import vulkan_ops as _vo  # noqa: PLC0415
    _cache = {}
    _items, _ids = [], []
    for dg0 in s_g:
        if isinstance(dg0, _vo.BatchTensor):
            if id(dg0) not in _cache:
                _ids.append(id(dg0))
                _items.append((int(dg0._buf), dg0.shape))
                _cache[id(dg0)] = None
        else:
            _cache[id(dg0)] = np.asarray(dg0)
    for fl in list(fmap_r) + list(fmap_g):
        for t in fl:
            if isinstance(t, _vo.BatchTensor):
                if id(t) not in _cache:
                    _ids.append(id(t))
                    _items.append((int(t._buf), t.shape))
                    _cache[id(t)] = None
            else:
                _cache[id(t)] = np.asarray(t)
    if _items:
        try:
            _arrs = _vo.get_context()._batch_download(_items)
            for k, a in zip(_ids, _arrs):
                _cache[k] = a
        except Exception:  # noqa: BLE001 回退逐项 download
            for k, it in zip(_ids, _items):
                _cache[k] = _vo.get_context().download(it[0], it[1])
    loss_gen = 0.0
    for dg0 in s_g:
        # J19：seeds 键必须用节点本身（id 稳定，BatchTensor 存活期
        # 不回收）；值用下载 numpy 计算。
        dg = _cache[id(dg0)] if _cache.get(id(dg0)) is not None else np.asarray(dg0)
        n = dg.size
        loss_gen += float(np.mean((1.0 - dg) ** 2))
        seeds[id(dg0)] = (seeds.get(id(dg0), 0.0) - 2.0 * (1.0 - dg) / n)
    loss_fm = 0.0
    for fi, (fr_list, fg_list) in enumerate(zip(fmap_r, fmap_g)):
        for fj, (fr0, fg0) in enumerate(zip(fr_list, fg_list)):
            fr = _cache.get(id(fr0))
            if fr is None:
                fr = np.asarray(fr0)
            fg = _cache.get(id(fg0))
            if fg is None:
                fg = np.asarray(fg0)
            n = fg.size
            loss_fm += float(np.mean(np.abs(fr - fg)))
            seeds[id(fg0)] = (seeds.get(id(fg0), 0.0)
                              + 2.0 * np.sign(fr - fg) / n)
    loss_fm *= 2.0
    return loss_gen, loss_fm, seeds


def _kl_tape(tape, z_p, m_p, logs_p, logs_q, z_mask):
    """以 tape 算子计算 kl_loss（种子 seed=1.0 注入）。"""
    # KL 溯源插桩（env RVC_TRAIN_KLTRACE=1；默认关，零行为改动）。
    # 记录 logs_p/logs_q/m_p/z_p 的逐标量统计，用于判别 KL 长程上升的
    # 机制：若 logs_p 单调变负（先验方差收缩）⇒ exp(-2logs_p) 正反馈；
    # 若 logs_p 稳定而 (z_p-m_p)² 放大 ⇒ 估计器噪声累积。
    # 仅读取已算好的数组并用 float() 求标量，不参与反传、不改变计算图。
    if _KLTRACE:
        try:
            _lp = np.asarray(logs_p, dtype=np.float64)
            _lq = np.asarray(logs_q, dtype=np.float64)
            _mp = np.asarray(m_p, dtype=np.float64)
            _zp = np.asarray(z_p, dtype=np.float64)
            print("[KLTRACE] logs_p_mean=%.6f logs_p_min=%.6f logs_p_max=%.6f "
                  "logs_q_mean=%.6f m_p_absmean=%.6f z_p_absmean=%.6f "
                  "d2_mean=%.6f exp_term_mean=%.6f"
                  % (float(_lp.mean()), float(_lp.min()), float(_lp.max()),
                     float(_lq.mean()), float(np.abs(_mp).mean()),
                     float(np.abs(_zp).mean()),
                     float((( _zp - _mp) ** 2).mean()),
                     float((((_zp - _mp) ** 2) * np.exp(-2.0 * _lp)).mean())))
        except Exception as _e:  # 插桩绝不影响训练
            print("[KLTRACE] error: %r" % (_e,))
    # kl = logs_p - logs_q - 0.5 + 0.5*(z_p-m_p)²*exp(-2*logs_p)
    t = tape.sub(logs_p, logs_q)
    t = tape.add_const(t, -0.5)
    d = tape.sub(z_p, m_p)
    d2 = tape.pow(d, 2)
    e = tape.exp(tape.mul_const(logs_p, -2.0))
    term = tape.mul_const(tape.mul(d2, e), 0.5)
    kl = tape.add(t, term)
    kl_m = tape.mul(kl, z_mask)
    total = tape.sum(kl_m)
    denom = float(np.sum(z_mask))
    v = tape.div(total, denom)
    return v


def _cat_batch(items, cfg: SamplingConfig):
    """F1（batch>1）：多样本对齐——统一到 batch 内最短帧数（方案A，
    固定 segment 语义：短样本已被 aligned 过滤）后首维 stack。

    返回 dict：phone [B,P,D]、pitch/pitchf [B,P]、spec/spec_mel [B,1,F]、
    wave [B,1,T']、spec_lengths [B]（int64）。
    """
    if len(items) == 1:
        return items[0]
    minF = min(int(it["spec"].shape[-1]) for it in items)
    # 注意：aligned 各键已带前导 batch=1 维（[1,P,D] 等）→ 先取 [0] 去维
    # 再 stack，避免叠出 4D（[B,1,P,D]）。
    out = {}
    # ---- T1.2/R6 修正（a25e 第 4 项）：帧轴口径不同，切片宽度必须区分 ----
    # phone 帧轴 = 2*T（enc_p 的 P = 2*T，见 train.py:337 repeat(2)）；
    # spec 帧轴 = T。minF 来自 spec ⇒ phone 必须用 [:2*minF]。
    # 原 [:minF] 把 phone 腰斩成 P=T ⇒ batch>1 得到 ("enc",B,T)，
    # 而 batch=1 走 :479-480 早退得到 ("enc",1,2*T)
    # ⇒ _enc_cache 出现两个 key（条目 4 而非 3）⇒ 违背「每族 1 key」靶标。
    # 锁定生效后 minF ≡ T0 恒定 ⇒ 该 bug 从「偶发」变「每步必现」，
    # 与 BLOCK-4 强联动，必须同批修改。
    out["phone"] = np.stack([it["phone"][0][:2 * minF] for it in items])
    out["pitch"] = np.stack([it["pitch"][0][:minF] for it in items])
    out["pitchf"] = np.stack([it["pitchf"][0][:minF] for it in items])
    out["spec"] = np.stack([it["spec"][0][..., :minF] for it in items])
    out["spec_mel"] = np.stack([it["spec_mel"][0][..., :minF] for it in items])
    out["wave"] = np.stack([it["wave"][0][:minF * cfg.hop] for it in items])
    out["spec_lengths"] = np.array([minF] * len(items), dtype=np.int64)
    # P0：参考音色 mel（batch 内统一到最短参考长度，保证可 stack）。
    if all("ref_mel" in it for it in items):
        rmin = min(int(it["ref_mel"].shape[-1]) for it in items)
        out["ref_mel"] = np.stack(
            [it["ref_mel"][0][..., :rmin] for it in items])
    return out


def train_step(net_g, net_d, opt_g, opt_d, batch, cfg: SamplingConfig,
               train_d=True, seed=None, rng=None, ns=None, randn=None):
    """单步训练（batch_size=1 约定）。

    Args:
        net_g: SynthesizerTrnTrain；net_d: MultiPeriodDiscriminator。
        opt_g / opt_d: AdamW。
        batch: 来自 load_samples 对齐后的 dict（phone/pitch/pitchf/spec/
               spec_mel/wave/spec_lengths）。
        cfg: SamplingConfig。
        train_d: 是否同步训练判别器（D 步）。
        seed: 固定随机（确定性测试）。
        ns / randn: 验证专用——固定注入 dec 噪声与 enc_q 高斯（None=默认
            随机路径，零回归）。

    Returns:
        dict: loss_disc / loss_gen / loss_fm / loss_mel / loss_kl /
              loss_total / lr。
    """
    rng = rng or (np.random.RandomState(seed) if seed is not None
                  else np.random)
    phone = batch["phone"]
    pitch = batch["pitch"]
    pitchf = batch["pitchf"]
    spec = batch["spec"]
    spec_mel = batch["spec_mel"]
    wave = batch["wave"]
    spec_lengths = batch["spec_lengths"]          # F1: [B] int64（或标量）
    B = int(np.asarray(spec_lengths).reshape(-1).shape[0])
    sid = np.zeros((B,), dtype=np.int64)
    seg_f = cfg.segment_frames

    # ---- J11 诊断：train_step 分段计时（env RVC_TRAIN_STEP_PROFILE=1）----
    _sp = os.environ.get("RVC_TRAIN_STEP_PROFILE")
    if _sp:
        import time as _pit  # noqa: PLC0415
        _PROF = {"t": _pit.perf_counter(), "s": {}}

        def _mark(seg):
            _n = _pit.perf_counter()
            _PROF["s"][seg] = _PROF["s"].get(seg, 0.0) + (_n - _PROF["t"])
            _PROF["t"] = _n
    else:
        _mark = lambda _seg: None  # noqa: E731

    # ---- G 前向（net_g.forward 返回的 tape 贯穿 G 步全部反向）----
    # VK-01 tape 断链修复：此前在此自建外层 tape 并把 forward 返回的
    # 生成器 tape 解包丢弃（赋给 _），导致 G 步梯度表命中 net_g 0/560；
    # 现在直接使用 forward 返回的 tape（其内已记录全部生成器算子）。
    tape, y_hat, ids_slice, x_mask, y_mask, (
        z, z_p, m_p, logs_p, m_q, logs_q) = net_g.forward(
        phone, pitch, pitchf, spec, spec_lengths, sid, seg_f, rng=rng,
        ns=ns, randn=randn)
    if os.environ.get("RVC_TRAIN_CAPTURE") == "1":
        # 门5 dump（a26ag :652|:653 之间）：必须在 _release_chain_br() /
        # tape.release_brs() 之前 —— 磁带持有的 BatchTensor 尚指向有效槽位。
        _cap_steps = int(os.environ.get("RVC_TRAIN_CAPTURE_STEPS", "1") or "1")
        _module_cap = __import__("_poc._capture", fromlist=["get_capture"])
        if _module_cap.get_capture().should_dump(_cap_steps):
            _module_cap.get_capture().dump(tag=os.environ.get(
                "RVC_TRAIN_CAPTURE_TAG", "mvp"))
    if _sp:
        _mark("g_fwd")

    # 真实 waveform / mel 目标切片（ids * hop 起；F1: 每样本独立 ids）
    if _sp:
        _mark("wave")
    ids0 = np.asarray(ids_slice).reshape(-1)
    hop = cfg.hop
    wave_r = np.stack([
        wave[i, ids0[i] * hop: ids0[i] * hop + cfg.segment_size]
        for i in range(B)])[:, None, :]          # [B, 1, seg]（判别器 [B,1,T]）
    tape.mark_const(wave_r)
    y_mel = np.stack([
        spec_mel[i, :, ids0[i]: ids0[i] + seg_f] for i in range(B)
    ])                                           # [B, 1, seg_f]
    tape.mark_const(y_mel)
    # 静音片段守卫（RVC_TRAIN_SKIP_SILENT=1）：随机切片落到静音区时，
    # log-mel 目标在静音上极小（clamp 到 2e-6），mel L1 梯度被放大成巨值
    # → 损伤 G 权重（实测 y_r|max|=0 时 mel=380~438，训练在 60~100 步崩坏）。
    # 检测目标片段峰值，过静音则跳过本步全部反向/更新（只前进计数器）。
    # 默认开启（RVC_TRAIN_SKIP_SILENT=0 可关）。这是训练稳定的关键修复：
    # 2026-10-08 定位到随机切片落到静音区时 log-mel 目标极小 → mel L1 梯度
    # 被放大成巨值（实测 mel=324~438）→ 损伤 G 权重 → 60~100 步内输出退化
    # 成窄带"蚊子叫"。跳过静音步可让训练稳定收敛（self_40 sim=0.917）。
    if os.environ.get("RVC_TRAIN_SKIP_SILENT", "1") != "0":
        _peak = float(np.abs(np.asarray(wave_r, dtype=np.float64)).max())
        _melmean = float(np.asarray(y_mel, dtype=np.float64).mean())
        if _peak < 1e-3 or _melmean < -11.0:
            try:
                tape.release_brs()
            except Exception:
                pass
            return {"loss_disc": 0.0, "loss_gen": 0.0, "loss_fm": 0.0,
                    "loss_mel": 0.0, "loss_kl": 0.0, "loss_total": 0.0,
                    "lr": opt_g.lr_now(), "_skipped": True}

    # ---- D 步（独立 tape，y_hat detach）----
    if _sp:
        _mark("prep")
    loss_disc = 0.0
    if train_d and os.environ.get("RVC_TRAIN_NO_D", "0") != "1":
        # M4 埋点 A'（a26ah §2）：阶段标定——D 步（判别器）边界。
        # 调用侧不打标则 _commit_chain_br 无法区分 D/G 的 18+18 次 commit。
        if os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1":
            from runtime import m4_probe as _m4p  # noqa: PLC0415
            _m4p.set_phase("D")
        tape_d = AutogradTape()
        loss_disc, *_ , seeds_d = _disc_loss(
            tape_d, net_d, wave_r, np.asarray(y_hat), detach_fake=True)
        tape_d.backward(seeds=seeds_d)
        grads_d = {}
        for name, arr in net_d.parameters().items():
            g = tape_d.grad_of(arr)
            if g is not None:
                grads_d[name] = g
        tape_d.release_brs()  # J15：判别器 BR_FWD 链已全部下载，释放
        opt_d.step(grads_d)
        # J11：判别器权重常驻 GPU 刷新（G 步可导判别器读新值）。
        # T4-3：图化且 p_slots 全覆盖时 p 已直接更新 wpers 槽，跳过重传；
        # 部分覆盖/未图化时照常刷新（无槽参数需 numpy 上传）。
        from runtime import vulkan_ops as _vo  # noqa: PLC0415
        if not (getattr(opt_d, "_ag", None) is not None
                and getattr(opt_d, "_ag_full_p", False)):
            _vo.wpers_refresh("d")
        if _sp:
            _mark("d_step")

    # ---- G 步：判别器（可导）----
    # M4 埋点 A'：G 步边界（此后全部 commit 归 G）。
    if os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1":
        from runtime import m4_probe as _m4p  # noqa: PLC0415
        _m4p.set_phase("G")
    _nod = os.environ.get("RVC_TRAIN_NO_D", "0") == "1"
    if _nod:
        loss_disc_g, loss_gen, loss_fm = 0.0, 0.0, 0.0
        seeds_g = {}
    else:
        loss_disc_g, s_r, s_g, fmap_r, fmap_g, seeds_gen = _disc_loss(
            tape, net_d, wave_r, y_hat, detach_fake=False)
        loss_gen, loss_fm, seeds_fm = _gen_disc_loss(tape, s_g, fmap_r, fmap_g)
        seeds_g = dict(seeds_gen)
        seeds_g.update(seeds_fm)
    if _sp:
        _mark("g_disc")

    # ---- KL（tape 内标量）----
    if _sp:
        _mark("kl")
    kl_node = _kl_tape(tape, z_p, m_p, logs_p, logs_q, y_mask)

    # ---- mel 损失（独立 backward 注入；F1: y2 [B, -1]）----
    if _sp:
        _mark("melcalc")
    y2 = tape.reshape(y_hat, (B, -1))
    y_hat_mel = np.asarray(mel_spectrogram_torch(
        np.asarray(y2), cfg.n_fft, cfg.n_mels, cfg.sr, cfg.hop, cfg.win,
        cfg.fmin, cfg.fmax, center=False), dtype=np.float64)
    diff = (y_hat_mel - np.asarray(y_mel, dtype=np.float64)) * cfg.c_mel
    loss_mel = float(np.mean(np.abs(diff)))
    if loss_mel > 100.0 and os.environ.get("RVC_MEL_DIAG", "0") == "1":
        try:
            _yh = np.asarray(y2, dtype=np.float64)
            _yr = np.asarray(wave_r, dtype=np.float64)
            print("[meldiag] step_mel=%.1f y_hat|max|=%.4f y_r|max|=%.4f "
                  "y_hat_absmean=%.4f y_r_absmean=%.4f seg_f=%d B=%d"
                  % (loss_mel, float(np.abs(_yh).max()), float(np.abs(_yr).max()),
                     float(np.abs(_yh).mean()), float(np.abs(_yr).mean()),
                     int(seg_f), int(B)), flush=True)
        except Exception:
            pass
    from runtime import nn_backward as _nb

    # mel 损失 = mean(|Δ|) * c_mel -> dL/d(mel) = c_mel * sign(Δ) / N
    g_mel = _nb.mel_spectrogram_backward(
        np.asarray(y2), cfg.c_mel * np.sign(diff),
        cfg.n_fft, cfg.sr, cfg.hop, cfg.win, cfg.n_mels, cfg.fmin, cfg.fmax,
        center=False)
    if _sp:
        _mark("mel")

    seeds_all = dict(seeds_g)
    seeds_all[id(kl_node)] = float(cfg.c_kl)
    seeds_all[id(y2)] = g_mel

    tape.backward(seeds=seeds_all)
    if _sp:
        _mark("g_bwd")

    grads_g = {}
    for name, arr in net_g.parameters().items():
        g = tape.grad_of(arr)
        if g is not None:
            grads_g[name] = g
    if os.environ.get("RVC_EMB_DEBUG", "0") == "1":
        _e = grads_g.get("emb_g.weight")
        print("[embdbg] emb_g grad=%s  grads_keys=%d" % (
            "None" if _e is None else "norm=%.4e" % float(np.linalg.norm(_e)),
            len(grads_g)), flush=True)
    opt_g.step(grads_g)
    # J11：生成器权重常驻 GPU 刷新（下一 fwd 用新值）。
    # T4-3：图化且 p_slots 全覆盖时 p 已直接更新 wpers 槽，跳过重传；
    # 部分覆盖/未图化时照常刷新（无槽参数需 numpy 上传）。
    from runtime import vulkan_ops as _vo  # noqa: PLC0415
    if not (getattr(opt_g, "_ag", None) is not None
            and getattr(opt_g, "_ag_full_p", False)):
        _vo.wpers_refresh("g")
    if _sp:
        _mark("opt_g")

    s_mel = float(np.mean(np.abs(y_hat_mel - np.asarray(y_mel))))
    if os.environ.get("RVC_TRAIN_BR_CHAIN", "1") == "1":
        # T-H7 链式：每步末归还链式 BatchRunner 缓冲（防跨步累积）。
        # 此时全部参数梯度已由 grad_of 下载，安全归还。
        from runtime.models.vits_train import _release_chain_br  # noqa: PLC0415
        _release_chain_br()
    tape.release_brs()  # J15：G 步判别器 BR_FWD 链释放（grad_of 已全部下载）
    if os.environ.get("RVC_TRAIN_BR_SHARED", "0") == "1":
        # 阶段H：backward 共享 BatchRunner 每步末归还缓冲（防跨步累积）
        from runtime.vulkan_ops import release_shared_br  # noqa: PLC0415
        release_shared_br()
    if _sp:
        _mark("tail")
        _order = ["g_fwd", "wave", "prep", "d_step", "g_disc", "kl", "melcalc",
                  "mel", "g_bwd", "opt_g", "tail"]
        _sum = sum(_PROF["s"].get(k, 0.0) for k in _order)
        _detail = " ".join(
            f"{k}={_PROF['s'].get(k, 0.0):.2f}" for k in _order)
        print(f"[STEP-PROF] sum={_sum:.2f}s {_detail}", file=sys.stderr)
    # M4 埋点 D（a26ah §2）：把本步 _PROF 段挂到进程级暂存，供驱动脚本在
    # monkey-patch 的 [WALL] 包装里（拿到 step/wall/loss 的地方）调
    # step_snapshot 时一并落账——此处不做快照/清空（否则丢掉 wall/loss 列）。
    if _sp and os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1":
        try:
            from runtime import m4_probe as _m4p  # noqa: PLC0415
            _m4p._S["_last_prof"] = dict(_PROF["s"])
        except Exception:  # noqa: BLE001
            pass
    return dict(loss_disc=float(loss_disc), loss_gen=float(loss_gen),
                loss_fm=float(loss_fm), loss_mel=float(s_mel * cfg.c_mel),
                loss_kl=float(np.asarray(kl_node)) * float(cfg.c_kl),
                loss_total=float(
                    loss_gen + loss_fm + s_mel * cfg.c_mel
                    + float(np.asarray(kl_node)) * cfg.c_kl),
                lr=opt_g.lr_now())


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def build_pretrained(g_ckpt, d_ckpt, cfg: SamplingConfig):
    """从底模 checkpoint 构建 (net_g, net_d)。"""
    import torch_compat  # noqa: PLC0415

    gw = load_g_weights(torch_compat.load_pth(g_ckpt))
    dw = load_d_weights(torch_compat.load_pth(d_ckpt))
    # VK-01 tape 修复配套：RVC 底模以 float16 存储（P1-3：half 为常见格式）。
    # 训练侧梯度在 float64 累加后 cast 回参数 dtype，float16 会溢出为 inf
    # → AdamW 更新产生 NaN（tape 断链修复前梯度从未到达 net_g，问题被掩盖；
    # 修复后 dec.bias 等立即暴露）。提升为 float32（保值提升，数值语义与
    # 推理侧 vits.py 的 half→f32 一致），不改模型数学与推理侧代码。
    gw = {k: (np.asarray(v).astype(np.float32)
              if np.asarray(v).dtype == np.float16 else v)
          for k, v in gw.items()}
    dw = {k: (np.asarray(v).astype(np.float32)
              if np.asarray(v).dtype == np.float16 else v)
          for k, v in dw.items()}
    net_g = SynthesizerTrnTrain(gw, config=cfg.model_config)
    net_d = MultiPeriodDiscriminator(dw, version=cfg.version)
    return net_g, net_d


# J30：轮数自适应基准——默认总步数预算（6000 步 ≈ 361 样本 × 16 次遍历，
# batch=3 时约 45~50 轮）。轮数 = 预算 ÷ 每轮步数，数据集越大每轮步数越多、
# 轮数自动越少（与数据量成反比）；用户可显式传 epochs 覆盖。
_EPOCHS_STEP_BUDGET = 6000


def train_main(exp_dir, epochs=None, steps=80, g_ckpt=None, d_ckpt=None,
               config=None, batch_size=None, log_interval=10, save_every=50,
               out_dir=None, train_d=True, seed=0, cfg_override=None,
               smart=None, resume=None):
    """简化训练入口（log 目录数据 + 预训练底模）。

    Args:
        exp_dir: 工作区 exp 绝对路径（含 0_gt_wavs / 1_16k_wavs / 2a_f0 / 2b-f0nsf /
            3_feature768）。
        epochs: 训练轮数（None=旧语义，直接用 steps 总数；正整数=每轮遍历
            全部样本一遍，总步数 = epochs × ceil(样本数/batch)，对齐原版
            "训练轮数"语义，且轮数建议与数据量成反比——数据集越大每轮步数
            越多，轮数相应越少）。
        steps: 总训练步数（epochs=None 时使用）。
        g_ckpt / d_ckpt: G/D 底模路径（默认 assets/pretrained_v2/f0G48k.pth 等）。
        config: SamplingConfig（None 时用默认 48k v2）。
        save_every: 每 N 步保存 G_{step}.npz（含权重 dict 与 meta）。
        out_dir: 保存目录（默认 exp_dir/model）。
        seed: 随机种子。
        smart: dict（rvc-project.json 的 smart_sampling 配置；None=关闭方案二）。
        resume: resume_state.npz 路径；提供则从断点继续（权重+优化器+进度）。

    Returns:
        dict: history 损失曲线、最终 loss、保存的 checkpoint 列表、
            smart_summary（方案二统计）、resume_path（最近续训状态）。
    """
    rng = np.random.RandomState(seed)
    # ---- T1.2：RVC_TRAIN_LOCK_FRAMES env 覆盖（a25e 第 1/2 项）----------
    # 优先级（高→低）：config 对象 > cfg_override（含 CLI）> env > 默认 368。
    # 默认取 ""（**不干预**）而非 "368"——与 RVC_TRAIN_BATCH_SIZE 的 "3" 不同，
    # 因为 lock_frames 参与四层优先级链，「未设置」必须可表达（a25e §2.2）。
    # 写在此处是为了**先于 apply_adaptive_mode**：后者可能注入 env，
    # 本门控只看用户显式设置的值，不被自适应模式改动。
    if config is None:
        _lf = os.environ.get("RVC_TRAIN_LOCK_FRAMES", "")
        if _lf.strip() and not (cfg_override or {}).get("lock_frames"):
            cfg_override = dict(cfg_override or {})
            cfg_override["lock_frames"] = int(_lf)
    cfg = config or SamplingConfig(**cfg_override or {})
    # T4 自适应模式切换：必须在 batch_size env 读取（下方 L~712）之前调用，
    # 4G 模式注入的 RVC_TRAIN_BATCH_SIZE=1 才能生效。返回决策 dict。
    apply_adaptive_mode(verbose=True)
    if batch_size is None:
        # F1/J27：batch>1 摊薄每步固定开销（Python 簿记/传输/同步 ~1.5s）。
        # batch=3 实测 wall 7.0s/步 → 等效 2.33s/样本（batch1 的 3.5s/样本，
        # -33%；目标 2.5s/样本达成）。默认 3；env RVC_TRAIN_BATCH_SIZE 可调。
        batch_size = int(os.environ.get("RVC_TRAIN_BATCH_SIZE", "3"))
        print(f"[train_main] 采样 batch_size={batch_size}"
              f"（env RVC_TRAIN_BATCH_SIZE 可调）")
    if g_ckpt is None:
        g_ckpt = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "assets",
            "pretrained_v2", "f0G48k.pth")
    if d_ckpt is None:
        d_ckpt = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "assets",
            "pretrained_v2", "f0D48k.pth")
    if out_dir is None:
        out_dir = os.path.join(exp_dir, "model")
    os.makedirs(out_dir, exist_ok=True)

    net_g, net_d = build_pretrained(g_ckpt, d_ckpt, cfg)
    opt_g = AdamW(net_g.parameters(),
                  lr=float(os.environ.get("RVC_TRAIN_LR_G", "1e-4") or "1e-4"))
    opt_d = AdamW(net_d.parameters(),
                  lr=float(os.environ.get("RVC_TRAIN_LR_D", "1e-4") or "1e-4"))
    print("[train_main] lr_g=%.2e lr_d=%.2e" % (opt_g.lr, opt_d.lr))

    # ---- D 阶段：续训恢复 ------------------------------------------------
    start_step = 1
    if resume and os.path.isfile(resume):
        from runtime.train.smart import load_resume
        w_rt, og, od, w_d_rt, meta = load_resume(resume)
        params = net_g.parameters()
        n_load = 0
        for k, v in w_rt.items():
            if k in params and params[k].shape == v.shape:
                params[k][:] = v
                n_load += 1
        # R-TRAIN-002：续训恢复 net_d 权重（缺键=旧协议，兼容跳过并告警）
        n_dload = 0
        if w_d_rt:
            dparams = net_d.parameters()
            for k, v in w_d_rt.items():
                if k in dparams and dparams[k].shape == v.shape:
                    dparams[k][:] = v
                    n_dload += 1
            print(f"[train_main] 续训恢复 net_d：{n_dload} 参数")
        else:
            print("[train_main] 警告：resume 档为旧协议（无 w_d.* 键），"
                  "net_d 保持底模权重，opt_d m/v 可能与 D 权重错配"
                  "（P-TRAIN-002；建议从头训练或重建 resume 档）")
        opt_g.load_state(og)
        opt_d.load_state(od)
        start_step = int(meta.get("step", 0)) + 1
        print(f"[train_main] 续训恢复：{n_load} 参数，从 step {start_step} 继续"
              f"（lr={meta.get('lr', opt_g.lr):.2e}）")

    recs = load_samples(exp_dir, cfg.version)
    full = [_FullSample(r, cfg) for r in recs]
    print(f"[train_main] 样本数 {len(full)} | 结构 "
          f"{net_g.cfg.__repr__()}")

    # J30：epochs 语义落地——每轮 = 遍历全部样本一遍（对齐原版"训练轮数"）。
    # 轮数建议与数据量成反比：数据集越大每轮步数越多，总步数目标固定
    # （_EPOCHS_STEP_BUDGET）时轮数自动越少；显式传 epochs 时按其为准。
    if epochs is not None and int(epochs) > 0:
        per_epoch = max(1, int(np.ceil(len(full) / max(1, int(batch_size)))))
        steps = int(epochs) * per_epoch
        print(f"[train_main] 训练轮数={int(epochs)}（每轮 {per_epoch} 步，"
              f"样本 {len(full)} / batch {int(batch_size)}）→ 总步数 {steps}")
    else:
        per_epoch = max(1, int(np.ceil(len(full) / max(1, int(batch_size)))))
        print(f"[train_main] 总步数 {steps}（样本 {len(full)} / batch "
              f"{int(batch_size)}，约每轮 {per_epoch} 步，建议轮数 "
              f"{max(1, int(np.ceil(_EPOCHS_STEP_BUDGET / per_epoch)))})")

    # ---- D 阶段：方案二（多级最优筛选）------------------------------------
    from runtime.train.smart import SmartSampler, save_resume
    sm = SmartSampler(**(smart or {})) if smart else SmartSampler(enabled=False)
    resume_path = os.path.join(out_dir, "resume_state.npz")

    def _persist_resume(step, losses):
        try:
            # T4-3：图化外部 p 槽时先同步 GPU→numpy（save_resume 读 numpy）
            opt_g._sync_gpu()
            opt_d._sync_gpu()
            save_resume(resume_path, net_g.parameters(),
                        opt_g.state(), opt_d.state(),
                        step=step, lr=opt_g.lr, last_loss=losses,
                        segment=sm.segment, d_weights=net_d.parameters())
        except Exception as exc:  # noqa: BLE001
            print("[train_main] 续训状态保存失败：%s" % exc)

    history = []
    ckpts = []
    skipped_short = 0   # 短样本跳过计数（R-TRAIN-006）
    try:
        for step in range(start_step, steps + 1):
            # F1（batch>1）：每步采样 batch_size 个样本，统一对齐后合并
            items = []
            for _ in range(max(1, int(batch_size))):
                _i = int(rng.randint(0, len(full)))
                fs = full[_i]
                b = fs.aligned(cfg, rng=rng)
                if b is not None:
                    items.append(b)
                else:            # 短样本：跳过并计数，不中断训练
                    skipped_short += 1
            if not items:
                continue
            batch = _cat_batch(items, cfg)
            losses = train_step(net_g, net_d, opt_g, opt_d, batch, cfg,
                                train_d=train_d, rng=rng)
            if losses.pop("_skipped", False):
                print(f"[train_main] step {step} 跳过（静音片段，不更新）")
                continue
            history.append(losses)
            # T4 自适应：每 log_interval 步查一次显存预算（RVC_ADAPTIVE_MONITOR=1
            # 才启用，默认关零开销；见 adaptive.monitor_step）
            monitor_step(step, log_interval)
            if step % log_interval == 0 or step == 1:
                print(
                    f"step {step:4d} | d={losses['loss_disc']:.4f} "
                    f"g={losses['loss_gen']:.4f} fm={losses['loss_fm']:.4f} "
                    f"mel={losses['loss_mel']:.4f} kl={losses['loss_kl']:.4f} "
                    f"total={losses['loss_total']:.4f}")
            if step % save_every == 0:
                # T4-3：图化外部 p 槽时先同步 GPU→numpy（权重保存在 numpy）
                opt_g._sync_gpu()
                opt_d._sync_gpu()
                path = os.path.join(out_dir, f"G_{step}.npz")
                meta = {"config": cfg.model_config, "step": int(step),
                        "version": cfg.version}
                np.savez(
                    path,
                    **{k: np.asarray(v) for k, v in net_g.parameters().items()},
                    **{"_meta": np.asarray(json.dumps(meta))})
                ckpts.append(path)
                print(f"[train_main] 已保存 {path}")
                # D 阶段：方案二观察（每保存周期一次）+ 续训状态
                if sm.enabled:
                    sm.observe(step, float(losses["loss_total"]),
                               net_g.parameters(), out_dir)
                _persist_resume(step, losses)
    except KeyboardInterrupt:
        print("[train_main] 收到中断，保存当前状态后退出（续训可用）")
        _persist_resume(step if 'step' in locals() else start_step,
                        history[-1] if history else None)
        raise
    except Exception:
        print("[train_main] 训练异常，尽力保存当前状态")
        try:
            _persist_resume(step if 'step' in locals() else start_step,
                            history[-1] if history else None)
        except Exception:  # noqa: BLE001
            pass
        raise
    result = dict(history=history, ckpts=ckpts,
                  final=history[-1] if history else None,
                  resume_path=resume_path,
                  skipped_short=skipped_short,
                  smart_summary=sm.summary() if sm.enabled else None)
    print("[train_main] 完成。跳过的短样本次数：%d | 方案二摘要：%s"
          % (skipped_short, sm.summary() if sm.enabled else "未启用"))
    # T3-a：训练结束释放图执行器（判别器图/槽位 buffer；wpers 权重常驻
    # 归 opt 管理，不随图释放）。跨 train_main 复用的图缓存随下次惰性重建。
    try:
        from runtime.models.vits_train import _release_graph_runner  # noqa: PLC0415
        _release_graph_runner()
    except Exception:  # noqa: BLE001
        pass
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="pure-numpy RVC 训练入口（工作区化，2026-10-01）")
    ap.add_argument("exp", help="训练数据目录（工作区绝对路径 workspaces/<项目>/<任务>/exp；旧 logs/<exp> 已废弃）",)
    ap.add_argument("--steps", type=int, default=80, help="总步数")
    ap.add_argument("--save_every", type=int, default=50, help="每 N 步保存 G_{step}.npz")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--out_dir", default="", help="模型输出目录（默认 <exp>/model）；工作区默认任务 checkpoints")
    ap.add_argument("--sr", type=int, default=48000, help="目标采样率（40000/48000/32000）")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    # ---- T1.2 数据侧 T 锁定（a25e 第 1 项：default=None，**不遮蔽 env**）----
    # 若给 default=368，则 cfg_override 每次都被写入 lock_frames ⇒ BLOCK-5 的
    # 守卫 `not (cfg_override or {}).get("lock_frames")` 恒为假 ⇒
    # RVC_TRAIN_LOCK_FRAMES 从诞生起就是死代码。None 是唯一能让三层同时可达的值。
    ap.add_argument("--lock-frames", type=int, default=None,
                    help="T1.2 固定帧数 T0（模型帧口径；0=关闭；"
                         "未传则取 env RVC_TRAIN_LOCK_FRAMES，"
                         "再兜底 SamplingConfig 默认 368）")
    ap.add_argument("--lock-mode", default=None,
                    choices=["pad", "crop", "padcrop"],
                    help="等长化方向（默认 padcrop；crop 是 189 帧样本唯一出口）")
    ap.add_argument("--resume", default="", help="resume_state.npz 路径（续训）")
    ap.add_argument("--smart-json", default="", help="方案二 smart_sampling JSON 字符串")
    ap.add_argument("--g", default="", help="生成器底模/.npz 续训 ckpt 路径")
    ap.add_argument("--d", default="", help="判别器底模/.npz 续训 ckpt 路径")
    args = ap.parse_args()
    smart = json.loads(args.smart_json) if args.smart_json.strip() else None
    _ov = {"sr": args.sr, "version": args.version}
    if args.lock_frames is not None:      # None = 未传 ⇒ 不写键 ⇒ env 可接管
        _ov["lock_frames"] = int(args.lock_frames)
    if args.lock_mode:
        _ov["lock_mode"] = args.lock_mode
    train_main(args.exp, steps=args.steps, save_every=args.save_every,
               log_interval=args.log_interval,
               out_dir=args.out_dir.strip() or None,
               g_ckpt=args.g or None, d_ckpt=args.d or None,
               resume=args.resume.strip() or None, smart=smart,
               cfg_override=_ov)
