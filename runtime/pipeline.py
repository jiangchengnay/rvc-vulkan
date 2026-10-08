# -*- coding: utf-8 -*-
"""RVC 离线推理管线——纯 numpy 移植（T34，对齐 ``infer/vc/pipeline.py``）。

提供与原版 ``Pipeline`` 完全一致的离线单次推理路径：

    ``pipeline()``（高通滤波/静音-长音切分/分块/后处理）
      → ``vc()``（hubert 特征 → faiss 检索混合 → 2× 上采样 → pitchff 保护掩码
                 → vits 合成）
      → ``get_f0()``（pm 自相关 / rmvpe 基频 + UV 插值/移调/mel 量化）
      + ``change_rms()``（RMS 包络混合，librosa.feature.rms 的 numpy 等价）

零 torch / faiss / librosa / parselmouth / transformers 依赖；
``scipy.signal`` 仅用于 ``butter``/``filtfilt``（高通滤波，与原版一致）。

组件解耦约定（T35 的 VC 类复用本类）：
    - model = hubert 编码器实例（``.encode(x[1,T], version=1|2) -> [1,L,D]``）
    - net_g = vits 合成器实例（``.infer(phone[1,P,D], pitch[1,P]i64,
      nsff0[1,P]f32, sid[1]i64) -> [1,1,480P]``）
    - sid: int 说话人 id
    - version: 1（256d）/ 2（768d），兼容字符串 "v1"/"v2"
    - 索引: file_index 为 None/""（不使用）或 FeatureIndex 实例
      （runtime.retrieval，.npz 索引）或 IVFIndex 实例（runtime.ivf_index，
      .ivf.npz 索引）；index_rate 为混合率。IVFIndex 走倒排分桶近似检索
      （nprobe 个桶内暴力 kNN），FeatureIndex 走全量暴力，加权语义一致。
    - if_f0: 模型是否音高引导（本项目全为 True；nono 模型不支持，会明确报错）

与原版代码的逐行对齐说明见各方法 docstring。
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time as ttime

import numpy as np
from scipy import signal

from runtime.dsp.f0 import f0_autocorrelation
from runtime.dsp.resample import resample
from runtime.ivf_index import IVFIndex
from runtime.nn import interpolate_linear
from runtime.retrieval import FeatureIndex, search_l2

__all__ = ["Pipeline", "change_rms"]

# 与原版一致：5 阶 48Hz 高通 Butterworth（fs=16000），模块级共享
_BH, _AH = signal.butter(N=5, Wn=48, btype="high", fs=16000)

# hubert 帧移（样本数@16k）：10ms
_WINDOW = 160

# 检索近邻数（与原版 faiss index.search(k=8) 一致）
_K_SEARCH = 8

# hubert 输入的整段 LayerNorm eps（runtime/models/hubert 的调用方约定）
_HUBERT_LN_EPS = 1e-5


_IVF_D2_CACHE: dict = {}  # T15：桶向量 L2 范数缓存（vecs.shape -> d2[vecs 行数]）


def _ivf_strict_search(index, query, k=8, nprobe=None):
    """严格逐行桶检索（与正版 faiss IndexIVFFlat 语义一致，B4 修复核心）。

    faiss 语义：每个 query 行独立找最近 nprobe 个簇中心 → 只在这些簇的
    向量内做暴力 L2 top-k。注意误区：``IVFIndex.search`` 用"块内各行桶
    取并集"（超集、只增召回），会把其他行的近邻混进来，与 faiss 单行
    桶检索不一致（实测 0.62 vs 1.0 命中率）——这里按行严格取桶。

    T15 性能（2026-09-25）：nprobe=1（gan1.index 实测）时走快速路径——
    每行唯一桶，用 ``np.argmin`` 代替 argpartition 并内联 L2 核心，消除
    逐行 ``search_l2`` 的固定调用/分配开销（~1500 帧 × 0.15ms → ~40µs，
    60s 档检索 0.99s → ~0.4s）。快速路径与原始逐行语义严格一致（同桶
    同 k、同公式，仅 BLAS 批处理/内联的累加序差异 ≤1e-7 级浮动）。
    nprobe>1 / 异常时回退原逐行循环（零回归）。

    Returns: (scores [F,k] f32 升序, indices [F,k] int64 全局行号)
    """
    if nprobe is None:
        nprobe = getattr(index, "nprobe", 1) or 1
    F = query.shape[0]
    if F == 0:
        return (np.empty((0, k), dtype=np.float32),
                np.empty((0, k), dtype=np.int64))
    centroids = np.asarray(getattr(index, "centroids", None), dtype=np.float32)
    vecs = np.asarray(getattr(index, "_vectors", None), dtype=np.float32)
    if centroids is None or vecs is None or centroids.shape[0] == 0:
        # 无 IVF 结构 → 退化为全量暴力
        return search_l2(query, vecs, k=k)
    # 每行 query 距所有中心（分块防内存峰值）
    q2 = np.einsum("fd,fd->f", query, query)
    c2 = np.einsum("nd,nd->n", centroids, centroids)
    dist = q2[:, None] + c2[None, :] - 2.0 * (query @ centroids.T)
    np.maximum(dist, 0.0, out=dist)
    nprobe = min(nprobe, centroids.shape[0])
    buckets = getattr(index, "buckets", None)
    if buckets is None:
        return search_l2(query, vecs, k=k)

    # ── T15 快速路径：nprobe == 1（每行唯一桶）──
    if nprobe == 1 and os.environ.get("RVC_RETRIEVAL_FAST", "1").strip() != "0":
        top = np.argmin(dist, axis=1)  # [F] 最近中心（等价 argpartition(0)[:,:1]）
        scores = np.full((F, k), np.inf, dtype=np.float32)
        inds = np.full((F, k), -1, dtype=np.int64)
        # 桶 L2 范数缓存（同进程内 vecs 不变；dict[桶行号集 -> d2] 避免
        # 每行重算 search_l2 的 d2 einsum —— 桶重复行频率通常很低，用
        # 全量 d2 一次计算 + 子集索引更省（74606 点 einsum ~1ms/调用）。
        d2 = _IVF_D2_CACHE.get(vecs.shape)
        if d2 is None:
            d2 = np.einsum("nd,nd->n", vecs, vecs)
            if len(_IVF_D2_CACHE) > 4:  # 防缓存膨胀（不同尺寸索引共存）
                _IVF_D2_CACHE.clear()
            _IVF_D2_CACHE[vecs.shape] = d2
        for f in range(F):
            b = top[f]
            cand = buckets[b]
            if cand.size == 0:
                continue
            kk = min(k, cand.size)
            d = q2[f] + d2[cand] - 2.0 * (query[f] @ vecs[cand].T)
            np.maximum(d, 0.0, out=d)  # 数值修剪（同 search_l2）
            if kk >= cand.size:
                order = np.argsort(d)
            else:
                p = np.argpartition(d, kk - 1)[:kk]
                order = p[np.argsort(d[p])]
            inds[f, :kk] = cand[order]
            scores[f, :kk] = d[order]
        return scores, inds

    # ── nprobe > 1：原逐行循环（候选 = nprobe 个桶并集）──
    top = np.argpartition(dist, nprobe - 1, axis=1)[:, :nprobe]  # [F,nprobe]
    scores = np.full((F, k), np.inf, dtype=np.float32)
    inds = np.full((F, k), -1, dtype=np.int64)
    for f in range(F):
        cand = np.concatenate([buckets[li] for li in top[f]
                               if buckets[li].size])
        if cand.size == 0:
            continue
        s, ix = search_l2(query[f:f + 1], vecs[cand], k=min(k, cand.size))
        inds[f, :len(ix[0])] = cand[ix[0]]
        scores[f, :len(s[0])] = s[0]
    return scores, inds


def change_rms(data1, data2, sr1, sr2, rate):
    """RMS 包络混合（模块级函数，兼容原版位置参数语义）。

    参数顺序与任务约定一致：``change_rms(data1, data2, sr1, sr2, rate)``，
    即 1 是输入音频、2 是输出音频、rate 是 2 的占比（原版模块函数顺序为
    ``(data1, sr1, data2, sr2, rate)``，公式完全一致）。

    公式（原版 pipeline.py 19-42 行）：对两路信号各自求半秒帧的 RMS 包络
    （librosa.feature.rms: frame_length = sr//2*2, hop_length = sr//2，
    center pad constant），线性插值到 ``len(data2)``，然后
    ``data2 *= pow(rms1, 1-rate) * pow(rms2, rate-1)``（rms2 取下界 1e-6）。
    """
    r1 = _rms_curve(np.asarray(data1, dtype=np.float32), int(sr1))
    r2 = _rms_curve(np.asarray(data2, dtype=np.float32), int(sr2))
    n_out = np.asarray(data2).shape[0]
    r1 = _interp_linear_size(r1, n_out)
    r2 = _interp_linear_size(r2, n_out)
    r2 = np.maximum(r2, np.float32(1e-6))
    ratio = np.power(r1, 1.0 - rate) * np.power(r2, rate - 1.0)
    # ratio 形如 [1, N]（rms 曲线带帧维），reshape 回 data2 的形状避免广播成 2D
    return np.asarray(data2, dtype=np.float32) * ratio.reshape(-1)


def _rms_curve(x, sr):
    """``librosa.feature.rms`` 的 numpy 等价（center=True, pad constant）。

    原版调用 ``librosa.feature.rms(y=x, frame_length=sr//2*2, hop_length=sr//2)``，
    librosa 默认 ``pad_mode="constant"``（零填充），center=True 时两侧各补
    frame_length//2。逐帧 RMS = sqrt(mean(frame^2))，返回 [1, 1 + len(x)//hop]。
    """
    frame_length = sr // 2 * 2
    hop = sr // 2
    pad = frame_length // 2
    xp = np.pad(x, (pad, pad), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(
        xp, frame_length, axis=-1
    )[::hop, ...]
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=-1))
    return rms[None, :].astype(np.float32)  # [1, frames]


def _interp_linear_size(x, n_out, axis=-1):
    """线性插值到**精确**输出长度，对齐 ``F.interpolate(size=n_out, mode='linear',
    align_corners=False)``。

    坐标映射 ``src = (i + 0.5) * L_in / L_out - 0.5``，clamp 到 [0, L_in-1]。
    scale_factor 为浮点时 ``interpolate_linear`` 的 ``int(L_in*scale)`` 可能因
    浮点误差少 1 个样本，这里直接按目标长度生成采样网格。
    """
    x = np.asarray(x, dtype=np.float64)
    L_in = x.shape[axis]
    if L_in == n_out:
        return np.asarray(x, dtype=np.float32)
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * (L_in / n_out) - 0.5
    src = np.clip(src, 0.0, L_in - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, L_in - 1)
    frac = src - lo
    xm = np.moveaxis(x, axis, -1)
    out = xm[..., lo] * (1.0 - frac) + xm[..., hi] * frac
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32)


def _as_batch2(x, dtype):
    """规整为 ``[1, P]`` 2D 数组（1D 输入补 batch 维）。"""
    x = np.asarray(x, dtype=dtype)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"期望 1D/2D 输入，实际 {x.ndim}D")
    return x


def _normalize_for_hubert(x):
    """[已弃用 B5-2026-09-23] hubert 输入整段 LayerNorm——曾误用于 pipeline，
    与官方 RVC 的 raw 输入（do_normalize: false）不一致，导致 hubert 特征偏差
    ~38%（咬字不清根因之一）。pipeline 现直喂 raw 音频；本函数保留定义仅供
    历史引用/对照（diag_b5_layer_cmp.py 复刻），不再被推理路径调用。
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean()
    var = x.var()  # ddof=0，与调用方约定一致
    return ((x - mean) / np.sqrt(var + _HUBERT_LN_EPS)).astype(np.float32)


def _parallel_workers(total_chunks: int) -> int:
    """多块并行推理的 worker 数（P2，**默认关闭**）。

    - 未设置 ``RVC_BATCH_PARALLEL`` 时返回 1：保持与原实现一致的严格串行
      路径（行为/性能完全不变，单块与多块均零线程开销）。
    - 环境变量 ``RVC_BATCH_PARALLEL`` 为 >=1 的整数时启用并行：块循环改
      线程池（worker 数 = 该值，建议 2-4）。实测 150s 长音频并行因 GPU
      批次提交引擎锁争用（BatchRunner 批次生命周期互斥）反而不如串行
      （150s 并行4 ≈537s vs 串行 483s），故默认关闭、作为可选开关。
    - 非法值回退 1（串行）。
    """
    raw = os.environ.get("RVC_BATCH_PARALLEL", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return 1


def _hubert_batch_n() -> int:
    """P2：多块 hubert 批量 encode 的单批块数上限（``RVC_HUBERT_BATCH_N``）。

    - 0 / 未设置（默认）：**块数>6 时按 6 自动分批**（VkFailed 修复 2026-09-25：
      345s≈10 块全批一次 GPU 驻留显存峰值超 16GB HBM2 → 块 2 即
      ``rvc_mem_upload: VkFailed``。分 2 批后峰值减半，同 kernel 同参数
      数值逐位一致）；块数≤6 全部一批（178s≈5 块、60s≈2 块不回归）。
    - >0：按该值分批（每组独立 batch encode），供显存受限场景调小。
    - 非法值回退 0（上述默认分批逻辑）。
    """
    raw = os.environ.get("RVC_HUBERT_BATCH_N", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return 0


def _hubert_batch_group_n(total_chunks: int) -> int:
    """实际单批块数：显式环境变量优先；否则块数>6 自动分批（6/批）防 VkFailed。"""
    n = _hubert_batch_n()
    if n > 0:
        return n
    return 6 if total_chunks > 6 else total_chunks


def _hubert_batch_ok() -> bool:
    """P2：多块 hubert 批量 encode 是否启用（仅 vulkan 后端，GPU 路径）。"""
    try:
        from runtime import backend  # noqa: PLC0415
        return backend.get_backend() == "vulkan"
    except Exception:  # noqa: BLE001
        return False


def _dec_batch_n() -> int:
    """P2：多块 dec 批量开关（``RVC_DEC_BATCH_N``）。

    - 未设置 / 0（默认）：**关**（dec 仍逐块，行为与现状完全一致；dec
      批量在数值验证 + 基准通过前的保守默认）。
    - >0：启用多块 dec 批量（各块先准备 dec 输入，再合并 GPU 解码），
      并按该值分批调用 ``decode_batch``（每组 N 块；组内 vits 侧再按引擎
      dispatch 上限分组）。
    - 非法值回退 0（关）。
    """
    raw = os.environ.get("RVC_DEC_BATCH_N", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return 0


def _dec_batch_ok() -> bool:
    """P2：多块 dec 批量是否可用（仅 vulkan 后端，GPU 路径）。"""
    try:
        from runtime import backend  # noqa: PLC0415
        return backend.get_backend() == "vulkan"
    except Exception:  # noqa: BLE001
        return False


def _persist_feats_gpu(feats):
    """perf(P1)：把 feats[0]（[P, D]）上传为 GPU 常驻 buffer（PersistentBuffer）。

    vulkan 后端返回可跨调用持有的 GPU buffer 句柄，供 vits.infer 的
    ``phone_gpu`` 直接消费（消除了 vits 再次上传 feats 的一次 upload）；
    numpy 后端 / GPU 不可用时返回 None（调用方照旧传 numpy，行为不变）。
    """
    try:
        from runtime import backend  # noqa: PLC0415  # 惰性避免包初始化环

        if backend.get_backend() != "vulkan":
            return None
        from runtime import vulkan_ops  # noqa: PLC0415

        return vulkan_ops.get_context().persistent_upload(
            np.ascontiguousarray(feats[0], dtype=np.float32)
        )
    except Exception:  # noqa: BLE001  # GPU 路径异常 → 回退 numpy 原路径
        return None


class Pipeline(object):
    """离线推理管线（对齐 ``infer/vc/pipeline.py.Pipeline``）。

    用法：
        pipe = Pipeline(tgt_sr=48000, config=Config())
        tgt_sr, audio_int16 = pipe.pipeline(
            hubert_enc, vits_syn, 0, audio16k, [0, 0, 0],
            f0_up_key=0, f0_method="rmvpe", file_index=idx_or_path,
            index_rate=0.75, if_f0=True, version=2, protect=0.33,
            tgt_sr=48000, resample_sr=48000, rms_mix_rate=0.75,
        )

    index_rate 说明（P2，2026-10-07）
    --------------------------------
    检索（index_rate > 0）会把训练集特征混入内容特征 `m_p`，用于
    抑制音色泄漏（top1 检索替换策略）。index_rate 越高，检索特征占比越大。
    """

    def __init__(self, tgt_sr, config):
        """tgt_sr：合成器输出采样率（48k）；config = runtime.native_config.Config。"""
        self.x_pad, self.x_query, self.x_center, self.x_max = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
        )
        self.is_half = False  # numpy 统一 float32
        self.sr = 16000  # hubert 输入采样率
        self.window = _WINDOW  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后 pad 时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device

    # ------------------------------------------------------------------
    # get_f0
    # ------------------------------------------------------------------
    def get_f0(self, x, p_len, f0_up_key, f0_method):
        """基频提取（原版 pipeline.py 44-139 行）。

        - ``pm``：自相关法（``f0_autocorrelation``，对齐 parselmouth
          ``to_pitch_ac(time_step=0.01, voicing_threshold=0.6, f0_min=50,
          f0_max=1100)``），pad 到 ``p_len``（与原版 89-93 行完全一致）。
        - ``rmvpe``：``load_rmvpe().infer_from_audio(x, thred=0.03)``。
        - ``fcpe``：``load_fcpe().infer(x, sr=16000, decoder_mode='local_argmax',
          threshold=0.006)``（懒加载，权重缺失时抛带下载提示的异常）。

        后处理（与原版一致）：uv=0 帧线性插值 → 移调 ``* 2**(up_key/12)`` →
        f0bak 备份 → mel 刻度量化到 [1,255] 得 ``f0_coarse``。
        返回 ``(f0_coarse[p_len], f0bak[p_len])``。
        """
        if f0_method not in ("pm", "rmvpe", "fcpe"):
            raise ValueError(f"Unsupported F0 method: {f0_method}")
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if f0_method == "pm":
            f0 = f0_autocorrelation(
                np.asarray(x, dtype=np.float64),
                self.sr,
                voicing_threshold=0.6,
                f0_min=f0_min,
                f0_max=f0_max,
                time_step=0.01,
            )
            f0 = np.asarray(f0, dtype=np.float64)
            pad_size = (p_len - len(f0) + 1) // 2
            right = p_len - len(f0) - pad_size
            if pad_size > 0 or right > 0:
                f0 = np.pad(f0, (pad_size, right), mode="constant")
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from runtime.models.rmvpe import load_rmvpe

                self.model_rmvpe = load_rmvpe()
            f0 = np.asarray(
                self.model_rmvpe.infer_from_audio(
                    np.asarray(x, dtype=np.float32), thred=0.03
                ),
                dtype=np.float64,
            )
        else:  # fcpe
            if not hasattr(self, "model_fcpe"):
                from runtime.models.fcpe import load_fcpe

                self.model_fcpe = load_fcpe()
            f0 = np.asarray(
                self.model_fcpe.infer(
                    np.asarray(x, dtype=np.float32),
                    sr=16000,
                    decoder_mode="local_argmax",
                    threshold=0.006,
                ),
                dtype=np.float64,
            )

        # UV 帧线性插值（全部为 0 时保持原样，避免 np.interp 空表异常）
        uv = f0 == 0
        if uv.any():
            voiced_idx = np.where(~uv)[0]
            if voiced_idx.size:
                f0[uv] = np.interp(np.where(uv)[0], voiced_idx, f0[voiced_idx])
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.astype(np.float32).copy()  # [p_len] f32
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel = np.clip(f0_mel, 1, 255)
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    # ------------------------------------------------------------------
    # vc（单块）
    # ------------------------------------------------------------------
    def _vc_prepare(self, model, sid, audio0, pitch, pitchf, times,
                    index, index_vectors, index_rate, version, protect,
                    feats=None, retrieval_mode="ivf", brute_mix=0.0):
        """P2：单块变声**准备**（feats → 检索混合 → 2× 插值 → pitchff 保护）。

        与 ``vc()`` 完全相同的准备段（数值逐位一致），但不执行
        ``net_g.infer`` —— 供 dec 多块批量路径先收集各块 dec 输入
        （块循环前统一 ``net_g.decode_batch``）。``times[0]`` 计入检索+
        插值耗时（与 vc() 语义一致）。

        返回 dict：
            feats   [1, P, D] float32（protect 后的最终 phone）
            pitch   [1, P] int64（已按 feats 帧数对齐）
            pitchf  [1, P] float32（同上；可能为 None）
            sid_arr [1] int64
        """
        audio0 = np.asarray(audio0, dtype=np.float32)
        if audio0.ndim == 2:  # double channels
            audio0 = audio0.mean(-1)
        assert audio0.ndim == 1, audio0.ndim

        t0 = ttime()
        if feats is None:
            # B5 修复（2026-09-23）：hubert 输入与官方正版 RVC 对齐——
            # 官方 infer/vc/pipeline.py 直接喂 raw 音频（do_normalize: false），
            # 不再做整段 LayerNorm（_normalize_for_hubert 曾导致特征相对偏差
            # 38%+，检索索引也是 raw 特征构建的，norm 特征整体错位）。
            feats = model.encode(
                audio0[None, :], version=_as_version(version)
            )  # [1, L, D]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.copy()

        if index is not None and index_rate != 0 and index.ntotal > 0:
            npy = np.asarray(feats[0], dtype=np.float32)  # [L, D]
            # ---- 双检索路径（B4）：IVF 近似（对齐 faiss）与全量暴力 ----
            # retrieval_mode: "ivf"（默认，正版语义）| "brute"（暴力为主）
            # brute_mix: 0~1，把另一路检索结果按比例混入主结果
            #  （mode=ivf 时 mix 混入暴力结果；mode=brute 时 mix 混入 IVF 结果）。
            ivf_blend = None   # IVF 近似检索的加权混合特征
            brute_blend = None  # 全量暴力检索的加权混合特征
            if getattr(index, "vectors_by_rows", None) is not None and index_vectors is None:
                # IVFIndex：严格逐行桶检索（对齐 faiss nprobe 语义，B4）
                ivf_score, ivf_ix = _ivf_strict_search(index, npy, k=_K_SEARCH)
                ivf_blk = index.vectors_by_rows(ivf_ix)
                w = np.square(1.0 / ivf_score)
                w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
                ivf_blend = np.sum(ivf_blk * w[:, :, None], axis=1)
                # 暴力路径所需全量向量：IVFIndex 内部全量存储
                if retrieval_mode == "brute" or brute_mix > 0:
                    full = getattr(index, "_vectors", None)
                    if full is not None:
                        sc, ix = search_l2(npy, full, k=_K_SEARCH)
                        blk = full[ix]
                        w = np.square(1.0 / sc)
                        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
                        brute_blend = np.sum(blk * w[:, :, None], axis=1)
            elif index_vectors is not None:
                # FeatureIndex：全量暴力（原语义）
                sc, ix = search_l2(npy, index_vectors, k=_K_SEARCH)
                blk = index_vectors[ix]
                w = np.square(1.0 / sc)
                w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
                brute_blend = np.sum(blk * w[:, :, None], axis=1)
                if retrieval_mode == "ivf" and brute_mix == 0:
                    ivf_blend = brute_blend  # 无 IVF 结构时等价退化
            if brute_blend is None:
                brute_blend = ivf_blend
            if ivf_blend is None:
                ivf_blend = brute_blend
            blend = None
            if retrieval_mode == "brute":
                blend = (1.0 - brute_mix) * brute_blend + brute_mix * ivf_blend
            else:
                blend = (1.0 - brute_mix) * ivf_blend + brute_mix * brute_blend
            if blend is not None:
                feats = (
                    blend[None, :, :] * index_rate + feats * (1 - index_rate)
                )

        feats = interpolate_linear(feats.transpose(0, 2, 1), scale_factor=2).transpose(
            0, 2, 1
        )
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = interpolate_linear(
                feats0.transpose(0, 2, 1), scale_factor=2
            ).transpose(0, 2, 1)
        t1 = ttime()

        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
        # 统一帧对齐：pitch/pitchf 与 feats 差 ±1-2 帧时尾部补齐/裁剪
        # （hubert 块音频含 window 重叠，pitch 切片不含——取整差产生的离散）
        if pitch is not None and pitchf is not None:
            n = feats.shape[1]
            m = pitch.shape[1]
            if m != n:
                if m < n:
                    tail_p = np.repeat(pitch[:, -1:], n - m, axis=1)
                    tail_f = np.repeat(pitchf[:, -1:], n - m, axis=1)
                    pitch = np.concatenate([pitch, tail_p], axis=1)
                    pitchf = np.concatenate([pitchf, tail_f], axis=1)
                else:
                    pitch = pitch[:, :n]
                    pitchf = pitchf[:, :n]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            # 帧对齐：hubert feats 帧数（块音频含 window 重叠）与 pitchf 切片
            # 可能差 ±1-2 帧（window 取整）；补齐/裁剪后再广播。
            n_f = feats.shape[1]
            n_p = pitchf.shape[1]
            if n_p != n_f:
                if n_p < n_f:
                    tail = np.repeat(pitchf[:, -1:], n_f - n_p, axis=1)
                    pitchf = np.concatenate([pitchf, tail], axis=1)
                else:
                    pitchf = pitchf[:, :n_f]
            pitchff = pitchf.copy()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff[:, :, None]  # [1, P, 1]
            feats = feats * pitchff + feats0 * (1 - pitchff)

        sid_arr = np.asarray([sid], dtype=np.int64)
        times[0] += t1 - t0
        return {"feats": feats, "pitch": pitch, "pitchf": pitchf, "sid_arr": sid_arr}

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        index_vectors,
        index_rate,
        version,
        protect,
        feats=None,
        retrieval_mode="ivf",
        brute_mix=0.0,
    ):
        """单块变声（原版 pipeline.py 141-253 行）。

        流程：
            1. audio0（16k f32，块音频）→ 整段 LayerNorm → hubert encode
               → [1, L, D]（v1: 256d，v2: 768d）
               —— P2：``feats`` 非 None 时跳过 encode，直接用调用方预取的
               批量特征（块循环前所有块一次 ``model.encode_batch``），
               检索/插值/dec 仍逐块。
            2. ``protect < 0.5`` 且存在 f0：备份纯 hubert 特征 feats0
            3. 检索混合（index/index_vectors 有效且 index_rate != 0）：
               ``search_l2(npy, vectors, k=8)``；w = (1/score)^2 行归一；
               ``npy = sum(vectors[ix] * w[...,None])``；
               ``feats = npy*index_rate + feats*(1-index_rate)``
            4. 2× 线性插值（20ms→10ms 帧），feats0 同样插值
            5. ``p_len = min(audio0_len//window, feats_len)``，pitch/pitchf
               截断对齐
            6. pitchff 保护掩码（原版 211-216 行）：pitchff 由 pitchf 生成，
               ``pitchff[pitchf>0]=1``、``pitchff[pitchf<1]=protect``，
               ``feats = feats*pitchff + feats0*(1-pitchff)``（pitchff [1,P,1]）
            7. ``net_g.infer(phone=feats, pitch, nsff0=pitchf, sid)``
               → 返回 ``[480P]`` f32 波形（未裁剪 t_pad，由调用方裁剪）

        pitch/pitchf 约定 [1, P] numpy（int64 / float32）。
        index 为 FeatureIndex/IVFIndex 或 None；index_vectors 为 [N, D] f32
        （FeatureIndex 由 _resolve_index 物化；IVFIndex 传 None，检索走
        index.search + index.vectors_by_rows，见 _resolve_index）。
        """
        prep = self._vc_prepare(
            model, sid, audio0, pitch, pitchf, times, index, index_vectors,
            index_rate, version, protect, feats=feats,
            retrieval_mode=retrieval_mode, brute_mix=brute_mix,
        )
        if prep["pitch"] is None or prep["pitchf"] is None:
            raise NotImplementedError(
                "本项目仅支持 f0 引导模型（if_f0=True）；"
                "nono 变体（无 f0 输入）不在本移植范围"
            )
        t1 = ttime()
        # perf(P1) 数据流改造：vulkan 后端下把最终 feats 上传为 GPU 常驻
        # buffer，vits 第一层 emb_phone matmul 直接消费该 buffer（不再
        # 重复上传 feats），推理期间中间张量驻留显存；numpy 后端恒为
        # None，行为与原来完全一致。
        feats_pb = None
        try:
            feats_pb = _persist_feats_gpu(prep["feats"])
            audio1 = net_g.infer(
                np.asarray(prep["feats"], dtype=np.float32),
                prep["pitch"],
                prep["pitchf"],
                prep["sid_arr"],
                phone_gpu=feats_pb,
            )[0, 0]
        finally:
            if feats_pb is not None:
                feats_pb.free()
        t2 = ttime()
        audio1 = np.asarray(audio1, dtype=np.float32).flatten()
        times[2] += t2 - t1
        return audio1

    # ------------------------------------------------------------------
    # pipeline（主流程）
    # ------------------------------------------------------------------
    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        progress_cb=None,
        slice_length=0,
        retrieval_mode="ivf",
        brute_mix=0.0,
    ):
        """离线全流程（原版 pipeline.py 255-410 行）。

        1. 索引解析：file_index 为 FeatureIndex/IVFIndex 实例 → 直接使用；
           为 .npz/.ivf.npz 路径字符串（按后缀分派）→ 懒加载缓存；None/"" →
           无索引。
        2. 高通滤波 butter(5, 48Hz) + filtfilt（原版 286 行）。
        3. 静音/长音切分（原版 287-301 行，细节一致）：
           - window//2=80 点 reflect pad；
           - 总长（pad 后）> t_max 才切分（t_max 可被 ``slice_length`` 秒数
             覆盖，用于"切分长度"滑块）；
           - 滑动 160 点绝对能量和 ``audio_sum[j] = sum_i |pad[j+i]|``；
           - 每 ``t_center`` 取 ``audio_sum[t-t_query : t+t_query]`` 最小
             能量位置为切点（含越界截断，与原版 numpy 行为一致）。
        4. 分块（原版 302-394 行）：整段 audio reflect pad t_pad 后整体
           get_f0；每块 ``audio_pad[s : t+t_pad2+window]``，pitch/pitchf
           切片 ``[s//window : (t+t_pad2)//window]``，vc 输出裁剪
           ``[t_pad_tgt : -t_pad_tgt]`` 后拼接。P2：可选多块并行——默认
           严格串行（行为/性能与原来完全一致）；设 ``RVC_BATCH_PARALLEL``
           为 >=1 的整数时块循环走线程池（块间独立：f0 全局一次，各块
           只读共享 pitch 切片；并行 = 同一代码路径多线程，输出与串行
           逐位一致）。150s 实测并行因 GPU 批次提交引擎锁争用不如串行，
           故默认关闭、供 CPU 侧较重的场景按需开启。
           P2（多块批量 encode）：块循环前把所有块音频一次性
           ``model.encode_batch``（hubert 多块批量、GPU 一次 encode，数值
           与逐块逐位一致），各块 vc() 用预取特征继续检索/插值/dec（dec
           仍逐块）。开关 ``RVC_HUBERT_BATCH_N``：0（默认）全部块一批，
           >0 按该值分批；单块 / numpy 后端 / 异常自动回退逐块 encode。
           每完成一块调用 ``progress_cb(done, total, msg)``（并行时按完成
           顺序推进，回调在主线程串行执行）。
        5. 后处理（原版 395-406 行）：rms_mix_rate != 1 时
           ``change_rms(audio, audio_opt, 16000, tgt_sr, rate)``；
           ``tgt_sr != resample_sr >= 16000`` 时重采样；峰值归一 0.99 → int16。
        返回 ``(tgt_sr, audio_int16)``。
        """
        index, index_vectors = self._resolve_index(file_index, index_rate)
        if slice_length is None:
            slice_length = 0
        t_max = self.t_max
        try:
            slice_length = float(slice_length)
        except (TypeError, ValueError):
            slice_length = 0
        if slice_length > 0:
            t_max = int(slice_length * self.sr)

        audio = np.asarray(audio, dtype=np.float32)
        audio = signal.filtfilt(_BH, _AH, audio)

        # --- 静音/长音切分 ---
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > t_max:
            # 切割步进 = min(t_center, t_max)：slice_length 传小时（如 10s）
            # 切割间隔随之变小 → 每块 hubert attention（O(T²)）大幅加速。
            step = min(self.t_center, t_max)
            # 滑动窗口绝对能量和 = Σ_i |audio_pad[i+j]|：abs 对每个切片
            # 逐元素独立，先对整个 pad 求一次绝对值（结果与逐切片相同，
            # 逐位一致），再逐窗累加 —— 省 160 次重复 abs 计算。
            ap_abs = np.abs(audio_pad)
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += ap_abs[i : i - self.window]
            for t in range(step, audio.shape[0], step):
                lo = max(0, t - self.t_query)
                hi = min(audio_sum.shape[0], t + self.t_query)
                seg = audio_sum[lo:hi]
                if seg.size == 0:
                    continue  # 窗口越界（短音频/小 slice_length）时跳过
                opt_ts.append(
                    lo
                    + int(np.where(seg == seg.min())[0][0])
                )

        # --- 分块 f0（与 dec 块边界一致；每块 T 小 → rmvpe BiGRU 逐帧快 ~15 倍）---
        # 对齐原版语义：pitch 对应 **audio_pad**（含 t_pad 两侧 pad）的帧序列，
        # dec 块切片 [s//win:(t+t_pad2)//win] 直接按 audio_pad 索引取用。
        s = 0
        t1 = ttime()
        pitch, pitchf = None, None
        if if_f0:
            audio_pad_f0 = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
            seg_pts = [0] + [o + self.t_pad for o in opt_ts] + [audio_pad_f0.shape[0]]
            pitch_parts, pitchf_parts = [], []
            for si in range(len(seg_pts) - 1):
                ap = seg_pts[si]
                bp = seg_pts[si + 1]
                b = min(bp + self.t_pad2 + self.window, audio_pad_f0.shape[0])
                seg_pad = audio_pad_f0[ap:b]
                p_len_seg = seg_pad.shape[0] // self.window
                pi, pfi = self.get_f0(seg_pad, p_len_seg, f0_up_key, f0_method)
                core_n = max(1, (bp - ap) // self.window)
                pitch_parts.append(pi[:core_n])
                pitchf_parts.append(pfi[:core_n])
            pitch = np.concatenate(pitch_parts)
            pitchf = np.concatenate(pitchf_parts)
            pitchf = pitchf.astype(np.float32)
            pitch = pitch[None, :]  # [1, P] int32
            pitchf = pitchf[None, :]  # [1, P] f32
        t2 = ttime()
        times[1] += t2 - t1
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")

        # --- 分块合成（P2：多块并行）---
        # 块间完全独立：f0 已在块循环前全局一次提取（pitch/pitchf 为整段
        # audio_pad 的帧序列，各块只读共享切片）；hubert encode / 检索混合 /
        # interpolate / dec 各块互不依赖。用线程池并行执行各块 vc()，结果按
        # 块序收集拼接——并行 = 同一代码路径多线程，输出与串行**逐位一致**。
        # 线程安全：模型对象（hubert/vits）推理只读；vulkan_ops 单算子经
        # ctx._lock、BatchRunner 批次生命周期经引擎级锁串行化（GPU 提交
        # 串行），CPU 侧（检索/interpolate/numpy）跨线程并行。
        # 默认严格串行（RVC_BATCH_PARALLEL 未设/<=0 → 1 worker：单块与多块
        # 均零线程开销、与原实现逐位一致）；显式设 >=1 才启用线程池并行。
        total_chunks = len(opt_ts) + 1
        done_chunks = 0
        if progress_cb is not None:
            progress_cb(done_chunks, total_chunks, "切分完成，共 %d 块" % total_chunks)

        # 预计算块边界（与原串行循环逐位一致：切点先 floor 到 window 对齐）
        bounds = []  # (s, e)；e=None 表示最后一块（audio_pad[s:]）
        s = 0
        for t in opt_ts:
            t = t // self.window * self.window
            bounds.append((s, t))
            s = t
        bounds.append((s, None))

        # --- P2：块循环前一次性批量 hubert encode（多块一次 GPU batch）---
        # 所有块音频先全部 encode（hubert 多块批量，块间独立、长度可不等，
        # 引擎 dispatch 上限内单组 GPU 驻留），每块再取自己的特征行继续
        # 检索/插值/dec（dec 仍逐块 —— dec 的 batch 化是另一阶段）。
        # 开关：RVC_HUBERT_BATCH_N=0（默认）全部块一批；>0 按该值分批。
        # 单块 / numpy 后端 / 异常 → 不预取，vc() 内逐块 encode（行为与
        # 原来完全一致，数值逐位一致：batch 路径与逐块同 kernel 同参数）。
        blk_feats = None
        if len(bounds) > 1 and _hubert_batch_ok():
            blk_audios, blk_idx = [], []
            for idx, (s0, e0) in enumerate(bounds):
                audio_blk = (
                    audio_pad[s0:]
                    if e0 is None
                    else audio_pad[s0 : e0 + self.t_pad2 + self.window]
                )
                if audio_blk.shape[0] < self.window * 10:
                    continue  # 空/过短块（与 run_block 同一跳过条件）
                blk_audios.append(audio_blk[None, :])  # B5：raw 输入对齐官方
                blk_idx.append(idx)
            if blk_audios:
                try:
                    # VkFailed 修复：块数>6 自动分批（_hubert_batch_group_n），
                    # 345s≈10 块分 2 批减半 GPU 驻留；显式 RVC_HUBERT_BATCH_N 优先。
                    N = _hubert_batch_group_n(len(blk_audios))
                    feats_all = []
                    for g in range(0, len(blk_audios), N):
                        feats_all.extend(
                            model.encode_batch(
                                blk_audios[g : g + N],
                                version=_as_version(version),
                            )
                        )
                    blk_feats = dict(zip(blk_idx, feats_all))
                except Exception:  # noqa: BLE001  # 异常 → 回退逐块 encode
                    blk_feats = None

        # --- P2：块循环前一次性 dec 批量（所有片段一起送入声码器）---
        # 各块先"准备 dec 输入"（enc_p+flow+相位预推进，逐块——轻量 numpy/
        # 单算子），再统一 ``net_g.decode_batch`` 合并 GPU 批量（GeneratorNSF
        # 多块无 padding 按块展开，见 runtime/models/vits.py _dec_batch_forward；
        # 数值与逐块 infer 逐位一致：dec 输入逐位相同 + 同 kernel 同参数）。
        # 开关 RVC_DEC_BATCH_N：未设置/0（默认）关（dec 仍逐块，行为与现状
        # 完全一致）；>0 启用并按该值分批 decode。任一异常 → 整体回退逐块。
        blk_waves = None
        dec_batch_on = (
            len(bounds) > 1
            and _dec_batch_ok()
            and hasattr(net_g, "decode_batch")
            and hasattr(net_g, "prep_dec")
            and _dec_batch_n() > 0
        )
        if dec_batch_on:
            blk_waves = {}
            try:
                preps = []  # (idx, prep_dict)
                for idx, (s0, e0) in enumerate(bounds):
                    if e0 is None:
                        audio_blk = audio_pad[s0:]
                        p = pitch[:, s0 // self.window :] if if_f0 else None
                        pf = pitchf[:, s0 // self.window :] if if_f0 else None
                    else:
                        audio_blk = audio_pad[s0 : e0 + self.t_pad2 + self.window]
                        p = (
                            pitch[:, s0 // self.window : (e0 + self.t_pad2) // self.window]
                            if if_f0
                            else None
                        )
                        pf = (
                            pitchf[:, s0 // self.window : (e0 + self.t_pad2) // self.window]
                            if if_f0
                            else None
                        )
                    if audio_blk.shape[0] < self.window * 10:
                        continue  # 空/过短块（与 run_block 同一跳过条件）
                    if p is None or pf is None:
                        continue  # 非 f0（不在本移植范围）→ 该块走逐块 vc()
                    prep0 = self._vc_prepare(
                        model, sid, audio_blk, p, pf, times,
                        index, index_vectors, index_rate, version, protect,
                        feats=(blk_feats.get(idx) if blk_feats is not None else None),
                        retrieval_mode=retrieval_mode, brute_mix=brute_mix,
                    )
                    # 全量 dec 语义（skip_head=None）：与逐块 vc() 的 infer 输入
                    # 完全一致（z 全段 + 全量 cumsum 相位）→ 批量与逐块逐位一致。
                    # （partial skip_head 语义的目标段与全量裁剪存在 T52 既有
                    # 近似差异，不用于批量路径。）
                    feats_pb = None
                    try:
                        feats_pb = _persist_feats_gpu(prep0["feats"])
                        prep = net_g.prep_dec(
                            prep0["feats"], prep0["pitch"], prep0["pitchf"],
                            prep0["sid_arr"], phone_gpu=feats_pb,
                        )
                    finally:
                        if feats_pb is not None:
                            feats_pb.free()
                    preps.append((idx, prep))
                N = _dec_batch_n()
                for g in range(0, len(preps), N):
                    grp = preps[g : g + N]
                    waves = net_g.decode_batch([pp for _, pp in grp])
                    for (idx, _pp), wv in zip(grp, waves):
                        # decode_batch 输出整块波形 [1,1,480P]（s0=0,n_out=P*480）
                        blk_waves[idx] = np.asarray(wv, dtype=np.float32).flatten()
            except Exception:  # noqa: BLE001  # 批量路径异常 → 回退逐块 vc()
                blk_waves = None

        def run_block(idx, s0, e0, blk_times):
            """单块 vc()：与串行路径完全相同的切片与调用（blk_times 线程局部）。"""
            if e0 is None:
                audio_blk = audio_pad[s0:]
                p = pitch[:, s0 // self.window :] if if_f0 else None
                pf = pitchf[:, s0 // self.window :] if if_f0 else None
            else:
                audio_blk = audio_pad[s0 : e0 + self.t_pad2 + self.window]
                p = (
                    pitch[:, s0 // self.window : (e0 + self.t_pad2) // self.window]
                    if if_f0
                    else None
                )
                pf = (
                    pitchf[:, s0 // self.window : (e0 + self.t_pad2) // self.window]
                    if if_f0
                    else None
                )
            if audio_blk.shape[0] < self.window * 10:
                # 空/过短块（切割点与末尾对齐的 0 长度段，或极短段导致
                # hubert conv 下采样后 oL=0）：跳过，避免崩溃（小块场景实测）。
                return None
            if blk_waves is not None and idx in blk_waves:
                # P2：dec 批量路径 —— 波形已在块循环前合并解码（整块
                # [480P]），与逐块 vc() 输出逐位一致，此处同逐块裁剪 t_pad。
                return blk_waves[idx][self.t_pad_tgt : -self.t_pad_tgt]
            out = self.vc(
                model,
                net_g,
                sid,
                audio_blk,
                p,
                pf,
                blk_times,
                index,
                index_vectors,
                index_rate,
                version,
                protect,
                feats=(blk_feats.get(idx) if blk_feats is not None else None),
                retrieval_mode=retrieval_mode,
                brute_mix=brute_mix,
            )[self.t_pad_tgt : -self.t_pad_tgt]
            return out

        results = [None] * len(bounds)  # 与 bounds 同序 → 拼接顺序不变

        # ────────────────────────────────────────────────────────────
        # T3.2：块间 CPU/GPU 软件流水（dec 下载延迟，**零新线程**）
        # ────────────────────────────────────────────────────────────
        # 核心：dec 的 GPU 执行（async commit 后的 fence wait）期间，CPU
        # 同步做**下一块**的 _vc_prepare（检索/插值/protect 纯 numpy）——
        # GPU 与 CPU 天然重叠，无需任何多线程（引擎批次生命周期锁由
        # _DecCollector 持有，所有 GPU 调用仍在单线程按锁串行化）。
        #
        # 启用前置条件（缺一不可，否则回退原严格串行路径，零回归）：
        #   1. 多块（单块无重叠对象）；
        #   2. hubert 批量预取成功（blk_feats 非 None）→ _vc_prepare 全程纯
        #      numpy，绝不创建 BatchRunner。若 feats=None 会走 hubert GPU
        #      encode（创建 BatchRunner → 阻塞在引擎批次生命周期锁）；而该
        #      锁正被在途 dec 的 _DecCollector 持有、须等下一轮 collect 才
        #      释放——单线程下即成死锁，故必须排除；
        #   3. dec 批量路径未启用（blk_waves 非 None 时波形已预生成、块循环
        #      内无 GPU dec 可重叠）；
        #   4. 未显式启用 RVC_BATCH_PARALLEL（线程池块并行与软件流水互斥）；
        #   5. net_g 为 GPU dec 驻留路径（支持 _defer 延迟下载）。
        _overlap_ok = (
            len(bounds) > 1
            and blk_feats is not None
            and blk_waves is None
            and _parallel_workers(len(bounds)) <= 1
            and hasattr(net_g, "infer")
            and getattr(getattr(net_g, "dec", None), "_dec_resident", False)
        )
        if _overlap_ok:

            def _prep(idx, s0, e0, blk_t):
                """切片 + _vc_prepare（纯 numpy，不触碰引擎批次锁）。"""
                if e0 is None:
                    audio_blk = audio_pad[s0:]
                    p = pitch[:, s0 // self.window :] if if_f0 else None
                    pf = pitchf[:, s0 // self.window :] if if_f0 else None
                else:
                    audio_blk = audio_pad[s0 : e0 + self.t_pad2 + self.window]
                    p = (
                        pitch[:, s0 // self.window : (e0 + self.t_pad2) // self.window]
                        if if_f0
                        else None
                    )
                    pf = (
                        pitchf[:, s0 // self.window : (e0 + self.t_pad2) // self.window]
                        if if_f0
                        else None
                    )
                if audio_blk.shape[0] < self.window * 10:
                    return None  # 空/过短块（与 run_block 同一跳过条件）
                return self._vc_prepare(
                    model, sid, audio_blk, p, pf, blk_t,
                    index, index_vectors, index_rate, version, protect,
                    feats=blk_feats.get(idx),
                    retrieval_mode=retrieval_mode, brute_mix=brute_mix,
                )

            def _submit(idx, prep, blk_t):
                """persist feats + infer(_defer=True) → (collector, feats_pb)。

                只做 GPU 提交（enc/flow/dec commit），不等待 dec 下载；
                dec 的 fence wait 与传输留给 _collect（与下一块 prep 重叠）。
                """
                t1 = ttime()
                feats_pb = _persist_feats_gpu(prep["feats"])
                try:
                    holder = net_g.infer(
                        np.asarray(prep["feats"], dtype=np.float32),
                        prep["pitch"],
                        prep["pitchf"],
                        prep["sid_arr"],
                        phone_gpu=feats_pb,
                        _defer=True,
                    )
                except BaseException:
                    if feats_pb is not None:
                        feats_pb.free()
                    raise
                blk_t[2] += ttime() - t1  # 提交段墙钟（不含 dec GPU 等待/下载）
                return holder, feats_pb

            def _collect(idx, holder, feats_pb, blk_t):
                """collector.collect()（wait fence → 下载 → tanh）+ 裁剪 t_pad。"""
                t1 = ttime()
                try:
                    audio1 = holder.collect()[0, 0]
                finally:
                    if feats_pb is not None:
                        feats_pb.free()
                blk_t[2] += ttime() - t1  # 等待+下载段墙钟
                return np.asarray(audio1, dtype=np.float32).flatten()[
                    self.t_pad_tgt : -self.t_pad_tgt
                ]

            block_times = [[0.0, 0.0, 0.0] for _ in bounds]
            hold = None  # (idx, holder, feats_pb)：在途 dec（GPU 执行中）
            try:
                for idx, (s0, e0) in enumerate(bounds):
                    blk_t = block_times[idx]
                    # CPU 段（下一块准备）：与上一块 dec 的 GPU 执行重叠
                    prep = _prep(idx, s0, e0, blk_t)
                    if hold is not None:
                        # 收前一块 dec（等 fence + 下载，释放引擎批次锁）——
                        # 必须先于本块 _submit（其创建 BatchRunner 需要该锁）。
                        h_idx, holder, pb = hold
                        hold = None
                        results[h_idx] = _collect(h_idx, holder, pb,
                                                  block_times[h_idx])
                        times[0] += block_times[h_idx][0]
                        times[2] += block_times[h_idx][2]
                        done_chunks += 1
                        if progress_cb is not None:
                            progress_cb(done_chunks, total_chunks,
                                        "块 %d/%d" % (done_chunks, total_chunks))
                    if prep is None:
                        # 空/过短块：无 dec 可提交，进度独立推进
                        done_chunks += 1
                        if progress_cb is not None:
                            progress_cb(done_chunks, total_chunks,
                                        "块 %d/%d" % (done_chunks, total_chunks))
                        continue
                    holder, feats_pb = _submit(idx, prep, blk_t)
                    hold = (idx, holder, feats_pb)
                # 尾块 drain
                if hold is not None:
                    h_idx, holder, pb = hold
                    hold = None
                    results[h_idx] = _collect(h_idx, holder, pb,
                                              block_times[h_idx])
                    times[0] += block_times[h_idx][0]
                    times[2] += block_times[h_idx][2]
                    done_chunks += 1
                    if progress_cb is not None:
                        progress_cb(done_chunks, total_chunks,
                                    "块 %d/%d" % (done_chunks, total_chunks))
            finally:
                # 异常兜底：放弃在途 dec（等提交完成 + 释放 runner/锁），
                # 防引擎批次生命周期锁泄漏导致后续调用死锁。
                if hold is not None:
                    _h_idx, holder, _pb = hold
                    try:
                        holder.abort()
                    except Exception:  # noqa: BLE001
                        pass
                    if _pb is not None:
                        try:
                            _pb.free()
                        except Exception:  # noqa: BLE001
                            pass
        elif len(bounds) == 1 or _parallel_workers(len(bounds)) <= 1:
            # 单块 / RVC_BATCH_PARALLEL=1：严格串行路径（行为与原来完全一致）
            for idx, (s0, e0) in enumerate(bounds):
                blk_times = times if len(bounds) == 1 else [0.0, 0.0, 0.0]
                results[idx] = run_block(idx, s0, e0, blk_times)
                if len(bounds) > 1:
                    times[0] += blk_times[0]
                    times[2] += blk_times[2]
                done_chunks += 1
                if progress_cb is not None:
                    progress_cb(done_chunks, total_chunks,
                                "块 %d/%d" % (done_chunks, total_chunks))
        else:
            workers = _parallel_workers(len(bounds))
            _done_lock = threading.Lock()
            block_times = [[0.0, 0.0, 0.0] for _ in bounds]
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="rvc-blk") as ex:
                futures = {
                    ex.submit(run_block, idx, s0, e0, block_times[idx]): idx
                    for idx, (s0, e0) in enumerate(bounds)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()  # 块异常在此重抛，退出时等待其余块
                    lt = block_times[idx]
                    times[0] += lt[0]  # 主线程串行合并各块耗时（无竞争）
                    times[2] += lt[2]
                    with _done_lock:
                        done_chunks += 1
                    if progress_cb is not None:
                        progress_cb(done_chunks, total_chunks,
                                    "块 %d/%d" % (done_chunks, total_chunks))
        # 按块序拼接（results 为 list，索引即块序）；空块（None）跳过
        audio_opt = np.concatenate([r for r in results if r is not None])

        # --- 后处理 ---
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, audio_opt, 16000, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = resample(audio_opt, tgt_sr, resample_sr, method="fft")
        audio_opt = np.asarray(audio_opt, dtype=np.float32)
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768.0
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        return tgt_sr, audio_opt

    # ------------------------------------------------------------------
    # change_rms（实例方法，任务接口约定）
    # ------------------------------------------------------------------
    def change_rms(self, data1, data2, sr1, sr2, rate):
        """实例方法版 RMS 混合（见模块级 ``change_rms``，语义一致）。"""
        return change_rms(data1, data2, sr1, sr2, rate)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_index(file_index, index_rate):
        """解析索引：FeatureIndex/IVFIndex 实例 / .npz / .ivf.npz 路径 / None/""。

        - index_rate == 0 或 file_index 为空 → (None, None)。
        - FeatureIndex（.npz）→ 返回 (index, 全量 [N, D] 矩阵)（暴力检索）。
        - IVFIndex（.ivf.npz）→ 返回 (index, None)：检索走 index.search
          + vectors_by_rows（倒排分桶近似，不物化全量矩阵）。
        - .ivf.npz 路径由 _load_index_cached 按后缀分派。
        """
        if index_rate == 0 or file_index is None or file_index == "":
            return None, None
        index = None
        if isinstance(file_index, (FeatureIndex, IVFIndex)):
            index = file_index
        elif isinstance(file_index, str):
            p = file_index.strip()
            if not p or not os.path.exists(p):
                return None, None
            index = _load_index_cached(p)
        else:
            return None, None
        if index is None or index.ntotal <= 0:
            return None, None
        if isinstance(index, IVFIndex):
            return index, None  # 倒排检索：不物化全量矩阵
        index_vectors = index.reconstruct_n(0, index.ntotal)
        return index, index_vectors


_index_cache = {}


def _load_index_cached(path):
    """.npz / .ivf.npz / faiss .index 索引懒加载缓存（按后缀分派类型）。

    - ``.ivf.npz`` → IVFIndex（倒排分桶近似检索）
    - ``.npz``    → FeatureIndex（暴力检索）
    - ``.index``  → faiss_reader.read_faiss_index → FeatureIndex（兼容原版 faiss
      索引，IndexFlat/IVFFlat/IDMap 等；避免被误当 .npz 用 np.load 读二进制
      而报 "pickled (object) data" 错误——已实测定位的根因）
    """
    key = os.path.abspath(path)
    if key not in _index_cache:
        low = key.lower()
        if low.endswith(".ivf.npz"):
            _index_cache[key] = IVFIndex.load(key)
        elif low.endswith(".index") or low.endswith(".faiss"):
            from runtime import faiss_reader
            _index_cache[key] = faiss_reader.read_faiss_index(key)
        else:
            _index_cache[key] = FeatureIndex.load(key)
    return _index_cache[key]


def _as_version(version):
    """归一化 version 到 int（1/2），兼容字符串 'v1'/'v2'。"""
    if version in ("v1", "v2"):
        return 1 if version == "v1" else 2
    v = int(version)
    if v not in (1, 2):
        raise ValueError(f"version 仅支持 1/2，收到 {version!r}")
    return v


def _self_test():
    """pipeline 模块级自测：配置/长度数学冒烟。"""
    from runtime.native_config import Config

    cfg = Config()
    assert cfg.x_pad == 1 and cfg.x_max == 41
    pipe = Pipeline(48000, cfg)
    assert pipe.t_pad == 16000 and pipe.t_pad_tgt == 48000
    assert pipe.t_query == 96000 and pipe.t_center == 608000 and pipe.t_max == 656000
    assert pipe.window == 160 and not pipe.is_half
    print("pipeline.Pipeline self_test PASS")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)