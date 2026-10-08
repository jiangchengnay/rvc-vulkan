# -*- coding: utf-8 -*-
"""RVC VITS 训练版模型（纯 numpy，T45/T46/P2）。

三部分：
    1. ``AutogradTape`` —— 记录式自动微分：前向把算子调用（含中间张量与参数
       引用）压入 tape，反向按逆序调用 ``runtime.nn_backward`` 的 bp 算子并
       把梯度累加到各输入；支持向任意中间节点注入外部梯度（mel 谱损失用独立
       的 ``mel_spectrogram_backward`` 计算后注入 ``y_hat`` 节点）。含
       ``deweight_norm`` 可导算子（weight_norm 还原，反向手写 g/v 梯度）。
    2. ``PosteriorEncoder``（enc_q）/ ``SynthesizerTrnTrain`` —— 完整训练
       路径：enc_p -> enc_q -> flow 正变换 -> rand_slice_segments 切片 ->
       GeneratorNSF（逐字节对齐 ``SynthesizerTrnMs768NSFsid.forward``）。
    3. ``MultiPeriodDiscriminator`` —— V1/V2（periods 与 conv2d/conv1d 堆），
       支持纯前向打分与 tape 记录（判别器训练）。

权重约定（P2 weight_norm 参数化，对齐原版 torch ``nn.utils.weight_norm``）：
    - ``load_g_weights`` 输出 **weight_g + weight_v 双参数**：dec.ups /
      dec.resblocks.convs / flow WN / enc_q.enc 等层以 ``*.weight_v`` +
      ``*.weight_g`` 存储，前向由 ``AutogradTape.deweight_norm`` 还原
      ``W = w_v * (w_g / ||w_v||_2)``，反向对 g/v 传播梯度——训练直接优化
      weight_norm 参数（与原版一致），训练产物本身就是推理格式
      （``process_ckpt.savee`` 无需再逆重参数化，转换损耗消除）。
    - 普通权重 checkpoint（T46 及以前的训练产物）按 ``weight_v = W``、
      ``weight_g = ||W||_逐通道`` 自动拆分；判别器 ``load_d_weights`` 保持
      普通权重训练（T46 现状）。
    - 判别器键 ``discriminators.<i>.<convs|conv_post>.<j>.weight_v|weight_g|bias``。

**本模块禁止 import torch**。
"""
from __future__ import annotations

import os  # 阶段A：RVC_TRAIN_BR_FWD 等开关
import sys

import math
import re

import numpy as np

from .. import nn_backward as nb
from .vits import VitsConfig, _deweight_norm, _get_relative_embeddings

# T-H8：模块级 BatchTensor 引用（tape 算子方法保持 GPU 引用保 ref id 链；
# vulkan_ops 不 import 本模块，无环）。
from runtime.vulkan_ops import BatchTensor as BatchTensor  # noqa: PLC0415, E402

# ---- P1 M1 捕获器接线（a26ag）：env RVC_TRAIN_CAPTURE=1 开启，默认关零开销 ----
# 门1 open / 门2 close 在 _forward_br（dec 前向段）内；第五道门 dump 在 train.py
# net_g.forward 返回后。惰性导入：env 关闭时不触碰 _poc._capture。
_CAPTURE_ON = os.environ.get("RVC_TRAIN_CAPTURE") == "1"
_m1_cap_inst = None


def _m1_cap():
    """惰性取进程级捕获器（已 install 到 BatchRunner）。"""
    global _m1_cap_inst
    if _m1_cap_inst is None:
        from _poc import _capture as _c
        _m1_cap_inst = _c.install()   # install() 返回捕获器实例
    return _m1_cap_inst

# J11 诊断：tape.backward bp 分桶计时（env RVC_TRAIN_BWD_PROFILE=1，
# 按节点输出形状累计，退出时打印 top15）。
_BDPROF = os.environ.get("RVC_TRAIN_BWD_PROFILE") == "1"
if _BDPROF:
    import time as _t  # noqa: PLC0415
    import atexit as _atexit  # noqa: PLC0415

    _BPT: dict = {}

    def _dump_bwd_prof():
        if not _BPT:
            return
        top = sorted(_BPT.items(), key=lambda kv: -kv[1])[:40]
        tot = sum(_BPT.values())
        print(f"[BWD-PROF] total={tot:.2f}s", file=sys.stderr)
        for k, v in top:
            print(f"  shape {k}: {v:.3f}s", file=sys.stderr)

    _atexit.register(_dump_bwd_prof)


def _td_dtype():
    """tape 张量目标 dtype：RVC_TRAIN_TAPE_F32=1 → f32（训练墙钟），
    否则 f64（精确参考，供数值测试对照）。J8：默认开（f32 是 RVC 官方
    训练精度，GPU 计算本就 f32，梯度 f32 vs f64 训练 loss 同量级、轨迹
    分叉属精度噪声；实测 wall 10.3→8.9s/步）。数值测试须显式设
    RVC_TRAIN_TAPE_F32=0 保持 f64 参考。"""
    import os as _os  # noqa: PLC0415
    return np.float32 if _os.environ.get("RVC_TRAIN_TAPE_F32", "1") == "1" \
        else np.float64


def _td(a):
    """按 _td_dtype() 转换张量为 tape 中间/梯度 dtype。"""
    return np.asarray(a, dtype=_td_dtype())

__all__ = [
    "AutogradTape",
    "PosteriorEncoder",
    "TextEncoderTrain",
    "ResidualCouplingBlockTrain",
    "GeneratorNSFTrain",
    "SynthesizerTrnTrain",
    "MultiPeriodDiscriminator",
    "load_g_weights",
    "load_d_weights",
    "rand_slice_segments_np",
    "sequence_mask_np",
    "kl_loss_np",
]

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# 记录式自动微分
# ---------------------------------------------------------------------------
# T-H7：backward 链式 BatchRunner（单线程训练用模块全局）。conv 反向 bp 把
# gx/gw/gb 全录进这里：gx 留 GPU 传下一层（BatchTensor 链），gw/gb 段末由
# grad_of 统一下载；backward 尾部 commit 一次（一次 submit+wait 替代逐算子
# 1486 次同步往返）。numpy bp 消费 BatchTensor 梯度时经 __array__ 自动
# commit+下载（断链点），随后 conv bp 又接回 GPU——链表现为若干「段」。
_CHAIN_BR: "BatchRunner | None" = None
_CHAIN_DIRTY = False
# T-J1：weight_norm 还原（_deweight_any）per-step 缓存——dec GPU fwd 每步
# 重复计算 ~127 次（每 resblock 每层）而权重仅在 optimizer 步后变化；release
# 时（每步末）清空，跨步不失效。
_DWN_CACHE: dict = {}
_DWN_CACHE_ON = True
# T3-a：判别器图执行器单例（RVC_TRAIN_GRAPH=1 时惰性创建；训练末
# release_all 释放）。图执行器槽位 buffer 生命周期 = 缓存图生命周期。
_GRAPH_RUNNER = None


def _graph_runner():
    """返回模块级 GraphRunner（惰性创建；释放后重建）。"""
    global _GRAPH_RUNNER
    if _GRAPH_RUNNER is None:
        from runtime.graph_runner import GraphRunner  # noqa: PLC0415
        _GRAPH_RUNNER = GraphRunner()
    return _GRAPH_RUNNER


def _release_graph_runner() -> None:
    """训练末释放图执行器全部缓存图与槽位（幂等）。"""
    global _GRAPH_RUNNER
    if _GRAPH_RUNNER is not None:
        _GRAPH_RUNNER.release_all()
        _GRAPH_RUNNER = None


_DBG_DISC_BWD_MS = []  # TEMP-DBG：判别器 bp 图单次耗时累计

# ---- P1 M4 阶段0 埋点（a26ah §2 埋点 A）：env RVC_TRAIN_M4_PROBE=1，默认关 ----
# 只读观测 _CHAIN_DIRTY，用于区分「真提交 vs 幂等早退」；不改任何引擎语义。
# 埋点关闭时 _A26AH_ON=False ⇒ 下方全部分支短路，字节级零回归。
_M4_ON = os.environ.get("RVC_TRAIN_M4_PROBE", "0") == "1"
if _M4_ON:
    from runtime import m4_probe as _M4  # noqa: PLC0415
else:
    _M4 = None


def _m4():
    """惰性取 M4 探针模块（env 关闭时返回 None，调用方短路）。"""
    global _M4, _M4_ON
    if not _M4_ON:
        return None
    if _M4 is None:
        from runtime import m4_probe as _mod  # noqa: PLC0415
        _M4 = _mod
    return _M4


def _chain_br() -> "BatchRunner":
    """返回链式共享 BatchRunner（惰性创建；释放后重建）。"""
    global _CHAIN_BR, _CHAIN_DIRTY
    if _CHAIN_BR is None or _CHAIN_BR._released:
        from runtime.vulkan_ops import (  # noqa: PLC0415
            BatchRunner, get_context, wpers_get)
        _CHAIN_BR = BatchRunner(get_context())
    _CHAIN_DIRTY = True
    return _CHAIN_BR


def _as_ref(x):
    """T-H8：keep BatchTensor 原引用（保 GPU id 链）；numpy 照常 asarray。"""
    return x if isinstance(x, BatchTensor) else np.asarray(x)


def _commit_chain_br(async_: bool = False) -> None:
    """backward 尾统一提交链式批次（幂等；无录制时空批次即跳过）。

    ``async_=True``（T6，RVC_TRAIN_BWD_ASYNC=1）：异步提交不等待 GPU，
    调用方须在批量下载前 ``wait()``（BatchTensor.numpy() 会自动 wait，
    但 J18 ``_batch_download`` 直接读 buffer 必须显式等）。
    """
    global _CHAIN_DIRTY
    # ---- M4 埋点 A（a26ah §2）：入口三段计数（真提交/早退/无 br）----
    _m = _m4()
    if _m is not None:
        _m.a_entry(bool(_CHAIN_DIRTY))
    if not _CHAIN_DIRTY:
        if _m is not None:
            _m.a_early()          # A3 幂等早退：0 GPU 工作
        return
    if _CHAIN_BR is not None and not _CHAIN_BR._released:
        if _m is not None:
            import time as _t_m4mod  # noqa: PLC0415
            _m.a_released()       # A5 真提交
            _t_m4 = _t_m4mod.perf_counter()
        _CHAIN_BR.commit(async_=async_)
        if _m is not None:
            _m.a_commit_ms((_t_m4mod.perf_counter() - _t_m4) * 1000.0)
    elif _m is not None:
        _m.a_no_br()              # A4 DIRTY 真但 br 空/已释放
    _CHAIN_DIRTY = False


def _release_chain_br() -> None:
    """每训练步末归还链式 br 全部 buffer（防跨步累积；下步惰性重建复用
    ctx 全局池）。须在全部参数梯度下载（grad_of）之后调用。"""
    global _CHAIN_BR, _CHAIN_DIRTY, _DWN_CACHE
    if _CHAIN_BR is not None and not _CHAIN_BR._released:
        _CHAIN_BR.release()
    _CHAIN_BR = None
    _CHAIN_DIRTY = False
    _DWN_CACHE.clear()


# ---------------------------------------------------------------------------
# T3：backward numpy bp GPU 链化（RVC_TRAIN_BR_BWD_FULLGPU=1，默认 0 零回归）
# ---------------------------------------------------------------------------
# 背景（D3/任务 T3）：conv 类 bp 已录进 _chain_br 尾部一次提交，但消费
# BatchTensor 梯度的 numpy bp（matmul/mul 等）执行时经 `BatchTensor.__array__`
# 触发隐式 commit()+wait()——这是 1560 次 commit/10 步的主要来源。开启后，
# 满足条件（无广播/2D）的 matmul/mul bp 改为录进共享链式 BatchRunner 并返回
# BatchTensor，消除逐 bp commit+wait，backward 恢复「一段 GPU 链、尾部一次
# 提交 + J18 批量下载」。关闭（默认）时与历史路径逐位一致。
def _bwd_fullgpu() -> bool:
    """FULLGPU 链化开关（函数形式：测试脚本可动态切换 env）。"""
    import os as _os  # noqa: PLC0415
    return _os.environ.get("RVC_TRAIN_BR_BWD_FULLGPU", "0") == "1"


def _bwd_gpu_matmul(go, a_snap, b_snap):
    """FULLGPU：2D 无广播 matmul 反向录进链式 br，返回 (ga, gb) 两个
    BatchTensor；条件不符返回 None（调用方走 numpy，数值不变）。

    out = a @ b（a [M,K]、b [K,N]、out [M,N]）。ga = go @ b^T → [M,K]＝
    a.shape；gb = a^T @ go → [K,N]＝b.shape，形状精确匹配，下游
    _reduce_to 同形直接透传，链不断。仅支持 2D（引擎 matmul 为 2D
    kernel）；batched/广播 matmul 回退 numpy。
    """
    if not (isinstance(go, BatchTensor) and go.ndim == 2
            and a_snap.ndim == 2 and b_snap.ndim == 2):
        return None
    if go.shape[-1] != b_snap.shape[-1] or go.shape[0] != a_snap.shape[0]:
        return None  # 广播/形状不匹配 → numpy
    br = _chain_br()
    bT = np.swapaxes(b_snap, -1, -2)
    aT = np.swapaxes(a_snap, -1, -2)
    return br.matmul(go, bT), br.matmul(aT, go)


def _bwd_gpu_mul(go, a_snap, b_snap):
    """FULLGPU：同形（无广播）逐元素乘反向录进链式 br，返回 (ga, gb)。

    ga = go * b、gb = go * a；copy+mul_inplace 是纯 elementwise，GPU f32
    与 numpy f32 逐位一致（无累加顺序差异）。广播场景（如 mask 乘）GPU 无
    通用归约 → 回退 numpy。条件不符返回 None。
    """
    if not (isinstance(go, BatchTensor)
            and tuple(a_snap.shape) == tuple(go.shape)
            and tuple(b_snap.shape) == tuple(go.shape)):
        return None
    br = _chain_br()
    ga = br.mul_inplace(br.copy(go), b_snap)
    gb = br.mul_inplace(br.copy(go), a_snap)
    return ga, gb


class AutogradTape:
    """极简记录式 autograd；算子方法命名与 ``runtime.nn`` 对齐。

    每个包装方法完成前向并把 ``(bp_fn, out)`` 压入 ``self.ops``；
    ``bp_fn(go)`` 返回 ``[(input_ref, grad), ...]``。
    ``backward(seeds)`` 逆序传播，梯度按数组对象 ``id`` 累加；
    ``grad_of(arr)`` 取某数组梯度。
    """

    def __init__(self):
        self.ops = []
        self.grads = {}
        self.constants = set()
        self._brs = []  # J15：判别器 BR_FWD 链的 BatchRunner（bp 引用其
        # BatchTensor，须活到 backward 尾统一 release）

    def reset(self):
        self.ops = []
        self.grads = {}
        self.constants = set()
        self._brs = []

    def mark_const(self, arr):
        """把数组标记为常数：反向时不对它累加梯度（如 mask / 噪声 / har）。"""
        self.constants.add(id(np.asarray(arr)))

    def _push(self, bp, out):
        self.ops.append((bp, out))
        return out

    # ------------------------------------------------------------- 算子
    def linear(self, x, w, b=None):
        x64, w64 = _as_ref(x), np.asarray(w)
        b64 = np.asarray(b) if b is not None else None
        out = x64 @ w64.T + (b64 if b64 is not None else 0.0)

        def bp(go, _x=x64, _w=w64, _b=b64):
            gx, gw, gb = nb.linear_backward(_x, _w, go, _b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def conv1d(self, x, w, b=None, stride=1, padding=0, dilation=1):
        x64, w64 = _as_ref(x), np.asarray(w)
        b64 = np.asarray(b) if b is not None else None
        out = _conv1d_np(x64, w64, b64, stride=stride, padding=padding,
                         dilation=dilation)

        def bp(go, _x=x64, _w=w64, _b=b64, s=stride, p=padding, d=dilation):
            # VK-02（R-TRAIN-008 / P-TRAIN-008）：conv1d 反向 GPU 化。
            # T1（perf）：默认开启——实测 2.09×（numpy 108s vs conv1d 51.7s 干净
            # wall 3 步中位，见 _diag/train_perf_opt/T1_conv1d_bwd_gpu.md）；引擎
            # f32 累加与 numpy f64→f16 参考在最低位有 ≤2 f16 ULP 差异（容忍
            # 口径）；设 RVC_TRAIN_CONV1D_BWD_GPU=0 关闭即逐位等于原 numpy。
            import os as _os  # noqa: PLC0415
            from runtime import backend as _be  # noqa: PLC0415

            if _os.environ.get("RVC_TRAIN_CONV1D_BWD_GPU", "1") == "1":
                # T-H7 链式：go 为 BatchTensor（GPU 链上）或显式开 BR 时，
                # 统一录进共享 BatchRunner，backward 尾提交；gx 留 GPU。
                if isinstance(go, BatchTensor) or (
                        _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"):
                    gx, gw, gb = _be.conv1d_backward(
                        _x, _w, go, stride=s, padding=p, dilation=d,
                        b=_b, br=_chain_br())
                else:
                    gx, gw, gb = _be.conv1d_backward(
                        _x, _w, go, stride=s, padding=p, dilation=d, b=_b)
            else:
                gx, gw, gb = nb.conv1d_backward(
                    _x, _w, go, stride=s, padding=p, dilation=d, b=_b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def conv1d_groups(self, x, w, b, groups, stride=1, padding=0):
        """带 groups 的 1D 卷积（DiscriminatorS 用）。"""
        x64, w64 = _as_ref(x), np.asarray(w)
        xs = self._snap(x64)
        B, C, T = x64.shape
        O, Ci, K = w64.shape
        if O % groups != 0 or C % groups != 0:
            raise ValueError(f"conv1d_groups: O={O}, C={C}, groups={groups}")
        outs = []
        for g in range(groups):
            xg = x64[:, g * (C // groups): (g + 1) * (C // groups), :]
            wg = w64[g * (O // groups): (g + 1) * (O // groups), :, :]
            outs.append(_conv1d_np(xg, wg, b[g * (O // groups):
                                             (g + 1) * (O // groups)],
                                   stride=stride, padding=padding))
        out = np.concatenate(outs, axis=1)

        def bp(go, _x=x64, _xs=xs, _w=w64, _b=b, gs=groups, s=stride,
               p=padding):
            import os as _os  # noqa: PLC0415
            from runtime import backend as _be  # noqa: PLC0415

            use_gpu = _os.environ.get("RVC_TRAIN_CONV1D_BWD_GPU", "1") == "1"
            # 阶段J（J6）：分组反向 kernel（op24/25/26）——全量 x/w/go 各上传
            # 一次 + GPU 内分组计算，替代每组 3 op 单发（680→3 dispatch）。
            # 默认开；数值 f32 vs f64 参考 ≤1e-4（容差内）；异常自动回退。
            if (use_gpu and
                    _os.environ.get("RVC_TRAIN_CONV1D_GROUPS_BWD_GPU", "1") == "1"):
                gx_t, gw_t, gb_t = _be.conv1d_groups_backward(
                    _xs, _w, go, gs, stride=s, padding=p, b=_b, br=_chain_br())
                return [(_x, gx_t), (_w, gw_t), (_b, gb_t)]
            gx = np.zeros_like(_xs)
            gw = np.zeros_like(_w)
            gb = np.zeros_like(_b)
            # 阶段A（A4 v3）：组间合并——BR_BWD_GROUPS=1（默认 0，实测无收益
            # +0.5s：省 submit 被 br 管理/上传包络抵消，同 A3 教训——输出全
            # 回 host 时批量收益有限；阶段 C GPU 内链才是正路）。=1 时把全部
            # 组的 conv1d_backward 录进一个 BatchRunner（每批 ≤170 组=340
            # 算子，引擎 dispatch 上限 ~384），少次 commit；同 kernel 同参数
            # 逐位一致（零回归）。组间无数据依赖（各组独立切片）→ 录制安全。
            # 录制模式返回 (gx BatchTensor, gw BatchTensor, gb ndarray|None)。
            br_on = (use_gpu and _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"
                     and _os.environ.get("RVC_TRAIN_BR_BWD_GROUPS", "0") == "1")
            if br_on:
                from runtime.vulkan_ops import (  # noqa: PLC0415
                    BatchRunner as _BR3, get_context as _gctx3)
                _br = _BR3(_gctx3())
                try:
                    recs = []
                    for g in range(gs):
                        cs = g * (C // gs)
                        os_ = g * (O // gs)
                        bg = _b[os_: os_ + (O // gs)]
                        r = _be.conv1d_backward(
                            _xs[:, cs: cs + (C // gs), :],
                            _w[os_: os_ + (O // gs), :, :],
                            go[:, os_: os_ + (O // gs), :], stride=s,
                            padding=p, b=bg, br=_br)
                        recs.append((cs, os_, r))
                        if len(recs) >= 170:
                            _br.commit()
                            for cs2, os_2, (gx_r, gw_r, gb_r) in recs:
                                if isinstance(gx_r, np.ndarray):
                                    gx_g, gw_g = gx_r, gw_r
                                else:
                                    # 分组卷积权重 C 维已是 C//gs（_w.shape[1]）
                                    gx_g, gw_g = (gx_r.numpy(),
                                                  gw_r.numpy().reshape(
                                                      O // gs, _w.shape[1], K))
                                gx[:, cs2: cs2 + (C // gs), :] += gx_g
                                gw[os_2: os_2 + (O // gs), :, :] += gw_g
                                gb[os_2: os_2 + (O // gs)] += gb_r
                            recs = []
                    if recs:
                        _br.commit()
                        for cs2, os_2, (gx_r, gw_r, gb_r) in recs:
                            if isinstance(gx_r, np.ndarray):
                                gx_g, gw_g = gx_r, gw_r
                            else:
                                gx_g, gw_g = (gx_r.numpy(),
                                              gw_r.numpy().reshape(
                                                  O // gs, _w.shape[1], K))
                            gx[:, cs2: cs2 + (C // gs), :] += gx_g
                            gw[os_2: os_2 + (O // gs), :, :] += gw_g
                            gb[os_2: os_2 + (O // gs)] += gb_r
                finally:
                    _br.release()
                return [(_x, gx), (_w, gw), (_b, gb)]
            for g in range(gs):
                cs = g * (C // gs)
                os_ = g * (O // gs)
                bg = _b[os_: os_ + (O // gs)]
                if use_gpu:
                    gx_g, gw_g, gb_g = _be.conv1d_backward(
                        _xs[:, cs: cs + (C // gs), :],
                        _w[os_: os_ + (O // gs), :, :],
                        go[:, os_: os_ + (O // gs), :], stride=s, padding=p,
                        b=bg)
                else:
                    gx_g, gw_g, gb_g = nb.conv1d_backward(
                        _xs[:, cs: cs + (C // gs), :],
                        _w[os_: os_ + (O // gs), :, :],
                        go[:, os_: os_ + (O // gs), :], stride=s, padding=p,
                        b=bg)
                gx[:, cs: cs + (C // gs), :] += gx_g
                gw[os_: os_ + (O // gs), :, :] += gw_g
                gb[os_: os_ + (O // gs)] += gb_g
            return [(_x, gx), (_w, gw), (_b, gb)]

        return self._push(bp, out)

    def conv2d(self, x, w, b=None, stride=1, padding=0, dilation=1):
        x64, w64 = _as_ref(x), np.asarray(w)
        b64 = np.asarray(b) if b is not None else None
        out = _conv2d_np(x64, w64, b64, stride=stride, padding=padding,
                         dilation=dilation)

        def bp(go, _x=x64, _w=w64, _b=b64, s=stride, p=padding, d=dilation):
            # T2（R-TRAIN-008 / P-TRAIN-008 续）：conv2d 反向 GPU 化。
            # T1（perf）：默认开启——conv2d 边际收益 1.62×（仅 conv1d 51.7s
            # vs 双开 32.0s 干净 wall 中位，见 T1 报告）；引擎 f32 累加与 numpy
            # f64→f16 参考在最低位有 ≤2 f16 ULP 差异（容忍口径，T2 回归基准
            # 已文档化）；设 RVC_TRAIN_CONV2D_BWD_GPU=0 关闭即逐位等于原
            # numpy。
            import os as _os  # noqa: PLC0415
            from runtime import backend as _be  # noqa: PLC0415

            if _os.environ.get("RVC_TRAIN_CONV2D_BWD_GPU", "1") == "1":
                # T-H7 链式：go 为 BatchTensor 或 BR_BWD=1 时录共享 BatchRunner。
                if isinstance(go, BatchTensor) or (
                        _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"):
                    gx, gw, gb = _be.conv2d_backward(
                        _x, _w, go, stride=s, padding=p, dilation=d,
                        b=_b, br=_chain_br())
                else:
                    gx, gw, gb = _be.conv2d_backward(
                        _x, _w, go, stride=s, padding=p, dilation=d, b=_b)
            else:
                gx, gw, gb = nb.conv2d_backward(
                    _x, _w, go, stride=s, padding=p, dilation=d, b=_b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def conv_transpose1d(self, x, w, b=None, stride=1, padding=0,
                         output_padding=0, dilation=1):
        x64, w64 = _as_ref(x), np.asarray(w)
        b64 = np.asarray(b) if b is not None else None
        out = _conv_transpose1d_np(x64, w64, b64, stride=stride,
                                   padding=padding,
                                   output_padding=output_padding,
                                   dilation=dilation)

        def bp(go, _x=x64, _w=w64, _b=b64, s=stride, p=padding,
               op=output_padding, d=dilation):
            gx, gw, gb = nb.conv_transpose1d_backward(
                _x, _w, go, stride=s, padding=p, output_padding=op,
                dilation=d, b=_b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        x64, g64, b64 = _as_ref(x), np.asarray(gamma), np.asarray(beta)
        mean = x64.mean(axis=-1, keepdims=True)
        var = x64.var(axis=-1, keepdims=True)
        out = (x64 - mean) / np.sqrt(var + eps) * g64 + b64

        def bp(go, _x=x64, _g=g64, _b=b64, e=eps):
            gx, gg, gb = nb.layer_norm_backward(_x, _g, _b, go, e)
            return [(_x, gx), (_g, gg), (_b, gb)]

        return self._push(bp, out)

    def softmax(self, x, axis=-1):
        x64 = _as_ref(x)
        m = np.max(x64, axis=axis, keepdims=True)
        e = np.exp(x64 - m)
        out = e / np.sum(e, axis=axis, keepdims=True)

        def bp(go, _o=out):
            return [(x64, nb.softmax_backward(_o, go))]

        return self._push(bp, out)

    def relu(self, x):
        x64 = _as_ref(x)

        def bp(go, _x=x64):
            return [(_x, nb.relu_backward(_x, go))]

        return self._push(bp, np.maximum(x64, 0.0))

    def leaky_relu(self, x, slope=0.1):
        x64 = _as_ref(x)

        def bp(go, _x=x64, s=slope):
            # T-H7 链式：go 为 BatchTensor 时 GPU 录（消除 numpy 断链点）
            if isinstance(go, BatchTensor):
                return [(_x, _chain_br().leaky_relu_backward(go, _x, s))]
            return [(_x, nb.leaky_relu_backward(_x, go, s))]

        return self._push(bp, np.where(x64 >= 0, x64, slope * x64))

    # ---- 阶段A（BatchRunner 化）：外部已算好 forward（引擎批量提交）时的
    # ---- 纯记录方法：只压 bp 不重算 forward（out 由 BatchRunner 提供）。
    @staticmethod
    def _snap(x):
        """record/bp 输入值快照：BatchTensor → 立即下载为 ndarray。

        backward 在 forward 全部完成后执行，此时 forward runner 的 GPU buffer
        已被后续段（包括 _chain_br 自身的梯度分配）覆写；因此 bp 闭包不能在
        backward 时点再经 __array__ 下载输入值，必须在此（record 调用时点、
        forward 值就绪）固化为快照。ref 仍保留原对象（id 链），快照仅供计算。
        """
        return x if not isinstance(x, BatchTensor) else np.asarray(x)

    def record_conv2d(self, out, x, w, b=None, stride=1, padding=0, dilation=1):
        """记录 conv2d 算子（forward 已由 BatchRunner/图执行；out 保持原引用
        （BatchTensor 或 ndarray），与 record_conv1d/leaky 一致——否则 out 转
        f64 新副本后 id 与上游 bp 挂载的 ref 不匹配，conv bp 永不触发，
        convs 权重梯度静默缺失（T3-c 门禁暴露）。bp 闭包不依赖 out 值。"""
        x64, w64 = _as_ref(x), np.asarray(w)
        xs = self._snap(x64)
        b64 = np.asarray(b) if b is not None else None

        def bp(go, _x=x64, _xs=xs, _w=w64, _b=b64, s=stride, p=padding,
               d=dilation):
            import os as _os  # noqa: PLC0415
            from runtime import backend as _be  # noqa: PLC0415

            if _os.environ.get("RVC_TRAIN_CONV2D_BWD_GPU", "1") == "1":
                # J16：与 record_conv1d 对齐——BR_BWD=1 时录进共享链式
                # BatchRunner（x 为 BatchTensor 时零上传，一次 commit 替代
                # 每算子 local BR 往返——P bp 0.34s/步 → 预期 ~0.15s）。
                if isinstance(go, BatchTensor) or (
                        _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"):
                    gx, gw, gb = _be.conv2d_backward(
                        _xs, _w, go, stride=s, padding=p, dilation=d,
                        b=_b, br=_chain_br())
                else:
                    gx, gw, gb = _be.conv2d_backward(
                        _xs, _w, go, stride=s, padding=p, dilation=d, b=_b)
            else:
                gx, gw, gb = nb.conv2d_backward(
                    _xs, _w, go, stride=s, padding=p, dilation=d, b=_b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def record_leaky_relu(self, out, x, slope=0.1):
        """记录 leaky_relu 算子（forward 已由 BatchRunner 执行；out 保持 f32
        引用，与 record_identity/record_conv1d 一致，避免断下游 id 链）。"""
        x64 = _as_ref(x)
        xs = self._snap(x64)

        def bp(go, _x=x64, _xs=xs, s=slope):
            # T-H7 链式：go 为 BatchTensor 时 GPU 录（消除 numpy 断链点）
            if isinstance(go, BatchTensor):
                return [(_x, _chain_br().leaky_relu_backward(go, _xs, s))]
            return [(_x, nb.leaky_relu_backward(_xs, go, s))]

        return self._push(bp, out)

    # ---- 阶段H（H5 后续：dec 前向 BR 化）纯记录方法扩展：bp 与对应
    # ---- 算子一致；out 保持 f32（bp 均只依赖输入捕获，不依赖 out 值，
    # ---- 省 f64 转换）。
    def record_identity(self, out, x):
        """记录恒等算子（copy）：梯度原样透传（out 与 x 同形状）。"""
        # T-H7/H8：x 可为 BatchTensor（dec GPU fwd 链）——保持原引用保 id；
        # bp 仅透传 go，不依赖 x 值。
        x64 = _as_ref(x)

        def bp(go, _x=x64):
            return [(_x, go)]

        return self._push(bp, out)

    def record_conv1d(self, out, x, w, b=None, stride=1, padding=0, dilation=1,
                      w_v=None, w_g=None, wtag="g"):
        """记录 conv1d 算子（forward 已由 BatchRunner 执行）。bp 同 conv1d。

        T0.3 泄漏修复：``w_v/w_g``（weight_norm 参数，持久对象）非 None 时
        backward 用 ``wpers_get_wn``（按 (id(v),id(g)) 缓存展开权重常驻
        GPU buffer，跨步命中）——原实现按每步新建的展开数组 id 缓存导致
        每步 +~114 个 PersistentBuffer 永久累积（buf_count 泄漏源）。
        ``wtag``：生成器权重 "g"（D 步后刷新，与判别器 "d" 组分离）。
        """
        x64 = _as_ref(x)
        xs = self._snap(x64)
        w64 = np.asarray(w)
        b64 = np.asarray(b) if b is not None else None
        wv64 = _as_ref(w_v) if w_v is not None else None
        wg64 = _as_ref(w_g) if w_g is not None else None

        def bp(go, _x=x64, _xs=xs, _w=w64, _b=b64, s=stride, p=padding,
               d=dilation, _wv=wv64, _wg=wg64, _wt=wtag):
            import os as _os  # noqa: PLC0415
            from runtime import backend as _be  # noqa: PLC0415
            import time as _tm  # noqa: PLC0415
            _t0 = _tm.perf_counter()  # TEMP-DBG
            _t1 = _t0
            try:
                if _os.environ.get("RVC_TRAIN_CONV1D_BWD_GPU", "1") == "1":
                    # T-H7 链式：go BatchTensor（GPU 链上）或 BR_BWD=1 时录进共享
                    # BatchRunner（x 为 BatchTensor 时零上传：im2col 直接用 GPU）。
                    if isinstance(go, BatchTensor) or (
                            _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"):
                        _t1 = _tm.perf_counter()  # TEMP-DBG
                        if _wv is not None and _wg is not None:
                            from runtime import vulkan_ops as _vo  # noqa: PLC0415
                            _buf_w = _vo.wpers_get_wn(
                                _wv, _wg, _chain_br(), _wt)
                        else:
                            _buf_w = None
                        gx, gw, gb = _be.conv1d_backward(
                            _xs, _w, go, stride=s, padding=p, dilation=d,
                            b=_b, br=_chain_br(), buf_w=_buf_w, wtag=_wt)
                    else:
                        _t1 = _tm.perf_counter()  # TEMP-DBG
                        gx, gw, gb = _be.conv1d_backward(
                            _xs, _w, go, stride=s, padding=p, dilation=d, b=_b)
                else:
                    gx, gw, gb = nb.conv1d_backward(
                        _xs, _w, go, stride=s, padding=p, dilation=d, b=_b)
            finally:
                if _os.environ.get("RVC_TRAIN_BWD_PROFILE"):  # TEMP-DBG
                    _dt = (_tm.perf_counter() - _t0) * 1000.0
                    if _dt > 25.0:
                        print(f"[BP-DBG] L553 x={np.shape(x64)} "
                              f"go_bt={isinstance(go, BatchTensor)} "
                              f"pre={(_t1-_t0)*1000:.1f}ms "
                              f"call={(_tm.perf_counter()-_t1)*1000:.1f}ms "
                              f"total={_dt:.0f}ms", file=sys.stderr)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def record_conv1d_groups(self, out, x, w, b, groups, stride=1, padding=0):
        """记录 conv1d_groups 算子（forward 已由 BatchRunner op23 执行）。

        bp 复用 conv1d_groups 的组循环（每组 _be.conv1d_backward 单发）；
        后续单独 kernel 化（本轮先验证 op23 的 fwd 收益与零回归）。
        """
        x64 = _as_ref(x)
        xs = self._snap(x64)
        w64 = np.asarray(w)
        b64 = np.asarray(b) if b is not None else None
        B, C, T = x64.shape
        O, Ci, K = w64.shape

        def bp(go, _x=x64, _xs=xs, _w=w64, _b=b64, gs=groups, s=stride,
               p=padding):
            import os as _os  # noqa: PLC0415
            from runtime import backend as _be  # noqa: PLC0415

            use_gpu = _os.environ.get("RVC_TRAIN_CONV1D_BWD_GPU", "1") == "1"
            # 阶段J（J6）：分组反向 kernel（op24/25/26）——全量 x/w/go 各上传
            # 一次 + GPU 内分组 im2col/matmul/convT，替代 680 次单发
            # conv1d_backward（省 ~3s/步）。默认开；数值 f32 vs f64 参考
            # ≤1e-4（容差内）；异常自动回退组循环。
            if (use_gpu and
                    _os.environ.get("RVC_TRAIN_CONV1D_GROUPS_BWD_GPU", "1") == "1"):
                gx_t, gw_t, gb_t = _be.conv1d_groups_backward(
                    _xs, _w, go, gs, stride=s, padding=p, b=_b, br=_chain_br())
                return [(_x, gx_t), (_w, gw_t), (_b, gb_t)]
            gx = np.zeros_like(_xs)
            gw = np.zeros_like(_w)
            gb = np.zeros_like(_b)
            br_on = (use_gpu and _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"
                     and _os.environ.get("RVC_TRAIN_BR_BWD_GROUPS", "0") == "1")
            if br_on:
                from runtime.vulkan_ops import (  # noqa: PLC0415
                    BatchRunner as _BR3, get_context as _gctx3)
                _br = _BR3(_gctx3())
                try:
                    recs = []
                    for g in range(gs):
                        cs = g * (C // gs)
                        os_ = g * (O // gs)
                        bg = _b[os_: os_ + (O // gs)]
                        r = _be.conv1d_backward(
                            _xs[:, cs: cs + (C // gs), :],
                            _w[os_: os_ + (O // gs), :, :],
                            go[:, os_: os_ + (O // gs), :], stride=s,
                            padding=p, b=bg, br=_br)
                        recs.append((cs, os_, r))
                        if len(recs) >= 170:
                            _br.commit()
                            for cs2, os_2, (gx_r, gw_r, gb_r) in recs:
                                if isinstance(gx_r, np.ndarray):
                                    gx_g, gw_g = gx_r, gw_r
                                else:
                                    gx_g, gw_g = (gx_r.numpy(),
                                                  gw_r.numpy().reshape(
                                                      O // gs, _w.shape[1], K))
                                gx[:, cs2: cs2 + (C // gs), :] += gx_g
                                gw[os_2: os_2 + (O // gs), :, :] += gw_g
                                gb[os_2: os_2 + (O // gs)] += gb_r
                            recs = []
                    if recs:
                        _br.commit()
                        for cs2, os_2, (gx_r, gw_r, gb_r) in recs:
                            if isinstance(gx_r, np.ndarray):
                                gx_g, gw_g = gx_r, gw_r
                            else:
                                gx_g, gw_g = (gx_r.numpy(),
                                              gw_r.numpy().reshape(
                                                  O // gs, _w.shape[1], K))
                            gx[:, cs2: cs2 + (C // gs), :] += gx_g
                            gw[os_2: os_2 + (O // gs), :, :] += gw_g
                            gb[os_2: os_2 + (O // gs)] += gb_r
                finally:
                    _br.release()
                return [(_x, gx), (_w, gw), (_b, gb)]
            for g in range(gs):
                cs = g * (C // gs)
                os_ = g * (O // gs)
                bg = _b[os_: os_ + (O // gs)]
                if use_gpu:
                    gx_g, gw_g, gb_g = _be.conv1d_backward(
                        _xs[:, cs: cs + (C // gs), :],
                        _w[os_: os_ + (O // gs), :, :],
                        go[:, os_: os_ + (O // gs), :], stride=s, padding=p,
                        b=bg)
                else:
                    gx_g, gw_g, gb_g = nb.conv1d_backward(
                        _xs[:, cs: cs + (C // gs), :],
                        _w[os_: os_ + (O // gs), :, :],
                        go[:, os_: os_ + (O // gs), :], stride=s, padding=p,
                        b=bg)
                gx[:, cs: cs + (C // gs), :] += gx_g
                gw[os_: os_ + (O // gs), :, :] += gw_g
                gb[os_: os_ + (O // gs)] += gb_g
            return [(_x, gx), (_w, gw), (_b, gb)]

        return self._push(bp, out)

    def record_conv_transpose1d(self, out, x, w, b=None, stride=1, padding=0,
                                output_padding=0, dilation=1):
        """记录 conv_transpose1d（forward 已由 BatchRunner 执行）。bp 同
        conv_transpose1d。"""
        x64 = _as_ref(x)
        xs = self._snap(x64)
        w64 = np.asarray(w)
        b64 = np.asarray(b) if b is not None else None

        def bp(go, _x=x64, _xs=xs, _w=w64, _b=b64, s=stride, p=padding,
               op=output_padding, d=dilation):
            import os as _os  # noqa: PLC0415
            from runtime import nn_backward as _nb  # noqa: PLC0415
            # J22：gx 用引擎 conv1d 表达（数学：convT 反向的 x 梯度 =
            # conv1d(go, w, stride=s, padding=p, dilation=d)，op 不影响
            # 输出长度，单测 maxdiff 1e-5）。go 为 BatchTensor（dec GPU
            # 链）时 gx 留 GPU；gw/gb 为小张量 numpy（einsum 快）。
            gpu = _os.environ.get("RVC_TRAIN_CONVT1D_BWD_GPU", "1") == "1"
            if gpu and (isinstance(go, BatchTensor) or
                        _os.environ.get("RVC_TRAIN_BR_BWD", "1") == "1"):
                _wf = np.asarray(_w, np.float32)
                gx = _chain_br().conv1d(
                    go, _wf, None, stride=s, padding=p, dilation=d)
                go_np = np.asarray(go) if isinstance(go, BatchTensor) else go
                x_np = _xs
                B_, C_, T_ = x_np.shape
                O_ = _w.shape[1]
                K_ = _w.shape[2]
                oL_ = go_np.shape[2]
                gw = np.zeros_like(_w, dtype=np.float32)
                for k in range(K_):
                    pos = np.arange(T_) * s - p + k * d
                    valid = (pos >= 0) & (pos < oL_)
                    if not valid.any():
                        continue
                    tv = np.nonzero(valid)[0]
                    g = go_np[:, :, pos[tv]]
                    gw[:, :, k] += np.einsum(
                        "bot,bct->co", g, x_np[:, :, tv], optimize=True)
                gb = go_np.sum(axis=(0, 2)) if _b is not None else None
                res = [(_x, gx), (_w, gw)]
                if _b is not None:
                    res.append((_b, gb))
                return res
            gx, gw, gb = _nb.conv_transpose1d_backward(
                _xs, _w, go, stride=s, padding=p, output_padding=op,
                dilation=d, b=_b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def record_add(self, out, a, b):
        """记录 add 算子（forward 已由 BatchRunner 执行）。bp 同 add。"""
        a64 = _as_ref(a)
        b64 = _as_ref(b)

        def bp(go, _a=a64, _b=b64):
            return [(_a, _reduce_to(go, _a.shape)),
                    (_b, _reduce_to(go, _b.shape))]

        return self._push(bp, out)

    def record_mul_const(self, out, a, c):
        """记录 mul_const 算子（forward 已由 BatchRunner 执行）。bp 同
        mul_const。"""
        a64 = _as_ref(a)
        c = float(c)

        def bp(go, _a=a64, _c=c):
            return [(_a, go * _c)]

        return self._push(bp, out)

    # ---- T3-d：enc attn 链纯记录方法扩展（GPU 链式基准 forward_br 与
    # ---- 图化 forward_graph 共用；bp 与对应原生算子一致；out 保持 f32
    # ---- 引用保 id 链，bp 依赖 out 值的（softmax）用闭包捕获 BatchTensor
    # ---- backward 时经 __array__ 自动下载）。
    def record_matmul(self, out, a, b, trans_b=False):
        """记录 matmul（forward 已由 BatchRunner 执行）。bp 同 matmul。"""
        a64 = _as_ref(a)
        b64 = _as_ref(b)
        asnap = self._snap(a64)
        bsnap = self._snap(b64)

        def bp(go, _a=a64, _b=b64, _as=asnap, _bs=bsnap, _tb=trans_b):
            if _bwd_fullgpu() and not _tb:
                _g = _bwd_gpu_matmul(go, _as, _bs)
                if _g is not None:
                    _ga, _gb = _g
                    return [(_a, _reduce_to(_ga, _a.shape)),
                            (_b, _reduce_to(_gb, _b.shape))]
            _b_op = np.swapaxes(_bs, -1, -2) if _tb else _bs
            ga = np.matmul(go, np.swapaxes(_b_op, -1, -2))
            gb = np.matmul(np.swapaxes(_as, -1, -2), go)
            if _tb:
                gb = np.swapaxes(gb, -1, -2)
            ga = _reduce_to(ga, _a.shape)
            gb = _reduce_to(gb, _b.shape)
            return [(_a, ga), (_b, gb)]

        return self._push(bp, out)

    def record_mul(self, out, a, b):
        """记录 mul（forward 已由 BatchRunner 执行）。bp 同 mul。"""
        a64 = _as_ref(a)
        b64 = _as_ref(b)
        asnap = self._snap(a64)
        bsnap = self._snap(b64)

        def bp(go, _a=a64, _b=b64, _as=asnap, _bs=bsnap):
            if _bwd_fullgpu():
                _g = _bwd_gpu_mul(go, _as, _bs)
                if _g is not None:
                    _ga, _gb = _g
                    return [(_a, _reduce_to(_ga, _a.shape)),
                            (_b, _reduce_to(_gb, _b.shape))]
            ga, gb = nb.mul_backward(_as, _bs, go)
            return [(_a, _reduce_to(ga, _a.shape)),
                    (_b, _reduce_to(gb, _b.shape))]

        return self._push(bp, out)

    def record_softmax(self, out, x):
        """记录 softmax（forward 已由 BatchRunner op8 执行）。bp 同 softmax
        （闭包捕获 out——return 时点下载软max输出值快照）。"""
        x64 = _as_ref(x)
        out64 = _as_ref(out)
        osnap = self._snap(out64)

        def bp(go, _o=osnap):
            return [(x64, nb.softmax_backward(_o, go))]

        return self._push(bp, out)

    def record_layer_norm(self, out, x, gamma, beta, eps=1e-5):
        """记录 layer_norm（forward 已由 BatchRunner op9 执行）。bp 同
        layer_norm（依赖 x/gamma/beta 值——record 时点快照）。"""
        x64 = _as_ref(x)
        xs = self._snap(x64)
        g64 = np.asarray(gamma)
        b64 = np.asarray(beta)

        def bp(go, _x=x64, _xs=xs, _g=g64, _b=b64, e=eps):
            gx, gg, gb = nb.layer_norm_backward(_xs, _g, _b, go, e)
            return [(_x, gx), (_g, gg), (_b, gb)]

        return self._push(bp, out)

    def record_relu(self, out, x):
        """记录 relu（forward 已由 BatchRunner op14 执行）。bp 同 relu。"""
        x64 = _as_ref(x)
        xs = self._snap(x64)

        def bp(go, _x=x64, _xs=xs):
            return [(_x, nb.relu_backward(_xs, go))]

        return self._push(bp, out)

    def record_linear(self, out, x, w, b=None):
        """记录 linear（forward 已由 BatchRunner matmul+bias_add 执行）。
        bp 同 linear（nb.linear_backward：gw 挂原 w 数组、gb 挂原 bias）。"""
        x64 = _as_ref(x)
        xs = self._snap(x64)
        w64 = np.asarray(w)
        b64 = np.asarray(b) if b is not None else None

        def bp(go, _x=x64, _xs=xs, _w=w64, _b=b64):
            gx, gw, gb = nb.linear_backward(_xs, _w, go, _b)
            res = [(_x, gx), (_w, gw)]
            if _b is not None:
                res.append((_b, gb))
            return res

        return self._push(bp, out)

    def record_embedding(self, out, ids, table):
        """记录 embedding 查表（forward 已由 Python np.take 完成，table 恒
        为 GPU 外数组）。bp 同 embedding（只回 table 梯度，ids 侧无梯度——
        与原 numpy forward 语义一致）。"""
        ids64 = np.asarray(ids)
        t64 = np.asarray(table)

        def bp(go, _ids=ids64, _t=t64):
            return [(_t, nb.embedding_backward(_ids, go, _t.shape))]

        return self._push(bp, out)

    def record_split_gather(self, out, parts, part_shape):
        """记录「分头组装」：out = 各 part（[P,kc]，每 (b,h) 独立 Buffer）经
        transpose+stack+reshape 组装成 [B,hid,P]（forward 由 Python 完成）。
        bp 逆重排拆回每 (b,h) 并累加（图侧 attn_in 梯度无法用单个视图
        identity 表达——内存重排无图 op）。parts 为 numpy 数组列表，梯度
        按 ref 累加。"""
        parts_t = tuple(parts)
        ps = tuple(part_shape)

        def bp(go, _parts=parts_t, _ps=ps):
            B_ = go.shape[0]
            H_ = len(_parts) // B_
            kc_, P_ = _ps
            # go [B,hid,P] → 逆 reshape(B,H,kc,P)+transpose → [B,H,P,kc]
            g4 = go.reshape(B_, H_, kc_, P_).transpose(0, 1, 3, 2)
            res = []
            for idx, p in enumerate(_parts):
                b_, h_ = divmod(idx, H_)
                res.append((p, np.asarray(g4[b_, h_])))
            return res

        return self._push(bp, out)

    def gelu(self, x):
        x64 = _as_ref(x)

        def bp(go, _x=x64):
            return [(_x, nb.gelu_backward(_x, go))]

        return self._push(bp, _gelu_np(x64))

    def sigmoid(self, x):
        x64 = _as_ref(x)
        safe = np.clip(x64, -50.0, 50.0)
        out = 1.0 / (1.0 + np.exp(-safe))

        def bp(go, _o=out):
            return [(x64, nb.sigmoid_backward(_o, go))]

        return self._push(bp, out)

    def tanh(self, x):
        x64 = _as_ref(x)
        out = np.tanh(x64)

        def bp(go, _o=out):
            return [(x64, nb.tanh_backward(_o, go))]

        return self._push(bp, out)

    def embedding(self, ids, table):
        ids64 = np.asarray(ids)
        t64 = np.asarray(table)
        out = _embedding_np(ids64, t64)

        def bp(go, _ids=ids64, _t=t64):
            return [(_t, nb.embedding_backward(_ids, go, _t.shape))]

        return self._push(bp, out)

    def einsum(self, expr, *arrays):
        arr64 = [np.asarray(a) for a in arrays]
        # T2：einsum_path 缓存（expr+形状确定性→路径一致→逐位等价 optimize=True，
        # maxdiff=0 零回归）；消除每次调用重算路径规划（调研 7.7s/3 步）。
        path = nb.einsum_optimize_path(expr, tuple(a.shape for a in arr64))
        out = np.einsum(expr, *arr64, optimize=path)

        def bp(go, _expr=expr, _arrs=tuple(arr64)):
            gs = nb.einsum_backward(_expr, _arrs, go)
            return list(zip(_arrs, gs))

        return self._push(bp, out)

    def matmul(self, a, b, trans_b=False):
        """矩阵乘/多维 batched matmul（收缩最后一维；T1 attn einsum 等价改写）。

        ``trans_b=True`` 时等价于 ``a @ b.T``（收缩 a 最后维与 b 最后维）——
        覆盖 attn 的 ``bhqd,bhkd->bhqk``（b^T over kc）与 ``bhqd,md->bhqm``。
        out = np.matmul(a, b)（或 a @ b.T），数值口径与 einsum 两输入 BLAS 路径
        一致（实测 maxdiff=0）。bp：ga = go @ (b_T)^T 归约到 a.shape；
        gb = a^T @ go 归约到 b.shape（trans_b 时逆转置映射回原布局）。
        """
        a64, b64 = _as_ref(a), _as_ref(b)
        b_op = np.swapaxes(b64, -1, -2) if trans_b else b64
        out = np.matmul(a64, b_op)

        def bp(go, _a=a64, _b=b64, _tb=trans_b):
            if _bwd_fullgpu() and not _tb:
                _g = _bwd_gpu_matmul(go, np.asarray(_a), np.asarray(_b))
                if _g is not None:
                    _ga, _gb = _g
                    return [(_a, _reduce_to(_ga, _a.shape)),
                            (_b, _reduce_to(_gb, _b.shape))]
            _b_op = np.swapaxes(_b, -1, -2) if _tb else _b
            ga = np.matmul(go, np.swapaxes(_b_op, -1, -2))
            gb = np.matmul(np.swapaxes(_a, -1, -2), go)
            if _tb:
                gb = np.swapaxes(gb, -1, -2)
            ga = _reduce_to(ga, _a.shape)
            gb = _reduce_to(gb, _b.shape)
            return [(_a, ga), (_b, gb)]

        return self._push(bp, out)

    # ------------------------------------------------------------- 元素级
    def mul(self, a, b):
        a64, b64 = _as_ref(a), _as_ref(b)

        def bp(go, _a=a64, _b=b64):
            if _bwd_fullgpu():
                _g = _bwd_gpu_mul(go, np.asarray(_a), np.asarray(_b))
                if _g is not None:
                    _ga, _gb = _g
                    return [(_a, _reduce_to(_ga, _a.shape)),
                            (_b, _reduce_to(_gb, _b.shape))]
            ga, gb = nb.mul_backward(_a, _b, go)
            return [(_a, _reduce_to(ga, _a.shape)),
                    (_b, _reduce_to(gb, _b.shape))]

        return self._push(bp, a64 * b64)

    def mul_const(self, a, c):
        a64 = _as_ref(a)
        c = float(c)

        def bp(go, _a=a64, _c=c):
            return [(_a, go * _c)]

        return self._push(bp, a64 * c)

    def add(self, a, b):
        a64, b64 = _as_ref(a), _as_ref(b)

        def bp(go, _a=a64, _b=b64):
            return [(_a, _reduce_to(go, _a.shape)),
                    (_b, _reduce_to(go, _b.shape))]

        return self._push(bp, a64 + b64)

    def add_const(self, a, c):
        a64 = _as_ref(a)
        c = float(c)

        def bp(go, _a=a64):
            return [(_a, go)]

        return self._push(bp, a64 + c)

    def sub(self, a, b):
        a64, b64 = _as_ref(a), _as_ref(b)

        def bp(go, _a=a64, _b=b64):
            return [(_a, _reduce_to(go, _a.shape)),
                    (_b, _reduce_to(-go, _b.shape))]

        return self._push(bp, a64 - b64)

    def div(self, a, b):
        a64, b64 = _as_ref(a), _as_ref(b)

        def bp(go, _a=a64, _b=b64):
            ga, gb = nb.div_backward(_a, _b, go)
            return [(_a, _reduce_to(ga, _a.shape)),
                    (_b, _reduce_to(gb, _b.shape))]

        return self._push(bp, a64 / b64)

    def deweight_norm(self, w_v, w_g):
        """weight_norm 还原为普通权重（可导）：``W = w_v * (w_g / ||w_v||_2)``。

        语义与 ``runtime.models.vits._deweight_norm`` / ``_deweight_any`` 完全
        一致：L2 范数沿除第 0 维外的所有维（逐第 0 维通道），``w_g`` 为逐通道
        标量（``[D0]`` 或 ``[D0,1,...]``）广播。前向用 float32 计算（与推理侧
        还原逐位一致），反向对 ``w_v`` / ``w_g`` 手写梯度：

        - ``dW/dv = (g/n)·I - (g/n³)·v·vᵀ``（逐行，``n = ||v||``）
        - ``dW/dg = v/n``（逐通道求和）

        该 op 是训练侧 weight_norm 参数化的核心：dec.ups / resblocks.convs /
        flow WN 等层以 ``weight_v + weight_g`` 双参数训练，前向先还原 W 再走
        卷积，反向直接把损失梯度传回 ``weight_v`` / ``weight_g``（与原版
        torch ``nn.utils.weight_norm`` 训练语义一致，消除 savee 时的
        逆重参数化转换损耗）。

        Args:
            w_v: [D0, ...] 普通权重（weight_norm 的 direction 向量）。
            w_g: [D0] 或 [D0, 1, ...] 逐通道尺度（weight_norm 的 scale）。

        Returns:
            [D0, ...] 还原后的普通权重（float32，可直接喂 conv/linear）。
        """
        # 反向用 tape 精度（J8：默认 f32——梯度已 f32，f64 参考仅数值测试）
        v64 = _td(w_v)
        g64 = _td(w_g)
        # 前向（float32，与 _deweight_norm/_deweight_any 逐位一致）
        v32 = np.asarray(w_v)
        g32 = np.asarray(w_g)
        dtype = v32.dtype if v32.dtype in (np.float32, np.float64) else np.float32
        v32 = v32.astype(dtype, copy=False)
        g32 = g32.astype(dtype, copy=False)
        if g32.ndim == 1:
            g32 = g32.reshape(-1, *([1] * (v32.ndim - 1)))
        norm32 = np.linalg.norm(v32.reshape(v32.shape[0], -1), ord=2, axis=1)
        norm32 = np.maximum(norm32, 1e-12)
        out = v32 * (g32 / norm32.reshape(-1, *([1] * (v32.ndim - 1))))
        # 反向用 float64（同 tape 其它算子）
        norm = np.linalg.norm(v64.reshape(v64.shape[0], -1), ord=2, axis=1)
        norm = np.maximum(norm, 1e-12)

        def bp(go, _v=v64, _g=g64, _n=norm):
            go64 = _td(go)
            g_sh = _g.reshape(-1, *([1] * (_v.ndim - 1)))
            n_sh = _n.reshape(-1, *([1] * (_v.ndim - 1)))
            v_flat = _v.reshape(_v.shape[0], -1)
            go_flat = go64.reshape(_v.shape[0], -1)
            dot = np.einsum("ok,ok->o", go_flat, v_flat)  # <go, v> 逐行
            # dL/dv = (g/n)*go - (g/n³)*v*<go,v>
            gv = g_sh / n_sh * go64
            gv -= g_sh / (n_sh ** 3) * _v * dot.reshape(-1, *([1] * (_v.ndim - 1)))
            # dL/dg = <go, v>/n（逐通道；归约到 w_g 原形状）
            gg = dot / _n
            gg = gg.reshape(_g.shape) if gg.shape != tuple(_g.shape) else gg
            return [(_v, gv), (_g, gg)]

        return self._push(bp, out)

    def exp(self, x):
        x64 = _as_ref(x)
        out = np.exp(x64)

        def bp(go, _o=out, _x=x64):
            return [(_x, nb.exp_backward(_o, go))]

        return self._push(bp, out)

    def log(self, x):
        x64 = _as_ref(x)

        def bp(go, _x=x64):
            return [(_x, nb.log_backward(_x, go))]

        return self._push(bp, np.log(x64))

    def sqrt(self, x):
        x64 = _as_ref(x)

        def bp(go, _x=x64):
            return [(_x, nb.sqrt_backward(_x, go))]

        return self._push(bp, np.sqrt(x64))

    def pow(self, x, e):
        x64 = _as_ref(x)

        def bp(go, _x=x64, _e=e):
            return [(_x, nb.pow_backward(_x, go, _e))]

        return self._push(bp, x64 ** e)

    def neg(self, x):
        x64 = _as_ref(x)

        def bp(go, _x=x64):
            return [(_x, -go)]

        return self._push(bp, -x64)

    # ------------------------------------------------------------- 形状类
    def transpose(self, x, axes):
        x64 = _as_ref(x)
        out = x64.transpose(axes)

        def bp(go, _x=x64, ax=axes):
            inv = np.argsort(np.array(ax))
            return [(_x, go.transpose(inv))]

        return self._push(bp, out)

    def reshape(self, x, shape):
        x64 = _as_ref(x)
        out = x64.reshape(shape)

        def bp(go, _x=x64, s=shape):
            return [(_x, go.reshape(_x.shape))]

        return self._push(bp, out)

    def slice(self, x, start, stop, axis=-1):
        x64 = _as_ref(x)
        ax = axis if axis >= 0 else axis + x64.ndim
        sl = [slice(None)] * x64.ndim
        sl[ax] = slice(start, stop)
        out = x64[tuple(sl)]

        def bp(go, _x=x64, s=start, e=stop, a=ax):
            return [(_x, nb.slice_backward(go, _x.shape, s, e, a))]

        return self._push(bp, out)

    def concat(self, arrays, axis=1):
        arr64 = [np.asarray(a) for a in arrays]
        sizes = [a.shape[axis] for a in arr64]
        out = np.concatenate(arr64, axis=axis)
        ax = axis if axis >= 0 else axis + out.ndim

        def bp(go, _arrs=tuple(arr64), sz=sizes, a=ax):
            pieces = np.split(go, np.cumsum(sz)[:-1], axis=a)
            return list(zip(_arrs, pieces))

        return self._push(bp, out)

    def flip(self, x, axis=1):
        x64 = _as_ref(x)
        out = np.flip(x64, axis=axis)

        def bp(go, _x=x64, a=axis):
            return [(_x, np.flip(go, axis=a))]

        return self._push(bp, out)

    def pad_const(self, x, pad_l, pad_r, axis=-1):
        x64 = _as_ref(x)
        ax = axis if axis >= 0 else axis + x64.ndim
        shape = [(0, 0)] * x64.ndim
        shape[ax] = (pad_l, pad_r)
        out = np.pad(x64, shape, mode="constant")

        def bp(go, _x=x64, a=ax, l=pad_l, r=pad_r):
            return [(_x, nb.pad_backward(go, l, r, a))]

        return self._push(bp, out)

    def pad_reflect(self, x, pad_l, pad_r, axis=-1):
        """reflect 边界 pad（对齐原版 F.pad(x, (0,n_pad), 'reflect')）。

        reflect 只是翻转复制边界值，梯度为去掉 pad 区的裁剪。
        """
        x64 = _as_ref(x)
        ax = axis if axis >= 0 else axis + x64.ndim
        shape = [(0, 0)] * x64.ndim
        shape[ax] = (pad_l, pad_r)
        out = np.pad(x64, shape, mode="reflect")

        def bp(go, _x=x64, a=ax, l=pad_l, r=pad_r):
            sl = [slice(None)] * go.ndim
            sl[a] = slice(l, go.shape[a] - r)
            gx = go[tuple(sl)] if go.shape[a] > l + r else np.zeros_like(_x)
            return [(_x, gx)]

        return self._push(bp, out)

    def interpolate_linear(self, x, scale_factor=2, axis=-1):
        x64 = _as_ref(x)
        L_in = x64.shape[axis]
        L_out = int(L_in * scale_factor)
        src = np.clip((np.arange(L_out) + 0.5) / float(scale_factor) - 0.5,
                      0.0, L_in - 1)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, L_in - 1)
        frac = src - lo
        move = axis != x64.ndim - 1
        xm = np.moveaxis(x64, axis, -1) if move else x64
        out_m = xm[..., lo] * (1.0 - frac) + xm[..., hi] * frac
        out = np.moveaxis(out_m, -1, axis) if move else out_m

        def bp(go, _x=x64, sf=scale_factor, a=axis):
            return [(_x, nb.interpolate_linear_backward(_x, go, sf, a))]

        return self._push(bp, out)

    # ------------------------------------------------------------- 归约
    def sum(self, x, axis=None, keepdims=False):
        x64 = _as_ref(x)
        out = x64.sum(axis=axis, keepdims=keepdims)
        # 归约到标量时返回 0-d ndarray（而非 numpy 标量）：保证对象身份稳定，
        # 否则后续算子 _as_ref(np.asarray(scalar)) 会新建对象 → 梯度按 id
        # 查找失败 → 梯度链断裂（KL → enc_p/flow 无梯度的根因，2026-10-09）。
        if not isinstance(out, np.ndarray):
            out = np.asarray(out)

        def bp(go, _x=x64, a=axis, k=keepdims):
            if a is None:
                return [(_x, np.broadcast_to(np.asarray(go), _x.shape).copy())]
            tmp = np.asarray(go)
            if not k:
                tmp = np.expand_dims(tmp, axis=a)
            return [(_x, np.broadcast_to(tmp, _x.shape).copy())]

        return self._push(bp, out)

    def mean(self, x, axis=None, keepdims=False):
        x64 = _as_ref(x)

        def bp(go, _x=x64, a=axis, k=keepdims):
            if a is None:
                return [(_x, np.broadcast_to(
                    np.asarray(go) / float(np.prod(_x.shape)),
                    _x.shape).copy())]
            axes = (a,) if isinstance(a, int) else tuple(a)
            cnt = 1
            for ax in axes:
                cnt *= _x.shape[ax]
            tmp = np.asarray(go)
            if not k:
                tmp = np.expand_dims(tmp, axis=a)
            return [(_x, np.broadcast_to(tmp / cnt, _x.shape).copy())]

        return self._push(bp, x64.mean(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------- 反向
    def backward(self, seeds=None):
        """逆序 bp。seeds: {id(array): grad}；无 seeds 时对最后一个 op 用 1。"""
        if seeds:
            for k, v in seeds.items():
                self.grads[k] = self.grads.get(k, 0.0) + _td(v)
        else:
            if self.ops:
                last = self.ops[-1][1]
                self.grads[id(last)] = np.ones_like(np.asarray(last),
                                                    dtype=_td_dtype())
        for i, (bp, out) in enumerate(reversed(self.ops)):
            go = self.grads.get(id(out))
            if go is None:
                continue
            if _BDPROF:
                _t0 = _t.perf_counter()
            try:
                grads = bp(go)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"tape.backward op#{i}（out 形状 {np.asarray(out).shape}）"
                    f"bp 失败: {exc}") from exc
            if _BDPROF:
                try:
                    _shp = str(np.asarray(out).shape)
                except Exception:  # noqa: BLE001
                    _shp = "?"
                _lin = getattr(getattr(bp, "__code__", None), "co_firstlineno", -1)
                _key = f"L{_lin}:{_shp}"
                _dt = _t.perf_counter() - _t0
                if _dt > 0.5:  # TEMP-DBG：慢 bp 定位
                    print(f"[BWD-SLOW] L{_lin} out={_shp} go="
                          f"{type(go).__name__} t={_dt:.3f}s i={i}",
                          file=sys.stderr)
                _BPT[_key] = _BPT.get(_key, 0.0) + _dt
            for ref, g in grads:
                if id(ref) in self.constants:
                    continue
                # T-J3：模块级 BatchTensor（L44）——省逐 op 局部 import
                is_bt = isinstance(g, BatchTensor)
                gshape = g.shape if is_bt else np.shape(g)
                prev = self.grads.get(id(ref))
                if prev is not None and prev.shape != gshape:
                    # 同一数组被不同形状梯度引用：说明该数组是广播常数但未标记
                    raise RuntimeError(
                        f"tape.backward op#{i}（out 形状 {np.asarray(out).shape}）"
                        f"对 ref（形状 {np.shape(ref)}）产生梯度 {gshape}，"
                        f"已有梯度 {prev.shape}；"
                        f"请用 tape.mark_const 标记广播常数")
                if is_bt:
                    if prev is None:
                        # T-H7 链式：GPU 梯度直接存，供下一 bp 消费（gx 链）
                        self.grads[id(ref)] = g
                        continue
                    # 共享 ref 已有梯度：下载合并（断链，罕见）
                    g = _td(g.numpy())
                    if isinstance(prev, BatchTensor):
                        prev = _td(prev.numpy())
                else:
                    g = _td(g)
                    if isinstance(prev, BatchTensor):
                        prev = _td(prev.numpy())
                self.grads[id(ref)] = prev + g if prev is not None else g

        # T-H7 链式：统一提交本 backward 录制的 GPU 批次（一次 submit+wait
        # 替代逐算子同步往返；无录制时空批次跳过）。
        # T6：RVC_TRAIN_BWD_ASYNC=1 → 链式 async 提交，等待推迟到下方
        # J18 批量下载前（bp 遍历 Python 与 GPU 链执行重叠；数值语义不变，
        # 默认 0 = 原同步路径零回归）。
        _bwd_async = os.environ.get("RVC_TRAIN_BWD_ASYNC", "0") == "1"
        _commit_chain_br(async_=_bwd_async)

        # R-T4：异步判别器 bwd 图在 J18 批量下载前统一等待（无在途时
        # no-op；engine 队列顺序执行保证链内 RAW 安全）。
        if os.environ.get("RVC_TRAIN_GRAPH_ASYNC", "0") == "1":
            _graph_runner().wait_all()
        if _bwd_async:
            # T6：J18 直接读 buffer，不经过 numpy() 的自动 wait——
            # 链式 async 提交必须在批量下载前显式等待（幂等 no-op）。
            global _CHAIN_BR
            if _CHAIN_BR is not None and not _CHAIN_BR._released:
                _CHAIN_BR.wait()

        # J18：批量下载剩余 BatchTensor 梯度（grad_of 消费 numpy；逐次
        # numpy() = 每次 readback 各自 submit+fence，数百次串行——批量
        # 一次 fence 处理全部）。
        _bt_grads = [(id(ref), g) for ref, g in self.grads.items()
                     if isinstance(g, BatchTensor)]
        if _bt_grads:
            _ctx0 = _bt_grads[0][1]._runner._ctx
            _items = [(int(g._buf), g.shape) for _, g in _bt_grads]
            try:
                _arrs = _ctx0._batch_download(_items)
            except Exception:  # noqa: BLE001 回退逐次 numpy()
                _arrs = [np.asarray(g) for _, g in _bt_grads]
            for (ref_id, _), a in zip(_bt_grads, _arrs):
                self.grads[ref_id] = a
        if os.environ.get("RVC_TRAIN_STEP_PROFILE") and _DBG_DISC_BWD_MS:  # TEMP-DBG
            import time as _tm2  # noqa: PLC0415
            _n = len(_DBG_DISC_BWD_MS)
            _s = sum(_DBG_DISC_BWD_MS)
            print(f"[DBG] disc_bwd 图: {_n}次 总{_s:.0f}ms "
                  f"均{_s/_n:.0f}ms/次 (backward内判别器bp图耗时)",
                  file=sys.stderr)
            _DBG_DISC_BWD_MS.clear()

    def release_brs(self):
        """J15：释放本 tape 挂载的判别器 BR_FWD 链 BatchRunner（必须在
        grad_of 全部下载之后调用——train_step 负责时序）。"""
        brs, self._brs = self._brs, []
        for br in brs:
            try:
                br.release()
            except Exception:  # noqa: BLE001
                pass

    def grad_of(self, arr):
        g = self.grads.get(id(arr))
        if g is None:
            return None
        if isinstance(g, BatchTensor):
            # T-H7 链式：参数梯度留 GPU，取用时下载（backward 已 commit）
            g = _td(g.numpy())
        return np.asarray(g, dtype=np.asarray(arr).dtype)


# ---------------------------------------------------------------------------
# 前向算子（内联 numpy 实现，避免 backend/Vulkan 分派对 tape 的干扰；
# 数值与 runtime.nn 一致）
# ---------------------------------------------------------------------------
def _conv1d_np(x, w, b=None, stride=1, padding=0, dilation=1):
    import os  # noqa: PLC0415   T3 matmul 开关
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(padding, (tuple, list)):
        pad_l, pad_r = int(padding[0]), int(padding[1])
    else:
        pad_l = pad_r = int(padding)
    stride, dilation = int(stride), int(dilation)
    B, C, T = x.shape
    O, _, K = w.shape
    K_dil = (K - 1) * dilation + 1
    oL = (T + pad_l + pad_r - K_dil) // stride + 1
    if oL <= 0:
        out = np.zeros((B, O, 0), dtype=x.dtype)
        return out
    if dilation == 1:
        w_dil = w
    else:
        w_dil = np.zeros((O, C, K_dil), dtype=w.dtype)
        w_dil[:, :, ::dilation] = w
    x_pad = np.zeros((B, C, T + pad_l + pad_r), dtype=x.dtype)
    x_pad[:, :, pad_l:pad_l + T] = x
    win = np.lib.stride_tricks.sliding_window_view(x_pad, K_dil, axis=-1)
    win = win[:, :, ::stride, :]
    if (os.environ.get("RVC_TRAIN_FWD_MATMUL", "1") != "0"
            and x.flags.c_contiguous and w_dil.flags.c_contiguous):
        # T3: einsum → matmul（im2col_b 展平 [b*t, ck] @ w[o, ck]^T）——
        # 消除 einsum_path/parse 每调用开销（profile: einsum 家族 12.5s/3 步）。
        # 数值 f64 等价（BLAS gemm 与 einsum 路径 ~1e-14 浮点序差异，同 T1 容差）；
        # RVC_TRAIN_FWD_MATMUL=0 回退原 einsum。
        import os  # noqa: PLC0415
        win_f = win.transpose(0, 2, 1, 3).reshape(B * oL, C * K_dil)
        out = (np.matmul(win_f, w_dil.reshape(O, C * K_dil).T)
               .reshape(B, oL, O).transpose(0, 2, 1))
    else:
        out = np.einsum("bctk,ock->bot", win, w_dil, optimize=True)
    if b is not None:
        out += np.asarray(b).reshape(1, -1, 1)
    return out


def _conv2d_np(x, w, b=None, stride=1, padding=0, dilation=1):
    import os  # noqa: PLC0415   T3 matmul 开关
    x = np.asarray(x)
    w = np.asarray(w)
    if isinstance(stride, (tuple, list)):
        sh, sw = int(stride[0]), int(stride[1])
    else:
        sh = sw = int(stride)
    if isinstance(padding, (tuple, list)):
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if isinstance(dilation, (tuple, list)):
        dh, dw = int(dilation[0]), int(dilation[1])
    else:
        dh = dw = int(dilation)
    B, C, H, W = x.shape
    O, _, KH, KW = w.shape
    KH_dil = (KH - 1) * dh + 1
    KW_dil = (KW - 1) * dw + 1
    if dh == 1 and dw == 1:
        w_dil = w
    else:
        w_dil = np.zeros((O, C, KH_dil, KW_dil), dtype=w.dtype)
        w_dil[:, :, ::dh, ::dw] = w
    x_pad = np.zeros((B, C, H + 2 * ph, W + 2 * pw), dtype=x.dtype)
    x_pad[:, :, ph:ph + H, pw:pw + W] = x
    Hp, Wp = x_pad.shape[2], x_pad.shape[3]
    OH = (Hp - KH_dil) // sh + 1
    OW = (Wp - KW_dil) // sw + 1
    win = np.lib.stride_tricks.sliding_window_view(
        x_pad, (KH_dil, KW_dil), axis=(-2, -1))
    win = win[:, :, ::sh, ::sw, :, :].transpose(0, 2, 3, 1, 4, 5)
    if (os.environ.get("RVC_TRAIN_FWD_MATMUL", "1") != "0"
            and w_dil.flags.c_contiguous):
        # T3: einsum → matmul（同 conv1d fwd 优化，消 einsum_path/parse）
        import os  # noqa: PLC0415
        win_f = np.ascontiguousarray(
            win.reshape(B * OH * OW, C * KH_dil * KW_dil))
        # 输出保持 [B,OH,OW,O]（+bias 前布局，公共 L765 统一转置为 [B,O,OH,OW]）
        out = np.matmul(win_f, w_dil.reshape(O, C * KH_dil * KW_dil).T) \
            .reshape(B, OH, OW, O)
    else:
        out = np.einsum("bijcxy,ocxy->bijo", win, w_dil, optimize=True)
    out = out.transpose(0, 3, 1, 2)
    if b is not None:
        out += np.asarray(b).reshape(1, -1, 1, 1)
    return out


def _conv_transpose1d_np(x, w, b=None, stride=1, padding=0, output_padding=0,
                         dilation=1):
    import os  # noqa: PLC0415   T5 matmul 开关（与 RVC_TRAIN_FWD_MATMUL 共用）
    x = np.asarray(x)
    w = np.asarray(w)
    stride, padding, output_padding, dilation = (
        int(stride), int(padding), int(output_padding), int(dilation))
    B, C_in, T = x.shape
    C_out = w.shape[1]
    K = w.shape[2]
    oL = (T - 1) * stride - 2 * padding + dilation * (K - 1) \
        + output_padding + 1
    out = np.zeros((B, C_out, oL), dtype=np.result_type(x, w))
    for k in range(K):
        pos = np.arange(T) * stride - padding + k * dilation
        valid = (pos >= 0) & (pos < oL)
        if not valid.any():
            continue
        tv = np.nonzero(valid)[0]
        if os.environ.get("RVC_TRAIN_FWD_MATMUL", "1") != "0":
            # T5: einsum → matmul（收缩 c：x^T[B,T',C] @ w[C,O] → [B,T',O]）
            contrib = np.matmul(x[:, :, tv].transpose(0, 2, 1),
                                w[:, :, k]).transpose(0, 2, 1)
        else:
            contrib = np.einsum("bct,co->bot", x[:, :, tv],
                                w[:, :, k], optimize=True)
        out[:, :, pos[tv]] += contrib
    if b is not None:
        out += np.asarray(b).reshape(1, -1, 1)
    return out


def _gelu_np(x):
    x = np.asarray(x)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _embedding_np(ids, table):
    ids = np.asarray(ids)
    V = table.shape[0]
    ids2 = np.where(ids < 0, ids + V, ids).astype(np.int64)
    return np.take(table, ids2, axis=0)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _reduce_to(g, shape):
    """把广播形状的梯度 g 归约回目标 shape（多余维求和）。

    T-J3：目标形状与 g 相同时直接透传（免下载/归约/f64 转换；g 为
    BatchTensor 时保持 GPU 链，由消费方惰性下载）。
    """
    shape = tuple(int(s) for s in shape)
    if isinstance(g, BatchTensor) and tuple(g.shape) == shape:
        return g
    g = _td(g)
    shape = tuple(int(s) for s in shape)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for d in range(len(shape)):
        if g.shape[d] != shape[d]:
            if shape[d] == 1 and g.shape[d] > 1:
                g = g.sum(axis=d, keepdims=True)
            elif g.shape[d] == 1 and shape[d] > 1:
                pass  # 目标更大：保持 size-1（后续会再被广播）
            else:
                raise ValueError(
                    f"广播归约失败: g.shape={g.shape} target={shape} "
                    f"(d={d}: {g.shape[d]} vs {shape[d]})")
    return g


def sequence_mask_np(length, max_length=None):
    """对齐 commons.sequence_mask [1, 1, T] float32。

    F1（batch>1）：length 支持标量（B=1 → [1,1,T]）或 array（B>1 →
    [B,1,T]，每样本各自长度；批量时样本已对齐，各 length 相同）。
    """
    lens = np.asarray(length, dtype=np.int64).reshape(-1)
    if max_length is None:
        max_length = int(lens.max()) if lens.size else 0
    max_length = int(max_length)
    b = lens.size
    mask = np.zeros((b, 1, max_length), dtype=np.float32)
    if b == 1 and np.asarray(length).ndim == 0:
        # 标量路径保持原 [1,1,T] 语义
        if int(lens[0]) > 0:
            mask[..., : min(int(lens[0]), max_length)] = 1.0
    else:
        for i in range(b):
            if int(lens[i]) > 0:
                mask[i, :, : min(int(lens[i]), max_length)] = 1.0
    return mask


def rand_slice_segments_np(x, x_lengths, segment_size, rng=None):
    """对齐 commons.rand_slice_segments。返回 (slice, ids_str)。

    F1（batch>1）：x_lengths 支持标量（B=1）或 array（B>1，每样本
    max_start）——ids 每样本独立。
    """
    if rng is None:
        rng = np.random
    b, c, t = x.shape
    lens = np.asarray(x_lengths, dtype=np.int64).reshape(-1)
    max_start = lens - int(segment_size) + 1
    max_start = np.maximum(1, max_start)
    ids = rng.randint(0, int(max_start.max()), size=(b,)).astype(np.int64)
    ids = np.minimum(ids, max_start - 1)  # 每样本上界
    out = np.stack([x[i, :, ids[i]: ids[i] + int(segment_size)]
                    for i in range(b)], axis=0)
    return out, ids


def kl_loss_np(z_p, logs_q, m_p, logs_p, z_mask):
    """对齐 train/losses.kl_loss（仓库实际公式）。

    kl = logs_p - logs_q - 0.5 + 0.5*((z_p - m_p)²) * exp(-2*logs_p)，
    加权 z_mask 后除以 sum(z_mask)。

    注意：这是**重参数化 z_p 版本**，与 VITS 论文标准 KL
    （commons.kl_divergence，基于 m_p - m_q）不同；本仓库训练代码实际采用
    前者，此处以仓库实际实现为准（报告核对结论）。
    """
    z_p = np.asarray(z_p, dtype=np.float64)
    logs_q = np.asarray(logs_q, dtype=np.float64)
    m_p = np.asarray(m_p, dtype=np.float64)
    logs_p = np.asarray(logs_p, dtype=np.float64)
    z_mask = np.asarray(z_mask, dtype=np.float64)
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * np.exp(-2.0 * logs_p)
    kl = np.sum(kl * z_mask)
    return float(kl / np.sum(z_mask))


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------

# 训练侧做 weight_norm 参数化的层（G 侧，含 enc_q.enc 的 WN——推理端无
# enc_q，故 process_ckpt._WN_LAYER_RE 不含它，但训练必须覆盖）。这些层的
# 普通权重 ``X.weight`` 在加载时拆成 ``X.weight_v`` + ``X.weight_g``。
_WN_LAYER_RE = re.compile(
    r"^(?:"
    r"enc_q\.enc\.(?:cond_layer|in_layers\.\d+|res_skip_layers\.\d+)"
    r"|flow\.flows\.\d+\.enc\.(?:cond_layer|in_layers\.\d+|res_skip_layers\.\d+)"
    r"|dec\.ups\.\d+"
    r"|dec\.resblocks\.\d+\.convs[12]\.\d+"
    r")\.weight$"
)


def _deweight_any(w_v, w_g):
    """通用 weight_norm 还原（3D/4D 权重都支持）。

    T-J1：per-step 缓存（key=(id(w_v), id(w_g))）——训练步内权重不变，重复
    调用直接命中；_release_chain_br 每步末清空（权重被 optimizer 就地更新
    后 id 不变，必须按步失效）。
    T-J7 修复：缓存 value 同时持有 (w_v, w_g) 对象引用 + 命中时身份校验
    （is）——仅用 id() 时，若旧对象被 GC 释放后地址被另一权重数组复用，
    缓存会命中错误形状/数值（实测偶发 conv1d w 4D 崩溃）。
    """
    if _DWN_CACHE_ON:
        k = (id(w_v), id(w_g))
        hit = _DWN_CACHE.get(k)
        if hit is not None and hit[1] is w_v and hit[2] is w_g:
            return hit[0]
    _wv0, _wg0 = w_v, w_g  # 原始引用（value 持有防 id 复用）
    # T-J7：deweight 计算 f32 化（默认开）——dec 训练权重 f64，但 deweight
    # 结果最终进 dec GPU fwd（引擎 buffer 恒 f32），f64→f32 只差 ~1e-6 相对
    # 精度（远小于 GPU f32 累加误差），norm+逐元素速度 ~2×。
    import os as _os  # noqa: PLC0415
    if _os.environ.get("RVC_TRAIN_DW_F32", "1") == "1":
        w_v = np.asarray(w_v, dtype=np.float32)
        w_g = np.asarray(w_g, dtype=np.float32)
    else:
        w_v = np.asarray(w_v)
        w_g = np.asarray(w_g)
    shape = (w_g.shape[0],) + (1,) * (w_v.ndim - 1)
    w_g = w_g.reshape(shape)
    norm = np.linalg.norm(w_v.reshape(w_v.shape[0], -1), ord=2, axis=1)
    norm = np.maximum(norm, 1e-12)
    res = w_v * (w_g / norm.reshape(-1, *([1] * (w_v.ndim - 1))))
    if _DWN_CACHE_ON:
        _DWN_CACHE[k] = (res, _wv0, _wg0)
    return res


def _deweight_dict(w_in: dict) -> dict:
    """把 ``X.weight_v + X.weight_g`` 还原为 ``X.weight``（返回可写副本）。"""
    out = {}
    for k, v in w_in.items():
        if k.endswith(".weight_v"):
            base = k[:-9]  # 去掉 ".weight_v"
            g = w_in.get(base + ".weight_g")
            if g is None:
                raise KeyError(f"缺少 weight_g: {base + '.weight_g'}")
            arr = _deweight_any(np.asarray(v), np.asarray(g))
            out[base + ".weight"] = np.array(arr, copy=True, order="C")
        elif k.endswith(".weight_g"):
            continue
        else:
            arr = np.asarray(v)
            out[k] = np.array(arr, copy=True, order="C")
    return out


def _channel_l2_norm(w: np.ndarray) -> np.ndarray:
    """逐第 0 维通道的 L2 范数（与 vits._deweight_norm 的 norm 一致）。"""
    w = np.asarray(w, dtype=np.float64)
    return np.linalg.norm(w.reshape(w.shape[0], -1), ord=2, axis=1)


def _reparam_weight_norm(w_in: dict) -> dict:
    """普通权重 dict → weight_norm 参数化 dict（``weight_v`` + ``weight_g``）。

    - 已是 ``X.weight_v`` / ``X.weight_g`` 的键原样保留（真实 RVC checkpoint
      的 weight_norm 层直接以 g/v 存储，训练直接优化它们，不还原为 W）；
    - 属于 ``_WN_LAYER_RE`` 的普通 ``X.weight``（本项目 T46 及以前的训练产物
      以普通权重保存）按 ``weight_v = W``、``weight_g = ||W||``（逐第 0 维
      通道 L2，形状 ``[D0,1,1]``，对齐 process_ckpt.reparam_weight_norm 与
      RVC 官方 checkpoint 布局）拆成双参数；
    - 其余键原样保留。返回可写副本（与旧 _deweight_dict 的复制语义一致）。
    """
    out = {}
    for key, val in w_in.items():
        arr = np.asarray(val)
        if key.endswith(".weight") and _WN_LAYER_RE.match(key):
            norm = _channel_l2_norm(arr)
            g = norm.astype(arr.dtype if arr.dtype in (np.float32, np.float64)
                            else np.float32)
            out[key + "_v"] = np.array(arr, copy=True, order="C")
            out[key + "_g"] = np.array(g.reshape(g.shape[0], 1, 1),
                                       copy=True, order="C")
        elif key.endswith((".weight_v", ".weight_g")):
            out[key] = np.array(arr, copy=True, order="C")
        else:
            out[key] = np.array(arr, copy=True, order="C")
    return out


def load_g_weights(checkpoint_dict: dict) -> dict:
    """从 cpt 加载 G 权重（weight_norm 参数化，对齐原版训练语义）。

    生成器（enc_p / enc_q / flow / dec / emb_g）以 ``weight_v + weight_g``
    双参数形式加载（dec.ups、dec.resblocks.convs、flow WN、enc_q.enc 等），
    前向由 ``AutogradTape.deweight_norm`` 还原 W，反向对 g/v 传播梯度——
    与原版 torch ``nn.utils.weight_norm`` 训练一致，训练产物本身就是推理
    格式（weight_v/weight_g），savee 不再需要逆重参数化。
    """
    w = checkpoint_dict.get("model", checkpoint_dict)
    return _reparam_weight_norm(w)


def load_d_weights(checkpoint_dict: dict) -> dict:
    """从 D checkpoint 加载判别器权重（还原为普通权重）。

    判别器保持普通权重训练（T46 现状；P2 只对 G 侧做 weight_norm 参数化）。
    """
    w = checkpoint_dict.get("model", checkpoint_dict)
    return _deweight_dict(w)


# ---------------------------------------------------------------------------
# WN（对齐 modules.WN）：enc_q.enc（16 层）/ flow.flows.<i>.enc（3 层）
# ---------------------------------------------------------------------------
def _load_wn_pair(w: dict, base: str):
    """weight_norm 层权重加载：返回 ``(权重数组, w_g 或 None)``。

    ``base + ".weight_v"`` 存在（weight_norm 参数化 checkpoint）时返回
    (w_v, w_g)；否则返回 (普通 ``.weight``, None)（普通权重 checkpoint）。
    训练侧 ``tape.deweight_norm(w_v, w_g)`` 在 forward 还原 W；普通权重
    直接使用（T46 历史产物兼容，行为与旧版完全一致）。
    """
    v = w.get(base + ".weight_v")
    if v is None:
        return np.asarray(w[base + ".weight"]), None
    g = w.get(base + ".weight_g")
    if g is None:
        raise KeyError(f"缺少 weight_g: {base + '.weight_g'}")
    return np.asarray(v), np.asarray(g)


def _tape_conv(tape, x, w, b, w_g=None, **kw):
    """带 weight_norm 还原的 conv1d：``w_g`` 非 None 时先 ``deweight_norm``。"""
    if w_g is not None:
        w = tape.deweight_norm(w, w_g)
    return tape.conv1d(x, w, b, **kw)


class WNTrain:
    def __init__(self, w: dict, base: str, hidden: int, kernel: int,
                 n_layers: int, gin_channels: int):
        self.w = w
        self.base = base
        self.hidden = hidden
        self.n_layers = n_layers
        if gin_channels != 0:
            self.cond_w, self.cond_g = _load_wn_pair(w, f"{base}.cond_layer")
            self.cond_b = np.asarray(w[f"{base}.cond_layer.bias"])
        else:
            self.cond_w = None
            self.cond_g = None
        self.in_w, self.in_g, self.in_b = [], [], []
        self.rs_w, self.rs_g, self.rs_b = [], [], []
        for i in range(n_layers):
            iw, ig = _load_wn_pair(w, f"{base}.in_layers.{i}")
            self.in_w.append(iw)
            self.in_g.append(ig)
            self.in_b.append(np.asarray(w[f"{base}.in_layers.{i}.bias"]))
            rw, rg = _load_wn_pair(w, f"{base}.res_skip_layers.{i}")
            self.rs_w.append(rw)
            self.rs_g.append(rg)
            self.rs_b.append(np.asarray(
                w[f"{base}.res_skip_layers.{i}.bias"]))

    def forward(self, tape, x, x_mask, g=None):
        hidden = self.hidden
        g_cond = (_tape_conv(tape, g, self.cond_w, self.cond_b, self.cond_g)
                  if g is not None else None)
        out = None
        cur = x
        for i in range(self.n_layers):
            x_in = _tape_conv(tape, cur, self.in_w[i], self.in_b[i],
                              self.in_g[i], dilation=1, padding=2)
            if g_cond is not None:
                off = i * 2 * hidden
                g_l = tape.slice(g_cond, off, off + 2 * hidden, axis=1)
                acts_in = tape.add(x_in, g_l)
            else:
                acts_in = x_in
            t = tape.tanh(tape.slice(acts_in, 0, hidden, axis=1))
            s = tape.sigmoid(tape.slice(acts_in, hidden, 2 * hidden, axis=1))
            acts = tape.mul(t, s)
            res_skip = _tape_conv(tape, acts, self.rs_w[i], self.rs_b[i],
                                  self.rs_g[i])
            if i < self.n_layers - 1:
                res = tape.slice(res_skip, 0, hidden, axis=1)
                cur = tape.mul(tape.add(cur, res), x_mask)
                skip = tape.slice(res_skip, hidden, 2 * hidden, axis=1)
                out = skip if out is None else tape.add(out, skip)
            else:
                out = res_skip if out is None else tape.add(out, res_skip)
        return tape.mul(out, x_mask)


# ---------------------------------------------------------------------------
# T4-1：WN 链 GPU 化衔接（enc_q / flow 共用；br 与 graph 同构）
# ---------------------------------------------------------------------------
def _wn_chain_fwd(tape, wn, conv, cur, x_mask, g_cond):
    """WN 前向链（br/graph 共用衔接）：``conv(i, x, kind)`` 执行第 i 层
    in/rs conv（内部已 tape.record_*）并返回 BatchTensor；衔接
    （slice/tanh/sigmoid/add/mul）为 tape op 保 bp 链。语义 ≡
    ``WNTrain.forward`` numpy 版（K/pad 由权重 shape 推导）。返回
    tape out（≡ ``out * x_mask``）。
    """
    hidden = wn.hidden
    out = None
    for i in range(wn.n_layers):
        x_in = conv(i, cur, "in")
        if g_cond is not None:
            off = i * 2 * hidden
            g_l = tape.slice(g_cond, off, off + 2 * hidden, axis=1)
            acts_in = tape.add(x_in, g_l)
        else:
            acts_in = x_in
        t = tape.tanh(tape.slice(acts_in, 0, hidden, axis=1))
        s = tape.sigmoid(tape.slice(acts_in, hidden, 2 * hidden, axis=1))
        acts = tape.mul(t, s)
        res_skip = conv(i, acts, "rs")
        if i < wn.n_layers - 1:
            res = tape.slice(res_skip, 0, hidden, axis=1)
            cur = tape.mul(tape.add(cur, res), x_mask)
            skip = tape.slice(res_skip, hidden, 2 * hidden, axis=1)
            out = skip if out is None else tape.add(out, skip)
        else:
            out = res_skip if out is None else tape.add(out, res_skip)
    return tape.mul(out, x_mask)


def _make_wn_conv_br(tape, br, wn):
    """br 版 WN conv 执行器：deweight 还原 → ``br.conv1d`` → record。
    返回闭包 ``conv(i, cur, kind) -> BatchTensor``（已 record）。"""
    def conv(i, cur, kind):
        if kind == "in":
            wv, wg, b = wn.in_w[i], wn.in_g[i], wn.in_b[i]
        else:
            wv, wg, b = wn.rs_w[i], wn.rs_g[i], wn.rs_b[i]
        kw = int(wv.shape[2])
        pad = (kw - 1) // 2
        w = tape.deweight_norm(wv, wg) if wg is not None else wv
        out = br.conv1d(cur, w, b, stride=1, padding=pad)
        tape.record_conv1d(out, cur, w, b, padding=pad,
                           w_v=wv, w_g=wg)
        return out
    return conv


def _make_wn_conv_graph(tape, gr, eg, wn, base_in, base_rs):
    """graph 版 WN conv 执行器：deweight 还原 → set_input（权重 + cur/acts
    槽）→ run 段图 → bt 包装 → record。``base_in/base_rs`` 为 ghs/outs 键
    前缀（如 ``"in.{}"`` / ``"l0.in.{}"``）。返回闭包同上。"""
    from runtime.vulkan_ops import BatchTensor  # noqa: PLC0415

    def conv(i, cur, kind):
        if kind == "in":
            base = base_in
            wv, wg, b = wn.in_w[i], wn.in_g[i], wn.in_b[i]
            xslot = "cur"
        else:
            base = base_rs
            wv, wg, b = wn.rs_w[i], wn.rs_g[i], wn.rs_b[i]
            xslot = "acts"
        key = base.format(i)
        kw = int(wv.shape[2])
        pad = (kw - 1) // 2
        w = tape.deweight_norm(wv, wg) if wg is not None else wv
        gr.set_input(np.asarray(w, np.float32), eg.wslots[key + ".w"])
        gr.set_input(np.asarray(b, np.float32), eg.wslots[key + ".b"])
        gr.set_input(cur, eg.slots[xslot])
        gr.run(eg.ghs[key])
        out = BatchTensor(_chain_br(), *eg.outs[key])
        tape.record_conv1d(out, cur, w, b, padding=pad,
                           w_v=wv, w_g=wg)
        return out
    return conv


# ---------------------------------------------------------------------------
# PosteriorEncoder（enc_q）
# ---------------------------------------------------------------------------
class PosteriorEncoder:
    def __init__(self, w: dict, cfg: VitsConfig):
        self.w = w
        self.hidden = cfg.hidden
        self.pre_w = np.asarray(w["enc_q.pre.weight"])
        self.pre_b = np.asarray(w["enc_q.pre.bias"])
        self.proj_w = np.asarray(w["enc_q.proj.weight"])
        self.proj_b = np.asarray(w["enc_q.proj.bias"])
        self.wn = WNTrain(w, "enc_q.enc", cfg.hidden, 5, 16,
                          cfg.gin_channels)

    def forward(self, tape, x, x_mask, g, randn=None):
        """T4-1 门控：环境变量分派 GPU 路径。

        RVC_TRAIN_GRAPH_ENCQ=1 → _forward_graph（EncGraph 分段图化）；
        RVC_TRAIN_GRAPH_ENCQ=0 且 RVC_TRAIN_BR_FWD_ENCQ=1 → _forward_br
        （链式 BatchRunner GPU 基准，同 kernel → 门禁位级一致）；
        否则 numpy 基准（_forward_np）。返回 (z, m, logs)。
        """
        import os as _os_e  # noqa: PLC0415
        _ge = _os_e.environ.get("RVC_TRAIN_GRAPH_ENCQ", "0")
        if _ge == "1":
            return self._forward_graph(tape, x, x_mask, g, randn)
        if _ge == "0" and _os_e.environ.get("RVC_TRAIN_BR_FWD_ENCQ", "0") == "1":
            return self._forward_br(tape, x, x_mask, g, randn)
        return self._forward_np(tape, x, x_mask, g, randn)

    def _forward_np(self, tape, x, x_mask, g, randn):
        """numpy 基准：现有 forward 主体（x: [B, spec_ch, F]）。"""
        h = tape.mul(tape.conv1d(x, self.pre_w, self.pre_b), x_mask)
        h = self.wn.forward(tape, h, x_mask, g)
        stats = tape.mul(tape.conv1d(h, self.proj_w, self.proj_b), x_mask)
        m = tape.slice(stats, 0, self.hidden, axis=1)
        logs = tape.slice(stats, self.hidden, 2 * self.hidden, axis=1)
        if randn is None:
            randn = np.random.randn(*m.shape).astype(np.float32)
        tape.mark_const(randn)
        z = tape.mul(tape.add(m, tape.mul(tape.exp(logs), randn)), x_mask)
        return z, m, logs

    def _forward_br(self, tape, x, x_mask, g, randn=None):
        """T4-1：enc_q 前向 GPU 链式基准（chain BatchRunner 分段录制）。

        与 _forward_graph 完全同构（同 record 序列、同 tape-op 衔接）：GPU
        段在 _chain_br 上执行（段间 br.commit()），tape.record_* 纯记 bp
        （out/x 保持 BatchTensor 引用）。cond/层权重先 deweight_norm 还原
        （梯度经 deweight bp 回 w_v/w_g）。返回 (z, m, logs)。
        """
        br = _chain_br()
        hid = self.hidden
        B, _C, F = x.shape
        xmask32 = np.asarray(x_mask, np.float32)
        xmask_full = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, hid, F)).astype(np.float32))
        xmask_full2 = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, 2 * hid, F)).astype(np.float32))
        k_pre = int(self.pre_w.shape[2])
        pad_pre = (k_pre - 1) // 2
        k_proj = int(self.proj_w.shape[2])
        pad_proj = (k_proj - 1) // 2
        # -- pre：conv + 输出侧 mask --
        h0 = br.conv1d(np.asarray(x, np.float32),
                       np.asarray(self.pre_w, np.float32),
                       np.asarray(self.pre_b, np.float32),
                       stride=1, padding=pad_pre)
        tape.record_conv1d(h0, x, self.pre_w, self.pre_b, padding=pad_pre)
        h = br.mul_inplace(h0, xmask_full)
        tape.record_mul(h, h0, xmask_full)
        # -- cond + WN 链 --
        if g is not None:
            cw = (tape.deweight_norm(self.wn.cond_w, self.wn.cond_g)
                  if self.wn.cond_g is not None else self.wn.cond_w)
            g_cond = br.conv1d(np.asarray(g, np.float32),
                               np.asarray(cw, np.float32),
                               np.asarray(self.wn.cond_b, np.float32),
                               stride=1, padding=0)
            tape.record_conv1d(g_cond, g, cw, self.wn.cond_b, padding=0,
                               w_v=self.wn.cond_w, w_g=self.wn.cond_g)
        else:
            g_cond = None
        br.commit()
        h_out = _wn_chain_fwd(tape, self.wn,
                              _make_wn_conv_br(tape, br, self.wn),
                              h, x_mask, g_cond)
        # -- proj：conv + 输出侧 mask --
        s0 = br.conv1d(np.asarray(h_out, np.float32),
                       np.asarray(self.proj_w, np.float32),
                       np.asarray(self.proj_b, np.float32),
                       stride=1, padding=pad_proj)
        tape.record_conv1d(s0, h_out, self.proj_w, self.proj_b,
                           padding=pad_proj)
        stats = br.mul_inplace(s0, xmask_full2)
        tape.record_mul(stats, s0, xmask_full2)
        br.commit()
        m = tape.slice(stats, 0, hid, axis=1)
        logs = tape.slice(stats, hid, 2 * hid, axis=1)
        if randn is None:
            randn = np.random.randn(*m.shape).astype(np.float32)
        tape.mark_const(randn)
        z = tape.mul(tape.add(m, tape.mul(tape.exp(logs), randn)), x_mask)
        return z, m, logs

    def _forward_graph(self, tape, x, x_mask, g, randn=None):
        """T4-1：enc_q 前向图化（EncGraph 分段图 + Python 衔接）。

        段 = gr.run(eg.ghs[name])；record out 用 bt(name) 包装（BatchTensor
        挂 _chain_br；同一 buffer 的多个 out_entry 各建独立对象——J19：就地
        mul 覆写后 out/x 引用分离、梯度 id 分离）。衔接与 _forward_br 完全
        同构。权重槽每步 set_input 覆写（WN 权重先 deweight_norm 还原，梯度
        经 deweight bp 回 w_v/w_g）。返回 (z, m, logs)。
        """
        from runtime.vulkan_ops import BatchTensor  # noqa: PLC0415
        gr = _graph_runner()
        chain = _chain_br()
        hid = self.hidden
        B, _C, F = x.shape
        eg = gr.encq_graph(("encq", B, F),
                           lambda: gr._build_encq(self, B, F))

        def bt(name):
            bid, shape = eg.outs[name]
            return BatchTensor(chain, bid, shape)

        xmask32 = np.asarray(x_mask, np.float32)
        xmask_full = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, hid, F)).astype(np.float32))
        xmask_full2 = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, 2 * hid, F)).astype(np.float32))
        k_pre = int(self.pre_w.shape[2])
        pad_pre = (k_pre - 1) // 2
        k_proj = int(self.proj_w.shape[2])
        pad_proj = (k_proj - 1) // 2
        # -- 权重/常量槽每步覆写 --
        gr.set_input(np.asarray(self.pre_w, np.float32), eg.wslots["pre_w"])
        gr.set_input(np.asarray(self.pre_b, np.float32), eg.wslots["pre_b"])
        gr.set_input(np.asarray(self.proj_w, np.float32), eg.wslots["proj_w"])
        gr.set_input(np.asarray(self.proj_b, np.float32), eg.wslots["proj_b"])
        gr.set_input(xmask_full2, eg.slots["xmask"])
        if g is not None:
            cw = (tape.deweight_norm(self.wn.cond_w, self.wn.cond_g)
                  if self.wn.cond_g is not None else self.wn.cond_w)
            gr.set_input(np.asarray(cw, np.float32), eg.wslots["cond_w"])
            gr.set_input(np.asarray(self.wn.cond_b, np.float32),
                         eg.wslots["cond_b"])
        # -- pre --
        gr.set_input(x, eg.slots["x"])
        gr.run(eg.ghs["pre"])
        h0_bt = bt("h0")
        h_bt = bt("h")
        tape.record_conv1d(h0_bt, x, self.pre_w, self.pre_b, padding=pad_pre)
        tape.record_mul(h_bt, h0_bt, xmask_full)
        # -- cond + WN 链 --
        if g is not None:
            gr.set_input(g, eg.slots["g"])
            gr.run(eg.ghs["cond"])
            g_cond = bt("g_cond")
            tape.record_conv1d(g_cond, g, cw, self.wn.cond_b, padding=0,
                               w_v=self.wn.cond_w, w_g=self.wn.cond_g)
        else:
            g_cond = None
        h_out = _wn_chain_fwd(tape, self.wn,
                              _make_wn_conv_graph(tape, gr, eg, self.wn,
                                                  "in.{}", "rs.{}"),
                              h_bt, x_mask, g_cond)
        # -- proj --
        gr.set_input(h_out, eg.slots["h"])
        gr.run(eg.ghs["proj"])
        stats0_bt = bt("stats0")
        stats_bt = bt("stats")
        tape.record_conv1d(stats0_bt, h_out, self.proj_w, self.proj_b,
                           padding=pad_proj)
        tape.record_mul(stats_bt, stats0_bt, xmask_full2)
        m = tape.slice(stats_bt, 0, hid, axis=1)
        logs = tape.slice(stats_bt, hid, 2 * hid, axis=1)
        if randn is None:
            randn = np.random.randn(*m.shape).astype(np.float32)
        tape.mark_const(randn)
        z = tape.mul(tape.add(m, tape.mul(tape.exp(logs), randn)), x_mask)
        return z, m, logs


# ---------------------------------------------------------------------------
# TextEncoder（enc_p）
# ---------------------------------------------------------------------------
def _rel_to_abs(tape, x, length):
    b, h, l, _ = x.shape
    xp = tape.pad_const(x, 0, 1, axis=-1)          # [b,h,l,2l]
    xf = tape.reshape(xp, (b, h, l * (2 * l)))
    xf = tape.pad_const(xf, 0, l - 1, axis=-1)
    xf = tape.reshape(xf, (b, h, l + 1, 2 * l - 1))
    xf = tape.slice(xf, 0, l, axis=2)              # [b,h,l,2l-1]
    return tape.slice(xf, l - 1, 2 * l - 1, axis=-1)


def _abs_to_rel(tape, x, length):
    b, h, l, _ = x.shape
    xp = tape.pad_const(x, 0, l - 1, axis=-1)
    xf = tape.reshape(xp, (b, h, l * l + l * (l - 1)))
    xf = tape.pad_const(xf, l, 0, axis=-1)
    xf = tape.reshape(xf, (b, h, l, 2 * l))
    return tape.slice(xf, 1, 2 * l, axis=-1)


# -- T3-d：enc attn 链 GPU 路径（forward_br / forward_graph）共享衔接桥 --
# 引擎无 pad/reshape/slice 图 op → rel ↔ abs 位置编码换算在 numpy 上用原生
# tape op 计算（backward 梯度链由此生成）。GPU 结果经 tape.reshape 升维进入
# 原生链：tape.reshape 是带 bp 的 record op（bp 将梯度 reshape 回原形状写回
# BatchTensor id），保 GPU 链且形状正确（record_identity 是纯透传，不能跨
# 形状使用）。
def _enc_rel_abs_bridge(tape, rel_bt, P):
    """rel [P,M] BatchTensor → tape.reshape 升维 → 原生 _rel_to_abs 链。

    返回 [1,1,P,P]（tape out）。bp：原生链写 rel_v → reshape bp 写回 rel_bt。
    """
    M = 2 * P - 1
    rel_v = tape.reshape(rel_bt, (1, 1, P, M))
    return _rel_to_abs(tape, rel_v, P)


def _enc_abs_rel_bridge(tape, p_bt, P):
    """p_attn BatchTensor [P,P] → tape.reshape 升维 → 原生 _abs_to_rel 链。

    返回 [1,1,P,M]（tape out）。bp：原生链写 p_v → reshape bp 写回 p_bt。
    调用方另建 p_sq = tape.reshape(p_bt, (P,P)) 作 o1 matmul 的输入（两消费
    的 bp 各写 p_bt，backward 累加）。
    """
    p_v = tape.reshape(p_bt, (1, 1, P, P))
    return _abs_to_rel(tape, p_v, P)


def _enc_bh_grid(tape, bt_, B, H, kc, P):
    """[B,hid,P] BatchTensor → 每 (b,h) 的 [kc,P] tape-out 网格（bh 拆分）。

    bp：经 tape.reshape/slice 链把每 bh 梯度 reshape 回 [B,hid,P] 写回 bt_
    （保 GPU 链、形状正确）。与裸 numpy 视图 + record_identity（形状断裂）
    不同——tape op 有逆重排的 bp。
    """
    r = tape.reshape(bt_, (B, H, kc, P))        # [B,H,kc,P]（bp 写 bt_）
    g = [[None] * H for _ in range(B)]
    for b0 in range(B):
        rb = tape.slice(r, b0, b0 + 1, axis=0)  # [1,H,kc,P]
        for h0 in range(H):
            rbh = tape.slice(rb, h0, h0 + 1, axis=1)   # [1,1,kc,P]
            g[b0][h0] = tape.reshape(rbh, (kc, P))     # [kc,P]（bp 写 rbh）
    return g


def _enc_split_heads(q_np, k_np, v_np, B, H, kc, P, inv_sqrt_kc):
    """q/k/v [B,hid,P] → 分头视图 + qs 预乘（forward_br/forward_graph 共用）。

    返回 (qh, khT, vh, qs)：qh/vh [B,H,P,kc]、khT [B,H,kc,P]（= k^T 表示，
    引擎/图 matmul 无 trans_b）、qs = qh * inv_sqrt_kc（f32 预乘，镜像图
    G2 输入槽值）。视图共享原下载数组内存；qs 为新数组。
    """
    qh = q_np.reshape(B, H, kc, P).transpose(0, 1, 3, 2)
    khT = k_np.reshape(B, H, kc, P)
    vh = v_np.reshape(B, H, kc, P).transpose(0, 1, 3, 2)
    qs = qh * inv_sqrt_kc
    return qh, khT, vh, qs


class TextEncoderTrain:
    def __init__(self, w: dict, cfg: VitsConfig):
        self.w = w
        self.cfg = cfg
        self.hidden = cfg.hidden
        self.n_layers = cfg.n_layers
        self.kc = cfg.k_channels
        self.heads = cfg.n_heads
        self.emb_phone_w = np.asarray(w["enc_p.emb_phone.weight"])
        self.emb_phone_b = np.asarray(w["enc_p.emb_phone.bias"])
        self.emb_pitch_w = np.asarray(w["enc_p.emb_pitch.weight"])

    def _forward_np(self, tape, phone, pitch, x_mask):
        """phone: [B, P, D]；pitch: [B, P] int；x_mask: [1,1,P]。

        返回 (m, logs) [B, hidden, P]。
        """
        hid = self.hidden
        w = self.w
        x = tape.linear(phone, self.emb_phone_w, self.emb_phone_b)
        if pitch is not None:
            x = tape.add(x, tape.embedding(pitch, self.emb_pitch_w))
        x = tape.mul_const(x, math.sqrt(hid))
        x = tape.leaky_relu(x, LRELU_SLOPE)
        x = tape.transpose(x, (0, 2, 1))  # [B, hid, P]
        P = x.shape[-1]
        attn_mask = (np.asarray(x_mask).reshape(1, 1, P, 1)
                     * np.asarray(x_mask).reshape(1, 1, 1, P))  # 常量
        tape.mark_const(attn_mask)
        x = tape.mul(x, x_mask)
        # T1：attn einsum → matmul（可回退：RVC_TRAIN_ATTN_MATMUL=0 用回 einsum）
        import os as _os_mm  # noqa: PLC0415
        _mmq_on = _os_mm.environ.get("RVC_TRAIN_ATTN_MATMUL", "1") == "1"
        for i in range(self.n_layers):
            _B = x.shape[0]  # F1: batch>1 时 multi-head reshape 用 B
            q = tape.conv1d(x, w[f"enc_p.encoder.attn_layers.{i}.conv_q.weight"],
                            w[f"enc_p.encoder.attn_layers.{i}.conv_q.bias"])
            k = tape.conv1d(x, w[f"enc_p.encoder.attn_layers.{i}.conv_k.weight"],
                            w[f"enc_p.encoder.attn_layers.{i}.conv_k.bias"])
            v = tape.conv1d(x, w[f"enc_p.encoder.attn_layers.{i}.conv_v.weight"],
                            w[f"enc_p.encoder.attn_layers.{i}.conv_v.bias"])
            qh = tape.transpose(tape.reshape(q, (_B, self.heads, self.kc, P)),
                                (0, 1, 3, 2))
            kh = tape.transpose(tape.reshape(k, (_B, self.heads, self.kc, P)),
                                (0, 1, 3, 2))
            vh = tape.transpose(tape.reshape(v, (_B, self.heads, self.kc, P)),
                                (0, 1, 3, 2))
            qs = tape.mul_const(qh, 1.0 / math.sqrt(self.kc))
            scores = (tape.matmul(qs, kh, trans_b=True) if _mmq_on
              else tape.einsum("bhqd,bhkd->bhqk", qs, kh))
            emb_k = np.asarray(w[
                f"enc_p.encoder.attn_layers.{i}.emb_rel_k"])
            used_k = _get_relative_embeddings(emb_k, P, window_size=10)[0]
            rel_logits = (tape.matmul(qs, used_k, trans_b=True) if _mmq_on
                  else tape.einsum("bhqd,md->bhqm", qs, used_k))
            scores = tape.add(scores, _rel_to_abs(tape, rel_logits, P))
            keep = tape.mul(scores, attn_mask)
            neg = tape.mul_const(np.ones_like(attn_mask) - attn_mask, -1e4)
            scores = tape.add(keep, neg)
            p_attn = tape.softmax(scores, axis=-1)
            out = (tape.matmul(p_attn, vh) if _mmq_on
           else tape.einsum("bhqk,bhkd->bhqd", p_attn, vh))
            emb_v = np.asarray(w[
                f"enc_p.encoder.attn_layers.{i}.emb_rel_v"])
            used_v = _get_relative_embeddings(emb_v, P, window_size=10)[0]
            rel_w = _abs_to_rel(tape, p_attn, P)
            out = tape.add(out, (tape.matmul(rel_w, used_v) if _mmq_on
                      else tape.einsum("bhqm,md->bhqd", rel_w, used_v)))
            attn_out = tape.reshape(
                tape.transpose(out, (0, 1, 3, 2)), (_B, hid, P))
            attn_out = tape.conv1d(
                attn_out, w[f"enc_p.encoder.attn_layers.{i}.conv_o.weight"],
                w[f"enc_p.encoder.attn_layers.{i}.conv_o.bias"])
            x = tape.transpose(
                tape.layer_norm(
                    tape.transpose(tape.add(x, attn_out), (0, 2, 1)),
                    w[f"enc_p.encoder.norm_layers_1.{i}.gamma"],
                    w[f"enc_p.encoder.norm_layers_1.{i}.beta"]),
                (0, 2, 1))
            y = tape.conv1d(x, w[f"enc_p.encoder.ffn_layers.{i}.conv_1.weight"],
                            w[f"enc_p.encoder.ffn_layers.{i}.conv_1.bias"],
                            padding=1)
            y = tape.relu(y)
            y = tape.conv1d(y, w[f"enc_p.encoder.ffn_layers.{i}.conv_2.weight"],
                            w[f"enc_p.encoder.ffn_layers.{i}.conv_2.bias"],
                            padding=1)
            x = tape.transpose(
                tape.layer_norm(
                    tape.transpose(tape.add(x, y), (0, 2, 1)),
                    w[f"enc_p.encoder.norm_layers_2.{i}.gamma"],
                    w[f"enc_p.encoder.norm_layers_2.{i}.beta"]),
                (0, 2, 1))
        x = tape.mul(x, x_mask)
        stats = tape.mul(tape.conv1d(x, w["enc_p.proj.weight"],
                                     w["enc_p.proj.bias"]), x_mask)
        m = tape.slice(stats, 0, hid, axis=1)
        logs = tape.slice(stats, hid, 2 * hid, axis=1)
        return m, logs

    def forward(self, tape, phone, pitch, x_mask):
        """T3-d 门控：环境变量分派 GPU 路径。

        RVC_TRAIN_GRAPH_ENC=1 且 RVC_TRAIN_ATTN_MATMUL=1 → forward_graph
        （EncGraph 整链图化）；RVC_TRAIN_GRAPH_ENC=0 且
        RVC_TRAIN_BR_FWD_ENC=1（默认 0）→ forward_br（链式 BatchRunner
        分段 GPU 基准）；否则 numpy 基准（_forward_np）。
        """
        import os as _os_e  # noqa: PLC0415
        _ge = _os_e.environ.get("RVC_TRAIN_GRAPH_ENC", "0")
        if (_ge == "1" and _os_e.environ.get("RVC_TRAIN_ATTN_MATMUL", "1") == "1"):
            return self._forward_graph(tape, phone, pitch, x_mask)
        if _ge == "0" and _os_e.environ.get("RVC_TRAIN_BR_FWD_ENC", "0") == "1":
            return self._forward_br(tape, phone, pitch, x_mask)
        return self._forward_np(tape, phone, pitch, x_mask)

    def _forward_br(self, tape, phone, pitch, x_mask):
        """T3-d：enc_p 前向 GPU 链式基准（链式共享 BatchRunner 分段录制）。

        分段结构与 forward_graph 完全同构（同 record 序列、同 numpy 衔接）：
        GPU 段在 _chain_br 上执行（段间 br.commit() 后经 __array__ 下载做
        预乘/预填/转置衔接），tape.record_* 纯记 bp（out/x 保持 BatchTensor
        引用——J19：bp 值依赖经 __array__ 自动下载，backward 时 chain br
        仍存活）。chain br 由 train_step 末 _release_chain_br 统一释放。
        返回 (m, logs) [B, hid, P]。
        """
        br = _chain_br()
        hid = self.hidden
        w = self.w
        cfg = self.cfg
        B, P, D = phone.shape
        H = self.heads
        kc = self.kc
        L = self.n_layers
        filt = int(cfg.filter)
        inv_sqrt_kc = np.float32(1.0 / math.sqrt(self.kc))
        xmask32 = np.asarray(x_mask, np.float32)              # [1,1,P]
        mask_full = (xmask32.reshape(1, 1, P, 1)
                     * xmask32.reshape(1, 1, 1, P)).astype(np.float32)
        neg_full = ((1.0 - mask_full) * -1e4).astype(np.float32)
        xmask_full = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, hid, P)).astype(np.float32))
        mask_arr = np.asarray(mask_full[0, 0], np.float32)
        neg_arr = np.asarray(neg_full[0, 0], np.float32)
        sqrt_arr = np.full(B * P * hid, np.sqrt(np.float32(hid)),
                           np.float32).reshape(B * P, hid)
        phone2d = np.asarray(phone, np.float32).reshape(B * P, D)
        # -- G_emb：linear(matmul+bias) + add(pitch) + mul(√hid) + leaky --
        lin_m = br.matmul(phone2d,
                          np.ascontiguousarray(
                              np.asarray(self.emb_phone_w, np.float32).T))
        lin_b = br.bias_add(lin_m, np.asarray(self.emb_phone_b, np.float32))
        tape.record_linear(lin_b, phone2d, self.emb_phone_w, self.emb_phone_b)
        if pitch is not None:
            pids = np.asarray(pitch).ravel()
            pe = np.asarray(self.emb_pitch_w, np.float32)[pids]  # [B*P,hid]
            lin_pe = br.add_inplace(lin_b, pe)
            tape.record_embedding(lin_pe, pids, self.emb_pitch_w)
            tape.record_add(lin_pe, lin_b, pe)
        else:
            lin_pe = lin_b
        lin_sq = br.mul_inplace(lin_pe, sqrt_arr)
        x0_bt = br.leaky_relu(lin_sq, LRELU_SLOPE)
        tape.record_mul(lin_sq, lin_pe, sqrt_arr)
        tape.record_leaky_relu(x0_bt, lin_sq, LRELU_SLOPE)
        br.commit()  # G_emb 段
        # -- 衔接：x0 → 层输入（×x_mask 一次，镜像 _forward_np）--
        x0t = tape.transpose(tape.reshape(x0_bt, (B, P, hid)),
                             (0, 2, 1))     # [B,hid,P]（bp 写回 x0_bt）
        x_cur = x0t * xmask_full
        tape.record_mul(x_cur, x0t, xmask_full)
        for i in range(L):
            wp = f"enc_p.encoder.attn_layers.{i}."
            np1 = f"enc_p.encoder.norm_layers_1.{i}."
            np2 = f"enc_p.encoder.norm_layers_2.{i}."
            ffn = f"enc_p.encoder.ffn_layers.{i}."
            # -- S1：q/k/v 卷积 --
            q_bt = br.conv1d(x_cur,
                             np.asarray(w[wp + "conv_q.weight"], np.float32),
                             np.asarray(w[wp + "conv_q.bias"], np.float32))
            k_bt = br.conv1d(x_cur,
                             np.asarray(w[wp + "conv_k.weight"], np.float32),
                             np.asarray(w[wp + "conv_k.bias"], np.float32))
            v_bt = br.conv1d(x_cur,
                             np.asarray(w[wp + "conv_v.weight"], np.float32),
                             np.asarray(w[wp + "conv_v.bias"], np.float32))
            tape.record_conv1d(q_bt, x_cur, w[wp + "conv_q.weight"],
                               w[wp + "conv_q.bias"])
            tape.record_conv1d(k_bt, x_cur, w[wp + "conv_k.weight"],
                               w[wp + "conv_k.bias"])
            tape.record_conv1d(v_bt, x_cur, w[wp + "conv_v.weight"],
                               w[wp + "conv_v.bias"])
            # -- S1 后：q/k/v 分头（tape op 链保梯度回写卷积 out）--
            br.commit()  # S1 段
            qg = _enc_bh_grid(tape, q_bt, B, H, kc, P)   # [kc,P] 网格
            kg = _enc_bh_grid(tape, k_bt, B, H, kc, P)
            vg = _enc_bh_grid(tape, v_bt, B, H, kc, P)
            # qs = q^T 预乘 [P,kc]；khT = k 头 [kc,P]；vh = v^T [P,kc]
            qs = [[tape.mul_const(tape.transpose(qg[b][h], (1, 0)),
                                  inv_sqrt_kc)
                   for h in range(H)] for b in range(B)]
            khT = [[kg[b][h] for h in range(H)] for b in range(B)]
            vh = [[tape.transpose(vg[b][h], (1, 0))
                   for h in range(H)] for b in range(B)]
            emb_k = np.asarray(w[wp + "emb_rel_k"])
            used_k = _get_relative_embeddings(emb_k, P, window_size=10)[0]
            used_kT = np.ascontiguousarray(np.asarray(used_k, np.float32).T)
            emb_v = np.asarray(w[wp + "emb_rel_v"])
            used_v = np.ascontiguousarray(np.asarray(
                _get_relative_embeddings(emb_v, P, window_size=10)[0],
                np.float32))
            # -- S2：每 bh scores=qs×khT、rel=qs×used_kT --
            scs = [[None] * H for _ in range(B)]
            rls = [[None] * H for _ in range(B)]
            for b0 in range(B):
                for h0 in range(H):
                    scs[b0][h0] = br.matmul(qs[b0][h0], khT[b0][h0])
                    rls[b0][h0] = br.matmul(qs[b0][h0], used_kT)
                    tape.record_matmul(scs[b0][h0], qs[b0][h0], khT[b0][h0])
                    tape.record_matmul(rls[b0][h0], qs[b0][h0], used_kT)
            br.commit()  # S2 段
            # -- S3：每 bh add(rel_abs)+mul(mask)+add(neg)+softmax --
            p_list = [[None] * H for _ in range(B)]
            for b0 in range(B):
                for h0 in range(H):
                    rel_abs = _enc_rel_abs_bridge(tape, rls[b0][h0], P)
                    rel_abs0 = tape.reshape(rel_abs, (P, P))   # [P,P]（bp 写回）
                    s_m = br.add_inplace(
                        scs[b0][h0], np.asarray(rel_abs0, np.float32))
                    s_k = br.mul_inplace(s_m, mask_arr)
                    s_n = br.add_inplace(s_k, neg_arr)
                    p_list[b0][h0] = br.softmax(s_n)
                    tape.record_add(s_m, scs[b0][h0], rel_abs0)
                    tape.record_mul(s_k, s_m, mask_arr)
                    tape.record_add(s_n, s_k, neg_arr)
                    tape.record_softmax(p_list[b0][h0], s_n)
            br.commit()  # S3 段
            # -- S4：每 bh o1=p×vh、o2=rel_w×used_v、o=o1+o2 --
            o_bts = []
            M = 2 * P - 1
            for b0 in range(B):
                for h0 in range(H):
                    p_sq = tape.reshape(p_list[b0][h0], (P, P))  # [P,P]（bp 写 p_bt）
                    rel_w = _enc_abs_rel_bridge(tape, p_list[b0][h0], P)
                    rel_w0 = tape.reshape(rel_w, (P, M))         # [P,M]（bp 写回）
                    o1b = br.matmul(p_sq, vh[b0][h0])
                    o2b = br.matmul(np.asarray(rel_w0, np.float32), used_v)
                    ob = br.add_inplace(o1b, o2b)
                    tape.record_matmul(o1b, p_sq, vh[b0][h0])
                    tape.record_matmul(o2b, rel_w0, used_v)
                    tape.record_add(ob, o1b, o2b)
                    o_bts.append(ob)
            br.commit()  # S4 段
            parts = [np.asarray(ob) for ob in o_bts]  # [P,kc] 下载对象
            # -- S5a：attn_in 组装 → conv_o → x1 = x + attn_o --
            attn_in = np.ascontiguousarray(
                np.stack([p_.T for p_ in parts]).reshape(B, hid, P))
            attn_o = br.conv1d(attn_in,
                               np.asarray(w[wp + "conv_o.weight"], np.float32),
                               np.asarray(w[wp + "conv_o.bias"], np.float32))
            x1b = br.add_inplace(x_cur, attn_o)
            tape.record_conv1d(attn_o, attn_in, w[wp + "conv_o.weight"],
                               w[wp + "conv_o.bias"])
            for idx in range(B * H):
                tape.record_identity(parts[idx], o_bts[idx])
            tape.record_split_gather(attn_in, parts, (kc, P))
            tape.record_add(x1b, x_cur, attn_o)
            br.commit()  # S5a 段
            # -- S5b：layer_norm1（x1t → xln）--
            x1t = tape.transpose(x1b, (0, 2, 1))      # [B,P,hid]（bp 写 x1b）
            in_xt = br.layer_norm(x1t, np.asarray(w[np1 + "gamma"], np.float32),
                                  np.asarray(w[np1 + "beta"], np.float32))
            tape.record_layer_norm(in_xt, x1t, w[np1 + "gamma"],
                                   w[np1 + "beta"])
            br.commit()  # S5b 段
            # -- S5c：conv_1(pad1)+relu+conv_2(pad1)+add --
            x2 = tape.transpose(in_xt, (0, 2, 1))     # [B,hid,P]（bp 写 in_xt）
            y1b = br.conv1d(x2, np.asarray(w[ffn + "conv_1.weight"], np.float32),
                            np.asarray(w[ffn + "conv_1.bias"], np.float32),
                            padding=1)
            y1r = br.relu(y1b)
            y2b = br.conv1d(y1r, np.asarray(w[ffn + "conv_2.weight"], np.float32),
                            np.asarray(w[ffn + "conv_2.bias"], np.float32),
                            padding=1)
            x3b = br.add_inplace(x2, y2b)
            tape.record_conv1d(y1b, x2, w[ffn + "conv_1.weight"],
                               w[ffn + "conv_1.bias"], padding=1)
            tape.record_relu(y1r, y1b)
            tape.record_conv1d(y2b, y1r, w[ffn + "conv_2.weight"],
                               w[ffn + "conv_2.bias"], padding=1)
            tape.record_add(x3b, x2, y2b)
            br.commit()  # S5c 段
            # -- S5d：layer_norm2（x3t → xln2）--
            x3t = tape.transpose(x3b, (0, 2, 1))      # [B,P,hid]（bp 写 x3b）
            in_xt2 = br.layer_norm(x3t, np.asarray(w[np2 + "gamma"], np.float32),
                                   np.asarray(w[np2 + "beta"], np.float32))
            tape.record_layer_norm(in_xt2, x3t, w[np2 + "gamma"],
                                   w[np2 + "beta"])
            br.commit()  # S5d 段
            # -- 跨层衔接：层间不 mask（镜像 _forward_np），xln2t → x --
            xln2t = tape.transpose(in_xt2, (0, 2, 1))  # [B,hid,P]（bp 写 in_xt2）
            x_cur = xln2t
        # -- 收尾：×x_mask → proj conv → ×x_mask → m/logs（最后入列）--
        x_m2 = xln2t * xmask_full
        tape.record_mul(x_m2, xln2t, xmask_full)
        stats_bt = br.conv1d(x_m2, np.asarray(w["enc_p.proj.weight"], np.float32),
                             np.asarray(w["enc_p.proj.bias"], np.float32))
        tape.record_conv1d(stats_bt, x_m2, w["enc_p.proj.weight"],
                           w["enc_p.proj.bias"])
        br.commit()  # proj 段
        stats_np = np.asarray(stats_bt)  # [B,2hid,P]
        tape.record_identity(stats_np, stats_bt)
        stats = stats_np * xmask32
        tape.record_mul(stats, stats_np, xmask32)
        m = tape.slice(stats, 0, hid, axis=1)
        logs = tape.slice(stats, hid, 2 * hid, axis=1)
        return m, logs

    def _forward_graph(self, tape, phone, pitch, x_mask):
        """T3-d：enc_p 前向整链图化（EncGraph 分段图 + Python 衔接）。

        段 = gr.run(dg.ghs[name])；record out 用 bt(name) 包装（BatchTensor
        挂 _chain_br；同一 buffer 的多 out_entry 各建一次对象并复用——
        J19：同 buffer 不同对象梯度 id 分离）。衔接与 forward_br 完全同构
        （同 record 序列、同 numpy 预乘/预填）。权重/常量槽每步 set_input
        覆写；sqrt_full 槽 build 预填。返回 (m, logs) [B, hid, P]。
        """
        from runtime.vulkan_ops import BatchTensor  # noqa: PLC0415
        gr = _graph_runner()
        hid = self.hidden
        w = self.w
        cfg = self.cfg
        B, P, D = phone.shape
        H = self.heads
        kc = self.kc
        L = self.n_layers
        filt = int(cfg.filter)
        inv_sqrt_kc = np.float32(1.0 / math.sqrt(self.kc))
        xmask32 = np.asarray(x_mask, np.float32)
        mask_full = (xmask32.reshape(1, 1, P, 1)
                     * xmask32.reshape(1, 1, 1, P)).astype(np.float32)
        neg_full = ((1.0 - mask_full) * -1e4).astype(np.float32)
        xmask_full = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, hid, P)).astype(np.float32))
        mask_arr = np.asarray(mask_full[0, 0], np.float32)
        neg_arr = np.asarray(neg_full[0, 0], np.float32)
        sqrt_arr = np.full(B * P * hid, np.sqrt(np.float32(hid)),
                           np.float32).reshape(B * P, hid)
        phone2d = np.asarray(phone, np.float32).reshape(B * P, D)
        dg = gr.enc_graph(("enc", B, P),
                          lambda: gr._build_enc(self, cfg, B, P, D))
        chain = _chain_br()

        def bt(name):
            bid, shape = dg.outs[name]
            return BatchTensor(chain, bid, shape)

        # 权重槽每步覆写（保证与当前权重一致）
        gr.set_input(np.ascontiguousarray(
                         np.asarray(self.emb_phone_w, np.float32).T),
                     dg.wslots["emb_w"])
        gr.set_input(np.asarray(self.emb_phone_b, np.float32),
                     dg.wslots["emb_b"])
        gr.set_input(np.asarray(w["enc_p.proj.weight"], np.float32),
                     dg.wslots["proj_w"])
        gr.set_input(np.asarray(w["enc_p.proj.bias"], np.float32),
                     dg.wslots["proj_b"])
        for i in range(L):
            wp = f"enc_p.encoder.attn_layers.{i}."
            np1 = f"enc_p.encoder.norm_layers_1.{i}."
            np2 = f"enc_p.encoder.norm_layers_2.{i}."
            ffn = f"enc_p.encoder.ffn_layers.{i}."
            for sk, kk in (("q", "conv_q"), ("k", "conv_k"),
                           ("v", "conv_v"), ("o", "conv_o")):
                gr.set_input(np.asarray(w[wp + kk + ".weight"], np.float32),
                             dg.wslots[f"l{i}.{sk}_w"])
                gr.set_input(np.asarray(w[wp + kk + ".bias"], np.float32),
                             dg.wslots[f"l{i}.{sk}_b"])
            for sk, kk in (("c1", "conv_1"), ("c2", "conv_2")):
                gr.set_input(np.asarray(w[ffn + kk + ".weight"], np.float32),
                             dg.wslots[f"l{i}.{sk}_w"])
                gr.set_input(np.asarray(w[ffn + kk + ".bias"], np.float32),
                             dg.wslots[f"l{i}.{sk}_b"])
            for sk, nb in (("n1", np1), ("n2", np2)):
                gr.set_input(np.asarray(w[nb + "gamma"], np.float32),
                             dg.wslots[f"l{i}.{sk}_g"])
                gr.set_input(np.asarray(w[nb + "beta"], np.float32),
                             dg.wslots[f"l{i}.{sk}_b"])
        # 常量槽每步覆写
        gr.set_input(xmask_full, dg.slots["xmask"])
        gr.set_input(mask_arr, dg.slots["mask"])
        gr.set_input(neg_arr, dg.slots["neg"])
        # -- G_emb：linear(matmul+bias) + add(pitch) + mul(√hid) + leaky --
        gr.set_input(phone2d, dg.slots["phone"])
        if pitch is not None:
            pids = np.asarray(pitch).ravel()
            pe = np.asarray(self.emb_pitch_w, np.float32)[pids]  # [B*P,hid]
        else:
            pids = None
            pe = np.zeros(B * P * hid, np.float32)
        gr.set_input(np.ascontiguousarray(pe), dg.slots["pe"])
        gr.run(dg.ghs["emb"])
        lin_bt = bt("lin")
        lin_a_bt = bt("lin_a")
        lin_mc_bt = bt("lin_mc")
        x0_bt = bt("x0")
        tape.record_linear(lin_bt, phone2d, self.emb_phone_w, self.emb_phone_b)
        if pitch is not None:
            lin_a_bt = bt("lin_a")
            tape.record_embedding(lin_a_bt, pids, self.emb_pitch_w)
            tape.record_add(lin_a_bt, lin_bt, pe)
            mul_in = lin_a_bt
        else:
            # no-pitch：无 add 记录，mul 直接挂 lin_bt，保链到 emb_phone 梯度
            mul_in = lin_bt
        tape.record_mul(lin_mc_bt, mul_in, sqrt_arr)
        tape.record_leaky_relu(x0_bt, lin_mc_bt, LRELU_SLOPE)
        # -- 衔接：x0 → 层输入（×x_mask 一次，镜像 _forward_np）--
        x0t = tape.transpose(tape.reshape(x0_bt, (B, P, hid)),
                             (0, 2, 1))       # [B,hid,P]（bp 写 x0_bt）
        x_cur = x0t * xmask_full
        tape.record_mul(x_cur, x0t, xmask_full)
        for i in range(L):
            wp = f"enc_p.encoder.attn_layers.{i}."
            np1 = f"enc_p.encoder.norm_layers_1.{i}."
            np2 = f"enc_p.encoder.norm_layers_2.{i}."
            ffn = f"enc_p.encoder.ffn_layers.{i}."
            # 每层相对位置嵌入 → 共享槽覆写
            emb_k = np.asarray(w[wp + "emb_rel_k"])
            used_k = _get_relative_embeddings(emb_k, P, window_size=10)[0]
            used_kT = np.ascontiguousarray(np.asarray(used_k, np.float32).T)
            emb_v = np.asarray(w[wp + "emb_rel_v"])
            used_v = np.ascontiguousarray(np.asarray(
                _get_relative_embeddings(emb_v, P, window_size=10)[0],
                np.float32))
            gr.set_input(used_kT, dg.slots["used_kT"])
            gr.set_input(used_v, dg.slots["used_v"])
            # -- G1：q/k/v 卷积 --
            gr.set_input(x_cur, dg.slots["x"])
            gr.run(dg.ghs[f"l{i}.g1"])
            q_bt = bt(f"l{i}.q")
            k_bt = bt(f"l{i}.k")
            v_bt = bt(f"l{i}.v")
            tape.record_conv1d(q_bt, x_cur, w[wp + "conv_q.weight"],
                               w[wp + "conv_q.bias"])
            tape.record_conv1d(k_bt, x_cur, w[wp + "conv_k.weight"],
                               w[wp + "conv_k.bias"])
            tape.record_conv1d(v_bt, x_cur, w[wp + "conv_v.weight"],
                               w[wp + "conv_v.bias"])
            qg = _enc_bh_grid(tape, q_bt, B, H, kc, P)   # [kc,P] 网格
            kg = _enc_bh_grid(tape, k_bt, B, H, kc, P)
            vg = _enc_bh_grid(tape, v_bt, B, H, kc, P)
            # qs = q^T 预乘 [P,kc]；khT = k 头 [kc,P]；vh = v^T [P,kc]
            qs = [[tape.mul_const(tape.transpose(qg[b][h], (1, 0)),
                                  inv_sqrt_kc)
                   for h in range(H)] for b in range(B)]
            khT = [[kg[b][h] for h in range(H)] for b in range(B)]
            vh = [[tape.transpose(vg[b][h], (1, 0))
                   for h in range(H)] for b in range(B)]
            # -- G2：每 bh scores=qs×khT、rel=qs×used_kT --
            for b0 in range(B):
                for h0 in range(H):
                    b = b0 * H + h0
                    gr.set_input(np.ascontiguousarray(qs[b0][h0]),
                                 dg.slots[f"qs.{i}.{b}"])
                    gr.set_input(np.ascontiguousarray(khT[b0][h0]),
                                 dg.slots[f"khT.{i}.{b}"])
            gr.run(dg.ghs[f"l{i}.g2"])
            scs = [[None] * H for _ in range(B)]
            rls = [[None] * H for _ in range(B)]
            for b0 in range(B):
                for h0 in range(H):
                    b = b0 * H + h0
                    scs[b0][h0] = bt(f"l{i}.scores.{b}")
                    rls[b0][h0] = bt(f"l{i}.rel.{b}")
                    tape.record_matmul(scs[b0][h0], qs[b0][h0], khT[b0][h0])
                    tape.record_matmul(rls[b0][h0], qs[b0][h0], used_kT)
            # -- G3：每 bh add(rel_abs)+mul(mask)+add(neg)+softmax --
            rel_abs_list = [[None] * H for _ in range(B)]
            for b0 in range(B):
                for h0 in range(H):
                    b = b0 * H + h0
                    rel_abs = _enc_rel_abs_bridge(tape, rls[b0][h0], P)
                    rel_abs0 = tape.reshape(rel_abs, (P, P))   # [P,P]（bp 写回）
                    rel_abs_list[b0][h0] = rel_abs0
                    gr.set_input(np.ascontiguousarray(rel_abs0),
                                 dg.slots[f"rel_abs.{i}.{b}"])
            gr.run(dg.ghs[f"l{i}.g3"])
            p_list = [[None] * H for _ in range(B)]
            for b0 in range(B):
                for h0 in range(H):
                    b = b0 * H + h0
                    s_a = bt(f"l{i}.s_a.{b}")
                    s_m = bt(f"l{i}.s_m.{b}")
                    s_n = bt(f"l{i}.s_n.{b}")
                    p_list[b0][h0] = bt(f"l{i}.p_attn.{b}")
                    tape.record_add(s_a, scs[b0][h0], rel_abs_list[b0][h0])
                    tape.record_mul(s_m, s_a, mask_arr)
                    tape.record_add(s_n, s_m, neg_arr)
                    tape.record_softmax(p_list[b0][h0], s_n)
            # -- G4：每 bh o1=p×vh、o2=rel_w×used_v、o=o1+o2 --
            rel_w_list = [[None] * H for _ in range(B)]
            p_sq_list = [[None] * H for _ in range(B)]
            M = 2 * P - 1
            for b0 in range(B):
                for h0 in range(H):
                    b = b0 * H + h0
                    p_sq = tape.reshape(p_list[b0][h0], (P, P))  # [P,P]（bp 写 p_bt）
                    p_sq_list[b0][h0] = p_sq
                    rel_w = _enc_abs_rel_bridge(tape, p_list[b0][h0], P)
                    rel_w0 = tape.reshape(rel_w, (P, M))         # [P,M]（bp 写回）
                    rel_w_list[b0][h0] = rel_w0
                    gr.set_input(np.ascontiguousarray(rel_w0),
                                 dg.slots[f"rel_w.{i}.{b}"])
                    gr.set_input(np.ascontiguousarray(vh[b0][h0]),
                                 dg.slots[f"vh.{i}.{b}"])
            gr.run(dg.ghs[f"l{i}.g4"])
            o_bts = []
            for b0 in range(B):
                for h0 in range(H):
                    b = b0 * H + h0
                    o1b = bt(f"l{i}.o1.{b}")
                    o2b = bt(f"l{i}.o2.{b}")
                    ob = bt(f"l{i}.out.{b}")
                    tape.record_matmul(o1b, p_sq_list[b0][h0], vh[b0][h0])
                    tape.record_matmul(o2b, rel_w_list[b0][h0], used_v)
                    tape.record_add(ob, o1b, o2b)
                    o_bts.append(ob)
            parts = [np.asarray(ob) for ob in o_bts]  # [P,kc] 下载对象
            # -- G5a：attn_in 组装 → conv_o → x1 = x + attn_o --
            attn_in = np.ascontiguousarray(
                np.stack([p_.T for p_ in parts]).reshape(B, hid, P))
            gr.set_input(attn_in, dg.slots["attn_in"])
            gr.run(dg.ghs[f"l{i}.g5a"])
            attn_o_bt = bt(f"l{i}.attn_o")
            x1_bt = bt(f"l{i}.x1")
            tape.record_conv1d(attn_o_bt, attn_in, w[wp + "conv_o.weight"],
                               w[wp + "conv_o.bias"])
            for idx in range(B * H):
                tape.record_identity(parts[idx], o_bts[idx])
            tape.record_split_gather(attn_in, parts, (kc, P))
            tape.record_add(x1_bt, x_cur, attn_o_bt)
            # -- G5b：layer_norm1（x1t → xln）--
            x1t = tape.transpose(x1_bt, (0, 2, 1))   # [B,P,hid]（bp 写 x1_bt）
            gr.set_input(x1t, dg.slots["in_xt"])
            gr.run(dg.ghs[f"l{i}.g5b"])
            xln_bt = bt(f"l{i}.xln")
            tape.record_layer_norm(xln_bt, x1t, w[np1 + "gamma"],
                                   w[np1 + "beta"])
            # -- G5c：conv_1(pad1)+relu+conv_2(pad1)+add --
            x2 = tape.transpose(xln_bt, (0, 2, 1))   # [B,hid,P]（bp 写 xln_bt）
            gr.set_input(x2, dg.slots["in_x2"])
            gr.run(dg.ghs[f"l{i}.g5c"])
            y1_bt = bt(f"l{i}.y1")
            y1r_bt = bt(f"l{i}.y1r")
            y2_bt = bt(f"l{i}.y2")
            x3_bt = bt(f"l{i}.x3")
            tape.record_conv1d(y1_bt, x2, w[ffn + "conv_1.weight"],
                               w[ffn + "conv_1.bias"], padding=1)
            tape.record_relu(y1r_bt, y1_bt)
            tape.record_conv1d(y2_bt, y1r_bt, w[ffn + "conv_2.weight"],
                               w[ffn + "conv_2.bias"], padding=1)
            tape.record_add(x3_bt, x2, y2_bt)
            # -- G5d：layer_norm2（x3t → xln2）--
            x3t = tape.transpose(x3_bt, (0, 2, 1))   # [B,P,hid]（bp 写 x3_bt）
            gr.set_input(x3t, dg.slots["in_xt2"])
            gr.run(dg.ghs[f"l{i}.g5d"])
            xln2_bt = bt(f"l{i}.xln2")
            tape.record_layer_norm(xln2_bt, x3t, w[np2 + "gamma"],
                                   w[np2 + "beta"])
            # -- 跨层衔接：层间不 mask（镜像 _forward_np），xln2t → x --
            xln2t = tape.transpose(xln2_bt, (0, 2, 1))  # [B,hid,P]（bp 写 xln2_bt）
            x_cur = xln2t
        # -- 收尾：×x_mask → proj conv → ×x_mask → m/logs（最后入列）--
        x_m2 = xln2t * xmask_full
        tape.record_mul(x_m2, xln2t, xmask_full)
        gr.set_input(x_m2, dg.slots["x"])
        gr.run(dg.ghs["proj"])
        stats_bt = bt("stats")
        tape.record_conv1d(stats_bt, x_m2, w["enc_p.proj.weight"],
                           w["enc_p.proj.bias"])
        stats_np = np.asarray(stats_bt)  # [B,2hid,P]
        tape.record_identity(stats_np, stats_bt)
        stats = stats_np * xmask32
        tape.record_mul(stats, stats_np, xmask32)
        m = tape.slice(stats, 0, hid, axis=1)
        logs = tape.slice(stats, hid, 2 * hid, axis=1)
        return m, logs


# ---------------------------------------------------------------------------
# Flow 正变换
# ---------------------------------------------------------------------------
class ResidualCouplingLayerTrain:
    def __init__(self, w: dict, cfg: VitsConfig, idx: int):
        base = f"flow.flows.{idx}"
        self.idx = idx
        self.pre_w = np.asarray(w[f"{base}.pre.weight"])
        self.pre_b = np.asarray(w[f"{base}.pre.bias"])
        self.post_w = np.asarray(w[f"{base}.post.weight"])
        self.post_b = np.asarray(w[f"{base}.post.bias"])
        self.wn = WNTrain(w, f"{base}.enc", cfg.hidden, 5, 3,
                          cfg.gin_channels)
        self.half = cfg.hidden // 2

    def forward(self, tape, x, x_mask, g):
        """T4-1 门控：RVC_TRAIN_GRAPH_FLOW=1 → _forward_graph（EncGraph
        分段图化）；RVC_TRAIN_GRAPH_FLOW=0 且 RVC_TRAIN_BR_FWD_FLOW=1 →
        _forward_br（链式 BatchRunner，同 kernel → 门禁位级一致）；否则
        numpy 基准。返回 [B, hid, F]（mean_only：x0 原样 + x1'）。
        """
        import os as _os_e  # noqa: PLC0415
        _ge = _os_e.environ.get("RVC_TRAIN_GRAPH_FLOW", "0")
        if _ge == "1":
            return self._forward_graph(tape, x, x_mask, g, None, None)
        if _ge == "0" and _os_e.environ.get("RVC_TRAIN_BR_FWD_FLOW", "0") == "1":
            return self._forward_br(tape, x, x_mask, g)
        return self._forward_np(tape, x, x_mask, g)

    def _forward_np(self, tape, x, x_mask, g):
        """numpy 基准：现有 forward 主体。"""
        x0 = tape.slice(x, 0, self.half, axis=1)
        x1 = tape.slice(x, self.half, 2 * self.half, axis=1)
        h = tape.mul(tape.conv1d(x0, self.pre_w, self.pre_b), x_mask)
        h = self.wn.forward(tape, h, x_mask, g)
        m = tape.mul(tape.conv1d(h, self.post_w, self.post_b), x_mask)
        # mean_only=True：logs=0 -> exp(logs)=1，x1' = m + x1
        x1 = tape.mul(tape.add(m, x1), x_mask)
        return tape.concat([x0, x1], axis=1)

    def _forward_br(self, tape, x, x_mask, g):
        """T4-1：flow 层前向 GPU 链式基准（chain BatchRunner 分段录制）。

        与 _forward_graph 完全同构（同 record 序列、同 tape-op 衔接）；GPU
        段在 _chain_br 上执行（段间 br.commit()）。pre/post 无 w_g（普通
        权重直接 conv）；cond（gin≠0 时）先 deweight_norm 还原。返回
        [B, hid, F]。
        """
        br = _chain_br()
        hid = self.wn.hidden
        half = self.half
        B, _C, F = x.shape
        xmask32 = np.asarray(x_mask, np.float32)
        xmask_hid = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, hid, F)).astype(np.float32))
        xmask_half = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, half, F)).astype(np.float32))
        k_pre = int(self.pre_w.shape[2])
        pad_pre = (k_pre - 1) // 2
        k_post = int(self.post_w.shape[2])
        pad_post = (k_post - 1) // 2
        x0 = tape.slice(x, 0, half, axis=1)
        x1 = tape.slice(x, half, 2 * half, axis=1)
        # -- pre --
        h0 = br.conv1d(np.asarray(x0, np.float32),
                       np.asarray(self.pre_w, np.float32),
                       np.asarray(self.pre_b, np.float32),
                       stride=1, padding=pad_pre)
        tape.record_conv1d(h0, x0, self.pre_w, self.pre_b, padding=pad_pre)
        h = br.mul_inplace(h0, xmask_hid)
        tape.record_mul(h, h0, xmask_hid)
        # -- cond + WN 链 --
        if g is not None:
            cw = (tape.deweight_norm(self.wn.cond_w, self.wn.cond_g)
                  if self.wn.cond_g is not None else self.wn.cond_w)
            g_cond = br.conv1d(np.asarray(g, np.float32),
                               np.asarray(cw, np.float32),
                               np.asarray(self.wn.cond_b, np.float32),
                               stride=1, padding=0)
            tape.record_conv1d(g_cond, g, cw, self.wn.cond_b, padding=0,
                               w_v=self.wn.cond_w, w_g=self.wn.cond_g)
        else:
            g_cond = None
        br.commit()
        h_out = _wn_chain_fwd(tape, self.wn,
                              _make_wn_conv_br(tape, br, self.wn),
                              h, x_mask, g_cond)
        # -- post --
        m0 = br.conv1d(np.asarray(h_out, np.float32),
                       np.asarray(self.post_w, np.float32),
                       np.asarray(self.post_b, np.float32),
                       stride=1, padding=pad_post)
        tape.record_conv1d(m0, h_out, self.post_w, self.post_b,
                           padding=pad_post)
        m = br.mul_inplace(m0, xmask_half)
        tape.record_mul(m, m0, xmask_half)
        br.commit()
        x1n = tape.mul(tape.add(m, x1), x_mask)
        return tape.concat([x0, x1n], axis=1)

    def _forward_graph(self, tape, x, x_mask, g, gr, eg):
        """T4-1：flow 层前向图化（EncGraph 分段图 + Python 衔接；gr/eg 由
        Block 传入共享）。段 = gr.run(eg.ghs[key])；record out 用 bt(key)
        包装（BatchTensor 挂 _chain_br）。搭载方法 _forward_br 同名衔接。
        返回 [B, hid, F]。
        """
        from runtime.vulkan_ops import BatchTensor  # noqa: PLC0415
        chain = _chain_br()
        hid = self.wn.hidden
        half = self.half
        B, _C, F = x.shape
        pfx = f"l{self.idx}."

        def bt(name):
            bid, shape = eg.outs[pfx + name]
            return BatchTensor(chain, bid, shape)

        xmask32 = np.asarray(x_mask, np.float32)
        xmask_hid = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, hid, F)).astype(np.float32))
        xmask_half = np.ascontiguousarray(
            np.broadcast_to(xmask32, (B, half, F)).astype(np.float32))
        k_pre = int(self.pre_w.shape[2])
        pad_pre = (k_pre - 1) // 2
        k_post = int(self.post_w.shape[2])
        pad_post = (k_post - 1) // 2
        x0 = tape.slice(x, 0, half, axis=1)
        x1 = tape.slice(x, half, 2 * half, axis=1)
        gr.set_input(xmask_hid, eg.slots["mask_hid"])
        gr.set_input(xmask_half, eg.slots["mask_half"])
        # -- pre --
        gr.set_input(x0, eg.slots["x0"])
        gr.set_input(np.asarray(self.pre_w, np.float32),
                     eg.wslots[pfx + "pre_w"])
        gr.set_input(np.asarray(self.pre_b, np.float32),
                     eg.wslots[pfx + "pre_b"])
        gr.run(eg.ghs[pfx + "pre"])
        h0_bt = bt("h0")
        h_bt = bt("h")
        tape.record_conv1d(h0_bt, x0, self.pre_w, self.pre_b,
                           padding=pad_pre)
        tape.record_mul(h_bt, h0_bt, xmask_hid)
        # -- cond + WN 链 --
        if g is not None:
            cw = (tape.deweight_norm(self.wn.cond_w, self.wn.cond_g)
                  if self.wn.cond_g is not None else self.wn.cond_w)
            gr.set_input(g, eg.slots["g"])
            gr.set_input(np.asarray(cw, np.float32),
                         eg.wslots[pfx + "cond_w"])
            gr.set_input(np.asarray(self.wn.cond_b, np.float32),
                         eg.wslots[pfx + "cond_b"])
            gr.run(eg.ghs[pfx + "cond"])
            g_cond = bt("g_cond")
            tape.record_conv1d(g_cond, g, cw, self.wn.cond_b, padding=0,
                               w_v=self.wn.cond_w, w_g=self.wn.cond_g)
        else:
            g_cond = None
        h_out = _wn_chain_fwd(tape, self.wn,
                              _make_wn_conv_graph(tape, gr, eg, self.wn,
                                                  pfx + "in.{}",
                                                  pfx + "rs.{}"),
                              h_bt, x_mask, g_cond)
        # -- post --
        gr.set_input(h_out, eg.slots["h"])
        gr.set_input(np.asarray(self.post_w, np.float32),
                     eg.wslots[pfx + "post_w"])
        gr.set_input(np.asarray(self.post_b, np.float32),
                     eg.wslots[pfx + "post_b"])
        gr.run(eg.ghs[pfx + "post"])
        m0_bt = bt("m0")
        m_bt = bt("m")
        tape.record_conv1d(m0_bt, h_out, self.post_w, self.post_b,
                           padding=pad_post)
        tape.record_mul(m_bt, m0_bt, xmask_half)
        x1n = tape.mul(tape.add(m_bt, x1), x_mask)
        return tape.concat([x0, x1n], axis=1)


class ResidualCouplingBlockTrain:
    def __init__(self, w: dict, cfg: VitsConfig):
        self.cfg = cfg
        self.layers = [ResidualCouplingLayerTrain(w, cfg, idx)
                       for idx in (0, 2, 4, 6)]

    def forward(self, tape, x, x_mask, g):
        """T4-1 门控：RVC_TRAIN_GRAPH_FLOW=1 → _forward_graph；=0 且
        RVC_TRAIN_BR_FWD_FLOW=1 → _forward_br；否则 numpy 基准。"""
        import os as _os_e  # noqa: PLC0415
        _ge = _os_e.environ.get("RVC_TRAIN_GRAPH_FLOW", "0")
        if _ge == "1":
            return self._forward_graph(tape, x, x_mask, g)
        if _ge == "0" and _os_e.environ.get("RVC_TRAIN_BR_FWD_FLOW", "0") == "1":
            return self._forward_br(tape, x, x_mask, g)
        return self._forward_np(tape, x, x_mask, g)

    def _forward_np(self, tape, x, x_mask, g):
        """numpy 基准：现有 forward 主体。"""
        for layer in self.layers:
            x = layer._forward_np(tape, x, x_mask, g)
            x = tape.flip(x, axis=1)
        return x

    def _forward_br(self, tape, x, x_mask, g):
        """T4-1：flow 前向 GPU 链式基准（Block 层循环 + tape.flip）。"""
        for layer in self.layers:
            x = layer._forward_br(tape, x, x_mask, g)
            x = tape.flip(x, axis=1)
        return x

    def _forward_graph(self, tape, x, x_mask, g):
        """T4-1：flow 前向图化（EncGraph 分段图集，4 层共享 gr/eg）。"""
        gr = _graph_runner()
        B, _C, F = x.shape
        eg = gr.flow_graph(("flow", B, F),
                           lambda: gr._build_flow(self, B, F))
        for layer in self.layers:
            x = layer._forward_graph(tape, x, x_mask, g, gr, eg)
            x = tape.flip(x, axis=1)
        return x


# ---------------------------------------------------------------------------
# GeneratorNSF（dec）
# ---------------------------------------------------------------------------
_BRFWD_DBG = False  # 阶段H（H5）dec BR 化调试开关（ResBlock._forward_br 用）


def _br_rec(br, recs, t, kind, **kw):
    """阶段H（H5）：登记一笔 BatchRunner 录制（t=输出 tensor），返回 rec 索引。

    recs: [(tensor, kind, kw)]——录制顺序 = 提交/下载/record 顺序（拓扑序）。
    kw 约定：
    - "conv1d"/"convT"：x_ref_idx（前 rec 索引；None=独立输入，用 kw["x"] 原
      对象——**边界输入（z/g/har）必须传原数组**保证梯度回传 enc_p/常数标记）、
      w/b（f32 参数）、stride/padding/dilation/output_padding。
    - "leaky"：x_ref_idx（前 rec 索引）。
    - "add"：a_ref_idx/b_ref_idx（两个输入 rec 索引）。
    """
    recs.append((t, kind, kw))
    return len(recs) - 1


def _play_recs(tape, recs, arrs, offset):
    """按录制序回放 tape.record_*（arrs=同序下载数组；rec 索引相对 offset 段）。

    ref 索引为**全局 rec 索引**（跨段连续），arrs 只含当前段 → 相对索引 =
    ref - offset。x_ref_idx=None 时用 kw["x"]（边界输入原对象 z/g/har）。"""
    for i, (t, kind, kw) in enumerate(recs):
        arr = arrs[i]
        if kind == "conv1d":
            x = arrs[kw["x_ref_idx"] - offset] if kw.get("x_ref_idx") is not None \
                else kw["x"]
            if os.environ.get("RVC_TRAIN_BR_FWD_DBG", "0") == "1":
                print(f"[play] i={i} kind={kind} out_id={id(arr)} "
                      f"x_id={id(x)} x_ref={kw.get('x_ref_idx')} off={offset}")
            if kw.get("w_v_ref") is not None and kw.get("w_g_ref") is not None:
                w = tape.deweight_norm(kw["w_v_ref"], kw["w_g_ref"])
            else:
                w = kw.get("w_ref", kw["w"])
            b = kw.get("b_ref", kw.get("b"))
            tape.record_conv1d(arr, x, w, b,
                               stride=kw.get("stride", 1),
                               padding=kw.get("padding", 0),
                               dilation=kw.get("dilation", 1),
                               w_v=kw.get("w_v_ref"), w_g=kw.get("w_g_ref"))
        elif kind == "convT":
            x = arrs[kw["x_ref_idx"] - offset] if kw.get("x_ref_idx") is not None \
                else kw["x"]
            if kw.get("w_v_ref") is not None and kw.get("w_g_ref") is not None:
                w = tape.deweight_norm(kw["w_v_ref"], kw["w_g_ref"])
            else:
                w = kw.get("w_ref", kw["w"])
            b = kw.get("b_ref", kw.get("b"))
            tape.record_conv_transpose1d(
                arr, x, w, b,
                stride=kw.get("stride", 1), padding=kw.get("padding", 0),
                output_padding=kw.get("output_padding", 0),
                dilation=kw.get("dilation", 1))
        elif kind == "leaky":
            x = arrs[kw["x_ref_idx"] - offset] if kw.get("x_ref_idx") is not None \
                else kw["x"]
            if os.environ.get("RVC_TRAIN_BR_FWD_DBG", "0") == "1":
                print(f"[play] i={i} kind=leaky out_id={id(arr)} "
                      f"x_id={id(x)} x_ref={kw.get('x_ref_idx')} off={offset}")
            tape.record_leaky_relu(arr, x, kw.get("slope", LRELU_SLOPE))
        elif kind == "add":
            tape.record_add(arr, arrs[kw["a_ref_idx"] - offset],
                            arrs[kw["b_ref_idx"] - offset])
        elif kind == "copy":
            x = arrs[kw["x_ref_idx"] - offset] if kw.get("x_ref_idx") is not None \
                else kw["x"]
            tape.record_identity(arr, x)
        elif kind == "mul_const":
            tape.record_mul_const(arr, arrs[kw["a_ref_idx"] - offset], kw["c"])
        else:
            raise ValueError(f"未知 rec kind: {kind}")


class ResBlock1Train:
    def __init__(self, w: dict, base: str, channels: int, kernel_size: int):
        self.k = int(kernel_size)
        self.c1_w, self.c1_g, self.c1_b = [], [], []
        self.c2_w, self.c2_g, self.c2_b = [], [], []
        for j in range(3):
            c1w, c1g = _load_wn_pair(w, f"{base}.convs1.{j}")
            self.c1_w.append(c1w)
            self.c1_g.append(c1g)
            self.c1_b.append(np.asarray(w[f"{base}.convs1.{j}.bias"]))
            c2w, c2g = _load_wn_pair(w, f"{base}.convs2.{j}")
            self.c2_w.append(c2w)
            self.c2_g.append(c2g)
            self.c2_b.append(np.asarray(w[f"{base}.convs2.{j}.bias"]))

    def forward(self, tape, x):
        k = self.k
        dilations = [1, 3, 5]
        for j in range(3):
            d = dilations[j]
            c1 = tape.leaky_relu(x, LRELU_SLOPE)
            c1 = _tape_conv(tape, c1, self.c1_w[j], self.c1_b[j],
                            self.c1_g[j], dilation=d,
                            padding=(k * d - d) // 2)
            c1 = tape.leaky_relu(c1, LRELU_SLOPE)
            c2 = _tape_conv(tape, c1, self.c2_w[j], self.c2_b[j],
                            self.c2_g[j], padding=(k - 1) // 2)
            x = tape.add(c2, x)
        return x

    def _forward_br(self, br, recs, x_t, x_ref_idx):
        """阶段H（H5）：ResBlock 前向整链 BatchRunner 录制（copy 保留残差 +
        leaky×2 + conv×2 + add_inplace），逐算子进 recs（与 Generator 链同序，
        commit 后统一下载 + tape.record_*）。返回 (新 x_t, 该 rec 索引)。"""
        k = self.k
        dilations = [1, 3, 5]
        if _BRFWD_DBG:
            print(f"[BRFWD] rb._forward_br x_ref_idx={x_ref_idx} type(x_t)={type(x_t)}")
        # 工作副本（输入 buffer 保持原值；leaky 覆写副本，绝不覆写输入 →
        # 所有 rec 的下载值 = 该 op 真实输出，conv 的 bp x 精确）。
        cur = _br_rec(br, recs, br.copy(x_t), "copy", x_ref_idx=x_ref_idx)
        for j in range(3):
            d = dilations[j]
            # leaky 就地覆写副本 lx（copy 记入 recs，恒等透传梯度）；
            # cur/输入 buffer 恒为原值（残差 add 的 b 直接用 cur）。
            lx = _br_rec(br, recs, br.copy(recs[cur][0]), "copy",
                         x_ref_idx=cur)
            lr = _br_rec(br, recs, br.leaky_relu(recs[lx][0], LRELU_SLOPE),
                         "leaky", x_ref_idx=lx)
            if self.c1_g[j] is not None:
                c1w_gpu = _deweight_any(self.c1_w[j], self.c1_g[j])
            else:
                c1w_gpu = self.c1_w[j]
            c1w32 = np.asarray(c1w_gpu, np.float32)
            c1b32 = np.asarray(self.c1_b[j], np.float32)
            c1 = _br_rec(br, recs, br.conv1d(
                recs[lr][0], c1w32, c1b32, dilation=d,
                padding=(k * d - d) // 2),
                "conv1d", x_ref_idx=lr, w=c1w32, b=c1b32,
                w_v_ref=self.c1_w[j], w_g_ref=self.c1_g[j],
                b_ref=self.c1_b[j],
                dilation=d, padding=(k * d - d) // 2)
            snap2 = _br_rec(br, recs, br.copy(recs[c1][0]), "copy",
                            x_ref_idx=c1)
            lr2 = _br_rec(br, recs, br.leaky_relu(
                recs[c1][0], LRELU_SLOPE), "leaky", x_ref_idx=snap2)
            if self.c2_g[j] is not None:
                c2w_gpu = _deweight_any(self.c2_w[j], self.c2_g[j])
            else:
                c2w_gpu = self.c2_w[j]
            c2w32 = np.asarray(c2w_gpu, np.float32)
            c2b32 = np.asarray(self.c2_b[j], np.float32)
            c2 = _br_rec(br, recs, br.conv1d(
                recs[lr2][0], c2w32, c2b32, padding=(k - 1) // 2),
                "conv1d", x_ref_idx=lr2, w=c2w32, b=c2b32,
                w_v_ref=self.c2_w[j], w_g_ref=self.c2_g[j],
                b_ref=self.c2_b[j],
                padding=(k - 1) // 2)
            # 残差 x_{j+1} = c2_j(...) + x_j（与 numpy forward 一致：b 用
            # leaky 前的原值 buffer cur；add_inplace 只覆写 a=c2 输出）。
            cur = _br_rec(br, recs, br.add_inplace(
                recs[c2][0], recs[cur][0]), "add",
                a_ref_idx=c2, b_ref_idx=cur)
        return cur, cur


class GeneratorNSFTrain:
    def __init__(self, w: dict, cfg: VitsConfig):
        self.w = w
        self.cfg = cfg
        self.n_ups = cfg.n_ups
        self.conv_pre_w = np.asarray(w["dec.conv_pre.weight"])
        self.conv_pre_b = np.asarray(w["dec.conv_pre.bias"])
        self.cond_w = np.asarray(w["dec.cond.weight"])
        self.cond_b = np.asarray(w["dec.cond.bias"])
        self.ups_w, self.ups_g, self.ups_b = [], [], []
        self.noise_w, self.noise_b = [], []
        for i in range(self.n_ups):
            uw, ug = _load_wn_pair(w, f"dec.ups.{i}")
            self.ups_w.append(uw)
            self.ups_g.append(ug)
            self.ups_b.append(np.asarray(w[f"dec.ups.{i}.bias"]))
            self.noise_w.append(np.asarray(w[f"dec.noise_convs.{i}.weight"]))
            self.noise_b.append(np.asarray(w[f"dec.noise_convs.{i}.bias"]))
        self.resblocks = []
        for i in range(self.n_ups):
            ch = cfg.upsample_initial // (2 ** (i + 1))
            for j, ks in enumerate(cfg.resblock_kernel_sizes):
                self.resblocks.append(
                    ResBlock1Train(w, f"dec.resblocks.{i * 3 + j}", ch, ks))
        self.conv_post_w = np.asarray(w["dec.conv_post.weight"])

    def sine_gen(self, f0, ns):
        """SineGen + m_source（与 vits._sine_gen 完全一致，无梯度）。"""
        cfg = self.cfg
        upp = cfg.upp
        sr = cfg.sr
        f0r = np.asarray(f0, dtype=np.float32)[:, None].transpose(0, 2, 1)
        rad = f0r / sr * np.arange(1, upp + 1, dtype=np.float32)[None, None, :]
        rad2 = np.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
        rad_acc = np.fmod(np.cumsum(rad2, axis=1), 1.0)
        rad = rad.copy()
        rad[:, 1:, :] += rad_acc[:, :-1, :]
        rad = rad.reshape(rad.shape[0], -1, 1)  # F1: B 行（B=1 与原 (1,-1,1) 同）
        sine = np.sin(2 * np.pi * rad) * 0.1
        uv = (f0r > 0).astype(np.float32)
        uv = np.repeat(uv, upp, axis=1)
        noise_amp = uv * 0.003 + (1 - uv) * 0.1 / 3
        noise = noise_amp * ns
        sine = sine * uv + noise
        w = self.w
        l_lin = sine @ np.asarray(w["dec.m_source.l_linear.weight"]).T \
            + np.asarray(w["dec.m_source.l_linear.bias"])
        return np.tanh(l_lin).transpose(0, 2, 1).astype(np.float32)

    def forward(self, tape, z, f0, g, ns):
        """z: [B, 192, Tf]；f0: [B, Tf]；g: [B, 256, 1]；ns 噪声。"""
        cfg = self.cfg
        har = self.sine_gen(f0, ns)  # 常量（detach）
        tape.mark_const(har)
        import os as _os  # noqa: PLC0415
        # T3-c：dec 前向整链单图执行（RVC_TRAIN_GRAPH_DEC=1，默认 0 零回归）。
        # 优先级高于 BR_FWD_DEC 门：图化后 forward_graph 内部不再走 _forward_br。
        if _os.environ.get("RVC_TRAIN_GRAPH_DEC", "0") == "1":
            return self.forward_graph(tape, z, g, har)
        # 阶段H（H5）：dec 前向默认 GPU 段式（RVC_TRAIN_BR_FWD_DEC=1）——
        # B=4 干净 wall 34.5s→30.6s（-3.9s/-11%）；f32 GPU 前向贴近原版
        # PyTorch f32 训练语义；设 0 回退 f64 numpy（T46 旧路径）。
        if _os.environ.get("RVC_TRAIN_BR_FWD_DEC", "1") == "1":
            return self._forward_br(tape, z, g, har)
        x = tape.conv1d(z, self.conv_pre_w, self.conv_pre_b, padding=3)
        x = tape.add(x, tape.conv1d(g, self.cond_w, self.cond_b))
        for i in range(self.n_ups):
            x = tape.leaky_relu(x, LRELU_SLOPE)
            up_w = self.ups_w[i]
            if self.ups_g[i] is not None:
                up_w = tape.deweight_norm(up_w, self.ups_g[i])
            x = tape.conv_transpose1d(
                x, up_w, self.ups_b[i],
                stride=cfg.upsample_rates[i],
                padding=(cfg.upsample_kernels[i] - cfg.upsample_rates[i]) // 2)
            x_source = tape.conv1d(
                har, self.noise_w[i], self.noise_b[i],
                stride=cfg.noise_strides[i],
                padding=cfg.noise_strides[i] // 2)
            x = tape.add(x, x_source)
            xs = None
            for j in range(3):
                rb = self.resblocks[i * 3 + j].forward(tape, x)
                xs = rb if xs is None else tape.add(xs, rb)
            x = tape.mul_const(xs, 1.0 / 3)
        x = tape.leaky_relu(x, 0.01)
        x = tape.conv1d(x, self.conv_post_w, None, padding=3)
        return tape.tanh(x)

    def forward_graph(self, tape, z, g, har):
        """T3-c：dec 前向整链单图执行（RVC_TRAIN_GRAPH_DEC=1，默认 0 零回归）。

        结构 = _forward_br 的录制定序（conv_pre + cond add + 4×ups(leaky+
        convT+noise+add) + 12×ResBlock + leaky(0.01) + conv_post），但整链
        一次 graph run：权重/输入按固定槽每步覆写（deweight 每步新数组 →
        图结点绑固定槽 id，不随数组对象变化；上传数 = _forward_br 同权重
        一次/步）；÷3 图内 mul_inplace（预填 1/3 槽）消除 numpy 断链下载/
        再上传。输出/中间值 = 图槽包装的 BatchTensor（runner=chain br），
        tape.record_* 沿用（J19：backward 零上传消费 GPU 槽）。tanh Python。
        """
        from runtime.vulkan_ops import BatchTensor  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        gr = _graph_runner()
        z32 = np.asarray(z, np.float32)
        g32 = np.asarray(g, np.float32)
        har32 = np.asarray(har, np.float32)
        B = int(z32.shape[0])
        Tf = int(z32.shape[2])
        key = ("dec", B, Tf)

        def _build():
            return gr._build_dec(self, self.cfg, B, Tf, int(har32.shape[2]))

        dg = gr.dec_graph(key, _build)
        # -- 1) 权重槽覆写。ups/RB deweight 用 tape.deweight_norm（梯度链
        #    回传 w_v/w_g），结果存 dwn 供 record 复用（数值逐位一致）。
        #    conv_pre/cond/noise/post 无 deweight：record 用训练数组引用
        #    （w_ref 语义），GPU 用 f32 覆写（镜像 _forward_br）。--
        pre_w32 = np.asarray(self.conv_pre_w, np.float32)
        gr.set_input(pre_w32, dg.wslots["pre_w"])
        gr.set_input(np.asarray(self.conv_pre_b, np.float32),
                     dg.wslots["pre_b"])
        dwn = {}
        dwn_pairs = {}  # T0.4：展开数组的 (w_v, w_g) 持久引用表（图化 ResBlock
        # record 透传用——展开数组 id 每步漂移，wpers 按 (id(v),id(g)) 缓存）
        for i in range(self.n_ups):
            uw = self.ups_w[i]
            if self.ups_g[i] is not None:
                uw = tape.deweight_norm(uw, self.ups_g[i])
            dwn[f"up.{i}"] = uw
            gr.set_input(np.asarray(uw, np.float32), dg.wslots[f"up.{i}.w"])
            gr.set_input(np.asarray(self.ups_b[i], np.float32),
                         dg.wslots[f"up.{i}.b"])
            gr.set_input(np.asarray(self.noise_w[i], np.float32),
                         dg.wslots[f"ns.{i}.w"])
            gr.set_input(np.asarray(self.noise_b[i], np.float32),
                         dg.wslots[f"ns.{i}.b"])
        for r in range(len(self.resblocks)):
            rb = self.resblocks[r]
            for j in range(3):
                c1w = rb.c1_w[j]
                if rb.c1_g[j] is not None:
                    c1w = tape.deweight_norm(c1w, rb.c1_g[j])
                dwn[f"rb.{r}.{j}.c1w"] = c1w
                dwn_pairs[f"rb.{r}.{j}.c1w"] = (rb.c1_w[j], rb.c1_g[j])
                gr.set_input(np.asarray(c1w, np.float32),
                             dg.wslots[f"rb.{r}.{j}.c1w"])
                gr.set_input(np.asarray(rb.c1_b[j], np.float32),
                             dg.wslots[f"rb.{r}.{j}.c1b"])
                c2w = rb.c2_w[j]
                if rb.c2_g[j] is not None:
                    c2w = tape.deweight_norm(c2w, rb.c2_g[j])
                dwn[f"rb.{r}.{j}.c2w"] = c2w
                dwn_pairs[f"rb.{r}.{j}.c2w"] = (rb.c2_w[j], rb.c2_g[j])
                gr.set_input(np.asarray(c2w, np.float32),
                             dg.wslots[f"rb.{r}.{j}.c2w"])
                gr.set_input(np.asarray(rb.c2_b[j], np.float32),
                             dg.wslots[f"rb.{r}.{j}.c2b"])
        gr.set_input(np.asarray(self.conv_post_w, np.float32),
                     dg.wslots["post_w"])
        # -- 2) 输入：z / cond（Python 路径：conv1d+commit+repeat）/ har --
        gr.set_input(z32, dg.in_z)
        gr.set_input(har32, dg.in_har)
        cond_w32 = np.asarray(self.cond_w, np.float32)
        cond_b32 = np.asarray(self.cond_b, np.float32)
        br = _chain_br()
        cond_t = br.conv1d(g32, cond_w32, cond_b32)
        br.commit()
        cond_full = np.ascontiguousarray(np.repeat(
            np.asarray(cond_t.numpy()), Tf, axis=2))
        gr.set_input(cond_full, dg.in_cond)
        # -- 3) run（防御：graph.run 内部 batchBegin 清空未提交 ops）--
        _commit_chain_br()
        gr.run(dg.gh)
        # -- 4) 按 _forward_br 录制定序 tape.record_*。out/x 均为图槽
        #    BatchTensor（同一 buffer 的就地覆写用两个对象分离引用——
        #    J19：add/leaky 的 a/x 与 out 不同对象避免双重累加）。--
        chain = _chain_br()
        bt = lambda name: BatchTensor(chain, dg.outs[name][0],
                                      dg.outs[name][1])
        x0 = bt("x0")
        x0a = bt("x0a")
        tape.record_conv1d(x0, z, self.conv_pre_w, self.conv_pre_b,
                           padding=3)
        tape.record_conv1d(cond_t, g, self.cond_w, self.cond_b)
        tape.record_add(x0a, x0, cond_t)
        x_prev = x0a
        dils = (1, 3, 5)
        for i in range(self.n_ups):
            lx = bt(f"lx.{i}")
            lx_lr = bt(f"lx_lr.{i}")
            tape.record_identity(lx, x_prev)
            tape.record_leaky_relu(lx_lr, lx, LRELU_SLOPE)
            up = bt(f"up.{i}")
            tape.record_conv_transpose1d(
                up, lx_lr, dwn[f"up.{i}"], self.ups_b[i],
                stride=int(self.cfg.upsample_rates[i]),
                padding=(int(self.cfg.upsample_kernels[i])
                         - int(self.cfg.upsample_rates[i])) // 2)
            ns = bt(f"ns.{i}")
            tape.record_conv1d(
                ns, har, self.noise_w[i], self.noise_b[i],
                stride=int(self.cfg.noise_strides[i]),
                padding=int(self.cfg.noise_strides[i]) // 2)
            up_a = bt(f"up_a.{i}")
            tape.record_add(up_a, up, ns)
            x_base = up_a
            rb_outs = []
            for j in range(3):
                rb = self.resblocks[i * 3 + j]
                k = rb.k
                cur0 = bt(f"rb.{i * 3 + j}.cur")
                tape.record_identity(cur0, x_base)
                cur_ref = cur0
                for jj in range(3):
                    d = dils[jj]
                    lx_j = bt(f"rb.{i * 3 + j}.lx.{jj}")
                    lx_j_lr = bt(f"rb.{i * 3 + j}.lx_lr.{jj}")
                    tape.record_identity(lx_j, cur_ref)
                    tape.record_leaky_relu(lx_j_lr, lx_j, LRELU_SLOPE)
                    c1 = bt(f"rb.{i * 3 + j}.c1.{jj}")
                    _pk1 = f"rb.{i * 3 + j}.{jj}.c1w"
                    _pv1, _pg1 = dwn_pairs.get(_pk1, (None, None))
                    tape.record_conv1d(
                        c1, lx_j_lr, dwn[_pk1],
                        rb.c1_b[jj], dilation=d, padding=(k * d - d) // 2,
                        w_v=_pv1, w_g=_pg1)
                    snap = bt(f"rb.{i * 3 + j}.snap.{jj}")
                    tape.record_identity(snap, c1)
                    c1_lr = bt(f"rb.{i * 3 + j}.c1_lr.{jj}")
                    tape.record_leaky_relu(c1_lr, snap, LRELU_SLOPE)
                    c2 = bt(f"rb.{i * 3 + j}.c2.{jj}")
                    _pk2 = f"rb.{i * 3 + j}.{jj}.c2w"
                    _pv2, _pg2 = dwn_pairs.get(_pk2, (None, None))
                    tape.record_conv1d(
                        c2, c1_lr, dwn[_pk2],
                        rb.c2_b[jj], padding=(k - 1) // 2,
                        w_v=_pv2, w_g=_pg2)
                    c2_a = bt(f"rb.{i * 3 + j}.c2_a.{jj}")
                    tape.record_add(c2_a, c2, cur_ref)
                    cur_ref = c2_a
                rb_outs.append(cur_ref)
            t1 = bt(f"t1.{i}")
            t1a = bt(f"t1a.{i}")
            t2 = bt(f"t2.{i}")
            t2a = bt(f"t2a.{i}")
            tape.record_identity(t1, rb_outs[0])
            tape.record_add(t1a, t1, rb_outs[1])
            tape.record_identity(t2, t1a)
            tape.record_add(t2a, t2, rb_outs[2])
            xm = bt(f"xm.{i}")
            tape.record_mul_const(xm, t2a, 1.0 / 3)
            x_prev = xm
        # 段2：leaky(0.01) + conv_post（无 bias）+ Python tanh
        xm_lr = bt("xm_lr")
        tape.record_leaky_relu(xm_lr, x_prev, 0.01)
        y_pre = bt("y_pre")
        tape.record_conv1d(y_pre, xm_lr, self.conv_post_w, None, padding=3)
        return tape.tanh(y_pre)

    def _forward_br(self, tape, z, g, har):
        """阶段H（H5）：dec 前向整链 BatchRunner 两段录制。

        段1 = conv_pre/cond/ups(leaky+convT+noise+add)/3×ResBlock → xs；
        mul_const(1/3) 用 numpy 断链（引擎无标量乘，仅 4 次/步）；
        段2 = leaky(0.01) + conv_post。每段 commit 后按录制序 tape.record_*
        纯记录（out/x 保持 BatchTensor 引用——T-H8：backward 时 conv bp 直接
        消费 GPU buffer 零上传）。边界输入 z/g/har 传**原数组对象**（np.asarray
        保引用）→ 梯度正确回传 enc_p/常数跳过。返回 ``tape.tanh`` 输出。
        使用链式共享 BatchRunner（_chain_br）：tensor 存活到训练步末统一
        release，避免局部 br 早释放导致 backward 用失效 GPU buffer。
        """
        if _CAPTURE_ON:
            _m1_cap().dec_frame_open(-1)     # 门1：dec 前向段开口（a26ag :3521|:3522）
        global _BRFWD_DBG
        cfg = self.cfg
        br = _chain_br()
        recs = []
        z32 = np.asarray(z, np.float32)
        g32 = np.asarray(g, np.float32)
        har32 = np.asarray(har, np.float32)
        pre_w = np.asarray(self.conv_pre_w, np.float32)
        pre_b = np.asarray(self.conv_pre_b, np.float32)
        _br_rec(br, recs, br.conv1d(z32, pre_w, pre_b, padding=3),
                "conv1d", x=z, w=pre_w, b=pre_b,
                w_ref=self.conv_pre_w, b_ref=self.conv_pre_b, padding=3)
        cond_w = np.asarray(self.cond_w, np.float32)
        cond_b = np.asarray(self.cond_b, np.float32)
        # cond 输出 [B,512,1]（g 是 [B,256,1]）：引擎 add_inplace 不支持广播，
        # 单独 commit + 下载 + repeat 到 [B,512,Tf] 再并入链（record 的 b 仍用
        # 未广播 [B,512,1]——bp 的 _reduce_to 与 numpy 广播路径语义一致）。
        cond_t = br.conv1d(g32, cond_w, cond_b)
        br.commit()
        cond_arr = np.asarray(cond_t.numpy())
        cond_full = np.ascontiguousarray(
            np.repeat(cond_arr, z.shape[2], axis=2))
        _br_rec(br, recs, cond_t, "conv1d", x=g, w=cond_w, b=cond_b,
                w_ref=self.cond_w, b_ref=self.cond_b)
        _br_rec(br, recs, br.add_inplace(recs[0][0], cond_full),
                "add", a_ref_idx=0, b_ref_idx=1)
        x_ref = 2
        seg_start = 0
        for i in range(self.n_ups):
            # leaky 覆写副本（不覆写衔接/输入 buffer → 衔接 rec 的下载值
            # 保持 x32 原值；lr 的 out = 副本 buffer 不被后续覆写，conv bp
            # 的 x 精确）。
            lx = _br_rec(br, recs, br.copy(recs[x_ref][0]), "copy",
                         x_ref_idx=x_ref)
            lr = _br_rec(br, recs, br.leaky_relu(recs[lx][0], LRELU_SLOPE),
                         "leaky", x_ref_idx=lx)
            up_w = self.ups_w[i]
            if self.ups_g[i] is not None:
                up_w = tape.deweight_norm(up_w, self.ups_g[i])
            up_w32 = np.asarray(up_w, np.float32)
            up_b32 = np.asarray(self.ups_b[i], np.float32)
            _br_rec(br, recs, br.conv_transpose1d(
                recs[lr][0], up_w32, up_b32,
                stride=cfg.upsample_rates[i],
                padding=(cfg.upsample_kernels[i] - cfg.upsample_rates[i]) // 2),
                "convT", x_ref_idx=lr, w=up_w32, b=up_b32,
                w_ref=up_w, b_ref=self.ups_b[i],
                stride=cfg.upsample_rates[i],
                padding=(cfg.upsample_kernels[i] - cfg.upsample_rates[i]) // 2)
            x_ref = len(recs) - 1
            nw32 = np.asarray(self.noise_w[i], np.float32)
            nb32 = np.asarray(self.noise_b[i], np.float32)
            ns = _br_rec(br, recs, br.conv1d(
                har32, nw32, nb32, stride=cfg.noise_strides[i],
                padding=cfg.noise_strides[i] // 2),
                "conv1d", x=har, w=nw32, b=nb32,
                w_ref=self.noise_w[i], b_ref=self.noise_b[i],
                stride=cfg.noise_strides[i],
                padding=cfg.noise_strides[i] // 2)
            _br_rec(br, recs, br.add_inplace(recs[x_ref][0], recs[ns][0]),
                    "add", a_ref_idx=x_ref, b_ref_idx=ns)
            x_ref = len(recs) - 1
            # 3×ResBlock 并联作用于同一 x（ups 输出），Σ/3 ——与 numpy
            # forward 一致（rb 输入固定为 ups 段输出，不随 j 更新）。
            x_base = x_ref
            xs = None
            for j in range(3):
                rb = self.resblocks[i * 3 + j]
                _, rb_out = rb._forward_br(br, recs, recs[x_base][0], x_base)
                if xs is None:
                    xs = rb_out
                else:
                    xs_t = br.copy(recs[xs][0])
                    _br_rec(br, recs, br.add_inplace(xs_t, recs[rb_out][0]),
                            "add", a_ref_idx=xs, b_ref_idx=rb_out)
                    xs = len(recs) - 1
            # 段末 ÷3（numpy 断链；语义 = numpy forward 的 mul_const(xs,1/3)）
            # T3-c：与 dec 整链图（GPU f32 mul_inplace × f32(1/3) 槽）位级
            # 一致 → 改 f32 乘（原 f64 断链 rel 2.4e-8 表示差会在训练混沌中
            # 指数分叉；f32 乘更贴近 PyTorch f32 训练语义，L2050）。
            br.commit()
            n_end = len(recs)
            seg = recs[seg_start:n_end]
            # T-H8：段值保持 BatchTensor（tape 记录 GPU 引用）——backward 时
            # conv bp 的 x 直接消费 GPU buffer（零上传）；需 numpy 的点（段间
            # ÷3、loss）经 __array__ 自动下载。
            arrs_seg = [t for t, _, _ in seg]
            _play_recs(tape, seg, arrs_seg, seg_start)
            xs_arr = arrs_seg[xs - seg_start]
            x_arr = np.asarray(xs_arr, np.float32) * np.float32(1.0 / 3.0)
            x32 = np.ascontiguousarray(x_arr, np.float32)
            tape.record_mul_const(x32, xs_arr, 1.0 / 3)
            if i == self.n_ups - 1:
                break
            x_ref = _br_rec(br, recs, br.copy(x32), "copy", x=x32)
            # 衔接 rec 属于下一批次的第一个 rec：seg_start 指向它（而非
            # len(recs)），否则下一段 play 时 record 的 x_ref/a_ref/b_ref
            # 指向它会产生 arrs 负索引错位（跨段引用 bug，backward 报
            # 「已有梯度形状不同」）。
            seg_start = len(recs) - 1
        # 段2（x32 = 最后一段 ÷3 输出；上传后 GPU 侧被 leaky 就地覆写，
        # record 的 out=x32 值不变）
        lr2 = _br_rec(br, recs, br.leaky_relu(x32, 0.01),
                      "leaky", x=x32, slope=0.01)
        post_w = np.asarray(self.conv_post_w, np.float32)
        # 输入必须用 lr2 的 tensor（leaky 已就地覆写上传 buffer；若重新
        # resolve numpy x32 会再上传原值 → GPU 算错 out，与 tape 的
        # x_ref=lr2 不一致 → y 前向错）。
        _br_rec(br, recs, br.conv1d(
            recs[lr2][0], post_w, None, padding=3),
            "conv1d", x_ref_idx=lr2, w=post_w,
            w_ref=self.conv_post_w, padding=3)
        br.commit()  # 段2
        n2 = len(recs)
        arrs2 = [t for t, _, _ in recs[seg_start:n2]]
        _play_recs(tape, recs[seg_start:n2], arrs2, seg_start)
        y = tape.tanh(arrs2[-1])
        if _CAPTURE_ON:
            _m1_cap().dec_frame_close()      # 门2：dec 前向段闭口（a26ag :3638|:3639）
        # T-H8：不 release（br 为链式共享）——tensor 存活到训练步末由
        # train.py 统一 _release_chain_br() 归还缓冲。
        return y


# ---------------------------------------------------------------------------
# SynthesizerTrnTrain
# ---------------------------------------------------------------------------
class SynthesizerTrnTrain:
    """训练版合成器：enc_p + enc_q + flow 正变换 + GeneratorNSF。

    ``forward`` 返回 ``(tape, y_hat, ids_slice, x_mask, y_mask, stats)``，
    ``stats = (z, z_p, m_p, logs_p, m_q, logs_q)``（对齐原版 forward）。
    """

    def __init__(self, weight_dict: dict, config=None):
        self.w = weight_dict
        self.cfg = VitsConfig(weight_dict, config)
        self.enc_p = TextEncoderTrain(weight_dict, self.cfg)
        self.enc_q = PosteriorEncoder(weight_dict, self.cfg)
        self.flow = ResidualCouplingBlockTrain(weight_dict, self.cfg)
        self.dec = GeneratorNSFTrain(weight_dict, self.cfg)

    def parameters(self):
        """返回 {参数名: ndarray}：参与训练的所有可导权重（含 enc_q）。"""
        params = {}
        for k, v in self.w.items():
            if (k.startswith("enc_p") or k.startswith("flow")
                    or k.startswith("dec") or k.startswith("enc_q")
                    or k == "emb_g.weight"):
                params[k] = v
        return params

    def forward(self, phone, pitch, pitchf, spec, spec_lengths, sid,
                segment_size, ns=None, randn=None, rng=None):
        """完整训练路径（对齐 SynthesizerTrnMs768NSFsid.forward）。

        phone: [1, P, D]；pitch: [1, P] int；pitchf: [1, P] f32；
        spec: [1, spec_ch, F]；spec_lengths: int；sid: [1] int；
        segment_size: **帧数**（= config.segment_size // hop_length）；
        rng: np.random.RandomState 或 None（None 用全局 np.random）。
        """
        rng = rng if rng is not None else np.random
        _B = int(np.asarray(spec_lengths).reshape(-1).shape[0])
        if ns is None:
            # SineGen 噪声长度与 dec 输入帧数（= segment_size 帧）一致
            ns = rng.standard_normal(
                (_B, int(segment_size) * self.cfg.upp, 1)).astype(np.float32)
        tape = AutogradTape()
        w = self.w
        g = tape.reshape(tape.embedding(sid, w["emb_g.weight"]),
                         (_B, -1, 1))
        x_mask = sequence_mask_np(phone.shape[1])
        tape.mark_const(x_mask)
        m_p, logs_p = self.enc_p.forward(tape, phone, pitch, x_mask)
        y_mask = sequence_mask_np(spec_lengths, spec.shape[-1])
        tape.mark_const(y_mask)
        z, m_q, logs_q = self.enc_q.forward(tape, spec, y_mask, g, randn=randn)
        z_p = self.flow.forward(tape, z, y_mask, g)
        _, ids_slice = rand_slice_segments_np(
            np.asarray(z), spec_lengths, segment_size, rng=rng)
        ids0 = np.asarray(ids_slice).reshape(-1).astype(np.int64)
        # F1（batch>1）：每样本独立起点切片（z [B,C,T]、pitchf [B,P]）
        z_parts, pf_parts = [], []
        for i in range(_B):
            zi = tape.slice(z, int(i), int(i + 1), axis=0)        # [1,C,T]
            zi = tape.slice(zi, int(ids0[i]), int(ids0[i]) + int(segment_size),
                            axis=-1)                             # [1,C,seg]
            z_parts.append(zi)
            pi = tape.slice(pitchf, int(i), int(i + 1), axis=0)  # [1,P]
            pi = tape.slice(pi, int(ids0[i]), int(ids0[i]) + int(segment_size),
                            axis=-1)                             # [1,seg]
            pf_parts.append(pi)
        z_slice = tape.concat(z_parts, axis=0) if _B > 1 else z_parts[0]
        pitchf_t = tape.concat(pf_parts, axis=0) if _B > 1 else pf_parts[0]
        if randn is not None:
            tape.mark_const(randn)
        y_hat = self.dec.forward(tape, z_slice, pitchf_t, g, ns)
        tape.mark_const(ns)
        return tape, y_hat, ids_slice, x_mask, y_mask, (
            np.asarray(z), z_p, m_p, logs_p, m_q, logs_q)


# ---------------------------------------------------------------------------
# 判别器
# ---------------------------------------------------------------------------
DISCRIMINATOR_S_CONVS = [
    (1, 16, 15, 1, 7, 1),
    (16, 64, 41, 4, 20, 4),
    (64, 256, 41, 4, 20, 16),
    (256, 1024, 41, 4, 20, 64),
    (1024, 1024, 41, 4, 20, 256),
    (1024, 1024, 5, 1, 2, 1),
]
DISCRIMINATOR_P_K = 5
DISCRIMINATOR_P_S = 3
DISCRIMINATOR_P_PAD = 2


class DiscriminatorS:
    def __init__(self, w: dict, base: str):
        self.convs = []
        for j, (_, o, k, s, p, gr) in enumerate(DISCRIMINATOR_S_CONVS):
            self.convs.append((np.asarray(w[f"{base}.convs.{j}.weight"]),
                               np.asarray(w[f"{base}.convs.{j}.bias"]),
                               s, p, gr))
        self.post_w = np.asarray(w[f"{base}.conv_post.weight"])
        self.post_b = np.asarray(w[f"{base}.conv_post.bias"])

    def forward(self, tape, x):
        fmap = []
        if tape is None:
            for (cwt, cb, s, p, gr) in self.convs:
                x = _conv1d_groups_np(x, cwt, cb, gr, stride=s, padding=p)
                x = np.where(x >= 0, x, LRELU_SLOPE * x)
                fmap.append(x)
            x = _conv1d_np(x, self.post_w, self.post_b, padding=1)
            fmap.append(x)
            return x, fmap
        if os.environ.get("RVC_TRAIN_GRAPH", "0") == "1":
            # T3-a：判别器整链图执行器化（RVC_TRAIN_GRAPH=1，默认 0 零回归）。
            # 中间槽位 buffer 由 GraphRunner 双缓冲管理（A/B 交替，D/G 步
            # real/fake 互不覆盖）；输出包装成 _chain_br 的 BatchTensor，
            # tape.record_* 沿用（J19 id 链匹配，backward 零上传消费槽位）。
            return self.forward_graph(tape, x)
        if os.environ.get("RVC_TRAIN_BR_FWD", "0") == "1":
            # 阶段J（J5）：S 整链 BatchRunner 化（op23 conv1d_groups fwd）。
            return self.forward_br(tape, x)
        for (cwt, cb, s, p, gr) in self.convs:
            x = tape.conv1d_groups(x, cwt, cb, gr, stride=s, padding=p)
            x = tape.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = tape.conv1d(x, self.post_w, self.post_b, padding=1)
        fmap.append(x)
        return x, fmap

    def forward_graph(self, tape, x):
        """S 的图执行器前向（RVC_TRAIN_GRAPH=1）。结构与 forward_br 一致，
        但整链一次 graph_run（Python 簿记下沉 Zig），输出为图执行器槽位
        包装的 BatchTensor（_runner=chain br），tape 记录沿用（J19）。"""
        from runtime.vulkan_ops import BatchTensor  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        gr = _graph_runner()
        x_np = np.asarray(x)
        B = int(x_np.shape[0])
        T = int(x_np.shape[-1])
        key = ("S", id(self), B, T)

        def build():
            return gr._build_s(self.convs, self.post_w, self.post_b, T, B=B)

        pair = gr.disc_graphs(key, build)  # [A, B] 双缓冲
        toggle = getattr(self, "_g_toggle", 0)
        self._g_toggle = 1 - toggle
        dg = pair[toggle]
        if _os.environ.get("RVC_TRAIN_GRAPH_ASYNC", "0") == "1":
            # R-T4：异步提交（不等待，在途队列；MPD.forward 末尾统一 wait）。
            pairs_bt, post_bt = dg.run_async(x_np, runner=_chain_br())
        else:
            pairs_bt, post_bt = dg.run(x_np, runner=_chain_br())
        graph_bwd = _os.environ.get("RVC_TRAIN_GRAPH_BWD", "0") == "1"
        if not graph_bwd:
            return self._record_graph_fwd(tape, pairs_bt, post_bt, x_np)
        return self._record_graph_bwd(tape, dg, pairs_bt, post_bt, x_np)

    def _record_graph_fwd(self, tape, pairs_bt, post_bt, x_np):
        """T3-a 逐层 record（backward 走原逐层 GPU bp 路径）。"""
        fmap = []
        h_in = x_np
        for i, (cwt, cb, s, p, gr) in enumerate(self.convs):
            tape.record_conv1d_groups(
                pairs_bt[i][0], h_in, cwt, cb, gr, stride=s, padding=p)
            h_l = tape.record_leaky_relu(
                pairs_bt[i][1], pairs_bt[i][0], LRELU_SLOPE)
            fmap.append(h_l)
            h_in = pairs_bt[i][1]
        h_post = tape.record_conv1d(
            post_bt, h_in, self.post_w, self.post_b, padding=1)
        fmap.append(h_post)
        return h_post, fmap

    def _record_graph_bwd(self, tape, dg, pairs_bt, post_bt, x_np):
        """T3-b：backward 整链图执行器化（RVC_TRAIN_GRAPH_BWD=1）。

        图段一次 run 产出全部梯度（go 种子 + FM 种子输入槽，消费 fwd 槽位
        与 wpers 权重）；tape 逐层仍注册（J19：out 与下游 _x 同一 BatchTensor
        对象），bp 改为从 dg.bwd_out 槽位查表返回（不重算）。
        """
        fmap = []
        h_in = x_np
        chain = _chain_br()
        # 每层注册图查表 bp（闭包捕获 ref 对象，保 J19 id 链）
        for i, (cwt, cb, s, p, gr) in enumerate(self.convs):
            _xr = h_in
            _wr = np.asarray(cwt)
            _br = np.asarray(cb)
            _gx_shape = _xr.shape                 # gx_{i-1} 形状（= conv 输入）
            _gw_shape = _wr.shape
            _gb_shape = _br.shape
            _dgi = dg
            _key_i = i

            def _conv_bp(go, _dg=_dgi, _xr=_xr, _wr=_wr, _br=_br,
                         _gx_shape=_gx_shape, _gw_shape=_gw_shape,
                         _gb_shape=_gb_shape, _i=_key_i, _chain=chain):
                if _i == 0:
                    gx = _dg.bwd_take("gxin", _gx_shape, _chain)
                else:
                    gx = _dg.bwd_take(f"gx{_i - 1}", _gx_shape, _chain)
                gw = _dg.bwd_take(f"gw{_i}", _gw_shape, _chain)
                gb = _dg.bwd_take(f"gb{_i}", _gb_shape, _chain)
                return [(_xr, gx), (_wr, gw), (_br, gb)]

            tape._push(_conv_bp, pairs_bt[i][0])
            _lr = pairs_bt[i][0]
            _gy_shape = pairs_bt[i][1].shape
            _dgi = dg
            _key_i = i

            def _leaky_bp(go, _dg=_dgi, _lr=_lr, _gy_shape=_gy_shape,
                          _i=_key_i, _chain=chain):
                gy = _dg.bwd_take(f"gy{_i}", _gy_shape, _chain)
                return [(_lr, gy)]

            h_l = tape._push(_leaky_bp, pairs_bt[i][1])
            fmap.append(h_l)
            h_in = pairs_bt[i][1]
        # post conv1d：bp 触发整链 bwd_run（go + FM 种子），再查表返回
        _post_wr = np.asarray(self.post_w)
        _post_br = np.asarray(self.post_b)
        _gx_post_shape = h_in.shape          # 到 leaky6 输出梯度形状
        _gw_post_shape = _post_wr.shape
        _gb_post_shape = _post_br.shape
        _dgi = dg
        _chain = chain
        _grads = tape.grads
        _fm_refs = [pairs_bt[i][1] for i in range(6)]

        def _post_bp(go, _dg=_dgi, _xr=h_in, _wr=_post_wr, _br=_post_br,
                     _gx_shape=_gx_post_shape, _gw_shape=_gw_post_shape,
                     _gb_shape=_gb_post_shape, _chain=_chain,
                     _grads=_grads, _fm_refs=_fm_refs):
            # FM 种子：G 步注入 id(fmap_i)（fake 链 leaky 输出梯度）
            fm = {}
            for _i, _fref in enumerate(_fm_refs):
                _v = _grads.get(id(_fref))
                if _v is not None:
                    fm[_i] = np.asarray(_v)
            _dg.bwd_run(go, fm)
            if os.environ.get("RVC_TRAIN_STEP_PROFILE"):  # TEMP-DBG
                _DBG_DISC_BWD_MS.append(_dg._last_bwd_ms)
            gx = _dg.bwd_take("gx_post", _gx_shape, _chain)
            gw = _dg.bwd_take("gw_post", _gw_shape, _chain)
            gb = _dg.bwd_take("gb_post", _gb_shape, _chain)
            return [(_xr, gx), (_wr, gw), (_br, gb)]

        h_post = tape._push(_post_bp, post_bt)
        fmap.append(h_post)
        return h_post, fmap

    def forward_br(self, tape, x):
        """S 的 BatchRunner 化前向（RVC_TRAIN_BR_FWD=1，op23 整链录制）。

        模式同 DiscriminatorP.forward 的 BR_FWD 分支：整链录制一次 commit
        （conv1d_groups+leaky×6，post conv1d×1），逐层下载 numpy 填 tape
        （bp 走 record_conv1d_groups/record_leaky_relu/record_conv1d）。
        """
        from runtime.vulkan_ops import (  # noqa: PLC0415
            BatchRunner, get_context, wpers_get)
        # J19b：统一用全局链式 br——record 的 out/x 均为 BatchTensor 时
        # bp 链（cg_bwd→leaky→conv1d_groups）与 fwd 的 pairs 必须同属
        # 一个 BatchRunner，否则 _resolve_input 拒绝混用。独立 br 只适合
        # J15 的 numpy-out（断链）方案；chain br 生命周期由
        # _release_chain_br（train_step 末）管理。
        br = _chain_br()
        try:
            h = np.asarray(x)
            pairs = []
            for (cwt, cb, s, p, gr) in self.convs:
                # J17：fwd 权重走 wpers 常驻 buffer（判别器权重数组对象
                # 稳定，id 命中缓存——省 P 权重每步 ~1GB 重复上传）。
                cbt = br.conv1d_groups(
                    h, cwt, cb, stride=s, padding=p,
                    buf_w=wpers_get(cwt, br, "d"))
                lbt = br.leaky_relu(cbt, LRELU_SLOPE)
                pairs.append((cbt, lbt))
                h = lbt
            cbt = br.conv1d(h, self.post_w, self.post_b, padding=1,
                            buf_w=wpers_get(self.post_w, br, "d"))
            pairs.append((cbt, None))
            br.commit()
            # J15/J19：tape 记录阶段 x 与 out 均为 BatchTensor（bp 零重传
            # 且 id 链匹配；下载只在 loss/score 消费点）。br 所有权归全局
            # chain（不挂 tape._brs）。
            fmap = []
            h_in = np.asarray(x)  # 第一层 conv 输入（小）
            for i, (cwt, cb, s, p, gr) in enumerate(self.convs):
                # J19：out 必须传该算子输出的 BatchTensor（与下游引用的
                # 同一对象）——tape.backward 按 id(out) 取 go、按 id(_x)
                # 存梯度，out 与 _x 同为 BatchTensor 引用时链才匹配。
                # 原实现 out 传 numpy（下载值）→ id 与 _x(BatchTensor)
                # 不匹配 → bp 断链（判别器/可导生成器梯度全零，训练静默
                # 不学——J15-J18 的隐蔽回归，loss 有限但 d/g 平稳）。
                tape.record_conv1d_groups(
                    pairs[i][0], h_in, cwt, cb, gr, stride=s, padding=p)
                h_l = tape.record_leaky_relu(
                    pairs[i][1], pairs[i][0], LRELU_SLOPE)
                fmap.append(h_l)
                h_in = pairs[i][1]  # 本层 leaky 输出 = 下一层 conv 输入
            h_post = tape.record_conv1d(
                pairs[-1][0], h_in, self.post_w, self.post_b, padding=1)
            fmap.append(h_post)
            return h_post, fmap
        finally:
            pass  # J19b：chain br 由 _release_chain_br（train_step 末）管理


class DiscriminatorP:
    def __init__(self, w: dict, base: str, period: int):
        self.period = period
        self.convs = []
        for j in range(5):
            self.convs.append((np.asarray(w[f"{base}.convs.{j}.weight"]),
                               np.asarray(w[f"{base}.convs.{j}.bias"])))
        self.post_w = np.asarray(w[f"{base}.conv_post.weight"])
        self.post_b = np.asarray(w[f"{base}.conv_post.bias"])

    def forward(self, tape, x):
        fmap = []
        b, c, t = x.shape
        p = self.period
        if t % p != 0:
            n_pad = p - (t % p)
            if tape is None:
                x = np.pad(x, ((0, 0), (0, 0), (0, n_pad)), mode="reflect")
            else:
                x = tape.pad_reflect(x, 0, n_pad, axis=-1)  # 对齐原版 F.pad reflect
            t += n_pad
        if tape is None:
            x = x.reshape(b, c, t // p, p)
            for (cwt, cb) in self.convs:
                x = _conv2d_np(x, cwt, cb, stride=(DISCRIMINATOR_P_S, 1),
                               padding=(DISCRIMINATOR_P_PAD, 0))
                x = np.where(x >= 0, x, LRELU_SLOPE * x)
                fmap.append(x)
            x = _conv2d_np(x, self.post_w, self.post_b, padding=(1, 0))
            fmap.append(x)
            return x, fmap
        x = tape.reshape(x, (b, c, t // p, p))
        if os.environ.get("RVC_TRAIN_GRAPH", "0") == "1":
            # T3-a：P 整链图执行器化（conv2d op27 链 + leaky 就地）。
            # reshape 仍在 tape（record_reshape），图从 reshape 后输入开始。
            return self.forward_graph(tape, x)
        if os.environ.get("RVC_TRAIN_BR_FWD", "0") == "1":
            # 阶段A（BatchRunner 化）：整链录制一次 commit（省逐算子
            # submit/wait 包络）；中间值 commit 后逐层下载填 tape（bp 复用
            # record_* 纯记录方法，forward 不重算）。GPU f32 vs numpy f64
            # 差 ~1e-7 rel（容差内）；RVC_TRAIN_BR_FWD=0 回原逐算子路径。
            from runtime.vulkan_ops import (  # noqa: PLC0415
            BatchRunner, get_context, wpers_get)
            # J19b：统一全局 chain br（同 S.forward_br；bp 链跨算子共用）
            br = _chain_br()
            try:
                h = np.asarray(x)
                pairs = []  # (conv_bt, leaky_bt|None)
                for (cwt, cb) in self.convs:
                    # J17：fwd 权重走 wpers 常驻（省 P 权重每步 ~1GB 重传）
                    cbt = br.conv2d(
                        h, cwt, cb, stride=(DISCRIMINATOR_P_S, 1),
                        padding=(DISCRIMINATOR_P_PAD, 0),
                        buf_w=wpers_get(cwt, br, "d"))
                    lbt = br.leaky_relu(cbt, LRELU_SLOPE)
                    pairs.append((cbt, lbt))
                    h = lbt
                cbt = br.conv2d(h, self.post_w, self.post_b, padding=(1, 0),
                                buf_w=wpers_get(self.post_w, br, "d"))
                pairs.append((cbt, None))
                br.commit()
                # J19：x/out 均 BatchTensor（bp 零重传 + id 链匹配）
                fmap = []
                h_in = np.asarray(x)  # reshape 后的第一层 conv 输入
                for i, (cwt, cb) in enumerate(self.convs):
                    # J19：out 传 BatchTensor（链 id 匹配，同 S.forward_br）
                    tape.record_conv2d(
                        pairs[i][0], h_in, cwt, cb,
                        stride=(DISCRIMINATOR_P_S, 1),
                        padding=(DISCRIMINATOR_P_PAD, 0))
                    h_l = tape.record_leaky_relu(
                        pairs[i][1], pairs[i][0], LRELU_SLOPE)
                    fmap.append(h_l)
                    h_in = pairs[i][1]
                h_post = tape.record_conv2d(
                    pairs[-1][0], h_in, self.post_w, self.post_b,
                    padding=(1, 0))
                fmap.append(h_post)
                return h_post, fmap
            finally:
                pass  # J19b：chain br 由 _release_chain_br（train_step 末）管理
        for (cwt, cb) in self.convs:
            x = tape.conv2d(x, cwt, cb, stride=(DISCRIMINATOR_P_S, 1),
                            padding=(DISCRIMINATOR_P_PAD, 0))
            x = tape.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = tape.conv2d(x, self.post_w, self.post_b, padding=(1, 0))
        fmap.append(x)
        return x, fmap

    def forward_graph(self, tape, x):
        """P 的图执行器前向（RVC_TRAIN_GRAPH=1）。reshape 已在 tape 记录
        （调用方做），图从 reshape 后输入 [B,1,H,W] 开始；结构同 BR_FWD
        分支，但整链一次 graph_run，输出为槽位 BatchTensor（J19 链匹配）。
        RVC_TRAIN_GRAPH_BWD=1 时 backward 走整链图（_record_graph_bwd）。"""
        gr = _graph_runner()
        x_np = np.asarray(x)
        b, c, h, w = x_np.shape
        key = ("P", id(self), b, h, w)

        def build():
            return gr._build_p(self.convs, self.post_w, self.post_b, h, w, B=b)

        pair = gr.disc_graphs(key, build)  # [A, B] 双缓冲
        toggle = getattr(self, "_g_toggle", 0)
        self._g_toggle = 1 - toggle
        dg = pair[toggle]
        if os.environ.get("RVC_TRAIN_GRAPH_ASYNC", "0") == "1":
            # R-T4：异步提交（在途队列；MPD.forward 末尾统一 wait）。
            pairs_bt, post_bt = dg.run_async(x_np, runner=_chain_br())
        else:
            pairs_bt, post_bt = dg.run(x_np, runner=_chain_br())
        graph_bwd = os.environ.get("RVC_TRAIN_GRAPH_BWD", "0") == "1"
        if graph_bwd:
            return self._record_graph_bwd(tape, dg, pairs_bt, post_bt, x_np)
        fmap = []
        h_in = x_np
        for i, (cwt, cb) in enumerate(self.convs):
            tape.record_conv2d(
                pairs_bt[i][0], h_in, cwt, cb,
                stride=(DISCRIMINATOR_P_S, 1),
                padding=(DISCRIMINATOR_P_PAD, 0))
            h_l = tape.record_leaky_relu(
                pairs_bt[i][1], pairs_bt[i][0], LRELU_SLOPE)
            fmap.append(h_l)
            h_in = pairs_bt[i][1]
        h_post = tape.record_conv2d(
            post_bt, h_in, self.post_w, self.post_b, padding=(1, 0))
        fmap.append(h_post)
        return h_post, fmap

    def _record_graph_bwd(self, tape, dg, pairs_bt, post_bt, x_np):
        """T3-c 后续：P 链 backward 整链图执行器化（RVC_TRAIN_GRAPH_BWD=1）。

        图段一次 run 产出全部梯度（go 种子 + FM 种子输入槽，消费 fwd 槽位
        与 wpers 权重）；tape 逐层仍注册（J19：out 与下游 _x 同一 BatchTensor
        对象），bp 改为从 dg.bwd_out 槽位查表返回（不重算）。结构与
        DiscriminatorS._record_graph_bwd 对称；conv2d_bwd 每层 5 算子图节点
        见 graph_runner._build_p_bwd。
        """
        fmap = []
        h_in = x_np
        chain = _chain_br()
        for i, (cwt, cb) in enumerate(self.convs):
            _xr = h_in
            _wr = np.asarray(cwt)
            _br = np.asarray(cb)
            _gx_shape = _xr.shape                 # gx_{i-1} 形状（= conv 输入）
            _gw_shape = _wr.shape
            _gb_shape = _br.shape
            _dgi = dg
            _key_i = i

            def _conv_bp(go, _dg=_dgi, _xr=_xr, _wr=_wr, _br=_br,
                         _gx_shape=_gx_shape, _gw_shape=_gw_shape,
                         _gb_shape=_gb_shape, _i=_key_i, _chain=chain):
                if _i == 0:
                    gx = _dg.bwd_take("gxin", _gx_shape, _chain)
                else:
                    gx = _dg.bwd_take(f"gx{_i - 1}", _gx_shape, _chain)
                gw = _dg.bwd_take(f"gw{_i}", _gw_shape, _chain)
                gb = _dg.bwd_take(f"gb{_i}", _gb_shape, _chain)
                return [(_xr, gx), (_wr, gw), (_br, gb)]

            tape._push(_conv_bp, pairs_bt[i][0])
            _lr = pairs_bt[i][0]
            _gy_shape = pairs_bt[i][1].shape
            _dgi = dg
            _key_i = i

            def _leaky_bp(go, _dg=_dgi, _lr=_lr, _gy_shape=_gy_shape,
                          _i=_key_i, _chain=chain):
                gy = _dg.bwd_take(f"gy{_i}", _gy_shape, _chain)
                return [(_lr, gy)]

            h_l = tape._push(_leaky_bp, pairs_bt[i][1])
            fmap.append(h_l)
            h_in = pairs_bt[i][1]
        # post conv2d：bp 触发整链 bwd_run（go + FM 种子），再查表返回
        _post_wr = np.asarray(self.post_w)
        _post_br = np.asarray(self.post_b)
        _gx_post_shape = h_in.shape          # 到 leaky5 输出梯度形状
        _gw_post_shape = _post_wr.shape
        _gb_post_shape = _post_br.shape
        _dgi = dg
        _chain = chain
        _grads = tape.grads
        _fm_refs = [pairs_bt[i][1] for i in range(5)]

        def _post_bp(go, _dg=_dgi, _xr=h_in, _wr=_post_wr, _br=_post_br,
                     _gx_shape=_gx_post_shape, _gw_shape=_gw_post_shape,
                     _gb_shape=_gb_post_shape, _chain=_chain,
                     _grads=_grads, _fm_refs=_fm_refs):
            # FM 种子：G 步注入 id(fmap_i)（fake 链 leaky 输出梯度）
            fm = {}
            for _i, _fref in enumerate(_fm_refs):
                _v = _grads.get(id(_fref))
                if _v is not None:
                    fm[_i] = np.asarray(_v)
            _dg.bwd_run(go, fm)
            if os.environ.get("RVC_TRAIN_STEP_PROFILE"):  # TEMP-DBG
                _DBG_DISC_BWD_MS.append(_dg._last_bwd_ms)
            gx = _dg.bwd_take("gx_post", _gx_shape, _chain)
            gw = _dg.bwd_take("gw_post", _gw_shape, _chain)
            gb = _dg.bwd_take("gb_post", _gb_shape, _chain)
            return [(_xr, gx), (_wr, gw), (_br, gb)]

        h_post = tape._push(_post_bp, post_bt)
        fmap.append(h_post)
        return h_post, fmap


def _conv1d_groups_np(x, w, b, groups, stride=1, padding=0):
    """groups conv1d 的纯 numpy 前向。"""
    x = np.asarray(x)
    w = np.asarray(w)
    B, C, T = x.shape
    O, Ci, K = w.shape
    outs = []
    for g in range(groups):
        xg = x[:, g * (C // groups): (g + 1) * (C // groups), :]
        wg = w[g * (O // groups): (g + 1) * (O // groups), :, :]
        bg = b[g * (O // groups): (g + 1) * (O // groups)]
        outs.append(_conv1d_np(xg, wg, bg, stride=stride, padding=padding))
    return np.concatenate(outs, axis=1)


class MultiPeriodDiscriminator:
    """V1: DiscriminatorS + periods [2,3,5,7,11,17]
    V2: periods [2,3,5,7,11,17,23,37]（与 f0D48k.pth 一致）。"""

    def __init__(self, weight_dict: dict, version="v2"):
        w = weight_dict
        if "discriminators.0.convs.0.weight" not in w:
            w = w.get("model", w)
        self.w = w
        periods = ([2, 3, 5, 7, 11, 17, 23, 37] if version == "v2"
                   else [2, 3, 5, 7, 11, 17])
        self.discs = [DiscriminatorS(w, "discriminators.0")]
        for i, per in enumerate(periods):
            self.discs.append(
                DiscriminatorP(w, f"discriminators.{i + 1}", per))

    def parameters(self):
        """返回 {参数名: ndarray}：判别器全部可导权重。"""
        w = self.w
        return {k: v for k, v in w.items()
                if k.startswith("discriminators.")}

    def forward(self, tape, x):
        """x: [B, 1, T]；返回 (scores_list, fmaps_list)。"""
        scores, fmaps = [], []
        for d in self.discs:
            s, fm = d.forward(tape, x)
            scores.append(s)
            fmaps.append(fm)
        if os.environ.get("RVC_TRAIN_GRAPH_ASYNC", "0") == "1":
            # R-T4：S+8×P 全部异步在途后统一 wait（9 图流水线，GPU 连续
            # 执行；wait 后才允许 host 下载 score/fmap）。
            _graph_runner().wait_all()
        return scores, fmaps


def discriminator_flatten(tape, x):
    return tape.reshape(x, (x.shape[0], -1))