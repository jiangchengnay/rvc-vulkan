# -*- coding: utf-8 -*-
"""process_ckpt 测试（T48）：savee / extract_small_model / merge + 端到端推理验证。

覆盖：
1. ``extract_small_model``：assets/pretrained_v2/f0G48k.pth（底模）→ 推理 .pth，
   用 ``runtime.vc.VC.get_vc`` + ``vc_single`` 实际合成小段音频（pm 音高）。
2. ``merge``：f0G48k.pth 与其自身 alpha=0.3 融合 → 输出可加载可合成。
3. ``savee``：从底模构造训练权重形态（weight_norm 还原 + 保留 enc_q 的
   ``G_{step}.npz``）→ 推理 .pth → 可加载，且数值与底模提取版一致（同一底模
   权重往返应几乎不变）。
4. weight_norm 重参数化往返、config 推断、speaker_info 规整等单元断言。

**本测试禁止 import torch**；运行：``python runtime/train/tests_process_ckpt.py``。
"""

from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch_compat  # noqa: PLC0415  # 项目根在 sys.path
from runtime.models.vits import VitsConfig, SynthesizerTrn  # noqa: PLC0415
from runtime.train.process_ckpt import (  # noqa: PLC0415
    savee,
    extract_small_model,
    merge,
    load_checkpoint_dict,
    reparam_weight_norm,
    deparam_weight_norm,
    infer_config,
    normalize_speaker_info,
)

PRETRAINED_G = os.path.join("assets", "pretrained_v2", "f0G48k.pth")
OUT_DIR = os.path.join("assets", "weights")


def _make_tone_wav(path, sr=48000, seconds=0.8, freq=220.0):
    """生成 int16 正弦波测试音频。"""
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    x16 = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())
    return path


def _build_fake_train_npz(path):
    """从底模构造"训练权重形态"：weight_norm 还原为普通权重 + 保留 enc_q。"""
    from runtime.models.vits_train import _deweight_dict  # noqa: PLC0415

    cpt = torch_compat.load_pth(PRETRAINED_G)
    w = _deweight_dict(cpt["model"])  # {X.weight} 普通权重，含 enc_q.*
    meta = json.dumps({"config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                                   [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                                   [12, 10, 2, 2], 512, [24, 20, 4, 4],
                                   109, 256, 48000],
                       "step": 100, "version": "v2"})
    np.savez(path, **{k: np.asarray(v) for k, v in w.items()},
             **{"_meta": np.asarray(meta)})
    return path


def _verify_infer_pth(path, expect_sr=48000, expect_nospk=109):
    """加载推理 .pth：torch_compat 可读、无 enc_q、有 weight_v/g、config 正确、
    vits.SynthesizerTrn 可构造（结构校验）。"""
    cpt = torch_compat.load_pth(path)
    w = cpt["weight"]
    assert "weight" in cpt and isinstance(w, dict), "缺 weight 键"
    assert not any("enc_q" in k for k in w), "推理权重不应含 enc_q"
    assert any(k.endswith(".weight_v") for k in w), "缺 weight_v"
    assert any(k.endswith(".weight_g") for k in w), "缺 weight_g"
    assert "emb_g.weight" in w, "缺 emb_g.weight"
    assert cpt["config"][-1] == expect_sr, \
        f"config sr={cpt['config'][-1]} != {expect_sr}"
    assert cpt["config"][15] == expect_nospk, "n_spk 推断错误"
    assert cpt["version"] in ("v1", "v2")
    cfg = VitsConfig(w, cpt["config"])
    assert cfg.upp in (480, 400, 320), f"upp={cfg.upp} 异常"
    # SynthesizerTrn 构造即验证全部 weight_v/weight_g 键齐备且形状正确
    syn = SynthesizerTrn(path)
    assert syn.cfg.n_spk == expect_nospk
    return cpt, syn


def test_extract_and_vc():
    """extract_small_model：底模 -> 推理 .pth -> vc.get_vc + vc_single 合成。"""
    out = os.path.join(OUT_DIR, "extract_test.pth")
    p = extract_small_model(PRETRAINED_G, out, 48000, 1, "v2",
                            info="从底模提取的测试模型")
    print(f"[extract] 输出 {p}")
    cpt, syn = _verify_infer_pth(p)
    assert cpt["info"] == "从底模提取的测试模型"
    assert cpt["f0"] == 1

    from runtime.native_config import Config
    from runtime.vc import VC

    wav = _make_tone_wav(os.path.join(OUT_DIR, "_probe.wav"))
    vc = VC(Config())
    info = vc.get_vc("extract_test")
    assert info["success"], f"get_vc 失败: {info.get('error')}"
    print(f"[extract] get_vc: n_spk={info['n_spk']} tgt_sr={info['tgt_sr']} "
          f"version={info['version']} if_f0={info['if_f0']}")
    status, (sr_out, audio) = vc.vc_single(
        0, wav, 0, "pm", None, 0.0, 48000, 1.0, 0.33)
    assert status == "转换成功", f"vc_single 失败: {status}"
    assert sr_out == 48000
    assert isinstance(audio, np.ndarray) and audio.size > 1000
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    print(f"[extract] vc_single OK: {audio.size} 样本, rms={rms:.4f}")
    assert np.isfinite(rms) and rms > 0, "输出音频异常（静音/非有限值）"


def test_merge_self():
    """merge：f0G48k.pth 与其自身 alpha=0.3 -> 可加载可合成。"""
    out = os.path.join(OUT_DIR, "merge_test.pth")
    p = merge(PRETRAINED_G, PRETRAINED_G, 0.3, out,
              sr=48000, if_f0=1, version="v2", info="自融合测试")
    print(f"[merge] 输出 {p}")
    cpt, syn = _verify_infer_pth(p)
    # 自融合结果应与原模型几乎一致
    w_ref = torch_compat.load_pth(PRETRAINED_G)["model"]
    key = "enc_p.emb_phone.weight"
    diff = np.abs(np.asarray(cpt["weight"][key]).astype(np.float64)
                  - np.asarray(w_ref[key]).astype(np.float64)).max()
    print(f"[merge] emb_phone 最大偏差 {diff:.2e}（自融合应接近 0）")
    assert diff < 0.01, "自融合结果偏差过大"

    from runtime.native_config import Config
    from runtime.vc import VC

    wav = os.path.join(OUT_DIR, "_probe.wav")
    vc = VC(Config())
    info = vc.get_vc("merge_test")
    assert info["success"], f"get_vc 失败: {info.get('error')}"
    status, (sr_out, audio) = vc.vc_single(
        0, wav, 0, "pm", None, 0.0, 48000, 1.0, 0.33)
    assert status == "转换成功"
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    print(f"[merge] vc_single OK: {audio.size} 样本, rms={rms:.4f}")
    assert np.isfinite(rms) and rms > 0


def test_savee():
    """savee：训练权重形态 npz -> 推理 .pth；数值与底模提取版一致。"""
    npz = os.path.join(OUT_DIR, "_G_100.npz")
    _build_fake_train_npz(npz)
    out = os.path.join(OUT_DIR, "savee_test.pth")
    p = savee(npz, "savee_test", 48000, 1, "v2", out_root=OUT_DIR,
              info="训练权重测试", speaker_info=[{"id": 0, "name": "speaker0"}],
              dtype="float32")
    print(f"[savee] 输出 {p}")
    cpt, syn = _verify_infer_pth(p)
    assert cpt["speaker_info"] == [{"id": 0, "name": "speaker0"}]

    # 数值一致性：savee(训练权重) vs extract(底模) 应得到相同权重（同一底模往返）
    w_extract = torch_compat.load_pth(os.path.join(OUT_DIR, "extract_test.pth"))["weight"]
    w_savee = cpt["weight"]
    keys = [k for k in w_savee if k.endswith(".weight_v")][:3]
    for k in keys:
        d = float(np.abs(np.asarray(w_savee[k]).astype(np.float64)
                         - np.asarray(w_extract[k]).astype(np.float64)).max())
        print(f"[savee] 键 {k} 与底模提取版偏差 {d:.2e}")
        assert d < 1e-3, f"{k} 偏差过大 {d}"
    # config 也应一致
    c_extract = torch_compat.load_pth(os.path.join(OUT_DIR, "extract_test.pth"))["config"]
    assert cpt["config"] == c_extract, "savee 与 extract 的 config 不一致"


def test_unit():
    """单元断言：重参数化往返 / config 推断 / speaker_info。"""
    rng = np.random.RandomState(7)
    w = load_checkpoint_dict(PRETRAINED_G)
    # 1) weight_norm 往返（对每个 wn 层）
    wn_keys = [k for k in w if k.endswith(".weight")
               and any(s in k for s in ("flow.flows.", "dec.ups.",
                                        "dec.resblocks."))]
    for k in wn_keys[:5]:
        W = np.asarray(w[k])
        rw = reparam_weight_norm({k: W})
        back = deparam_weight_norm(rw)[k]
        err = float(np.abs(back - W).max())
        assert err < 1e-5, f"{k} 往返误差 {err}"
    print(f"[unit] weight_norm 往返 PASS（抽查 {min(5, len(wn_keys))} 个 wn 层）")
    # 2) config 推断（48k v2 布局）
    cfg = infer_config(w, 48000)
    assert cfg[0] == 1025 and cfg[12] == [12, 10, 2, 2] \
        and cfg[14] == [24, 20, 4, 4] and cfg[15] == 109 and cfg[17] == 48000
    assert int(np.prod(cfg[12])) == 480
    print("[unit] infer_config PASS:", cfg[12], cfg[14], "n_spk=", cfg[15])
    # 3) speaker_info 规整
    spk = normalize_speaker_info(
        [{"id": 2, "name": "b"}, {"id": 1, "name": "a"}, {"id": 2, "name": "dup"},
         {"id": -1, "name": "bad"}, {"id": 200, "name": "bad2"}, {"name": "noid"}])
    assert spk == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    print("[unit] normalize_speaker_info PASS")


def main():
    ok = True
    for fn in (test_unit, test_extract_and_vc, test_merge_self, test_savee):
        print("\n=== %s ===" % fn.__name__)
        try:
            fn()
            print("  PASS")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            ok = False
            print("  FAIL: %s" % (exc,))
    print("\nprocess_ckpt tests %s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
