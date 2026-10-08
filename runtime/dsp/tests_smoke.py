"""汇总冒烟测试：运行 fft / resample / mel / f0 / audio_io 的全部自测并汇总。

用法：
    python -m runtime.dsp.tests_smoke      （在项目根目录）
    python runtime/dsp/tests_smoke.py
"""

from __future__ import annotations

import sys
import time


def run_all() -> bool:
    """运行各模块 self_test，返回是否全部通过。"""
    results: list[tuple[str, bool, float]] = []
    modules = [
        ("fft", "runtime.dsp.fft"),
        ("resample", "runtime.dsp.resample"),
        ("mel", "runtime.dsp.mel"),
        ("f0", "runtime.dsp.f0"),
        ("audio_io", "runtime.dsp.audio_io"),
        ("utils", "runtime.dsp.utils"),
    ]
    for name, mod_path in modules:
        mod = __import__(mod_path, fromlist=["_self_test"])
        t0 = time.time()
        try:
            passed = bool(mod._self_test())
        except Exception as e:  # noqa: BLE001 - 冒烟测试需要捕获所有异常
            print(f"  !! {name} 自测异常: {type(e).__name__}: {e}")
            passed = False
        dt = time.time() - t0
        results.append((name, passed, dt))
        print(f"  [{name}] {'PASS' if passed else 'FAIL'} ({dt:.2f}s)")

    print("\n" + "=" * 60)
    all_ok = all(p for _, p, _ in results)
    for name, passed, dt in results:
        print(f"  {name:<10} {'PASS' if passed else 'FAIL'}  {dt:.2f}s")
    print("=" * 60)
    print(f"总体结果: {'ALL PASS' if all_ok else 'HAS FAILURE'}")
    return all_ok


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)