"""ChunkExecutor 抽象的 CPU smoke test (不需要仿真器 / GPU).

验证:
    1. reconstruct_chunk 的 v>1 / v<1 / v=1 语义与形状.
    2. 与 RoboTwin 原版 chunk_accel.reconstruct_chunk 数值一致 (若可 import).
    3. ChunkExecutor.compress / capabilities 归一 / ChunkExecResult 流转.
"""
import os
import sys

import numpy as np

# 把 RLinf 仓根目录加进 path, 以便 `import speedtune.execution`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from speedtune.execution import (  # noqa: E402
    ChunkExecResult,
    ChunkExecutor,
    ExecutorCapabilities,
    SpeedAction,
    reconstruct_chunk,
    result_from_speedup_info,
)
from speedtune.execution.base import (  # noqa: E402
    STATUS_RETIME_FAILED,
    STATUS_SUCCESS,
)


def test_reconstruct_semantics():
    chunk = np.arange(50 * 14, dtype=np.float64).reshape(50, 14)
    # v = 1 -> identity
    assert np.allclose(reconstruct_chunk(chunk, 1.0), chunk)
    # v > 1 -> 变短
    out = reconstruct_chunk(chunk, 1.3)
    assert out.shape[0] == int(np.floor((50 - 1) / 1.3)) + 1 < 50
    assert out.shape[1] == 14
    # v < 1 -> 变长
    out = reconstruct_chunk(chunk, 0.5)
    assert out.shape[0] > 50
    # 边界: N<=1
    assert reconstruct_chunk(chunk[:1], 2.0).shape == (1, 14)
    print("[ok] reconstruct semantics")


def test_matches_robotwin_original():
    # 直接按文件路径加载 chunk_accel.py (只依赖 numpy), 绕过 envs/__init__.py 的 sapien import.
    import importlib.util
    path = "/home/xukainan/RoboTwin/envs/utils/chunk_accel.py"
    try:
        spec = importlib.util.spec_from_file_location("_rt_chunk_accel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rt_reconstruct = mod.reconstruct_chunk
    except Exception as e:  # noqa: BLE001
        print(f"[skip] RoboTwin original not loadable: {e!r}")
        return
    chunk = np.random.RandomState(0).randn(37, 14)
    for v in (0.7, 1.0, 1.2, 1.5, 2.3):
        a = reconstruct_chunk(chunk, v)
        b = rt_reconstruct(chunk, v)
        assert a.shape == b.shape and np.allclose(a, b), f"mismatch at v={v}"
    print("[ok] matches RoboTwin original")


def test_capabilities_sanitize():
    class _Cap(ChunkExecutor):
        capabilities = ExecutorCapabilities(compress=True, vel_scale=True, acc_scale=False)

        def execute_chunk(self, chunk, action):
            a = self._sanitize(action)
            return ChunkExecResult(status=STATUS_SUCCESS, frames_consumed=len(chunk),
                                   extra={"sanitized": a.as_tuple()})

    ex = _Cap()
    res = ex.execute_chunk(np.zeros((10, 14)), SpeedAction(v=1.2, vel_scale=1.5, acc_scale=3.0))
    # acc_scale 不支持 -> 归 1.0; v / vel_scale 保留
    assert res.extra["sanitized"] == (1.2, 1.5, 1.0)
    assert res.ok and res.frames_consumed == 10
    print("[ok] capabilities sanitize + result flow")


def test_result_from_speedup_info():
    # 模拟 take_chunk_action / infos["speedup"] 的 info dict
    ok = result_from_speedup_info(
        {"status": "success", "take_action_cnt_delta": 40, "dense_steps": 500, "duration": 2.0},
        sim_hz=250.0, success=True,
    )
    assert ok.status == STATUS_SUCCESS and ok.frames_consumed == 40
    assert ok.exec_steps == 500 and abs(ok.exec_time - 2.0) < 1e-9 and ok.success
    assert ok.extra["duration"] == 2.0
    # TOPP fallback -> RETIME_FAILED, dense_steps=0 -> exec_time=0
    fb = result_from_speedup_info(
        {"status": "topp_fallback", "take_action_cnt_delta": 40, "fallback_reason": "x"},
        success=False,
    )
    assert fb.status == STATUS_RETIME_FAILED and fb.exec_time == 0.0 and not fb.success
    print("[ok] result_from_speedup_info mapping")


if __name__ == "__main__":
    test_reconstruct_semantics()
    test_matches_robotwin_original()
    test_capabilities_sanitize()
    test_result_from_speedup_info()
    print("\nALL PASSED")
