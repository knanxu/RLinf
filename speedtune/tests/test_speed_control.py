"""speed_control (奖励 / 状态) 的 CPU smoke test.

验证:
    1. speed_reward 在动作空间四角的数值, 对齐 rainbow/config.py 文档的 r_total ∈ [0.13, 1.51].
    2. chunk_reward 的状态映射 (RETIME_FAILED / success / 普通未成功).
    3. crash_reward.
    4. assemble_state 的形状 / 取值 / 维度校验.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from speedtune.execution import ChunkExecResult, SpeedAction  # noqa: E402
from speedtune.execution.base import STATUS_RETIME_FAILED, STATUS_SUCCESS  # noqa: E402
from speedtune.speed_control import (  # noqa: E402
    SpeedRewardConfig,
    SpeedStateConfig,
    assemble_state,
    chunk_reward,
    crash_reward,
    speed_reward,
)


def test_speed_reward_corners():
    cfg = SpeedRewardConfig()
    # r_v(v,vs,as) = 0.05*v^2 + 0.05*vs^2 + 0.05*as^1
    assert abs(speed_reward(SpeedAction(1.0, 1.0, 1.0), cfg) - 0.15) < 1e-9
    # 动作空间下界角 -> 文档 r_total 下界 0.13
    lo = speed_reward(SpeedAction(0.8, 1.0, 1.0), cfg)
    assert abs(lo - (0.05 * 0.64 + 0.05 + 0.05)) < 1e-9 and abs(lo - 0.132) < 1e-3
    # 上界角 + 任务奖励 -> 文档 r_total 上界 1.51
    hi = speed_reward(SpeedAction(1.5, 2.0, 4.0), cfg)
    assert abs(hi - (0.05 * 2.25 + 0.05 * 4 + 0.05 * 4)) < 1e-9
    assert abs((hi + cfg.task_success_reward) - 1.5125) < 1e-3
    print("[ok] speed_reward corners match documented [0.13, 1.51]")


def _result(status, success):
    return ChunkExecResult(status=status, frames_consumed=10, success=success)


def test_chunk_reward_mapping():
    cfg = SpeedRewardConfig()
    a = SpeedAction(1.2, 1.5, 2.0)
    rv = speed_reward(a, cfg)
    # 重定时失败 -> 屏蔽 r_speed, 给 fallback_penalty(0)
    rb = chunk_reward(a, _result(STATUS_RETIME_FAILED, False), cfg)
    assert rb.r_speed == 0.0 and rb.total == cfg.fallback_penalty
    # 成功 -> r_speed + task_success_reward
    rb = chunk_reward(a, _result(STATUS_SUCCESS, True), cfg)
    assert abs(rb.r_task - 1.0) < 1e-9 and abs(rb.total - (rv + 1.0)) < 1e-9
    # 正常未成功 -> 仅 r_speed
    rb = chunk_reward(a, _result(STATUS_SUCCESS, False), cfg)
    assert rb.r_task == 0.0 and abs(rb.total - rv) < 1e-9
    # crash
    assert crash_reward(cfg).total == cfg.crash_penalty
    print("[ok] chunk_reward status mapping")


def test_assemble_state():
    cfg = SpeedStateConfig(cond_emb_dim=2048)
    assert cfg.state_dim == 2048 + 3 + 1 + 1 == 2053
    cond = np.arange(2048, dtype=np.float32)
    s = assemble_state(cond, [1.2, 1.5, 2.0], progress=0.3, last_fallback=1.0, cfg=cfg)
    assert s.shape == (2053,) and s.dtype == np.float32
    assert np.allclose(s[:2048], cond)
    assert np.allclose(s[2048:2051], [1.2, 1.5, 2.0])
    assert s[2051] == np.float32(0.3) and s[2052] == 1.0
    # 维度校验
    for bad in (np.zeros(2047), np.zeros(2049)):
        try:
            assemble_state(bad, [1, 1, 1], 0.0, 0.0, cfg)
            raise AssertionError("expected ValueError on wrong cond_emb dim")
        except ValueError:
            pass
    print("[ok] assemble_state shape/values/validation")


if __name__ == "__main__":
    test_speed_reward_corners()
    test_chunk_reward_mapping()
    test_assemble_state()
    print("\nALL PASSED")
