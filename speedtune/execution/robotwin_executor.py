"""RoboTwin (Sapien) 的 ChunkExecutor 实现.

薄封装: 复用 RoboTwin 已有且已测的 ``Base_Task.take_chunk_action``
(整段 TOPPRA + 250Hz 密集下发), 不重写 TOPPRA 逻辑, 只把它的 info dict 翻译成
框架通用的 ``ChunkExecResult``.

两种用法:
    1. 单 env: ``RoboTwinChunkExecutor(task_env).execute_chunk(chunk, action)``.
    2. RLinf 向量化路径: VectorEnv.step_with_speed 把每 env 的 take_chunk_action info
       放进 ``infos["speedup"]``, 上层用 ``result_from_speedup_info(info, success=...)``
       逐 env 转成 ChunkExecResult. 两条路共用同一映射, 保证语义一致.
"""
from __future__ import annotations

import numpy as np

from .base import (
    STATUS_RETIME_FAILED,
    STATUS_SUCCESS,
    STATUS_TRUNCATED,
    ChunkExecResult,
    ChunkExecutor,
    ExecutorCapabilities,
    SpeedAction,
)

# RoboTwin take_chunk_action 内部 info["status"] -> 框架通用状态
_STATUS_MAP = {
    "success": STATUS_SUCCESS,
    "topp_fallback": STATUS_RETIME_FAILED,
    "truncated": STATUS_TRUNCATED,
}


def result_from_speedup_info(
    info: dict, sim_hz: float = 250.0, success: bool = False
) -> ChunkExecResult:
    """把 RoboTwin take_chunk_action 的 info dict 映射成通用 ChunkExecResult.

    ``info`` 即 take_chunk_action 的返回 (也是 RLinf 向量化路径里 infos["speedup"][env_i]).
    ``success`` 由调用方提供 (单 env 用 task.eval_success; 向量化用 RLinf infos["success"]).
    """
    status = _STATUS_MAP.get(info.get("status", "success"), STATUS_SUCCESS)
    dense_steps = int(info.get("dense_steps", 0))
    return ChunkExecResult(
        status=status,
        frames_consumed=int(info.get("take_action_cnt_delta", 0)),
        exec_steps=dense_steps,
        exec_time=dense_steps / float(sim_hz) if dense_steps else 0.0,
        success=bool(success),
        fallback_reason=info.get("fallback_reason"),
        extra={
            "duration": info.get("duration", 0.0),
            "topp_return_code": info.get("topp_return_code"),
            "success_obs_time": info.get("success_obs_time", 0.0),
            "chunk_truncated": info.get("chunk_truncated", False),
        },
    )


class RoboTwinChunkExecutor(ChunkExecutor):
    """把速度元动作执行委托给 RoboTwin task_env.take_chunk_action.

    Args:
        task_env: 已 setup_demo 的 RoboTwin Base_Task 实例.
        sim_hz:   底层物理频率, 用于由 dense_steps 反推真实执行时间 (RoboTwin = 250).
    """

    # RoboTwin 关节空间 TOPPRA 完整支持三维速度元动作.
    capabilities = ExecutorCapabilities(compress=True, vel_scale=True, acc_scale=True)

    def __init__(self, task_env, sim_hz: float = 250.0) -> None:
        super().__init__()
        self._task_env = task_env
        self._sim_hz = float(sim_hz)

    def execute_chunk(self, chunk: np.ndarray, action: SpeedAction) -> ChunkExecResult:
        a = self._sanitize(action)
        # 注意: 压缩 (v) 由 take_chunk_action 内部处理, 这里不重复 compress, 直接透传 v,
        # 以保证 budget(cnt) 截断语义与原实现完全一致.
        info = self._task_env.take_chunk_action(
            np.asarray(chunk),
            vel_scale=a.vel_scale,
            acc_scale=a.acc_scale,
            v=a.v,
        )
        return result_from_speedup_info(
            info,
            sim_hz=self._sim_hz,
            success=bool(getattr(self._task_env, "eval_success", False)),
        )
