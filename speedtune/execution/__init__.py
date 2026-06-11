"""跨 benchmark 的 chunk 加速执行抽象.

公共接口:
    ChunkExecutor          -- 抽象基类, 速度策略只跟它打交道.
    SpeedAction            -- 通用三维速度元动作 (v, vel_scale, acc_scale).
    ChunkExecResult        -- benchmark 无关的执行结果.
    ExecutorCapabilities   -- 声明 executor 支持哪些速度维度.

实现:
    RoboTwinChunkExecutor  -- RoboTwin (Sapien) 关节 TOPPRA, 薄封装 take_chunk_action.
    RobosuiteChunkExecutor -- [P2.4] LIBERO / LIBERO-plus / robomimic, 任务空间重定时.

具体 benchmark 实现按需 import, 避免在没装对应仿真依赖时报错.
"""
from .base import (
    STATUS_RETIME_FAILED,
    STATUS_SUCCESS,
    STATUS_TRUNCATED,
    ChunkExecResult,
    ChunkExecutor,
    ExecutorCapabilities,
    SpeedAction,
)
from .chunk_ops import reconstruct_chunk

# robotwin_executor 只依赖 numpy + .base (无 sapien/toppra 硬依赖), CPU 可安全导入.
# (robosuite_executor 仍按需 import, 因后续会引入 robosuite 依赖.)
from .robotwin_executor import RoboTwinChunkExecutor, result_from_speedup_info

__all__ = [
    "ChunkExecutor",
    "SpeedAction",
    "ChunkExecResult",
    "ExecutorCapabilities",
    "reconstruct_chunk",
    "RoboTwinChunkExecutor",
    "result_from_speedup_info",
    "STATUS_SUCCESS",
    "STATUS_RETIME_FAILED",
    "STATUS_TRUNCATED",
]
