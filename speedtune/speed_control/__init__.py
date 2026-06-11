"""速度控制智能体的 benchmark 无关组件.

与 ``speedtune.execution`` 的分工:
    execution/     -- "怎么把一段 chunk 按 (v,vel,acc) 执行" (benchmark 专属).
    speed_control/ -- 速度控制 RL 智能体本身: 奖励 / 状态 (本模块), 以及后续的策略.

奖励 / 状态口径对齐 RoboTwin/speedtune/rl/rainbow 与 .../rl/config.py (Plan A 非负奖励).
"""
from .reward import RewardBreakdown, SpeedRewardConfig, chunk_reward, crash_reward, speed_reward
from .state import SpeedStateConfig, assemble_state

__all__ = [
    "SpeedRewardConfig",
    "RewardBreakdown",
    "speed_reward",
    "chunk_reward",
    "crash_reward",
    "SpeedStateConfig",
    "assemble_state",
]
