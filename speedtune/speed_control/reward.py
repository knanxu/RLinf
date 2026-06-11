"""速度控制智能体的奖励 (benchmark 无关).

口径对齐 RoboTwin/speedtune/rl/rainbow/config.py 与 .../rl/env.py 的 Plan A (非负奖励):

    r_speed = α_v·v^β_v + α_vs·vel_scale^β_vs + α_as·acc_scale^β_as
    r_task  = task_success_reward  若该 chunk 执行后任务判定成功, 否则 0
    r_total = r_speed + r_task                       (正常执行)
    r_total = fallback_penalty (默认 0)              (重定时失败 / TOPP fallback, 屏蔽 r_speed)
    r_total = crash_penalty   (默认 0)               (env 崩溃, 由 env wrapper 调 crash_reward)

Plan A 不给负 penalty: episode 预算消耗本身就是隐性惩罚, 同时保证 Q 值 >= 0 (利于 C51 support [0, V_max]).

设计: 只吃 SpeedAction + ChunkExecResult, 不碰 task_env, 因此跨 benchmark 通用且可 CPU 单测.
传入的 SpeedAction 应已 clamp 到动作空间边界 (clamp 是上层策略/env 的职责).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..execution.base import STATUS_RETIME_FAILED, ChunkExecResult, SpeedAction


@dataclass
class SpeedRewardConfig:
    # 速度奖励系数 (默认对齐 rainbow/config.py:RewardConfig)
    alpha_v: float = 0.05
    alpha_vs: float = 0.05
    alpha_as: float = 0.05
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0
    # 终止 / 异常奖励 (Plan A: 0, 不给负 penalty)
    fallback_penalty: float = 0.0
    crash_penalty: float = 0.0
    # 任务成功奖励
    task_success_reward: float = 1.0


@dataclass
class RewardBreakdown:
    """奖励分解, 便于日志 / info."""
    r_speed: float
    r_task: float
    total: float


def speed_reward(action: SpeedAction, cfg: SpeedRewardConfig) -> float:
    """纯速度奖励项 r_speed (不含任务奖励)."""
    v, vs, ac = action.as_tuple()
    return (
        cfg.alpha_v * (v ** cfg.beta_v)
        + cfg.alpha_vs * (vs ** cfg.beta_vs)
        + cfg.alpha_as * (ac ** cfg.beta_as)
    )


def chunk_reward(
    action: SpeedAction, result: ChunkExecResult, cfg: SpeedRewardConfig
) -> RewardBreakdown:
    """一次正常 (非崩溃) chunk 执行后的奖励.

    与 env.py 一致: 仅重定时失败 (RETIME_FAILED) 屏蔽 r_speed 并给 fallback_penalty;
    其余 (success / truncated) 给 r_speed + r_task.
    """
    if result.status == STATUS_RETIME_FAILED:
        return RewardBreakdown(r_speed=0.0, r_task=0.0, total=cfg.fallback_penalty)
    r_s = speed_reward(action, cfg)
    r_t = cfg.task_success_reward if result.success else 0.0
    return RewardBreakdown(r_speed=r_s, r_task=r_t, total=r_s + r_t)


def crash_reward(cfg: SpeedRewardConfig) -> RewardBreakdown:
    """env 崩溃 (execute_chunk 抛异常) 时的终止奖励, 由 env wrapper 调用."""
    return RewardBreakdown(r_speed=0.0, r_task=0.0, total=cfg.crash_penalty)
