"""速度控制智能体的状态拼装 (benchmark 无关).

口径对齐 RoboTwin/speedtune/rl/env.py:_assemble_state:

    state = concat[ cond_emb (cond_emb_dim,),       # pi0.5 VLM 前缀 mean-pool 条件特征
                    last_action (3,),               # 上次 (v, vel_scale, acc_scale)
                    progress (1,),                  # take_action_cnt / step_lim, 进度归一
                    last_fallback (1,) ]            # 上次是否重定时失败 (1.0 / 0.0)

cond_emb 来自模型 (infer_with_hidden), progress / last_fallback 由 env wrapper 依据
ChunkExecResult 与预算状态计算后传入. 本函数只做拼装 + 维度校验, 因此跨 benchmark 通用、可 CPU 单测.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedStateConfig:
    cond_emb_dim: int = 2048
    action_dim: int = 3  # (v, vel_scale, acc_scale)

    @property
    def state_dim(self) -> int:
        # cond_emb + last_action + progress(1) + last_fallback(1)
        return self.cond_emb_dim + self.action_dim + 1 + 1


def assemble_state(
    cond_emb: np.ndarray,
    last_action: np.ndarray,
    progress: float,
    last_fallback: float,
    cfg: SpeedStateConfig,
) -> np.ndarray:
    """拼装速度策略的观测向量, 返回 (state_dim,) float32."""
    cond = np.asarray(cond_emb, dtype=np.float32).reshape(-1)
    if cond.shape[0] != cfg.cond_emb_dim:
        raise ValueError(
            f"cond_emb dim mismatch: got {cond.shape[0]} vs cfg.cond_emb_dim={cfg.cond_emb_dim}"
        )
    act = np.asarray(last_action, dtype=np.float32).reshape(-1)
    if act.shape[0] != cfg.action_dim:
        raise ValueError(
            f"last_action dim mismatch: got {act.shape[0]} vs cfg.action_dim={cfg.action_dim}"
        )

    s = np.concatenate(
        [
            cond,
            act,
            np.array([float(progress)], dtype=np.float32),
            np.array([float(last_fallback)], dtype=np.float32),
        ],
        axis=0,
    )
    if s.shape[0] != cfg.state_dim:
        raise RuntimeError(f"assembled state dim {s.shape[0]} != cfg.state_dim {cfg.state_dim}")
    return s
