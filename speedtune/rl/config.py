"""Chunk-speedup 训练的共享配置 (动作空间 / 奖励 / env / pi0.5 server).

这些 dataclass 被 ChunkSpeedupEnv (rl/env.py) 直接消费. Rainbow 训练
(rl/rainbow/) 通过 adapter 把自己的 FullConfig 转成这里的形状再喂给 env;
算法专属超参 (C51 / Rainbow / 网络 / 训练循环) 见 rl/rainbow/config.py.

默认值:
  - action 空间: 保守边界 (smoke_replay_expert 验证过的安全区)
  - reward: Plan A 非负奖励 (fallback / crash 给 0, 不给负 penalty)
"""
from dataclasses import dataclass


@dataclass
class ActionSpaceConfig:
    # 保守边界: 在 smoke_replay_expert 验证过的安全区
    v_low: float = 0.8
    v_high: float = 1.5
    vel_low: float = 1.0
    vel_high: float = 2.0
    acc_low: float = 1.0
    acc_high: float = 4.0


@dataclass
class RewardConfig:
    """
    Plan A (non-negative reward):
      r_v = alpha_v * v^beta_v + alpha_vs * vel_scale^beta_vs + alpha_as * acc_scale^beta_as
      r_task ∈ {0, 1}  若 chunk 执行后 check_success == True 则为 1, 否则 0
      r_total = r_v + r_task                    (正常 chunk)
      r_total = 0                               (TOPP fallback / crash)

    Fallback / crash 不给负 penalty; episode 预算消耗本身就是隐性惩罚.
    """
    alpha_v: float = 0.05
    alpha_vs: float = 0.05
    alpha_as: float = 0.05
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0

    fallback_penalty: float = 0.0    # 方案 A: 不给负 penalty
    crash_penalty: float = 0.0       # 方案 A: 不给负 penalty (仅 terminal)


@dataclass
class EnvConfig:
    task_name: str = "shake_bottle"
    task_config: str = "smoke_test"
    instruction_type: str = "unseen"
    chunk_size: int = 50                    # pi0.5 action_horizon
    max_chunks_per_episode: int = 100       # 额外上限, 配合 step_lim

    # 状态维度: cond_emb + last_action(3) + chunk_idx_norm(1) + last_topp_fallback(1)
    cond_emb_dim: int = 2048
    state_dim: int = 2048 + 3 + 1 + 1


@dataclass
class PolicyConfig:
    """pi0.5 服务器配置 (本机不加载权重, 通过 websocket 调云端 server).

    云端 openpi server 启动方式见 speedtune/rl/README.md.
    """
    server_host: str = "127.0.0.1"   # 本地通过 SSH 端口转发到云端
    server_port: int = 8000
    api_key: str | None = None
    # 每次从返回的 action chunk 中截取的前 pi0_step 帧 (对齐 pi_model.py:47)
    pi0_step: int = 50
