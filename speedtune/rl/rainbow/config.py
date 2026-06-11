"""Rainbow DQN config for chunk speedup (SPEEDTUNING-style).

All hyperparameters live here so swapping between SAC and Rainbow only
needs to flip the imports in train_rainbow.py.

Defaults are picked to converge an order of magnitude faster than SAC on
the same task: small action grid (8/6/7), large C51 atoms (101), n-step
returns, PER, double + dueling. ε-greedy for exploration (linear decay).
"""
from dataclasses import dataclass, field
from typing import Tuple


# ----------------------------------------------------------------------
# Discrete grids for (v, vel_scale, acc_scale)
# ----------------------------------------------------------------------
# Conservative bounds (validated by smoke_replay_expert.py), with steps
# small enough to give the agent meaningful resolution but small enough
# to keep |A| tractable: 8 + 6 + 7 = 21 Q-logits in factored mode.
@dataclass
class ActionGridConfig:
    v_grid: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    vel_grid: Tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    acc_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


# ----------------------------------------------------------------------
# Reward (Plan A: non-negative reward only)
# ----------------------------------------------------------------------
# Per-chunk reward when TOPP solves and the chunk runs:
#   r_v   = α_v · v^β_v + α_vs · vs^β_vs + α_as · as^β_as
#   r_task = 1 if check_success() else 0
#   r_total = r_v + r_task                              ∈ [0.13, 1.51]
#
# When TOPP fallback or env crash:
#   r_total = 0   (no positive signal; episode budget keeps draining as
#                  an implicit cost, so the agent still learns to avoid it)
#   crash additionally terminates the episode.
#
# This keeps Q values ≥ 0 so C51 support can sit on [V_min=0, V_max=20].
@dataclass
class RewardConfig:
    alpha_v: float = 0.05
    alpha_vs: float = 0.05
    alpha_as: float = 0.05
    beta_v: float = 2.0
    beta_vs: float = 2.0
    beta_as: float = 1.0

    fallback_penalty: float = 0.0    # 方案 A: 不再给负 penalty
    crash_penalty: float = 0.0       # 方案 A: 不再给负 penalty (仅 terminal)


# ----------------------------------------------------------------------
# Env (chunk-level wrapper, same as SAC)
# ----------------------------------------------------------------------
@dataclass
class EnvConfig:
    task_name: str = "shake_bottle"
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    chunk_size: int = 50
    max_chunks_per_episode: int = 100

    cond_emb_dim: int = 2048
    state_dim: int = 2048 + 3 + 1 + 1     # cond_emb + last_action + cnt + last_fallback


@dataclass
class PolicyConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    api_key: str | None = None
    pi0_step: int = 50


# ----------------------------------------------------------------------
# Rainbow DQN
# ----------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """Dueling + factored C51 head."""
    backbone_sizes: Tuple[int, ...] = (512, 512, 256)
    # Per-dim Q head MLP (after the shared backbone)
    head_hidden: int = 128


@dataclass
class C51Config:
    """Categorical distributional Q (C51 / Bellemare et al. 2017).

    Plan A reward is non-negative:
      single-chunk r ∈ [0.13, 1.51] (max only at terminal success step)
      typical r_v per chunk ≈ 0.2~0.4
      successful episode ~10-30 chunks, only the last gets r_task=1
      realistic episode return ≈ 5~7 (with γ=0.99 discounting)
      V_max=8 covers the practical upper bound with margin.
      101 atoms → resolution 0.08/atom, very fine-grained.
    """
    v_min: float = 0.0
    v_max: float = 8.0
    n_atoms: int = 101


@dataclass
class RainbowConfig:
    gamma: float = 0.99
    n_step: int = 3              # multi-step return horizon
    tau_target: float = 1.0      # 1.0 == hard target update every target_update_every
    target_update_every: int = 500

    lr: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 1000
    updates_per_step: int = 1

    # PER (Schaul et al. 2015)
    per_alpha: float = 0.6           # priority exponent
    per_beta_start: float = 0.4      # IS-weight exponent (annealed to 1.0)
    per_beta_end: float = 1.0
    per_beta_steps: int = 30_000     # anneal over this many env steps
    per_eps: float = 1e-6            # min priority

    # ε-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000

    # Gradient clipping (per Rainbow paper Adam settings)
    grad_clip: float = 10.0
    adam_eps: float = 1.5e-4


@dataclass
class TrainConfig:
    total_env_steps: int = 30_000
    eval_every: int = 2000
    eval_episodes: int = 10
    checkpoint_every: int = 5000
    log_every: int = 50

    seed: int = 42
    device: str = "cuda"

    run_name: str = "rainbow_chunk_speedup"
    log_dir: str = "./speedtune/rl/runs"


@dataclass
class FullConfig:
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    c51: C51Config = field(default_factory=C51Config)
    rainbow: RainbowConfig = field(default_factory=RainbowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
