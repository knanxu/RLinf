"""SPEEDTUNING-style Rainbow DQN for chunk speedup.

Reference: Yuan et al., "SPEEDTUNING: Speeding Up Policy Execution with
Lightweight Reinforcement Learning", 2024.

Differences vs the paper:
  - 3-D factored action space (v, vel_scale, acc_scale) instead of 1-D v
  - Each dimension has its own dueling C51 head (factored Q heads)
  - VLM mean-pooled conditioning embedding as state (cheaper + richer than
    proprioception + image encoder)
  - frame_skip = 1 because one env.step already executes one chunk
"""
