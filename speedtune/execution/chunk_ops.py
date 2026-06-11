"""跨 benchmark 通用的 chunk 操作.

这里只放与仿真后端无关的纯函数. 当前唯一通用操作是 chunk 压缩 / 插帧重采样
(reconstruct_chunk), 它只对 (N, D) 动作序列做线性重采样, 不关心动作是关节角还是
EEF 位姿, 因此对 RoboTwin / LIBERO / robomimic 等所有 benchmark 都适用.

实现与 RoboTwin/envs/utils/chunk_accel.py 对齐 (移植到框架层作为可复用的通用算子).
"""
from __future__ import annotations

import numpy as np


def reconstruct_chunk(chunk: np.ndarray, v: float) -> np.ndarray:
    """按速度 v 对 action chunk 做线性插值重采样.

      v > 1  聚合 (aggregation):   M = floor((N-1)/v)+1 < N   chunk 变短 -> 执行更快
      v < 1  分解 (decomposition): M > N                      chunk 变长 -> 执行更平滑
      v = 1  identity

    采样位置 positions[k] = k * v, k = 0..M-1, clip 到 [0, N-1].
    每个浮点位置 p: idx=floor(p), frac=p-idx, out = chunk[idx] + frac*(chunk[idx+1]-chunk[idx]).

    Args:
        chunk: (N, D) action chunk.
        v:     正实数速度比率, 支持非整数 (如 1.3).

    Returns:
        (M, D) 重采样后的 chunk.
    """
    assert v > 0, f"v must be positive, got {v}"
    chunk = np.asarray(chunk)
    assert chunk.ndim == 2, f"chunk must be (N, D), got shape {chunk.shape}"

    N = chunk.shape[0]
    if N <= 1:
        return chunk.copy()

    M = int(np.floor((N - 1) / v)) + 1
    positions = np.clip(np.arange(M) * v, 0.0, N - 1)

    idx = np.floor(positions).astype(int)
    frac = positions - idx
    idx_next = np.minimum(idx + 1, N - 1)

    return chunk[idx] + frac[:, None] * (chunk[idx_next] - chunk[idx])
