"""ChunkExecutor: 跨 benchmark 的 chunk 加速执行抽象.

动机
----
速度控制 RL 策略输出一个通用的三维元动作 ``SpeedAction(v, vel_scale, acc_scale)``:
    - v:          chunk 压缩 / 插帧比率 (纯动作序列重采样, 任何 benchmark 通用).
    - vel_scale:  轨迹重定时的速度约束倍率.
    - acc_scale:  轨迹重定时的加速度约束倍率.

不同 benchmark "如何把一段 action chunk 在底层控制器上执行" 各不相同:
    - RoboTwin (Sapien): 关节空间, 整段 TOPPRA + 250Hz 密集下发.
    - LIBERO / LIBERO-plus / robomimic (robosuite/MuJoCo): OSC delta-EEF, ~20Hz,
      需任务空间重定时或 IK->关节->TOPPRA.

因此把"加速操作"抽象成一个统一接口: 速度策略只跟 ``ChunkExecutor`` 打交道,
每个 benchmark 提供自己的实现. 压缩 (compress) 作为唯一通用部分给默认实现.

设计要点
--------
- ``execute_chunk`` 返回 benchmark 无关的 ``ChunkExecResult``, 其中:
    * ``frames_consumed`` = 压缩后实际"下发"的 VLA 动作帧数 M.  这同时是 RL 预算消耗,
      也是后续 VLA-RL "executed-prefix 信用分配" 要用到的量 (见 PLAN.md §3.2).
    * ``exec_time`` = 这一 chunk 在真实机器人上的执行时间 (秒), 速度奖励的来源.
- ``capabilities`` 声明该 executor 实际支持哪些速度维度, 不支持的会被忽略并告警一次.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .chunk_ops import reconstruct_chunk


# 通用执行状态 (benchmark 无关). 各实现把自己的内部状态映射到这三种之一.
STATUS_SUCCESS = "success"          # 正常执行完 (含中途任务成功)
STATUS_RETIME_FAILED = "retime_failed"  # 重定时 / 轨迹求解失败 (如 TOPPRA fallback)
STATUS_TRUNCATED = "truncated"      # 预算耗尽 / episode 已结束, 未执行


@dataclass
class SpeedAction:
    """速度控制策略输出的通用元动作."""
    v: float = 1.0
    vel_scale: float = 1.0
    acc_scale: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.v), float(self.vel_scale), float(self.acc_scale))


@dataclass
class ChunkExecResult:
    """一次 chunk 执行的 benchmark 无关结果."""
    status: str                         # STATUS_*
    frames_consumed: int                # M: 压缩后下发的 VLA 动作帧数 (= RL 预算消耗)
    exec_steps: int = 0                 # 底层控制步数 (如 250Hz 物理步, 或控制 tick 数)
    exec_time: float = 0.0              # 真实执行时间 (秒), 速度奖励来源
    success: bool = False               # 这一步内任务是否判定成功
    fallback_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # benchmark 专属字段 (duration / return_code ...)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS


@dataclass(frozen=True)
class ExecutorCapabilities:
    """声明 executor 支持哪些速度维度. 不支持的维度在执行时被忽略."""
    compress: bool = True       # 支持 v (chunk 压缩)
    vel_scale: bool = True      # 支持速度约束倍率
    acc_scale: bool = True      # 支持加速度约束倍率


class ChunkExecutor(abc.ABC):
    """把一段 VLA action chunk 按速度元动作执行的 benchmark 无关接口.

    一个 executor 实例对应一个 (单)环境. 向量化由上层 (如 RLinf VectorEnv) 负责.
    """

    capabilities: ExecutorCapabilities = ExecutorCapabilities()

    def __init__(self) -> None:
        self._warned_unsupported = False

    # ------------------------------------------------------------------
    # 通用部分: 压缩 (纯动作序列重采样, 所有 benchmark 共用)
    # ------------------------------------------------------------------
    @staticmethod
    def compress(chunk: np.ndarray, v: float) -> np.ndarray:
        """按 v 压缩 / 插帧 chunk. 见 chunk_ops.reconstruct_chunk."""
        if v == 1.0:
            return np.asarray(chunk)
        return reconstruct_chunk(chunk, float(v))

    def _sanitize(self, action: SpeedAction) -> SpeedAction:
        """把不支持的速度维度归一到 1.0, 首次告警."""
        cap = self.capabilities
        v = action.v if cap.compress else 1.0
        vs = action.vel_scale if cap.vel_scale else 1.0
        ac = action.acc_scale if cap.acc_scale else 1.0
        if (v, vs, ac) != action.as_tuple() and not self._warned_unsupported:
            import warnings
            warnings.warn(
                f"{type(self).__name__} ignores unsupported speed dims "
                f"(cap={cap}); action {action.as_tuple()} -> ({v}, {vs}, {ac})",
                stacklevel=2,
            )
            self._warned_unsupported = True
        return SpeedAction(v, vs, ac)

    # ------------------------------------------------------------------
    # benchmark 专属: 真正执行
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def execute_chunk(self, chunk: np.ndarray, action: SpeedAction) -> ChunkExecResult:
        """以速度元动作 ``action`` 执行一段 action chunk ``(N, D)``, 返回通用结果.

        实现约定:
            1. 先 ``chunk = self.compress(chunk, action.v)``;
            2. 按 vel_scale / acc_scale 做轨迹重定时 (benchmark 专属);
            3. 在底层控制器逐点下发并统计 exec_steps / exec_time;
            4. 失败映射到 STATUS_RETIME_FAILED, 预算耗尽映射到 STATUS_TRUNCATED.
        """
        raise NotImplementedError
