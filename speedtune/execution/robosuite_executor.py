"""robosuite (LIBERO / LIBERO-plus / robomimic) 的 ChunkExecutor 实现.

[占位 / P2.4]  robosuite 系 benchmark 用 MuJoCo + OSC 控制器, 动作通常是 7 维
delta-EEF (3 pos + 3 rot + 1 gripper), 控制频率 ~20Hz, 没有现成的关节空间轨迹,
因此不能直接套用 RoboTwin 的关节 TOPPRA.

两条实现路线 (见 PLAN.md §2(b)):
    路线 1 (建议先做, 近似): 任务空间重定时
        - compress(v): 直接重采样 EEF 路点序列 (通用算子, 已可用).
        - retime(vel_scale, acc_scale): 缩放 EEF 路点之间的插值步数 / 控制 tick 数,
          即把 vel_scale 解释为 "每段 EEF 位移分配的控制步数 / vel_scale".
          不依赖机器人模型, 实现简单, 但不保证关节级速度/加速度约束.
        - execute: 用 env.step 逐 tick 下发插值后的 EEF 目标.
        - exec_time: 控制 tick 数 / control_hz.

    路线 2 (忠实, 较重): IK -> 关节 -> TOPPRA
        - 用机器人模型把 EEF 路点 IK 成关节路点, 复用 toppra retime, 再 FK/控制器跟踪.
        - 需要 robosuite 暴露 IK 与关节限制; 与 OSC 控制器的交互需小心.

capabilities: 路线 1 下 vel_scale 以 "插值步数缩放" 近似支持, acc_scale 暂不支持
(OSC 无显式加速度约束); 待路线 2 再补全.
"""
from __future__ import annotations

import numpy as np

from .base import ChunkExecResult, ChunkExecutor, ExecutorCapabilities, SpeedAction


class RobosuiteChunkExecutor(ChunkExecutor):
    """[未实现] robosuite 系 benchmark 的加速执行. 见模块 docstring 的两条路线."""

    # 路线 1: 压缩 + 速度(插值步数缩放) 近似支持; 加速度暂不支持.
    capabilities = ExecutorCapabilities(compress=True, vel_scale=True, acc_scale=False)

    def __init__(self, env, control_hz: float = 20.0) -> None:
        super().__init__()
        self._env = env
        self._control_hz = float(control_hz)

    def execute_chunk(self, chunk: np.ndarray, action: SpeedAction) -> ChunkExecResult:
        raise NotImplementedError(
            "RobosuiteChunkExecutor 待 P2.4 实现 (任务空间重定时). 见模块 docstring."
        )
