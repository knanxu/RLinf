"""robosuite (LIBERO / LIBERO-plus / robomimic) 的 ChunkExecutor —— 方案 A (关节空间 TOPPRA).

robosuite 系 benchmark 用 MuJoCo + OSC 控制器, 动作是 7 维 delta-EEF
(3 pos + 3 rot + 1 gripper), 控制频率 ~20Hz, 没有现成关节轨迹. 已选**方案 A**
(``speedtune/docs/robosuite_toppra_design.md``): delta-EEF → 积分绝对 EEF 路点 →
顺序 seed DLS IK → 关节路点 → **复用 RoboTwin 同一 ``retime_chunk``** → 关节执行.
这样 ``(v, vel_scale, acc_scale)`` 与 RoboTwin **完全同义** (缩放关节速度/加速度上限),
框架真正统一成"一套加速模型跨 benchmark".

注意执行落点 (设计文档 §2): LIBERO ``venv.py`` 是**多进程 worker**, robosuite env 活在
子进程, 关节执行必须在 worker 内对传入的 env 对象操作. 因此 robosuite 的真正执行路径是
``rlinf/envs/libero/speedup.py`` 的 worker 内纯函数 (经 ``libero_env.chunk_step_with_speed``
→ ``venv`` 的 ``chunk_step_speedup`` 命令分发), **不走本 executor 实例** —— 这点与 RoboTwin
(线程池同进程, 直接用 RoboTwinChunkExecutor) 不同. 本类保留为 capabilities/schema 声明,
``execute_chunk`` 不是单进程直调入口.

状态: §8 step 1 (speedup.py 版本无关骨架 + IK/PD) 已落地; venv/libero_env 接线 (step 2-3)
与云端 §7 核实 (robosuite API / output_max / 关节限制 / 索引) 待办.
"""
from __future__ import annotations

import numpy as np

from .base import ChunkExecResult, ChunkExecutor, ExecutorCapabilities, SpeedAction


class RobosuiteChunkExecutor(ChunkExecutor):
    """robosuite 系 benchmark 的加速能力声明 (方案 A, 关节空间 TOPPRA).

    capabilities 三维全 True: 关节空间 TOPPRA 完整支持 v / vel_scale / acc_scale,
    与 RoboTwinChunkExecutor 同义. 实际执行见模块 docstring (在 libero venv worker 内).
    """

    # 方案 A 关节空间 TOPPRA: 三维速度元动作全支持 (与 RoboTwin 同义).
    capabilities = ExecutorCapabilities(compress=True, vel_scale=True, acc_scale=True)

    def __init__(self, env, control_hz: float = 20.0) -> None:
        super().__init__()
        self._env = env
        self._control_hz = float(control_hz)

    def execute_chunk(self, chunk: np.ndarray, action: SpeedAction) -> ChunkExecResult:
        raise NotImplementedError(
            "robosuite 加速执行在 libero venv worker 内进行 "
            "(rlinf/envs/libero/speedup.py), 不走本 executor 实例直调. "
            "见模块 docstring 与 speedtune/docs/robosuite_toppra_design.md."
        )
