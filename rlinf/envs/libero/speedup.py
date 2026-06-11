# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""robosuite / LIBERO 关节空间 TOPPRA 加速 —— worker 内执行 (方案 A).

实现 ``speedtune/docs/robosuite_toppra_design.md`` §8 step 1 的**版本无关骨架**:
delta-EEF chunk → 绝对 EEF 路点 → 顺序 seed DLS IK → 关节路点 → ``retime_chunk``
(复用 RoboTwin 同一 TOPPRA) → PD 兜底关节执行 → RoboTwin 兼容 info dict
(经 ``speedtune.execution.robotwin_executor.result_from_speedup_info`` 转 ChunkExecResult).

设计约束 (见设计文档 §2):
    LIBERO ``venv.py`` 是多进程 worker, robosuite env 活在子进程. 关节执行必须在
    worker 内、对传入的 robosuite env 对象操作, **不改 libero / robosuite 库本身**.
    本模块即那段 "worker 内纯函数", 不持有 env, 所有句柄经参数传入.

本地可测 vs 云端待定:
    * 纯函数 (``so3_exp`` / ``so3_log`` / ``integrate_eef_deltas``): 仅依赖 numpy, CPU 可单测.
    * mujoco IK / PD 执行 / 编排: 依赖 mujoco + robosuite env, 本地无 mujoco, 只写不跑.
    * 标注 ``TODO(cloud)`` 的是设计文档 §7 必须在云端核实后才能定稿的 robosuite 专属细节
      (控制器 API / OSC output_max / 关节 vel-acc 限制来源 / site 与执行器索引).

mujoco / robosuite / RoboTwin retime 全部**懒加载** (函数内 import), 保证纯函数与本模块
在无这些重依赖的环境下也能 import 和单测.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 纯几何工具 (numpy-only, CPU 可单测)
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """3 向量 → 反对称矩阵 (so(3) hat)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """轴角 ``omega`` (3,) → 旋转矩阵 (3,3). Rodrigues 公式, 数值稳定于 theta→0."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    K = _skew(omega)
    if theta < 1e-8:
        # 一阶近似: exp(K) ≈ I + K (theta 很小时足够精确, 避免除零)
        return np.eye(3) + K
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (3,3) → 轴角 (3,). so3_exp 的逆, 用于 IK 姿态误差."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        # 接近单位阵: 取反对称部分作一阶近似
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    ) / (2.0 * np.sin(theta))
    return axis * theta


def integrate_eef_deltas(
    start_pos: np.ndarray,
    start_rotmat: np.ndarray,
    delta_chunk: np.ndarray,
    pos_output_max: float | np.ndarray = 1.0,
    rot_output_max: float | np.ndarray = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 OSC delta-EEF 动作序列积分成绝对 EEF 路点 (设计文档 §3 第 2 步).

    动作约定 (OSC_POSE): ``delta_chunk[k] = [dpos(3), drot_axisangle(3), gripper(1)]``,
    每帧的 dpos / drot 是控制器归一化输入 [-1, 1], 需乘 controller 的 ``output_max``
    换算成米 / 弧度后再积分.

    Args:
        start_pos:        (3,)   当前 EEF 位置 (世界系).
        start_rotmat:     (3,3)  当前 EEF 朝向.
        delta_chunk:      (M, 7) delta-EEF + gripper 序列.
        pos_output_max:   标量或 (3,) 位置缩放 (controller output_max[:3]).
        rot_output_max:   标量或 (3,) 姿态缩放 (controller output_max[3:6]).

    Returns:
        positions:  (M, 3)   累积绝对 EEF 位置路点.
        rotmats:    (M, 3, 3) 累积绝对 EEF 朝向路点.
        grippers:   (M,)     gripper 指令 (原样透传, 不积分).

    TODO(cloud, 设计文档 §7.2): ``pos_output_max`` / ``rot_output_max`` 必须取自云端
    实际 controller config (``output_max`` / ``input_max``), 否则路径尺度错; 同时世界系
    vs body 系的 delta 朝向合成约定 (这里按 OSC_POSE 世界系左乘) 需对 LIBERO 实测核实.
    """
    delta_chunk = np.asarray(delta_chunk, dtype=np.float64)
    assert delta_chunk.ndim == 2 and delta_chunk.shape[1] >= 7, (
        f"delta_chunk must be (M, >=7), got {delta_chunk.shape}"
    )
    M = delta_chunk.shape[0]
    pos = np.asarray(start_pos, dtype=np.float64).reshape(3).copy()
    rot = np.asarray(start_rotmat, dtype=np.float64).reshape(3, 3).copy()

    positions = np.empty((M, 3), dtype=np.float64)
    rotmats = np.empty((M, 3, 3), dtype=np.float64)
    grippers = np.empty((M,), dtype=np.float64)

    for k in range(M):
        dpos = delta_chunk[k, 0:3] * pos_output_max
        drot = delta_chunk[k, 3:6] * rot_output_max
        pos = pos + dpos
        # 世界系左乘 (OSC_POSE 约定): R_new = exp(drot) @ R_cur
        rot = so3_exp(drot) @ rot
        positions[k] = pos
        rotmats[k] = rot
        grippers[k] = delta_chunk[k, 6]

    return positions, rotmats, grippers


# ---------------------------------------------------------------------------
# mujoco DLS IK (lazy mujoco; 本地无 mujoco, 写对但不跑)
# ---------------------------------------------------------------------------


def dls_ik_sequential(
    model,
    scratch_data,
    site_id: int,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    target_positions: np.ndarray,
    target_rotmats: np.ndarray,
    q_seed: np.ndarray,
    jnt_range: Optional[np.ndarray] = None,
    damping: float = 1e-2,
    max_iter: int = 20,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
) -> Tuple[np.ndarray, bool]:
    """顺序 seed 阻尼最小二乘 (DLS) IK (设计文档 §4).

    在 scratch ``MjData`` 上迭代 (``mj_copyData`` 出来的副本, **不扰动真实 sim**),
    对每个 EEF 路点求关节角, 用上一路点的解作初值 (顺序 seed → Panda 7-DOF 冗余零空间
    连续, 无关节跳变).

    Args:
        model:            mujoco MjModel.
        scratch_data:     mujoco MjData (调用方用 mj_copyData 复制真实 data, 本函数只读写它).
        site_id:          EEF site id (``robot.eef_site_id``).        TODO(cloud, §7.4)
        arm_qpos_adr:     (n_arm,) 手臂关节在 qpos 中的地址.           TODO(cloud, §7.4)
        arm_dof_adr:      (n_arm,) 手臂关节在 qvel/jacobian 列中的地址. TODO(cloud, §7.4)
        target_positions: (M, 3)   目标 EEF 位置路点.
        target_rotmats:   (M, 3, 3) 目标 EEF 朝向路点.
        q_seed:           (n_arm,)  第一个路点的初值 (一般 = q_cur).
        jnt_range:        (n_arm, 2) 关节限位, 迭代中 clip; None 则不限.
        damping:          阻尼 λ (~1e-2).
        max_iter:         每路点最大迭代步数.
        pos_tol / rot_tol: 收敛阈值 (米 / 弧度).

    Returns:
        q_path: (M, n_arm) 关节路点 (失败路点之后用最后可行解填充).
        ok:     bool, 是否全部路点收敛 (任一失败 → False, 触发 §6 fallback).
    """
    import mujoco  # lazy: 本地无 mujoco 时不影响纯函数 import

    arm_qpos_adr = np.asarray(arm_qpos_adr, dtype=int)
    arm_dof_adr = np.asarray(arm_dof_adr, dtype=int)
    n_arm = arm_qpos_adr.shape[0]
    M = target_positions.shape[0]
    nv = model.nv

    q = np.asarray(q_seed, dtype=np.float64).reshape(n_arm).copy()
    q_path = np.zeros((M, n_arm), dtype=np.float64)
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    all_ok = True

    for k in range(M):
        p_target = target_positions[k]
        R_target = target_rotmats[k]
        converged = False
        for _ in range(max_iter):
            scratch_data.qpos[arm_qpos_adr] = q
            mujoco.mj_kinematics(model, scratch_data)
            p_cur = scratch_data.site_xpos[site_id].copy()
            R_cur = scratch_data.site_xmat[site_id].reshape(3, 3).copy()

            e_pos = p_target - p_cur
            e_rot = so3_log(R_target @ R_cur.T)  # 世界系姿态误差
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                converged = True
                break

            mujoco.mj_jacSite(model, scratch_data, jacp, jacr, site_id)
            J = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])  # (6, n_arm)
            e = np.concatenate([e_pos, e_rot])  # (6,)
            # dq = Jᵀ (JJᵀ + λ²I)⁻¹ e
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            q = q + dq
            if jnt_range is not None:
                q = np.clip(q, jnt_range[:, 0], jnt_range[:, 1])

        q_path[k] = q
        if not converged:
            all_ok = False
            # 后续路点继续用当前 q 顺序 seed (返回 ok=False, 上层走 fallback)

    return q_path, all_ok


# ---------------------------------------------------------------------------
# PD 兜底关节执行 (lazy; 依赖 robosuite env.sim, 本地不跑)
# ---------------------------------------------------------------------------


def execute_joint_trajectory_pd(
    sim,
    arm_qpos_adr: np.ndarray,
    arm_qvel_adr: np.ndarray,
    arm_actuator_ids: np.ndarray,
    dense_q: np.ndarray,
    dense_qvel: np.ndarray,
    dense_gripper: np.ndarray,
    set_gripper_ctrl: Callable[[object, np.ndarray], None],
    kp: np.ndarray,
    kd: np.ndarray,
    check_success: Callable[[], bool],
    max_steps: Optional[int] = None,
) -> Tuple[int, bool]:
    """PD 兜底执行循环 (设计文档 §5 兜底): ``tau = Kp(q*-q) + Kd(q̇*-q̇)``.

    不碰 robosuite 控制器 API, 对 robosuite 1.4/1.5 版本最鲁棒 (§7.1 风险规避).
    稳定后再可选换原生 ``JointPositionController`` (设计文档 §8 step 5).

    Args:
        sim:               robosuite ``env.sim`` (mujoco_py / mujoco MjSim 封装).
        arm_qpos_adr / arm_qvel_adr: 手臂关节 qpos / qvel 索引.   TODO(cloud, §7.4)
        arm_actuator_ids:  手臂执行器 (ctrl) 索引.                TODO(cloud, §7.4)
        dense_q:           (T, n_arm) 目标关节位置 (retime 输出).
        dense_qvel:        (T, n_arm) 目标关节速度.
        dense_gripper:     (T, g_dof) 目标 gripper.
        set_gripper_ctrl:  回调, 把 gripper 指令写进 sim.data.ctrl (robosuite gripper
                           执行器布局各异, 交给调用方).            TODO(cloud, §7.4)
        kp / kd:           (n_arm,) PD 增益 (参考 JointPositionController 默认).
        check_success:     回调, 任务是否成功 (= LIBERO env 的成功判定).
        max_steps:         安全上限; None = T.

    Returns:
        exec_steps: 实际 sim 步数.
        success:    执行中是否判定成功.
    """
    arm_qpos_adr = np.asarray(arm_qpos_adr, dtype=int)
    arm_qvel_adr = np.asarray(arm_qvel_adr, dtype=int)
    arm_actuator_ids = np.asarray(arm_actuator_ids, dtype=int)
    T = dense_q.shape[0]
    limit = T if max_steps is None else min(T, max_steps)

    success = False
    steps = 0
    for t in range(limit):
        q = sim.data.qpos[arm_qpos_adr]
        qd = sim.data.qvel[arm_qvel_adr]
        tau = kp * (dense_q[t] - q) + kd * (dense_qvel[t] - qd)
        sim.data.ctrl[arm_actuator_ids] = tau
        set_gripper_ctrl(sim, dense_gripper[t])
        sim.step()
        steps += 1
        if check_success():
            success = True
            break
    return steps, success


# ---------------------------------------------------------------------------
# 编排器: delta-EEF chunk → 加速执行 → RoboTwin 兼容 info dict
# ---------------------------------------------------------------------------


def _default_retime_fn():
    """懒加载 RoboTwin 的 retime_chunk (复用同一 TOPPRA, 设计文档 §3 第 4 步)."""
    from robotwin.envs.robot.toppra_chunk_executor import retime_chunk  # TODO(cloud): 确认 import 路径

    return retime_chunk


def chunk_step_speedup_single_env(
    *,
    delta_chunk: np.ndarray,
    speed_action: Tuple[float, float, float],
    eef_pos: np.ndarray,
    eef_rotmat: np.ndarray,
    q_cur_arm: np.ndarray,
    gripper_cur: np.ndarray,
    # robosuite / mujoco 句柄与参数 —— 全部 worker 边界传入 (§7 TODO(cloud))
    model=None,
    scratch_data=None,
    sim=None,
    site_id: int = -1,
    arm_qpos_adr: Optional[np.ndarray] = None,
    arm_qvel_adr: Optional[np.ndarray] = None,
    arm_dof_adr: Optional[np.ndarray] = None,
    arm_actuator_ids: Optional[np.ndarray] = None,
    jnt_range: Optional[np.ndarray] = None,
    joint_vel_limits: Optional[np.ndarray] = None,
    joint_acc_limits: Optional[np.ndarray] = None,
    pos_output_max: float | np.ndarray = 1.0,
    rot_output_max: float | np.ndarray = 1.0,
    kp: Optional[np.ndarray] = None,
    kd: Optional[np.ndarray] = None,
    set_gripper_ctrl: Optional[Callable] = None,
    check_success: Optional[Callable[[], bool]] = None,
    exec_hz: int = 20,
    retime_fn: Optional[Callable] = None,
) -> dict:
    """单 env 加速执行一段 delta-EEF chunk, 返回 RoboTwin take_chunk_action 兼容 info.

    流水线 (设计文档 §3): [v 压缩] → 积分 EEF 路点 → 顺序 IK → retime_chunk →
    PD 执行. 任一环节失败 → ``status="topp_fallback"``, 预算照扣 M (坏 scale 负信号).

    info dict 键对齐 ``speedtune.execution.robotwin_executor.result_from_speedup_info``:
        ``status`` ∈ {success, topp_fallback, truncated},
        ``dense_steps`` (sim 步数), ``take_action_cnt_delta`` (= 压缩后帧数 M, 预算消耗),
        ``duration`` (TOPPRA 时长 = exec_time 来源), ``fallback_reason``.

    本函数是 §8 step 1 的版本无关编排骨架; 标 ``TODO(cloud)`` 的句柄/参数需在 venv worker
    内由 robosuite env 解析后传入 (§8 step 2-3, 另行接线).
    """
    from speedtune.execution.chunk_ops import reconstruct_chunk

    v, vel_scale, acc_scale = float(speed_action[0]), float(speed_action[1]), float(speed_action[2])
    retime_fn = retime_fn or _default_retime_fn()

    # 1) v 压缩 (纯动作序列重采样, 通用算子). M = 压缩后帧数, 即预算消耗.
    chunk = np.asarray(delta_chunk, dtype=np.float64)
    if v != 1.0:
        chunk = reconstruct_chunk(chunk, v)
    M = chunk.shape[0]

    def _fallback(reason: str) -> dict:
        return {
            "status": "topp_fallback",
            "fallback_reason": reason,
            "duration": 0.0,
            "dense_steps": 0,
            "take_action_cnt_delta": M,
        }

    # 2) 积分绝对 EEF 路点 (纯函数)
    positions, rotmats, grippers = integrate_eef_deltas(
        eef_pos, eef_rotmat, chunk, pos_output_max, rot_output_max
    )

    # 3) 顺序 seed IK → 关节路点
    q_path, ik_ok = dls_ik_sequential(
        model, scratch_data, site_id, arm_qpos_adr, arm_dof_adr,
        positions, rotmats, q_seed=q_cur_arm, jnt_range=jnt_range,
    )
    if not ik_ok:
        return _fallback("ik_failed")

    # 4) retime_chunk (复用 RoboTwin TOPPRA, 关节空间, vel/acc_scale 同义)
    retimed = retime_fn(
        current_state_arm=np.asarray(q_cur_arm, dtype=np.float64),
        chunk_arm=q_path,
        current_gripper=np.asarray(gripper_cur, dtype=np.float64),
        chunk_gripper=grippers.reshape(M, -1),
        joint_vel_limits=joint_vel_limits,
        joint_acc_limits=joint_acc_limits,
        vel_scale=vel_scale,
        acc_scale=acc_scale,
        exec_hz=exec_hz,
    )
    if retimed.get("status") != "success":
        return _fallback(f"toppra_{retimed.get('return_code', 'fail')}")

    # 5) PD 兜底执行
    exec_steps, success = execute_joint_trajectory_pd(
        sim, arm_qpos_adr, arm_qvel_adr, arm_actuator_ids,
        dense_q=retimed["dense_arm_pos"],
        dense_qvel=retimed["dense_arm_vel"],
        dense_gripper=retimed["dense_gripper"],
        set_gripper_ctrl=set_gripper_ctrl,
        kp=kp, kd=kd, check_success=check_success,
    )

    return {
        "status": "success",
        "fallback_reason": None,
        "duration": float(retimed["duration"]),
        "dense_steps": int(exec_steps),
        "take_action_cnt_delta": M,
        "success_obs": bool(success),
    }
