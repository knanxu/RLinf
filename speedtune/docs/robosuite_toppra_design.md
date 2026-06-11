# robosuite/LIBERO 关节空间 TOPPRA 加速 — 设计文档（方案 A）

> 目标：把 RoboTwin 的"整段 TOPPRA + (v, vel_scale, acc_scale) 三旋钮"加速模型**统一**到 robosuite 系
> benchmark（LIBERO / LIBERO-plus / robomimic），让速度控制 RL 在 robosuite 上也能调速度/加速度约束。
> 状态：**设计待 review + 云端核实第 7 节 5 点后再实现**。
> 关联代码：`speedtune/execution/`（`reconstruct_chunk`/`retime_chunk` 思路、`ChunkExecResult`、
> `result_from_speedup_info`）、RoboTwin 侧 `gen_sparse_reward_data_speedup`（对照模板）。

---

## 1. 背景与选型

robosuite/LIBERO 的执行模型与 RoboTwin 本质不同：
- 动作 = 7 维 **delta-EEF**（6 EEF 增量 + 1 gripper），由 **OSC 控制器**在固定 control_freq 内部消化。
- `LiberoEnv.chunk_step` 逐帧 `self.step(action_i)` → robosuite `env.step`，**不暴露关节轨迹、无时间重参数化接口**。

要上 TOPPRA 并暴露 **vel_scale / acc_scale 两个约束旋钮**，需自己构造路径 + 选约束空间 + 接管执行。
已确认选 **方案 A：关节空间 TOPPRA**（vs 笛卡尔空间方案 B）。理由：
- 复用 RoboTwin 同一个 `retime_chunk`，**vel_scale/acc_scale 与 RoboTwin 完全同义**（缩放关节速度/加速度上限）。
- 约束在关节级有物理保证（笛卡尔 + OSC 跟踪会有跟踪误差，约束只作用于参考）。
- 框架真正统一成"一套加速模型跨 benchmark"。
代价：需要 IK（EEF→关节）+ robosuite 关节执行（A 方案的两块 robosuite 专属胶水）。

---

## 2. 架构落点（关键：外部库零改动，跑在 worker 进程内）

与 RoboTwin 对比：
- RoboTwin `VectorEnv` 是**线程池（同进程）** → 加速逻辑放进外部 robotwin 库的 `Base_Task`。
- LIBERO `venv.py` 是**多进程 worker**（`_worker` 靠 pipe 收 `(cmd, data)`），robosuite env 活在子进程 →
  关节执行（碰 `env.sim`/`robot`）**必须在 worker 内跑**，但只需操作传入的 robosuite env 对象，
  **不需要改 libero/robosuite 库本身**。

新增/改动（均在 RLinf 仓内）：
| 文件 | 改动 |
|---|---|
| `rlinf/envs/libero/speedup.py`（新） | worker 内对单个 robosuite env 做 IK + retime + 关节执行的纯函数 |
| `rlinf/envs/libero/venv.py` | `_worker` 加 `"chunk_step_speedup"` 命令；venv 加分发方法 |
| `rlinf/envs/libero/libero_env.py` | 加 `chunk_step_with_speed(chunk_actions, speed_actions)`，opt-in，不动原 `chunk_step` |
| `speedtune/execution/robosuite_executor.py` | capabilities 改回三维全 True；执行委托给上面（或仅作 schema 文档） |

---

## 3. 执行流水线（每 env，worker 内）

```
当前 EEF 位姿 q_cur + delta-EEF chunk (M, 7)
  │  [可选] reconstruct_chunk(chunk, v)            # v 压缩, 复用纯函数, 作用于 EEF 动作序列
  ├─ 积分 delta → 绝对 EEF 路点 {p_k ∈ R³, R_k ∈ SO(3)}, k=1..M
  │     注意: delta 需乘 OSC controller 的 output_max (action[-1,1]→米/弧度)
  ├─ 顺序 seed IK → 关节路点 q_0..q_M               # MuJoCo DLS, 见 §4
  ├─ retime_chunk(q, joint_vel·vel_scale, joint_acc·acc_scale)   # ★复用 toppra_chunk_executor
  │     → 密集关节 pos/vel (T, dof) + duration
  ├─ gripper: 沿路径参数插值 (同 RoboTwin retime_chunk 的 gripper 处理)
  └─ 关节执行循环 (§5): 逐 sim 步跟踪 + check_success
```

返回 info dict（与 RoboTwin take_chunk_action 同结构）：
`{status, fallback_reason, duration, dense_steps, take_action_cnt_delta, ...}` →
`result_from_speedup_info` → `ChunkExecResult`。

---

## 4. IK 选型：MuJoCo 雅可比 DLS，顺序 seed

不引新依赖（robosuite 自带 mujoco）。worker 有 `env.sim`（MjModel/MjData）：
- 在 scratch MjData 上（`mj_copyData`，**不扰动真实 sim**），从 `q_cur` 起，对每个 EEF 路点迭代：
  `mj_kinematics` → 取 EEF site 位姿 → 位姿误差 e（位置 + 姿态 axis-angle）→ `mj_jacSite` 得 J →
  `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e` → `q ← q + dq`，迭代到收敛或达上限。
- **顺序 seed**：`q_k` 用 `q_{k-1}` 初始化 → Panda 7-DOF 冗余零空间保持连续、无关节跳变（A 方案路径平滑的关键）。
- 不用 pybullet（robosuite IK_POSE 那套），避免额外耦合。
- 关节限位：迭代中 clip 到 `jnt_range`；越界/不可达 → IK 失败（§6 fallback）。

参数：λ（阻尼）~1e-2、max_iter ~20、tol ~1e-3 m / 1e-2 rad（云端调）。

---

## 5. 关节执行循环

密集关节轨迹要在 robosuite 里被跟踪。两条路：

**推荐：复用 robosuite `JOINT_POSITION` 控制器**
- 临时把该 robot 控制器换成 JointPositionController（或单独建一个绑同一 sim），密集循环：
  ```
  for t in range(T):
      controller.set_goal(dense_q[t])           # (+ dense_qvel[t] 若支持)
      tau = controller.run_controller()
      sim.data.ctrl[arm_actuators] = tau
      set_gripper_ctrl(dense_gripper[t])
      sim.step()
      if check_success(): break
  ```
- 结构与 RoboTwin `take_chunk_action` 控制循环一一对应。
- ⚠️ `set_goal/run_controller` API 在 robosuite 1.4 vs 1.5 差异大（§7.1）。

**兜底：自写关节 PD**（对版本最鲁棒，不碰 robosuite 控制器 API）
- `tau = Kp(q*-q) + Kd(q̇*-q̇)`，直接灌 `sim.data.ctrl[arm]`，`sim.step()`。
- 增益 Kp/Kd 需调（可参考 robosuite JointPositionController 默认值）。
- **建议先用兜底实现打通**（§8），稳定后再换原生控制器。

计时：`exec_time` = TOPPRA duration（同 RoboTwin），`exec_steps` = 实际 sim 步数。

---

## 6. Fallback（对齐 RoboTwin topp_fallback）

| 失败 | 处理 |
|---|---|
| IK 失败/奇异（路点不可达、J 病态、越界） | 整段 `status="topp_fallback"`，预算照扣 M（坏 scale 的负信号，r_speed 屏蔽） |
| TOPPRA 求解失败 | 同上（retime_chunk 已返回 fallback） |
| **更稳的降级（比 RoboTwin 多一层）** | speedup 整体异常 → **回退到原生 OSC 逐帧 chunk_step**，保证不崩，只是这段不加速 |

预算（`take_action_cnt`/`step_lim`）语义对齐 RoboTwin：压缩后帧数 M 即消耗预算。

---

## 7. 云端核实清单（决定具体 API 填法，实现前必查）

1. **robosuite 版本**：LIBERO 多锁 1.4.x；liberoplus/pro 可能不同。决定 `JointPositionController`
   的 `set_goal/run_controller` API（1.5 引入 `composite_controller`/`part_controllers`，结构变化大）。
2. **OSC action 缩放**：controller config 的 `output_max/input_max`（delta-EEF action[-1,1]→米/弧度）。
   积分 EEF 路点必须用它，否则路径尺度错。
3. **Panda 关节 vel/acc 限制来源**：robot model / MjModel `jnt_range`、actuator 限制；
   **acc 限制 robosuite 未必直接给** → 可能需设默认（参考 RoboTwin 从 mplib planner 取的值）。
4. **EEF site 名 + 关节/执行器索引**：`robot.eef_site_id` / site 名、arm joint indices、gripper actuator id。
5. **worker 内 mujoco IK 开销**：每 chunk × 每 env × M 路点 × DLS 迭代。评估是否需限迭代/缓存/降 M。

---

## 8. 建议实现顺序（风险递增）

1. `speedup.py`：IK（MuJoCo DLS）+ 积分 EEF 路点 + 调 `retime_chunk` + **PD 兜底执行循环**（§5 兜底）。
   先不碰 robosuite 控制器 API，对版本最鲁棒。
2. `venv.py` worker 加 `"chunk_step_speedup"` 命令 + venv 分发。
3. `libero_env.py` 加 `chunk_step_with_speed`，opt-in；info → `result_from_speedup_info` → `ChunkExecResult`。
4. 云端冒烟：speed=(1,1,1) 应≈原生执行；再扫 vel_scale/acc_scale 看加速 vs 成功率。
5. 稳定后：PD 兜底 → robosuite 原生 `JointPositionController`（若版本 API 允许，跟踪更准）。

## 9. 风险

- **最大风险：§5 关节执行的版本敏感性** → 用 PD 兜底先规避。
- **IK 平滑性/开销** → 顺序 seed + 限迭代；必要时对 M 降采样（v 压缩本就降 M）。
- **执行偏离 VLA 训练分布**：改执行会改路点间真实轨迹（路点不变），retiming 越激进越偏离 →
  正是速度 RL 要学的"速度 vs 成功率"权衡，与 RoboTwin 一致（设计动机，非 bug）。
- **统一性收益**：成功后 robosuite 与 RoboTwin 共用 `retime_chunk` + 同义 (v,vel_scale,acc_scale)，
  框架达成"一套加速模型跨 benchmark"。
