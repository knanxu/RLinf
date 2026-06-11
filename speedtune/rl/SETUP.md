# 速度控制模块（Rainbow DQN）云端配置与启动

> 本文档只覆盖**速度控制模块**（`speedtune/rl/`，RoboTwin 加速）的环境配置与启动。
> 所有命令从 **RLinf 仓根**执行。VLA 的 SFT/RL 微调见 `docs/drift_dbpo_bpo_migration.md`；
> LIBERO 加速（方案 1，尚未接线）见本文末尾 + `speedtune/docs/robosuite_toppra_design.md`。

---

## 0. 架构：哪些东西在跑（先理解，避免配错）

```
云端 GPU 主机
├─ openpi pi0.5 server   (openpi venv)   serve_policy.py, 监听 0.0.0.0:8000
│     冻结的 pi0.5, 经 websocket 暴露 infer_with_hidden (chunk + cond_emb)
│
└─ 速度模块 (RLinf 仓)   python -m speedtune.rl.rainbow.train
      ├─ WebsocketClient(127.0.0.1:8000)  ← 连上面的 server
      ├─ RoboTwin sim (经 ROBOTWIN_PATH 接入, take_chunk_action + TOPPRA)
      └─ Rainbow DQN 学 (v, vel_scale, acc_scale)
```

**关键认知**：速度模块走 **websocket 连一个 served pi0.5**（不是 RLinf 标准的 in-process
模型）。所以它需要一个**额外的 pi0.5 server**，这一步 `drift_dbpo_bpo_migration.md`
（那是 in-process DBPO+BPO）**没有**。把它改成 RLinf 原生 rollout 是 PLAN 的 P2，尚未做。

需要三样东西：① RoboTwin sim（经 ROBOTWIN_PATH）② 一个 pi0.5 checkpoint ③ openpi server。

---

## 1. 环境基座

RLinf 标准安装提供 RoboTwin sim 的依赖（sapien/mplib/curobo/pytorch3d/toppra…）。两条路：

**(推荐) Docker** —— 与迁移文档同一镜像，顺带满足 VLA SFT/RL：
```bash
cd $WORKSPACE/RLinf
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-robotwin \
    -t rlinf:embodied-robotwin .
# 进容器, 让挂载的 fork 覆盖镜像里的版本:
#   source switch_env openpi
#   pip install -e /workspace/openpi   (drift 分支: infer_with_hidden)
#   pip install -e /workspace/RLinf
```

**(或) bare-metal**：
```bash
bash requirements/install.sh embodied --env robotwin --model openpi
```
> `--env robotwin` 只装 **sim 依赖**，**不 clone RoboTwin 仓本体**（见第 2 步）。

---

## 2. RoboTwin 仓 + ROBOTWIN_PATH（速度模块必需）

速度模块经 `import envs`（RoboTwin 的任务类）+ `task_config/` + `data/` 接入 sim，
全靠 **ROBOTWIN_PATH 指向 RoboTwin 仓根**。

```bash
cd $WORKSPACE
git clone https://github.com/<your>/RoboTwin.git -b RLinf_support_speedtune RoboTwin

export ROBOTWIN_PATH=$WORKSPACE/RoboTwin          # 必须指向仓根 (内含 envs/ task_config/ data/)
export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH       # 使 `import envs` 解析
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl         # 云端 headless 渲染

# RoboTwin 仿真 assets (3D 资产):
cd $ROBOTWIN_PATH && bash script/_download_assets.sh    # 脚本名以仓内为准
```

---

## 3. pi0.5 checkpoint

速度模块在一个**冻结的 pi0.5** 上学加速，需要一个 ckpt 来 serve。两条路：

**(a) 快速：用公开 SFT ckpt**
```bash
pip install huggingface-hub          # 国内: export HF_ENDPOINT=https://hf-mirror.com
hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
    --local-dir $WORKSPACE/checkpoints/pi05-robotwin-adjust_bottle
```

**(b) 自己 SFT**：见 `drift_dbpo_bpo_migration.md` §3.2(b) + §7（drifting SFT 产物）。

---

## 4. 起 openpi pi0.5 server（速度模块特有的一步）

在 **openpi venv** 里、从 **openpi 仓**起 server，监听 0.0.0.0:8000：

```bash
cd $WORKSPACE/openpi
/opt/venvs/openpi/bin/python scripts/serve_policy.py \
    --port 8000 policy:checkpoint \
    --policy.config=pi05_aloha_robotwin_drifting_<task> \
    --policy.dir=$WORKSPACE/checkpoints/pi05-robotwin-<task>/<...>/29999
```

**或**用 RoboTwin 仓的 compose（一把起 server，env 变量传 ckpt）：
```bash
cd $WORKSPACE/RoboTwin
export POLICY_CONFIG=pi05_aloha_robotwin_drifting_<task>
export POLICY_DIR=/openpi_assets/checkpoints/$POLICY_CONFIG/default/29999
docker compose -f docker/compose.yml up openpi_server      # 常驻
```

> 首次推理要 warmup JAX/PyTorch，3–10s 属正常；后续稳定百 ms 级。

---

## 5. 跑：可行性 → 训练 → 评估（全部从 RLinf 根）

每个 shell 先确保第 2 步的 `ROBOTWIN_PATH` / `PYTHONPATH` / `MUJOCO_GL` 已 export。

```bash
cd $WORKSPACE/RLinf

# ① CPU 单测 (不需 sim/server, 验证代码完好)
python3 speedtune/tests/test_speed_control.py
python3 speedtune/tests/test_execution.py

# ② 加速可行性冒烟 (无 RL, 需 RoboTwin sim + 已采 50 条专家数据; 不需 server)
#    若加速档 success 保持、wall_time 下降 -> TOPPRA 加速物理可行, 再训才有意义
python speedtune/tests/smoke_replay_expert.py --task_name <task> --task_config smoke_test

# ③ Rainbow 速度模块训练 (需第 4 步的 server 在跑)
python -m speedtune.rl.rainbow.train \
    --task_name <task> --task_config demo_clean \
    --server_host 127.0.0.1 --server_port 8000
#    产物: speedtune/rl/runs/<run>/  ; TensorBoard 看 success_rate / wall_time / 速度档分布

# ④ 训练后对比评估 (trained Rainbow vs default baseline)
python -m speedtune.rl.eval_compare \
    --ckpt speedtune/rl/runs/<run>/rainbow_step20000.pt \
    --task_name <task> --task_config demo_clean \
    --server_host 127.0.0.1 --server_port 8000
```

调速度网格 / 奖励：`speedtune/rl/rainbow/config.py`（动作网格、C51、训练循环）
与 `speedtune/rl/config.py`（动作空间 / 奖励 / env / server）。

---

## 6. LIBERO 加速（独立的一条线，更轻）

LIBERO 用 robosuite/MuJoCo，与 RoboTwin sim **无关**，**不需要 ROBOTWIN_PATH / server**。
当前 LIBERO 加速（方案 1：运行时 IK）**尚未接线**，只能跑 OSC 约定验证脚本：

```bash
# 基座: libero + robosuite + mujoco
bash requirements/install.sh embodied --env liberopro      # 或 liberoplus / maniskill_libero

# 验证 (CPU 纯函数, 任何机器):
python3 speedtune/tests/test_robosuite_speedup.py

# 云端 (需真实 libero env): dump OSC 约定 + 实测验证 EEF 积分
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python3 speedtune/tests/cloud_check_libero_integration.py --suite libero_spatial --task 0
```

把 `cloud_check` 的输出贴回，据此填 `rlinf/envs/libero/speedup.py` 的 `TODO(cloud)`
参数，再做 `libero_env.chunk_step_with_speed` + venv 接线（§8 step 2-3）才能训。

---

## 7. 常见坑

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'envs'` | `ROBOTWIN_PATH` 没设或没进 `PYTHONPATH`；须指向 RoboTwin **仓根** |
| `ChunkSpeedupEnv needs the RoboTwin repo: set ROBOTWIN_PATH` | 同上 |
| 训练卡在连 server / 连接拒绝 | 第 4 步的 pi0.5 server 没起，或 `--server_host/--server_port` 不符 |
| `Failed to open X display` | 没设 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`；或 task_config 的 `render_freq` 设 0 |
| 首个 chunk 卡 3–10s | pi0.5 server 首次推理 warmup，正常 |
| `cond_emb dim mismatch` | server 的 pi0.5 与 `EnvConfig.cond_emb_dim`(2048) 不符，检查 ckpt/config |
| 单卡显存紧 | server 与 sim 分卡：server `CUDA_VISIBLE_DEVICES=0`、训练 `=1`；或 Rainbow `--device cpu` |
