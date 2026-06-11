# Chunk-Speedup RL (Rainbow DQN)

> **已移植到 RLinf。** 本速度控制模块现在活在 `RLinf/speedtune/rl/`，所有命令从
> **RLinf 仓根**执行 (`python -m speedtune.rl.rainbow.train ...`)。它仍通过 websocket 连
> 一个 openpi 起的 pi0.5 server，并经 **`ROBOTWIN_PATH`** 接入 RoboTwin sim
> (`export ROBOTWIN_PATH=/path/to/RoboTwin` 且加进 `PYTHONPATH`，使 `import envs` 解析)。
> 下文关于 openpi server / docker 部署的说明仍然适用，只是入口改为从 RLinf 启动。

用 Rainbow DQN (C51 + dueling + PER + n-step + noisy) 训一个 agent, 看当前
pi0.5 推理出的 chunk 条件下, 选什么 `(v, vel_scale, acc_scale)` 三元组能最快把
任务做完而不掉成功率. (此前的 SAC 连续控制版本已废弃删除, 算法专属超参见
`rl/rainbow/config.py`, 共享的 env/奖励/动作空间配置见 `rl/config.py`.)

- 动作: `(v, vel_scale, acc_scale)` 离散网格 (Rainbow 用离散动作), 边界
  `v∈[0.8,1.5]`, `vel∈[1.0,2.0]`, `acc∈[1.0,4.0]` (可在 `rl/rainbow/config.py` 改)
- 状态: pi0.5 action expert 的 mean-pooled prefix features (2048D) +
  上一步动作 (3D) + cnt 进度 (1D) + 上一步是否 topp fallback (1D) = 2053D
- Reward (Plan A 非负奖励):
  ```
  r_v = α_v * v^β_v + α_vs * vel_scale^β_vs + α_as * acc_scale^β_as
  r_task = 1 if chunk 执行后 check_success else 0
  r = r_v + r_task
  ```
  TOPP fallback / crash 时屏蔽 `r_v`, 给 `0` penalty (预算消耗本身即隐性惩罚).


## 部署拓扑

训练全在云端 GPU 主机上跑, 本机只用来看日志.

```
云端 GPU 主机
└─ docker compose (docker/compose.yml), 一个 image 起两个容器:
     ├─ openpi_server   [venv: /opt/venvs/openpi, py3.11]
     │    serve_policy.py, 监听 0.0.0.0:8000
     └─ robotwin_train  [venv: conda RoboTwin, py3.10]
          speedtune/rl/rainbow/train.py
            └─ WebsocketClient(127.0.0.1:8000)  ← 宿主回环, 无额外延迟
            └─ RoboTwin Sapien sim (headless, EGL)
            └─ Rainbow DQN on cuda

本机 (开发/监控)
└─ ssh 进去看 tmux 日志 / 端口转发 tensorboard
```

两个容器共享同一个 image (`speedtune:latest`) —— image 里同时装了:
- `/opt/venvs/openpi` (uv 管理, py3.11, flax/jax/torch)
- `conda env RoboTwin` (py3.10, sapien/mplib/curobo/pytorch3d + websockets)

各自激活各自的 venv 跑不同入口, 互不干扰.
本机不跑推理, pi0.5 权重也不用下到本机. 云端 GPU ≥ 24GB.


## Step 1: fork 仓库

- fork **RoboTwin** (本仓库)
- fork **openpi** 的带 drift model 和 `infer_with_hidden` 的分支

两个 fork push 上去.


## Step 2: 云端拉代码 + 放 checkpoint

```bash
# 要求两个仓库 side-by-side
mkdir -p ~/work && cd ~/work
git clone <your-openpi-fork>.git openpi
git -C openpi checkout <your-drifting-branch>
git clone <your-RoboTwin-fork>.git RoboTwin

# 放 checkpoint (默认挂到容器里的 /openpi_assets)
mkdir -p ~/.cache/openpi/checkpoints
# 把 ckpt 同步过来, 例如:
#   ~/.cache/openpi/checkpoints/pi05_aloha_robotwin_drifting_shake_bottle/default/29999
```


## Step 3: 一把起 server + 训练

```bash
cd ~/work/RoboTwin

# 选任务对应的 ckpt config
export POLICY_CONFIG=pi05_aloha_robotwin_drifting_shake_bottle
export POLICY_DIR=/openpi_assets/checkpoints/$POLICY_CONFIG/default/29999

# 默认 server 和训练都拿 GPU 0; 两张卡就分开:
# export OPENPI_GPU=0 ROBOTWIN_GPU=1

docker compose -f docker/compose.yml up --build
```

第一次 build 会装 flax/jax/torch/sapien/pytorch3d/curobo 等, 估计 20-40 分钟.
后续启动 <1 分钟. server 会先起, `robotwin_train` 会轮询 `127.0.0.1:8000`
通了才启动训练.


## Step 4: 调试 / 只起 server

只起 server, 手动进训练容器跑:

```bash
docker compose -f docker/compose.yml up --build openpi_server
# 另开一个 shell
docker compose -f docker/compose.yml run --rm robotwin_train bash
# 容器里:
conda activate RoboTwin
python -m speedtune.rl.rainbow.train --task_name shake_bottle --task_config smoke_test
```

容器里也可以直接走 openpi venv 跑别的 openpi 脚本:

```bash
docker compose -f docker/compose.yml run --rm openpi_server bash
# 容器里:
/opt/venvs/openpi/bin/python -c "import openpi, flax, jax; print('ok')"
```


## Step 5: 本机看日志 (可选)

```bash
# 5a. 直接 ssh 进去看 docker 日志
ssh <user>@<云端IP> 'cd work/RoboTwin && docker compose -f docker/compose.yml logs -f robotwin_train'

# 5b. 把云端 tensorboard 端口转发回本机
ssh -N -f -L 6006:127.0.0.1:6006 <user>@<云端IP>
# 云端起 tensorboard (容器外, 直接读挂载的 runs/):
ssh <user>@<云端IP> 'cd work/RoboTwin && tensorboard --logdir speedtune/rl/runs --port 6006'
# 本机浏览器打开 http://127.0.0.1:6006
```


## 如果要在本机小规模 dev (仍让 pi0.5 在云端)

保留 SSH 端口转发方案, 本机 train.py 不需要改任何东西
(默认 `server_host=127.0.0.1`):

```bash
# 云端把 server 跑起来 (用上面 Step 3 的 compose 也行)
# 本机:
ssh -N -f -L 8000:localhost:8000 <user>@<云端IP>
python -m speedtune.rl.rainbow.train   # 本机 Sapien + 云端 pi0.5
```


## 常用命令

```bash
# 只覆盖连接参数, 走云端直连 (不用隧道)
python -m speedtune.rl.rainbow.train --server_host 1.2.3.4 --server_port 8000

# 切换任务 (需要云端已加载对应 ckpt)
python -m speedtune.rl.rainbow.train \
  --task_name open_microwave \
  --task_config demo_clean
```

## 关键代码位置

| 文件 | 作用 |
|---|---|
| `envs/_base_task.py:take_chunk_action` | 执行器, 加了 `v` 参数做 chunk reconstruct |
| `envs/utils/chunk_accel.py` | `reconstruct_chunk(chunk, v)` 实现 |
| `envs/robot/toppra_chunk_executor.py` | 整段 TOPPRA 时间重参数化 |
| `speedtune/rl/env.py` | Gym-like env, 包装 RoboTwin + 远端 pi0.5 (SAC/Rainbow 共用) |
| `speedtune/rl/config.py` | 共享 env/奖励/动作空间/server 配置 |
| `speedtune/rl/rainbow/networks.py` | FactoredDuelingC51 (dueling + C51 分布式 Q) |
| `speedtune/rl/rainbow/agent.py` | Rainbow agent (C51 + PER + n-step + noisy) |
| `speedtune/rl/rainbow/buffer.py` | PER + n-step replay buffer |
| `speedtune/rl/rainbow/config.py` | Rainbow 专属超参 (C51 / 动作网格 / 训练循环) |
| `speedtune/rl/rainbow/train.py` | 训练入口 |
| `speedtune/rl/eval_compare.py` | 评估 / 速度-成功率对比 |

openpi 侧:

| 文件 | 作用 |
|---|---|
| `src/openpi/models_pytorch/pi0_pytorch.py:_sample_actions_drifting` | `return_hidden=True` 时返回 `suffix_out` + `cond_emb` |
| `src/openpi/policies/policy.py:infer_with_hidden` | Policy 级别透出 hidden |
| `src/openpi/serving/websocket_policy_server.py` | server 根据 `_return_hidden` flag 分发 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py:infer_with_hidden` | client 发带 flag 的请求 |


## 已知的坑

1. **云端 checkpoint 必须和 `--policy.config` 对得上**. drift model 的
   config 在 `src/openpi/training/config.py` 里都是
   `pi05_aloha_robotwin_drifting_<task>` 形式, 不要拿 upstream 的
   `pi05_aloha` 去 serve.
2. **WebSocket max_size**. observation 包含 3 张 RGB 图像, 一个 chunk
   推理往返大概 1~3 MB, 已经通过 `max_size=None` 关掉限制.
3. **首 chunk 延迟**. 第一次 pi0.5 推理要 warmup JAX/PyTorch, 可能
   3-10s, 后续稳定在百 ms 级 (GPU 型号而定).
4. **训练侧 policy_client 不复位**. 每个 episode 不关 websocket, 保持
   长连接, 避免每次 handshake 开销.
5. **云端 Sapien headless**. 云端 GPU 机器通常没显示器, 需要 EGL 渲染:
   ```bash
   export PYOPENGL_PLATFORM=egl
   export DISPLAY=                # 清空 X display
   ```
   如果还是报 `Failed to open X display`, 检查
   `task_config/<task>.yml:render_freq`, 把它设成 0 (关 viewer) 再跑.
6. **GPU 分片**. pi0.5 server 会吃掉绝大多数显存, RoboTwin Sapien
   默认也拿 device 0. 如果单卡紧张, 要么:
   - server 用 `CUDA_VISIBLE_DEVICES=0 docker compose ...`, RL 用
     `CUDA_VISIBLE_DEVICES=1 python -m speedtune.rl.rainbow.train`
   - 或者把 Rainbow agent 放 cpu (`--device cpu`), 只让 Sapien 和 pi0.5 抢 GPU0
7. **Docker + conda 同机**. docker 容器里 server 监听主机 8000,
   conda 宿主里 client 连 `127.0.0.1:8000`. 这条走的是 loopback, 不需要
   开外网端口. 但如果改 `network_mode: bridge`, 宿主连的就是容器 IP,
   那 config 里 `server_host` 也得改. 保持 `host` 模式最省心.
