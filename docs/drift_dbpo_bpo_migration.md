# 云端从 0 部署：drifting Pi0.5 + DBPO/BPO + speedtune（RoboTwin）

从零在云端服务器（目标：单节点 4×RTX 5880）跑通本项目：克隆三个仓 → 配环境 → 拿
checkpoint/assets → sanity 检查 → 测试训练（Stage-2 DBPO+BPO）。

> 本指南覆盖**当前可跑的 DBPO+BPO VLA-RL baseline**。speedtune 的速度模块
> （`chunk_step_with_speed`）目前是 opt-in、尚未接入 rollout（P2）；robosuite TOPPRA 见
> `speedtune/docs/robosuite_toppra_design.md`（待实现）。所以下面的"测试训练"跑的是
> 标准 DBPO+BPO，速度模块不参与。

---

## 0. 仓库与分支

| 仓 | URL | 分支 | 作用 |
|---|---|---|---|
| RLinf | `github.com/knanxu/RLinf` | **`speedtune`** | 主项目：DBPO+BPO + speedtune 框架 |
| openpi | `github.com/N0ne1eft/openpi` | **`zj-humanoid-drifting`** | drift 模型（`_sample_actions_drifting(return_hidden=True)`） |
| RoboTwin | `github.com/knanxu/RoboTwin` | **`RLinf_support_speedtune`** | benchmark（RLinf_support + 整段 TOPPRA 加速） |

**前置（在本地仓执行一次）**：RLinf 的 `speedtune` 与 RoboTwin 的 `RLinf_support_speedtune`
目前只在本地，需先推到各自 fork，云端才能 clone：

```bash
git -C /home/xukainan/RLinf    push -u origin speedtune
git -C /home/xukainan/RoboTwin push -u origin RLinf_support_speedtune   # 该分支在 worktree /home/xukainan/RoboTwin-rlinf
```
（openpi 的 `zj-humanoid-drifting` 已在 GitHub，无需 push。）

---

## 1. 克隆三个仓（云端）

```bash
export WORKSPACE=/workspace          # 按云端实际改
mkdir -p $WORKSPACE && cd $WORKSPACE

git clone https://github.com/knanxu/RLinf.git      -b speedtune                RLinf
git clone https://github.com/N0ne1eft/openpi.git   -b zj-humanoid-drifting     openpi
git clone https://github.com/knanxu/RoboTwin.git   -b RLinf_support_speedtune  RoboTwin
```

---

## 2. 环境

RLinf 依赖重（Ray ≥2.47 / torch ≥2.5 / FSDP / sapien / mujoco / toppra…）。推荐用官方
Docker 镜像，bare-metal 见 2.B。

### 2.A Docker（推荐）

```bash
cd $WORKSPACE/RLinf
# embodied-robotwin 目标镜像（内含 openpi venv、sapien、依赖）
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-robotwin \
    -t rlinf:embodied-robotwin .

# 单节点 4 卡启动，bind-mount 三个仓 + 数据/ckpt/结果目录
docker run --rm -it --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -v $WORKSPACE/RLinf:/workspace/RLinf \
    -v $WORKSPACE/openpi:/workspace/openpi \
    -v $WORKSPACE/RoboTwin:/workspace/RoboTwin \
    -v $WORKSPACE/data:/workspace/data \
    -v $WORKSPACE/checkpoints:/workspace/checkpoints \
    -v $WORKSPACE/results:/workspace/results \
    rlinf:embodied-robotwin bash
```

容器内（让挂载的 fork 覆盖镜像里 PyPI 的版本）：

```bash
source switch_env openpi                 # 切到 openpi 兼容的 venv (torch/flash-attn 等)
pip install -e /workspace/openpi         # drift fork: 提供 _sample_actions_drifting(return_hidden=True)
pip install -e /workspace/RLinf          # speedtune 分支: DBPO+BPO + speedtune
export ROBOTWIN_PATH=/workspace/RoboTwin # RLinf 的 robotwin_env 靠它找 robotwin 库
export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
```

### 2.B Bare-metal（无 Docker）

```bash
conda create -n rlinf python=3.10 -y && conda activate rlinf
cd $WORKSPACE/openpi && pip install -e .
cd $WORKSPACE/RLinf  && pip install -e .
# RoboTwin sim 依赖 (sapien / mplib / curobo / toppra)：按 RoboTwin README 装
cd $WORKSPACE/RoboTwin && pip install -e . && bash script/install.sh   # 名称以仓内为准
export ROBOTWIN_PATH=$WORKSPACE/RoboTwin
export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
```

---

## 3. Checkpoint 与 Assets

### 3.1 RoboTwin 仿真 assets（必需）

```bash
cd $WORKSPACE/RoboTwin
bash script/_download_assets.sh          # 下载 3D 资产到 assets/（脚本名以仓内为准）
```
记下 assets 路径，填进 env yaml 的 `env.*.assets_path`。

### 3.2 VLA checkpoint —— 两条路，先选快的

**(a) 快速测训练：用公开 SFT ckpt 跳过 Stage-0/1 + 数据生成**
```bash
pip install huggingface-hub
# 国内加速: export HF_ENDPOINT=https://hf-mirror.com
hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
    --local-dir /workspace/checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle
```
⚠️ 公开 ckpt 是 **flow-matching SFT、非 drifting**，能当 Stage-2 起点验证 RL 闭环，但推理
**不是严格 1-NFE**。要真正的 1-NFE drift，走 (b)。

**(b) 完整路径：Stage-0 base → Stage-1 drifting SFT（真 1-NFE）**
```bash
# Stage 0: 下 pi05_base (JAX) 并转 PyTorch safetensors
python - <<'PY'
import openpi.shared.download as dl
print(dl.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))
PY
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
    --checkpoint_dir <pi05_base下载路径> \
    --output_path    /workspace/checkpoints/pi05_base_pytorch \
    --config_name    pi05_aloha_robotwin --precision bfloat16

# Stage 1: drifting SFT (需 RoboTwin 专家数据转 LeRobot, 见 §6)
# 改 examples/sft/config/robotwin_sft_drifting_openpi_pi05.yaml 的
#   data.train_data_paths 与 actor.model.model_path=/workspace/checkpoints/pi05_base_pytorch
bash examples/sft/run_vla_sft.sh robotwin_sft_drifting_openpi_pi05
# 产物: /workspace/results/robotwin_sft_drifting_openpi_pi05/<run-id>/checkpoints/last/
```
checkpoint 目录须含 `assets/<asset_id>/norm_stats.json`（来自 openpi 训练，原样带过来）。

---

## 4. Sanity 检查（不跑全量任务）

```bash
# 4.1 BPO loss 注册
python - <<'PY'
import rlinf.algorithms
from rlinf.algorithms.registry import LOSS_REGISTRY
assert "bpo_actor" in LOSS_REGISTRY and "bpo_actor_critic" in LOSS_REGISTRY
print("BPO registered:", sorted(k for k in LOSS_REGISTRY if "bpo" in k))
PY

# 4.2 drift fork 可用
python -c "from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; print('openpi drift OK')"

# 4.3 robotwin 库可导入 (含加速)
python -c "from robotwin.envs.vector_env import VectorEnv; print('robotwin VectorEnv OK')"
python -c "import importlib;m=importlib.import_module('envs._base_task');print('take_chunk_action' in dir(m.Base_Task))" 2>/dev/null || \
  echo "(take_chunk_action 在 RLinf_support_speedtune 分支的 Base_Task 上)"

# 4.4 speedtune 框架 CPU 测试
cd /workspace/RLinf
python3 speedtune/tests/test_execution.py
python3 speedtune/tests/test_speed_control.py
```

---

## 5. 配置（改 yaml 的路径）

测试用配置：`examples/embodiment/config/robotwin_adjust_bottle_dbpo_bpo_openpi_pi05.yaml`
（DBPO+BPO）或 `..._dbpo_openpi_pi05.yaml`（DBPO+PPO clip，更稳，建议先跑）。要改：

```yaml
cluster:
  num_nodes: 1
  component_placement: { actor,env,rollout: all }   # 单节点 4 卡 colocated

env:
  train: { assets_path: "/workspace/RoboTwin/assets" }
  eval:  { assets_path: "/workspace/RoboTwin/assets" }

actor:
  model:
    model_path: "/workspace/checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"  # 或 Stage-1 产物
    openpi: { noise_method: "drift_dbpo" }          # 走 DBPO 单步漂移
  optim: { critic_warmup_steps: 200 }               # value/logstd 头随机初始化, 先热身
```

DBPO 关键旋钮（`actor.model.openpi`，默认对齐论文）：`dbpo_init_log_std=-3.5`、
`dbpo_max_logprob_std=0.30`、`dbpo_freeze_logstd_cond=True`、`dbpo_anchor_coef=1.0`。
BPO（`algorithm`）：`loss_type=bpo_actor_critic`、`bpo_epsilon=0.2`、`bpo_lambda=1e-3`，
**`normalize_advantages=False`**（bpo_lambda 有量纲，归一化会破坏 target ratio 尺度）。

---

## 6. 测试训练（Stage-2 DBPO/BPO RL）

```bash
cd /workspace/RLinf
# 先 DBPO+PPO clip（确认 adapter 产出合理 logprob），有提升再换 BPO
bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dbpo_openpi_pi05 ALOHA
bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dbpo_bpo_openpi_pi05 ALOHA
```

日志/ckpt 在 `/workspace/results/<exp_name>/`。TensorBoard 看 success_rate / return /
ratio 统计 / value explained_variance。

**首跑预期（正常现象，非报错）**：
1. `logstd_head` 不在 ckpt → `strict=False` 警告。σ 从 `exp(-3.5)≈0.030` 起。
2. value head explained_variance 前几百步≈0（随机初始化 + `critic_warmup_steps`）。
3. rollout 近确定性（`dbpo_freeze_logstd_cond=True` + 小 init σ，actor 贴近 BC）。
   成功率卡在 BC baseline 时，调大 `dbpo_max_logprob_std` 或调小 init log-std 增探索。

---

## 7. （可选）生成 RoboTwin 专家数据 → LeRobot（仅 Stage-1 真 drifting 需要）

```bash
cd /workspace/RoboTwin
python script/run_task.py --task_name adjust_bottle --episode_num 100 \
    --planner_backend mplib --embodiment aloha-agilex aloha-agilex 0.6 \
    --save_path /workspace/data/robotwin-raw/adjust_bottle          # 产 HDF5+mp4
python script/convert_to_lerobot.py \
    --input_dir /workspace/data/robotwin-raw/adjust_bottle \
    --output_dir /workspace/data/robotwin-lerobot/adjust_bottle \
    --repo_id robotwin/adjust_bottle --embodiment aloha-agilex      # 转 LeRobot v2.1
```
相机键需为 `cam_high / cam_left_wrist / cam_right_wrist`（对齐
`robotwin_aloha_dataconfig.py` 的 RepackTransform）。脚本名/参数以 `RLinf_support` 分支
`--help` 为准。

可用 RoboTwin 任务（`examples/embodiment/config/env/`）：adjust_bottle、handover_block、
beat_block_hammer、lift_pot、move_can_pot、pick_dual_bottles、place_container_plate 等；
drift-dbpo-bpo 配置目前覆盖 adjust_bottle / handover_block，换任务复制 yaml 改 `env/robotwin_*@`
与 `model_path` 即可。

---

## 8. 4×5880 与排错要点

- **单节点 4 卡**：`cluster.num_nodes=1`，component_placement 用 colocated（actor/env/rollout
  共享 4 卡）。pi0.5 RL 微调用 FSDP（`actor.training_backend=fsdp`）；显存紧就调小
  `micro_batch_size` / `env.train.total_num_envs` / `num_action_chunks`。
- **anchor loss** 需一份 CPU pin 的 BC 权重副本（pi0.5 ~4GB/rank CPU），`dbpo_anchor_coef=0`
  可关（论文消融关掉掉 ~15pt 成功率，别轻易关）。
- **CPU 最小形状自测**（不起任务，验证 BPO loss）：
  ```python
  import torch; from rlinf.algorithms.registry import LOSS_REGISTRY
  b=4; f=LOSS_REGISTRY["bpo_actor_critic"]
  loss,m=f(logprobs=torch.randn(b),old_logprobs=torch.randn(b),advantages=torch.randn(b),
           returns=torch.randn(b),values=torch.randn(b),prev_values=torch.randn(b),
           value_clip=0.2,huber_delta=10.0,bpo_epsilon=0.2,bpo_lambda=1e-3,bpo_alpha1=0.0)
  print(float(loss), sorted(m))
  ```
- **weight syncer**：yaml 用 `weight_syncer/patch_syncer`；drift 的 `logstd_head`/value head 随
  参数字典正常同步，rollout/actor 头不一致时开 `patch_syncer.py` 日志排查。

---

## 9. 当前状态与后续（speedtune）

- 本部署跑的是 **DBPO+BPO VLA-RL baseline**（speedtune 速度模块未接入 rollout）。
- RoboTwin 整段 TOPPRA 加速链路已就位（`take_chunk_action` / `gen_sparse_reward_data_speedup` /
  `VectorEnv.step_with_speed`，RLinf 侧 `chunk_step_with_speed`），但需 P2 的速度策略产出
  `speed_actions` 才会被驱动。
- robosuite/LIBERO 的 TOPPRA 加速：设计见 `speedtune/docs/robosuite_toppra_design.md`（待实现）。
