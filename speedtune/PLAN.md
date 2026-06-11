# speedtune-vla 项目重构计划 (v2)

> 目标：把「drift 版 openpi + 带 speedtune 的 RoboTwin + 速度控制 RL 模块」整合成 RLinf 风格的统一框架，
> 支持 model × benchmark × algorithm 任意组合；支持 RL 微调 VLA（已用 DBPO+BPO 实现）+ 共训速度模块。
> v2 依据：读了 Gao 2026 (DBPO) 与 Ao 2026 (BPO) 两篇论文 + 摸清作者已在 RLinf 完成的实现。

---

## 0. 决策汇总（已确认）
- Fork RLinf 作骨架；drift + speedtune 移植进 RLinf 适配器。
- 速度模块 = 独立的共训第二策略（分层/双智能体），与 VLA 各用自己算法、共享一条 rollout。
- actor-learner + 权重 sync 用 RLinf 原生（Ray worker + NCCL）；websocket 退化为部署/eval。
- 基建优先。
- **硬件**：云端 4×RTX 5880(Ada 48GB)；本地不做大显存测试（本地只做代码/重构/CPU smoke）。
- RLinf 环境云端**尚未配置**。

---

## 1. 现状盘点（关键认知）

### 1.1 VLA-RL 算法（DBPO+BPO）已基本完成 —— 但从未端到端跑过
作者在 RLinf 的 `drift-dbpo-bpo` 分支已实现生产级 DBPO+BPO：
- `rlinf/models/embodiment/openpi/openpi_dbpo_action_model.py`：`OpenPi0DBPOForRLActionPrediction`
  （`LogStdHead` + 单步漂移 `sample_actions` + 精确高斯 `get_log_prob_value`(共享 z 重放) + anchor extras）。
- `rlinf/algorithms/losses_bpo.py`：`bpo_actor` / `bpo_actor_critic`，bounded-ratio TV loss
  `|A|·|ρ-(1+ε·sign(Ã))|`，平滑版 `target_ratio=(1-ε)+2ε·sigmoid(adv/λ)`。
- anchor loss `MSE(μ, μ_frozen)` 已接进 `fsdp_actor_worker.py`。
- 4 个配置（adjust_bottle / handover_block × DBPO / DBPO+BPO）+ smoke test + 迁移文档。
- ⚠️ **仅离线 CPU smoke test，无端到端 RL 训练验证** → P0 的真正目标是首次端到端跑通。

### 1.2 两篇论文 = 两个泛化目标的设计蓝图
- **DBPO 两层结构**：模型专属 `μ=f_θ(o,z)` + 条件特征 `c_θ(o)`；模型无关的"analytic stochastic actor"
  （LogStdHead + 共享 z 精确似然 + anchor + BPO loss）。→ 直接对应「可插拔通用 drift 头」。
- **BPO bounded-ratio**：用比率界 `1-ε≤π/π0≤1+ε` 替代 KL 信任域，解析最优解
  `π*=(1+ε·tanh(Ã/2λ))π0`，需 mean/median 值函数 `μ_ψ`、`normalize_advantages=False`。

### 1.3 加速模块在 RLinf 里为零
`vel_scale/acc_scale/TOPPRA/chunk 压缩` 只在 `/home/xukainan/RoboTwin/speedtune`。移植 + 跨 benchmark 泛化是新工程。

---

## 2. 两个泛化设计

### (a) 可插拔通用 drift 头 —— 切开 DBPO 的两层
```
DriftBackbone (模型专属, 每个 VLA 各实现):
  cond_dim: int                              # 不再硬编码 "pi05" in config_name
  drift_generate(obs, z) -> (mu_chunk, cond_feat)   # 1-NFE 单步
  drift_hypotheses(obs, G) -> hyps                  # Stage-1 DBP 训练用

DriftStochasticActor(BasePolicy) (模型无关, 复用):
  LogStdHead(cond_dim, action_dim)
  sample_actions / get_log_prob_value (共享 z 精确似然) / anchor_loss
  + losses_bpo bounded-ratio
```
- 重构：`OpenPi0DBPOForRLActionPrediction` → `OpenPi0DriftBackbone`(专属) + 共享 `DriftStochasticActor`。
- 其他模型接入：① 用 DBP 漂移损失重训/微调出 one-step 骨架（每模型大头）② 实现 `drift_generate` 即复用整套 RL 适配。

### (b) 跨 benchmark 加速 —— ChunkExecutor 抽象
速度策略输出通用三维元动作 `(v, vel_scale, acc_scale)`，各 benchmark executor 解释：
```
ChunkExecutor (per benchmark):
  compress(chunk, v) -> chunk'                       # 共享默认实现 (序列重采样)
  retime(chunk', vel_scale, acc_scale) -> dense_targets   # 环境专属 (限速/限加速)
  execute(dense_targets) -> step_result                   # 环境专属控制环
  execution_cost() -> float                               # 速度奖励来源 (时间/步数)
```
- `RoboTwinChunkExecutor`：关节空间 TOPPRA（现有 speedtune 直接接，250Hz, dense_steps/250 = 真实时间）。
- `RobosuiteChunkExecutor`（LIBERO / LIBERO-plus / robomimic 共用）：OSC delta-EEF，
  **建议先任务空间重定时**（重采样 EEF 路点 + 缩放插值率），后续再可选 IK→关节→TOPPRA。

---

## 3. 优化点（结合论文）
1. **state-dependent σ**：`dbpo_freeze_logstd_cond` 由 True→False（论文用 `g_ψ(c_θ(o))`），baseline 收敛后试。
2. **executed-prefix ↔ chunk 压缩耦合（正确性命门）**：DBPO 信用分配只算实际执行的 prefix；速度模块 v 压缩
   改变"chunk 里多少被执行" → 双智能体里 VLA 的 logprob/优势**必须用速度模块压缩后真正执行的动作**，否则 ratio 算错。
3. **anchor loss 调好**（论文消融 0.90→0.75）。
4. **BPO `normalize_advantages=False` + λ 尺度**：加速度奖励 shaping 别破坏原始优势尺度。

---

## 4. 修订路线（基建优先）

### P0 — 云端环境 + DBPO+BPO 首次端到端验证
- [ ] P0.1 云端 4×5880 配 RLinf 依赖（Ray/torch/FSDP；pi0.5 RL 微调需 FSDP 跨 4 卡，评估 sim+train 共置显存）。
- [ ] P0.2 `pip -e` 接入作者 openpi(drift) 与 RoboTwin。
- [ ] P0.3 **端到端跑** `robotwin_adjust_bottle_dbpo_bpo_openpi_pi05.yaml`，确认算法真能训、成功率提升。
- [ ] P0.4 drift sanity check（同 obs 两 noise 量 action 方差，确认不塌缩）。
**产出**：DBPO+BPO 在云端跑通的端到端 baseline（项目最关键去风险）。

### P1 — 两个泛化（基建主体，本地可写代码）
- [ ] P1a 重构 drift 头为 `DriftBackbone` + `DriftStochasticActor`，去掉 pi05 硬编码；回归验证 pi05 不退化，再接一个第二模型。
- [ ] P1b 移植 speedtune → 定义 `ChunkExecutor` 抽象 + 实现 `RoboTwinChunkExecutor`；接进 RLinf robotwin env 的 chunk 执行。
**产出**：drift 头跨模型可插拔 + 加速操作有统一抽象（先支持 RoboTwin）。

### P2 — 双智能体共训 + benchmark 泛化
- [ ] P2.1 `SpeedControllerDQN(BasePolicy)`：状态=cond_emb+last_action+进度+fallback，离散 (v,vel,acc)，PER buffer。
- [ ] P2.2 rollout 钩子（`huggingface_worker.generate_one_epoch`）插速度策略推理；`RolloutResult` 加 secondary_*。
- [ ] P2.3 dual-agent 联合更新：VLA 走 on-policy BPO，speed 走 off-policy DQN，共享 rollout；reward 归因 + 两时间尺度/课程。
- [ ] P2.4 实现 `RobosuiteChunkExecutor`，让加速操作支持 LIBERO / LIBERO-plus / robomimic。
**产出**：rollout 同时改进 VLA 与速度模块 + 加速跨多 benchmark。

### P3 — 全矩阵泛化
- [ ] 配置驱动 model × benchmark × algorithm 任意组合，整理可复现 example 矩阵。

---

## 5. 风险与验证点
1. drift 塌缩（P0.4 先验证）。
2. 4×5880 显存：pi0.5 RL 微调 + sim 共置；FSDP 分片策略、是否 disaggregate sim/train（P0.1 评估）。
3. drift fork 与 RLinf DBPO 适配器接口偏差（已基本对齐，端到端跑会暴露）。
4. **executed-prefix × chunk 压缩耦合**导致 on-policy ratio 算错（P2.3 专门设计与单测）。
5. robosuite 加速：OSC delta-EEF 无关节轨迹，任务空间重定时是近似（P2.4 先近似后忠实）。
6. 双智能体耦合发散（两时间尺度 + 课程：先冻 VLA 训 speed = 现状，再共训）。
```
