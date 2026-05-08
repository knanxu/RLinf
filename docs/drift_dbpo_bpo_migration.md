# Migrating drifting Pi0.5 into RLinf for DBPO + BPO

This doc describes what the `drift-dbpo-bpo` branch adds to RLinf and how to
stand it up against your existing openpi `zj-humanoid-drifting` stage-1
checkpoints on RoboTwin. Target audience: you.

## What this branch adds

All changes live inside this fork. No modifications to the upstream openpi are
made from *this* branch; the companion openpi changes are in your
`zj-humanoid-drifting` branch over at `/home/xukainan/openpi`.

```
rlinf/models/embodiment/openpi/openpi_dbpo_action_model.py   [new]
    OpenPi0DBPOConfig
    OpenPi0DBPOForRLActionPrediction
        - sample_actions()       -> 1-NFE drift + Gaussian reparam
        - get_log_prob_value()   -> exact Gaussian logprob on (z, a)
    LogStdHead                   -> zero-weight + log(init_std) bias

rlinf/models/embodiment/openpi/__init__.py                   [patched]
    Dispatches on cfg.openpi.noise_method == "drift_dbpo" so we route into the
    new model. Flow-matching path is untouched.

rlinf/algorithms/losses_bpo.py                               [new]
    @register_policy_loss("bpo_actor")
    @register_policy_loss("bpo_actor_critic")
    BPO advantage-weighted TV toward the analytic bounded-ratio optimum
    (Ao et al. 2026, Eq. 16). Critic side reuses compute_ppo_critic_loss.

rlinf/algorithms/__init__.py                                 [patched]
    Imports losses_bpo so the register_policy_loss decorators fire.

rlinf/workers/actor/fsdp_actor_worker.py                     [patched]
    Adds bpo_epsilon / bpo_lambda / bpo_alpha1 to the kwargs packed into
    policy_loss(). PPO losses ignore them (**kwargs), BPO consumes them.

examples/embodiment/config/
    robotwin_adjust_bottle_dbpo_openpi_pi05.yaml             [new, DBPO+PPO]
    robotwin_adjust_bottle_dbpo_bpo_openpi_pi05.yaml         [new, DBPO+BPO]
```

## Required openpi changes

`zj-humanoid-drifting` branch, single surgical edit:

```
src/openpi/models_pytorch/pi0_pytorch.py
    _sample_actions_drifting(..., return_hidden=False)
      If return_hidden is True, returns (mean, suffix_out, cond_emb) instead
      of just mean. suffix_out feeds the RLinf value head;
      cond_emb = mean-pool over valid prefix tokens feeds LogStdHead.
```

That's it. No changes to the JAX-side drifting code, no changes to
`_forward_drifting`, no changes to the existing drifting configs.

## Install path

```bash
# 1. The openpi branch must be in editable install inside the RLinf venv.
cd /home/xukainan/openpi
git checkout zj-humanoid-drifting
pip install -e .

# 2. This RLinf branch.
cd /home/xukainan/RLinf
git checkout drift-dbpo-bpo
pip install -e .
```

Sanity check the registrations:

```bash
python - <<'PY'
import rlinf.algorithms  # triggers all @register_* decorators
from rlinf.algorithms.registry import LOSS_REGISTRY
assert "bpo_actor" in LOSS_REGISTRY
assert "bpo_actor_critic" in LOSS_REGISTRY
print("BPO loss registered:", sorted(LOSS_REGISTRY.keys()))
PY
```

## Checkpoint migration

The RLinf openpi loader (`rlinf/models/embodiment/openpi/__init__.py`) accepts
one of:

1. FSDP-saved `<ckpt>/model_state_dict/full_weights.pt` (runner's own format).
2. FSDP-saved `<ckpt>/actor/model_state_dict/full_weights.pt`.
3. openpi-native `<ckpt>/*.safetensors`.

Your stage-1 drifting runs save safetensors. Point `actor.model.model_path`
at the relevant checkpoint step directory, e.g.:

```
checkpoints/pi05_aloha_robotwin_drifting_adjust_bottle/drifting_v1/13000/
```

The directory must also contain `assets/<asset_id>/norm_stats.json` for
`_checkpoints.load_norm_stats` — this comes from the openpi training run
itself, so carry it over verbatim.

`load_state_dict(strict=False)` means the value head and log-std head
(which do not exist in the BC checkpoint) are initialised randomly. That is
intentional: both heads train from scratch during the RL phase. Set
`actor.optim.critic_warmup_steps > 0` so the value head converges before the
actor starts moving.

## RoboTwin env paths

Configure the two path placeholders in the yaml files:

```yaml
env.train.assets_path: "/path/to/robotwin_assets"
env.eval.assets_path:  "/path/to/robotwin_assets"
actor.model.model_path: "/path/to/openpi_drifting_ckpt/<task>"
```

Shell env vars (set by `run_embodiment.sh`):
  - `ROBOTWIN_PATH` — root of your RoboTwin install.
  - `REPO_PATH` — auto-set to RLinf root.

## Running

```bash
cd /home/xukainan/RLinf
# PPO-clip variant first (establishes that the adapter is producing sane logprobs)
bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dbpo_openpi_pi05 ALOHA

# Once that shows improvement, swap to BPO
bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dbpo_bpo_openpi_pi05 ALOHA
```

To port to the other four drifting configs
(`shake_bottle`, `open_microwave`, `stack_bowls_two`, `put_object_cabinet`,
`handover_block`) duplicate the yaml and swap the task-name references in the
`defaults:` block and `env.*.assets_path` and `actor.model.model_path`.

## DBPO knobs (where each one comes from)

Read: Gao et al. "Drift-Based Policy Optimization", Sec. IV.B.

| Knob | Default here | Paper setting | What moving it does |
|---|---|---|---|
| `dbpo_init_log_std` | -3.5 | log(0.03) = -3.51 | Initial sigma. Lower = safer, higher = more exploration. |
| `dbpo_min_logprob_std` | 0.03 | 0.03 | Lower bound on logprob sigma; avoids log(0). |
| `dbpo_max_logprob_std` | 0.30 | 0.10 (D4RL) / 0.12 (RoboMimic) | Upper bound; loosen if policy won't explore. |
| `dbpo_min_sampling_std` | 0.03 | same | Floor for *sampling* std, independent of logprob sigma. |
| `dbpo_randn_clip` | 3.0 | 3.0 | Truncates Gaussian noise at ±3 sigma per dim. |
| `dbpo_freeze_logstd_cond` | True | False (DBPO paper) | True = per-dim bias only; False = learned per-state. Start True. |
| `dbpo_anchor_coef` | 1.0 | 1.0 | Anchor loss (Eq. 16). See "Anchor loss" below. Set to 0 to disable. |

## BPO knobs

Read: Ao et al. "Bounded Ratio Reinforcement Learning", Sec. 4.

| Knob | Default | Paper setting | Notes |
|---|---|---|---|
| `bpo_epsilon` | 0.2 | 0.2–0.3 | Bounded-ratio band. Same role as PPO `clip_ratio_*`. |
| `bpo_lambda` | 1e-3 | 1e-3 for MuJoCo, 1e-4 for LLM | Temperature of tanh sweep; smaller -> harder edges. |
| `bpo_alpha1` | 0.0 | 0.0 | Additive weight in TV loss. Paper ablation: 0 is fine. |

Crucial detail: BPO config sets `algorithm.normalize_advantages: False`. This
matters because `bpo_lambda` is dimensional — if we divide advantages by
their std before feeding them into `sigmoid(adv / lambda)` we destroy the
intended scale of the target ratio. Leave advantage normalization off.

## Anchor loss (Gao et al. Eq. 16)

Wired through `actor.model.openpi.dbpo_anchor_coef` (float, default 1.0). When
set, the actor worker snapshots the BC checkpoint with
`retrieve_model_state_dict_in_cpu` at init, and every minibatch runs a second
forward under those frozen weights via `cpu_weight_swap` to obtain `mu_old`.
The loss is `dbpo_anchor_coef * MSE(mu_theta(o, z), mu_theta_bar(o, z))`,
added on top of the PPO/BPO policy+critic loss.

Cost: roughly one extra backbone forward per microbatch (no grad, so only
activations live transiently). Memory: one CPU-pinned copy of the full state
dict (shared with the KL reference path when `kl_beta > 0`). On pi0.5 this is
~4 GB per rank on CPU, negligible on GPU because the swap happens in place
under `torch.no_grad`.

Set `dbpo_anchor_coef: 0.0` in the yaml to disable and rely on a tight ratio
clip alone. Paper ablation (Table II, "w/o anc.") shows +15 pts success on
RoboMimic when anchor is on, so leave it on unless you're actively probing
the no-anchor regime.

## Weight syncer note

The yaml uses `weight_syncer/patch_syncer`, same path RLinf's flow-matching
PPO already uses. The syncer only sees parameter dicts, so the drift model's
new `logstd_head` flows through unchanged. If rollout / actor processes
ever diverge on the value head or log-std bias, re-check by running
`rlinf/hybrid_engines/weight_syncer/patch_syncer.py` with logging enabled.

## Quick self-test without running a full job

A minimal shape check you can run on CPU:

```python
import torch
from rlinf.algorithms.registry import LOSS_REGISTRY

bsz, chunks, dim = 4, 50, 14
logprobs = torch.randn(bsz).float()
old_logprobs = torch.randn(bsz).float()
advantages = torch.randn(bsz).float()
returns = torch.randn(bsz).float()
values = torch.randn(bsz).float()
prev_values = torch.randn(bsz).float()

fn = LOSS_REGISTRY["bpo_actor_critic"]
loss, metrics = fn(
    logprobs=logprobs, old_logprobs=old_logprobs, advantages=advantages,
    returns=returns, values=values, prev_values=prev_values,
    value_clip=0.2, huber_delta=10.0,
    bpo_epsilon=0.2, bpo_lambda=1e-3, bpo_alpha1=0.0,
)
print(float(loss), sorted(metrics.keys()))
```

## Expected first-pass failure modes

1. **`logstd_head` doesn't exist in checkpoint -> warning on strict=False**.
   Expected. Sigma starts at `exp(-3.5) ~ 0.030`.

2. **Value head explained variance stays near 0 for first few hundred steps**.
   Expected — randomly initialised, hence `critic_warmup_steps: 200`.

3. **Rollout returns near-deterministic trajectories**. Expected when
   `dbpo_freeze_logstd_cond: True` and `dbpo_init_log_std: -3.5`: the actor
   is very close to the BC policy. Raise `dbpo_max_logprob_std` or drop the
   init log-std to increase exploration if you see success rate plateau at
   the BC baseline.

## End-to-end pipeline: base -> drifting SFT -> DBPO/BPO RL

This branch only provides the Stage-2 RL glue. Stage-0 base download, Stage-1
drifting SFT, and data preparation all sit outside the changes. Here is the
full recipe in one place so you do not have to re-derive it. Commands assume
the official Docker image — see the "Docker" section below for setup.

```
  Stage 0        Stage 1                        Stage 2 (this branch)
  +-------+      +------------------------+     +-------------------------+
  |pi05   |----->|drifting SFT on RoboTwin|---->|DBPO/BPO RL on RoboTwin |
  |base   |      |expert demonstrations   |     |(PPO clip or BPO)       |
  |(JAX)  |      |(use_drifting_loss=True)|     |(drift_dbpo sampling)   |
  +-------+      +------------------------+     +-------------------------+
    |                      |                               |
    v                      v                               v
  jax->pytorch       pytorch ckpt w/            pytorch ckpt, 1-NFE
  converter          drifting decision          drift inference + Gaussian
                     boundary                   exploration + value head
```

### Stage 0: download and convert the base checkpoint

```bash
# 0.1 download JAX Orbax checkpoint (pi05_base is the flow-matching base;
# pi0_base exists too but we're going the pi0.5 route).
python - <<'PY'
import openpi.shared.download as dl
p = dl.maybe_download('gs://openpi-assets/checkpoints/pi05_base')
print("downloaded to", p)
PY

# 0.2 convert to PyTorch safetensors.
# config_name picks the *data config* that the converter uses to build the
# Pi0Config; pi05_aloha_robotwin matches RoboTwin with Aloha embodiment.
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
    --checkpoint_dir /opt/assets/.cache/openpi/pi05_base \
    --output_path    /workspace/checkpoints/pi05_base_pytorch \
    --config_name    pi05_aloha_robotwin \
    --precision      bfloat16
```

Result: `/workspace/checkpoints/pi05_base_pytorch/` with `*.safetensors` and an
`assets/` directory (with `norm_stats.json`). This is what Stage-1 SFT reads.

### Stage 1: drifting SFT

Use `examples/sft/config/robotwin_sft_drifting_openpi_pi05.yaml` (added on
this branch). It is `robotwin_sft_openpi_pi05.yaml` with the four drifting
knobs turned on:

```yaml
actor.model.openpi:
  use_drifting_loss: True
  drifting_gen_per_label: 4
  drifting_temperatures: [0.02, 0.05, 0.2]
  drifting_per_timestep_loss: True
```

The openpi dataclass `Pi0Config` already has these fields (your
`zj-humanoid-drifting` branch added them). RLinf's yaml override loop in
`rlinf/models/embodiment/openpi/__init__.py` copies them onto the instantiated
model config without requiring RLinf-side code changes. So at SFT time,
`PI0Pytorch.forward(observation, actions)` sees `use_drifting_loss=True` and
dispatches to `_forward_drifting`, which computes the attraction-repulsion
drift loss from `openpi/models_pytorch/drifting_util.py`.

Launch:

```bash
cd /workspace/RLinf
# Set data path first (see "RoboTwin expert data" below)
# Edit examples/sft/config/robotwin_sft_drifting_openpi_pi05.yaml:
#   data.train_data_paths: "/workspace/data/robotwin-lerobot/adjust_bottle"
#   actor.model.model_path: "/workspace/checkpoints/pi05_base_pytorch"

bash examples/sft/run_vla_sft.sh robotwin_sft_drifting_openpi_pi05
```

Output: `/workspace/results/robotwin_sft_drifting_openpi_pi05/<run-id>/checkpoints/last/`.
That directory is what Stage-2 RL reads.

### Stage 2: DBPO / BPO RL

Point the RL yaml's `actor.model.model_path` at the SFT output. Then:

```bash
# DBPO + PPO clip
bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dbpo_openpi_pi05 ALOHA

# DBPO + BPO (once PPO version is known to improve)
bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dbpo_bpo_openpi_pi05 ALOHA
```

## Docker

RLinf ships a unified Dockerfile at `docker/Dockerfile` with per-env build
targets. For RoboTwin we want `embodied-robotwin`, which installs three
Python venvs (`openvla-oft`, `openpi`, `lingbotvla`) inside the image.

### Build

```bash
cd /home/xukainan/RLinf
docker build \
    -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-robotwin \
    -t rlinf:embodied-robotwin \
    .
```

The build runs `requirements/install.sh embodied --venv openpi --model openpi
--env robotwin`, which in turn pins a specific version of openpi from PyPI.
That is fine for stock RLinf, but **your drifting changes live in your
openpi fork and are not on PyPI**. See "Using your openpi fork inside Docker"
below.

### Run

The container expects the RLinf repo to be bind-mounted at `/workspace/RLinf`
(or wherever you want) so your edits here are visible inside. A typical
single-node launch:

```bash
docker run --rm -it --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=16g \
    -v /home/xukainan/RLinf:/workspace/RLinf \
    -v /home/xukainan/openpi:/workspace/openpi \
    -v /home/xukainan/data:/workspace/data \
    -v /home/xukainan/checkpoints:/workspace/checkpoints \
    -v /home/xukainan/results:/workspace/results \
    rlinf:embodied-robotwin \
    bash
```

Inside the container:

```bash
# Switch into the openpi venv (this venv has the openpi-compatible torch,
# flash-attn, jax-for-converter, etc.)
source switch_env openpi

# Re-install the mounted RLinf as editable so your drift-dbpo-bpo branch
# takes precedence over whatever the image baked in.
pip install -e /workspace/RLinf

# Editable-install your openpi fork (zj-humanoid-drifting branch) so
# _sample_actions_drifting(return_hidden=True) is available.
pip install -e /workspace/openpi
```

Then run Stages 0-2 as shown above.

### Using your openpi fork inside Docker

Two ways. Pick one.

**Option A — bind-mount and pip install -e (recommended for iteration):**
see the `run` recipe above. The `-v /home/xukainan/openpi:/workspace/openpi`
bind plus `pip install -e /workspace/openpi` swaps the PyPI openpi for your
fork at runtime. Fast; no rebuild; you can edit code on the host and re-run
without rebuilding the image.

**Option B — bake into the image:** add a `RUN git clone -b zj-humanoid-drifting
https://github.com/N0ne1eft/openpi /opt/openpi && source switch_env openpi
&& pip install -e /opt/openpi` line after the RoboTwin install in the
`embodied-robotwin-image` stage of `docker/Dockerfile`. Only worth doing
once the openpi drifting code is stable.

### Asset cache layout in the image

`download_assets --dir /opt/assets --assets openpi` (which the Dockerfile
already runs) populates `/opt/assets/.cache/openpi/` with openpi's public
checkpoints, including `pi05_base`. The entrypoint then symlinks
`~/.cache/openpi -> /opt/assets/.cache/openpi` so `openpi.shared.download`
resolves those paths offline. So Stage-0's `maybe_download("pi05_base")`
will return `/opt/assets/.cache/openpi/pi05_base` in the container without
a second download.

RoboTwin assets (the simulator side, separate from openpi) are downloaded
by `bash script/_download_assets.sh` inside the RoboTwin repo — see the
next section.

## RoboTwin expert data

RLinf does not ship RoboTwin expert demonstrations. You generate them
from the RoboTwin simulator and convert to LeRobot v2.1 Parquet, which is
the format RLinf's SFT pipeline consumes
(`rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py`).

### 1. Clone the RoboTwin RLinf_support branch

```bash
cd /workspace
git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
cd RoboTwin
bash script/_download_assets.sh
```

The `RLinf_support` branch is the RoboTwin fork that RLinf's env wrapper
(`rlinf/envs/robotwin/robotwin_env.py`) targets. It pins API versions that
match the wrapper's expectations.

### 2. Generate expert demonstrations

RoboTwin tasks come with deterministic planners (mplib / curobo). For a
single task:

```bash
cd /workspace/RoboTwin
# Example: 100 successful adjust_bottle trajectories with aloha-agilex.
python script/run_task.py \
    --task_name adjust_bottle \
    --episode_num 100 \
    --planner_backend mplib \
    --embodiment aloha-agilex aloha-agilex 0.6 \
    --save_path /workspace/data/robotwin-raw/adjust_bottle
```

Flag names may drift across RoboTwin versions; consult
`RoboTwin/script/run_task.py --help` in the `RLinf_support` branch.
The output is RoboTwin's native episode format (HDF5 + mp4).

### 3. Convert RoboTwin -> LeRobot v2.1 Parquet

RoboTwin's `RLinf_support` branch provides a LeRobot exporter. Point it
at the raw folder:

```bash
cd /workspace/RoboTwin
python script/convert_to_lerobot.py \
    --input_dir  /workspace/data/robotwin-raw/adjust_bottle \
    --output_dir /workspace/data/robotwin-lerobot/adjust_bottle \
    --repo_id    robotwin/adjust_bottle \
    --embodiment aloha-agilex
```

(Script name is the one RoboTwin currently ships as of the 2.0 release —
check the repo for the exact entry point.)

Expected output layout:

```
robotwin-lerobot/adjust_bottle/
  meta/
    info.json
    episodes.jsonl
    stats.json
  data/chunk-000/
    episode_000000.parquet
    ...
  videos/chunk-000/
    observation.images.cam_high/
    observation.images.cam_left_wrist/
    observation.images.cam_right_wrist/
```

RLinf's `LeRobotAlohaDataConfig` expects these camera keys
(`cam_high`, `cam_left_wrist`, `cam_right_wrist` in its RepackTransform).
If the exporter produces a different naming, either pass
`--camera_key_map` to the exporter or override the repack transform in
`robotwin_aloha_dataconfig.py`.

### 4. Point SFT at the converted data

```yaml
# examples/sft/config/robotwin_sft_drifting_openpi_pi05.yaml
data:
  train_data_paths: "/workspace/data/robotwin-lerobot/adjust_bottle"
```

### 5. Shortcut: pre-trained SFT checkpoints

If you only want to verify the Stage-2 DBPO/BPO RL loop first and collect
demos later, use one of RLinf's public SFT models on HuggingFace as the
starting point:

```bash
pip install huggingface-hub
# Pi0.5 + RoboTwin adjust_bottle, SFT'd on official expert data. No drifting.
hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
    --local-dir /workspace/checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle

# For mainland China faster mirror:
# export HF_ENDPOINT=https://hf-mirror.com
```

Note: these public checkpoints were SFT'd with flow matching, *not*
drifting loss. They work as a Stage-2 starting point but the Stage-2
deployment will not be strict 1-NFE — the model still expects multi-step
flow denoising at inference. For true 1-NFE DBPO you must run Stage-1 with
`use_drifting_loss: True`.

### 6. Which RoboTwin tasks are usable

RLinf currently ships env yamls for:

```
adjust_bottle, beat_block_hammer, click_bell, handover_block, lift_pot,
move_can_pot, pick_dual_bottles, place_container_plate,
place_empty_cup, place_shoe
```

(in `examples/embodiment/config/env/`). Our drift-dbpo-bpo configs cover
`adjust_bottle` and `handover_block` — the two that overlap with openpi's
drifting stage-1 configs. Any of the ten tasks above can be plugged in by
copying one of the four provided yamls and changing the `env/robotwin_*@`
references plus the `model_path`.

Not listed: `shake_bottle`, `open_microwave`, `stack_bowls_two`,
`put_object_cabinet`. These exist in RoboTwin itself but RLinf has not
shipped env yamls for them yet. Adding one is mechanical — copy
`examples/embodiment/config/env/robotwin_adjust_bottle.yaml` and change
the `task_config.task_name` / `step_lim` fields.

## JAX vs PyTorch drifting-loss parity

RLinf only uses the PyTorch path, but for the record the two implementations
in openpi (`openpi/models/drifting_util.py` and
`openpi/models_pytorch/drifting_util.py`) are numerically equivalent. I
diffed them line-by-line on `zj-humanoid-drifting` after the latest changes:

* Same multi-scale R-list (`(0.02, 0.05, 0.2)`), same scale normalisation
  (`sqrt(clip(scale / sqrt(S), 1e-3))`), same diagonal mask (block_mask with
  mask_val=100), same softmax-symmetrisation
  (`sqrt(softmax(logits, -1) * softmax(logits, -2))`), same force accumulator
  divided by `sqrt(clip(f_norm, 1e-8))`, same MSE on
  `(gen_scaled - stop_grad(goal_scaled))`.
* The only divergence is stop-gradient placement: JAX uses
  `jax.lax.stop_gradient(...)` at the goal and scale_inputs, PyTorch wraps
  the goal construction in `torch.no_grad()` + `.detach()` at the MSE site.
  These are equivalent.
* Callers: JAX `_compute_loss_drifting` returns
  `jnp.full((B, action_horizon), loss)` for compatibility with flow-matching
  loss shape; PyTorch `_forward_drifting` returns `loss.mean().unsqueeze(0)`.
  The reduced scalar is what the SFT trainer consumes, so the shape
  difference does not leak.

Bottom line: you can train with either backend and the decision boundary is
the same.

