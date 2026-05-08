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
| `dbpo_anchor_coef` | 0.0 | 1.0 | Anchor loss (Eq. 16). Not wired into the PPO loss here — tight clip instead. |

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

## What is *not* done yet and why

### Anchor loss (Eq. 16, `mu_theta(o,z) - mu_theta_bar(o,z)`)

Requires a frozen pretrained snapshot in memory alongside the trainable
policy. At pi0.5 scale this doubles backbone memory. DBPO paper ablation
(Table II, "w/o anc.") shows anchor gains +15 pts success on RoboMimic — so it
matters — but first we want to verify the adapter loop is correct without
this complication. Path to add later:

1. In `DBPOPPOWrapper.__init__` keep a deepcopy `actor_old` on CPU offload.
2. When computing loss, call `_drift_forward_from_tokens` once with frozen
   weights (via `torch.no_grad` + temporary `load_state_dict`) to get
   `mu_old`; drop this into an extra loss term `lambda_anchor * MSE(mu, mu_old)`.
3. Expose `dbpo_anchor_coef` through the actor-worker kwargs the same way
   `bpo_*` was exposed.

### Weight syncer for drift path

The yaml uses `weight_syncer/patch_syncer` which RLinf's flow-matching PPO
path already uses. The syncer only sees parameter dicts, so the drift model's
new `logstd_head` flows through it unchanged. If rollout / actor processes
diverge on the value head or log-std bias, re-check by running
`rlinf/hybrid_engines/weight_syncer/patch_syncer.py` with logging enabled.

### `ratio_clip_eps` collision

`fsdp_actor_worker.py:735` reads `self.cfg.algorithm.ratio_clip_eps` for a
different (older) loss path. Our BPO yaml does not set it, so the field is
missing; the key isn't used by `bpo_actor_critic`, but if another code path
in the worker also reaches the same line with BPO loss selected we would
KeyError. If you see that at startup, add `ratio_clip_eps: 0.2` to the
algorithm block as a no-op.

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

4. **Prefix pool dtype mismatch on pi0.5**. `_sample_actions_drifting`
   (openpi) already casts prefix_out -> float32 before the pooling; the
   cond_emb reaching `LogStdHead` is fp32. If the LogStdHead has been
   cast to bf16 by FSDP mixed precision, input/param dtype will mismatch.
   The `LogStdHead.forward` expands the bias to match cond_emb.dtype to
   defend against this. If you still see dtype errors, wrap the
   `self.proj(cond_emb)` call in `.to(cond_emb.dtype)` for weights.
