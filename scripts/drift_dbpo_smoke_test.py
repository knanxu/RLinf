#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
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
"""Offline smoke tests for the DBPO + BPO additions.

These tests intentionally avoid the full FSDP / Ray stack so you can run them
before the heavy RLinf env is up. Two layers of verification:

  Tier 1 (fast, no openpi required):
      * BPO loss registration + shape contract check.
      * LogStdHead numerical properties (zero-weight, bias = log(init_std)).

  Tier 2 (slower, requires openpi + torch):
      * OpenPi0DBPOForRLActionPrediction: build a tiny pi0.5 model from fake
        config, run sample_actions once, run get_log_prob_value once, check
        shapes and that replaying the stored (z, a) reproduces the same
        logprob exactly under unchanged parameters (joint-policy ratio trick).

Tier 2 is skipped with a clear message if openpi or a GPU is unavailable.

Usage:
    python scripts/drift_dbpo_smoke_test.py
    python scripts/drift_dbpo_smoke_test.py --only tier1
    python scripts/drift_dbpo_smoke_test.py --only tier2
"""

from __future__ import annotations

import argparse
import sys
import traceback


# ---------------------------------------------------------------------------
# Tier 1: BPO loss + LogStdHead (no openpi, no GPU)
# ---------------------------------------------------------------------------


def tier1_bpo_loss_registered() -> None:
    import rlinf.algorithms  # noqa: F401  -- triggers decorators
    from rlinf.algorithms.registry import LOSS_REGISTRY

    assert "bpo_actor" in LOSS_REGISTRY, (
        "bpo_actor not registered. Check that rlinf.algorithms.__init__ "
        "imports losses_bpo."
    )
    assert "bpo_actor_critic" in LOSS_REGISTRY
    print("    LOSS_REGISTRY has:", sorted(LOSS_REGISTRY.keys()))


def tier1_bpo_loss_shapes() -> None:
    import torch

    from rlinf.algorithms.registry import LOSS_REGISTRY

    # Chunk-level logprobs: [bsz]
    bsz = 8
    torch.manual_seed(0)
    logprobs = torch.randn(bsz).float()
    old_logprobs = logprobs.clone() + 0.01 * torch.randn(bsz)
    advantages = torch.randn(bsz).float()
    returns = torch.randn(bsz).float()
    values = torch.randn(bsz).float()
    prev_values = torch.randn(bsz).float()

    fn = LOSS_REGISTRY["bpo_actor_critic"]
    loss, metrics = fn(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        returns=returns,
        values=values,
        prev_values=prev_values,
        value_clip=0.2,
        huber_delta=10.0,
        bpo_epsilon=0.2,
        bpo_lambda=1e-3,
        bpo_alpha1=0.0,
    )
    assert loss.ndim == 0, f"loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss must be finite, got {loss}"

    required_keys = {
        "actor/policy_loss",
        "actor/ratio",
        "actor/ratio_abs",
        "actor/target_ratio",
        "actor/approx_kl",
        "actor/clip_fraction",
        "actor/bpo_epsilon",
        "actor/bpo_lambda",
        "critic/value_loss",
        "critic/value_clip_ratio",
        "critic/explained_variance",
    }
    missing = required_keys - set(metrics.keys())
    assert not missing, f"missing metrics: {missing}"

    print(f"    loss={float(loss):+.4f}")
    print(f"    actor/target_ratio={float(metrics['actor/target_ratio']):+.4f}")
    print(f"    actor/approx_kl={float(metrics['actor/approx_kl']):+.4f}")


def tier1_bpo_target_ratio_math() -> None:
    """The BPO paper states target_ratio = 1 + epsilon * tanh(adv / (2 lambda)).
    We implement the equivalent sigmoid form. Check the two match.
    """
    import torch

    bsz = 128
    eps = 0.2
    lam = 1e-3
    torch.manual_seed(1)
    adv = torch.randn(bsz).float()

    tanh_form = 1.0 + eps * torch.tanh(adv / (2 * lam))
    sigmoid_form = (1.0 - eps) + 2 * eps * torch.sigmoid(adv / lam)

    max_err = (tanh_form - sigmoid_form).abs().max().item()
    assert max_err < 1e-5, f"tanh vs sigmoid mismatch: max diff {max_err}"
    print(f"    max |tanh form - sigmoid form| = {max_err:.2e}")


def tier1_logstd_head_init() -> None:
    """LogStdHead: at init, log_std == log(init_std) regardless of input."""
    import torch

    from rlinf.models.embodiment.openpi.openpi_dbpo_action_model import LogStdHead

    init_log_std = -3.5

    # Conditional head
    head = LogStdHead(
        cond_dim=32, out_dim=7, init_log_std=init_log_std, freeze_cond=False
    )
    cond = torch.randn(4, 32)
    out = head(cond)
    assert out.shape == (4, 7)
    max_err = (out - init_log_std).abs().max().item()
    assert max_err < 1e-6, f"conditional LogStdHead not constant at init: {max_err}"
    print(f"    conditional head: max |out - init_log_std| = {max_err:.2e}")

    # Bias-only head
    head_b = LogStdHead(
        cond_dim=32, out_dim=7, init_log_std=init_log_std, freeze_cond=True
    )
    out_b = head_b(cond)
    max_err_b = (out_b - init_log_std).abs().max().item()
    assert max_err_b < 1e-6
    print(f"    bias-only head:   max |out - init_log_std| = {max_err_b:.2e}")


# ---------------------------------------------------------------------------
# Tier 2: actual model build + sample + logprob replay
# ---------------------------------------------------------------------------


def _tier2_imports():
    import openpi  # noqa: F401 -- must be importable

    # pi0 Pytorch model with drift support
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: F401


def _build_tiny_pi0_config():
    """Return a minimal Pi0Config that keeps a real pi0.5 code path live but
    small enough to run on a laptop CPU in a few seconds."""
    from openpi.models import pi0_config as _pi0cfg

    # Pi0.5 codepath (adaRMS, prefix state token in language). Keep stock sizes
    # because the gemma configs only expose predefined variants; we slim down
    # action_horizon and action_dim to cut the suffix token count.
    return _pi0cfg.Pi0Config(
        dtype="float32",
        action_dim=14,
        action_horizon=4,          # tiny H to keep attention cheap
        max_token_len=48,
        pi05=True,
        discrete_state_input=True,
        pytorch_compile_mode=None,
        use_drifting_loss=True,
        drifting_gen_per_label=1,
        drifting_temperatures=(0.02,),
        drifting_per_timestep_loss=True,
    )


def _wrap_dbpo_model(pi0_config):
    """Attach the DBPO config fields on top of a Pi0Config so we can build
    OpenPi0DBPOForRLActionPrediction without round-tripping through openpi
    training configs (those require assets dirs)."""
    from dataclasses import fields

    from rlinf.models.embodiment.openpi.openpi_dbpo_action_model import (
        OpenPi0DBPOConfig,
        OpenPi0DBPOForRLActionPrediction,
    )

    # Start from the OpenPi0Config defaults, overlay Pi0Config field values.
    kwargs = {}
    for f in fields(OpenPi0DBPOConfig):
        if hasattr(pi0_config, f.name):
            kwargs[f.name] = getattr(pi0_config, f.name)
    kwargs["config_name"] = "pi05_aloha_robotwin"
    kwargs["num_images_in_input"] = 3
    kwargs["action_env_dim"] = 14
    kwargs["action_chunk"] = pi0_config.action_horizon
    kwargs["num_steps"] = 1
    kwargs["add_value_head"] = True
    kwargs["detach_critic_input"] = True
    kwargs["noise_method"] = "drift_dbpo"
    # DBPO knobs
    kwargs["dbpo_init_log_std"] = -3.5
    kwargs["dbpo_freeze_logstd_cond"] = True
    dbpo_cfg = OpenPi0DBPOConfig(**kwargs)

    model = OpenPi0DBPOForRLActionPrediction(dbpo_cfg)
    model.eval()
    return model, dbpo_cfg


def _fake_observation(cfg, bsz: int):
    """Build a valid _model.Observation matching pi0.5 expectations."""
    import torch

    from openpi.models import model as _model

    H, W = 224, 224
    image = {
        "base_0_rgb": torch.zeros(bsz, H, W, 3),
        "left_wrist_0_rgb": torch.zeros(bsz, H, W, 3),
        "right_wrist_0_rgb": torch.zeros(bsz, H, W, 3),
    }
    image_mask = {
        "base_0_rgb": torch.ones(bsz, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(bsz, dtype=torch.bool),
        "right_wrist_0_rgb": torch.ones(bsz, dtype=torch.bool),
    }
    state = torch.zeros(bsz, cfg.action_dim)
    token_len = cfg.max_token_len
    tokenized_prompt = torch.zeros(bsz, token_len, dtype=torch.long)
    tokenized_prompt_mask = torch.ones(bsz, token_len, dtype=torch.bool)

    return _model.Observation(
        images=image,
        image_masks=image_mask,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def tier2_dbpo_shapes_and_replay() -> None:
    import torch

    _tier2_imports()

    pi0_cfg = _build_tiny_pi0_config()
    model, dbpo_cfg = _wrap_dbpo_model(pi0_cfg)
    torch.manual_seed(0)

    bsz = 2
    obs = _fake_observation(pi0_cfg, bsz)

    # Build tokenized inputs the way _preprocess_observation wants them, then
    # run the rollout path.
    with torch.no_grad():
        noise = torch.randn(bsz, pi0_cfg.action_horizon, pi0_cfg.action_dim)
        out = model.sample_actions(obs, noise=noise, mode="train", compute_values=True)

    assert out["actions"].shape == (bsz, pi0_cfg.action_horizon, pi0_cfg.action_dim)
    assert out["chains"].shape == (bsz, 2, pi0_cfg.action_horizon, pi0_cfg.action_dim)
    expected_lp_shape = (
        bsz,
        dbpo_cfg.action_chunk,
        dbpo_cfg.action_env_dim,
    )
    assert out["prev_logprobs"].shape == expected_lp_shape, (
        f"prev_logprobs shape {out['prev_logprobs'].shape} != {expected_lp_shape}"
    )
    assert torch.isfinite(out["prev_logprobs"]).all()
    assert torch.isfinite(out["prev_values"]).all()
    print("    sample_actions OK")
    print(f"      actions:     {tuple(out['actions'].shape)}")
    print(f"      chains:      {tuple(out['chains'].shape)}")
    print(f"      prev_logprobs: {tuple(out['prev_logprobs'].shape)}")
    print(f"      prev_values:   {tuple(out['prev_values'].shape)}")

    # Now replay through get_log_prob_value. In the rollout path the chain
    # went through the exact same weights, so the replayed logprob must match.
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        obs, train=False
    )

    with torch.no_grad():
        logprob_replay, value_replay, entropy_replay, mean_replay = (
            model.get_log_prob_value(
                images, img_masks, lang_tokens, lang_masks, state,
                chains=out["chains"], denoise_inds=out["denoise_inds"],
                compute_values=True,
            )
        )
    assert logprob_replay.shape == (bsz, 1, pi0_cfg.action_horizon, pi0_cfg.action_dim)
    assert value_replay.shape == (bsz, 1)
    assert mean_replay.shape == (bsz, pi0_cfg.action_horizon, pi0_cfg.action_dim)
    assert torch.isfinite(logprob_replay).all()

    # Trim replayed logprob to the executed (chunk, env_dim) slice and compare.
    lp_replay_trim = logprob_replay[
        :, 0, : dbpo_cfg.action_chunk, : dbpo_cfg.action_env_dim
    ]
    diff = (lp_replay_trim - out["prev_logprobs"]).abs().max().item()
    # float32 drift through gemma is ~1e-4, accept that.
    assert diff < 1e-3, f"replay logprob mismatch: max abs diff {diff:.3e}"
    print(f"    replay logprob max |diff| = {diff:.3e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_tier(name, tests):
    print(f"\n=== {name} ===")
    n_pass = n_fail = n_skip = 0
    for label, fn in tests:
        print(f"[{name}] {label}")
        try:
            fn()
        except ImportError as e:
            n_skip += 1
            print(f"    SKIP (missing dependency): {e}")
        except Exception:
            n_fail += 1
            print("    FAIL")
            traceback.print_exc()
        else:
            n_pass += 1
            print("    PASS")
    return n_pass, n_fail, n_skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["tier1", "tier2"], default=None)
    args = parser.parse_args()

    tier1 = [
        ("bpo_loss registered", tier1_bpo_loss_registered),
        ("bpo_loss shapes + metrics", tier1_bpo_loss_shapes),
        ("target_ratio tanh == sigmoid form", tier1_bpo_target_ratio_math),
        ("LogStdHead init", tier1_logstd_head_init),
    ]
    tier2 = [
        ("DBPO model shape + replay consistency", tier2_dbpo_shapes_and_replay),
    ]

    total_pass = total_fail = total_skip = 0
    if args.only in (None, "tier1"):
        p, f, s = run_tier("Tier 1", tier1)
        total_pass, total_fail, total_skip = p, f, s
    if args.only in (None, "tier2"):
        p, f, s = run_tier("Tier 2", tier2)
        total_pass += p
        total_fail += f
        total_skip += s

    print(
        f"\nResult: pass={total_pass} fail={total_fail} skip={total_skip}"
    )
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
