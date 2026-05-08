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
"""DBPO (Drift-Based Policy Optimization) RL action model.

Implements the stochastic adapter from Gao et al. 2026:
  - Deterministic center mu_theta(o, z) comes from the pretrained Drift-Based
    Policy (1-NFE drifting inference) — openpi's _sample_actions_drifting.
  - State-conditioned Gaussian head gives log sigma_psi(o), producing the
    stochastic actor pi_{theta, psi}(x | o, z) = N(x; mu, diag(sigma^2)).
  - Joint-policy ratio trick (Eq. 14): storing z alongside actions lets the
    PPO/BPO importance ratio reduce to the conditional Gaussian ratio under
    a fixed prior p0(z). So logprob only spans the executed action prefix.
  - Deployment (mode="eval") drops the noise and returns mu directly, which
    preserves strict 1-NFE inference from the pretrained backbone.

This module intentionally mirrors the interface shape of OpenPi0ForRLActionPrediction
so the existing RLinf FSDP / rollout / env workers can talk to it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)


@dataclass(frozen=True)
class OpenPi0DBPOConfig(OpenPi0Config):
    # Sentinel: identifies DBPO's single-step drift path. Not consumed by the
    # inherited flow_* branches in OpenPi0ForRLActionPrediction.sample_mean_var_val.
    noise_method: str = "drift_dbpo"

    # State-conditioned log-std head
    dbpo_init_log_std: float = -3.5      # exp(-3.5) ~ 0.030
    dbpo_min_logprob_std: float = 0.03
    dbpo_max_logprob_std: float = 0.30
    dbpo_min_sampling_std: float = 0.03
    dbpo_randn_clip: float = 3.0
    # If True, log-sigma is a per-dim bias only (state-independent). Recommended
    # for the first runs: closest to PPO-on-deterministic-policy and most stable
    # given a zero-initialised backbone.
    dbpo_freeze_logstd_cond: bool = True

    # Anchor regularisation (Gao et al. Eq. 16). We do not implement anchor
    # here because it doubles backbone memory; instead we recommend tight
    # clip_ratio. Kept as a config knob for a future trainer extension.
    dbpo_anchor_coef: float = 0.0


def _resolve_cond_dim(config: OpenPi0DBPOConfig) -> int:
    """Prefix token width used by the state-conditioned log-std head.

    pi0.5 runs prefix through the 2B paligemma trunk — width 2048.
    pi0    runs prefix through the shared backbone — width 1024.
    Matches the assumption in OpenPi0ForRLActionPrediction.__init__ (proj_width).
    """
    return 2048 if "pi05" in config.config_name else 1024


class LogStdHead(nn.Module):
    """Zero-weight, log(init_std)-bias head producing per-dim log sigma.

    On init the head outputs a constant log_std == log(init_std) regardless of
    cond_emb, which means the pretrained deterministic policy is preserved
    exactly at iteration 0.
    """

    def __init__(
        self,
        cond_dim: int,
        out_dim: int,
        init_log_std: float,
        freeze_cond: bool,
    ):
        super().__init__()
        self.freeze_cond = freeze_cond
        self._out_dim = out_dim
        if freeze_cond:
            self.bias = nn.Parameter(torch.full((out_dim,), float(init_log_std)))
        else:
            self.proj = nn.Linear(cond_dim, out_dim)
            nn.init.zeros_(self.proj.weight)
            nn.init.constant_(self.proj.bias, float(init_log_std))

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        if self.freeze_cond:
            return self.bias.to(cond_emb.dtype).expand(cond_emb.shape[0], self._out_dim)
        return self.proj(cond_emb)


class OpenPi0DBPOForRLActionPrediction(OpenPi0ForRLActionPrediction):
    """DBPO adapter on top of OpenPi0ForRLActionPrediction.

    Reuses:
      * embed_prefix / embed_suffix / paligemma_with_expert forward
      * _build_prefix_cache for value-head-only re-evaluation
      * value_head / _compute_value_from_suffix
      * gaussian_entropy / get_logprob_norm
      * input_transform / output_transform / obs_processor / precision_processor
      * freeze_vlm behaviour (train_expert_only)

    Overrides:
      * sample_actions             -> 1-NFE drifting + Gaussian reparam
      * get_log_prob_value         -> exact Gaussian logprob from stored (z, a)
      * default_forward            -> inherited; new get_log_prob_value powers it
    """

    config: OpenPi0DBPOConfig

    @property
    def _no_split_names(self) -> list[str]:
        # Keep the log-std head out of FSDP intra-module sharding.
        return [*super()._no_split_names, "logstd_head"]

    def __init__(self, config: OpenPi0DBPOConfig):
        super().__init__(config)

        cond_dim = _resolve_cond_dim(config)
        out_dim = config.action_horizon * config.action_dim
        self.logstd_head = LogStdHead(
            cond_dim=cond_dim,
            out_dim=out_dim,
            init_log_std=config.dbpo_init_log_std,
            freeze_cond=config.dbpo_freeze_logstd_cond,
        )

        # Refresh _fsdp_wrap_name tags so new logstd_head is tagged too
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    # ------------------------------------------------------------------
    # Drift forward helpers
    # ------------------------------------------------------------------
    def _logstd_bounds(self, device, dtype):
        mn = torch.log(torch.tensor(self.config.dbpo_min_logprob_std, device=device, dtype=dtype))
        mx = torch.log(torch.tensor(self.config.dbpo_max_logprob_std, device=device, dtype=dtype))
        return mn, mx

    def _drift_forward_from_observation(self, observation, noise):
        """Wrap openpi's _sample_actions_drifting for rollout (we need .no_grad
        handled by the caller — this method is plain forward)."""
        device = observation.state.device
        # Requires openpi pi0_pytorch.PI0Pytorch._sample_actions_drifting
        # to support return_hidden=True (returning (mean, suffix_out, cond_emb)).
        mean, suffix_out, cond_emb = self._sample_actions_drifting(
            device, observation, noise, return_hidden=True
        )
        return mean, suffix_out, cond_emb

    def _drift_forward_from_tokens(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        """Re-run drifting forward from tokenised inputs (used in
        get_log_prob_value when the chain was stored but the Observation is not)."""
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # local import to avoid circulars

        device = state.device
        bsize = noise.shape[0]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        time_ones = torch.ones(bsize, device=device, dtype=torch.float32)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, noise, time_ones
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -self.config.action_horizon:].to(torch.float32)
        mean = self.action_out_proj(suffix_out)

        prefix_out = prefix_out.to(torch.float32)
        prefix_mask_f = prefix_pad_masks.to(prefix_out.dtype)[..., None]
        cond_emb = (prefix_out * prefix_mask_f).sum(dim=1) / prefix_mask_f.sum(dim=1).clamp(min=1e-6)
        return mean, suffix_out, cond_emb

    def _make_std(self, cond_emb, mean_shape):
        raw = self.logstd_head(cond_emb).reshape(mean_shape)
        mn, mx = self._logstd_bounds(raw.device, raw.dtype)
        log_std = raw.clamp(min=mn, max=mx)
        return log_std.exp(), log_std

    # ------------------------------------------------------------------
    # Rollout: single-step drift + Gaussian reparam
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise=None,
        mode: str = "train",
        compute_values: bool = True,
    ) -> dict:
        bsize = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(noise_shape, device)
        else:
            noise = noise.to(self.action_in_proj.weight.dtype)

        mean, suffix_out, cond_emb = self._drift_forward_from_observation(observation, noise)
        std, _log_std = self._make_std(cond_emb, mean.shape)

        if mode == "train":
            sampling_std = std.clamp(min=self.config.dbpo_min_sampling_std)
            eps = torch.randn_like(mean).clamp(
                -self.config.dbpo_randn_clip, self.config.dbpo_randn_clip
            )
            sampled_actions = mean + sampling_std * eps
        else:
            sampling_std = std
            sampled_actions = mean

        # Exact Gaussian logprob, then trim to executed (chunk, env_dim) prefix
        logprob_full = self.get_logprob_norm(sampled_actions, mean, sampling_std)
        logprob = logprob_full[
            :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = self.gaussian_entropy(sampling_std)[
            :, : self.config.action_chunk, : self.config.action_env_dim
        ]

        # chains: [B, 2, H, D]. Index 0 is latent z (stored for joint-ratio replay),
        # index 1 is the action actually executed. We pad to the same rank/shape
        # assumed by existing rollout buffers: (n_chain_steps, H, D).
        chains = torch.stack([noise, sampled_actions], dim=1)
        # denoise_inds: single-step; any constant works, default_forward reads it as an index.
        denoise_inds = torch.zeros(bsize, 1, dtype=torch.long, device=device)

        if self.config.add_value_head and compute_values:
            value = self._compute_value_from_suffix(suffix_out)  # [B]
        else:
            value = torch.zeros(bsize, device=device)

        # Keep output shape aligned with flow-matching rollout: logprobs 3-D.
        result = {
            "actions": sampled_actions,
            "chains": chains,
            "prev_logprobs": logprob,
            "prev_values": value[:, None],
            "denoise_inds": denoise_inds,
        }
        return result

    # ------------------------------------------------------------------
    # Training: exact Gaussian logprob on stored (z, a)
    # ------------------------------------------------------------------
    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values: bool = False,
    ):
        """Override of the flow-matching chain-replay version.

        chains is [B, 2, H, D] where [:, 0] is z (noise) and [:, 1] is the
        executed action. We re-run drift once under current parameters, get
        mu_theta(o, z), build sigma from logstd_head(cond_emb), and compute
        log N(a; mu, sigma).
        """
        z = chains[:, 0]
        actions = chains[:, 1]

        mean, suffix_out, cond_emb = self._drift_forward_from_tokens(
            images, img_masks, lang_tokens, lang_masks, state, z
        )
        std, _log_std = self._make_std(cond_emb, mean.shape)

        logprob_full = self.get_logprob_norm(actions, mean, std)
        entropy_full = self.gaussian_entropy(std)

        if self.config.add_value_head and compute_values:
            value = self._compute_value_from_suffix(suffix_out)  # [B]
        else:
            value = torch.zeros(state.shape[0], device=state.device, dtype=torch.float32)

        # Shape contract of the parent (flow-matching) path: [B, num_steps, H, D]
        # DBPO has a single effective step, so num_steps == 1.
        logprob = logprob_full[:, None]  # [B, 1, H, D]
        entropy = entropy_full[:, None]  # [B, 1, H, D]
        value = value[:, None]           # [B, 1]
        return logprob, value, entropy
