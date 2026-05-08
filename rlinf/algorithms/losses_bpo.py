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
"""Bounded Policy Optimization (BPO) actor loss for RLinf.

Reference:
  Ao et al. "Bounded Ratio Reinforcement Learning" (arXiv:2604.18578v2, 2026).

Core idea. Instead of PPO's clipped surrogate, BPO minimises an
advantage-weighted total-variation distance between the new policy ratio and
the analytic bounded-ratio optimum of the BRRL regularised problem:

    target_ratio(s, a) = 1 + epsilon * tanh( A_tilde_pi0(s, a) / (2 * lambda) )
                       = (1 - epsilon) + 2 * epsilon * sigmoid( A_tilde / lambda )

    pg_loss = E[ |ratio - target_ratio| * ( |R - V| + alpha1 ) ]

where A_tilde is the (soft) median advantage. Following the paper's ablation
(Fig. 9, mean vs median), we approximate A_tilde by the ordinary advantage
R - V. This skips an additional median-value network without noticeable
performance loss in practice.

Relative to PPO:
  * No hard clip on ratio. Outside |ratio - 1| > epsilon the loss still has
    symmetric gradient toward target_ratio, so updates remain stable when the
    ratio drifts (this is what BPO Fig. 7 measures).
  * advantage is used *detached* inside target_ratio — target_ratio is a
    constant policy direction, not a differentiable path.
  * weight |R - V| + alpha1 scales the penalty by how confident the critic is
    about the action. alpha1 defaults to 0 (BPO ablation: adding alpha1 did
    not help).

Shape contract mirrors PPO loss in rlinf/algorithms/losses.py so that this
loss plugs into the same trainer path without changes.
"""

from typing import Callable, Optional

import torch

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.algorithms.registry import register_policy_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def _compute_bpo_actor_loss_core(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    bpo_epsilon: float,
    bpo_lambda: float,
    bpo_alpha1: float,
    loss_mask: Optional[torch.Tensor],
    loss_agg_func: Callable[..., torch.Tensor],
    loss_mask_ratio: Optional[torch.Tensor],
    critic_warmup: bool,
) -> tuple[torch.Tensor, dict]:
    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32, "logprobs must be float32"
    assert old_logprobs.dtype == torch.float32, "old_logprobs must be float32"
    assert advantages.dtype == torch.float32, "advantages must be float32"

    loss_mask_count = loss_mask.count_nonzero() or 1

    log_ratio = logprobs - old_logprobs
    ratio = torch.where(loss_mask, torch.exp(log_ratio), torch.zeros_like(log_ratio))

    # Advantage-weighted TV toward the analytic optimal ratio.
    # Using sigmoid form (identical to 1 + eps*tanh(A/(2 lambda))):
    #   1 - eps + 2*eps*sigmoid(A / lambda)
    adv_detached = advantages.detach() / max(bpo_lambda, 1e-8)
    target_ratio = (1.0 - bpo_epsilon) + 2.0 * bpo_epsilon * torch.sigmoid(adv_detached)

    # |R - V| + alpha1 (values are detached: this term is a *weight*).
    weight = (returns - values.detach()).abs() + bpo_alpha1
    raw_pg = torch.where(
        loss_mask,
        torch.abs(ratio - target_ratio) * weight,
        torch.zeros_like(ratio),
    )
    pg_loss = loss_agg_func(raw_pg, loss_mask, loss_mask_ratio)

    # approx_kl, clip_fraction mirror PPO logging.
    with torch.no_grad():
        approx_kl_per = torch.where(loss_mask, log_ratio, torch.zeros_like(log_ratio))
        approx_kl = -approx_kl_per.sum() / float(loss_mask_count)

        out_of_band = (torch.abs(ratio - 1.0) > bpo_epsilon).float()
        clip_fraction = (out_of_band * loss_mask.float()).sum() / float(loss_mask_count)

    if critic_warmup:
        pg_loss = torch.tensor(0.0, device=pg_loss.device)

    metrics_data = {
        "actor/pg_loss": pg_loss.detach(),
        "actor/policy_loss": pg_loss.detach(),  # alias, keeps PPO-compatible logs
        "actor/ratio": masked_mean(ratio.detach(), loss_mask),
        "actor/ratio_abs": masked_mean((ratio - 1.0).abs().detach(), loss_mask),
        "actor/target_ratio": masked_mean(target_ratio.detach(), loss_mask),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
        "actor/bpo_epsilon": torch.tensor(float(bpo_epsilon), device=pg_loss.device),
        "actor/bpo_lambda": torch.tensor(float(bpo_lambda), device=pg_loss.device),
    }
    return pg_loss, metrics_data


@register_policy_loss("bpo_actor")
def compute_bpo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    bpo_epsilon: float = 0.2,
    bpo_lambda: float = 1.0e-3,
    bpo_alpha1: float = 0.0,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Advantage-weighted TV actor loss of BPO (no critic term).

    Intended to pair with a separate critic loss term. For the combined
    path, use :func:`compute_bpo_actor_critic_loss`.
    """
    loss_agg_func: Callable[..., torch.Tensor] = masked_mean
    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    return _compute_bpo_actor_loss_core(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        returns=returns,
        values=values,
        bpo_epsilon=bpo_epsilon,
        bpo_lambda=bpo_lambda,
        bpo_alpha1=bpo_alpha1,
        loss_mask=loss_mask,
        loss_agg_func=loss_agg_func,
        loss_mask_ratio=loss_mask_ratio,
        critic_warmup=bool(critic_warmup),
    )


@register_policy_loss("bpo_actor_critic")
def compute_bpo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """Joint BPO actor loss + standard PPO-style clipped value loss.

    The critic loss is reused verbatim from PPO (Huber + value-clip) so the
    experiments only isolate the policy-side change (clip -> advantage-weighted
    TV toward the bounded-ratio optimum).
    """
    actor_loss, actor_metrics = compute_bpo_actor_loss(**kwargs)
    critic_loss, critic_metrics = compute_ppo_critic_loss(**kwargs)
    loss = actor_loss + critic_loss
    metrics = {**actor_metrics, **critic_metrics}
    return loss, metrics
