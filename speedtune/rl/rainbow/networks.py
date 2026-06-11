"""Factored dueling C51 network for chunk-speedup Rainbow DQN.

Design:
  state ─► shared MLP backbone ─► two paths
        ├─ value head:    V(s) of shape (atoms,)
        └─ for each action dim i ∈ {v, vs, as}:
                advantage head A_i(s) of shape (K_i, atoms)
  Combine into Q_i(s, a):
        logits_i(s, a, z) = V(s, z) + A_i(s, a, z) - mean_a A_i(s, ·, z)
  Then per-dim probabilities = softmax over atoms.

  Q_i(s, a) = Σ_z probs_i(s, a, z) * z, where z is the support grid.

The factored decomposition assumes the three speed components are
approximately conditionally independent given the state — a reasonable
prior for a 1D-like structure (v, vs, as all push "speed up" the same
direction). Greedy action selection picks
  a* = (argmax_v Q_v, argmax_vs Q_vs, argmax_as Q_as)
which is consistent with optimizing the *sum* of per-dim Q values, which
is what the loss minimizes.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: Sequence[int], activation=nn.ReLU, out_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class FactoredDuelingC51(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions_per_dim: Tuple[int, int, int],   # (K_v, K_vs, K_as)
        n_atoms: int,
        v_min: float,
        v_max: float,
        backbone_sizes: Sequence[int] = (512, 512, 256),
        head_hidden: int = 128,
    ):
        super().__init__()
        self.num_actions_per_dim = tuple(num_actions_per_dim)
        self.num_dims = len(self.num_actions_per_dim)
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32),
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.trunk = _mlp([state_dim, *backbone_sizes], activation=nn.ReLU)
        self.post_relu = nn.ReLU()

        feat_dim = backbone_sizes[-1]

        # Single shared value head V(s) of size (atoms,). Following the
        # standard dueling formulation the value head sees the same
        # backbone features used by the advantage heads.
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_atoms),
        )

        # One advantage head per action dim: K_i * n_atoms outputs.
        self.advantage_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, K * n_atoms),
                )
                for K in self.num_actions_per_dim
            ]
        )

    def _features(self, s: torch.Tensor) -> torch.Tensor:
        h = self.trunk(s)
        return self.post_relu(h)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return per-dim probability tensors of shape (B, K_i, n_atoms)."""
        h = self._features(s)
        v = self.value_head(h)                   # (B, n_atoms)
        out = []
        for K, head in zip(self.num_actions_per_dim, self.advantage_heads):
            a = head(h).view(-1, K, self.n_atoms)              # (B, K, n_atoms)
            # Dueling combine, per atom: V + A - mean_K A
            logits = v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
            # Numerical safety: clamp away from 0 for log() below.
            probs = probs.clamp(min=1e-8)
            out.append(probs)
        return tuple(out)

    def q_values(self, s: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Per-dim expected Q values (B, K_i)."""
        probs = self.forward(s)
        return tuple(p @ self.support for p in probs)

    @torch.no_grad()
    def greedy_action(self, s: torch.Tensor) -> torch.Tensor:
        """Argmax_a Q_i(s, a) per dim. Returns (B, num_dims) int64 indices."""
        qs = self.q_values(s)
        return torch.stack([q.argmax(dim=-1) for q in qs], dim=-1)
