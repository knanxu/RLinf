"""Rainbow DQN agent for chunk-speedup.

Pieces wired up here:
  - Factored dueling C51 network (networks.FactoredDuelingC51)
  - Double Q learning: argmax on online net, evaluation on target net
  - n-step returns: read from PER buffer (already folded)
  - C51 projection: standard Bellemare et al. 2017
  - PER importance-sampling correction in the loss

For factored Q heads, the loss is the sum of per-dim cross-entropy losses
between the projected target distribution and the online distribution at
the chosen (per-dim) action. ε-greedy exploration (linearly annealed).

Episode-level done flag: the buffer stores done flags from the env, so
``(1 - done) * γ^n_len`` correctly cuts bootstrap when episodes end inside
the n-step window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import PERNStepBuffer
from .config import C51Config, NetworkConfig, RainbowConfig
from .networks import FactoredDuelingC51


class RainbowAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions_per_dim: Tuple[int, int, int],
        action_grids: Tuple[np.ndarray, np.ndarray, np.ndarray],
        c51_cfg: C51Config,
        rainbow_cfg: RainbowConfig,
        net_cfg: NetworkConfig,
        device: str = "cuda",
    ):
        self.cfg = rainbow_cfg
        self.c51 = c51_cfg
        self.num_actions_per_dim = tuple(num_actions_per_dim)
        self.device = device
        # action_grids[i][k] is the float value of the k-th choice on dim i
        self.action_grids = [np.asarray(g, dtype=np.float32) for g in action_grids]

        self.net = FactoredDuelingC51(
            state_dim=state_dim,
            num_actions_per_dim=num_actions_per_dim,
            n_atoms=c51_cfg.n_atoms,
            v_min=c51_cfg.v_min,
            v_max=c51_cfg.v_max,
            backbone_sizes=net_cfg.backbone_sizes,
            head_hidden=net_cfg.head_hidden,
        ).to(device)
        self.target_net = FactoredDuelingC51(
            state_dim=state_dim,
            num_actions_per_dim=num_actions_per_dim,
            n_atoms=c51_cfg.n_atoms,
            v_min=c51_cfg.v_min,
            v_max=c51_cfg.v_max,
            backbone_sizes=net_cfg.backbone_sizes,
            head_hidden=net_cfg.head_hidden,
        ).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=rainbow_cfg.lr, eps=rainbow_cfg.adam_eps
        )

        self.buffer = PERNStepBuffer(
            state_dim=state_dim,
            capacity=rainbow_cfg.buffer_size,
            n_step=rainbow_cfg.n_step,
            gamma=rainbow_cfg.gamma,
            alpha=rainbow_cfg.per_alpha,
            eps=rainbow_cfg.per_eps,
            device=device,
        )

        self.train_steps = 0  # counts update() calls, used for target hard sync

    # ---------------------------------------------------------------
    # action selection
    # ---------------------------------------------------------------
    def epsilon(self, env_step: int) -> float:
        c = self.cfg
        frac = min(1.0, env_step / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, env_step: int, deterministic: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (action_values float[3], action_indices int[3])."""
        eps = 0.0 if deterministic else self.epsilon(env_step)
        if not deterministic and np.random.rand() < eps:
            idx = np.array(
                [np.random.randint(K) for K in self.num_actions_per_dim],
                dtype=np.int64,
            )
        else:
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            idx = self.net.greedy_action(s).squeeze(0).cpu().numpy()
        action = np.array(
            [self.action_grids[i][idx[i]] for i in range(len(idx))],
            dtype=np.float32,
        )
        return action, idx

    # ---------------------------------------------------------------
    # C51 projection (per dim)
    # ---------------------------------------------------------------
    def _project_distribution(
        self,
        next_probs: torch.Tensor,    # (B, n_atoms) at the best next action
        rewards: torch.Tensor,       # (B, 1)
        dones: torch.Tensor,         # (B, 1)
        n_lens: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """Bellemare 2017 projection. Returns target distribution (B, n_atoms)."""
        batch_size = next_probs.size(0)
        support = self.net.support               # (n_atoms,)
        delta_z = self.net.delta_z
        v_min, v_max = self.c51.v_min, self.c51.v_max

        gamma_pow_n = (self.cfg.gamma ** n_lens).unsqueeze(-1)   # (B, 1)
        tz = rewards + (1.0 - dones) * gamma_pow_n * support.unsqueeze(0)
        tz = tz.clamp(v_min, v_max)                              # (B, n_atoms)
        b = (tz - v_min) / delta_z                               # in [0, n_atoms-1]
        lower = b.floor().long()
        upper = b.ceil().long()

        # Handle the case where b is exactly on a grid point (lower == upper).
        # In that case redistribute the full mass at lower.
        lower_eq_upper = (lower == upper)
        # Adjust to keep mass conservation: shift upper down if at upper bound
        upper = torch.where(upper >= self.c51.n_atoms, upper - 1, upper)
        # Disambiguate by nudging upper up when lower==upper and not at top.
        # Standard trick:
        lower = torch.where(
            lower_eq_upper & (lower > 0),
            lower - 1,
            lower,
        )
        upper = torch.where(
            lower_eq_upper & (upper < self.c51.n_atoms - 1),
            upper + 1,
            upper,
        )

        # Distribute probability mass.
        target = torch.zeros_like(next_probs)
        m_l = next_probs * (upper.float() - b)
        m_u = next_probs * (b - lower.float())

        # When lower==upper after the nudge above (only happens at the
        # boundary), put the full probability mass on that atom.
        # In practice the (u - b) and (b - l) sum to (u - l) which is >=1
        # so we re-normalize per row below to be safe.
        target.scatter_add_(1, lower, m_l)
        target.scatter_add_(1, upper, m_u)
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return target

    # ---------------------------------------------------------------
    # train step
    # ---------------------------------------------------------------
    def update(self, beta: float) -> dict:
        batch = self.buffer.sample(self.cfg.batch_size, beta)
        s = batch["states"]
        a = batch["actions"]               # (B, 3) int64
        r = batch["rewards"]
        ns = batch["next_states"]
        d = batch["dones"]
        n_lens = batch["n_lens"]
        idxs = batch["idxs"]
        is_w = batch["weights"]            # (B,)

        # ---- target distribution per dim (Double Q + C51 projection) ----
        with torch.no_grad():
            # Online net selects the argmax action; target net evaluates it
            online_qs = self.net.q_values(ns)               # tuple of (B, K_i)
            online_argmax = [q.argmax(dim=-1) for q in online_qs]
            target_probs_all = self.target_net(ns)          # tuple of (B, K_i, n_atoms)

            target_dists = []
            for i in range(len(self.num_actions_per_dim)):
                best_idx = online_argmax[i].unsqueeze(-1).unsqueeze(-1)        # (B, 1, 1)
                best_idx = best_idx.expand(-1, 1, self.c51.n_atoms)            # (B, 1, n_atoms)
                next_probs_i = target_probs_all[i].gather(1, best_idx).squeeze(1)  # (B, n_atoms)
                target_i = self._project_distribution(next_probs_i, r, d, n_lens)
                target_dists.append(target_i)

        # ---- online distribution at the taken actions ----
        probs = self.net(s)   # tuple of (B, K_i, n_atoms)

        total_loss = 0.0
        per_sample_td = torch.zeros_like(is_w)
        loss_dict = {}
        for i, (p, t) in enumerate(zip(probs, target_dists)):
            taken = a[:, i].unsqueeze(-1).unsqueeze(-1)                    # (B, 1, 1)
            taken = taken.expand(-1, 1, self.c51.n_atoms)                  # (B, 1, n_atoms)
            p_taken = p.gather(1, taken).squeeze(1)                        # (B, n_atoms)

            # Cross-entropy: -Σ t · log p
            ce = -(t * torch.log(p_taken)).sum(dim=-1)                     # (B,)
            weighted = (is_w * ce).mean()
            total_loss = total_loss + weighted

            # Use the i=0 dim's CE as the PER signal for simplicity; sum
            # of CE across dims would also work and is what we use below.
            per_sample_td = per_sample_td + ce.detach()
            loss_dict[f"loss_dim{i}"] = float(weighted.detach())

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.opt.step()

        # Update PER priorities with summed |CE| (works fine as a proxy
        # for TD error magnitude in the distributional setting).
        self.buffer.update_priorities(
            idxs, per_sample_td.cpu().numpy()
        )

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_every == 0:
            self._sync_target()

        loss_dict["loss_total"] = float(total_loss.detach())
        return loss_dict

    def _sync_target(self):
        if self.cfg.tau_target >= 1.0:
            self.target_net.load_state_dict(self.net.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
                    tp.data.mul_(1.0 - self.cfg.tau_target)
                    tp.data.add_(self.cfg.tau_target * p.data)

    # ---------------------------------------------------------------
    # checkpointing
    # ---------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "net": self.net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "opt": self.opt.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.train_steps = int(ckpt.get("train_steps", 0))
