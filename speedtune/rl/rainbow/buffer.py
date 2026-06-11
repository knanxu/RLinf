"""Prioritized Experience Replay with n-step return support.

  - SumTree-based proportional priority sampling (Schaul et al. 2015).
  - Multi-step (n-step) bootstrapping: a transition stored after a delay
    of n env steps already holds the discounted sum of n rewards and the
    state n steps later. Combined with double Q in the trainer this gives
    the Rainbow flavour of multi-step returns.

API:
  buf = PERNStepBuffer(state_dim, capacity, n_step, gamma, alpha)
  buf.add(s, a_idx_tuple, r, ns, done)          # `a_idx_tuple` is (i_v, i_vs, i_as)
  s,a,r,ns,d, idxs, weights = buf.sample(batch_size, beta)
  buf.update_priorities(idxs, td_errors)
"""
from collections import deque
from typing import Tuple

import numpy as np
import torch


class _SumTree:
    """Fixed-capacity sum-tree for proportional priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        # internal nodes (capacity - 1) + leaves (capacity)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float) -> int:
        idx = self.data_idx
        self.update(idx, priority)
        self.data_idx = (self.data_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def get(self, s: float) -> Tuple[int, float]:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PERNStepBuffer:
    """Prioritized n-step replay buffer.

    Stores per-step transitions but, on add(), folds the last ``n_step``
    transitions into a single n-step transition before pushing to the
    sumtree. This means every sampled tuple already has the correct
    multi-step target reward and "next state n steps later".
    """

    # action stored per dim: i_v (int8), i_vs (int8), i_as (int8)
    def __init__(
        self,
        state_dim: int,
        capacity: int,
        n_step: int,
        gamma: float,
        alpha: float,
        eps: float = 1e-6,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 3), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        # actual n-step length used for this transition (handles end-of-episode
        # cutoffs where fewer than n steps remain).
        self.n_lens = np.zeros((capacity,), dtype=np.int32)

        self.tree = _SumTree(capacity)
        self.max_priority = 1.0

        # rolling window for n-step fold
        self._buf = deque()

    def __len__(self):
        return self.tree.size

    def _push_one(self, s, a, r, ns, d, n_len):
        idx = self.tree.add(self.max_priority ** self.alpha)
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = ns
        self.dones[idx] = float(d)
        self.n_lens[idx] = n_len

    def add(self, s, a, r, ns, d):
        """Add one env transition; n-step fold happens automatically.

        ``a`` must be a length-3 iterable of int discrete-action indices.
        """
        self._buf.append((s, np.asarray(a, dtype=np.int64), float(r), ns, bool(d)))

        if len(self._buf) >= self.n_step:
            self._flush_oldest()

        if d:
            # Drain the rest using progressively shorter horizons so each
            # tail transition still gets folded with whatever remains.
            while self._buf:
                self._flush_oldest()

    def _flush_oldest(self):
        """Pop the oldest transition out of the window and push the
        n-step folded version into the sumtree."""
        s0, a0, _, _, _ = self._buf[0]
        # Discounted sum r0 + γ r1 + γ² r2 + ... up to either n_step or
        # the first terminal in the window.
        R = 0.0
        gamma_pow = 1.0
        ns_final = None
        d_final = False
        n_len = 0
        for s_, a_, r_, ns_, d_ in self._buf:
            R += gamma_pow * r_
            gamma_pow *= self.gamma
            ns_final = ns_
            d_final = d_
            n_len += 1
            if d_:
                break
            if n_len >= self.n_step:
                break

        self._push_one(s0, a0, R, ns_final, d_final, n_len)
        self._buf.popleft()

    def sample(self, batch_size: int, beta: float):
        if len(self) == 0:
            raise RuntimeError("Buffer is empty")

        # Segmented sampling for diversity (Schaul et al.)
        segment = self.tree.total / batch_size
        idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            data_idx, p = self.tree.get(s)
            idxs[i] = data_idx
            priorities[i] = p

        sampling_probs = priorities / max(self.tree.total, 1e-12)
        # Importance-sampling weights, normalized by max for stability.
        weights = (len(self) * sampling_probs) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)

        to_t = lambda x: torch.from_numpy(x).to(self.device)
        batch = {
            "states": to_t(self.states[idxs]),
            "actions": to_t(self.actions[idxs]),
            "rewards": to_t(self.rewards[idxs]),
            "next_states": to_t(self.next_states[idxs]),
            "dones": to_t(self.dones[idxs]),
            "n_lens": to_t(self.n_lens[idxs].astype(np.float32)),
            "idxs": idxs,                       # numpy, used for later update
            "weights": to_t(weights.astype(np.float32)),
        }
        return batch

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        """Update priorities for the given samples.

        td_errors should already be |TD-error| (positive). The buffer
        applies the α exponent internally.
        """
        prios = np.abs(td_errors) + self.eps
        prios = prios ** self.alpha
        for idx, p in zip(idxs, prios):
            self.tree.update(int(idx), float(p))
        self.max_priority = max(self.max_priority, float(prios.max()))
