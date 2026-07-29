"""Gradient-imbalance instrumentation.

Reproduces the diagnostic behind Fig. 1(b) of Q^2: the per-branch gradient
energy ``G_i = ||dL/dF~_i||_2`` at a fusion (concat) node, traced over training
steps. Under low-bit QAT the deep branch dominates; Q-GBFusion equalises them.

Usage::

    probe = BranchGradientProbe(num_branches=2)
    # register on the branch feature tensors of a concat node:
    for i, feat in enumerate(branches):
        feat.retain_grad()
        feat.register_hook(probe.hook(i))
    ...
    loss.backward()
    energies = probe.collect()   # tensor of shape (K,)
    probe.log(step)              # appends to internal history
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class BranchGradientProbe:
    """Captures per-branch L2 gradient norms at a fusion node (Fig. 1b)."""

    def __init__(self, num_branches: int) -> None:
        self.K = num_branches
        self._grads: List[Optional[torch.Tensor]] = [None] * num_branches
        self.history: List[Dict] = []  # rows: {step, G_0, G_1, ...}

    def hook(self, idx: int):
        def _h(grad: torch.Tensor) -> None:
            self._grads[idx] = grad.detach()

        return _h

    def reset(self) -> None:
        self._grads = [None] * self.K

    def collect(self) -> torch.Tensor:
        out = torch.zeros(self.K)
        for i, g in enumerate(self._grads):
            if g is not None:
                out[i] = g.norm(2).item()
        return out

    def log(self, step: int) -> Dict:
        G = self.collect()
        row = {"step": step}
        for i in range(self.K):
            row[f"G_{i}"] = float(G[i])
        # ratio of max/min branch energy (imbalance metric)
        nz = [v for v in row.values() if isinstance(v, float) and v > 0]
        if len(nz) >= 2:
            row["ratio"] = max(nz) / min(nz)
        else:
            row["ratio"] = float("nan")
        self.history.append(row)
        return row
