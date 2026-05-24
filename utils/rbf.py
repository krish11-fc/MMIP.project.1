"""
utils/rbf.py — Learnable RBF influence function φ_i(s).

    φ_i(s) = Σ_j  w_{i,j} · exp( -(s - c_j)² / (2h²) )

BUG 4 FIX: centres narrowed from ±310 to ±50.
  DCT filter responses on [0,255] images fall in ~[-50, +50].
  With ±310, ~80% of centres had zero gradient → wasted capacity.
  With ±50 and 63 centres, step h≈1.6 → fine resolution over active range.

Warm-start: w_j ← 2·c_j/(1+c_j²)  (Perona-Malik) for all filters.
"""

import torch
import torch.nn as nn
from config import RBF_NUM_CENTERS, RBF_CENTER_MIN, RBF_CENTER_MAX


class RBFInfluenceFunction(nn.Module):

    def __init__(
        self,
        num_filters:  int   = 48,
        num_centers:  int   = RBF_NUM_CENTERS,
        c_min:        float = RBF_CENTER_MIN,
        c_max:        float = RBF_CENTER_MAX,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.num_centers = num_centers

        centres = torch.linspace(c_min, c_max, num_centers)
        self.register_buffer("centres", centres)

        h = (c_max - c_min) / max(num_centers - 1, 1)
        self.register_buffer("two_h2", torch.tensor(2.0 * h * h))

        # Warm-start at Perona-Malik: φ(s) = 2s/(1+s²)
        with torch.no_grad():
            warm = 2.0 * centres / (1.0 + centres ** 2)
            warm = warm.unsqueeze(0).expand(num_filters, -1).clone()
        self.weights = nn.Parameter(warm)   # (F, C)

    def forward(self, s: torch.Tensor, filter_idx: int) -> torch.Tensor:
        w     = self.weights[filter_idx]        # (C,)
        shape = s.shape
        flat  = s.reshape(-1, 1)                # (N,1)
        diff  = flat - self.centres.unsqueeze(0)  # (N,C)
        rbf   = torch.exp(-diff ** 2 / self.two_h2)  # (N,C)
        out   = (rbf * w).sum(dim=1)            # (N,)
        return out.reshape(shape)

    def forward_modulated(
        self,
        s:          torch.Tensor,
        filter_idx: int,
        scale:      torch.Tensor,
        shift:      torch.Tensor,
    ) -> torch.Tensor:
        phi_val = self.forward(s, filter_idx)
        if isinstance(scale, torch.Tensor) and scale.dim() > 0:
            scale = scale.reshape(-1, 1, 1, 1)
            shift = shift.reshape(-1, 1, 1, 1)
        return phi_val * scale + shift
