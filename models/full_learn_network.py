"""
models/full_learn_network.py — T-stage network learning all PDE scalars + RBF + λ.
"""
import torch
import torch.nn as nn
from .full_learn_stage import FullLearnInertialDiffusionStage


class FullLearnInertialTNRDNetwork(nn.Module):
    """
    T-stage Inertial TNRD with all PDE scalars (K, γ, τ, ν, σ) learnable.

    Filters k_i remain frozen (fixed DCT bank). The optimizer updates
    the 5 PDE scalars + RBF φ_i weights + λ simultaneously.
    """

    def __init__(self, num_stages=10, num_filters=48, filter_size=7,
                 gamma_init=0.5, tau_init=0.2, nu_init=1.0,
                 K_init=128.0, sigma_init=1.0,
                 num_centers=63, use_g_func=True, device="cpu"):
        super().__init__()

        self.T = num_stages

        from utils.filters import build_dct_filters
        filter_bank = build_dct_filters(
            num_filters=num_filters,
            filter_size=filter_size,
            device="cpu",
        ).to(device)

        self.stages = nn.ModuleList()
        for t in range(num_stages):
            self.stages.append(FullLearnInertialDiffusionStage(
                filter_bank=filter_bank,
                num_centers=num_centers,
                gamma_init=gamma_init,
                tau_init=tau_init,
                nu_init=nu_init,
                K_init=K_init,
                sigma_init=sigma_init,
                use_g_func=use_g_func,
            ))

    def forward(self, f, active_stages=None):
        T = active_stages if active_stages is not None else self.T
        u_prv = f.clone()
        u_cur = f.clone()
        all_outputs = []
        for t in range(T):
            u_nxt = self.stages[t](u_cur, u_prv, f)
            all_outputs.append(u_nxt)
            u_prv = u_cur
            u_cur = u_nxt
        return u_cur, all_outputs

    def freeze_stages(self, up_to):
        for i, s in enumerate(self.stages):
            for p in s.parameters():
                p.requires_grad = (i >= up_to)

    def unfreeze_stage(self, stage_idx):
        for p in self.stages[stage_idx].parameters():
            p.requires_grad = True
