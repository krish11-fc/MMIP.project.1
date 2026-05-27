import os
import torch
import torch.nn as nn

from config import (
    NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    RBF_NUM_CENTERS, EMBED_DIM, NUM_NOISE_LEVELS,
)
from models.noise_conditional_stage_v2 import NoiseConditionalDiffusionStageV2


class NoiseConditionalTNRDNetworkV2(nn.Module):

    def __init__(
        self,
        num_stages:      int   = NUM_STAGES,
        num_filters:     int   = NUM_FILTERS,
        filter_size:     int   = FILTER_SIZE,
        gamma_inertia:   float = GAMMA_INERTIA,
        sigma_smooth:    float = SIGMA_SMOOTH,
        nu:              float = NU,
        K_thresh:        float = K,
        num_centers:     int   = RBF_NUM_CENTERS,
        use_g_func:      bool  = True,
        embed_dim:       int   = EMBED_DIM,
        num_noise_levels: int  = NUM_NOISE_LEVELS,
        device:          torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T = num_stages

        self.stages = nn.ModuleList([
            NoiseConditionalDiffusionStageV2(
                num_filters=num_filters,
                filter_size=filter_size,
                num_centers=num_centers,
                gamma_inertia=gamma_inertia,
                tau=0.2,
                nu=nu,
                sigma_smooth=sigma_smooth,
                K_thresh=K_thresh,
                use_g_func=use_g_func,
                embed_dim=embed_dim,
                num_noise_levels=num_noise_levels,
            )
            for _ in range(num_stages)
        ])

    def forward(
        self,
        f:             torch.Tensor,
        L:             int,
        active_stages: int = None,
    ):
        n = active_stages if active_stages is not None else self.T
        u_prv = f.clone()
        u_cur = f.clone()
        stage_outputs = []
        for t in range(n):
            u_nxt = self.stages[t](u_cur, u_prv, f, L)
            stage_outputs.append(u_nxt)
            u_prv = u_cur
            u_cur = u_nxt
        return u_cur, stage_outputs

    def freeze_stages(self, up_to: int) -> None:
        for t in range(min(up_to, self.T)):
            for p in self.stages[t].parameters():
                p.requires_grad_(False)

    def unfreeze_stage(self, stage_idx: int) -> None:
        for p in self.stages[stage_idx].parameters():
            p.requires_grad_(True)

    def get_stage_params(self, stage_idx: int) -> list:
        return [p for p in self.stages[stage_idx].parameters()
                if p.requires_grad]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        sd = torch.load(path, map_location="cpu")
        self.load_state_dict(sd)

    def print_param_summary(self) -> None:
        total   = sum(p.numel() for p in self.parameters())
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fixed   = total - learned
        dev = next(self.parameters()).device
        print(f"  NoiseConditionalTNRDNetworkV2  T={self.T}")
        print(f"    Total params  : {total:,}")
        print(f"    Learnable     : {learned:,}")
        print(f"    Fixed (k_i)   : {fixed:,}")
        for t, stage in enumerate(self.stages):
            np_phi    = sum(p.numel() for p in stage.phi.parameters() if p.requires_grad)
            np_embed  = sum(p.numel() for p in stage.noise_embedding.parameters() if p.requires_grad)
            np_mlp    = sum(p.numel() for p in stage.embed_mlp.parameters() if p.requires_grad)
            np_lambda = sum(p.numel() for p in stage.lambda_mlp.parameters() if p.requires_grad)
            np_pde    = sum(p.numel() for p in stage.pde_param_mlp.parameters() if p.requires_grad)
            np_g      = sum(p.numel() for p in stage.learned_g.parameters() if p.requires_grad) if stage.learned_g else 0
            np_k      = stage.Ki.numel()
            lam_val   = stage.lambda_t(
                stage.noise_embedding(torch.tensor([1], dtype=torch.long, device=dev))
            ).item()
            print(f"    Stage {t+1}: φ={np_phi}  embed={np_embed}  mlp={np_mlp}  "
                  f"λ_mlp={np_lambda}  pde={np_pde}  g={np_g}  K={np_k}  "
                  f"λ(L=1)={lam_val:.4f}")
