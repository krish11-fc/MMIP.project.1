import os
import torch
import torch.nn as nn
from config import (
    NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS,
)
from .learn_k_stage import LearnKInertialDiffusionStage

class LearnKInertialTNRDNetwork(nn.Module):
    """
    Variant that learns: filter bank k_i, gamma, RBF φ_i, lambda.
    Filters are initialized from DCT warm-start but then trained.
    """

    def __init__(
        self,
        num_stages:    int   = NUM_STAGES,
        num_filters:   int   = NUM_FILTERS,
        filter_size:   int   = FILTER_SIZE,
        gamma_init:    float = 0.5,
        sigma_smooth:  float = SIGMA_SMOOTH,
        nu:            float = NU,
        K_thresh:      float = K,
        num_centers:   int   = RBF_NUM_CENTERS,
        use_g_func:    bool  = True,
        device:        torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T             = num_stages

        self.stages = nn.ModuleList([
            LearnKInertialDiffusionStage(
                num_filters   = num_filters,
                filter_size   = filter_size,
                num_centers   = num_centers,
                gamma_init    = gamma_init,
                tau           = 0.2,
                nu            = nu,
                sigma_smooth  = sigma_smooth,
                K_thresh      = K_thresh,
                use_g_func    = use_g_func,
            )
            for _ in range(num_stages)
        ])

    def forward(self, f, active_stages=None):
        n     = active_stages if active_stages is not None else self.T
        u_prv = f.clone()
        u_cur = f.clone()
        stage_outputs = []
        for t in range(n):
            u_nxt = self.stages[t](u_cur, u_prv, f)
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
        return [p for p in self.stages[stage_idx].parameters() if p.requires_grad]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        sd = torch.load(path, map_location="cpu")
        self.load_state_dict(sd)

    def print_param_summary(self) -> None:
        total   = sum(p.numel() for p in self.parameters())
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_buf   = sum(b.numel() for b in self.buffers())
        print(f"  LearnKInertialTNRDNetwork  T={self.T}")
        print(f"    Learnable params   : {learned:,}  (k_i + γ + φ_i RBF + λ^t)")
        print(f"    Fixed buffers      : {n_buf:,}   (Gaussian kernels)")
        for t, stage in enumerate(self.stages):
            n_ki  = stage.Ki.numel()
            n_phi = sum(p.numel() for p in stage.phi.parameters())
            print(f"    Stage {t+1}: k_i={n_ki}  φ={n_phi}  "
                  f"γ={stage.gamma_inertia.item():.4f}  "
                  f"λ={stage.lambda_t.item():.4f}")
