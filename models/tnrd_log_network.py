import os
import torch
import torch.nn as nn
from config import NUM_STAGES, NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS
from utils.filters import build_dct_filters
from models.tnrd_log_stage import TNRDLogDiffusionStage

class TNRDLogNetwork(nn.Module):
    """
    TNRD for log-transformed speckle (Chen & Pock 2016 adapted).
    
    Forward:
      1. z = log(1 + f)        — log transform (multiplicative → additive)
      2. Run T-stage diffusion on z (no inertia, no g-function)
      3. u = exp(z_T) - 1      — inverse transform
    
    Loss is computed in log domain (MSE on z).
    """
    def __init__(
        self,
        num_stages:    int   = NUM_STAGES,
        num_filters:   int   = NUM_FILTERS,
        filter_size:   int   = FILTER_SIZE,
        num_centers:   int   = RBF_NUM_CENTERS,
        device:        torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T = num_stages

        # Shared fixed filter bank (same as original TNRD)
        filt = build_dct_filters(
            num_filters=num_filters,
            filter_size=filter_size,
            device=device,
        )

        self.stages = nn.ModuleList([
            TNRDLogDiffusionStage(
                filter_bank=filt,
                num_centers=num_centers,
                tau=0.2,
            )
            for _ in range(num_stages)
        ])

    def forward(self, f, active_stages=None):
        n = active_stages if active_stages is not None else self.T
        
        # Log transform
        z = torch.log(1.0 + f)
        
        z_cur = z.clone()
        stage_outputs = []
        for t in range(n):
            z_nxt = self.stages[t](z_cur)
            stage_outputs.append(z_nxt)
            z_cur = z_nxt
        
        # Inverse transform
        u = torch.exp(z_cur) - 1.0
        return u.clamp(0.0, 255.0), stage_outputs

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
        print(f"  TNRDLogNetwork  T={self.T}")
        print(f"    Learnable params: {learned:,}  (RBF φ_i + λ^t per stage)")
        print(f"    Fixed buffers   : {n_buf:,}   (DCT filters)")
        for t, stage in enumerate(self.stages):
            n_phi = sum(p.numel() for p in stage.phi.parameters())
            print(f"    Stage {t+1}: φ_params={n_phi}  λ={stage.lambda_t.item():.4f}")
