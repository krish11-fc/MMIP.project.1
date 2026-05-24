"""
models/network.py — T-stage Inertial TNRD network for Project 8.

BUG 3 FIX: load() now unconditionally uses map_location="cpu".
    Previous code: sd = torch.load(path, map_location=map_location or "cpu")
    If map_location was a torch.device object (truthy), "or 'cpu'" was skipped,
    causing device mismatch when the saving device ≠ loading device.
    Fix: always map to "cpu" first; caller already has model on correct device.

BUG 2 NOTE: DataParallel unwrapping is handled in train.py, not here.
    train.py does: base = model.module if isinstance(model, nn.DataParallel) else model
    before calling freeze_stages / unfreeze_stage / save.
"""

import os
import torch
import torch.nn as nn

from config import (
    NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS,
)
from utils.filters import build_dct_filters
from models.stage import InertialDiffusionStage


class InertialTNRDNetwork(nn.Module):
    """
    T-stage unrolled inertial PDE network.

    Fixed  : k_i (DCT filter bank, shared buffer), γ (plain float)
    Learned: φ_i^t (RBF weights, one set per stage), λ^t (fidelity per stage)
    """

    def __init__(
        self,
        num_stages:    int   = NUM_STAGES,
        num_filters:   int   = NUM_FILTERS,
        filter_size:   int   = FILTER_SIZE,
        gamma_inertia: float = GAMMA_INERTIA,
        sigma_smooth:  float = SIGMA_SMOOTH,
        nu:            float = NU,
        K_thresh:      float = K,            # 128.0 after BUG 1 fix
        num_centers:   int   = RBF_NUM_CENTERS,
        use_g_func:    bool  = True,
        device:        torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T             = num_stages
        self.gamma_inertia = gamma_inertia   # plain float — never a parameter

        # Shared fixed filter bank (buffer — not in model.parameters())
        filt = build_dct_filters(
            num_filters=num_filters,
            filter_size=filter_size,
            device=device,
        )  # (Nk, 1, m, m), requires_grad=False

        # T independent stages — each has its own φ_i^t and λ^t
        self.stages = nn.ModuleList([
            InertialDiffusionStage(
                filter_bank   = filt,
                num_centers   = num_centers,
                gamma_inertia = gamma_inertia,
                tau           = 0.2,
                nu            = nu,
                sigma_smooth  = sigma_smooth,
                K_thresh      = K_thresh,
                use_g_func    = use_g_func,
            )
            for _ in range(num_stages)
        ])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, f: torch.Tensor, active_stages: int = None):
        """
        f             : noisy input (B,1,H,W) in [0,255]
        active_stages : stages to run (default: self.T); used in greedy training

        Returns (u_T, stage_outputs) where stage_outputs is a list of T tensors.
        """
        n     = active_stages if active_stages is not None else self.T
        u_prv = f.clone()   # u^{-1} = f  (I_t(x,0)=0 → u^0=u^{-1}=f)
        u_cur = f.clone()   # u^0  = f

        stage_outputs = []
        for t in range(n):
            u_nxt = self.stages[t](u_cur, u_prv, f)
            stage_outputs.append(u_nxt)
            u_prv = u_cur
            u_cur = u_nxt

        return u_cur, stage_outputs

    # ── Greedy training helpers ───────────────────────────────────────────────
    # IMPORTANT (BUG 2): Always call these on the BASE model, NOT on a
    # DataParallel wrapper.  In train.py:
    #   base = model.module if isinstance(model, nn.DataParallel) else model
    #   base.freeze_stages(stage_idx)

    def freeze_stages(self, up_to: int) -> None:
        """Freeze parameters of stages 0 .. up_to-1."""
        for t in range(min(up_to, self.T)):
            for p in self.stages[t].parameters():
                p.requires_grad_(False)

    def unfreeze_stage(self, stage_idx: int) -> None:
        """Unfreeze parameters of stage stage_idx."""
        for p in self.stages[stage_idx].parameters():
            p.requires_grad_(True)

    def get_stage_params(self, stage_idx: int) -> list:
        return [p for p in self.stages[stage_idx].parameters()
                if p.requires_grad]

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        # BUG 3 FIX: always load to "cpu" first — no device mismatch possible.
        # The model is already on the correct device; load_state_dict handles transfer.
        sd = torch.load(path, map_location="cpu")
        self.load_state_dict(sd)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def print_param_summary(self) -> None:
        total   = sum(p.numel() for p in self.parameters())
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_buf   = sum(b.numel() for b in self.buffers())
        print(f"  InertialTNRDNetwork  T={self.T}  γ={self.gamma_inertia}  K={self.stages[0].K_thresh}")
        print(f"    Learnable params   : {learned:,}  (φ_i RBF + λ^t per stage)")
        print(f"    Fixed buffers (k_i): {n_buf:,}   (DCT filters + Gaussian kernels)")
        for t, stage in enumerate(self.stages):
            n_phi = sum(p.numel() for p in stage.phi.parameters())
            print(f"    Stage {t+1}: φ_params={n_phi}  λ={stage.lambda_t.item():.4f}  "
                  f"K={stage.K_thresh}  use_g={stage.use_g_func}")
