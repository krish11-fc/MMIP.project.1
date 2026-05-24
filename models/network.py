"""
models/network.py
Full T-stage Inertial TNRD network for Project 8.

Wraps T InertialDiffusionStage modules into a single nn.Module with:
  • greedy freeze / unfreeze utilities used by train.py
  • `active_stages` parameter for partial unrolling (ablation / greedy training)
  • save / load / print_param_summary helpers
  • output range clamped to [0, 255]

BUG 2 FIX (DataParallel):
    train.py wrapped the model in nn.DataParallel BEFORE calling
    freeze_stages / unfreeze_stage / model.save().  nn.DataParallel does not
    expose custom methods — only forward().  This caused AttributeError on
    multi-GPU or silently skipped all freezing on single GPU.

    Fix applied in train.py: unwrap with
        base = model.module if isinstance(model, nn.DataParallel) else model
    before calling any custom method.

    The network itself is unchanged; the fix lives in train.py.

Architecture recap
------------------
Fixed  : k_i (filter bank, shared across all stages), γ (damping)
Learned: φ_i^t (RBF weights, one set per stage), λ^t (fidelity, one per stage)

Forward signature
-----------------
    u_T, stage_outputs = net(f)
    u_T, stage_outputs = net(f, active_stages=3)

`stage_outputs` is a list of length `active_stages` containing u^1, u^2, … u^T.
`u_T` == stage_outputs[-1].
"""

import os
import torch
import torch.nn as nn

from config import (
    NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    RBF_NUM_CENTERS,
)
from utils.filters import build_dct_filters
from models.stage import InertialDiffusionStage


class InertialTNRDNetwork(nn.Module):
    """
    T-stage unrolled inertial PDE network.

    Parameters
    ----------
    num_stages    : T — number of unrolled stages (default 5)
    num_filters   : Nk — filter bank size (default 48 for 7×7 DCT)
    filter_size   : m — spatial kernel size (must be odd, default 7)
    gamma_inertia : fixed γ (default from config)
    sigma_smooth  : σ for gray-level indicator (default from config)
    nu            : ν exponent (default from config)
    K_thresh      : gradient threshold (default from config — 128.0 for [0,255])
    num_centers   : RBF basis count per stage (default from config)
    use_g_func    : if False, g≡1 in all stages (ablation)
    device        : construction device
    """

    def __init__(
        self,
        num_stages:    int   = NUM_STAGES,
        num_filters:   int   = NUM_FILTERS,
        filter_size:   int   = FILTER_SIZE,
        gamma_inertia: float = GAMMA_INERTIA,
        sigma_smooth:  float = SIGMA_SMOOTH,
        nu:            float = NU,
        K_thresh:      float = K,
        num_centers:   int   = RBF_NUM_CENTERS,
        use_g_func:    bool  = True,
        device:        torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.T             = num_stages
        self.gamma_inertia = gamma_inertia

        # ── Shared fixed filter bank ──────────────────────────────────────────
        filt = build_dct_filters(
            num_filters=num_filters,
            filter_size=filter_size,
            device=device,
        )  # (Nk, 1, m, m)

        # ── T independent stages ─────────────────────────────────────────────
        self.stages = nn.ModuleList([
            InertialDiffusionStage(
                filter_bank=filt,
                num_centers=num_centers,
                gamma_inertia=gamma_inertia,
                tau=0.2,
                nu=nu,
                sigma_smooth=sigma_smooth,
                K_thresh=K_thresh,
                use_g_func=use_g_func,
            )
            for _ in range(num_stages)
        ])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        f:             torch.Tensor,
        active_stages: int = None,
    ):
        """
        Run the unrolled inertial PDE for `active_stages` steps.

        Parameters
        ----------
        f             : noisy input (B,1,H,W) in [0,255]
        active_stages : number of stages to run (default: self.T)

        Returns
        -------
        u_T          : final output (B,1,H,W)
        stage_outputs: list of length active_stages
        """
        n = active_stages if active_stages is not None else self.T

        u_prv = f.clone()   # u^{-1} = f
        u_cur = f.clone()   # u^{0}  = f

        stage_outputs = []
        for t in range(n):
            u_nxt = self.stages[t](u_cur, u_prv, f)
            stage_outputs.append(u_nxt)
            u_prv = u_cur
            u_cur = u_nxt

        return u_cur, stage_outputs

    # ── Greedy training helpers ───────────────────────────────────────────────
    # NOTE (BUG 2 FIX): These methods must be called on the BASE model, not on
    # a DataParallel wrapper.  In train.py always unwrap first:
    #   base = model.module if isinstance(model, nn.DataParallel) else model
    #   base.freeze_stages(...)

    def freeze_stages(self, up_to: int) -> None:
        """Freeze all stages with index < up_to."""
        for t in range(min(up_to, self.T)):
            for p in self.stages[t].parameters():
                p.requires_grad_(False)

    def unfreeze_stage(self, stage_idx: int) -> None:
        """Unfreeze stage `stage_idx` for training."""
        for p in self.stages[stage_idx].parameters():
            p.requires_grad_(True)

    def get_stage_params(self, stage_idx: int) -> list:
        """Return list of trainable parameters of stage `stage_idx`."""
        return [p for p in self.stages[stage_idx].parameters()
                if p.requires_grad]

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        # BUG 3 FIX: always load to CPU first to avoid device mismatches,
        # then let the caller move the model to the desired device.
        sd = torch.load(path, map_location="cpu")
        self.load_state_dict(sd)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def print_param_summary(self) -> None:
        total   = sum(p.numel() for p in self.parameters())
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fixed   = total - learned
        print(f"  InertialTNRDNetwork  T={self.T}  γ={self.gamma_inertia}")
        print(f"    Total params  : {total:,}")
        print(f"    Learnable     : {learned:,}")
        print(f"    Fixed (k_i)   : {fixed:,}")
        for t, stage in enumerate(self.stages):
            np_phi = sum(p.numel() for p in stage.phi.parameters())
            print(f"    Stage {t+1}: φ params={np_phi}  λ={stage.lambda_t.item():.4f}")
