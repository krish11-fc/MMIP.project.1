import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTNRDWrapper(nn.Module):
    """Multi-scale processing wrapper (MSND-style).
    
    Processes input at multiple resolutions, then fuses results.
    Based on "Image Denoising via Multi-scale Nonlinear Diffusion Models" (SIAM 2017).
    
    For each scale:
      1. Downsample input by factor `s`
      2. Run through base_network
      3. Upsample back to original resolution
    
    All scale outputs are averaged (or learned-weighted) for final result.
    
    Args:
        base_network: the underlying TNRD/NCTDN network
        scales: list of downsampling factors (e.g. [1.0, 2.0, 3.0])
        align_corners: for grid_sample / interpolate
    """
    
    def __init__(self, base_network, scales=None, align_corners=False):
        super().__init__()
        self.base = base_network
        self.scales = scales if scales is not None else [1.0, 2.0]
        self.align_corners = align_corners
        
        # Optional: learnable fusion weights (one per scale)
        self.fusion_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
    
    # ── Forwarded methods to base network ──
    @property
    def T(self):
        return self.base.T

    @property
    def stages(self):
        return self.base.stages

    def freeze_stages(self, up_to: int) -> None:
        self.base.freeze_stages(up_to)

    def unfreeze_stage(self, stage_idx: int) -> None:
        self.base.unfreeze_stage(stage_idx)

    def print_param_summary(self) -> None:
        self.base.print_param_summary()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        sd = torch.load(path, map_location="cpu")
        # Try loading as wrapper state_dict first (has "base.stages" or "fusion_weights")
        if any(k.startswith("base.stages") for k in sd) or "fusion_weights" in sd:
            self.load_state_dict(sd, strict=False)
        else:
            # Legacy: bare base state_dict (keys like "stages.0.Ki")
            self.base.load_state_dict(sd, strict=False)

    def get_stage_params(self, stage_idx: int) -> list:
        return self.base.get_stage_params(stage_idx)

    def forward_single_scale(self, f, L=None, active_stages=None):
        if L is not None:
            return self.base(f, L=L, active_stages=active_stages)
        return self.base(f, active_stages=active_stages)

    def _downsample(self, x, factor):
        if factor <= 1.0 + 1e-6:
            return x
        h, w = x.shape[-2:]
        nh = max(1, round(h / factor))
        nw = max(1, round(w / factor))
        return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=self.align_corners)
    
    def _upsample(self, x, target_size):
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=self.align_corners)
    
    def forward(self, f, L=None, active_stages=None):
        """Forward pass at multiple scales.
        
        Args:
            f: (B, 1, H, W) noisy input
            L: noise level (for NCTDN models)
            active_stages: number of active diffusion stages
        
        Returns:
            u_pred: (B, 1, H, W) fused prediction
            all_outputs: same as base_network's all_outputs (at full resolution)
        """
        target_size = f.shape[-2:]
        outputs = []
        base_outputs = None
        
        for s, factor in enumerate(self.scales):
            x_s = self._downsample(f, factor)
            
            if hasattr(self.base, 'forward') and L is not None:
                # For NCTDN models that take L
                u_s, all_out = self.base(x_s, L=L, active_stages=active_stages)
            else:
                u_s, all_out = self.base(x_s, active_stages=active_stages)
            
            u_full = self._upsample(u_s, target_size)
            outputs.append(u_full)
            
            if factor <= 1.0 + 1e-6:
                base_outputs = all_out
        
        # Fusion: weighted average
        w = F.softmax(self.fusion_weights, dim=0)
        u_pred = sum(w[s] * outputs[s] for s in range(len(self.scales)))
        
        return u_pred, base_outputs
