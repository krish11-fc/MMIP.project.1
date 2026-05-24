"""
models/stage.py — Single unrolled step of the inertial telegraph diffusion PDE.

Implements paper §4 eq. d:
    (1 + γτ) u^{n+1} = (2 + γτ) u^n  -  u^{n-1}
                       + τ² · Σ_i  K_i^T [ φ_i(K_i * u^n) · g(u_ξ, |∇u_ξ|) ]

Fixed  : K_i (filter bank, buffer), γ (plain float)
Learned: φ_i (RBF weights), λ^t (fidelity, inactive per paper h=0)

BUG 6 FIX: blur_kernel registered as buffer via register_buffer().
    It now moves automatically when model.to(device) is called.
    Inside forward() we only cast dtype, not device — no per-call GPU transfer.

BUG 1 context: K_thresh is now 128.0 (from config.py) not 0.5.
    c(|∇u_ξ|) = 1/(1+(|∇u_ξ|/128)²) — correct scale for [0,255] images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, EPSILON
from utils.rbf import RBFInfluenceFunction


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian kernel builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_gaussian_kernel(sigma: float, size: int) -> torch.Tensor:
    """Return (1,1,size,size) Gaussian kernel on CPU, float32."""
    half = size // 2
    t    = torch.arange(-half, half + 1, dtype=torch.float32)
    g1d  = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    g1d  = g1d / g1d.sum()
    return g1d.outer(g1d).view(1, 1, size, size)


def _gaussian_blur(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv2d(u, kernel, padding=kernel.shape[-1] // 2)


# ─────────────────────────────────────────────────────────────────────────────
# Gray-level indicator  g(u_ξ, |∇u_ξ|) — Majee eq.(2.5)
# ─────────────────────────────────────────────────────────────────────────────

def _gray_level_indicator(
    u:           torch.Tensor,    # (B,1,H,W) in [0,255]
    nu:          float,
    K_thresh:    float,           # 128.0 for [0,255] images (BUG 1 fixed in config)
    blur_kernel: torch.Tensor,    # (1,1,ksz,ksz) — already on correct device
) -> torch.Tensor:
    """
    g(u_ξ, |∇u_ξ|) = b(u_ξ) · c(|∇u_ξ|)

    b(u_ξ) = 2|u_ξ|^ν / (M_ξ^ν + |u_ξ|^ν)   gray-level part  (0,1]
    c(|∇u_ξ|) = 1 / (1 + (|∇u_ξ|/K)²)        edge-stopping    (0,1]

    BUG 6 FIX: kernel is a registered buffer; only dtype cast needed here.
    """
    # BUG 6 FIX: device already correct via register_buffer; only cast dtype
    kernel = blur_kernel.to(dtype=u.dtype)
    u_xi   = _gaussian_blur(u, kernel)

    # Gray-level factor b(u_ξ)
    u_abs = u_xi.abs().clamp(min=1e-8)
    B     = u.shape[0]
    M     = u_abs.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
    b     = (2.0 * u_abs ** nu) / (M ** nu + u_abs ** nu)

    # Edge-stopping factor c(|∇u_ξ|) — central differences on u_ξ
    u_pad_x = F.pad(u_xi, (1, 1, 0, 0), mode="replicate")
    u_pad_y = F.pad(u_xi, (0, 0, 1, 1), mode="replicate")
    gx  = (u_pad_x[:, :, :, 2:] - u_pad_x[:, :, :, :-2]) / 2.0
    gy  = (u_pad_y[:, :, 2:, :] - u_pad_y[:, :, :-2, :]) / 2.0
    mag = (gx ** 2 + gy ** 2).sqrt()
    # K_thresh = 128.0 (BUG 1 fix): c stays near 1 for typical gradients
    c   = 1.0 / (1.0 + (mag / K_thresh) ** 2)

    return b * c   # (B,1,H,W) in (0,1]


# ─────────────────────────────────────────────────────────────────────────────
# InertialDiffusionStage
# ─────────────────────────────────────────────────────────────────────────────

class InertialDiffusionStage(nn.Module):
    """
    One stage of the unrolled inertial telegraph diffusion PDE.

    Parameters
    ----------
    filter_bank  : (Nk,1,m,m) fixed filter bank — stored as buffer
    num_centers  : RBF centre count for φ_i
    gamma_inertia: fixed γ (damping, plain float — NOT a parameter)
    tau          : fixed τ (time step, 0.2 per paper §5)
    nu           : ν in gray-level indicator
    sigma_smooth : σ for Gaussian blur u_ξ
    K_thresh     : 128.0 for [0,255] images (BUG 1 fix)
    use_g_func   : if False, g≡1 (ablation switch)
    """

    def __init__(
        self,
        filter_bank:   torch.Tensor,
        num_centers:   int   = 63,
        gamma_inertia: float = GAMMA_INERTIA,
        tau:           float = 0.2,
        nu:            float = NU,
        sigma_smooth:  float = SIGMA_SMOOTH,
        K_thresh:      float = K,           # 128.0 after BUG 1 fix
        use_g_func:    bool  = True,
    ):
        super().__init__()

        # Fixed filter bank — buffer, never in model.parameters()
        self.register_buffer("Ki", filter_bank)   # (Nk,1,m,m)
        self.Nk = filter_bank.shape[0]
        ksz     = filter_bank.shape[-1]

        # BUG 6 FIX: register_buffer so it moves with model.to(device)
        # No dtype arg needed — built as float32, cast in forward only
        self.register_buffer(
            "blur_kernel",
            _make_gaussian_kernel(sigma_smooth, ksz)   # (1,1,ksz,ksz) CPU float32
        )

        # Fixed hyperparameters — plain Python floats, NOT nn.Parameters
        self.gamma_inertia = gamma_inertia
        self.tau           = tau
        self.nu            = nu
        self.K_thresh      = K_thresh      # 128.0
        self.use_g_func    = use_g_func

        # ── Learned parameters ────────────────────────────────────────────────
        # φ_i : RBF influence function (one per stage, shared across all filters)
        # Warm-started at Perona-Malik  φ(s) = 2s/(1+s²) with centres ±300.
        self.phi = RBFInfluenceFunction(
            num_filters=self.Nk,
            num_centers=num_centers,
        )
        # λ^t : fidelity weight — paper sets h(I₀,I)=0 so inactive,
        # but kept for completeness / ablation
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_t(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    # ── Divergence term  Σ_i K_i^T [ φ_i(K_i*u) · g ] ──────────────────────

    def _divergence_term(
        self,
        u: torch.Tensor,    # (B,1,H,W)
        g: torch.Tensor,    # (B,1,H,W)
    ) -> torch.Tensor:
        """
        Σ_i  K_i^T ( φ_i(K_i * u) · g )

        K_i * u    : filter response (forward conv)
        φ_i(·)     : learned RBF influence (replaces fixed Perona-Malik c)
        × g        : modulate by gray-level indicator
        K_i^T      : transpose conv (180°-flipped kernel, same padding)
        """
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki    = self.Ki[i:i+1]                       # (1,1,m,m)
            r_i   = F.conv2d(u, ki, padding=pad)         # (B,1,H,W)
            phi_r = self.phi(r_i, filter_idx=i)          # (B,1,H,W)
            flux  = phi_r * g                            # (B,1,H,W)
            ki_T  = ki.flip(-1).flip(-2)
            out   = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        u_cur: torch.Tensor,    # u^n       (B,1,H,W) in [0,255]
        u_prv: torch.Tensor,    # u^{n-1}   (B,1,H,W)
        f:     torch.Tensor,    # noisy obs (B,1,H,W) — unused (h=0)
    ) -> torch.Tensor:
        """
        One finite-difference step of the inertial PDE (paper §4, eq. d):

            (1+γτ) u^{n+1} = (2+γτ) u^n - u^{n-1} + τ² · div_term(u^n)

        Returns u^{n+1} clamped to [0, 255].
        """
        tau = self.tau
        gam = self.gamma_inertia

        # Compute diffusivity g or use g≡1 for ablation
        if self.use_g_func:
            # BUG 6 FIX: blur_kernel is on correct device via register_buffer;
            # only dtype cast needed (float16/32 consistency)
            g = _gray_level_indicator(
                u_cur, self.nu, self.K_thresh,
                self.blur_kernel.to(dtype=u_cur.dtype),
            )
        else:
            g = torch.ones_like(u_cur)

        div_term = self._divergence_term(u_cur, g)

        # Inertial telegraph update (paper §4, eq. d)
        numer  = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term
        u_next = numer / (1.0 + gam * tau)
        return u_next.clamp(0.0, 255.0)
