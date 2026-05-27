import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, EPSILON
from utils.rbf import RBFInfluenceFunction


def _make_gaussian_kernel(sigma: float, size: int) -> torch.Tensor:
    half = size // 2
    t    = torch.arange(-half, half + 1, dtype=torch.float32)
    g1d  = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    g1d  = g1d / g1d.sum()
    return g1d.outer(g1d).view(1, 1, size, size)


def _gaussian_blur(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv2d(u, kernel, padding=kernel.shape[-1] // 2)


def _gray_level_indicator(
    u: torch.Tensor,
    nu: float,
    K_thresh: float,
    blur_kernel: torch.Tensor,
) -> torch.Tensor:
    """
    g(u_ξ, |∇u_ξ|) = b(u_ξ) · c(|∇u_ξ|)   — Majee eq.(2.5)

    b(u_ξ) = 2|u_ξ|^ν / (M^ν + |u_ξ|^ν)    gray-level part
    c(|∇u_ξ|) = 1 / (1 + (|∇u_ξ|/K)²)      edge-stopping part

    K_thresh must be ~128 for [0,255] images (BUG 1 fix in config.py).
    """
    # BUG 6 FIX: kernel already on correct device via register_buffer
    kernel = blur_kernel.to(dtype=u.dtype)
    u_xi   = _gaussian_blur(u, kernel)

    # Gray-level factor b
    u_abs = u_xi.abs().clamp(min=1e-8)
    B     = u.shape[0]
    M     = u_abs.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
    b     = (2.0 * u_abs ** nu) / (M ** nu + u_abs ** nu)

    # Edge-stopping factor c
    u_pad_x = F.pad(u_xi, (1, 1, 0, 0), mode="replicate")
    u_pad_y = F.pad(u_xi, (0, 0, 1, 1), mode="replicate")
    gx  = (u_pad_x[:, :, :, 2:] - u_pad_x[:, :, :, :-2]) / 2.0
    gy  = (u_pad_y[:, :, 2:, :] - u_pad_y[:, :, :-2, :]) / 2.0
    mag = (gx ** 2 + gy ** 2).sqrt()
    c   = 1.0 / (1.0 + (mag / K_thresh) ** 2)

    return b * c


class InertialDiffusionStage(nn.Module):

    def __init__(
        self,
        filter_bank:   torch.Tensor,
        num_centers:   int   = 63,
        gamma_inertia: float = GAMMA_INERTIA,
        tau:           float = 0.2,
        nu:            float = NU,
        sigma_smooth:  float = SIGMA_SMOOTH,
        K_thresh:      float = K,
        use_g_func:    bool  = True,
    ):
        super().__init__()

        self.register_buffer("Ki", filter_bank)   # (Nk,1,m,m) — fixed
        self.Nk = filter_bank.shape[0]
        ksz     = filter_bank.shape[-1]

        # BUG 6 FIX: build on CPU, register_buffer handles device movement
        self.register_buffer("blur_kernel",
                             _make_gaussian_kernel(sigma_smooth, ksz))

        self.gamma_inertia = gamma_inertia
        self.tau           = tau
        self.nu            = nu
        self.K_thresh      = K_thresh
        self.use_g_func    = use_g_func

        # Learnable: RBF influence function φ_i (one per stage)
        self.phi = RBFInfluenceFunction(
            num_filters=self.Nk,
            num_centers=num_centers,
        )
        # Learnable: fidelity weight λ^t (inactive, h=0 per paper §2)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_t(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def _divergence_term(self, u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki    = self.Ki[i:i+1]
            r_i   = F.conv2d(u, ki, padding=pad)
            phi_r = self.phi(r_i, filter_idx=i)
            flux  = phi_r * g
            ki_T  = ki.flip(-1).flip(-2)
            out   = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    def forward(
        self,
        u_cur: torch.Tensor,
        u_prv: torch.Tensor,
        f:     torch.Tensor,
    ) -> torch.Tensor:
        """
        (1+γτ) u^{n+1} = (2+γτ) u^n - u^{n-1}
                          + τ² · div_term(u^n) + λ · (f - u^n)
        Returns u^{n+1} clamped to [0,255].
        """
        tau = self.tau
        gam = self.gamma_inertia

        if self.use_g_func:
            g = _gray_level_indicator(
                u_cur, self.nu, self.K_thresh,
                self.blur_kernel.to(dtype=u_cur.dtype),
            )
        else:
            g = torch.ones_like(u_cur)

        div_term  = self._divergence_term(u_cur, g)
        fidelity  = self.lambda_t * (f - u_cur)
        numer     = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term + fidelity
        u_next    = numer / (1.0 + gam * tau)
        return u_next.clamp(0.0, 255.0)
