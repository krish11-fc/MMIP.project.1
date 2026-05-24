"""
models/adaptive_stage.py — Stage variants that prevent degradation with depth.

Problem: The inertial PDE (u_tt + γ·u_t) creates an oscillator. At T=5 it peaks,
beyond T=5 it diverges (PSNR drops 16.7→11.7 dB).

Three fixes:
  A. ResidualSkip — Each stage learns a residual scale α^t ∈ [0,1].
     u^{n+1} = u^n + α^t · inertial_update(u^n, u^{n-1})
     If α^t → 0, stage becomes a no-op (stable identity).

  B. LearnedDamping — γ (damping) increases with stage index.
     γ_t = γ_0 + learned_β · t
     Higher damping → less oscillation → stable at depth.

  C. GatedUpdate — A learned gate controls blending of inertial vs diffusive.
     β = sigmoid(gate_param)
     u^{n+1} = β · inertial_update + (1-β) · diffusive_update
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K
from models.stage import _gray_level_indicator, _make_gaussian_kernel
from utils.rbf import RBFInfluenceFunction


class ResidualSkipStage(nn.Module):
    """
    Residual formulation with learnable skip coefficient α^t.

    α^t is a learned scalar (logit → sigmoid) that gates the update.
    If training finds α=0 beneficial, the stage becomes identity.
    """

    def __init__(self, filter_bank, num_centers=63,
                 gamma_inertia=GAMMA_INERTIA, tau=0.2,
                 nu=NU, sigma_smooth=SIGMA_SMOOTH, K_thresh=K,
                 use_g_func=True):
        super().__init__()
        self.register_buffer("Ki", filter_bank)
        self.Nk = filter_bank.shape[0]
        ksz = filter_bank.shape[-1]
        self.register_buffer("blur_kernel", _make_gaussian_kernel(sigma_smooth, ksz))

        self.gamma_inertia = gamma_inertia
        self.tau = tau
        self.nu = nu
        self.K_thresh = K_thresh
        self.use_g_func = use_g_func

        self.phi = RBFInfluenceFunction(num_filters=self.Nk, num_centers=num_centers)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

        # Learnable residual gate: α = sigmoid(log_alpha)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_t(self):
        return F.softplus(self.log_lambda)

    @property
    def alpha(self):
        """Residual scale: α ∈ (0, 1).  α=0 → identity, α=1 → full update."""
        return torch.sigmoid(self.log_alpha)

    def _divergence_term(self, u, g):
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki = self.Ki[i:i+1]
            r_i = F.conv2d(u, ki, padding=pad)
            phi_r = self.phi(r_i, filter_idx=i)
            flux = phi_r * g
            ki_T = ki.flip(-1).flip(-2)
            out = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    def forward(self, u_cur, u_prv, f):
        tau = self.tau
        gam = self.gamma_inertia

        if self.use_g_func:
            g = _gray_level_indicator(u_cur, self.nu, self.K_thresh,
                                      self.blur_kernel.to(dtype=u_cur.dtype))
        else:
            g = torch.ones_like(u_cur)

        div_term = self._divergence_term(u_cur, g)
        numer = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term
        full_update = numer / (1.0 + gam * tau)

        a = self.alpha
        u_next = u_cur + a * (full_update - u_cur)
        return u_next.clamp(0.0, 255.0)


class LearnedDampingStage(nn.Module):
    """
    γ_t = γ_0 + β · t   — damping increases with depth.

    β is a learned scalar.  Early stages have low γ (exploratory),
    late stages have high γ (stable, damps oscillations).
    """

    def __init__(self, filter_bank, num_centers=63,
                 gamma_init=GAMMA_INERTIA, tau=0.2,
                 nu=NU, sigma_smooth=SIGMA_SMOOTH, K_thresh=K,
                 use_g_func=True, stage_idx=0):
        super().__init__()
        self.register_buffer("Ki", filter_bank)
        self.Nk = filter_bank.shape[0]
        ksz = filter_bank.shape[-1]
        self.register_buffer("blur_kernel", _make_gaussian_kernel(sigma_smooth, ksz))

        self.tau = tau
        self.nu = nu
        self.K_thresh = K_thresh
        self.use_g_func = use_g_func
        self.stage_idx = stage_idx

        # Base damping γ_0
        self.log_gamma0 = nn.Parameter(torch.tensor(gamma_init).log())
        # Damping slope β (shared across stages)
        self.log_beta = nn.Parameter(torch.tensor(0.1).log())

        self.phi = RBFInfluenceFunction(num_filters=self.Nk, num_centers=num_centers)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_t(self):
        return F.softplus(self.log_lambda)

    @property
    def gamma(self):
        """γ_t = γ_0 + β · t, clamped positive."""
        g0 = self.log_gamma0.exp()
        b = self.log_beta.exp()
        return g0 + b * self.stage_idx

    def _divergence_term(self, u, g):
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki = self.Ki[i:i+1]
            r_i = F.conv2d(u, ki, padding=pad)
            phi_r = self.phi(r_i, filter_idx=i)
            flux = phi_r * g
            ki_T = ki.flip(-1).flip(-2)
            out = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    def forward(self, u_cur, u_prv, f):
        tau = self.tau
        gam = self.gamma

        if self.use_g_func:
            g = _gray_level_indicator(u_cur, self.nu, self.K_thresh,
                                      self.blur_kernel.to(dtype=u_cur.dtype))
        else:
            g = torch.ones_like(u_cur)

        div_term = self._divergence_term(u_cur, g)
        numer = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term
        u_next = numer / (1.0 + gam * tau)
        return u_next.clamp(0.0, 255.0)


class GatedUpdateStage(nn.Module):
    """
    Blends inertial and diffusive updates with a learned gate β ∈ [0,1].

    u^{n+1} = β · inertial_update + (1-β) · diffusive_update
    """
    def __init__(self, filter_bank, num_centers=63,
                 gamma_inertia=GAMMA_INERTIA, tau=0.2,
                 nu=NU, sigma_smooth=SIGMA_SMOOTH, K_thresh=K,
                 use_g_func=True):
        super().__init__()
        self.register_buffer("Ki", filter_bank)
        self.Nk = filter_bank.shape[0]
        ksz = filter_bank.shape[-1]
        self.register_buffer("blur_kernel", _make_gaussian_kernel(sigma_smooth, ksz))

        self.gamma_inertia = gamma_inertia
        self.tau = tau
        self.nu = nu
        self.K_thresh = K_thresh
        self.use_g_func = use_g_func

        self.phi = RBFInfluenceFunction(num_filters=self.Nk, num_centers=num_centers)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.log_gate = nn.Parameter(torch.tensor(0.0))  # β = sigmoid(log_gate)

    @property
    def lambda_t(self):
        return F.softplus(self.log_lambda)

    @property
    def beta(self):
        """β ∈ (0,1): 1 = full inertial, 0 = pure diffusive."""
        return torch.sigmoid(self.log_gate)

    def _divergence_term(self, u, g):
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki = self.Ki[i:i+1]
            r_i = F.conv2d(u, ki, padding=pad)
            phi_r = self.phi(r_i, filter_idx=i)
            flux = phi_r * g
            ki_T = ki.flip(-1).flip(-2)
            out = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    def forward(self, u_cur, u_prv, f):
        tau = self.tau
        gam = self.gamma_inertia

        if self.use_g_func:
            g = _gray_level_indicator(u_cur, self.nu, self.K_thresh,
                                      self.blur_kernel.to(dtype=u_cur.dtype))
        else:
            g = torch.ones_like(u_cur)

        div_term = self._divergence_term(u_cur, g)

        # Inertial: (1+γτ) u^{n+1} = (2+γτ) u^n - u^{n-1} + τ² div
        inertial = ((2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term) / (1.0 + gam * tau)

        # Diffusive (no momentum): u^{n+1} = u^n + τ² div
        diffusive = u_cur + tau ** 2 * div_term

        b = self.beta
        u_next = b * inertial + (1.0 - b) * diffusive
        return u_next.clamp(0.0, 255.0)
