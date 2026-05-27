import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage import _gray_level_indicator, _make_gaussian_kernel


class FullLearnInertialDiffusionStage(nn.Module):
    """
    One stage of inertial telegraph diffusion with ALL PDE scalars learnable.

    Same update as InertialDiffusionStage:
      (1+γτ) u^{n+1} = (2+γτ) u^n - u^{n-1} + τ² · div(u^n)

    But γ, τ, K, ν, σ are nn.Parameters constrained via softplus/log.
    Filters k_i remain frozen (fixed DCT bank).
    """

    def __init__(self, filter_bank, num_centers=63,
                 gamma_init=0.5, tau_init=0.2, nu_init=1.0,
                 K_init=128.0, sigma_init=1.0,
                 use_g_func=True):
        super().__init__()

        self.register_buffer("Ki", filter_bank)  # (Nk,1,m,m) — frozen
        self.Nk = filter_bank.shape[0]
        ksz = filter_bank.shape[-1]
        self.register_buffer("_blur_kernel",
                             _make_gaussian_kernel(sigma_init, ksz))

        self.use_g_func = use_g_func

        # ── Learnable PDE scalars (softplus-parameterised for positivity) ──
        # softplus: smoother gradients near zero, bounded gradient magnitude
        import numpy as np
        def _inv_softplus(y):
            return np.log(np.expm1(max(y, 1e-6)))
        self._gamma_raw  = nn.Parameter(torch.tensor(_inv_softplus(gamma_init), dtype=torch.float32))
        self._tau_raw    = nn.Parameter(torch.tensor(_inv_softplus(tau_init), dtype=torch.float32))
        self._nu_raw     = nn.Parameter(torch.tensor(_inv_softplus(nu_init), dtype=torch.float32))
        self._K_raw      = nn.Parameter(torch.tensor(_inv_softplus(K_init), dtype=torch.float32))
        self._sigma_raw  = nn.Parameter(torch.tensor(_inv_softplus(sigma_init), dtype=torch.float32))

        # ── Learnable: RBF influence function φ_i ──
        from utils.rbf import RBFInfluenceFunction
        self.phi = RBFInfluenceFunction(
            num_filters=self.Nk, num_centers=num_centers,
        )
        # ── Learnable: fidelity weight λ^t (inactive per paper h=0) ──
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    # ── Parameter properties ──────────────────────────────────────────

    @property
    def gamma(self): return F.softplus(self._gamma_raw) + 1e-3

    @property
    def tau(self): return F.softplus(self._tau_raw) + 1e-3

    @property
    def nu(self): return F.softplus(self._nu_raw) + 1e-3

    @property
    def K_thresh(self): return F.softplus(self._K_raw) + 0.1

    @property
    def sigma(self): return F.softplus(self._sigma_raw) + 1e-3

    @property
    def lambda_t(self): return F.softplus(self.log_lambda)

    # ── Blur kernel (regenerated when σ changes) ──────────────────────

    def _get_blur_kernel(self):
        s = self.sigma.item()
        ksz = self.Ki.shape[-1]
        half = ksz // 2
        t = torch.arange(-half, half + 1, dtype=self.Ki.dtype,
                         device=self.Ki.device)
        g1d = torch.exp(-t ** 2 / (2.0 * s ** 2))
        g1d = g1d / g1d.sum()
        kernel = g1d.outer(g1d).view(1, 1, ksz, ksz)
        return kernel

    # ── Divergence term ──────────────────────────────────────────────

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

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, u_cur, u_prv, f):
        tau = self.tau
        gam = self.gamma

        if self.use_g_func:
            kernel = self._get_blur_kernel()
            g = _gray_level_indicator(
                u_cur, self.nu, self.K_thresh, kernel,
            )
        else:
            g = torch.ones_like(u_cur)

        div_term = self._divergence_term(u_cur, g)
        fidelity = self.lambda_t * (f - u_cur)
        numer = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term + fidelity
        u_next = numer / (1.0 + gam * tau)
        return u_next.clamp(0.0, 255.0)
