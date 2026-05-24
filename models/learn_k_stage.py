import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, EPSILON
from utils.rbf import RBFInfluenceFunction
from models.stage import _gaussian_blur, _gray_level_indicator, _make_gaussian_kernel

class LearnKInertialDiffusionStage(nn.Module):
    def __init__(
        self,
        num_filters:   int   = 48,
        filter_size:   int   = 7,
        num_centers:   int   = 63,
        gamma_init:    float = GAMMA_INERTIA,
        tau:           float = 0.2,
        nu:            float = NU,
        sigma_smooth:  float = SIGMA_SMOOTH,
        K_thresh:      float = K,
        use_g_func:    bool  = True,
    ):
        super().__init__()
        ksz = filter_size

        # ── Learnable filter bank (was fixed buffer) ────────────────────────
        from utils.filters import build_dct_filters
        filt_init = build_dct_filters(num_filters, filter_size, device="cpu")
        self.Ki = nn.Parameter(filt_init.clone())   # (Nk,1,m,m) learnable!

        self.Nk = num_filters

        # BUG 6 FIX: register_buffer so it moves with model.to(device)
        self.register_buffer(
            "blur_kernel",
            _make_gaussian_kernel(sigma_smooth, ksz)
        )

        # ── Learnable gamma (was plain float) ───────────────────────────────
        self.log_gamma = nn.Parameter(torch.tensor(gamma_init).log())

        self.tau           = tau
        self.nu            = nu
        self.K_thresh      = K_thresh
        self.use_g_func    = use_g_func

        # Learned RBF influence function φ_i (same as before)
        self.phi = RBFInfluenceFunction(
            num_filters=self.Nk,
            num_centers=num_centers,
        )
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def gamma_inertia(self):
        return self.log_gamma.exp()

    @property
    def lambda_t(self):
        return F.softplus(self.log_lambda)

    def _divergence_term(self, u, g):
        pad = (self.Ki.shape[-1] - 1) // 2 if self.Ki.shape[-1] % 2 == 1 else self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki    = self.Ki[i:i+1]
            r_i   = F.conv2d(u, ki, padding=pad)
            phi_r = self.phi(r_i, filter_idx=i)
            flux  = phi_r * g
            ki_T  = ki.flip(-1).flip(-2)
            out   = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    def forward(self, u_cur, u_prv, f):
        tau = self.tau
        gam = self.gamma_inertia

        if self.use_g_func:
            g = _gray_level_indicator(
                u_cur, self.nu, self.K_thresh,
                self.blur_kernel.to(dtype=u_cur.dtype),
            )
        else:
            g = torch.ones_like(u_cur)

        div_term = self._divergence_term(u_cur, g)
        fidelity  = self.lambda_t * (f - u_cur)
        numer  = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term + fidelity
        u_next = numer / (1.0 + gam * tau)
        return u_next.clamp(0.0, 255.0)
