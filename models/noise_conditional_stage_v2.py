import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, EPSILON,
    EMBED_DIM, NUM_NOISE_LEVELS, NUM_FILTERS, FILTER_SIZE,
)
from utils.rbf import RBFInfluenceFunction
from models.stage import _make_gaussian_kernel


class LearnedGrayLevelIndicator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, u: torch.Tensor, L_embed: torch.Tensor) -> torch.Tensor:
        B, C, H, W = u.shape
        u_flat = u.reshape(B, -1).mean(dim=1, keepdim=True)
        L_expand = L_embed.expand(B, -1) if L_embed.dim() > 1 and L_embed.shape[0] != B else L_embed
        inp = torch.cat([u_flat, L_expand], dim=-1)
        g_val = self.net(inp)
        return g_val.view(B, 1, 1, 1).expand_as(u)


class NoiseConditionalDiffusionStageV2(nn.Module):

    def __init__(
        self,
        num_filters:   int   = NUM_FILTERS,
        filter_size:   int   = FILTER_SIZE,
        num_centers:   int   = 63,
        gamma_inertia: float = GAMMA_INERTIA,
        tau:           float = 0.2,
        nu:            float = NU,
        sigma_smooth:  float = SIGMA_SMOOTH,
        K_thresh:      float = K,
        use_g_func:    bool  = True,
        embed_dim:     int   = EMBED_DIM,
        num_noise_levels: int = NUM_NOISE_LEVELS,
    ):
        super().__init__()

        self.Nk = num_filters
        ksz = filter_size

        self.Ki = nn.Parameter(torch.randn(num_filters, 1, ksz, ksz) * 0.01)

        self.register_buffer("blur_kernel",
                             _make_gaussian_kernel(sigma_smooth, ksz))

        self.gamma_inertia = gamma_inertia
        self.tau           = tau
        self.nu            = nu
        self.sigma_smooth  = sigma_smooth
        self.K_thresh      = K_thresh
        self.use_g_func    = use_g_func

        self.phi = RBFInfluenceFunction(
            num_filters=self.Nk,
            num_centers=num_centers,
        )

        self.log_lambda_base = nn.Parameter(torch.tensor(0.0))

        self.noise_embedding = nn.Embedding(num_noise_levels, embed_dim)
        self.embed_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2 * self.Nk),
        )
        self.lambda_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.pde_param_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5),
        )

        self.learned_g = LearnedGrayLevelIndicator(embed_dim) if use_g_func else None

        self._reset_embedding()

    def _reset_embedding(self):
        for m in self.embed_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        for m in self.lambda_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        for m in self.pde_param_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def lambda_t(self, L_embed: torch.Tensor) -> torch.Tensor:
        offset = self.lambda_mlp(L_embed).squeeze(-1)
        return F.softplus(self.log_lambda_base + offset)

    def _get_pde_params(self, L_embed: torch.Tensor) -> dict:
        raw = self.pde_param_mlp(L_embed)
        gamma_off = raw[..., 0]
        tau_off   = raw[..., 1]
        nu_off    = raw[..., 2]
        K_off     = raw[..., 3]
        sigma_off = raw[..., 4]
        return {
            "gamma": F.softplus(self.gamma_inertia + gamma_off),
            "tau": F.softplus(self.tau + tau_off),
            "nu": F.softplus(self.nu + nu_off),
            "K": F.softplus(self.K_thresh + K_off),
            "sigma": F.softplus(self.sigma_smooth + sigma_off),
        }

    def l1_norm_filter_bank(self) -> torch.Tensor:
        return self.Ki.norm(p=1)

    def _divergence_term(
        self,
        u:     torch.Tensor,
        g:     torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki    = self.Ki[i:i+1]
            r_i   = F.conv2d(u, ki, padding=pad)
            s     = scale[..., i] if scale.dim() > 0 else scale
            h     = shift[..., i] if shift.dim() > 0 else shift
            phi_r = self.phi.forward_modulated(r_i, filter_idx=i, scale=s, shift=h)
            flux  = phi_r * g
            ki_T  = ki.flip(-1).flip(-2)
            out   = out + F.conv2d(flux, ki_T, padding=pad)
        return out

    def forward(
        self,
        u_cur: torch.Tensor,
        u_prv: torch.Tensor,
        f:     torch.Tensor,
        L:     int,
    ) -> torch.Tensor:

        if isinstance(L, (int, float)):
            L_tensor = torch.tensor([L], dtype=torch.long, device=u_cur.device)
        elif isinstance(L, torch.Tensor):
            L_tensor = L.long()
        else:
            L_tensor = torch.tensor(L, dtype=torch.long, device=u_cur.device)

        embed = self.noise_embedding(L_tensor)
        film_params = self.embed_mlp(embed)

        if film_params.dim() == 2 and film_params.shape[0] == 1 and u_cur.shape[0] != 1:
            film_params = film_params.squeeze(0)

        scale = film_params[..., :self.Nk]
        shift = film_params[..., self.Nk:]

        lam = self.lambda_t(embed)
        if lam.dim() > 0:
            if lam.shape[0] == 1 and u_cur.shape[0] != 1:
                lam = lam.squeeze(0)
            else:
                lam = lam.reshape(-1, 1, 1, 1)

        pde_params = self._get_pde_params(embed)
        tau = pde_params["tau"]
        gam = pde_params["gamma"]
        if tau.dim() > 0:
            tau = tau.reshape(-1, 1, 1, 1)
            gam = gam.reshape(-1, 1, 1, 1)

        if self.use_g_func and self.learned_g is not None:
            g = self.learned_g(u_cur, embed)
        else:
            g = torch.ones_like(u_cur)

        div_term  = self._divergence_term(u_cur, g, scale, shift)
        fidelity  = lam * (f - u_cur)
        numer     = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term + fidelity
        u_next    = numer / (1.0 + gam * tau)
        return u_next.clamp(0.0, 255.0)
