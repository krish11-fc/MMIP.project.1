import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GAMMA_INERTIA, K, EPSILON
from utils.rbf import RBFInfluenceFunction

class TNRDLogDiffusionStage(nn.Module):
    """
    TNRD-log stage: operates on log-transformed images.
    No gray-level indicator g(u_σ) — pure learned RBF diffusion in log domain.
    This is the standard TNRD (Chen & Pock 2016) adapted for log-speckle.
    """
    def __init__(
        self,
        filter_bank:   torch.Tensor,
        num_centers:   int   = 63,
        tau:           float = 0.2,
    ):
        super().__init__()
        self.register_buffer("Ki", filter_bank)
        self.Nk = filter_bank.shape[0]
        self.tau = tau

        # No gamma — this is first-order diffusion (no inertia)
        # λ^t : fidelity weight
        self.phi = RBFInfluenceFunction(
            num_filters=self.Nk,
            num_centers=num_centers,
        )
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_t(self):
        return F.softplus(self.log_lambda)

    def _divergence_term(self, u):
        pad = self.Ki.shape[-1] // 2
        out = torch.zeros_like(u)
        for i in range(self.Nk):
            ki    = self.Ki[i:i+1]
            r_i   = F.conv2d(u, ki, padding=pad)
            phi_r = self.phi(r_i, filter_idx=i)
            ki_T  = ki.flip(-1).flip(-2)
            out   = out + F.conv2d(phi_r, ki_T, padding=pad)
        return out

    def forward(self, u_cur, f=None):
        div_term = self._divergence_term(u_cur)
        fidelity = 0.0
        if f is not None:
            fidelity = self.lambda_t * (f - u_cur)
        u_next = u_cur + self.tau ** 2 * div_term + fidelity
        return u_next
