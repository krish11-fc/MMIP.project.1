import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = 1e-6


def _sobel_kernels(device):
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    sx = sx.view(1, 1, 3, 3)
    sy = sy.view(1, 1, 3, 3)
    return sx, sy


def gradient_magnitude(x):
    """Compute gradient magnitude |∇x| using Sobel operators.
    
    Args:
        x: (B, 1, H, W) tensor
    
    Returns:
        (B, 1, H, W) gradient magnitude
    """
    sx, sy = _sobel_kernels(x.device)
    pad = 1
    gx = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), sx)
    gy = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), sy)
    return torch.sqrt(gx ** 2 + gy ** 2 + EPS)


def edge_preserving_loss(pred, target, weight=1.0):
    """Gradient-magnitude edge loss: L1 on |∇pred| - |∇target|.
    
    From MONet / KL-DNN papers: encourages network to preserve
    edges by penalizing differences in gradient magnitude.
    
    Args:
        pred: (B, 1, H, W) predicted image
        target: (B, 1, H, W) ground truth
        weight: scalar multiplier for this loss term
    
    Returns:
        scalar loss tensor
    """
    grad_pred = gradient_magnitude(pred)
    grad_target = gradient_magnitude(target)
    return weight * F.l1_loss(grad_pred, grad_target)


def combined_loss(pred, target, L_batch=None, edge_weight=0.1, filter_reg=1e-4, model=None):
    """Combined MSE + edge-preserving loss with optional noise weighting.
    
    Args:
        pred: (B, 1, H, W) prediction
        target: (B, 1, H, W) ground truth
        L_batch: (B,) noise levels for weighting (None = uniform)
        edge_weight: weight for gradient-magnitude loss term
        filter_reg: L1 regularization on filter banks
        model: model with l1_norm_filter_bank() methods (optional)
    
    Returns:
        scalar loss
    """
    mse = (pred - target) ** 2
    
    if L_batch is not None:
        inv = 1.0 / L_batch.float()
        w = inv / inv.mean()
        mse = (mse * w.view(-1, 1, 1, 1)).mean()
    else:
        mse = mse.mean()
    
    edge = edge_preserving_loss(pred, target, weight=edge_weight)
    
    reg = 0.0
    if model is not None:
        reg = sum(s.l1_norm_filter_bank() for s in model.modules()
                  if hasattr(s, 'l1_norm_filter_bank'))
        reg = filter_reg * reg
    
    return mse + edge + reg
