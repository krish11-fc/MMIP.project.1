import numpy as np
import torch
from scipy import stats as scipy_stats
from skimage.metrics import structural_similarity as sk_ssim


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 255.0) -> float:
    with torch.no_grad():
        mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 255.0) -> float:
    p = pred.detach().cpu().float().numpy()
    t = target.detach().cpu().float().numpy()
    if p.ndim == 4:
        scores = [sk_ssim(p[i, 0], t[i, 0], data_range=max_val) for i in range(p.shape[0])]
        return float(np.mean(scores))
    return float(sk_ssim(p.squeeze(), t.squeeze(), data_range=max_val))


def speckle_index(img: torch.Tensor) -> float:
    arr = img.detach().cpu().float().numpy().flatten()
    m = arr.mean()
    if m < 1e-8:
        return float("nan")
    return float(arr.std() / m)


def paired_ttest(
    psnr_method1: list,
    psnr_method2: list,
) -> dict:
    arr1 = np.array(psnr_method1)
    arr2 = np.array(psnr_method2)
    diff = arr1 - arr2
    t_stat, p_val = scipy_stats.ttest_rel(arr1, arr2)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    sig = "significant" if p_val < 0.05 else "not significant"
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "verdict": f"{sig} (p={p_val:.4f})",
    }


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    L: int = 1,
    max_val: float = 255.0,
) -> dict:
    model.eval()
    psnr_list, ssim_list, si_list = [], [], []
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        u_pred, _ = model(f, L=L)
        u_pred = u_pred.clamp(0.0, max_val)
        psnr_list.append(psnr(u_pred, u_gt, max_val))
        ssim_list.append(ssim(u_pred, u_gt, max_val))
        si_list.append(speckle_index(u_pred))
    return {
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list, ddof=1)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list, ddof=1)),
        "si_mean": float(np.mean(si_list)),
        "psnr_per_image": psnr_list,
        "ssim_per_image": ssim_list,
    }
