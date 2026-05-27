import torch, os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run_pde_baseline
from config import GAMMA_INERTIA, TAU, NU, K, SIGMA_SMOOTH
from dataset import make_test_loader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

results = {}
for L in [1, 10]:
    loader = make_test_loader(os.path.join("data", "Set12"), L=L, seed=0)
    psnrs, ssims = [], []
    for u_gt, f in loader:
        u_pred = run_pde_baseline(f, gamma=GAMMA_INERTIA, tau=TAU, nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH, max_iter=500)
        u_pred = u_pred.clamp(0, 255)
        arr_pred = u_pred.cpu().squeeze().numpy()
        arr_gt = u_gt.cpu().squeeze().numpy()
        psnrs.append(psnr(arr_gt, arr_pred, data_range=255))
        ssims.append(ssim(arr_gt, arr_pred, data_range=255))
    results[L] = {"psnr_mean": float(np.mean(psnrs)), "ssim_mean": float(np.mean(ssims)),
                  "psnr_per_image": [float(p) for p in psnrs], "ssim_per_image": [float(s) for s in ssims]}
    print(f"L={L}: Mean PSNR={np.mean(psnrs):.2f}, SSIM={np.mean(ssims):.4f}")

with open("outputs/comparison/pde_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to outputs/comparison/pde_baseline.json")
