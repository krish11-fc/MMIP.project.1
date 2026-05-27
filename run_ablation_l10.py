import torch, os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHECKPOINT_DIR, DEVICE, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, ABLATION_DIR
from models.network import InertialTNRDNetwork
from utils.filters import build_dct_filters
from dataset import make_test_loader
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 10
device = DEVICE
sd = torch.load(os.path.join(CHECKPOINT_DIR, f"model_L{L}_final.pth"), map_location="cpu")
n = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
print(f"Detected {n} stages in L={L} checkpoint")

filt = build_dct_filters(NUM_FILTERS, FILTER_SIZE, device)
model = InertialTNRDNetwork(num_stages=n, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
    gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
    K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device).to(device)
model.load_state_dict(sd, strict=False)
model.eval()

loader = make_test_loader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Set12"), L=L, seed=0)
psnrs, ssims = [], []
with torch.no_grad():
    for u_gt, f in loader:
        u_gt, f = u_gt.to(device), f.to(device)
        u_pred, _ = model(f, active_stages=n)
        u_pred = u_pred.clamp(0, 255)
        mse = ((u_pred - u_gt) ** 2).mean().item()
        psnrs.append(10 * np.log10(255**2 / max(mse, 1e-10)))
        ssims.append(ssim(u_pred.cpu().squeeze().numpy(), u_gt.cpu().squeeze().numpy(), data_range=255))

print(f"L={L}: Mean PSNR={np.mean(psnrs):.2f}, SSIM={np.mean(ssims):.4f}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ablation")
os.makedirs(out_dir, exist_ok=True)
results = {"L": L, "stages": n, "psnr_mean": float(np.mean(psnrs)), "ssim_mean": float(np.mean(ssims)),
           "psnr_per_image": [float(p) for p in psnrs], "ssim_per_image": [float(s) for s in ssims]}
with open(os.path.join(out_dir, f"ablation_L10_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_dir}/ablation_L10_summary.json")
