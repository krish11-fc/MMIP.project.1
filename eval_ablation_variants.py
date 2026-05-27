import torch, os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHECKPOINT_DIR, DEVICE, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, TAU
from models.network import InertialTNRDNetwork
from evaluate import run_pde_baseline
from dataset import make_test_loader
from skimage.metrics import peak_signal_noise_ratio as psnr

device = DEVICE
test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Set12")
results = {}

for L in [1, 10]:
    print(f"\n{'='*50}\nL={L}\n{'='*50}")
    ckpt = f"model_L{L}_final.pth"
    sd = torch.load(os.path.join(CHECKPOINT_DIR, ckpt), map_location="cpu")
    n = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    print(f"  Model has {n} stages")

    def make_model(use_g, load_phi=True):
        m = InertialTNRDNetwork(num_stages=n, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
            gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
            K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=use_g, device=device).to(device)
        msd = m.state_dict()
        for k, v in sd.items():
            if k in msd:
                if load_phi or ("phi" not in k):
                    msd[k].copy_(v.to(device))
        m.load_state_dict(msd, strict=False)
        m.eval()
        return m

    def eval_model(m, n_stg):
        loader = make_test_loader(test_dir, L=L, seed=0)
        psnrs = []
        with torch.no_grad():
            for u_gt, f in loader:
                u_gt, f = u_gt.to(device), f.to(device)
                out, _ = m(f, active_stages=n_stg)
                out = out.clamp(0, 255).cpu().squeeze().numpy()
                psnrs.append(psnr(u_gt.cpu().squeeze().numpy(), out, data_range=255))
        return float(np.mean(psnrs))

    # A1
    m1 = make_model(use_g=True, load_phi=True)
    p1 = eval_model(m1, n)
    print(f"  A1_Full: {p1:.2f} dB")
    results.setdefault(L, {})["A1_Full"] = {"psnr": p1}

    # A2
    m2 = make_model(use_g=False, load_phi=True)
    p2 = eval_model(m2, n)
    print(f"  A2_NoG: {p2:.2f} dB")
    results[L]["A2_NoG"] = {"psnr": p2}

    # A3
    m3 = make_model(use_g=True, load_phi=False)
    p3 = eval_model(m3, n)
    print(f"  A3_FixedRBF: {p3:.2f} dB")
    results[L]["A3_FixedRBF"] = {"psnr": p3}

    # A5: PDE baseline
    loader = make_test_loader(test_dir, L=L, seed=0)
    psnrs = []
    for u_gt, f in loader:
        out = run_pde_baseline(f.to(device), gamma=GAMMA_INERTIA, tau=TAU, nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH, max_iter=500)
        out = out.clamp(0, 255).cpu().squeeze().numpy()
        psnrs.append(psnr(u_gt.squeeze().numpy(), out, data_range=255))
    p5 = float(np.mean(psnrs))
    print(f"  A5_PDE: {p5:.2f} dB")
    results[L]["A5_PDE"] = {"psnr": p5}

    with open("outputs/ablation/ablation_variants.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  > Saved intermediate results")

print("\nDONE")
with open("outputs/ablation/ablation_variants.json", "w") as f:
    json.dump(results, f, indent=2)
