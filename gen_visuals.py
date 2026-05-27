import torch, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from config import DEVICE, CHECKPOINT_DIR, CLEAN_TEST_DIR, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS
from models.network import InertialTNRDNetwork
from utils.filters import build_dct_filters
from utils.noise import add_gamma_noise

device = DEVICE
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "plots")
os.makedirs(out_dir, exist_ok=True)

for L in [1, 10]:
    filt = build_dct_filters(NUM_FILTERS, FILTER_SIZE, device)
    sd = torch.load(os.path.join(CHECKPOINT_DIR, f"model_L{L}_final.pth"), map_location="cpu")
    n = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    print(f"L={L}: {n} stages")
    model = InertialTNRDNetwork(num_stages=n, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    img = Image.open(os.path.join(CLEAN_TEST_DIR, "01.png")).convert("L")
    clean = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    noisy = add_gamma_noise(clean, L).clip(0, 255)

    # Visual result: noisy vs restored vs clean
    with torch.no_grad():
        out, _ = model(noisy, active_stages=n)
    out = out.clamp(0, 255)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy.cpu().squeeze(), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Noisy L={L}")
    axes[0].axis("off")
    axes[1].imshow(out.cpu().squeeze(), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Restored L={L}")
    axes[1].axis("off")
    axes[2].imshow(clean.cpu().squeeze(), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Clean")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"visual_test_000_L{L}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visual_test_000_L{L}.png")

    # Stage progression
    n_show = min(n, 6)
    fig, axes = plt.subplots(1, n_show + 2, figsize=(4 * (n_show + 2), 4))
    axes[0].imshow(noisy.cpu().squeeze(), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Noisy"); axes[0].axis("off")
    with torch.no_grad():
        u_prv, u_cur = noisy.clone(), noisy.clone()
        for t in range(n):
            u_nxt = model.stages[t](u_cur, u_prv, noisy)
            u_prv, u_cur = u_cur, u_nxt
            step = t + 1
            if step in [1, 2, 3, n//2, n-1, n] or step == n:
                idx = [1, 2, 3, n//2, n-1, n].index(step) + 1 if step in [1, 2, 3, n//2, n-1, n] else min(step, n_show)
                if idx <= n_show:
                    axes[idx].imshow(u_cur.cpu().squeeze().clamp(0,255), cmap="gray", vmin=0, vmax=255)
                    axes[idx].set_title(f"Stage {step}"); axes[idx].axis("off")
    axes[-1].imshow(clean.cpu().squeeze(), cmap="gray", vmin=0, vmax=255)
    axes[-1].set_title("Clean"); axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"stage_progression_L{L}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved stage_progression_L{L}.png")

print("ALL DONE")
