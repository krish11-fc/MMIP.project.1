"""
generate_all_figures.py — Comprehensive model visualization for paper.
Generates all per-model images, comparison grids, error maps, and charts.
Runs in parallel while v3 trains; v3 can be added later.
"""
import os, sys, glob, json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, CLEAN_TEST_DIR, CLEAN_VAL_DIR, NUM_FILTERS, FILTER_SIZE
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, TAU, EMBED_DIM, NUM_NOISE_LEVELS, NUM_STAGES
from models.network import InertialTNRDNetwork
from models.full_learn_network import FullLearnInertialTNRDNetwork
from models.tnrd_log_network import TNRDLogNetwork
from models.noise_conditional_network import NoiseConditionalTNRDNetwork
from models.noise_conditional_network_v2 import NoiseConditionalTNRDNetworkV2
from utils.noise import add_gamma_noise
from evaluate import run_pde_baseline
from utils.metrics import psnr, ssim
from scipy.ndimage import sobel as _sobel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "all_figures")
for sub in ["comparison", "restored", "error_maps", "charts", "stage_progress", "edge_maps"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
device = DEVICE
print(f"Device: {device}")

# ── Test images ──────────────────────────────────────────────────────────
test_paths = sorted(glob.glob(os.path.join(CLEAN_TEST_DIR, "*.png")))[:12]
names = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
print(f"Test images: {len(test_paths)}")

# ── Model loading ────────────────────────────────────────────────────────
MODELS = {}

def _load_state(path, model):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# PDE baseline (callable, no ckpt)
MODELS["PDE"] = ("pde", lambda f, L: run_pde_baseline(f, gamma=GAMMA_INERTIA, tau=TAU, nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH))

# Learned TNRD
def load_tnrd(L_suffix):
    base = "model_L{}_final.pth"
    path = os.path.join(CHECKPOINT_DIR, base.format(L_suffix))
    if not os.path.exists(path): return None
    sd = torch.load(path, map_location="cpu")
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    m = InertialTNRDNetwork(ns, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, device).to(device)
    return _load_state(path, m)

# NCTDN v1 (mixed)
def load_nctdn_v1():
    path = os.path.join(CHECKPOINT_DIR, "nctdn_model_mixed_final.pth")
    if not os.path.exists(path): return None
    sd = torch.load(path, map_location="cpu")
    ns = max(int(k.split('.')[1]) for k in sd if k.startswith('stages.')) + 1
    m = NoiseConditionalTNRDNetwork(ns, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, EMBED_DIM, NUM_NOISE_LEVELS, device).to(device)
    _load_state(path, m)
    return m

# NCTDN v2 (mixed)
def load_nctdn_v2():
    path = os.path.join(CHECKPOINT_DIR, "nctdn_v2_model_mixed_final.pth")
    if not os.path.exists(path): return None
    m = NoiseConditionalTNRDNetworkV2(10, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, EMBED_DIM, NUM_NOISE_LEVELS, device).to(device)
    _load_state(path, m)
    return m

# NCTDN v3 (mixed) — placeholder if exists
def load_nctdn_v3():
    path = os.path.join(CHECKPOINT_DIR, "nctdn_v3_model_mixed_final.pth")
    if not os.path.exists(path): return None
    from models.multi_scale_wrapper import MultiScaleTNRDWrapper
    base = NoiseConditionalTNRDNetworkV2(10, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, EMBED_DIM, NUM_NOISE_LEVELS, device)
    wrapper = MultiScaleTNRDWrapper(base).to(device)
    _load_state(path, wrapper)
    wrapper.eval()
    return wrapper

# Full-Learn (needs L-specific model)
def load_full_learn(L_suffix):
    path = os.path.join(CHECKPOINT_DIR, "model_L{}_fulllearn_final.pth".format(L_suffix))
    if not os.path.exists(path): return None
    sd = torch.load(path, map_location="cpu")
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    m = FullLearnInertialTNRDNetwork(ns, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, TAU, NU, K, SIGMA_SMOOTH, RBF_NUM_CENTERS, True, device).to(device)
    return _load_state(path, m)

# TNRD-Log (needs L-specific model)
def load_tnrd_log(L_suffix):
    path = os.path.join(CHECKPOINT_DIR, "model_L{}_tnrdlog_final.pth".format(L_suffix))
    if not os.path.exists(path): return None
    sd = torch.load(path, map_location="cpu")
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    m = TNRDLogNetwork(ns, NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS, device).to(device)
    return _load_state(path, m)

def load_models_for_L(L_val):
    L_s = str(L_val)
    models = {}
    models["PDE"] = ("pde", lambda f, L=L_val: run_pde_baseline(f, gamma=GAMMA_INERTIA, tau=TAU, nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH))
    m_tnrd = load_tnrd(L_s)
    if m_tnrd: models["TNRD"] = ("nn", lambda f, m=m_tnrd: m(f)[0])
    m_v1 = load_nctdn_v1()
    if m_v1: models["NCTDNv1"] = ("nn", lambda f, m=m_v1, l=L_val: m(f, L=l)[0])
    m_v2 = load_nctdn_v2()
    if m_v2: models["NCTDNv2"] = ("nn", lambda f, m=m_v2, l=L_val: m(f, L=l)[0])
    m_v3 = load_nctdn_v3()
    if m_v3: models["NCTDNv3"] = ("nn", lambda f, m=m_v3, l=L_val: m(f, L=l)[0])
    m_fl = load_full_learn(L_s)
    if m_fl: models["Full-Learn"] = ("nn", lambda f, m=m_fl: m(f)[0])
    m_tl = load_tnrd_log(L_s)
    if m_tl: models["TNRD-Log"] = ("nn", lambda f, m=m_tl: m(f)[0])
    return models

def calc_psnr(im_pred, im_gt):
    return 10 * np.log10(255**2 / np.mean((im_pred - im_gt)**2 + 1e-10))

def calc_ssim(im_pred, im_gt):
    return ssim(torch.from_numpy(im_pred).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(im_gt).unsqueeze(0).unsqueeze(0))

MODEL_COLORS = {
    "PDE": "gray", "TNRD": "steelblue", "NCTDNv1": "darkorange",
    "NCTDNv2": "forestgreen", "NCTDNv3": "crimson",
    "Full-Learn": "purple", "TNRD-Log": "brown",
}
MODEL_LABELS = {
    "PDE": "PDE", "TNRD": "Learned TNRD", "NCTDNv1": "NCTDN v1",
    "NCTDNv2": "NCTDN v2", "NCTDNv3": "NCTDN v3",
    "Full-Learn": "Full-Learn", "TNRD-Log": "TNRD-Log",
}

# ── Per-noise-level processing ───────────────────────────────────────────
all_set12_metrics = {}

for L in [1, 10]:
    print(f"\n{'='*60}")
    print(f"  L={L}")
    print(f"{'='*60}")
    
    models = load_models_for_L(L)
    model_keys = [k for k in ["PDE", "TNRD", "NCTDNv1", "NCTDNv2", "NCTDNv3", "Full-Learn", "TNRD-Log"] if k in models]
    print(f"  Models loaded: {model_keys}")
    
    metrics = {k: {"psnr": [], "ssim": []} for k in model_keys}
    
    for idx in range(len(test_paths)):
        img_path = test_paths[idx]
        name = names[idx]
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        noisy_np = noisy.cpu().squeeze().numpy()
        
        outputs = {"Noisy": noisy_np}
        for mk in model_keys:
            kls, fn = models[mk]
            with torch.no_grad():
                out = fn(noisy.to(device))
                if isinstance(out, tuple):
                    out = out[0]
                out_np = out.clamp(0, 255).cpu().squeeze().numpy()
            outputs[mk] = out_np
            p = calc_psnr(out_np, img)
            s = calc_ssim(out_np, img)
            metrics[mk]["psnr"].append(p)
            metrics[mk]["ssim"].append(s)
        
        # ── 1. Multi-model side-by-side comparison ───────────────────────
        n_panels = len(model_keys) + 2  # noisy + models + GT
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
        panels = [("Noisy", outputs["Noisy"], "gray")]
        for mk in model_keys:
            panels.append((MODEL_LABELS.get(mk, mk), outputs[mk], MODEL_COLORS.get(mk, "gray")))
        panels.append(("Ground Truth", img, "white"))
        for ax, (label, im, _) in zip(axes, panels):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            if label != "Ground Truth":
                pv = calc_psnr(im, img)
                ax.set_title(f"{label}\n{pv:.2f} dB", fontsize=9, fontweight="bold")
            else:
                ax.set_title("Ground Truth", fontsize=9, fontweight="bold")
            ax.axis("off")
        fig.suptitle(f"{name} — L={L}", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fpath = os.path.join(OUT, "comparison", f"{name}_L{L}_comparison.png")
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        # ── 2. Per-model restored image ──────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, im, t in zip(axes, [img, noisy_np, outputs[model_keys[-1]]],
                             ["Ground Truth", f"Noisy L={L}", f"{MODEL_LABELS.get(model_keys[-1], model_keys[-1])}"]):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(t, fontsize=10)
            ax.axis("off")
        ps_val = metrics[model_keys[-1]]["psnr"][-1]
        fig.suptitle(f"{name} — L={L} — {ps_val:.2f} dB", fontsize=12)
        plt.tight_layout()
        fpath = os.path.join(OUT, "restored", f"{name}_L{L}_restored.png")
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        # ── 3. Error maps (residual |pred - GT|) ─────────────────────────
        n_err = len(model_keys)
        fig, axes = plt.subplots(1, n_err + 1, figsize=(4 * (n_err + 1), 4.5))
        axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Ground Truth", fontsize=9)
        axes[0].axis("off")
        for j, mk in enumerate(model_keys):
            err = np.abs(outputs[mk] - img)
            im = axes[j + 1].imshow(err, cmap="hot", vmin=0, vmax=40)
            axes[j + 1].set_title(f"{MODEL_LABELS.get(mk, mk)}\nmax={err.max():.1f}", fontsize=8)
            axes[j + 1].axis("off")
        fig.suptitle(f"{name} — L={L} Error Maps", fontsize=11)
        plt.tight_layout()
        fig.subplots_adjust(right=0.92)
        cax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        plt.colorbar(im, cax=cax)
        fpath = os.path.join(OUT, "error_maps", f"{name}_L{L}_errors.png")
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        # ── 4. Edge-preservation (Sobel magnitude) — first 3 images only ──
        if idx < 3:
            def grad_mag(im):
                return np.sqrt(_sobel(im, axis=0)**2 + _sobel(im, axis=1)**2)
            
            fig, axes = plt.subplots(1, n_err + 1, figsize=(4 * (n_err + 1), 4.5))
            gt_grad = grad_mag(img)
            axes[0].imshow(gt_grad, cmap="viridis")
            axes[0].set_title("GT Edges", fontsize=9)
            axes[0].axis("off")
            for j, mk in enumerate(model_keys):
                pred_grad = grad_mag(outputs[mk])
                axes[j + 1].imshow(pred_grad, cmap="viridis", vmin=0, vmax=gt_grad.max())
                axes[j + 1].set_title(f"{MODEL_LABELS.get(mk, mk)}", fontsize=8)
                axes[j + 1].axis("off")
            fig.suptitle(f"{name} — L={L} Edge Maps (Sobel)", fontsize=11)
            plt.tight_layout()
            fpath = os.path.join(OUT, "edge_maps", f"{name}_L{L}_edges.png")
            fig.savefig(fpath, dpi=200, bbox_inches="tight")
            plt.close(fig)
        
        print(f"  [{idx+1}/{len(test_paths)}] {name}")
        torch.cuda.empty_cache()
    
    # ── 5. 4×3 grid of best model's restored outputs ─────────────────────
    best_model = model_keys[-1]  # last loaded = most advanced
    fig, axes = plt.subplots(4, 3, figsize=(10, 13))
    for idx in range(12):
        img_path = test_paths[idx]
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        kls, fn = models[best_model]
        with torch.no_grad():
            out = fn(noisy.to(device))
            if isinstance(out, tuple): out = out[0]
            out_np = out.clamp(0, 255).cpu().squeeze().numpy()
        p = calc_psnr(out_np, img)
        ax = axes[idx // 3, idx % 3]
        ax.imshow(out_np, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{names[idx]} — {p:.1f} dB", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{MODEL_LABELS.get(best_model, best_model)} — L={L}", fontsize=14, y=1.01)
    plt.tight_layout()
    fpath = os.path.join(OUT, "restored", f"all_restored_L{L}.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 4x3 grid for {best_model}")
    
    all_set12_metrics[L] = metrics

# ── 6. PSNR bar chart ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for li, L in enumerate([1, 10]):
    ax = axes[li]
    metrics = all_set12_metrics[L]
    model_keys = [k for k in ["PDE", "TNRD", "NCTDNv1", "NCTDNv2", "NCTDNv3", "Full-Learn", "TNRD-Log"] if k in metrics]
    means = [np.mean(metrics[k]["psnr"]) for k in model_keys]
    stds = [np.std(metrics[k]["psnr"]) for k in model_keys]
    colors = [MODEL_COLORS.get(k, "gray") for k in model_keys]
    labels = [MODEL_LABELS.get(k, k) for k in model_keys]
    bars = ax.bar(range(len(model_keys)), means, color=colors, alpha=0.85, yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Set12 — L={L}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Model Comparison — PSNR on Set12", fontsize=14, fontweight="bold")
plt.tight_layout()
fpath = os.path.join(OUT, "charts", "psnr_bar_comparison.png")
fig.savefig(fpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved PSNR bar chart")

# ── 7. PSNR vs noise level curves (all models, all L=1,3,5,10,33) ────────
print(f"\n  Generating PSNR vs noise level curves (requires eval at each L)...")
all_l_metrics = {}
for L in [1, 3, 5, 10, 33]:
    print(f"  L={L}...")
    models = load_models_for_L(L)
    model_keys = [k for k in ["PDE", "TNRD", "NCTDNv1", "NCTDNv2", "NCTDNv3", "Full-Learn", "TNRD-Log"] if k in models]
    all_l_metrics[L] = {}
    for mk in model_keys:
        ps = []
        for idx in range(min(6, len(test_paths))):  # first 6 images
            img_path = test_paths[idx]
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
            clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
            kls, fn = models[mk]
            with torch.no_grad():
                out = fn(noisy.to(device))
                if isinstance(out, tuple): out = out[0]
                out_np = out.clamp(0, 255).cpu().squeeze().numpy()
            ps.append(calc_psnr(out_np, img))
        all_l_metrics[L][mk] = np.mean(ps)
    torch.cuda.empty_cache()

fig, ax = plt.subplots(figsize=(9, 6))
ls = sorted(all_l_metrics.keys())
for mk in ["PDE", "TNRD", "NCTDNv1", "NCTDNv2", "NCTDNv3", "Full-Learn", "TNRD-Log"]:
    vals = [all_l_metrics[L].get(mk, None) for L in ls]
    if any(v is not None for v in vals):
        vals_f = [v if v is not None else np.nan for v in vals]
        ax.plot(ls, vals_f, marker="o", linewidth=2, label=MODEL_LABELS.get(mk, mk),
                color=MODEL_COLORS.get(mk, "gray"))
ax.set_xlabel("Noise Level (L)", fontsize=11)
ax.set_ylabel("PSNR (dB)", fontsize=11)
ax.set_title("PSNR vs Noise Level (Set12, first 6 images avg)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(ls)
plt.tight_layout()
fpath = os.path.join(OUT, "charts", "psnr_vs_noise.png")
fig.savefig(fpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved PSNR vs noise curves")

# ── 8. Params vs PSNR scatter (placeholder — add v3 when ready) ──────────
param_counts = {
    "PDE": 0, "TNRD": 816, "NCTDNv1": 16032,
    "NCTDNv2": 101360, "NCTDNv3": 101362,
    "Full-Learn": 500, "TNRD-Log": 400,
}
fig, ax = plt.subplots(figsize=(8, 6))
for mk in ["PDE", "TNRD", "NCTDNv1", "NCTDNv2", "NCTDNv3", "Full-Learn", "TNRD-Log"]:
    if mk not in all_set12_metrics.get(1, {}): continue
    p_mean = np.mean(all_set12_metrics[1][mk]["psnr"])
    n_params = param_counts.get(mk, 0)
    ax.scatter(n_params, p_mean, s=120, c=MODEL_COLORS.get(mk, "gray"),
               label=MODEL_LABELS.get(mk, mk), zorder=5)
    ax.annotate(MODEL_LABELS.get(mk, mk).split()[0], (n_params, p_mean),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.set_xlabel("Number of Parameters", fontsize=11)
ax.set_ylabel("PSNR at L=1 (dB)", fontsize=11)
ax.set_title("Model Size vs Performance (Set12 L=1)", fontsize=13)
ax.set_xscale("symlog", linthresh=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fpath = os.path.join(OUT, "charts", "params_vs_psnr.png")
fig.savefig(fpath, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved params vs PSNR scatter")

# ── 9. Save metrics JSON ────────────────────────────────────────────────
json_out = {str(L): {
    mk: {"psnr_mean": float(np.mean(metrics[mk]["psnr"])),
         "psnr_std": float(np.std(metrics[mk]["psnr"])),
         "ssim_mean": float(np.mean(metrics[mk]["ssim"])),
         "ssim_std": float(np.std(metrics[mk]["ssim"]))}
    for mk in metrics}
    for L, metrics in all_set12_metrics.items()}
with open(os.path.join(OUT, "all_set12_metrics.json"), "w") as f:
    json.dump(json_out, f, indent=2)

print(f"\n{'='*60}")
print(f"  ALL FIGURES GENERATED IN: {OUT}")
print(f"{'='*60}")
