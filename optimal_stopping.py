"""
optimal_stopping.py — Optimal Stopping Theory for Inertial TNRD.

Problem:
  The inertial telegraph PDE generates u^1..u^T from noisy f.
  PSNR(u^t, u_gt) peaks at T* then degrades (T*=5 for L=1, T*=4 for L=10).
  Without u_gt, find a stopping rule that halts near T*.

Noise model: speckle (multiplicative gamma)  f = u · eta, eta ~ Gamma(L, 1/L)
  → noise variance is signal-dependent: Var[f|u] = u^2 / L
  → standard additive-noise criteria (Morozov) do NOT apply directly

General stopping theory for iterative diffusion:
  Let Phi^t be the nonlinear operator u^t = Phi^t(f).
  The process transitions through:
    Phase 1 (t < T*): denoising — diffusion removes speckle
    Phase 2 (t = T*): optimal — estimate closest to clean
    Phase 3 (t > T*): degradation — inertial term oscillates, HF energy grows

Three model-based stopping criteria (no noise-model assumptions):
  1. Min Update Norm   — |u^{t+1} - u^t| minimized at convergence
  2. Min Diffusion     — ||div_term|| minimized when diffusion saturates
  3. Min HF Energy     — Laplacian energy minimized before instability onset

Von Neumann stability:
  Linearise: phi_i(s) ~ psi·s.  Diffusion operator L[u] = -psi·(sum K_i^T K_i)*u
  Fourier symbol: L_hat(k) = -psi · sum |K_hat_i(k)|^2  (<= 0 for diffusive)
  
  Inertial scheme amplification eigenvalue:
    lambda(k) = [beta(k) +/- sqrt(beta(k)^2 - 4 delta)] / 2
    beta(k) = (2 + gamma tau + tau^2 L_hat(k)) / (1 + gamma tau)
    delta = 1 / (1 + gamma tau)
  
  |lambda(k)| > 1 when L_hat(k) is too negative (strong diffusion at high k).
  This causes HF amplification beyond T*.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, NUM_STAGES, CHECKPOINT_DIR, CLEAN_TEST_DIR, TAU, GAMMA_INERTIA
from config import NUM_FILTERS, FILTER_SIZE, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS
from models import InertialTNRDNetwork
from dataset import make_test_loader
from utils.metrics import psnr, ssim

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "optimal_stopping")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Model Loading ──────────────────────────────────────────────────────────────

def _find_best_ckpt(ckpt_dir, L):
    import re
    max_stage = 0
    best_path = None
    for fname in os.listdir(ckpt_dir):
        m = re.match(rf"stage(\d+)_L{L}_best\.pth", fname)
        if m:
            s = int(m.group(1))
            if s > max_stage:
                max_stage = s
                best_path = os.path.join(ckpt_dir, fname)
    if best_path and max_stage > 0:
        sd = torch.load(best_path, map_location="cpu")
        stage_nums = sorted(set(int(k.split('.')[1]) for k in sd if k.startswith('stages.')))
        return best_path, len(stage_nums)
    final_path = os.path.join(ckpt_dir, f"model_L{L}_final.pth")
    if os.path.exists(final_path):
        sd = torch.load(final_path, map_location="cpu")
        stage_nums = sorted(set(int(k.split('.')[1]) for k in sd if k.startswith('stages.')))
        return final_path, len(stage_nums)
    return None, NUM_STAGES


def load_model(L, ckpt_dir, device=DEVICE):
    path, actual_stages = _find_best_ckpt(ckpt_dir, L)
    model = InertialTNRDNetwork(
        num_stages=actual_stages, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        print(f"  Loaded: {path}  (stages={actual_stages})")
    else:
        print(f"  WARNING: no checkpoint for L={L}")
    return model


# ── Forward pass collecting all signals ───────────────────────────────────────

@torch.no_grad()
def collect_signals(model, f):
    T = len(model.stages)
    u_prv = f.clone()
    u_cur = f.clone()

    u_list = [f.clone()]
    update_list = []
    div_list = []
    resid_list = []
    hf_list = []

    for t in range(T):
        stage = model.stages[t]
        tau = stage.tau
        gam = stage.gamma_inertia

        from models.stage import _gray_level_indicator
        if stage.use_g_func:
            g = _gray_level_indicator(u_cur, stage.nu, stage.K_thresh,
                                      stage.blur_kernel.to(dtype=u_cur.dtype))
        else:
            g = torch.ones_like(u_cur)
        div_term = stage._divergence_term(u_cur, g)

        div_list.append(div_term.square().mean().item())

        numer = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term
        u_nxt = numer / (1.0 + gam * tau)
        u_nxt = u_nxt.clamp(0.0, 255.0)

        update_list.append((u_nxt - u_cur).square().mean().item())
        resid_list.append((f - u_nxt).square().mean().item())

        with torch.no_grad():
            lap = u_nxt[:, :, 1:-1, 1:-1] * 4 - (
                u_nxt[:, :, 1:-1, :-2] + u_nxt[:, :, 1:-1, 2:] +
                u_nxt[:, :, :-2, 1:-1] + u_nxt[:, :, 2:, 1:-1]
            )
            hf_list.append(lap.square().mean().item())

        u_list.append(u_nxt.clone())
        u_prv = u_cur
        u_cur = u_nxt

    return {"u": u_list, "update": update_list, "div": div_list,
            "resid": resid_list, "hf": hf_list}


# ── Stopping Criteria ──────────────────────────────────────────────────────────

def _argmin_first_local(xs):
    for i in range(1, len(xs) - 1):
        if xs[i] < xs[i - 1] and xs[i] < xs[i + 1]:
            return i + 1
    return int(np.argmin(xs)) + 1


def stop_update_min(update_history):
    return _argmin_first_local(update_history)


def stop_div_min(div_history):
    return _argmin_first_local(div_history)


def stop_hf_min(hf_history):
    return _argmin_first_local(hf_history)


# ── Von Neumann Stability Analysis ──────────────────────────────────────────

@torch.no_grad()
def compute_amplification_factor(model, img_size=64):
    """
    Estimate the spectral radius of the inertial scheme's amplification matrix
    from learned filter parameters.

    We approximate the diffusion operator's symbol from the filter bank:
      L_hat(k) ≈ -sum_i |K_hat_i(k)|^2

    Then compute the maximal eigenvalue of:
      A(k) = [[beta(k), -delta], [1, 0]]
    
    Returns (freqs, spec_radii, stable_mask)
    """
    # Get filters from stage 0 (they're the same across all stages)
    Ki = model.stages[0].Ki.cpu().numpy()  # (Nk, 1, m, m)
    Nk, _, m, m = Ki.shape
    tau = model.stages[0].tau
    gam = model.stages[0].gamma_inertia

    # Compute 2D DFT of each filter
    freqs = np.fft.fftfreq(img_size)
    kx, ky = np.meshgrid(freqs, freqs)
    k_mag = np.sqrt(kx**2 + ky**2)

    # Compute filter symbol
    sum_ki_sq = np.zeros((img_size, img_size), dtype=np.complex128)
    for i in range(Nk):
        K_hat = np.fft.fft2(Ki[i, 0], s=(img_size, img_size))
        sum_ki_sq += K_hat * np.conj(K_hat)

    # L_hat(k) = -sum_i |K_hat_i(k)|^2  (negative since diffusion)
    L_hat = -np.real(sum_ki_sq)

    # Amplification factor eigenvalues
    beta_k = (2.0 + gam * tau + tau**2 * L_hat) / (1.0 + gam * tau)
    delta = 1.0 / (1.0 + gam * tau)

    # λ = [β ± √(β² - 4δ)] / 2
    discriminant = beta_k**2 - 4 * delta
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0))

    lam_plus = (beta_k + sqrt_disc) / 2.0
    lam_minus = (beta_k - sqrt_disc) / 2.0

    # For complex discriminant, |λ| = sqrt(delta)
    # (when β² < 4δ, eigenvalues are complex conjugates with magnitude sqrt(δ))
    complex_mask = discriminant < 0
    spec_rad = np.maximum(np.abs(lam_plus), np.abs(lam_minus))
    spec_rad[complex_mask] = np.sqrt(delta)

    stable = spec_rad <= 1.0

    return k_mag, spec_rad, stable, L_hat


# ── Main Analysis ──────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze(L=1, ckpt_dir=CHECKPOINT_DIR, test_dir=CLEAN_TEST_DIR, device=DEVICE):
    print(f"\n{'='*70}")
    print(f"  OPTIMAL STOPPING THEORY  L={L}")
    print(f"{'='*70}")

    model = load_model(L, ckpt_dir, device)
    model.eval()
    T_max = len(model.stages)
    print(f"  Stages={T_max}")

    # Single pass: evaluate all images
    loader = make_test_loader(test_dir, L=L, seed=0)
    images = []
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        sig = collect_signals(model, f)

        best_t, best_p = 1, -1e9
        for t in range(1, T_max + 1):
            p = float(psnr(sig["u"][t], u_gt).item())
            if p > best_p:
                best_p, best_t = p, t

        images.append({
            "u_gt": u_gt.cpu(),
            "f": f.cpu(),
            "sig": {
                "u": [u_i.cpu() for u_i in sig["u"]],
                "update": sig["update"],
                "div": sig["div"],
                "resid": sig["resid"],
                "hf": sig["hf"],
            },
            "oracle_T": best_t,
            "oracle_psnr": best_p,
        })

    oracle_Ts = np.array([d["oracle_T"] for d in images])
    oracle_psnrs = np.array([d["oracle_psnr"] for d in images])
    print(f"  Images: {len(images)},  Oracle T* range: [{oracle_Ts.min()}, {oracle_Ts.max()}],  "
          f"mean={oracle_Ts.mean():.2f}")

    # ── Stopping methods ────────────────────────────────────────────────────
    methods = {
        "Min Update": lambda s: stop_update_min(s["update"]),
        "Min Diffusion": lambda s: stop_div_min(s["div"]),
        "Min HF Energy": lambda s: stop_hf_min(s["hf"]),
    }

    method_results = {}
    for name, fn in methods.items():
        Ts, ps = [], []
        for d in images:
            t_pred = fn(d["sig"])
            t_pred = max(1, min(t_pred, T_max))
            Ts.append(t_pred)
            p = float(psnr(d["sig"]["u"][t_pred], d["u_gt"]).item())
            ps.append(p)
        method_results[name] = {"Ts": Ts, "psnrs": ps}

    # Fixed-T baselines
    fixed_baselines = {}
    for T in [1, 3, 5, T_max]:
        if T <= T_max:
            ps = [float(psnr(d["sig"]["u"][T], d["u_gt"]).item()) for d in images]
            fixed_baselines[f"Fixed T={T}"] = ps

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\n  {'Method':<20} {'Mean T':>6} {'PSNR':>9} {'Oracle Gap':>10} {'Hit%':>6} {'rho':>5}")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*10} {'-'*6} {'-'*5}")

    for name in methods:
        Ts = method_results[name]["Ts"]
        ps = method_results[name]["psnrs"]
        mt = np.mean(Ts)
        mp = np.mean(ps)
        gap = float(np.mean(oracle_psnrs) - mp)
        hit = np.mean(np.array(Ts) == oracle_Ts) * 100
        unique_Ts = len(set(Ts))
        rho = np.corrcoef(Ts, oracle_Ts)[0, 1] if unique_Ts > 1 else 0.0
        print(f"  {name:<20} {mt:>6.2f} {mp:>9.4f} {gap:>10.4f} {hit:>5.1f}% {rho:>5.3f}")

    for name, psnrs in fixed_baselines.items():
        mp = np.mean(psnrs)
        gap = float(np.mean(oracle_psnrs) - mp)
        print(f"  {name:<20} {'---':>6} {mp:>9.4f} {gap:>10.4f} {'---':>6} {'---':>5}")

    print(f"  {'Oracle (upper bound)':<20} {np.mean(oracle_Ts):>6.2f} "
          f"{np.mean(oracle_psnrs):>9.4f} {'0.0000':>10} {'100.0%':>6} {'1.000':>5}")

    # ── Von Neumann Stability ──────────────────────────────────────────────
    print(f"\n  Von Neumann stability analysis...")
    k_mag, spec_rad, stable, L_hat = compute_amplification_factor(model, img_size=64)
    frac_stable = stable.mean() * 100
    max_spec = spec_rad.max()
    print(f"  Stable freq. fraction: {frac_stable:.1f}%")
    print(f"  Max |lambda|: {max_spec:.4f}  (>1 => unstable)")

    # ── Figures ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Oracle T distribution
    ax = axes[0, 0]
    ax.hist(oracle_Ts, bins=range(1, T_max + 2), align="left", rwidth=0.8,
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Optimal T")
    ax.set_ylabel("Images")
    ax.set_title(f"Oracle T* Dist'n  (mu={np.mean(oracle_Ts):.2f})")
    ax.grid(alpha=0.3)

    # 2. Stopping criteria vs oracle
    ax = axes[0, 1]
    x = np.arange(len(images))
    methods_plot = list(methods.keys())
    w = 0.25
    ax.bar(x - w, oracle_Ts, w, label="Oracle", color="black", alpha=0.8)
    for i, mn in enumerate(methods_plot):
        preds = method_results[mn]["Ts"]
        ax.bar(x + (i - 0.5) * w, preds, w, label=mn, alpha=0.7)
    ax.set_xlabel("Test image")
    ax.set_ylabel("T")
    ax.set_title("Stopping Criteria vs Oracle")
    ax.set_xticks(x)
    ax.set_xticklabels([f"img{idx}" for idx in range(len(images))], fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 3. Stopping signals for first image
    ax = axes[0, 2]
    d0 = images[0]
    stages = np.arange(1, T_max + 1)
    uh = np.array(d0["sig"]["update"])
    uh = uh / uh.max() if uh.max() > 0 else uh
    dh = np.array(d0["sig"]["div"])
    dh = dh / dh.max() if dh.max() > 0 else dh
    hf = np.array(d0["sig"]["hf"])
    hf = hf / hf.max() if hf.max() > 0 else hf

    ax.plot(stages, uh, "s-", label="|Update|", color="firebrick", lw=1.5)
    ax.plot(stages, dh, "^-", label="||div||", color="forestgreen", lw=1.5)
    ax.plot(stages, hf, "d-", label="HF Energy", color="purple", lw=1.5)
    ax.axvline(d0["oracle_T"], color="black", ls="--", lw=2, label=f"Oracle T*={d0['oracle_T']}")
    ax.set_xlabel("Stage T")
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Stopping Signals -- test_000 (L={L})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Von Neumann stability plot
    ax = axes[1, 0]
    k_bins = np.linspace(0, 0.5, 40)
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    spec_by_k = []
    stable_by_k = []
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if mask.sum() > 0:
            spec_by_k.append(spec_rad[mask].mean())
            stable_by_k.append(stable[mask].mean())
    ax.plot(k_centers, spec_by_k, "b-", lw=2, label="Max |lambda|")
    ax.plot(k_centers, stable_by_k, "r--", lw=2, label="Frac stable")
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xlabel("Normalized frequency k")
    ax.set_ylabel("|lambda| / Stable fraction")
    ax.set_title("Von Neumann Stability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5. Diffusion operator symbol
    ax = axes[1, 1]
    k_bin_centers = []
    L_hat_k = []
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if mask.sum() > 0:
            k_bin_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            L_hat_k.append(L_hat[mask].mean())
    ax.plot(k_bin_centers, L_hat_k, "g-", lw=2)
    ax.set_xlabel("Normalized frequency k")
    ax.set_ylabel("L_hat(k)")
    ax.set_title("Diffusion Operator Symbol")
    ax.grid(alpha=0.3)

    # 6. Method comparison bar chart
    ax = axes[1, 2]
    all_methods = list(methods.keys())
    all_psnrs = [np.mean(method_results[m]["psnrs"]) for m in all_methods]
    baseline_names = list(fixed_baselines.keys())
    baseline_psnrs = [np.mean(fixed_baselines[n]) for n in baseline_names]
    all_names = all_methods + baseline_names + ["Oracle"]
    all_vals = all_psnrs + baseline_psnrs + [float(np.mean(oracle_psnrs))]
    colors_bar = (["steelblue"] * len(all_methods) + ["gray"] * len(baseline_names) + ["black"])
    ax.bar(range(len(all_names)), all_vals, color=colors_bar, alpha=0.85)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("Method Comparison")
    ax.axhline(float(np.mean(oracle_psnrs)), color="black", ls="--", lw=1,
               label=f"Oracle {np.mean(oracle_psnrs):.2f} dB")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Optimal Stopping Theory for Inertial TNRD  (L={L})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"optimal_stopping_L{L}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Figure -> {path}")

    # ── Mathematical summary ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  MATHEMATICAL DERIVATION")
    print("="*70)
    print(r"""
  Problem:
  ------------------------------------------------------------------
  Multiplicative speckle noise: f = u . eta, eta ~ Gamma(L, 1/L).
  Inertial TNRD generates sequence u^1..u^T from f.
  Goal: find stopping time tau* = argmin_t ||u^t - u_gt||^2
         WITHOUT observing u_gt.

  PDE Scheme:
  ------------------------------------------------------------------
  (1+gamma tau) u^{n+1} = (2+gamma tau) u^n - u^{n-1} + tau^2 * div(u^n)
  
  div(u^n) = sum_i K_i^T [ phi_i(K_i * u^n) . g(u^n) ]

  The inertial term u_tt (via u^{n-1}) creates momentum:
  - Accelerates diffusion initially (good: faster denoising)
  - Causes overshoot at late stages (bad: PSNR degradation)

  Stopping Criterion 1: Minimum Update Norm
  ------------------------------------------------------------------
  Delta^n = u^{n+1} - u^n = [tau^2 div(u^n) - gamma tau (u^n - u^{n-1})]
                             / (1 + gamma tau)
  
  At optimal T*, diffusion force and damping balance:
    tau^2 div(u^T*) = gamma tau (u^{T*} - u^{T*-1})
    => Delta^T* ~ 0

  Stopping Criterion 2: Minimum Diffusion Force
  ------------------------------------------------------------------
  ||div(u^n)|| = ||sum_i K_i^T [phi_i(K_i * u^n) . g(u^n)]||
  
  The diffusion force is large when edges are present (good, active
  denoising) and small when the image is already smooth.
  After T*, oscillations create spurious edges -> div grows again.

  Stopping Criterion 3: Minimum High-Frequency Energy
  ------------------------------------------------------------------
  HF(u) = ||Laplacian(u)||^2  (second derivative energy)
  
  From the von Neumann stability analysis, when the PDE scheme
  becomes unstable, HIGH-FREQUENCY COMPONENTS GROW first.
  HF energy is the earliest indicator of instability onset.

  Von Neumann Stability:
  ------------------------------------------------------------------
  Linearise: phi_i(s) ~ psi * s (first-order Taylor).
  Then div(u) ~ L u where L is a linear diffusion operator.
  
  In Fourier domain: L_hat(k) = -psi * sum_i |K_hat_i(k)|^2
  
  The amplification matrix eigenvalues satisfy:
    lambda(+/-) = [beta(k) +/- sqrt(beta(k)^2 - 4 delta)] / 2
    beta(k) = (2 + gamma tau + tau^2 L_hat(k)) / (1 + gamma tau)
    delta = 1 / (1 + gamma tau)
  
  |lambda| > 1 indicates instability at frequency k.
  This occurs when tau^2 |L_hat(k)| exceeds the damping gamma tau,
  i.e., when diffusion operates too aggressively at high frequencies.
""")

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_data = {
        "L": L, "T_max": T_max, "num_images": len(images),
        "oracle": {
            "mean_T": float(np.mean(oracle_Ts)),
            "std_T": float(np.std(oracle_Ts)),
            "mean_psnr": float(np.mean(oracle_psnrs)),
            "per_image": [{"T": int(t), "psnr": float(p)}
                          for t, p in zip(oracle_Ts, oracle_psnrs)],
        },
        "methods": {
            name: {
                "mean_T": float(np.mean(method_results[name]["Ts"])),
                "mean_psnr": float(np.mean(method_results[name]["psnrs"])),
                "oracle_gap": float(np.mean(oracle_psnrs) - np.mean(method_results[name]["psnrs"])),
                "hit_rate_pct": float(np.mean(np.array(method_results[name]["Ts"]) == oracle_Ts) * 100),
            }
            for name in methods
        },
        "fixed_baselines": {
            name: {
                "mean_psnr": float(np.mean(psnrs)),
                "oracle_gap": float(np.mean(oracle_psnrs) - np.mean(psnrs)),
            }
            for name, psnrs in fixed_baselines.items()
        },
        "von_neumann": {
            "stable_fraction_pct": float(frac_stable),
            "max_spectral_radius": float(max_spec),
        },
    }
    json_path = os.path.join(OUT_DIR, f"optimal_stopping_L{L}.json")
    with open(json_path, "w") as fp:
        json.dump(json_data, fp, indent=2)
    print(f"  JSON -> {json_path}")

    return images, oracle_Ts, oracle_psnrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--ckpt_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--test_dir", default=CLEAN_TEST_DIR)
    args = parser.parse_args()
    analyze(L=args.L, ckpt_dir=args.ckpt_dir, test_dir=args.test_dir)


if __name__ == "__main__":
    main()
