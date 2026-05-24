"""
generate_architecture_diagram.py — Publication-quality architecture diagram for IEEE paper.

Black-and-white optimized for print. Shows:
  - Top: T-stage unrolled network
  - Bottom: single-stage detail with all components and data flow

Output: outputs/architecture_diagram.pdf, outputs/architecture_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style constants (all grayscale) ──────────────────────────────────────────
LW = 1.2            # line width
LW_THICK = 1.8      # thick line for main boxes
FONT = "serif"
FS_SM = 6
FS = 7
FS_LG = 8
FS_TITLE = 9

# Grayscale fills for different component types
FILL_RECUR = "0.92"    # PDE recurrence
FILL_FILTER = "0.85"   # Fixed filter bank
FILL_LEARN = "0.75"    # Learned RBF
FILL_ANALYTIC = "0.88" # Analytic g/c functions
FILL_DATA = "0.95"     # Data/tensors


def _box(ax, x, y, w, h, fill, ls="-", lw=LW_THICK, ec="black"):
    """Add a rounded box patch."""
    p = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor=fill, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(p)
    return p


def _arr(ax, x1, y1, x2, y2, ls="-", lw=LW, ec="black", style="->"):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle=style, linestyle=ls, lw=lw, color=ec))


def _txt(ax, x, y, s, fs=FS, weight="normal", style="normal"):
    ax.text(x, y, s, fontsize=fs, ha="center", va="center",
            fontfamily=FONT, fontweight=weight, fontstyle=style)


# ──────────────────────────────────────────────────────────────────────────────
# Detailed single-stage drawing
# ──────────────────────────────────────────────────────────────────────────────
def draw_stage(ax):
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    _txt(ax, 0.50, 0.965, "Single Stage t  (Eq.~4)", fs=FS_TITLE, weight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN: three inputs
    # ══════════════════════════════════════════════════════════════════════════
    inputs = [(0.030, 0.820, r"$u^{t-1}$"),
              (0.030, 0.670, r"$u^{t-2}$"),
              (0.030, 0.520, r"$f$ (noisy)")]
    for x, y, lab in inputs:
        _box(ax, x, y - 0.035, 0.065, 0.065, FILL_DATA, lw=LW)
        _txt(ax, x + 0.0325, y, lab, fs=FS, weight="bold")

    # Arrows from inputs to inertial block
    for _, y, _ in inputs:
        _arr(ax, 0.095, y, 0.130, y, lw=LW)

    # ══════════════════════════════════════════════════════════════════════════
    # INERTIAL RECURRENCE block (Eq.~4, left part)
    # ══════════════════════════════════════════════════════════════════════════
    rx, ry, rw, rh = 0.130, 0.480, 0.100, 0.435
    _box(ax, rx, ry, rw, rh, FILL_RECUR)
    _txt(ax, rx + rw / 2, ry + rh - 0.045, "Inertial", fs=FS, weight="bold")
    _txt(ax, rx + rw / 2, ry + rh - 0.075, "recurrence", fs=FS, weight="bold")
    _txt(ax, rx + rw / 2, ry + rh - 0.115, r"$(1+\gamma\tau)u^{t+1}$", fs=FS_SM)
    _txt(ax, rx + rw / 2, ry + rh - 0.145, r"$= (2+\gamma\tau)u^{t}$", fs=FS_SM)
    _txt(ax, rx + rw / 2, ry + rh - 0.175, r"$- u^{t-1} + \tau^2\cdot$", fs=FS_SM)
    _txt(ax, rx + rw / 2, ry + 0.045, r"$(\gamma=0.5$)", fs=FS_SM, style="italic")
    _txt(ax, rx + rw / 2, ry + 0.020, r"$(\tau=0.2$)", fs=FS_SM, style="italic")

    # Arrow from inertial block to filter bank (rightward)
    _arr(ax, rx + rw, ry + rh - 0.08, 0.250, ry + rh - 0.08, lw=LW)

    # ══════════════════════════════════════════════════════════════════════════
    # FILTER BANK block
    # ══════════════════════════════════════════════════════════════════════════
    fx, fy, fw, fh = 0.255, 0.540, 0.110, 0.315
    _box(ax, fx, fy, fw, fh, FILL_FILTER, ls="--")
    _txt(ax, fx + fw / 2, fy + fh - 0.040, "Filter bank", fs=FS, weight="bold")
    _txt(ax, fx + fw / 2, fy + fh - 0.070, r"$\{k_i\}_{i=1}^{48}$", fs=FS)
    _txt(ax, fx + fw / 2, fy + fh - 0.100, r"$7\times7$ DCT-II", fs=FS_SM)
    _txt(ax, fx + fw / 2, fy + fh - 0.125, "fixed, zero-mean", fs=FS_SM, style="italic")
    _txt(ax, fx + fw / 2, fy + 0.040, r"$k_i \ast u^t$", fs=FS, weight="bold")
    _txt(ax, fx + fw / 2, fy + 0.015, "48 responses", fs=FS_SM)

    # Diverging arrows: top -> RBF, bottom -> gray-level indicator
    _arr(ax, fx + fw, fy + fh - 0.060, 0.395, fy + fh - 0.030, lw=LW)
    _arr(ax, fx + fw, fy + 0.050, 0.395, fy - 0.040, lw=LW)

    # ══════════════════════════════════════════════════════════════════════════
    # RBF INFLUENCE FUNCTION block
    # ══════════════════════════════════════════════════════════════════════════
    rx2, ry2, rw2, rh2 = 0.395, 0.600, 0.120, 0.240
    _box(ax, rx2, ry2, rw2, rh2, FILL_LEARN, ls="-.")
    _txt(ax, rx2 + rw2 / 2, ry2 + rh2 - 0.035, r"RBF $\phi_i^t(z)$", fs=FS, weight="bold")
    _txt(ax, rx2 + rw2 / 2, ry2 + rh2 - 0.065, r"$\sum_{j=1}^{63} w_{ij}^t\,G_j(z)$", fs=FS_SM)
    _txt(ax, rx2 + rw2 / 2, ry2 + rh2 - 0.095, r"$G_j(z) = e^{-(z-\mu_j)^2/2h^2}$", fs=FS_SM)
    _txt(ax, rx2 + rw2 / 2, ry2 + 0.050, r"centres $\mu_j\in[-300,300]$", fs=FS_SM, style="italic")
    _txt(ax, rx2 + rw2 / 2, ry2 + 0.025, r"learned $w_{ij}^t$", fs=FS_SM, style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # GRAY-LEVEL INDICATOR block (w/ Gaussian smoothing and edge stopping)
    # ══════════════════════════════════════════════════════════════════════════
    gx, gy, gw, gh = 0.395, 0.330, 0.120, 0.220
    _box(ax, gx, gy, gw, gh, FILL_ANALYTIC, ls=":")

    _txt(ax, gx + gw / 2, gy + gh - 0.030, r"Gray-level $g(u_\sigma)$", fs=FS, weight="bold")
    _txt(ax, gx + gw / 2, gy + gh - 0.060, r"$g(s) = 2|s|^\nu/(M^\nu+|s|^\nu)$", fs=FS_SM)
    _txt(ax, gx + gw / 2, gy + gh - 0.085, r"$\nu=1.0$, $M=255$", fs=FS_SM)
    _txt(ax, gx + gw / 2, gy + 0.060, r"$u_\sigma = G_\sigma \ast u$", fs=FS_SM)
    _txt(ax, gx + gw / 2, gy + 0.035, r"$\sigma=1.0$ (Gaussian)", fs=FS_SM, style="italic")

    # Small sub-box for edge-stopping
    subx, suby, subw, subh = gx + 0.005, gy + 0.003, gw - 0.010, 0.055
    _box(ax, subx, suby, subw, subh, "0.95", ls=":", lw=LW)
    _txt(ax, subx + subw / 2, suby + subh / 2,
         r"$c(|\nabla u|) = 1/(1+(|\nabla u|/K)^2)$", fs=FS_SM - 0.5)
    _txt(ax, subx + subw / 2, suby + subh / 2 - 0.025,
         r"$K=128$ (edge stopping)", fs=FS_SM - 0.5, style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # MODULATION: multiply Phi(gradient) * g(intensity) * c(edge)
    # ══════════════════════════════════════════════════════════════════════════
    # RBF output arrow
    _arr(ax, rx2 + rw2, ry2 + rh2 / 2, 0.545, 0.630, lw=LW)
    # Gray-level indicator arrow
    _arr(ax, gx + gw, gy + gh / 2, 0.545, 0.430, lw=LW)

    # Modulation box (multiply)
    mx, my, mw, mh = 0.550, 0.510, 0.055, 0.150
    _box(ax, mx, my, mw, mh, "white", lw=LW_THICK)
    _txt(ax, mx + mw / 2, my + mh / 2, "Decoupled", fs=FS - 0.5, weight="bold")
    _txt(ax, mx + mw / 2, my + mh / 2 - 0.025, "diffusivity", fs=FS - 0.5, weight="bold")
    _txt(ax, mx + mw / 2, my + mh / 2 - 0.055, r"$g \cdot c$", fs=FS, weight="bold")
    _txt(ax, mx + mw / 2, my + mh / 2 + 0.025, r"$(\times)$", fs=FS_SM)

    # Arrow to adjoint filter
    _arr(ax, mx + mw, my + mh / 2, 0.630, my + mh / 2, lw=LW)

    # ══════════════════════════════════════════════════════════════════════════
    # ADJOINT FILTER (divergence) block
    # ══════════════════════════════════════════════════════════════════════════
    afx, afy, afw, afh = 0.635, 0.540, 0.085, 0.170
    _box(ax, afx, afy, afw, afh, FILL_FILTER, ls="--")
    _txt(ax, afx + afw / 2, afy + afh - 0.030, "Adjoint", fs=FS, weight="bold")
    _txt(ax, afx + afw / 2, afy + afh - 0.060, "filters", fs=FS, weight="bold")
    _txt(ax, afx + afw / 2, afy + 0.040, r"$\sum_i \bar{k}_i$", fs=FS)
    _txt(ax, afx + afw / 2, afy + 0.015, "(divergence)", fs=FS_SM, style="italic")

    # Arrow back to recurrence (sum into the + tau^2 term)
    _arr(ax, afx + afw / 2, afy + afh, afx + afw / 2, ry + 0.070, lw=LW)
    _txt(ax, afx + afw / 2 + 0.015, ry + 0.045, r"$+\;\tau^2\sum$", fs=FS_SM,
         weight="bold")

    # Arrow from recurrence to output
    out_x = ry + rh / 2

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    ox, oy, ow, oh = 0.900, 0.535, 0.055, 0.080
    _box(ax, ox, oy, ow, oh, FILL_DATA, lw=LW_THICK)
    _txt(ax, ox + ow / 2, oy + oh / 2, r"$u^t$", fs=FS_LG, weight="bold")

    # Data flow arrow from recurrence to output
    _arr(ax, rx + rw, ry + rh / 2, ox, oy + oh / 2, lw=LW)

    # Arrow to next stage (horizontal)
    _arr(ax, ox + ow, oy + oh / 2, 0.980, oy + oh / 2, lw=LW_THICK)
    _txt(ax, 0.988, oy + oh / 2, r"$\to$ Stage $t+1$", fs=FS_SM, weight="bold",
         style="italic")

    # ── Feedback (teacher forcing) ────────────────────────────────────────────
    # Dashed arrow from output back around to the input side
    mid_fb_x = 0.505
    _arr(ax, ox + ow / 4, oy, mid_fb_x, 0.090, ls="--", lw=LW * 0.6, ec="0.4")
    _arr(ax, mid_fb_x, 0.090, 0.060, 0.090, ls="--", lw=LW * 0.6, ec="0.4")
    _arr(ax, 0.060, 0.090, 0.060, 0.520, ls="--", lw=LW * 0.6, ec="0.4")
    _txt(ax, 0.505, 0.075,
         r"$u^t$ becomes $u^{t-1}$ for next stage (teacher forcing)", fs=FS_SM,
         style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # Bottom annotations / equations
    # ══════════════════════════════════════════════════════════════════════════
    _txt(ax, 0.50, 0.010,
         r"Core recurrence: $(1+\gamma\tau)u^{t+1} = (2+\gamma\tau)u^t - u^{t-1} + \tau^2 \sum_i \bar{k}_i \ast [\phi_i^t(k_i \ast u^t) \cdot g(u_\sigma^t) \cdot c(|\nabla u^t|)]$",
         fs=FS_SM, style="italic")


# ──────────────────────────────────────────────────────────────────────────────
# Main figure
# ──────────────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(8.75, 5.0))
    fig.patch.set_facecolor("white")

    # ── TOP: unrolled network overview ──────────────────────────────────────
    ax_top = fig.add_axes([0.020, 0.580, 0.960, 0.380])
    ax_top.axis("off")

    _txt(ax_top, 0.500, 0.920,
         "InertialTNRD: Unrolled Telegraph-Diffusion Network ($T = 5$ stages)",
         fs=FS_TITLE, weight="bold")

    # Input arrow + label
    _arr(ax_top, 0.025, 0.45, 0.055, 0.45, lw=LW_THICK)
    _txt(ax_top, 0.020, 0.30, r"$f$", fs=FS_LG, weight="bold")

    # Stage blocks
    xs = 0.055
    dx_stage = 0.145
    sw = 0.090
    sh = 0.40

    for i in range(5):
        _box(ax_top, xs + i * dx_stage, 0.45 - sh / 2, sw, sh, FILL_DATA)
        _txt(ax_top, xs + i * dx_stage + sw / 2, 0.45,
             f"Stage {i+1}", fs=FS, weight="bold")

        if i < 4:
            _arr(ax_top, xs + i * dx_stage + sw, 0.45,
                 xs + (i + 1) * dx_stage, 0.45, lw=LW_THICK)

    # Output arrow + label
    out_s = xs + 5 * dx_stage
    _arr(ax_top, out_s, 0.45, out_s + 0.03, 0.45, lw=LW_THICK)
    _txt(ax_top, out_s + 0.045, 0.30, r"$u^T$", fs=FS_LG, weight="bold")

    # Arrow labels
    _txt(ax_top, 0.310, 0.68, "sequential refinement", fs=FS_SM, style="italic")
    _arr(ax_top, 0.260, 0.67, 0.360, 0.67, ls="--", lw=LW * 0.5, ec="0.5")

    # ── BOTTOM: detailed stage ──────────────────────────────────────────────
    ax_bot = fig.add_axes([0.020, 0.020, 0.960, 0.540])
    draw_stage(ax_bot)

    # ── Save ────────────────────────────────────────────────────────────────
    fig.savefig(os.path.join(OUT_DIR, "architecture_diagram.pdf"),
                dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT_DIR, "architecture_diagram.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/architecture_diagram.pdf and .png")


if __name__ == "__main__":
    main()
