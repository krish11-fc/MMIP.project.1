import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

# ── Input arrow ────────────────────────────────────────────
ax.annotate("", xy=(0.5, 2.5), xytext=(1.0, 2.5),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.0, 2.8, "Noisy\n$f$", ha="center", va="bottom", fontsize=11)

# ── N stages ───────────────────────────────────────────────
stage_colors = ["#E8F5E9", "#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A",
                "#4CAF50", "#43A047", "#388E3C", "#2E7D32", "#1B5E20"]
num_stages = 5
for i in range(num_stages):
    x0 = 1.5 + i * 1.8
    rect = mpatches.FancyBboxPatch((x0, 1.0), 1.4, 3.0,
                                     boxstyle="round,pad=0.15",
                                     facecolor=stage_colors[i],
                                     edgecolor="#333", lw=1.2)
    ax.add_patch(rect)
    ax.text(x0 + 0.7, 3.8, f"Stage {i+1}",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x0 + 0.7, 2.4, "$\\nabla^2 u_i$\n$\\varphi_i(\\cdot)$\n$g(\\cdot)$\n$u_{i+1}$",
            ha="center", va="center", fontsize=8, color="#333")
    if i < num_stages - 1:
        ax.annotate("", xy=(x0 + 1.4, 2.5), xytext=(x0 + 1.7, 2.5),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#666"))

# ── Output arrow ───────────────────────────────────────────
x_out = 1.5 + (num_stages - 1) * 1.8 + 1.4
ax.annotate("", xy=(x_out, 2.5), xytext=(x_out + 0.5, 2.5),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(x_out + 0.8, 2.8, "Restored\n$u_T$", ha="center", va="bottom", fontsize=11)

# ── Legend ─────────────────────────────────────────────────
legend_y = 0.3
items = [
    ("$\\varphi_i$", "Learned RBF influence function"),
    ("$g(\\cdot)$", "Gray-level indicator function"),
    ("$\\nabla^2$", "Laplacian diffusion term"),
    ("$\\gamma,\\tau$", "Inertia & time-step params"),
]
for j, (sym, desc) in enumerate(items):
    ax.text(2.0 + j * 2.5, legend_y, f"  {sym}: {desc}", fontsize=8, color="#555")

ax.set_title("Inertial TNRD Network Architecture", fontsize=14, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig("/mnt/c/SEM-6/MMIP/project/ablation/architecture_diagram.pdf", dpi=200, bbox_inches="tight")
fig.savefig("/mnt/c/SEM-6/MMIP/architecture_diagram.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("Saved architecture_diagram.pdf")
