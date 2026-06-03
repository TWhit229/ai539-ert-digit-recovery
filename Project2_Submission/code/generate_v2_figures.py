"""Generate the figures for the v2 narrative (class-conditional + TV polish
is THE method; base PnP-DM + GN-CG is 'what we tried first').

Outputs to figures/ for the slideshow and figures/lesson/ for the PDF.
"""
import os, numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


SUB_ROOT = ".."
FIG_DIR = f"{SUB_ROOT}/figures"
LES_DIR = f"{SUB_ROOT}/figures/lesson"
SWEEP   = f"{SUB_ROOT}/sweep"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LES_DIR, exist_ok=True)


def load_mat(p):
    d = loadmat(p)
    img = d.get('sigma_answer', None)
    if img is None:
        img = 1.0 + d['x_answer']
    return np.asarray(img), float(np.asarray(d['final_misfit']).flat[0])


def mathsci(v):
    """Format v as $a \\times 10^{b}$ (LaTeX-friendly)."""
    s = f"{v:.2e}"  # e.g. "1.83e-11"
    a, b = s.split("e")
    return rf"${a}\times 10^{{{int(b)}}}$"


# ---------- y_obs for the input plot ----------
y_obs = loadmat(f"{SUB_ROOT}/code/y_truth_measurement.mat")['y_truth'].flatten()

# ---------- the new winner: cond.geomspace.tv ----------
win_img, win_mis = load_mat(f"{SWEEP}/results/polished/cond_geomspace_tv.mat")

# ---------- the old "base" answer for comparison ----------
old_img, old_mis = load_mat(f"{SWEEP}/results/polished/base_geomspace_lm.mat")

# ---------- "what we tried" supplements ----------
# (Reuse existing dps_chains.png etc. — no need to regenerate.)

# ================================================================
# FIG: fig12_recovered_sigma.png — input + new recovered output
# (replaces the speckled 5/6-morph figure)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(np.arange(len(y_obs)), y_obs, lw=0.5, color='steelblue')
axes[0].set_xlabel('measurement index $k$  (out of 1900)', fontsize=10)
axes[0].set_ylabel('voltage $y_k$', fontsize=10)
axes[0].set_title('The input: $y_{obs}$ (1900 boundary voltages)',
                  fontsize=11, fontweight='bold')
axes[0].grid(alpha=0.3)

im = axes[1].imshow(win_img, cmap='viridis', vmin=1, vmax=2)
axes[1].axis('off')
axes[1].set_title(r'The output: recovered conductivity $\sigma$  '
                   '(misfit ' + mathsci(win_mis) + ')',
                   fontsize=11, fontweight='bold')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
out = f"{LES_DIR}/fig12_recovered_sigma.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {out}")


# ================================================================
# FIG: fig13_base_vs_winner.png — side-by-side
# old (base + LM) vs new (cond + CFG + TV)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
axes[0].imshow(old_img, cmap='viridis', vmin=1, vmax=2)
axes[0].set_title('Base pipeline\n(PnP-DM + GN-CG polish)\n'
                   'misfit ' + mathsci(old_mis),
                   fontsize=10)
axes[0].axis('off')
axes[1].imshow(win_img, cmap='viridis', vmin=1, vmax=2)
axes[1].set_title('New pipeline\n(class-cond + CFG, TV polish)\n'
                   'misfit ' + mathsci(win_mis),
                   fontsize=10)
axes[1].axis('off')
plt.suptitle('What changed: cleaner background, less stroke-speckle',
              fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
out = f"{LES_DIR}/fig13_base_vs_winner.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {out}")


# ================================================================
# FIG: fig14_sweep_grid.png — annotated 3x4 grid (priors x schedule.polish)
# ================================================================
PRIORS    = ["base", "bigger", "cond"]
PRIOR_LBL = {"base": "base prior\n(1M params)",
             "bigger": "bigger prior\n(6M + EMA)",
             "cond": "class-conditional\n+ CFG (w=3)"}
SCHEDS    = ["geomspace", "neg_rho"]
POLISHES  = ["lm", "tv"]

fig, axes = plt.subplots(3, 4, figsize=(11, 10))
fig.subplots_adjust(top=0.88)
for i, prior in enumerate(PRIORS):
    for j, (s, pol) in enumerate([(s, p) for s in SCHEDS for p in POLISHES]):
        path = f"{SWEEP}/results/polished/{prior}_{s}_{pol}.mat"
        img, mis = load_mat(path)
        ax = axes[i, j]
        ax.imshow(img, cmap='viridis', vmin=1, vmax=2)
        # highlight the winner with an orange border
        if (prior, s, pol) == ("cond", "geomspace", "tv"):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#D73F09')
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title('WINNER\nmisfit ' + mathsci(mis), fontsize=9,
                          color='#D73F09', fontweight='bold')
        else:
            ax.axis('off')
            ax.set_title('misfit ' + mathsci(mis), fontsize=9)
        if i == 0:
            ax.text(0.5, 1.25, f'schedule = {s}\npolish = {pol}',
                     ha='center', va='bottom', transform=ax.transAxes,
                     fontsize=9, fontweight='bold')
    axes[i, 0].text(-0.18, 0.5, PRIOR_LBL[prior], ha='center', va='center',
                     transform=axes[i, 0].transAxes, fontsize=10,
                     fontweight='bold', rotation=90)
plt.suptitle('12-combination sweep:  rows = prior,  cols = schedule x polish',
              fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = f"{LES_DIR}/fig14_sweep_grid.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {out}")


# ================================================================
# FIG: fig15_tv_vs_lm.png — closeup TV vs LM polish, base+geomspace
# pre / lm / tv  side by side, plus zoomed background patches
# ================================================================
pre_img, pre_mis = load_mat(f"{SWEEP}/results/pre_polish/base_geomspace.mat")
lm_img,  lm_mis  = load_mat(f"{SWEEP}/results/polished/base_geomspace_lm.mat")
tv_img,  tv_mis  = load_mat(f"{SWEEP}/results/polished/base_geomspace_tv.mat")

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, img, title, mis in zip(
        axes,
        [pre_img, lm_img, tv_img],
        ['No polish (pre)',
         'LM polish (current default)',
         'TV-regularized polish (new)'],
        [pre_mis, lm_mis, tv_mis]):
    ax.imshow(img, cmap='viridis', vmin=1, vmax=2)
    ax.set_title(title + '\nmisfit ' + mathsci(mis), fontsize=10)
    ax.axis('off')
plt.suptitle('Polish comparison on the base prior (geomspace schedule)',
              fontsize=11, fontweight='bold', y=1.04)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{LES_DIR}/fig15_tv_vs_lm.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {out}")


# ================================================================
# FIG: fig16_cond_samples.png — unconditional digits 0-9 from CFG
# (copy from the trained model dir if it exists)
# ================================================================
src_cond_png = f"{SWEEP}/ddpm_cond_rot15_ema/cond_samples.png"
if os.path.isfile(src_cond_png):
    import shutil
    out = f"{LES_DIR}/fig16_cond_samples.png"
    shutil.copy(src_cond_png, out)
    print(f"Copied {out}")


# ================================================================
# FIG: fig17_pre_vs_polish.png — pre-polish vs polished, for each prior
# (visualizes the "polish is over-fitting" finding)
# ================================================================
combos = [
    ("base",   "geomspace"),
    ("bigger", "geomspace"),
    ("cond",   "geomspace"),
]
fig, axes = plt.subplots(len(combos), 3, figsize=(8, 8.5))
for i, (prior, s) in enumerate(combos):
    pre_img, pre_mis = load_mat(f"{SWEEP}/results/pre_polish/{prior}_{s}.mat")
    lm_img,  lm_mis  = load_mat(f"{SWEEP}/results/polished/{prior}_{s}_lm.mat")
    tv_img,  tv_mis  = load_mat(f"{SWEEP}/results/polished/{prior}_{s}_tv.mat")
    for ax, img, mis, title in zip(
            axes[i],
            [pre_img, lm_img, tv_img],
            [pre_mis, lm_mis, tv_mis],
            ['no polish (pre)', 'LM polish', 'TV polish']):
        ax.imshow(img, cmap='viridis', vmin=1, vmax=2)
        ax.set_title(title + '\n' + mathsci(mis), fontsize=9)
        ax.axis('off')
    axes[i, 0].text(-0.18, 0.5, PRIOR_LBL[prior], ha='center', va='center',
                     transform=axes[i, 0].transAxes, fontsize=10,
                     fontweight='bold', rotation=90)
plt.suptitle('Pre-polish images are often cleaner than polished ones',
              fontsize=11, fontweight='bold', y=0.995)
plt.tight_layout()
out = f"{LES_DIR}/fig17_pre_vs_polish.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {out}")


# Also duplicate the lesson copies into figures/ for slide use
import shutil
for f in ["fig12_recovered_sigma.png", "fig13_base_vs_winner.png",
          "fig14_sweep_grid.png", "fig15_tv_vs_lm.png",
          "fig16_cond_samples.png", "fig17_pre_vs_polish.png"]:
    src = f"{LES_DIR}/{f}"
    if os.path.isfile(src):
        dst = f"{FIG_DIR}/{f}"
        shutil.copy(src, dst)
        print(f"Mirrored -> {dst}")

print("\nAll figures generated.")
