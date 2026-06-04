"""Generate the side-by-side comparison figures for v1 (digit-5 example)
vs v2 (Competition 2 actual = digit 2) using the pre-polish v2 answer.

Outputs:
  figures/lesson/fig_v2_vs_v1.png   side-by-side conductivity images
  figures/fig_v2_vs_v1.png          mirror for talk
  figures/lesson/fig_v2_recovered.png   y_obs + recovered for v2
  figures/fig_v2_recovered.png          mirror
"""
import os, numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


SUB = ".."
FIG_T = f"{SUB}/figures"
FIG_L = f"{SUB}/figures/lesson"


def mathsci(v):
    s = f"{v:.2e}"; a, b = s.split("e")
    return rf"${a}\times 10^{{{int(b)}}}$"


def load_pair(p):
    d = loadmat(p)
    img = d.get('sigma_answer')
    if img is None:
        img = 1.0 + d['x_answer']
    return np.asarray(img), float(np.asarray(d['final_misfit']).flat[0])


v1_sigma, v1_mis = load_pair(f"{SUB}/final_answer.mat")              # digit 5, polished
v2_sigma, v2_mis = load_pair(f"{SUB}/final_answer_v2_pre_polish.mat") # digit 2, pre-polish
y2 = loadmat(f"{SUB}/code/y_truth_measurement_noisy.mat")['y_truth_noisy'].flatten()


# ------ fig_v2_recovered ------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(np.arange(len(y2)), y2, lw=0.5, color='steelblue')
axes[0].set_xlabel('measurement index $k$  (out of 1900)', fontsize=10)
axes[0].set_ylabel('voltage $y_k$', fontsize=10)
axes[0].set_title('The input: Competition 2 $y_{obs}$ (1900 noisy voltages)',
                  fontsize=11, fontweight='bold')
axes[0].grid(alpha=0.3)
im = axes[1].imshow(v2_sigma, cmap='viridis', vmin=1, vmax=2)
axes[1].axis('off')
axes[1].set_title(r'The output: recovered conductivity $\sigma$' + '\n'
                  + 'digit 2, pre-polish misfit ' + mathsci(v2_mis),
                  fontsize=11, fontweight='bold')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
for d_out in (FIG_T, FIG_L):
    os.makedirs(d_out, exist_ok=True)
    out = f"{d_out}/fig_v2_recovered.png"
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")
plt.close()


# ------ fig_v2_vs_v1 ------
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
axes[0].imshow(v1_sigma, cmap='viridis', vmin=1, vmax=2)
axes[0].set_title('Worked example: digit-5 test vector\n'
                  '(noiseless), polished misfit ' + mathsci(v1_mis),
                  fontsize=10)
axes[0].axis('off')
axes[1].imshow(v2_sigma, cmap='viridis', vmin=1, vmax=2)
axes[1].set_title('Competition 2 actual vector\n'
                  '(noisy), pre-polish misfit ' + mathsci(v2_mis),
                  fontsize=10)
axes[1].axis('off')
plt.suptitle('Same pipeline, two vectors',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
for d_out in (FIG_T, FIG_L):
    out = f"{d_out}/fig_v2_vs_v1.png"
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")
plt.close()
