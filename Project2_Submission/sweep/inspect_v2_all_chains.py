"""Run all 32 cond+CFG chains on the new Competition 2 vector and save
every chain's final image as a 4x8 grid. Sort by final misfit. Also
classify each chain. Goal: see whether ANY chain produces a clean
digit, regardless of misfit rank.
"""
import sys, os, time
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from pnp_dm_cfg import diff_one_denoise_cfg, load_cond_ddpm
from schedule import get_schedule
from train_cond_ddpm import N_CLASSES
from stage_c_template_retrieval import train_classifier
from run_di_rtg_v2 import classify_with_top_k


COND_DIR = "./ddpm_cond_rot15_ema"
N_CHAINS = 32
N_ITER = 80


def main():
    setup = ERTSetup("../code")
    try:
        # Swap in noisy vector
        d = loadmat("../code/y_truth_measurement_noisy.mat")
        y_new = d['y_truth_noisy'].astype(np.float32).flatten()
        setup.y_obs = torch.tensor(y_new, dtype=torch.float32)

        unet, sch = load_cond_ddpm(COND_DIR)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        alphas_cumprod = setup.alphas_cumprod
        print(f"Loaded cond DDPM, new y_obs swapped in")

        sigmas = get_schedule("geomspace", n_iter=N_ITER,
                              sigma_max=1.0, sigma_min=0.05)
        torch.manual_seed(0); np.random.seed(0)

        # Train classifier
        print("Training TinyMNIST classifier...")
        mn = loadmat("../code/MNIST Data/mnist.mat")
        M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_train = mn['training']['labels'][0, 0].flatten()
        if M_train.max() > 1.5: M_train = M_train / 255.0
        clf = train_classifier(M_train, L_train, epochs=8, verbose=False)

        chain_results = []
        t0 = time.time()
        for c in range(N_CHAINS):
            class_label = c % N_CLASSES
            x = torch.randn(28, 28).clamp(-1.5, 1.5)
            for k, sigma_k in enumerate(sigmas):
                z = gn_likelihood_prox(setup, x, float(sigma_k),
                                        sigma_n=1e-4, max_inner=2)
                x = diff_one_denoise_cfg(unet, alphas_cumprod, z, float(sigma_k),
                                          class_label, w=3.0)
            x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
            mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                                 - setup.y_obs) ** 2).sum())
            top_classes, probs = classify_with_top_k(clf, x_pix, k=10)
            clf_digit = top_classes[0]; clf_prob = float(probs[clf_digit])
            chain_results.append({
                'idx': c, 'target': class_label,
                'misfit': mis, 'img': x_pix,
                'clf_digit': clf_digit, 'clf_prob': clf_prob,
            })
            print(f"  chain {c:2d} target={class_label}  misfit {mis:.3e}  "
                  f"clf={clf_digit} (p={clf_prob:.2f})  ({time.time()-t0:.0f}s)")

        # Save 4x8 grid sorted by misfit
        chain_results.sort(key=lambda r: r['misfit'])
        fig, axes = plt.subplots(4, 8, figsize=(16, 9))
        for i, r in enumerate(chain_results):
            ax = axes[i // 8, i % 8]
            ax.imshow(r['img'], cmap='gray', vmin=0, vmax=1)
            ax.set_title(
                f"#{i+1}  t={r['target']}  m={r['misfit']:.2e}\n"
                f"clf={r['clf_digit']} (p={r['clf_prob']:.2f})",
                fontsize=8)
            ax.axis('off')
        plt.suptitle("All 32 cond+CFG chains on Competition 2 (sorted by misfit, ascending)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = "../figures/v2_all_chains_grid.png"
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"\nSaved {out}")

        # Also save grid grouped by target class (column = chain rank within class)
        by_class = {c: [] for c in range(N_CLASSES)}
        # original order
        chain_results.sort(key=lambda r: r['idx'])
        for r in chain_results:
            by_class[r['target']].append(r)
        fig, axes = plt.subplots(N_CLASSES, 4, figsize=(8, 18))
        for cls in range(N_CLASSES):
            for j, r in enumerate(by_class[cls][:4]):
                ax = axes[cls, j] if N_CLASSES > 1 else axes[j]
                ax.imshow(r['img'], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"chain {r['idx']}  m={r['misfit']:.2e}\n"
                             f"clf={r['clf_digit']} (p={r['clf_prob']:.2f})",
                             fontsize=7)
                ax.axis('off')
            if not by_class[cls]:
                continue
            axes[cls, 0].text(-0.25, 0.5, f"target={cls}",
                              transform=axes[cls, 0].transAxes,
                              fontsize=10, fontweight='bold',
                              rotation=90, ha='center', va='center')
        plt.suptitle("Same chains, grouped by target class",
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out2 = "../figures/v2_chains_by_class.png"
        plt.savefig(out2, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {out2}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
