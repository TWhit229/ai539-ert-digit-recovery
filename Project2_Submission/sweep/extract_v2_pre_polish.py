"""Extract just the WINNING chain's pre-polish image for the new
Competition 2 vector. Replays chain 1 (target class=1, which was the
winning class in the full 32-chain run) and saves both the figure
and a .mat file with the pre-polish answer.

Doesn't run polish at all — for the noisy vector the polish overshoots.
"""
import sys, os, time
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from pnp_dm_cfg import diff_one_denoise_cfg, load_cond_ddpm
from schedule import get_schedule
from train_cond_ddpm import N_CLASSES


COND_DIR = "./ddpm_cond_rot15_ema"
WINNING_CHAIN_IDX = 1  # from the full run, chain 1 (target_class=1) was lowest


def mathsci(v):
    s = f"{v:.2e}"; a, b = s.split("e")
    return rf"${a}\times 10^{{{int(b)}}}$"


def main():
    setup = ERTSetup("../code")
    try:
        # Swap in new y_obs
        d = loadmat("../code/y_truth_measurement_noisy.mat")
        y_new = d['y_truth_noisy'].astype(np.float32).flatten()
        true_y = setup.y_obs
        setup.y_obs = torch.tensor(y_new, dtype=torch.float32)

        unet, sch = load_cond_ddpm(COND_DIR)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        alphas_cumprod = setup.alphas_cumprod
        print("Loaded conditional DDPM + new y_obs")

        # Replay the production sampler up to chain WINNING_CHAIN_IDX so the
        # random number sequence matches. Chain 0 gets the first 1+80 draws,
        # chain 1 gets the next 1+80, etc.
        n_iter = 80
        sigmas = get_schedule("geomspace", n_iter=n_iter,
                              sigma_max=1.0, sigma_min=0.05)
        torch.manual_seed(0); np.random.seed(0)

        t0 = time.time()
        # Burn through chain 0 to advance the RNG
        for c in range(WINNING_CHAIN_IDX + 1):
            class_label = c % N_CLASSES
            x = torch.randn(28, 28).clamp(-1.5, 1.5)
            if c == WINNING_CHAIN_IDX:
                target_class = class_label
                snapshots = []
                x_pix0 = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
                mis0 = 0.5 * float(((setup.forward(torch.tensor(x_pix0, dtype=torch.float32))
                                      - setup.y_obs) ** 2).sum())
                snapshots.append((0, float(sigmas[0]), mis0, x_pix0))
            for k, sigma_k in enumerate(sigmas):
                z = gn_likelihood_prox(setup, x, float(sigma_k),
                                        sigma_n=1e-4, max_inner=2)
                x = diff_one_denoise_cfg(unet, alphas_cumprod, z, float(sigma_k),
                                          class_label, w=3.0)
                if c == WINNING_CHAIN_IDX:
                    x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
                    mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                                         - setup.y_obs) ** 2).sum())
                    snapshots.append((k + 1, float(sigma_k), mis, x_pix))
        print(f"Replay done in {time.time()-t0:.0f}s; "
              f"chain {WINNING_CHAIN_IDX} (class={target_class}) "
              f"final pre-polish misfit {snapshots[-1][2]:.3e}")

        # Final image
        x_final = snapshots[-1][3]
        sigma_answer = 1.0 + x_final
        final_misfit = snapshots[-1][2]

        savemat("../final_answer_v2_pre_polish.mat", {
            'sigma_answer': sigma_answer,
            'x_answer': x_final,
            'final_misfit': float(final_misfit),
            'target_class': int(target_class),
            'note': 'pre-polish; TV polish overshoots on noisy data',
        })
        print(f"Saved ../final_answer_v2_pre_polish.mat (misfit {final_misfit:.3e})")

        # Figure
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].plot(np.arange(len(y_new)), y_new, lw=0.5, color='steelblue')
        axes[0].set_xlabel('measurement index $k$  (out of 1900)', fontsize=10)
        axes[0].set_ylabel('voltage $y_k$', fontsize=10)
        axes[0].set_title('The input: Competition 2 $y_{obs}$ (1900 boundary voltages)',
                          fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.3)
        im = axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].axis('off')
        axes[1].set_title('The output: recovered conductivity $\\sigma$  '
                          '(pre-polish, misfit ' + mathsci(final_misfit) + ')',
                          fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        for d_out in ('../figures', '../figures/lesson'):
            out = f"{d_out}/fig_v2_recovered.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
        plt.close()
    finally:
        setup.y_obs = true_y
        setup.quit()


if __name__ == "__main__":
    main()
