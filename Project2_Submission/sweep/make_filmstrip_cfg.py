"""Regenerate fig_denoising_progression.png using the cond + CFG chain.

Same chain seed as make_denoising_gifs_cfg.py for the competition vector
(class=5), pick 6 snapshots, save as a 1x6 grid matching the talk slide.
"""
import os, sys, time
sys.path.insert(0, "../code")

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from pnp_dm_cfg import diff_one_denoise_cfg, load_cond_ddpm
from schedule import get_schedule

OUT_DIR_TALK   = "../figures"
OUT_DIR_LESSON = "../figures/lesson"
COND_DIR = "./ddpm_cond_rot15_ema"

SAVE_AT = [0, 4, 12, 25, 45, 79]  # 0-indexed iters; titles show k+1


def main():
    setup = ERTSetup("../code")
    try:
        unet, sch = load_cond_ddpm(COND_DIR)
        alphas_cumprod = sch.alphas_cumprod.float()
        setup.scheduler = sch
        setup.alphas_cumprod = alphas_cumprod

        n_iter = 80
        sigmas = get_schedule("geomspace", n_iter=n_iter,
                               sigma_max=1.0, sigma_min=0.05)
        torch.manual_seed(0); np.random.seed(0)
        x = torch.randn(28, 28).clamp(-1.5, 1.5)
        snapshots = []
        class_label = 5

        print(f"running cond+CFG chain (class={class_label}) for {n_iter} iters")
        t0 = time.time()
        for k, sigma_k in enumerate(sigmas):
            z = gn_likelihood_prox(setup, x, float(sigma_k),
                                    sigma_n=1e-4, max_inner=2)
            x = diff_one_denoise_cfg(unet, alphas_cumprod, z, float(sigma_k),
                                      class_label, w=3.0)
            if k in SAVE_AT:
                x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
                mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                                     - setup.y_obs) ** 2).sum())
                snapshots.append((k, float(sigma_k), mis, x_pix))
        print(f"  done in {time.time()-t0:.0f}s")

        fig, axes = plt.subplots(1, len(snapshots), figsize=(2.0*len(snapshots), 2.4))
        for ax, (k, sig, mis, img) in zip(axes, snapshots):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"iter {k+1}\nmisfit {mis:.1e}", fontsize=11)
            ax.axis('off')
        plt.suptitle("One cond+CFG chain (class=5) over 80 iterations",
                      fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()

        for d in (OUT_DIR_TALK, OUT_DIR_LESSON):
            os.makedirs(d, exist_ok=True)
            out = f"{d}/fig_denoising_progression.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
