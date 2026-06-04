"""Run the final pipeline (cond+CFG + TV polish) on the new Competition 2
vector y_truth_measurement_noisy.mat.

The new file uses key 'y_truth_noisy' rather than 'y_truth' (it has
measurement noise added). This script loads that file, swaps it into
the ERTSetup as the new y_obs, runs the same PnP-DM-CFG sampler we
use for the digit-5 worked example (32 chains seeded to classes 0..9
cycling, CFG w=3, geomspace schedule, 80 iterations), picks the
lowest-misfit chain, and finishes with the TV-regularized polish.

Outputs:
  ../final_answer_v2.mat              — recovered sigma + misfit
  ../figures/lesson/fig_v2_recovered.png — y_obs + recovered side-by-side
  ../figures/fig_v2_recovered.png        — mirror copy for the talk
  ../figures/lesson/fig_v2_vs_v1.png      — old digit-5 vs new side-by-side
  ../figures/fig_v2_vs_v1.png             — mirror

Run time: ~13 minutes on MPS (12 min CFG + 30 s TV polish).
"""
import argparse, time, sys, os
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from stage_a_daps_warmstart import ERTSetup
from pnp_dm_cfg import pnp_dm_cfg, load_cond_ddpm
from tv_polish import tv_polish


COND_DIR = "./ddpm_cond_rot15_ema"


def mathsci(v):
    s = f"{v:.2e}"
    a, b = s.split("e")
    return rf"${a}\times 10^{{{int(b)}}}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--y_file", default="../code/y_truth_measurement_noisy.mat")
    p.add_argument("--y_key",  default="y_truth_noisy")
    p.add_argument("--n_chains", type=int, default=32)
    p.add_argument("--n_iter",   type=int, default=80)
    p.add_argument("--sigma_n",  type=float, default=1e-4)
    p.add_argument("--cfg_w",    type=float, default=3.0)
    p.add_argument("--K_tv",     type=int,   default=120)
    p.add_argument("--tau",      type=float, default=1e-5)
    p.add_argument("--out_mat",  default="../final_answer_v2.mat")
    args = p.parse_args()

    # Load the new vector ourselves
    d = loadmat(args.y_file)
    y_new = d[args.y_key].astype(np.float32).flatten()
    print(f"Loaded {args.y_file}: key='{args.y_key}', "
          f"shape={y_new.shape}, range [{y_new.min():.4g}, {y_new.max():.4g}]")

    setup = ERTSetup("../code")
    try:
        unet, sch = load_cond_ddpm(COND_DIR)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        alphas_cumprod = setup.alphas_cumprod
        print(f"Loaded conditional DDPM from {COND_DIR}")

        # Swap in the new y_obs
        true_y_obs = setup.y_obs
        setup.y_obs = torch.tensor(y_new, dtype=torch.float32)
        print(f"Swapped in new y_obs (was digit-5 example; now Competition 2)")

        print(f"\nPnP-DM-CFG: {args.n_chains} chains x {args.n_iter} iters, w={args.cfg_w}")
        t0 = time.time()
        chains = pnp_dm_cfg(setup, unet, alphas_cumprod,
                            schedule_name="geomspace",
                            n_chains=args.n_chains, n_iter=args.n_iter,
                            sigma_n=args.sigma_n, cfg_w=args.cfg_w,
                            seed=0, verbose=True)
        best = chains[0]
        t_pre = time.time() - t0
        print(f"\nPre-polish winner: chain {best.get('chain_idx','?')}, "
              f"target class={best['class_label']}, "
              f"misfit {best['final_misfit']:.3e}  ({t_pre:.0f}s)")

        print(f"\nTV polish: K={args.K_tv}, tau={args.tau}")
        t0 = time.time()
        x_polished, log_tv = tv_polish(setup, best['x_pix'],
                                        K=args.K_tv, n_cg=60,
                                        tau=args.tau,
                                        target_misfit=1e-15,
                                        verbose=True, log_every=20)
        final_misfit = log_tv[-1][1]
        t_pol = time.time() - t0
        print(f"\nPolished misfit: {final_misfit:.3e}  ({t_pol:.0f}s)")

        sigma_answer = 1.0 + x_polished
        savemat(args.out_mat, {
            'sigma_answer': sigma_answer,
            'x_answer': x_polished,
            'final_misfit': float(final_misfit),
            'pre_polish_misfit': float(best['final_misfit']),
            'target_class': int(best['class_label']),
            'cfg_w': float(args.cfg_w),
            'y_obs_key': args.y_key,
        })
        print(f"\nSaved {args.out_mat}")

        # --- Figures ---
        os.makedirs("../figures/lesson", exist_ok=True)

        # fig_v2_recovered: y_obs plot + recovered conductivity
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].plot(np.arange(len(y_new)), y_new, lw=0.5, color='steelblue')
        axes[0].set_xlabel('measurement index $k$  (out of 1900)', fontsize=10)
        axes[0].set_ylabel('voltage $y_k$', fontsize=10)
        axes[0].set_title('The input: Competition 2 $y_{obs}$ (1900 boundary voltages)',
                          fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.3)
        im = axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].axis('off')
        axes[1].set_title(r'The output: recovered conductivity $\sigma$  '
                          '(misfit ' + mathsci(final_misfit) + ')',
                          fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        for d_out in ('../figures', '../figures/lesson'):
            out = f"{d_out}/fig_v2_recovered.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
        plt.close()

        # fig_v2_vs_v1: side-by-side example (digit-5) vs actual (Comp 2)
        v1 = loadmat('../final_answer.mat')
        v1_sigma = v1.get('sigma_answer')
        if v1_sigma is None:
            v1_sigma = 1.0 + v1['x_answer']
        v1_misfit = float(np.asarray(v1['final_misfit']).flat[0])

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
        axes[0].imshow(v1_sigma, cmap='viridis', vmin=1, vmax=2)
        axes[0].set_title('Worked example (digit-5 vector)\n'
                          + 'misfit ' + mathsci(v1_misfit),
                          fontsize=10)
        axes[0].axis('off')
        axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].set_title('Competition 2 actual vector\n'
                          + 'misfit ' + mathsci(final_misfit),
                          fontsize=10)
        axes[1].axis('off')
        plt.suptitle('Same pipeline, two vectors',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        for d_out in ('../figures', '../figures/lesson'):
            out = f"{d_out}/fig_v2_vs_v1.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
        plt.close()
    finally:
        setup.y_obs = true_y_obs  # restore (cleanliness)
        setup.quit()


if __name__ == "__main__":
    main()
