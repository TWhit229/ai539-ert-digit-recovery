"""
Ensemble-then-polish: average the top-K PnP-DM chains by ERT misfit, then
hand the averaged image to GN-CG polish. The hypothesis is that stochastic
noise across independent chains is uncorrelated and washes out in the mean,
leaving cleaner edge structure that GN-CG can converge from faster.

Implementation:
  1. Re-run PnP-DM and KEEP ALL chains (not just best).
  2. Sort by ERT misfit; average the top-K (weighted by 1/misfit or equal).
  3. Compute misfit of averaged image.
  4. Run GN-CG polish.

Usage:
    python3 ensemble_polish.py --ddpm_path ./ddpm_mnist_rot15 --n_chains 32 \
                               --n_iter 80 --top_k 5 --K 300 --n_cg 100
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy.io import savemat
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import pnp_dm
from gn_cg_polish import gn_cg_polish


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path",   default="./ddpm_mnist_rot15")
    p.add_argument("--n_chains",    type=int, default=32)
    p.add_argument("--n_iter",      type=int, default=80)
    p.add_argument("--sigma_n",     type=float, default=1e-4)
    p.add_argument("--top_k",       type=int, default=5)
    p.add_argument("--weighting",   choices=['equal', 'misfit_inv'], default='equal')
    p.add_argument("--K",           type=int, default=300)
    p.add_argument("--n_cg",        type=int, default=100)
    p.add_argument("--out_mat",     default="../ensemble_polish_answer.mat")
    p.add_argument("--fig",         default="../figures/ensemble_polish.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Swap in fine-tuned DDPM
        print(f"Loading fine-tuned DDPM from {args.ddpm_path}...")
        pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
        setup.unet = pipe.unet.eval()
        setup.scheduler = pipe.scheduler
        setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
        for p_ in setup.unet.parameters():
            p_.requires_grad_(False)

        # Run PnP-DM (returns chains sorted by final_misfit)
        print(f"\nPnP-DM: {args.n_chains} chains × {args.n_iter} iters")
        t0 = time.time()
        chains = pnp_dm(setup, n_chains=args.n_chains, n_iter=args.n_iter,
                         sigma_n=args.sigma_n, gn_inner=2, seed=0)
        print(f"  done in {time.time()-t0:.0f}s")
        for k, c in enumerate(chains[:args.top_k+2]):
            print(f"  chain {k}  misfit {c['final_misfit']:.3e}")

        # Ensemble: average top-K
        top_chains = chains[:args.top_k]
        if args.weighting == 'equal':
            weights = np.ones(args.top_k) / args.top_k
        else:
            inv_m = np.array([1.0 / c['final_misfit'] for c in top_chains])
            weights = inv_m / inv_m.sum()
        x_avg = np.zeros((28, 28), dtype=np.float32)
        for w, c in zip(weights, top_chains):
            x_avg += w * c['x_pix']
        x_avg = np.clip(x_avg, 0.0, 1.0)
        m_avg = 0.5 * float(((setup.forward(torch.tensor(x_avg, dtype=torch.float32))
                              - setup.y_obs) ** 2).sum())
        print(f"\nAveraged top-{args.top_k} ({args.weighting}):  misfit  {m_avg:.3e}")
        print(f"  (vs best single chain: {chains[0]['final_misfit']:.3e})")

        # GN-CG polish on averaged image
        print(f"\nGN-CG polish on averaged image ({args.K} outer × {args.n_cg} CG)...")
        t0 = time.time()
        x_final, log = gn_cg_polish(setup, x_avg, K=args.K, n_cg=args.n_cg,
                                     lam_init=1e-5, lam_min=1e-12, verbose=True,
                                     log_every=20)
        final_misfit = log[-1][1]
        print(f"  GN-CG done in {time.time()-t0:.0f}s → misfit {final_misfit:.3e}")

        sigma_answer = 1.0 + x_final
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_final,
            'final_misfit':  float(final_misfit),
            'ensemble_misfit': float(m_avg),
            'best_chain_misfit': float(chains[0]['final_misfit']),
            'top_k': int(args.top_k),
        })
        print(f"\nSaved {args.out_mat}")

        # Plot: top-5 chains, averaged, polished
        fig, axes = plt.subplots(2, args.top_k, figsize=(2.6*args.top_k, 5.5))
        if args.top_k == 1:
            axes = axes.reshape(2, 1)
        for k in range(args.top_k):
            axes[0, k].imshow(top_chains[k]['x_pix'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"chain {k}  {top_chains[k]['final_misfit']:.2e}", fontsize=10)
            axes[0, k].axis('off')
        axes[1, 0].imshow(x_avg, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f"avg(top-{args.top_k})\n{m_avg:.2e}", fontsize=10)
        axes[1, 0].axis('off')
        axes[1, 1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1, 1].set_title(f"GN-CG polished\n{final_misfit:.2e}", fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')
        for k in range(2, args.top_k):
            axes[1, k].axis('off')
        plt.suptitle(f"Ensemble + GN-CG (avg top-{args.top_k} of {args.n_chains} chains)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
