"""PnP-DM v2 — same Split-Gibbs MCMC as code/pnp_dm.py, but the sigma
schedule is selectable (geomspace or DAPS++ negative-rho polynomial).
The negative-rho schedule spends more iterations at LOW noise, which
DAPS++ found beats geomspace on nonlinear inverse problems.

Other knobs are unchanged. The DDPM prior is whatever pipeline you pass
via --ddpm_path (defaults to the fine-tuned rotated MNIST checkpoint).
"""
import argparse, time, warnings, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import savemat

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox, diff_one_denoise
from schedule import get_schedule


def pnp_dm_v2(setup, schedule_name="geomspace",
              n_chains=32, n_iter=80,
              sigma_max=1.0, sigma_min=0.05,
              sigma_n=1e-4, gn_inner=2, seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    sigmas = get_schedule(schedule_name, n_iter=n_iter,
                          sigma_max=sigma_max, sigma_min=sigma_min)
    chains = []
    for c in range(n_chains):
        x = (torch.randn(28, 28) * sigma_max).clamp(-1.5, 1.5)
        log = []
        for k, sigma_k in enumerate(sigmas):
            z = gn_likelihood_prox(setup, x, float(sigma_k),
                                    sigma_n=sigma_n, max_inner=gn_inner)
            x = diff_one_denoise(setup, z, float(sigma_k))
            x_pix_eval = x_diff_to_pix(x).clamp(0.0, 1.0)
            mis = 0.5 * float(((setup.forward(x_pix_eval) - setup.y_obs) ** 2).sum())
            log.append((k, float(sigma_k), mis))
        if verbose:
            print(f"  chain {c:2d}  final misfit {log[-1][2]:.3e}")
        x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
        chains.append({'x_pix': x_pix, 'log': log,
                       'final_misfit': log[-1][2]})
    chains.sort(key=lambda c: c['final_misfit'])
    return chains


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path", default="../code/ddpm_mnist_rot15")
    p.add_argument("--schedule", choices=["geomspace", "neg_rho"],
                   default="geomspace")
    p.add_argument("--n_chains", type=int, default=32)
    p.add_argument("--n_iter",   type=int, default=80)
    p.add_argument("--sigma_n",  type=float, default=1e-4)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out_mat",  required=True,
                   help="where to save the best-chain answer")
    args = p.parse_args()

    setup = ERTSetup("../code")
    try:
        if args.ddpm_path and os.path.isdir(args.ddpm_path):
            from diffusers import DDPMPipeline
            print(f"Loading DDPM from {args.ddpm_path}")
            pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
            setup.unet = pipe.unet.eval()
            setup.scheduler = pipe.scheduler
            setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
            for p_ in setup.unet.parameters(): p_.requires_grad_(False)

        print(f"\nPnP-DM v2: schedule={args.schedule}  chains={args.n_chains}  "
              f"iter={args.n_iter}")
        t0 = time.time()
        chains = pnp_dm_v2(setup, schedule_name=args.schedule,
                           n_chains=args.n_chains, n_iter=args.n_iter,
                           sigma_n=args.sigma_n, seed=args.seed)
        print(f"\nFinished in {time.time()-t0:.0f}s")
        best = chains[0]
        sigma_answer = 1.0 + best['x_pix']
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      best['x_pix'],
            'final_misfit':  float(best['final_misfit']),
            'schedule':      args.schedule,
        })
        print(f"Best misfit {best['final_misfit']:.3e} → saved {args.out_mat}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
