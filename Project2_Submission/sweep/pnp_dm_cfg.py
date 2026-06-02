"""PnP-DM with a class-conditional DDPM prior and classifier-free guidance.

Each chain is assigned a target class label (0-9). At every denoising
step the prior call is

    eps_guided = (1 + w) * eps(x_t, t, class=c) - w * eps(x_t, t, class=uncond)

with w typically 3-5. With n_chains=32 the labels cycle 0..9..0..9.. so
every class gets at least 3 chains. We sort chains by final misfit at
the end and the winner is the best across all class hypotheses.

This is the third "prior" branch of the sweep. It only works with a
class-conditional checkpoint produced by train_cond_ddpm.py.
"""
import argparse, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import savemat
from diffusers import UNet2DModel, DDPMScheduler

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from schedule import get_schedule
from train_cond_ddpm import N_CLASSES, UNCOND_LABEL, cfg_eps


def diff_one_denoise_cfg(unet, alphas_cumprod, z_diff, sigma_k, class_label, w=3.0):
    """One DDPM denoise step with classifier-free guidance, class-conditional."""
    a_bar = 1.0 / (1.0 + sigma_k ** 2)
    sqrt_a   = torch.tensor(a_bar, dtype=torch.float32).sqrt()
    sqrt_1ma = torch.tensor(1.0 - a_bar, dtype=torch.float32).sqrt()
    t_idx = int(torch.argmin(torch.abs(alphas_cumprod - a_bar)).item())
    x_t = sqrt_a * z_diff.view(1, 1, 28, 28) + sqrt_1ma * torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        t_b = torch.tensor([t_idx]).long()
        eps = cfg_eps(unet, x_t, t_b, class_label, w=w)
    x0_hat = (x_t - sqrt_1ma * eps) / sqrt_a
    return x0_hat.squeeze().clamp(-1.5, 1.5)


def pnp_dm_cfg(setup, unet, alphas_cumprod, schedule_name="geomspace",
               n_chains=32, n_iter=80, sigma_max=1.0, sigma_min=0.05,
               sigma_n=1e-4, gn_inner=2, cfg_w=3.0, seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    sigmas = get_schedule(schedule_name, n_iter=n_iter,
                           sigma_max=sigma_max, sigma_min=sigma_min)
    chains = []
    for c in range(n_chains):
        class_label = c % N_CLASSES
        x = (torch.randn(28, 28) * sigma_max).clamp(-1.5, 1.5)
        log = []
        for k, sigma_k in enumerate(sigmas):
            z = gn_likelihood_prox(setup, x, float(sigma_k),
                                    sigma_n=sigma_n, max_inner=gn_inner)
            x = diff_one_denoise_cfg(unet, alphas_cumprod,
                                      z, float(sigma_k),
                                      class_label, w=cfg_w)
            x_pix_eval = x_diff_to_pix(x).clamp(0.0, 1.0)
            mis = 0.5 * float(((setup.forward(x_pix_eval) - setup.y_obs) ** 2).sum())
            log.append((k, float(sigma_k), mis))
        if verbose:
            print(f"  chain {c:2d}  class={class_label}  "
                  f"final misfit {log[-1][2]:.3e}")
        x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
        chains.append({'x_pix': x_pix, 'log': log,
                       'class_label': class_label,
                       'final_misfit': log[-1][2]})
    chains.sort(key=lambda c: c['final_misfit'])
    return chains


def load_cond_ddpm(cond_dir):
    """Load the class-conditional UNet + scheduler saved by train_cond_ddpm.py."""
    unet = UNet2DModel.from_pretrained(f"{cond_dir}/unet")
    sch  = DDPMScheduler.from_pretrained(f"{cond_dir}/scheduler")
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)
    return unet, sch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond_dir", default="./ddpm_cond_rot15_ema")
    p.add_argument("--schedule", choices=["geomspace", "neg_rho"],
                   default="geomspace")
    p.add_argument("--n_chains", type=int, default=32)
    p.add_argument("--n_iter",   type=int, default=80)
    p.add_argument("--sigma_n",  type=float, default=1e-4)
    p.add_argument("--cfg_w",    type=float, default=3.0)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--out_mat",  required=True)
    args = p.parse_args()

    setup = ERTSetup("../code")
    try:
        unet, sch = load_cond_ddpm(args.cond_dir)
        alphas_cumprod = sch.alphas_cumprod.float()
        # Override setup's prior with the conditional one. We don't call
        # setup.unet here — pnp_dm_cfg calls our own diff_one_denoise_cfg.
        print(f"Loaded conditional DDPM from {args.cond_dir}")

        print(f"\nPnP-DM CFG: schedule={args.schedule}  chains={args.n_chains}  "
              f"iter={args.n_iter}  w={args.cfg_w}")
        t0 = time.time()
        chains = pnp_dm_cfg(setup, unet, alphas_cumprod,
                             schedule_name=args.schedule,
                             n_chains=args.n_chains, n_iter=args.n_iter,
                             sigma_n=args.sigma_n, cfg_w=args.cfg_w,
                             seed=args.seed)
        print(f"\nFinished in {time.time()-t0:.0f}s")
        best = chains[0]
        sigma_answer = 1.0 + best['x_pix']
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      best['x_pix'],
            'final_misfit':  float(best['final_misfit']),
            'class_label':   int(best['class_label']),
            'schedule':      args.schedule,
            'cfg_w':         float(args.cfg_w),
        })
        print(f"Best misfit {best['final_misfit']:.3e} "
              f"(class {best['class_label']}) → saved {args.out_mat}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
