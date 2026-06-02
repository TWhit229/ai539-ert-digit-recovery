"""
TDS-PnP-DM — Twisted Diffusion Sampler variant of PnP-DM Split-Gibbs MCMC.

Standard PnP-DM (`pnp_dm.py`) runs N independent chains and at the very end picks
the lowest-misfit chain. That wastes a lot of compute on chains that drift into
wrong-digit basins early.

This file extends PnP-DM with **TDS-style particle resampling** (Wu et al.,
ICLR 2024, arXiv 2306.17775): every R Gibbs iterations, weight each particle by
its current likelihood and resample (with replacement). Chains in wrong basins
are killed; chains in right basins proliferate. The compute is spent on
promising trajectories.

Algorithm (N particles, K iterations, resample every R steps):
  Init N particles from pure noise
  For k = 0..K-1:
      For each particle:
          z = GN_prox(x, sigma_k)
          x = DDPM_denoise(z, sigma_k)
      If k mod R == 0 and k > 0:
          Compute misfits m_i for each particle
          Compute weights w_i ∝ exp(-α (m_i - min m))
          Multinomial resample with weights w
  Pick top-K_keep particles by final misfit → LM polish each → keep best

For mode collapse prevention, we use systematic resampling and add a small
amount of seed-varying noise after each resample.
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox, diff_one_denoise


def systematic_resample(weights):
    """Systematic resampling — lower variance than multinomial. Returns indices."""
    N = len(weights)
    weights = weights / weights.sum()
    cum = np.cumsum(weights)
    u = (np.arange(N) + np.random.uniform(0, 1)) / N
    indices = np.zeros(N, dtype=int)
    i = 0
    for j in range(N):
        while u[j] > cum[i] and i < N - 1:
            i += 1
        indices[j] = i
    return indices


def tds_pnp_dm(setup, n_particles=32, n_iter=50, resample_every=10,
                sigma_max=1.0, sigma_min=0.05, sigma_n=1e-4, gn_inner=2,
                resample_alpha=1.0, perturb_after_resample=0.02, seed=0,
                verbose=True):
    """N parallel particles with periodic resampling by likelihood weight."""
    torch.manual_seed(seed); np.random.seed(seed)
    sigmas = np.geomspace(sigma_max, sigma_min, n_iter)

    # Initialize N particles
    particles = []
    for p in range(n_particles):
        x = (torch.randn(28, 28) * sigma_max).clamp(-1.5, 1.5)
        particles.append({'x': x, 'history': []})

    resample_events = []

    for k, sigma_k in enumerate(sigmas):
        # ----- One Gibbs step per particle -----
        for p in range(n_particles):
            x = particles[p]['x']
            z = gn_likelihood_prox(setup, x, sigma_k, sigma_n=sigma_n, max_inner=gn_inner)
            x_new = diff_one_denoise(setup, z, sigma_k)
            particles[p]['x'] = x_new

        # ----- Compute particle misfits + log -----
        misfits = np.zeros(n_particles)
        for p in range(n_particles):
            x_pix = x_diff_to_pix(particles[p]['x']).clamp(0.0, 1.0)
            m = 0.5 * float(((setup.forward(x_pix) - setup.y_obs) ** 2).sum())
            misfits[p] = m
            particles[p]['history'].append((k, sigma_k, m))

        if verbose:
            print(f"  iter {k+1:3d}/{n_iter}  σ={sigma_k:.3f}  "
                  f"best={misfits.min():.3e}  median={np.median(misfits):.3e}  "
                  f"worst={misfits.max():.3e}")

        # ----- Resample every R iters (after warmup) -----
        if (k + 1) % resample_every == 0 and k < n_iter - resample_every:
            m_min = misfits.min()
            # Stable softmax weighting
            log_w = -resample_alpha * (misfits - m_min) / (sigma_n ** 2)
            log_w = log_w - log_w.max()
            w = np.exp(log_w)
            indices = systematic_resample(w)
            # Resample particles, perturb tied ones to break degeneracy
            new_particles = []
            counts = {}
            for idx in indices:
                dup_count = counts.get(idx, 0)
                counts[idx] = dup_count + 1
                x_clone = particles[idx]['x'].clone()
                if dup_count > 0:
                    # Slightly perturb to break degeneracy on duplicates
                    x_clone = x_clone + torch.randn_like(x_clone) * perturb_after_resample
                    x_clone = x_clone.clamp(-1.5, 1.5)
                new_particles.append({
                    'x':       x_clone,
                    'history': list(particles[idx]['history']),
                })
            particles = new_particles
            unique_kept = len(set(indices.tolist()))
            resample_events.append({
                'iter':       k + 1,
                'sigma':      sigma_k,
                'best_mis':   m_min,
                'unique':     unique_kept,
                'ess':        float(1.0 / (w * w).sum() if w.sum() > 0 else 0.0),
            })
            if verbose:
                print(f"    [resample] kept {unique_kept}/{n_particles} unique; "
                      f"ESS={resample_events[-1]['ess']:.1f}")

    # Compute final misfits and return all particles sorted by misfit
    out = []
    for p in particles:
        x_pix = x_diff_to_pix(p['x']).clamp(0.0, 1.0).numpy().astype(np.float32)
        m = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                          - setup.y_obs) ** 2).sum())
        out.append({'x_pix': x_pix, 'final_misfit': m, 'history': p['history']})
    out.sort(key=lambda r: r['final_misfit'])
    return out, resample_events


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path",   default=None)
    p.add_argument("--n_particles", type=int,   default=32)
    p.add_argument("--n_iter",      type=int,   default=60)
    p.add_argument("--resample_every", type=int, default=10)
    p.add_argument("--resample_alpha", type=float, default=1.0)
    p.add_argument("--sigma_n",     type=float, default=1e-4)
    p.add_argument("--gn_inner",    type=int,   default=2)
    p.add_argument("--lm_polish_K", type=int,   default=1500)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--out_mat",     default="../tds_pnp_dm_answer.mat")
    p.add_argument("--fig",         default="../figures/tds_pnp_dm.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        if args.ddpm_path:
            print(f"\nLoading fine-tuned DDPM from {args.ddpm_path}...")
            pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
            setup.unet = pipe.unet.eval()
            setup.scheduler = pipe.scheduler
            setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
            for p_ in setup.unet.parameters():
                p_.requires_grad_(False)

        print(f"\nTDS-PnP-DM: {args.n_particles} particles × {args.n_iter} iters, "
              f"resample every {args.resample_every}")
        t0 = time.time()
        particles, resample_events = tds_pnp_dm(
            setup, n_particles=args.n_particles, n_iter=args.n_iter,
            resample_every=args.resample_every, resample_alpha=args.resample_alpha,
            sigma_n=args.sigma_n, gn_inner=args.gn_inner, seed=args.seed, verbose=True)
        print(f"\nFinished in {time.time()-t0:.0f}s")
        print(f"Top-5 particle misfits: " +
              ", ".join(f"{p['final_misfit']:.3e}" for p in particles[:5]))

        best = particles[0]
        pre_polish_misfit = best['final_misfit']
        x_final_pix = best['x_pix']

        if args.lm_polish_K > 0:
            from stage_d_lm_polish import stage_d_lm_polish
            print(f"\nLM polish on best particle ({args.lm_polish_K} iter)...")
            t0 = time.time()
            x_final_pix, lm_log = stage_d_lm_polish(setup, best['x_pix'],
                                                     K=args.lm_polish_K,
                                                     eta=5.0, gamma=0.95,
                                                     use_backtrack=False,
                                                     target_misfit=1e-10,
                                                     log_every=args.lm_polish_K,
                                                     verbose=True)
            final_misfit = lm_log[-1][1]
            print(f"  polished in {time.time()-t0:.0f}s → misfit {final_misfit:.3e}")
        else:
            final_misfit = pre_polish_misfit

        sigma_answer = 1.0 + x_final_pix
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_final_pix,
            'final_misfit':  float(final_misfit),
            'pre_polish_misfit': float(pre_polish_misfit),
            'n_particles':   int(args.n_particles),
            'n_iter':        int(args.n_iter),
            'resample_events': len(resample_events),
        })
        print(f"\nSaved {args.out_mat}")

        # Plot top-8 particles + best polished
        nk = min(8, len(particles))
        fig, axes = plt.subplots(2, nk, figsize=(2.5*nk, 5.5))
        for k in range(nk):
            axes[0, k].imshow(particles[k]['x_pix'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"{particles[k]['final_misfit']:.2e}", fontsize=9)
            axes[0, k].axis('off')
        axes[1, 0].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1, 0].set_title(f"Best polished\n{final_misfit:.2e}",
                             fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        for k in range(1, nk):
            axes[1, k].axis('off')
        plt.suptitle(f"TDS-PnP-DM ({args.n_particles} particles, resample every "
                      f"{args.resample_every})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
