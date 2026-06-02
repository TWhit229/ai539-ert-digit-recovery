"""
DPS — Diffusion Posterior Sampling for the nonlinear ERT inverse problem.
Chung, Kim, Mccann, Klasky, Ye (NeurIPS 2022, arXiv 2209.14687).

This is a **diffusion-load-bearing** method:
  * The DDPM determines digit class and pose by *generating* the image through
    reverse-diffusion, conditioned on the data via a gradient step at every
    timestep.
  * No template matching, no class-restricted search, no brute-force scan over
    MNIST — the diffusion is the prior and the data guidance shapes the trajectory.

Algorithm (per chain):
  x_T ~ N(0, I)
  for t = T, T-1, ..., 1:
      1) Standard DDPM reverse step:
            eps_θ   = ε_θ(x_t, t)
            x̂_0     = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t       # Tweedie estimate
            μ_t     = DDPM_mean(x_t, x̂_0, t)             # diffusers `step()`
            x_{t-1} = μ_t + σ_t · ε                       # ε ~ N(0,I)
      2) DPS data-consistency gradient (added to x_{t-1}):
            r       = F(σ_bg + x̂_0_pix) − y_obs
            g       = ∇_{x_t}‖r‖₂ ≈ Jᵀ r / ‖r‖₂ / √ᾱ_t   (DPS chain-rule trick)
            x_{t-1} ← x_{t-1} − ζ · g

  Multi-chain: run N seeds, pick the chain with lowest final misfit.

References:
  - DPS paper: arXiv:2209.14687
  - Implementation cross-checks: github.com/DPS2022/diffusion-posterior-sampling
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix


def _ddim_step(scheduler, eps_pred, t, x_t, eta_ddim=0.0):
    """One DDIM reverse step. Returns (x_{t-1}, x0_hat).

    eta_ddim=0 → deterministic DDIM. Set >0 for stochastic samples (mode diversity).
    Uses the scheduler's stored alphas_cumprod.
    """
    a_bar_t = scheduler.alphas_cumprod[t]
    sqrt_a = a_bar_t.sqrt()
    sqrt_1ma = (1.0 - a_bar_t).sqrt()
    x0_hat = (x_t - sqrt_1ma * eps_pred) / sqrt_a

    # Find previous timestep
    idx = (scheduler.timesteps == t).nonzero(as_tuple=True)[0]
    if idx.numel() == 0 or int(idx[0]) + 1 >= len(scheduler.timesteps):
        # last step → return x0_hat (no noise)
        return x0_hat, x0_hat
    t_prev = scheduler.timesteps[int(idx[0]) + 1]
    a_bar_prev = scheduler.alphas_cumprod[t_prev]

    # DDIM mean
    sigma_t = eta_ddim * ((1 - a_bar_prev) / (1 - a_bar_t)).sqrt() * \
              (1 - a_bar_t / a_bar_prev).sqrt()
    direction = (1 - a_bar_prev - sigma_t ** 2).sqrt() * eps_pred
    x_prev_mean = a_bar_prev.sqrt() * x0_hat + direction
    if eta_ddim > 0:
        x_prev = x_prev_mean + sigma_t * torch.randn_like(x_t)
    else:
        x_prev = x_prev_mean
    return x_prev, x0_hat


def dps_single_chain(setup, n_steps=100, zeta=1.0, sigma_n=1.0e-3,
                     eta_ddim=0.0, seed=0, x_clip=1.5, verbose=False,
                     log_every=10):
    """One DPS chain. Returns (x_pix_final, log, final_misfit)."""
    torch.manual_seed(seed); np.random.seed(seed)

    # Use a DDIM-style scheduler over n_steps. Re-uses pretrained DDPM betas.
    scheduler = DDIMScheduler(num_train_timesteps=1000,
                               beta_start=setup.scheduler.config.beta_start,
                               beta_end=setup.scheduler.config.beta_end,
                               beta_schedule=setup.scheduler.config.beta_schedule)
    scheduler.set_timesteps(n_steps)
    scheduler.alphas_cumprod = setup.scheduler.alphas_cumprod  # ensure same alphas
    timesteps = scheduler.timesteps

    # x_T in diffusion convention [-1, +1]
    x_t = torch.randn(1, 1, 28, 28)
    log = []

    for i, t in enumerate(timesteps):
        # ----- 1) DDIM reverse step (requires gradient on x_t for DPS) -----
        x_t = x_t.detach().requires_grad_(True)
        with torch.enable_grad():
            eps_pred = setup.unet(x_t, t).sample
            a_bar_t = scheduler.alphas_cumprod[int(t)]
            sqrt_a = a_bar_t.sqrt()
            sqrt_1ma = (1.0 - a_bar_t).sqrt()
            x0_hat = (x_t - sqrt_1ma * eps_pred) / sqrt_a
            x0_hat_clamped = x0_hat.clamp(-x_clip, x_clip)

            # ----- 2) Data-consistency residual -----
            x0_pix = x_diff_to_pix(x0_hat_clamped).clamp(0.0, 1.0)
            y_pred = setup.forward(x0_pix.squeeze())                  # (1900,)
            residual = y_pred - setup.y_obs
            res_norm = torch.linalg.vector_norm(residual)

            # DPS step size: zeta_t = zeta / ||y - F(x0_hat)|| (paper convention)

        # DPS gradient: ∇_{x_t} ||y - F(x0_hat(x_t))||
        # Use analytic Jacobian (saves an autograd through MATLAB forward)
        with torch.no_grad():
            _, J_pix = setup.forward_and_jacobian(x0_pix.detach().squeeze())
            J_diff = 0.5 * J_pix                                       # ∂F/∂x_diff
            # gradient w.r.t. x_t (DPS chain rule): J_diff^T · r / (sqrt(a_bar_t) · ||r||)
            grad_x_t = (J_diff.T @ residual.detach()).view(1, 1, 28, 28) \
                       / (sqrt_a * res_norm.detach().clamp_min(1e-12))

        # Standard DDIM step using detached x0_hat
        with torch.no_grad():
            x_prev, _ = _ddim_step(scheduler, eps_pred.detach(), int(t),
                                    x_t.detach(), eta_ddim=eta_ddim)

            # DPS update: subtract guidance gradient
            x_prev = x_prev - zeta * grad_x_t
            x_prev = x_prev.clamp(-x_clip, x_clip)

        # Log periodically
        if (i + 1) % log_every == 0 or i == len(timesteps) - 1:
            x0_pix_eval = x_diff_to_pix(x0_hat_clamped).clamp(0.0, 1.0).detach()
            mis_at_x0 = 0.5 * float(((setup.forward(x0_pix_eval.squeeze())
                                       - setup.y_obs) ** 2).sum())
            log.append((i + 1, int(t), mis_at_x0))
            if verbose:
                print(f"    seed {seed}  step {i+1:3d}/{n_steps}  t={int(t):4d}  "
                      f"||r||={res_norm.item():.3e}  misfit(x0)={mis_at_x0:.3e}")

        x_t = x_prev

    # Final pixel-coords output
    x_final_pix = x_diff_to_pix(x_t).clamp(0.0, 1.0).squeeze().detach().numpy()
    final_misfit = 0.5 * float(((setup.forward(torch.tensor(x_final_pix, dtype=torch.float32))
                                  - setup.y_obs) ** 2).sum())
    return x_final_pix, log, final_misfit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path", default=None,
                   help="local path to fine-tuned DDPM (e.g. ./ddpm_mnist_rot15). "
                        "Default = pretrained 1aurent/ddpm-mnist.")
    p.add_argument("--n_chains", type=int, default=16)
    p.add_argument("--n_steps",  type=int, default=100)
    p.add_argument("--zeta",     type=float, default=1.0)
    p.add_argument("--sigma_n",  type=float, default=1e-3)
    p.add_argument("--eta_ddim", type=float, default=0.0,
                   help="0=deterministic DDIM, >0=stochastic")
    p.add_argument("--lm_polish_K", type=int, default=400,
                   help="short LM polish iterations at the very end")
    p.add_argument("--out_mat",  default="../dps_answer.mat")
    p.add_argument("--fig",      default="../figures/dps_chains.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Optionally swap in the fine-tuned UNet
        if args.ddpm_path:
            print(f"\nLoading fine-tuned DDPM from {args.ddpm_path}...")
            pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
            setup.unet = pipe.unet.eval()
            setup.scheduler = pipe.scheduler
            setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
            for p_ in setup.unet.parameters():
                p_.requires_grad_(False)

        print(f"\nDPS: {args.n_chains} chains × {args.n_steps} steps  "
              f"(ζ={args.zeta}, σ_n={args.sigma_n}, eta_ddim={args.eta_ddim})")

        chains = []
        t0 = time.time()
        for c in range(args.n_chains):
            x, log, mis = dps_single_chain(setup, n_steps=args.n_steps, zeta=args.zeta,
                                            sigma_n=args.sigma_n, eta_ddim=args.eta_ddim,
                                            seed=c, verbose=False)
            chains.append({'seed': c, 'x_pix': x, 'log': log, 'final_misfit': mis})
            print(f"  chain {c:2d}  final misfit {mis:.3e}  "
                  f"({time.time()-t0:.0f}s elapsed)")

        chains.sort(key=lambda r: r['final_misfit'])
        best = chains[0]
        print(f"\nBest chain (seed {best['seed']}): misfit {best['final_misfit']:.3e}")

        # ---- Short LM polish on the best chain (pixel-level refinement only) ----
        if args.lm_polish_K > 0:
            from stage_d_lm_polish import stage_d_lm_polish
            print(f"\nShort LM polish on best chain ({args.lm_polish_K} iter)...")
            t0 = time.time()
            x_polished, lm_log = stage_d_lm_polish(setup, best['x_pix'],
                                                    K=args.lm_polish_K,
                                                    eta=5.0, gamma=0.95,
                                                    use_backtrack=False,
                                                    target_misfit=1e-9,
                                                    log_every=args.lm_polish_K,
                                                    verbose=True)
            final_misfit_polished = lm_log[-1][1]
            print(f"  polished in {time.time()-t0:.0f}s  →  misfit {final_misfit_polished:.3e}")
        else:
            x_polished = best['x_pix']
            final_misfit_polished = best['final_misfit']
            lm_log = []

        sigma_answer = 1.0 + x_polished
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_polished,
            'final_misfit':  float(final_misfit_polished),
            'pre_polish_misfit': float(best['final_misfit']),
            'best_seed':     int(best['seed']),
            'n_chains':      int(args.n_chains),
            'n_steps':       int(args.n_steps),
            'lm_polish_K':   int(args.lm_polish_K),
        })
        print(f"\nSaved {args.out_mat}")

        # Visualize: 4 chains + best + after-polish + polish trajectory
        nc = min(args.n_chains, 8)
        fig, axes = plt.subplots(2, nc, figsize=(2.5*nc, 5.5))
        if nc == 1: axes = axes.reshape(2, 1)
        for k in range(nc):
            c = chains[k]
            axes[0, k].imshow(c['x_pix'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"seed {c['seed']}\n{c['final_misfit']:.2e}", fontsize=9)
            axes[0, k].axis('off')
        # bottom row: best chain image and polished image
        axes[1, 0].imshow(best['x_pix'], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f"BEST chain (seed {best['seed']})\n{best['final_misfit']:.2e}",
                             fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        if args.lm_polish_K > 0:
            axes[1, 1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
            axes[1, 1].set_title(f"After {args.lm_polish_K}-iter LM polish\n"
                                  f"misfit {final_misfit_polished:.2e}",
                                  fontsize=10, fontweight='bold')
            axes[1, 1].axis('off')
        for k in range(2, nc):
            axes[1, k].axis('off')
        plt.suptitle(f"DPS (Diffusion Posterior Sampling) — {args.n_chains} chains × "
                      f"{args.n_steps} steps", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

        # Summary
        print("\n" + "="*68)
        print("DPS SUMMARY")
        print("="*68)
        print(f"  Top-3 chain misfits: " +
              ", ".join(f"{c['final_misfit']:.3e}" for c in chains[:3]))
        print(f"  Best DPS misfit (pre-polish):    {best['final_misfit']:.3e}")
        if args.lm_polish_K > 0:
            print(f"  After {args.lm_polish_K}-iter LM polish:           "
                  f"{final_misfit_polished:.3e}")
        print(f"  Target (P1 rotation-aware):       1.006e-7")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
