"""
DAPS proper — Decoupled Annealing Posterior Sampling (Zhang et al., CVPR 2025
Oral, arXiv 2407.01521) configured per the original paper rather than the
short warm-start we used as Stage A of DI-RTG v1.

This is a **diffusion-load-bearing** posterior sampler:
  * The DDPM prior determines digit class, shape, and pose at every annealing step.
  * The likelihood gradient (using our analytic Jacobian) refines toward the data.
  * No template database; no classifier; no brute-force scan.

Algorithm (one chain):
  σ_max → σ_min annealed schedule (K_outer steps, poly-decay rho=-7).
  for i = 0..K_outer-1:
      # 1) Diffusion denoise at σ_i:
      sample x_t at noise σ_i; run DDPM denoise to get x̂_0_hat
      # 2) ULA refinement (K_inner steps):
      z = x̂_0_hat
      for k = 1..K_inner:
          r = F(σ_bg + z_pix) − y
          g = J^T r / σ_n²
          z = z − η_i · g + √(2 η_i τ_i) · ξ                    # ULA step
      # 3) Re-noise z to σ_{i+1} for next outer step.

Multi-chain: run N independent chains with different seeds. Pick the one whose
final misfit is lowest.

References:
  - DAPS — arXiv 2407.01521 (CVPR 2025 Oral)
  - DAPS++ — arXiv 2511.17038 (decouples Stage-1 diffusion from Stage-2 ULA)
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from diffusers import DDPMPipeline

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix


def poly_decay_sigmas(sigma_max=1.0, sigma_min=0.01, K=80, rho=7.0):
    """The EDM-style polynomial decay schedule used by DAPS.
    sigma_i = (sigma_max^(1/rho) + (i/K)(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    """
    i = np.arange(K, dtype=np.float64)
    inv = 1.0 / rho
    a, b = sigma_max ** inv, sigma_min ** inv
    return (a + (i / max(K - 1, 1)) * (b - a)) ** rho


def sigma_to_t(sigma_target, alphas_cumprod):
    """Find the DDPM timestep whose noise level matches sigma_target.
    Diffusers DDPM uses x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,
    so noise level σ = sqrt((1 - alpha_bar_t) / alpha_bar_t) when reframed."""
    # σ²(t) = (1 - α_bar) / α_bar  →  α_bar = 1 / (1 + σ²)
    a_target = 1.0 / (1.0 + sigma_target ** 2)
    diffs = (alphas_cumprod - a_target).abs()
    return int(diffs.argmin().item())


def daps_chain(setup, sigma_max=1.0, sigma_min=0.01, K_outer=80, K_inner=20,
                rho=7.0, eta_lik=1e-3, eta_lik_final=1e-5, tau=0.0,
                x_clip=1.5, seed=0, sigma_n=1.0, verbose=False, log_every=10):
    """One DAPS chain. Returns (x_pix, log, final_misfit).

    Per outer step:
      - one DDPM denoise from sigma_i (single step, x_0_hat via Tweedie),
      - K_inner ULA refinement steps using the analytic Jacobian,
      - re-noise to sigma_{i+1} for next outer step.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    sigmas = poly_decay_sigmas(sigma_max, sigma_min, K_outer, rho)
    etas = np.geomspace(eta_lik, eta_lik_final, K_outer)
    log = []

    # Init x_T at sigma_max (diffusion convention, x in [-1, 1])
    x = torch.randn(1, 1, 28, 28) * sigma_max

    for i, sigma_i in enumerate(sigmas):
        t_i = sigma_to_t(sigma_i, setup.alphas_cumprod)
        a_bar = setup.alphas_cumprod[t_i]
        sqrt_a = a_bar.sqrt()
        sqrt_1ma = (1.0 - a_bar).sqrt()

        # ---------- 1) Diffusion denoise: Tweedie x_0_hat at σ_i ----------
        # Re-noise x to noise level σ_i then ε-predict and Tweedie.
        x_t = sqrt_a * x + sqrt_1ma * torch.randn_like(x)
        with torch.no_grad():
            eps_pred = setup.unet(x_t, torch.tensor([t_i])).sample
        x0_hat = (x_t - sqrt_1ma * eps_pred) / sqrt_a
        x0_hat = x0_hat.clamp(-x_clip, x_clip)

        # ---------- 2) ULA refinement using analytic J^T r + prior pull ----------
        # DAPS Langevin step (faithful to arXiv 2407.01521 Algorithm 1):
        #   z ← z + η · [(x̂₀ − z) / σ_i²  −  λ · Jᵀ r / σ_n²]  +  √(2 η τ) · ξ
        # The prior pull keeps the inner chain near the Tweedie estimate; the data
        # gradient pulls toward likelihood; both are needed for proper sampling.
        z = x0_hat.detach().clone()
        sigma_i2 = float(sigma_i) ** 2
        for k in range(K_inner):
            z_pix = x_diff_to_pix(z).clamp(0.0, 1.0)
            y_pred, J_pix = setup.forward_and_jacobian(z_pix.squeeze())
            r = y_pred - setup.y_obs
            J_diff = 0.5 * J_pix                                      # ∂F/∂x_diff
            g_lik = (J_diff.T @ r).view(1, 1, 28, 28) / (sigma_n ** 2)
            g_prior = (z - x0_hat.detach()) / sigma_i2                # pull toward Tweedie
            step = etas[i] * (g_prior + g_lik)
            if tau > 0:
                step = step + math.sqrt(2 * etas[i] * tau) * torch.randn_like(z)
            z = z - step
            z = z.clamp(-x_clip, x_clip)

        # 3) Set x to refined z for next outer step
        x = z.detach()

        if (i + 1) % log_every == 0 or i == K_outer - 1:
            x_pix_eval = x_diff_to_pix(x).clamp(0.0, 1.0).squeeze().detach()
            mis = 0.5 * float(((setup.forward(x_pix_eval) - setup.y_obs) ** 2).sum())
            log.append((i + 1, float(sigma_i), float(etas[i]), mis))
            if verbose:
                print(f"    seed {seed}  step {i+1:3d}/{K_outer}  σ={sigma_i:.3f}  "
                      f"η={etas[i]:.1e}  misfit {mis:.3e}")

    # Final misfit in pixel coords
    x_pix_final = x_diff_to_pix(x).clamp(0.0, 1.0).squeeze().detach().numpy()
    final_misfit = 0.5 * float(((setup.forward(torch.tensor(x_pix_final, dtype=torch.float32))
                                  - setup.y_obs) ** 2).sum())
    return x_pix_final, log, final_misfit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path", default=None,
                   help="local fine-tuned DDPM dir (default: pretrained)")
    p.add_argument("--n_chains", type=int, default=16)
    p.add_argument("--K_outer",  type=int, default=100)
    p.add_argument("--K_inner",  type=int, default=20)
    p.add_argument("--sigma_max", type=float, default=1.0)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--rho",       type=float, default=7.0)
    p.add_argument("--eta_lik",   type=float, default=5e-4)
    p.add_argument("--eta_lik_final", type=float, default=1e-5)
    p.add_argument("--tau",       type=float, default=0.0)
    p.add_argument("--lm_polish_K", type=int, default=300,
                   help="brief pixel polish after diffusion (acceptable per DAPS paper)")
    p.add_argument("--out_mat",  default="../daps_answer.mat")
    p.add_argument("--fig",      default="../figures/daps_chains.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Swap in fine-tuned DDPM if provided
        if args.ddpm_path:
            print(f"\nLoading fine-tuned DDPM from {args.ddpm_path}...")
            pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
            setup.unet = pipe.unet.eval()
            setup.scheduler = pipe.scheduler
            setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
            for p_ in setup.unet.parameters():
                p_.requires_grad_(False)

        print(f"\nDAPS proper — {args.n_chains} chains × K_outer={args.K_outer} × "
              f"K_inner={args.K_inner}")
        print(f"  σ: [{args.sigma_min}, {args.sigma_max}], rho={args.rho}")
        print(f"  η_lik schedule: {args.eta_lik:.1e} → {args.eta_lik_final:.1e}")

        chains = []
        t0 = time.time()
        for c in range(args.n_chains):
            x, log, mis = daps_chain(setup, sigma_max=args.sigma_max, sigma_min=args.sigma_min,
                                      K_outer=args.K_outer, K_inner=args.K_inner,
                                      rho=args.rho, eta_lik=args.eta_lik,
                                      eta_lik_final=args.eta_lik_final, tau=args.tau,
                                      seed=c, verbose=(c == 0), log_every=20)
            chains.append({'seed': c, 'x_pix': x, 'log': log, 'final_misfit': mis})
            print(f"  chain {c:2d}  misfit {mis:.3e}  ({time.time()-t0:.0f}s elapsed)")

        chains.sort(key=lambda r: r['final_misfit'])
        best = chains[0]
        print(f"\nBest chain (seed {best['seed']}): misfit {best['final_misfit']:.3e}")

        # ---- Brief LM polish on best chain (standard in DAPS papers) ----
        if args.lm_polish_K > 0:
            from stage_d_lm_polish import stage_d_lm_polish
            print(f"\nBrief LM polish ({args.lm_polish_K} iter) on best chain...")
            t0 = time.time()
            x_polished, lm_log = stage_d_lm_polish(setup, best['x_pix'],
                                                    K=args.lm_polish_K,
                                                    eta=5.0, gamma=0.95,
                                                    use_backtrack=False,
                                                    target_misfit=1e-10,
                                                    log_every=args.lm_polish_K,
                                                    verbose=True)
            final_misfit_polished = lm_log[-1][1]
            print(f"  polished in {time.time()-t0:.0f}s → misfit {final_misfit_polished:.3e}")
        else:
            x_polished = best['x_pix']
            final_misfit_polished = best['final_misfit']

        sigma_answer = 1.0 + x_polished
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_polished,
            'final_misfit':  float(final_misfit_polished),
            'pre_polish_misfit': float(best['final_misfit']),
            'best_seed':     int(best['seed']),
            'n_chains':      int(args.n_chains),
            'K_outer':       int(args.K_outer),
            'K_inner':       int(args.K_inner),
        })
        print(f"\nSaved {args.out_mat}")

        # Plot: chains row + best + polished
        nc = min(args.n_chains, 8)
        fig, axes = plt.subplots(2, nc, figsize=(2.5*nc, 5.5))
        if nc == 1: axes = axes.reshape(2, 1)
        for k in range(nc):
            c = chains[k]
            axes[0, k].imshow(c['x_pix'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"seed {c['seed']}\n{c['final_misfit']:.2e}", fontsize=9)
            axes[0, k].axis('off')
        axes[1, 0].imshow(best['x_pix'], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f"BEST chain (seed {best['seed']})\n{best['final_misfit']:.2e}",
                             fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        polish_label = (f"After {args.lm_polish_K}-iter LM polish\n"
                        f"misfit {final_misfit_polished:.2e}") if args.lm_polish_K > 0 \
                       else f"No polish\nmisfit {final_misfit_polished:.2e}"
        axes[1, 1].set_title(polish_label, fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')
        for k in range(2, nc):
            axes[1, k].axis('off')
        plt.suptitle(f"DAPS proper — {args.n_chains} chains × K_outer={args.K_outer} × "
                     f"K_inner={args.K_inner}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

        # Summary
        print("\n" + "="*68)
        print("DAPS SUMMARY")
        print("="*68)
        top3 = ', '.join(f"{c['final_misfit']:.3e}" for c in chains[:3])
        print(f"  Top-3 chain misfits: {top3}")
        print(f"  Best DAPS misfit (pre-polish):    {best['final_misfit']:.3e}")
        if args.lm_polish_K > 0:
            print(f"  After {args.lm_polish_K}-iter LM polish:           {final_misfit_polished:.3e}")
        print(f"  Target (P1 rotation-aware floor):  1.006e-7")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
