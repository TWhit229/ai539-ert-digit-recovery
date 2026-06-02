"""
PnP-DM — Plug-and-Play Diffusion Model posterior sampler (Wu et al., NeurIPS 2024,
arXiv 2405.18782) for the ERT inverse problem.

Why this approach: agent research surveyed 15+ recent papers; PnP-DM was
flagged by all three reviewers as the strongest principled method for nonlinear
inverse problems with a known forward operator and analytic Jacobian. It runs
Split-Gibbs MCMC alternating

    (LIKELIHOOD)  z_k = prox_f(x_k; σ_k²)         # uses our analytic J
    (PRIOR)       x_{k+1} ~ p_diff(x | z_k, σ_k²) # one DDPM denoising step

where prox_f(x; σ²) ≈ argmin_z  ½‖y − F(z)‖²/σ_n² + ‖z − x‖²/(2σ²). For our
nonlinear F we approximate the prox by ONE Gauss-Newton step at z = x:

    (Jᵀ J / σ_n² + I / σ²) Δ = Jᵀ (y − F(x)) / σ_n²
    z = x + Δ

Multiple chains in parallel give us mode exploration — exactly what fixes
DI-RTG's "wrong-digit-low-misfit" failure mode.

Implementation notes:
  - Diffusion model space: x ∈ [-1, 1] (DDPM convention). We map to pixel
    coords (σ - σ_bg ∈ [0, 1]) for the forward operator.
  - σ_n (measurement noise): estimated from Project 1's floor (~ √(2·1.0e-7)
    on competition data). We use σ_n = 1e-4 as a soft cap.
  - σ_k schedule: log-spaced from σ_max ≈ 1.0 down to σ_min ≈ 0.05.
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix


def gn_likelihood_prox(setup, x_diff, sigma_k, sigma_n=1e-4, max_inner=2):
    """One (or few) Gauss-Newton step on the likelihood prox.

    min_z  (1/(2σ_n²))‖F(σ_bg + (z+1)/2) − y_obs‖²  +  (1/(2σ_k²))‖z − x_diff‖²

    Returns the GN update z (in x_diff coords).
    Uses chain rule: dF/dx_diff = (1/2) dF/dx_pix.
    """
    z = x_diff.detach().clone()
    I = torch.eye(z.numel(), dtype=torch.float32)
    for _ in range(max_inner):
        z_pix = (z.view(28, 28) + 1.0) * 0.5
        z_pix = z_pix.clamp(0.0, 1.0)
        y_pred, J_pix = setup.forward_and_jacobian(z_pix)   # ∂F/∂x_pix
        J_diff = 0.5 * J_pix                                # ∂F/∂x_diff
        r = setup.y_obs - y_pred
        A = J_diff.T @ J_diff / (sigma_n ** 2) + I / (sigma_k ** 2)
        b = J_diff.T @ r / (sigma_n ** 2) + (x_diff.view(-1) - z.view(-1)) / (sigma_k ** 2)
        # solve A Δ = b
        delta = torch.linalg.solve(A, b)
        z = z + delta.view(*z.shape)
        z = z.clamp(-1.5, 1.5)
    return z


def diff_one_denoise(setup, z_diff, sigma_k):
    """One denoise step of the DDPM from noise level sigma_k.

    Equivalent: nearest pretrained t such that sqrt((1-α_t)/α_t) ≈ sigma_k.
    Then run an Euler/DDIM step from t toward t-1 (single step). We use
    Tweedie's formula and a small noise top-up so the chain stays Markovian.
    """
    a_bar = 1.0 / (1.0 + sigma_k ** 2)
    sqrt_a   = torch.tensor(a_bar, dtype=torch.float32).sqrt()
    sqrt_1ma = torch.tensor(1.0 - a_bar, dtype=torch.float32).sqrt()

    # Find nearest stored timestep
    targets = setup.alphas_cumprod  # (1000,)
    t_idx = int(torch.argmin(torch.abs(targets - a_bar)).item())

    # Build x_t at this noise level
    x_t = sqrt_a * z_diff.view(1, 1, 28, 28) + sqrt_1ma * torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        eps = setup.unet(x_t, torch.tensor([t_idx])).sample
    x0_hat = (x_t - sqrt_1ma * eps) / sqrt_a
    x0_hat = x0_hat.squeeze().clamp(-1.5, 1.5)
    return x0_hat


def pnp_dm(setup, n_chains=4, n_iter=40, sigma_max=1.0, sigma_min=0.05,
           sigma_n=1e-4, gn_inner=2, seed=0, verbose=True,
           sigma_n_schedule='constant', sigma_n_c=0.5):
    """Run N independent PnP-DM chains, return all final samples + best by misfit.

    sigma_n_schedule:
      'constant'    — sigma_n stays at the value passed in (original).
      'proportional'— sigma_n_k = sigma_n_c * sigma_k. Annealing recommended by
                      PnP-DM / DAPS literature: tracks the prior's noise level.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    sigmas = np.geomspace(sigma_max, sigma_min, n_iter)

    chains = []
    for c in range(n_chains):
        # init from pure noise at sigma_max
        x = torch.randn(28, 28) * sigma_max
        x = x.clamp(-1.5, 1.5)
        log = []
        for k, sigma_k in enumerate(sigmas):
            # Sigma_n schedule: constant or proportional to sigma_k
            sigma_n_k = (sigma_n_c * float(sigma_k)
                         if sigma_n_schedule == 'proportional' else sigma_n)
            # Likelihood prox via GN
            z = gn_likelihood_prox(setup, x, sigma_k, sigma_n=sigma_n_k, max_inner=gn_inner)
            # Prior step via DDPM denoise
            x = diff_one_denoise(setup, z, sigma_k)

            # Diagnostic misfit at current iterate (in pixel coords)
            x_pix_eval = x_diff_to_pix(x).clamp(0.0, 1.0)
            mis = 0.5 * float(((setup.forward(x_pix_eval) - setup.y_obs) ** 2).sum())
            log.append((k, float(sigma_k), mis))
            if verbose and (k + 1) % 5 == 0:
                print(f"  chain {c}  it {k+1:2d}/{n_iter}  σ={sigma_k:.3f}  misfit {mis:.3e}")

        # Convert to pixel coords for return
        x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
        chains.append({'x_pix': x_pix, 'log': log, 'final_misfit': log[-1][2]})

    chains.sort(key=lambda c: c['final_misfit'])
    return chains


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path", default=None,
                   help="local fine-tuned DDPM dir (default: pretrained)")
    p.add_argument("--n_chains",  type=int,   default=4)
    p.add_argument("--n_iter",    type=int,   default=40)
    p.add_argument("--sigma_n",   type=float, default=1e-4)
    p.add_argument("--sigma_n_schedule", choices=['constant', 'proportional'],
                   default='constant',
                   help="constant=use --sigma_n; proportional=sigma_n_k = c·sigma_k")
    p.add_argument("--sigma_n_c", type=float, default=0.5)
    p.add_argument("--gn_inner",  type=int,   default=2)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--lm_polish_K", type=int, default=0,
                   help="optional LM polish iters on best chain")
    p.add_argument("--out_mat",   default="../pnp_dm_answer.mat")
    p.add_argument("--out_npz",   default="pnp_dm_result.npz")
    p.add_argument("--fig",       default="../figures/pnp_dm_chains.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        if args.ddpm_path:
            from diffusers import DDPMPipeline
            print(f"\nLoading fine-tuned DDPM from {args.ddpm_path}...")
            pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
            setup.unet = pipe.unet.eval()
            setup.scheduler = pipe.scheduler
            setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
            for p_ in setup.unet.parameters():
                p_.requires_grad_(False)

        print(f"\nPnP-DM: {args.n_chains} chains × {args.n_iter} iters")
        t0 = time.time()
        chains = pnp_dm(setup, n_chains=args.n_chains, n_iter=args.n_iter,
                        sigma_n=args.sigma_n, gn_inner=args.gn_inner,
                        sigma_n_schedule=args.sigma_n_schedule,
                        sigma_n_c=args.sigma_n_c,
                        seed=args.seed)
        print(f"\nFinished in {time.time()-t0:.0f}s")
        for k, c in enumerate(chains):
            print(f"  chain {k}  final misfit {c['final_misfit']:.3e}")

        best = chains[0]
        pre_polish_misfit = best['final_misfit']
        x_final_pix = best['x_pix']

        if args.lm_polish_K > 0:
            from stage_d_lm_polish import stage_d_lm_polish
            print(f"\nLM polish on best chain ({args.lm_polish_K} iter)...")
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
        from scipy.io import savemat
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_final_pix,
            'final_misfit':  float(final_misfit),
            'pre_polish_misfit': float(pre_polish_misfit),
            'n_chains':      int(args.n_chains),
            'n_iter':        int(args.n_iter),
        })
        print(f"\nSaved {args.out_mat}")

        np.savez(args.out_npz,
                 x_pix=x_final_pix,
                 final_misfit=final_misfit,
                 log=np.array(best['log']))
        print(f"Best chain pre-polish misfit: {pre_polish_misfit:.3e}")
        print(f"Final misfit after polish:    {final_misfit:.3e}")

        # 2 × n_chains plot: image and trajectory per chain
        nc = len(chains)
        fig, axes = plt.subplots(2, nc, figsize=(3.5*nc, 7))
        if nc == 1:
            axes = axes.reshape(2, 1)
        for k, c in enumerate(chains):
            axes[0, k].imshow(c['x_pix'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f'chain {k}\nmisfit {c["final_misfit"]:.2e}', fontsize=11)
            axes[0, k].axis('off')
            its, sigs, miss = zip(*c['log'])
            axes[1, k].plot(its, miss, 'o-', markersize=3)
            axes[1, k].set_yscale('log')
            axes[1, k].set_xlabel('iter'); axes[1, k].set_ylabel('misfit')
            axes[1, k].grid(alpha=0.3)
            axes[1, k].set_title(f'σ {sigs[0]:.2f}→{sigs[-1]:.2f}', fontsize=10)
        plt.suptitle(f'PnP-DM (Split-Gibbs MCMC with GN likelihood prox)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
