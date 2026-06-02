"""
Stage B of DI-RTG: Gauss-Newton refinement with diffusion prior anchor.

Replaces DAPS's noisy Langevin inner loop with EXACT Newton-style steps using
the analytic Jacobian J(σ). The diffusion prior enters only as a soft anchor
z_t (Tweedie estimate) — not as a stochastic gradient.

At each of K_B decreasing noise levels:
    1. Re-noise current clean estimate x to noise level t:
           x_t = √ᾱ_t · x_diff + √(1−ᾱ_t) · ε,   x_diff = 2·x − 1
    2. Run one diffusion-model evaluation, Tweedie-denoise to get anchor:
           ε̂ = network(x_t, t)
           z_t = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t      (in diff space)
           z_t_pix = (z_t + 1) / 2                (back to pixel space)
    3. Solve the Tikhonov subproblem with N_gn Gauss-Newton iterations:
           min_x  ½‖F(σ_b+x) − y_obs‖²  +  (ρ_t/2)·‖x − z_t_pix‖²
       Normal equations at each GN iter:
           (JᵀJ + ρ_t·I + μ_LM·diag(JᵀJ)) δ = Jᵀr + ρ_t·(x − z_t_pix)
       Line search on full nonlinear loss.

The diffusion prior keeps x near the MNIST manifold; the GN step drives misfit
toward the measurement-noise floor (~1e-6).
"""
import argparse, time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Reuse the setup from Stage A
import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix


def stage_b_gn_refinement(setup, x_init, timesteps, rho_schedule,
                          N_gn=4, mu_lm=1e-3, tweedie_clip=1.5,
                          line_search_min_alpha=1.0/64.0, seed=0, verbose=True):
    """
    Args:
      x_init       : (28, 28) numpy array, pixel-space initialization (e.g. Stage A output)
      timesteps    : list of int diffusion timesteps, DECREASING
      rho_schedule : list of float anchor strengths, same length as timesteps
      N_gn         : Gauss-Newton iterations per level
      mu_lm        : Levenberg-Marquardt damping coefficient on diag(JᵀJ)
    Returns:
      x_final      : (28, 28) numpy array, pixel-space recovered image
      diagnostics  : list of dicts, one per level
    """
    assert len(timesteps) == len(rho_schedule)
    torch.manual_seed(seed); np.random.seed(seed)

    x = torch.tensor(x_init.copy(), dtype=torch.float32)              # (28, 28) pixel
    I784 = torch.eye(784)
    diagnostics = []

    initial_misfit = 0.5 * float(((setup.forward(x) - setup.y_obs) ** 2).sum())
    if verbose:
        print(f"\nStage B start  initial misfit = {initial_misfit:.3e}")

    for level_idx, (t_idx, rho_t) in enumerate(zip(timesteps, rho_schedule)):
        a_bar_t  = setup.alphas_cumprod[t_idx]
        sqrt_a   = a_bar_t.sqrt()
        sqrt_1ma = (1.0 - a_bar_t).sqrt()

        # ---- 1. Re-noise current x to level t (in x_diff space) ----
        x_diff = 2.0 * x - 1.0
        eps = torch.randn_like(x_diff)
        x_t = (sqrt_a * x_diff.view(1, 1, 28, 28)
               + sqrt_1ma * eps.view(1, 1, 28, 28))

        # ---- 2. Tweedie denoising to get anchor z_t_pix ----
        with torch.no_grad():
            eps_pred = setup.unet(x_t, t_idx).sample
        z_t_diff = (x_t.squeeze() - sqrt_1ma * eps_pred.squeeze()) / sqrt_a
        z_t_diff = z_t_diff.clamp(-tweedie_clip, tweedie_clip)
        z_t_pix  = x_diff_to_pix(z_t_diff).clamp(0.0, 1.0)            # (28, 28)

        # ---- 3. Gauss-Newton inner loop on Tikhonov subproblem ----
        for gn_iter in range(N_gn):
            y_pred, J = setup.forward_and_jacobian(x)                  # (1900,), (1900, 784)
            r = y_pred - setup.y_obs                                    # (1900,)

            JTJ = J.T @ J                                              # (784, 784)
            diag_JTJ = torch.diag(JTJ)
            H = JTJ + rho_t * I784 + mu_lm * torch.diag(diag_JTJ)
            x_flat = x.flatten()
            b = J.T @ r + rho_t * (x_flat - z_t_pix.flatten())

            try:
                delta = torch.linalg.solve(H, b)                       # (784,)
            except RuntimeError as e:
                if verbose:
                    print(f"    [level {level_idx} gn {gn_iter}] solve failed: {e}; bumping mu_lm")
                mu_lm *= 10.0
                continue

            # ---- Line search on the FULL Tikhonov objective (data + anchor) ----
            # Project 1 style: gradient descent on data alone is what Stage D will do.
            # Here Stage B's objective IS data + ρ·anchor, so the line search must use
            # the same objective the GN step is descending. Otherwise the anchor pull
            # always looks like "worse data fit" and the step gets rejected.
            def tikhonov_loss(x_test):
                y = setup.forward(x_test)
                data = 0.5 * float(((y - setup.y_obs) ** 2).sum())
                anchor = 0.5 * rho_t * float(((x_test.flatten() - z_t_pix.flatten()) ** 2).sum())
                return data + anchor, data

            orig_loss, orig_data = tikhonov_loss(x)
            alpha = 1.0
            accepted = False
            while alpha >= line_search_min_alpha:
                x_new_flat = (x_flat - alpha * delta).clamp(0.0, 1.0)
                x_new = x_new_flat.reshape(28, 28)
                new_loss, new_data = tikhonov_loss(x_new)
                if new_loss <= orig_loss:
                    x = x_new
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                if verbose:
                    print(f"    [level {level_idx} gn {gn_iter}] line search failed; stop level")
                break

        misfit = 0.5 * float(((setup.forward(x) - setup.y_obs) ** 2).sum())
        diagnostics.append({'level': level_idx, 't': t_idx, 'rho': float(rho_t),
                            'misfit': misfit})
        if verbose:
            print(f"  level {level_idx+1:2d}/{len(timesteps)}  t={t_idx:4d}  "
                  f"ρ={rho_t:.2e}  misfit={misfit:.3e}")

    x_final = x.detach().cpu().numpy()
    return x_final, diagnostics


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="stage_a_best.npz")
    p.add_argument("--K_B", type=int, default=8)
    p.add_argument("--N_gn", type=int, default=4)
    p.add_argument("--mu_lm", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="stage_b_result.npz")
    p.add_argument("--fig", default="../figures/stage_b_refined.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Load Stage A initialization
        init_data = np.load(args.init, allow_pickle=True)
        x_init = init_data['x']
        init_misfit = float(init_data['misfit']) if 'misfit' in init_data.files else None
        print(f"\nStage A init  shape={x_init.shape}  misfit_recorded={init_misfit}")

        # Schedule: K_B decreasing timesteps, ρ increasing
        # Cover the mid-to-low noise regime. Skip very-high-t (Tweedie unreliable)
        # and very-low-t (we already start clean).
        timesteps = np.linspace(500, 5, args.K_B, dtype=int).tolist()
        # ρ grows ~ geometrically from 0.01 to 50
        rho_schedule = np.geomspace(0.01, 50.0, args.K_B).tolist()

        print(f"\nStage B schedule  K_B={args.K_B}  N_gn={args.N_gn}")
        for t, r in zip(timesteps, rho_schedule):
            print(f"  t={t:4d}  ρ={r:.2e}")

        t0 = time.time()
        x_final, diags = stage_b_gn_refinement(setup, x_init, timesteps, rho_schedule,
                                               N_gn=args.N_gn, mu_lm=args.mu_lm, seed=args.seed)
        elapsed = time.time() - t0
        print(f"\nStage B done in {elapsed:.1f}s.")
        final_misfit = diags[-1]['misfit']
        print(f"Final misfit: {final_misfit:.3e}  (target: ≤ 1e-5)")

        np.savez(args.out, x=x_final, misfit=final_misfit,
                 diagnostics=diags, init_misfit=init_misfit)
        print(f"Saved {args.out}")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(x_init, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Stage A init\nmisfit = {init_misfit:.2e}' if init_misfit else 'Stage A init')
        axes[1].imshow(x_final, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Stage B GN-refined\nmisfit = {final_misfit:.2e}')
        for ax in axes[:2]:
            ax.axis('off')
        misfits = [init_misfit] + [d['misfit'] for d in diags] if init_misfit else [d['misfit'] for d in diags]
        axes[2].plot(misfits, 'o-')
        axes[2].set_yscale('log')
        axes[2].set_xlabel('level')
        axes[2].set_ylabel('misfit')
        axes[2].set_title('Misfit per GN-level')
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
