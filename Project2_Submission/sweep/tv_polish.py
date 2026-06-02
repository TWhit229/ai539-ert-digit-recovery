"""TV-regularized Gauss-Newton polish.

The plain GN-CG polish from the main pipeline drives the misfit to the
float64 floor (~1e-13), but in doing so it fits the round-off noise of
the forward simulator and the recovered image picks up per-pixel
speckle. This script adds a smoothed total-variation (TV) penalty to
the polish objective so the polish cannot introduce per-pixel speckle
even when it has freedom to.

Objective:
    min_x  (1/2) || F(sigma_bg + x) - y_obs ||^2
           + tau * sum_{i,j} sqrt( (dx)^2 + (dy)^2 + eps^2 )

where dx, dy are forward differences and tau is the TV weight. The
smoothing term eps keeps the penalty differentiable everywhere.

Algorithm: alternate one GN-CG step on the data term with one
gradient-descent step on the TV term (proximal-gradient style, in the
spirit of FISTA for TV-regularized least squares).
"""
import argparse, time, sys
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from stage_a_daps_warmstart import ERTSetup
from gn_cg_polish import cg_solve


def smoothed_tv_grad(x_np, eps=1e-6):
    """Gradient of sum_{i,j} sqrt(dx^2 + dy^2 + eps^2) with respect to x.
    Standard isotropic TV. Returns a 28x28 array."""
    # Forward differences (with replicate-edge boundary)
    dx = np.diff(x_np, axis=1, append=x_np[:, -1:])
    dy = np.diff(x_np, axis=0, append=x_np[-1:, :])
    mag = np.sqrt(dx ** 2 + dy ** 2 + eps ** 2)
    nx = dx / mag
    ny = dy / mag
    # Backward divergence (matches forward-difference adjoint)
    div_x = nx - np.concatenate([nx[:, :1] * 0, nx[:, :-1]], axis=1)
    div_y = ny - np.concatenate([ny[:1, :] * 0, ny[:-1, :]], axis=0)
    return -(div_x + div_y)


def smoothed_tv_value(x_np, eps=1e-6):
    dx = np.diff(x_np, axis=1, append=x_np[:, -1:])
    dy = np.diff(x_np, axis=0, append=x_np[-1:, :])
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2 + eps ** 2)))


def tv_polish(setup, x_init, K=100, n_cg=60, lam_init=1e-4, lam_min=1e-10,
              tau=1e-5, tv_step=0.01, target_misfit=1e-15, verbose=True,
              log_every=5):
    """GN-CG polish with a TV term added by alternation.

    Each iteration: take one GN-CG step on the data term, then take a
    small gradient step on the TV term. tau controls how much TV is
    applied per iteration relative to data-fit progress.
    """
    x = torch.tensor(x_init.copy(), dtype=torch.float32).flatten()
    x_min = -setup.sigma_bg + 1.0e-6
    lam = lam_init
    log = []

    y0 = setup.forward(x.reshape(28, 28).clamp_min(x_min))
    misfit = 0.5 * float(((y0 - setup.y_obs) ** 2).sum())
    tv_val = smoothed_tv_value(x.reshape(28, 28).numpy())
    log.append((0, misfit, tv_val))
    if verbose:
        print(f"  iter   0  misfit {misfit:.4e}  TV {tv_val:.3f}")

    for it in range(1, K + 1):
        # ---- Gauss-Newton step on the data term ----
        x_img = x.reshape(28, 28).clamp_min(x_min)
        y_pred, J = setup.forward_and_jacobian(x_img)
        r = y_pred - setup.y_obs
        delta = cg_solve(J, r, lam, n_cg=n_cg)
        # Armijo line search
        g = J.T @ r
        decr = float((g @ delta).item())
        if decr >= 0:
            delta = -delta; decr = -decr
        alpha = 1.0
        for _ in range(8):
            x_try = (x + alpha * delta).clamp_min(x_min)
            mis_try = 0.5 * float(((setup.forward(x_try.reshape(28, 28))
                                     - setup.y_obs) ** 2).sum())
            if mis_try <= misfit + 1e-4 * alpha * decr:
                break
            alpha *= 0.5
        x = x_try
        misfit = mis_try
        lam = max(lam * 0.5, lam_min)

        # ---- TV gradient step ----
        x_2d = x.reshape(28, 28).numpy()
        g_tv = smoothed_tv_grad(x_2d)
        x_2d_new = x_2d - tv_step * tau * g_tv
        x = torch.tensor(x_2d_new.astype(np.float32)).flatten().clamp_min(x_min)

        # Re-evaluate misfit after TV step
        misfit = 0.5 * float(((setup.forward(x.reshape(28, 28).clamp_min(x_min))
                                - setup.y_obs) ** 2).sum())
        tv_val = smoothed_tv_value(x.reshape(28, 28).numpy())
        if it % log_every == 0 or it == 1:
            log.append((it, misfit, tv_val))
            if verbose:
                print(f"  iter {it:3d}  misfit {misfit:.4e}  TV {tv_val:.3f}  "
                      f"α={alpha:.2e}  λ={lam:.2e}")
        if misfit < target_misfit:
            break

    x_final = x.reshape(28, 28).clamp_min(x_min).detach().cpu().numpy()
    return x_final, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_mat", required=True)
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--n_cg", type=int, default=60)
    p.add_argument("--tau", type=float, default=1e-5,
                   help="TV weight; larger = smoother but bigger misfit floor")
    p.add_argument("--tv_step", type=float, default=0.01,
                   help="TV gradient step size")
    p.add_argument("--out_mat", default="tv_polished.mat")
    p.add_argument("--fig", default="tv_polish.png")
    args = p.parse_args()

    setup = ERTSetup("../code")
    try:
        d = loadmat(args.init_mat)
        if 'x_answer' in d:
            x_init = d['x_answer'].astype(np.float32)
        elif 'sigma_answer' in d:
            x_init = d['sigma_answer'].astype(np.float32) - 1.0
        else:
            raise ValueError("no x_answer/sigma_answer in input")
        if x_init.ndim != 2:
            x_init = x_init.reshape(28, 28)

        print(f"\nTV polish: tau={args.tau}, K={args.K}")
        t0 = time.time()
        x_final, log = tv_polish(setup, x_init, K=args.K, n_cg=args.n_cg,
                                  tau=args.tau, tv_step=args.tv_step)
        print(f"\nDone in {time.time()-t0:.0f}s. Final misfit {log[-1][1]:.3e}")

        sigma_answer = 1.0 + x_final
        savemat(args.out_mat,
                {'sigma_answer': sigma_answer, 'x_answer': x_final,
                 'final_misfit': float(log[-1][1]),
                 'tau': float(args.tau)})
        print(f"Saved {args.out_mat}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x_init, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("input"); axes[0].axis('off')
        axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].set_title(f"TV polished\nmisfit {log[-1][1]:.2e}\nTV {log[-1][2]:.2f}")
        axes[1].axis('off')
        its, mis, tvs = zip(*log)
        axes[2].plot(its, mis, 'o-', label='misfit')
        axes[2].set_yscale('log'); axes[2].set_xlabel('iter')
        axes[2].set_title('misfit trajectory'); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
