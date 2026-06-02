"""
Gauss-Newton with Conjugate-Gradient inner solve — replaces the heavy-ball
gradient-descent polish with a Newton-style optimizer that gets quadratic
convergence near the optimum.

Per outer step:
    J = ∂F/∂x   (the analytic ERT Jacobian, 1900 × 784)
    r = F(x) − y_obs
    Solve  (JᵀJ + λI) Δ = -Jᵀ r  via truncated CG  (≤ n_cg inner iters)
    α = Wolfe line search for sufficient decrease
    x ← x + α · Δ
    Update damping λ (trust-region style: shrink on success, grow on failure)

Why this beats gradient descent for the final polish:
  - GD with momentum has LINEAR convergence rate near the minimum.
  - GN-CG has QUADRATIC convergence rate (Newton-like).
  - With analytic J and ~784 unknowns the normal-equations system is
    well-conditioned at convergence and CG is cheap (~50 iters per outer step).
  - For our problem, expect to reach machine-precision misfit (~1e-12) in 30-50
    outer iterations vs 1500 iters of GD polish.
"""
import argparse, time
import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup


def cg_solve(J, r, lam, n_cg=60, tol=1e-12, verbose=False):
    """Solve (JᵀJ + λI) Δ = -Jᵀ r via truncated CG. All inputs torch.

    Implementation note: we work in torch on CPU (matches our setting). For
    1900x784 J, J.T @ J is 784x784 and could be formed explicitly, but using
    matvec products is cheaper memory-wise and equally fast for n_cg ~ 50.
    """
    rhs = -(J.T @ r)
    delta = torch.zeros_like(rhs)
    rcg = rhs - (J.T @ (J @ delta) + lam * delta)   # initial residual (= rhs)
    p = rcg.clone()
    rs_old = float((rcg @ rcg).item())
    for k in range(n_cg):
        Ap = J.T @ (J @ p) + lam * p
        alpha = rs_old / float((p @ Ap).item() + 1e-30)
        delta = delta + alpha * p
        rcg = rcg - alpha * Ap
        rs_new = float((rcg @ rcg).item())
        if verbose and (k + 1) % 10 == 0:
            print(f"    CG iter {k+1}  ||r||²={rs_new:.3e}")
        if rs_new < tol * (rs_old + 1e-30):
            break
        p = rcg + (rs_new / rs_old) * p
        rs_old = rs_new
    return delta


def gn_cg_polish(setup, x_init, K=50, lam_init=1e-4, lam_min=1e-10, lam_max=1e2,
                  n_cg=60, alpha_max=1.0, c1=1e-4, ls_shrink=0.5, ls_max=10,
                  target_misfit=1e-12, log_every=2, verbose=True):
    """
    Args:
      x_init    : (28,28) numpy initial image (pixel coords ∈ [0, 1] approx).
      K         : maximum outer GN iterations.
      lam_init  : initial Levenberg-Marquardt damping.
      n_cg      : max CG inner iterations per outer step.
      alpha_max : maximum step length (Wolfe line-search).
      c1        : Armijo sufficient-decrease constant.
      ls_shrink : how much to shrink alpha when Armijo fails.
    """
    x = torch.tensor(x_init.copy(), dtype=torch.float32).flatten()
    x_min = -setup.sigma_bg + 1.0e-6
    lam = lam_init
    log = []

    # initial misfit
    y0 = setup.forward(x.reshape(28, 28).clamp_min(x_min))
    misfit = 0.5 * float(((y0 - setup.y_obs) ** 2).sum())
    log.append((0, misfit, lam))
    if verbose:
        print(f"  iter   0  misfit {misfit:.6e}  (initial)")

    for it in range(1, K + 1):
        x_img = x.reshape(28, 28).clamp_min(x_min)
        y_pred, J = setup.forward_and_jacobian(x_img)
        r = y_pred - setup.y_obs
        # Solve damped normal equations via CG
        delta = cg_solve(J, r, lam, n_cg=n_cg, verbose=False)

        # Wolfe / Armijo line search
        g = J.T @ r                       # gradient of 0.5||r||²
        decr_pred = float((g @ delta).item())   # expected first-order decrease
        if decr_pred >= 0:
            # ascent direction shouldn't happen — flip
            delta = -delta
            decr_pred = -decr_pred

        alpha = alpha_max
        success = False
        for ls in range(ls_max):
            x_try = (x + alpha * delta).clamp_min(x_min)
            y_try = setup.forward(x_try.reshape(28, 28))
            misfit_try = 0.5 * float(((y_try - setup.y_obs) ** 2).sum())
            # Armijo: misfit_try ≤ misfit + c1 * alpha * (decr_pred)
            if misfit_try <= misfit + c1 * alpha * decr_pred:
                success = True
                break
            alpha *= ls_shrink

        if success:
            x = x_try
            misfit = misfit_try
            lam = max(lam * 0.5, lam_min)            # shrink damping on success
        else:
            lam = min(lam * 4.0, lam_max)            # grow damping on failure
            if verbose:
                print(f"  iter {it:3d}  line-search failed, damp={lam:.2e}")

        if it % log_every == 0 or it == 1:
            log.append((it, misfit, lam))
            if verbose:
                print(f"  iter {it:3d}  misfit {misfit:.6e}  α={alpha:.2e}  λ={lam:.2e}")

        if misfit < target_misfit:
            log.append((it, misfit, lam))
            if verbose:
                print(f"  iter {it:3d}  misfit {misfit:.3e} (below target)")
            break

    x_final = x.reshape(28, 28).clamp_min(x_min).detach().cpu().numpy()
    return x_final, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_mat", required=True,
                   help=".mat file with 'x_answer' (28×28) field to polish further")
    p.add_argument("--K",         type=int, default=50)
    p.add_argument("--n_cg",      type=int, default=60)
    p.add_argument("--lam_init",  type=float, default=1e-4)
    p.add_argument("--lam_min",   type=float, default=1e-10)
    p.add_argument("--target_misfit", type=float, default=1e-12,
                   help="stop when misfit below this")
    p.add_argument("--out_mat",   default="../gn_cg_answer.mat")
    p.add_argument("--fig",       default="../figures/gn_cg_polish.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        d = loadmat(args.init_mat)
        # Accept multiple naming conventions
        if 'x_answer' in d:
            x_init = d['x_answer'].astype(np.float32)
        elif 'sigma_answer' in d:
            x_init = (d['sigma_answer'].astype(np.float32) - 1.0)
        elif 'sigma_answer_rot' in d:
            x_init = (d['sigma_answer_rot'].astype(np.float32) - 1.0)
        else:
            raise ValueError(f"Couldn't find x_answer or sigma_answer in {args.init_mat}; keys: {list(d.keys())}")
        if x_init.ndim != 2:
            x_init = x_init.reshape(28, 28)
        print(f"\nGN-CG polish of {args.init_mat}")
        print(f"  init image shape={x_init.shape}, range=[{x_init.min():.3f}, {x_init.max():.3f}]")

        y0 = setup.forward(torch.tensor(x_init, dtype=torch.float32))
        m0 = 0.5 * float(((y0 - setup.y_obs) ** 2).sum())
        print(f"  initial misfit = {m0:.3e}")

        t0 = time.time()
        x_final, log = gn_cg_polish(setup, x_init, K=args.K, n_cg=args.n_cg,
                                     lam_init=args.lam_init, lam_min=args.lam_min,
                                     target_misfit=args.target_misfit, verbose=True)
        elapsed = time.time() - t0

        final_misfit = log[-1][1]
        print(f"\nGN-CG done in {elapsed:.1f}s.  Final misfit: {final_misfit:.3e}")
        print(f"  reduction: {m0/final_misfit:.1f}× over input")

        sigma_answer = 1.0 + x_final
        savemat(args.out_mat, {
            'sigma_answer': sigma_answer,
            'x_answer':     x_final,
            'final_misfit': float(final_misfit),
            'init_misfit':  float(m0),
            'init_file':    args.init_mat,
        })
        print(f"Saved {args.out_mat}")

        # Plot trajectory
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(x_init, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Input (from {args.init_mat.split("/")[-1]})\n'
                          f'misfit {m0:.2e}', fontsize=11)
        axes[0].axis('off')
        axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].set_title(f'GN-CG polished\nmisfit {final_misfit:.2e}',
                          fontsize=11, fontweight='bold')
        axes[1].axis('off')
        iters, mis, lams = zip(*log)
        axes[2].plot(iters, mis, 'o-', markersize=5)
        axes[2].set_yscale('log')
        axes[2].axhline(4.72e-8, color='gray', linestyle=':', label='PnP-DM v2 (LM polish) 4.72e-8')
        axes[2].axhline(1.006e-7, color='red', linestyle='--', alpha=0.5,
                        label='P1 rotation-aware 1.0e-7')
        axes[2].set_xlabel('GN iteration'); axes[2].set_ylabel('misfit')
        axes[2].set_title(f'GN-CG convergence ({args.n_cg}-iter CG inner)')
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
