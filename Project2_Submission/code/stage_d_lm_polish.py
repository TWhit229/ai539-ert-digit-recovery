"""
Stage D of DI-RTG: final Levenberg-Marquardt polish, identical in spirit to
Project 1's `solve_competition.m` refinement.

Drives misfit toward the measurement noise floor (~1e-6 for our competition vector)
using only the data fidelity `½‖F(σ_bg + x) − y_obs‖²`, with momentum-style
heavy-ball descent matching what Project 1 used.

Per Project 1: eta=5, gamma=0.95, K=1500 iterations starting from a clean image.
"""
import argparse, time
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup


def stage_d_lm_polish(setup, x_init, K=2000, eta=5.0, gamma=0.95,
                      target_misfit=1.0e-6, verbose=True, log_every=100,
                      use_backtrack=True, eta_min=1e-3, eta_grow=1.05):
    """
    Project 1 used eta=5, gamma=0.95, K=1500 starting from an MNIST training image
    (misfit ~2e-5 already). Our Stage A start is 100× worse, so the same momentum
    overshoots. Add a backtracking line-search that halves eta whenever a step
    doesn't decrease misfit. Once a step succeeds, grow eta slightly back up.
    """
    x = torch.tensor(x_init.copy(), dtype=torch.float32).flatten()
    vel = torch.zeros_like(x)
    log = []
    x_min = -setup.sigma_bg + 1.0e-6

    # Initial misfit
    y_init = setup.forward(x.reshape(28, 28))
    cur_misfit = 0.5 * float(((y_init - setup.y_obs) ** 2).sum())
    log.append((0, cur_misfit))
    if verbose:
        print(f"  it    0  misfit {cur_misfit:.3e}  (initial)")

    for it in range(1, K + 1):
        # Compute gradient at current x
        x_img = x.reshape(28, 28).clamp_min(x_min)
        y_pred, J = setup.forward_and_jacobian(x_img)
        r = y_pred - setup.y_obs
        grad = J.T @ r

        if not use_backtrack:
            # Project 1 verbatim
            vel = gamma * vel + eta * grad
            x = (x - vel).clamp_min(x_min)
            new_misfit = 0.5 * float(((setup.forward(x.reshape(28,28)) - setup.y_obs) ** 2).sum())
            cur_misfit = new_misfit
        else:
            # Backtracking on eta with momentum
            tentative_vel = gamma * vel + eta * grad
            x_new = (x - tentative_vel).clamp_min(x_min)
            new_misfit = 0.5 * float(((setup.forward(x_new.reshape(28,28)) - setup.y_obs) ** 2).sum())

            tries = 0
            while new_misfit > cur_misfit and eta > eta_min and tries < 8:
                eta = eta * 0.5
                tentative_vel = gamma * vel + eta * grad
                x_new = (x - tentative_vel).clamp_min(x_min)
                new_misfit = 0.5 * float(((setup.forward(x_new.reshape(28,28)) - setup.y_obs) ** 2).sum())
                tries += 1

            if new_misfit <= cur_misfit:
                x = x_new
                vel = tentative_vel
                cur_misfit = new_misfit
                eta = min(eta * eta_grow, 5.0)              # slowly grow back toward Project 1's eta
            else:
                # Couldn't make progress even at eta_min — give up
                if verbose:
                    print(f"  it {it:4d}  no progress at eta={eta:.2e}; halting")
                break

        if it % log_every == 0 or it == 1:
            log.append((it, cur_misfit))
            if verbose:
                print(f"  it {it:4d}  misfit {cur_misfit:.3e}  eta={eta:.2e}")

        if cur_misfit < target_misfit:
            log.append((it, cur_misfit))
            if verbose:
                print(f"  it {it:4d}  misfit {cur_misfit:.3e}  (below target)")
            break

    x_final = x.reshape(28, 28).clamp_min(x_min).detach().cpu().numpy()
    return x_final, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="stage_a_best.npz",
                   help="initial image (defaults to Stage A best output)")
    p.add_argument("--K", type=int, default=1500)
    p.add_argument("--eta", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--target_misfit", type=float, default=1.0e-6)
    p.add_argument("--out", default="stage_d_result.npz")
    p.add_argument("--fig", default="../figures/stage_d_final.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        init_data = np.load(args.init, allow_pickle=True)
        x_init = init_data['x']
        init_misfit_recorded = float(init_data['misfit']) if 'misfit' in init_data.files else None
        print(f"\nInit  shape={x_init.shape}  misfit_recorded={init_misfit_recorded}")

        print(f"\nStage D LM polish  K={args.K}  eta={args.eta}  gamma={args.gamma}")
        t0 = time.time()
        x_final, log = stage_d_lm_polish(setup, x_init, K=args.K, eta=args.eta,
                                          gamma=args.gamma, target_misfit=args.target_misfit)
        elapsed = time.time() - t0

        final_misfit = log[-1][1]
        print(f"\nDone in {elapsed:.1f}s.")
        print(f"Final misfit: {final_misfit:.3e}  (Project 1 hit 1.76e-6 on this vector)")

        np.savez(args.out, x=x_final, misfit=final_misfit, log=np.array(log))
        print(f"Saved {args.out}")

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(x_init, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Stage A init\nmisfit = {init_misfit_recorded:.2e}'
                          if init_misfit_recorded else 'Stage A init')
        axes[1].imshow(x_final, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'After Stage D polish\nmisfit = {final_misfit:.2e}')
        for ax in axes[:2]:
            ax.axis('off')
        iters, mis = zip(*log)
        axes[2].plot(iters, mis, 'o-', markersize=4)
        axes[2].set_yscale('log')
        axes[2].set_xlabel('iteration')
        axes[2].set_ylabel('misfit')
        axes[2].axhline(1.76e-6, color='red', linestyle='--', alpha=0.5,
                        label='Project 1 floor (1.76e-6)')
        axes[2].set_title('LM polish trajectory')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
