"""
Validate DI-RTG on 10 held-out MNIST test-set digits (one per class 0–9).

Mirrors Project 1's `validate_0to9.m`. For each unseen test digit:
  1. Synthesize the measurement: y = F(σ_b + test_image).
  2. Run an abbreviated DI-RTG pipeline:
       - Stage A multi-seed DAPS warm-start (4 seeds for speed)
       - Stage C classifier + top-2 class template search (training images only)
       - Stage D short LM polish (K=300 iters)
  3. Check whether the predicted class matches the true class.

Output: figures/validate_di_rtg.png and a per-digit summary.
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, daps_warmstart
from stage_c_template_retrieval import train_classifier, stage_c
from stage_d_lm_polish import stage_d_lm_polish


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N_seeds_A", type=int, default=4)
    p.add_argument("--K_outer", type=int, default=30)
    p.add_argument("--K_lm", type=int, default=300)
    p.add_argument("--out_fig", default="../figures/validate_di_rtg.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Load MNIST training + test sets
        mn = loadmat("MNIST Data/mnist.mat")
        M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_train = mn['training']['labels'][0, 0].flatten()
        M_test  = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_test  = mn['test']['labels'][0, 0].flatten()
        if M_train.max() > 1.5: M_train = M_train / 255.0
        if M_test.max()  > 1.5: M_test  = M_test  / 255.0
        print(f"  MNIST: train {M_train.shape[1]}, test {M_test.shape[1]}")

        # Train the classifier once (reused for all 10 digits)
        print("\nTraining TinyMNIST classifier (8 epochs)...")
        t0 = time.time()
        clf = train_classifier(M_train, L_train, epochs=8, verbose=False)
        print(f"  trained in {time.time()-t0:.1f}s")

        # Pick one test image per class
        test_indices = [int(np.where(L_test == d)[0][0]) for d in range(10)]
        results = []

        for digit in range(10):
            tidx = test_indices[digit]
            true_x = M_test[:, tidx].reshape(28, 28).astype(np.float32)
            print(f"\n=== digit {digit} (test idx {tidx}) ===")

            # Synthetic measurement
            with torch.no_grad():
                y = setup.forward(torch.tensor(true_x, dtype=torch.float32))
            # Stash y_obs on setup temporarily
            true_y_obs = setup.y_obs
            setup.y_obs = y

            try:
                # ---- Stage A multi-seed ----
                t0 = time.time()
                best = (None, np.inf, None)
                for s in range(args.N_seeds_A):
                    x_warm, _ = daps_warmstart(setup, K_outer=args.K_outer, K_inner=3,
                                                eta=0.5, lam=1.0, seed=s, verbose=False)
                    mis = 0.5 * float(((setup.forward(torch.tensor(x_warm, dtype=torch.float32))
                                         - setup.y_obs) ** 2).sum())
                    if mis < best[1]:
                        best = (s, mis, x_warm)
                print(f"  Stage A best: seed {best[0]}, misfit {best[1]:.3e}  ({time.time()-t0:.1f}s)")
                stage_a_img = best[2]

                # ---- Stage C: classify + restricted template search ----
                t0 = time.time()
                template, tmpl_idx, tmpl_misfit, tmpl_class, _ = stage_c(
                    setup, stage_a_img, M_train, L_train, clf,
                    min_top1_conf=0.99, verbose=False)
                print(f"  Stage C: template #{tmpl_idx}, class {tmpl_class}, misfit {tmpl_misfit:.3e}  ({time.time()-t0:.1f}s)")

                # ---- Stage D: short LM polish ----
                t0 = time.time()
                x_final, log = stage_d_lm_polish(setup, template, K=args.K_lm, eta=5.0, gamma=0.95,
                                                  use_backtrack=False, target_misfit=1e-7,
                                                  log_every=args.K_lm, verbose=False)
                final_misfit = log[-1][1]
                print(f"  Stage D: misfit {final_misfit:.3e}  ({time.time()-t0:.1f}s)")

                results.append({
                    'digit': digit, 'true_x': true_x, 'stage_a_img': stage_a_img,
                    'template': template, 'final': x_final,
                    'predicted_class': tmpl_class,
                    'correct': (tmpl_class == digit),
                    'final_misfit': final_misfit,
                    'stage_c_misfit': tmpl_misfit,
                })
            finally:
                setup.y_obs = true_y_obs

        # Summary
        correct = sum(r['correct'] for r in results)
        print(f"\n========================================")
        print(f"DI-RTG validation: {correct}/10 correct")
        print(f"========================================")
        for r in results:
            mark = '✓' if r['correct'] else '✗'
            print(f"  {mark}  true {r['digit']}  predicted {r['predicted_class']}  "
                  f"final misfit {r['final_misfit']:.2e}")

        # Plot: 3 rows × 10 cols (true / Stage-C template / Stage-D refined)
        fig, axes = plt.subplots(3, 10, figsize=(18, 6.5))
        for k, r in enumerate(results):
            axes[0, k].imshow(r['true_x'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"{r['digit']}", fontsize=13)
            axes[1, k].imshow(r['template'], cmap='gray', vmin=0, vmax=1)
            c = [0, 0.55, 0] if r['correct'] else [0.85, 0, 0]
            axes[1, k].set_title(f"{r['predicted_class']}", color=c, fontweight='bold', fontsize=13)
            axes[2, k].imshow(1.0 + r['final'], cmap='viridis', vmin=1, vmax=2)
            axes[2, k].set_title(f"{r['final_misfit']:.1e}", fontsize=9)
            for row in range(3):
                axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
        axes[0, 0].set_ylabel('True (test)', fontsize=11)
        axes[1, 0].set_ylabel('Stage C template', fontsize=11)
        axes[2, 0].set_ylabel('Stage D σ', fontsize=11)
        plt.suptitle(f'DI-RTG validation on 10 held-out digits — {correct}/10 correct',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.out_fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"\nSaved {args.out_fig}")

        np.savez("validate_di_rtg.npz",
                 results=np.array(results, dtype=object), correct_count=correct)
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
