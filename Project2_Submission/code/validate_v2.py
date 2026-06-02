"""
Validate DI-RTG v2 on 10 held-out MNIST test digits (one per class).

Mirrors `validate_di_rtg.py` but uses the v2 pipeline:
  Stage A → A.5 cleanup → classifier on clean image → rotation-aware Stage C
  (top-2 classes) → Stage D LM polish.

For wall-clock reasons:
  - n_seeds_A = 4   (not 12)
  - K_upright_per_class = 30   (not 50)
  - coarse rotation step = 15° (24 angles, not 36)
  - K_lm = 300        (not 1500)
Total expected: ~6-8 min per digit × 10 = 60-80 min.
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.io import loadmat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, daps_warmstart
from stage_c_template_retrieval import train_classifier
from stage_c_rotation import stage_c_rotation
from stage_d_lm_polish import stage_d_lm_polish
from run_di_rtg_v2 import short_lm_cleanup, classify_with_top_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N_seeds_A",   type=int,   default=4)
    p.add_argument("--K_outer",     type=int,   default=30)
    p.add_argument("--n_classes",   type=int,   default=2)
    p.add_argument("--K_upright",   type=int,   default=30)
    p.add_argument("--coarse_step", type=float, default=15.0)
    p.add_argument("--fine_step",   type=float, default=0.5)
    p.add_argument("--K_lm",        type=int,   default=300)
    p.add_argument("--out_fig",     default="../figures/validate_di_rtg_v2.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        mn = loadmat("MNIST Data/mnist.mat")
        M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_train = mn['training']['labels'][0, 0].flatten()
        M_test  = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_test  = mn['test']['labels'][0, 0].flatten()
        if M_train.max() > 1.5: M_train = M_train / 255.0
        if M_test.max()  > 1.5: M_test  = M_test  / 255.0
        print(f"  MNIST: train {M_train.shape[1]}, test {M_test.shape[1]}")

        print("\nTraining TinyMNIST classifier (8 epochs)...")
        t0 = time.time()
        clf = train_classifier(M_train, L_train, epochs=8, verbose=False)
        print(f"  trained in {time.time()-t0:.0f}s")

        coarse_angles = np.arange(0, 360, args.coarse_step).astype(float)
        test_indices = [int(np.where(L_test == d)[0][0]) for d in range(10)]
        results = []

        for digit in range(10):
            tidx = test_indices[digit]
            true_x = M_test[:, tidx].reshape(28, 28).astype(np.float32)
            print(f"\n========== digit {digit} (test idx {tidx}) ==========")

            # synthetic y_obs
            with torch.no_grad():
                y = setup.forward(torch.tensor(true_x, dtype=torch.float32))
            true_y_obs = setup.y_obs
            setup.y_obs = y

            try:
                # Stage A multi-seed
                t0 = time.time()
                best = (None, np.inf, None)
                for s in range(args.N_seeds_A):
                    x_warm, _ = daps_warmstart(setup, K_outer=args.K_outer, K_inner=3,
                                                eta=0.5, lam=1.0, seed=s, verbose=False)
                    mis = 0.5 * float(((setup.forward(torch.tensor(x_warm, dtype=torch.float32))
                                         - setup.y_obs) ** 2).sum())
                    if mis < best[1]:
                        best = (s, mis, x_warm)
                print(f"  Stage A best: seed {best[0]}, misfit {best[1]:.3e}  "
                      f"({time.time()-t0:.0f}s)")
                x_a = best[2]

                # Stage A.5 cleanup
                t0 = time.time()
                x_clean = short_lm_cleanup(setup, x_a, K=80)
                mis_a5 = 0.5 * float(((setup.forward(torch.tensor(x_clean, dtype=torch.float32))
                                        - setup.y_obs) ** 2).sum())
                print(f"  Stage A.5: misfit {mis_a5:.3e}  ({time.time()-t0:.0f}s)")

                # Stage B classify on cleaned image
                top_classes_a5, probs_a5 = classify_with_top_k(clf, x_clean, k=10)
                classes_to_search = top_classes_a5[:args.n_classes]
                print(f"  classifier top-3 on cleaned: {top_classes_a5[:3]}  "
                      f"probs {[f'{probs_a5[c]:.2f}' for c in top_classes_a5[:3]]}")
                print(f"  searching {len(classes_to_search)} classes: {classes_to_search}")

                # Stage C v2 (rotation-aware)
                t0 = time.time()
                sc = stage_c_rotation(setup, x_clean, M_train, L_train, clf,
                                      classes_to_search=classes_to_search,
                                      K_upright_per_class=args.K_upright,
                                      coarse_angles=coarse_angles,
                                      fine_step=args.fine_step,
                                      verbose=False)
                print(f"  Stage C v2: template #{sc['best_idx']} class {sc['best_class']} "
                      f"@ {sc['best_theta']:.1f}°, misfit {sc['best_misfit']:.3e}  "
                      f"({time.time()-t0:.0f}s)")

                # Stage D short polish
                t0 = time.time()
                x_final, log = stage_d_lm_polish(setup, sc['best_template'],
                                                  K=args.K_lm, eta=5.0, gamma=0.95,
                                                  use_backtrack=False, target_misfit=1e-8,
                                                  log_every=args.K_lm, verbose=False)
                final_misfit = log[-1][1]
                print(f"  Stage D: misfit {final_misfit:.3e}  ({time.time()-t0:.0f}s)")

                results.append({
                    'digit': digit, 'true_x': true_x, 'stage_a': x_a, 'clean': x_clean,
                    'template': sc['best_template'], 'final': x_final,
                    'predicted_class': sc['best_class'], 'theta': sc['best_theta'],
                    'correct': (sc['best_class'] == digit),
                    'final_misfit': final_misfit, 'stage_c_misfit': sc['best_misfit'],
                    'classes_searched': sc['classes_searched'],
                })
            finally:
                setup.y_obs = true_y_obs

        correct = sum(r['correct'] for r in results)
        print(f"\n========================================")
        print(f"DI-RTG v2 validation: {correct}/10 correct")
        print(f"========================================")
        for r in results:
            mark = '✓' if r['correct'] else '✗'
            print(f"  {mark}  true {r['digit']}  predicted {r['predicted_class']} "
                  f"@ {r['theta']:5.1f}°  final misfit {r['final_misfit']:.2e}   "
                  f"(searched {r['classes_searched']})")

        # 3-row figure
        fig, axes = plt.subplots(3, 10, figsize=(18, 6.5))
        for k, r in enumerate(results):
            axes[0, k].imshow(r['true_x'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"{r['digit']}", fontsize=13)
            axes[1, k].imshow(r['template'], cmap='gray', vmin=0, vmax=1)
            c = [0, 0.55, 0] if r['correct'] else [0.85, 0, 0]
            axes[1, k].set_title(f"{r['predicted_class']} @ {r['theta']:.0f}°",
                                 color=c, fontweight='bold', fontsize=12)
            axes[2, k].imshow(1.0 + r['final'], cmap='viridis', vmin=1, vmax=2)
            axes[2, k].set_title(f"{r['final_misfit']:.1e}", fontsize=9)
            for row in range(3):
                axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
        axes[0, 0].set_ylabel('True (test)', fontsize=11)
        axes[1, 0].set_ylabel('Stage C v2 (rotated)', fontsize=11)
        axes[2, 0].set_ylabel('Stage D σ', fontsize=11)
        plt.suptitle(f'DI-RTG v2 validation — {correct}/10 correct',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.out_fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"\nSaved {args.out_fig}")

        np.savez("validate_di_rtg_v2.npz",
                 results=np.array(results, dtype=object), correct_count=correct)
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
