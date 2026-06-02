"""
DI-RTG v2 — rotation-aware, class-restricted, with cleanup-before-classify.

Two changes vs v1:
  1) Stage A.5 short LM cleanup (50 iter) on Stage A's best image, so the
     classifier sees a *clean* digit, not noisy DAPS output.
  2) Stage C v2 (`stage_c_rotation.py`): for top-2 classes, scan all training
     templates upright, then a rotation search on the top-50 per class. This
     matches Project 1's rotation-aware result (misfit ~1e-7 on the
     competition vector).

Stage D unchanged from v1 (LM polish, eta=5, gamma=0.95, K=1500).
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, daps_warmstart
from stage_c_template_retrieval import train_classifier, TinyMNIST
from stage_c_rotation import stage_c_rotation
from stage_d_lm_polish import stage_d_lm_polish


def short_lm_cleanup(setup, x_init, K=80, eta=2.0, gamma=0.9):
    """Cheap LM polish to clean a noisy Stage-A image before classification.
    Smaller eta + fewer iters than full Stage D — we want a digit shape, not
    a fully-converged answer."""
    x = torch.tensor(x_init.copy(), dtype=torch.float32).flatten()
    vel = torch.zeros_like(x)
    x_min = -setup.sigma_bg + 1.0e-6
    for _ in range(K):
        x_img = x.reshape(28, 28).clamp_min(x_min)
        _, J = setup.forward_and_jacobian(x_img)
        r = setup.forward(x_img) - setup.y_obs
        vel = gamma * vel + eta * (J.T @ r)
        x = (x - vel).clamp_min(x_min)
    return x.reshape(28, 28).detach().cpu().numpy()


def classify_with_top_k(classifier, img28, k=3):
    """Return list of class indices sorted by probability."""
    x_t = torch.tensor(img28.reshape(1, 1, 28, 28), dtype=torch.float32)
    with torch.no_grad():
        probs = Fnn.softmax(classifier(x_t), dim=1).numpy().flatten()
    order = np.argsort(-probs)
    return [int(c) for c in order[:k]], probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12, help="Stage A seeds")
    p.add_argument("--K_outer", type=int, default=30)
    p.add_argument("--K_inner", type=int, default=3)
    p.add_argument("--n_classes", type=int, default=2,
                   help="how many top classes to rotation-search in Stage C")
    p.add_argument("--K_upright", type=int, default=50,
                   help="top-K upright templates carried into rotation search")
    p.add_argument("--coarse_step", type=float, default=10.0,
                   help="coarse rotation step (degrees)")
    p.add_argument("--fine_step", type=float, default=0.5)
    p.add_argument("--K_lm", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out_mat", default="../competition_answer_v2.mat")
    p.add_argument("--fig", default="../figures/di_rtg_v2_pipeline.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # ================================================================
        # Stage A — multi-seed DAPS warm-start
        # ================================================================
        print("\n" + "="*68)
        print(f"STAGE A — multi-seed DAPS warm-start ({args.seeds} seeds)")
        print("="*68)
        t0 = time.time()
        results_a = []
        for s in range(args.seeds):
            x_warm, _ = daps_warmstart(setup, K_outer=args.K_outer, K_inner=args.K_inner,
                                       eta=0.5, lam=1.0, seed=s, verbose=False)
            mis = 0.5 * float(((setup.forward(torch.tensor(x_warm, dtype=torch.float32))
                                 - setup.y_obs) ** 2).sum())
            results_a.append((s, mis, x_warm))
            print(f"  seed {s:2d}  misfit {mis:.3e}")
        results_a.sort(key=lambda r: r[1])
        best_seed, mis_a, x_a = results_a[0]
        print(f"\n  Stage A best: seed {best_seed}, misfit {mis_a:.3e}   "
              f"({time.time()-t0:.0f}s)")

        # ================================================================
        # Stage A.5 — short LM cleanup so the classifier sees a clean digit
        # ================================================================
        print("\n" + "="*68)
        print("STAGE A.5 — short LM cleanup for the classifier")
        print("="*68)
        t0 = time.time()
        x_clean = short_lm_cleanup(setup, x_a, K=80)
        mis_a5 = 0.5 * float(((setup.forward(torch.tensor(x_clean, dtype=torch.float32))
                                - setup.y_obs) ** 2).sum())
        print(f"  cleaned in {time.time()-t0:.0f}s, misfit {mis_a5:.3e}")

        # ================================================================
        # Stage B — classify on cleaned image (high confidence)
        # ================================================================
        print("\n" + "="*68)
        print("STAGE B — classify cleaned image")
        print("="*68)
        mn = loadmat("MNIST Data/mnist.mat")
        M = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L = mn['training']['labels'][0, 0].flatten()
        if M.max() > 1.5: M = M / 255.0

        print(f"\nTraining TinyMNIST classifier ({args.epochs} epochs)...")
        t0 = time.time()
        clf = train_classifier(M, L, epochs=args.epochs, verbose=False)
        Xtest = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        if Xtest.max() > 1.5: Xtest = Xtest / 255.0
        Ltest = mn['test']['labels'][0, 0].flatten()
        with torch.no_grad():
            pred = clf(torch.tensor(Xtest.T.reshape(-1,1,28,28), dtype=torch.float32)).argmax(1).numpy()
        test_acc = (pred == Ltest).mean()
        print(f"  classifier test acc: {test_acc:.4f}  ({time.time()-t0:.0f}s)")

        top_classes_a,  probs_a  = classify_with_top_k(clf, x_a, k=10)
        top_classes_a5, probs_a5 = classify_with_top_k(clf, x_clean, k=10)
        print(f"  Stage A   classifier top-3: {top_classes_a[:3]}   "
              f"probs {[f'{probs_a[c]:.2f}' for c in top_classes_a[:3]]}")
        print(f"  Stage A.5 classifier top-3: {top_classes_a5[:3]}  "
              f"probs {[f'{probs_a5[c]:.2f}' for c in top_classes_a5[:3]]}")

        # Use the cleaned classification ordering, take top-n_classes
        classes_to_search = top_classes_a5[:args.n_classes]

        # ================================================================
        # Stage C v2 — rotation-aware template search
        # ================================================================
        print("\n" + "="*68)
        print("STAGE C v2 — class-restricted template search + rotation")
        print("="*68)
        coarse_angles = np.arange(0, 360, args.coarse_step).astype(float)
        t0 = time.time()
        sc = stage_c_rotation(setup, x_clean, M, L, clf,
                              classes_to_search=classes_to_search,
                              K_upright_per_class=args.K_upright,
                              coarse_angles=coarse_angles,
                              fine_step=args.fine_step,
                              verbose=True)
        print(f"\nStage C v2 best:  template #{sc['best_idx']}  class {sc['best_class']}  "
              f"theta {sc['best_theta']:.1f}°  misfit {sc['best_misfit']:.3e}   "
              f"({time.time()-t0:.0f}s)")

        # ================================================================
        # Stage D — LM polish from rotated template
        # ================================================================
        print("\n" + "="*68)
        print(f"STAGE D — LM polish ({args.K_lm} iter, eta=5, gamma=0.95)")
        print("="*68)
        t0 = time.time()
        x_final, log = stage_d_lm_polish(setup, sc['best_template'],
                                          K=args.K_lm, eta=5.0, gamma=0.95,
                                          use_backtrack=False, target_misfit=1e-8,
                                          log_every=200, verbose=True)
        final_misfit = log[-1][1]
        print(f"\nStage D done in {time.time()-t0:.0f}s   final misfit {final_misfit:.3e}")

        # ================================================================
        # Save + figure
        # ================================================================
        sigma_answer = 1.0 + x_final
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_final,
            'digit':         int(sc['best_class']),
            'theta':         float(sc['best_theta']),
            'final_misfit':  float(final_misfit),
            'template_idx':  int(sc['best_idx']),
            'stage_a_seed':  int(best_seed),
            'stage_a_misfit':  float(mis_a),
            'stage_a5_misfit': float(mis_a5),
            'stage_c_misfit':  float(sc['best_misfit']),
            'classes_searched': sc['classes_searched'],
        })
        print(f"\nSaved {args.out_mat}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(x_a, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'Stage A (seed {best_seed})\nmisfit {mis_a:.2e}', fontsize=11)
        axes[0, 1].imshow(x_clean, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Stage A.5 cleanup\nmisfit {mis_a5:.2e}', fontsize=11)
        axes[0, 2].imshow(sc['best_template'], cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Stage C v2 best\ntemplate #{sc["best_idx"]} class {sc["best_class"]} '
                             f'@ {sc["best_theta"]:.1f}°\nmisfit {sc["best_misfit"]:.2e}',
                             fontsize=11)
        axes[1, 0].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1, 0].set_title(f'Stage D final σ\nmisfit {final_misfit:.2e}', fontsize=12, fontweight='bold')
        iters, mis = zip(*log)
        axes[1, 1].plot(iters, mis, 'o-', markersize=4)
        axes[1, 1].set_yscale('log')
        axes[1, 1].axhline(1.01e-7, color='red', linestyle='--', alpha=0.5,
                            label='P1 rotation-aware floor (1.0e-7)')
        axes[1, 1].axhline(1.76e-6, color='gray', linestyle=':', alpha=0.5,
                            label='P1 non-rotation floor (1.76e-6)')
        axes[1, 1].set_xlabel('iteration'); axes[1, 1].set_ylabel('misfit')
        axes[1, 1].set_title('Stage D LM trajectory'); axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend(fontsize=9)
        axes[1, 2].axis('off')
        for ax in axes[0, :3]: ax.axis('off')
        axes[1, 0].axis('off')
        plt.suptitle(f"DI-RTG v2 (rotation-aware) — digit {sc['best_class']} @ "
                     f"{sc['best_theta']:.1f}°, final misfit {final_misfit:.2e}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

        # Summary
        print("\n" + "="*68)
        print("DI-RTG v2 SUMMARY")
        print("="*68)
        print(f"  Stage A  (multi-seed DAPS, n={args.seeds}):       misfit {mis_a:.3e}")
        print(f"  Stage A.5 (LM cleanup, 80 iter):                  misfit {mis_a5:.3e}")
        print(f"  Stage C v2 (template + rotation, top-{args.n_classes} classes): "
              f"misfit {sc['best_misfit']:.3e}")
        print(f"  Stage D  (LM polish, {args.K_lm} iter):             misfit {final_misfit:.3e}")
        print(f"\n  Project 1 rotation-aware reference:               misfit 1.01e-7")
        print(f"  Project 1 non-rotation reference:                 misfit 1.76e-6")
        print(f"  DI-RTG v1 (no rotation):                          misfit 1.83e-6")
        print(f"\n  Recovered digit: {sc['best_class']}  rotation {sc['best_theta']:.1f}°")
        print(f"  Saved: {args.out_mat}, {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
