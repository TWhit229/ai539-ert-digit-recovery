"""
DI-RTG end-to-end pipeline for Project 2.

  DAPS-Init (multi-seed)  →  Classifier + targeted template retrieval  →  LM polish

A diffusion-based prior method for the ERT inverse problem. The diffusion model
contributes (a) sample candidates that identify the digit class and
(b) a class restriction that cuts the template search by ~3-4× vs Project 1.

Final recovered conductivity image saved as `competition_answer.mat` with the
digit class and final misfit.

Stages we tested but did NOT include in the final pipeline:
  - Stage B (GN refinement with diffusion anchor): the diffusion anchor pulled
    AWAY from data fit because the true answer is a rotated 5, not on the
    upright-MNIST manifold the prior learned. Skipped for v1. Could be revisited
    with rotation-augmented training.

Usage:
    python3 run_di_rtg.py [--seeds 12] [--K_outer 30] [--K_lm 1500]
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup, daps_warmstart, x_diff_to_pix
from stage_c_template_retrieval import train_classifier, stage_c, TinyMNIST
from stage_d_lm_polish import stage_d_lm_polish


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12, help="Stage A: # seeds to try")
    p.add_argument("--K_outer", type=int, default=30)
    p.add_argument("--K_inner", type=int, default=3)
    p.add_argument("--K_lm", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=8, help="Stage C: classifier training epochs")
    p.add_argument("--out_mat", default="../competition_answer.mat",
                   help="competition .mat with sigma_answer, digit, misfit, …")
    p.add_argument("--fig", default="../figures/di_rtg_pipeline.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # --------------------------------------------------------------
        # Stage A — multi-seed DAPS warm-start
        # --------------------------------------------------------------
        print("\n" + "="*68)
        print(f"STAGE A — multi-seed DAPS warm-start ({args.seeds} seeds)")
        print("="*68)
        stage_a_results = []
        t0 = time.time()
        for s in range(args.seeds):
            x_warm, _ = daps_warmstart(setup, K_outer=args.K_outer, K_inner=args.K_inner,
                                       eta=0.5, lam=1.0, seed=s, verbose=False)
            mis = 0.5 * float(((setup.forward(torch.tensor(x_warm, dtype=torch.float32))
                                 - setup.y_obs) ** 2).sum())
            print(f"  seed {s:2d}  misfit {mis:.3e}")
            stage_a_results.append((s, mis, x_warm))
        stage_a_results.sort(key=lambda r: r[1])
        best_seed, stage_a_misfit, stage_a_img = stage_a_results[0]
        print(f"\n  best Stage A: seed {best_seed}, misfit {stage_a_misfit:.3e}  "
              f"({time.time()-t0:.1f}s total)")

        # --------------------------------------------------------------
        # Stage C — classifier + targeted template retrieval
        # --------------------------------------------------------------
        print("\n" + "="*68)
        print("STAGE C — classifier-guided template retrieval")
        print("="*68)
        mn = loadmat("MNIST Data/mnist.mat")
        M = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L = mn['training']['labels'][0, 0].flatten()
        if M.max() > 1.5: M = M / 255.0
        print(f"\nTraining TinyMNIST classifier ({args.epochs} epochs)...")
        t0 = time.time()
        clf = train_classifier(M, L, epochs=args.epochs, verbose=False)
        # Quick test acc check
        Xtest = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        if Xtest.max() > 1.5: Xtest = Xtest / 255.0
        Ltest = mn['test']['labels'][0, 0].flatten()
        with torch.no_grad():
            pred = clf(torch.tensor(Xtest.T.reshape(-1,1,28,28), dtype=torch.float32)).argmax(1).numpy()
        test_acc = (pred == Ltest).mean()
        print(f"  classifier test acc: {test_acc:.4f}  ({time.time()-t0:.1f}s trained)")

        t0 = time.time()
        template, tmpl_idx, tmpl_misfit, tmpl_class, candidates = stage_c(
            setup, stage_a_img, M, L, clf, min_top1_conf=0.99, verbose=True)
        print(f"\n  Stage C best: template #{tmpl_idx}, class {tmpl_class}, "
              f"misfit {tmpl_misfit:.3e}  ({time.time()-t0:.1f}s)")

        # --------------------------------------------------------------
        # Stage D — LM polish from template
        # --------------------------------------------------------------
        print("\n" + "="*68)
        print(f"STAGE D — LM polish (Project 1 hyperparameters: eta=5, γ=0.95, K={args.K_lm})")
        print("="*68)
        t0 = time.time()
        x_final, log = stage_d_lm_polish(setup, template, K=args.K_lm, eta=5.0, gamma=0.95,
                                          use_backtrack=False, target_misfit=1e-7,
                                          log_every=200, verbose=True)
        final_misfit = log[-1][1]
        print(f"\n  Stage D done in {time.time()-t0:.1f}s   "
              f"final misfit = {final_misfit:.3e}   (Project 1 floor: 1.76e-6)")

        # --------------------------------------------------------------
        # Save answer + figure
        # --------------------------------------------------------------
        sigma_answer = 1.0 + x_final
        savemat(args.out_mat, {
            'sigma_answer':  sigma_answer,
            'x_answer':      x_final,
            'digit':         int(tmpl_class),
            'final_misfit':  float(final_misfit),
            'template_idx':  int(tmpl_idx),
            'stage_a_seed':  int(best_seed),
            'stage_a_misfit': float(stage_a_misfit),
            'stage_c_misfit': float(tmpl_misfit),
        })
        print(f"\nSaved {args.out_mat}")

        # Visualize the whole pipeline
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        axes[0].imshow(stage_a_img, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Stage A (DAPS, seed {best_seed})\n'
                          f'misfit {stage_a_misfit:.2e}', fontsize=11)
        axes[1].imshow(template, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Stage C template\n#{tmpl_idx} (class {tmpl_class})\n'
                          f'misfit {tmpl_misfit:.2e}', fontsize=11)
        axes[2].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[2].set_title(f'Stage D final σ\nmisfit {final_misfit:.2e}', fontsize=11)
        iters, mis = zip(*log)
        axes[3].plot(iters, mis, 'o-', markersize=4)
        axes[3].set_yscale('log')
        axes[3].set_xlabel('iteration'); axes[3].set_ylabel('misfit')
        axes[3].axhline(1.76e-6, color='red', linestyle='--', alpha=0.5,
                        label='Project 1 floor 1.76e-6')
        axes[3].set_title('Stage D trajectory')
        axes[3].grid(alpha=0.3); axes[3].legend()
        for ax in axes[:3]: ax.axis('off')
        plt.suptitle(f"DI-RTG Pipeline — digit {tmpl_class}, final misfit {final_misfit:.2e}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

        # Summary
        print("\n" + "="*68)
        print("DI-RTG SUMMARY")
        print("="*68)
        print(f"  Stage A  (multi-seed DAPS, n={args.seeds}):  misfit  {stage_a_misfit:.3e}")
        print(f"  Stage C  (template #{tmpl_idx}, class {tmpl_class}):  misfit  {tmpl_misfit:.3e}")
        print(f"  Stage D  (LM polish, {args.K_lm} iters):       misfit  {final_misfit:.3e}")
        print(f"  Project 1 reference (60k brute-force + LM):     misfit  1.76e-6")
        print(f"\n  Predicted digit: {tmpl_class}")
        print(f"  Saved: {args.out_mat}, {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
