"""Validate the NEW pipeline (cond + CFG + TV polish) on 10 held-out
MNIST test digits.

Mirror of code/validate_final.py but with:
  - class-conditional DDPM (loaded from sweep/ddpm_cond_rot15_ema/)
  - pnp_dm_cfg sampler (32 chains seeded to classes 0..9 cycling, CFG w=3)
  - TV-regularized polish (sweep/tv_polish.py)

For each digit class 0-9:
  1. Pick one unseen test image with that label.
  2. Synthesize y = F(sigma_bg + test_image).
  3. Run PnP-DM-CFG with n_chains x n_iter Gibbs iters.
  4. Pick lowest-misfit chain across all 32 (10 class hypotheses x 3+ chains/class).
  5. TV polish to ~1e-11 regime.
  6. Classify recovered image, check correctness.

Output: 10-digit summary + figure overwriting figures/validate_final.png.
"""
import argparse, time, warnings, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from stage_a_daps_warmstart import ERTSetup
from pnp_dm_cfg import pnp_dm_cfg, load_cond_ddpm
from tv_polish import tv_polish
from stage_c_template_retrieval import train_classifier
from run_di_rtg_v2 import classify_with_top_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond_dir", default="./ddpm_cond_rot15_ema")
    p.add_argument("--n_chains", type=int, default=32)
    p.add_argument("--n_iter",   type=int, default=80)
    p.add_argument("--sigma_n",  type=float, default=1e-4)
    p.add_argument("--cfg_w",    type=float, default=3.0)
    p.add_argument("--K_tv",     type=int, default=120)
    p.add_argument("--tau",      type=float, default=1e-5)
    p.add_argument("--out_fig",  default="../figures/validate_final.png")
    p.add_argument("--out_npz",  default="../validate_cfg.npz")
    args = p.parse_args()

    setup = ERTSetup("../code")
    try:
        unet, sch = load_cond_ddpm(args.cond_dir)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        alphas_cumprod = setup.alphas_cumprod
        print(f"Loaded conditional DDPM from {args.cond_dir}")

        mn = loadmat("../code/MNIST Data/mnist.mat")
        M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_train = mn['training']['labels'][0, 0].flatten()
        M_test  = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_test  = mn['test']['labels'][0, 0].flatten()
        if M_train.max() > 1.5: M_train = M_train / 255.0
        if M_test.max()  > 1.5: M_test  = M_test  / 255.0

        print("\nTraining TinyMNIST classifier (8 epochs)...")
        t0 = time.time()
        clf = train_classifier(M_train, L_train, epochs=8, verbose=False)
        print(f"  trained in {time.time()-t0:.0f}s")

        # Same held-out indices as the original validate_final.py
        test_indices = [int(np.where(L_test == d)[0][0]) for d in range(10)]
        results = []
        true_y_obs = setup.y_obs

        for digit in range(10):
            tidx = test_indices[digit]
            true_x = M_test[:, tidx].reshape(28, 28).astype(np.float32)
            print(f"\n========== digit {digit} (test idx {tidx}) ==========")
            t_total0 = time.time()

            with torch.no_grad():
                y = setup.forward(torch.tensor(true_x, dtype=torch.float32))
            setup.y_obs = y

            try:
                # Stage 1: PnP-DM-CFG, 32 chains cycling classes 0..9
                t0 = time.time()
                chains = pnp_dm_cfg(setup, unet, alphas_cumprod,
                                    schedule_name="geomspace",
                                    n_chains=args.n_chains, n_iter=args.n_iter,
                                    sigma_n=args.sigma_n, cfg_w=args.cfg_w,
                                    seed=0, verbose=False)
                best = chains[0]
                t_pnpdm = time.time() - t0
                print(f"  PnP-DM-CFG ({args.n_chains}x{args.n_iter}): "
                      f"best chain class={best['class_label']}, "
                      f"misfit {best['final_misfit']:.3e}  ({t_pnpdm:.0f}s)")

                # Classify pre-polish image
                top_classes, probs = classify_with_top_k(clf, best['x_pix'], k=10)
                recovered_digit = top_classes[0]
                recovered_prob = float(probs[recovered_digit])
                correct = (recovered_digit == digit)
                mark = 'OK' if correct else 'XX'
                print(f"  [{mark}]  Recovered class: {recovered_digit} "
                      f"(p={recovered_prob:.2f})  truth={digit}  "
                      f"target_class={best['class_label']}")

                # Stage 2: TV polish
                t0 = time.time()
                x_polished, log_tv = tv_polish(setup, best['x_pix'],
                                                K=args.K_tv, n_cg=60,
                                                tau=args.tau,
                                                target_misfit=1e-15,
                                                verbose=False)
                final_misfit = log_tv[-1][1]
                t_polish = time.time() - t0
                print(f"  TV polish (K={args.K_tv}): misfit {final_misfit:.3e}  "
                      f"({t_polish:.0f}s)")

                results.append({
                    'digit': digit,
                    'true_x': true_x,
                    'pnpdm_best': best['x_pix'],
                    'polished': x_polished,
                    'pnpdm_misfit': float(best['final_misfit']),
                    'final_misfit': float(final_misfit),
                    'recovered_digit': int(recovered_digit),
                    'recovered_prob': float(recovered_prob),
                    'best_target_class': int(best['class_label']),
                    'correct': bool(correct),
                    't_pnpdm': float(t_pnpdm),
                    't_polish': float(t_polish),
                })
                print(f"  total per-digit time: {time.time()-t_total0:.0f}s")
            finally:
                setup.y_obs = true_y_obs

        correct = sum(r['correct'] for r in results)
        print(f"\n========================================")
        print(f"NEW pipeline validation: {correct}/10 correct")
        print(f"Misfit stats: median {np.median([r['final_misfit'] for r in results]):.2e}, "
              f"max {max(r['final_misfit'] for r in results):.2e}")
        print(f"========================================")
        for r in results:
            m = 'OK' if r['correct'] else 'XX'
            print(f"  [{m}]  true {r['digit']}  recovered {r['recovered_digit']} "
                  f"(p={r['recovered_prob']:.2f}, target={r['best_target_class']})  "
                  f"pre {r['pnpdm_misfit']:.2e} -> polish {r['final_misfit']:.2e}")

        # Figure mirrors validate_final.png layout (3 rows x 10 cols)
        fig, axes = plt.subplots(3, 10, figsize=(18, 6.5))
        for k, r in enumerate(results):
            axes[0, k].imshow(r['true_x'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"{r['digit']}", fontsize=12)
            axes[1, k].imshow(r['pnpdm_best'], cmap='gray', vmin=0, vmax=1)
            c = [0, 0.55, 0] if r['correct'] else [0.85, 0, 0]
            axes[1, k].set_title(f"{r['recovered_digit']}", color=c,
                                  fontweight='bold', fontsize=12)
            axes[2, k].imshow(1.0 + r['polished'], cmap='viridis', vmin=1, vmax=2)
            axes[2, k].set_title(f"{r['final_misfit']:.1e}", fontsize=9)
            for row in range(3):
                axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
        axes[0, 0].set_ylabel('True (test)', fontsize=11)
        axes[1, 0].set_ylabel('PnP-DM-CFG best', fontsize=11)
        axes[2, 0].set_ylabel('After TV polish', fontsize=11)
        plt.suptitle(f'New pipeline (cond DDPM + CFG + TV polish) - {correct}/10 correct',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.out_fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"\nSaved {args.out_fig}")

        np.savez(args.out_npz, results=np.array(results, dtype=object),
                 correct_count=correct)
        print(f"Saved {args.out_npz}")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
