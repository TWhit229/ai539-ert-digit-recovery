"""
Validate the FINAL pipeline (rot-aug DDPM + PnP-DM Split-Gibbs MCMC + GN-CG polish)
on 10 held-out MNIST test digits.

For each digit class 0–9:
  1. Pick one unseen test image with that label.
  2. Synthesize y = F(σ_bg + test_image).
  3. Run PnP-DM with rot-aug DDPM (n_chains × n_iter Gibbs iters).
  4. Pick lowest-misfit chain.
  5. GN-CG polish to float64 floor.
  6. Check (a) whether the recovered digit is qualitatively correct,
           (b) what final misfit we hit.

Output: 10-digit summary + figure.
"""
import argparse, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup
from pnp_dm import pnp_dm
from gn_cg_polish import gn_cg_polish
from stage_c_template_retrieval import train_classifier
from run_di_rtg_v2 import classify_with_top_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm_path",   default="./ddpm_mnist_rot15")
    p.add_argument("--n_chains",    type=int, default=16,
                   help="PnP-DM chains per digit (16 for speed; v3 used 32)")
    p.add_argument("--n_iter",      type=int, default=50,
                   help="PnP-DM Gibbs iters per chain (50 for speed; v3 used 80)")
    p.add_argument("--sigma_n",     type=float, default=1e-4)
    p.add_argument("--K_polish",    type=int, default=300)
    p.add_argument("--n_cg",        type=int, default=100)
    p.add_argument("--target_misfit", type=float, default=1e-13)
    p.add_argument("--out_fig",     default="../figures/validate_final.png")
    p.add_argument("--out_npz",     default="../validate_final.npz")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Swap in fine-tuned DDPM
        print(f"Loading fine-tuned DDPM from {args.ddpm_path}...")
        pipe = DDPMPipeline.from_pretrained(args.ddpm_path)
        setup.unet = pipe.unet.eval()
        setup.scheduler = pipe.scheduler
        setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
        for p_ in setup.unet.parameters():
            p_.requires_grad_(False)

        # Load MNIST
        mn = loadmat("MNIST Data/mnist.mat")
        M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_train = mn['training']['labels'][0, 0].flatten()
        M_test  = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_test  = mn['test']['labels'][0, 0].flatten()
        if M_train.max() > 1.5: M_train = M_train / 255.0
        if M_test.max()  > 1.5: M_test  = M_test  / 255.0

        # Train classifier (only for class-identification check, not pipeline use)
        print("\nTraining TinyMNIST classifier (8 epochs)...")
        t0 = time.time()
        clf = train_classifier(M_train, L_train, epochs=8, verbose=False)
        print(f"  trained in {time.time()-t0:.0f}s")

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
                # --- Stage 1: PnP-DM (rot-aug DDPM + Split-Gibbs MCMC) ---
                t0 = time.time()
                chains = pnp_dm(setup, n_chains=args.n_chains, n_iter=args.n_iter,
                                 sigma_n=args.sigma_n, gn_inner=2, seed=0, verbose=False)
                best = chains[0]
                t_pnpdm = time.time() - t0
                print(f"  PnP-DM ({args.n_chains}×{args.n_iter}): best chain misfit "
                      f"{best['final_misfit']:.3e}  ({t_pnpdm:.0f}s)")

                # Classify best chain to identify recovered digit
                top_classes, probs = classify_with_top_k(clf, best['x_pix'], k=10)
                recovered_digit = top_classes[0]
                recovered_prob = float(probs[recovered_digit])
                correct = (recovered_digit == digit)
                mark = '✓' if correct else '✗'
                print(f"  {mark}  Recovered digit class: {recovered_digit} "
                      f"(p={recovered_prob:.2f})  truth={digit}")

                # --- Stage 2: GN-CG polish ---
                t0 = time.time()
                x_polished, log_polish = gn_cg_polish(
                    setup, best['x_pix'], K=args.K_polish, n_cg=args.n_cg,
                    lam_init=1e-5, lam_min=1e-12,
                    target_misfit=args.target_misfit,
                    log_every=args.K_polish, verbose=False)
                final_misfit = log_polish[-1][1]
                t_polish = time.time() - t0
                print(f"  GN-CG polish ({args.K_polish}):  misfit {final_misfit:.3e}  "
                      f"({t_polish:.0f}s)")

                results.append({
                    'digit': digit, 'true_x': true_x, 'pnpdm_best': best['x_pix'],
                    'polished': x_polished,
                    'pnpdm_misfit': float(best['final_misfit']),
                    'final_misfit': float(final_misfit),
                    'recovered_digit': int(recovered_digit),
                    'recovered_prob': float(recovered_prob),
                    'correct': bool(correct),
                    't_pnpdm': float(t_pnpdm),
                    't_polish': float(t_polish),
                })
            finally:
                setup.y_obs = true_y_obs

        # Summary
        correct = sum(r['correct'] for r in results)
        print(f"\n========================================")
        print(f"Final pipeline validation: {correct}/10 correct")
        print(f"Floor stats: median {np.median([r['final_misfit'] for r in results]):.2e}, "
              f"max {max(r['final_misfit'] for r in results):.2e}")
        print(f"========================================")
        for r in results:
            m = '✓' if r['correct'] else '✗'
            print(f"  {m}  true {r['digit']}  recovered {r['recovered_digit']} "
                  f"(p={r['recovered_prob']:.2f})  "
                  f"PnP-DM {r['pnpdm_misfit']:.2e} → polish {r['final_misfit']:.2e}")

        # Save figure
        fig, axes = plt.subplots(3, 10, figsize=(18, 6.5))
        for k, r in enumerate(results):
            axes[0, k].imshow(r['true_x'], cmap='gray', vmin=0, vmax=1)
            axes[0, k].set_title(f"{r['digit']}", fontsize=12)
            axes[1, k].imshow(r['pnpdm_best'], cmap='gray', vmin=0, vmax=1)
            c = [0, 0.55, 0] if r['correct'] else [0.85, 0, 0]
            axes[1, k].set_title(f"{r['recovered_digit']}", color=c, fontweight='bold', fontsize=12)
            axes[2, k].imshow(1.0 + r['polished'], cmap='viridis', vmin=1, vmax=2)
            axes[2, k].set_title(f"{r['final_misfit']:.1e}", fontsize=9)
            for row in range(3):
                axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
        axes[0, 0].set_ylabel('True (test)', fontsize=11)
        axes[1, 0].set_ylabel('PnP-DM best', fontsize=11)
        axes[2, 0].set_ylabel('After GN-CG polish', fontsize=11)
        plt.suptitle(f'Final pipeline (rot-aug DDPM + PnP-DM + GN-CG) — {correct}/10 correct',
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
