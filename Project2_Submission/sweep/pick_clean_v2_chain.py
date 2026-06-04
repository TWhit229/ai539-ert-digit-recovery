"""Pick the cleanest chain among the 32 cond+CFG runs on Competition 2.

'Cleanest' = highest classifier confidence on the majority-vote class
(which is 2 — 18 of 32 chains classify as 2). Tie-break by lowest misfit.

Save the chosen chain's image as the official Competition 2 answer
(`final_answer_v2_pre_polish.mat`) and regenerate the two v2 figures.
"""
import sys, os, time
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from pnp_dm_cfg import diff_one_denoise_cfg, load_cond_ddpm
from schedule import get_schedule
from train_cond_ddpm import N_CLASSES
from stage_c_template_retrieval import train_classifier
from run_di_rtg_v2 import classify_with_top_k


COND_DIR = "./ddpm_cond_rot15_ema"
N_CHAINS = 32
N_ITER = 80
MAJORITY = 2  # majority-vote class across the 32 chains


def mathsci(v):
    s = f"{v:.2e}"; a, b = s.split("e")
    return rf"${a}\times 10^{{{int(b)}}}$"


def main():
    setup = ERTSetup("../code")
    try:
        d = loadmat("../code/y_truth_measurement_noisy.mat")
        y_new = d['y_truth_noisy'].astype(np.float32).flatten()
        setup.y_obs = torch.tensor(y_new, dtype=torch.float32)

        unet, sch = load_cond_ddpm(COND_DIR)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        alphas_cumprod = setup.alphas_cumprod

        print("Training classifier...")
        mn = loadmat("../code/MNIST Data/mnist.mat")
        M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_train = mn['training']['labels'][0, 0].flatten()
        if M_train.max() > 1.5: M_train = M_train / 255.0
        clf = train_classifier(M_train, L_train, epochs=8, verbose=False)

        sigmas = get_schedule("geomspace", n_iter=N_ITER,
                              sigma_max=1.0, sigma_min=0.05)
        torch.manual_seed(0); np.random.seed(0)

        all_chains = []
        t0 = time.time()
        for c in range(N_CHAINS):
            class_label = c % N_CLASSES
            x = torch.randn(28, 28).clamp(-1.5, 1.5)
            for k, sigma_k in enumerate(sigmas):
                z = gn_likelihood_prox(setup, x, float(sigma_k),
                                        sigma_n=1e-4, max_inner=2)
                x = diff_one_denoise_cfg(unet, alphas_cumprod, z, float(sigma_k),
                                          class_label, w=3.0)
            x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
            mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                                 - setup.y_obs) ** 2).sum())
            top_classes, probs = classify_with_top_k(clf, x_pix, k=10)
            all_chains.append({
                'idx': c, 'target': class_label,
                'misfit': mis, 'img': x_pix,
                'clf_digit': int(top_classes[0]),
                'clf_prob_majority': float(probs[MAJORITY]),
                'clf_prob_max': float(probs[top_classes[0]]),
            })

        # Rank by classifier confidence on the majority class, then by misfit
        ranked = sorted(all_chains,
                         key=lambda r: (-r['clf_prob_majority'], r['misfit']))
        winner = ranked[0]
        print(f"\nWinner by (P(class=2) desc, misfit asc):")
        print(f"  chain {winner['idx']}  target={winner['target']}  "
              f"misfit {winner['misfit']:.3e}  "
              f"clf P(2)={winner['clf_prob_majority']:.3f}")
        print(f"\nTop 5 candidates by this ranking:")
        for r in ranked[:5]:
            print(f"  chain {r['idx']:2d}  target={r['target']}  "
                  f"misfit {r['misfit']:.3e}  P(2)={r['clf_prob_majority']:.3f}")

        # Save chosen image as the v2 answer
        img = winner['img']
        sigma_answer = 1.0 + img
        savemat("../final_answer_v2_pre_polish.mat", {
            'sigma_answer': sigma_answer,
            'x_answer': img,
            'final_misfit': float(winner['misfit']),
            'target_class': int(winner['target']),
            'classifier_digit': 2,
            'classifier_prob': float(winner['clf_prob_majority']),
            'note': 'cleanest of 32 cond+CFG chains by classifier confidence; '
                    'pre-polish (TV polish overshoots on noisy data)',
        })
        print("Saved ../final_answer_v2_pre_polish.mat (NEW chosen chain)")

        # Regenerate fig_v2_recovered
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].plot(np.arange(len(y_new)), y_new, lw=0.5, color='steelblue')
        axes[0].set_xlabel('measurement index $k$  (out of 1900)', fontsize=10)
        axes[0].set_ylabel('voltage $y_k$', fontsize=10)
        axes[0].set_title('The input: Competition 2 $y_{obs}$ (1900 noisy voltages)',
                          fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.3)
        im = axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].axis('off')
        axes[1].set_title(r'The output: recovered $\sigma$  (digit 2, '
                          + f'classifier $p={winner["clf_prob_majority"]:.2f}$' + ')\n'
                          + 'pre-polish misfit ' + mathsci(winner['misfit']),
                          fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        for d_out in ('../figures', '../figures/lesson'):
            os.makedirs(d_out, exist_ok=True)
            out = f"{d_out}/fig_v2_recovered.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
        plt.close()

        # Regenerate v1 vs v2 side-by-side
        v1 = loadmat('../final_answer.mat')
        v1_sigma = v1.get('sigma_answer')
        if v1_sigma is None: v1_sigma = 1.0 + v1['x_answer']
        v1_misfit = float(np.asarray(v1['final_misfit']).flat[0])
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
        axes[0].imshow(v1_sigma, cmap='viridis', vmin=1, vmax=2)
        axes[0].set_title('Worked example: digit-5 test vector\n'
                          '(noiseless), polished misfit ' + mathsci(v1_misfit),
                          fontsize=10)
        axes[0].axis('off')
        axes[1].imshow(sigma_answer, cmap='viridis', vmin=1, vmax=2)
        axes[1].set_title('Competition 2 actual vector (noisy)\n'
                          'digit 2, pre-polish misfit ' + mathsci(winner['misfit']),
                          fontsize=10)
        axes[1].axis('off')
        plt.suptitle('Same pipeline, two vectors',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        for d_out in ('../figures', '../figures/lesson'):
            out = f"{d_out}/fig_v2_vs_v1.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
        plt.close()
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
