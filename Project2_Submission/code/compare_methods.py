"""
Compare all DI-RTG variants and the PnP-DM baseline on the competition vector.
Pulls the saved .mat / .npz files from each method, plots side-by-side, and
prints a misfit table.

Run after all candidate methods have produced their result file:
  - DI-RTG v1:  competition_answer.mat
  - DI-RTG v2:  competition_answer_v2.mat
  - PnP-DM:     pnp_dm_result.npz (optional)
  - P1 (rotation): ../../Project1_Submission/rotated_answer.mat (reference)
"""
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


METHODS = [
    {
        'name': 'P1 rotation-aware\n(reference)',
        'load': lambda: loadmat('../../Project1_Submission/rotated_answer.mat'),
        'extract': lambda d: (d['sigma_answer_rot'], float(d['final_misfit_rot'].flatten()[0]),
                              int(d['digit_rot'].flatten()[0])),
    },
    {
        'name': 'DI-RTG v1\n(diffusion+classifier)',
        'load': lambda: loadmat('../competition_answer.mat'),
        'extract': lambda d: (d['sigma_answer'], float(d['final_misfit'].flatten()[0]),
                              int(d['digit'].flatten()[0])),
    },
    {
        'name': 'DI-RTG v2\n(NOT diffusion-load-bearing:\ntemplate match + rotation)',
        'load': lambda: loadmat('../competition_answer_v2.mat'),
        'extract': lambda d: (d['sigma_answer'], float(d['final_misfit'].flatten()[0]),
                              int(d['digit'].flatten()[0])),
    },
    {
        'name': 'DAPS proper\n(diffusion sampler)',
        'load': lambda: loadmat('../experiments/intermediate_results/daps_v2_answer.mat'),
        'extract': lambda d: (d['sigma_answer'], float(d['final_misfit'].flatten()[0]), -1),
    },
    {
        'name': 'DPS\n(diffusion sampler)',
        'load': lambda: loadmat('../experiments/intermediate_results/dps_answer.mat'),
        'extract': lambda d: (d['sigma_answer'], float(d['final_misfit'].flatten()[0]), -1),
    },
    {
        'name': 'PnP-DM v3 (pre-polish)\n(rot-aug DDPM, 32ch×80,\n+ LM polish)',
        'load': lambda: loadmat('../pnp_dm_v3_answer.mat'),
        'extract': lambda d: (d['sigma_answer'], float(d['final_misfit'].flatten()[0]), -1),
    },
    {
        'name': 'PnP-DM + GN-CG polish\n(FINAL — float64 floor)',
        'load': lambda: loadmat('../final_answer.mat'),
        'extract': lambda d: (d['sigma_answer'], float(d['final_misfit'].flatten()[0]), -1),
    },
]


def main():
    out_dir = Path('..')
    results = []
    for m in METHODS:
        try:
            data = m['load']()
            sigma, mis, digit = m['extract'](data)
            results.append({'name': m['name'], 'sigma': sigma, 'misfit': mis, 'digit': digit})
            print(f"  {m['name']:35s} → digit {digit:>3}  misfit {mis:.3e}")
        except FileNotFoundError as e:
            print(f"  {m['name']:35s} → FILE MISSING ({e})")

    if not results:
        print("No methods produced output yet."); return

    # Side-by-side plot
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5.2))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        ax.imshow(r['sigma'], cmap='viridis', vmin=1, vmax=2)
        digit_str = f"digit {r['digit']}" if r['digit'] >= 0 else "(no digit label)"
        ax.set_title(f"{r['name']}\n{digit_str}\nmisfit {r['misfit']:.3e}", fontsize=11)
        ax.axis('off')
    plt.suptitle("Competition recovery — method comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = out_dir / 'figures' / 'method_comparison.png'
    plt.savefig(fig_path, dpi=130, bbox_inches='tight', facecolor='white')
    print(f"\nSaved {fig_path}")

    # Also print a clean misfit table
    print("\n=== Misfit table ===")
    print(f"{'Method':38s}  {'Misfit':>12s}  {'Digit':>6s}")
    print("-" * 60)
    for r in results:
        print(f"{r['name'].replace(chr(10), ' '):38s}  {r['misfit']:.3e}  {r['digit']:>6}")


if __name__ == "__main__":
    main()
