"""Visualize the 12 polished images side by side on a 3x4 grid:
rows = prior (base / bigger / cond), columns = (schedule, polish).

Saves figures/sweep_grid.png and figures/sweep_pre_vs_post.png.
"""
import argparse, os, json
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


PRIORS    = ["base", "bigger", "cond"]
SCHEDULES = ["geomspace", "neg_rho"]
POLISHES  = ["lm", "tv"]


def try_load(path):
    if not os.path.isfile(path):
        return None
    return loadmat(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--out_grid",    default="./sweep_grid.png")
    p.add_argument("--out_pre_post", default="./sweep_pre_vs_post.png")
    args = p.parse_args()

    # 3 rows (prior) × 4 cols (schedule x polish)
    col_labels = [f"{s}\n{pol}" for s in SCHEDULES for pol in POLISHES]
    fig, axes = plt.subplots(len(PRIORS), 4, figsize=(13, 9.5))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i, prior in enumerate(PRIORS):
        for j, (s, pol) in enumerate([(s, pol) for s in SCHEDULES for pol in POLISHES]):
            path = f"{args.results_dir}/polished/{prior}_{s}_{pol}.mat"
            d = try_load(path)
            ax = axes[i, j]
            if d is None:
                ax.text(0.5, 0.5, "(missing)", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                img = d.get('sigma_answer', None)
                if img is None:
                    img = 1.0 + d['x_answer']
                ax.imshow(img, cmap='viridis', vmin=1.0, vmax=2.0)
                m = float(np.array(d['final_misfit']).flat[0])
                ax.set_title(f"misfit {m:.2e}", fontsize=9)
                ax.axis('off')
            if i == 0:
                ax.set_title((ax.get_title() + f"\n{col_labels[j]}").strip(),
                             fontsize=9)
        axes[i, 0].set_ylabel(prior, fontsize=12, rotation=90, labelpad=12)

    plt.suptitle("12-combo sweep (rows = prior; cols = schedule / polish)",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out_grid, dpi=130, bbox_inches='tight', facecolor='white')
    print(f"Saved {args.out_grid}")

    # Pre vs post for each (prior, schedule)
    fig2, axes2 = plt.subplots(len(PRIORS), 6, figsize=(15, 8))
    for i, prior in enumerate(PRIORS):
        for k, s in enumerate(SCHEDULES):
            # pre-polish (1 col)
            pp_path = f"{args.results_dir}/pre_polish/{prior}_{s}.mat"
            d = try_load(pp_path)
            ax = axes2[i, k * 3]
            if d is None:
                ax.text(0.5, 0.5, "(missing)", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                img = d.get('sigma_answer', 1.0 + d['x_answer'])
                ax.imshow(img, cmap='viridis', vmin=1, vmax=2)
                m = float(np.array(d['final_misfit']).flat[0])
                ax.set_title(f"{prior}/{s}\npre  {m:.1e}", fontsize=8)
                ax.axis('off')
            # lm and tv polished
            for jj, pol in enumerate(["lm", "tv"]):
                ax = axes2[i, k * 3 + 1 + jj]
                path = f"{args.results_dir}/polished/{prior}_{s}_{pol}.mat"
                d = try_load(path)
                if d is None:
                    ax.text(0.5, 0.5, "(missing)", ha='center', va='center',
                            transform=ax.transAxes, fontsize=10, color='gray')
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                img = d.get('sigma_answer', 1.0 + d['x_answer'])
                ax.imshow(img, cmap='viridis', vmin=1, vmax=2)
                m = float(np.array(d['final_misfit']).flat[0])
                ax.set_title(f"polish={pol}\n{m:.1e}", fontsize=8)
                ax.axis('off')

    plt.suptitle("Pre-polish vs LM polish vs TV polish",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out_pre_post, dpi=130, bbox_inches='tight', facecolor='white')
    print(f"Saved {args.out_pre_post}")

    # Print sorted summary if present
    js = f"{args.results_dir}/sweep_summary.json"
    if os.path.isfile(js):
        with open(js) as f:
            rows = json.load(f)
        print("\nFinal misfits (sorted):")
        for r in sorted(rows, key=lambda r: r['final_misfit']):
            print(f"  {r['prior']:6s}.{r['schedule']:9s}.{r['polish']:2s}  "
                  f"pre {r['pre_polish_misfit']:.2e}  "
                  f"polished {r['final_misfit']:.2e}")


if __name__ == "__main__":
    main()
