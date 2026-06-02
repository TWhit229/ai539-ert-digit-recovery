"""
Stage C of DI-RTG: classifier-guided template retrieval.

The Stage A diffusion samples identified the digit as a "5" (top-misfit seeds
are all 5-shaped). Stage C exploits this to do a TARGETED template search:
only over MNIST 5s (~6000 images) instead of all 60k. This is where the
diffusion prior earns its keep — it tells us the class so the template search
becomes 10× cheaper, AND we never even consider templates of other digits.

Pipeline:
  1. Load Stage A's multi-seed result.
  2. Train (or load) a small MNIST CNN classifier in a few minutes.
  3. Classify Stage A best → predicted class c_hat.
  4. Iterate over all MNIST training images of class c_hat, compute ERT misfit
     for each, keep the K=8 best.
  5. Save: best template, its index, predicted class, top-K candidates.

For robustness against misclassification:
  - If top-1 confidence < 0.9, also try the top-3 classes.
  - Pick the overall best by ERT misfit across all candidates.
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.io import loadmat
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from stage_a_daps_warmstart import ERTSetup


# ----------------------------------------------------------------------
# A tiny MNIST CNN (~30s to train on CPU, >99% test accuracy)
# ----------------------------------------------------------------------
class TinyMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = Fnn.relu(self.c1(x))
        x = Fnn.max_pool2d(x, 2)
        x = Fnn.relu(self.c2(x))
        x = Fnn.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = Fnn.relu(self.fc1(x))
        return self.fc2(x)


def train_classifier(M, L, epochs=8, batch_size=256, lr=1e-3, verbose=True):
    """Train TinyMNIST on MNIST training set. M: (784, N), L: (N,)."""
    torch.manual_seed(0)
    net = TinyMNIST()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    X = torch.tensor(M.T.reshape(-1, 1, 28, 28), dtype=torch.float32)
    Y = torch.tensor(L, dtype=torch.long)
    N = X.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(N)
        losses = []
        correct = 0; total = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], Y[idx]
            opt.zero_grad()
            logits = net(xb)
            loss = Fnn.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}  loss {np.mean(losses):.4f}  acc {correct/total:.4f}")
    net.eval()
    return net


# ----------------------------------------------------------------------
# Stage C main logic
# ----------------------------------------------------------------------
def stage_c(setup, x_init, M, L, classifier, K_topk_per_class=8,
            min_top1_conf=0.99, verbose=True):
    """
    Args:
      x_init    : (28,28) numpy Stage A output (in [0,1] pixel range)
      M         : (784, 60000) MNIST training images
      L         : (60000,) labels
      classifier: TinyMNIST instance
    Returns:
      best_template, best_idx, best_misfit, predicted_class, candidate_summary
    """
    # 1) Classify Stage A output
    x_t = torch.tensor(x_init.reshape(1, 1, 28, 28), dtype=torch.float32)
    with torch.no_grad():
        logits = classifier(x_t)
        probs = Fnn.softmax(logits, dim=1).numpy().flatten()
    top1_class = int(np.argmax(probs))
    top1_conf  = float(probs[top1_class])
    sorted_classes = np.argsort(-probs).tolist()

    if verbose:
        print(f"\nClassifier output:")
        for c in sorted_classes[:5]:
            print(f"  class {c}: p={probs[c]:.3f}")
        print(f"Predicted class: {top1_class} (confidence {top1_conf:.3f})")

    # 2) Decide which classes to search
    if top1_conf >= min_top1_conf:
        classes_to_search = [top1_class]
    else:
        classes_to_search = sorted_classes[:3]
    if verbose:
        print(f"Searching classes: {classes_to_search}")

    # 3) Targeted ERT misfit search within those classes
    candidates = []  # list of (idx, misfit, class)
    for c in classes_to_search:
        class_indices = np.where(L == c)[0]
        if verbose:
            print(f"\n  class {c}: scanning {len(class_indices)} templates via ERT misfit...")
        t0 = time.time()
        misfits = np.zeros(len(class_indices))
        for j, idx in enumerate(class_indices):
            template = M[:, idx].reshape(28, 28).astype(np.float32)
            if template.max() > 1.5: template = template / 255.0
            t = torch.tensor(template, dtype=torch.float32)
            y_pred = setup.forward(t)
            misfits[j] = 0.5 * float(((y_pred - setup.y_obs) ** 2).sum())
            if verbose and (j+1) % 1000 == 0:
                print(f"    {j+1}/{len(class_indices)}  best so far {misfits[:j+1].min():.3e}  "
                      f"({time.time()-t0:.1f}s)")
        order = np.argsort(misfits)
        for rank in range(min(K_topk_per_class, len(order))):
            candidates.append((int(class_indices[order[rank]]),
                               float(misfits[order[rank]]), int(c)))
        if verbose:
            print(f"    class {c} best misfit: {misfits.min():.3e} "
                  f"(template idx {class_indices[order[0]]})")

    candidates.sort(key=lambda x: x[1])
    if verbose:
        print(f"\nTop-{min(8, len(candidates))} candidates across searched classes:")
        for idx, m, c in candidates[:8]:
            print(f"  class {c}  idx {idx:5d}  misfit {m:.3e}")

    best_idx, best_misfit, best_class = candidates[0]
    best_template = M[:, best_idx].reshape(28, 28).astype(np.float32)
    if best_template.max() > 1.5: best_template = best_template / 255.0
    return best_template, best_idx, best_misfit, best_class, candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="stage_a_best.npz")
    p.add_argument("--mnist", default="MNIST Data/mnist.mat")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out", default="stage_c_result.npz")
    p.add_argument("--fig", default="../figures/stage_c_template.png")
    args = p.parse_args()

    setup = ERTSetup(".")
    try:
        # Load MNIST
        print("Loading MNIST training set...")
        mn = loadmat(args.mnist)
        imgs = mn['training']['images'][0, 0]                        # (28, 28, 60000)
        labels = mn['training']['labels'][0, 0].flatten()             # (60000,)
        M = imgs.astype(np.float32).reshape(784, -1)
        if M.max() > 1.5: M = M / 255.0
        print(f"  M shape {M.shape}  labels shape {labels.shape}")

        # Train classifier
        print("\nTraining TinyMNIST classifier...")
        t0 = time.time()
        clf = train_classifier(M, labels, epochs=args.epochs)
        print(f"  trained in {time.time()-t0:.1f}s")

        # Load Stage A initialization
        init_data = np.load(args.init, allow_pickle=True)
        x_init = init_data['x']
        print(f"\nStage A init shape={x_init.shape}  "
              f"misfit_recorded={float(init_data['misfit']):.3e}")

        # Run Stage C
        t0 = time.time()
        best_template, best_idx, best_misfit, best_class, candidates = stage_c(
            setup, x_init, M, labels, clf)
        elapsed = time.time() - t0

        print(f"\nStage C done in {elapsed:.1f}s.")
        print(f"  Best template: idx {best_idx}, class {best_class}, misfit {best_misfit:.3e}")
        print(f"  Project 1 found template #51138 (class 5) with misfit 2.07e-5.")

        np.savez(args.out,
                 x=best_template, idx=best_idx, misfit=best_misfit, predicted_class=best_class,
                 candidates=np.array(candidates), stage_a_init=x_init)
        print(f"Saved {args.out}")

        # Visualize: Stage A init | top-3 templates | best
        top3 = candidates[:3]
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        axes[0].imshow(x_init, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Stage A init\n(class {best_class} predicted)')
        for k, (idx, m, c) in enumerate(top3):
            tpl = M[:, idx].reshape(28, 28)
            axes[k+1].imshow(tpl, cmap='gray', vmin=0, vmax=1)
            axes[k+1].set_title(f'#{idx}  class {c}\nmisfit {m:.2e}')
        axes[4].imshow(best_template, cmap='gray', vmin=0, vmax=1)
        axes[4].set_title(f'BEST template\n#{best_idx}  misfit {best_misfit:.2e}',
                          fontweight='bold')
        for ax in axes:
            ax.axis('off')
        plt.suptitle('Stage C — classifier-guided template retrieval', fontsize=14)
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

    finally:
        setup.quit()


if __name__ == "__main__":
    main()
