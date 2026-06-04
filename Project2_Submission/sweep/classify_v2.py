"""Classify the v2 pre-polish recovery using the same TinyMNIST classifier
the held-out validation script uses."""
import sys, os
sys.path.insert(0, "../code")

import numpy as np
from scipy.io import loadmat
from stage_c_template_retrieval import train_classifier
from run_di_rtg_v2 import classify_with_top_k


def main():
    print("Training TinyMNIST classifier...")
    mn = loadmat("../code/MNIST Data/mnist.mat")
    M_train = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
    L_train = mn['training']['labels'][0, 0].flatten()
    if M_train.max() > 1.5: M_train = M_train / 255.0
    clf = train_classifier(M_train, L_train, epochs=8, verbose=False)
    print("  classifier ready")

    d = loadmat("../final_answer_v2_pre_polish.mat")
    img = d['x_answer'].astype(np.float32)
    print(f"\nv2 recovered image: shape {img.shape}, "
          f"range [{img.min():.3f}, {img.max():.3f}]")
    top_classes, probs = classify_with_top_k(clf, img, k=10)
    print(f"\nTop classifier predictions:")
    for c in top_classes[:5]:
        print(f"  digit {c}: p = {probs[c]:.3f}")


if __name__ == "__main__":
    main()
