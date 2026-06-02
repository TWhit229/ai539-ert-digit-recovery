"""
Stage C v2: class-restricted template search WITH rotation.

Why this exists:
  Project 1's `find_rotation.m` showed that the competition vector's true answer
  is digit 5 *rotated by ~7.5°*. Upright-only template matching plateaued at
  misfit 1.76e-6. With rotation it dropped to 1.01e-7.

  DI-RTG v1's Stage C used upright templates only — it inherits the 1.83e-6 floor
  for the same reason.

Algorithm:
  - Take the top classes (from classifier, possibly iterative refinement).
  - For each class:
      Phase A: cheap upright-misfit scan of every training image of that class.
      Phase B: rotation search (coarse 10° → fine 0.5°) on the top-K_upright
               survivors only.
  - Return the global best (template_idx, rotation_deg, misfit) and the rotated
    template image.

This stays compatible with Stage D LM polish — pass the rotated template as the
initial x.
"""
import time
import numpy as np
import torch
import torch.nn.functional as Fnn

from utils_rotation import rotate28, rotate28_batch


def _ert_misfit(setup, img28):
    """Half-sum-of-squares ERT misfit of a (28,28) numpy image."""
    t = torch.tensor(img28, dtype=torch.float32)
    y_pred = setup.forward(t)
    return 0.5 * float(((y_pred - setup.y_obs) ** 2).sum())


def stage_c_rotation(setup, x_init, M_train, L_train, classifier,
                     classes_to_search=None, min_top1_conf=0.99,
                     K_upright_per_class=50,
                     coarse_angles=None, fine_step=0.5, fine_half_window=9,
                     verbose=True):
    """
    Args:
      setup           : ERTSetup
      x_init          : (28,28) numpy, used as classifier input
      M_train, L_train: MNIST training set (784 x N), labels (N,)
      classifier      : TinyMNIST instance (for class prediction)
      classes_to_search: explicit list of classes (overrides classifier if set)
      min_top1_conf   : if top-1 confidence below this, also try top-2 classes
      K_upright_per_class: # of templates to carry into rotation search per class
      coarse_angles   : numpy array of coarse angles to scan (default 0..350 step 10)
      fine_step       : step size for fine rotation refinement (deg)
      fine_half_window: search +/- this many degrees around coarse winner
    Returns:
      dict with best_template (rotated, 28x28), best_idx, best_theta, best_class,
      best_misfit, candidates (top-10 list of dicts).
    """
    if coarse_angles is None:
        coarse_angles = np.arange(0, 360, 10).astype(float)

    # ---------- 1. Classify Stage A image to pick which classes to search ----------
    if classes_to_search is None:
        x_t = torch.tensor(x_init.reshape(1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            logits = classifier(x_t)
            probs = Fnn.softmax(logits, dim=1).numpy().flatten()
        top1 = int(np.argmax(probs))
        top1_conf = float(probs[top1])
        if top1_conf >= min_top1_conf:
            classes_to_search = [top1]
        else:
            classes_to_search = list(np.argsort(-probs)[:2])
        if verbose:
            print(f"Classifier top-3: " +
                  ", ".join(f"{c}:{probs[c]:.3f}" for c in np.argsort(-probs)[:3]))
            print(f"Searching classes: {classes_to_search}  (conf {top1_conf:.3f})")

    # ---------- 2. Phase A: upright misfit per template, per class ----------
    candidates = []   # (class, idx, theta, misfit)
    t_all = time.time()
    for c in classes_to_search:
        class_idx = np.where(L_train == c)[0]
        if verbose:
            print(f"\n[class {c}] phase A: upright scan over {len(class_idx)} templates")
        misfits_up = np.empty(len(class_idx))
        t0 = time.time()
        for j, idx in enumerate(class_idx):
            tpl = M_train[:, idx].reshape(28, 28).astype(np.float32)
            misfits_up[j] = _ert_misfit(setup, tpl)
            if verbose and (j + 1) % 1500 == 0:
                print(f"  {j+1}/{len(class_idx)}  best so far {misfits_up[:j+1].min():.3e}  "
                      f"({time.time()-t0:.0f}s)")
        order = np.argsort(misfits_up)
        topK = order[:K_upright_per_class]
        if verbose:
            print(f"  class {c} best upright misfit: {misfits_up[topK[0]]:.3e}  "
                  f"(template {class_idx[topK[0]]})")

        # ---------- 3. Phase B: coarse rotation search on top-K templates ----------
        if verbose:
            print(f"[class {c}] phase B: rotation search over top-{K_upright_per_class} "
                  f"× {len(coarse_angles)} angles")
        best_per_template = []  # (idx, best_theta_coarse, best_misfit_coarse)
        t1 = time.time()
        for j_idx, j in enumerate(topK):
            idx = int(class_idx[j])
            tpl = M_train[:, idx].reshape(28, 28).astype(np.float32)
            misfits_rot = np.empty(len(coarse_angles))
            for ai, theta in enumerate(coarse_angles):
                if theta == 0:
                    rot = tpl
                else:
                    rot = np.clip(rotate28(tpl, theta), 0.0, 1.0).astype(np.float32)
                misfits_rot[ai] = _ert_misfit(setup, rot)
            ai_best = int(np.argmin(misfits_rot))
            best_per_template.append((idx, float(coarse_angles[ai_best]),
                                      float(misfits_rot[ai_best])))
            if verbose and (j_idx + 1) % 10 == 0:
                cur_min = min(b[2] for b in best_per_template)
                print(f"  template {j_idx+1}/{len(topK)}  best class min so far "
                      f"{cur_min:.3e}  ({time.time()-t1:.0f}s)")

        # ---------- 4. Phase C: fine rotation refinement on the class winner ----------
        best_per_template.sort(key=lambda r: r[2])
        winner_idx, winner_theta_coarse, _ = best_per_template[0]
        tpl_win = M_train[:, winner_idx].reshape(28, 28).astype(np.float32)
        fine_angles = np.arange(winner_theta_coarse - fine_half_window,
                                winner_theta_coarse + fine_half_window + fine_step,
                                fine_step)
        misfits_fine = np.empty(len(fine_angles))
        for ai, theta in enumerate(fine_angles):
            rot = np.clip(rotate28(tpl_win, theta), 0.0, 1.0).astype(np.float32)
            misfits_fine[ai] = _ert_misfit(setup, rot)
        ai_best = int(np.argmin(misfits_fine))
        best_theta_class = float(fine_angles[ai_best])
        best_misfit_class = float(misfits_fine[ai_best])
        if verbose:
            print(f"  class {c} fine winner: template {winner_idx}, "
                  f"theta {best_theta_class:.1f}°, misfit {best_misfit_class:.3e}")

        # Collect this class's results
        for (idx, theta_c, m_c) in best_per_template[:3]:
            candidates.append({'class': int(c), 'idx': idx,
                               'theta': theta_c, 'misfit': m_c, 'phase': 'coarse'})
        candidates.append({'class': int(c), 'idx': winner_idx,
                           'theta': best_theta_class, 'misfit': best_misfit_class,
                           'phase': 'fine'})

    # ---------- 5. Global winner ----------
    candidates.sort(key=lambda r: r['misfit'])
    best = candidates[0]
    tpl = M_train[:, best['idx']].reshape(28, 28).astype(np.float32)
    if best['theta'] == 0:
        best_template = tpl
    else:
        best_template = np.clip(rotate28(tpl, best['theta']), 0.0, 1.0).astype(np.float32)

    if verbose:
        print(f"\n=== Stage C v2 winner ===")
        print(f"  class {best['class']}  template #{best['idx']}  "
              f"theta {best['theta']:.1f}°  misfit {best['misfit']:.3e}")
        print(f"  ({time.time()-t_all:.0f}s total)")

    return {
        'best_template': best_template,
        'best_idx':     int(best['idx']),
        'best_theta':   float(best['theta']),
        'best_class':   int(best['class']),
        'best_misfit':  float(best['misfit']),
        'candidates':   candidates[:10],
        'classes_searched': list(classes_to_search),
    }
