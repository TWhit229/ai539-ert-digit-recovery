# Project 2 Submission — DI-RTG (Diffusion-prior ERT inversion)

Recovering an MNIST-like conductivity image from ERT boundary voltage measurements
using a **diffusion-based prior** (Project 2 requirement).

Authors: Travis Whitney, Cole Seifert, Alexander Nutt (AI 539, Oregon State).

---

## The method: DI-RTG

```
  DAPS warm-start  →  Classifier-guided template retrieval  →  LM polish
  (Stage A)           (Stage C)                                (Stage D)
```

A four-stage hybrid where the **diffusion model contributes the prior** by
(a) generating digit-shaped sample candidates and (b) narrowing the template
search to the predicted class. Stage B (GN refinement with a diffusion anchor)
is in the codebase but skipped for v1 — the anchor pulled toward the upright-MNIST
manifold and away from the true (rotated) answer.

| Stage | What it does | Misfit reached |
|---|---|---|
| **A** — DAPS warm-start | Short reverse-diffusion (`K_outer=30`, `K_inner=3`, `η=0.5·σ²`, `λ=1`) from each of 12 seeds; keep best by ERT misfit | 2.55e-3 |
| **C** — Targeted template retrieval | Tiny MNIST CNN classifier (98.5% test acc); top-3 classes → ERT misfit over ~17k training images instead of all 60k | 2.07e-5 |
| **D** — LM polish | Heavy-ball gradient descent à la Project 1 (`η=5, γ=0.95, K=1500`) | **1.83e-6** |
| Project 1 reference | brute-force 60k templates + LM | 1.76e-6 |

**Final result on the professor's competition vector: digit 5, misfit 1.83e-6.**

DI-RTG is a diffusion-based prior method (per the Project 2 rule), independently
arrives at the same answer Project 1 found, and reduces the template search by ~3.5×.

---

## Contents

```
Project2_Submission/
├── README.md                       # this file
├── competition_answer.mat          # FINAL ANSWER: σ_answer, digit=5, misfit=1.83e-6
├── Project2_Plan.md                # DI-RTG plan + research synthesis
├── figures/
│   ├── di_rtg_pipeline.png         # 4-panel pipeline summary
│   ├── prior_smoketest.png         # pretrained DDPM smoke test
│   ├── stage_a_seed_sweep.png      # 12-seed Stage A — all results
│   ├── stage_c_template.png        # Stage C retrieval result
│   ├── stage_d_from_C.png          # Stage D trajectory + final image
│   └── validate_di_rtg.png         # 10-digit held-out validation
└── code/
    ├── run_di_rtg.py               # MAIN: end-to-end pipeline driver
    ├── validate_di_rtg.py          # held-out validation on 10 unseen digits
    ├── stage_a_daps_warmstart.py   # Stage A + shared ERTSetup class
    ├── stage_b_gn_refinement.py    # Stage B (skipped in v1; kept for record)
    ├── stage_c_template_retrieval.py # Stage C + tiny MNIST CNN
    ├── stage_d_lm_polish.py        # Stage D LM polish
    ├── ERT_call.m                  # MATLAB-side wrapper, caches ERTParams
    ├── ERT2D.m, paramPackGenerator.m, *.mat     # forward model + adjoint
    ├── y_truth_measurement.mat     # competition vector (byte-identical to ~/Downloads)
    └── MNIST Data/mnist.mat        # MNIST 60k train + 10k test
```

---

## How to run

End-to-end:

```bash
cd code
python3 run_di_rtg.py
```

Total wall time on CPU: **~6 minutes** (Stage A 12 seeds + classifier training +
class-restricted template search + 1500 LM iters).

To validate on 10 held-out digits:

```bash
python3 validate_di_rtg.py    # ~30-60 min (full DI-RTG per digit)
```

### Requirements

- MATLAB R2026a (tested) with the Python engine installed
  (`cd $MATLABROOT/extern/engines/python && pip install --user .`).
- Python 3.10+ with `torch`, `diffusers`, `scipy`, `numpy`, `matplotlib`, `huggingface_hub`.
- ~5 MB to pull `1aurent/ddpm-mnist` from Hugging Face on first run.

---

## Why this beats vanilla DPS / DAPS for our setting

The DI-RTG plan started from a multi-agent research workflow surveying 15+ papers
(`Project2_Plan.md`). DAPS alone (arXiv 2407.01521) was the front-runner, but for
our specific setting it's *over-engineered for the forward model* (its Langevin
loop wastes our analytic Jacobian) and *under-engineered for the prior* (the
professor explicitly told us x is "very similar to one of the MNIST samples" —
strongest possible class prior).

DI-RTG strips DAPS to a warm-start (Stage A) and bolts on:

1. A **classifier** (~30 s to train, 98.5 % test acc) that turns the Stage-A
   sample into a class identification — a property the diffusion prior trivially
   gives but DAPS doesn't exploit.
2. A **class-restricted template search** that's 3.5× cheaper than Project 1's
   brute-force 60k search.
3. **Project 1's LM polish verbatim** because for this problem its near-MAP
   refinement is provably right and we have the analytic Jacobian.

The key implementation gotcha was a **Jacobian column-ordering bug** —
MATLAB's `ERT2D` returns J indexed column-major (pixel (i,j) ↔ column j·28+i),
while numpy's reshape is row-major. Permuting J's columns once inside
`ERTSetup.forward_and_jacobian` fixed every downstream stage at once.

---

## What's in `competition_answer.mat`

| variable | meaning |
|---|---|
| `sigma_answer` (28×28) | recovered conductivity image (`σ_bg + x`) |
| `x_answer` (28×28) | conductivity contrast |
| `digit` | recovered digit class: **5** |
| `final_misfit` | `½‖F(σ_answer) − y_obs‖²` = **1.83e-6** |
| `template_idx` | MNIST training index of Stage C winner = **51137** (matches Project 1's m_51138 in 1-indexed) |
| `stage_a_seed` | best Stage A seed = **1** |
| `stage_a_misfit`, `stage_c_misfit` | per-stage misfits for inspection |

---

## Relationship to Project 1

Project 1 found the same answer (digit 5, refined misfit 1.76e-6) by template
matching over all 60k MNIST training images. Project 2 (DI-RTG) arrives
independently by a diffusion-prior route — necessary to satisfy the Project 2
rule that the prior be diffusion-based.

The diffusion model contributes specifically the digit-class identification.
Once that's pinned down, the rest (template retrieval + LM polish) is
Project 1's machinery. This is honest about where the diffusion adds value:
**class identification**, not the final misfit driving.

---

## Key references

- **DAPS** — arXiv `2407.01521` (Zhang et al., CVPR 2025 Oral)
- **ReSample** — arXiv `2307.08123` (Song et al., ICLR 2024)
- **RED-DiffEq** — arXiv `2509.21659` (Shan/Zhu/Lin/Lu 2025)
- **PnP-DM** — arXiv `2405.18782` (Wu et al., NeurIPS 2024)
- **DPnP** — arXiv `2403.17042` (Xu/Chi, NeurIPS 2024)
- **InverseBench** (leaderboard) — arXiv `2503.11043`
- Pretrained MNIST DDPM: `1aurent/ddpm-mnist` on HuggingFace
