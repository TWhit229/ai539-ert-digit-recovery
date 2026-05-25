# Project 1 Submission — ERT Digit Recovery

Recovering an MNIST-like conductivity image from ERT boundary voltage
measurements. This folder is self-contained.

## Contents

```
Project1_Submission/
├── Project1_Lesson.pdf         # FULL written lesson: complete theory + math,
│                               #   built from scratch in plain English
├── Project1_CompetitionTalk.pdf# the ~7-minute presentation (OSU theme)
├── Project1_Lesson.tex         # source for the lesson
├── Project1_CompetitionTalk.tex# source for the talk
├── make_teaching_figs.py       # regenerates the teaching diagrams
├── README.md
├── figures/                    # all figures used in the PDFs
└── code/                       # everything needed to run it
    ├── solve_competition.m  # MAIN: template-match + refine -> the answer
    ├── ERT2D.m              # forward model (PDE solver + adjoint Jacobian)
    ├── paramPackGenerator.m # builds the ERT parameter pack
    ├── y_truth_measurement.mat  # the competition measurement vector
    ├── MNIST Data/mnist.mat # 60k MNIST training images (the prior)
    ├── *.mat                # forward-model data (nodes, dipoles, etc.)
    │
    ├── ExactTemplateMatch.m # Stage 1 (digit identification) standalone
    ├── RefineFromTemplate.m # Stage 2 (refinement) standalone
    ├── FinalAnswer.m        # packages the answer figure
    ├── LambdaSweep.m        # generic regularizers (Tikhonov/L1/TV)
    ├── ParametricInversion.m# Route 1: PCA reduced-basis
    ├── train_vae.py         # Route 2: train the VAE (Python/PyTorch)
    ├── VAEInversion.m       # Route 2: VAE generative-prior inversion
    ├── TwoRouteComparison.m # the headline comparison figure
    ├── FullDemo.m           # render the setup + reconstruction animation
    └── LiveSolve.m          # live on-screen reconstruction
```

## How to run (the answer)

In MATLAB, from the `code/` folder:

```matlab
>> solve_competition
```

This:
1. Loads the measurement vector `y_truth_measurement.mat`.
2. **Stage 1 — template matching:** computes the exact ERT forward misfit of
   all 60,000 MNIST images and picks the best (identifies the digit + a clean
   starting image). ~15 min.
3. **Stage 2 — refinement:** free gradient descent from that template to
   recover the exact digit. ~1 min.
4. Saves `competition_answer.mat` and `competition_answer.png`.

## The method in one line

The unknown is essentially *one of the 60k MNIST images*, so we **template-
match to identify the digit and a correct starting point, then refine** with
gradient descent. This beats blind regularization (which lands on the wrong or
a blurry digit) because it avoids the multi-modality of the inverse problem.

## Requirements

- MATLAB (tested on R2026a). No special toolboxes needed for `solve_competition`.
- Python 3 with PyTorch + scipy *only* if retraining the VAE (`train_vae.py`).

## Key result

Generic regularizers give blobs; linear PCA gives the wrong digit at lowest
misfit; the VAE gives the correct but blurry digit; **template-match + refine
gives a clean, correct digit** at ~200× lower misfit than the TV baseline.
Central lesson: for this ill-posed problem, *data misfit is not reconstruction
quality* — the prior matters more.
```
