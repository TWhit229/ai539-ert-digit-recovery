# Project 1 Submission — ERT Digit Recovery

Recovering an MNIST-like conductivity image from ERT boundary voltage
measurements. This folder is self-contained.

Authors: Travis Whitney, Cole Seifert, Alexander Nutt (AI 539, Oregon State).

## Contents

```
Project1_Submission/
├── Project1_Lesson.pdf          # FULL written lesson: theory + math, plain English
├── Project1_CompetitionTalk.pdf # the ~7-minute presentation (OSU theme)
├── competition_answer.mat       # THE ANSWER: digit=5, sigma_answer (28x28), etc.
├── Project1_Lesson.tex          # source for the lesson
├── Project1_CompetitionTalk.tex # source for the talk
├── make_teaching_figs.py        # regenerates the teaching diagrams
├── README.md
├── figures/                     # all figures used in the PDFs
└── code/                        # everything needed to run it
    ├── solve_competition.m      # MAIN: template-match + refine -> the answer
    ├── validate_0to9.m          # held-out test: recover 10 unseen digits (9/10)
    ├── ERT2D.m                  # forward model (PDE solver + adjoint Jacobian)
    ├── paramPackGenerator.m     # builds the ERT parameter pack
    ├── y_truth_measurement.mat  # the competition measurement vector
    ├── MNIST Data/mnist.mat     # 60k training + 10k test MNIST images (the prior)
    ├── nodes.mat
    ├── dipole_configuration.mat
    └── dipole_voltages.mat      # sensor geometry for the forward model
```

## How to run

In MATLAB, from the `code/` folder.

**The competition answer:**

```matlab
>> solve_competition
```

1. Loads the measurement vector `y_truth_measurement.mat`.
2. **Stage 1 (template matching):** exact ERT forward misfit of all 60,000 MNIST
   images; keeps the best (identifies the digit + a clean starting image). ~8 min.
3. **Stage 2 (refinement):** gradient descent from that template to recover the
   exact digit. ~1 min.
4. Saves `competition_answer.mat` and `competition_answer.png`.

Result on the professor's vector: **digit 5**, refined misfit `1.76e-6`
(top-8 candidates: seven 5's and one 8).

## The answer file: `competition_answer.mat`

The recovered conductivity image is saved at the submission root in
`competition_answer.mat`. Load it in MATLAB with `load('competition_answer.mat')`;
it contains:

| variable        | type        | meaning                                                              |
|-----------------|-------------|----------------------------------------------------------------------|
| `digit`         | scalar      | recovered digit class: **5**                                         |
| `final_misfit`  | scalar      | `1/2 ||F(sigma) - y_obs||^2` after refinement: **1.7627e-6**         |
| `sigma_answer`  | 28x28       | the recovered conductivity image (background 1, digit ~2)            |
| `x_template`    | 784x1       | the winning MNIST training image (pixels of the Stage-1 winner)      |
| `best_idx`      | scalar      | its index in the MNIST training set: **51138**                       |

**The generalization test (optional):**

```matlab
>> validate_0to9
```

Uses the 60k training images as the dictionary and recovers one *unseen* test
digit of each class 0-9. Saves `figures/validation_0to9.png`. **9 of 10 correct**
(the lone miss is an unseen 5 that resembles a 6). ~3 min.

## The method in one line

The unknown is essentially *one of the MNIST images*, so we **template-match to
identify the digit and a correct starting point, then refine** with gradient
descent. This beats blind regularization (which lands on the wrong or a blurry
digit) because it avoids the multi-modality of the inverse problem.

## Requirements

MATLAB (tested on R2026a). No special toolboxes needed.

## Key result

Generic regularizers give blobs; linear PCA gives the wrong digit at lowest
misfit; a VAE gives the correct but blurry digit; **template-match + refine gives
a clean, correct digit** at ~200x lower misfit than the TV baseline. The held-out
test confirms the method generalizes (9/10 unseen digits). Central lesson: for
this ill-posed problem, *data misfit is not reconstruction quality* — the prior
matters more. The full story is in `Project1_Lesson.pdf`.
