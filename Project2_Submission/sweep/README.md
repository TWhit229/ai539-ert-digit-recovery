# Project 2 Sweep — 12 Combinations to Fix the Recovered-Image Quality

The current Project 2 pipeline (`code/pnp_dm.py` → `code/gn_cg_polish.py`)
reaches a misfit of `6.16e-13` on the competition vector, but the recovered
digit shows speckle and looks like a "5 morphed with a 6". This sweep
tests twelve combinations of model and algorithm changes to find a setup
that produces cleaner recovered images across all digits.

## What the sweep covers

| Axis | Options |
| --- | --- |
| **Prior** | `base` — current 1.07 M-param `1aurent/ddpm-mnist` fine-tuned on ±15° rotations |
|  | `bigger` — new 6 M-param `UNet2DModel`, trained from scratch on rotated MNIST with EMA weights |
|  | `cond` — 6 M-param class-conditional UNet with classifier-free guidance (CFG, `w = 3`) |
| **Schedule** | `geomspace` — log-spaced sigmas (current) |
|  | `neg_rho` — DAPS++ polynomial schedule with `rho = -7` (more iters at low noise) |
| **Polish** | `lm` — current GN-CG Levenberg-Marquardt polish |
|  | `tv` — Gauss-Newton polish with smoothed total-variation regularization |

Total: `3 x 2 x 2 = 12` polished images per run, plus `6` pre-polish images.

## What the sweep does NOT cover

- The EDM-style sampler. That would require swapping the entire DDPM
  framework for the Karras et al. ODE solver — too big a lift for an
  incremental sweep. If `bigger`+`neg_rho`+`tv` wins, EDM can be a
  follow-up.

## Files in this folder

| File | Purpose |
| --- | --- |
| `schedule.py`         | Geomspace and DAPS++ negative-rho sigma schedules |
| `tv_polish.py`        | GN-CG polish with smoothed-TV regularization |
| `pnp_dm_v2.py`        | PnP-DM with selectable schedule (used for `base`, `bigger`) |
| `train_bigger_ddpm.py`| Train the bigger 6 M-param DDPM with EMA |
| `train_cond_ddpm.py`  | Train the class-conditional 6 M-param DDPM with EMA |
| `pnp_dm_cfg.py`       | PnP-DM with classifier-free guidance (used for `cond`) |
| `run_sweep.py`        | Driver: runs all 12 combos, writes results/ |
| `plot_sweep.py`       | Makes the 3×4 grid of recovered images |
| `requirements.txt`    | Python deps (install CUDA torch separately, see below) |
| `results/`            | Output directory (mat files + CSV/JSON summary) |

Everything imports from `../code/` for the unchanged pieces (forward
operator, ERTSetup, GN-CG polish, original PnP-DM helpers).

## Setup on the desktop with the 5090

### 1. Pull the repo and `cd` into Project 2

```bash
git pull
cd "ERT 2D/Project2_Submission/sweep"
```

### 2. Install MATLAB R2026a + Python engine

MATLAB R2026a must be installed. To install the Python engine into the
Python environment you'll use:

```bash
cd "/Applications/MATLAB_R2026a.app/extern/engines/python"   # macOS
# or  "C:/Program Files/MATLAB/R2026a/extern/engines/python"  # Windows
python setup.py install
```

(`python` here means whatever Python you'll be using for the sweep, e.g.
a conda env.)

### 3. Install CUDA PyTorch

For the 5090 (Blackwell, sm_120), install the latest official CUDA build:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 5090
```

### 4. Install the rest

```bash
pip install -r requirements.txt
```

## Running the sweep — three phases

The driver caches PnP-DM results per `(prior, schedule)` so it only runs
6 PnP-DM passes total (one per combination, not one per polish variant).
You can run any subset of combos with `--combos`.

### Phase 1 — base prior, no new training (about 25 minutes on a 5090)

```bash
python run_sweep.py \
  --combos base.geomspace.lm,base.geomspace.tv,base.neg_rho.lm,base.neg_rho.tv
python plot_sweep.py
```

This tests whether the cheaper changes (TV polish, negative-rho
schedule) on the existing DDPM are enough. If the TV polish on
`base.geomspace` already removes the speckle, you may not need to
train the bigger or conditional DDPMs at all.

### Phase 2 — bigger DDPM (about 90 minutes on a 5090)

Train once:

```bash
python train_bigger_ddpm.py --epochs 30
```

Then sweep its 4 combos:

```bash
python run_sweep.py \
  --combos bigger.geomspace.lm,bigger.geomspace.tv,bigger.neg_rho.lm,bigger.neg_rho.tv
python plot_sweep.py
```

### Phase 3 — class-conditional DDPM with CFG (about 2 hours on a 5090)

Train once:

```bash
python train_cond_ddpm.py --epochs 30
```

Then sweep its 4 combos:

```bash
python run_sweep.py \
  --combos cond.geomspace.lm,cond.geomspace.tv,cond.neg_rho.lm,cond.neg_rho.tv
python plot_sweep.py
```

### All 12 in one shot

If you want to leave it running, train both DDPMs first, then run
without `--combos`:

```bash
python train_bigger_ddpm.py --epochs 30
python train_cond_ddpm.py   --epochs 30
python run_sweep.py
python plot_sweep.py
```

Estimated wall-clock on a 5090: about 3.5 to 4 hours end to end.

## Outputs

After a sweep run you'll have:

```
results/
  pre_polish/
    base_geomspace.mat       # best chain BEFORE polish
    base_neg_rho.mat
    bigger_geomspace.mat
    bigger_neg_rho.mat
    cond_geomspace.mat
    cond_neg_rho.mat
  polished/
    base_geomspace_lm.mat    # 12 polished answers
    base_geomspace_tv.mat
    ...
    cond_neg_rho_tv.mat
  sweep_summary.csv          # one row per combo, with both misfits + runtimes
  sweep_summary.json         # same data, JSON

sweep_grid.png         # 3x4 grid of all polished images
sweep_pre_vs_post.png  # pre-polish vs each polish, per (prior, schedule)
```

`sweep_summary.csv` is the headline file — sort by `final_misfit` to find
the winner, but also look at `sweep_grid.png` because the lowest misfit
is NOT always the visually best image (that is the original problem
this sweep exists to fix).

## How to pick the winner

Open `sweep_grid.png` and look for the combo that:

1. Looks most like a clean digit (no speckle, recognizable class).
2. Has a low misfit (within an order of magnitude of the best).

Speckle reduction matters more than driving misfit to the float64 floor.
A combo with misfit `1e-10` and a clean image beats one with misfit
`1e-13` and a noisy image.

## Knobs worth tweaking

If results are not what you hoped, in roughly decreasing order of impact:

- **TV weight `--tau`** (default `1e-5`): bigger = smoother but higher
  misfit floor. Try `1e-4` if there is still visible speckle, `1e-6` if
  the image is too smooth.
- **CFG weight `--cfg_w`** (default `3.0`): bigger = more class-faithful
  but less variety across chains. Try `5.0` if `cond` chains are not
  committing to a class, `1.0` if they are over-committing.
- **`--n_chains`** (default `32`): more chains = more mode coverage.
  Each chain is independent so this scales linearly with wall-clock.
- **`--n_iter`** (default `80`): more iters = lower pre-polish misfit
  but rarely changes the image's shape after the first 60.
- **Training `--epochs`** for the new DDPMs (default `30`): 60 epochs
  usually gives noticeably crisper unconditional samples; doubles
  training time.

## Pushing results back

After the sweep, if you want to share the polished mat files, the
`results/` directory is committed (it has a `.gitkeep`). The mat files
themselves are small (28x28 doubles = a few KB each).

The trained DDPM directories (`ddpm_bigger_rot15_ema/`,
`ddpm_cond_rot15_ema/`) are not committed by default since they are tens
of MB. If you want to share them, add them explicitly with `git add -f`.
