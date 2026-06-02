# Project 2 — DI-RTG Pipeline

**Goal:** Recover the competition digit from `y_truth_measurement.mat` using a diffusion-based prior, beating the result you'd get from vanilla DPS or vanilla DAPS.

**Method name:** DI-RTG = **D**APS-**I**nit → **R**eSample/RED-DiffEq Refinement → **T**emplate-Match → **G**auss-Newton Polish.

**Core insight:** Vanilla DAPS is over-engineered for our forward model (Langevin loop wastes the analytic Jacobian we already have) and under-engineered for our prior (the professor explicitly said x is "very similar to one of the MNIST samples" — strongest possible class prior). DI-RTG strips DAPS to a warm-start, then layers on classical PDE-inversion machinery from Project 1.

---

## Stage A — DAPS warm-start

- **Paper:** Decoupled Annealing Posterior Sampling (Zhang et al., CVPR 2025 Oral), arXiv `2407.01521`.
- **Code:** https://github.com/zhangbingliang2019/DAPS
- **What it does here:** ≈10 outer reverse-diffusion steps with a short Langevin inner loop. We deliberately under-run DAPS — its only job is to land near the MNIST manifold so Stage B has a good starting point. Don't tune for full DAPS quality.

## Stage B — Hard-consistency refinement with Gauss-Newton

Cross-pollination of three recent papers:
- **ReSample** (Song et al., ICLR 2024), arXiv `2307.08123` — hard data-consistency at selected reverse timesteps, re-noise back onto the manifold.
- **RED-DiffEq** (Shan/Zhu/Lin/Lu, 2025/2026), arXiv `2509.21659` — diffusion as a RED regularizer, classical Gauss-Newton on the PDE residual. Validated on full-waveform inversion — the closest published analogue to our ERT setup.
- **PnP-DM** (Wu et al., NeurIPS 2024), arXiv `2405.18782` — Split-Gibbs structure alternating likelihood (classical) and prior (diffusion) steps.

At each of ~8 decreasing noise levels:
1. Compute the Tweedie-denoised anchor `z_t = x + σ_t²·s_θ(x, t)`.
2. Solve the Tikhonov subproblem `min_x ½‖F(σ_bg+x) − y_obs‖² + (ρ_t/2)‖x − z_t‖²` with **3–5 Gauss-Newton iterations using analytic J(σ)** (Levenberg-Marquardt damping `μ_LM·diag(JᵀJ)`).
3. Re-noise: `x = x + σ(t_k)·ε`.
4. Denoise one step with the diffusion model.

This is the *new* code — Stage B's GN inner loop is what replaces DAPS's noisy Langevin gradient.

## Stage C — Class-conditional template match + ReGuidance

- **Paper:** ReGuidance (Karan/Shah/Chen 2025), arXiv `2506.10955`.
- Classify Stage B's output with a small MNIST CNN.
- Retrieve the top-3 nearest training images of that class (Project 1's idea, ported into the diffusion setting).
- For each candidate template `x_T`: invert the unconditional PF-ODE to get latent `z_T`, then run a brief class-conditional DPS pass anchored at the template. Keep the candidate with the lowest `‖F(σ_bg + x_R) − y_obs‖²`.
- Collapses any near-manifold ambiguity to a *specific* in-distribution digit.

## Stage D — Final Levenberg-Marquardt polish

Reuse Project 1's winning refinement verbatim:
```
for iter in N_LM:
    r = F(σ_bg + x) − y_obs
    H = JᵀJ + λ_pol·I + μ_LM·diag(JᵀJ)
    b = Jᵀr + λ_pol·(x − x_template)
    Δ = solve(H, b)
    x = clip01(x − α·Δ)
    if ‖r‖² < 1e-6: break
```

Drives final misfit to ~1e-6 (the noise floor we already established in Project 1).

---

## Why DI-RTG beats vanilla DAPS for our specific setting

| Our setting | DAPS alone | DI-RTG fix |
|---|---|---|
| Nonlinear F with **known analytic J** | Uses noisy Langevin `Jᵀr` — discards precision | Stage B uses `JᵀJ` Newton steps — quadratic convergence |
| Very low noise (~1e-6 misfit) | Posterior collapses to MAP; Langevin noise is pure error | Stage D's deterministic LM polish chases the 1e-6 floor |
| x "very similar to MNIST sample" (prof's hint) | Pure DAPS can settle between digit modes | Stage C commits to a class via classifier + retrieval |
| Overdetermined (1900 ≫ 784) | Doesn't exploit it | LM normal equations directly use overdetermination |

---

## Implementation roadmap (~one weekend)

| Hours | Task |
|---|---|
| 0–2  | Wrap MATLAB `ERT2D.m` (F and J) as a PyTorch `autograd.Function` via `matlab.engine`, OR port forward+adjoint to Python. |
| 2–6  | Train or pull a small MNIST DDPM (~1M params, 50 epochs). Use `lucidrains/denoising-diffusion-pytorch`. Optional: class-conditional via CFG (arXiv `2207.12598`). |
| 6–7  | Train a 3-layer MNIST CNN classifier (>99% test acc, trivial). |
| 7–10 | Implement Stage A (DAPS, ~80 LOC). |
| 10–14| Implement Stage B (RED-DiffEq + ReSample with GN inner loop, ~150 LOC). Bulk of new code. |
| 14–16| Implement Stage C (ReGuidance + nearest-neighbor lookup, ~40 LOC). |
| 16–17| Stage D — paste in Project 1's LM polish. |
| 17–20| Validate end-to-end on the 9 held-out Project-1 digits; tune `K_A, K_B, ρ_t, λ_pol`. |
| 20–24| Run on competition vector. Compare to DAPS-alone baseline. Write up. |

No new networks beyond a standard MNIST DDPM + small classifier. All heavy compute is in 784-dim linear solves (milliseconds each).

---

## Backup: DAPS++ + template + LM polish

**DAPS++**, arXiv `2511.17038`. If Stage B's GN refinement is numerically tricky (ill-conditioned `JᵀJ`, schedule issues), fall back to DAPS++ as the entire diffusion backbone — it's already an EM-style decoupling of sampling and refinement, reports ~90% NFE reduction vs DAPS at comparable quality. Then keep Stages C and D unchanged. Simpler, fewer hyperparameters, off-the-shelf published method.

---

## Risk factors and mitigations

1. **`JᵀJ` ill-conditioning during Stage B.** ERT Jacobian decays away from electrodes. *Mitigation:* LM damping `μ_LM·diag(JᵀJ)` + Tikhonov anchor `ρ_t·I` (already in pseudocode). Increase `ρ_t` early in the schedule.
2. **Weak MNIST DDPM mode-collapses.** *Mitigation:* class-conditional CFG, or grab a pretrained checkpoint.
3. **Classifier picks wrong digit at Stage C.** Anchors Stage D on wrong template. *Mitigation:* run Stage C for top-3 classes, pick by final residual. With noise floor 1e-6, correct class wins by orders of magnitude.
4. **DAPS Stage A drifts to non-digit basin.** *Mitigation:* run Stage A with 4–8 random seeds, keep the candidate with lowest post-Stage-B residual.
5. **MATLAB↔PyTorch overhead.** F and J evals dominate runtime. *Mitigation:* batch calls or port the forward+adjoint to Python.

---

## Key references (arXiv IDs)

**Primary pipeline:**
- DAPS — `2407.01521` — Zhang et al. 2024 (CVPR 2025 Oral)
- ReSample — `2307.08123` — Song et al. 2023 (ICLR 2024)
- RED-DiffEq — `2509.21659` — Shan/Zhu/Lin/Lu 2025
- PnP-DM — `2405.18782` — Wu et al. 2024 (NeurIPS 2024)
- DPnP — `2403.17042` — Xu/Chi 2024 (NeurIPS 2024)
- ReGuidance — `2506.10955` — Karan/Shah/Chen 2025

**Backup:**
- DAPS++ — `2511.17038`

**Supporting:**
- InverseBench — `2503.11043` (ICLR 2025 Spotlight benchmark — DAPS top-rated)
- CFG (classifier-free guidance) — `2207.12598`
- RDPS-EIT — `2605.19621` (diffusion for electrical impedance tomography specifically)
