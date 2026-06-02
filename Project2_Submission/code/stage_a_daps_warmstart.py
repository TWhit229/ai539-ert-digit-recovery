"""
Stage A of DI-RTG: short DAPS warm-start.

Goal: starting from pure noise, run ~K_outer diffusion steps with a short Langevin
inner loop at each step. Just get x near the MNIST manifold so Stage B has a
reasonable starting point — NOT a fully-converged answer.

Algorithm:
  x_T ~ N(0, I)
  for t = T, T-1, ..., 1   (K_outer steps total):
      ε_pred = ε_θ(x_t, t)                           # diffusion model
      x̂_0    = (x_t − √(1−ᾱ_t)·ε_pred) / √ᾱ_t        # Tweedie estimate of clean x
      z      = x̂_0
      for k = 1..K_inner:                            # short Langevin / GD
          r        = F(σ_b + z) − y_obs              # MATLAB call
          g_lik    = Jᵀr                              # MATLAB call (mode 2)
          g_prior  = (x̂_0 − z) / σ²_x|t              # pull toward Tweedie estimate
          z        = z + η · ( g_prior − λ · g_lik ) + √(2η τ)·ξ
          z        = clip(z, -σ_b + ε, ∞)            # keep σ > 0
      # re-noise to next level
      x_{t-1} = √ᾱ_{t-1}·z + √(1−ᾱ_{t-1})·N(0,I)

Outputs:
  - figures/stage_a_warmstart.png  — starting noise → warm-start image
  - stage_a_result.npz             — final image, intermediate diagnostics
"""
import argparse
import time
import numpy as np
import torch
from scipy.io import loadmat
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
import matlab.engine
import matlab as _ml

# ----------------------------------------------------------------------
# Setup: MATLAB engine + diffusion model + y_obs
# ----------------------------------------------------------------------
class ERTSetup:
    def __init__(self, code_dir, sigma_bg=1.0, domain=(10.0, 5.0), grid=(28, 28)):
        print("Starting MATLAB engine...")
        t0 = time.time()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(code_dir, nargout=0)
        print(f"  engine up in {time.time()-t0:.1f}s")

        print("Loading pretrained MNIST DDPM (1aurent/ddpm-mnist)...")
        self.pipe = DDPMPipeline.from_pretrained("1aurent/ddpm-mnist")
        self.unet = self.pipe.unet.eval()
        self.scheduler = self.pipe.scheduler
        self.alphas_cumprod = self.scheduler.alphas_cumprod.float()  # (1000,)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.sigma_bg = float(sigma_bg)
        self.Lx, self.Ly = float(domain[0]), float(domain[1])
        self.grid = (int(grid[0]), int(grid[1]))

        print("Loading competition y_truth...")
        yd = loadmat(f"{code_dir}/y_truth_measurement.mat")
        self.y_obs = torch.tensor(yd['y_truth'].flatten(), dtype=torch.float32)
        print(f"  y_obs shape: {tuple(self.y_obs.shape)}")

        # Warm up MATLAB cache with one call
        print("Warming MATLAB cache (one forward call)...")
        sig0 = self.sigma_bg * np.ones(self.grid)
        t0 = time.time()
        _ = self._matlab_call(sig0, with_jacobian=False)
        print(f"  warmed in {time.time()-t0:.2f}s")

        # ---- Jacobian column-permutation: MATLAB → numpy ----
        # MATLAB flattens images column-major: pixel (i,j) → flat index j·28 + i.
        # numpy default is row-major: pixel (i,j) → flat index i·28 + j.
        # We permute J's columns once so the rest of our code can use row-major
        # indexing transparently. Verified by directional derivative test.
        ny, nx = self.grid
        self._col_perm = np.arange(ny * nx).reshape(ny, nx).T.flatten()

    def _matlab_call(self, sig_np, with_jacobian=False):
        sig_np = np.asarray(sig_np, dtype=np.float64)
        sig_np = np.maximum(sig_np, 1e-6)  # keep σ > 0
        sig_mat = _ml.double(sig_np.tolist())
        mode = 2 if with_jacobian else 1
        y, J = self.eng.ERT_call(sig_mat, mode, _ml.double([self.grid[0], self.grid[1]]),
                                  self.sigma_bg, self.Lx, self.Ly, nargout=2)
        y = np.array(y).flatten()
        if with_jacobian:
            J = np.array(J)
            return y, J
        return y, None

    def forward(self, x):
        """x: torch (28,28) tensor → y (1900,) torch."""
        x_np = x.detach().cpu().numpy()
        y, _ = self._matlab_call(self.sigma_bg + x_np, with_jacobian=False)
        return torch.tensor(y, dtype=torch.float32)

    def forward_and_jacobian(self, x):
        """x: torch (28,28). Returns (y (1900,), J (1900,784)) torch tensors.
        J's columns are permuted to numpy row-major pixel order: J[:, i·28+j] is the
        sensitivity of every measurement to pixel (i, j)."""
        x_np = x.detach().cpu().numpy()
        y, J = self._matlab_call(self.sigma_bg + x_np, with_jacobian=True)
        J = J[:, self._col_perm]                           # row-major column order
        return (torch.tensor(y, dtype=torch.float32),
                torch.tensor(J, dtype=torch.float32))

    def misfit(self, x):
        y = self.forward(x)
        return 0.5 * torch.sum((y - self.y_obs) ** 2).item()

    def quit(self):
        try:
            self.eng.quit()
        except Exception:
            pass

# ----------------------------------------------------------------------
# Coordinate conventions
# ----------------------------------------------------------------------
# We work internally in `x_diff ∈ [-1, 1]` (the convention the pretrained
# DDPM was trained on). Our forward map F takes a CONDUCTIVITY image
# σ = σ_bg + x_pix where x_pix ∈ [0, 1]. The map between them:
#     x_pix = (x_diff + 1) / 2          x_diff = 2·x_pix − 1
# The Jacobian J we get from MATLAB is ∂F/∂x_pix. By chain rule:
#     ∂F/∂x_diff = (1/2) · J
# so the likelihood gradient in x_diff coordinates is (1/2)·Jᵀr.
def x_diff_to_pix(x_diff):
    return 0.5 * (x_diff + 1.0)


# ----------------------------------------------------------------------
# Stage A — DAPS warm-start (fixed coordinates, clamped Tweedie)
# ----------------------------------------------------------------------
def daps_warmstart(setup, K_outer=30, K_inner=3, eta=0.5, lam=1.0, tau=0.0,
                   tweedie_clip=1.5, seed=0, verbose=True):
    """
    Args:
      K_outer        : number of reverse-diffusion steps
      K_inner        : Langevin/GD steps per outer step
      eta            : Langevin / GD step size (in x_diff coordinates)
      lam            : likelihood weight (multiplies (1/2)·Jᵀr)
      tau            : Langevin temperature (0 = deterministic, no extra noise)
      tweedie_clip   : clip Tweedie x0_hat to [-tweedie_clip, +tweedie_clip] to
                       avoid the high-t blow-up
    """
    torch.manual_seed(seed); np.random.seed(seed)
    setup.scheduler.set_timesteps(K_outer)
    timesteps = setup.scheduler.timesteps  # decreasing

    # x_T ~ N(0, I) in x_diff space
    x_t = torch.randn(1, 1, 28, 28)
    diagnostics = []

    for i, t in enumerate(timesteps):
        t_idx = int(t.item())
        a_bar_t  = setup.alphas_cumprod[t_idx]
        sqrt_a   = a_bar_t.sqrt()
        sqrt_1ma = (1.0 - a_bar_t).sqrt()
        sigma2_x_given_t = ((1.0 - a_bar_t) / a_bar_t).clamp_min(1e-6)

        # ---- Tweedie estimate of clean image (in x_diff coords) ----
        with torch.no_grad():
            eps_pred = setup.unet(x_t, t).sample
        x0_hat = (x_t - sqrt_1ma * eps_pred) / sqrt_a
        x0_hat = x0_hat.squeeze().clamp(-tweedie_clip, tweedie_clip)

        # ---- Short Langevin / GD inner loop ----
        # Stability fix: scale η by σ²_x|t so the prior pull-back term
        # (x̂₀ − z)/σ²  is always damped at a stable rate.
        # The likelihood weight λ is divided by σ² inside, so the effective
        # data-fit contribution stays independent of t.
        eta_t = eta * sigma2_x_given_t
        z = x0_hat.clone()
        for k in range(K_inner):
            z_pix = x_diff_to_pix(z).clamp(0.0, 1.0)
            y_pred, J = setup.forward_and_jacobian(z_pix)
            r       = y_pred - setup.y_obs
            g_lik   = 0.5 * (J.T @ r).reshape(28, 28)        # (1/2)·Jᵀr (chain rule)
            g_prior = (x0_hat - z) / sigma2_x_given_t
            step    = eta_t * (g_prior - (lam / sigma2_x_given_t) * g_lik)
            #   = η · (x̂₀ − z)  −  η · λ · g_lik         (after cancellation)
            if tau > 0:
                step = step + (2.0 * eta_t * tau) ** 0.5 * torch.randn_like(z)
            z = (z + step).clamp(-tweedie_clip, tweedie_clip)

        # Diagnostic misfit using current z (pixel coords)
        z_pix_eval = x_diff_to_pix(z).clamp(0.0, 1.0)
        misfit = 0.5 * float(((setup.forward(z_pix_eval) - setup.y_obs) ** 2).sum())
        diagnostics.append({'step': i, 't': t_idx,
                            'misfit_after_inner': misfit,
                            'z_min': float(z.min()), 'z_max': float(z.max())})
        if verbose:
            print(f"  step {i+1:2d}/{K_outer}  t={t_idx:4d}  α_bar={a_bar_t:.4f}  "
                  f"misfit={misfit:.3e}  z_diff range [{z.min():+.2f},{z.max():+.2f}]")

        # ---- Re-noise to next noise level (DDPM forward, x_diff space) ----
        if i < len(timesteps) - 1:
            t_prev_idx = int(timesteps[i + 1].item())
            a_bar_prev = setup.alphas_cumprod[t_prev_idx]
            x_t = (a_bar_prev.sqrt() * z.view(1, 1, 28, 28)
                   + (1.0 - a_bar_prev).sqrt() * torch.randn_like(x_t))
        else:
            x_final_diff = z

    # Return in PIXEL coords (= our x convention for the inverse problem)
    x_final_pix = x_diff_to_pix(x_final_diff).clamp(0.0, 1.0)
    return x_final_pix.detach().cpu().numpy(), diagnostics

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_dir", default=".", help="dir with ERT2D / ERT_call / y_truth_measurement.mat")
    parser.add_argument("--K_outer", type=int, default=30)
    parser.add_argument("--K_inner", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--tweedie_clip", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="stage_a_result.npz")
    parser.add_argument("--fig", default="../figures/stage_a_warmstart.png")
    args = parser.parse_args()

    setup = ERTSetup(args.code_dir)
    try:
        # Record the initial pure-noise image for visualization
        torch.manual_seed(args.seed)
        x_T_for_viz = torch.randn(28, 28).numpy()

        print(f"\nRunning DAPS warm-start  K_outer={args.K_outer}  K_inner={args.K_inner}  "
              f"eta={args.eta}  lam={args.lam}")
        t0 = time.time()
        x_final, diags = daps_warmstart(setup, K_outer=args.K_outer, K_inner=args.K_inner,
                                        eta=args.eta, lam=args.lam, tau=args.tau,
                                        tweedie_clip=args.tweedie_clip, seed=args.seed)
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s.")

        # Final misfit
        x_t = torch.tensor(x_final, dtype=torch.float32)
        final_misfit = setup.misfit(x_t)
        print(f"Final misfit: {final_misfit:.3e}")
        print(f"  (Project 1 winner achieved 1.76e-6 after full LM polish; "
              f"warm-start target is ~1e-3 to 1e-4)")

        np.savez(args.out,
                 x_final=x_final, x_T_for_viz=x_T_for_viz,
                 final_misfit=final_misfit,
                 K_outer=args.K_outer, K_inner=args.K_inner, eta=args.eta, lam=args.lam,
                 diagnostics=diags)
        print(f"Saved {args.out}")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(x_T_for_viz, cmap='gray', vmin=-3, vmax=3)
        axes[0].set_title('Initial: pure noise\n(x_T ~ N(0, I))')
        axes[1].imshow(x_final, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'After Stage A (K_outer={args.K_outer})\nmisfit = {final_misfit:.2e}')
        axes[2].plot([d['misfit_after_inner'] for d in diags], 'o-')
        axes[2].set_yscale('log')
        axes[2].set_xlabel('outer step')
        axes[2].set_ylabel('misfit')
        axes[2].set_title('Misfit per outer step')
        axes[2].grid(alpha=0.3)
        for ax in axes[:2]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(args.fig, dpi=130, bbox_inches='tight', facecolor='white')
        print(f"Saved {args.fig}")

    finally:
        setup.quit()

if __name__ == "__main__":
    main()
