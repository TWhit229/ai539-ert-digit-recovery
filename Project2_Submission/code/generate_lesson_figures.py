"""
Generate all matplotlib figures used in Project2_Lesson.tex.

Figures produced into ../figures/lesson/:
  fig02_forward_chain.png      - one digit at noise levels t=0..1000
  fig05_tweedie_strip.png       - at each t, show x_t (top) and x0_hat (bottom)
  fig07_bayes_1d.png            - three bell curves: prior, likelihood, posterior
  fig10_pretrained_vs_finetuned.png  - 8 samples from each DDPM
  fig11_gd_vs_newton.png        - GD zigzag vs Newton path on an elongated bowl
  fig12_recovered_sigma.png     - recovered conductivity + y_obs trace
"""
import os, math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from diffusers import DDPMPipeline

OUT_DIR = "../figures/lesson"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# Figure 2: forward chain strip
# -----------------------------------------------------------
def fig_forward_chain():
    mn = loadmat("MNIST Data/mnist.mat")
    M = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
    if M.max() > 1.5: M = M / 255.0
    x0 = M[:, 51137].reshape(28, 28)        # our familiar template 51138 (a 5)
    # DDPM beta schedule (linear from 1e-4 to 0.02 over 1000 steps)
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    ts = [0, 100, 300, 500, 800, 999]
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    fig, axes = plt.subplots(1, len(ts), figsize=(2.0*len(ts), 2.3))
    for ax, t in zip(axes, ts):
        if t == 0:
            xt = x0.copy()
        else:
            ab = alpha_bar[t-1] if t > 0 else 1.0
            xt = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
        ax.imshow(xt, cmap='gray', vmin=-0.5, vmax=1.2)
        ax.set_title(f"$t = {t}$\n$\\bar\\alpha_t = {alpha_bar[max(t-1,0)]:.3f}$"
                     if t > 0 else f"$t = 0$\n(clean)", fontsize=10)
        ax.axis('off')
    plt.suptitle("Forward noising chain: one digit, six noise levels",
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig02_forward_chain.png", dpi=140,
                bbox_inches='tight', facecolor='white')
    print(f"saved fig02_forward_chain.png")


# -----------------------------------------------------------
# Figure 5: Tweedie estimates
# -----------------------------------------------------------
def fig_tweedie_strip():
    mn = loadmat("MNIST Data/mnist.mat")
    M = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
    if M.max() > 1.5: M = M / 255.0
    x0_pix = M[:, 51137].reshape(28, 28)
    # x0 in diffusion convention [-1, +1]
    x0_diff = (x0_pix * 2.0 - 1.0).astype(np.float32)

    print("loading DDPM for Tweedie figure...")
    pipe = DDPMPipeline.from_pretrained("./ddpm_mnist_rot15")
    unet = pipe.unet.eval()
    scheduler = pipe.scheduler
    alpha_bar = scheduler.alphas_cumprod.float()

    ts = [700, 400, 150, 30]   # less extreme high-t to keep Tweedie estimate informative
    rng = np.random.default_rng(0)
    eps = torch.tensor(rng.standard_normal((1, 1, 28, 28)).astype(np.float32))

    fig, axes = plt.subplots(2, len(ts), figsize=(2.2*len(ts), 4.6))
    for col, t in enumerate(ts):
        ab = alpha_bar[t]
        sqrt_a = ab.sqrt()
        sqrt_1ma = (1 - ab).sqrt()
        x0_t = torch.tensor(x0_diff[None, None])
        xt = sqrt_a * x0_t + sqrt_1ma * eps
        with torch.no_grad():
            eps_pred = unet(xt, torch.tensor([t])).sample
        x0_hat = (xt - sqrt_1ma * eps_pred) / sqrt_a
        x0_hat = x0_hat.clamp(-1.5, 1.5)
        # show in [0, 1]
        xt_show = (xt.squeeze().numpy() * 0.5 + 0.5)
        x0hat_show = (x0_hat.squeeze().numpy() * 0.5 + 0.5).clip(0, 1)
        axes[0, col].imshow(xt_show, cmap='gray', vmin=-0.3, vmax=1.3)
        axes[0, col].set_title(f"$x_t$ at $t = {t}$\n$\\bar\\alpha_t={float(ab):.3f}$",
                                fontsize=10)
        axes[0, col].axis('off')
        axes[1, col].imshow(x0hat_show, cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f"$\\hat x_0$ (Tweedie estimate)", fontsize=10)
        axes[1, col].axis('off')
    plt.suptitle("Tweedie's formula at four noise levels (top: input $x_t$; "
                 "bottom: network's clean-image guess $\\hat x_0$)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig05_tweedie_strip.png", dpi=140,
                bbox_inches='tight', facecolor='white')
    print(f"saved fig05_tweedie_strip.png")


# -----------------------------------------------------------
# Figure 7: Bayes 1D
# -----------------------------------------------------------
def fig_bayes_1d():
    x = np.linspace(50, 95, 600)
    # Prior: N(67, 4)
    mu_p, sig_p = 67, 4
    prior = np.exp(-0.5 * ((x - mu_p) / sig_p) ** 2) / (sig_p * np.sqrt(2*np.pi))
    # Likelihood: N(80, 1) seen as a function of x
    mu_l, sig_l = 80, 1
    lik = np.exp(-0.5 * ((x - mu_l) / sig_l) ** 2) / (sig_l * np.sqrt(2*np.pi))
    # Posterior: product of Gaussians is Gaussian
    var_post = 1.0 / (1.0 / sig_p**2 + 1.0 / sig_l**2)
    mu_post = var_post * (mu_p / sig_p**2 + mu_l / sig_l**2)
    sig_post = np.sqrt(var_post)
    post = np.exp(-0.5 * ((x - mu_post) / sig_post) ** 2) / (sig_post * np.sqrt(2*np.pi))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(x, prior, alpha=0.25, color='#1976d2', label=f"prior $p(x)$:  $\\mathcal{{N}}({mu_p},{sig_p}^2)$")
    ax.fill_between(x, lik, alpha=0.25, color='#d32f2f', label=f"likelihood $p(y|x)$:  $\\mathcal{{N}}({mu_l},{sig_l}^2)$")
    ax.fill_between(x, post, alpha=0.45, color='#388e3c',
                    label=f"posterior $p(x|y)$:  $\\mathcal{{N}}({mu_post:.1f},{sig_post:.2f}^2)$")
    ax.plot(x, prior, color='#1976d2', linewidth=1.5)
    ax.plot(x, lik, color='#d32f2f', linewidth=1.5)
    ax.plot(x, post, color='#388e3c', linewidth=2.2)
    ax.set_xlabel("$x$  (adult height, inches)")
    ax.set_ylabel("density")
    ax.set_title("Bayes' rule in 1D: prior $\\times$ likelihood $\\propto$ posterior",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig07_bayes_1d.png", dpi=140,
                bbox_inches='tight', facecolor='white')
    print(f"saved fig07_bayes_1d.png")


# -----------------------------------------------------------
# Figure 10: pretrained vs fine-tuned samples
# -----------------------------------------------------------
def fig_pretrained_vs_finetuned():
    N = 8
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    def sample(pipe, seed):
        torch.manual_seed(seed)
        pipe.unet.to(device).eval()
        pipe.scheduler.set_timesteps(100)
        x = torch.randn(N, 1, 28, 28, device=device)
        with torch.no_grad():
            for t in pipe.scheduler.timesteps:
                eps = pipe.unet(x, t).sample
                x = pipe.scheduler.step(eps, t, x).prev_sample
        return (x.cpu() * 0.5 + 0.5).clamp(0, 1).numpy()

    print("loading pretrained DDPM...")
    pipe_pre = DDPMPipeline.from_pretrained("1aurent/ddpm-mnist")
    print("loading fine-tuned DDPM...")
    pipe_ft = DDPMPipeline.from_pretrained("./ddpm_mnist_rot15")

    pre = sample(pipe_pre, 42)
    ft  = sample(pipe_ft, 42)

    fig, axes = plt.subplots(2, N, figsize=(1.6*N, 4))
    for k in range(N):
        axes[0, k].imshow(pre[k, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, k].axis('off')
        axes[1, k].imshow(ft[k, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, k].axis('off')
    axes[0, 0].set_title("pretrained DDPM (upright only)", fontsize=11,
                          loc='left', x=0)
    axes[1, 0].set_title("after fine-tuning on $\\pm 15^\\circ$ rotated MNIST",
                          fontsize=11, loc='left', x=0)
    plt.suptitle("Unconditional samples: what the prior thinks digits look like",
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig10_pretrained_vs_finetuned.png", dpi=140,
                bbox_inches='tight', facecolor='white')
    print(f"saved fig10_pretrained_vs_finetuned.png")


# -----------------------------------------------------------
# Figure 11: GD zigzag vs Newton path on elongated bowl
# -----------------------------------------------------------
def fig_gd_vs_newton():
    # f(x, y) = 0.5 * (a*x^2 + b*y^2),  elongated bowl, condition number a/b
    a, b = 12.0, 1.0     # smaller condition number for cleaner visualization

    def grad(p):
        return np.array([a*p[0], b*p[1]])

    # GD path: choose eta near stability limit so x1 visibly zigzags
    # 1 - eta*a in (-1, 0) needs eta in (1/a, 2/a). Pick close to 2/a.
    p = np.array([1.0, 1.0])
    gd_path = [p.copy()]
    eta = 1.7 / a   # x1 multiplier = 1 - 1.7 = -0.7 -> visible zigzag
    for _ in range(35):
        p = p - eta * grad(p)
        gd_path.append(p.copy())
    gd_path = np.array(gd_path)

    # Newton path
    p = np.array([1.0, 1.0])
    nw_path = [p.copy()]
    for _ in range(2):
        g = grad(p)
        H = np.array([[a, 0], [0, b]])
        delta = -np.linalg.solve(H, g)
        p = p + delta
        nw_path.append(p.copy())
    nw_path = np.array(nw_path)

    # contour grid
    xs = np.linspace(-1.3, 1.3, 200)
    ys = np.linspace(-1.3, 1.3, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (a*X**2 + b*Y**2)

    fig, ax = plt.subplots(figsize=(8, 5.0))
    levels = [0.2, 0.5, 1, 2, 4, 8]
    cs = ax.contour(X, Y, Z, levels=levels, colors='#888', linewidths=0.7,
                     alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')
    # GD path
    ax.plot(gd_path[:, 0], gd_path[:, 1], 'o-', color='#d32f2f',
            markersize=4, linewidth=1.4,
            label=f'gradient descent  ({len(gd_path)-1} steps)')
    # Newton path
    ax.plot(nw_path[:, 0], nw_path[:, 1], 's-', color='#1976d2',
            markersize=8, linewidth=2.0,
            label=f"Newton's method  ({len(nw_path)-1} step)")
    ax.plot(0, 0, '*', color='gold', markersize=22,
            markeredgecolor='black', label='optimum')
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Elongated quadratic bowl: gradient descent zigzags;\n"
                 "Newton's method lands at the bottom in one shot",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig11_gd_vs_newton.png", dpi=140,
                bbox_inches='tight', facecolor='white')
    print(f"saved fig11_gd_vs_newton.png")


# -----------------------------------------------------------
# Figure 12: recovered conductivity + y_obs trace
# -----------------------------------------------------------
def fig_recovered_sigma():
    d = loadmat("../final_answer.mat")
    sigma = d['sigma_answer']
    yd = loadmat("y_truth_measurement.mat")
    y = yd['y_truth'].flatten()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(y, color='#1976d2', linewidth=0.7)
    axes[0].set_xlabel("measurement index $k$  (out of 1900)")
    axes[0].set_ylabel("voltage $y_k$")
    axes[0].set_title("The input: $y_\\text{obs}$ (1900 boundary voltages)",
                       fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    im = axes[1].imshow(sigma, cmap='viridis', vmin=1, vmax=2)
    axes[1].set_title("The output: recovered conductivity $\\sigma$ "
                       "(misfit $6.16\\times 10^{-13}$)",
                       fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig12_recovered_sigma.png", dpi=140,
                bbox_inches='tight', facecolor='white')
    print(f"saved fig12_recovered_sigma.png")


# -----------------------------------------------------------
# Figure: PnP-DM chains compact (8 chains in a single row)
# -----------------------------------------------------------
def fig_pnp_dm_chains_compact():
    """Run PnP-DM once with 8 chains and save a compact 1-row figure."""
    import sys as _sys, time
    if "." not in _sys.path: _sys.path.insert(0, ".")
    from diffusers import DDPMPipeline
    from pnp_dm import pnp_dm as run_pnp_dm
    from stage_a_daps_warmstart import ERTSetup

    print("loading DDPM + setting up MATLAB for compact-chains figure...")
    setup = ERTSetup(".")
    try:
        pipe = DDPMPipeline.from_pretrained("./ddpm_mnist_rot15")
        setup.unet = pipe.unet.eval()
        setup.scheduler = pipe.scheduler
        setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
        for p_ in setup.unet.parameters(): p_.requires_grad_(False)

        print("running PnP-DM with 8 chains x 80 iters...")
        t0 = time.time()
        chains = run_pnp_dm(setup, n_chains=8, n_iter=80, sigma_n=1e-4,
                            gn_inner=2, seed=0, verbose=False)
        print(f"  done in {time.time()-t0:.0f}s")

        fig, axes = plt.subplots(1, 8, figsize=(13, 2.0))
        for k, c in enumerate(chains):
            axes[k].imshow(c['x_pix'], cmap='gray', vmin=0, vmax=1)
            axes[k].set_title(f"chain {k}\nmisfit {c['final_misfit']:.1e}",
                              fontsize=9)
            axes[k].axis('off')
        plt.suptitle("8 sample PnP-DM chains after 80 Gibbs iterations: "
                     "most converged to a tilted 5; a few drifted to an 8-shape",
                     fontsize=11, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig_chains_compact.png", dpi=140,
                    bbox_inches='tight', facecolor='white')
        print("saved fig_chains_compact.png")
    finally:
        setup.quit()


# -----------------------------------------------------------
# Figure: PnP-DM denoising progression for one chain
# -----------------------------------------------------------
def fig_denoising_progression():
    """Run one PnP-DM chain on the competition vector, save the iterate
    at fixed checkpoints, and assemble a filmstrip showing how a 5 emerges."""
    import sys as _sys, time
    if "." not in _sys.path: _sys.path.insert(0, ".")
    import torch
    from diffusers import DDPMPipeline
    from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
    from pnp_dm import gn_likelihood_prox, diff_one_denoise

    print("loading DDPM + MATLAB for denoising progression...")
    setup = ERTSetup(".")
    try:
        pipe = DDPMPipeline.from_pretrained("./ddpm_mnist_rot15")
        setup.unet = pipe.unet.eval()
        setup.scheduler = pipe.scheduler
        setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
        for p_ in setup.unet.parameters(): p_.requires_grad_(False)

        n_iter = 80
        sigmas = np.geomspace(1.0, 0.05, n_iter)
        torch.manual_seed(0); np.random.seed(0)
        x = torch.randn(28, 28).clamp(-1.5, 1.5)
        save_at = [0, 4, 12, 25, 45, 79]
        snapshots = []

        print(f"running one chain for {n_iter} iters, snapshots at {save_at}...")
        t0 = time.time()
        for k, sigma_k in enumerate(sigmas):
            z = gn_likelihood_prox(setup, x, sigma_k, sigma_n=1e-4, max_inner=2)
            x = diff_one_denoise(setup, z, sigma_k)
            if k in save_at:
                x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
                mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                                    - setup.y_obs) ** 2).sum())
                snapshots.append((k, float(sigma_k), mis, x_pix))
        print(f"  done in {time.time()-t0:.0f}s")

        fig, axes = plt.subplots(1, len(snapshots), figsize=(2.0*len(snapshots), 2.4))
        for ax, (k, sig, mis, img) in zip(axes, snapshots):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"iter {k+1}\nmisfit {mis:.1e}", fontsize=11)
            ax.axis('off')
        plt.suptitle("One PnP-DM chain over 80 iterations: pure noise to a clean tilted 5",
                     fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig_denoising_progression.png", dpi=140,
                    bbox_inches='tight', facecolor='white')
        print("saved fig_denoising_progression.png")
    finally:
        setup.quit()


if __name__ == "__main__":
    fig_forward_chain()
    fig_bayes_1d()
    fig_gd_vs_newton()
    fig_recovered_sigma()
    fig_tweedie_strip()
    fig_pretrained_vs_finetuned()
    fig_pnp_dm_chains_compact()
    fig_denoising_progression()
    print("\nAll figures written to", OUT_DIR)
