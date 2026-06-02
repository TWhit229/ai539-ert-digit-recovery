"""
Make animated GIFs showing PnP-DM denoising over time.

For each requested image (the competition vector and a couple of held-out
test digits), run one chain, capture the pixel image at every Gibbs
iteration, and assemble into an animated GIF. Each frame is upscaled 8x
with nearest-neighbor interpolation and annotated with the iteration
number and current misfit.
"""
import os, sys, time
sys.path.insert(0, ".")
import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image, ImageDraw, ImageFont
from diffusers import DDPMPipeline

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox, diff_one_denoise

OUT_DIR = "../figures/gifs"
os.makedirs(OUT_DIR, exist_ok=True)


def get_font(size=18):
    """Try to find a usable system font; fall back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def run_chain_capture_all(setup, n_iter=80, sigma_n=1e-4, gn_inner=2, seed=0):
    """Run one PnP-DM chain and capture the pixel image at every iteration.
    Returns a list of (k, sigma_k, misfit, x_pix_28x28) tuples."""
    torch.manual_seed(seed); np.random.seed(seed)
    sigmas = np.geomspace(1.0, 0.05, n_iter)
    x = torch.randn(28, 28).clamp(-1.5, 1.5)
    frames = []
    # Capture an initial "noise" frame before any iteration
    x_pix0 = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
    mis0 = 0.5 * float(((setup.forward(torch.tensor(x_pix0, dtype=torch.float32))
                          - setup.y_obs) ** 2).sum())
    frames.append((0, float(sigmas[0]), mis0, x_pix0))

    for k, sigma_k in enumerate(sigmas):
        z = gn_likelihood_prox(setup, x, sigma_k, sigma_n=sigma_n, max_inner=gn_inner)
        x = diff_one_denoise(setup, z, sigma_k)
        x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
        mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                            - setup.y_obs) ** 2).sum())
        frames.append((k + 1, float(sigma_k), mis, x_pix))
    return frames


def make_gif(frames, out_path, scale=10, fps=8, hold_last_ms=2500,
              title="PnP-DM denoising"):
    """Assemble frames into an animated GIF with text overlay."""
    font = get_font(size=22)
    font_small = get_font(size=16)
    img_size = 28 * scale
    # Canvas is wider than the image so the title doesn't get cut off.
    panel_w = max(img_size + 20, 480)
    panel_h = img_size + 90
    pil_frames = []
    for (k, sigma_k, mis, x_pix) in frames:
        x_uint8 = (np.clip(x_pix, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(x_uint8, mode='L').resize(
            (img_size, img_size), Image.NEAREST).convert('RGB')
        canvas = Image.new('RGB', (panel_w, panel_h), 'white')
        # Center the image horizontally
        canvas.paste(img, ((panel_w - img_size) // 2, 35))
        d = ImageDraw.Draw(canvas)
        d.text((10, 5), title, fill=(40, 40, 40), font=font)
        d.text((10, panel_h - 50),
               f"iter {k:>3d} / {len(frames)-1}", fill=(40, 40, 40),
               font=font_small)
        d.text((10, panel_h - 28),
               f"misfit  {mis:.2e}", fill=(220, 70, 5), font=font_small)
        pil_frames.append(canvas)
    # Per-frame durations so the LAST frame holds for hold_last_ms.
    per_frame_ms = int(1000 / fps)
    durations = [per_frame_ms] * (len(pil_frames) - 1) + [hold_last_ms]
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=durations, loop=0, disposal=2)
    print(f"  saved {out_path}  ({len(pil_frames)} frames, "
          f"{per_frame_ms} ms each, {hold_last_ms} ms hold)")


def main():
    setup = ERTSetup(".")
    try:
        pipe = DDPMPipeline.from_pretrained("./ddpm_mnist_rot15")
        setup.unet = pipe.unet.eval()
        setup.scheduler = pipe.scheduler
        setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
        for p_ in setup.unet.parameters():
            p_.requires_grad_(False)

        # ---- Competition vector ----
        print("\n=== competition vector ===")
        t0 = time.time()
        frames = run_chain_capture_all(setup, seed=0)
        print(f"  chain done in {time.time()-t0:.0f}s")
        make_gif(frames, f"{OUT_DIR}/denoising_competition.gif",
                 title="PnP-DM on competition vector  (digit 5)")

        # ---- A few held-out digits ----
        # Synthesize y from a test image and re-run.
        mn = loadmat("MNIST Data/mnist.mat")
        M_test = mn['test']['images'][0, 0].astype(np.float32).reshape(784, -1)
        L_test = mn['test']['labels'][0, 0].flatten()
        if M_test.max() > 1.5: M_test = M_test / 255.0
        true_y_obs = setup.y_obs

        for digit in [1, 7, 8]:
            tidx = int(np.where(L_test == digit)[0][0])
            true_x = M_test[:, tidx].reshape(28, 28).astype(np.float32)
            with torch.no_grad():
                setup.y_obs = setup.forward(
                    torch.tensor(true_x, dtype=torch.float32))
            print(f"\n=== held-out digit {digit} (test idx {tidx}) ===")
            t0 = time.time()
            frames = run_chain_capture_all(setup, seed=0)
            print(f"  chain done in {time.time()-t0:.0f}s")
            make_gif(frames, f"{OUT_DIR}/denoising_digit{digit}.gif",
                     title=f"PnP-DM on a held-out {digit}")
        setup.y_obs = true_y_obs
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
