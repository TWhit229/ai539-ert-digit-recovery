"""Make animated GIFs of the NEW (cond + CFG) pipeline denoising over time.

Replaces code/make_denoising_gifs.py for the post-sweep narrative. For
each requested image (competition vector + three held-out test digits),
run one PnP-DM-CFG chain seeded to the truth's class label, capture the
pixel image at every Gibbs iteration, and assemble into an annotated
animated GIF. Frames are also extracted as numbered PNGs for use with
LaTeX's animate package.
"""
import os, sys, time
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image, ImageDraw, ImageFont

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from pnp_dm_cfg import diff_one_denoise_cfg, load_cond_ddpm
from schedule import get_schedule

OUT_DIR = "../figures/gifs"
COND_DIR = "./ddpm_cond_rot15_ema"
os.makedirs(OUT_DIR, exist_ok=True)


def get_font(size=18):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def run_chain_capture_all(setup, unet, alphas_cumprod, class_label,
                           n_iter=80, sigma_n=1e-4, cfg_w=3.0, gn_inner=2,
                           seed=0):
    """Run one PnP-DM-CFG chain seeded to class_label, capture every iter."""
    torch.manual_seed(seed); np.random.seed(seed)
    sigmas = get_schedule("geomspace", n_iter=n_iter,
                           sigma_max=1.0, sigma_min=0.05)
    x = torch.randn(28, 28).clamp(-1.5, 1.5)
    frames = []
    x_pix0 = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
    mis0 = 0.5 * float(((setup.forward(torch.tensor(x_pix0, dtype=torch.float32))
                          - setup.y_obs) ** 2).sum())
    frames.append((0, float(sigmas[0]), mis0, x_pix0))

    for k, sigma_k in enumerate(sigmas):
        z = gn_likelihood_prox(setup, x, float(sigma_k),
                                sigma_n=sigma_n, max_inner=gn_inner)
        x = diff_one_denoise_cfg(unet, alphas_cumprod, z, float(sigma_k),
                                  class_label, w=cfg_w)
        x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
        mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                            - setup.y_obs) ** 2).sum())
        frames.append((k + 1, float(sigma_k), mis, x_pix))
    return frames


def make_gif_and_frames(frames, out_gif, frames_dir, scale=10, fps=8,
                         hold_last_ms=2500, title=""):
    """Assemble frames into an animated GIF AND write numbered PNGs for
    \\animategraphics."""
    os.makedirs(frames_dir, exist_ok=True)
    font = get_font(size=22)
    font_small = get_font(size=16)
    img_size = 28 * scale
    panel_w = max(img_size + 20, 480)
    panel_h = img_size + 90
    pil_frames = []
    for (k, sigma_k, mis, x_pix) in frames:
        x_uint8 = (np.clip(x_pix, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(x_uint8, mode='L').resize(
            (img_size, img_size), Image.NEAREST).convert('RGB')
        canvas = Image.new('RGB', (panel_w, panel_h), 'white')
        canvas.paste(img, ((panel_w - img_size) // 2, 35))
        d = ImageDraw.Draw(canvas)
        d.text((10, 5), title, fill=(40, 40, 40), font=font)
        d.text((10, panel_h - 50),
               f"iter {k:>3d} / {len(frames)-1}", fill=(40, 40, 40),
               font=font_small)
        d.text((10, panel_h - 28),
               f"misfit  {mis:.2e}", fill=(220, 70, 5), font=font_small)
        pil_frames.append(canvas)
        # Write unpadded frame for \animategraphics
        canvas.save(f"{frames_dir}/frame_{k}.png")

    per_frame_ms = int(1000 / fps)
    durations = [per_frame_ms] * (len(pil_frames) - 1) + [hold_last_ms]
    pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:],
                       duration=durations, loop=0, disposal=2)
    print(f"  saved {out_gif}  ({len(pil_frames)} frames)")
    print(f"  saved frames to {frames_dir}/")


def main():
    setup = ERTSetup("../code")
    try:
        unet, sch = load_cond_ddpm(COND_DIR)
        alphas_cumprod = sch.alphas_cumprod.float()
        # Use cond's scheduler so diff_one_denoise_cfg's t_idx lookup matches
        setup.scheduler = sch
        setup.alphas_cumprod = alphas_cumprod
        print(f"Loaded conditional DDPM from {COND_DIR}")

        # ---- Competition vector (digit 5) ----
        print("\n=== competition vector (class=5) ===")
        t0 = time.time()
        frames = run_chain_capture_all(setup, unet, alphas_cumprod,
                                        class_label=5, seed=0)
        print(f"  chain done in {time.time()-t0:.0f}s")
        make_gif_and_frames(frames, f"{OUT_DIR}/denoising_competition.gif",
                             f"{OUT_DIR}/frames_competition",
                             title="PnP-DM + CFG  (class=5)")

        # ---- Held-out digits 1, 7, 8 ----
        mn = loadmat("../code/MNIST Data/mnist.mat")
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
            print(f"\n=== held-out digit {digit} (test idx {tidx}, class={digit}) ===")
            t0 = time.time()
            frames = run_chain_capture_all(setup, unet, alphas_cumprod,
                                            class_label=digit, seed=0)
            print(f"  chain done in {time.time()-t0:.0f}s")
            make_gif_and_frames(frames,
                                 f"{OUT_DIR}/denoising_digit{digit}.gif",
                                 f"{OUT_DIR}/frames_digit{digit}",
                                 title=f"PnP-DM + CFG on held-out {digit}")
        setup.y_obs = true_y_obs
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
