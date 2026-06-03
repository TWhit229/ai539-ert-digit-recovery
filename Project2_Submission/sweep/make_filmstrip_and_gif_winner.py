"""Run all 32 cond+CFG chains (the actual sampler) and use the WINNER chain
(lowest final misfit) for both the filmstrip and the animated GIF.

Previous make_filmstrip_cfg.py just ran chain 0 with class=5. That chain
isn't necessarily the winner. The actual sampler runs 32 chains seeded
to classes 0..9 cycling, and the chain with the lowest final misfit is
the one we use as the answer. To match the deck's narrative, the
filmstrip and animation must come from THAT chain.

Outputs:
  ../figures/fig_denoising_progression.png       (6-frame filmstrip)
  ../figures/lesson/fig_denoising_progression.png (same)
  ../figures/gifs/denoising_competition.gif       (animated, 81 frames)
  ../figures/gifs/frames_competition/frame_0..80.png (PNGs for animategraphics)
"""
import os, sys, time
sys.path.insert(0, "../code")

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from pnp_dm import gn_likelihood_prox
from pnp_dm_cfg import diff_one_denoise_cfg, load_cond_ddpm
from schedule import get_schedule
from train_cond_ddpm import N_CLASSES

OUT_FIG_TALK   = "../figures"
OUT_FIG_LESSON = "../figures/lesson"
OUT_GIF_DIR    = "../figures/gifs"
COND_DIR = "./ddpm_cond_rot15_ema"

N_CHAINS = 32
N_ITER   = 80
SAVE_AT_FILMSTRIP = [0, 4, 12, 25, 45, 79]  # 0-indexed; title shows k+1


def get_font(size=18):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()


def main():
    setup = ERTSetup("../code")
    try:
        unet, sch = load_cond_ddpm(COND_DIR)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        alphas_cumprod = setup.alphas_cumprod
        print(f"Loaded conditional DDPM from {COND_DIR}")

        sigmas = get_schedule("geomspace", n_iter=N_ITER,
                               sigma_max=1.0, sigma_min=0.05)

        # Replay the exact same sequence the production sampler uses
        # (32 chains, seed=0, class label cycles 0..9).
        torch.manual_seed(0); np.random.seed(0)

        all_chain_frames = []
        t0 = time.time()
        for c in range(N_CHAINS):
            class_label = c % N_CLASSES
            x = torch.randn(28, 28).clamp(-1.5, 1.5)
            frames_this_chain = []
            x_pix0 = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
            mis0 = 0.5 * float(((setup.forward(torch.tensor(x_pix0, dtype=torch.float32))
                                  - setup.y_obs) ** 2).sum())
            frames_this_chain.append((0, float(sigmas[0]), mis0, x_pix0))

            for k, sigma_k in enumerate(sigmas):
                z = gn_likelihood_prox(setup, x, float(sigma_k),
                                        sigma_n=1e-4, max_inner=2)
                x = diff_one_denoise_cfg(unet, alphas_cumprod, z, float(sigma_k),
                                          class_label, w=3.0)
                x_pix = x_diff_to_pix(x).clamp(0.0, 1.0).numpy().astype(np.float32)
                mis = 0.5 * float(((setup.forward(torch.tensor(x_pix, dtype=torch.float32))
                                     - setup.y_obs) ** 2).sum())
                frames_this_chain.append((k + 1, float(sigma_k), mis, x_pix))
            final_mis = frames_this_chain[-1][2]
            print(f"  chain {c:2d}  class={class_label}  final misfit {final_mis:.3e}  "
                  f"({time.time()-t0:.0f}s)")
            all_chain_frames.append({
                'chain_idx': c,
                'class_label': class_label,
                'final_misfit': final_mis,
                'frames': frames_this_chain,
            })

        # Pick the winner
        all_chain_frames.sort(key=lambda c: c['final_misfit'])
        winner = all_chain_frames[0]
        print(f"\nWinner: chain {winner['chain_idx']}, class={winner['class_label']}, "
              f"final misfit {winner['final_misfit']:.3e}")

        winning_frames = winner['frames']
        win_class = winner['class_label']

        # ----- Filmstrip (6 snapshots) -----
        snapshots = [winning_frames[k] for k in [s + 1 for s in SAVE_AT_FILMSTRIP]]
        # +1 because winning_frames[0] is the initial noise (iter 0) and
        # winning_frames[k+1] is after iteration k.
        fig, axes = plt.subplots(1, len(snapshots), figsize=(2.0*len(snapshots), 2.4))
        for ax, (k, sig, mis, img) in zip(axes, snapshots):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"iter {k}\nmisfit {mis:.1e}", fontsize=11)
            ax.axis('off')
        plt.suptitle(f"Winner chain (class={win_class}) over {N_ITER} iterations",
                      fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        for d in (OUT_FIG_TALK, OUT_FIG_LESSON):
            os.makedirs(d, exist_ok=True)
            out = f"{d}/fig_denoising_progression.png"
            plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
            print(f"Saved {out}")
        plt.close()

        # ----- GIF + frame PNGs (all 81 frames) -----
        font = get_font(size=22); font_small = get_font(size=16)
        img_size = 28 * 10
        panel_w = max(img_size + 20, 480); panel_h = img_size + 90
        os.makedirs(f"{OUT_GIF_DIR}/frames_competition", exist_ok=True)
        pil_frames = []
        for (k, sig, mis, x_pix) in winning_frames:
            x_uint8 = (np.clip(x_pix, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(x_uint8, mode='L').resize((img_size, img_size),
                                                              Image.NEAREST).convert('RGB')
            canvas = Image.new('RGB', (panel_w, panel_h), 'white')
            canvas.paste(img, ((panel_w - img_size) // 2, 35))
            d = ImageDraw.Draw(canvas)
            d.text((10, 5), f"PnP-DM + CFG  (class={win_class}, winner)",
                    fill=(40, 40, 40), font=font)
            d.text((10, panel_h - 50),
                   f"iter {k:>3d} / {N_ITER}", fill=(40, 40, 40), font=font_small)
            d.text((10, panel_h - 28),
                   f"misfit  {mis:.2e}", fill=(220, 70, 5), font=font_small)
            pil_frames.append(canvas)
            canvas.save(f"{OUT_GIF_DIR}/frames_competition/frame_{k}.png")

        per_frame_ms = int(1000 / 8)  # 8 fps
        durations = [per_frame_ms] * (len(pil_frames) - 1) + [2500]
        out_gif = f"{OUT_GIF_DIR}/denoising_competition.gif"
        pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:],
                            duration=durations, loop=0, disposal=2)
        print(f"Saved {out_gif}  ({len(pil_frames)} frames)")
        print(f"Saved frame PNGs to {OUT_GIF_DIR}/frames_competition/")
    finally:
        setup.quit()


if __name__ == "__main__":
    main()
