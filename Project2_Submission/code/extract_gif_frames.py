"""
Extract animated-GIF frames as numbered PNGs for use with LaTeX's
animate package (\animategraphics). One subdirectory per GIF.
"""
import os
from PIL import Image

GIFS = [
    ("../figures/gifs/denoising_competition.gif", "../figures/gifs/frames_competition"),
    ("../figures/gifs/denoising_digit1.gif",      "../figures/gifs/frames_digit1"),
    ("../figures/gifs/denoising_digit7.gif",      "../figures/gifs/frames_digit7"),
    ("../figures/gifs/denoising_digit8.gif",      "../figures/gifs/frames_digit8"),
]

for src, dst_dir in GIFS:
    os.makedirs(dst_dir, exist_ok=True)
    g = Image.open(src)
    n = 0
    while True:
        g.convert('RGB').save(f"{dst_dir}/frame_{n:03d}.png")
        n += 1
        try:
            g.seek(g.tell() + 1)
        except EOFError:
            break
    print(f"{src}: extracted {n} frames to {dst_dir}")
