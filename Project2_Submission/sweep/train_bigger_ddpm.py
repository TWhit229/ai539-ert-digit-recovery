"""Train a larger UNet2DModel DDPM on rotated MNIST with EMA weights.

Existing prior: 1aurent/ddpm-mnist (~1.07 M params).
This script:  ~6 M param UNet2DModel, trained from scratch with rotation
augmentation in ±15°, with an EMA copy of the weights maintained during
training. EMA almost always improves DDPM sample quality.

On a 5090 this takes ~10-20 minutes for 30 epochs. On MPS, ~2 hours.
"""
import argparse, time, os, math, copy
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

import sys; sys.path.insert(0, "../code")
from finetune_rotated_mnist import RotatedMNIST  # reuse the data loader


class EMA:
    """Exponential moving average of model parameters.
    Stores a shadow copy of parameters and updates it after each opt step:
        shadow = decay * shadow + (1 - decay) * current.
    At eval time, copy shadow into the live model.
    """
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].data)


def build_bigger_unet():
    """About 6 M parameters. Tuned for 28x28."""
    return UNet2DModel(
        sample_size=28,
        in_channels=1, out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D",
                          "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D",
                        "AttnUpBlock2D", "UpBlock2D"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--ema_decay",  type=float, default=0.9995)
    p.add_argument("--rot_range",  type=float, default=15.0)
    p.add_argument("--out_dir",    default="./ddpm_bigger_rot15_ema")
    args = p.parse_args()

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device: {device}")

    unet = build_bigger_unet().to(device).train()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {n_params/1e6:.2f} M")

    scheduler = DDPMScheduler(num_train_timesteps=1000,
                               beta_schedule="squaredcos_cap_v2")

    print("Loading MNIST...")
    mn = loadmat("../code/MNIST Data/mnist.mat")
    M = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
    if M.max() > 1.5: M = M / 255.0
    ds = RotatedMNIST(M, rot_range=args.rot_range)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-6)
    ema = EMA(unet, decay=args.ema_decay)

    print(f"\nTraining {args.epochs} epochs, lr={args.lr}, ema={args.ema_decay}")
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = []
        for step, batch in enumerate(dl):
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            t = torch.randint(0, 1000, (batch.size(0),), device=device).long()
            noisy = scheduler.add_noise(batch, noise, t)
            opt.zero_grad()
            eps_pred = unet(noisy, t).sample
            loss = Fnn.mse_loss(eps_pred, noise)
            loss.backward()
            opt.step()
            ema.update(unet)
            losses.append(loss.item())
            if (step + 1) % 100 == 0:
                print(f"  ep {epoch+1}/{args.epochs}  step {step+1}/{len(dl)}  "
                      f"loss {np.mean(losses[-50:]):.4f}  ({time.time()-t0:.0f}s)")
        print(f"  ep {epoch+1} done — mean loss {np.mean(losses):.4f}  "
              f"({time.time()-t0:.0f}s total)")

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy EMA weights into the model before saving
    ema.copy_to(unet)
    pipe = DDPMPipeline(unet=unet.cpu().eval(), scheduler=scheduler)
    pipe.save_pretrained(args.out_dir)
    print(f"\nSaved EMA pipeline to {args.out_dir}")

    # Sanity samples
    pipe.unet = pipe.unet.to(device)
    pipe.scheduler.set_timesteps(100)
    with torch.no_grad():
        x = torch.randn(8, 1, 28, 28, device=device)
        for t in pipe.scheduler.timesteps:
            eps = pipe.unet(x, t).sample
            x = pipe.scheduler.step(eps, t, x).prev_sample
    x = (x.cpu() * 0.5 + 0.5).clamp(0, 1).numpy()
    fig, axes = plt.subplots(1, 8, figsize=(13, 2))
    for k in range(8):
        axes[k].imshow(x[k, 0], cmap='gray', vmin=0, vmax=1); axes[k].axis('off')
    plt.suptitle(f"Unconditional samples — bigger UNet ({n_params/1e6:.1f}M, EMA)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/uncond_samples.png", dpi=110, facecolor='white')
    print(f"Saved {args.out_dir}/uncond_samples.png")


if __name__ == "__main__":
    main()
