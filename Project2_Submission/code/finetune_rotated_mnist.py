"""
Fine-tune the pretrained MNIST DDPM (1aurent/ddpm-mnist) on rotated MNIST so
the prior natively generates rotated digits.

Why this matters: the competition vector is digit 5 rotated by ~7.5°. The
upright-only prior pushes samples toward upright digits, so during posterior
sampling the diffusion drifts away from the rotated truth. With rotation
augmentation in the training distribution, the prior natively supports
arbitrary rotation and DPS samples land much closer to the rotated truth.

Recipe (~20-30 min on MPS):
  - Load pretrained 1aurent/ddpm-mnist (UNet 1.07M params + DDPM scheduler).
  - Build rotation-augmented MNIST: each training image gets a random rotation
    uniform in [-15°, +15°] each epoch.
  - 5-10 epochs of standard DDPM denoising loss training.
  - Save fine-tuned checkpoint.
"""
import argparse, time, os, math
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline


def rotate_batch_torch(imgs, theta_deg):
    """Bilinear CCW rotation of (B, 1, 28, 28) images by per-sample theta_deg (B,).
    Implemented with affine_grid + grid_sample on whatever device imgs is on."""
    B = imgs.size(0)
    th = theta_deg * math.pi / 180.0
    cos, sin = torch.cos(th), torch.sin(th)
    # affine matrix maps OUTPUT grid → INPUT grid (inverse rotation).
    # For CCW rotation of image by theta, the sampling matrix is R(-theta):
    A = torch.zeros(B, 2, 3, dtype=imgs.dtype, device=imgs.device)
    A[:, 0, 0] = cos
    A[:, 0, 1] = sin
    A[:, 1, 0] = -sin
    A[:, 1, 1] = cos
    grid = Fnn.affine_grid(A, imgs.size(), align_corners=False)
    return Fnn.grid_sample(imgs, grid, mode='bilinear', padding_mode='zeros',
                           align_corners=False)


class RotatedMNIST(Dataset):
    """Wraps MNIST training images, applies a random rotation per __getitem__.

    All access happens through torch tensors in (1, 28, 28) shape in [-1, +1]
    (the DDPM training convention).
    """
    def __init__(self, M, rot_range=15.0):
        # M: (784, N) in [0, 1]
        self.X = torch.tensor(M.T.reshape(-1, 1, 28, 28), dtype=torch.float32)
        self.rot_range = rot_range

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        img = self.X[idx]  # (1, 28, 28), [0,1]
        # Random rotation in [-rot_range, rot_range]
        theta = (torch.rand(1) * 2 - 1) * self.rot_range
        img_rot = rotate_batch_torch(img.unsqueeze(0), theta)[0]
        # Map [0,1] → [-1, 1] (DDPM convention)
        return img_rot * 2.0 - 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--rot_range", type=float, default=15.0,
                   help="random rotation range in degrees, ±this")
    p.add_argument("--out_dir", default="./ddpm_mnist_rot15")
    args = p.parse_args()

    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # Pretrained DDPM (UNet + scheduler)
    print("Loading 1aurent/ddpm-mnist...")
    pipe = DDPMPipeline.from_pretrained("1aurent/ddpm-mnist")
    unet = pipe.unet.to(device).train()
    scheduler = pipe.scheduler

    # MNIST
    print("Loading MNIST...")
    mn = loadmat("MNIST Data/mnist.mat")
    M = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
    if M.max() > 1.5: M = M / 255.0
    ds = RotatedMNIST(M, rot_range=args.rot_range)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    n_train_t = scheduler.config.num_train_timesteps  # 1000

    # Show a few rotated samples for sanity
    fig, axes = plt.subplots(1, 8, figsize=(13, 2))
    for k in range(8):
        s = ds[k] * 0.5 + 0.5
        axes[k].imshow(s.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[k].axis('off')
    plt.suptitle("Rotation-augmented training samples (random θ ∈ ±15°)")
    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(f"{args.out_dir}/train_samples.png", dpi=110, facecolor='white')
    print(f"  Saved {args.out_dir}/train_samples.png")

    print(f"\nTraining {args.epochs} epochs, lr={args.lr}, batch {args.batch}, "
          f"rot ±{args.rot_range}°...")
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = []
        for step, batch in enumerate(dl):
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            t = torch.randint(0, n_train_t, (batch.size(0),), device=device).long()
            noisy = scheduler.add_noise(batch, noise, t)
            opt.zero_grad()
            eps_pred = unet(noisy, t).sample
            loss = Fnn.mse_loss(eps_pred, noise)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch+1}/{args.epochs}  step {step+1}/{len(dl)}  "
                      f"loss {np.mean(losses[-50:]):.4f}  ({time.time()-t0:.0f}s)")
        print(f"  epoch {epoch+1} done — mean loss {np.mean(losses):.4f}  "
              f"({time.time()-t0:.0f}s total)")

    # Save the fine-tuned UNet + scheduler
    pipe.unet = unet.cpu().eval()
    pipe.save_pretrained(args.out_dir)
    print(f"\nSaved fine-tuned pipeline to {args.out_dir}")

    # Quick sanity sample
    print("\nSanity check — sampling 8 unconditional digits from the new prior:")
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
        axes[k].imshow(x[k, 0], cmap='gray', vmin=0, vmax=1)
        axes[k].axis('off')
    plt.suptitle(f"Unconditional samples from fine-tuned DDPM (rot ±{args.rot_range}°)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/uncond_samples.png", dpi=110, facecolor='white')
    print(f"  Saved {args.out_dir}/uncond_samples.png")


if __name__ == "__main__":
    main()
