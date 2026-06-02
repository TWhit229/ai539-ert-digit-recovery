"""Train a class-conditional UNet2DModel on rotated MNIST with EMA and
classifier-free guidance dropout.

Each training sample includes its class label (0-9). With probability
p_drop=0.1 the label is replaced by a special 'unconditional' label
(class 10). At inference, classifier-free guidance combines the
conditional and unconditional epsilons:

    eps_guided = (1 + w) * eps_cond - w * eps_uncond

This is the standard CFG recipe (Ho & Salimans 2022). w=3-5 gives sharp
class-faithful samples; w=0 is plain conditional; w=-1 is unconditional.

Class conditioning gives the PnP-DM chains the option to commit to a
specific digit class early, which helps when the data is consistent with
multiple plausible digits.

On a 5090: ~20-30 min for 30 epochs. On MPS: ~3-4 hours.
"""
import argparse, time, os, math
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

import sys; sys.path.insert(0, "../code")
from finetune_rotated_mnist import rotate_batch_torch
from train_bigger_ddpm import EMA


N_CLASSES = 10
UNCOND_LABEL = N_CLASSES  # class 10 = "no condition"
N_EMB = N_CLASSES + 1


class RotatedMNISTWithLabel(Dataset):
    def __init__(self, M, labels, rot_range=15.0, p_drop=0.1):
        self.X = torch.tensor(M.T.reshape(-1, 1, 28, 28), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.rot_range = rot_range
        self.p_drop = p_drop

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        img = self.X[idx]
        theta = (torch.rand(1) * 2 - 1) * self.rot_range
        img_rot = rotate_batch_torch(img.unsqueeze(0), theta)[0]
        img_rot = img_rot * 2.0 - 1.0
        label = self.y[idx].item()
        if torch.rand(1).item() < self.p_drop:
            label = UNCOND_LABEL
        return img_rot, label


def build_cond_unet():
    """About 6 M params, with class embedding."""
    return UNet2DModel(
        sample_size=28,
        in_channels=1, out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D",
                          "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D",
                        "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=N_EMB,
    )


def cfg_eps(unet, x_t, t, class_label, w):
    """Classifier-free guided epsilon: (1+w)*eps_c - w*eps_u."""
    B = x_t.size(0)
    label_c = torch.full((B,), int(class_label), dtype=torch.long, device=x_t.device)
    label_u = torch.full((B,), UNCOND_LABEL,     dtype=torch.long, device=x_t.device)
    if w == 0:
        return unet(x_t, t, class_labels=label_c).sample
    if w == -1:
        return unet(x_t, t, class_labels=label_u).sample
    # Batch both passes for speed
    x_dbl = torch.cat([x_t, x_t], dim=0)
    t_dbl = t if t.ndim == 0 else torch.cat([t, t], dim=0)
    lab_dbl = torch.cat([label_c, label_u], dim=0)
    eps_dbl = unet(x_dbl, t_dbl, class_labels=lab_dbl).sample
    eps_c, eps_u = eps_dbl[:B], eps_dbl[B:]
    return (1 + w) * eps_c - w * eps_u


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--ema_decay",  type=float, default=0.9995)
    p.add_argument("--rot_range",  type=float, default=15.0)
    p.add_argument("--p_drop",     type=float, default=0.1)
    p.add_argument("--out_dir",    default="./ddpm_cond_rot15_ema")
    args = p.parse_args()

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device: {device}")

    unet = build_cond_unet().to(device).train()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {n_params/1e6:.2f} M (class-conditional, {N_EMB} embeddings)")

    scheduler = DDPMScheduler(num_train_timesteps=1000,
                               beta_schedule="squaredcos_cap_v2")

    print("Loading MNIST + labels...")
    mn = loadmat("../code/MNIST Data/mnist.mat")
    M  = mn['training']['images'][0, 0].astype(np.float32).reshape(784, -1)
    if M.max() > 1.5: M = M / 255.0
    y_arr = mn['training']['labels'][0, 0].astype(np.int64).reshape(-1)
    ds = RotatedMNISTWithLabel(M, y_arr, rot_range=args.rot_range, p_drop=args.p_drop)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-6)
    ema = EMA(unet, decay=args.ema_decay)

    print(f"\nTraining {args.epochs} epochs (label drop p={args.p_drop})")
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = []
        for step, (batch, lab) in enumerate(dl):
            batch = batch.to(device); lab = lab.to(device)
            noise = torch.randn_like(batch)
            t = torch.randint(0, 1000, (batch.size(0),), device=device).long()
            noisy = scheduler.add_noise(batch, noise, t)
            opt.zero_grad()
            eps_pred = unet(noisy, t, class_labels=lab).sample
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
    ema.copy_to(unet)
    # Save unet config + state dict + scheduler config separately (no DDPMPipeline
    # since it doesn't handle class_labels by default — we load components manually
    # in pnp_dm_cfg.py).
    unet.cpu().eval().save_pretrained(f"{args.out_dir}/unet")
    scheduler.save_pretrained(f"{args.out_dir}/scheduler")
    with open(f"{args.out_dir}/metadata.txt", 'w') as f:
        f.write(f"class_conditional=True\n")
        f.write(f"num_class_embeds={N_EMB}\n")
        f.write(f"uncond_label={UNCOND_LABEL}\n")
        f.write(f"params_M={n_params/1e6:.2f}\n")
    print(f"\nSaved class-conditional UNet + scheduler to {args.out_dir}")

    # Sanity sample: one digit per class, CFG w=3
    unet = unet.to(device).eval()
    scheduler.set_timesteps(100)
    fig, axes = plt.subplots(1, 10, figsize=(14, 2))
    with torch.no_grad():
        for cls in range(10):
            x = torch.randn(1, 1, 28, 28, device=device)
            for t in scheduler.timesteps:
                t_b = torch.tensor([int(t.item())], device=device).long()
                eps = cfg_eps(unet, x, t_b, cls, w=3.0)
                x = scheduler.step(eps, t, x).prev_sample
            img = (x.cpu() * 0.5 + 0.5).clamp(0, 1).squeeze().numpy()
            axes[cls].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[cls].set_title(str(cls), fontsize=10); axes[cls].axis('off')
    plt.suptitle(f"Class-conditional samples (CFG w=3, {n_params/1e6:.1f}M, EMA)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/cond_samples.png", dpi=110, facecolor='white')
    print(f"Saved {args.out_dir}/cond_samples.png")


if __name__ == "__main__":
    main()
