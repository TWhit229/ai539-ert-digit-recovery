"""
Bilinear CCW rotation around image center, zero-padded, cropped to original
size. Python port of Project 1's `myrotate.m`. Vectorized for batches.
"""
import numpy as np
import torch


def rotate28(img, theta_deg):
    """Rotate a single 28x28 image CCW by theta_deg.

    Args:
        img: (28, 28) numpy array or torch tensor, any float dtype.
        theta_deg: float, rotation in degrees.

    Returns:
        same shape/dtype as input.
    """
    is_torch = isinstance(img, torch.Tensor)
    if is_torch:
        device, dtype = img.device, img.dtype
        img_np = img.detach().cpu().numpy().astype(np.float64)
    else:
        img_np = np.asarray(img, dtype=np.float64)

    H, W = img_np.shape
    cx, cy = (W + 1) / 2.0, (H + 1) / 2.0
    th = theta_deg * np.pi / 180.0

    # 1-indexed grid to match MATLAB
    X, Y = np.meshgrid(np.arange(1, W + 1, dtype=np.float64),
                       np.arange(1, H + 1, dtype=np.float64))
    Xc = X - cx
    Yc = Y - cy
    Xs =  np.cos(th) * Xc + np.sin(th) * Yc + cx
    Ys = -np.sin(th) * Xc + np.cos(th) * Yc + cy

    # MATLAB's interp2 with 'linear', 0 fill
    Xs0 = np.floor(Xs).astype(int)
    Ys0 = np.floor(Ys).astype(int)
    dx = Xs - Xs0
    dy = Ys - Ys0
    out = np.zeros_like(img_np)

    for di in (0, 1):
        for dj in (0, 1):
            xi = Xs0 + di
            yi = Ys0 + dj
            w = (1 - dx if di == 0 else dx) * (1 - dy if dj == 0 else dy)
            inb = (xi >= 1) & (xi <= W) & (yi >= 1) & (yi <= H)
            xi_c = np.where(inb, xi - 1, 0)
            yi_c = np.where(inb, yi - 1, 0)
            out += np.where(inb, img_np[yi_c, xi_c] * w, 0.0)

    out = out.astype(img.dtype if not is_torch else np.float32)
    if is_torch:
        return torch.tensor(out, device=device, dtype=dtype)
    return out


def rotate28_batch(imgs, theta_deg):
    """Rotate a batch of images, identical theta for all.

    Args:
        imgs: (N, 28, 28) or (28, 28) — rotates each by theta_deg.
        theta_deg: float.

    Returns:
        same shape as input.
    """
    imgs = np.asarray(imgs, dtype=np.float64)
    single = (imgs.ndim == 2)
    if single:
        imgs = imgs[None, ...]
    N, H, W = imgs.shape
    cx, cy = (W + 1) / 2.0, (H + 1) / 2.0
    th = theta_deg * np.pi / 180.0
    X, Y = np.meshgrid(np.arange(1, W + 1, dtype=np.float64),
                       np.arange(1, H + 1, dtype=np.float64))
    Xc, Yc = X - cx, Y - cy
    Xs =  np.cos(th) * Xc + np.sin(th) * Yc + cx
    Ys = -np.sin(th) * Xc + np.cos(th) * Yc + cy
    Xs0 = np.floor(Xs).astype(int)
    Ys0 = np.floor(Ys).astype(int)
    dx, dy = Xs - Xs0, Ys - Ys0
    out = np.zeros_like(imgs)
    for di in (0, 1):
        for dj in (0, 1):
            xi = Xs0 + di
            yi = Ys0 + dj
            w = (1 - dx if di == 0 else dx) * (1 - dy if dj == 0 else dy)
            inb = (xi >= 1) & (xi <= W) & (yi >= 1) & (yi <= H)
            xi_c = np.where(inb, xi - 1, 0)
            yi_c = np.where(inb, yi - 1, 0)
            out += np.where(inb[None, :, :],
                            imgs[:, yi_c, xi_c] * w[None, :, :], 0.0)
    out = out.astype(np.float32)
    return out[0] if single else out
