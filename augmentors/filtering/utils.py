import numpy as np
import torch
import torch.nn.functional as F


def motion_kernel(k=3, length=2, angle_deg=0):
    kernel = np.zeros((k, k), np.float32)
    center = (k - 1) / 2

    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle), np.sin(angle)

    for i in range(-length // 2, length // 2 + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))

        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0

    s = kernel.sum()
    return kernel / s if s > 0 else np.ones((k, k), np.float32) / (k * k)


def disk_kernel(k=3, radius=1.0):
    ax = np.arange(k) - (k - 1) / 2
    xx, yy = np.meshgrid(ax, ax)

    mask = (xx ** 2 + yy ** 2) <= radius ** 2
    kernel = mask.astype(np.float32)

    s = kernel.sum()
    return kernel / s if s > 0 else np.ones((k, k), np.float32) / (k * k)


def depthwise_conv(images, kernel):
    k = kernel.shape[0]
    weight = torch.from_numpy(kernel).to(images.device)
    weight = weight.view(1, 1, k, k).repeat(images.size(1), 1, 1, 1)

    return F.conv2d(images, weight, padding=k // 2, groups=images.size(1))


def build_filtering_targets(images, kernel_size=3, seed=0):
    rng = np.random.default_rng(seed)
    targets = []

    for i in range(images.size(0)):
        if rng.random() < 0.5:
            length = int(rng.integers(2, kernel_size + 1))
            angle = float(rng.uniform(0, 180))
            kernel = motion_kernel(kernel_size, length, angle)
        else:
            radius = float(rng.uniform(1, (kernel_size - 1) / 2))
            kernel = disk_kernel(kernel_size, radius)

        target = depthwise_conv(images[i:i + 1], kernel)
        targets.append(target)

    return torch.cat(targets, dim=0).clamp(0, 1)