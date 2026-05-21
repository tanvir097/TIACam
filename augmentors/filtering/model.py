import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelGenerator(nn.Module):
    def __init__(self, latent_dim=512, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, kernel_size * kernel_size)
        )

    def forward(self, z):
        k = self.mlp(z)
        k = F.softplus(k) + 1e-6
        k = k / k.sum(dim=1, keepdim=True)
        return k.view(-1, 1, self.kernel_size, self.kernel_size)


def apply_psf(images, kernels):
    b, c, h, w = images.shape
    k = kernels.size(-1)

    kernels = kernels.repeat(1, c, 1, 1)
    images = images.view(1, b * c, h, w)
    kernels = kernels.view(b * c, 1, k, k)

    out = F.conv2d(images, kernels, padding=k // 2, groups=b * c)
    return out.view(b, c, h, w)