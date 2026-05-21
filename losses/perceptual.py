import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.to(device).eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.mse_loss(self.vgg(pred), self.vgg(target))