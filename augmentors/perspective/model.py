import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class KernelGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)

        H = self.fc(feat)
        H = H.view(-1, 3, 3)

        H = H / (H[:, 2:3, 2:3] + 1e-8)

        return H