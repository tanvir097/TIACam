import torch.nn as nn
import torch.nn.functional as F

from invariant_learning.models.residual import ResidualBlock


class Generator(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=1024,
        output_dim=1024,
        use_l2_norm=True
    ):
        super().__init__()

        self.use_l2_norm = use_l2_norm

        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.resblocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblocks(x)
        x = self.fusion(x)
        x = self.projection(x)

        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)

        return x