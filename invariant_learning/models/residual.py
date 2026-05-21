import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)