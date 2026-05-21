import torch
import torch.nn as nn

from invariant_learning.models.transformer import TransformerBlock


class Discriminator(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        hidden_dim=512,
        depth=4,
        num_heads=8,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        self.token_proj = nn.Linear(embed_dim, hidden_dim)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, z_image, z_text):
        batch_size = z_image.size(0)

        img_token = self.token_proj(z_image).unsqueeze(1)
        txt_token = self.token_proj(z_text).unsqueeze(1)

        cls_token = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat([
            cls_token,
            img_token,
            txt_token
        ], dim=1)

        x = self.blocks(x)

        x = self.norm(x)

        cls_output = x[:, 0]

        return self.fc_out(cls_output)