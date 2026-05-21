import torch
import torch.nn.functional as F


def warp_image(image, homography):
    b, c, h, w = image.shape

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=image.device),
        torch.linspace(-1, 1, w, device=image.device),
        indexing="ij"
    )

    ones = torch.ones_like(x)

    grid = torch.stack([x, y, ones], dim=-1)
    grid = grid.view(-1, 3).T
    grid = grid.unsqueeze(0).repeat(b, 1, 1)

    warped_grid = homography @ grid
    warped_grid = warped_grid[:, :2, :] / (warped_grid[:, 2:3, :] + 1e-8)

    warped_grid = warped_grid.view(b, 2, h, w)
    warped_grid = warped_grid.permute(0, 2, 3, 1)

    return F.grid_sample(
        image,
        warped_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )