import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets.image_datasets import SingleImageDataset
from utils.seed import set_seed
from utils.checkpoint import save_model
from augmentors.filtering.model import KernelGenerator, apply_psf
from augmentors.filtering.utils import build_filtering_targets


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SingleImageDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
        max_images=args.max_images
    )

    images = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)
    targets = build_filtering_targets(images, kernel_size=args.kernel_size).to(device)

    train_dataset = TensorDataset(images.cpu(), targets.cpu())
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = KernelGenerator(
        latent_dim=args.latent_dim,
        kernel_size=args.kernel_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            z = torch.randn(images.size(0), args.latent_dim, device=device)
            kernels = model(z)
            preds = apply_psf(images, kernels).clamp(0, 1)

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {total_loss / len(train_dataset):.6f}")

    save_model(model, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default="../dataset/images/VG_100K_2")
    parser.add_argument("--save_path", type=str, default="outputs/checkpoints/filtering.pth")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)