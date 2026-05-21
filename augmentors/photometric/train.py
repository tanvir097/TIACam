import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.image_datasets import SingleImageDataset
from utils.seed import set_seed
from utils.checkpoint import save_model
from augmentors.photometric.model import KernelGenerator


def build_target(images, mode="multiplicative"):
    device = images.device

    if mode == "multiplicative":
        mask = torch.empty(
            1, 3, images.size(2), images.size(3),
            device=device
        ).uniform_(0.9, 1.1)

        target = torch.clamp(images * mask, 0, 1)

    elif mode == "additive":
        noise = torch.empty(
            1, 3, images.size(2), images.size(3),
            device=device
        ).uniform_(-0.05, 0.05)

        target = torch.clamp(images + noise, 0, 1)

    else:
        raise ValueError("mode must be either 'multiplicative' or 'additive'")

    return target


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SingleImageDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
        max_images=args.max_images
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = KernelGenerator(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images in loader:
            images = images.to(device)
            targets = build_target(images, mode=args.mode)

            z = torch.randn(images.size(0), args.latent_dim, device=device)
            mask = model(z)

            if args.mode == "multiplicative":
                pred = images * (1.0 + args.alpha * mask)
            else:
                pred = images + args.beta * mask

            pred = torch.clamp(pred, 0, 1)

            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {total_loss / len(dataset):.6f}")

    save_model(model, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default="../dataset/images/VG_100K_2")
    parser.add_argument("--save_path", type=str, default="outputs/checkpoints/photometric.pth")
    parser.add_argument("--mode", type=str, default="multiplicative", choices=["multiplicative", "additive"])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)