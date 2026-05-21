import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.image_datasets import PairedImageDataset
from utils.seed import set_seed
from utils.checkpoint import save_model
from augmentors.perspective.model import KernelGenerator
from augmentors.perspective.warp import warp_image


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairedImageDataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        image_size=args.image_size,
        max_images=args.max_images
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = KernelGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for original, target in loader:
            original = original.to(device)
            target = target.to(device)

            H = model(original)
            pred = warp_image(original, H)

            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {total_loss / len(loader):.6f}")

    save_model(model, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", type=str, default="../dataset/images/VG_100K_2")
    parser.add_argument("--target_dir", type=str, default="../dataset/perspective_transformed")
    parser.add_argument("--save_path", type=str, default="outputs/checkpoints/perspective.pth")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)