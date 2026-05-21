import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from invariant_learning.datasets.vg_dataset import VGDataset
from invariant_learning.datasets.collate import custom_collate

from invariant_learning.models.generator import Generator
from invariant_learning.models.discriminator import Discriminator

from invariant_learning.clip.clip_loader import load_clip

from invariant_learning.distortions.manager import build_distortions

from invariant_learning.training.distortion_pipeline import (
    RandomDistortionPipeline
)

from invariant_learning.training.trainer import TIACamTrainer


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = VGDataset(
    root_dir="dataset/final",
    image_size=128
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate
)

clip_model, clip_processor = load_clip(device)

generator = Generator(output_dim=1024).to(device)

discriminator = Discriminator().to(device)

distortions = build_distortions()

pipeline = RandomDistortionPipeline(
    distortions
)

g_optimizer = optim.Adam(
    generator.parameters(),
    lr=1e-5
)

d_optimizer = optim.Adam(
    discriminator.parameters(),
    lr=1e-4
)

trainer = TIACamTrainer(
    generator=generator,
    discriminator=discriminator,
    clip_model=clip_model,
    clip_processor=clip_processor,
    distortion_pipeline=pipeline,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    device=device
)

for epoch in range(100):
    print(f"\nEpoch {epoch + 1}")
    trainer.train_epoch(dataloader)