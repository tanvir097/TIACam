import torch
import numpy as np

from invariant_learning.utils.image_utils import tensor_batch_to_pil_list
from invariant_learning.utils.text_utils import get_mismatched_texts
from invariant_learning.training.losses import (
    discriminator_loss,
    generator_loss
)


class TIACamTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        clip_model,
        clip_processor,
        distortion_pipeline,
        g_optimizer,
        d_optimizer,
        device
    ):
        self.generator = generator
        self.discriminator = discriminator

        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.distortion_pipeline = distortion_pipeline

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.device = device

    def train_epoch(self, dataloader):
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        sim_I_T_all = []
        sim_Ip_T_all = []
        sim_I_T_fake_all = []

        for images, texts in dataloader:
            images = images.to(self.device)

            distorted_images = self.distortion_pipeline(images)
            distorted_images = distorted_images.to(self.device)

            fake_texts = get_mismatched_texts(texts)

            images_pil = tensor_batch_to_pil_list(images)
            distorted_pil = tensor_batch_to_pil_list(distorted_images)

            with torch.no_grad():
                real_inputs = self.clip_processor(
                    text=texts,
                    images=images_pil,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                distorted_inputs = self.clip_processor(
                    text=texts,
                    images=distorted_pil,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                fake_inputs = self.clip_processor(
                    text=fake_texts,
                    images=images_pil,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                outputs_real = self.clip_model(**real_inputs)
                outputs_distorted = self.clip_model(**distorted_inputs)
                outputs_fake = self.clip_model(**fake_inputs)

            f_I = outputs_real.image_embeds
            f_T = outputs_real.text_embeds

            f_Ip = outputs_distorted.image_embeds
            f_T_fake = outputs_fake.text_embeds

            z_I = self.generator(f_I)
            z_T = self.generator(f_T)

            z_Ip = self.generator(f_Ip)
            z_T_fake = self.generator(f_T_fake)

            real_pairs = torch.cat([
                torch.cat([z_I, z_T], dim=1),
                torch.cat([z_Ip, z_T], dim=1)
            ], dim=0)

            fake_pairs = torch.cat([
                torch.cat([z_I, z_T_fake], dim=1),
                torch.cat([z_Ip, z_T_fake], dim=1)
            ], dim=0)

            all_pairs = torch.cat([
                real_pairs,
                fake_pairs
            ], dim=0)

            labels_real = torch.tensor(
                [[1, 0]] * real_pairs.size(0),
                dtype=torch.float32,
                device=self.device
            )

            labels_fake = torch.tensor(
                [[0, 1]] * fake_pairs.size(0),
                dtype=torch.float32,
                device=self.device
            )

            all_labels = torch.cat([
                labels_real,
                labels_fake
            ], dim=0)

            perm = torch.randperm(all_pairs.size(0))

            all_pairs = all_pairs[perm]
            all_labels = all_labels[perm]

            self.d_optimizer.zero_grad()

            d_logits = self.discriminator(
                all_pairs[:, :1024],
                all_pairs[:, 1024:]
            )

            d_loss = discriminator_loss(
                d_logits,
                all_labels
            )

            d_loss.backward()

            self.d_optimizer.step()

            self.g_optimizer.zero_grad()

            generator_pairs = torch.cat([
                torch.cat([z_I, z_T], dim=1),
                torch.cat([z_Ip, z_T], dim=1),
                torch.cat([z_I, z_T_fake], dim=1),
                torch.cat([z_Ip, z_T_fake], dim=1)
            ], dim=0)

            fool_labels = torch.tensor(
                [[1.0, 0.0]] * generator_pairs.size(0),
                device=self.device
            )

            logits = self.discriminator(
                generator_pairs[:, :1024],
                generator_pairs[:, 1024:]
            )

            g_loss, metrics = generator_loss(
                logits,
                fool_labels,
                z_I,
                z_Ip,
                z_T,
                z_T_fake
            )

            g_loss.backward()

            self.g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            sim_I_T_all.append(metrics["sim_I_T"])
            sim_Ip_T_all.append(metrics["sim_Ip_T"])
            sim_I_T_fake_all.append(metrics["sim_I_T_fake"])

        print(
            f"D Loss: {total_d_loss / len(dataloader):.4f} | "
            f"G Loss: {total_g_loss / len(dataloader):.4f}"
        )

        print(
            f"I-T: {np.mean(sim_I_T_all):.4f} | "
            f"I'-T: {np.mean(sim_Ip_T_all):.4f} | "
            f"I-T_fake: {np.mean(sim_I_T_fake_all):.4f}"
        )