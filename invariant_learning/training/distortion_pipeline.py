import random
import torch


class RandomDistortionPipeline:
    def __init__(self, distortions):
        self.distortions = distortions

    def __call__(self, images):
        distorted_images = []

        for image in images:
            distortion = random.choice(self.distortions)

            distorted = distortion(image).squeeze(0)

            distorted_images.append(distorted)

        return torch.stack(distorted_images)