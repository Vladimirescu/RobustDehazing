import torch
import torch.nn as nn


class GaussNoiseAttack:
    def __init__(self, model, std=0.1):
        """
        :model nn.Module: image-to-image network
        :std float: standard deviation of noise
        """
        self.model = model
        self.std = std  # L1, L2, or L∞

    def __call__(self, x, target):
        """
        Returns Gaussian perturbed images.
        
        :x: Input image to perturb
        :target: dummy variable
        :return: Perturbed input image
        """

        return (x + torch.randn_like(x) * self.std).clamp(-1, 1) 