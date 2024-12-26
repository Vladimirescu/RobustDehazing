import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Attack, differential_evolution

# TODO: check if any bad normalization happens

class OnePixelAttack(Attack):
    """
    Attack in the paper 'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]

    Modified from 
    "https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/onepixel.html
    to be used in image-to-image tasks.

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        pixels (int): number of pixels to change (Default: 1)
        steps (int): number of steps. (Default: 10)
        popsize (int): population size, i.e. the number of candidate agents or "parents" in differential evolution (Default: 10)
    """

    def __init__(self, model, pixels=1, steps=10, popsize=10, inf_batch=16):
        super().__init__("OnePixel", model)
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        
        self.inf_batch = inf_batch

        self.loss = lambda x, y: -np.sum(np.abs(x - y), axis=(1, 2, 3))

    def forward(self, images, targets):

        images = images.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)

        batch_size, channel, height, width = images.shape

        bounds = [(0, height), (0, width)] + [(0, 1)] * channel
        bounds = bounds * self.pixels

        popmul = max(1, int(self.popsize / len(bounds)))

        adv_images = []
        for idx in range(batch_size):
            image, target = images[idx : idx + 1], targets[idx : idx + 1]

            def func(delta):
                return self._loss(image, target, delta)

            delta = differential_evolution(
                func=func,
                bounds=bounds,
                callback=None,
                maxiter=self.steps,
                popsize=popmul,
                init="random",
                recombination=1,
                atol=-1,
                polish=False,
            ).x
            
            delta = np.split(delta, len(delta) / len(bounds))
            adv_image = self._perturb(image, delta)
            adv_images.append(adv_image)

        adv_images = torch.cat(adv_images)
        return adv_images

    def _loss(self, image, target, delta):
        # adds multiple perturbations over the same image
        adv_images = self._perturb(image, delta)  # image - (1, C, H, W), delta - (N, C, H, W)
        out = self._get_prob(adv_images)

        target_ = target.repeat(out.shape[0], 1, 1, 1).cpu().numpy()

        return self.loss(
            out, target_
        )

    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.get_logits(batch)
                outs.append(out)
        outs = torch.cat(outs)
        
        return outs.detach().cpu().numpy()

    def _perturb(self, image, delta):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image] * num_delta, dim=0)
        for idx in range(num_delta):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                pos_x, pos_y = pixel[:2]
                channel_v = pixel[2:]
                for channel, v in enumerate(channel_v):
                    adv_images[idx, channel, int(pos_x), int(pos_y)] = v
        return adv_images