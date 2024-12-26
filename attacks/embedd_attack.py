import torch
import torch.nn as nn


class EmbeddAttack:
    def __init__(self, model, norm_type="l_inf", eps=4/255, step_size=0.1, max_iter=5):
        """
        :model nn.Module: image-to-image network
        :eps float: maximum allowed perturbation for the given norm
        :max_iter int: the number of iterations to refine the input image
        """
        self.model = model
        self.norm_type = norm_type  # L1, L2, or L∞
        self.eps = eps 
        self.step_size = step_size
        self.max_iter = max_iter

        self.loss = lambda x1, x2: nn.CosineEmbeddingLoss()(
            x1.flatten(start_dim=1), x2.flatten(start_dim=1), torch.ones(x1.shape[0]).to(x1.device)
        )


    def _normalize_perturbation(self, x_adv, x_clean):
        """
        Constrain perturbation magnitude.
        """
        diff = x_adv - x_clean

        if self.norm_type == "l1":
            norm = diff.abs().sum(dim=(1, 2, 3), keepdim=True)
        elif self.norm_type == "l2":
            norm = diff.norm(p=2, dim=(1, 2, 3), keepdim=True)
        elif self.norm_type == "l_inf":
            norm = torch.amax(diff.abs(), dim=(1, 2, 3), keepdim=True)
        else:
            raise ValueError("Norm type must be either 'l1', 'l2', or 'l_inf'")
        
        scale_factor = self.eps / (norm + 1e-8)
        scale_factor = torch.minimum(scale_factor, torch.ones_like(scale_factor))

        return (x_clean + diff * scale_factor).data


    def __call__(self, x, target):
        """
        Perform the attack to maximize the difference between the model's output and the target.
        
        :x: Input image to perturb
        :target: Ground truth target image
        :return: Perturbed input image
        """
        # Ensure the input is a tensor with requires_grad
        x_ = x.clone().detach().requires_grad_(True)
        with torch.no_grad():
            target_feat = self.model(x, return_feat=True)

        for i in range(self.max_iter):
            self.model.zero_grad()

            output_feat = self.model(x_, return_feat=True)

            loss = self.loss(output_feat, target_feat) 

            loss.backward()

            with torch.no_grad():
                x_.data = x_.data + self.step_size * torch.sign(x_.grad)
                x_.data = self._normalize_perturbation(x_, x).clamp(-1, 1)

        return x_.detach()  