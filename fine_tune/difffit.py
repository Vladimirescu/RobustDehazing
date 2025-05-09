import torch
import torch.nn as nn
from models import *


class DiffFitModel(nn.Module):
    def __init__(self, pretrained_model, train_params=["bias"], **kwargs):
        super().__init__()
        
        self.model = pretrained_model
        self.new_weights = nn.ParameterList()
        self.train_params = train_params
        
        for name, p in self.model.named_parameters():
            p.requires_grad = False
            for tp in self.train_params:
                if tp in name:
                    p.requires_grad = True
            
        if isinstance(self.model, DehazeFormer):
            target_layer = TransformerBlock
        elif isinstance(self.model, FFA):
            target_layer = FFABlock
        elif isinstance(self.model, MB_TaylorFormer):
            target_layer = MHCAEncoder
        else:
            raise ValueError(f"Unknown model type {self.model}.")
            
        self._add_new_params(target_layer)

    def _get_new_forward(self, scale, original_forward):
        if isinstance(self.model, DehazeFormer):
            def new_forward(x, scale=scale, original_forward=original_forward):
                return scale * original_forward(x)
        elif isinstance(self.model, FFA):
            ValueError("Not implemented.")
        elif isinstance(self.model, MB_TaylorFormer):
            def new_forward(x, size, scale=scale, original_forward=original_forward):
                return scale * original_forward(x, size)

        return new_forward

    def _add_new_params(self, target_layer):
        
        for name, module in self.model.named_modules():
            if isinstance(module, target_layer):
        
                scale = nn.Parameter(torch.tensor(torch.ones(1)), requires_grad=True)
                
                self.new_weights.append(scale)

                original_forward = module.forward
                module.forward = self._get_new_forward(scale, original_forward)
                
    def forward(self, x):
        return self.model(x)