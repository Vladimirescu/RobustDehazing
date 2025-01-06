import torch
import torch.nn as nn
from models import *


class LastLayerTune(nn.Module):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__()
        
        self.model = pretrained_model

        for name, p in self.model.named_parameters():
            p.requires_grad = False
            
        if isinstance(self.model, DehazeFormer):
            for p in self.model.patch_unembed.parameters():
                p.requires_grad = True
        elif isinstance(self.model, FFA):
            ...
        elif isinstance(self.model, MB_TaylorFormer):
            for p in self.model.output.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown model type {self.model}.")
                    
    def forward(self, x):
        return self.model(x)