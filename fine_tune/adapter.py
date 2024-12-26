import torch
import torch.nn as nn
from models import *


class TargetAdapt(nn.Module):
    """
    Adds a learnable linear layer after each target block.
    """
    def __init__(self, pretrained_model):
        super().__init__()
        
        self.model = pretrained_model
        self.new_weights = nn.ParameterList()

        for name, p in self.model.named_parameters():
            p.requires_grad = False
            
        if isinstance(self.model, DehazeFormer):
            target_layer = TransformerBlock
        elif isinstance(self.model, FFA):
            target_layer = FFABlock
        else:
            raise ValueError(f"Unknown model type {self.model}.")

        self._add_new_params(target_layer)

    def _add_new_params(self, target_layer):
        for name, module in self.model.named_modules():
            if isinstance(module, target_layer):
        
                out_ch = module.mlp.mlp[-1].out_channels
                adaptor = nn.Conv2d(
                    out_ch, out_ch, 3, 1, 1, bias=False
                )
                identity_init(adaptor)

                original_forward = module.forward
                
                def new_forward(x, adaptor=adaptor, original_forward=original_forward):
                    return adaptor(original_forward(x))
                
                module.forward = new_forward
                self.new_weights.append(adaptor.weight)

    def forward(self, x):
        return self.model(x)
        

def identity_init(conv_layer):
    """
    Initializes a Conv2d layer to act as an identity mapping.
    
    Args:
        conv_layer (nn.Conv2d): A Conv2d layer with in_channels == out_channels.
    """
    # Ensure it's a square kernel and has the same number of input and output channels
    assert conv_layer.in_channels == conv_layer.out_channels, "Input and output channels must be the same"
    assert conv_layer.kernel_size[0] == conv_layer.kernel_size[1], "Kernel must be square"

    # Get the kernel size
    k = conv_layer.kernel_size[0]
    # Initialize weights to zeros
    with torch.no_grad():
        conv_layer.weight.zero_()
        # Set the center of the kernel for each channel to 1
        for i in range(conv_layer.in_channels):
            conv_layer.weight[i, i, k // 2, k // 2] = 1
        # Optionally, set bias to zero (if it exists)
        if conv_layer.bias is not None:
            conv_layer.bias.zero_()


class FirstLastAdapt(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        
        self.model = pretrained_model

        for name, p in self.model.named_parameters():
            p.requires_grad = False
            
        self.pre_adapt, self.post_adapt = get_pre_post_modules(self.model) 
                    
    def forward(self, x):
        pre = self.pre_adapt(x)
        middle = self.model(pre)
        post = self.post_adapt(middle)
        
        return post
    
    
def get_pre_post_modules(model: nn.Module):
    
    if isinstance(model, DehazeFormer):
        in_ch = model.patch_embed.in_chans
        out_ch = 3 # although they use 4 to splint and combine features from 4 channels
        
        pre = Adaptor(in_ch, 4, 3)
        post = Adaptor(out_ch, 4, 3)
        
    elif isinstance(self.model, FFA):
        ...
        
    else:
        raise ValueError(f"Unknown model type {self.model}.")
        
    return pre, post

    
class Adaptor(nn.Module):
    """
    Linear adaptor with identity intialization.
    """
    def __init__(self, in_ch, inner_mul=4, kernel_size=3):
        super(Adaptor, self).__init__()
        
        self.in_ch = in_ch
        self.inner_mul = inner_mul
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv2d(in_ch, in_ch * inner_mul, 
                               kernel_size=self.kernel_size, 
                               padding=self.kernel_size // 2,
                               bias=False)
        self.conv2 = nn.Conv2d(in_ch * inner_mul, in_ch,
                                kernel_size=self.kernel_size, 
                                padding=self.kernel_size // 2,
                                bias=False)
        self._identity_init()
        
    def _identity_init(self):
        xy_loc = self.kernel_size // 2
        
        with torch.no_grad():
            self.conv1.weight.zero_()
            for i in range(self.conv1.in_channels):
                for j in range(self.inner_mul):
                    self.conv1.weight[i * self.inner_mul + j, i, xy_loc, xy_loc] = 1

            self.conv2.weight.zero_()
            for i in range(self.conv2.out_channels):
                for j in range(self.inner_mul):
                    self.conv2.weight[i, i * self.inner_mul + j, xy_loc, xy_loc] = 1 / self.inner_mul
        
    def forward(self, x):
        return self.conv2(self.conv1(x))
    