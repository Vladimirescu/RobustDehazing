import numpy as np
import cv2
import torch
import torch.nn.functional as F
import einops
from omegaconf import OmegaConf
import os

from models import *
from fine_tune import *


str_to_ft = {
    "difffit": DiffFitModel,
    "last": LastLayerTune,
    "adapt": TargetAdapt
}


def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if "state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint['state_dict']
    elif "model" in checkpoint:
        checkpoint_state_dict = checkpoint['model']
    elif "params" in checkpoint:
        checkpoint_state_dict = checkpoint['params']
    else:
        checkpoint_state_dict = checkpoint
    
    model_state_dict = model.state_dict()

    new_state_dict = {}
    for key in checkpoint_state_dict:
        key_stripped = key.replace("model.", "").replace("module.", "")  # Remove prefixes
        for model_key in model_state_dict:
            model_key_stripped = model_key.replace("model.", "").replace("module.", "")
            # Match by name and shape
            if key_stripped == model_key_stripped and model_state_dict[model_key].shape == checkpoint_state_dict[key].shape:
                new_state_dict[model_key] = checkpoint_state_dict[key]
                break

    model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Weights loaded with {len(new_state_dict) / len(model_state_dict) * 100} % parameters matched.")
    
    return model


def load_model(args):
    allowed_extensions = [".pth", ".pk", ".ckpt"]
    
    if not args.fine_tuned:
        network = eval(args.model.replace('-', '_'))()
        network.cuda()
        
        saved_model_dir = os.path.join(args.save_dir, "base", args.model)

        exists = False

        print(saved_model_dir)

        for ext in allowed_extensions:
            if os.path.exists(saved_model_dir + ext):
                print(f"Loading {saved_model_dir + ext}")
                network = load_model_weights(network, saved_model_dir + ext)
                exists = True
                break

        if not exists:
            raise ValueError(f"No existing checkpoint")

        return network

    else:
        folder = os.path.join(args.save_dir, "fine_tuned", args.model)
        setting = OmegaConf.load(
            os.path.join(folder, "setting")
        )
  
        base = setting.base_model
        fine = setting.ft.fine_tune_type
        kwgs = {} if "fine_tune_kwargs" not in setting.ft.keys() else setting.ft.fine_tune_kwargs
  
        network = eval(base.replace('-', '_'))()
        saved_model_dir = os.path.join(args.save_dir, "fine_tuned", args.model, "fine_tuned")

        if fine not in str_to_ft.keys():
            raise ValueError(f"Unknown {fine}")

        network = str_to_ft[fine](network, **kwgs)
  
        exists = False
        for ext in allowed_extensions:
            if os.path.exists(saved_model_dir + ext):
                print(f"Loading {saved_model_dir + ext}")
                network = load_model_weights(network, saved_model_dir + ext)
                exists = True
                break
    
        if not exists:
            raise ValueError(f"No existing checkpoint")

        return network


def extract_patches(image, patch_size=512):
    """
    Extracts patches of size (patch_size, patch_size) from a given image.
    If H, W are not divisible by patch_size, padding is added.
    """
    B, C, H, W = image.shape
    
    # Calculate padding sizes
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # Apply padding if necessary
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

    patches = einops.rearrange(
        padded_image, 
        'b c (h ph) (w pw) -> (b h w) c ph pw', 
        ph=patch_size, pw=patch_size
    )
    
    return patches, padded_image


def reconstruct_image(output_patches, original_image, patch_size=512):
    """
    Reconstructs the image from the processed patches.
    """
    B, C, H, W = original_image.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    # Get the number of patches in the height and width dimensions
    num_patches_h = (H + pad_h) // patch_size
    num_patches_w = (W + pad_w) // patch_size

    # Rearrange the patches back to the full image grid
    reconstructed_image = einops.rearrange(
        output_patches, 
        '(b h w) c ph pw -> b c (h ph) (w pw)', 
        b=B, h=num_patches_h, w=num_patches_w, ph=patch_size, pw=patch_size
    )

    # Remove padding
    return reconstructed_image[:, :, :H, :W]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count
            

def read_img(filename):
    img = cv2.imread(filename)
    return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
    img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    cv2.imwrite(filename, img)


def hwc_to_chw(img):
    if len(img.shape) == 2:
        return img[np.newaxis, ...]
    elif len(img.shape) == 3:
        return np.transpose(img, axes=[2, 0, 1]).copy()
    elif len(img.shape) == 4:
        return np.transpose(img, axes=[0, 3, 1, 2]).copy()


def chw_to_hwc(img):
    if len(img.shape) == 3:
        return np.transpose(img, axes=[1, 2, 0]).copy()
    elif len(img.shape) == 4:
        return np.transpose(img, axes=[0, 2, 3, 1]).copy()