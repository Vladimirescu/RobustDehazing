import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from omegaconf import OmegaConf

from pytorch_msssim import ssim

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from fine_tune import *


str_to_ft = {
    "difffit": DiffFitModel,
    "last": LastLayerTune,
    "adapt": TargetAdapt
}


def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
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


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in tqdm(enumerate(test_loader)):
        inpt = batch['source'].cuda()
        target = batch['target'].cuda()
        filename = batch['filename']

        with torch.no_grad():
            output = network(inpt).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(
                1 / (F.mse_loss(output, target, reduction='none').mean(dim=(1, 2, 3)) + 1e-7)
            )

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)				

        PSNR.update(psnr_val.mean().item())
        SSIM.update(ssim_val.mean().item())

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        out_img = chw_to_hwc(output.detach().cpu().numpy())
        for i in range(out_img.shape[0]):
            f_result.write('%s,%.02f,%.03f\n'%(filename[i], psnr_val[i].item(), ssim_val[i].item()))
            write_img(os.path.join(result_dir, 'imgs', filename[i]), out_img[i])

    f_result.write('Avg PSNR/SSIM ,%.02f,%.03f\n'%(PSNR.avg, SSIM.avg))
    
    f_result.close()

    print(f"Average results: PSNR = {PSNR.avg:.2f} SSIM = {SSIM.avg:.4f}")


def load_model(args):
    allowed_extensions = [".pth", ".pk", ".ckpt"]
    
    if not args.fine_tuned:
        network = eval(args.model.replace('-', '_'))()
        network.cuda()
        
        saved_model_dir = os.path.join(args.save_dir, args.type, args.model)

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

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--data_dir', default='D:/RESIDE/', type=str, help='path to dataset')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
    parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
    parser.add_argument('--type', default='base', type=str, help='experiment setting')
    parser.add_argument("--fine_tuned", action="store_true", help="Whether this was a fine-tuned model. If so, lod in 2 stages.")
    args = parser.parse_args()
    
    network = load_model(args).cuda()
 
    test_dataset = PairLoader(args.data_dir, 'test', 'valid', size=256)
    test_loader = DataLoader(test_dataset,
                             batch_size=4,
                             num_workers=args.num_workers,
                             pin_memory=True)

    if args.fine_tuned:
        result_dir = os.path.join(args.result_dir, "fine_tuned", args.model)
    else:
        result_dir = os.path.join(args.result_dir, args.exp, args.model)
    test(test_loader, network, result_dir)