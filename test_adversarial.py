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
from attacks import get_attack_from_config
from test import load_model


def test_adv(test_loader, network, result_dir, attack_config):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LINF = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'input_adv'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    adv_attack = get_attack_from_config(network, attack_config)

    for idx, batch in tqdm(enumerate(test_loader)):
        inpt = batch['source'].cuda()
        target = batch['target'].cuda()
        filename = batch['filename']

        input_adv = adv_attack(inpt, target)

        LINF.update(
            (torch.amax(torch.abs(input_adv - inpt), dim=(1, 2, 3))).mean().item()
        )

        with torch.no_grad():
            output = network(input_adv).clamp_(-1, 1)

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
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
              'LINF: {linf.val:.03f} ({linf.avg:.03f})\t'
              .format(idx, psnr=PSNR, ssim=SSIM, linf=LINF))

        out_img = chw_to_hwc(output.detach().cpu().numpy())
        input_adv = (input_adv * 0.5 + 0.5).clamp_(0, 1)
        input_adv = chw_to_hwc(input_adv.detach().cpu().numpy())
        input_clean = (inpt * 0.5 + 0.5).clamp_(0, 1)
        input_clean = chw_to_hwc(input_clean.detach().cpu().numpy())
        
        for i in range(out_img.shape[0]):
            f_result.write('%s,%.02f,%.03f\n'%(filename[i], psnr_val[i].item(), ssim_val[i].item()))
            
            write_img(os.path.join(result_dir, 'imgs', filename[i]), out_img[i])
            write_img(os.path.join(result_dir, "input_adv", filename[i]), input_adv[i])

    f_result.write('Avg PSNR/SSIM/LINF ,%.02f,%.03f,%.03f\n'%(PSNR.avg, SSIM.avg, LINF.avg))
    f_result.close()

    print(f"Average results: PSNR = {PSNR.avg:.2f} SSIM = {SSIM.avg:.4f} LINF = {LINF.avg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--data_dir', default='D:/RESIDE/', type=str, help='path to dataset')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
    parser.add_argument('--result_dir', default='./results_adv/', type=str, help='path to results saving')
    parser.add_argument('--type', default='base', type=str, help='experiment setting')
    parser.add_argument("--fine_tuned", action="store_true", help="Whether this was a fine-tuned model. If so, lod in 2 stages.")
    parser.add_argument('--attack_config', default='./configs/attacks/target_1.yaml', 
                        type=str, help='path to attack config')
    args = parser.parse_args()

    network = load_model(args).cuda()

    test_dataset = PairLoader(args.data_dir, 'test', 'valid')
    test_loader = DataLoader(test_dataset,
                             batch_size=4,
                             num_workers=args.num_workers,
                             pin_memory=True)

    attack_config = OmegaConf.load(args.attack_config)
    result_dir = os.path.join(args.result_dir, args.exp, args.model, attack_config.name)
    
    test_adv(test_loader, network, result_dir, attack_config)