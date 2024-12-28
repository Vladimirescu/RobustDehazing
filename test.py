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

from utils import AverageMeter, write_img, chw_to_hwc, extract_patches, reconstruct_image, load_model
from datasets.loader import PairLoader


def predict_patches(model, imgs, bs=4):
    outs = []
    
    n = imgs.shape[0] // bs
    rest = imgs.shape[0] % bs
    
    for i in range(n):
        outs.append(model(imgs[i * bs: (i + 1) * bs]))
        
    if rest > 0:
        outs.append(model(imgs[-rest:]))
        
    return torch.cat(outs, dim=0)
    

def test(test_loader, network, result_dir, 
         max_size=1024, 
         show_original_psnr=False):
    PSNR = AverageMeter()
    if show_original_psnr:
        PSNR_o = AverageMeter()
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
            _, _, H, W = inpt.shape
            if H > max_size or W > max_size:
                """First approach - patchify, predict, reconstruct"""
                # patches, padded_image = extract_patches(inpt, max_size)
                # print(f"Constructed patches {patches.shape}.")
                # output_patches = predict_patches(network, patches)
                # output = reconstruct_image(output_patches, inpt, max_size)
                
                """Second approach - reshape"""
                inpt = F.interpolate(inpt, size=(max_size, max_size), mode="bilinear", align_corners=False)
                target = F.interpolate(target, size=(max_size, max_size), mode="bilinear", align_corners=False)
                output = network(inpt)
            else:
                output = network(inpt)

            output = output.clamp(-1, 1) * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(
                1 / (F.mse_loss(output, target, reduction='none').mean(dim=(1, 2, 3)) + 1e-7)
            )
            if show_original_psnr:
                # inpt is always in [-1, 1]
                inpt = inpt * 0.5 + 0.5
                psnr_o = 10 * torch.log10(
                    1 / (F.mse_loss(inpt, target, reduction='none').mean(dim=(1, 2, 3)) + 1e-7)
                )

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)				

        PSNR.update(psnr_val.mean().item())
        SSIM.update(ssim_val.mean().item())
        if show_original_psnr:
            PSNR_o.update(psnr_o.mean().item())

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        out_img = chw_to_hwc(output.detach().cpu().numpy())
        for i in range(out_img.shape[0]):
            f_result.write('%s,%.02f,%.03f\n'%(filename[i], psnr_val[i].item(), ssim_val[i].item()))
            write_img(os.path.join(result_dir, 'imgs', filename[i]), out_img[i])

    if show_original_psnr:
        f_result.write('Avg PSNR/PSNR_original/SSIM ,%.02f,%.02f,%.03f\n'%(PSNR.avg, PSNR_o.avg, SSIM.avg))
        f_result.close()
        print(f"Average results: PSNR = {PSNR.avg:.2f} (vs PSNR = {PSNR_o.avg:.2f}) SSIM = {SSIM.avg:.4f}")
    else:
        f_result.write('Avg PSNR/SSIM ,%.02f,%.03f\n'%(PSNR.avg, SSIM.avg))
        f_result.close()
        print(f"Average results: PSNR = {PSNR.avg:.2f} SSIM = {SSIM.avg:.4f}")




        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--dataset', default='reside', type=str, help='path to dataset')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
    parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
    parser.add_argument("--fine_tuned", action="store_true", help="Whether this was a fine-tuned model. If so, lod in 2 stages.")
    args = parser.parse_args()
    
    network = load_model(args).cuda()
    data_config = OmegaConf.load(
        os.path.join("./configs/data", args.dataset + ".yaml")
    )
    
    test_dataset = PairLoader(data_config.test_path, 'valid', size=data_config.test_size)
    test_loader = DataLoader(test_dataset,
                             batch_size=16 if data_config.test_size else 1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    if args.fine_tuned:
        result_dir = os.path.join(args.result_dir, data_config.name, "fine_tuned", args.model)
    else:
        result_dir = os.path.join(args.result_dir, data_config.name, "base", args.model)
    
    test(test_loader, network, result_dir)