import os
import argparse
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from collections import OrderedDict

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import AverageMeter
from datasets.loader import PairLoader
from datasets.sampler import RandomSampler
from models import *
from fine_tune.tuner import FineTuningLightningModule
from test import load_model_weights


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='name of Fine Tune config file')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--save_dir', default='./saved_models/fine_tuned/', type=str, help='path to models saving')
    parser.add_argument('--data_dir', default='D:/RESIDE/', type=str, help='path to dataset')
    parser.add_argument('--gpu', default='0,', type=str, help='GPUs used for training')
    args = parser.parse_args()
    
    setting_file = os.path.join('configs/finetune', args.config + ".json")
    if not os.path.exists(setting_file):
        raise ValueError(f"Unknown file {setting_file}")

    setting = OmegaConf.load(setting_file)

    base_network = eval(setting.base_model)()
    base_network = load_model_weights(base_network, setting.resume_path)
    
    model = FineTuningLightningModule(base_network, **setting["ft"])

    train_dataset = PairLoader(args.data_dir, 'train/train/train', 'train', 
                                setting.patch_size, 
                                setting.edge_decay)
    sampler = RandomSampler(train_dataset, num_samples=setting.samples_per_epoch)
    train_loader = DataLoader(train_dataset,
                                batch_size=setting.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True,
                                sampler=sampler)

    val_dataset = PairLoader(args.data_dir, 'test', "valid", setting.patch_size)
    val_loader = DataLoader(val_dataset,
                            batch_size=setting.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)

    post_fix = '' if "save_postfix" not in setting.keys() else "_" + setting.save_postfix
    folder_name = setting.base_model + "_" + setting.ft.fine_tune_type + "_" + setting.ft.attack + "_" + setting.ft.train_type + post_fix
                    
    save_path = os.path.join(args.save_dir, folder_name)
    
    os.makedirs(save_path, exist_ok=True)
 
    OmegaConf.save(config=setting, f=os.path.join(save_path, "setting"))
 
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                          filename="fine_tuned")
    trainer = Trainer(
        max_epochs=setting.epochs, callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, train_loader, val_loader)
