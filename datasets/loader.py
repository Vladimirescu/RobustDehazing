import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H-Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W-Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
            
    return imgs


def align(imgs=[], size=256):
    if size is None:
        return imgs
    else:
        H, W, _ = imgs[0].shape
        Hc, Wc = [size, size]

        Hs = (H - Hc) // 2
        Ws = (W - Wc) // 2
        for i in range(len(imgs)):
            imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

        return imgs


class PairLoader(Dataset):
    """
    Loader of (hazy, non-hazy) image pairs.
    Both are loaded and normalized in [-1, 1].
    
    To adapt each network, its recommended to add in the forward() of each module the corresponding 
    pre-processing.
    """
    def __init__(self, data_dir, mode, 
                    size=256, 
                    edge_decay=0, 
                    only_h_flip=True,
                    first_n_imgs=None):
                    
        assert mode in ['train', 'valid', 'test']

        if size is None:
            print("Changing to `test` mode.")
            mode = "test"

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = data_dir
  
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        
        self.img_names = self._filter_filenames(self.img_names)
        
        if isinstance(first_n_imgs, int):
            print(f"Keeping only the first {first_n_imgs} images.")
            self.img_names = self.img_names[:first_n_imgs]

        self.img_num = len(self.img_names)

    def _filter_filenames(self, files):
        new_files = []
        for f in files:
            if f.startswith("._"):
                continue
            else:
                new_files.append(f)
        
        return new_files

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
  
        if "GT" in img_name:
            # OHAZE has different names for hazy and gt
            img_name_haze = img_name.replace("GT", "hazy")
        else:
            img_name_haze = img_name
  
        source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name_haze)) * 2 - 1
        target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
  
        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}