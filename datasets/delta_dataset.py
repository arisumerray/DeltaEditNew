#фейк файл, надо удалить
import copy
import random

import torch
import torch.utils.data
import torch.distributions

import sys
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from PIL import Image
import collections
#from . import DATASETS_REGISTRY

#разделить на трейн-валидацию, убрать тест 
#get-item и collate-fn из hairswap
#@DATASETS_REGISTRY.add_to_registry("delta", ("train", "val"))
class DeltaDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, debug=False, cycle=True, **kwargs):
        super().__init__()
        
        style_latents_list = []
        clip_latents_list = []
        wplus_latents_list = []

        self.train = train

        data = collections.defaultdict(list)

        if train:
            
            style_latents_list.append(torch.Tensor(np.load(f"./latent_code/ffhq/sspace_ffhq_feat.npy")))
            clip_latents_list.append(torch.Tensor(np.load(f"./latent_code/ffhq/cspace_ffhq_feat.npy")))
            wplus_latents_list.append(torch.Tensor(np.load(f"./latent_code/ffhq/wspace_ffhq_feat.npy")))
        else:
            style_latents_list.append(torch.Tensor(np.load("./examples/sspace_img_feat.npy")))
            clip_latents_list.append(torch.Tensor(np.load("./examples/cspace_img_feat.npy")))
            wplus_latents_list.append(torch.Tensor(np.load("./examples/wplus_img_feat.npy")))
        
        self.style_latents = torch.cat(style_latents_list, dim=0)[:300]
        self.clip_latents = torch.cat(clip_latents_list, dim=0)[:300]
        self.wplus_latents = torch.cat(wplus_latents_list, dim=0)[:300]

        data['Style'] = torch.cat(style_latents_list, dim=0)[:10]
        data['Clip'] = torch.cat(clip_latents_list, dim=0)[:10]
        data['Wplus'] = torch.cat(wplus_latents_list, dim=0)[:10]

    #    self.dataset_size = self.style_latents.shape[0]
    #    self.cycle = cycle
        print(len(data[list(data.keys())[0]]))

        if train:

            self.dataset_size = self.style_latents.shape[0]
            print("dataset size", self.dataset_size)
            self.cycle = cycle
        
    def __len__(self):
        return self.style_latents.shape[0]
        
    def __getitem__(self, index):
        if self.cycle:
            index = index % self.dataset_size #посмотреть

        latent_s1 = self.style_latents[index]
        latent_c1 = self.clip_latents[index]
        latent_w1 = self.wplus_latents[index]
        latent_c1 = latent_c1 / latent_c1.norm(dim=-1, keepdim=True).float()

        if self.train:
            random_index = random.randint(0, self.dataset_size - 1)
            latent_s2 = self.style_latents[random_index]
            latent_c2 = self.clip_latents[random_index]
            latent_w2 = self.wplus_latents[random_index]
            latent_c2 = latent_c2 / latent_c2.norm(dim=-1, keepdim=True).float()

            delta_s1 = latent_s2 - latent_s1
            delta_c = latent_c2 - latent_c1
            
            delta_c = delta_c / delta_c.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)
            delta_c = torch.cat([latent_c1, delta_c], dim=0)

            return latent_s1, delta_c, delta_s1
        else:
            delta_c = torch.cat([latent_c1, latent_c1], dim=0)
        
            return latent_s1, delta_c, latent_w1
        
d = DeltaDataset()
print(d[0])