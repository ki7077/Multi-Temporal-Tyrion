import os

import numpy as np
import torch

from CaFFe.constants import *


def turn_class_labels_to_zones_torch_caffee(mask):
    if isinstance(mask,torch.Tensor):
        mask = mask.type(torch.uint8)
        mask_class_labels = torch.ones(mask.shape,dtype=torch.uint8)*15
    elif isinstance(mask,np.ndarray):
        mask = mask.astype(np.uint8)
        mask_class_labels = np.ones(mask.shape,dtype=np.uint8)*15
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 1] = 64
    mask_class_labels[mask == 2] = 127
    mask_class_labels[mask == 3] = 254
    return mask_class_labels


def uncertainty_norm(img):

    return torch.clip(img *255.0,min=0.0,max=255.0)

if __name__ == "__main__":

    #TODO add checkpoints
    ensembling_dirs = []


    raw_parent_dir = None#TODO
    save_dir = os.path.join(raw_parent_dir,"output_images","complete_images")
    save_dir_uncertainty = os.path.join(raw_parent_dir,"output_images","uncertainty")

    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(save_dir_uncertainty,exist_ok=True)

    for i in range(len(ensembling_dirs)):
        ensembling_dirs[i] = os.path.join(ensembling_dirs[i],"output_images","prob")


    for file in os.listdir(ensembling_dirs[0]):
        files = []
        for dir in ensembling_dirs:
            files.append(torch.load(os.path.join(dir,file)).type(torch.float16))

        C, H, W = files[0].shape

        q = torch.zeros((len(files),C,H,W),dtype=torch.float16)
        # now we got all the files evaluate an average and save it
        counter = 0
        for f in files:
            q[counter] = f
            counter = counter + 1

        std_q = q.std(dim=0)
        mean_q = q.mean(dim=0)

        back(turn_class_labels_to_zones_torch_caffee(torch.argmax(mean_q,dim=0).type(torch.uint8))).save(os.path.join(save_dir,file+".png"))
        for i in range(std_q.shape[0]):
            back(uncertainty_norm(std_q[i]).type(torch.uint8)).save(os.path.join(save_dir_uncertainty,file+"class+"+str(i)+"_uncertainty.png"))