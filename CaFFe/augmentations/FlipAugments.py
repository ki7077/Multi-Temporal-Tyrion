import torch
import torch
import torchvision
from torch import nn
import numpy as np
"""
    Augments for both Images + Masks
"""
class FlipAugments(nn.Module):
    def __init__(self, p_flip_v=0.,
                p_rotate= 0.,
                p_flip_h=0.,
                p_rotate_any = 0.,
                 ):
        self.p_flip_v= p_flip_v
        self.p_rotate = p_rotate
        self.p_flip_h = p_flip_h
        self.p_rotate_any = p_rotate_any

    def __call__(self, target_image, target_mask):
        if self.p_flip_h > torch.rand(1):
            target_image,target_mask= self.Fliph(target_image,target_mask)

        if self.p_flip_v > torch.rand(1):
            target_image,target_mask = self.Flipv(target_image,target_mask)

        if self.p_rotate > torch.rand(1):
            target_image,target_mask= self.Rotate(target_image,target_mask)

        if self.p_rotate_any > torch.rand(1):
            target_image,target_mask= self.Rotate_any_direction(target_image,target_mask)

        return (target_image,target_mask)

    def Fliph(self, target_image, target_mask):
        target = torchvision.transforms.functional.hflip(target_image)
        mask_target = torchvision.transforms.functional.hflip(target_mask)
        return target, mask_target

    def Flipv(self, target_image,  target_mask):
        target = torchvision.transforms.functional.vflip(target_image)
        mask_target = torchvision.transforms.functional.vflip(target_mask)
        return target, mask_target
    def Rotate(self, target_image, target_mask):
        random = np.random.randint(0, 3)
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
        target = torchvision.transforms.functional.rotate(target_image, angle=angle)
        mask_target = torchvision.transforms.functional.rotate(target_mask.unsqueeze(dim=0), angle=angle)[0]

        return target, mask_target.squeeze(0)
    def Rotate_any_direction(self, target_image,target_mask):
        random = np.random.randint(0,360)
        og_size = target_mask.shape

        pad_transform = torchvision.transforms.Pad((target_mask.shape[0]//2,target_mask.shape[1]//2), padding_mode='symmetric')

        centering = torchvision.transforms.CenterCrop(og_size)

        target_image = pad_transform(target_image)
        target_mask = pad_transform(target_mask.unsqueeze(dim=0))


        target = torchvision.transforms.functional.rotate(target_image, angle=random)
        mask_target = torchvision.transforms.functional.rotate(target_mask, angle=random)

        target = centering(target)
        mask_target = centering(mask_target)[0]

        return target, mask_target.squeeze(0)





class FlipMultiAugments(nn.Module):
    def __init__(self, p_flip_v=0.,
                p_rotate= 0.,
                p_flip_h=0.,
                p_rotate_any = 0.,
                 ):
        self.p_flip_v= p_flip_v
        self.p_rotate = p_rotate
        self.p_flip_h = p_flip_h
        self.p_rotate_any = p_rotate_any

    def __call__(self, target_images, target_masks):
        if self.p_flip_h > torch.rand(1):
            for i in range(len(target_images)):
                target_images[i],target_masks[i]= self.Fliph(target_images[i],target_masks[i])

        if self.p_flip_v > torch.rand(1):
            for i in range(len(target_images)):
                target_images[i],target_masks[i] = self.Flipv(target_images[i],target_masks[i])

        if self.p_rotate > torch.rand(1):
            random = np.random.randint(0, 3)
            for i in range(len(target_images)):
                target_images[i],target_masks[i] = self.Rotate(target_images[i],target_masks[i],random)

        if self.p_rotate_any > torch.rand(1):
            random = np.random.randint(0, 360)

            for i in range(len(target_images)):
                target_images[i],target_masks[i] = self.Rotate_any_direction(target_images[i],target_masks[i],random)

        return (target_images,target_masks)

    def Fliph(self, target_image, target_mask):
        target = torchvision.transforms.functional.hflip(target_image)
        mask_target = torchvision.transforms.functional.hflip(target_mask)
        return target, mask_target

    def Flipv(self, target_image,  target_mask):
        target = torchvision.transforms.functional.vflip(target_image)
        mask_target = torchvision.transforms.functional.vflip(target_mask)
        return target, mask_target
    def Rotate(self, target_image, target_mask,random):
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
        target = torchvision.transforms.functional.rotate(target_image, angle=angle)
        mask_target = torchvision.transforms.functional.rotate(target_mask.unsqueeze(dim=0), angle=angle)[0]

        return target, mask_target.squeeze(0)
    def Rotate_any_direction(self, target_image,target_mask,random):
        og_size = target_mask.shape

        pad_transform = torchvision.transforms.Pad((target_mask.shape[0]//2,target_mask.shape[1]//2), padding_mode='symmetric')

        centering = torchvision.transforms.CenterCrop(og_size)

        target_image = pad_transform(target_image)
        target_mask = pad_transform(target_mask.unsqueeze(dim=0))


        target = torchvision.transforms.functional.rotate(target_image, angle=random)
        mask_target = torchvision.transforms.functional.rotate(target_mask, angle=random)

        target = centering(target)
        mask_target = centering(mask_target)[0]

        return target, mask_target.squeeze(0)