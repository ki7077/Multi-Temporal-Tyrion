import numpy as np
import torch
from PIL import Image
from kornia import morphology
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomPerspective
from torchvision.transforms import RandomSolarize
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, adjust_gamma

from CaFFe.augmentations import ocrodeg


# context and input should have the same size otherwise it gets funny :)
def poisson_noise_ehancement(input, rate=1, peak=3.0):
    final_noise_input = np.zeros(input.shape).astype(np.float32)
    for i in range(rate):
        noise_input = np.random.poisson(peak, input.shape).astype(np.float32)
        final_noise_input = final_noise_input + (noise_input - peak)
    final_noise_input = final_noise_input / final_noise_input.max()
    input = np.clip(input + final_noise_input, a_min=0.0, a_max=1.0)
    return input


def poisson_noise_more_random(input, rate=1, peak=3.0):
    final_noise_input = np.zeros(input.shape).astype(np.float32)
    for i in range(rate):
        extra_scalar_input = np.random.rand(*input.shape)
        noise_input = np.random.poisson(peak, input.shape).astype(np.float32)
        final_noise_input = final_noise_input + extra_scalar_input * (noise_input - peak)
    final_noise_input = final_noise_input / final_noise_input.max()
    input = np.clip(input + final_noise_input, a_min=0.0, a_max=1.0)
    return input


# Pushes values towards the extreme values of the color spectrum
def extremize(input, factor, threshold):
    input = np.clip(np.where(input > threshold, factor * input, input / factor), a_min=input.min(), a_max=input.max())
    return input,


# data augmentation based on https://github.com/NVlabs/ocrodeg
class OcrodegAug(nn.Module):
    def __init__(self,
                 p_dilation=0.,
                 p_erosion=0.,
                 p_distort_with_noise=0.,
                 p_background_noise=0.,
                 p_perspective=0.,
                 p_gamma=0.,
                 p_contrast=0.,
                 p_brightness=0.,
                 p_poisson=0.,
                 p_poisson_speckel=0.,
                 p_solarize=0.,
                 p_extremize=0.,
                 color_channels=1):
        super(OcrodegAug, self).__init__()

        self.p_dilation = p_dilation
        self.p_erosion = p_erosion
        self.p_distort_with_noise = p_distort_with_noise
        self.p_background_noise = p_background_noise
        self.noise_bg = ocrodeg.FastPrintlike()

        self.toTensor = transforms.ToTensor()
        self.color_channels = color_channels


        self.p_perspective = p_perspective
        self.p_contrast = p_contrast
        self.p_brightness = p_brightness
        self.p_gamma = p_gamma
        self.p_poisson = p_poisson
        self.p_poisson_speckel = p_poisson_speckel  # TODO maybe adjust noise level here
        self.p_solarize = p_solarize
        self.p_extremize = p_extremize

    def __call__(self, x, skip_noise=False):
        x = np.array(x)[0]
        x = x / 255.0  # (x.max() if x.max()>0 else 1)

        if self.p_extremize > torch.rand(1):
            factor = np.random.uniform(0.5, 1.4)
            thresh_hold = int(np.random.uniform(0.33, 0.66))
            x = extremize(x, factor=factor, threshold=thresh_hold)

        if self.p_poisson > torch.rand(1) and not skip_noise:
            # 1 -3 repeats
            repeats = 1
            peak = int(np.random.rand() * 2) + 1
            x = poisson_noise_ehancement(x, repeats, peak)

        if self.p_poisson_speckel > torch.rand(1) and not skip_noise:
            repeats = 1
            peak = int(np.random.rand() * 3) + 1
            x = poisson_noise_more_random(x, repeats, peak)

        if self.p_erosion > torch.rand(1):
            kernel = torch.ones(tuple(torch.randint(low=2, high=4, size=(2,))))
            x = torch.from_numpy(x).view(1, self.color_channels, x.shape[0], x.shape[1])
            x = morphology.erosion(x, kernel).squeeze().numpy()

        if self.p_dilation > torch.rand(1):
            kernel = torch.ones(tuple(torch.randint(low=2, high=4, size=(2,))))
            x = torch.from_numpy(x).view(1, self.color_channels, x.shape[0], x.shape[1])
            x = morphology.dilation(x, kernel).squeeze().numpy()

        for sigma in [2, 3]:
            if self.p_distort_with_noise > torch.rand(1) and not skip_noise:
                noise = ocrodeg.bounded_gaussian_noise(x.shape, sigma, 2.0)
                x = ocrodeg.distort_with_noise(x, noise)

        if self.p_background_noise > torch.rand(1) and not skip_noise:
            x = 1 - self.noise_bg(x)

        x = Image.fromarray((x * 255).astype(np.uint8))

        if self.p_perspective > torch.rand(1):
            scale = np.random.uniform(0.0, 0.15)
            x = RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=255)(x)

        if self.p_contrast > torch.rand(1):
            factor = np.random.uniform(0.6, 1.4)
            x = adjust_contrast(x, factor)

        if self.p_brightness > torch.rand(1):
            factor = np.random.uniform(0.6, 1.4)
            x = adjust_brightness(x, factor)

        if self.p_solarize > torch.rand(1):
            thresh_hold = 10 + int(np.random.uniform() * 20)
            x = RandomSolarize(threshold=thresh_hold, p=1.0)(x)

        if self.p_gamma > torch.rand(1):
            factor = np.random.uniform(0.7, 1.3)
            x = adjust_gamma(x, factor)

        return np.expand_dims(np.array(x), axis=0)
