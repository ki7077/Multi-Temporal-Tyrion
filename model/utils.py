"""
    code adapted from https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py"
"""
import importlib
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

"""
    utils file. Here a bunch of magic happens in regards to how the config create classes
"""

def get_yaml(application,filename,project_name="Multi-Temporal-Tyrion",config_directory="Config"):
    mypath = os.getcwd()
    path = mypath[:mypath.find(project_name) + len(project_name)]
    return os.path.join(path, config_directory,application,filename)

def get_dict_from_model_config(config):
    dict_1 = extract_dict(config["params"])
    # dict_2 = extract_dict(config["params"]["config"]["params"])
    # return {**dict_1, **dict_2}
    return dict_1

def extract_dict(config):
    return {k: v for k, v in config.items() if k != "config"}
def instantiate_completely(application,filename,**kwargs):
    file = get_yaml(application,filename)
    cfg = OmegaConf.load(file)
    return instantiate_from_config(cfg,**kwargs)

def instantiate_from_config(config,ckpt_bulk=None,**kwargs):
    if ckpt_bulk is not None:
        return get_obj_from_str(config["target"]).load_from_checkpoint(checkpoint_path=ckpt_bulk,
                                                                       **(config.get("params", dict())),
                                                                       strict=False, **kwargs)

    if "ckpt" in config:
        return get_obj_from_str(config["target"]).load_from_checkpoint(checkpoint_path=config["ckpt"],
                                                                       **(config.get("params", dict())),
                                                                       strict=False,**kwargs )

    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()),**kwargs)


def instantiate_from_config_function(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])


def instantiate_from_ckpt(config):
    return None

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def turn_colors_to_class_labels_zones_torch(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.type(torch.uint8)
        mask_class_labels = torch.ones(mask.shape, dtype=torch.uint8) * 15
    elif isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
        mask_class_labels = np.ones(mask.shape, dtype=np.uint8) * 15
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 64] = 1
    mask_class_labels[mask == 127] = 2
    mask_class_labels[mask == 254] = 3
    return mask_class_labels

def turn_class_labels_to_zones_torch(mask):
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

def whole_preprocess(pil_img):
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd[:, :, 0]
    H, W = img_trans.shape

    mask = np.ones([H, W]) * 15
    stone = np.where(img_trans == 0)
    na_area = np.where(img_trans == 63)
    na_areas = np.where(img_trans == 64)
    glacier = np.where(img_trans == 127)
    ocean_ice = np.where(img_trans == 254)
    mask[stone] = 0
    mask[na_area] = 1
    mask[na_areas] = 1
    mask[glacier] = 2
    mask[ocean_ice] = 3

    return mask

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module




# From SeMask https://github.com/Picsart-AI-Research/SeMask-Segmentation
def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


"""
    Flop counting from https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/
    
"""
from torch.utils.flop_counter import FlopCounterMode
def get_flops(model, inp, with_backward=False,**kwargs):

    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp,**kwargs).sum().backward()
        else:
            model(inp,**kwargs)
    total_flops = flop_counter.get_total_flops()

    return total_flops


from datetime import date
import pandas as pd
def get_monthly_dates_dict_CaFFe():
    s_date = date(1995, 1, 1)
    e_date = date(2020, 12, 31)
    dates_daily = pd.date_range(s_date, e_date, freq='d').strftime('%Y-%m-%d').tolist()

    date_to_number = dict()
    names = np.array(dates_daily)
    for i in range(names.shape[0]):
        date_to_number[names[i]] = i
    return date_to_number

