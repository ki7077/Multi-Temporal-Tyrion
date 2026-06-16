import os.path

import PIL
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from CaFFe.constants import *
from CaFFe.utils import *


def get_split(split_dir,split,legacy=False):

    split_file_name = ""
    if split == "train":
        split_file_name = "train_new.txt"
        if legacy:
            split_file_name = "train_old.txt"
    elif split == "val":
        split_file_name = "val_new.txt"
        if legacy:
            split_file_name = "val_old.txt"

    elif split == "test":
        split_file_name = "test.txt"

    files = []

    with open(os.path.join(split_dir,split_file_name)) as f:
        for line in f:
            files.append(line.split("\n")[0])

    return files



def get_nr_of_patches_for_images(img,patch_size,context_size):
    if len(img.shape) == 3:
        _, H, W = img.shape
    else:
        H, W = img.shape

    extra_patches = max(int(context_size/patch_size-1.0),0)

    HH = int(H/patch_size)  - extra_patches#(H + 1) // patch_size + extra_add
    WW = int(W/patch_size) - extra_patches# (W + 1) // patch_size + extra_add

    return HH*WW



def get_patch(img, patch_size, context_size, idx):

        if len(img.shape) == 3:
            _,H,W = img.shape
        else:
            H,W = img.shape

        extra_add = -1
        if ((H + 1) // patch_size * patch_size + context_size) <= H and (
                ((W + 1) // patch_size * patch_size + context_size) <= W):
            extra_add = 0

        HH = (H + 1) // patch_size + extra_add
        WW = (W+1)//patch_size + extra_add

        # defines the top left corner of the patch
        row = idx // WW
        column = idx % WW

        if len(img.shape) == 3:
            context_crop = img[:, row * patch_size:(row) * patch_size + context_size,
                       column * patch_size:(column) * patch_size + context_size]
            context_crop = context_crop[0]

        else:
            context_crop = img[row * patch_size:(row) * patch_size + context_size,
                           column * patch_size:(column) * patch_size + context_size]

        print("context shape ",context_crop.shape)
        print(" row and column ",row, " and ", column)

        output_img = np.array(transforms.CenterCrop((patch_size, patch_size))(transforms.ToTensor()(context_crop)))
        context_crop = np.array(transforms.Resize((patch_size, patch_size),
                                                  interpolation=PIL.Image.NEAREST)(transforms.ToTensor()(context_crop)))

        return output_img, context_crop


def fetch_patches(parent_dir,split,patch_size,context_size,automatic_resizing=False,legacy=False):

    sample_names = []
    meta_data = dict()

    # all_files = os.listdir(split_sar_dir)
    all_files = get_split(parent_dir, split, legacy=legacy)
    all_files.sort()

    if split=="val":
        split="train"

    split_zones_dir = os.path.join(parent_dir, ZONES_DIR, split)
    split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)

    for file_name in all_files:
       # file_name = all_files[file_idx]
        name = file_name[:-4]
        img = Image.open(os.path.join(split_sar_dir, file_name)).convert('L')
        mask = Image.open(os.path.join(split_zones_dir, name + "_zones" + ".png")).convert('L')
        if automatic_resizing:
            resolution_factor = int(name.split('_')[-3])
            rescaling_factor = resolution_factor/STATIC_ZOOM_FACTOR
            rescaled_height = int(rescaling_factor*img.height)
            rescaled_width = int(rescaling_factor*img.width)

            img = img.resize((rescaled_width,rescaled_height),resample=PIL.Image.BICUBIC)
            mask = mask.resize((rescaled_width,rescaled_height),resample=PIL.Image.NEAREST)

        h = img.height
        w = img.width
        img = preprocess(pad_whole_image(img,patch_size,context_size))
        mask = mask_preprocess(pad_whole_image(mask,patch_size,context_size), 1.0)
        nr_of_patches = get_nr_of_patches_for_images(img,patch_size,context_size)

        #Dumping all the patch_names into a list so we can get them at runtime. That way we don't need to precut all the patches
        for i in range(nr_of_patches):
            sample_names.append(name+'_{:05d}'.format(i))

        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8),
            IMAGE_MASK: mask.astype(np.uint8),
            OG_SHAPE : (h,w)
        }
        meta_data[name] = entry

    return sample_names, meta_data

def fetch_whole_set(parent_dir,split,patch_size=224,padding_mode="symmetric",automatic_resizing=False,bounding_boxes=False,
                    legacy=False):

    all_files = get_split(parent_dir, split, legacy=legacy)
    all_files.sort()

    if split == "val":
        split = "train"

    sample_names = []
    meta_data = dict()

    split_zones_dir = os.path.join(parent_dir,ZONES_DIR,split)
    split_sar_dir = os.path.join(parent_dir,SAR_IMAGES_DIR,split)

    all_names_to_print = []
    for file_name in tqdm(all_files):
        name = file_name[:-4]
        all_names_to_print.append(file_name)

        img = Image.open(os.path.join(split_sar_dir, file_name)).convert('L')
        mask = Image.open(os.path.join(split_zones_dir, name + "_zones" + ".png")).convert('L')
        if bounding_boxes:
            img,mask = adjust_for_bounding_boxes(img,mask,parent_dir,name)

        if automatic_resizing:
            resolution_factor = int(name.split('_')[-3])
            rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
            rescaled_height = int(rescaling_factor * img.height)
            rescaled_width = int(rescaling_factor * img.width)

            img = img.resize((rescaled_width, rescaled_height), resample=PIL.Image.BICUBIC)
            mask = mask.resize((rescaled_width, rescaled_height), resample=PIL.Image.NEAREST)

        h = img.height
        w = img.width
        img = preprocess(pad_whole_image_old(img, patch_size))
        mask = mask_preprocess(pad_whole_image_old(mask, patch_size), 1.0)

        entry = {
            IMAGE:  np.round(img * 255).astype(np.uint8),
            IMAGE_MASK: mask.astype(np.uint8),
            OG_SHAPE : (h,w)
        }
        meta_data[name] = entry
        sample_names.append(name)

    print(all_names_to_print)
    return sample_names,meta_data



def adjust_for_bounding_boxes(img,mask,parent_dir,name):
    img = np.array(img)
    mask = np.array(mask)

    with open(os.path.join(parent_dir, BOUNDING_BOX_DIR, name + "_front_extent_coord.txt")) as f:
        coord_file_lines = f.readlines()
    left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
    left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
    right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in
                                                  coord_file_lines[3].split(",")]
    right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in
                                                  coord_file_lines[4].split(",")]

    if left_upper_corner_x < 0:
        left_upper_corner_x = 0
    if left_lower_corner_x < 0:
        left_lower_corner_x = 0
    if right_upper_corner_x > img.shape[1]:
        right_upper_corner_x = img.shape[1] - 1
    if right_lower_corner_x > img.shape[1]:
        right_lower_corner_x = img.shape[1] - 1
    if left_upper_corner_y > img.shape[0]:
        left_upper_corner_y = img.shape[0] - 1
    if left_lower_corner_y < 0:
        left_lower_corner_y = 0
    if right_upper_corner_y > img.shape[0]:
        right_upper_corner_y = img.shape[0] - 1
    if right_lower_corner_y < 0:
        right_lower_corner_y = 0


    # Make sure the Bounding Box coordinates are within the image
    img = Image.fromarray(img[right_lower_corner_y:left_upper_corner_y,left_upper_corner_x:right_lower_corner_x])
    mask = Image.fromarray(mask[right_lower_corner_y:left_upper_corner_y,left_upper_corner_x:right_lower_corner_x])

    return img,mask


def fetch_dataset(dir_img_target, dir_img_context, dir_mask_target, dir_mask_context,scale=1,masks_suffix=""):

    sample_names = []
    meta_data = dict()

    os.path.join(dir_img_target)
    for name in tqdm(os.listdir(dir_img_target)):
        img = preprocess(Image.open(os.path.join(dir_img_target,name)).convert('L'))
        context = preprocess(Image.open(os.path.join(dir_img_context,name)).convert('L'))
        mask = mask_preprocess(Image.open(os.path.join(dir_mask_target,name[:-4]+masks_suffix+".png")).convert('L'),scale)
        context_mask = mask_preprocess(Image.open(os.path.join(dir_mask_context,name[:-4]+masks_suffix+".png")).convert('L'),scale)
        sample_names.append(name)
        entry = {
            IMAGE : np.round(img*255).astype(np.uint8),
            CONTEXT : np.round(context*255).astype(np.uint8),
            IMAGE_MASK : mask.astype(np.uint8),
            CONTEXT_MASK : context_mask.astype(np.uint8),
        }
        meta_data[name] = entry

    return sample_names, meta_data


def preprocess(img_nd):
    if isinstance(img_nd,PIL.Image.Image):
        img_nd = np.array(img_nd)

    #if len(img_nd.shape) == 2:
    #    img_nd = np.expand_dims(img_nd, axis=2)

    img_trans = np.expand_dims(img_nd, axis=0)
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans

def mask_preprocess(pil_img, scale):

    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    img_trans = img_trans[0, :, :]

    C, D = img_trans.shape
    mask = np.ones([C, D]) * 15
    stone = np.where(img_trans == 0)
    na_area = np.where(img_trans == 63)
    na_areas = np.where(img_trans == 64)
    glacier = np.where(img_trans == 127)
    ocean_ice = np.where(img_trans == 254)
    mask[stone] = STONE_ID
    mask[na_area] = NA_AREA_ID
    mask[na_areas] = NA_AREA_ID
    mask[glacier] = GLACIER_ID
    mask[ocean_ice] = OCEAN_ICE_ID
    return mask

