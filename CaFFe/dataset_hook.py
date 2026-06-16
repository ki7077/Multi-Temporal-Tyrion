import datetime

import omegaconf
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import resize

from CaFFe.augmentations.OtherTransforms import NoTransform2, NoTransform
from CaFFe.fetcher import *
from model.utils import *

"""
    Abstract Dataset to avoid redundancy
    args: 
        mask_suffix : suffix of the mask
        scale: I also want to know what that is
        prob_mix_up: probability for the mix_up augmentation
        prob_erasure: probability for the erasure augmentation
        prob_cut_mix: probability of the cut mix augmentation
        erasure_repeats: how often erasure is repeated
        augmentation: The augmentations like (contrast, brightness etc). Applies to image and context
        double_augmentation: Second set of augmentations that also applies to masks (like rotation etc)
        context_without_resize : If set the context is not resized to the same resolution as the image. This is kinda experimental. And has some bugs... WIP
        prob_rezoom : rescaling the resolution of the image as augmentation
        style_dic_type: This decides how the time series is created. Are we taking all images from the same glacier, or only the ones with the same resolution etc.
        unique_time : So every time step occurs only once for efficient processing. (Data might still get duplicated to fill up the last time series)
        cut_mix_bounds : maximum size for the cutmix crop
        T : Length of the timeseries. If no time series is used T=1
    
"""
class AbstractDataset(Dataset):
    def __init__(self,mask_suffix='_zones_NA',scale=1,
                 prob_mix_up=0.0,
                 prob_erasure =0.0,prob_cut_mix=0.0,erasure_repeats = 2,
                 augmentation=NoTransform(),double_augmentation=NoTransform2(),
                 context_without_resize=False,prob_rezoom=0.0,
                 style_dic_type="standard",unique_time=False,cut_mix_bounds=[192,192],T=1,
                 ):

        super(AbstractDataset, self).__init__()
        self.T = T
        self.scale = scale
        self.context_without_resize = context_without_resize
        self.prob_mix_up = prob_mix_up
        self.prob_rezoom = prob_rezoom
        self.prob_erasure = prob_erasure
        self.erasure_repeats = erasure_repeats
        self.prob_cut_mix = prob_cut_mix
        self.unique_time = unique_time

        self.cut_mix_bounds = cut_mix_bounds

        if isinstance(augmentation, omegaconf.dictconfig.DictConfig):
            augmentation = instantiate_from_config(augmentation)

        if isinstance(double_augmentation, omegaconf.dictconfig.DictConfig):
            double_augmentation = instantiate_from_config(double_augmentation)

        self.double_augmentation = double_augmentation
        self.augmentation = augmentation
        self.mask_suffix = mask_suffix
        self.style_dic_type = style_dic_type
        #self.style_dic = None

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, i):
        return NotImplementedError

    # Name of the Glacier, #1 Year, #2 month, #3 Dayy, #4 Satellite, #5 Zoom, #6 Orbit, #7Quality Factor #8patch
    def get_style(self,name,ignore_unique=False):
        tokens = name.split('_')

        if self.unique_time and (not ignore_unique):
            if len(tokens)>=7:
                return ('_').join(name.split('_')[:-1])
            return name

        #if len(tokens) >= 7:
        #    return ('_').join((tokens[0], tokens[1].split('-')[0], tokens[2], tokens[3],tokens[6]))
        if self.style_dic_type == "standard":
            return ('_').join((tokens[0],tokens[2],tokens[3]))
        elif self.style_dic_type == "satellite":
            return ('_').join((tokens[0],tokens[2]))
        elif self.style_dic_type == "reso":
            return ('_').join((tokens[0],tokens[3]))

        else:
            return tokens[0]

    def pad_whole_image(self,img, target_size):
        W = img.shape[2]
        H = img.shape[1]

        WW = (W // target_size) + 2
        HH = (H // target_size) + 2
        # If the image is smaller than the size given, then CenterCrop pads the image accordingly
        # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
        crop_height, crop_width = (HH * target_size, WW * target_size)
        padding_ltrb = [
            int(round((crop_width - W) / 2.0)) if crop_width > W else 0,
            int(round((crop_height - H) / 2.0)) if crop_height > H else 0,
            int(round((crop_width - W + 1) / 2.0)) if crop_width > W else 0,
            int(round((crop_height - H + 1) / 2.0)) if crop_height > H else 0,
        ]
        pil_img = transforms.Pad(padding_ltrb, fill=0, padding_mode="symmetric")(img)

        return pil_img

    def align_image_masks(self,image_list,mask_list,og_shape_list,pad_size,center_idx):
        og_h,og_w = og_shape_list[center_idx]
        resize_operator = transforms.Resize((og_h,og_w),interpolation=PIL.Image.NEAREST)

        for i in range(0,len((image_list))):#,mask_list):
            if og_shape_list[i][0] == og_h and og_shape_list[i][1] == og_w:
                continue

            centering = CenterCrop((og_shape_list[i][0], og_shape_list[i][1]))
            image_list[i] = np.array(self.pad_whole_image(resize_operator(centering(torch.from_numpy(image_list[i]))),pad_size))
            mask_list[i] = np.array(self.pad_whole_image(resize_operator(centering(torch.from_numpy(mask_list[i]).unsqueeze(dim=0))),pad_size)[0])

        return image_list,mask_list

    def get_style_patch(self,name,rng=True,repeat=1):
        possible_styles = self.style_dic[self.get_style(name)]
        if rng:
            idx = np.random.randint(0,len(possible_styles))
        else:
            timestamps = []
            for style_name in possible_styles:
                timestamps.append(style_name.split('_')[1])
            idx = self.get_ith_closest_date(timestamps,name.split('_')[1],repeat)

        return possible_styles[idx]

    def get_ith_closest_date(self,timestamps,current_date,i):

        current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d")
        dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps]

        deltas = []
        for d in dates:
            deltas.append(np.abs(current_date-d))

        if i+1 >= len(deltas):
            return np.argsort(deltas)[-1]

        return np.argsort(deltas)[1:][i]

    def mk_style_dic(self,names):
        style_dict = dict()

        for name in names:
            if len(name.split('_')) ==7:
                if name.split('_')[-1] != '00000':
                    continue
            #Get style key
            key = self.get_style(name)

            #Checking whether we need to make a new list for that type of styling
            if style_dict.get(key) is None:
                style_dict[key] = []
            style_dict[key].append(name)

        return style_dict

    def extract_layers(self,image, mask):
        ocean_ice = np.where(mask == OCEAN_ICE_ID, image, 0)
        glacier_ice = np.where(mask == GLACIER_ID, image, 0)
        rest = np.where(mask < GLACIER_ID, image, 0)

        return ocean_ice, glacier_ice, rest

    def combine_layers(self,oecan_ice,ice,rest,mask):
        rest = np.where(mask==OCEAN_ICE_ID,oecan_ice,rest)
        rest = np.where(mask==GLACIER_ID,ice,rest)

        return rest

    def mix_up_ocean_ice(self,img1, img2, mask, w_1, w_2):
        img1_np = img1.astype(np.float32)
        img2_np = img2.astype(np.float32)

        mask = np.expand_dims(mask, axis=0)

        #Case of a RGB image
        if img1.shape[0] == 3:
            mask = np.concatenate((mask,mask,mask))

        img = np.where(mask == OCEAN_ICE_ID, (w_1 * img1_np + w_2 * img2_np) / (w_1 + w_2), img1_np)
        img = np.clip(img, a_min=0, a_max=255)

        return img.astype(np.uint8)


    #TODO maybe we need a leading 0 dimension
    def random_erasure(self,img,patch_size):

        height_box = max(1, np.int32(patch_size / 2 * torch.rand(1).item()))
        width_box = max(1, np.int32(patch_size / 2 * torch.rand(1).item()))

        start_h = np.int32((img.shape[1]-height_box)*torch.rand(1).item())
        start_w = np.int32((img.shape[2]-width_box)*torch.rand(1).item())

        img[:,start_h:start_h+height_box,start_w:start_w+width_box] = np.random.randint(size=(height_box,width_box),low=0,high=255).astype(np.uint8)

        return img

    def cut_mix(self,img_list,mask_list,cut_img,cut_mask):
        height_box = np.int32(self.cut_mix_bounds[0] * torch.rand(1).item())+128
        width_box = np.int32(self.cut_mix_bounds[1] * torch.rand(1).item())+128

        start_h = np.int32((cut_img.shape[1] - height_box) * torch.rand(1).item())
        start_w = np.int32((cut_img.shape[2] - width_box) * torch.rand(1).item())

        cut_out = cut_img[:,start_h:start_h+height_box,start_w:start_w+width_box]
        cut_out_mask = cut_mask[start_h:start_h+height_box,start_w:start_w+width_box]

        start_h = np.int32((img_list[0].shape[1] - height_box) * torch.rand(1).item())
        start_w = np.int32((img_list[0].shape[2] - width_box) * torch.rand(1).item())

        for i in range(len(img_list)):
            img_list[i][:,start_h:start_h+height_box,start_w:start_w+width_box] = cut_out
            mask_list[i][start_h:start_h+height_box,start_w:start_w+width_box] = cut_out_mask

        return img_list,mask_list

    def cut_mix_time_series(self,img_list,mask_list,cut_img_list,cut_mask_list,patch_size):
        height_box = min(self.cut_mix_bounds[0],np.int32((patch_size/2) * torch.rand(1).item()+patch_size/2))
        width_box = min(self.cut_mix_bounds[1],np.int32((patch_size/2) * torch.rand(1).item()++patch_size/2))

        cut_start_h = np.int32((cut_img_list[0].shape[1] - height_box) * torch.rand(1).item())
        cut_start_w = np.int32((cut_img_list[0].shape[2] - width_box) * torch.rand(1).item())

        img_start_h = np.int32((img_list[0].shape[1] - height_box) * torch.rand(1).item())
        img_start_w = np.int32((img_list[0].shape[2] - width_box) * torch.rand(1).item())

        start = 0
        end = len(mask_list)

        for i in range(start,end):

            cut_out = cut_img_list[i][:,cut_start_h:cut_start_h+height_box,cut_start_w:cut_start_w+width_box]
            cut_out_mask = cut_mask_list[i][cut_start_h:cut_start_h+height_box,cut_start_w:cut_start_w+width_box]

            img_list[i][:,img_start_h:img_start_h+height_box,img_start_w:img_start_w+width_box] = cut_out
            mask_list[i][img_start_h:img_start_h+height_box,img_start_w:img_start_w+width_box] = cut_out_mask

        return img_list,mask_list

    def apply_augmentations(self, image,mask):

        image = self.augmentation(image)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.int64)
        image, mask = self.double_augmentation(image, mask)
        return image,mask
