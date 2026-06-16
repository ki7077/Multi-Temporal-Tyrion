import PIL
import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import JaccardIndex
from torchvision.transforms import *

from CaFFe.constants import *
from model.utils import *

"""
    this get's called by Evaluation

"""
from scipy.ndimage.filters import gaussian_filter

"""
    code from https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them/blob/master/data_processing/data_postprocessing.py#L7
"""
def get_gaussian(patch_size,num_classes=4, sigma_scale=1. / 8):
    """
    Returns Gaussian map with size of patch and sig
    :param patch_size: The size of the image patches -> gaussian importance map will have the same size
    :param sigma_scale: A scaling factor
    :return: Gaussian importance map
    """
    if isinstance(patch_size,int):
        patch_size = (patch_size,patch_size)

    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    return gaussian_importance_map.repeat((num_classes,)+(1,)*len(gaussian_importance_map.shape))

def evaluate_model(model,DL,ground_truth,whole_save,ground_truth_list):

    whole_dict = dict()
    #TODO can Tensorboardlogger be delteted here and replaced with smth?
    logger = TensorBoardLogger(save_dir="../TMP_RESULTS", name="TMP_NAME")
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger)
    trainer.test(model, dataloaders=DL)

    #Not the prettiest way to get the values out of the model but a possible way :)
    predicted_names = model.suffix_to_names
    predicted_patches = model.test_results

    save_probabilities = model.save_probability
    if save_probabilities:
        predicted_probabilities = model.prob_test_results
        probabilities_path = os.path.join(os.path.dirname(whole_save),"prob")
        os.makedirs(probabilities_path,exist_ok=True)

    IOU = JaccardIndex(task='multiclass', num_classes=model.num_classes, average='none')
    iou_ratio = 0.0
    success_ratio = 0.0
    patch_size = model.patch_size
    context_size = 512
    overlapping_patches = 0 #TODO model.overlapping_patches. Not implemented here

    for large_gt_name in ground_truth_list:
        gt = Image.open(os.path.join(ground_truth,large_gt_name)).convert('L')
        original_W, original_H = gt.size
        gt = whole_preprocess(gt)
        suffix = large_gt_name.split('.')[0][0:-6]

        if model.automatic_resizing:
            resolution_factor = int(suffix.split('_')[-3])
            rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
        else:
            rescaling_factor = 1


        W = int(original_W * rescaling_factor)
        H = int(original_H * rescaling_factor)

      #  HH = H // patch_size + 1
     #   WW = W // patch_size + 1
        #extra_patches = max(int(context_size/patch_size-1),0)

        HH = (int(H/patch_size)+1) #- extra_patches
        WW = (int(W/patch_size)+1) #- extra_patches

        length = patch_size
        # HH_overlap = H // overlap_size +1
        # WW_overlap = W //overlap_size +1
        all_names = predicted_names[suffix]
        all_names.sort()
        all_patches = []
        all_prob = []

        for i in range(len(all_names)):
            all_patches.append(predicted_patches[all_names[i]])
            if save_probabilities:
                all_prob.append(predicted_probabilities[all_names[i]])

        whole = Image.new('L', (WW * length, HH * length))

        whole_prob = torch.zeros((model.num_classes, HH * length,WW * length),dtype=torch.float16)#torch.float8_e4m3fn)

        for k in range(len(all_patches)):
            whole.paste(back(all_patches[k]),
                        (length * (k % WW), length * (k // WW), length * (k % WW + 1), length * (k // WW + 1)))
            if save_probabilities:
                #TODO for future might want to add gaussian overlap magic
                #TODO Gaussian mask
                #TODO adjust for overlap_patches
                whole_prob[:,length * (k // WW):length * (k // WW + 1),(length * (k % WW)):length * (k % WW + 1)] = (whole_prob[:,length * (k // WW):length * (k // WW + 1),(length * (k % WW)):length * (k % WW + 1)].type(torch.float16) + all_prob[k].type(torch.float16))#.type(torch.float8_e4m3fn)

                #whole_prob[:,overlap_size * (k // WW_overlap):overlap_size * (k // WW_overlap)+length,
                #(overlap_size * (k % WW_overlap)):overlap_size* (k % WW_overlap)+length, ] += get_gaussian(patch_size,model.num_classes)*all_prob[k]

        whole = CenterCrop((H, W))(whole)
        if model.automatic_resizing:
            whole = whole.resize((original_W,original_H),resample=PIL.Image.NEAREST)

        whole.save(os.path.join(whole_save, suffix + '.png'))
        if save_probabilities:
            whole_prob = CenterCrop((H, W))(whole_prob)
            torch.save(whole_prob,os.path.join(probabilities_path, suffix ))

        #TODO check whetherthis works or break
        whole_dict[suffix] = whole_preprocess(whole).astype(np.uint8)
        h, w = whole.size#shape

        whole = whole_preprocess(whole)
        success_ratio+=np.sum(np.where(whole==gt,1,0))/(h*W)

        iou = IOU(torch.from_numpy(whole).type(torch.int64), torch.from_numpy(gt).type(torch.int64))
        iou_ratio += iou

    iou_whole = iou_ratio / len(ground_truth_list)

    print("################# My output#############")
    print("STONE_ID = 0, NA_AREA_ID = 1, GLACIER_ID = 2, OCEAN_ICE_ID = 3")
    print("IoU whole: ",iou_whole)
    print("IoU avg",torch.mean(iou_whole))
    print("##########################################")
    iou_ratio = iou_ratio / len(ground_truth_list)
    ave_iou = sum(iou_ratio) / len(iou_ratio)
    success_ratio = success_ratio / len(ground_truth_list)


    return success_ratio,ave_iou,iou_ratio, whole_dict




