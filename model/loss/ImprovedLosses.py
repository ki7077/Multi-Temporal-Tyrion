import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CaFFe.constants import *


class OGDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(OGDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class ClassWiseSmoothCE(nn.Module):

    def __init__(self, eps=0.0, reduction='mean',weights=None,mode="normal"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.weights = weights if weights is None else torch.nn.functional.normalize(torch.from_numpy(np.array(weights)),dim=0)
        self.mode= mode
        if mode== "class":
            self.eps =0.0


    def forward(self, pred, gold,reduc = 'mean'):
        # casting gold if necessary
        if gold.dtype == torch.int32 or gold.dtype ==torch.int8 or gold.dtype == torch.uint8:
            gold = gold.type(torch.int64)

        elif gold.dtype == torch.float32:
            gold = torch.round(gold).type(torch.int64)


        if self.eps >= 0.:
            pred = pred.permute(0, 2, 3, 1).flatten(end_dim=2)
            gold = gold.flatten().type(torch.int64)

            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred,device='cuda')

            one_hot = one_hot.scatter(1, gold.view(-1, 1), 1)
            one_hot_eps = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            if self.mode == "class":
                loss = -(one_hot_eps * log_prb).sum(dim=1)
                loss = torch.matmul(loss,one_hot_eps)/(one_hot_eps.sum(dim=0)+0.00000001)
                # loss = loss/(one_hot_eps.sum(dim=0) + 0.00001)
                return loss.sum()/torch.where(one_hot_eps.sum(dim=0)!=0,1,0).sum()

            loss = -(one_hot_eps * log_prb)#.sum(dim=1)
            if self.weights is not None:
                self.weights = self.weights.to(loss.device)
                loss = self.weights*loss
            loss = loss.sum(dim=1)

            if reduc == 'nothing':
                return loss

            loss = torch.mean(loss)

            return loss

        return None



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


