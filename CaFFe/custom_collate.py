import torch

from CaFFe.constants import *


def custom_collate(batch):
    names = []
    imgs = []
    img_labels = []
    contexts = []
    context_labels = []
    idx_references = []

    counter = 0

    for item in batch:

        if item.get(IDX_REFERENCE) is not None:
            idx_references.append(item[IDX_REFERENCE]+counter*len(item["name"]))

        if isinstance(item["name"],list):
            for i in range(len(item["name"])):
                names.append(item["name"][i])
                imgs.append(item[IMAGE][i])
                img_labels.append(item[IMAGE_MASK][i])
                contexts.append(item[CONTEXT][i])
                context_labels.append(item[CONTEXT_MASK][i])
        else:
            names.append(item["name"])
            imgs.append(item[IMAGE])
            img_labels.append(item[IMAGE_MASK])
            contexts.append(item[CONTEXT])
            context_labels.append(item[CONTEXT_MASK])

        counter+=1

    imgs_t = None
    if len(imgs) > 0:
        imgs_t = torch.stack(imgs)

    contexts_t = None
    if len(contexts) > 0:
        contexts_t = torch.stack(contexts)

    img_labels_t = None
    if len(img_labels) > 0:
        img_labels_t = torch.stack(img_labels)

    context_labels_t = None
    if len(context_labels) > 0:
        context_labels_t = torch.stack(context_labels)



    out_dict = {
            "name": names,
            IMAGE: imgs_t/255.0,
            CONTEXT: contexts_t/255.0,
            IMAGE_MASK: img_labels_t,
            CONTEXT_MASK : context_labels_t,
            IDX_REFERENCE: idx_references

    }

    return out_dict


