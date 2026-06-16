from CaFFe.dataset_hook import *


"""
    patch_size : size of the patches
    Context_factor: What scale the context is larger than the patch size
    parent_dir: parent_dir for data fetching
    split : train/validation/test split
    padding_mode : how the data should be padded
    nr_of_sample_per_epoch: how many samples per epoch should be sampled
    automatic_resizing: should every image be resized to the same resoltion?
"""

class RandomMultiplePatchDataset(AbstractDataset):
    def __init__(self,patch_size,context_factor, parent_dir,split,padding_mode="symmetric",
                 nr_of_samples_per_epoch=59463,automatic_resizing=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.nr_of_samples_per_epoch = nr_of_samples_per_epoch
        self.padding_mode = padding_mode
        self.names, self.meta_data = fetch_whole_set(parent_dir,split,int(context_factor*patch_size),padding_mode,
                                                     automatic_resizing=automatic_resizing)
        self.patch_size = patch_size
        self.context_factor = context_factor
        self.center = torchvision.transforms.CenterCrop((self.patch_size,self.patch_size))
        self.style_dic = self.mk_style_dic(self.names)


    def __len__(self):
        return self.nr_of_samples_per_epoch

    def get_random_multi_patch(self,image_list,mask_list,patch_size):
        #images are padded by patch_size in h and width. That's why we can only take image.shape-1 patch_size. also not to confuse with the other patch_size
        random_zoom_factor = 1.0
        if self.prob_rezoom > torch.rand(1):
            random_zoom_factor = torch.rand(1)*2.5 + 0.33

        # all images should have same size so taking the first ones should be fine
        random_x = torch.randint(0,max(1, image_list[0].shape[1] - int(random_zoom_factor * patch_size)), (1,)).item()
        random_y = torch.randint(0, max(1,image_list[0].shape[2] - int(patch_size * random_zoom_factor)), (1,)).item()

        output_images = []
        output_masks = []

        for image,mask in zip(image_list,mask_list):

            img = image[:, random_x:min(image.shape[1], random_x + int(random_zoom_factor * patch_size)),
                  random_y:min(image.shape[2], random_y + int(patch_size * random_zoom_factor))]
            mask = mask[random_x:min(image.shape[1], random_x + int(random_zoom_factor * patch_size)),
                   random_y:min(image.shape[2], random_y + int(patch_size * random_zoom_factor))]

           # img = image[:, random_x:random_x+int(random_zoom_factor*patch_size), random_y:random_y+int(patch_size*random_zoom_factor)]
           # mask = mask[ random_x:random_x+int(random_zoom_factor*patch_size), random_y:random_y+int(patch_size*random_zoom_factor)]
            #TODO resizing
            if self.prob_rezoom > 0.0:
                img = np.array(resize(torch.from_numpy(img).unsqueeze(dim=0), (patch_size, patch_size)))[0]
                mask = np.array(resize(torch.from_numpy(mask).unsqueeze(dim=0).unsqueeze(dim=0), (patch_size, patch_size))[0,0])

            output_images.append(img)
            output_masks.append(mask)

        return output_images,output_masks

    def get_random_patch(self,image,mask,patch_size):
        #images are padded by patch_size in h and width. That's why we can only take image.shape-1 patch_size. also not to confuse with the other patch_size
        random_zoom_factor = 1.0
        if self.prob_rezoom > torch.rand(1):
            random_zoom_factor = torch.rand(1)*2.5 + 0.33

        random_x = torch.randint(0, max(1, image.shape[1] - int(random_zoom_factor * patch_size)), (1,)).item()
        random_y = torch.randint(0, max(1, image.shape[2] - int(patch_size * random_zoom_factor)), (1,)).item()

        img = image[:, random_x:min(image.shape[1], random_x + int(random_zoom_factor * patch_size)),
              random_y:min(image.shape[2], random_y + int(patch_size * random_zoom_factor))]
        mask = mask[random_x:min(image.shape[1], random_x + int(random_zoom_factor * patch_size)),
               random_y:min(image.shape[2], random_y + int(patch_size * random_zoom_factor))]

        if self.prob_rezoom > 0.0:
            img = np.array(resize(torch.from_numpy(img), (patch_size, patch_size)))
            mask = np.array(resize(torch.from_numpy(mask).unsqueeze(dim=0), (patch_size, patch_size))[0])

        return img,mask

    def get_multi_image_from_context(self,context_list,context_mask_list):
        images = []
        masks = []

        for context_img,context_mask in zip(context_list,context_mask_list):
            images.append(np.array(self.center(context_img)))
            masks.append(np.array(self.center(context_mask)))
        return images, masks

    def apply_augmentations(self, image_list, mask_list):

        image_list_new = []
        for img in image_list:
            image_list_new.append(self.augmentation(img))
        image_list = image_list_new
        new_image_list = []
        new_mask_list = []

        for image,mask in zip(image_list,mask_list):
            new_image_list.append(torch.from_numpy(image).type(torch.FloatTensor))
            new_mask_list.append(torch.from_numpy(mask).type(torch.int64))

        image_list= new_image_list
        mask_list = new_mask_list

        image_list, mask_list = self.double_augmentation(image_list, mask_list)
        return image_list, mask_list

    def get_style_patch(self,name,already_taken_names=None):
        possible_styles = self.style_dic[self.get_style(name)]

        #Should not throw an error because first part gets evaluated first
        if already_taken_names is not None and len(possible_styles)> len(already_taken_names):
            new_styles = []
            for potential_style in possible_styles:
                if not (potential_style in already_taken_names):
                    new_styles.append(potential_style)

            possible_styles = new_styles

        idx = np.random.randint(0,len(possible_styles))

        return possible_styles[idx]



    def __getitem__(self, i):
        name = self.names[i%len(self.names)]

        names = [name]
        for j in range(self.T-1):
            names.append(self.get_style_patch(name,names))

        names.sort()
        whole_image_list = []
        whole_mask_list = []
        whole_shape_list = []

        for n in names:
            whole_image_list.append(self.meta_data[n][IMAGE])
            whole_mask_list.append(self.meta_data[n][IMAGE_MASK])
            whole_shape_list.append(self.meta_data[n][OG_SHAPE])

        if self.style_dic_type != "standard":
            whole_image_list,whole_mask_list = self.align_image_masks(whole_image_list,whole_mask_list,whole_shape_list,int(self.context_factor*self.patch_size),np.argwhere(np.array(names) == name)[0, 0])
        context_list,mask_context_list = self.get_random_multi_patch(whole_image_list,whole_mask_list,self.patch_size*self.context_factor)

        if self.prob_mix_up> torch.rand(1):
            mix_up_factor = torch.rand(1)*0.15+0.15
            data_mixup = self.meta_data[self.names[int(len(self.names)*torch.rand(1))]]
            mix_up_img = data_mixup[IMAGE]
            mix_up_mask = data_mixup[IMAGE_MASK]
            mix_up_context, _ = self.get_random_patch(mix_up_img,mix_up_mask,self.patch_size*self.context_factor) #self.get_all(mix_up_img, mix_up_mask)

            new_context_list = []
            for context,mask_context in zip(context_list,mask_context_list):
                context = self.mix_up_ocean_ice(context,mix_up_context,mask_context,1.0,mix_up_factor.item())
                new_context_list.append(context)
            context_list = new_context_list

        if self.prob_cut_mix > torch.rand(1):
            cut_mix_names = [self.names[int(len(self.names)*torch.rand(1))]]
            for i in range(self.T-1):
                cut_mix_names.append(self.get_style_patch(cut_mix_names[0]))

            cut_mix_names.sort()
            cut_mix_sar = []
            cut_mix_masks = []
            cut_mix_shapes = []

            for n in cut_mix_names:
                cut_mix_masks.append(self.meta_data[n][IMAGE_MASK])
                cut_mix_sar.append(self.meta_data[n][IMAGE])
                cut_mix_shapes.append(self.meta_data[n][OG_SHAPE])

            if self.style_dic_type != "standard":
                cut_mix_sar, cut_mix_masks = self.align_image_masks(cut_mix_sar, cut_mix_masks,
                                                                           cut_mix_shapes,
                                                                           int(self.context_factor * self.patch_size),0)

            context_list,mask_context_list = self.cut_mix_time_series(context_list,mask_context_list,cut_mix_sar,cut_mix_masks,patch_size=self.patch_size)

        for i in range(len(context_list)):
            for j in range(self.erasure_repeats):
                if self.prob_erasure > torch.rand(1):
                    context_list[i] = self.random_erasure(context_list[i],self.patch_size)


        context_list,mask_context_list = self.apply_augmentations(context_list,mask_context_list)

        image_list, mask_image_list = self.get_multi_image_from_context(context_list,mask_context_list)
        for i in range(len(image_list)):
            image_list[i] = torch.from_numpy(image_list[i])
            mask_image_list[i] = torch.from_numpy(mask_image_list[i])

            if not self.context_without_resize:
                context_list[i] = resize(context_list[i],torch.from_numpy(image_list[0][0]).shape)
                mask_context_list[i] = resize(mask_context_list[i],torch.from_numpy(mask_image_list[0]).shape)

        return {
            "name": names,
            IMAGE : image_list,
            IMAGE_MASK: mask_image_list,
            CONTEXT: context_list,
            CONTEXT_MASK: mask_context_list,
            IDX_REFERENCE: np.argwhere(np.array(names) == name)[0, 0]
        }




"""
    Organized Patch Dataset. This Dataset cuts the entire image into symmetrically padded images and then cuts them into patches on the fly.
    This avoids huge memory consumption and loading times to start. Also you can cut any patch_size without waiting.
    args: 
         look at RandomPatchDataset
         split: train/val/test
"""

class MultiPatchDataSet(AbstractDataset):
    def __init__(self,patch_size,context_factor, parent_dir,split,automatic_resizing=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.names, self.meta_data = fetch_patches(parent_dir,split,patch_size,int(context_factor*patch_size),
                                                     automatic_resizing=automatic_resizing)
        self.patch_size = patch_size
        self.context_factor = context_factor

        if self.unique_time:
            self.names, self.style_dic = self.create_strict_style_dic(self.names)
        else:
            self.style_dic = self.mk_style_dic(self.names)

        print(len(self.names),"nr_of samples ")

    def __len__(self):
        return len(self.names)

    def check_style(self,name,style_key):
        tokens = name.split('_')

        if self.style_dic_type == "standard":
            return ('_').join((tokens[0], tokens[2], tokens[3])) == style_key
        elif self.style_dic_type == "satellite":
            return ('_').join((tokens[0], tokens[2])) == style_key
        elif self.style_dic_type == "reso":
            return ('_').join((tokens[0], tokens[3])) == style_key
        else:
            return tokens[0] == style_key

    """
        Returns a list of names and an associated style dic.
    """
    def create_strict_style_dic(self,names):
        style_dict = dict()
        names.sort()

        for name in names:
            if len(name.split('_')) == 7:
                if name.split('_')[-1] != '00000':
                    continue
            # Get style key
            key = self.get_style(name,ignore_unique=True)

            # Checking whether we need to make a new list for that type of styling
            if style_dict.get(key) is None:
                style_dict[key] = []
            style_dict[key].append(name)

        # okay this is gonna be a bit complicated here
        # Essentially we go through every style and split them into groups for the timeseries.
        # Then we make one of each group the representative of each group and take the rest out of the names set
        # That way we will end up with the correct amount of samples
        # We will always choose the last samples as representative to make processing easier
        groups = []

        for style_key in style_dict:

            style_representatives = style_dict[style_key]
            # pair the samples based on time series. If it's not a clean cut we will fix in the next part
            for i in range(0, len(style_representatives), self.T):
                print(" ",style_key," i ",i, " bis ",i+self.T-1)
                if len(style_representatives) >= i+self.T:
                    groups.append(style_representatives[i:i + self.T])

            # in case of an uneven split
            if len(style_representatives) % (self.T) != 0:
                #awkward case of having too little style representatives overall
                if len(style_representatives) < self.T:
                    print("Difficult Case might need extra testing")
                    # we need to add more by duplicating the existing ones
                    start = len(style_representatives)
                    last_sample = style_representatives[-1]
                    for i in range(start,self.T-1):
                        style_representatives.append(style_representatives[i%start])
                    style_representatives.append(last_sample)
                    style_representatives.sort()
                #TODO does this properly sample backwards?
                groups.append(style_representatives[-(self.T):])

        for small_group in groups:
            new_list = []
            for z in small_group:
                new_list.append(('_').join(z.split('_')[:-1]))
            print("Style_group: ",new_list)

        representatives_names = []
        new_style_dic = dict()
        for group in groups:
            name_group = group[-1]
            if len(name_group.split('_')) == 7:
                name_group = ('_').join(name_group.split('_')[:-1])#name_group[:-4]

            representatives_names.append(name_group)
            new_style_dic[name_group] = group.copy() # we copy all the samples since one will always get filtered out later anyway

        #TODO this might not work in case of certain styles.........
        new_names = []
        for name in self.names:
            for repr in representatives_names:
                if repr in name:
                    new_names.append(name)
                    break

        return new_names,new_style_dic

    def get_patch_multi(self,img, patch_size, context_size, idx):

        if len(img[0].shape) == 3:
            _,H,W = img[0].shape
        else:
            H,W = img[0].shape

        extra_add = -1

        if ((H + 1) // patch_size * patch_size + context_size) <= H and (
                ((W + 1) // patch_size * patch_size + context_size) <= W):
            extra_add = 0

       # HH = (H + 1) // patch_size + extra_add
        WW = (W+1)//patch_size + extra_add

        extra_patches = max(int(context_size/patch_size-1),0)
        WW = W // patch_size - extra_patches
        # defines the top left corner of the patch
        row = idx // WW
        column = idx % WW

        context_crop_list = []

        # Go through all the images and do the thing
        for i in range(len(img)):

            if len(img[i].shape) == 3:
                context_crop = img[i][:, row * patch_size:(row) * patch_size + context_size,
                           column * patch_size:(column) * patch_size + context_size]

            else:
                context_crop = img[i][row * patch_size:(row) * patch_size + context_size,
                               column * patch_size:(column) * patch_size + context_size]

            if not self.context_without_resize:
                if len(context_crop.shape)== 2:
                    context_crop = np.array(transforms.Resize((patch_size, patch_size),interpolation=PIL.Image.NEAREST)(torch.from_numpy(context_crop).unsqueeze(dim=0)))[0]

                else:
                    context_crop = np.array(transforms.Resize((patch_size, patch_size),interpolation=PIL.Image.NEAREST)(torch.from_numpy(context_crop)))
            else:
                context_crop = np.array((torch.from_numpy(context_crop)))

            context_crop_list.append(context_crop)

        return context_crop_list

    def get_image_from_context(self,context_list,patch_size):

        output_img_list = []

        for context in context_list:
            output_img_list.append(transforms.CenterCrop((patch_size, patch_size))(context))

        return output_img_list

    def apply_augmentations(self, image_list, mask_list):
        new_image_list =[]
        for image in image_list:
            new_image_list.append(self.augmentation(image))
        image_list=new_image_list
        new_image_list = []
        new_mask_list = []

        for image,mask in zip(image_list,mask_list):
            new_image_list.append(torch.from_numpy(image).type(torch.FloatTensor))
            new_mask_list.append(torch.from_numpy(mask).type(torch.int64))

        image_list= new_image_list
        mask_list = new_mask_list

        image_list, mask_list = self.double_augmentation(image_list, mask_list)
        return image_list, mask_list

    def __getitem__(self, item):
        name = self.names[item]

        suffix = ('_').join(name.split('_')[:-1])
        idx = int(name.split('_')[-1])

        names = [name]
        style_names = [suffix]
        for j in range(self.T-1):
            style_name = self.get_style_patch(name, rng=False, repeat=j)  # False,repeat=j)
            style_suffix = ('_').join(style_name.split('_')[:-1])
            style_names.append(style_suffix)
            names.append(style_suffix + "_" + name.split('_')[-1])

        names.sort()
        style_names.sort()

        whole_image_list = []
        whole_mask_list = []
        og_shape_list = []

        for style_suffix in style_names:
            whole_image_list.append(self.meta_data[style_suffix][IMAGE])
            whole_mask_list.append(self.meta_data[style_suffix][IMAGE_MASK])
            og_shape_list.append(self.meta_data[style_suffix][OG_SHAPE])

        if self.style_dic_type != "standard":
            whole_image_list,whole_mask_list = self.align_image_masks(whole_image_list,whole_mask_list,og_shape_list,self.patch_size,np.argwhere(np.array(names) == name)[0, 0])

        context_list = self.get_patch_multi(whole_image_list,self.patch_size,int(self.context_factor*self.patch_size),idx)
        mask_context_list = self.get_patch_multi(whole_mask_list,self.patch_size,int(self.context_factor*self.patch_size),idx)

        context_list,mask_context_list = self.apply_augmentations(context_list,mask_context_list)
        image_list = self.get_image_from_context(context_list,self.patch_size)
        mask_image_list = self.get_image_from_context(mask_context_list,self.patch_size)

        return {
            "name": names,
            IMAGE : image_list,
            IMAGE_MASK: mask_image_list,
            CONTEXT: context_list,
            CONTEXT_MASK: mask_context_list,
            IDX_REFERENCE: np.argwhere(np.array(names) == name)[0, 0]
        }
