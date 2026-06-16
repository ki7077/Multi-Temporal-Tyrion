# --------------------------------------------------------
# Swin Transformer V2+1 based on the Swin V2 Transformer
# --------------------------------------------------------

import copy

import torch.nn as nn
from torchvision.transforms import Resize

from model.modules.SwinV2Modules import *
from model.modules.TemporalAttention import MultiLTAEWrapperCaFFe
from model.modules.TemporalModules import SpatioTemporalGru,TemporalSingularDownSample, TemporalSingularUpSample, Identity

from model.utils import zero_module
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import torch.utils.checkpoint as checkpoint


class ResUpSampleBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,up_sample=True,**kwargs):
        super().__init__()
        self.block = ResnetBlock(in_channels=in_channels,out_channels=out_channels,**kwargs)
        self.up_sample = up_sample
        if self.up_sample:
            self.up = Upsample(in_channels=out_channels,with_conv=True)

    def forward(self,x):
        x = self.block(x)
        if self.up_sample:
            x = self.up(x)

        return x
class ResUpSampleBlockTemporal(torch.nn.Module):
    def __init__(self,in_channels,out_channels,up_sample=True,T_length=3,temporal=True,
                 padding_type="zeros",
                 **kwargs):
        super().__init__()
        self.block = ResnetBlock(in_channels=in_channels,out_channels=out_channels,padding_type=padding_type,**kwargs)
        self.temporal = temporal
        if temporal:
            self.temporal_block = Temporal1DResBlock(in_channels=out_channels,out_channels=out_channels,T_length=T_length,
                                                     padding_type=padding_type)
        else:
            self.temporal_block = Identity()
        self.up_sample = up_sample
        if self.up_sample:
            self.up = Upsample(in_channels=out_channels,with_conv=True)

    def forward(self,x):
        x = self.block(x)
        x = self.temporal_block(x)
        if self.up_sample:
            x = self.up(x)

        return x

class Swin2x1Model(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    #TODO Depth-unet is not fully functional yet ....
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,increased_window_factor=2,
                 use_checkpoint=False, final_upsample="expand_first",pretrained_window_sizes=[0, 0, 0, 0],
                 checkpoint_transformer=None,out_channels=256,return_pred_embedding=False,T_length=3,
                 temporal_type="conv",decoder_channels = None,temporal_down_scale=[],
                 padding_type="reflect",
                 shift_temporal=False,temporal_up_sample=True,temporal_start_encoder=0,
                 temporal_patch_embedding=False,combine_ch=1,**kwargs):

        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};drop_path_rate:{};num_classes:{}".format(
                depths, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.downsample_steps = len(depths)+1
        self.embed_dim = embed_dim
        self.decoder_channels = embed_dim if decoder_channels is None else decoder_channels
        temporal_information_used = False if temporal_type == "nothing" else True
        self.temporal_upsample = temporal_up_sample

        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.window_size = window_size
        self.increased_window_factor = increased_window_factor

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_upt = nn.ModuleList()

        if temporal_patch_embedding:
            self.patch_embed = TemporalPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,T_length=T_length,combine_ch=combine_ch)
            T_length = int(np.round(T_length/combine_ch))
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)


        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        self.temporal_downsample_steps = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)

            layer = BasicLayer(dim=dim,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               T_length=T_length,
                               temporal_type=temporal_type if i_layer >= temporal_start_encoder else "nothing",
                               padding_type=padding_type,
                               shift_temporal=shift_temporal)

            self.layers.append(layer)
            if i_layer in temporal_down_scale and i_layer+1 != self.num_layers:
                down_ch_size = min(embed_dim*2**(self.num_layers-1),int(embed_dim * 2 ** (i_layer+1)))
                self.temporal_downsample_steps.append(TemporalSingularDownSample(down_ch_size,down_ch_size,
                                                                                 T_length=T_length,padding_type=padding_type))
                T_length = int(T_length/2)
            else:
                self.temporal_downsample_steps.append(Identity())


        self.return_pred_embedding = return_pred_embedding
        self.pred_target = nn.ModuleList()
        self.pred_target.append(ResUpSampleBlockTemporal(embed_dim*2**(self.num_layers-1),self.decoder_channels*2**(self.num_layers-1),up_sample=False,T_length=T_length,
                                                         temporal=temporal_information_used,padding_type=padding_type))
        self.temporal_upsample_blocks = torch.nn.ModuleList()
        self.temporal_upsample_blocks.append(Identity())

        for i_layer in reversed(range(self.num_layers)):
            dim = 0 if i_layer+1 == self.num_layers else int(embed_dim * 2 ** i_layer)
            decoder_dim = int(self.decoder_channels* 2**i_layer)
            out_channels = int(decoder_dim/2) if (i_layer != 0) else decoder_dim

            self.pred_target.append(ResUpSampleBlockTemporal(dim+decoder_dim,out_channels,T_length=T_length,
                                            temporal= temporal_information_used if i_layer >= temporal_start_encoder else False,
                                                             padding_type=padding_type))
            if i_layer-1 in temporal_down_scale and i_layer != 0 and temporal_up_sample:
                self.temporal_upsample_blocks.append(TemporalSingularUpSample(in_channels=out_channels,out_channels=out_channels,
                                                                              with_conv=True,T_length=T_length,padding_type=padding_type))
                T_length = T_length*2
            else:
                self.temporal_upsample_blocks.append(Identity())

        # Need one extra set of upsampling because patch embeddings are usually dowonscaling it by a factor of 4
        self.pred_target.append(ResUpSampleBlockTemporal(out_channels,out_channels,T_length=T_length,
                                            temporal=temporal_information_used if i_layer >= temporal_start_encoder else False,
                                                         padding_type=padding_type))

        self.pred_target.append(torch.nn.Conv2d(in_channels=out_channels,out_channels=num_classes,kernel_size=1))

        for bly in self.layers:
            bly._init_respostnorm()

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")

            self.outputt = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)


         #TODO other layers as well
        for bly in self.layers_upt:
            if isinstance(bly, BasicLayer_up):
                bly._init_respostnorm()
        #self.cross_att._init_respostnorm()

        self.load_from_pretrained(checkpoint_transformer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_pretrained(self, ckpt_path):
        if ckpt_path is None:
            print("No checkpoint path for Decoder ")
            return

        pretrained_dict = torch.load(ckpt_path)['model']
        model_dict = self.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)

        for k in list(full_dict.keys()):
            if k in model_dict:
                if "patch_embed.proj.weight" in k:
                    # a small trick to copy the RGB channels from the pretrained swin
                    if full_dict[k].shape != model_dict[k].shape:
                        new_weight = torch.zeros((model_dict[k].shape),device=model_dict[k].device)
                        new_weight[:,:full_dict[k].shape[1]] = full_dict[k]
                        full_dict[k]= new_weight

                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                              model_dict[k].shape))
                    del full_dict[k]
        msg = self.load_state_dict(full_dict, strict=False)
        print("********************************************************************************")
        print("Loaded from swin encoder")
        print(msg)
        print("********************************************************************************")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_down(self,x,**kwargs):

        x_downsample = []
        for inx, layer in enumerate(self.layers):
            x,x_down = layer(x,**kwargs)
            x = self.temporal_downsample_steps[inx](x)
            x_downsample.append(x_down)

        return x, x_downsample

    def forward(self, x,**kwargs):
        x = self.prep_forward(x)

        x,x_downsample = self.forward_down(x,**kwargs)

        x_upsample = []

        for i in range(0,self.num_layers+1):
            if i == 0:
                B, L, C = x.shape
                x = self.pred_target[i](x.reshape(B, int(L ** 0.5), int(L ** 0.5), C).permute(0, 3, 1, 2))
            elif i==1:
                x = self.pred_target[i](x)
            else:
                B, L, C = x_downsample[self.num_layers-i].shape
                skipped_x = x_downsample[self.num_layers-i]
                if not self.temporal_upsample and B !=x.shape[0]:
                    skipped_x = Resize((L,x.shape[0]))(skipped_x.permute(2,1,0)).permute(2,1,0)
                    B = x.shape[0]
                x = self.pred_target[i](torch.cat((x,(skipped_x.reshape(B,int(L**0.5),int(L**0.5),C).permute(0, 3, 1, 2))),dim=1))
            x_upsample.append(x)
            x = self.temporal_upsample_blocks[i](x)


        x = self.pred_target[-2](x)
        x_final = self.pred_target[-1](x)
        #TODO final convolution layer
        if self.return_pred_embedding:
            return x_final,x_downsample,x

        return x_final, x_downsample

    def prep_forward(self,x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class Swin2x1TransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0,
                 T_length=3,temporal_type="conv",padding_type="reflect",temporal_dropout=0.1,
                 shift_temporal = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.shift_temporal = shift_temporal
        self.T_length = T_length
        # setting up the 1D Convolution

        if temporal_type =="conv":
            self.temporal_convolution = Temporal1DResBlock(in_channels=dim,out_channels=dim,T_length=T_length,
                                                           padding_type=padding_type)

        elif temporal_type == "recurrent":
            self.temporal_convolution = SpatioTemporalGru(in_channels=dim,out_channels=dim,T_length=T_length)

        elif temporal_type == "ltaeC":
            self.temporal_convolution = MultiLTAEWrapperCaFFe(in_channels=dim,dropout=temporal_dropout,T_length=T_length)

        elif temporal_type == "nothing":
            self.temporal_convolution = Identity()

    def forward(self, x,names=[]):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        if self.shift_temporal:
            shifted_x = self.temporal_recompose(shifted_x)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_temporal:
            shifted_x = self.temporal_recompose(shifted_x,reverse=True)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        # 1D Temporal convolution
        x = self.temporal_convolution(x,names=names)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


    def temporal_recompose(self,x,reverse=False):

        B, H, W, C = x.shape
        x_old = x

        x_i = []
        for i in range(self.T_length):
            x_i.append(x[i::self.T_length].unsqueeze(dim=1))

        # split x into T_length sections
        x = torch.cat(x_i, dim=1)

        x = self.temporal_shift(x,reverse=reverse)

        # shape back into the right form for Swin
        for i in range(self.T_length):
            x_old[i::self.T_length] = x[:, i]

        return x_old.reshape(B, H,W, C)

    def temporal_shift(self,feature_map, reverse=False):
        pattern1 = torch.zeros(feature_map.shape, device=feature_map.device)
        pattern2 = torch.zeros(feature_map.shape, device=feature_map.device)

        shift = 1 if reverse else 0
        rev = 1 - shift

        pattern1[:, :, shift::2, shift::2] = 1.0
        pattern2[:, :, rev::2, rev::2] = 1.0
        forward = torch.mul(feature_map, pattern1)
        backward = torch.mul(feature_map, pattern2)
        feature_map[:, 1:, rev::2, rev::2] = backward[:, :-1, rev::2, rev::2]
        feature_map[:, :-1, shift::2, shift::2] = forward[:, 1:, shift::2, shift::2]
        if reverse:
            feature_map[:, 0, rev::2, rev::2] = forward[:, 1, shift::2, shift::2]
            feature_map[:, -1, shift::2, shift::2] = backward[:, -2, rev::2, rev::2]

        return feature_map


from model.custom.custom_modules import *

class Temporal1DResBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None,
                num_groups=32,T_length=3,padding_type="reflect"):
        super().__init__()
        self.in_channels = in_channels
        self.padding_type = padding_type
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.T_length = T_length

        self.norm1 = Normalize(in_channels, num_groups)
        self.conv1 = zero_module(torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=(3,1,1),
                                     stride=1))
    def forward(self, x,names=[]):

        threeD = False

        if len(x.shape) == 4:
            threeD = True
            B,C,H,W = x.shape
            x = x.permute(0,2,3,1)
        else:
            B,L,C = x.shape
            # Reshaping X from patch-swin form to 3D
            x = x.reshape(B,int(L**0.5),int(L**0.5),C)
        x_old = x

        """
        x_i = []
        for i in range(self.T_length):
            x_i.append(x[i::self.T_length].unsqueeze(dim=1))

        #split x into T_length sections
        x = torch.cat(x_i,dim=1).permute(0,4, 1, 2, 3)
        """

        x = x.view(
            self.T_length,
            x.size(0) // self.T_length,
            *x.shape[1:]
        ).permute(1, 4, 0, 2, 3)


        h = x
        h = self.norm1(h)
        h = nonlinearity(h)

        #manual circular padding. Reasoning is otherwiser the outer time dimensions wont see each other with a kernel size of 3
        h = torch.nn.functional.pad(h,(0,0,0,0,1,1),self.padding_type)
        h = self.conv1(h)

        x = x+h

        x = x.permute(0,2,3,4,1)
        # shape back into the right form for Swin
        for i in range(self.T_length):
            x_old[i::self.T_length] = x[:,i]

        if threeD:
            x = x_old.permute(0,3,1,2)
        else:
            x = x_old.reshape(B,L,C)

        return x
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm,out_dim=None,nor_out_dim=None):
        super().__init__()

        if out_dim is None:
            out_dim = dim_scale * dim
        if nor_out_dim is None:
            nor_out_dim = (dim_scale*dim)//dim_scale**2

        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, out_dim, bias=False) #if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(nor_out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale**2)
        x = x.view(B, -1, C // self.dim_scale**2)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., qk_scale=None,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0,T_length=3,temporal_type="conv",padding_type="reflect",
                 shift_temporal=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Swin2x1TransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 T_length=T_length,
                                 temporal_type=temporal_type,
                                 padding_type=padding_type,
                                 shift_temporal=shift_temporal)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x,**kwargs):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,**kwargs)
        x_pre_down = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_pre_down

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, pretrained_window_size=0,
                 return_pre_up=False,T_length=3,shift_temporal=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.return_pre_up = return_pre_up

        self.blocks = nn.ModuleList([
            Swin2x1TransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size= pretrained_window_size,
                                 T_length=T_length,
                                 shift_temporal=shift_temporal)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_up = x
        if self.upsample is not None:
            x = self.upsample(x)

        if self.return_pre_up:
            return x, x_up
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class TemporalPatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,T_length=3,combine_ch=3, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.T_length = T_length
        self.combine_ch = combine_ch

        self.combine_temporal = nn.Conv3d(embed_dim,embed_dim,kernel_size=(combine_ch,1,1),stride=(combine_ch,1,1),padding=0)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):

        x = self.proj(x)  # B Ph*Pw C

        ### TEMPORAL stuff

        x_i = []
        for i in range(self.T_length):
            x_i.append(x[i::self.T_length].unsqueeze(dim=1))

        # split x into T_length sections
        x = torch.cat(x_i, dim=1).permute(0,2,1,3,4)
        x = self.combine_temporal(x)

        x = x.permute(0, 2,1, 3, 4)

        result = torch.zeros((x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]),device=x.device)
        # shape back into the right form for Swin
        for i in range(x.shape[1]):
            result[i::x.shape[1]] = x[:, i]

        x = result.flatten(2).transpose(1, 2) # B Ph*Pw C

        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class Swin2x1TransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],T_length=3,
                 shift_temporal=False,**kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               T_length=T_length,
                               shift_temporal=shift_temporal
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops