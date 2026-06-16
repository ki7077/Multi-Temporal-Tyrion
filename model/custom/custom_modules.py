"""
    This code originates from latent diffusion https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py but was slightly adjusted. Eigentlich Taming Transformers
"""

# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


#Normalize trying to enforce 4 groups minimum
def Normalize(in_channels, num_groups=32):
    if in_channels<num_groups:
        if in_channels % 4 == 0 and in_channels / 4 > 1.0:
            num_groups = in_channels // 4
        else:
            num_groups = in_channels

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv,sample_all_directions=True):
        super().__init__()
        self.with_conv = with_conv
        self.sample_all_directions = sample_all_directions

        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        #x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.sample_all_directions:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            custom_size = (x.shape[-2]*2,x.shape[-1])
            x = torch.nn.functional.interpolate(x, size=custom_size, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv,custom_stride=(2,2),custom_padding=None):
        super().__init__()
        self.with_conv = with_conv
        self.padding = custom_padding
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=custom_stride,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            if self.padding is None:
                pad = (0,1,0,1)
            else:
                pad = self.padding
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.1,num_groups=32,padding_type="zeros"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels,num_groups)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode=padding_type)

        self.norm2 = Normalize(out_channels,num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode=padding_type)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     padding_mode=padding_type)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    padding_mode=padding_type)


    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

