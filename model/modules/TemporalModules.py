from model.utils import zero_module
from torch.nn import Linear, Dropout, LayerNorm

from model.custom.custom_modules import nonlinearity
from model.modules.SwinV2Modules import *
from model.utils import zero_module
from torch._C import _ImperativeEngine as ImperativeEngine
import math
__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    _execution_engine = ImperativeEngine()


class Identity(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
    def forward(self,x,**kwargs):
        return x

class TemporalSingularDownSample(nn.Module):
    def __init__(self, in_channels,out_channels, T_length,padding_type="reflect"):
        super().__init__()
        self.T_length = T_length
        self.padding_type = padding_type

        self.conv = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=(3,1,1),
                                     stride=(2,1,1))

    def forward(self, x,**kwargs):
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

        x_i = []
        for i in range(self.T_length):
            x_i.append(x[i::self.T_length].unsqueeze(dim=1))

        #split x into T_length sections
        x = torch.cat(x_i,dim=1).permute(0,4, 1, 2, 3)

        downsampled_T_length = int(self.T_length/2)

        # manual circular padding. Reasoning is otherwiser the outer time dimensions wont see each other with a kernel size of 3
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), self.padding_type)
        x = self.conv(x)

        x = x.permute(0, 2, 3, 4, 1)
        # shape back into the right form for Swin
        x_old = x_old[:int(x_old.shape[0]/2)]
        for i in range(downsampled_T_length):
            x_old[i::downsampled_T_length] = x[:, i]
        if threeD:
            x = x_old.permute(0, 3, 1, 2)
        else:
            x = x_old.reshape(int(B/2), L, C)

        return x


class TemporalSingularUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv, T_length,
                 padding_type="reflect"):
        super().__init__()
        self.with_conv = with_conv
        self.T_length = T_length
        self.padding_type = padding_type

        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=(3, 1, 1),
                                        stride=1)

    def forward(self, x,**kwargs):
        threeD = False

        if len(x.shape) == 4:
            threeD = True
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
        else:
            B, L, C = x.shape
            # Reshaping X from patch-swin form to 3D
            x = x.reshape(B, int(L ** 0.5), int(L ** 0.5), C)
        x_old = x
        x = x.view(-1, self.T_length, *x.shape[1:]).permute(0, 4, 1, 2, 3)

        upsampled_T_length = self.T_length * 2
        output_size = (upsampled_T_length, x.shape[3], x.shape[4])

        x = torch.nn.functional.interpolate(x, size=output_size, mode="nearest")

        if self.with_conv:
            # manual circular padding. Reasoning is otherwiser the outer time dimensions wont see each other with a kernel size of 3
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), self.padding_type)
            x = self.conv(x)

        x = x.permute(0, 2, 3, 4, 1)
        x_old = torch.zeros((x_old.shape[0]*2,x_old.shape[1],x_old.shape[2],x_old.shape[3]),device=x_old.device)
        # shape back into the right form for Swin
        for i in range(upsampled_T_length):
            x_old[i::upsampled_T_length] = x[:, i]
        if threeD:
            x = x_old.permute(0, 3, 1, 2)
        else:
            x = x_old.reshape(int(B / 2), L, C)

        return x


class SpatioTemporalGru(nn.Module):
    def __init__(self, *, in_channels, out_channels=None,
                 num_groups=32, T_length=3, dropout=0.0, nhead=1,
                 re_zero=True):
        super().__init__()
        half_channels = int(0.5*in_channels)

        self.compress = torch.nn.Linear(in_features=in_channels,out_features=half_channels)
        self.compress_norm = LayerNorm(half_channels)
        self.gru_layer = torch.nn.GRU(half_channels,half_channels,batch_first=True,dropout=dropout,bidirectional=True)
        self.combine_bi = torch.nn.Linear(in_features=in_channels,out_features=in_channels)
        self.T_length = T_length
        self.norm = LayerNorm(in_channels)



    def forward(self,x,names=None):

        x_old = x

        x = x.view(self.T_length, -1, *x.shape[1:]).transpose(0, 1)

        mem_s = x.shape # N T Patch C
        x = x.permute(1,3,0,2).reshape(mem_s[1],mem_s[3],-1).permute(2,0,1)# (NxPatch) T C

        x = nonlinearity(self.compress_norm(self.compress(x)))
        x,_ = self.gru_layer(x)

        x = nonlinearity(self.norm(self.combine_bi(x)))
        #(NxPatch) T C -> T C (NxPatch) -> T C N Patch ->
        x = x.permute(1,2,0).reshape(mem_s[1],mem_s[3],mem_s[0],mem_s[2]).permute(2,0,3,1)

        # shape back into the right form for Swin
       # for i in range(self.T_length):
        #    x_old[i::self.T_length] += x[:, i]

        x_old = x_old.view(mem_s[0], mem_s[1], mem_s[2], mem_s[3])
        x_old += x
        x_old = x_old.view(mem_s[0]*mem_s[1], mem_s[2], mem_s[3])

        return x_old