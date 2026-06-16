"""
Multi-temporal U-TAE Implementation
Inspired by U-TAE Implementation (Vivien Sainte Fare Garnot (github/VSainteuf))
"""
import numpy as np
import torch
import torch.nn as nn



class MultiLTAEWrapperCaFFe(nn.Module):
    def __init__(self,T_length=8,**kwargs):
        super(MultiLTAEWrapperCaFFe, self).__init__()

        self.network = MultiLTAE(T=10000,**kwargs)
       # self.months_to_positions = get_monthly_dates_dict_CaFFe()
        self.T_length = T_length
        self.resweight = nn.Parameter(torch.Tensor([0]))


    def parse_name(self,names):
        parsed_to_month = torch.zeros(len(names),dtype=torch.int32)
        for i in range(len(names)):
            parsed_to_month[i] = self.months_to_positions[names[i].split('_')[1]]

        return parsed_to_month


    #input N*T x P x C
    def forward(self,x,names):

        batch_positions = names#self.parse_name(names)

        x_old = x
        x_i = []

        for i in range(self.T_length):
            x_i.append(x[i::self.T_length].unsqueeze(dim=1))

        x = torch.cat(x_i, dim=1)
        mem_s = x.shape # N T Patch C
        if len(mem_s) == 4: # N T Patch C -> N T C Patch -> N T C H W
            x = x.permute(0,1,3,2).reshape(mem_s[0],mem_s[1],mem_s[3],int(np.sqrt(mem_s[2])),int(np.sqrt(mem_s[2]))) # N x T x C x H X W
        x = self.network(x,batch_positions=batch_positions)

        if len(mem_s) == 4:
            x = x.reshape(mem_s[0],mem_s[1],mem_s[3],-1).permute(0,1,3,2) # N x T x P x C

        # shape back into the right form for Swin
        for i in range(self.T_length):
            x_old[i::self.T_length] += self.resweight * x[:, i]

        return x_old


class MultiLTACEWrapper(nn.Module):
    def __init__(self,T_length=8,T_length_fast=16,**kwargs):
        super(MultiLTACEWrapper, self).__init__()

        #positional encoding should still be good from previous layers
        self.network = MultiLTACE(**kwargs)
        self.months_to_positions = get_monthly_dates_dict()
        self.T_length = T_length
        self.T_length_fast = T_length_fast

    def parse_name(self,names):
        parsed_to_month = torch.zeros(len(names),dtype=torch.int32)
        for i in range(len(names)):
            correct_month = int(names[i].split('_')[1].split('-')[1])-1

            if 2019 == int(names[i].split('_')[1].split('-')[0]):
                correct_month+= 12

            parsed_to_month[i] = self.months_to_positions[correct_month]

        return parsed_to_month


    #input N*T x P x C
    def forward(self,x,y,names_y,**kwargs):

        batch_positions = self.parse_name(names_y)

        x_old = x
        x_i = []
        y_i = []

        for i in range(self.T_length):
            x_i.append(x[i::self.T_length].unsqueeze(dim=1))
        for i in range(self.T_length_fast):
            y_i.append(y[i::self.T_length_fast].unsqueeze(dim=1))

        x = torch.cat(x_i, dim=1)
        y = torch.cat(y_i,dim=1)
        mem_s = x.shape # N T Patch C
        x = x.permute(0,1,3,2).reshape(mem_s[0],mem_s[1],mem_s[3],int(np.sqrt(mem_s[2])),int(np.sqrt(mem_s[2]))) # N x T x C x H X W

        x = self.network(x,y=y,batch_positions=batch_positions)
        x = x.reshape(mem_s[0],mem_s[1],mem_s[3],-1).permute(0,1,3,2) # N x T x P x C

        # shape back into the right form for Swin
        for i in range(self.T_length):
            x_old[i::self.T_length] += x[:, i]

        return x_old


"""
    method from https://github.com/ElliotVincent/SitsSCD
"""
from datetime import date
import pandas as pd

def get_monthly_dates_dict():
    s_date = date(2018, 1, 1)
    e_date = date(2019, 12, 31)
    dates_monthly = [f'{year}-{month}-01' for year, month in zip(
        [2018 for _ in range(12)] + [2019 for _ in range(12)],
        [f'0{m}' for m in range(1, 10)] + ['10', '11', '12'] + [f'0{m}' for m in range(1, 10)] + ['10', '11', '12']
    )]
    dates_daily = pd.date_range(s_date, e_date, freq='d').strftime('%Y-%m-%d').tolist()
    monthly_dates = []
    i, j = 0, 0
    while i < 730 and j < 24:
        if dates_monthly[j] == dates_daily[i]:
            monthly_dates.append(i)
            j += 1
        i += 1
    return monthly_dates


def get_monthly_dates_dict_CaFFe():
    s_date = date(1995, 1, 1)
    e_date = date(2020, 12, 31)
    dates_daily = pd.date_range(s_date, e_date, freq='d').strftime('%Y-%m-%d').tolist()

    date_to_number = dict()
    names = np.array(dates_daily)
    for i in range(names.shape[0]):
        date_to_number[names[i]] = i
    return date_to_number


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=730, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * torch.div(torch.arange(offset, offset + d).float(), 2, rounding_mode='floor') / (d+offset)
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table





class MultiLTAE(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=8,
        d_k=16,
        dropout=0.2,
        T=730,
        offset=0,
        return_att=False,
        positional_encoding=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(MultiLTAE, self).__init__()
        self.in_channels = in_channels
        self.mlp = [in_channels, in_channels]
        self.return_att = return_att
        self.n_head = n_head
        self.d_model = in_channels
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head, offset=offset
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp).to(out.device)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)  # h x (sz_b*h*w) x t x (d//h), h x (sz_b*h*w) x t x t
        out = out.permute(1, 2, 0, 3).contiguous().view(sz_b * h * w, seq_len, -1)  # Concatenate heads

        out = self.dropout(self.mlp(out.view(sz_b * h * w * seq_len, -1)))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, seq_len, -1).permute(0, 3, 4, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len, seq_len).permute(
            0, 1, 4, 5, 2, 3
        )  # head x b x t x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiLTACE(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=8,
        d_k=16,
        dropout=0.2,
        T=730,
        offset=0,
        return_att=False,
        positional_encoding=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(MultiLTACE, self).__init__()
        self.in_channels = in_channels
        self.mlp = [in_channels, in_channels]
        self.return_att = return_att
        self.n_head = n_head
        self.d_model = in_channels
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head, offset=offset
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadCrossAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.in_norm_y = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )

        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.mlp[-1],
        )
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,y, batch_positions=None, pad_mask=None,**kwargs):
        sz_b, seq_len, d, h, w = x.shape
        sz_b_y, seq_len_y, d_y, h_y, w_y = y.shape

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out_y = y.permute(0, 3, 4, 1, 2).contiguous().view(sz_b_y * h_y * w_y, seq_len_y, d_y)

        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out_y = self.in_norm(out_y.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h_y))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w_y))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b_y * h_y * w_y, seq_len_y)
            out_y = out_y + self.positional_encoder(bp).to(out_y.device)

        #TODO have to permute the other tokens still + PE
        out, attn = self.attention_heads(out,k=out_y,v=out_y, pad_mask=pad_mask,**kwargs)  # h x (sz_b*h*w) x t x (d//h), h x (sz_b*h*w) x t x t
        out = out.permute(1, 2, 0, 3).contiguous().view(sz_b * h * w, seq_len, -1)  # Concatenate heads

        out = self.dropout(self.mlp(out.view(sz_b * h * w * seq_len, -1)))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, seq_len, -1).permute(0, 3, 4, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len, seq_len_y).permute(
            0, 1, 4, 5, 2, 3
        )  # head x b x t x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = self.fc1_q(v).view(sz_b, seq_len, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k).permute(0, 2, 1)  # (n*b) x dk x lk

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, sz_b, seq_len, seq_len)
        output = output.view(n_head, sz_b, seq_len, d_in // n_head)
        return output, attn


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductCrossAttention(temperature=np.power(d_k, 0.5))

    def forward(self,q,k, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()
        sz_b_q,seq_len_q,_ =q.size()

        q = self.fc1_q(q).view(sz_b_q, seq_len_q, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, d_k).permute(0, 2, 1)  # (n*b) x dk x lk

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, sz_b, seq_len, seq_len_q)
        output = output.view(n_head, sz_b, seq_len_q, d_in // n_head)
        return output, attn



class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(k, q)
        attn = attn / self.temperature

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class ScaledDotProductCrossAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(q.permute(0,2,1),k.permute(0,2,1))
        attn = attn / self.temperature

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn