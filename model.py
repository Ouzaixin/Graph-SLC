import config
import math
import numpy as np
import torch
from torch import nn, einsum

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    a = np.random.uniform(low,high,(fan_in,fan_out))
    a = a.astype('float32')
    a = torch.from_numpy(a)
    return a

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim, dim_out, kernel = 4, stride = 2, padding = 1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim_out, kernel, stride, padding)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=16, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=16):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2) 

        attn = torch.einsum(
            "bnchwd, bncyxz -> bnhwdyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, depth, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, depth, height, width, depth)

        out = torch.einsum("bnhwdyxz, bncyxz -> bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, norm_groups=16, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        return x
    
class Reconstruction(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=1,
        inner_channel=(32, 64, 128, 32),
        norm_groups=16,
        dropout=0,
    ):
        super().__init__()

        # share layers
        shared_layer = []
        shared_layer.append(nn.Conv3d(in_channel, inner_channel[0], kernel_size=3, padding=1))
        shared_layer.append(Upsample(inner_channel[0], inner_channel[0]))
        shared_layer.append(ResnetBlocWithAttn(inner_channel[0], inner_channel[0], norm_groups=norm_groups, dropout=dropout, with_attn=True))
        shared_layer.append(Upsample(inner_channel[0], inner_channel[1]))
        shared_layer.append(ResnetBlocWithAttn(inner_channel[1], inner_channel[1], norm_groups=norm_groups, dropout=dropout, with_attn=True))

        # specific layers
        # MRI
        specific_layer_MRI = []
        specific_layer_MRI.append(Upsample(inner_channel[1], inner_channel[2]))
        specific_layer_MRI.append(ResnetBlocWithAttn(inner_channel[2], inner_channel[2], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_MRI.append(Upsample(inner_channel[2], inner_channel[3]))
        specific_layer_MRI.append(ResnetBlocWithAttn(inner_channel[3], inner_channel[3], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_MRI.append(nn.Conv3d(inner_channel[3], out_channel, kernel_size=3, padding=1))

        # FDG
        specific_layer_FDG = []
        specific_layer_FDG.append(Upsample(inner_channel[1], inner_channel[2]))
        specific_layer_FDG.append(ResnetBlocWithAttn(inner_channel[2], inner_channel[2], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_FDG.append(Upsample(inner_channel[2], inner_channel[3]))
        specific_layer_FDG.append(ResnetBlocWithAttn(inner_channel[3], inner_channel[3], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_FDG.append(nn.Conv3d(inner_channel[3], out_channel, kernel_size=3, padding=1))

        # Aβ
        specific_layer_AV45 = []
        specific_layer_AV45.append(Upsample(inner_channel[1], inner_channel[2]))
        specific_layer_AV45.append(ResnetBlocWithAttn(inner_channel[2], inner_channel[2], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_AV45.append(Upsample(inner_channel[2], inner_channel[3]))
        specific_layer_AV45.append(ResnetBlocWithAttn(inner_channel[3], inner_channel[3], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_AV45.append(nn.Conv3d(inner_channel[3], out_channel, kernel_size=3, padding=1))

        # Tau
        specific_layer_Tau = []
        specific_layer_Tau.append(Upsample(inner_channel[1], inner_channel[2]))
        specific_layer_Tau.append(ResnetBlocWithAttn(inner_channel[2], inner_channel[2], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_Tau.append(Upsample(inner_channel[2], inner_channel[3]))
        specific_layer_Tau.append(ResnetBlocWithAttn(inner_channel[3], inner_channel[3], norm_groups=norm_groups, dropout=dropout, with_attn=False))
        specific_layer_Tau.append(nn.Conv3d(inner_channel[3], out_channel, kernel_size=3, padding=1))

        self.shared_part = nn.ModuleList(shared_layer)
        self.specific_layer_MRI = nn.ModuleList(specific_layer_MRI)
        self.specific_layer_FDG = nn.ModuleList(specific_layer_FDG)
        self.specific_layer_AV45 = nn.ModuleList(specific_layer_AV45)
        self.specific_layer_Tau = nn.ModuleList(specific_layer_Tau)

    def forward(self, x):
        x = x.reshape(config.batch_size, 1, 12, 12, 12).to(config.device)
        for layer in self.shared_part:
            x = layer(x)
        
        rec_MRI = x.copy()
        for layer in self.specific_layer_MRI:
            rec_MRI = layer(rec_MRI)
        
        rec_FDG = x.copy()
        for layer in self.specific_layer_FDG:
            rec_FDG = layer(rec_FDG)

        rec_AV45 = x.copy()
        for layer in self.specific_layer_MRI:
            rec_AV45 = layer(rec_AV45)
        
        rec_Tau = x.copy()
        for layer in self.specific_layer_FDG:
            rec_Tau = layer(rec_Tau)
        return rec_MRI, rec_FDG, rec_AV45, rec_Tau