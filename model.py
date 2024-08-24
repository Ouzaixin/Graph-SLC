import math
import torch
from torch import nn
import numpy as np
import config

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    a = np.random.uniform(low,high,(fan_in,fan_out))
    a = a.astype('float32')
    a = torch.from_numpy(a).cuda()
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

class CrossAttention(nn.Module):
    def __init__(self, x_channel, y_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, x_channel)
        self.q = nn.Linear(x_channel, y_channel)
        self.k = nn.Linear(y_channel, y_channel)
        self.v = nn.Linear(y_channel, y_channel)
        self.out = nn.Linear(y_channel, x_channel)

    def forward(self, input, y):
        b, n, H, W, D = input.shape
        x = self.norm(input)
        x = x.reshape(b, n, H*W*D).permute(0, 2, 1)
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        attn = torch.matmul(q,k.T) / math.sqrt(n)
        attn = torch.softmax(attn, -1)
        out = torch.matmul(attn, v)
        out = self.out(out)
        out = out.permute(0, 2, 1).reshape(b, n, H, W, D)
        return out

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

        shared_layer = [
            nn.Conv3d(in_channel, inner_channel[0], kernel_size=3, padding=1),
            Upsample(inner_channel[0], inner_channel[0]),
            ResnetBlocWithAttn(inner_channel[0], inner_channel[1], norm_groups=norm_groups, dropout=dropout, with_attn=True),
            Upsample(inner_channel[1], inner_channel[1]),
            ResnetBlocWithAttn(inner_channel[1], inner_channel[2], norm_groups=norm_groups, dropout=dropout, with_attn=True),
        ]

        def create_specific_layers():
            return nn.Sequential(
                Upsample(inner_channel[2], inner_channel[2]),
                ResnetBlocWithAttn(inner_channel[2], inner_channel[3], norm_groups=norm_groups, dropout=dropout, with_attn=False),
                Upsample(inner_channel[3], inner_channel[3]),
                ResnetBlocWithAttn(inner_channel[3], inner_channel[3], norm_groups=norm_groups, dropout=dropout, with_attn=False),
                nn.Conv3d(inner_channel[3], out_channel, 3, 1, 1)
            )

        self.shared_part = nn.Sequential(*shared_layer)
        self.specific_layer_MRI = create_specific_layers()
        self.specific_layer_FDG = create_specific_layers()
        self.specific_layer_AV45 = create_specific_layers()
        self.specific_layer_Tau = create_specific_layers()

    def forward(self, x):
        x = x.reshape(config.batch_size, 1, 12, 12, 12).to(config.device)
        x = self.shared_part(x)
        
        rec_MRI = self.specific_layer_MRI(x)
        rec_FDG = self.specific_layer_FDG(x)
        rec_AV45 = self.specific_layer_AV45(x)
        rec_Tau = self.specific_layer_Tau(x)

        return rec_MRI, rec_FDG, rec_AV45, rec_Tau
