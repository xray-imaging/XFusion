import torch
from torch import nn as nn
from torch.nn import functional as F

from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from .edvr_arch import PCDAlignment, InversePCDAlignment
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from functools import reduce, lru_cache
from operator import mul
from xfusion.train.basicsr.utils.registry import ARCH_REGISTRY

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

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
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 conv_window_size=3):
        super(RSTB, self).__init__()

        self.dim = dim
        #self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, conv_window_size, 1, (conv_window_size-1)//2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, conv_window_size, 1, (conv_window_size-1)//2), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, conv_window_size, 1, (conv_window_size-1)//2))
        elif resi_connection == '0conv':
            self.conv = nn.Identity()
        '''
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        '''
    def forward(self, x, x_size_t):
        x_t = self.residual_group(x)
        x_t = rearrange(x_t, 'b c d h w -> b d c h w').view((-1,*x_size_t[1:]))
        x_t = self.conv(x_t)
        x_t = rearrange(x_t.view((x_size_t[0],-1,*x_size_t[1:])), 'b d c h w -> b c d h w')
        return x_t + x

    '''
    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops
    '''
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@ARCH_REGISTRY.register()
class PatchTransformerSTF(nn.Module):

    def __init__(self,
                 num_in_ch=1,
                 num_feat_ext=192,
                 embed_dim=96,
                 num_frame=3,
                 #num_frame_hi=2,
                 center_frame_idx = None,
                 #num_extract_block=5,
                 #num_reconstruct_block=10,
                 depths=[6,6,54,6],
                 num_heads=[1,1,1,1],
                 mlp_ratio = 4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint = False,
                 upscale = 4,
                 #patchsize = [(1,1)],
                 window_size = [25,16,16],
                 resi_connection='1conv',
                 conv_window_size = 3,
                 upsampler='',
                 num_feat_rec = 64,
                 align_features_ok = False,
                 deformable_groups=8,
                 num_extract_block=5,
                 adapt_deformable_conv = False):
        super(PatchTransformerSTF, self).__init__()
        num_out_ch= num_in_ch
        self.num_out_ch = num_out_ch
        self.align_features_ok = align_features_ok
        self.adapt_deformable_conv = adapt_deformable_conv
        if not adapt_deformable_conv:
            assert num_feat_ext == embed_dim
        self.embed_dim = embed_dim
        self.upsampler = upsampler
        self.upscale = upscale
        self.num_feat_ext = num_feat_ext
        self.num_feat_rec = num_feat_rec
        self.deformable_groups = deformable_groups
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.num_layers = len(depths)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat_ext, 3, 1, 1)

        self.conv_first_hi = nn.Conv2d(num_in_ch, num_feat_ext, 3, 1, 1)
        self.conv1_hi = nn.Conv2d(num_feat_ext, num_feat_ext, 3, 1, 1)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2_hi = nn.Conv2d(num_feat_ext, num_feat_ext, 3, 1, 1)
        self.max_pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        if align_features_ok:
            # align features within the HR and LR branches, separately
            self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat_ext)
            self.conv_l2_1 = nn.Conv2d(num_feat_ext, num_feat_ext, 3, 2, 1)
            self.conv_l2_2 = nn.Conv2d(num_feat_ext, num_feat_ext, 3, 1, 1)
            self.conv_l3_1 = nn.Conv2d(num_feat_ext, num_feat_ext, 3, 2, 1)
            self.conv_l3_2 = nn.Conv2d(num_feat_ext, num_feat_ext, 3, 1, 1)

            # pcd and tsa module
            self.pcd_align = PCDAlignment(num_feat=num_feat_ext, deformable_groups=deformable_groups)
        if adapt_deformable_conv:
            self.channel_projection = nn.Conv2d(num_feat_ext, embed_dim, 1, 1, 0)
        #blocks = []
        #for _ in range(stack_num):
        #    blocks.append(TransformerBlock(patchsize, hidden=num_feat, window_size=window_size))
        #self.transformer = nn.Sequential(*blocks)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
                conv_window_size=conv_window_size)
            self.layers.append(layer)

        self.num_features = embed_dim
        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, conv_window_size, 1, (conv_window_size-1)//2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, conv_window_size, 1, (conv_window_size-1)//2), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, conv_window_size, 1, (conv_window_size-1)//2))
        elif resi_connection == '0conv':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat_rec, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat_rec)
            self.conv_last = nn.Conv2d(num_feat_rec, num_out_ch, 3, 1, 1)
        #elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
        #    self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
        #                                    (patches_resolution[0], patches_resolution[1]))
        '''
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat_rec, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat_rec, num_feat_rec, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat_rec, num_feat_rec, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat_rec, num_feat_rec, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat_rec, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        '''


        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, sample):
        x, y = sample['lq'], sample['hq']
        b, t, c, h, w = x.size()
        b_,t_,c_,h_,w_ = y.size()

        #x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        y = self.lrelu(self.conv_first_hi(y.view(-1,c_,h_,w_)))
        y = self.max_pool1(self.lrelu(self.conv1_hi(y)))
        y = self.max_pool2(self.lrelu(self.conv2_hi(y)))
        feat_l1 = torch.cat((feat_l1.view(b, t, -1, h, w), y.view(b, t_, -1, h, w)), dim=1)
        t = t + t_

        if self.align_features_ok:
            feat_l1 = self.feature_extraction(feat_l1.view(b * t, -1, h, w))
            # L2
            feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
            feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
            # L3
            feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
            feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

            feat_l1 = feat_l1.view(b, t, -1, h, w)
            feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
            feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

            # PCD alignment
            ref_feat_l = [  # reference feature list
                feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
                feat_l3[:, self.center_frame_idx, :, :, :].clone()
            ]
            aligned_feat = []
            for i in range(t):
                nbr_feat_l = [  # neighboring feature list
                    feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
                ]
                aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))

            feat_l1 = torch.stack(aligned_feat, dim=1)

        if self.adapt_deformable_conv:
            feat_l1 = feat_l1.view(b * t, -1, h, w)
            feat_l1 = self.channel_projection(feat_l1)
            feat_l1 = feat_l1.view(b, t, -1, h, w)

        feat_l1_ = rearrange(feat_l1,'b d c h w -> b c d h w')
        #TODO: need to address the size differences between reference patch and all others (as they could be expanded in size)
        #TODO: maybe a better way is to split the expanded patch into multiple patches of same size as reference

        #feat_attention = self.transformer({'x':feat_l1,'m':None,'b':b,'c':feat_l1.size()[1]})['x'].view(b, t, -1, h, w)
        for layer in self.layers:
            feat_l1_ = layer(feat_l1_,(b,self.embed_dim,h,w))#TODO: check if contiguous
        feat_l1_ = rearrange(feat_l1_, 'b c d h w -> b d h w c')
        feat_l1_ = self.norm(feat_l1_)
        feat_l1_ = rearrange(feat_l1_, 'b d h w c -> b d c h w').view((-1,self.embed_dim,h,w))
        feat_l1_ = self.conv_after_body(feat_l1_)
        feat_l1_ = feat_l1_.view((b,-1,self.embed_dim,h,w)) + feat_l1

        #feat_attention = self.layers(feat_l1)
        #feat = feat_attention[:, self.center_frame_idx, :,:,:]
        feat = feat_l1_[:, self.center_frame_idx, :,:,:].contiguous()
        feat = self.conv_before_upsample(feat)
        out = self.conv_last(self.upsample(feat))
        '''
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        '''
        results = {'out': out}
        return results
    
@ARCH_REGISTRY.register()
class PatchTransformerVSTF(PatchTransformerSTF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.adapt_deformable_conv:
            self.channel_projection_postfusion = nn.Conv2d(self.embed_dim, self.num_feat_ext, 1, 1, 0)
        
        if self.align_features_ok:
            self.conv_l2_1_postfusion = nn.Conv2d(self.num_feat_ext, self.num_feat_ext, 3, 2, 1)
            self.conv_l2_2_postfusion = nn.Conv2d(self.num_feat_ext, self.num_feat_ext, 3, 1, 1)
            self.conv_l3_1_postfusion = nn.Conv2d(self.num_feat_ext, self.num_feat_ext, 3, 2, 1)
            self.conv_l3_2_postfusion = nn.Conv2d(self.num_feat_ext, self.num_feat_ext, 3, 1, 1)
            self.pcd_align_postfusion = InversePCDAlignment(num_feat=self.num_feat_ext, deformable_groups=self.deformable_groups)
        
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.num_feat_ext, self.num_feat_rec, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(self.upscale, self.num_feat_rec)
            self.conv_last = nn.Conv2d(self.num_feat_rec, self.num_out_ch, 3, 1, 1)
        #elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
        #    self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
        #                                    (patches_resolution[0], patches_resolution[1]))
        '''
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.num_feat_ext, self.num_feat_rec, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(self.num_feat_rec, self.num_feat_rec, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(self.num_feat_rec, self.num_feat_rec, 3, 1, 1)
            self.conv_hr = nn.Conv2d(self.num_feat_rec, self.num_feat_rec, 3, 1, 1)
            self.conv_last = nn.Conv2d(self.num_feat_rec, self.num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(self.num_feat_ext, self.num_out_ch, 3, 1, 1)
        '''
    def forward(self, sample):
        x, y = sample['lq'], sample['hq']
        b, t1, c, h, w = x.size()
        b_,t2,c_,h_,w_ = y.size()

        #x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        x = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        y = self.lrelu(self.conv_first_hi(y.view(-1,c_,h_,w_)))
        y = self.max_pool1(self.lrelu(self.conv1_hi(y)))
        y = self.max_pool2(self.lrelu(self.conv2_hi(y)))
        aligned_feat = torch.cat((x.view(b, t1, -1, h, w), y.view(b, t2, -1, h, w)), dim=1)
        t = t1 + t2

        if self.align_features_ok:
            feat_l1 = self.feature_extraction(aligned_feat.view(b * t, -1, h, w))
            # L2
            feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
            feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
            # L3
            feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
            feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

            feat_l1 = feat_l1.view(b, t, -1, h, w)
            feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
            feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

            # PCD alignment
            ref_feat_l = [  # reference feature list
                feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
                feat_l3[:, self.center_frame_idx, :, :, :].clone()
            ]
            aligned_feat = []
            for i in range(t):
                nbr_feat_l = [  # neighboring feature list
                    feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
                ]
                aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))

            aligned_feat = torch.stack(aligned_feat, dim=1)

        if self.adapt_deformable_conv:
            aligned_feat = self.channel_projection(aligned_feat.view(b * t, -1, h, w)).view(b, t, -1, h, w)

        aligned_feat_ = rearrange(aligned_feat,'b d c h w -> b c d h w')
        #TODO: need to address the size differences between reference patch and all others (as they could be expanded in size)
        #TODO: maybe a better way is to split the expanded patch into multiple patches of same size as reference

        #feat_attention = self.transformer({'x':feat_l1,'m':None,'b':b,'c':feat_l1.size()[1]})['x'].view(b, t, -1, h, w)
        for layer in self.layers:
            aligned_feat_ = layer(aligned_feat_,(b,self.embed_dim,h,w))#TODO: check if contiguous
        aligned_feat_ = rearrange(aligned_feat_, 'b c d h w -> b d h w c')
        aligned_feat_ = self.norm(aligned_feat_)
        aligned_feat_ = rearrange(aligned_feat_, 'b d h w c -> b d c h w').view((-1,self.embed_dim,h,w))
        aligned_feat_ = self.conv_after_body(aligned_feat_)
        aligned_feat_ = aligned_feat_.view((b,-1,self.embed_dim,h,w)) + aligned_feat

        if self.adapt_deformable_conv:
            aligned_feat_ = self.channel_projection_postfusion(aligned_feat_.view(b * t, -1, h, w))
        else:
            aligned_feat_ = aligned_feat_.view(b * t, -1, h, w)

        if self.align_features_ok:
            feat_l1_postfusion = aligned_feat_
            feat_l2_postfusion = self.lrelu(self.conv_l2_1_postfusion(feat_l1_postfusion))
            feat_l2_postfusion = self.lrelu(self.conv_l2_2_postfusion(feat_l2_postfusion))

            feat_l3_postfusion = self.lrelu(self.conv_l3_1_postfusion(feat_l2_postfusion))
            feat_l3_postfusion = self.lrelu(self.conv_l3_2_postfusion(feat_l3_postfusion))

            feat_l1_postfusion = feat_l1_postfusion.view(b, t, -1, h, w)
            feat_l2_postfusion = feat_l2_postfusion.view(b, t, -1, h // 2, w // 2)
            feat_l3_postfusion = feat_l3_postfusion.view(b, t, -1, h // 4, w // 4)

            inverse_aligned_feat = []

            # only inverse align enhanced LR feature maps for subsequent reconstruction
            for i in range(t1):
                nbr_feat_l = [  # neighboring feature list
                    feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
                ]
                nbr_feat_postfusion_l = [
                    feat_l1_postfusion[:, i, :, :, :].clone(), feat_l2_postfusion[:, i, :, :, :].clone(), feat_l3_postfusion[:, i, :, :, :].clone()
                ]
                inverse_aligned_feat.append(self.pcd_align_postfusion(ref_feat_l, nbr_feat_l, nbr_feat_postfusion_l))
            inverse_aligned_feat = torch.stack(inverse_aligned_feat, dim=1)

        if self.align_features_ok:
            feat = inverse_aligned_feat.view(b * t1, -1, h, w)
        else:
            #feat_attention = self.layers(feat_l1)
            #feat = feat_attention[:, self.center_frame_idx, :,:,:]
            #feat = aligned_feat_.view(b, t, -1, h, w)[:, self.center_frame_idx, :,:,:].contiguous()
            feat =  aligned_feat_.view(b, t, -1, h, w)[:,:t1,:,:,:].view(b * t1, -1, h, w)
        feat = self.conv_before_upsample(feat)
        out = self.conv_last(self.upsample(feat))


        out = out.view(b, t1, self.num_out_ch, h_, w_)
        '''
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        '''
        results = {'out': out}
        return results