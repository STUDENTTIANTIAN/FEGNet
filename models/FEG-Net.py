from os import stat_result
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
import math
from models.GBC import *
import traceback
from scipy.ndimage import gaussian_filter, laplace
import numpy as np
from functools import partial
from models.model.lib_mamba.vmambanew import SS2D
import pywt
class HeightWidthChannelEulerProcessor(nn.Module):
    def __init__(self, input_dimension, use_bias=False, operation_mode='fc'):
        super().__init__()
        self.channel_normalization = nn.BatchNorm2d(input_dimension)
        self.horizontal_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)
        self.vertical_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)
        self.channel_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)
        self.horizontal_transform_conv = nn.Conv2d(2 * input_dimension, input_dimension, (1, 7), stride=1,
                                                  padding=(0, 7 // 2), groups=input_dimension, bias=False)
        self.vertical_transform_conv = nn.Conv2d(2 * input_dimension, input_dimension, (7, 1), stride=1,
                                                 padding=(7 // 2, 0), groups=input_dimension, bias=False)
        self.operation_mode = operation_mode
        if operation_mode == 'fc':
            self.horizontal_phase_calculator = nn.Sequential(
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
            self.vertical_phase_calculator = nn.Sequential(
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
        else:
            self.horizontal_phase_calculator = nn.Sequential(
                nn.Conv2d(input_dimension, input_dimension, 3, stride=1, padding=1, groups=input_dimension, bias=False),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
            self.vertical_phase_calculator = nn.Sequential(
                nn.Conv2d(input_dimension, input_dimension, 3, stride=1, padding=1, groups=input_dimension, bias=False),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
        self.feature_fusion_layer = nn.Sequential(
            nn.Conv2d(4 * input_dimension, input_dimension, kernel_size=1, stride=1),
            nn.BatchNorm2d(input_dimension),
            nn.ReLU(inplace=True),
        )
    def forward(self, input_tensor):
        horizontal_phase = self.horizontal_phase_calculator(input_tensor)
        vertical_phase = self.vertical_phase_calculator(input_tensor)
        horizontal_amplitude = self.horizontal_feature_conv(input_tensor)
        vertical_amplitude = self.vertical_feature_conv(input_tensor)
        horizontal_euler = torch.cat([horizontal_amplitude * torch.cos(horizontal_phase),
                                      horizontal_amplitude * torch.sin(horizontal_phase)], dim=1)
        vertical_euler = torch.cat([vertical_amplitude * torch.cos(vertical_phase),
                                    vertical_amplitude * torch.sin(vertical_phase)], dim=1)
        original_input = input_tensor
        transformed_h = self.horizontal_transform_conv(horizontal_euler)
        transformed_w = self.vertical_transform_conv(vertical_euler)
        transformed_c = self.channel_feature_conv(input_tensor)
        merged_features = torch.cat([original_input, transformed_h, transformed_w, transformed_c], dim=1)
        output_tensor = self.feature_fusion_layer(merged_features)
        return output_tensor
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights
class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm1 = nn.LayerNorm(dim)
            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.norm2 = nn.LayerNorm(dim)
    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.sr1(x_1).reshape(B, C, -1).permute(0, 2, 1)
            x_1 = self.norm1(x_1)
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2).reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        return main_out, aux_out
class FeatureInteraction(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
    def forward(self, x1, x2, H, W):
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        c1, c2 = self.cross_attn(z1, z2, H, W)
        y1 = torch.cat((y1, c1), dim=-1)
        y2 = torch.cat((y2, c2), dim=-1)
        main_out = self.norm1(x1 + self.end_proj1(y1))
        aux_out = self.norm2(x2 + self.end_proj2(y2))
        return main_out, aux_out
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)
        self.oula = HeightWidthChannelEulerProcessor(in_channels//2)
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(x+self.oula(residual))
        return out
class FeatureFusion(nn.Module):
    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        num_heads = num_heads or max(1, dim // 32)
        self.cross = FeatureInteraction(dim=dim, reduction=reduction, num_heads=num_heads, sr_ratio=sr_ratio)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2, H, W)
        fuse = torch.cat((x1, x2), dim=-1)
        fuse = self.channel_emb(fuse, H, W)
        return fuse
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return torch.mul(self.weight, x)
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1', ssm_ratio=1, forward_type="v05"):
        super().__init__()
        assert in_channels == in_channels
        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.global_atten = SS2D(d_model=in_channels, d_state=1, ssm_ratio=ssm_ratio, initialize="v2",
                                 forward_type=forward_type, channel_first=True, k_group=2)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', groups=in_channels * 4)
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])
    def forward(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)).reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), x_h_in_levels.pop()], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :shapes_in_levels[i][2], :shapes_in_levels[i][3]]
        x_tag = next_x_ll
        x = self.base_scale(self.global_atten(x)) + x_tag
        return x
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()
    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        return std
class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        super(MCAGate, self).__init__()
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))
    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.sigmoid(out)
        out = out.expand_as(x)
        return x * out
class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=True):
        super(MCALayer, self).__init__()
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)
    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()
        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)
        return x_out
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.att=MCALayer(in_planes)
    def forward(self, x):
        res=x
        x=self.att(x)+res
        return x
class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, num_experts=4):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
            for _ in range(num_experts)
        ])
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1, bias=False),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        gates = self.gating(x)
        gates = gates.view(x.size(0), self.num_experts, 1, 1, 1)
        outputs = []
        for i, expert in enumerate(self.experts):
            outputs.append(expert(x).unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        out = (gates * outputs).sum(dim=1)
        return out
class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.add_module('dwconv3x3', DynamicConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                                   stride=1, padding=kernel_size // 2, groups=in_channels, bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', Mish())
        self.add_module('dwconv1x1', nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                               stride=1, padding=0, groups=in_channels, bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, groups=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = Mish()
        self.pw2 = Conv2d_BN(h, ed)
    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))
class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob
    def forward(self, x):
        if not self.training:
            return x
        batch_size = x.shape[0]
        random_tensor = self.survival_prob + torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x * binary_tensor / self.survival_prob
class Residual(nn.Module):
    def __init__(self, m, survival_prob=0.8):
        super().__init__()
        self.m = m
        self.stochastic_depth = StochasticDepth(survival_prob)
    def forward(self, x):
        return x + self.stochastic_depth(self.m(x))
class MobileMambaModule(nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25, kernels=3, ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.dim = dim
        self.global_channels = int(global_ratio * dim)
        self.local_channels = int(local_ratio * dim)
        self.identity_channels = dim - self.global_channels - self.local_channels
        self.local_op = nn.ModuleList([
            DWConv2d_BN_ReLU(self.local_channels, self.local_channels, k)
            for k in [3, 5, 7]
        ]) if self.local_channels > 0 else nn.Identity()
        self.global_op = MBWTConv2d(self.global_channels, kernel_size=kernels,
                                     ssm_ratio=ssm_ratio, forward_type=forward_type) \
            if self.global_channels > 0 else nn.Identity()
        self.cbam = CBAM(dim)
        self.proj = nn.Sequential(
            Mish(),
            Conv2d_BN(dim, dim),
            CBAM(dim)
        )
    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        if isinstance(self.local_op, nn.ModuleList):
            local_features = [op(x2) for op in self.local_op]
            local_features = torch.cat(local_features, dim=1)
            local_features = torch.mean(local_features, dim=1, keepdim=True)
            local_features = local_features.expand(-1, self.local_channels, -1, -1)
        else:
            local_features = self.local_op(x2)
        out = torch.cat([self.global_op(x1), local_features, x3], dim=1)
        out = self.cbam(out)
        return self.proj(out)
class MobileMambaBlock(nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25, kernels=5, ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(dim, dim, 3, pad=1, groups=dim))
        self.ffn0 = Residual(FFN(dim, int(dim * 2)))
        self.mixer = Residual(MobileMambaModule(dim, global_ratio, local_ratio, kernels, ssm_ratio, forward_type))
        self.dw1 = Residual(Conv2d_BN(dim, dim, 3, pad=1, groups=dim))
        self.ffn1 = Residual(FFN(dim, int(dim * 2)))
    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
    @staticmethod
    def get_module_name():
        return "simam"
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)
class diff_moudel(nn.Module):
    def __init__(self,in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()
    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        out = self.simam(out)
        return out
class CBM(nn.Module):
    def __init__(self,in_channel):
        super(CBM, self).__init__()
        self.diff_1 = diff_moudel(in_channel)
        self.simam = simam_module()
    def forward(self,x1,x2):
        d1 = self.diff_1(x1)
        d = torch.add(d1,x2)
        d = self.simam(d)
        return d
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc
class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, in_c, kernel_size=3, padding=1),
            nn.Conv2d(in_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, in_c, kernel_size=3, padding=1),
            nn.Conv2d(in_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, in_c, kernel_size=3, padding=1),
            nn.Conv2d(in_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc
class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
    def forward(self, x, fg):
        x = self.input_cbr(x)
        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)
        x_weighted_fg=x_weighted_fg.permute(0,3,1,2)
        out = self.output_cbr(x_weighted_fg)
        return out
    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn
    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted
class CDFAPreprocess(nn.Module):
    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))
    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x
class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, in_c, kernel_size=3, padding=1),
            nn.Conv2d(in_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, in_c, kernel_size=3, padding=1),
            nn.Conv2d(in_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, in_c, kernel_size=3, padding=1),
            nn.Conv2d(in_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            LayerNorm2d(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge
def adjust_feature(feature_a, feature_b):
    shape_a, shape_b = feature_a.size(), feature_b.size()
    assert shape_a[1] == shape_b[1], "特征图A和B的通道数必须相同"
    cosine_similarity = F.cosine_similarity(feature_a, feature_b, dim=1)
    cosine_similarity = cosine_similarity.unsqueeze(1)
    fused_feature = feature_a + feature_b * cosine_similarity
    fused_feature = F.relu(fused_feature)
    return fused_feature
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class SEM(nn.Module):
    def __init__(self, ch_out, reduction=None):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // reduction, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out // reduction, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))
if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
    except:
        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)
class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s
    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x
class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
        return att1, att2, att3, att4, att5
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_
class FEGNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, layer_scale_init_value=1e-6, gnconv=gnconv, block=Block,
                 pretrained=None,
                 use_checkpoint=False, c_list=[8, 16, 32, 64, 128, 256], depths=[2, 3, 18, 2], drop_path_rate=0.,
                 split_att='fc', bridge=True):
        super().__init__()
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.bridge = bridge
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        )
        self.encoder1_eem = EdgeEnhancer(c_list[0])
        self.encoder2_eem = EdgeEnhancer(c_list[1])
        self.encoder3_eem = EdgeEnhancer(c_list[2])
        self.encoder1_gbc = GBC(c_list[0])
        self.encoder2_gbc = GBC(c_list[1])
        self.encoder3_gbc = GBC(c_list[2])
        self.decoder1_eem = EdgeEnhancer(c_list[0])
        self.decoder2_eem = EdgeEnhancer(c_list[1])
        self.decoder3_eem = EdgeEnhancer(c_list[2])
        self.decoder1_gbc = GBC(c_list[0])
        self.decoder2_gbc = GBC(c_list[1])
        self.decoder3_gbc = GBC(c_list[2])
        self.decouplelayer=DecoupleLayer(c_list[3],c_list[3])
        self.cdfapreprocess_fg=CDFAPreprocess(c_list[3],c_list[4],1)
        self.cdfapreprocess_bg = CDFAPreprocess(c_list[3], c_list[4], 1)
        self.cdfa=ContrastDrivenFeatureAggregation(c_list[4], c_list[4], 1)
        self.auxiliaryhead=AuxiliaryHead(c_list[3])
        self.cbm=CBM(c_list[4])
        self.cbr64to128 = CBR(64, 128, kernel_size=1, stride=1, padding=0)
        self.twoencoder_mamba_ch16 = MobileMambaBlock(c_list[1])
        self.twoencoder_mamba_ch32 = MobileMambaBlock(c_list[2])
        self.twoencoder_mamba_ch64 = MobileMambaBlock(c_list[3])
        self.mamaba_conv=nn.Conv2d(3, c_list[1], 3, stride=2, padding=1)
        self.mamaba_ch64_32 = nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1)
        self.mamaba_ch16_32 = nn.Conv2d(c_list[1], c_list[2], 3, stride=2, padding=1)
        self.mamaba_ch32_64 = nn.Conv2d(c_list[2], c_list[3], 3, stride=2, padding=1)
        self.c128_32=nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.c32_16 = nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.c16_8 = nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.FF8 = FeatureFusion(8,num_heads=1,sr_ratio=4)
        self.FF16 = FeatureFusion(16,num_heads=1,sr_ratio=4)
        self.FF32=FeatureFusion(32,num_heads=1,sr_ratio=4)
        self.FF128 = FeatureFusion(128,num_heads=1,sr_ratio=4)
        self.FF256 = FeatureFusion(256, num_heads=1,sr_ratio=4)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.shuffle_groups = 4
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(c_list[2] * 2, c_list[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(c_list[2]),
            nn.ReLU(inplace=True)
        )
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3),
                      partial(gnconv, order=3, s=1 / 3),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4
        if isinstance(gnconv[0], str):
            gnconv = [eval(g) for g in gnconv]
        if isinstance(block, str):
            block = eval(block)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
        )
        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        def format_error(layer, issues):
            return (
                    f"🚨 发现无效的层配置！\n"
                    f"• 层类型: {layer.__class__.__name__}\n"
                    f"• 参数配置: in_channels={getattr(layer, 'in_channels', 'N/A')}, "
                    f"out_channels={getattr(layer, 'out_channels', 'N/A')}, "
                    f"kernel_size={getattr(layer, 'kernel_size', 'N/A')}, "
                    f"groups={getattr(layer, 'groups', 'N/A')}\n"
                    f"• 检测到的问题：\n   - " + "\n   - ".join(issues) + "\n"
                                                                         f"💡 解决方案建议：\n"
                                                                         f"   1. 检查模型定义中该层的参数\n"
                                                                         f"   2. 确保所有数值为正整数\n"
                                                                         f"   3. 验证 groups ≤ min(in_channels, out_channels)"
            )
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            issues = []
            if m.out_channels <= 0:
                issues.append(f"输出通道数必须 > 0 (当前: {m.out_channels})")
            if any(ks <= 0 for ks in m.kernel_size):
                issues.append(f"卷积核尺寸必须 > 0 (当前: {m.kernel_size})")
            if issues:
                raise ValueError(format_error(m, issues))
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            issues = []
            kernel_size = m.kernel_size
            if m.out_channels <= 0:
                issues.append(f"输出通道数必须 > 0 (当前: {m.out_channels})")
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            for i, ks in enumerate(kernel_size):
                if ks <= 0:
                    issues.append(f"卷积核维度 {i + 1} 的值必须 > 0 (当前: {ks})")
            if m.groups <= 0:
                issues.append(f"分组数必须 > 0 (当前: {m.groups})")
            elif m.groups > m.out_channels:
                issues.append(f"分组数({m.groups}) 不能超过输出通道数({m.out_channels})")
            if issues:
                raise ValueError(format_error(m, issues))
            try:
                fan_out = kernel_size[0] * kernel_size[1] * m.out_channels
                fan_out = fan_out // m.groups
                if fan_out <= 0:
                    raise ValueError()
            except Exception as e:
                raise ValueError(
                    format_error(m, [f"参数组合导致无效计算 (fan_out={fan_out})"]) +
                    f"\n计算过程: ({kernel_size[0]}×{kernel_size[1]}×{m.out_channels}) ÷ {m.groups} = {fan_out}"
                ) from e
            m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x_mamaba=self.mamaba_conv(x)
        x_mamaba=self.twoencoder_mamba_ch16(x_mamaba)
        x_mamaba=self.mamaba_ch16_32(x_mamaba)
        x_mamaba = F.gelu(self.ebn3(x_mamaba))
        x_mamaba=self.twoencoder_mamba_ch32(x_mamaba)
        x_mamaba = F.gelu(self.ebn3(x_mamaba))
        x_mamaba=self.mamaba_ch32_64(x_mamaba)
        x_mamaba = self.twoencoder_mamba_ch64(x_mamaba)
        x_mamaba = F.gelu(self.ebn4(x_mamaba))
        x_mamaba=self.mamaba_ch64_32(x_mamaba)
        x = self.encoder1(x)
        x_eem = self.encoder1_eem(x)
        x_gbc = self.encoder1_gbc(x)
        x_add = adjust_feature(x_eem, x_gbc)
        out = F.gelu(F.max_pool2d(self.ebn1(x_add), 2, 2))
        t1 = out
        x = self.encoder2(out)
        x_eem = self.encoder2_eem(x)
        x_gbc = self.encoder2_gbc(x)
        x_add = adjust_feature(x_eem, x_gbc)
        out = F.gelu(F.max_pool2d(self.ebn2(x_add), 2, 2))
        t2 = out
        x = self.encoder3(out)
        x_eem = self.encoder3_eem(x)
        x_gbc = self.encoder3_gbc(x)
        x_add = adjust_feature(x_eem, x_gbc)
        out = F.gelu(F.max_pool2d(self.ebn3(x_add), 2, 2))
        t3 = out
        torch.add(t3,x_mamaba)
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out
        f_fg,f_bg,f_uc=self.decouplelayer(t5)
        mask_fg,mask_bg,mask_uc=self.auxiliaryhead(f_fg,f_bg,f_uc)
        tiaoyue_f_fg=self.cdfapreprocess_fg(f_fg)
        suppressbg_minus_f_fg=tiaoyue_f_fg
        suppressbg_minus_f_fg=F.interpolate(suppressbg_minus_f_fg, scale_factor=0.5, mode='bilinear',
                      align_corners=True)
        t5=self.FF128(t5,suppressbg_minus_f_fg)
        suppressbg_minus_f_fg = F.interpolate(suppressbg_minus_f_fg, scale_factor=4, mode='bilinear',
                                              align_corners=True)
        suppressbg_minus_f_fg=self.c128_32(suppressbg_minus_f_fg)
        t3 = self.FF32(t3, suppressbg_minus_f_fg)
        suppressbg_minus_f_fg = F.interpolate(suppressbg_minus_f_fg, scale_factor=2, mode='bilinear',
                                              align_corners=True)
        suppressbg_minus_f_fg=self.c32_16(suppressbg_minus_f_fg)
        t2=self.FF16(t2,suppressbg_minus_f_fg)
        suppressbg_minus_f_fg = F.interpolate(suppressbg_minus_f_fg, scale_factor=2, mode='bilinear',
                                              align_corners=True)
        suppressbg_minus_f_fg=self.c16_8(suppressbg_minus_f_fg)
        t1 = self.FF8(t1, suppressbg_minus_f_fg)
        out = F.gelu(self.encoder6(out))
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out4 = torch.add(out4, t4)
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out3 = torch.add(out3, t3)
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out2 = torch.add(out2, t2)
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out1 = torch.add(out1, t1)
        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)
        return torch.sigmoid(out0),mask_fg,mask_bg,mask_uc
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x
