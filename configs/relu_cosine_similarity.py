import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
import math
from user1.gbc.GBC import *
import traceback
from scipy.ndimage import gaussian_filter, laplace
import numpy as np


# 用余弦相似度进行特征融合时，对cosine_similiaty 进行了relu操作保证其为正值

def adjust_feature(feature_a, feature_b):
    """
    基于余弦相似度进行特征融合

    参数:
        feature_a: 特征图A, 形状为 [batch_size, channels, height, width]
        feature_b: 特征图B, 形状为 [batch_size, channels, height, width]

    返回:
        融合后的特征图, 与输入特征图形状相同

    功能描述:
        1. 计算特征图A和特征图B在通道维度的余弦相似度
        2. 将该相似度结果作为权重作用于特征图B
        3. 将加权后的特征图B与特征图A相加
        4. 对结果应用ReLU激活函数
    """
    # 获取形状信息并验证通道数一致性
    shape_a, shape_b = feature_a.size(), feature_b.size()
    assert shape_a[1] == shape_b[1], "特征图A和B的通道数必须相同"

    # 计算通道维度上的余弦相似度
    cosine_similarity = F.cosine_similarity(feature_a, feature_b, dim=1)

    # 增加维度以便进行乘法运算
    cosine_similarity = F.relu(cosine_similarity.unsqueeze(1))

    # 特征融合: feature_a + (feature_b * cosine_similarity)
    fused_feature = feature_a + feature_b * cosine_similarity

    # # 应用ReLU激活函数
    # fused_feature = fused_feature
    fused_feature = fused_feature

    return fused_feature


# 下面是边缘特征提取，上面是余弦相似度向加
def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:, :, int((kernelsize - 1) / 2), int((kernelsize - 1) / 2)] = kernelvalue
    g = gaussian_filter(n, sigma)
    gaussiankernel = torch.from_numpy(g)

    return gaussiankernel.float()


def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:, :, int((kernelsize - 1) / 2), int((kernelsize - 1) / 2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)

    return laplacekernel.float()


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


class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, groups, reduction):
        super(EEM, self).__init__()

        self.groups = groups
        self.gk = gaussiankernel(ch_in, int(ch_in / groups), kernel, kernel - 2, 0.9)
        self.lk = laplaceiankernel(ch_in, int(ch_in / groups), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out / 2), kernel_size=1, padding=0, groups=2),
            nn.PReLU(num_parameters=int(ch_out / 2), init=0.05),
            nn.InstanceNorm2d(int(ch_out / 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out / 2), kernel_size=1, padding=0, groups=2),
            nn.PReLU(num_parameters=int(ch_out / 2), init=0.05),
            nn.InstanceNorm2d(int(ch_out / 2))
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(int(ch_out / 2), ch_out, kernel_size=1, padding=0, groups=2),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out)
        )

        self.sem1 = SEM(ch_out, reduction=reduction)  # let's put reduction as 2
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)

    def forward(self, x):
        DoG = F.conv2d(x, self.gk.to('cuda'), padding='same', groups=self.groups)
        LoG = F.conv2d(DoG, self.lk.to('cuda'), padding='same', groups=self.groups)
        DoG = self.conv1(DoG - x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG * LoG)

        tot1 = self.sem1(tot)
        x1 = self.sem2(x)

        return self.prelu(x + x1 + tot + tot1)


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
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())


        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


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

        print('[gconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

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
    r""" Horblock
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
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
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

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


class MHorunet(nn.Module):

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
        self.encoder1_eem = EEM(c_list[0], c_list[0], 3, 1, 2)
        self.encoder2_eem = EEM(c_list[1], c_list[1], 3, 1, 2)
        self.encoder3_eem = EEM(c_list[2], c_list[2], 3, 1, 2)
        self.encoder1_gbc = GBC(c_list[0])
        self.encoder2_gbc = GBC(c_list[1])
        self.encoder3_gbc = GBC(c_list[2])
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
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
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        # c_list = [8, 16, 32, 64, 128, 256]
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1)
        )

        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1)
        )

        self.encoder6 = nn.Sequential(
            GBC(c_list[4]),
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
            GBC(c_list[4])
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
            GBC(c_list[3])
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
            GBC(c_list[2])
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            GBC(c_list[1])
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
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder1(x)
        x_eem = self.encoder1_eem(x)
        x_gbc = self.encoder1_gbc(x)
        x_add = adjust_feature(x_eem, x_gbc)
        out = F.gelu(F.max_pool2d(self.ebn1(x_add), 2, 2))
        t1 = out  # b, c0, H/2, W/2
        x = self.encoder2(out)
        x_eem = self.encoder2_eem(x)
        x_gbc = self.encoder2_gbc(x)
        x_add = adjust_feature(x_eem, x_gbc)
        out = F.gelu(F.max_pool2d(self.ebn2(x_add), 2, 2))
        t2 = out  # b, c1, H/4, W/4
        x = self.encoder3(out)
        x_eem = self.encoder3_eem(x)
        x_gbc = self.encoder3_gbc(x)
        x_add = adjust_feature(x_eem, x_gbc)
        out = F.gelu(F.max_pool2d(self.ebn3(x_add), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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
