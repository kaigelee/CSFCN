# ------------------------------------------------------------------------------
# Written by Kaige Li (kglee1994@163.com)
# ------------------------------------------------------------------------------

#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import BatchNorm2d
import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y

class spatial_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(spatial_layer, self).__init__()


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y

class SCA(nn.Module):

    def __init__(self, in_chan, k_size,  *args, **kwargs):
        super(SCA, self).__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(in_chan,
                in_chan,
                kernel_size = 1,
                bias = False),
            nn.Sigmoid()
        )

        self.channel = eca_layer(k_size)

        self.keras_init_weight()

    '''
    64   
    128 
    256 
    512 
    '''
    def forward(self, x):

        channel = self.channel(x)
        spatial = self.spatial(x)

        return x * spatial + x * channel

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params



class CBR(nn.Module):
    """
    1*1 Convolution Block
    """
    def __init__(self, in_ch, out_ch,kernel_size=1,stride=1,padding=0):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d,nn.Conv1d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
            elif isinstance(module, nn.modules.normalization.LayerNorm):
                nowd_params += list(module.parameters())
            # elif isinstance(module, GeneralizedMeanPooling):
            #     wd_params += list(module.get_params())
        return wd_params, nowd_params


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18_v1c']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_stem=True,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        ksize = [3, 3, 3, 5]
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], 3, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], 3, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], 3, stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], 5, stride=2, norm_layer=norm_layer)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, k_size, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 64 != 64 * 4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if stride==2: layers.append(SCA(self.inplanes,k_size))
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet18_v1c(pretrained=True, **kwargs):
    model = ResNet(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    return model


class SegHead(nn.Module):
    def __init__(self, in_channels, inter_channels,  n_classes=19, up_factor=4, *args, **kwargs):

        super(SegHead, self).__init__()
        self.block = nn.Sequential(
            CBR(in_channels, inter_channels,1,1,0),
            nn.Conv2d(inter_channels, n_classes, 3, stride=1, padding=1,bias=False)
        )
        self.scale = up_factor
    def forward(self, x):
        x = self.block(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)

        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class Backbone(nn.Module):
    def __init__(self, backbone='resnet18', pretrained_base=True, norm_layer=nn.BatchNorm2d):
        super(Backbone, self).__init__()
        if backbone == 'resnet18':
            pretrained = resnet18_v1c(pretrained=pretrained_base)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.stem = pretrained.conv1
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4
        self.cbr1 = CBR(128,128,3,1,1)
        self.cbr2 = CBR(256,128,3,1,1)
        self.cbr3 = CBR(512,128,3,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, self.cbr1(c2), self.cbr2(c3), self.cbr3(c4),self.gap(c4)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params






class FAA(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):

        super(FAA, self).__init__()

        self.offset = nn.Conv2d(in_channels * 2, 2, 3, stride=1, padding=1,bias=False)

        if in_channels < 512:
            self.sca = SCA(in_channels,3)
        else:
            self.sca = SCA(in_channels,5)

    def forward(self, fh, fl):

        n, _, out_h, out_w = fh.size()


        x = torch.cat([fh,fl],dim=1)

        offset = self.offset(x)


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(fh).to(fh.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(fh).to(fh.device)

        grid = grid + offset.permute(0, 2, 3, 1) / norm

        fl = F.grid_sample(fl, grid , align_corners=True)  ## 考虑是否指定align_corners

        fh = self.sca(fh + fl)

        return fh



class FEB_4(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):

        super(FEB_4, self).__init__()

        self.dw_conv_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=2,
                                 dilation=2, groups=in_channels,bias=False)

        self.bn_1 = nn.BatchNorm2d(in_channels)

        self.dw_conv_2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=5,
                                 dilation=5, groups=in_channels,bias=False)

        self.conv_1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

        self.bn_2 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(True)

        if in_channels < 512:
            self.sca = SCA(in_channels,3)
        else:
            self.sca = SCA(in_channels,5)


    def forward(self, x ):
        res = x
        x1 = self.dw_conv_1(x)
        x2 = self.bn_1(x1)
        x2 = self.dw_conv_2(x2)
        x = torch.cat([x1,x2],dim=1)

        x = self.conv_1x1(x)
        x = self.bn_2(x)
        x = self.relu(res + x)


        return self.sca(x)


class FEB_3(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(FEB_3, self).__init__()

        self.dw_conv_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1,
                                   dilation=1, groups=in_channels, bias=False)

        self.bn_1 = nn.BatchNorm2d(in_channels)

        self.dw_conv_2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1,
                                   dilation=1, groups=in_channels, bias=False)

        self.conv_1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

        self.bn_2 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(True)

        if in_channels < 512:
            self.sca = SCA(in_channels,3)
        else:
            self.sca = SCA(in_channels,5)

    def forward(self, x):
        res = x
        x1 = self.dw_conv_1(x)
        x2 = self.bn_1(x1)
        x2 = self.dw_conv_2(x2)
        x = torch.cat([x1, x2], dim=1)

        x = self.conv_1x1(x)
        x = self.bn_2(x)
        x = self.relu(res + x)

        return self.sca(x)



class FEB_2(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(FEB_2, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1,
                                    bias=False)

        self.bn_1 = nn.BatchNorm2d(in_channels)

        self.conv_2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1,
                                    bias=False)

        self.bn_2 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(True)

        if in_channels < 512:
            self.sca = SCA(in_channels,3)
        else:
            self.sca = SCA(in_channels,5)


    def forward(self, x):
        res = x

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(res + x)

        return self.sca(x)



class FEB_1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FEB_1, self).__init__()

    def forward(self, x):

        return x



class SFA(nn.Module):
    def __init__(self, cbr_in_channels,feb_in_channels,faa_in_channels,
                 inter_channels, block, *args, **kwargs):

        super(SFA, self).__init__()

        # if cbr_in_channels ==512:
        #     self.cbr = CBR(cbr_in_channels, inter_channels,1,1,0)
        # else:
        #     self.cbr = CBR(cbr_in_channels, inter_channels,3,1,1)


        self.cbr = CBR(cbr_in_channels, inter_channels,1,1,0)

        self.feb = block(feb_in_channels)

        self.faa = FAA(faa_in_channels)

    def forward(self, fh,fl):

        fl = self.cbr(fl)
        fl = F.interpolate(fl, size=fh.size()[2:], mode='bilinear', align_corners=True)

        fh = self.feb(fh)

        fh = self.faa(fh, fl)

        return fh

class SFANet(nn.Module):
    '''
        Stage-Aware Feature Alignment Network for Real-Time Semantic Segmentation of Street Scenes
    '''

    def __init__(self, n_classes=19, output_aux=False, *args, **kwargs):
        super(SFANet, self).__init__()
        self.resnet = Backbone()

        self.sfa4 = SFA(512,128,128,128,block=FEB_4)
        self.sfa3 = SFA(128,128,128,128,block=FEB_3)
        self.sfa2 = SFA(128,128,128,128,block=FEB_2)
        self.sfa1 = SFA(128,64,64,64,block=FEB_1)

        self.conv_out = SegHead(128, 64, n_classes, up_factor=4)

        self.output_aux = output_aux
        # if self.output_aux:
        #     self.conv_out16 = SegHead(256, 64, n_classes, up_factor=16)

    def forward(self, x):
        # 128
        feat4,feat8,dsn,feat32,gap = self.resnet(x)

        f = self.sfa4(feat32,gap)
        f = self.sfa3(dsn,f)
        f = self.sfa2(feat8,f)
        f = self.sfa1(feat4,f)

        f = torch.cat([feat4,f],dim=1)
        f = self.conv_out(f)

        return f
        # if self.output_aux:
        #     aux1 = self.conv_out16(dsn)
        #     return f, aux1
        #
        # f = f.argmax(dim=1)
        # return f



    # def get_params(self):
    #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
    #     for name, child in self.named_children():
    #         #child为自定义的各个模块的名称 如 acm
    #         child_wd_params, child_nowd_params = child.get_params()
    #         if isinstance(child, (BiSeNetOutput)):
    #             lr_mul_wd_params += child_wd_params
    #             lr_mul_nowd_params += child_nowd_params
    #         else:
    #             wd_params += child_wd_params
    #             nowd_params += child_nowd_params
    #     return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



if __name__ == "__main__":
    import numpy as np
    import math
    a = math.log2(128)
    b = np.array([6,7,8,9]) / 2 + 1/2

    print(b) # 3335

    model = SFANet()
    model.eval()
    print(model)

    input = torch.rand(1,3,1024,2048)

    out = model(input)

    print(out.shape)


    from thop import profile, clever_format

    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.4f")
    print('flops,', macs, 'params', params)
