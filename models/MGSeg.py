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

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return  c2, c3, c4

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


from torch import nn


class CBR(nn.Module):
    """
    1*1 Convolution Block
    """
    def __init__(self, in_ch, out_ch, kernel_size=1,stride=1,padding=0):
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


class S_LWA(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(S_LWA, self).__init__()

        self.reduce = CBR(in_channel,out_channel,3,1,1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channel, out_channel, bias=False),
            nn.Sigmoid()
        )
        self.init_weight()

    def forward(self, x):
        x = self.reduce(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x

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


class LWA(nn.Module):
    def __init__(self, in_channel,out_channel, reduction=2):
        super(LWA, self).__init__()

        self.reduce = CBR(in_channel,out_channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channel, out_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // reduction, out_channel, bias=False),
            nn.Sigmoid()
        )

        self.init_weight()

    def forward(self, x):
        x = self.reduce(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x

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

class HAFA(nn.Module):
    def __init__(self):
        super(HAFA, self).__init__()

        self.q5_slwa = S_LWA(512,128) # 512
        self.q5_conv = nn.Conv2d(128, 128, kernel_size=3,
                  stride=1, padding=1, bias=False)


        self.q4_slwa = S_LWA(256,128) # 256
        self.q4_conv = nn.Conv2d(128, 128, kernel_size=3,
                  stride=1, padding=1, bias=False)

        self.q3_lwa = LWA(256,128)    # 128

        self.q3_conv = nn.Conv2d(128, 128, kernel_size=3,
                                 stride=1, padding=1, bias=False)


        self.p1_conv = nn.Conv2d(128, 32, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.p1_1_conv = nn.Conv2d(32, 32, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(128, 32, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.p2_1_conv = nn.Conv2d(32, 32, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(128, 32, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.p3_1_conv = nn.Conv2d(32, 32, kernel_size=3,
                                 stride=1, padding=1, bias=False)

        self.predictor = nn.Conv2d(32, 19, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.init_weight()

    def forward(self, q3, q4, q5):

        f3 = self.q5_slwa(q5)
        f3 = F.interpolate(f3,scale_factor=2,mode='bilinear',align_corners=True) # 1/16
        p3 = self.q5_conv(f3)

        f2 = self.q4_slwa(q4)
        q4 = F.interpolate(p3 + f2,scale_factor=2,mode='bilinear',align_corners=True) # 1/16
        p2 = self.q4_conv(q4)

        f1 = self.q3_lwa(torch.cat([q3,p2],dim=1))

        p1 = self.q3_conv(f1 + p2)

        p3 = self.p3_conv(p3)
        p2 = self.p2_conv(p2)
        p1 = self.p1_conv(p1)

        p1 = self.p1_1_conv(p1)
        p2 = self.p2_1_conv(p1 + p2)
        p2_down = F.interpolate(p2,scale_factor=0.5,mode='bilinear',align_corners=True)

        p3 = self.p3_1_conv(p2_down + p3)
        p3 = F.interpolate(p3,scale_factor=2,mode='bilinear',align_corners=True) # 1/16

        out = self.predictor(p3 + p2 + p1)

        return f3,f2,f1,out

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


class GWL(nn.Module):
    def __init__(self, in_channel=512):
        super(GWL, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.Conv2d(128, 152, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.GC_conv = nn.Conv2d(152, 19 * 4, kernel_size=3,
                      stride=1, padding=1, bias=False, groups=19)
        self.init_weight()

    def forward(self, q5):

        q5 = self.conv_1(q5) # 1/32
        q5 = F.interpolate(q5,scale_factor=2,mode='bilinear',align_corners=True) # 1/16
        q5 = self.GC_conv(q5) # 1/16
        q5 = F.interpolate(q5,scale_factor=2,mode='bilinear',align_corners=True) # 1/8  --> b 19 h w

        return q5

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


class FGR(nn.Module):
    def __init__(self):
        super(FGR, self).__init__()

        self.gwl = GWL()
        self.conv_1 = nn.Conv2d(128, 1, kernel_size=3,
                      stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(128, 1, kernel_size=3,
                      stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(128, 1, kernel_size=3,
                      stride=1, padding=1, bias=False)


        self.init_weight()

    def forward(self, q5, f3,f2,f1,pred):

        weight = self.gwl(q5) # 1, 19*4, h w # 1/32
        b,_,h,w = weight.size()

        size = f1.size()[2:]
        f3 = F.interpolate(f3,size=size,mode='bilinear',align_corners=True) # 1/16
        f2 = F.interpolate(f2,size=size,mode='bilinear',align_corners=True) # 1/16

        a1 = self.conv_1(f3)
        a2 = self.conv_2(f2)
        a3 = self.conv_3(f1)

        a = torch.cat([a1,a2,a3],dim=1)
        a = a.unsqueeze(1).repeat(1, 19, 1, 1, 1) # B K 3 H W
        pred = pred.unsqueeze(2)                 # B K 1 H W
        g = torch.cat([a, pred], dim=2)           # b k m h w
        weight = weight.reshape(b,19,4,h,w)  # b k m h w
        h = g * weight
        h = torch.sum(h,dim=2) # bkhw

        return h

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



class SegHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(SegHead, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes * up_factor * up_factor
        self.conv = CBR(in_chan, mid_chan,3,1,1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.PixelShuffle(up_factor)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
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

class MGSeg(nn.Module):
    '''
        MGSeg: Multiple Granularity-Based Real-Time Semantic Segmentation Network
    '''

    def __init__(self, n_classes=19, output_aux=True, *args, **kwargs):
        super(MGSeg, self).__init__()
        self.resnet = Backbone()
        self.hafa = HAFA()
        self.fgr = FGR()

        self.output_aux = output_aux
        if self.output_aux:
            self.conv_out16 = SegHead(256, 64, n_classes, up_factor=16)


    def forward(self, x):
        # 128
        feat8,dsn,feat32 = self.resnet(x)

        f3,f2,f1,pred = self.hafa(feat8,dsn,feat32) # 1 / 8

        # q5, f3, f2, f1, pred
        h = self.fgr(feat32, f3, f2, f1, pred)

        h = F.interpolate(h,scale_factor=8,mode='bilinear',align_corners=True) # 1/8

        if self.output_aux:
            aux1 = self.conv_out16(dsn)
            pred = F.interpolate(pred, scale_factor=8, mode='bilinear', align_corners=True)  # 1/8
            return h, pred, aux1

        h = h.argmax(dim=1)
        return h



    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            #child为自定义的各个模块的名称 如 acm
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (HAFA,FGR)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



if __name__ == "__main__":
    import numpy as np
    import math
    a = math.log2(128)
    b = np.array([6,7,8,9]) / 2 + 1/2

    print(b) # 3335

    model = MGSeg(output_aux=True)
    model.eval()
    print(model)

    input = torch.rand(1,3,1024,2048)

    out = model(input)

    print(out[2].shape)

    from thop import profile, clever_format

    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.4f")
    print('flops,', macs, 'params', params)
