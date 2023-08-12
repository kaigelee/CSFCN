# many are borrowed from https://github.com/orsic/swiftnet
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.optim as optim
from pathlib import Path
import numpy as np
import os


scale = 1
mean = [73.15, 82.90, 72.3]
std = [47.67, 48.49, 47.73]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from math import log2

import torch
import torch.nn.functional as F
import warnings

from torch import nn as nn

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
batchnorm_momentum = 0.01 / 2


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super(_BNReluConv, self).__init__()
        # if batch_norm:
        #     self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if drop_rate > 0:
            warnings.warn(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        warnings.warn(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        warnings.warn(f'Using skip connections: {self.use_skip}', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x, skip):
        if self.detach_skip:
            warnings.warn(f'Detaching skip connection {skip.shape[2:4]}', UserWarning)
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = self.upsampling_method(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

__all__ = ['ResNet', 'resnet18', 'resnet18dws', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet18dws': '/home/morsic/saves/imagenet/resnet18dws/model_best.pth.tar',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, separable=False):
    """3x3 convolution with padding"""
    conv_class = SeparableConv2d if separable else nn.Conv2d
    return conv_class(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = conv(x)
        # if norm is not None:
        #     x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function

def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, deleting=False,
                 separable=False):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, separable=separable)
        self.bn1 = None #nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, separable=separable)
        self.bn2 = None # nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.deleting = deleting

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.deleting is False:
            bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
            bn_2 = _bn_function_factory(self.conv2, self.bn2)

            out = do_efficient_fwd(bn_1, x, self.efficient)
            out = do_efficient_fwd(bn_2, out, self.efficient)
        else:
            out = torch.zeros_like(residual)

        out = out + residual
        relu = self.relu(out)
        # print(f'Basic Block memory: {torch.cuda.memory_allocated() // 2**20}')

        return relu, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, separable=False):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = None #nn.BatchNorm2d(planes) if self.use_bn else None
        conv_class = SeparableConv2d if separable else nn.Conv2d
        self.conv2 = conv_class(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = None #nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = None # nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = _bn_function_factory(self.conv3, self.bn3, self.relu)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=False, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, spp_drop_rate=0.0,
                 upsample_skip=True, upsample_only_skip=False,
                 detach_upsample_skips=(), detach_upsample_in=False,
                 target_size=None, output_stride=4, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192), scale=1, separable=False,
                 upsample_separable=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.separable = separable
        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        if scale != 1:
            self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.detach_upsample_in = detach_upsample_in
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = None # nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 6)]
        else:
            target_sizes = [None] * 4
        upsamples = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=2 in detach_upsample_skips, fixed_size=target_sizes[0],
                      separable=upsample_separable)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=1 in detach_upsample_skips, fixed_size=target_sizes[1],
                      separable=upsample_separable)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
                      separable=upsample_separable)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        # if self.use_bn:
        #     self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = kwargs.get('spp_size', num_features)
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn, drop_rate=spp_drop_rate
                                         , fixed_size=target_sizes[3])
        num_up_remove = max(0, int(log2(output_stride) - 2))
        self.upsample = nn.ModuleList(list(reversed(upsamples[num_up_remove:])))

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            # if self.use_bn:
            #     layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn,
                        separable=self.separable)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn,
                             separable=self.separable)]

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        # if hasattr(self, 'img_scale'):
        #     image /= self.img_scale
        # image -= self.img_mean
        # image /= self.img_std

        x = self.conv1(image)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        return features

    def forward_up(self, features):
        features = features[::-1]

        x = features[0]
        if self.detach_upsample_in:
            x = x.detach()

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        return self.forward_up(self.forward_down(image))


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet18dws(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], separable=True, **kwargs)
    if pretrained:
        try:
            model.load_state_dict(torch.load(model_urls['resnet18dws'])['state_dict'], strict=True)
        except Exception as e:
            print(e)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import warnings



class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, num_inst_classes=None, use_bn=True, k=1, bias=True,
                 loss_ret_additional=False, upsample_logits=True, logit_class=_BNReluConv,
                 multiscale_factors=(.5, .75, 1.5, 2.)):
        super(SemsegModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = logit_class(self.backbone.num_features, self.num_classes, batch_norm=use_bn, k=k, bias=bias)
        if num_inst_classes is not None:
            self.border_logits = _BNReluConv(self.backbone.num_features, num_inst_classes, batch_norm=use_bn,
                                             k=k, bias=bias)
        self.criterion = None
        self.loss_ret_additional = loss_ret_additional
        self.img_req_grad = loss_ret_additional
        self.upsample_logits = upsample_logits
        self.multiscale_factors = multiscale_factors

    def forward(self, image, target_size=(1024, 2048), image_size=(1024, 2048)):
        features, _ = self.backbone(image)
        logits = self.logits.forward(features)
        if (not self.training) or self.upsample_logits:
            logits = upsample(logits, image_size)
        # if hasattr(self, 'border_logits'):
        #     additional['border_logits'] = self.border_logits(features).sigmoid()
        # additional['logits'] = logits
        return logits # , additional

    def forward_down(self, image, target_size, image_size):
        return self.backbone.forward_down(image), target_size, image_size

    def forward_up(self, feats, target_size, image_size):
        feats, additional = self.backbone.forward_up(feats)
        features = upsample(feats, target_size)
        logits = self.logits.forward(features)
        logits = upsample(logits, image_size)
        return logits, additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda'), img_key='image'):
        if image_size is None:
            image_size = batch['target_size']
        warnings.warn(f'Image requires grad: {self.img_req_grad}', UserWarning)
        image = batch[img_key].detach().requires_grad_(self.img_req_grad).to(device)
        return {
            'image': image,
            'image_size': image_size,
            'target_size': batch.get('target_size_feats')
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        logits, additional = self.forward(**data)
        additional['model'] = self
        additional = {**additional, **data}
        return logits, additional

    def loss(self, batch):
        assert self.criterion is not None
        labels = batch['labels'].cuda()
        logits, additional = self.do_forward(batch, image_size=labels.shape[-2:])
        if self.loss_ret_additional:
            return self.criterion(logits, labels, batch=batch, additional=additional), additional
        return self.criterion(logits, labels, batch=batch, additional=additional)

    def random_init_params(self):
        params = [self.logits.parameters(), self.backbone.random_init_params()]
        if hasattr(self, 'border_logits'):
            params += [self.border_logits.parameters()]
        return chain(*(params))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def ms_forward(self, batch, image_size=None):
        image_size = batch.get('target_size', image_size if image_size is not None else batch['image'].shape[-2:])
        ms_logits = None
        pyramid = [batch['image'].cuda()]
        pyramid += [
            F.interpolate(pyramid[0], scale_factor=sf, mode=self.backbone.pyramid_subsample,
                          align_corners=self.backbone.align_corners) for sf in self.multiscale_factors
        ]
        for image in pyramid:
            batch['image'] = image
            logits, additional = self.do_forward(batch, image_size=image_size)
            if ms_logits is None:
                ms_logits = torch.zeros(logits.size()).to(logits.device)
            ms_logits += F.softmax(logits, dim=1)
        batch['image'] = pyramid[0].cpu()
        return ms_logits / len(pyramid), {}


def SwiftNet():

    resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
    model = SemsegModel(resnet, num_classes = 19)

    return model


if __name__ == '__main__':
    import numpy as np
    import random,time
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    random.seed(123)

    device = torch.device('cuda')
    model = SwiftNet()
    model.eval()
    model.to(device)
    print(model)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(20):
            model(input)

        # if iterations is None:
        #     elapsed_time = 0
        #     iterations = 100
        #     while elapsed_time < 1:
        #         torch.cuda.synchronize()
        #         torch.cuda.synchronize()
        #         for _ in range(iterations):
        #             t_start = time.time()
        #
        #             model(input)
        #             print(time.time()-t_start)
        #             torch.cuda.synchronize()
        #
        #         torch.cuda.synchronize()
        #         elapsed_time = time.time() - t_start
        #         iterations *= 2
        #     FPS = iterations / elapsed_time
        #     print(FPS)
        #     iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        iterations = 1000

        print(iterations)
        import torch.nn.functional as F
        torch.cuda.synchronize()
        # torch.cuda.synchronize()
        # i = 0
        t_start = time.time()

        for _ in range(iterations):
            out = model(input)
            # out = F.interpolate(out,size=(1024,2048))

        torch.cuda.synchronize()
        # torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
