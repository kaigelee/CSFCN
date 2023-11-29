#!/usr/bin/python
# -*- encoding: utf-8 -*-
# ------------------------------------------------------------------------------
# Written by Kaige Li (kglee1994@163.com)
# ------------------------------------------------------------------------------

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.base_models.resnetv1c import resnet18

from torch.nn import BatchNorm2d

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class Backbone(nn.Module):
    def __init__(self, backbone='resnet18', pretrained_base=True, norm_layer=nn.BatchNorm2d):
        super(Backbone, self).__init__()
        if backbone == 'resnet18':
            pretrained = resnet18(pretrained=pretrained_base)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.conv1 = pretrained.conv1
        self.bn1  = pretrained.bn1
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x,True)
        x = self.maxpool(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c2,  c3, c4



class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels
        print(self.grids)
    def forward(self, feats):

        b, c , h , w = feats.size()

        ar = w / h

        return torch.cat([
        F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
        F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
        F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
        F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ],dim=2)

class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256,inter_channels=32):
        super(LocalAttenModule, self).__init__()

        print('sigmoid')
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, inter_channels,1,1,0),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False)  )

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)

        res1 = res1 * x_mask

        return res1 + res2



class CFC_CRB(nn.Module):
    def __init__(self, in_channels = 512 , inter_channels = 256, grids=(6, 3, 2, 1)): # 先ce后ffm
        super(CFC_CRB, self).__init__()
        self.grids = grids
        self.inter_channels = inter_channels

        self.reduce_channel = ConvBNReLU(in_channels, inter_channels,3,1,1)

        self.query_conv = nn.Conv2d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.key_channels = 32

        self.value_psp = PSPModule(grids,256)
        self.key_psp = PSPModule(grids,256)

        self.softmax = nn.Softmax(dim=-1)

        self.local_attention = LocalAttenModule(inter_channels,inter_channels//8)
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):

        x = self.reduce_channel(x) # 降维- 128

        m_batchsize,_,h,w = x.size()

        query = self.query_conv(x).view(m_batchsize,32,-1).permute(0,2,1) ##  b c n ->  b n c

        key = self.key_conv(self.key_psp(x))  ## b c s


        sim_map = torch.matmul(query,key)

        sim_map = self.softmax(sim_map)
        # sim_map = self.attn_drop(sim_map)
        value = self.value_conv(self.value_psp(x)) #.permute(0,2,1)  ## b c s

        # context = torch.matmul(sim_map,value) ## B N S * B S C ->  B N C
        context = torch.bmm(value,sim_map.permute(0,2,1))  #  B C S * B S N - >  B C N

        # context = context.permute(0,2,1).view(m_batchsize,self.inter_channels,h,w)
        context = context.view(m_batchsize,self.inter_channels,h,w)
        # out = x + self.gamma * context
        context = self.local_attention(context)

        out = x + context


        return out


class SFC_G2(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SFC_G2, self).__init__()

        self.conv_8 = ConvBNReLU(128,128,3,1,1)

        self.conv_32 = ConvBNReLU(256,128,3,1,1)


        self.groups = 2

        print('groups',self.groups)

        self.conv_offset = nn.Sequential(
            ConvBNReLU(256,64,1,1,0),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)  )

        self.keras_init_weight()

        self.conv_offset[1].weight.data.zero_()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self, cp,sp):

        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征  1 / 8  256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)
        # x_8
        cp = self.conv_8(cp)

        conv_results = self.conv_offset(torch.cat([cp, sp], 1))

        sp = sp.reshape(n*self.groups,-1,out_h,out_w)
        cp = cp.reshape(n*self.groups,-1,out_h,out_w)

        offset_l = conv_results[:, 0:self.groups*2, :, :].reshape(n*self.groups,-1,out_h,out_w)
        offset_h = conv_results[:, self.groups*2:self.groups*4, :, :].reshape(n*self.groups,-1,out_h,out_w)


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n*self.groups, 1, 1, 1).type_as(sp).to(sp.device)


        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm


        cp = F.grid_sample(cp, grid_l , align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h , align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups*4:, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=self.up_factor, mode='bilinear', align_corners=True)
        return x


class CSFCN(nn.Module):
    def __init__(self, num_classes=19, augment=False, *args, **kwargs):

        super(CSFCN, self).__init__()

        self.resnet = Backbone()

        self.cfc = CFC_CRB()

        self.sfc = SFC_G2()

        self.conv_out32 = BiSeNetOutput(128, 128, num_classes, up_factor=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        feat8,_ , feat32 = self.resnet(x)

        f = self.cfc(feat32)

        f = self.sfc(feat8, f)

        out = self.conv_out32(f)

        return out


def get_seg_model(cfg, imgnet_pretrained=True):
    model = CSFCN(num_classes=cfg.DATASET.NUM_CLASSES, augment=True)

    return model


def get_pred_model(name, num_classes):
    model = CSFCN(num_classes=num_classes, augment=False)

    return model


if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='CSFCN', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
