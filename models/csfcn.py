# ------------------------------------------------------------------------------
# Written by Kaige Li (kglee1994@163.com)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

## TODO


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

        feat8, feat32 = self.resnet(x)

        f = self.cfc(feat32)

        f = self.sfc( feat8 , f)

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
    model = get_pred_model(name='SANet', num_classes=19)
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