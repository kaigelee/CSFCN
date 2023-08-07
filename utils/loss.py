"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['ICNetLoss']

# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target) ##调用nn.CrossEntropyLoss
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, reduction='mean',thresh=0.9, min_kept=110000, use_weight=False, **kwargs):  # min_kept 可以改为  110000

    # def __init__(self, ignore_index=-1, reduction='mean',thresh=0.9, min_kept=131072, use_weight=False, **kwargs):110000
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,ignore_index=ignore_index)
        print(use_weight,min_kept)
    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)



class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=True, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index,**kwargs)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)




# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=True,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight=weight, ignore_index = ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, pred2, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncLoss(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.01, nclass=19, aux=True,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncLoss, self).__init__(weight=weight, ignore_index = ignore_index)
        self.nclass = nclass
        self.se_weight = se_weight
        self.bceloss = nn.BCELoss(weight)
        print(self.se_weight)

    def forward(self, se_pred,target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        # se_pred, target = tuple(inputs)
        se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(se_pred)
        loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
        return  self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect

class MixSoftmaxCrossEntropyOHEMEncLoss(OhemCrossEntropy2d):
    def __init__(self, aux=True, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMEncLoss, self).__init__(ignore_index=ignore_index,**kwargs)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)
        self.encloss = EncLoss(ignore_index=ignore_index,**kwargs)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMEncLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)-1):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMEncLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        loss += self.encloss(preds[-1],target) #+ self.encloss(preds[-2],target)
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)
#
# # reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
# class EncSuperLoss(OhemCrossEntropy2d):
#     """2D Cross Entropy Loss with SE Loss"""
#
#     def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=True,
#                  aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
#         super(EncSuperLoss, self).__init__(weight=weight, ignore_index = ignore_index)
#         self.se_loss = se_loss
#         self.aux = aux
#         self.nclass = nclass
#         self.se_weight = se_weight
#         self.aux_weight = aux_weight
#         self.bceloss = nn.BCELoss(weight)
#         self.Mse = nn.MSELoss()
#
#     def forward(self, *inputs):
#         preds, target = tuple(inputs)
#         inputs = tuple(list(preds) + [target])
#         if not self.se_loss and not self.aux:
#             return super(EncSuperLoss, self).forward(*inputs)
#         elif not self.se_loss:
#             pred1, pred2, target = tuple(inputs)
#             loss1 = super(EncSuperLoss, self).forward(pred1, target)
#             loss2 = super(EncSuperLoss, self).forward(pred2, target)
#             return dict(loss=loss1 + self.aux_weight * loss2)
#         elif not self.aux:
#             pred, se_pred, target = tuple(inputs)
#             se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
#             loss1 = super(EncSuperLoss, self).forward(pred, target)
#             loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
#             return dict(loss=loss1 + self.se_weight * loss2)
#         else:
#             pred1, pred2, pred3, pred4, se_pred, super_r, target, image = tuple(inputs)
#             se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
#             loss1 = super(EncSuperLoss, self).forward(pred1, target)
#             loss2 = super(EncSuperLoss, self).forward(pred2, target)
#             loss3 = super(EncSuperLoss, self).forward(pred3, target)
#             loss4 = super(EncSuperLoss, self).forward(pred4, target)
#             loss5 = self.bceloss(torch.sigmoid(se_pred), se_target)
#
#             loss6 = self.Mse(super_r, image)
#
#             return loss1 + self.aux_weight * loss2  + self.aux_weight * loss3  + self.aux_weight * loss4 + self.se_weight * loss5 + self.se_weight * loss6
#
#     @staticmethod
#     def _get_batch_label_vector(target, nclass):
#         # target is a 3D Variable BxHxW, output is 2D BxnClass
#         batch = target.size(0)
#         tvect = Variable(torch.zeros(batch, nclass))
#         for i in range(batch):
#             hist = torch.histc(target[i].cpu().data.float(),
#                                bins=nclass, min=0,
#                                max=nclass - 1)
#             vect = hist > 0
#             tvect[i] = vect
#         return tvect
