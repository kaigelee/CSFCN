"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import os
import torch
__all__ = ['SegmentationDataset','CamVid']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=[960,720]):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.color_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,hue=0.1)
        # self.color_aug = transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1,hue=0.1)
        print('0.125 - aug 0.1')

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, img, mask):
        # outsize = self.crop_size
        # short_size = outsize
        # w, h = img.size
        # if w > h:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)

        # 960 720
        # 960 704

        # img = img.resize((928, 704), Image.BILINEAR)
        # mask = mask.resize((928, 704), Image.NEAREST)
        # # center crop
        # w, h = img.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # img = img.crop((x1, y1, x1+outsize, y1+outsize))
        # mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):

        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random scale (short edge)
        w, h = img.size  # 960 720
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size  ## 960 * 0.5-2
            oh = int(1.0 * h * long_size / w + 0.5)  #
            short_size = oh
        # print('--',ow,oh,'--') # 1073 805
        # print('--',long_size,short_size,'--') # 1073 805
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # crop_size 960 704
        # pad crop
        if short_size < self.crop_size[1] or long_size < self.crop_size[0]:
            padh = self.crop_size[1] - oh if oh < self.crop_size[1] else 0
            padw = self.crop_size[0] - ow if ow < self.crop_size[0] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size  # 1073 805
        # print('***',w,h)

        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])

        # print('000',w - self.crop_size[0],h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))



        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        img = self.color_aug(img)


        # final transform
        return img, self._mask_transform(mask)
    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')
from torchvision import transforms

class CamVid(SegmentationDataset):
    NUM_CLASS = 11
    IGNORE_INDEX= -1
    NAME = "CamVid"
    BASE_DIR = 'CamVid'

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    def __init__(self, root='/home/lkg/PycharmProjects', split='train',
                 mode=None, transform=input_transform, **kwargs):

        super(CamVid, self).__init__(
            root, split, mode, transform,  **kwargs)
        # assert exists and prepare dataset automatically
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please setup the dataset using" + \
            "encoding/scripts/prepare_ade20k.py"
        self.images, self.masks = _get_camvid_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask[mask==11] = -1
        return img, mask,os.path.basename(self.images[index])

    #def _sync_transform(self, img, mask):
    #    # random mirror
    #    if random.random() < 0.5:
    #        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #    crop_size = self.crop_size
    #    # random scale (short edge)
    #    w, h = img.size
    #    long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
    #    if h > w:
    #        oh = long_size
    #        ow = int(1.0 * w * long_size / h + 0.5)
    #        short_size = ow
    #    else:
    #        ow = long_size
    #        oh = int(1.0 * h * long_size / w + 0.5)
    #        short_size = oh
    #    img = img.resize((ow, oh), Image.BILINEAR)
    #    mask = mask.resize((ow, oh), Image.NEAREST)
    #    # pad crop
    #    if short_size < crop_size:
    #        padh = crop_size - oh if oh < crop_size else 0
    #        padw = crop_size - ow if ow < crop_size else 0
    #        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    #    # random crop crop_size
    #    w, h = img.size
    #    x1 = random.randint(0, w - crop_size)
    #    y1 = random.randint(0, h - crop_size)
    #    img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
    #    mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
    #    # gaussian blur as in PSP
    #    if random.random() < 0.5:
    #        img = img.filter(ImageFilter.GaussianBlur(
    #            radius=random.random()))
    #    # final transform
    #    return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64')
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_camvid_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".png"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths


    if split == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
        assert len(img_paths) == 367
    elif split == 'val':
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        assert len(img_paths) == 101
    elif split == 'trainval':
        assert split == 'trainval'
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        assert len(img_paths) == 468
    else:
        assert split == 'test'
        img_folder = os.path.join(folder, 'images/testing')
        mask_folder = os.path.join(folder, 'annotations/testing')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print(len(img_paths),len(mask_paths))
        assert len(img_paths) == 233

    return img_paths, mask_paths


if __name__ == '__main__':

    import os
    import numpy as np
    import torch
    from torch.utils import data
    import torchvision.transforms as transform

    input_transform = transform.Compose([
        transform.ToTensor(),
    #    transform.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    # dataset
    data_kwargs = {'transform': input_transform, 'base_size': 960,
                   'crop_size': [960,704]}
    trainset = CamVid(split='test', mode='val', **data_kwargs)
    trainloader = data.DataLoader(trainset, batch_size=1,
                                  drop_last=False, shuffle=True)


    # img = Image.open(r'E:\Seg_Program\CamVid_3\annotations\validation\0016E5_07959.png')
    # img = np.array(img)
    # print(np.unique(img))


    print(len(trainloader))
    import matplotlib.pyplot as plt
    import cv2
    for img,label, name in trainloader:
        print(label.size())
        print(torch.unique(label))
        print(name)


