"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
__all__ = ['SegmentationDataset']


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size=(960,960), cat_max_ratio=0.75, ignore_index=-1):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

        # x1 = random.randint(0, w - crop_size)
        # y1 = random.randint(0, h - crop_size)
        # img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, img, mask):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = np.array(img) # 1024 2048
        mask = np.array(mask) # 1024 2048

        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(mask, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        # img = self.crop(img, crop_bbox)
        img = Image.fromarray(self.crop(img, crop_bbox))
        mask = Image.fromarray(self.crop(mask, crop_bbox))

        return img,mask


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=1024, crop_size=1024):
        """
        root: string

        split: string
            'train', 'val' or 'test'
        mode:

        transform: callable, optional
             A function that transforms the image
        base_size:
            shorter size will be resized between [short_size*0.5, short_size*2.0]
        crop_size:

        """
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        # self.color_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        # self.color_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

        #  color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  For PyTorch 1.9/TorchVision 0.10 users
        # self.crop = RandomCrop(crop_size=(crop_size,crop_size))



    def _val_sync_transform(self, img, mask):
        """
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        """
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # img = self.color_aug(img)

        crop_size = self.crop_size
        short_size = random.randint(int(self.base_size * 0.125), int(self.base_size * 2.0))


        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)  ##


        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # img , mask = self.crop(img,mask)


        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = self.color_aug(img)
        # else:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        ## color aug
        ''' 删除此项数据增强，将获得更好地测试集mIoU结果'''
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        img = self.color_aug(img)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
        
    """
    PIL.Image.open
        对于RGB图:
        - 读取后，RGB顺序的(cols,rows,3)；
        - 施加np.array()后，变为(rows,cols,3),即(H x W x C) 
        - 继续经过torchvision.transform.Totensor()后，变为torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        对于灰度图：
        - 读取后,(cols,rows);
        - 施加np.array()后，变为(rows,cols)

    cv2.imread
        对于RGB图：
        - 读取后，BGR顺序的(rows,cols,3),即(H x W x C); 就是np.array
        对于灰度图：
        - 读取后,(rows,cols); 就是np.array
    """
    def _img_transform(self, img):
        return np.array(img)
    
    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')
    
    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
