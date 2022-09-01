import os
import os.path as osp

from PIL import Image, ImageOps
import numpy as np
import scipy.io as sio

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

import settings


def fetch(image_path, label_path=None):
    with Image.open(image_path).convert('RGB') as image:
        image = ImageOps.exif_transpose(image)
        image = torch.FloatTensor(np.asarray(image)) / 255
        # TODO: use dataset mean/stdev rather than calculating per image
        image = (image - settings.MEAN) / settings.STD
        image = image.permute(2, 0, 1).unsqueeze(dim=0)

    if label_path is not None:
        with Image.open(label_path).convert('P') as label:
            label = torch.FloatTensor(np.asarray(label))
            label = label.unsqueeze(dim=0).unsqueeze(dim=1)
    else:
        label = None

    return image, label


def scale(image, label=None):
    ratio = np.random.choice(settings.SCALES)
    image = F.interpolate(image, scale_factor=ratio, mode='bilinear', 
                          align_corners=True)
    if label is not None:
        label = F.interpolate(label, scale_factor=ratio, mode='nearest')
    return image, label


def pad(image, label=None):
    h, w = image.size()[-2:] 
    crop_size = settings.CROP_SIZE
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', 
                          value=settings.IGNORE_LABEL)
    return image, label


def pad_inf(image, label=None):
    h, w = image.size()[-2:] 
    stride = settings.STRIDE
    pad_h = (stride + 1 - h % stride) % stride
    pad_w = (stride + 1 - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', 
                          value=settings.IGNORE_LABEL)
    return image, label


def crop(image, label=None):
    h, w = image.size()[-2:]
    crop_size = settings.CROP_SIZE
    s_h = np.random.randint(0, h - crop_size + 1)
    s_w = np.random.randint(0, w - crop_size + 1)
    e_h = s_h + crop_size
    e_w = s_w + crop_size
    image = image[:, :, s_h: e_h, s_w: e_w]
    label = label[:, :, s_h: e_h, s_w: e_w]
    return image, label


def flip(image, label=None):
    if np.random.rand() < 0.5:
        image = torch.flip(image, [3])
        if label is not None:
            label = torch.flip(label, [3])
    return image, label

def adjust_brightness(image, label=None):
    brightness = np.random.uniform(0.7, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    image = transforms.functional.adjust_brightness(image, brightness)
    image = transforms.functional.adjust_contrast(image, contrast)
    return image, label

def sharpen(image, label=None):
    sharpness = np.random.uniform(0.5, 1.5)
    image = transforms.functional.adjust_sharpness(image, sharpness)
    return image, label

def rotate(image, label=None):
    angle = np.random.uniform(-20, 20)
    image = transforms.functional.rotate(image, angle)
    label = transforms.functional.rotate(label, angle, fill=255)
    return image, label

def labels_for_model_type(image, label):
    """
    Transforms labels to accomodate the model type:
    boolean for isgeological, nongeological masked for structuretype
    """
    if settings.MODEL_TYPE == 'isgeological':
        label[label < 5] = 1
        label[label == 5] = 0
    elif settings.MODEL_TYPE == 'structuretype':
        label[label == 5] = 255
    return image, label

class BaseDataset(data.Dataset):
    def __init__(self, data_root, split):
        self.data_root = data_root

        file_list = osp.join('datalist', split + '.txt')
        file_list = tuple(open(file_list, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list
        self.image_files = os.listdir(osp.join(self.data_root, 'JPEGImages'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_id = self.files[idx]
        return self._get_item(image_id)

    def _get_item(self, idx):
        raise NotImplemented


class TrainDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, split='trainaug'):
        super(TrainDataset, self).__init__(data_root, split)

    def _get_item(self, image_id):
        image_name = [name for name in self.image_files if name.startswith(f'{image_id}.')][0]
        image_path = osp.join(self.data_root, 'JPEGImages', image_name)
        label_path = osp.join(self.data_root, 'SegmentationClassAug', 
                              image_id + '.png')

        image, label = fetch(image_path, label_path)
        image, label = scale(image, label)
        image, label = pad(image, label)
        image, label = crop(image, label)
        image, label = flip(image, label)
        image, label = adjust_brightness(image, label)
        image, label = sharpen(image, label)
        image, label = rotate(image, label)
        image, label = labels_for_model_type(image, label)

        return image[0], label[0, 0].long()
 

class ValDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, split='val'):
        super(ValDataset, self).__init__(data_root, split)

    def _get_item(self, image_id):
        image_name = [name for name in self.image_files if name.startswith(f'{image_id}.')][0]
        image_path = osp.join(self.data_root, 'JPEGImages', image_name)
        label_path = osp.join(self.data_root, 'SegmentationClassAug', 
                              image_id + '.png')

        image, label = fetch(image_path, label_path)
        image, label = pad_inf(image, label)
        image, label = labels_for_model_type(image, label)
        return image[0], label[0, 0].long()



def test_dt():
    train_dt = TrainDataset()
    print('train', len(train_dt))
    for i in range(10):
        img, lbl = train_dt[i]
        print(img.shape, lbl.shape, img.mean(), np.unique(lbl))

    val_dt = ValDataset()
    print('val', len(val_dt))
    for i in range(10):
        img, lbl = val_dt[i]
        print(img.shape, lbl.shape, img.mean(), np.unique(lbl))


if __name__ == '__main__':
    test_dt()
