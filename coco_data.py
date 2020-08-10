# ------------------------------------------------------------------------------
# coco_data.py
# ------------------------------------------------------------------------------
#
# COCO Detection 2017 dataset loader
#
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import torch
from torchvision import transforms
import os
import numpy as np
import cv2


np.random.seed(7737)


class COCOData(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transform=None, target_transform=None, size=512, output_stride=1.0, phase='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.coco = COCO(annotation)
        self.ids = sorted(self.coco.imgs.keys())
        self.phase = phase
        self.size = size
        self.output_stride = output_stride

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        # img = Image.open(os.path.join(self.root, path))
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # build segmentation mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for annotation in coco_annotation:
            if annotation['category_id'] > 80:
                # skip label > 80
                continue
            mask += (mask == 0) * (coco.annToMask(annotation)
                                   * annotation['category_id'])

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (int(self.size * self.output_stride),
                                 int(self.size * self.output_stride)), interpolation=cv2.INTER_NEAREST)

        # data augmentation
        if self.phase == 'train' and np.random.rand() < 0.5:
            # randomly scale
            factor = np.random.rand() * 1.5 + 0.5  # [0.5, 2.0]
            w, h = mask.shape
            new_w = int(w * factor)
            new_h = int(h * factor)
            diff_w = (new_w - w) // 2
            diff_h = (new_h - h) // 2
            diff_w = -diff_w if diff_w < 0 else diff_w
            diff_h = -diff_h if diff_h < 0 else diff_h
            img = cv2.resize(img, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)
            # print(f'scale {factor}, diff {(diff_w, diff_h)}, new {(new_w, new_h)}')
            if new_w <= w:
                # scale and pad
                img = np.pad(img, ((diff_w, w - new_w - diff_w), (diff_h, h - new_h - diff_h),
                                   (0, 0)), 'constant', constant_values=0)
                mask = np.pad(mask, ((diff_w, w - new_w - diff_w), (diff_h, h - new_h - diff_h)),
                              'constant', constant_values=255)
            else:
                # scale and crop
                img = img[diff_w:-(new_w - w - diff_w),
                          diff_h:-(new_h - h - diff_h), :]
                mask = mask[diff_w:-(new_w - w - diff_w),
                            diff_h:-(new_h - h - diff_h)]

        if self.phase == 'train' and np.random.rand() < 0.5:
            # randomly left-right filpping
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.ids)


def get_coco_data(dataset_path, size=512, output_stride=1.0, batch_size=1):
    preprocess = transforms.Compose([
        # transforms.Resize(512),
        # transforms.Resize((512, 512)),
        # transforms.Lambda(lambda x: cv2.resize(x, dsize=(size, size))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_preprocess = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((int(512 * output_stride), int(512 * output_stride)), interpolation=Image.NEAREST),
        # transforms.Lambda(lambda x: cv2.resize(x, dsize=(int(
        #     size * output_stride), int(size * output_stride)), interpolation=cv2.INTER_NEAREST)),
        transforms.Lambda(lambda x: torch.from_numpy(x))
    ])

    datasets = {}
    datasets['train'] = COCOData(os.path.join(dataset_path, "train2017"),
                                 os.path.join(dataset_path, "annotations", "instances_train2017.json"), preprocess, mask_preprocess, size, output_stride, 'train')
    datasets['val'] = COCOData(os.path.join(dataset_path, "val2017"),
                               os.path.join(dataset_path, "annotations", "instances_val2017.json"), preprocess, mask_preprocess, size, output_stride, 'val')
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  shuffle=True if x == 'train' else False,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  drop_last=True
                                                  )
                   for x in ['train', 'val']}
    return dataloaders


if __name__ == "__main__":
    train = get_coco_data('/m2/shared/ziping/coco')['train'].dataset
    d = train.__getitem__(10)
    plt.figure()
    plt.imshow(d[0].numpy().transpose((1, 2, 0)))
    plt.savefig('1.jpg')
    plt.figure()
    plt.imshow(d[1].numpy())
    plt.savefig('2.jpg')
