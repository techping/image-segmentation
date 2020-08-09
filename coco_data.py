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


class COCOData(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.coco = COCO(annotation)
        self.ids = sorted(self.coco.imgs.keys())

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

        # build segmentation mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for annotation in coco_annotation:
            if annotation['category_id'] > 80:
                # skip label > 80
                continue
            mask += (mask == 0) * (coco.annToMask(annotation)
                                   * annotation['category_id'])
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
        transforms.Lambda(lambda x: cv2.resize(x, dsize=(size, size))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_preprocess = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((int(512 * output_stride), int(512 * output_stride)), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: cv2.resize(x, dsize=(int(
            size * output_stride), int(size * output_stride)), interpolation=cv2.INTER_NEAREST)),
        transforms.Lambda(lambda x: torch.from_numpy(x))
    ])

    datasets = {}
    datasets['train'] = COCOData(os.path.join(dataset_path, "train2017"),
                                 os.path.join(dataset_path, "annotations", "instances_train2017.json"), preprocess, mask_preprocess)
    datasets['val'] = COCOData(os.path.join(dataset_path, "val2017"),
                               os.path.join(dataset_path, "annotations", "instances_val2017.json"), preprocess, mask_preprocess)
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
    train = get_coco_data(1)['train'].dataset
    d = train.__getitem__(10)
    plt.figure()
    plt.imshow(d[0].numpy().transpose((1, 2, 0)))
    plt.savefig('1.jpg')
    cv2.imwrite('2.png', d[1].squeeze(0).numpy())
