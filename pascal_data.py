# ------------------------------------------------------------------------------
# pascal_data.py
# ------------------------------------------------------------------------------
#
# Pascal VOC 2012 dataset loader
# + original dataset
# + augmented dataset
#
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import cv2


np.random.seed(7737)


class PascalVOCAug(Dataset):
    def __init__(self, dataset_path, file_path, phase='train', size=512, output_stride=1.0):
        self.dataset_path = dataset_path
        self.data = []
        with open(dataset_path + file_path, 'r') as f:
            for line in f.readlines():
                self.data.append(line.split())
        self.preprocess = transforms.Compose([
            # transforms.Resize(512),
            # transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (int(size * output_stride), int(size * output_stride)), interpolation=Image.NEAREST),
            transforms.Lambda(lambda x: np.array(x)),
            transforms.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0))
        ])

    def __getitem__(self, index):
        image_path = self.dataset_path + self.data[index][0]
        mask_path = self.dataset_path + self.data[index][1]
        image = Image.open(image_path)
        mask = cv2.imread(mask_path, 0)
        return [
            self.preprocess(image),  # torch.from_numpy(image),
            self.mask_preprocess(mask),  # torch.from_numpy(mask)
        ]

    def __len__(self):
        return len(self.data)


def get_data(dataset_path, size=512, output_stride=1.0, batch_size=1):
    preprocess = transforms.Compose([
        # transforms.Resize(512),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_preprocess = transforms.Compose([
        transforms.Resize((int(size * output_stride),
                           int(size * output_stride)), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: np.array(x)),
        transforms.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0))
    ])

    aug_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine([0, np.pi]),
        transforms.RandomPerspective(),
        # transforms.RandomRotation([0, np.pi])
        # transforms.RandomSizedCrop()
    ])

    data_folder = dataset_path

    datasets = {}
    datasets['train'] = torchvision.datasets.VOCSegmentation(
        root=data_folder, image_set='train', transform=preprocess, target_transform=mask_preprocess)
    datasets['val'] = torchvision.datasets.VOCSegmentation(
        root=data_folder, image_set='val', transform=preprocess, target_transform=mask_preprocess)
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  shuffle=True if x == 'train' else False,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  drop_last=True
                                                  )
                   for x in ['train', 'val']}
    return dataloaders


train_path = '/SegmentationAug/train_aug.txt'
val_path = '/SegmentationAug/val.txt'


def get_augdata(dataset_path, size=512, output_stride=1.0, batch_size=1):
    datasets = {}
    datasets['train'] = PascalVOCAug(
        os.path.join(dataset_path, 'VOCdevkit', 'VOC2012'), train_path, size=size, output_stride=output_stride)
    datasets['val'] = PascalVOCAug(
        os.path.join(dataset_path, 'VOCdevkit', 'VOC2012'), val_path, phase='val', size=size, output_stride=output_stride)
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  shuffle=True if x == 'train' else False,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  drop_last=True
                                                  )
                   for x in ['train', 'val']}
    return dataloaders


if __name__ == '__main__':
    data = PascalVOCAug(dataset_path, train_path, output_stride=1/8)

    d = data.__getitem__(500)
    print(type(d[0]))
    print(type(d[1]))
    print(d[0].numpy().shape)
    print(d[1].numpy().shape)
    plt.figure()
    plt.imshow(d[0].numpy().transpose((1, 2, 0)))
    plt.savefig('1.jpg')
    plt.figure()
    plt.imshow(d[1].numpy().reshape((64, 64)))
    plt.savefig('2.png')
    print(set(d[1].numpy().reshape((-1,))))
