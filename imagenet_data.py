# ------------------------------------------------------------------------------
# imagenet_data.py
# ------------------------------------------------------------------------------
#
# ImageNet ILSVRC2012 dataset loader
#
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torch


def get_imagenet_data(dataset_path, scale=64, batch_size=1):
    preprocess = transforms.Compose([
        transforms.Resize(int(scale * 1.05)),
        transforms.CenterCrop(scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    datasets = {}
    datasets['train'] = torchvision.datasets.ImageNet(
        dataset_path, split='train', transform=preprocess)
    datasets['val'] = torchvision.datasets.ImageNet(
        dataset_path, split='val', transform=preprocess)
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
    d = get_imagenet_data()
    img, label = d['train'].dataset.__getitem__(10)
    plt.figure()
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.savefig('1.jpg')
    print(label)
