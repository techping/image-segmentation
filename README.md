# image-segmentation

# Network Structure

I define "out_channels" as a list of channels for the CNN architectures. e.g. the original U-Net architecture is defined as "out_channels" = [64, 128, 256, 512, 1024] in my search space.

# Code Description

## Network

+ `segnet.py`: Construst segmentation network by `num_channels` array.

+ `classnet.py`: Construct classification network from a segnet encoder

## Data

+ `coco_data.py`: COCO dataset loader

+ `pascal_data.py`: Pascal VOC 2012 dataset loader

+ `imagenet_data.py`: ImageNet dataset loader

## Train

+ `train.py`: Training script for segmentation

+ `train_category.py`: Training script for classification
