config = {
    'coco_path': '/m2/shared/ziping/coco',
    'pascal_path': '/m2/shared/ziping/pascal_voc_seg',
    'imagenet_path': '/m2/shared/ziping/imagenet',
    'lr': 1e-3,  # 1e-5, for imagenet
    'batch_size': 1,  # 256 for imagenet; 128
    'weight_decay': 0,  # 1e-4,
    'epochs': 200,
    'scale': 128,  # input resolution
    'num_classes': 80  # COCO: 80, Pascal VOC: 20, ImageNet: 1000
}
