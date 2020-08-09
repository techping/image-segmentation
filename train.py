# ------------------------------------------------------------------------------
# train.py
# ------------------------------------------------------------------------------
#
# Train the segmentation network
#
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

import torch
from pascal_data import get_data, get_augdata
from coco_data import get_coco_data
from segnet import SegNet
import time
import copy
from config import config


def iou_calc(preds, labels, num_classes=81):
    """iou_calc

    Perform IoU calculations
    """
    batch_ious = [[0.0, 0.0] for _ in range(1, num_classes)]
    pred = preds.view(-1)
    target = labels.view(-1)
    # ignore background class 0
    for i in range(1, num_classes):
        pred_inds = pred.eq(i)
        target_inds = target.eq(i)
        intersection = (pred_inds[target_inds].long().sum()).float().item()
        union = (pred_inds.long().sum() + target_inds.long().sum()
                 ).float().item() - intersection
        # iou = intersection / (union + 1e-12)
        if target_inds.long().sum().item() > 0:
            batch_ious[i - 1][0] += intersection
            batch_ious[i - 1][1] += union
    return batch_ious


def get_miou(class_iu):
    class_ious = []
    for i in range(len(class_iu)):
        class_ious.append(class_iu[i][0] / (class_iu[i][1] + 1e-12))
    return sum(class_ious) / (len(class_ious) + 1e-12)


def pixelwise_acc(preds, labels):
    label = labels.view(-1)
    pred = preds.view(-1)
    mask = (label != 0) & (label != 255)
    corrects = ((pred == label) & mask).long().sum().float().item()
    # accs = (corrects / (mask.long().sum().float().item() + 1e-12))
    return [corrects, mask.long().sum().float().item()]


def train_model(model, dataset, device, criterion, optimizer, scheduler, num_epochs=10, num_classes=81):
    since = time.time()

    val_miou_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Training:")
            else:
                model.eval()  # Set model to evaluate mode
                print("Validation:")

            running_loss = 0.0
            class_ious = [[0.0, 0.0] for _ in range(1, num_classes + 1)]
            acc = [0.0, 0.0]  # [num of correct pixels, total available pixels]

            # Iterate over data.
            for i, data in enumerate(dataset[phase]):
                inputs = data[0].to(device)
                labels = data[1].to(device).squeeze(1).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    _, preds = outputs.max(1)

                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                        optimizer.step()
                        # exp_lr_scheduler.step()

                # statistics
                step_loss = loss.item() * inputs.size(0)
                step_ious = iou_calc(
                    preds.detach(), labels.detach(), num_classes + 1)
                for idx, iou in enumerate(step_ious):
                    class_ious[idx][0] += iou[0]
                    class_ious[idx][1] += iou[1]
                running_loss += step_loss
                running_miou = get_miou(class_ious)
                step_acc = pixelwise_acc(preds.detach(), labels.detach())
                acc[0] += step_acc[0]
                acc[1] += step_acc[1]
                print('\r{} {}/{} Loss: {:.4f} mIoU: {:.4f} acc: {:.4f}'.format(phase, i + 1, len(
                    dataset[phase]), running_loss / (inputs.size(0) * (i + 1)), running_miou, acc[0] / (acc[1] + 1e-12)), end='')

            epoch_loss = running_loss / len(dataset[phase].dataset)
            epoch_miou = get_miou(class_ious)
            ious = []
            for i in range(len(class_ious)):
                ious.append(class_ious[i][0] / (class_ious[i][1] + 1e-12))
            print(f'\nClass IoU: {ious}')
            print(f'\nEpoch mIoU: {epoch_miou}', end='')

            print()

            # deep copy the model
            if phase == 'val':
                if epoch_miou > best_miou:
                    best_miou = epoch_miou
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'model-full.pth')
                    print('Best model saved at: {}'.format(
                        'model-full.pth'))
                else:
                    print('Performance not improved. Skip saving.')
                val_miou_history.append(epoch_miou)
                scheduler.step(epoch_miou)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val mIoU: {:4f}'.format(best_miou))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_miou_history


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # use original Pascal VOC 2012 data
    # ds = get_data(config['pascal_path'], 256)

    # use augmented Pascal VOC 2012 data
    # ds = get_augdata(config['pascal_path'], size=config['scale'], output_stride=1, batch_size=config['batch_size'])#get_data()

    # use COCO detection data
    ds = get_coco_data(config['coco_path'],
                       size=config['scale'], output_stride=1, batch_size=config['batch_size'])

    kw = {
        'out_channels': [64, 128, 256, 512, 1024]  # [40, 60, 90, 135, 203]#
    }

    # use my model
    model = SegNet([3, config['scale'], config['scale']],
                   config['num_classes'] + 1, **kw)

    # use deeplabv3-resnet-101 model
    # uncomment next line to use deeplabv3-resnet-101
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)

    # load imagenet pretrained weights
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('model-imagenet.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # pretrained_dict.pop('conv_1x1.weight')
    # pretrained_dict.pop('conv_1x1.bias')
    # pretrained_dict.pop('conv_1x1_aux.weight')
    # pretrained_dict.pop('conv_1x1_aux.bias')
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    model.to(device=device)
    optimizer = torch.optim.SGD(model.parameters(
    ), config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, verbose=True, factor=0.1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    try:
        train_model(model, ds, device, criterion, optimizer,
                    lr_scheduler, config['epochs'], config['num_classes'])
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
    finally:
        torch.save(model.state_dict(), 'end.pth')
