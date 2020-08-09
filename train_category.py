# ------------------------------------------------------------------------------
# train_category.py
# ------------------------------------------------------------------------------
#
# Train the encoder part on ImageNet 1000 classes dataset.
#
# I extract the encoder part of the segmentation network, attach several MLP
# layers to form a classification network.
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

import torch
from imagenet_data import get_imagenet_data
from classnet import ClassNet
import time
import copy
from config import config


def train_model(model, dataset, device, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0.0

            # Iterate over data.
            for i, data in enumerate(dataset[phase]):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)  # 1, 1000
                    _, preds = outputs.max(1)  # 1

                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # torch.backends.cudnn.benchmark=True
                        loss.backward()
                        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                        optimizer.step()
                        # exp_lr_scheduler.step()

                # statistics
                step_loss = loss.item() * inputs.size(0)
                step_correct = (preds == labels).sum().item()

                running_loss += step_loss
                running_corrects += step_correct
                print('\r{} {}/{} Loss: {:.4f} Acc: {:.4f}'.format(phase, i + 1, len(dataset[phase]), running_loss / (
                    inputs.size(0) * (i + 1)), running_corrects / (inputs.size(0) * (i + 1))), end='')

            epoch_loss = running_loss / len(dataset[phase].dataset)
            epoch_acc = running_corrects / len(dataset[phase].dataset)

            print()

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'model-imagenet.pth')
                    print(f'Best Acc: {best_acc}')
                    print('Best model saved at: {}'.format('model-imagenet.pth'))
                else:
                    print('Performance not improved. Skip saving.')
                val_acc_history.append(epoch_acc)
                scheduler.step(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    ds = get_imagenet_data(config['imagenet_path'],
                           scale=config['scale'], batch_size=config['batch_size'])  # get_data()
    # ds = get_coco_data(output_stride=1/8, batch_size=config['batch_size'])

    kw = {
        'out_channels': [64, 128, 256, 512, 1024]  # [40, 60, 90, 135, 203]#
    }

    # my model
    model = ClassNet([3, config['scale'], config['scale']], 1000, **kw)

    # vgg model
    # uncomment the next line to use vgg19
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = torch.nn.DataParallel(model)

    # model.load_state_dict(torch.load('model-imagenet.pth'))

    model.to(device=device)
    optimizer = torch.optim.Adam(
        model.parameters(), config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, verbose=True, factor=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        train_model(model, ds, device, criterion, optimizer,
                    lr_scheduler, config['epochs'])
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
    finally:
        torch.save(model.state_dict(), 'end.pth')
