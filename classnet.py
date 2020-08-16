# ------------------------------------------------------------------------------
# classnet.py
# ------------------------------------------------------------------------------
#
# Classification Network
#
# I extract the encoder part of the segmentation network, attach several MLP
# layers to form a classification network.
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

from segnet import SegNet
import torch
import torch.nn as nn


class ClassNet(SegNet):
    def __init__(self, input_size=[3, 32, 32], output_size=1000, **kw):
        super(ClassNet, self).__init__(input_size, output_size, **kw)

        self.nlp_input_size = kw['out_channels'][-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.nlp_input_size, 4096)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        out = []
        for i in range(self.num_layers_conv):
            # residual block >>
            feature = self.contracting_path['doubleconv-{0}'.format(i)](
                x if i == 0 else pre)
            # residual block <<
            out.append(feature)
            if i != self.num_layers_conv - 1:
                pre = self.maxpool(out[i])
        x = self.pool(out[-1])
        x = x.view(batch_size, -1)
        assert x.shape[1] == self.nlp_input_size, "size error"
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    kw = {
        'out_channels': [40, 60, 90, 135, 160]  # [64, 128, 256, 512, 1024]
    }
    model = ClassNet([3, 256, 256], 1000, **kw)
    fake = torch.rand((10, 3, 256, 256))
    output = model(fake)
    print(output.shape)
