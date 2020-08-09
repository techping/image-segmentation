# ------------------------------------------------------------------------------
# segnet.py
# ------------------------------------------------------------------------------
#
# Segmentation Network
#
# ------------------------------------------------------------------------------
# Ziping Chen, University of Southern California
# <zipingch@usc.edu>
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            # nn.ReLU(inplace=True),
        )

    def upconv(self, in_c, out_c):
        return nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2
        )

    def crop(self, in_tensor, out_tensor):

        diff_h = (in_tensor.size()[2] - out_tensor.size()[2])
        left_h = diff_h // 2
        diff_w = (in_tensor.size()[3] - out_tensor.size()[3])
        left_w = diff_w // 2
        return in_tensor[:, :, left_h:-(diff_h-left_h), left_w:-(diff_w-left_w)]

    def pad(self, in_tensor, out_tensor):
        diff_h = (in_tensor.size()[2] - out_tensor.size()[2])
        left_h = diff_h // 2
        diff_w = (in_tensor.size()[3] - out_tensor.size()[3])
        left_w = diff_w // 2
        return F.pad(out_tensor, [left_w, diff_w - left_w,
                                  left_h, diff_h - left_h])

    def shortcut(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_c),
            # nn.ReLU(inplace=True),
        )

    def __init__(self, input_size=[3, 32, 32], output_size=81, **kw):
        '''
        *** Create Pytorch net for Segmentation ***
        input_size: Iterable. Size of 1 input. Example: [3,32,32]
        output_size: Integer. #labels. Number of categories + 1 (background)
        kw:
            --- CONV ---:
                    out_channels: Iterable. #filters in each conv layer, i.e. #conv layers. If no conv layer is needed, enter []          
        '''
        super().__init__()

        #### Conv ####
        self.out_channels = kw['out_channels'] if 'out_channels' in kw else net_kws_defaults['out_channels']
        self.num_layers_conv = len(self.out_channels)
        # self.kernel_sizes = kw['kernel_sizes'] if 'kernel_sizes' in kw else self.num_layers_conv*net_kws_defaults['kernel_sizes']
        # self.strides = kw['strides'] if 'strides' in kw else self.num_layers_conv*net_kws_defaults['strides']
        # self.paddings = kw['paddings']  if 'paddings' in kw else [(ks-1)//2 for ks in self.kernel_sizes]
        # self.dilations = kw['dilations'] if 'dilations' in kw else self.num_layers_conv*net_kws_defaults['dilations']
        # self.groups = kw['groups'] if 'groups' in kw else self.num_layers_conv*net_kws_defaults['groups']
        # self.apply_bns = kw['apply_bns'] if 'apply_bns' in kw else self.num_layers_conv*net_kws_defaults['apply_bns']
        # self.apply_maxpools = kw['apply_maxpools'] if 'apply_maxpools' in kw else self.num_layers_conv*net_kws_defaults['apply_maxpools']
        # self.apply_gap = kw['apply_gap'] if 'apply_gap' in kw else net_kws_defaults['apply_gap']
        # self.apply_dropouts = kw['apply_dropouts'] if 'apply_dropouts' in kw else self.num_layers_conv*net_kws_defaults['apply_dropouts']
        # if 'dropout_probs' in kw:
        #     self.dropout_probs = kw['dropout_probs']
        # else:
        #     self.dropout_probs = np.count_nonzero(self.apply_dropouts)*[net_kws_defaults['dropout_probs'][1]]
        #     if len(self.apply_dropouts)!=0 and self.apply_dropouts[0]==1:
        #         self.dropout_probs[0] = net_kws_defaults['dropout_probs'][0]

        # self.shortcuts = kw['shortcuts'] if 'shortcuts' in kw else self.num_layers_conv*net_kws_defaults['shortcuts']

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_path = nn.ModuleDict({})
        self.expansive_path = nn.ModuleDict({})
        for i in range(self.num_layers_conv):
            self.contracting_path['doubleconv-{0}'.format(i)] = self.double_conv(input_size[0] if i == 0 else self.out_channels[i-1],
                                                                                 self.out_channels[i])
            self.contracting_path['shortcut-{0}'.format(i)] = self.shortcut(input_size[0] if i == 0 else self.out_channels[i-1],
                                                                            self.out_channels[i])

            if i != self.num_layers_conv - 1:
                self.expansive_path['upconv-{0}'.format(self.num_layers_conv - 2 - i)] = \
                    self.upconv(self.out_channels[i + 1],
                                self.out_channels[i])
                self.expansive_path['doubleconv-{0}'.format(self.num_layers_conv - 2 - i)] = \
                    self.double_conv(self.out_channels[i] * 2,
                                     self.out_channels[i])
                self.expansive_path['shortcut-{0}'.format(self.num_layers_conv - 2 - i)] = \
                    self.shortcut(self.out_channels[i] * 2,
                                  self.out_channels[i])

        self.num_classes = output_size
        self.conv_1x1 = nn.Conv2d(
            in_channels=self.out_channels[0], out_channels=self.num_classes, kernel_size=1)

        self.conv_1x1_aux = nn.Conv2d(
            in_channels=self.out_channels[-1], out_channels=self.num_classes, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        # encoder
        out = []
        for i in range(self.num_layers_conv):
            # residual block >>
            feature = self.contracting_path['doubleconv-{0}'.format(i)](
                x if i == 0 else pre)
            feature += self.contracting_path['shortcut-{0}'.format(i)](
                x if i == 0 else pre)
            feature = nn.ReLU(inplace=True)(feature)
            # residual block <<
            out.append(feature)
            if i != self.num_layers_conv - 1:
                pre = self.maxpool(out[i])
        # decoder
        for i in range(self.num_layers_conv - 1):
            x = self.expansive_path['upconv-{0}'.format(i)](
                out[-1] if i == 0 else x)
            x = torch.cat([out[-2 - i], self.pad(out[-2 - i], x)], 1)
            # residual block >>
            y = self.expansive_path['doubleconv-{0}'.format(i)](x)
            x = self.expansive_path['shortcut-{0}'.format(i)](x) + y
            x = nn.ReLU(inplace=True)(x)
            # residual block <<
        x = self.conv_1x1(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchviz import make_dot, make_dot_from_trace

    kw = {
        'out_channels': [64, 128, 256, 512, 1024]  # [40, 60, 90, 135]#
    }

    # test
    fake_input = torch.rand((1, 3, 128, 128))
    model = SegNet([3, 128, 128], 21, **kw)
    output = model(fake_input)
    print(output.shape)
    print(model)

    dot = make_dot(model(fake_input), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render("test")
