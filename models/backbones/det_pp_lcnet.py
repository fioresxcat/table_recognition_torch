from __future__ import absolute_import, division, print_function

import os
import pdb
import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Linear
from torch.nn.init import kaiming_normal_
from ..common_modules import *



NET_CONFIG = {
    # k, in_c, out_c, stride, use_se
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5":
    [[3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False],
     [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


class PPLCNet(nn.Module):
    def __init__(self,
        in_channel=3,
        scale=1.0,
    ):
        super().__init__()
        self.out_channels = [
            int(NET_CONFIG["blocks3"][-1][2] * scale),
            int(NET_CONFIG["blocks4"][-1][2] * scale),
            int(NET_CONFIG["blocks5"][-1][2] * scale),
            int(NET_CONFIG["blocks6"][-1][2] * scale)
        ]
        self.scale = scale

        self.conv1 = ConvBNLayer(
            in_c=in_channel,
            kernel_size=3,
            out_c=make_divisible(16 * scale),
            stride=2,
            init_type='kaiming_normal',
            act='hard_swish'
        )
        
        for block_name in ['blocks2', 'blocks3', 'blocks4', 'blocks5', 'blocks6']:
            blocks = []
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG[block_name]):
                blocks.append(
                    DepthwiseSeparable(
                        in_c=make_divisible(in_c * scale),
                        out_c=make_divisible(out_c * scale),
                        kernel_size=k,
                        stride=s,
                        use_se=se,
                        act='hard_swish',
                    )
                )
            setattr(self, block_name, nn.Sequential(*blocks))


    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        x = self.blocks6(x)
        outs.append(x)
        return outs
    

if __name__ == '__main__':
    model = PPLCNet()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    outs = model(x)
    for i, out in enumerate(outs):
        print(f"outs[{i}].shape: {out.shape}")
    pdb.set_trace()