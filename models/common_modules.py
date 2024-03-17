import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Linear
from torch.nn.init import kaiming_normal_, kaiming_uniform_
from einops import rearrange, reduce, repeat
import numpy as np

class ConvBNLayer(nn.Module):
    def __init__(self,
        in_c,
        out_c,
        kernel_size,
        stride=1,
        num_groups=1,
        act='leaky_relu',
        init_type='kaiming_normal',
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=num_groups,
            bias=False)  # fix: init with kaiming_normal
        self.bn = BatchNorm2d(out_c)  # fix: apply l2 decay = 0.0 here
        self.act = nn.Hardswish() if act == 'hard_swish' else nn.LeakyReLU(0.01)

        self.init_weights(init_type)


    def init_weights(self, init_type):
        if init_type == 'kaiming_normal':
            kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        elif init_type == 'kaiming_uniform':
            kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



def make_divisible(v, divisor=8, min_value=None):
    """
    Rounds a value `v` to the nearest multiple of `divisor`, ensuring that the result is not less than `min_value`.

    Args:
        v (float): The value to be rounded.
        divisor (int, optional): The divisor used for rounding. Defaults to 8.
        min_value (int or None, optional): The minimum value that the rounded result should not be less than. 
            If None, it is set to `divisor`. Defaults to None.

    Returns:
        int: The rounded value that is divisible by `divisor` and not less than `min_value`.

    Example:
        >>> make_divisible(23, 8, 16)
        24

    This function is commonly used in scenarios where neural network architectures require certain constraints on the 
    dimensions of layers or channels. By specifying a `divisor`, the function ensures that the resulting value is 
    divisible by `divisor`. The `min_value` parameter can be used to set a lower limit for the rounded value.

    The rounding is done by adding `divisor / 2` to the input value, converting it to an integer, and then dividing 
    by `divisor`. If the resulting value is less than `min_value`, it is increased by `divisor`.

    Note:
        The `make_divisible` function assumes that the input value `v` is a positive number.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v




class DepthwiseSeparable(nn.Module):
    def __init__(self,
        in_c,
        out_c,
        stride,
        kernel_size=3,
        use_se=False,
        act='leaky_relu',
        init_type='kaiming_normal'
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            in_c=in_c,
            out_c=in_c,
            kernel_size=kernel_size,
            stride=stride,
            num_groups=in_c,   # each channel is a group, cause it's depthwise
            act=act,
            init_type=init_type
        )  
        if use_se:
            self.se = SEModule(in_c)
        self.pw_conv = ConvBNLayer(
            in_c=in_c,
            kernel_size=1,
            out_c=out_c,
            stride=1,
            act=act,
            init_type=init_type
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x



class SEModule(nn.Module):
    def __init__(self, in_c, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1) # (n, c, h, w) -> (n, c, 1, 1)
        self.conv1 = Conv2d(
            in_channels=in_c,
            out_channels=in_c // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(
            in_channels=in_c // reduction,
            out_channels=in_c,
            kernel_size=1,
            stride=1,
            padding=0)
        self.act = nn.Hardsigmoid()


    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.act(x)
        x = identity * x   # element-wise multiplication
        return x
