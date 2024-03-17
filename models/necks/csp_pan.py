from ..common_modules import *

# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F
# from paddle import ParamAttr

# __all__ = ['CSPPAN']



class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.
    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
    """

    def __init__(self,
        in_c,
        out_c,
        kernel_size=3,
        expansion=0.5,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu"
    ):
        super().__init__()
        hidden_c = int(out_c * expansion)
        conv_func = DepthwiseSeparable if use_depthwise else ConvBNLayer
        self.conv1 = ConvBNLayer(
            in_c=in_c,
            out_c=hidden_c,
            kernel_size=1,
            act=act,
            init_type='kaiming_uniform'
        )
        self.conv2 = conv_func(
            in_c=hidden_c,
            out_c=out_c,
            kernel_size=kernel_size,
            stride=1,
            act=act,
            init_type='kaiming_uniform'
        )
        self.add_identity = add_identity and in_c == out_c


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
    """

    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        expand_ratio=0.5,
        num_blocks=1,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu"
    ):
        super().__init__()
        mid_c = int(out_c * expand_ratio)
        self.main_conv = ConvBNLayer(in_c, mid_c, 1, act=act, init_type='kaiming_uniform')
        self.short_conv = ConvBNLayer(in_c, mid_c, 1, act=act, init_type='kaiming_uniform')
        self.final_conv = ConvBNLayer(2 * mid_c, out_c, 1, act=act, init_type='kaiming_uniform')

        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                mid_c,
                mid_c,
                kernel_size,
                1.0,
                add_identity,
                use_depthwise,
                act=act
            ) for _ in range(num_blocks)
        ])


    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.concat((x_main, x_short), axis=1)
        return self.final_conv(x_final)


class Channel_Transform(nn.Module):
    """
    Channel_T class represents the channel transformation module in the network.

    Args:
        in_channels (list[int]): List of input channel sizes for each input feature map.
        out_channels (int): Output channel size for the transformed feature maps.
        act (str): Activation function to be applied after each convolutional layer.

    Returns:
        list[Tensor]: List of transformed feature maps.

    """

    def __init__(self,
                 in_channels=[116, 232, 464],
                 out_channels=96,
                 act="leaky_relu"):
        super(Channel_Transform, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(
                ConvBNLayer(in_channels[i], out_channels, 1, 1, act=act, init_type='kaiming_uniform')
            )


    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x (list): List of input tensors.

        Returns:
            list: List of output tensors.
        """
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class CSPPAN(nn.Module):
    """Path Aggregation Network with CSP module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(
        self,
        list_in_c,
        out_c,
        kernel_size=5,
        num_csp_blocks=1,
        use_depthwise=True,
        act='hard_swish'
    ):
        super(CSPPAN, self).__init__()
        self.list_in_c = list_in_c
        self.list_out_c = [out_c] * len(list_in_c)
        conv_func = DepthwiseSeparable if use_depthwise else ConvBNLayer
        self.conv_trans = Channel_Transform(list_in_c, out_c, act=act)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(list_in_c) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPLayer(
                    out_c * 2,
                    out_c,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(list_in_c) - 1):
            self.downsamples.append(
                conv_func(
                    out_c,
                    out_c,
                    kernel_size=kernel_size,
                    stride=2,
                    act=act
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    out_c * 2,
                    out_c,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act
                )
            )


    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.list_in_c)
        inputs = self.conv_trans(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.list_in_c) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = F.interpolate(feat_heigh, size=feat_low.shape[2:4], mode="nearest")
            inner_out = self.top_down_blocks[len(self.list_in_c) - 1 - idx](
                torch.concat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.list_in_c) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.concat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        return tuple(outs)



if __name__ == '__main__':
    neck = CSPPAN([96, 192, 384], 96)
    neck.eval()
    inputs = [
        torch.randn(1, 96, 56, 56),
        torch.randn(1, 192, 28, 28),
        torch.randn(1, 384, 14, 14)
    ]
    outputs = neck(inputs)
    for i, out in enumerate(outputs):
        print(f"outputs[{i}].shape: {out.shape}")
