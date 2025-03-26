import torch
from monai.networks.blocks import Convolution
from torch import nn


class DepthWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="GELU", norm="INSTANCE", adn_ordering="ADN", dilation=1):
        super(DepthWiseBlock, self).__init__()
        self.depth_wise_conv = Convolution(2, in_channels, in_channels, kernel_size=3,
                                           act=activation, adn_ordering=adn_ordering, norm=norm,
                                           dilation=dilation, groups=in_channels)
        self.point_wise_conv = Convolution(2, in_channels, out_channels, kernel_size=1,
                                           act=activation, adn_ordering=adn_ordering, norm=norm,
                                           dilation=dilation)

    def forward(self, x):
        depth_wise_conv = self.depth_wise_conv(x)
        point_wise_conv = self.point_wise_conv(depth_wise_conv)
        return point_wise_conv


if __name__ == "__main__":
    input = torch.randn((1, 3, 64, 64))
    net = DepthWiseBlock(3, 64)
    output = net(input)
    print(output.shape)
