import torch
from monai.networks.blocks import Convolution
from torch import nn


class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="GELU",
                 norm="INSTANCE", adn_ordering="ADN"):
        super(DilationBlock, self).__init__()

        self.conv1 = Convolution(2, in_channels, out_channels, kernel_size=3,
                                 adn_ordering=adn_ordering, norm=norm, act=activation, dilation=1)
        self.conv2 = Convolution(2, out_channels, out_channels, kernel_size=3,
                                 adn_ordering=adn_ordering, norm=norm, act=activation, dilation=3)
        self.conv3 = Convolution(2, out_channels, out_channels, kernel_size=3,
                                 adn_ordering=adn_ordering, norm=norm, act=activation, dilation=5)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return conv3


if __name__ == "__main__":
    input = torch.randn((1, 3, 64, 64))
    net = DilationBlock(3, 64)
    output = net(input)
    print(output.shape)
