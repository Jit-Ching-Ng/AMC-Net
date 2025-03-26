import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import get_norm_layer


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, activation="GELU", norm="INSTANCE", adn_ordering="ADN", dilation=1):
        super(InvertedResidualBlock, self).__init__()
        hidden_channels = in_channels * expansion_factor
        self.conv1 = Convolution(2, in_channels, hidden_channels, kernel_size=1,
                                 act=activation, adn_ordering=adn_ordering, norm=norm,
                                 dilation=dilation)
        self.conv2 = Convolution(2, hidden_channels, hidden_channels, kernel_size=3,
                                 act=activation, adn_ordering=adn_ordering, norm=norm,
                                 dilation=dilation)
        self.conv3 = Convolution(2, hidden_channels, out_channels, kernel_size=1,
                                 act=activation, adn_ordering=adn_ordering, norm=norm,
                                 dilation=dilation)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        result = x + conv3
        return result


if __name__ == "__main__":
    input = torch.randn((1, 3, 64, 64))
    net = InvertedResidualBlock(3, 3, 4)
    output = net(input)
    print(output.shape)