import torch
from monai.networks.layers import get_norm_layer
from torch import nn
from monai.networks.blocks.convolutions import Convolution


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="GELU", norm="INSTANCE", adn_ordering="ADN", dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Convolution(2, in_channels, out_channels, kernel_size=3,
                                 act=activation, adn_ordering=adn_ordering, norm=norm,
                                 dilation=dilation)
        self.conv2 = Convolution(2, in_channels, out_channels, kernel_size=3,
                                 act=activation, adn_ordering=adn_ordering, norm=norm,
                                 dilation=dilation)
        self.conv3 = Convolution(2, out_channels, out_channels, kernel_size=3,
                                 act=activation, adn_ordering=adn_ordering, norm=norm,
                                 dilation=dilation)
        self.merge_norm = get_norm_layer(name=norm, spatial_dims=2, channels=out_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2)
        result = conv3 + conv1
        result = self.merge_norm(result)
        return result


if __name__ == "__main__":
    input = torch.randn((1, 3, 64, 64))
    net = ResidualBlock(3, 64)
    output = net(input)
    print(output.shape)
