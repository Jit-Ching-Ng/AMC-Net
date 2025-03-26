import torch
from monai.networks.layers import get_norm_layer
from torch import nn
from monai.networks.blocks.convolutions import Convolution


class DoubleSeparatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, activation="GELU", norm="INSTANCE", adn_ordering="ADN"):
        super(DoubleSeparatedBlock, self).__init__()

        self.conv1 = Convolution(2, in_channels, out_channels,
                                 kernel_size=(1, size),
                                 adn_ordering=adn_ordering, norm=norm, act=activation
                                 )

        self.conv2 = Convolution(2, out_channels, out_channels,
                                 kernel_size=(size, 1),
                                 adn_ordering=adn_ordering, norm=norm, act=activation
                                 )

        self.conv3 = Convolution(2, in_channels, out_channels,
                                 kernel_size=(size, 1),
                                 adn_ordering=adn_ordering, norm=norm, act=activation
                                 )

        self.conv4 = Convolution(2, out_channels, out_channels,
                                 kernel_size=(1, size),
                                 adn_ordering=adn_ordering, norm=norm, act=activation
                                 )
        self.merge_norm = get_norm_layer(name=norm, spatial_dims=2, channels=out_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(x)
        conv4 = self.conv4(conv3)
        result = self.merge_norm(conv2 + conv4)
        return result


if __name__ == "__main__":
    input = torch.randn((1, 3, 64, 64))
    net = DoubleSeparatedBlock(3, 64, 5)
    output = net(input)
    print(output.shape)
