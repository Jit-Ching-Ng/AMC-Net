import torch
from torch import nn
from monai.networks.blocks import Convolution
from monai.networks.layers import get_norm_layer


class GroupWeightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="GELU", norm="INSTANCE", adn_ordering="ADN"):
        super(GroupWeightBlock, self).__init__()
        self.merge_conv = Convolution(2, in_channels, out_channels, kernel_size=1, act=activation, norm=norm,
                        adn_ordering=adn_ordering)
        self.merge_norm = get_norm_layer(name=norm, spatial_dims=2, channels=out_channels)

    def forward(self, x):
        result = self.merge_norm(self.merge_conv(x))
        return result


if __name__ == "__main__":
    input = torch.randn((1, 30, 64, 64))
    net = GroupWeightBlock(30, 4)
    output = net(input)
    print(output.shape)