import torch
from monai.networks.blocks import Convolution
from monai.networks.layers import get_norm_layer
from torch import nn
from network.DepthWiseBlock import DepthWiseBlock
from network.DilationBlock import DilationBlock
from network.DoubleSeparatedBlock import DoubleSeparatedBlock
from network.ResidualBlock import ResidualBlock
from network.InvertedResidualBlock import InvertedResidualBlock
from network.GroupWeightBlock import GroupWeightBlock

class AMCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="GELU", norm="INSTANCE", adn_ordering="ADN"):
        super(AMCBlock, self).__init__()
        self.start_norm = get_norm_layer(name=norm, spatial_dims=2, channels=in_channels)
        self.depth_wise_block = DepthWiseBlock(in_channels, out_channels,
                                               activation=activation, norm=norm, adn_ordering=adn_ordering)
        self.dilation_block = DilationBlock(in_channels, out_channels,
                                            activation=activation, norm=norm, adn_ordering=adn_ordering)
        self.double_separated_block = DoubleSeparatedBlock(in_channels, out_channels, 5,
                                                           activation=activation, norm=norm,
                                                           adn_ordering=adn_ordering)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels,
                          activation=activation, norm=norm, adn_ordering=adn_ordering),
            ResidualBlock(out_channels, out_channels,
                          activation=activation, norm=norm, adn_ordering=adn_ordering))
        self.inverted_residual_block = InvertedResidualBlock(out_channels, out_channels, 4,
                                                             activation=activation, norm=norm,
                                                             adn_ordering=adn_ordering)
        self.merge_conv = GroupWeightBlock(out_channels * 5, out_channels,
                                           activation=activation, norm=norm, adn_ordering=adn_ordering)

    def forward(self, x):
        start_norm = self.start_norm(x)
        depth_wise_result = self.depth_wise_block(start_norm)
        dilation_result = self.dilation_block(start_norm)
        double_separated_result = self.double_separated_block(start_norm)
        residual_result = self.residual_blocks(start_norm)
        inverted_residual_result = self.residual_blocks(start_norm)
        concatenated_result = torch.cat((depth_wise_result, dilation_result,
                                         double_separated_result, residual_result,
                                         inverted_residual_result),dim=1)
        result = self.merge_conv(concatenated_result)
        return result


if __name__ == "__main__":
    input = torch.randn((1, 3, 64, 64))
    net = AMCBlock(3, 64)
    output = net(input)
    print(output.shape)
