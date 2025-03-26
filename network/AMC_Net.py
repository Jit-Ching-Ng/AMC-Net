import torch
from monai.utils import ensure_tuple_rep
from torch import nn
from monai.networks.blocks import Convolution
from network.AMCBlock import AMCBlock
from network.GroupWeightBlock import GroupWeightBlock
from network.ResidualBlock import ResidualBlock
from monai.networks.blocks.upsample import UpSample
from network.TransposeConvolutionBlock import TransposeConvolutionBlock


class AMC_Net(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, activation="GELU",
                 norm="INSTANCE", adn_ordering="ADN", interpolation_mode="bilinear"):
        super(AMC_Net, self).__init__()
        up_sample_size = ensure_tuple_rep(2, 2)

        self.down_sample_1_2 = nn.AvgPool2d(2, 2)
        self.down_sample_2_4 = nn.AvgPool2d(2, 2)
        self.down_sample_4_8 = nn.AvgPool2d(2, 2)

        self.conv_1 = Convolution(2, in_channels, mid_channels,
                                  kernel_size=3, act=activation, adn_ordering=adn_ordering, norm=norm)
        self.down_conv_1_2 = Convolution(2, mid_channels, mid_channels * 2,
                                         kernel_size=2, strides=2, padding=0)
        self.conv_2 = Convolution(2, in_channels, mid_channels * 2,
                                  kernel_size=3, act=activation, adn_ordering=adn_ordering, norm=norm)
        self.group_weight_2 = GroupWeightBlock(mid_channels * 4, mid_channels * 2,
                                               activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.basic_2 = AMCBlock(mid_channels * 2, mid_channels * 2,
                                activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.down_conv_2_4 = Convolution(2, mid_channels * 2, mid_channels * 4,
                                         kernel_size=2, strides=2, padding=0)
        self.down_conv_basic_2_4 = Convolution(2, mid_channels * 2, mid_channels * 4,
                                               kernel_size=2, strides=2, padding=0)
        self.conv_4 = Convolution(2, in_channels, mid_channels * 4,
                                  kernel_size=3, act=activation, adn_ordering=adn_ordering, norm=norm)
        self.group_weight_4 = GroupWeightBlock(mid_channels * 8, mid_channels * 4,
                                               activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.basic_4 = AMCBlock(mid_channels * 8, mid_channels * 4,
                                activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.down_conv_4_8 = Convolution(2, mid_channels * 4, mid_channels * 8,
                                         kernel_size=2, strides=2, padding=0)
        self.down_conv_basic_4_8 = Convolution(2, mid_channels * 4, mid_channels * 8,
                                               kernel_size=2, strides=2, padding=0)
        self.conv_8 = Convolution(2, in_channels, mid_channels * 8,
                                  kernel_size=3, act=activation, adn_ordering=adn_ordering, norm=norm)
        self.group_weight_8 = GroupWeightBlock(mid_channels * 16, mid_channels * 8,
                                               activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.basic_8 = AMCBlock(mid_channels * 16, mid_channels * 8,
                                activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.down_conv_8_16 = Convolution(2, mid_channels * 8, mid_channels * 16,
                                          kernel_size=2, strides=2, padding=0)
        self.down_conv_basic_8_16 = Convolution(2, mid_channels * 8, mid_channels * 16,
                                                kernel_size=2, strides=2, padding=0)

        self.basic_16 = AMCBlock(mid_channels * 32, mid_channels * 16,
                                 activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.down_conv_16_32 = Convolution(2, mid_channels * 16, mid_channels * 32,
                                           kernel_size=2, strides=2, padding=0)
        self.down_conv_basic_16_32 = Convolution(2, mid_channels * 16, mid_channels * 32,
                                                 kernel_size=2, strides=2, padding=0)
        self.bottom_residual_block_ = ResidualBlock(mid_channels * 64, mid_channels * 64)
        self.bottom_residual_block__ = ResidualBlock(mid_channels * 64, mid_channels * 32)
        self.bottom_residual_block___ = ResidualBlock(mid_channels * 32, mid_channels * 32)
        self.bottom_residual_block____ = ResidualBlock(mid_channels * 32, mid_channels * 16)

        self.up_sample_32_16 = UpSample(spatial_dims=2, scale_factor=up_sample_size, mode="nontrainable",
                                        interp_mode=interpolation_mode)
        self.up_group_weight_16 = GroupWeightBlock(mid_channels * 32, mid_channels * 16,
                                                   activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.up_basic_16 = AMCBlock(mid_channels * 16, mid_channels * 8,
                                    activation=activation, adn_ordering=adn_ordering, norm=norm)

        self.up_sample_16_8 = UpSample(spatial_dims=2, scale_factor=up_sample_size, mode="nontrainable",
                                       interp_mode=interpolation_mode)
        self.up_group_weight_8 = GroupWeightBlock(mid_channels * 16, mid_channels * 8,
                                                  activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.up_basic_8 = AMCBlock(mid_channels * 8, mid_channels * 4,
                                   activation=activation, adn_ordering=adn_ordering, norm=norm)

        self.up_sample_8_4 = UpSample(spatial_dims=2, scale_factor=up_sample_size, mode="nontrainable",
                                      interp_mode=interpolation_mode)
        self.up_group_weight_4 = GroupWeightBlock(mid_channels * 8, mid_channels * 4,
                                                  activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.up_basic_4 = AMCBlock(mid_channels * 4, mid_channels * 2,
                                   activation=activation, adn_ordering=adn_ordering, norm=norm)

        self.up_sample_4_2 = UpSample(spatial_dims=2, scale_factor=up_sample_size, mode="nontrainable",
                                      interp_mode=interpolation_mode)
        self.up_group_weight_2 = GroupWeightBlock(mid_channels * 4, mid_channels * 2,
                                                  activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.up_basic_2 = AMCBlock(mid_channels * 2, mid_channels,
                                     activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.up_sample_2_1 = UpSample(spatial_dims=2, scale_factor=up_sample_size, mode="nontrainable",
                                      interp_mode=interpolation_mode)
        self.basic_1 = AMCBlock(mid_channels * 2, mid_channels,
                                activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.output_group_weight_1 = GroupWeightBlock(mid_channels, out_channels,
                                                      activation=activation, adn_ordering=adn_ordering, norm=norm)
        self.output_layer = nn.Sigmoid()
        self.content_up_16_8 = TransposeConvolutionBlock(mid_channels * 8, mid_channels * 4)
        self.content_up_8_4 = TransposeConvolutionBlock(mid_channels * 4, mid_channels * 2)
        self.content_up_4_2 = TransposeConvolutionBlock(mid_channels * 2, mid_channels * 1)
        self.content_up_2_1 = TransposeConvolutionBlock(mid_channels * 1, 3)

    def forward(self, x_1):
        conv_1 = self.conv_1(x_1)
        down_conv_1_2 = self.down_conv_1_2(conv_1)

        x_2 = self.down_sample_1_2(x_1)
        conv_2 = self.conv_2(x_2)
        concat_1_2 = torch.cat((down_conv_1_2, conv_2), dim=1)
        group_weight_2 = self.group_weight_2(concat_1_2)
        basic_2 = self.basic_2(group_weight_2)
        down_conv_2_4 = self.down_conv_2_4(group_weight_2)
        down_conv_basic_2_4 = self.down_conv_basic_2_4(basic_2)

        x_4 = self.down_sample_2_4(x_2)
        conv_4 = self.conv_4(x_4)
        concat_2_4 = torch.cat((down_conv_2_4, conv_4), dim=1)
        group_weight_4 = self.group_weight_4(concat_2_4)
        concat_group_weight_4 = torch.cat((down_conv_basic_2_4, group_weight_4), dim=1)
        basic_4 = self.basic_4(concat_group_weight_4)
        down_conv_4_8 = self.down_conv_4_8(group_weight_4)
        down_conv_basic_4_8 = self.down_conv_basic_4_8(basic_4)

        x_8 = self.down_sample_4_8(x_4)
        conv_8 = self.conv_8(x_8)
        concat_4_8 = torch.cat((down_conv_4_8, conv_8), dim=1)
        group_weight_8 = self.group_weight_8(concat_4_8)
        concat_group_weight_8 = torch.cat((down_conv_basic_4_8, group_weight_8), dim=1)
        basic_8 = self.basic_8(concat_group_weight_8)
        down_conv_8_16 = self.down_conv_8_16(group_weight_8)
        down_conv_basic_8_16 = self.down_conv_basic_8_16(basic_8)

        concat_8_16 = torch.cat((down_conv_8_16, down_conv_basic_8_16), dim=1)
        basic_16 = self.basic_16(concat_8_16)
        down_conv_16_32 = self.down_conv_16_32(down_conv_8_16)
        down_conv_basic_16_32 = self.down_conv_basic_16_32(basic_16)

        concat_16_32 = torch.cat((down_conv_16_32, down_conv_basic_16_32), dim=1)
        bottom_residual_block_ = self.bottom_residual_block_(concat_16_32)
        bottom_residual_block__ = self.bottom_residual_block__(bottom_residual_block_)
        bottom_residual_block___ = self.bottom_residual_block___(bottom_residual_block__)
        bottom_residual_block____ = self.bottom_residual_block____(bottom_residual_block___)

        up_sample_32_16 = self.up_sample_32_16(bottom_residual_block____)
        up_concat_32_16 = torch.cat((basic_16, up_sample_32_16), dim=1)
        up_group_weight_16 = self.up_group_weight_16(up_concat_32_16)
        up_basic_16 = self.up_basic_16(up_group_weight_16)

        up_sample_16_8 = self.up_sample_16_8(up_basic_16)
        up_concat_16_8 = torch.cat((basic_8, up_sample_16_8), dim=1)
        up_group_weight_8 = self.up_group_weight_8(up_concat_16_8)
        up_basic_8 = self.up_basic_8(up_group_weight_8)

        up_sample_8_4 = self.up_sample_8_4(up_basic_8)
        up_concat_8_4 = torch.cat((basic_4, up_sample_8_4), dim=1)
        up_group_weight_4 = self.up_group_weight_4(up_concat_8_4)
        up_basic_4 = self.up_basic_4(up_group_weight_4)

        up_sample_4_2 = self.up_sample_4_2(up_basic_4)
        up_concat_4_2 = torch.cat((basic_2, up_sample_4_2), dim=1)
        up_group_weight_2 = self.up_group_weight_2(up_concat_4_2)
        up_basic_2 = self.up_basic_2(up_group_weight_2)

        up_sample_2_1 = self.up_sample_2_1(up_basic_2)
        up_concat_2_1 = torch.cat((conv_1, up_sample_2_1), dim=1)
        basic_1 = self.basic_1(up_concat_2_1)
        output_group_weight_1 = self.output_group_weight_1(basic_1)
        mask = self.output_layer(output_group_weight_1)

        content_up_16_8 = self.content_up_16_8(up_basic_16)
        content_up_8_4 = self.content_up_8_4(content_up_16_8)
        content_up_4_2 = self.content_up_4_2(content_up_8_4)
        content_up_2_1 = self.content_up_2_1(content_up_4_2)

        triple_mask = mask.repeat(1, 3, 1, 1)
        background = (1 - triple_mask) * x_1
        poly = triple_mask * content_up_2_1
        re_image = background + poly
        return mask, re_image


if __name__ == "__main__":
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    input = torch.randn((1, 3, 352, 352)).to(device)
    net = AMC_Net(3, 1).to(device)
    flops, params = profile(net, inputs=(input,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))

