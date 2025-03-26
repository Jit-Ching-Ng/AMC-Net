import torch
import torch.nn as nn
from monai.networks.layers import get_norm_layer


class TransposeConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="INSTANCE"):
        super(TransposeConvolutionBlock, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.merge_norm = get_norm_layer(name=norm, spatial_dims=2, channels=out_channels)
        self.activation_function = nn.GELU()

    def forward(self, x):
        transpose_conv = self.transpose_conv(x)
        return self.activation_function(self.merge_norm(transpose_conv))


if __name__ == "__main__":
    input = torch.randn((1, 4, 64, 64))
    net = TransposeConvolutionBlock(4, 2)
    output = net(input)
    print(output.shape)
