import torch.nn.functional as F
import torch
from torch import nn
from spikingjelly.activation_based import layer, neuron, surrogate
from .base import BaseJellyNet


class SpikingConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(SpikingConvBlock, self).__init__()
        self.conv = nn.Sequential(
            layer.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, padding=padding, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(out_channels, out_channels,
                         kernel_size=kernel_size, padding=padding, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        return self.conv(x)


class SpikingEncoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            layer.MaxPool2d(2, 2),
            SpikingConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class SpikingDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.transpose = layer.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = SpikingConvBlock(in_channels + out_channels,
                                     out_channels, kernel_size, padding)

    def forward(self, x1, x2):
        x1 = self.transpose(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SpikingUNet(BaseJellyNet):
    def __init__(self, n_input: int = 1, n_output: int = 1, n_steps: int = 5, encoding_method: str = 'direct'):
        super(SpikingUNet, self).__init__(
            n_input, n_output, n_steps, encoding_method)

        self.in_conv = SpikingConvBlock(n_input, 64)

        self.enc_1 = SpikingEncoder(64, 128)
        self.enc_2 = SpikingEncoder(128, 256)
        self.enc_3 = SpikingEncoder(256, 512)

        self.dec_3 = SpikingDecoder(512, 256)
        self.dec_2 = SpikingDecoder(256, 128)
        self.dec_1 = SpikingDecoder(128, 64)

        self.out_conv = SpikingConvBlock(
            64, n_output, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)

        x = self.dec_3(x4, x3)
        x = self.dec_2(x, x2)
        x = self.dec_1(x, x1)

        return self.out_conv(x)
