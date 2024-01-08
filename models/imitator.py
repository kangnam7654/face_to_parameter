import torch
import torch.nn as nn
import torch.nn.functional as F


class TransUpConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.nonlinear(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.instance_norm = nn.InstanceNorm2d(out_dim)
        self.nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.nonlinear(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.nonlinear = nn.Tanh()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.nonlinear(x)
        return x


class Imitator(nn.Module):
    def __init__(self, latent_dim=37):
        super().__init__()
        self.deconv1 = TransUpConv(latent_dim, 512, 4, 1, 0)  # x4
        self.deconv2 = UpConv(512, 512)  # x 8
        self.deconv3 = UpConv(512, 512)  # x 16
        self.deconv4 = UpConv(512, 256)  # x 32
        self.deconv5 = UpConv(256, 128)  # x 64
        self.deconv6 = UpConv(128, 64)  # x 128
        self.deconv7 = OutConv(64, 3)  # x 256

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        return x
