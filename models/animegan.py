import torch.nn as nn
import torch.nn.functional as F


class CustomConvBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_dim)
        self.non_linearity = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        return x


class DSConv(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bottleneck = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.depthwise_conv = nn.Conv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_dim,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_dim)
        self.non_linearity = nn.LeakyReLU()
        self.conv_block = CustomConvBlock(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.conv_block(x)
        return x


class InvertedResidualBlockWithResidual(nn.Module):
    def __init__(self, in_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_block = CustomConvBlock(in_dim, 512, 1, 1, 0)
        self.depthwise_conv = nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)
        self.norm1 = nn.InstanceNorm2d(512)
        self.non_linearity = nn.LeakyReLU()
        self.conv = nn.Conv2d(512, in_dim, 1, 1)
        self.norm2 = nn.InstanceNorm2d(in_dim)

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x = self.depthwise_conv(x)
        x = self.norm1(x)
        x = self.non_linearity(x)
        x = self.conv(x)
        x = self.norm2(x)
        x += residual
        return x


class DownConv(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ds_conv1 = DSConv(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.ds_conv2 = DSConv(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.ds_conv1(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.ds_conv2(x)
        x += residual
        return x


class Upconv(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ds_conv = DSConv(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.ds_conv(x)
        return x


class AnimeGenerator(nn.Module):
    def __init__(self, in_dim=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            CustomConvBlock(in_dim, 64),
            CustomConvBlock(64, 64),
            DownConv(64, 128),
            CustomConvBlock(128, 128),
            DSConv(128, 128),
            DownConv(128, 256),
            CustomConvBlock(256, 256),
        )
        self.irb = nn.Sequential(
            *[InvertedResidualBlockWithResidual(256) for _ in range(8)]
        )
        self.decoder = nn.Sequential(
            CustomConvBlock(256, 256),
            Upconv(256, 128),
            DSConv(128, 128),
            CustomConvBlock(128, 128),
            Upconv(128, 64),
            CustomConvBlock(64, 64),
            CustomConvBlock(64, 64),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.irb(x)
        x = self.decoder(x)
        return x


class AnimeDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.InstanceNorm2d(256), nn.LeakyReLU()
        )
        self.layer5 = nn.Conv2d(256, 1, 3, 1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class ConvNormLReLU(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        pad_mode="reflect",
        groups=1,
        bias=False,
    ):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(
            ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True)
        )
        # pw
        layers.append(
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False)
        )
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64),
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128),
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(ConvNormLReLU(128, 128), ConvNormLReLU(128, 128))

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False), nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=False
            )
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(
                out, input.size()[-2:], mode="bilinear", align_corners=True
            )
        else:
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=False
            )
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


def main():
    import sys

    print(sys.path)


if __name__ == "__main__":
    main()