import torch
import torch.nn as nn


class UpConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, last=False):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(out_dim, out_dim)
        self.non_linearity = (
            nn.LeakyReLU(negative_slope=0.2, inplace=True) if not last else nn.Tanh()
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        return x
    
class UpConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, last=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(out_dim, out_dim)
        self.non_linearity = (
            nn.LeakyReLU(negative_slope=0.2, inplace=True) if not last else nn.Tanh()
        )

    def forward(self, x):
        # x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        return x



class Imitator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.deconv0 = nn.ConvTranspose2d(latent_dim, 512, 4, 1)
        self.deconv1 = UpConv(512, 512)
        self.deconv2 = UpConv(512, 256)
        self.deconv3 = UpConv(256, 256)
        self.deconv4 = UpConv(256, 128)
        self.deconv5 = UpConv(128, 128)
        self.deconv6 = UpConv(128, 64)
        self.deconv7 = UpConv(64, 3, last=True)

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        return x
    
    

def test():
    model = Imitator(latent_dim=37)
    z = torch.randn(1, 37)
    out = model(z)
    print(out.shape)


if __name__ == "__main__":
    test()