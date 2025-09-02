import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Conditioner(nn.Module):
    def __init__(self, in_dim=960, code_dim=53, cond_dim=128, use_z=False, z_dim=0):
        super().__init__()
        self.use_z = use_z
        self.z_dim = z_dim
        self.to_code = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, code_dim),
        )
        self.to_cond = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, cond_dim)
        )

    def forward(self, labels, noise_sigma=0.0):
        if noise_sigma > 0:
            labels = labels + noise_sigma * torch.randn_like(labels)
        code = self.to_code(labels)
        cond_emb = self.to_cond(labels)
        return code, cond_emb


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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.instance_norm(x)
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


class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_channels)
        self.to_beta = nn.Linear(cond_dim, num_channels)
        # --- Identity init ---
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h, cond_emb):
        gamma = self.to_gamma(cond_emb).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta = self.to_beta(cond_emb).unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta


class Imitator(nn.Module):
    def __init__(self, latent_dim=960, cond_dim=128):
        super().__init__()
        self.conditioner = Conditioner(
            in_dim=latent_dim, code_dim=512, cond_dim=cond_dim
        )

        self.deconv1 = TransUpConv(512, 512, 4, 1, 0)  # x4
        self.flim1 = FiLM(cond_dim, 512)

        self.deconv2 = UpConv(512, 512)  # x 8
        self.flim2 = FiLM(cond_dim, 512)

        self.deconv3 = UpConv(512, 512)  # x 16
        self.flim3 = FiLM(cond_dim, 512)

        self.deconv4 = UpConv(512, 256)  # x 32
        self.flim4 = FiLM(cond_dim, 256)

        self.deconv5 = UpConv(256, 256)  # x 64
        self.flim5 = FiLM(cond_dim, 256)

        self.deconv6 = UpConv(256, 128)  # x 128
        self.flim6 = FiLM(cond_dim, 128)

        self.deconv7 = UpConv(128, 64)  # x 256
        self.flim7 = FiLM(cond_dim, 64)

        self.deconv8 = OutConv(64, 3)  # x 512
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, noise_sigma=0.0):
        x, c = self.conditioner(x, noise_sigma=noise_sigma)

        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv1(x)
        x = self.flim1(x, c)
        x = self.act(x)

        x = self.deconv2(x)
        x = self.flim2(x, c)
        x = self.act(x)

        x = self.deconv3(x)
        x = self.flim3(x, c)
        x = self.act(x)

        x = self.deconv4(x)
        x = self.flim4(x, c)
        x = self.act(x)

        x = self.deconv5(x)
        x = self.flim5(x, c)
        x = self.act(x)

        x = self.deconv6(x)
        x = self.flim6(x, c)
        x = self.act(x)

        x = self.deconv7(x)
        x = self.flim7(x, c)
        x = self.act(x)

        x = self.deconv8(x)
        return x


class DDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ProjectionDiscriminator(nn.Module):
    """
    입력:  x=[B,3,512,512], label=[B,960]
    출력:  logit=[B,1], feat=[B,feat_dim]  # feat는 (선택) feature matching용
    """

    def __init__(self, in_ch=3, base_ch=64, cond_in=960, cond_dim=256, feat_dim=1024):
        super().__init__()
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.backbone = nn.Sequential(
            DDownBlock(in_ch, base_ch),  # 512 -> 256
            DDownBlock(base_ch, base_ch * 2),  # 256 -> 128
            DDownBlock(base_ch * 2, base_ch * 4),  # 128 -> 64
            DDownBlock(base_ch * 4, base_ch * 4),  # 64  -> 32
            DDownBlock(base_ch * 4, base_ch * 8),  # 32  -> 16
            DDownBlock(base_ch * 8, base_ch * 8),  # 16  -> 8
            DDownBlock(base_ch * 8, base_ch * 16),  # 8   -> 4
        )
        # 4x4 -> 1x1 -> feat
        self.to_feat = nn.Sequential(
            spectral_norm(
                nn.Conv2d(base_ch * 16, feat_dim, kernel_size=4, stride=1, padding=0)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 기본 스칼라 로짓
        self.lin = spectral_norm(nn.Linear(feat_dim, 1))

        # 조건 임베딩 투사: 960 -> cond_dim -> feat_dim (Wc)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, 512),
            nn.SiLU(),
            nn.Linear(512, cond_dim),
        )
        self.embed = spectral_norm(nn.Linear(cond_dim, feat_dim, bias=False))

    def forward(self, x, label):
        h = self.backbone(x)  # [B, C, 4, 4]
        f = self.to_feat(h).flatten(1)  # [B, feat_dim]
        base = self.lin(f)  # [B, 1]

        c = self.cond_proj(label)  # [B, cond_dim]
        wc = self.embed(c)  # [B, feat_dim]
        proj = (f * wc).sum(dim=1, keepdim=True)  # [B, 1]

        logit = base + proj
        return logit, f
