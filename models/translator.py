import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FcBnReLU(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None
    ):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias, device=device
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FcBn(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias, device=device
        )
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x


class FcReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias, device=device
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class FcSigmoid(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias, device=device
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class ResAttBlock(nn.Module):
    def __init__(self, in_features, bias=True, device=None):
        super().__init__()
        self.fc_bn_relu = FcBnReLU(
            in_features=in_features, out_features=1024, bias=bias, device=device
        )
        self.fc_bn = FcBn(
            in_features=1024,
            out_features=512,
            bias=bias,
            device=device,
        )
        self.fc_relu = FcReLU(
            in_features=512,
            out_features=16,
            bias=bias,
            device=device,
        )
        self.fc_sigmoid = FcSigmoid(
            in_features=16,
            out_features=512,
            bias=bias,
            device=device,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc_bn_relu(x)
        residual = self.fc_bn(x)
        x = self.fc_relu(residual)
        x = self.fc_sigmoid(x)
        x = x * residual  # Element wise product
        x = self.relu(x)
        return x


class Translator(nn.Module):
    def __init__(self, out_features=37, bias=True, device=None):
        super().__init__()
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval()
        self.fc1 = nn.Linear(
            in_features=512, out_features=512, bias=bias, device=device
        )
        self.res_att1 = ResAttBlock(
            in_features=512,
            bias=bias,
            device=device,
        )
        self.res_att2 = ResAttBlock(
            in_features=512,
            bias=bias,
            device=device,
        )
        self.res_att3 = ResAttBlock(
            in_features=512,
            bias=bias,
            device=device,
        )
        self.fc2 = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        embed = self.encoder(x)
        x = self.fc1(embed)
        residual1 = self.res_att1(x)
        residual1 = residual1 + x
        residual2 = self.res_att2(residual1)
        residual2 = residual2 + residual1
        residual3 = self.res_att3(residual2)
        residual3 = residual3 + residual2
        x = self.fc2(residual3)
        return x, embed
