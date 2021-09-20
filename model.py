import torch
import torch.nn.functional as F

from torch import nn, cat
from torchvision import models
from torch.utils.checkpoint import checkpoint_sequential


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, e=None):
        d = F.interpolate(x, scale_factor=2, mode='bilinear')
        if e is not None:
            d = cat([d, e], 1)

        return self.double_conv(d)


class UnetResnet34(nn.Module):
    def __init__(self, pretrained=True, bias=True, checkpotinting=False):
        super().__init__()
        self.checkpointing = checkpotinting
        backbone = models.resnet34(pretrained=pretrained)

        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderBlock(256 + 512, 512, 256, bias=bias)
        self.decoder4 = DecoderBlock(256 + 256, 256, 128, bias=bias)
        self.decoder3 = DecoderBlock(128 + 128, 128, 64, bias=bias)
        self.decoder2 = DecoderBlock(64 + 64, 128, 64, bias=bias)
        self.decoder1 = DecoderBlock(64, 64, 32, bias=bias)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = checkpoint_sequential(self.encoder2, 3, e1) if self.checkpointing else self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        b = self.bottleneck(e5)

        d5 = self.decoder5(b, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        return torch.sigmoid(self.final_conv(d1))
