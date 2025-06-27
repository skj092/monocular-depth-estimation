import torch
import torch.nn as nn
import torch.nn.functional as F

class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.convB = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.reluA = nn.LeakyReLU(0.2)
        self.reluB = nn.LeakyReLU(0.2)
        self.bn2a = nn.BatchNorm2d(out_channels)
        self.bn2b = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        d = self.convA(x)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x = x + d
        p = self.pool(x)
        return x, p


class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.convB = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.reluA = nn.LeakyReLU(0.2)
        self.reluB = nn.LeakyReLU(0.2)
        self.bn2a = nn.BatchNorm2d(out_channels)
        self.bn2b = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convA(x)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.convA = nn.Conv2d(channels, channels, kernel_size, stride, padding)
        self.convB = nn.Conv2d(channels, channels, kernel_size, stride, padding)
        self.reluA = nn.LeakyReLU(0.2)
        self.reluB = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x


class DepthEstimationModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        filters = [16, 32, 64, 128, 256]

        self.down1 = DownscaleBlock(in_channels, filters[0])           # 3 → 16
        self.down2 = DownscaleBlock(filters[0], filters[1])            # 16 → 32
        self.down3 = DownscaleBlock(filters[1], filters[2])            # 32 → 64
        self.down4 = DownscaleBlock(filters[2], filters[3])            # 64 → 128

        self.bottleneck = BottleNeckBlock(filters[3])                  # 128 → 128

        self.up1 = UpscaleBlock(filters[3] + filters[3], filters[2])   # 128 + 128 → 64
        self.up2 = UpscaleBlock(filters[2] + filters[2], filters[1])   # 64 + 64 → 32
        self.up3 = UpscaleBlock(filters[1] + filters[1], filters[0])   # 32 + 32 → 16
        self.up4 = UpscaleBlock(filters[0] + filters[0], filters[0])   # 16 + 16 → 16

        self.final_conv = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)

        b = self.bottleneck(p4)

        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        x = self.final_conv(x)
        x = self.activation(x)
        return x

