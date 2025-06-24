import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bnA = nn.BatchNorm2d(out_channels)
        self.convB = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnB = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d = self.relu(self.bnA(self.convA(x)))
        x = self.relu(self.bnB(self.convB(d)))
        x = x + d  # additive skip connection
        p = self.pool(x)
        return x, p

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.convA = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bnA = nn.BatchNorm2d(out_channels)
        self.convB = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnB = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, skip):
        # Instead of self.up = nn.Upsample(scale_factor=2)
        x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bnA(self.convA(x)))
        x = self.relu(self.bnB(self.convB(x)))
        return x

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.reluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.reluB = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.reluA(self.convA(x))
        x = self.reluB(self.convB(x))
        return x

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [16, 32, 64, 128, 256]
        self.down1 = DownscaleBlock(3, filters[0])
        self.down2 = DownscaleBlock(filters[0], filters[1])
        self.down3 = DownscaleBlock(filters[1], filters[2])
        self.down4 = DownscaleBlock(filters[2], filters[3])

        self.bottleneck = BottleNeckBlock(filters[3], filters[4])

        self.up1 = UpscaleBlock(filters[4], filters[3], filters[3])
        self.up2 = UpscaleBlock(filters[3], filters[2], filters[2])
        self.up3 = UpscaleBlock(filters[2], filters[1], filters[1])
        self.up4 = UpscaleBlock(filters[1], filters[0], filters[0])

        self.final_conv = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)

        bn = self.bottleneck(p4)

        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        out = self.final_conv(u4)
        return self.tanh(out)

if __name__ == "__main__":
    model = DepthEstimationModel()
    summary = torchsummary.summary(model, (3, 224, 224), device='cpu')
    print(summary)
    batch = torch.randn(16, 3, 304, 228)
    output = model(batch)
    print(f"Output shape: {output.shape}")

