# https://github.com/irolaina/FCRN-DepthPrediction/blob/master/tensorflow/models/fcrn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out

class UpProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class ResNet50UpProj(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Transition
        self.trans = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, bias=True),
            nn.BatchNorm2d(1024),
        )

        # Up-projection layers
        self.up1 = UpProjectionBlock(1024, 512)
        self.up2 = UpProjectionBlock(512, 256)
        self.up3 = UpProjectionBlock(256, 128)
        self.up4 = UpProjectionBlock(128, 64)

        self.final = nn.Sequential(
            nn.Dropout(0.0),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def _make_layer(self, mid_channels, blocks, stride=1):
        downsample = None
        out_channels = mid_channels * Bottleneck.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # res2
        x = self.layer2(x)  # res3
        x = self.layer3(x)  # res4
        x = self.layer4(x)  # res5

        x = self.trans(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x

if __name__ == "__main__":
    model = ResNet50UpProj(num_classes=1)
    summary = torchsummary.summary(model, (3, 304, 228), device='cpu')
    print(summary)
    batch = torch.randn(16, 3, 304, 228)
    output = model(batch)
    print(f"Output shape: {output.shape}")
