from fastai.vision.all import *
import pandas as pd
import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split
from fastai.callback.wandb import *
import wandb
import mlflow

wandb.login(key="97b5307e24cc3a77259ade3057e4eea6fd2addb0")
wandb.init(project="depth-estimation", name="depth_estimation_experiment")

import mlflow
from fastai.callback.core import Callback

class MLflowCallback(Callback):
    "Minimal MLflow integration for fastai"
    order = Recorder.order + 1

    def __init__(self, log_model: bool = True, model_name: str = "model"):
        self.log_model = log_model
        self.model_name = model_name

    def before_fit(self):
        if mlflow.active_run() is None:
            mlflow.start_run()
        self.run = True

    def after_epoch(self):
        logs = {'epoch': self.epoch}
        names = self.recorder.metric_names
        values = self.recorder.log
        logs.update({k: v for k, v in zip(names, values) if k not in ('epoch', 'time')})
        mlflow.log_metrics(logs, step=self.epoch)

    def after_fit(self):
        if self.log_model:
            # Save and log model weights
            fname = f'{self.model_name}.pth'
            path = self.learn.path / fname
            torch.save(self.learn.model.state_dict(), path)
            mlflow.log_artifact(str(path))
            print(f"Logged model to MLflow: {path}")
        mlflow.end_run()




# Create a DataFrame with image paths and corresponding depth and mask files
path = "val_extracted/val/indoors"

filelist = []
for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root, file))

filelist.sort()
data = {
    "image": [x for x in filelist if x.endswith(".png")],
    "depth": [x for x in filelist if x.endswith("_depth.npy")],
    "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
}
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

def open_image(fn):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    return PILImage.create(img)

def load_depth_masked(row):
    depth = np.load(row['depth']).squeeze()
    mask = np.load(row['mask']) > 0

    max_depth = min(300, np.percentile(depth, 99))
    depth = np.clip(depth, 0.1, max_depth)
    depth = np.log(depth, where=mask)
    depth = np.ma.masked_where(~mask, depth)
    depth = np.clip(depth, 0.1, np.log(max_depth))
    depth = cv2.resize(depth, (256, 256))

    # Normalize to [0, 255] and convert to uint8
    depth_norm = (depth / np.log(max_depth)) * 255
    depth_uint8 = depth_norm.astype(np.uint8)

    return PILImageBW.create(depth_uint8)

def get_x(row): return open_image(row['image'])
def get_y(row): return load_depth_masked(row)

dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=IndexSplitter(valid_df.index),
    item_tfms=Resize((256, 256))
)

dls = dblock.dataloaders(pd.concat([train_df, valid_df]), bs=32, num_workers=4)

class DepthWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):  # FastAI expects output shape = input shape
        out = self.model(x)
        return F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)


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
    # summary = torchsummary.summary(model, (3, 304, 228), device='cpu')
    # print(summary)
    batch = torch.randn(16, 3, 304, 228)
    output = model(batch)
    print(f"Output shape: {output.shape}")

import torch
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

# Optional: SSIM implementation
import pytorch_msssim

def image_gradients(image):
    if image.ndim != 4:
        raise ValueError(
            f"image_gradients expects a 4D tensor [B, C, H, W], not {image.shape}."
        )

    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]

    # Pad to retain original shape
    dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0)
    dx = F.pad(dx, (0, 1, 0, 0), mode="constant", value=0)

    return dy, dx

def calculate_loss(target, pred, ssim_loss_weight=1.0, l1_loss_weight=1.0, edge_loss_weight=1.0, max_val=1.0):
    # Image gradients
    dy_true, dx_true = image_gradients(target)
    dy_pred, dx_pred = image_gradients(pred)

    weights_x = torch.exp(torch.mean(torch.abs(dx_true)))
    weights_y = torch.exp(torch.mean(torch.abs(dy_true)))

    # Smoothness loss
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    depth_smoothness_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

    # SSIM loss (assuming images are in [0, 1] range)
    ssim_loss = 1 - pytorch_msssim.ssim(pred, target, data_range=max_val, size_average=True)

    # L1 Loss
    l1_loss = torch.mean(torch.abs(target - pred))

    # Final loss
    loss = (
        ssim_loss_weight * ssim_loss
        + l1_loss_weight * l1_loss
        + edge_loss_weight * depth_smoothness_loss
    )
    return loss

if __name__ == "__main__":
    # Example usage
    target = torch.rand(1, 3, 256, 256)  # Random target image
    pred = torch.rand(1, 3, 256, 256)    # Random predicted image

    loss = calculate_loss(target, pred)
    print(f"Calculated loss: {loss.item()}")

class CombinedDepthLoss:
    def __init__(self, ssim_loss_weight=1.0, l1_loss_weight=1.0, edge_loss_weight=1.0):
        self.ssim_loss_weight = ssim_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.edge_loss_weight = edge_loss_weight

    def __call__(self, pred, target):
        return calculate_loss(
            target,
            pred,
            ssim_loss_weight=self.ssim_loss_weight,
            l1_loss_weight=self.l1_loss_weight,
            edge_loss_weight=self.edge_loss_weight,
            max_val=1.0
        )

model = DepthWrapper(ResNet50UpProj(num_classes=1))

loss_func = CombinedDepthLoss(
    ssim_loss_weight=1.0,
    l1_loss_weight=1.0,
    edge_loss_weight=1.0
)

from fastai.metrics import mae, rmse

learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=Adam,
    metrics=[mae, rmse],
    cbs=[WandbCallback(log_preds=True, log_model=True)])

learn.fit_one_cycle(3, 1e-4)

def visualize_depth_map(samples, test=False, model=None):
    input, target = samples
    fig, ax = plt.subplots(6, 3, figsize=(18, 36))
    for i in range(6):
        ax[i, 0].imshow(input[i].permute(1, 2, 0).cpu().numpy())
        ax[i, 0].set_title("Input RGB")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(target[i].squeeze().cpu().numpy(), cmap="inferno")
        ax[i, 1].set_title("Ground Truth")
        ax[i, 1].axis("off")

        if test and model:
            with torch.no_grad():
                pred = model(input.to(next(model.parameters()).device))
                if pred.shape != target.shape:
                    pred = torch.nn.functional.interpolate(
                        pred, size=target.shape[-2:], mode='bilinear', align_corners=False
                    )
            ax[i, 2].imshow(pred[i].squeeze().cpu().numpy(), cmap="inferno")
            ax[i, 2].set_title("Prediction")
            ax[i, 2].axis("off")
        else:
            ax[i, 2].axis("off")

    plt.tight_layout()
    return fig

# Log predictions
xb, yb = dls.valid.one_batch()
preds = learn.model(xb)
fig = visualize_depth_map((xb.cpu(), yb.cpu()), test=True, model=learn.model)
fig.savefig("fastai_sample_preds.png")
