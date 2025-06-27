import pandas as pd
import numpy as np
import cv2
import os
import torch
from sklearn.model_selection import train_test_split
import wandb
# from dotenv import load_dotenv
from loss_fns import calculate_loss
from model2 import DepthEstimationModel
from fastai.vision.all import *
import keras
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

# Data Download
# annotation_folder = "/dataset/"
# if not os.path.exists(os.path.abspath(".") + annotation_folder):
#     annotation_zip = keras.utils.get_file(
#         "val.tar.gz",
#         cache_subdir=os.path.abspath("."),
#         origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
#         extract=True,
#     )

# wandb.login(key=os.getenv("WANDB_API_KEY"))
# wandb.init(project="DepthEstimation", name="FastAI_Training",)

# load_dotenv()


# data loading and preprocessing
train_path = "val_extracted/val/indoors"
filelist = []
for root, dirs, files in os.walk(train_path):
    for file in files:
        filelist.append(os.path.join(root, file))

filelist.sort()

images = sorted([x for x in filelist if x.endswith(".png")])
depths = sorted([x for x in filelist if x.endswith("_depth.npy")])
masks = sorted([x for x in filelist if x.endswith("_depth_mask.npy")])

# Get base names
depth_bases = {os.path.basename(x).replace("_depth.npy", "") for x in depths}
mask_bases = {os.path.basename(x).replace("_depth_mask.npy", "") for x in masks}
valid_bases = depth_bases & mask_bases

# Filter valid files only
valid_images = [x for x in images if os.path.basename(x).replace(".png", "") in valid_bases]
valid_depths = [x for x in depths if os.path.basename(x).replace("_depth.npy", "") in valid_bases]
valid_masks = [x for x in masks if os.path.basename(x).replace("_depth_mask.npy", "") in valid_bases]

# Build DataFrame
data = {
    "image": valid_images,
    "depth": valid_depths,
    "mask": valid_masks,
}
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total train samples: {len(df)}")
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_df)}, Validation samples: {len(valid_df)}")

# Dataset and DataLoader setup
def open_image(fn):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    return img

def load_depth_masked(row):
    depth = np.load(row['depth']).squeeze()
    mask = np.load(row['mask']) > 0
    epsilon = 1e-6
    max_depth = min(300, np.percentile(depth[mask], 99))
    depth = np.clip(depth, epsilon, max_depth)
    depth_log = np.zeros_like(depth)
    depth_log[mask] = np.log(depth[mask])
    depth_log = np.clip(depth_log, np.log(epsilon), np.log(max_depth))
    depth_log = cv2.resize(depth_log, (256, 256))
    with np.errstate(invalid='ignore', divide='ignore'):
        depth_norm = (depth_log / np.log(max_depth))
        depth_norm = np.nan_to_num(depth_norm, nan=0.0, posinf=0.0, neginf=0.0)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return depth_uint8

def get_x(row): return open_image(row['image'])
def get_y(row): return load_depth_masked(row)

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = get_x(row)
        y = get_y(row)
        x = self.transform(x)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0) / 255.0
        return x, y

train_ds = DepthDataset(train_df)
a, b = train_ds[0]
valid_ds = DepthDataset(valid_df)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=4)
xb, yb = next(iter(train_dl))
print(f"Batch shape: {xb.shape}, {yb.shape}")

# Model and loss function
class DepthWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    def forward(self, x):
        out = self.model(x)
        return F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)

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

loss_func = CombinedDepthLoss(ssim_loss_weight=1.0, l1_loss_weight=1.0, edge_loss_weight=1.0)

# Metrics
def abs_rel(pred, target):
    pred, target = pred.squeeze(), target.squeeze()
    mask = target > 1e-3  # Stricter threshold to avoid division by near-zero
    if mask.sum() == 0:
        return float('nan')
    return torch.mean(torch.abs(pred[mask] - target[mask]) / target[mask])

def log10_mae(pred, target):
    pred, target = pred.squeeze(), target.squeeze()
    mask = (target > 1e-3) & (pred > 1e-3)  # Mask non-positive values
    if mask.sum() == 0:
        return float('nan')
    return torch.mean(torch.abs(torch.log10(pred[mask]) - torch.log10(target[mask])))

def log10_rmse(pred, target):
    pred, target = pred.squeeze(), target.squeeze()
    mask = (target > 1e-3) & (pred > 1e-3)
    if mask.sum() == 0:
        return float('nan')
    return torch.sqrt(torch.mean((torch.log10(pred[mask]) - torch.log10(target[mask])) ** 2))

def threshold_accuracy(thresh):
    def _inner(pred, target):
        pred, target = pred.squeeze(), target.squeeze()
        mask = target > 1e-3
        if mask.sum() == 0:
            return float('nan')
        ratio = torch.max(pred[mask] / target[mask], target[mask] / pred[mask])
        return torch.mean((ratio < thresh).float())
    return _inner

delta1 = threshold_accuracy(1.25)
delta2 = threshold_accuracy(1.25 ** 2)
delta3 = threshold_accuracy(1.25 ** 3)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthEstimationModel(in_channels=3)
model = model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_dl), epochs=10)
# Training loop

def train_epoch(model, train_dl, optimizer, loss_func, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(train_dl):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dl)
def validate_epoch(model, valid_dl, loss_func, device):
    model.eval()
    total_loss = 0.0
    metrics = {
        'abs_rel': 0.0,
        'log10_mae': 0.0,
        'log10_rmse': 0.0,
        'delta1': 0.0,
        'delta2': 0.0,
        'delta3': 0.0
    }
    with torch.no_grad():
        for x, y in tqdm(valid_dl):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_func(pred, y)
            total_loss += loss.item()
            metrics = {}
            # metrics['abs_rel'] += abs_rel(pred, y).item()
            # metrics['log10_mae'] += log10_mae(pred, y).item()
            # metrics['log10_rmse'] += log10_rmse(pred, y).item()
            # metrics['delta1'] += delta1(pred, y).item()
            # metrics['delta2'] += delta2(pred, y).item()
            # metrics['delta3'] += delta3(pred, y).item()
    num_batches = len(valid_dl)
    return (total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()})

def train_model(model, train_dl, valid_dl, optimizer, loss_func, device, epochs=10):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dl, optimizer, loss_func, device)
        valid_loss, metrics = validate_epoch(model, valid_dl, loss_func, device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        print(f"Metrics: {metrics}")

        # Update learning rate
        # scheduler.step()
        # Log to Weights & Biases
        # wandb.log({
        #     'epoch': epoch + 1,
        #     'train_loss': train_loss,
        #     'valid_loss': valid_loss,
        #     **metrics
        # })
    return model


# visualize a batch of data
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(dl, pairs_per_row=2):
    for x, y in dl:
        x = x.permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
        y = y.squeeze(1).numpy()           # [B, H, W]

        # Unnormalize input images (assuming ImageNet stats)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        x = x * imagenet_std + imagenet_mean
        x = np.clip(x, 0, 1)

        batch_size = x.shape[0]
        num_cols = pairs_per_row
        num_rows = int(np.ceil(batch_size / num_cols))

        fig, axes = plt.subplots(nrows=num_rows * 2, ncols=num_cols, figsize=(5 * num_cols, 4 * num_rows))

        for i in range(batch_size):
            row = (i // num_cols) * 2
            col = i % num_cols

            axes[row, col].imshow(x[i])
            axes[row, col].set_title(f'Input {i}')
            axes[row, col].axis('off')

            axes[row + 1, col].imshow(y[i], cmap='viridis')
            axes[row + 1, col].set_title(f'Depth {i}')
            axes[row + 1, col].axis('off')

        # Hide any empty subplots
        for i in range(batch_size, num_rows * num_cols):
            row = (i // num_cols) * 2
            col = i % num_cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')

        plt.tight_layout()
        plt.savefig('batch_visualization.png')
        plt.show()
        break  # only show one batch

visualize_batch(train_dl)
import sys; sys.exit()

model = train_model(model, train_dl, valid_dl, optimizer, loss_func, device, epochs=10)
# Save the model
torch.save(model.state_dict(), 'depth_estimation_model.pth')

import matplotlib.pyplot as plt

def visualize_predictions(model, valid_dl, device):
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_dl):
            x = x.to(device)
            pred = model(x)
            pred = F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=False)

            # Get only the first image from the batch
            pred_img = pred[0].cpu().squeeze().numpy()
            y_img = y[0].cpu().squeeze().numpy()

            # Convert to uint8 for visualization
            pred_vis = (pred_img * 255).astype(np.uint8)
            y_vis = (y_img * 255).astype(np.uint8)

            # Plot side-by-side
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(pred_vis, cmap='gray')
            axs[0].set_title(f'Predicted Depth {i}')
            axs[0].axis('off')

            axs[1].imshow(y_vis, cmap='gray')
            axs[1].set_title(f'Ground Truth Depth {i}')
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

            if i >= 4:  # Show only 5 batches
                break

visualize_predictions(model, valid_dl, device)
