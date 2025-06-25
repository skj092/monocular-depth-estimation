from fastai.vision.all import *
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
import wandb
from fastai.callback.wandb import WandbCallback
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="DepthEstimation", name="debug")


# Assume these are defined
from utils import visualize_depth_map
from models import ResNet50UpProj
from loss_fns import calculate_loss

# Data preparation (training and validation)
train_path = "train_extracted/train/indoors"
filelist = []
for root, dirs, files in os.walk(train_path):
    for file in files:
        filelist.append(os.path.join(root, file))
filelist.sort()
data = {
    "image": [x for x in filelist if x.endswith(".png")],
    "depth": [x for x in filelist if x.endswith("_depth.npy")],
    "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
}
df = pd.DataFrame(data)
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"Total train samples: {len(df)}")
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_df)}, Validation samples: {len(valid_df)}")

# FastAI data loading
def open_image(fn):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    return PILImage.create(img)

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
    return PILImageBW.create(depth_uint8)

def get_x(row): return open_image(row['image'])
def get_y(row): return load_depth_masked(row)

# Training/validation DataLoader (FastAI)
dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=IndexSplitter(valid_df.index),
    item_tfms=Resize((256, 256))
)
dls = dblock.dataloaders(pd.concat([train_df, valid_df]), bs=32, num_workers=4)

# Test dataset (PyTorch)
print("Preparing test data...")
test_path = "val_extracted/val/indoors"
filelist = []
for root, dirs, files in os.walk(test_path):
    for file in files:
        filelist.append(os.path.join(root, file))
filelist.sort()
data = {
    "image": [x for x in filelist if x.endswith(".png")],
    "depth": [x for x in filelist if x.endswith("_depth.npy")],
    "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
}
test_df = pd.DataFrame(data)
print(f"Total test samples: {len(test_df)}")  # Should print 325

class DepthDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
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
        depth_tensor = torch.from_numpy(depth_norm).float().unsqueeze(0)
        return img, depth_tensor

test_dataset = DepthDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

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

model = DepthWrapper(ResNet50UpProj(num_classes=1))
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


# Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=Adam,
    metrics=[mae, rmse, abs_rel, log10_mae, log10_rmse, delta1, delta2, delta3],
    cbs=[WandbCallback(), SaveModelCallback(monitor='valid_loss', fname='best_model')],
)

learn.fit_one_cycle(2, 1e-4)
wandb.save('models/best_model.pth')
learn.export('depth_estimation_model.pkl')

# test set inference
import sys; sys.exit()
test_path = "val_extracted/val/indoors"
test_filelist = []
for root, dirs, files in os.walk(test_path):
    for file in files:
        test_filelist.append(os.path.join(root, file))
test_filelist.sort()
test_data = {
    "image": [x for x in test_filelist if x.endswith(".png")],
    "depth": [x for x in test_filelist if x.endswith("_depth.npy")],
    "mask": [x for x in test_filelist if x.endswith("_depth_mask.npy")],
}
test_df = pd.DataFrame(test_data)
# Inference on test set
predictions = []
for img, _ in tqdm(test_loader, desc="Inference"):
    img = img.to(device)
    with torch.no_grad():
        pred = learn.model(img)
    pred = pred.cpu().numpy()
    predictions.append(pred)
predictions = np.concatenate(predictions, axis=0)
# Convert predictions to torch tensor
pred_tensor = torch.from_numpy(predictions)

# Load ground truth depth maps
gt_depths = []
for i in range(len(test_df)):
    depth = np.load(test_df.iloc[i]['depth']).squeeze()
    mask = np.load(test_df.iloc[i]['mask']) > 0
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
    gt_depths.append(depth_norm)

# Efficient conversion to torch tensor
gt_tensor = torch.from_numpy(np.array(gt_depths)).unsqueeze(1)

# Compute and log metrics
for metric in learn.metrics:
    if callable(metric):
        try:
            metric_value = metric(pred_tensor, gt_tensor)
            metric_name = getattr(metric, 'name', str(metric))
            wandb.log({metric_name: metric_value})
        except Exception as e:
            print(f"Error calculating {metric}: {e}")

