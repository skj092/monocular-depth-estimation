from fastai.vision.all import *
import pandas as pd
import numpy as np
import cv2
import os
import torch
from sklearn.model_selection import train_test_split
import wandb
from fastai.callback.wandb import WandbCallback
from dotenv import load_dotenv
from loss_fns import calculate_loss
from model2 import DepthEstimationModel

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="DepthEstimation", name="FastAI_Training",)

load_dotenv()

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
learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=Adam,
    metrics=[mae, rmse, abs_rel, log10_mae, log10_rmse, delta1, delta2, delta3],
    cbs=[WandbCallback(), SaveModelCallback(monitor='valid_loss', fname='best_model')],
)

learn.fit_one_cycle(15, 1e-4)
learn.show_results()
