from fastai.vision.all import *
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Your existing data loading code
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

class DepthTestDataset(Dataset):
    def __init__(self, df, image_size=(256, 256)):
        self.df = df
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        depth_path = self.df.iloc[idx]['depth']
        mask_path = self.df.iloc[idx]['mask']

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) / 255.0  # (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Load and preprocess depth
        depth = np.load(depth_path).squeeze()
        mask = np.load(mask_path) > 0
        epsilon = 1e-6
        max_depth = min(300, np.percentile(depth[mask], 99))
        depth = np.clip(depth, epsilon, max_depth)
        depth_log = np.zeros_like(depth)
        depth_log[mask] = np.log(depth[mask])
        depth_log = np.clip(depth_log, np.log(epsilon), np.log(max_depth))
        depth_log = cv2.resize(depth_log, self.image_size)

        with np.errstate(invalid='ignore', divide='ignore'):
            depth_norm = (depth_log / np.log(max_depth))
            depth_norm = np.nan_to_num(depth_norm, nan=0.0, posinf=0.0, neginf=0.0)

        depth_tensor = torch.tensor(depth_norm, dtype=torch.float32).unsqueeze(0)
        return img_tensor, depth_tensor

# Create test dataset and dataloader
test_dataset = DepthTestDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the model from the downloaded pkl file
learn = load_learner('depth_estimation_model.pkl')
learn.model.eval()
device = default_device()
learn.model.to(device)

# Run inference
all_preds = []
all_targets = []

for xb, yb in tqdm(test_loader, desc="Evaluating"):
    xb, yb = xb.to(device), yb.to(device)
    with torch.no_grad():
        pred = learn.model(xb)
    all_preds.append(pred.cpu())
    all_targets.append(yb.cpu())

# Concatenate all predictions and targets
pred_tensor = torch.cat(all_preds, dim=0)
gt_tensor = torch.cat(all_targets, dim=0)

print(f"Predictions shape: {pred_tensor.shape}")
print(f"Ground truth shape: {gt_tensor.shape}")

# You can now compute metrics or visualize results
mse_loss = torch.nn.functional.mse_loss(pred_tensor, gt_tensor)
print(f"MSE Loss: {mse_loss.item()}")
