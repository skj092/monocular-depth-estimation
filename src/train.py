from pathlib import Path
from glob import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from models import ResNet50UpProj
from utils import visualize_depth_map
from engine import train_depth_model

# Set the working directory to the script's directory
path = Path(__file__).parent
images = glob('val_extracted/val/**/**/**/*.png', recursive=True)
# print(f"Current file path: {path}")
# print(f'Number of images: {len(images)}')

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


# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.min_depth = 0.1
        self.dim = (256, 256)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image']
        depth_path = self.dataframe.iloc[idx]['depth']
        mask_path = self.dataframe.iloc[idx]['mask']

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = np.array(image_, dtype=np.float32) / 255.0
        image_ = np.transpose(image_, (2, 0, 1))  # Convert to CHW format

        depth_map = np.load(depth_path).squeeze()

        mask = np.load(mask_path)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = np.array(depth_map, dtype=np.float32) / np.log(max_depth)
        depth_map = np.transpose(depth_map, (2, 0, 1))

        return image_, depth_map


# Create DataLoader for training and validation sets
train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
xb, yb = next(iter(train_loader))
print(f"Batch shape: {xb.shape}, Depth shape: {yb.shape}")
visualize_depth_map((xb, yb))

# Initialize the model
model = ResNet50UpProj(num_classes=1)
out = model(xb)
print(f"Model output shape: {out.shape}")

# Train the model
train_depth_model(model, train_loader, val_loader, epochs=30)
