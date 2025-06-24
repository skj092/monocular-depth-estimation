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
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MyExperiment")
mlflow.pytorch.autolog()


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

# trainer
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def train_depth_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    save_path: str = "resnet_uproj_depth.pt",
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_fn", "MSELoss")
    mlflow.log_param("batch_size", train_loader.batch_size)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if outputs.shape != depths.shape:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=depths.shape[-2:], mode='bilinear', align_corners=False
                )

            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = images.to(device)
                depths = depths.to(device)

                outputs = model(images)
                if outputs.shape != depths.shape:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=depths.shape[-2:], mode='bilinear', align_corners=False
                    )

                loss = criterion(outputs, depths)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            print(f"✅ Model saved to {save_path} (best val loss: {best_val_loss:.4f})")

    # Final model logging
    mlflow.pytorch.log_model(model, artifact_path="final_model")

    # Log final loss plot
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig_path = "loss_curve.png"
    fig.savefig(fig_path)
    plt.close(fig)
    mlflow.log_artifact(fig_path)

    # Log a batch of predictions vs. ground truth
    images, depths = next(iter(val_loader))
    fig = visualize_depth_map((images[:6], depths[:6]), test=True, model=model)
    fig_pred_path = "sample_predictions.png"
    fig.savefig(fig_pred_path)
    plt.close(fig)
    mlflow.log_artifact(fig_pred_path)


# Train the model
with mlflow.start_run():
    train_depth_model(model, train_loader, val_loader, epochs=20)
