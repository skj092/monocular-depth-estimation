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
import mlflow
from datetime import datetime
from tqdm import tqdm

# Assume these are defined
from utils import visualize_depth_map
from models import ResNet50UpProj
from loss_fns import calculate_loss

# Setup MLflow
mlflow.set_tracking_uri("http://192.168.95.103:5000")
mlflow.set_experiment("depth_prediction_experiment_fastai_mlflow")
if mlflow.active_run():
    mlflow.end_run()

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

# MLflowCallback
class MLflowCallback(Callback):
    def __init__(self, log_model=True, test_loader=None, run_name=None, device='cuda'):
        super().__init__()
        self.log_model = log_model
        self.test_loader = test_loader
        self.run_name = run_name or f"DepthPrediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.artifact_dir = Path("mlflow_artifacts")
        self.artifact_dir.mkdir(exist_ok=True)
        self.run = None
        self.device = device

    def before_fit(self):
        try:
            if mlflow.active_run():
                mlflow.end_run()
            self.run = mlflow.start_run(run_name=self.run_name)
            mlflow.log_params({
                "lr": float(self.learn.opt.hypers[0]["lr"]),
                "batch_size": int(self.dls.bs),
                "epochs": int(self.learn.n_epoch),
                "train_samples": len(self.dls.train_ds),
                "valid_samples": len(self.dls.valid_ds),
                "test_samples": len(self.test_loader.dataset) if self.test_loader else 0,
                "model": "ResNet50UpProj"
            })
        except Exception as e:
            print(f"Error in before_fit: {e}")
            mlflow.log_param("error_before_fit", str(e))

    def after_epoch(self):
        try:
            metrics = {}
            if self.learn.recorder.losses:
                metrics["train_loss"] = float(self.learn.recorder.losses[-1])
            if self.learn.recorder.values and len(self.learn.recorder.values[-1]) >= 2:
                metrics["valid_loss"] = float(self.learn.recorder.values[-1][1])
                metric_names = ["mae", "rmse", "abs_rel", "log10_mae", "log10_rmse", "delta1", "delta2", "delta3"]
                for i, name in enumerate(metric_names, start=2):
                    if len(self.learn.recorder.values[-1]) > i:
                        metrics[f"valid_{name}"] = float(self.learn.recorder.values[-1][i])
            if metrics:
                mlflow.log_metrics(metrics, step=self.epoch)
        except Exception as e:
            print(f"Error in after_epoch: {e}")
            mlflow.log_param(f"error_epoch_{self.epoch}", str(e))

    def after_fit(self):
        try:
            if self.log_model:
                model_path = self.artifact_dir / "final_model.pth"
                torch.save(self.learn.model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="models")

            if self.test_loader:
                print("Evaluating on test set...")
                test_metrics = self._compute_test_metrics()
                if test_metrics:
                    mlflow.log_metrics(test_metrics, step=self.learn.n_epoch)

                xb, yb = next(iter(self.test_loader))
                xb, yb = xb.to(self.device), yb.to(self.device)
                with torch.no_grad():
                    preds = self.learn.model(xb)
                fig = visualize_depth_map((xb.cpu(), yb.cpu()), test=True, model=self.learn.model)
                viz_path = self.artifact_dir / "test_preds_final.png"
                fig.savefig(viz_path)
                plt.close(fig)
                mlflow.log_artifact(viz_path, artifact_path="visualizations")

            if self.run:
                mlflow.end_run()
        except Exception as e:
            print(f"Error in after_fit: {e}")
            mlflow.log_param("error_after_fit", str(e))

    def _compute_test_metrics(self):
        try:
            test_metrics = {}
            all_preds = []
            all_targets = []

            self.learn.model.eval()
            self.learn.model.to(self.device)
            with torch.no_grad():
                for xb, yb in tqdm(self.test_loader, total=len(self.test_loader), desc="Test Inference"):
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.learn.model(xb)
                    all_preds.append(preds.cpu())
                    all_targets.append(yb.cpu())

            test_preds = torch.cat(all_preds)
            test_targets = torch.cat(all_targets)

            test_metrics["test_mae"] = float(torch.mean(torch.abs(test_preds - test_targets)).numpy())
            test_metrics["test_rmse"] = float(torch.sqrt(torch.mean((test_preds - test_targets) ** 2)).numpy())
            test_metrics["test_abs_rel"] = float(abs_rel(test_preds, test_targets).numpy())
            test_metrics["test_log10_mae"] = float(log10_mae(test_preds, test_targets).numpy())
            test_metrics["test_log10_rmse"] = float(log10_rmse(test_preds, test_targets).numpy())
            test_metrics["test_delta1"] = float(delta1(test_preds, test_targets).numpy())
            test_metrics["test_delta2"] = float(delta2(test_preds, test_targets).numpy())
            test_metrics["test_delta3"] = float(delta3(test_preds, test_targets).numpy())
            mlflow.log_params(test_metrics)

            return test_metrics
        except Exception as e:
            print(f"Error computing test metrics: {e}")
            mlflow.log_param("error_test_metrics", str(e))
            return {}

# Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=Adam,
    metrics=[mae, rmse, abs_rel, log10_mae, log10_rmse, delta1, delta2, delta3],
    cbs=[MLflowCallback(log_model=True, test_loader=test_loader, run_name="SimpleDepthRun", device=device)]
)

learn.fit_one_cycle(1, 1e-4)

# Print test metrics
test_metrics = learn.cbs[-1]._compute_test_metrics()
if test_metrics:
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
