import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_depth_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    save_path: str = "resnet_uproj_depth.pt",
    device: torch.device = None,
):
    """
    Train a depth estimation model using MSE loss.

    Args:
        model: PyTorch model (e.g., ResNet50UpProj)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the best model
        device: torch.device object, defaults to GPU if available
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Resize output to match target depth dimensions
            if outputs.shape != depths.shape:
                outputs = torch.nn.functional.interpolate(outputs, size=depths.shape[-2:], mode='bilinear', align_corners=False)

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
                    outputs = torch.nn.functional.interpolate(outputs, size=depths.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, depths)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path} (best val loss: {best_val_loss:.4f})")

