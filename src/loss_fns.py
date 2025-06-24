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
