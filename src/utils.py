import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch

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

