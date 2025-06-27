import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import mlflow
from fastai.callback.core import Callback
from fastai.callback.tracker import Recorder

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


# Alternative approach - more robust metric extraction:
# class MLflowCallback(Callback):
#     def __init__(self, log_model=False):
#         self.log_model = log_model
#
#     def before_fit(self):
#         # Log hyperparameters safely
#         try:
#             config = {
#                 'lr': float(self.learn.opt.hypers[0]['lr']),
#                 'batch_size': int(self.dls.bs),
#                 'epochs': int(self.learn.n_epoch)
#             }
#             mlflow.log_params(config)
#         except Exception as e:
#             print(f"Warning: Could not log hyperparameters: {e}")
#
#     def after_epoch(self):
#         try:
#             # Get metrics from the recorder's values
#             metrics_to_log = {}
#
#             # Extract loss
#             if hasattr(self.learn.recorder, 'losses') and self.learn.recorder.losses:
#                 metrics_to_log['train_loss'] = float(self.learn.recorder.losses[-1])
#
#             # Extract validation metrics
#             if hasattr(self.learn.recorder, 'values') and self.learn.recorder.values:
#                 values = self.learn.recorder.values[-1]  # Latest epoch values
#                 if len(values) >= 2:  # [train_loss, valid_loss, ...]
#                     metrics_to_log['valid_loss'] = float(values[1])
#
#                     # Add other metrics (mae, rmse, etc.)
#                     metric_names = ['mae', 'rmse']  # Adjust based on your metrics
#                     for i, name in enumerate(metric_names, start=2):
#                         if len(values) > i:
#                             metrics_to_log[f'valid_{name}'] = float(values[i])
#
#             # Log metrics
#             if metrics_to_log:
#                 mlflow.log_metrics(metrics_to_log, step=self.epoch)
#
#         except Exception as e:
#             print(f"Warning: Could not log metrics for epoch {self.epoch}: {e}")
#
#     def after_fit(self):
#         if self.log_model:
#             try:
#                 model_path = "fastai_depth_model.pth"
#                 torch.save(self.learn.model.state_dict(), model_path)
#                 mlflow.log_artifact(model_path)
#             except Exception as e:
#                 print(f"Warning: Could not log model: {e}")
