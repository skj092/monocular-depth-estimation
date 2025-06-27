import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
# from loss_fns import calculate_loss, abs_rel, log10_mae, log10_rmse, delta1, delta2, delta3


def train_epoch(model, train_dl, optimizer, loss_func, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(train_dl):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dl)
def validate_epoch(model, valid_dl, loss_func, device):
    model.eval()
    total_loss = 0.0
    metrics = {
        'abs_rel': 0.0,
        'log10_mae': 0.0,
        'log10_rmse': 0.0,
        'delta1': 0.0,
        'delta2': 0.0,
        'delta3': 0.0
    }
    with torch.no_grad():
        for x, y in tqdm(valid_dl):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_func(pred, y)
            total_loss += loss.item()
            metrics['abs_rel'] += abs_rel(pred, y).item()
            metrics['log10_mae'] += log10_mae(pred, y).item()
            metrics['log10_rmse'] += log10_rmse(pred, y).item()
            metrics['delta1'] += delta1(pred, y).item()
            metrics['delta2'] += delta2(pred, y).item()
            metrics['delta3'] += delta3(pred, y).item()
    num_batches = len(valid_dl)
    return (total_loss / num_batches, {k: v / num_batches for k, v in metrics.items()})
