import torch
from tqdm import tqdm

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
    delta1 = threshold_accuracy(1.25)
    delta2 = threshold_accuracy(1.25 ** 2)
    delta3 = threshold_accuracy(1.25 ** 3)
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
