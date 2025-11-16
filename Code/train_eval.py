# train_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import ResNet18Feature
from config import DEVICE, NUM_CLASSES, OUT_DIR
from torch.utils.data import DataLoader, TensorDataset
from data_loader import make_loaders
import os

def evaluate_on_synth(synth_path, epochs=25):
    data = torch.load(synth_path)
    X_syn, y_syn = data['X_syn'], data['y_syn']
    print(f"\n[DEBUG] Loaded synthetic data: shape {X_syn.shape}, labels: {sorted(set(y_syn.tolist()))}")
    
    # **Normalize synthetic data to prevent NaN loss**
    X_syn = torch.nan_to_num(X_syn, nan=0.0, posinf=1.0, neginf=-1.0)
    # Use numpy percentile instead of torch.quantile for large tensors on CPU
    X_syn_np = X_syn.numpy() if isinstance(X_syn, torch.Tensor) else X_syn
    p1 = float(np.percentile(X_syn_np, 1))
    p99 = float(np.percentile(X_syn_np, 99))
    if p99 > p1:
        X_syn = (X_syn - p1) / (p99 - p1 + 1e-8)
    X_syn = torch.clamp(X_syn, 0, 1)
    
    in_channels = X_syn.shape[1]
    model = ResNet18Feature(in_channels=in_channels, num_classes=NUM_CLASSES).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    dsyn = TensorDataset(X_syn, y_syn)
    loader = DataLoader(dsyn, batch_size=16, shuffle=True)
    for ep in range(epochs):
        total, correct = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss in epoch {ep}, skipping batch")
                continue
            loss.backward()
            opt.step()
            total += xb.size(0)
            correct += (out.argmax(1)==yb).sum().item()
        print(f"Epoch {ep}: train acc {correct/total:.3f}")
    _,_,test_loader = make_loaders()
    print(f"[DEBUG] Test loader created")
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            correct += (out.argmax(1)==yb).sum().item()
            total += xb.size(0)
    print(f"[DEBUG] Test samples: {total}, correct: {correct}")
    print("Test accuracy on real test set:", correct/total)
