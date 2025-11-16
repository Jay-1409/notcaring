#!/usr/bin/env python3
"""
Direct training of ResNet18 on real MMFi dataset (no SALD distillation).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models import ResNet18Feature, SmallCNNFeature
from config import DEVICE, NUM_CLASSES, OUT_DIR
from data_loader import gather_mmfi_files, make_splits, MMFiCSIDataset
import os

def train_direct(epochs=3, batch_size=32, max_files=5000):
    """Train ResNet18 directly on real MMFi data (subset for speed)."""
    device = torch.device(DEVICE)
    
    print(f"[INFO] Gathering MMFi files (this may take a moment)...")
    files, labels = gather_mmfi_files()
    
    # Use only a subset for faster training
    if len(files) > max_files:
        idx = np.random.choice(len(files), max_files, replace=False)
        files = [files[i] for i in idx]
        labels = [labels[i] for i in idx]
    
    (train_f, train_l), (val_f, val_l), (test_f, test_l) = make_splits(files, labels)
    
    print(f"[INFO] Dataset split: train={len(train_f)}, val={len(val_f)}, test={len(test_f)}")
    print(f"[INFO] Creating data loaders with batch_size={batch_size}...")
    
    train_ds = MMFiCSIDataset(train_f, train_l)
    val_ds = MMFiCSIDataset(val_f, val_l)
    test_ds = MMFiCSIDataset(test_f, test_l)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Creating SmallCNN model with {NUM_CLASSES} classes...")
    model = SmallCNNFeature(in_channels=6, num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Reduced LR from 1e-3
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    print(f"[INFO] Starting training for {epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Normalize each batch to prevent NaN
            x = (x - x.mean(dim=(1,2,3), keepdim=True)) / (x.std(dim=(1,2,3), keepdim=True) + 1e-8)
            x = torch.clamp(x, -3, 3)
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss detected at epoch {epoch}, batch {batch_idx}. Skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [Epoch {epoch}] Batch {batch_idx+1}: Loss={loss.item():.4f}")
        
        train_acc = train_correct / train_total
        print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, train_acc={train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        print(f"  Validation acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(OUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Best] Saved model to {checkpoint_path}")
    
    # Test phase
    print(f"\n[INFO] Evaluating on test set...")
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_correct += (logits.argmax(1) == y).sum().item()
            test_total += y.size(0)
    
    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
    
    # Save final model
    final_path = os.path.join(OUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved final model to {final_path}")
    
    return model, test_acc

if __name__ == "__main__":
    model, test_acc = train_direct(epochs=2, batch_size=32, max_files=2000)
    print(f"\n[SUMMARY] Final test accuracy: {test_acc:.4f}")
