# sald.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from config import *
from data_loader import get_file_list, GenericNPYDataset, MMFiCSIDataset, gather_mmfi_files, make_splits
from models import ResNet18Feature
from torch.utils.data import DataLoader, TensorDataset
import os

torch.manual_seed(SEED)
np.random.seed(SEED)

def init_synthetic_from_real(train_files, train_labels, fraction=DISTILLED_FRACTION):
    import scipy.io as sp
    files_per_class = {}
    for f,l in zip(train_files, train_labels):
        files_per_class.setdefault(l, []).append(f)
    X_list, y_list = [], []
    for cls, flist in files_per_class.items():
        n = max(1, int(len(flist)*fraction))
        chosen = np.random.choice(flist, n, replace=False)
        for p in chosen:
            # Load .mat files for MMFi dataset
            mat = sp.loadmat(p)
            amp = mat["CSIamp"]      # (3,114,10)
            phase = mat["CSIphase"]  # (3,114,10)
            # Stack channels → (6, 114, 10)
            arr = np.concatenate([amp, phase], axis=0).astype(np.float32)
            
            # **CRITICAL: Normalize to prevent NaN in SALD**
            # Remove NaN/inf values
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize to [0, 1] using percentiles (robust to outliers)
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
            if p99 > p1:
                arr = (arr - p1) / (p99 - p1 + 1e-8)
            arr = np.clip(arr, 0, 1)
            
            t = torch.tensor(arr, dtype=torch.float32)
            if t.ndim == 2:
                t = t.unsqueeze(0)
            X_list.append(t)
            y_list.append(cls)
    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y

def compute_gradient_importance(student, criterion, real_loader, device, top_k, max_samples=REAL_BATCH):
    student.eval()
    batch_x, batch_y, collected = [], [], 0
    for x,y in real_loader:
        batch_x.append(x)
        batch_y.append(y)
        collected += x.size(0)
        if collected >= max_samples:
            break
    batch_x = torch.cat(batch_x, dim=0)[:max_samples]
    batch_y = torch.cat(batch_y, dim=0)[:max_samples]
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    grads_norms = []
    for i in range(batch_x.size(0)):
        student.zero_grad()
        xi, yi = batch_x[i:i+1], batch_y[i:i+1]
        out = student(xi)
        loss = criterion(out, yi)
        grads = torch.autograd.grad(loss, student.parameters(), retain_graph=False)
        total_norm = sum([(g.detach()**2).sum().item() for g in grads])**0.5
        grads_norms.append(total_norm)
    grads_norms = np.array(grads_norms)
    top_idx = np.argsort(-grads_norms)[:top_k]
    return batch_x[top_idx], batch_y[top_idx], top_idx, grads_norms

def extract_features(student, x, device):
    student.eval()
    with torch.no_grad():
        feat, _ = student(x.to(device), return_feat=True)
    return feat.cpu()

def compute_subspace_from_features(F_cpu, svd_k=SVD_K):
    F = F_cpu - F_cpu.mean(dim=0, keepdim=True)
    # Convert to numpy for more stable SVD
    F_np = F.numpy()
    try:
        U, S, Vt = np.linalg.svd(F_np, full_matrices=False)
    except Exception as e:
        print(f"[WARNING] SVD failed: {e}, using fallback")
        # Fallback: use identity as principal subspace
        U = np.eye(F_np.shape[0])
        S = np.ones(min(F_np.shape))
        Vt = np.eye(F_np.shape[1])[:svd_k]
    
    # Convert back to torch
    Vt = torch.from_numpy(Vt).float()
    V = Vt[:svd_k].T
    P = V @ V.T
    centroid = F_cpu.mean(dim=0)
    centroid_proj = (V @ (V.T @ centroid))
    return P, centroid_proj

def sald_distill():
    train_files, train_labels = get_file_list("train")
    print(f"\n[DEBUG] Found {len(train_files)} training files")
    print(f"[DEBUG] Unique train labels: {sorted(set(train_labels))}, counts: {dict((l, train_labels.count(l)) for l in set(train_labels))}")
    
    X_syn_init, y_syn = init_synthetic_from_real(train_files, train_labels)
    print(f"[DEBUG] Synthetic data shape: {X_syn_init.shape}, labels: {sorted(set(y_syn.tolist()))}")
    X_syn_param = torch.nn.Parameter(X_syn_init.clone())
    y_syn = y_syn.clone()
    opt_syn = optim.SGD([X_syn_param], lr=SYN_LR)
    train_dataset = MMFiCSIDataset(train_files, train_labels)
    real_loader = DataLoader(train_dataset, batch_size=REAL_BATCH, shuffle=True)
    device = torch.device(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for outer in trange(OUTER_ITERS, desc="SALD outer"):
        in_channels = X_syn_param.shape[1]
        student = ResNet18Feature(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
        opt_student = optim.Adam(student.parameters(), lr=STUDENT_LR)
        dsyn = TensorDataset(X_syn_param.detach().clone(), y_syn)
        syn_loader = DataLoader(dsyn, batch_size=STUDENT_BATCH, shuffle=True)

        for ep in range(STUDENT_EPOCHS):
            for xb, yb in syn_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt_student.zero_grad()
                out = student(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt_student.step()

        top_x, top_y, top_idx, norms = compute_gradient_importance(student, criterion, real_loader, device, TOP_K)
        top_feats = extract_features(student, top_x, device)
        P_cpu, centroid_proj = compute_subspace_from_features(top_feats)
        syn_feats = student(X_syn_param.to(device), return_feat=True)[0]
        P, centroid_proj = P_cpu.to(device), centroid_proj.to(device)
        target = centroid_proj.unsqueeze(0).expand(syn_feats.size(0), -1)
        student.eval()
        feats_with_grad, _ = student(X_syn_param.to(device), return_feat=True)
        loss_to_backprop = nn.MSELoss()(feats_with_grad, target)
        opt_syn.zero_grad()
        loss_to_backprop.backward()
        opt_syn.step()
        if outer % 10 == 0:
            print(f"[outer {outer}] syn_loss {loss_to_backprop.item():.6f}, grad_norms_mean {norms.mean():.4f}")

    save_path = os.path.join(OUT_DIR, "D_syn_sald.pth")
    torch.save({"X_syn": X_syn_param.detach().cpu(), "y_syn": y_syn}, save_path)
    print(f"Saved synthetic dataset to {save_path}")
    return save_path
