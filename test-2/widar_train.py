#!/usr/bin/env python3
"""
widar_train_final.py

Robust loader + trainer for Widar3.0-style .dat files.

Features:
 - recursive .dat discovery
 - flexible loading (text CSV/whitespace or raw float32)
 - parse filename metadata like user1-1-1-1-1-r1.dat
 - determine a target channel count (mode by default) and normalize all files to it
 - sliding-window creation with optional per-file window cap
 - robust train/test split (handles singletons)
 - simple 1D-CNN, training + evaluation

Adjust build_label_from_metadata() if your label is only one field (e.g., p1).
"""

import os
import argparse
import random
from collections import defaultdict, Counter
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# File discovery & loading
# -------------------------

def find_dat_files_recursive(data_dir: str) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.lower().endswith('.dat'):
                files.append(os.path.join(root, fn))
    files = sorted(list(dict.fromkeys(files)))
    return files

def parse_filename(fname: str) -> Dict:
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split('-')
    md = {}
    try:
        if parts and parts[0].startswith('user'):
            md['user'] = int(parts[0][4:])
        else:
            md['user'] = parts[0] if parts else -1
        # best-effort numeric parsing; leave strings otherwise
        md['p1'] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else (parts[1] if len(parts)>1 else -1)
        md['p2'] = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else (parts[2] if len(parts)>2 else -1)
        md['p3'] = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else (parts[3] if len(parts)>3 else -1)
        md['p4'] = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else (parts[4] if len(parts)>4 else -1)
        last = parts[5] if len(parts) > 5 else ''
        if isinstance(last, str) and last.startswith('r'):
            try:
                md['repeat'] = int(last[1:])
            except:
                md['repeat'] = last
        else:
            md['repeat'] = last if last != '' else -1
    except Exception:
        md['parts'] = parts
    return md

def load_dat_file(path: str) -> np.ndarray:
    """Try common numeric file formats; raise on failure."""
    # 1) text whitespace/comma
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.float32)
    except Exception:
        pass
    try:
        arr = np.genfromtxt(path, delimiter=',')
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.float32)
    except Exception:
        pass
    # 3) raw float32
    try:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size == 0:
            raise ValueError("Empty file or no float32 content")
        # guess common widths, else return (N,1)
        for w in (90, 60, 30, 6, 3, 2):
            if raw.size % w == 0:
                return raw.reshape(-1, w).astype(np.float32)
        return raw.reshape(-1, 1).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Cannot read file {path}: {e}. Inspect file format and adapt load_dat_file().")

# -------------------------
# Label mapping
# -------------------------

def build_label_from_metadata(md: Dict):
    """
    Default: use tuple (p1,p2,p3,p4) as label key.
    If your label is just p1 (activity), change to `return md['p1']`.
    """
    return (md.get('p1', -1), md.get('p2', -1), md.get('p3', -1), md.get('p4', -1))

# -------------------------
# Build samples and decide channel policy
# -------------------------

def build_samples_with_channel_stats(data_dir: str):
    """
    Loads files, returns list of tuples (arr, label_key, path, md) and channel counts Counter.
    """
    files = find_dat_files_recursive(data_dir)
    print(f"[Info] Found {len(files)} files under {data_dir} (recursive).")
    samples = []
    channel_counter = Counter()
    errors = []
    for f in tqdm(files, desc="Loading files"):
        try:
            md = parse_filename(f)
            arr = load_dat_file(f)
            if arr is None or arr.size == 0:
                errors.append((f, "empty"))
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            channel_counter[arr.shape[1]] += 1
            key = build_label_from_metadata(md)
            samples.append((arr, key, f, md))
        except Exception as e:
            errors.append((f, str(e)))
            # continue loading others
    if errors:
        print(f"[Warning] {len(errors)} files failed to load (sample up to 5):")
        for e in errors[:5]:
            print("   ", e[0], "->", e[1])
    return samples, channel_counter

def decide_target_channels(channel_counter: Counter, policy: str = 'mode'):
    """
    Decide target channels based on policy:
      - 'mode' : most common channel count
      - 'max'  : maximum channel count seen
      - 'strict' : require all files to have same channels (error otherwise)
    """
    if policy == 'mode':
        most_common = channel_counter.most_common(1)
        if not most_common:
            raise RuntimeError("No channel information available.")
        return most_common[0][0]
    elif policy == 'max':
        return max(channel_counter.keys())
    elif policy == 'strict':
        if len(channel_counter) == 0:
            raise RuntimeError("No channel information available.")
        if len(channel_counter) > 1:
            raise RuntimeError(f"strict policy: multiple channel counts found: {channel_counter}")
        return next(iter(channel_counter.keys()))
    else:
        raise ValueError("Unknown channel policy: " + str(policy))

def normalize_channels(arr: np.ndarray, target_ch: int) -> np.ndarray:
    """
    arr: (T, C)
    If C < target_ch -> pad columns with zeros to reach target_ch.
    If C > target_ch -> truncate extra columns (keep first target_ch).
    """
    T, C = arr.shape
    if C == target_ch:
        return arr
    if C < target_ch:
        pad = np.zeros((T, target_ch - C), dtype=arr.dtype)
        return np.hstack([arr, pad])
    else:
        return arr[:, :target_ch]

# -------------------------
# Prepare final samples (map label keys to ints)
# -------------------------

def finalize_samples(samples_raw: List[Tuple[np.ndarray, object, str, Dict]]):
    """
    samples_raw: list of (arr, label_key, path, md)
    returns: samples_final list (arr, label_int, path, md), key2int dict
    """
    label_counts = Counter([t[1] for t in samples_raw])
    keys_sorted = sorted(label_counts.keys(), key=lambda x: str(x))
    key2int = {k: i for i, k in enumerate(keys_sorted)}
    final = []
    for arr, key, path, md in samples_raw:
        final.append((arr, key2int[key], path, md))
    return final, key2int

# -------------------------
# Dataset: windows created after channel normalization
# -------------------------

class WidarDataset(Dataset):
    def __init__(self, samples_with_labels: List[Tuple[np.ndarray, int, str, Dict]],
                 window_size: int = 256, stride: int = 128,
                 target_ch: int = 90, max_windows_per_file: int = None,
                 sample_seed: int = 42):
        """
        samples_with_labels: list of (arr, label_int, path, md)
        This constructor creates windows and normalizes channels to target_ch.
        max_windows_per_file: if set, sample up to this many windows randomly per file.
        """
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.labels = []
        self.paths = []

        rnd = random.Random(sample_seed)

        for arr, label, path, md in tqdm(samples_with_labels, desc="Windowing files"):
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            # normalize channels
            arr = normalize_channels(arr, target_ch)
            T, C = arr.shape
            if T < window_size:
                pad = np.zeros((window_size - T, C), dtype=np.float32)
                arrp = np.vstack([arr, pad])
            else:
                arrp = arr
            max_start = max(1, arrp.shape[0] - window_size + 1)
            starts = list(range(0, max_start, self.stride))
            if max_windows_per_file is not None and len(starts) > max_windows_per_file:
                starts = rnd.sample(starts, max_windows_per_file)
            for s in starts:
                w = arrp[s:s+window_size]  # shape (window, channels)
                w = w.T  # (channels, window)
                self.windows.append(w.astype(np.float32))
                self.labels.append(label)
                self.paths.append(path)
        print(f"[Dataset] Built {len(self.windows)} windows from {len(samples_with_labels)} files")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# -------------------------
# Model
# -------------------------

class Simple1DCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Training utilities
# -------------------------

def train_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    preds = []
    trues = []
    for xb, yb in dataloader:
        xb = xb.to(device).float()
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
        trues.extend(yb.cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(dataloader.dataset))
    acc = accuracy_score(trues, preds) if len(trues) else 0.0
    f1 = f1_score(trues, preds, average='macro') if len(trues) else 0.0
    return avg_loss, acc, f1

def eval_epoch(model, device, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(yb.cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(dataloader.dataset))
    acc = accuracy_score(trues, preds) if len(trues) else 0.0
    f1 = f1_score(trues, preds, average='macro') if len(trues) else 0.0
    return avg_loss, acc, f1, preds, trues

# -------------------------
# Robust train/test split (same as earlier)
# -------------------------

def robust_train_test_split(samples, test_size, seed):
    n_files = len(samples)
    labels = [s[1] for s in samples]
    label_counts = Counter(labels)
    singleton_labels = {lab for lab, cnt in label_counts.items() if cnt == 1}

    singleton_indices = [i for i, lab in enumerate(labels) if lab in singleton_labels]
    multi_indices = [i for i in range(n_files) if i not in singleton_indices]
    multi_labels = [labels[i] for i in multi_indices]

    print(f"[Split] Total files: {n_files}")
    print(f"[Split] Unique labels: {len(label_counts)}; singletons: {len(singleton_indices)} files")

    train_idx = []
    test_idx = []

    if len(multi_indices) == 0:
        all_idxs = list(range(n_files))
        random.Random(seed).shuffle(all_idxs)
        cut = max(1, int(len(all_idxs) * (1.0 - test_size)))
        train_idx = all_idxs[:cut]
        test_idx = all_idxs[cut:]
    else:
        try:
            tr_multi, te_multi = train_test_split(
                multi_indices,
                test_size=test_size,
                stratify=multi_labels if len(set(multi_labels))>1 else None,
                random_state=seed
            )
            train_idx = list(tr_multi) + singleton_indices
            test_idx = list(te_multi)
        except Exception as e:
            all_idxs = list(range(n_files))
            random.Random(seed).shuffle(all_idxs)
            cut = max(1, int(len(all_idxs) * (1.0 - test_size)))
            train_idx = all_idxs[:cut]
            test_idx = all_idxs[cut:]

    if len(test_idx) == 0 and len(train_idx) > 0:
        moved = train_idx.pop()
        test_idx.append(moved)
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    print(f"[Split] Final -> train files: {len(train_samples)}, test files: {len(test_samples)}")
    return train_samples, test_samples

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='root folder containing .dat files')
    parser.add_argument('--window', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-model', type=str, default='widar_model_final.pth')
    parser.add_argument('--channel-policy', choices=['mode', 'max', 'strict'], default='mode')
    parser.add_argument('--max-windows-per-file', type=int, default=None,
                        help='If set, randomly sample up to this many windows per file (memory control)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load files and examine channel counts
    samples_raw, channel_counter = build_samples_with_channel_stats(args.data_dir)
    if len(samples_raw) == 0:
        print("[Error] No valid files loaded. Exiting.")
        return
    print("[Info] Channel counts (sample):", channel_counter.most_common()[:10])

    # 2) Decide target channels
    target_ch = decide_target_channels(channel_counter, policy=args.channel_policy)
    print(f"[Info] Using target channel count = {target_ch} (policy={args.channel_policy})")

    # 3) Normalize channels and map label keys to ints
    # apply channel normalization to copies (so we don't mutate originals)
    normalized_samples = []
    for arr, key, path, md in samples_raw:
        arr_norm = normalize_channels(arr, target_ch)
        normalized_samples.append((arr_norm, key, path, md))
    samples_final, key2int = finalize_samples(normalized_samples)
    print(f"[Info] {len(samples_final)} files, {len(key2int)} classes (label mapping size)")

    # 4) Robust train/test split at file level
    train_files, test_files = robust_train_test_split(samples_final, test_size=args.test_size, seed=args.seed)
    if len(train_files) == 0:
        print("[Error] No training files after split. Exiting.")
        return

    # 5) Build datasets (windowing), optionally cap windows per file
    train_dataset = WidarDataset(train_files, window_size=args.window, stride=args.stride,
                                 target_ch=target_ch, max_windows_per_file=args.max_windows_per_file,
                                 sample_seed=args.seed)
    test_dataset = WidarDataset(test_files, window_size=args.window, stride=args.stride,
                                target_ch=target_ch, max_windows_per_file=args.max_windows_per_file,
                                sample_seed=args.seed+1)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("[Error] Empty dataset after windowing. Check window/stride and file durations.")
        print(" train windows:", len(train_dataset), " test windows:", len(test_dataset))
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # 6) Model
    sample_x, _ = train_dataset[0]
    in_ch = sample_x.shape[0]
    n_classes = len(key2int)
    print(f"[Info] Input channels: {in_ch}, Classes: {n_classes}, train windows: {len(train_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple1DCNN(in_channels=in_ch, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 7) Train loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc, test_f1, preds, trues = eval_epoch(model, device, test_loader, criterion)
        print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f}"
              f" | Test loss {test_loss:.4f} acc {test_acc:.4f} f1 {test_f1:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'key2int': key2int,
                'in_channels': in_ch,
                'window_size': args.window,
            }, args.save_model)
            print(f"[Info] Saved best model to {args.save_model} (test_acc={test_acc:.4f})")

    # 8) Final report
    _, test_acc, test_f1, preds, trues = eval_epoch(model, device, test_loader, criterion)
    print("\nFinal evaluation:")
    print(f"Test acc: {test_acc:.4f}, Test f1: {test_f1:.4f}")
    try:
        print("\nClassification report:")
        print(classification_report(trues, preds, digits=4))
    except Exception:
        pass

if __name__ == '__main__':
    main()
