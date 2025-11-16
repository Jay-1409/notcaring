# data_loader.py

import os
import torch
import numpy as np
import scipy.io as sp
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import STUDENT_BATCH, DATA_ROOT

class MMFiCSIDataset(Dataset):
    """
    Loads MMFi .mat files and returns tensors of shape:
        (6, 114, 10)  ← 3 antennas amplitude + 3 antennas phase
    """
    def __init__(self, file_list, label_list):
        self.files = file_list
        self.labels = label_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mat = sp.loadmat(path)

        amp = mat["CSIamp"]      # (3,114,10)
        phase = mat["CSIphase"]  # (3,114,10)

        # Stack channels → (6, 114, 10)
        x = np.concatenate([amp, phase], axis=0).astype(np.float32)
        
        # Remove NaN and inf values
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Global normalization to [0, 1]
        x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min + 1e-8)
        x = np.clip(x, 0, 1)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


class GenericNPYDataset(Dataset):
    """
    Loads generic .npy files for compatibility with SALD code.
    Handles both channels-first (C,H,W) and channels-last (H,W,C) formats.
    """
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx], allow_pickle=True)
        x = torch.tensor(arr, dtype=torch.float32)
        
        # Normalize shape handling:
        # - If 2D, add channel dim: (H,W) -> (1,H,W)
        # - If 3D, accept either (C,H,W) or (H,W,C). If channels look like last dim,
        #   transpose to (C,H,W).
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            # If channels-first (C,H,W) then first dim is small (1-8)
            if x.shape[0] in (1, 2, 3, 4, 6):
                pass
            # If channels-last (H,W,C) where last dim is small, transpose
            elif x.shape[2] in (1, 2, 3, 4, 6):
                x = x.permute(2, 0, 1)
        
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def gather_mmfi_files():
    """
    Walk through MMFi directory structure:
        E*/S*/A*/wifi-csi/*.mat
    Label extraction: A01 → 0, A02 → 1, ...
    """

    files = []
    labels = []

    envs = sorted(glob(os.path.join(DATA_ROOT, "E*")))
    for env in envs:
        subjects = sorted(glob(os.path.join(env, "S*")))
        for subj in subjects:
            actions = sorted(glob(os.path.join(subj, "A*")))
            for action_folder in actions:

                # Folder name like .../A03
                action_label = int(os.path.basename(action_folder)[1:]) - 1  # A01→0

                csi_folder = os.path.join(action_folder, "wifi-csi")
                mat_files = sorted(glob(os.path.join(csi_folder, "*.mat")))

                for f in mat_files:
                    files.append(f)
                    labels.append(action_label)

    return files, labels


def make_splits(files, labels, train_ratio=0.7, val_ratio=0.15):
    """
    Create deterministic train/val/test splits.
    """
    N = len(files)
    idx = np.arange(N)
    np.random.shuffle(idx)

    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    def select(indices):
        return [files[i] for i in indices], [labels[i] for i in indices]

    return select(train_idx), select(val_idx), select(test_idx)


def make_loaders(batch_size=STUDENT_BATCH):
    files, labels = gather_mmfi_files()
    (train_f, train_l), (val_f, val_l), (test_f, test_l) = make_splits(files, labels)

    train_ds = MMFiCSIDataset(train_f, train_l)
    val_ds   = MMFiCSIDataset(val_f, val_l)
    test_ds  = MMFiCSIDataset(test_f, test_l)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_file_list(split="train"):
    """For SALD synthetic data initialization"""
    files, labels = gather_mmfi_files()
    (train_f, train_l), (val_f, val_l), (test_f, test_l) = make_splits(files, labels)
    
    if split == "train":  return train_f, train_l
    if split == "val":    return val_f, val_l
    if split == "test":   return test_f, test_l
