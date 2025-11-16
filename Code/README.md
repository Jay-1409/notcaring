# Wireless CSI Action Recognition with SALD Distillation

## Overview

This project implements **Synthetic data-Assisted Learning via Distillation (SALD)** for action recognition from WiFi Channel State Information (CSI) data. The goal is to create a compact synthetic dataset that captures the knowledge of a real, large-scale MMFi dataset, enabling efficient model training on resource-constrained devices.

**Key Features:**
- Loads real MMFi CSI data (.mat files) from 6-antenna WiFi receivers
- Performs gradient-based synthetic data generation and refinement
- Trains student models on distilled synthetic data
- Evaluates performance on held-out real test sets
- Fully CPU-based implementation for portability

---

## Project Structure

```
/Users/jay/Desktop/wireless-project/Code/
├── config.py              # Global hyperparameters and device settings
├── data_loader.py         # Dataset classes and data loading utilities
├── models.py              # ResNet18 and SmallCNN feature extractors
├── sald.py                # Core SALD distillation algorithm
├── train_eval.py          # Student model training and evaluation
├── train_direct.py        # Direct training on real data (baseline)
└── run_sald.py            # Main entry point for SALD pipeline
```

### File Descriptions

| File | Purpose |
|------|---------|
| **config.py** | Centralized hyperparameters: batch sizes, learning rates, number of classes (27), distillation fractions, device (CPU), random seed (42). |
| **data_loader.py** | `MMFiCSIDataset`: loads .mat CSI files (6-channel tensors: 3 amplitude + 3 phase). `GenericNPYDataset`: loads .npy synthetic data. Utilities for gathering files, creating train/val/test splits. |
| **models.py** | `ResNet18Feature`: ResNet-18 backbone for CSI data. `SmallCNNFeature`: lightweight CNN alternative. Both output 512-dim feature vectors → classification head. |
| **sald.py** | Main distillation loop: (1) initializes synthetic data from real samples, (2) computes gradient importance scores, (3) extracts features and subspace projections, (4) iteratively refines synthetic data via gradient descent. |
| **train_eval.py** | `evaluate_on_synth()`: trains a student on synthetic data, then evaluates on real test set. Handles NaN/Inf robustness and normalization. |
| **train_direct.py** | Baseline: directly trains ResNet18 or SmallCNN on real MMFi data (subset). |
| **run_sald.py** | Orchestrates the full SALD pipeline: distills synthetic data, then trains and evaluates. |

---

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch (CPU)
- NumPy, SciPy, scikit-learn
- tqdm
- scipy.io (for .mat file loading)

### Installation

```bash
pip install torch numpy scipy scikit-learn tqdm
```

### Data Setup

Update paths in `config.py`:
```python
DATA_ROOT = '/path/to/MMFi_Dataset'  # Directory with train/val/test splits
OUT_DIR = '/path/to/output'           # Where distilled data is saved
```

Expected directory structure:
```
MMFi_Dataset/
├── E01/
│   ├── S01/
│   │   ├── A01/wifi-csi/
│   │   ├── A02/wifi-csi/
│   │   └── ...
│   └── ...
├── E02/
└── ...
```

---

## Usage

### Run Full SALD Pipeline

```bash
python run_sald.py
```

**Output:**
- Distilled synthetic dataset saved to `OUT_DIR/synthetic_data_epoch_X.pt`
- Student model trained on synthetic data
- Test accuracy on real data printed to console

### Train Directly on Real Data (Baseline)

```bash
python train_direct.py
```

**Output:**
- Trains SmallCNN for `epochs=3` on real MMFi subset
- Prints train/val/test accuracies
- Saves final model to `OUT_DIR/final_model.pth`

### Evaluate Pre-computed Synthetic Data

```python
from train_eval import evaluate_on_synth

evaluate_on_synth('/path/to/synthetic_data.pt', epochs=30)
```

---

## Key Hyperparameters

All hyperparameters are defined in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 27 | Action classes in MMFi dataset |
| `DISTILLED_FRACTION` | 0.10 | Fraction of real data to initialize synthetic set |
| `OUTER_ITERS` | 10 | Number of outer distillation iterations |
| `STUDENT_EPOCHS` | 1 | Epochs per outer iteration to train student |
| `STUDENT_BATCH` | 32 | Student batch size |
| `STUDENT_LR` | 5e-4 | Student learning rate (Adam) |
| `REAL_BATCH` | 64 | Real data batch size for gradient computation |
| `TOP_K` | 20 | Number of gradient-important samples to track |
| `SVD_K` | 5 | Subspace dimension for projection |
| `SYN_LR` | 0.5 | Synthetic data learning rate (SGD) |
| `SYN_UPDATES_PER_OUTER` | 1 | Updates to synthetic params per outer iteration |
| `SEED` | 42 | Random seed for reproducibility |
| `DEVICE` | "cpu" | Compute device (CPU only) |

---

## Implementation Details

### SALD Algorithm Overview

1. **Initialization** (`init_synthetic_from_real`): Sample a small fraction of real data; aggregate by class.

2. **Gradient Importance** (`compute_gradient_importance`): For each real sample, compute gradient norm w.r.t. model parameters. Select top-K most influential samples.

3. **Feature Extraction & Subspace** (`extract_features`, `compute_subspace_from_features`): Extract features from top-K samples; compute principal subspace via SVD; project samples onto this subspace.

4. **Outer Loop** (main `sald_distill` loop):
   - Train student on current synthetic dataset for `STUDENT_EPOCHS` epochs.
   - Recompute importance and subspace on real data.
   - Refine synthetic data via gradient descent (one step per outer iteration).
   - Repeat for `OUTER_ITERS` iterations.

5. **Evaluation** (`evaluate_on_synth`): Train a new student on final synthetic data; test on real held-out test set.

### CPU-Based Design

- All tensors and models explicitly moved to `torch.device("cpu")`.
- DataLoaders use `num_workers=0` for compatibility.
- Normalization and NaN/Inf handling prevent numerical instability.
- Random seeds fixed for reproducibility.

### Data Preprocessing

- CSI tensors shape: **(6, 114, 10)** — 3 antenna amplitude + 3 antenna phase, 114 subcarriers, 10 time steps.
- NaN/Inf values replaced with 0 / ±1.
- Global min-max normalization to [0, 1].
- Clipping to prevent outlier effects.

---

## Example Workflow

```python
# 1. Load and explore data
from data_loader import gather_mmfi_files
files, labels = gather_mmfi_files()
print(f"Found {len(files)} MMFi files, {len(set(labels))} classes")

# 2. Run distillation
from sald import sald_distill
synth_path = sald_distill()
print(f"Saved synthetic data to {synth_path}")

# 3. Evaluate on synthetic
from train_eval import evaluate_on_synth
evaluate_on_synth(synth_path, epochs=25)
```

---

## Output & Logging

- **Distilled synthetic data**: Saved as `.pt` (PyTorch) files in `OUT_DIR`. Contains tensors `X_syn` (synthetic samples) and `y_syn` (labels).
- **Console output**: Training/validation/test accuracies and loss values printed per epoch.
- **Model checkpoints**: Final model saved to `OUT_DIR/final_model.pth` (if applicable).

---

## Troubleshooting

### Issue: File not found error
- **Solution**: Ensure `DATA_ROOT` in `config.py` points to a valid MMFi directory with the expected structure.

### Issue: Out of memory (even on CPU)
- **Solution**: Reduce `REAL_BATCH`, `STUDENT_BATCH`, or `DISTILLED_FRACTION` in `config.py`.

### Issue: NaN loss during training
- **Solution**: Handled automatically in `train_eval.py`. If persists, check data normalization and reduce `STUDENT_LR`.

### Issue: Slow data loading
- **Solution**: Data is loaded from .mat files on disk. Preprocess and cache to .npy or .pt if needed.

---

## Performance & Results

- **Direct training baseline** (train_direct.py): Trains SmallCNN on real data subset. Baseline accuracy depends on data size and hyperparameters.
- **SALD distillation**: Produces a compact synthetic dataset (~10% of real data) that can train efficient models with minimal performance loss.

---

## References

- **MMFi Dataset**: Widely-used WiFi CSI dataset for action recognition.
- **Distillation**: Inspired by dataset distillation and meta-learning via gradient matching.
- **PyTorch**: Deep learning framework used throughout.

---

## Contact & Support

For questions or issues, contact the project maintainer or open an issue on the repository.

---

**Last Updated:** November 16, 2025
