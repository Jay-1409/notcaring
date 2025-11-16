# config.py
import os

DATA_ROOT = '/media/Lab208/Storage Drive/wifi0sensing/data/MMFi_Dataset'  # Folder with train/val/test
OUT_DIR = '/media/Lab208/Storage Drive/wifi0sensing/data/MMFI_OUT'  # Where to save distilled data
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 27                  # Found 27 action classes in the real MMFi dataset
DISTILLED_FRACTION = 0.10         # 1% of real data (reduced for quick testing)
DEVICE = "cpu"                    # CPU only

# Distillation hyperparameters
OUTER_ITERS = 10                  # Reduced for stability (was 200)
STUDENT_EPOCHS = 1                # Reduced for stability (was 2)
STUDENT_BATCH = 32
STUDENT_LR = 5e-4                 # Reduced LR for stability (was 1e-3)

REAL_BATCH = 64                   # Reduced batch size
TOP_K = 20
SVD_K = 5
SYN_LR = 0.5
SYN_UPDATES_PER_OUTER = 1

SEED = 42
