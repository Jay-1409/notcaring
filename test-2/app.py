import argparse
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from dtw import dtw  # For DTW verification

# Widar3.0 .dat Parser (Unchanged)
def parse_dat_file(filepath):
    """Parse .dat: Header + CSI packets -> amplitude matrix (time=500, subcarriers=30, antennas=3)."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        offset = 0
        time_series = []  # List of (30 sub) amps
        packet_count = 0
        while offset < len(data) - 100 and packet_count < 500:  # Cap at 500 packets (~1s)
            if data[offset:offset+3] != b'CSI':  # Magic header
                offset += 1
                continue
            offset += 8  # Skip magic + len_info
            if offset + 16 >= len(data): break
            timestamp = struct.unpack('d', data[offset:offset+8])[0]
            offset += 8
            csi_len = struct.unpack('H', data[offset:offset+2])[0]  # CSI payload len
            offset += 2
            
            # CSI: 30 sub x 3x3 MIMO (90 complex = 180 shorts; but use floats for simplicity)
            payload_size = csi_len * 2  # Approx
            if offset + payload_size > len(data): break
            try:
                # Unpack real/imag (simplified: assume 180 floats; adjust if shorts)
                csi_raw = np.frombuffer(data[offset:offset+payload_size], dtype=np.float32)
                if len(csi_raw) < 180:
                    offset += payload_size
                    continue
                csi_real = csi_raw[:90]
                csi_imag = csi_raw[90:180]
                csi_complex = csi_real + 1j * csi_imag
                amp = np.abs(csi_complex.reshape(3, 3, 30))  # (rx, tx, sub)
                amp = np.mean(amp, axis=(0,1))  # Avg MIMO -> (30 sub)
                time_series.append(amp)
                packet_count += 1
            except struct.error:
                offset += payload_size
                continue
            offset += payload_size
        
        if not time_series:
            return np.zeros((500, 30))  # Empty fallback
        csi_matrix = np.stack(time_series)  # (packets, 30)
        if csi_matrix.shape[0] < 500:
            csi_matrix = np.pad(csi_matrix, ((0, 500 - csi_matrix.shape[0]), (0,0)), mode='constant')
        else:
            csi_matrix = csi_matrix[:500]  # Trim to 500
        print(f"  Parsed {os.path.basename(filepath)}: {csi_matrix.shape} ({packet_count} packets)")
        return csi_matrix.astype(np.float32)
    except Exception as e:
        print(f"  Error parsing {os.path.basename(filepath)}: {e} -> using zeros")
        return np.zeros((500, 30)).astype(np.float32)

def extract_label_from_filename(filename):
    """Parse activity label from [user]-[activity]-... .dat."""
    parts = filename.split('-')
    if len(parts) >= 2:
        return int(parts[1]) - 1  # 0-21
    return 0

# Data Loading (Unchanged)
def load_data(dataset_dir='./data/widar3.0/raw', dataset='widar', max_files=None, 
              min_activities=2, max_files_per_activity=50):
    """
    Load from .dat files with better diversity guarantees.
    
    Args:
        max_files: Total max files to load (None = no limit)
        min_activities: Minimum number of different activities required
        max_files_per_activity: Max samples per activity (for balance)
    """
    if dataset == 'widar':
        all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.dat')]
        
        # Group files by activity
        activity_groups = {}
        for fname in all_files:
            activity = extract_label_from_filename(fname)
            if activity not in activity_groups:
                activity_groups[activity] = []
            activity_groups[activity].append(fname)
        
        print(f"Found {len(all_files)} total files across {len(activity_groups)} activities")
        print(f"Activities: {sorted(activity_groups.keys())}")
        print(f"Samples per activity: {[(k, len(v)) for k, v in sorted(activity_groups.items())]}")
        
        # Ensure minimum diversity
        if len(activity_groups) < min_activities:
            print(f"\n⚠️  ERROR: Need at least {min_activities} activities, found {len(activity_groups)}")
            print(f"   Please add .dat files with different activity numbers (user*-2-*.dat, user*-3-*.dat, etc.)")
            print(f"   Current activities: {sorted(activity_groups.keys())}")
            raise ValueError(f"Insufficient activity diversity: {len(activity_groups)} < {min_activities}")
        
        # Balance: take up to max_files_per_activity from each activity
        selected_files = []
        for activity in sorted(activity_groups.keys()):
            files = activity_groups[activity][:max_files_per_activity]
            selected_files.extend(files)
        
        # Apply global max if specified
        if max_files and len(selected_files) > max_files:
            # Stratified sampling: proportional from each activity
            files_per_activity = max_files // len(activity_groups)
            selected_files = []
            for activity in sorted(activity_groups.keys()):
                files = activity_groups[activity][:files_per_activity]
                selected_files.extend(files)
        
        print(f"\nLoading {len(selected_files)} files...")
        
        data, labels = [], []
        load_errors = 0
        for fname in selected_files:
            filepath = os.path.join(dataset_dir, fname)
            if not os.path.exists(filepath):
                print(f"  Skipping missing: {fname}")
                continue
            
            try:
                csi_amp = parse_dat_file(filepath)  # (500, 30)
                csi_amp = np.expand_dims(csi_amp, axis=-1)  # (500, 30, 1)
                csi_amp = np.repeat(csi_amp, 3, axis=-1)  # (500, 30, 3)
                label = extract_label_from_filename(fname)
                data.append(csi_amp)
                labels.append(label)
            except Exception as e:
                load_errors += 1
                if load_errors <= 5:  # Only print first 5 errors
                    print(f"  Error loading {fname}: {e}")
        
        if load_errors > 5:
            print(f"  ... and {load_errors - 5} more errors")
        
        if not data:
            raise ValueError("No data successfully loaded!")
        
        full_data = np.stack(data)
        full_labels = np.array(labels)
        
        # Show detailed class distribution
        unique_classes, counts = np.unique(full_labels, return_counts=True)
        print(f"\n{'='*60}")
        print(f"Final Dataset Summary:")
        print(f"{'='*60}")
        print(f"Total samples: {len(full_data)}")
        print(f"Data shape: {full_data.shape}")
        print(f"Number of classes: {len(unique_classes)}")
        print(f"Classes: {unique_classes}")
        print(f"\nClass distribution:")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / len(full_labels)) * 100
            print(f"  Class {cls}: {count:4d} samples ({percentage:5.1f}%)")
        print(f"{'='*60}\n")
        
        # Verify diversity
        if len(unique_classes) < min_activities:
            print(f"⚠️  WARNING: Expected {min_activities} classes, got {len(unique_classes)}")
    else:
        full_data = np.random.rand(100, 500, 30, 3).astype(np.float32)
        full_labels = np.random.randint(0, 7, 100)
    
    return full_data.astype(np.float32), full_labels.astype(np.int64)

def formulate_data(full_data, full_labels, n_domains=3, samples_per_domain=50):
    """Section V-A: Split (robust: handle small n_samples in splits)."""
    domains = np.random.randint(0, n_domains, len(full_labels))
    dt_data, dt_labels = [], []
    dl_data_list, dl_labels_list = [], []
    
    for d in range(n_domains):
        mask = domains == d
        d_data, d_labels = full_data[mask], full_labels[mask]
        print(f"  Domain {d}: {len(d_data)} samples")  # Debug
        
        if len(d_data) < 1: 
            continue
            
        if d < n_domains - 2:  # Train domains
            dt_data.append(d_data)
            dt_labels.append(d_labels)
        else:  # New domains (imbalance simulation)
            if len(d_data) < 3:  # Pad if too small
                pad_reps = 3 - len(d_data)
                d_data = np.concatenate([d_data, np.tile(d_data[:1], (pad_reps, 1, 1, 1))])
                d_labels = np.concatenate([d_labels, np.tile(d_labels[:1], pad_reps)])
            
            # FIXED: Simulate imbalance by reducing BOTH data and labels proportionally
            num_samples = len(d_data)
            imbalance_ratio = 0.5  # Keep 50% for imbalance simulation
            reduced_len = max(1, int(num_samples * imbalance_ratio))
            
            # Apply imbalance to both data and labels
            d_data = d_data[:reduced_len]
            d_labels = d_labels[:reduced_len]
            
            # Robust split: Proportional, no empty
            split1 = max(1, reduced_len // 3)  # Dlp ~1/3
            split2 = max(1, (reduced_len // 3) + (reduced_len % 3))  # Dlw ~next
            
            dlp_data = d_data[:split1]
            dlp_labels = d_labels[:split1]
            dlw_data = d_data[split1:split1 + split2]
            dlw_labels = d_labels[split1:split1 + split2]
            dlt_data = d_data[split1 + split2:]
            dlt_labels = d_labels[split1 + split2:]
            
            # Concatenate all splits
            dl_data_list.append(np.concatenate([dlp_data, dlw_data, dlt_data]))
            dl_labels_list.append(np.concatenate([dlp_labels, dlw_labels, dlt_labels]))
    
    # Handle empty train
    if not dt_data:
        print("No train domains -> using full as train + dummy test")
        dt_data = [full_data]
        dt_labels = [full_labels]
        dl_data_list = [full_data[:2]]  # Dummy new
        dl_labels_list = [full_labels[:2]]
    
    dt_concat = np.concatenate(dt_data) if len(dt_data) > 1 else dt_data[0]
    dt_l_concat = np.concatenate(dt_labels) if len(dt_labels) > 1 else dt_labels[0]
    
    return dt_concat, dt_l_concat, dl_data_list, dl_labels_list
# Noise Dispelling Scheme (Fixed: Handle 4D batch, diff axis=1, explicit pad, loop filtfilt)
def noise_dispelling(csi_data):
    """Adjacent diff + low-pass on time axis (1); handle batch (4D)."""
    # csi_data (batch, time, sub, ant)
    diff = np.diff(csi_data, axis=1)  # Diff over time
    pad_width = ((0, 0), (0, 1), (0, 0), (0, 0))  # Pad time dim only
    diff_padded = np.pad(diff, pad_width, mode='constant')
    b, a = butter(4, 0.1, btype='low')
    # Loop over batch, sub, ant for filtfilt (1D per time series)
    for b_idx in range(diff_padded.shape[0]):
        for i in range(diff_padded.shape[2]):
            for j in range(diff_padded.shape[3]):
                diff_padded[b_idx, :, i, j] = filtfilt(b, a, diff_padded[b_idx, :, i, j])
    return diff_padded

# Feature Extraction Scheme (FES Model, Unchanged)
class FESModel(nn.Module):
    def __init__(self, num_classes=22, subcarriers=30, time_steps=500, hidden_size=64):
        super(FESModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        # Calculate CNN output size
        cnn_out_size = 64 * (time_steps//4) * (subcarriers//4)
        self.rnn_input_size = hidden_size  # Features per time step for RNN
        
        # Project CNN features to RNN sequence
        self.seq_len = cnn_out_size // self.rnn_input_size
        if self.seq_len == 0:
            self.seq_len = 1
            self.rnn_input_size = cnn_out_size
        
        self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (batch, 3, time, sub)
        cnn_out = self.cnn(x)  # (batch, cnn_out_size)
        
        # Reshape flattened CNN output into sequence for RNN
        cnn_out = cnn_out.view(batch_size, self.seq_len, self.rnn_input_size)
        
        rnn_out, _ = self.rnn(cnn_out)
        return self.fc(rnn_out[:, -1, :])  # Use last time step
def train_model(model, data, labels, epochs=10, lr=0.001, batch_size=8):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, batch_labels in loader:
            optimizer.zero_grad()
            out = model(batch_data)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss {total_loss/len(loader):.3f}")
    return model

# CARING FL (Unchanged)
class CARINGFL:
    def __init__(self, global_model, num_clients=3, num_rounds=5):
        self.global_model = global_model
        # FIXED: Access the last Linear layer (fc[3]) instead of fc[1] (ReLU)
        self.num_classes = global_model.fc[3].out_features
        self.clients = [FESModel(num_classes=self.num_classes) for _ in range(num_clients)]
        for client in self.clients:
            client.load_state_dict(global_model.state_dict())
        self.num_rounds = num_rounds
    
    def personalize_local(self, client_idx, dlp_data, dlp_labels):
        self.clients[client_idx] = train_model(self.clients[client_idx], dlp_data, dlp_labels, epochs=3)
    
    def weight_adaption(self, client_models):
        weights = []
        global_flat = self.global_model.fc[0].weight.detach().numpy().flatten().reshape(1, -1)
        for client in client_models:
            client_flat = client.fc[0].weight.detach().numpy().flatten().reshape(1, -1)
            sim = cosine_similarity(global_flat, client_flat)[0][0]
            weights.append(max(0.1, 1 - sim))
        weights = np.array(weights) / np.sum(weights)
        global_state = {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}
        for i, (state, w) in enumerate(zip([c.state_dict() for c in client_models], weights)):
            for k in global_state:
                global_state[k] += w * state[k]
        self.global_model.load_state_dict(global_state)
    
    def federate(self, dt_data, dt_labels, dl_data_list, dl_labels_list):
        client_data = np.array_split(dt_data, len(self.clients))
        client_labels = np.array_split(dt_labels, len(self.clients))
        for round in range(self.num_rounds):
            print(f"FL Round {round+1}/{self.num_rounds}")
            updates = []
            for c in range(len(self.clients)):
                self.clients[c] = train_model(self.clients[c], client_data[c], client_labels[c])
                updates.append(self.clients[c])
            self.weight_adaption(updates)
            for c in range(len(self.clients)):
                self.clients[c].load_state_dict(self.global_model.state_dict())
        for i, (dl_data, dl_labels) in enumerate(zip(dl_data_list, dl_labels_list)):
            dlp_data, dlp_labels = dl_data[:1], dl_labels[:1]
            self.personalize_local(i % len(self.clients), dlp_data, dlp_labels)
        return self.global_model
    
def evaluate(model, test_data, test_labels, num_classes=22):
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(torch.tensor(test_data, dtype=torch.float32)), dim=1).numpy()
    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average='weighted')
    cm = confusion_matrix(test_labels, pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    return acc, f1

# DTW Verification (Unchanged)
def plot_dtw(full_data):
    if len(full_data) < 2 or np.allclose(full_data[0], full_data[1]):
        print("Skipping DTW: Insufficient diversity (all samples similar)")
        return
    same_dtw = dtw(full_data[0].flatten(), full_data[1].flatten()).distance
    diff_dtw = dtw(full_data[0].flatten(), full_data[-1].flatten()).distance
    plt.figure()
    plt.bar(['Same Domain', 'Diff Domain'], [same_dtw, diff_dtw])
    plt.title('DTW Distances (Pre-NDS)')
    plt.savefig('dtw_comparison.png')
    plt.close()
    print(f"DTW Same: {same_dtw:.2f}, Diff: {diff_dtw:.2f} (Expect diff > same)")

def main(dataset='widar'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load & Formulate
    full_data, full_labels = load_data(dataset=dataset)
    plot_dtw(full_data)
    
    dt_data, dt_labels, dl_data_list, dl_labels_list = formulate_data(full_data, full_labels)
    print(f"Train split: {dt_data.shape}, Test domains: {len(dl_data_list)}")
    
    # Preprocess (NDS)
    preprocessed_dt = noise_dispelling(dt_data)
    preprocessed_dl = [noise_dispelling(d) for d in dl_data_list]
    print(f"Post-NDS shapes: Train {preprocessed_dt.shape}, DL {preprocessed_dl[0].shape if dl_data_list else 'N/A'}")
    
    # Baseline
    num_classes = len(np.unique(full_labels))
    baseline_model = FESModel(num_classes=num_classes)
    half_len = max(1, len(preprocessed_dt) // 2)
    baseline_model = train_model(baseline_model, preprocessed_dt[:half_len], dt_labels[:half_len])
    if dl_data_list:
        base_acc, base_f1 = evaluate(baseline_model, preprocessed_dl[0], dl_labels_list[0], num_classes)
        print(f"Baseline Acc: {base_acc:.2%}, F1: {base_f1:.2%}")
    
    # CARING FL
    global_model = FESModel(num_classes=num_classes)
    fl = CARINGFL(global_model, num_clients=3, num_rounds=5)
    trained_global = fl.federate(preprocessed_dt, dt_labels, preprocessed_dl[:2] if len(dl_data_list) > 1 else preprocessed_dl, dl_labels_list[:2] if len(dl_labels_list) > 1 else dl_labels_list)
    if dl_data_list:
        caring_acc, caring_f1 = evaluate(trained_global, preprocessed_dl[0], dl_labels_list[0], num_classes)
        print(f"CARING Acc: {caring_acc:.2%}, F1: {caring_f1:.2%}")
        print(f"Improvement: Acc +{(caring_acc - base_acc)*100:.1f}%, F1 +{(caring_f1 - base_f1)*100:.1f}%")
        
        # Imbalance
        if len(dl_labels_list) > 1:
            imb_pred = torch.argmax(trained_global(torch.tensor(preprocessed_dl[1], dtype=torch.float32)), dim=1).numpy()
            imb_f1 = f1_score(dl_labels_list[1], imb_pred, average='weighted')
            print(f"Imbalance F1 (CARING): {imb_f1:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='widar', choices=['widar', 'wiar', 'csi_har'])
    args = parser.parse_args()
    main(args.dataset)