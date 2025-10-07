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
# ==================================================
# 1. FIXED: Noise Dispelling Scheme (matches paper)
# ==================================================
def noise_dispelling(csi_data):
    """
    Paper Section 5.2: Remove domain factors via adjacent difference.
    diff(sx - sy) ≈ (Gx - Gy) where G is motion component
    """
    # Adjacent difference to remove static components
    diff = np.diff(csi_data, axis=1)  # (batch, time-1, sub, ant)
    
    # Pad to maintain shape
    pad_width = ((0, 0), (0, 1), (0, 0), (0, 0))
    diff_padded = np.pad(diff, pad_width, mode='edge')  # Use 'edge' instead of 'constant'
    
    # Apply low-pass Butterworth filter (preserves motion, removes high-freq noise)
    b, a = butter(4, 0.1, btype='low')
    
    filtered = np.zeros_like(diff_padded)
    for b_idx in range(diff_padded.shape[0]):
        for i in range(diff_padded.shape[2]):
            for j in range(diff_padded.shape[3]):
                filtered[b_idx, :, i, j] = filtfilt(b, a, diff_padded[b_idx, :, i, j])
    
    return filtered


# ==================================================
# 2. FIXED: Feature Extraction Model (matches Fig. 9)
# ==================================================
class FESModel(nn.Module):
    def __init__(self, num_classes=22, subcarriers=30, time_steps=500):
        super(FESModel, self).__init__()
        
        # CNN Layers (2 blocks as in paper)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Calculate dimensions after CNN
        # No pooling, so dimensions stay same
        cnn_out_channels = 64
        cnn_out_h = time_steps
        cnn_out_w = subcarriers
        
        # Flatten CNN output for BiLSTM
        self.flatten_size = cnn_out_channels * cnn_out_w  # Features per time step
        
        # Bi-LSTM Layers (2 blocks as in paper)
        self.bilstm1 = nn.LSTM(
            input_size=self.flatten_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # Critical: Bi-directional
        )
        self.bilstm2 = nn.LSTM(
            input_size=32 * 2,  # *2 for bidirectional
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * cnn_out_h, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, time, sub, 3)
        batch_size = x.size(0)
        
        # Permute for Conv2d: (batch, channels=3, height=time, width=sub)
        x = x.permute(0, 3, 1, 2)
        
        # CNN blocks
        x = self.cnn1(x)  # (batch, 128, time, sub)
        x = self.cnn2(x)  # (batch, 64, time, sub)
        
        # Prepare for BiLSTM: treat time as sequence dimension
        # (batch, 64, time, sub) -> (batch, time, 64*sub)
        x = x.permute(0, 2, 1, 3)  # (batch, time, 64, sub)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, 64*sub)
        
        # BiLSTM blocks
        x, _ = self.bilstm1(x)  # (batch, time, 32*2)
        x, _ = self.bilstm2(x)  # (batch, time, 16*2)
        
        # Fully connected
        out = self.fc(x)  # (batch, num_classes)
        
        return out


# ==================================================
# 3. FIXED: Data Formulation (matches Section 5.1)
# ==================================================
def formulate_data(full_data, full_labels, n_domains=3):
    """
    Paper Section 5.1: Proper domain split with Dlp, Dlw, Dlt
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Stratified domain assignment
    unique_classes = np.unique(full_labels)
    domains = np.zeros(len(full_labels), dtype=int)
    
    for cls in unique_classes:
        cls_mask = full_labels == cls
        cls_indices = np.where(cls_mask)[0]
        cls_domains = np.random.randint(0, n_domains, len(cls_indices))
        domains[cls_indices] = cls_domains
    
    dt_data, dt_labels = [], []
    dl_data_dict = {}  # Store Dlp, Dlw, Dlt separately
    
    for d in range(n_domains):
        mask = domains == d
        d_data, d_labels = full_data[mask], full_labels[mask]
        
        print(f"  Domain {d}: {len(d_data)} samples, classes {np.unique(d_labels)}")
        
        if len(d_data) < 1:
            continue
        
        if d < n_domains - 2:  # Training domains
            dt_data.append(d_data)
            dt_labels.append(d_labels)
        else:  # New domains
            # Paper: Dlp and Dlw each have < 2 samples per class
            unique_cls = np.unique(d_labels)
            
            # Take 1 sample per class for Dlp (personalization)
            dlp_data, dlp_labels = [], []
            for cls in unique_cls:
                cls_mask = d_labels == cls
                cls_samples = d_data[cls_mask][:1]  # 1 sample
                dlp_data.append(cls_samples)
                dlp_labels.append([cls])
            
            # Take 1 more sample per class for Dlw (weight adaptation)
            dlw_data, dlw_labels = [], []
            for cls in unique_cls:
                cls_mask = d_labels == cls
                cls_samples = d_data[cls_mask][1:2]  # Next 1 sample
                if len(cls_samples) > 0:
                    dlw_data.append(cls_samples)
                    dlw_labels.append([cls])
            
            # Rest for Dlt (testing)
            dlt_data, dlt_labels = [], []
            for cls in unique_cls:
                cls_mask = d_labels == cls
                cls_samples = d_data[cls_mask][2:]  # Remaining
                if len(cls_samples) > 0:
                    dlt_data.append(cls_samples)
                    dlt_labels.append([cls] * len(cls_samples))
            
            # Concatenate
            if len(dlp_data):
                dlp_data = np.concatenate(dlp_data)
                dlp_labels = np.concatenate(dlp_labels)
            if len(dlp_data):
                dlw_data = np.concatenate(dlw_data)
                dlw_labels = np.concatenate(dlw_labels)
            if len(dlp_data):
                dlt_data = np.concatenate(dlt_data)
                dlt_labels = np.concatenate(dlt_labels)
            
            # FIXED:
            dl_data_dict[d] = {
                'dlp': (dlp_data, dlp_labels) if len(dlp_data) > 0 else (np.array([]), np.array([])),
                'dlw': (dlw_data, dlw_labels) if len(dlw_data) > 0 else (np.array([]), np.array([])),
                'dlt': (dlt_data, dlt_labels) if len(dlt_data) > 0 else (np.array([]), np.array([]))
            }
    
    dt_concat = np.concatenate(dt_data) if len(dt_data) > 1 else dt_data[0]
    dt_l_concat = np.concatenate(dt_labels) if len(dt_labels) > 1 else dt_labels[0]
    
    return dt_concat, dt_l_concat, dl_data_dict

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
# ==================================================
# 4. FIXED: Weight Adaptation (matches Section 5.4.2)
# ==================================================
class CARINGFL:
    def __init__(self, global_model, num_clients=3, num_rounds=5):
            self.global_model = global_model
            # FIXED: Access the last Linear layer (fc[4]) instead of fc[3]
            self.num_classes = global_model.fc[4].out_features
            self.clients = [FESModel(num_classes=self.num_classes) for _ in range(num_clients)]
            for client in self.clients:
                client.load_state_dict(global_model.state_dict())
            self.num_rounds = num_rounds
            self.client_weights = np.ones(num_clients)
    
    def weight_adaption(self, client_models, dlw_data, dlw_labels):
        """
        Paper Section 5.4.2: Use actual performance on Dlw to determine weights.
        λk = 1.0 if accuracy improves, λk = 0.3 if accuracy degrades
        """
        if len(dlw_data) == 0:
            return np.ones(len(client_models))
        
        weights = []
        dlw_tensor = torch.tensor(dlw_data, dtype=torch.float32)
        dlw_labels_tensor = torch.tensor(dlw_labels, dtype=torch.long)
        
        for i, client in enumerate(client_models):
            client.eval()
            with torch.no_grad():
                pred = torch.argmax(client(dlw_tensor), dim=1)
                accuracy = (pred == dlw_labels_tensor).float().mean().item()
            
            # Paper: λk = 1 if good performance, λk = 0.3 if poor
            # Use threshold of 50% accuracy
            if accuracy >= 0.5:
                weights.append(1.0)
            else:
                weights.append(0.3)
        
        return np.array(weights)
    
    def federate(self, dt_data, dt_labels, dl_data_dict):
        """Algorithm 1 from paper"""
        client_data = np.array_split(dt_data, len(self.clients))
        client_labels = np.array_split(dt_labels, len(self.clients))
        
        for round_idx in range(self.num_rounds):
            print(f"FL Round {round_idx+1}/{self.num_rounds}")
            
            # Train local models
            for c in range(len(self.clients)):
                self.clients[c] = train_model(self.clients[c], client_data[c], client_labels[c], epochs=10)
            
            # Personalize on Dlp
            for domain_id, splits in dl_data_dict.items():
                dlp_data, dlp_labels = splits['dlp']
                if len(dlp_data) > 0:
                    client_idx = domain_id % len(self.clients)
                    self.clients[client_idx] = train_model(
                        self.clients[client_idx], dlp_data, dlp_labels, epochs=3
                    )
            
            # Weight adaptation using Dlw
            all_dlw_data, all_dlw_labels = [], []
            for splits in dl_data_dict.values():
                dlw_data, dlw_labels = splits['dlw']
                if len(dlw_data) > 0:
                    all_dlw_data.append(dlw_data)
                    all_dlw_labels.append(dlw_labels)
            
            if all_dlw_data:
                all_dlw_data = np.concatenate(all_dlw_data)
                all_dlw_labels = np.concatenate(all_dlw_labels)
                weights = self.weight_adaption(self.clients, all_dlw_data, all_dlw_labels)
            else:
                weights = np.ones(len(self.clients))
            
            # FIXED: Handle different tensor types properly
            new_global_state = {}
            weight_sum = np.sum(weights)
            
            for k in self.global_model.state_dict().keys():
                param = self.global_model.state_dict()[k]
                
                # Only aggregate Float tensors (trainable parameters)
                if param.dtype in [torch.float32, torch.float16, torch.float64]:
                    weighted_param = torch.zeros_like(param)
                    for i, client in enumerate(self.clients):
                        weighted_param += (weights[i] / weight_sum) * client.state_dict()[k]
                    new_global_state[k] = weighted_param
                else:
                    # Copy non-float tensors from first client
                    new_global_state[k] = self.clients[0].state_dict()[k].clone()
            
            self.global_model.load_state_dict(new_global_state)
            
            # Sync clients with global model
            for c in range(len(self.clients)):
                self.clients[c].load_state_dict(self.global_model.state_dict())
        
        return self.global_model

def evaluate(model, test_data, test_labels, num_classes=22):
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        pred = torch.argmax(model(test_tensor), dim=1).numpy()
    
    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, pred, labels=range(num_classes))
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

def main(dataset='widar', use_location=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load & Formulate
    full_data, full_labels = load_data(
        dataset=dataset, 
        min_activities=2,  # At least 2 for your current data
        max_files_per_activity=100
    )
    plot_dtw(full_data)
    
    # NEW: Returns dict instead of lists
    dt_data, dt_labels, dl_data_dict = formulate_data(full_data, full_labels)
    
    # Extract test data from the last new domain
    test_domain_id = max(dl_data_dict.keys())
    test_data = dl_data_dict[test_domain_id]['dlt'][0]
    test_labels = dl_data_dict[test_domain_id]['dlt'][1]
    
    print(f"Train: {dt_data.shape}, Test: {test_data.shape}")
    
    # Preprocess (NDS)
    preprocessed_dt = noise_dispelling(dt_data)
    preprocessed_test = noise_dispelling(test_data)
    
    # Preprocess new domains
    preprocessed_dl_dict = {}
    for domain_id, splits in dl_data_dict.items():
        preprocessed_dl_dict[domain_id] = {
            'dlp': (noise_dispelling(splits['dlp'][0]), splits['dlp'][1]) if len(splits['dlp'][0]) > 0 else (np.array([]), np.array([])),
            'dlw': (noise_dispelling(splits['dlw'][0]), splits['dlw'][1]) if len(splits['dlw'][0]) > 0 else (np.array([]), np.array([])),
            'dlt': (noise_dispelling(splits['dlt'][0]), splits['dlt'][1]) if len(splits['dlt'][0]) > 0 else (np.array([]), np.array([]))
        }
    
    num_classes = len(np.unique(full_labels))
    print(f"Number of classes: {num_classes}\n")
    
    # === BASELINE MODEL ===
    print("="*60)
    print("Training BASELINE Model")
    print("="*60)
    baseline_model = FESModel(num_classes=num_classes)
    baseline_model = train_model(baseline_model, preprocessed_dt, dt_labels, epochs=15)
    base_acc, base_f1 = evaluate(baseline_model, preprocessed_test, test_labels, num_classes)
    print(f"\nBaseline: Acc={base_acc:.2%}, F1={base_f1:.2%}\n")
    
    # === CARING FL MODEL ===
    print("="*60)
    print("Training CARING FL Model")
    print("="*60)
    global_model = FESModel(num_classes=num_classes)
    num_clients = min(3, max(2, len(preprocessed_dt) // 10))
    fl = CARINGFL(global_model, num_clients=num_clients, num_rounds=1)
    
    # Train with new dict structure
    trained_global = fl.federate(preprocessed_dt, dt_labels, preprocessed_dl_dict)
    
    caring_acc, caring_f1 = evaluate(trained_global, preprocessed_test, test_labels, num_classes)
    print(f"\nCARING FL: Acc={caring_acc:.2%}, F1={caring_f1:.2%}\n")
    
    # === COMPARISON ===
    print("="*60)
    print("Final Comparison")
    print("="*60)
    print(f"Baseline:  Acc={base_acc:.2%}, F1={base_f1:.2%}")
    print(f"CARING FL: Acc={caring_acc:.2%}, F1={caring_f1:.2%}")
    
    acc_improvement = (caring_acc - base_acc) * 100
    f1_improvement = (caring_f1 - base_f1) * 100
    
    print(f"\nImprovement: Acc={acc_improvement:+.1f}pp, F1={f1_improvement:+.1f}pp")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='widar', choices=['widar', 'wiar', 'csi_har'])
    args = parser.parse_args()
    main(args.dataset)