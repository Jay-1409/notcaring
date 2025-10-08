import argparse
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from dtw import dtw

# ============================
# Widar3.0 .dat Parser
# ============================
def parse_dat_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        offset = 0
        time_series = []
        packet_count = 0
        while offset < len(data) - 100 and packet_count < 500:
            if data[offset:offset+3] != b'CSI':
                offset += 1
                continue
            offset += 8
            if offset + 16 >= len(data): break
            timestamp = struct.unpack('d', data[offset:offset+8])[0]
            offset += 8
            csi_len = struct.unpack('H', data[offset:offset+2])[0]
            offset += 2
            payload_size = csi_len * 2
            if offset + payload_size > len(data): break
            try:
                csi_raw = np.frombuffer(data[offset:offset+payload_size], dtype=np.float32)
                if len(csi_raw) < 360:
                    offset += payload_size
                    continue
                csi_real = csi_raw[::2][:270]
                csi_imag = csi_raw[1::2][:270]
                csi_complex = csi_real + 1j * csi_imag
                amp = np.abs(csi_complex.reshape(3, 3, 30))
                time_series.append(amp)
                packet_count += 1
            except struct.error:
                offset += payload_size
                continue
            offset += payload_size
        if not time_series:
            return np.zeros((500, 3, 3, 30))
        csi_matrix = np.stack(time_series)
        if csi_matrix.shape[0] < 500:
            csi_matrix = np.pad(csi_matrix, ((0, 500 - csi_matrix.shape[0]), (0,0), (0,0), (0,0)), mode='constant')
        else:
            csi_matrix = csi_matrix[:500]
        print(f"Parsed {os.path.basename(filepath)}: {csi_matrix.shape} ({packet_count} packets)")
        return csi_matrix.astype(np.float32)
    except Exception as e:
        print(f"Error parsing {os.path.basename(filepath)}: {e} -> using zeros")
        return np.zeros((500, 3, 3, 30)).astype(np.float32)

def extract_label_from_filename(filename):
    parts = filename.split('-')
    if len(parts) >= 2:
        return int(parts[1]) - 1
    return 0

# ============================
# Data Loading
# ============================
def load_data(dataset_dir='./data/widar3.0/raw', dataset='widar', max_files=None, min_activities=2, max_files_per_activity=50):
    if dataset == 'widar':
        all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.dat')]
        activity_groups = {}
        for fname in all_files:
            activity = extract_label_from_filename(fname)
            if activity not in activity_groups:
                activity_groups[activity] = []
            activity_groups[activity].append(fname)
        if len(activity_groups) < min_activities:
            raise ValueError(f"Insufficient activity diversity: {len(activity_groups)} < {min_activities}")
        selected_files = []
        for activity in sorted(activity_groups.keys()):
            files = activity_groups[activity][:max_files_per_activity]
            selected_files.extend(files)
        if max_files and len(selected_files) > max_files:
            files_per_activity = max_files // len(activity_groups)
            selected_files = []
            for activity in sorted(activity_groups.keys()):
                files = activity_groups[activity][:files_per_activity]
                selected_files.extend(files)
        data, labels = [], []
        for fname in selected_files:
            filepath = os.path.join(dataset_dir, fname)
            if not os.path.exists(filepath):
                continue
            csi_amp = parse_dat_file(filepath)
            label = extract_label_from_filename(fname)
            data.append(csi_amp)
            labels.append(label)
        if not data:
            raise ValueError("No data successfully loaded!")
        full_data = np.stack(data)
        full_labels = np.array(labels)
        unique_labels = np.unique(full_labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        full_labels = np.array([label_map[l] for l in full_labels])
        print(f"\nFinal Dataset: {full_data.shape}, Classes: {np.unique(full_labels)}")
    else:
        full_data = np.random.rand(100, 500, 3, 3, 30).astype(np.float32)
        full_labels = np.random.randint(0, 7, 100)
    return full_data.astype(np.float32), full_labels.astype(np.int64)

# ============================
# Formulate Data
# ============================
def formulate_data(full_data, full_labels, n_domains=3):
    domains = np.random.randint(0, n_domains, len(full_labels))
    dt_data, dt_labels = [], []
    dl_data_list, dl_labels_list = [], []
    for d in range(n_domains):
        mask = domains == d
        d_data, d_labels = full_data[mask], full_labels[mask]
        if len(d_data) < 1:
            continue
        if d < n_domains - 1:
            dt_data.append(d_data)
            dt_labels.append(d_labels)
        else:
            num_samples = len(d_data)
            imbalance_ratio = 0.5
            reduced_len = max(1, int(num_samples * imbalance_ratio))
            idx = np.random.choice(num_samples, reduced_len, replace=False)
            d_data = d_data[idx]
            d_labels = d_labels[idx]
            split1 = max(1, reduced_len // 3)
            split2 = max(1, (reduced_len - split1) // 2)
            dlp_data = d_data[:split1]
            dlp_labels = d_labels[:split1]
            dlw_data = d_data[split1:split1 + split2]
            dlw_labels = d_labels[split1:split1 + split2]
            dlt_data = d_data[split1 + split2:]
            dlt_labels = d_labels[split1 + split2:]
            dl_data_list.append((dlp_data, dlw_data, dlt_data))
            dl_labels_list.append((dlp_labels, dlw_labels, dlt_labels))
    if not dt_data:
        dt_data = [full_data]
        dt_labels = [full_labels]
        dl_data_list = [(full_data[:1], full_data[1:2], full_data[2:])]
        dl_labels_list = [(full_labels[:1], full_labels[1:2], full_labels[2:])]
    dt_concat = np.concatenate(dt_data) if len(dt_data) > 1 else dt_data[0]
    dt_l_concat = np.concatenate(dt_labels) if len(dt_labels) > 1 else dt_labels[0]
    return dt_concat, dt_l_concat, dl_data_list, dl_labels_list

# ============================
# Noise Dispelling Scheme (NDS)
# ============================
def noise_dispelling(csi_data):
    diff = np.diff(csi_data, axis=1)
    pad_width = ((0,0),(0,1),(0,0),(0,0),(0,0))
    diff_padded = np.pad(diff, pad_width, mode='edge')
    b, a = butter(4, 0.05, btype='low')
    for b_idx in range(diff_padded.shape[0]):
        for tx in range(3):
            for rx in range(3):
                for sub in range(30):
                    diff_padded[b_idx,:,tx,rx,sub] = filtfilt(b, a, diff_padded[b_idx,:,tx,rx,sub])
    return diff_padded

# ============================
# Feature Extraction Scheme (FES Model)
# ============================
class FESModel(nn.Module):
    def __init__(self, num_classes=22, time_steps=500, hidden_size=64):
        super(FESModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten(start_dim=2)
        )
        # Fixed calculation
        # Input to cnn: B, 1, T, 9, 30
        # After conv1 + pool1: B, 32, T/2, 9/2, 30/2 = B, 32, 250, 4, 15
        # After conv2 + pool2: B, 64, 250/2, 4/2, 15/2 = B, 64, 125, 2, 7
        # Flatten start_dim=2: B, 64, (125*2*7) = B, 64, 1750
        # Then view to B, 125, (64*2*7) = B, 125, 896
        self.rnn_input_size = 64 * 2 * 7  # 896
        self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, Tx, Rx, Sub = x.shape
        x = x.view(B, 1, T, Tx*Rx, Sub)  # B, 1, T, 9, 30
        cnn_out = self.cnn(x)  # B, 64, 125*2*7 = B, 64, 1750
        seq_len = T // 4  # 500/4 = 125
        cnn_out = cnn_out.view(B, seq_len, self.rnn_input_size)  # B, 125, 896
        rnn_out, _ = self.rnn(cnn_out)
        return self.fc(rnn_out[:, -1, :])

# ============================
# Train Model
# ============================
def train_model(model, data, labels, epochs=10, lr=0.001, batch_size=8):
    if len(data) == 0: return model
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

# ============================
# CARING FL
# ============================
class CARINGFL:
    def __init__(self, global_model, num_clients=3, num_rounds=5):
        self.global_model = global_model
        self.num_classes = global_model.fc[3].out_features
        self.clients = [FESModel(num_classes=self.num_classes) for _ in range(num_clients)]
        for client in self.clients:
            client.load_state_dict(global_model.state_dict())
        self.num_rounds = num_rounds
    def personalize_local(self, client_idx, dlp_data, dlp_labels):
        self.clients[client_idx] = train_model(self.clients[client_idx], dlp_data, dlp_labels, epochs=3)
    def weight_adaption(self, client_models):
        weights = []
        global_flat = nn.utils.parameters_to_vector(self.global_model.parameters()).detach().numpy().reshape(1, -1)
        for client in client_models:
            client_flat = nn.utils.parameters_to_vector(client.parameters()).detach().numpy().reshape(1, -1)
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
                if len(client_data[c]) == 0: continue
                self.clients[c] = train_model(self.clients[c], client_data[c], client_labels[c])
                updates.append(self.clients[c])
            self.weight_adaption(updates)
            for c in range(len(self.clients)):
                self.clients[c].load_state_dict(self.global_model.state_dict())
        for i in range(len(dl_data_list)):
            dlp_data, dlw_data, dlt_data = dl_data_list[i]
            dlp_labels, dlw_labels, dlt_labels = dl_labels_list[i]
            client_idx = i % len(self.clients)
            self.personalize_local(client_idx, dlp_data, dlp_labels)
            if len(dlw_data) > 0:
                self.clients[client_idx] = train_model(self.clients[client_idx], dlw_data, dlw_labels, epochs=3)
            self.weight_adaption([self.clients[client_idx]])
        return self.global_model

# ============================
# Evaluation
# ============================
def evaluate(model, test_data, test_labels, num_classes=22):
    if len(test_data) == 0: return 0, 0
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

# ============================
# DTW Plot
# ============================
def plot_dtw(full_data):
    if len(full_data) < 2 or np.allclose(full_data[0], full_data[1]):
        print("Skipping DTW: Insufficient diversity")
        return
    same_dtw = dtw(full_data[0].flatten(), full_data[1].flatten()).distance
    diff_dtw = dtw(full_data[0].flatten(), full_data[-1].flatten()).distance
    plt.figure()
    plt.bar(['Same Domain', 'Diff Domain'], [same_dtw, diff_dtw])
    plt.title('DTW Distances (Pre-NDS)')
    plt.savefig('dtw_comparison.png')
    plt.close()
    print(f"DTW Same: {same_dtw:.2f}, Diff: {diff_dtw:.2f}")

# ============================
# Main
# ============================
def main(dataset='widar'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    full_data, full_labels = load_data(dataset=dataset)
    plot_dtw(full_data)
    dt_data, dt_labels, dl_data_list, dl_labels_list = formulate_data(full_data, full_labels)
    print(f"Train split: {dt_data.shape}, Test domains: {len(dl_data_list)}")
    preprocessed_dt = noise_dispelling(dt_data)
    preprocessed_dl = [(noise_dispelling(d[0]), noise_dispelling(d[1]), noise_dispelling(d[2])) for d in dl_data_list]
    num_classes = len(np.unique(full_labels))
    baseline_model = FESModel(num_classes=num_classes)
    half_len = max(1, len(preprocessed_dt) // 2)
    baseline_model = train_model(baseline_model, preprocessed_dt[:half_len], dt_labels[:half_len])
    base_acc, base_f1 = 0, 0
    if dl_data_list:
        _, _, dlt = preprocessed_dl[0]
        _, _, dlt_l = dl_labels_list[0]
        base_acc, base_f1 = evaluate(baseline_model, dlt, dlt_l, num_classes)
        print(f"Baseline Acc: {base_acc:.2%}, F1: {base_f1:.2%}")
    global_model = FESModel(num_classes=num_classes)
    fl = CARINGFL(global_model)
    trained_global = fl.federate(preprocessed_dt, dt_labels, preprocessed_dl, dl_labels_list)
    caring_acc, caring_f1 = 0, 0
    if dl_data_list:
        _, _, dlt = preprocessed_dl[0]
        _, _, dlt_l = dl_labels_list[0]
        caring_acc, caring_f1 = evaluate(trained_global, dlt, dlt_l, num_classes)
        print(f"CARING Acc: {caring_acc:.2%}, F1: {caring_f1:.2%}")
        print(f"Improvement: Acc +{(caring_acc - base_acc)*100:.1f}%, F1 +{(caring_f1 - base_f1)*100:.1f}%")
        if len(preprocessed_dl) > 1:
            _, _, dlt2 = preprocessed_dl[1]
            _, _, dlt_l2 = dl_labels_list[1]
            imb_pred = torch.argmax(trained_global(torch.tensor(dlt2, dtype=torch.float32)), dim=1).numpy()
            imb_f1 = f1_score(dlt_l2, imb_pred, average='weighted')
            print(f"Imbalance F1 (CARING): {imb_f1:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='widar', choices=['widar', 'wiar', 'csi_har'])
    args = parser.parse_args()
    main(args.dataset)