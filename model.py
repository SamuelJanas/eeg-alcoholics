import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture from: https://arxiv.org/pdf/1611.08024
class EEGNet(nn.Module):
    def __init__(self, 
                 num_channels=64, 
                 num_time_points=128, 
                 num_classes=4, 
                 F1=8, 
                 D=2, 
                 F2=16, 
                 dropout_rate=0.5):
        super(EEGNet, self).__init__()

        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(
            F1, D * F1, kernel_size=(num_channels, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2
        self.separable_conv = nn.Sequential(
            nn.Conv2d(D * F1, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)  # pointwise
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier
        self.classifier = nn.Linear(F2 * (num_time_points // 32), num_classes)

    def forward(self, x):
        # Input: (batch, 1, channels, time) e.g. (B, 1, 64, 128)

        x = self.conv1(x)         # (B, F1, C, T)
        x = self.bn1(x)

        x = self.depthwise_conv(x)  # (B, D*F1, 1, T)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.separable_conv(x)  # (B, F2, 1, T//4)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)     # (B, num_classes)
        return F.log_softmax(x, dim=1)
    
# architecture from https://www.mdpi.com/2076-3425/12/6/778
class CNN_BiLSTM(nn.Module):
    def __init__(self,
                 input_length=256,        # time points per trial
                 num_channels=64,         # EEG channels
                 num_classes=2,           # classification targets
                 conv_channels=[64, 64, 128, 128],
                 lstm_hidden=64,
                 fc_dims=[256, 128, 64],
                 dropout_rate=0.5):
        super(CNN_BiLSTM, self).__init__()

        # 1D convolutions over time axis, input: (B, C, T)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=conv_channels[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=conv_channels[2], out_channels=conv_channels[3], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # BiLSTM input: (B, T, Features)
        reduced_length = input_length // 4  # due to 2 MaxPool1d layers (factor 2 each)
        self.bi_lstm = nn.LSTM(input_size=conv_channels[3],
                               hidden_size=lstm_hidden,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

        self.fc1 = nn.Linear(reduced_length * 2 * lstm_hidden, fc_dims[0])
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(fc_dims[1], fc_dims[2])
        self.drop3 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(fc_dims[2], num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        # reshape for LSTM: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        x, _ = self.bi_lstm(x)  # (B, T, 2 * hidden)

        x = x.contiguous().view(x.size(0), -1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.output(x)

        return F.log_softmax(x, dim=1)

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (B, 1, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_dataloaders(X, y, id, batch_size=32, val_split=0.2):
    unique_ids = np.unique(id)
    train_ids, val_ids = train_test_split(unique_ids, test_size=val_split, stratify=None, random_state=24)

    train_mask = np.isin(id, train_ids)
    val_mask = np.isin(id, val_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train set unique IDs: {np.unique(id[train_mask])}")
    print(f"Val set unique IDs: {np.unique(id[val_mask])}")
    
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_model(
    model,
    train_loader,
    val_loader,
    *,
    epochs=50,
    lr=1e-3,
    device="cpu",
    patience=10,
    min_delta=0.0,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    stats = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            if model.__class__.__name__ == 'CNN_BiLSTM':
                X_batch = X_batch.squeeze(1)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                if model.__class__.__name__ == 'CNN_BiLSTM':
                    X_batch = X_batch.squeeze(1)
                
                output = model(X_batch)
                loss = criterion(output, y_batch)
                running_val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(y_batch).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        stats["train_loss"].append(avg_train_loss)
        stats["val_loss"].append(avg_val_loss)
        stats["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f} | Acc {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stop: no improvement in val-loss for "
                    f"{patience} consecutive epochs."
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, stats

def plot_metrics(stats):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(stats["train_loss"], label="Train Loss")
    plt.plot(stats["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(stats["val_acc"], label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)   
    plt.savefig("figures/training_metrics.png")
    plt.show()

def save_training_stats(stats, filename="training_stats.csv"):
    pd.DataFrame(stats).to_csv(filename, index=False)
