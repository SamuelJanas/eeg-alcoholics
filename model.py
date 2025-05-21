import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from load_data import get_dataset
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

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (B, 1, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_dataloaders(X, y, batch_size=32, val_split=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_split)

    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    stats = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(y_batch).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        stats["train_loss"].append(avg_train_loss)
        stats["val_loss"].append(avg_val_loss)
        stats["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return stats

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
    plt.show()

def save_training_stats(stats, filename="training_stats.csv"):
    pd.DataFrame(stats).to_csv(filename, index=False)
