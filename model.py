import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from load_data import get_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture from: https://arxiv.org/pdf/1611.08024
class simple_EEGNet(nn.Module):
    def __init__(self, 
                 num_channels=64, 
                 num_time_points=128, 
                 num_classes=4, 
                 F1=8, 
                 D=2, 
                 F2=16, 
                 dropout_rate=0.5):
        super(simple_EEGNet, self).__init__()

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
