# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class TargetEncoder(nn.Module):
    def __init__(self, input_channels, input_length, target_length, output_channels=1):
        super().__init__()
        self.target_length = target_length
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=output_channels, kernel_size=3, padding=1)

        self.stride = max(1, target_length // input_length)
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=self.stride * 2,
            stride=self.stride,
            padding=self.stride // 2,
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv_transpose(x)
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x).squeeze(2)
        x = torch.sigmoid(self.fc(x))
        return x
