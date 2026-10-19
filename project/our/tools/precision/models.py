"""
Model definitions for E0 precision study.

Three models representing different architectural families:
- MLP: Fully-connected (MNIST)
- CNN: ResNet-20 style convolutional (CIFAR-10)
- LSTM: Recurrent sequence classification (IMDB)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """3-layer MLP for MNIST (28x28 grayscale → 10 classes)."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-20."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10 (32x32 RGB → 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # ResNet-20: 3 stages, each with 3 blocks (n=3)
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleLSTM(nn.Module):
    """2-layer LSTM for sequence classification (IMDB sentiment)."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128,
                 hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        # Use final hidden state from last layer
        x = h_n[-1]  # (batch, hidden_dim)
        x = self.fc(x)
        return x


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to get model by name."""
    if model_name == 'mlp':
        return MLP(**kwargs)
    elif model_name == 'resnet20':
        return ResNet20(**kwargs)
    elif model_name == 'lstm':
        return SimpleLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
