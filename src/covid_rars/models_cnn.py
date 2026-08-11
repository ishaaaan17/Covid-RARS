from __future__ import annotations

import torch
from torch import nn


class CompactSpectrogramCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualSpectrogramCNN(nn.Module):
    """Deeper residual log-mel classifier for stronger internal baselines."""

    def __init__(self, base_channels: int = 32, dropout: float = 0.35) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(ResidualBlock(base_channels), ResidualBlock(base_channels))
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(ResidualBlock(base_channels * 2), ResidualBlock(base_channels * 2))
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(ResidualBlock(base_channels * 4), ResidualBlock(base_channels * 4))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


class CNNBiGRUSpectrogram(nn.Module):
    """CNN front-end plus bidirectional GRU over time frames."""

    def __init__(self, channels: int = 32, hidden_size: int = 96, dropout: float = 0.35) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=channels * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = x.mean(dim=2).transpose(1, 2)
        seq, _ = self.gru(x)
        pooled = seq.mean(dim=1)
        return self.classifier(pooled).squeeze(1)


def make_spectrogram_model(architecture: str = "compact_cnn") -> nn.Module:
    if architecture == "compact_cnn":
        return CompactSpectrogramCNN()
    if architecture == "residual_cnn":
        return ResidualSpectrogramCNN()
    if architecture == "cnn_bigru":
        return CNNBiGRUSpectrogram()
    raise ValueError(f"Unknown spectrogram architecture: {architecture}")
