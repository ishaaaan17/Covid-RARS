from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_paper_193_features_from_audio(
    y: np.ndarray,
    sr: int = 22050,
) -> np.ndarray:
    """Extract the exact 193 acoustic features used in Islam et al. (ESWA 2026 / arXiv:2501.01117).

    Feature Breakdown:
    - 40 MFCCs (mean over time) -> 40
    - 12 Chroma STFT (mean over time) -> 12
    - 128 Mel Spectrogram (mean over time) -> 128
    - 7 Spectral Contrast (mean over time) -> 7
    - 6 Tonnetz (mean over time) -> 6
    Total = 40 + 12 + 128 + 7 + 6 = 193 features.
    """
    import librosa

    # Ensure valid audio array
    if y is None or len(y) == 0:
        return np.zeros(193, dtype=np.float32)

    # 1. 40 MFCCs
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    except Exception:
        mfcc = np.zeros(40, dtype=np.float32)

    # 2. 12 Chroma STFT
    try:
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    except Exception:
        chroma = np.zeros(12, dtype=np.float32)

    # 3. 128 Mel Spectrogram
    try:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    except Exception:
        mel = np.zeros(128, dtype=np.float32)

    # 4. 7 Spectral Contrast
    try:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    except Exception:
        contrast = np.zeros(7, dtype=np.float32)

    # 5. 6 Tonnetz
    try:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    except Exception:
        tonnetz = np.zeros(6, dtype=np.float32)

    features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
    if len(features) != 193:
        # Fallback pad/truncate to exactly 193
        features = np.pad(features, (0, max(0, 193 - len(features))))[:193]

    return features.astype(np.float32)


def get_paper_193_feature_names() -> list[str]:
    """Get the 193 column names matching the exact paper feature bank."""
    names: list[str] = []
    names.extend([f"mfcc_{i+1}" for i in range(40)])
    names.extend([f"chroma_{i+1}" for i in range(12)])
    names.extend([f"mel_{i+1}" for i in range(128)])
    names.extend([f"contrast_{i+1}" for i in range(7)])
    names.extend([f"tonnetz_{i+1}" for i in range(6)])
    return names
