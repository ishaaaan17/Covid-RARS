from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8_000.0
    top_db: int = 35
    cough_fixed_seconds: float = 4.0
    breath_fixed_seconds: float = 5.0
    speech_fixed_seconds: float = 5.0


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class QualityConfig:
    min_cough_seconds: float = 0.50
    min_breath_seconds: float = 1.00
    min_speech_seconds: float = 1.00
    max_silence_ratio: float = 0.70
    max_clipping_ratio: float = 0.01
    silence_db_threshold: float = 35.0


AUDIO_CONFIG = AudioConfig()
SPLIT_CONFIG = SplitConfig()
QUALITY_CONFIG = QualityConfig()
