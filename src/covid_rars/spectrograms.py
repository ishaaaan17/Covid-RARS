from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from covid_rars.audio_io import load_audio
from covid_rars.config import AUDIO_CONFIG, AudioConfig
from covid_rars.preprocess import preprocess_for_features


def log_mel_spectrogram(
    y: np.ndarray,
    sample_rate: int,
    config: AudioConfig = AUDIO_CONFIG,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel[np.newaxis, :, :].astype(np.float32)


def build_spectrogram_index(metadata: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    supervised = metadata[metadata["label_binary"].isin(["positive", "negative"])]
    for _, row in supervised.iterrows():
        y, sample_rate, _ = load_audio(Path(row["audio_path"]))
        processed, event_info = preprocess_for_features(y, sample_rate, str(row.get("modality", "unknown")))
        spec = log_mel_spectrogram(processed, sample_rate)
        spec_path = output_dir / f"{row['recording_id']}.npy"
        np.save(spec_path, spec)
        rows.append(
            {
                "recording_id": row["recording_id"],
                "participant_id": row["participant_id"],
                "dataset": row.get("dataset", "coswara"),
                "modality": row.get("modality", "unknown"),
                "submodality": row.get("submodality", "unknown"),
                "label_binary": row["label_binary"],
                "split": row.get("split", "unused"),
                "spectrogram_path": spec_path.as_posix(),
                "shape": "x".join(str(v) for v in spec.shape),
                **event_info,
            }
        )
    return pd.DataFrame(rows)
