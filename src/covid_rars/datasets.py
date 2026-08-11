from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from covid_rars.labels import label_to_int


@dataclass
class SpectrogramExample:
    x: np.ndarray
    y: int
    recording_id: str
    participant_id: str


class SpectrogramTableDataset:
    """Lightweight dataset usable from PyTorch training code and notebooks."""

    def __init__(self, index: pd.DataFrame, split: str, modality: str | None = None) -> None:
        df = index[index["split"] == split].copy()
        if modality is not None:
            df = df[df["modality"] == modality]
        df = df[df["label_binary"].isin(["positive", "negative"])]
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SpectrogramExample:
        row = self.df.iloc[idx]
        x = np.load(row["spectrogram_path"]).astype(np.float32)
        y = label_to_int(row["label_binary"])
        return SpectrogramExample(
            x=x,
            y=y,
            recording_id=str(row["recording_id"]),
            participant_id=str(row["participant_id"]),
        )
