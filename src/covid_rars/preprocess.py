from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from covid_rars.audio_io import fixed_length_samples, pad_or_crop_center, peak_normalize
from covid_rars.config import AUDIO_CONFIG, AudioConfig


@dataclass(frozen=True)
class EventWindow:
    start_sample: int
    end_sample: int
    method: str

    @property
    def duration_samples(self) -> int:
        return max(0, self.end_sample - self.start_sample)


def trim_silence(y: np.ndarray, top_db: int = AUDIO_CONFIG.top_db) -> tuple[np.ndarray, tuple[int, int]]:
    if y.size == 0:
        return y, (0, 0)
    trimmed, index = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size == 0:
        return y, (0, y.size)
    return trimmed.astype(np.float32, copy=False), (int(index[0]), int(index[1]))


def crop_or_pad_audio(y: np.ndarray, target_samples: int) -> np.ndarray:
    """Compatibility wrapper for center crop/pad used by notebooks and tests."""
    return pad_or_crop_center(y, target_samples)


def detect_active_event(
    y: np.ndarray,
    sample_rate: int,
    modality: str,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    threshold_ratio: float = 0.25,
    margin_ms: float = 120.0,
) -> EventWindow:
    """Detect an active event region using RMS energy.

    This is intentionally simple and auditable. It is not a medical segmentation model.
    """
    if y.size == 0:
        return EventWindow(0, 0, "empty")

    frame_length = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, int(sample_rate * hop_ms / 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    if rms.size == 0 or float(np.max(rms)) <= 1e-8:
        return EventWindow(0, y.size, "flat_signal")

    threshold = float(np.max(rms)) * threshold_ratio
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return EventWindow(0, y.size, "no_active_frame")

    start_frame = int(active[0])
    end_frame = int(active[-1])
    margin = int(sample_rate * margin_ms / 1000.0)
    start_sample = max(0, start_frame * hop_length - margin)
    end_sample = min(y.size, end_frame * hop_length + frame_length + margin)
    method = "rms_event_cough" if modality == "cough" else "rms_active_region"
    return EventWindow(start_sample, end_sample, method)


def preprocess_for_features(
    y: np.ndarray,
    sample_rate: int,
    modality: str,
    config: AudioConfig = AUDIO_CONFIG,
) -> tuple[np.ndarray, dict[str, float | str]]:
    trimmed, trim_index = trim_silence(y, top_db=config.top_db)
    event = detect_active_event(trimmed, sample_rate=sample_rate, modality=modality)
    active = trimmed[event.start_sample : event.end_sample]
    if active.size == 0:
        active = trimmed
    active = peak_normalize(active)
    target = fixed_length_samples(modality, sample_rate, config=config)
    fixed = pad_or_crop_center(active, target)

    offset = trim_index[0]
    event_start = (offset + event.start_sample) / sample_rate
    event_end = (offset + event.end_sample) / sample_rate
    info = {
        "event_start_sec": float(event_start),
        "event_end_sec": float(event_end),
        "event_duration_sec": float(max(0.0, event_end - event_start)),
        "active_audio_ratio": float(active.size / max(1, y.size)),
        "segmentation_method": event.method,
    }
    return fixed, info
