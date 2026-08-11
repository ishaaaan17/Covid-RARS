from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from covid_rars.audio_io import load_audio, pad_or_crop_center, peak_normalize
from covid_rars.preprocess import detect_active_event, trim_silence
from covid_rars.strong_features import augment_waveform


SOTA_PREPROCESSING_VERSION = "sota_active_segments_v1"
SUPPORTED_SOTA_SPLITS = {"train", "validation", "test", "external_test"}


@dataclass(frozen=True)
class SegmentConfig:
    window_sec: float = 3.0
    overlap: float = 0.5
    max_segments_per_recording: int = 8
    quality_mode: str = "quality_ok_only"
    augment_train_copies: int = 0
    random_state: int = 42

    @property
    def hop_fraction(self) -> float:
        return max(0.05, 1.0 - min(0.95, max(0.0, self.overlap)))


def rms_normalize(y: np.ndarray, target_dbfs: float = -23.0, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.size == 0:
        return arr
    rms = float(np.sqrt(np.mean(np.square(arr))))
    if rms < eps:
        return arr
    target = float(10.0 ** (target_dbfs / 20.0))
    out = arr * (target / max(rms, eps))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _stable_seed(*parts: object, base_seed: int = 42) -> int:
    text = "::".join(str(part) for part in parts)
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value *= 16777619
        value &= 0xFFFFFFFF
    return int((value + int(base_seed)) % (2**32 - 1))


def _segment_starts(
    active_start: int,
    active_end: int,
    sample_rate: int,
    config: SegmentConfig,
) -> list[tuple[int, int]]:
    window = max(1, int(round(config.window_sec * sample_rate)))
    hop = max(1, int(round(window * config.hop_fraction)))
    active_start = max(0, int(active_start))
    active_end = max(active_start, int(active_end))
    if active_end <= active_start:
        return [(active_start, active_start + window)]
    active_len = active_end - active_start
    if active_len <= window:
        return [(active_start, active_end)]

    starts = list(range(active_start, active_end - window + 1, hop))
    final_start = max(active_start, active_end - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    starts = sorted(set(starts))
    if config.max_segments_per_recording > 0 and len(starts) > config.max_segments_per_recording:
        indices = np.linspace(0, len(starts) - 1, config.max_segments_per_recording)
        starts = [starts[int(round(i))] for i in indices]
        starts = sorted(set(starts))
    return [(start, min(active_end, start + window)) for start in starts]


def _filter_metadata(
    metadata: pd.DataFrame,
    modalities: Iterable[str] | None,
    quality_mode: str,
) -> pd.DataFrame:
    df = metadata.copy()
    if "split" not in df.columns:
        df["split"] = "unused"
    if modalities is not None:
        allowed = {str(m) for m in modalities}
        df = df[df["modality"].astype(str).isin(allowed)]
    df = df[df["label_binary"].isin(["positive", "negative"])]
    df = df[df["split"].astype(str).isin(SUPPORTED_SOTA_SPLITS)]
    if quality_mode == "quality_ok_only" and "quality_flag" in df.columns:
        df = df[df["quality_flag"].astype(str).str.lower().eq("ok")]
    elif quality_mode not in {"all_samples", "quality_ok_only"}:
        raise ValueError(f"Unknown quality_mode: {quality_mode}")
    required = {"recording_id", "participant_id", "modality", "label_binary", "split", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required SOTA segment columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def build_sota_segment_index(
    metadata: pd.DataFrame,
    modalities: Iterable[str] | None = None,
    quality_mode: str = "quality_ok_only",
    window_sec: float = 3.0,
    overlap: float = 0.5,
    max_segments_per_recording: int = 8,
    augment_train_copies: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a leakage-safe segment table for deep/foundation audio branches.

    Segments inherit the participant split from their source recording. Training
    augmentation is represented as extra segment rows only for train rows, so
    validation/test metrics remain untouched.
    """
    config = SegmentConfig(
        window_sec=float(window_sec),
        overlap=float(overlap),
        max_segments_per_recording=int(max_segments_per_recording),
        quality_mode=str(quality_mode),
        augment_train_copies=int(augment_train_copies),
        random_state=int(random_state),
    )
    df = _filter_metadata(metadata, modalities=modalities, quality_mode=config.quality_mode)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        recording_id = str(row["recording_id"])
        try:
            y, sample_rate, original_sample_rate = load_audio(Path(row["audio_path"]))
        except Exception as exc:
            rows.append(
                {
                    "segment_id": f"{recording_id}::load_error",
                    "recording_id": recording_id,
                    "source_recording_id": recording_id,
                    "participant_id": str(row["participant_id"]),
                    "dataset": row.get("dataset", "unknown"),
                    "modality": row.get("modality", "unknown"),
                    "submodality": row.get("submodality", row.get("modality", "unknown")),
                    "label_binary": row["label_binary"],
                    "split": row["split"],
                    "audio_path": row["audio_path"],
                    "segment_start_sample": 0,
                    "segment_end_sample": 0,
                    "segment_index": 0,
                    "segment_count": 0,
                    "is_augmented": False,
                    "augmentation_id": "original",
                    "augmentation_seed": int(config.random_state),
                    "preprocessing_version": SOTA_PREPROCESSING_VERSION,
                    "skip_reason": f"load_error: {exc}",
                }
            )
            continue

        trimmed, trim_index = trim_silence(y)
        event = detect_active_event(trimmed, sample_rate=sample_rate, modality=str(row.get("modality", "unknown")))
        active_start = int(trim_index[0] + event.start_sample)
        active_end = int(trim_index[0] + event.end_sample)
        windows = _segment_starts(active_start, active_end, sample_rate=sample_rate, config=config)
        source_rows: list[dict[str, object]] = []
        for segment_index, (start, end) in enumerate(windows):
            start = max(0, min(int(start), int(y.size)))
            end = max(start + 1, min(int(end), int(y.size)))
            source_rows.append(
                {
                    "segment_id": f"{recording_id}::seg{segment_index:03d}",
                    "recording_id": recording_id,
                    "source_recording_id": recording_id,
                    "participant_id": str(row["participant_id"]),
                    "dataset": row.get("dataset", "unknown"),
                    "modality": row.get("modality", "unknown"),
                    "submodality": row.get("submodality", row.get("modality", "unknown")),
                    "label_binary": row["label_binary"],
                    "split": row["split"],
                    "audio_path": row["audio_path"],
                    "sample_rate": int(sample_rate),
                    "original_sample_rate": int(original_sample_rate),
                    "segment_start_sample": int(start),
                    "segment_end_sample": int(end),
                    "segment_start_sec": float(start / sample_rate),
                    "segment_end_sec": float(end / sample_rate),
                    "segment_duration_sec": float((end - start) / sample_rate),
                    "segment_index": int(segment_index),
                    "segment_count": int(len(windows)),
                    "active_start_sec": float(active_start / sample_rate),
                    "active_end_sec": float(active_end / sample_rate),
                    "segmentation_method": event.method,
                    "is_augmented": False,
                    "augmentation_id": "original",
                    "augmentation_seed": int(config.random_state),
                    "preprocessing_version": SOTA_PREPROCESSING_VERSION,
                    "skip_reason": "",
                }
            )
        rows.extend(source_rows)
        if str(row["split"]) == "train" and config.augment_train_copies > 0:
            for source in source_rows:
                for copy_idx in range(1, config.augment_train_copies + 1):
                    augmented = dict(source)
                    augmented["segment_id"] = f"{source['segment_id']}::aug{copy_idx}"
                    augmented["is_augmented"] = True
                    augmented["augmentation_id"] = f"aug{copy_idx}"
                    augmented["augmentation_seed"] = _stable_seed(source["segment_id"], copy_idx, base_seed=config.random_state)
                    rows.append(augmented)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["skip_reason"].fillna("").eq("")].reset_index(drop=True)
    return out


def validate_segment_index_no_leakage(segment_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if segment_index.empty:
        return pd.DataFrame([{"severity": "error", "check": "segment_index", "message": "segment index is empty"}])

    split_counts = segment_index.groupby("participant_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        rows.append(
            {
                "severity": "error",
                "check": "participant_split_leakage",
                "message": f"{len(leaking)} participants appear in multiple splits",
            }
        )
    augmented_non_train = segment_index[segment_index["is_augmented"].fillna(False).astype(bool) & ~segment_index["split"].eq("train")]
    if not augmented_non_train.empty:
        rows.append(
            {
                "severity": "error",
                "check": "augmentation_split_leakage",
                "message": f"{len(augmented_non_train)} augmented segments are outside train split",
            }
        )
    invalid_window = segment_index[
        pd.to_numeric(segment_index["segment_end_sample"], errors="coerce")
        <= pd.to_numeric(segment_index["segment_start_sample"], errors="coerce")
    ]
    if not invalid_window.empty:
        rows.append(
            {
                "severity": "error",
                "check": "invalid_segment_window",
                "message": f"{len(invalid_window)} segments have non-positive duration",
            }
        )
    duplicate_ids = int(segment_index["segment_id"].duplicated().sum()) if "segment_id" in segment_index.columns else len(segment_index)
    if duplicate_ids:
        rows.append(
            {
                "severity": "error",
                "check": "duplicate_segment_id",
                "message": f"{duplicate_ids} duplicated segment IDs",
            }
        )
    if not rows:
        rows.append({"severity": "ok", "check": "segment_index", "message": "no leakage detected"})
    return pd.DataFrame(rows)


def load_sota_segment_waveform(
    segment_row: pd.Series | dict[str, object],
    target_samples: int = 48000,
    rms_dbfs: float = -23.0,
) -> np.ndarray:
    row = pd.Series(segment_row)
    y, sample_rate, _ = load_audio(Path(row["audio_path"]))
    start = max(0, int(row.get("segment_start_sample", 0)))
    end = min(int(y.size), max(start + 1, int(row.get("segment_end_sample", y.size))))
    segment = y[start:end]
    segment = peak_normalize(segment)
    if bool(row.get("is_augmented", False)):
        segment = augment_waveform(
            segment,
            sample_rate=sample_rate,
            augmentation_id=str(row.get("augmentation_id", "aug")),
            seed=int(row.get("augmentation_seed", 42)),
        )
    segment = rms_normalize(segment, target_dbfs=float(rms_dbfs))
    fixed = pad_or_crop_center(segment, int(target_samples))
    return np.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
