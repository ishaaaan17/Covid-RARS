from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from covid_rars.audio_io import load_audio
from covid_rars.preprocess import preprocess_for_features


def _safe_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else np.asarray([0.0], dtype=float)


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = _safe_values(values)
    q25, q75 = np.percentile(arr, [25, 75])
    std = float(np.std(arr))
    if std <= 1e-12:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        skewness = float(skew(arr, bias=False, nan_policy="omit"))
        excess_kurtosis = float(kurtosis(arr, fisher=True, bias=False, nan_policy="omit"))
    out = {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_skew": skewness,
        f"{prefix}_kurtosis": excess_kurtosis,
    }
    return {key: (value if np.isfinite(value) else 0.0) for key, value in out.items()}


def _matrix_row_stats(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    for idx in range(arr.shape[0]):
        features.update(_stats(f"{prefix}_{idx + 1:02d}", arr[idx]))
    return features


def extract_extended_acoustic_features(
    y: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 40,
    n_mels: int = 64,
) -> dict[str, float]:
    """Extract a paper-style acoustic feature bank.

    The bank intentionally combines families used across COVID-audio papers:
    MFCC/delta trajectories, mel-band summaries, chroma, spectral contrast,
    tonnetz, and low-level spectral/energy descriptors.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    if not np.any(np.isfinite(y)):
        y = np.zeros_like(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    features: dict[str, float] = {
        "duration_sec": float(y.size / max(1, sample_rate)),
        "waveform_abs_mean": float(np.mean(np.abs(y))),
        "waveform_abs_max": float(np.max(np.abs(y))) if y.size else 0.0,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    features.update(_matrix_row_stats("mfcc", mfcc))
    features.update(_matrix_row_stats("delta_mfcc", librosa.feature.delta(mfcc)))
    features.update(_matrix_row_stats("delta2_mfcc", librosa.feature.delta(mfcc, order=2)))

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    features.update(_matrix_row_stats("mel_band", log_mel))
    features.update(_stats("mel_global", log_mel))

    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    features.update(_matrix_row_stats("chroma", chroma))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate)
    features.update(_matrix_row_stats("spectral_contrast", contrast))

    try:
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
        features.update(_matrix_row_stats("tonnetz", tonnetz))
    except Exception:
        for idx in range(6):
            features.update(_stats(f"tonnetz_{idx + 1:02d}", np.asarray([0.0])))

    low_level = {
        "rms": librosa.feature.rms(y=y)[0],
        "zcr": librosa.feature.zero_crossing_rate(y=y)[0],
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0],
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sample_rate)[0],
        "spectral_flatness": librosa.feature.spectral_flatness(y=y)[0],
    }
    for name, values in low_level.items():
        features.update(_stats(name, values))

    tempo_values = librosa.feature.tempogram(y=y, sr=sample_rate)
    features.update(_stats("tempogram_global", tempo_values))
    return {key: float(value) if np.isfinite(value) else 0.0 for key, value in features.items()}


def _stable_seed(*parts: object, base_seed: int = 42) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(base_seed)) % (2**32 - 1)


def augment_waveform(
    y: np.ndarray,
    sample_rate: int,
    augmentation_id: str,
    seed: int,
) -> np.ndarray:
    """Apply lightweight deterministic audio augmentation for training rows.

    The transformations intentionally avoid changing validation/test data.
    They are conservative enough for cough, breath, and speech: small time
    stretch, small pitch shift, and low-amplitude additive noise.
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.size == 0:
        return arr
    rng = np.random.default_rng(int(seed))
    kind = str(augmentation_id)
    out = arr.copy()
    if kind.endswith("1") or kind.endswith("4"):
        rate = float(rng.uniform(0.92, 1.08))
        try:
            out = librosa.effects.time_stretch(out, rate=rate).astype(np.float32, copy=False)
        except Exception:
            out = arr.copy()
    elif kind.endswith("2") or kind.endswith("5"):
        steps = float(rng.uniform(-1.5, 1.5))
        try:
            out = librosa.effects.pitch_shift(out, sr=sample_rate, n_steps=steps).astype(np.float32, copy=False)
        except Exception:
            out = arr.copy()
    else:
        noise_scale = float(rng.uniform(0.001, 0.012))
        out = (out + rng.normal(0.0, noise_scale, size=out.shape)).astype(np.float32, copy=False)

    if out.size == 0:
        out = arr.copy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def extract_strong_features_for_row(row: pd.Series) -> dict[str, object]:
    y, sample_rate, _ = load_audio(Path(row["audio_path"]))
    processed, event_info = preprocess_for_features(y, sample_rate, str(row.get("modality", "unknown")))
    is_augmented = bool(row.get("is_augmented", False))
    augmentation_id = str(row.get("augmentation_id", "original"))
    if is_augmented:
        seed = int(row.get("augmentation_seed", _stable_seed(row.get("recording_id", ""), augmentation_id)))
        processed = augment_waveform(processed, sample_rate, augmentation_id=augmentation_id, seed=seed)
    features = extract_extended_acoustic_features(processed, sample_rate=sample_rate)
    return {
        "recording_id": row["recording_id"],
        "source_recording_id": row.get("source_recording_id", row["recording_id"]),
        "participant_id": row["participant_id"],
        "dataset": row.get("dataset", "coswara"),
        "modality": row.get("modality", "unknown"),
        "submodality": row.get("submodality", "unknown"),
        "label_binary": row.get("label_binary", "unknown"),
        "split": row.get("split", "unused"),
        "is_augmented": is_augmented,
        "augmentation_id": augmentation_id,
        **event_info,
        **features,
    }


def _expand_training_augmentations(
    metadata: pd.DataFrame,
    augment_train_copies: int,
    augmentation_seed: int,
) -> pd.DataFrame:
    if augment_train_copies <= 0 or metadata.empty:
        out = metadata.copy()
        out["source_recording_id"] = out["recording_id"].astype(str)
        out["is_augmented"] = False
        out["augmentation_id"] = "original"
        out["augmentation_seed"] = int(augmentation_seed)
        return out

    rows: list[pd.Series] = []
    for _, source_row in metadata.iterrows():
        original = source_row.copy()
        original["source_recording_id"] = str(source_row["recording_id"])
        original["is_augmented"] = False
        original["augmentation_id"] = "original"
        original["augmentation_seed"] = int(augmentation_seed)
        rows.append(original)
        if str(source_row.get("split", "")).lower() != "train":
            continue
        for copy_idx in range(1, int(augment_train_copies) + 1):
            augmented = source_row.copy()
            source_id = str(source_row["recording_id"])
            augmented["source_recording_id"] = source_id
            augmented["recording_id"] = f"{source_id}::aug{copy_idx}"
            augmented["is_augmented"] = True
            augmented["augmentation_id"] = f"aug{copy_idx}"
            augmented["augmentation_seed"] = _stable_seed(source_id, copy_idx, base_seed=augmentation_seed)
            rows.append(augmented)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_strong_feature_table(
    metadata: pd.DataFrame,
    quality_mode: str = "all_samples",
    progress_interval: int = 250,
    augment_train_copies: int = 0,
    augmentation_seed: int = 42,
) -> pd.DataFrame:
    df = metadata.copy()
    df = df[df["label_binary"].isin(["positive", "negative"])]
    if quality_mode == "quality_ok_only" and "quality_flag" in df.columns:
        df = df[df["quality_flag"].astype(str).str.lower().eq("ok")]
    elif quality_mode not in {"all_samples", "quality_ok_only"}:
        raise ValueError(f"Unknown quality_mode: {quality_mode}")

    df = _expand_training_augmentations(
        df,
        augment_train_copies=int(augment_train_copies),
        augmentation_seed=int(augmentation_seed),
    )
    rows: list[dict[str, object]] = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            rows.append(extract_strong_features_for_row(row))
        except Exception as exc:
            print(f"WARNING: skipping {row.get('audio_path', '?')} — {exc}")
        if progress_interval > 0 and (idx % progress_interval == 0 or idx == total):
            print(f"Strong feature extraction progress: {idx}/{total} rows", flush=True)
    return pd.DataFrame(rows)
