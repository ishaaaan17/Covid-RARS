from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from covid_rars.audio_io import load_audio
from covid_rars.config import AudioConfig
from covid_rars.preprocess import preprocess_for_features


class BatchEmbeddingExtractor(Protocol):
    representation_name: str
    sample_rate: int

    def embed_batch(self, waveforms): ...


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def pool_time_sequence(values) -> np.ndarray:
    array = _as_numpy(values)
    if array.ndim == 3:
        return array.mean(axis=1)
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return array.reshape(1, -1)
    raise ValueError(f"Expected 1D, 2D, or 3D embeddings, got shape {array.shape}")


def _embedding_feature_row(prefix: str, embedding: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_dim_{idx:04d}": float(value)
        for idx, value in enumerate(np.asarray(embedding, dtype=np.float32).ravel())
    }


def _base_output_row(row: pd.Series, representation_name: str) -> dict[str, object]:
    output: dict[str, object] = {
        "recording_id": row["recording_id"],
        "participant_id": row["participant_id"],
        "dataset": row.get("dataset", "coswara"),
        "modality": row.get("modality", "unknown"),
        "submodality": row.get("submodality", "unknown"),
        "label_binary": row.get("label_binary", "unknown"),
        "split": row.get("split", "unused"),
        "representation": representation_name,
    }
    for optional_col in ("quality_flag", "manual_quality_label"):
        if optional_col in row:
            output[optional_col] = row.get(optional_col)
    return output


def _filter_metadata(
    metadata: pd.DataFrame,
    *,
    quality_mode: str,
    modality: str | None,
    max_rows: int | None,
) -> pd.DataFrame:
    if quality_mode not in {"all_samples", "quality_ok_only"}:
        raise ValueError(f"Unknown quality_mode: {quality_mode}")
    df = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    if modality is not None:
        df = df[df["modality"] == modality].copy()
    if quality_mode == "quality_ok_only" and "quality_flag" in df.columns:
        df = df[df["quality_flag"] == "ok"].copy()
    if max_rows is not None:
        df = df.head(max(0, int(max_rows))).copy()
    return df


def extract_torch_embedding_feature_table(
    metadata: pd.DataFrame,
    *,
    extractor: BatchEmbeddingExtractor,
    quality_mode: str = "all_samples",
    modality: str | None = None,
    max_rows: int | None = None,
    batch_size: int = 8,
    strict: bool = False,
) -> pd.DataFrame:
    df = _filter_metadata(metadata, quality_mode=quality_mode, modality=modality, max_rows=max_rows)
    config = AudioConfig(sample_rate=int(extractor.sample_rate))
    batch_size = max(1, int(batch_size))

    output_rows: list[dict[str, object]] = []
    batch_waveforms: list[np.ndarray] = []
    batch_base_rows: list[dict[str, object]] = []
    batch_event_info: list[dict[str, object]] = []

    def flush_batch() -> None:
        if not batch_waveforms:
            return
        waveforms = np.stack(batch_waveforms).astype(np.float32, copy=False)
        embeddings = pool_time_sequence(extractor.embed_batch(waveforms))
        if embeddings.ndim != 2 or embeddings.shape[0] != len(batch_waveforms):
            raise ValueError(
                f"Extractor returned shape {embeddings.shape} for batch size {len(batch_waveforms)}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Extractor produced non-finite embeddings")
        for base_row, event_info, embedding in zip(batch_base_rows, batch_event_info, embeddings):
            row = dict(base_row)
            row.update(event_info)
            row.update(_embedding_feature_row(extractor.representation_name, embedding))
            output_rows.append(row)
        batch_waveforms.clear()
        batch_base_rows.clear()
        batch_event_info.clear()

    for _, row in df.iterrows():
        try:
            y, sample_rate, _ = load_audio(Path(row["audio_path"]), config=config)
            processed, event_info = preprocess_for_features(
                y,
                sample_rate,
                str(row.get("modality", "unknown")),
                config=config,
            )
            batch_waveforms.append(processed.astype(np.float32, copy=False))
            batch_base_rows.append(_base_output_row(row, extractor.representation_name))
            batch_event_info.append(event_info)
            if len(batch_waveforms) >= batch_size:
                flush_batch()
        except Exception as exc:
            if strict:
                raise
            print(f"WARNING: skipping {row.get('audio_path', '?')} — {exc}")
    flush_batch()
    return pd.DataFrame(output_rows)
