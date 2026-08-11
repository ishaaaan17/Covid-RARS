from __future__ import annotations

from pathlib import Path
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from covid_rars.audio_io import load_audio
from covid_rars.preprocess import preprocess_for_features


def sanitize_feature_name(name: object) -> str:
    text = str(name)
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return sanitized or "feature"


def flatten_smile_frame(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("OpenSMILE produced no numeric features")
    means = numeric.mean(axis=0)
    return {
        f"{prefix}_{sanitize_feature_name(column)}": float(value)
        for column, value in means.items()
        if np.isfinite(float(value))
    }


OPENSMILE_FEATURE_SET_ALIASES = {
    "egemaps": ("eGeMAPSv02",),
    "egemapsv02": ("eGeMAPSv02",),
    "egemapsv2": ("eGeMAPSv02",),
    "compare": ("ComParE_2016",),
    "compare2016": ("ComParE_2016",),
    "compare_2016": ("ComParE_2016",),
    "is10": ("IS10",),
    "is10_paraling": ("IS10",),
    "is10_paralinguistic": ("IS10",),
}


def resolve_opensmile_feature_set(opensmile_module, feature_set: str):
    normalized = sanitize_feature_name(feature_set).lower()
    candidates = OPENSMILE_FEATURE_SET_ALIASES.get(normalized)
    if candidates is None:
        supported = ", ".join(sorted(OPENSMILE_FEATURE_SET_ALIASES))
        raise ValueError(f"Unsupported OpenSMILE feature set: {feature_set}. Supported: {supported}")

    for attr in candidates:
        if hasattr(opensmile_module.FeatureSet, attr):
            return getattr(opensmile_module.FeatureSet, attr)
    available = ", ".join(name for name in dir(opensmile_module.FeatureSet) if not name.startswith("_"))
    raise ValueError(
        f"OpenSMILE feature set {feature_set!r} is not available in the installed opensmile package. "
        f"Available feature sets: {available}"
    )


def make_smile_extractor(feature_set: str = "egemaps"):
    try:
        import opensmile
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("opensmile is not installed. Install with: python -m pip install opensmile") from exc

    return opensmile.Smile(
        feature_set=resolve_opensmile_feature_set(opensmile, feature_set),
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_opensmile_features_for_row(
    row: pd.Series,
    *,
    smile,
    representation_name: str = "opensmile_egemaps",
) -> dict[str, object]:
    y, sample_rate, _ = load_audio(Path(row["audio_path"]))
    processed, event_info = preprocess_for_features(y, sample_rate, str(row.get("modality", "unknown")))
    frame = smile.process_signal(processed.astype(np.float32, copy=False), sample_rate)
    features = flatten_smile_frame(frame, prefix=representation_name)

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
    output.update(event_info)
    output.update(features)
    return output


def _filtered_metadata(
    metadata: pd.DataFrame,
    *,
    quality_mode: str = "all_samples",
    modality: str | None = None,
    max_rows: int | None = None,
    skip_recording_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    if quality_mode not in {"all_samples", "quality_ok_only"}:
        raise ValueError(f"Unknown quality_mode: {quality_mode}")

    df = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    if modality is not None:
        df = df[df["modality"] == modality].copy()
    if quality_mode == "quality_ok_only" and "quality_flag" in df.columns:
        df = df[df["quality_flag"] == "ok"].copy()
    if skip_recording_ids:
        skipped = {str(recording_id) for recording_id in skip_recording_ids}
        df = df[~df["recording_id"].astype(str).isin(skipped)].copy()
    if max_rows is not None:
        df = df.head(max(0, int(max_rows))).copy()
    return df


def iter_opensmile_feature_rows(
    metadata: pd.DataFrame,
    *,
    smile=None,
    feature_set: str = "egemaps",
    quality_mode: str = "all_samples",
    modality: str | None = None,
    max_rows: int | None = None,
    representation_name: str | None = None,
    strict: bool = False,
    progress_interval: int = 0,
) -> Iterable[dict[str, object]]:
    representation = representation_name or f"opensmile_{sanitize_feature_name(feature_set).lower()}"
    extractor = smile if smile is not None else make_smile_extractor(feature_set)
    df = _filtered_metadata(
        metadata,
        quality_mode=quality_mode,
        modality=modality,
        max_rows=max_rows,
    )

    total = len(df)
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            yield extract_opensmile_features_for_row(
                row,
                smile=extractor,
                representation_name=representation,
            )
        except Exception as exc:
            if strict:
                raise
            print(f"WARNING: skipping {row.get('audio_path', '?')} — {exc}")
        if progress_interval > 0 and (row_idx % int(progress_interval) == 0 or row_idx == total):
            print(f"Extracted {representation}: {row_idx}/{total} rows", flush=True)


def extract_opensmile_feature_table(
    metadata: pd.DataFrame,
    *,
    smile=None,
    feature_set: str = "egemaps",
    quality_mode: str = "all_samples",
    modality: str | None = None,
    max_rows: int | None = None,
    representation_name: str | None = None,
    strict: bool = False,
    progress_interval: int = 0,
) -> pd.DataFrame:
    rows = list(
        iter_opensmile_feature_rows(
            metadata,
            smile=smile,
            feature_set=feature_set,
            quality_mode=quality_mode,
            modality=modality,
            max_rows=max_rows,
            representation_name=representation_name,
            strict=strict,
            progress_interval=progress_interval,
        )
    )
    return pd.DataFrame(rows)


def _append_feature_chunk(rows: list[dict[str, object]], path: Path, *, columns: list[str] | None) -> list[str]:
    frame = pd.DataFrame(rows)
    if columns is None:
        columns = frame.columns.astype(str).tolist()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    extra_columns = [column for column in frame.columns if column not in columns]
    if extra_columns:
        frame = frame.drop(columns=extra_columns)
    frame = frame[columns]
    frame.to_csv(path, mode="a", index=False, header=not path.exists())
    return columns


def extract_opensmile_feature_csv(
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    smile=None,
    feature_set: str = "egemaps",
    quality_mode: str = "all_samples",
    modality: str | None = None,
    max_rows: int | None = None,
    representation_name: str | None = None,
    strict: bool = False,
    progress_interval: int = 0,
    chunk_size: int = 128,
) -> int:
    if output_path.exists():
        return int(len(pd.read_csv(output_path, usecols=["recording_id"])))

    representation = representation_name or f"opensmile_{sanitize_feature_name(feature_set).lower()}"
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    completed_ids: set[str] = set()
    columns: list[str] | None = None
    if partial_path.exists() and partial_path.stat().st_size > 0:
        try:
            columns = pd.read_csv(partial_path, nrows=0).columns.astype(str).tolist()
            completed_ids = set(pd.read_csv(partial_path, usecols=["recording_id"])["recording_id"].astype(str))
            print(
                f"Resuming {representation} extraction from {partial_path}: "
                f"{len(completed_ids)} rows already written",
                flush=True,
            )
        except Exception:
            partial_path.unlink(missing_ok=True)
            columns = None
            completed_ids = set()

    filtered = _filtered_metadata(
        metadata,
        quality_mode=quality_mode,
        modality=modality,
        max_rows=max_rows,
        skip_recording_ids=completed_ids,
    )
    extractor = smile if smile is not None else make_smile_extractor(feature_set)
    buffer: list[dict[str, object]] = []
    written_now = 0
    total_remaining = len(filtered)
    chunk_rows = max(1, int(chunk_size))

    for row_idx, (_, row) in enumerate(filtered.iterrows(), start=1):
        try:
            buffer.append(
                extract_opensmile_features_for_row(
                    row,
                    smile=extractor,
                    representation_name=representation,
                )
            )
        except Exception as exc:
            if strict:
                raise
            print(f"WARNING: skipping {row.get('audio_path', '?')} — {exc}")
        if len(buffer) >= chunk_rows:
            columns = _append_feature_chunk(buffer, partial_path, columns=columns)
            written_now += len(buffer)
            print(f"Wrote {representation} chunk: {written_now} rows this run", flush=True)
            buffer = []
        if progress_interval > 0 and (row_idx % int(progress_interval) == 0 or row_idx == total_remaining):
            print(f"Extracted {representation}: {row_idx}/{total_remaining} rows", flush=True)

    if buffer:
        columns = _append_feature_chunk(buffer, partial_path, columns=columns)
        written_now += len(buffer)
        print(f"Wrote {representation} chunk: {written_now} rows this run", flush=True)
    if not partial_path.exists():
        pd.DataFrame(columns=columns or ["recording_id"]).to_csv(partial_path, index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(output_path)
    return len(completed_ids) + written_now
