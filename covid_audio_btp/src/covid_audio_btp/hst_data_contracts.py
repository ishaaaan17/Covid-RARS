from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


CLASS_TO_INDEX = {"negative": 0, "positive": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}

_ALLOWED_PRIMARY_LABEL_COLUMNS = {"status", "status_SSL"}
_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm", ".mp4")
_REQUIRED_CONTRACT_METADATA = {
    "dataset_release_id",
    "label_column",
    "label_normalization_version",
    "source_manifest_sha256",
    "eligibility_policy_version",
}
_PREDICTION_KEY_COLUMNS = (
    "run_id",
    "protocol",
    "fold",
    "dataset",
    "participant_key",
    "split",
    "modality",
    "model",
    "checkpoint_hash",
    "representation",
)

_NEGATIVE_ALIASES = {
    "0",
    "control",
    "covid 19 negative",
    "covid negative",
    "healthy",
    "negative",
    "no",
    "no covid",
    "not detected",
    "sars cov 2 negative",
}
_POSITIVE_ALIASES = {
    "1",
    "covid",
    "covid 19",
    "covid 19 positive",
    "covid positive",
    "positive",
    "sars cov 2 positive",
}
_UNKNOWN_ALIASES = {
    "",
    "ambiguous",
    "nan",
    "none",
    "not provided",
    "null",
    "other",
    "possibly covid",
    "respiratory condition",
    "symptomatic",
    "unknown",
    "unreviewed",
}


def _clean_token(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().casefold()
    text = re.sub(r"[_\-/]+", " ", text)
    return " ".join(text.split())


def normalize_coughvid_status(value: object) -> str:
    """Normalize a COUGHVID status using exact, ordered, fail-closed aliases."""
    token = _clean_token(value)
    if token in _NEGATIVE_ALIASES:
        return "negative"
    if token in _POSITIVE_ALIASES:
        return "positive"
    if token in _UNKNOWN_ALIASES:
        return "unknown"
    return "unknown"


def _resolve_named_column(columns: Sequence[object], requested: str) -> str:
    matches = [str(column) for column in columns if str(column).casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Required COUGHVID column {requested!r} was not found exactly once")
    return matches[0]


def _pick_identifier_column(columns: Sequence[object]) -> str:
    for candidate in ("uuid", "participant_id", "recording_id", "id"):
        matches = [str(column) for column in columns if str(column).casefold() == candidate.casefold()]
        if len(matches) == 1:
            return matches[0]
    raise ValueError("COUGHVID metadata must contain one explicit uuid/id column")


def _find_metadata_csv(root: Path) -> Path:
    for name in ("metadata_compiled.csv", "metadata.csv"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    candidates = sorted(path for path in root.rglob("*.csv") if "metadata" in path.name.casefold())
    if not candidates:
        raise FileNotFoundError(f"No COUGHVID metadata CSV found below {root}")
    return candidates[0]


def _zip_metadata_frame(path: Path) -> tuple[pd.DataFrame, set[str]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        preferred = [
            name
            for name in sorted(names)
            if Path(name).name in {"metadata_compiled.csv", "metadata.csv"}
        ]
        candidates = preferred or [
            name
            for name in sorted(names)
            if name.casefold().endswith(".csv") and "metadata" in Path(name).name.casefold()
        ]
        if not candidates:
            raise FileNotFoundError(f"No COUGHVID metadata CSV found inside {path}")
        frame = pd.read_csv(io.BytesIO(archive.read(candidates[0])))
    return frame, names


def _directory_audio_map(root: Path) -> dict[str, Path]:
    audio: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.casefold() in _AUDIO_EXTENSIONS:
            audio.setdefault(path.stem, path)
    return audio


def _zip_audio_map(names: set[str]) -> dict[str, str]:
    audio: dict[str, str] = {}
    for name in sorted(names):
        path = Path(name)
        if path.suffix.casefold() in _AUDIO_EXTENSIONS:
            audio.setdefault(path.stem, name)
    return audio


def _validated_declared_audio_path(value: object, *, base_dir: Path) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none"}:
        return ""
    if "::" in text:
        archive_text, member = text.split("::", 1)
        archive = Path(archive_text)
        if not archive.is_absolute():
            archive = (base_dir / archive).resolve()
        if not archive.is_file() or not member.strip():
            return ""
        try:
            with zipfile.ZipFile(archive) as bundle:
                if member not in bundle.namelist():
                    return ""
        except (OSError, zipfile.BadZipFile):
            return ""
        return f"{archive.as_posix()}::{member}"
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.as_posix() if path.is_file() else ""


def _validate_sha256(value: str, field: str) -> str:
    normalized = str(value).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return normalized


def build_audited_coughvid_index(
    raw_source: Path,
    *,
    label_column: str,
    dataset_release_id: str,
    source_manifest_sha256: str,
    require_audio: bool = True,
) -> pd.DataFrame:
    """Build a COUGHVID index without implicit label-column precedence."""
    raw_source = Path(raw_source)
    if label_column not in _ALLOWED_PRIMARY_LABEL_COLUMNS:
        raise ValueError(
            f"label_column must be explicitly allow-listed: {sorted(_ALLOWED_PRIMARY_LABEL_COLUMNS)}"
        )
    if not str(dataset_release_id).strip():
        raise ValueError("dataset_release_id must be non-empty")
    manifest_hash = _validate_sha256(source_manifest_sha256, "source_manifest_sha256")

    zip_names: set[str] | None = None
    metadata_path: Path | None = None
    if raw_source.is_file() and raw_source.suffix.casefold() == ".zip":
        metadata, zip_names = _zip_metadata_frame(raw_source)
    elif raw_source.is_file() and raw_source.suffix.casefold() == ".csv":
        metadata_path = raw_source
        metadata = pd.read_csv(raw_source)
    elif raw_source.is_dir():
        metadata_path = _find_metadata_csv(raw_source)
        metadata = pd.read_csv(metadata_path)
    else:
        raise FileNotFoundError(f"COUGHVID raw source does not exist: {raw_source}")

    selected_label_column = _resolve_named_column(metadata.columns, label_column)
    identifier_column = _pick_identifier_column(metadata.columns)
    identifier_is_subject = identifier_column.casefold() == "participant_id"
    identifiers = metadata[identifier_column].astype("string").str.strip()
    valid_identifier = identifiers.notna() & ~identifiers.str.casefold().isin({"", "nan", "none"})
    indexed = metadata.loc[valid_identifier].copy()
    identifiers = identifiers.loc[valid_identifier].astype(str)

    declared_audio_column = next(
        (column for column in indexed.columns if str(column).casefold() == "audio_path"),
        None,
    )
    if declared_audio_column is not None and metadata_path is not None:
        audio_paths = indexed[declared_audio_column].map(
            lambda value: _validated_declared_audio_path(
                value,
                base_dir=metadata_path.parent,
            )
        )
    elif zip_names is not None:
        audio_map = _zip_audio_map(zip_names)
        audio_paths = identifiers.map(
            lambda item: f"{raw_source.as_posix()}::{audio_map[item]}" if item in audio_map else ""
        )
    else:
        audio_root = raw_source if raw_source.is_dir() else metadata_path.parent  # type: ignore[union-attr]
        audio_map = _directory_audio_map(audio_root)
        audio_paths = identifiers.map(
            lambda item: audio_map[item].as_posix() if item in audio_map else ""
        )

    indexed["participant_id"] = identifiers.to_numpy()
    indexed["recording_id"] = identifiers.to_numpy()
    indexed["identity_source_column"] = identifier_column
    indexed["analysis_unit_type"] = (
        "participant" if identifier_is_subject else "recording_uuid"
    )
    indexed["participant_id_is_recording_proxy"] = not identifier_is_subject
    indexed["subject_linkage_available"] = identifier_is_subject
    indexed["metadata_source_level"] = (
        "raw_release_archive"
        if zip_names is not None
        else "derived_csv"
        if raw_source.is_file()
        else "raw_release_directory"
    )
    indexed["dataset"] = "coughvid"
    indexed["modality"] = "cough"
    indexed["submodality"] = "cough"
    indexed["audio_path"] = audio_paths.to_numpy()
    indexed["audio_exists"] = indexed["audio_path"].astype(str).str.len().gt(0)
    indexed["label_source"] = selected_label_column
    indexed["label_raw"] = indexed[selected_label_column]
    indexed["label_binary"] = indexed["label_raw"].map(normalize_coughvid_status)
    indexed["dataset_release_id"] = str(dataset_release_id)
    indexed["source_manifest_sha256"] = manifest_hash
    indexed["exclusion_reason"] = ""
    indexed.loc[~indexed["audio_exists"], "exclusion_reason"] = "missing_audio"
    indexed = qualify_identifiers(indexed)
    if require_audio:
        indexed = indexed.loc[indexed["audio_exists"]].copy()
    return indexed.reset_index(drop=True)


def qualify_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach immutable dataset-qualified participant and recording keys."""
    required = {"dataset", "participant_id", "recording_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot qualify identifiers; missing columns: {missing}")
    result = frame.copy()
    cleaned: dict[str, pd.Series] = {}
    for column in sorted(required):
        values = result[column].astype("string").str.strip()
        invalid = values.isna() | values.str.casefold().isin({"", "nan", "none"})
        if invalid.any():
            raise ValueError(f"Identifier column {column!r} contains empty values")
        if values.str.contains("::", regex=False).any():
            raise ValueError(f"Identifier column {column!r} contains reserved delimiter '::'")
        cleaned[column] = values.astype(str)
        result[column] = cleaned[column]
    expected_participants = cleaned["dataset"] + "::" + cleaned["participant_id"]
    expected_recordings = cleaned["dataset"] + "::" + cleaned["recording_id"]
    for column, expected in (
        ("participant_key", expected_participants),
        ("recording_key", expected_recordings),
    ):
        if column in result and not result[column].astype(str).equals(expected):
            raise ValueError(f"Existing {column} values disagree with qualified identifiers")
        result[column] = expected
    return result


def _label_source_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        token = str(column).casefold()
        if token in {"status", "status_ssl"} or "physician" in token or "expert" in token:
            columns.append(str(column))
    return sorted(columns, key=str.casefold)


def audit_coughvid_labels(
    frame: pd.DataFrame,
    *,
    prior: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the selected label and emit value, agreement, and prior-change audits."""
    if "label_source" not in frame:
        raise ValueError("COUGHVID label audit requires an explicit label_source column")
    sources = frame["label_source"].dropna().astype(str).unique().tolist()
    if len(sources) != 1:
        raise ValueError("COUGHVID label audit requires exactly one selected label_source")
    selected = _resolve_named_column(frame.columns, sources[0])
    normalized = frame.copy()
    normalized["label_raw"] = normalized[selected]
    normalized["label_binary"] = normalized[selected].map(normalize_coughvid_status)

    audit_rows: list[dict[str, object]] = []
    source_columns = _label_source_columns(normalized)
    normalized_sources: dict[str, pd.Series] = {}
    for source in source_columns:
        mapped = normalized[source].map(normalize_coughvid_status)
        normalized_sources[source] = mapped
        raw_values = normalized[source].astype("string").fillna("<missing>")
        counts = raw_values.value_counts(dropna=False, sort=False)
        for raw_value, count in sorted(counts.items(), key=lambda item: str(item[0])):
            label = normalize_coughvid_status(None if raw_value == "<missing>" else raw_value)
            audit_rows.append(
                {
                    "audit_type": "value",
                    "label_source": source,
                    "raw_value": raw_value,
                    "normalized_value": label,
                    "row_count": int(count),
                    "supervised_count": int(count) if label in CLASS_TO_INDEX else 0,
                    "exclusion_reason": "" if label in CLASS_TO_INDEX else "unknown_or_ambiguous_label",
                }
            )

    for left_index, left in enumerate(source_columns):
        for right in source_columns[left_index + 1 :]:
            left_values = normalized_sources[left]
            right_values = normalized_sources[right]
            overlap = left_values.isin(CLASS_TO_INDEX) & right_values.isin(CLASS_TO_INDEX)
            overlap_count = int(overlap.sum())
            disagreement = int((left_values.loc[overlap] != right_values.loc[overlap]).sum())
            audit_rows.append(
                {
                    "audit_type": "pairwise",
                    "left_label_source": left,
                    "right_label_source": right,
                    "left_supervised_count": int(left_values.isin(CLASS_TO_INDEX).sum()),
                    "right_supervised_count": int(right_values.isin(CLASS_TO_INDEX).sum()),
                    "overlap_supervised_count": overlap_count,
                    "disagreement_count": disagreement,
                    "disagreement_fraction": disagreement / overlap_count if overlap_count else math.nan,
                }
            )

    if prior is not None:
        if "recording_key" not in normalized or "recording_key" not in prior:
            raise ValueError("Prior-label comparison requires recording_key in both frames")
        if "label_binary" not in prior:
            raise ValueError("Prior-label comparison requires prior label_binary")
        current_labels = normalized[["recording_key", "label_binary"]].rename(
            columns={"label_binary": "current_label"}
        )
        prior_labels = prior[["recording_key", "label_binary"]].rename(
            columns={"label_binary": "prior_label"}
        )
        if current_labels["recording_key"].duplicated().any() or prior_labels["recording_key"].duplicated().any():
            raise ValueError("Prior-label comparison requires unique recording_key values")
        comparison = current_labels.merge(prior_labels, on="recording_key", how="inner", validate="one_to_one")
        both_supervised = comparison["current_label"].isin(CLASS_TO_INDEX) & comparison["prior_label"].isin(
            CLASS_TO_INDEX
        )
        changed = both_supervised & (comparison["current_label"] != comparison["prior_label"])
        audit_rows.append(
            {
                "audit_type": "prior_comparison",
                "aligned_recordings": int(len(comparison)),
                "aligned_supervised_recordings": int(both_supervised.sum()),
                "changed_supervised_labels": int(changed.sum()),
                "invalidates_prior_metrics": bool(changed.any()),
            }
        )
    return normalized, pd.DataFrame(audit_rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path {path} escapes declared root {root}") from exc


def _file_records(paths: Sequence[Path], root: Path, role: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for supplied in paths:
        supplied = Path(supplied)
        relative_supplied = _relative_path(supplied, root)
        if not supplied.exists():
            raise FileNotFoundError(supplied)
        candidates = [supplied] if supplied.is_file() else sorted(path for path in supplied.rglob("*") if path.is_file())
        if not candidates:
            raise ValueError(f"Contract path contains no files: {supplied}")
        for candidate in candidates:
            relative = _relative_path(candidate, root)
            if relative in seen:
                raise ValueError(f"Duplicate contract path: {relative}")
            seen.add(relative)
            records.append(
                {
                    "role": role,
                    "declared_path": relative_supplied,
                    "path": relative,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
    return sorted(records, key=lambda row: (str(row["declared_path"]), str(row["path"])))


def _json_canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TypeError("Data contract metadata must be JSON serializable") from exc


def freeze_data_contracts(
    *,
    source_root: Path,
    audit_root: Path,
    source_paths: tuple[Path, ...],
    label_audits: tuple[Path, ...],
    contract_metadata: Mapping[str, object],
    output_path: Path,
) -> str:
    """Freeze content-addressed source and label-audit provenance atomically."""
    missing = sorted(_REQUIRED_CONTRACT_METADATA - set(contract_metadata))
    if missing:
        raise ValueError(f"Data contract is missing required metadata: {missing}")
    metadata = dict(contract_metadata)
    _validate_sha256(str(metadata["source_manifest_sha256"]), "source_manifest_sha256")
    _json_canonical(metadata)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract_metadata": metadata,
        "sources": _file_records(source_paths, Path(source_root), "source"),
        "label_audits": _file_records(label_audits, Path(audit_root), "label_audit"),
    }
    manifest_hash = hashlib.sha256(_json_canonical(manifest)).hexdigest()
    payload = {**manifest, "manifest_sha256": manifest_hash}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest_hash


def assert_prediction_key_contract(frame: pd.DataFrame, *, repeated: bool) -> None:
    """Reject predictions that can be silently pooled across scientific units."""
    required = set(_PREDICTION_KEY_COLUMNS) | {"label_binary", "probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction key contract missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction frame is empty")
    if repeated and frame["fold"].isna().any():
        raise ValueError("Repeated predictions require a non-null fold")
    for column in _PREDICTION_KEY_COLUMNS:
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"Prediction key column {column!r} contains empty values")
    if not frame["label_binary"].isin(CLASS_TO_INDEX).all():
        raise ValueError("Predictions contain labels outside the frozen class map")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if probability.isna().any() or (~probability.between(0.0, 1.0)).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1]")
    if "participant_id" in frame:
        expected = frame["dataset"].astype(str) + "::" + frame["participant_id"].astype(str)
        if not frame["participant_key"].astype(str).equals(expected):
            raise ValueError("participant_key does not match dataset::participant_id")
    identity = list(_PREDICTION_KEY_COLUMNS)
    if "recording_key" in frame:
        identity.append("recording_key")
    if frame.duplicated(identity).any():
        raise ValueError(f"Duplicate prediction identity for columns: {identity}")


def aggregate_to_participant(frame: pd.DataFrame) -> pd.DataFrame:
    """Average recording probabilities so every participant has equal metric weight."""
    repeated = "fold" in frame and frame["fold"].nunique(dropna=False) > 1
    assert_prediction_key_contract(frame, repeated=repeated)
    group_columns = list(_PREDICTION_KEY_COLUMNS)
    grouped = frame.groupby(group_columns, dropna=False, sort=False)
    label_counts = grouped["label_binary"].nunique(dropna=False)
    if (label_counts != 1).any():
        raise ValueError("A participant has conflicting labels inside a prediction unit")
    participant = grouped.agg(
        label_binary=("label_binary", "first"),
        probability=("probability", "mean"),
        n_recordings=("recording_key" if "recording_key" in frame else "participant_key", "nunique"),
    ).reset_index()
    if "participant_id" in frame:
        participant_ids = grouped["participant_id"].agg(lambda values: values.astype(str).iloc[0]).reset_index(
            name="participant_id"
        )
        participant = participant.merge(participant_ids, on=group_columns, how="left", validate="one_to_one")
    provenance_columns = (
        "analysis_unit_type",
        "identity_source_column",
        "participant_id_is_recording_proxy",
        "subject_linkage_available",
        "metadata_source_level",
    )
    for column in provenance_columns:
        if column not in frame:
            continue
        counts = grouped[column].nunique(dropna=False)
        if counts.ne(1).any():
            raise ValueError(
                f"Prediction aggregation mixes multiple {column} values in one unit"
            )
        values = grouped[column].first().reset_index(name=column)
        participant = participant.merge(
            values,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
    assert_prediction_key_contract(participant, repeated=repeated)
    return participant
