from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    check: str
    message: str


def _issue(severity: str, check: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, check=check, message=message)


def validate_index(index: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = {
        "participant_id",
        "recording_id",
        "dataset",
        "modality",
        "submodality",
        "audio_path",
        "label_binary",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        issues.append(_issue("error", "index_columns", f"Missing required columns: {missing}"))
        return issues

    if index.empty:
        issues.append(_issue("error", "index_rows", "Index has zero rows"))
        return issues

    if index["recording_id"].duplicated().any():
        n = int(index["recording_id"].duplicated().sum())
        issues.append(_issue("error", "recording_id_unique", f"{n} duplicate recording_id values"))

    missing_paths = [p for p in index["audio_path"].dropna().head(1000) if not Path(p).exists()]
    if missing_paths:
        issues.append(
            _issue(
                "error",
                "audio_paths_exist",
                f"At least {len(missing_paths)} sampled audio paths do not exist",
            )
        )

    modality_counts = index["modality"].value_counts(dropna=False).to_dict()
    for modality in ("cough", "breath", "speech"):
        if modality_counts.get(modality, 0) == 0:
            issues.append(_issue("warning", "modality_coverage", f"No rows found for modality={modality}"))

    if modality_counts.get("unknown", 0) > 0:
        issues.append(
            _issue(
                "warning",
                "unknown_modality",
                f"{modality_counts.get('unknown', 0)} rows have unknown modality",
            )
        )

    label_counts = index["label_binary"].value_counts(dropna=False).to_dict()
    if label_counts.get("positive", 0) == 0:
        issues.append(_issue("error", "label_coverage", "No positive labels detected"))
    if label_counts.get("negative", 0) == 0:
        issues.append(_issue("error", "label_coverage", "No negative labels detected"))
    if label_counts.get("unknown", 0) > 0:
        issues.append(
            _issue("warning", "unknown_labels", f"{label_counts.get('unknown', 0)} rows have unknown labels")
        )
    return issues


def validate_metadata(metadata: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = {"participant_id", "recording_id", "modality", "label_binary", "split", "audio_path"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        issues.append(_issue("error", "metadata_columns", f"Missing required columns: {missing}"))
        return issues

    supervised = metadata[metadata["label_binary"].isin(["positive", "negative"])]
    if supervised.empty:
        issues.append(_issue("error", "supervised_rows", "No supervised positive/negative rows available"))
        return issues

    split_counts = supervised["split"].value_counts(dropna=False).to_dict()
    for split in ("train", "validation", "test"):
        if split_counts.get(split, 0) == 0:
            issues.append(_issue("error", "split_coverage", f"No supervised rows in split={split}"))

    leakage = (
        supervised[supervised["split"].isin(["train", "validation", "test"])]
        .groupby("participant_id")["split"]
        .nunique()
    )
    leaked = leakage[leakage > 1]
    if not leaked.empty:
        issues.append(
            _issue(
                "error",
                "participant_leakage",
                f"{len(leaked)} participants appear in multiple train/validation/test splits",
            )
        )

    per_split_label = supervised.groupby(["split", "label_binary"]).size().reset_index(name="n")
    for split in ("train", "validation", "test"):
        split_labels = set(per_split_label[per_split_label["split"] == split]["label_binary"])
        if not {"positive", "negative"}.issubset(split_labels):
            issues.append(
                _issue(
                    "warning",
                    "split_label_balance",
                    f"Split {split} does not contain both positive and negative labels",
                )
            )
    return issues


def validate_quality(quality: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if quality.empty:
        return [_issue("error", "quality_rows", "Quality audit has zero rows")]
    required = {"recording_id", "quality_flag", "duration_sec", "silence_ratio", "clipping_ratio"}
    missing = sorted(required - set(quality.columns))
    if missing:
        return [_issue("error", "quality_columns", f"Missing required columns: {missing}")]

    corrupt_fraction = float((quality["quality_flag"] == "corrupt").mean())
    if corrupt_fraction > 0.20:
        issues.append(
            _issue(
                "warning",
                "corrupt_fraction",
                f"High corrupt fraction: {corrupt_fraction:.1%}. Check audio paths/codecs.",
            )
        )
    ok_fraction = float((quality["quality_flag"] == "ok").mean())
    if ok_fraction < 0.50:
        issues.append(
            _issue(
                "warning",
                "ok_fraction",
                f"Low ok-quality fraction: {ok_fraction:.1%}. Review thresholds and dataset.",
            )
        )
    return issues


def issues_to_frame(issues: list[ValidationIssue]) -> pd.DataFrame:
    return pd.DataFrame([issue.__dict__ for issue in issues], columns=["severity", "check", "message"])


def raise_on_errors(issues: list[ValidationIssue]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        message = "\n".join(f"- {issue.check}: {issue.message}" for issue in errors)
        raise ValueError(f"Validation failed:\n{message}")
