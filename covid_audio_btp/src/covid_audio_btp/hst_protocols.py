from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


PRESPECIFIED_HST_REPO_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)
TRACK_A_PROTOCOL = "hst_literature_aligned_repeated_holdout"
SPLIT_POLICY_MIXED_PROTOCOL = "hst_calendar_mixed_split_policy"
SPLIT_POLICY_CHRONOLOGICAL_PROTOCOL = "hst_chronological_split_policy"
COMMON_LATE_BALANCED_PROTOCOL = "hst_common_late_test_date_balanced_source"
COMMON_LATE_CHRONOLOGICAL_PROTOCOL = "hst_common_late_test_chronological_source"
REVERSE_TEMPORAL_PROTOCOL = "hst_reverse_temporal_sensitivity"
EXTERNAL_PROTOCOL = "coswara_to_coughvid_hst_external"

_CLASS_TO_INDEX = {"negative": 0, "positive": 1}
_SPLITS = ("train", "validation", "test")
_HASH_COLUMNS = {"row_content_sha256", "manifest_sha256"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTENT_HASH_COLUMNS = (
    "tensor_sha256",
    "source_sha256",
    "source_audio_sha256",
    "audio_sha256",
)
_CACHE_COLUMNS = {
    "dataset",
    "participant_key",
    "recording_key",
    "modality",
    "label_binary",
    "eligible",
    "tensor_sha256",
    "source_audio_sha256",
    "preprocessing_hash",
    "representation_id",
}
_EXTERNAL_PROVENANCE_COLUMNS = {
    "label_source",
    "label_provenance",
    "dataset_release_id",
    "source_manifest_sha256",
    "preprocessing_variant",
}
_DATE_COLUMNS = (
    "recording_timestamp_utc",
    "recording_timestamp",
    "recording_date",
    "date",
)
_NON_SCIENTIFIC_CONFIG_KEYS = {
    "manifest_path",
    "manifest_paths",
    "protocol_label",
    "protocol_name",
    "run_id",
    "output_dir",
    "output_path",
    "log_dir",
    "resume_path",
}
_ANALYSIS_PROVENANCE_COLUMNS = {
    "analysis_scope",
    "analysis_role",
    "estimand_id",
    "multiplicity_family",
    "analysis_mode",
    "confirmatory_protocol",
}
_SHARED_INTERSECTION_POLICY = "shared_recording_modality_intersection_v1"
_EXPLICIT_RESTRICTION_POLICY = "explicit_frozen_restriction_v1"


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _canonicalize(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(item) for item in value), key=str)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _canonicalize(vars(value))
    raise TypeError(f"Cannot serialize {type(value).__name__} deterministically")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _content_digest(value: object) -> str:
    return sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _stamp_analysis_provenance(
    frame: pd.DataFrame,
    *,
    analysis_scope: str,
    analysis_role: str,
    estimand_id: str,
    multiplicity_family: str,
    analysis_mode: str,
    confirmatory_protocol: bool,
) -> pd.DataFrame:
    values = {
        "analysis_scope": analysis_scope,
        "analysis_role": analysis_role,
        "estimand_id": estimand_id,
        "multiplicity_family": multiplicity_family,
        "analysis_mode": analysis_mode,
    }
    if any(not str(value).strip() for value in values.values()):
        raise ValueError("Analysis provenance values cannot be empty")
    result = frame.copy()
    for column, value in values.items():
        result[column] = value
    result["confirmatory_protocol"] = bool(confirmatory_protocol)
    return result


def _without_hash_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in _HASH_COLUMNS if column in frame])


def _sort_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    split_values = result["split"] if "split" in result else pd.Series("", index=result.index)
    result["_split_order"] = split_values.map(
        {"train": 0, "validation": 1, "test": 2, "external_test": 3}
    ).fillna(9)
    candidates = (
        "protocol",
        "cohort",
        "fold",
        "_split_order",
        "dataset",
        "participant_key",
        "modality",
        "recording_key",
        "representation_id",
        "tensor_sha256",
        "preprocessing_hash",
        "scientific_configuration_fingerprint",
        "event_key",
        "label_source",
        "preprocessing_variant",
    )
    sort_columns = [column for column in candidates if column in result]
    content_columns = sorted(
        column
        for column in result.columns
        if column not in _HASH_COLUMNS and column != "_split_order"
    )
    result["_canonical_row_order"] = result[content_columns].apply(
        lambda row: _content_digest(row.to_dict()), axis=1
    )
    sort_columns.append("_canonical_row_order")
    if sort_columns:
        result = result.sort_values(sort_columns, kind="mergesort", na_position="last")
    return result.drop(columns=["_split_order", "_canonical_row_order"]).reset_index(
        drop=True
    )


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    content = _without_hash_columns(frame)
    return content.apply(lambda row: _content_digest(row.to_dict()), axis=1)


def _manifest_digest(frame: pd.DataFrame) -> str:
    content = _sort_manifest(_without_hash_columns(frame))
    return _content_digest(content.to_dict(orient="records"))


def _finalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Cannot freeze an empty HST manifest")
    _require_columns(
        frame,
        {
            "source_audio_sha256",
            "scientific_configuration_fingerprint",
            "eligibility_alignment_fingerprint",
        }
        | _ANALYSIS_PROVENANCE_COLUMNS,
        "manifest",
    )
    if not frame["source_audio_sha256"].astype(str).str.casefold().map(
        lambda value: bool(_SHA256_PATTERN.fullmatch(value))
    ).all():
        raise ValueError("Manifest contains invalid source_audio_sha256 values")
    for column in (
        "scientific_configuration_fingerprint",
        "eligibility_alignment_fingerprint",
    ):
        if not frame[column].astype(str).map(
            lambda value: bool(_SHA256_PATTERN.fullmatch(value))
        ).all():
            raise ValueError(f"Manifest contains invalid {column} values")
    for column in sorted(_ANALYSIS_PROVENANCE_COLUMNS - {"confirmatory_protocol"}):
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any() or values.nunique() != 1:
            raise ValueError(f"Manifest has inconsistent {column} provenance")
    confirmatory_values = set(frame["confirmatory_protocol"].astype(bool))
    if len(confirmatory_values) != 1:
        raise ValueError("Manifest has inconsistent confirmatory_protocol provenance")
    attrs = dict(frame.attrs)
    result = _sort_manifest(_without_hash_columns(frame))
    preferred = [
        column
        for column in (
            "protocol",
            "cohort",
            "fold",
            "split",
            "dataset",
            "participant_key",
            "recording_key",
            "modality",
            "representation_id",
            "label_binary",
            "tensor_sha256",
            "preprocessing_hash",
            "scientific_configuration_fingerprint",
            "eligibility_alignment_fingerprint",
        )
        if column in result
    ]
    remaining = sorted(column for column in result if column not in preferred)
    result = result[preferred + remaining]
    result["row_content_sha256"] = _row_hashes(result)
    if result["row_content_sha256"].duplicated().any():
        raise ValueError("Manifest contains duplicate content rows")
    result["manifest_sha256"] = _manifest_digest(result)
    result.attrs.update(attrs)
    return result


def _resolve_scientific_fingerprint(
    scientific_config: object | None,
    scientific_fingerprint: str | None,
) -> str:
    if scientific_config is None:
        raise ValueError(
            "A scientific configuration is required for confirmatory manifests"
        )
    calculated = scientific_configuration_fingerprint(scientific_config)
    supplied = None
    if scientific_fingerprint is not None:
        supplied = str(scientific_fingerprint).strip().casefold()
        if not _SHA256_PATTERN.fullmatch(supplied):
            raise ValueError("scientific_fingerprint must be a canonical SHA-256 digest")
    if supplied is not None and calculated != supplied:
        raise ValueError("scientific configuration and fingerprint disagree")
    return calculated


def _frame_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    return _sort_manifest(_without_hash_columns(frame)).to_dict(orient="records")


def _embed_audit_payload(
    frame: pd.DataFrame,
    prefix: str,
    payload: object,
) -> pd.DataFrame:
    """Embed one canonical payload and reference its digest from every row."""
    result = _sort_manifest(frame)
    payload_json = _canonical_json(payload)
    digest = sha256(payload_json.encode("ascii")).hexdigest()
    result[f"{prefix}_sha256"] = digest
    result[f"{prefix}_payload_json"] = ""
    result.loc[0, f"{prefix}_payload_json"] = payload_json
    return result


def _extract_audit_payload(frame: pd.DataFrame, prefix: str) -> object:
    hash_column = f"{prefix}_sha256"
    payload_column = f"{prefix}_payload_json"
    _require_columns(frame, {hash_column, payload_column}, "audit payload")
    hashes = set(frame[hash_column].astype(str))
    payloads = frame.loc[frame[payload_column].astype(str).ne(""), payload_column]
    if len(hashes) != 1 or len(payloads) != 1:
        raise ValueError(f"{prefix} must have one canonical payload and one digest")
    try:
        payload = json.loads(str(payloads.iloc[0]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{prefix} payload is not canonical JSON") from exc
    if _content_digest(payload) != next(iter(hashes)):
        raise ValueError(f"{prefix} payload digest does not verify")
    return payload


def _verify_frozen_frame(frame: pd.DataFrame, name: str) -> None:
    _require_columns(frame, _HASH_COLUMNS, name)
    stored = set(frame["manifest_sha256"].astype(str))
    calculated = _manifest_digest(frame)
    if stored != {calculated}:
        raise ValueError(f"{name} manifest hash does not verify")
    if not frame["row_content_sha256"].astype(str).eq(_row_hashes(frame)).all():
        raise ValueError(f"{name} row hashes do not verify")


def _alignment_fingerprint(frame: pd.DataFrame) -> str:
    required = [
        "recording_key",
        "modality",
        "dataset",
        "participant_key",
        "label_binary",
        "representation_id",
        "tensor_sha256",
        "preprocessing_hash",
        "scientific_configuration_fingerprint",
    ]
    columns = required + [
        column
        for column in _CONTENT_HASH_COLUMNS
        if column in frame and column not in required
    ]
    canonical = frame[columns].drop_duplicates().copy()
    canonical = _sort_manifest(canonical)
    return _content_digest(canonical.to_dict(orient="records"))


def _analysis_unit_key(recording_key: object, modality: object) -> str:
    return _content_digest(
        {"recording_key": recording_key, "modality": modality}
    )


def _assert_representation_source_audio_consistency(
    frame: pd.DataFrame,
    *,
    name: str,
) -> None:
    _require_columns(
        frame,
        {"recording_key", "modality", "representation_id", "source_audio_sha256"},
        name,
    )
    counts = frame.groupby(["recording_key", "modality"], dropna=False)[
        "source_audio_sha256"
    ].nunique(dropna=False)
    if counts.gt(1).any():
        raise ValueError(
            f"{name} representations disagree on source_audio_sha256 for a "
            "recording/modality"
        )


def _analysis_units_by_representation(
    frame: pd.DataFrame,
) -> dict[str, set[str]]:
    _require_columns(
        frame,
        {"recording_key", "modality", "representation_id"},
        "eligibility policy frame",
    )
    current = frame[["recording_key", "modality", "representation_id"]].copy()
    current["analysis_unit_key"] = current.apply(
        lambda row: _analysis_unit_key(row["recording_key"], row["modality"]),
        axis=1,
    )
    return {
        str(representation_id): set(unit["analysis_unit_key"].astype(str))
        for representation_id, unit in current.groupby("representation_id", sort=True)
    }


def _validate_declared_mapping_policy(
    mapping: pd.DataFrame,
    audit_payload: object,
    *,
    cache: pd.DataFrame | None = None,
) -> None:
    _require_columns(mapping, {"eligibility_mapping_policy"}, "eligibility_mapping")
    policy_values = set(mapping["eligibility_mapping_policy"].astype(str))
    if len(policy_values) != 1:
        raise ValueError("eligibility mapping has ambiguous mapping policy")
    policy_id = next(iter(policy_values))
    if policy_id not in {_SHARED_INTERSECTION_POLICY, _EXPLICIT_RESTRICTION_POLICY}:
        raise ValueError("eligibility mapping policy is not a frozen supported policy")
    if not isinstance(audit_payload, dict) or audit_payload.get(
        "mapping_policy_id"
    ) != policy_id:
        raise ValueError("eligibility mapping policy does not match its frozen audit")
    dispositions = audit_payload.get("representation_unit_dispositions")
    if not isinstance(dispositions, dict) or not dispositions:
        raise ValueError("eligibility mapping policy lacks unit dispositions")

    mapped_by_representation = _analysis_units_by_representation(mapping)
    for representation_id, mapped_units in mapped_by_representation.items():
        disposition = dispositions.get(representation_id)
        if not isinstance(disposition, dict):
            raise ValueError("eligibility mapping policy omits a representation")
        input_units = set(disposition.get("input_eligible_analysis_unit_keys", []))
        retained_units = set(disposition.get("retained_analysis_unit_keys", []))
        excluded_units = set(disposition.get("excluded_analysis_unit_keys", []))
        if retained_units & excluded_units or retained_units | excluded_units != input_units:
            raise ValueError("eligibility mapping policy has invalid unit dispositions")
        if mapped_units != retained_units:
            raise ValueError("eligibility mapping rows disagree with retained policy units")

    if cache is None:
        return
    observed_by_representation = _analysis_units_by_representation(cache)
    for representation_id, observed_units in observed_by_representation.items():
        disposition = dispositions.get(representation_id)
        if not isinstance(disposition, dict):
            raise ValueError(
                "eligibility mapping omits eligible cache analysis units for an "
                "undeclared representation"
            )
        declared_input = set(
            disposition.get("input_eligible_analysis_unit_keys", [])
        )
        if observed_units != declared_input:
            raise ValueError(
                "eligibility mapping omits eligible cache analysis units or declares "
                "units absent from the supplied cache"
            )


def _apply_eligibility_alignment(
    cache_index: pd.DataFrame,
    *,
    scientific_fingerprint: str,
    eligibility_mapping: pd.DataFrame | None,
    eligibility_fingerprint: str | None,
) -> tuple[pd.DataFrame, str, object]:
    if eligibility_mapping is None:
        raise ValueError("A frozen representation eligibility mapping is required")
    mapping = eligibility_mapping.copy()
    _verify_frozen_frame(mapping, "eligibility_mapping")
    _assert_representation_source_audio_consistency(
        mapping, name="eligibility_mapping"
    )
    _require_columns(
        mapping,
        {
            "recording_key",
            "modality",
            "representation_id",
            "tensor_sha256",
            "preprocessing_hash",
            "scientific_configuration_fingerprint",
            "eligibility_alignment_fingerprint",
            "analysis_unit_key",
        },
        "eligibility_mapping",
    )
    mapping_science = set(mapping["scientific_configuration_fingerprint"].astype(str))
    if mapping_science != {scientific_fingerprint}:
        raise ValueError("eligibility mapping scientific fingerprint does not match")
    calculated_alignment = _alignment_fingerprint(mapping)
    stored_alignments = set(mapping["eligibility_alignment_fingerprint"].astype(str))
    if stored_alignments != {calculated_alignment}:
        raise ValueError("eligibility alignment fingerprint does not verify")
    if eligibility_fingerprint is not None:
        supplied = str(eligibility_fingerprint).strip().casefold()
        if not _SHA256_PATTERN.fullmatch(supplied) or supplied != calculated_alignment:
            raise ValueError("supplied eligibility fingerprint does not verify")
    audit_payload = _extract_audit_payload(mapping, "eligibility_audit")

    cache = _eligible_cache(cache_index)
    _validate_declared_mapping_policy(mapping, audit_payload, cache=cache)
    base_identity = ["recording_key", "modality", "representation_id"]
    exact_identity = base_identity + [
        "dataset",
        "participant_key",
        "label_binary",
        "tensor_sha256",
        "preprocessing_hash",
    ]
    for column in _CONTENT_HASH_COLUMNS:
        if column in cache and column in mapping and column not in exact_identity:
            exact_identity.append(column)

    mapping_base = set(map(tuple, mapping[base_identity].astype(str).to_numpy()))
    cache_base = cache[base_identity].astype(str).apply(tuple, axis=1)
    candidates = cache.loc[cache_base.isin(mapping_base)].copy()
    if candidates.empty:
        raise ValueError("cache has no rows in the frozen eligibility intersection")
    mapping_exact = set(map(tuple, mapping[exact_identity].astype(str).to_numpy()))
    candidate_exact = candidates[exact_identity].astype(str).apply(tuple, axis=1)
    if not candidate_exact.isin(mapping_exact).all():
        raise ValueError("cache content disagrees with the frozen eligibility mapping")
    supplied_representations = set(cache["representation_id"].astype(str))
    expected_rows = mapping[
        mapping["representation_id"].astype(str).isin(supplied_representations)
    ]
    expected_exact = set(
        map(tuple, expected_rows[exact_identity].astype(str).to_numpy())
    )
    if set(candidate_exact) != expected_exact:
        raise ValueError("cache is incomplete for the frozen eligibility intersection")

    derived_columns = {
        "analysis_unit_key",
        "analysis_unit_weight",
        "paired_representation",
        "paired_representation_count",
        "scientific_configuration_fingerprint",
        "eligibility_alignment_fingerprint",
        "eligibility_audit_sha256",
        "eligibility_audit_payload_json",
    }
    candidates = candidates.drop(
        columns=[column for column in derived_columns if column in candidates]
    )
    metadata_columns = exact_identity + ["analysis_unit_key"]
    aligned = candidates.merge(
        mapping[metadata_columns].drop_duplicates(),
        on=exact_identity,
        how="inner",
        validate="one_to_one",
    )
    counts = aligned.groupby("analysis_unit_key")["representation_id"].transform(
        "nunique"
    )
    aligned["paired_representation_count"] = counts.astype(int)
    aligned["paired_representation"] = counts.gt(1)
    aligned["analysis_unit_weight"] = 1.0 / counts.astype(float)
    aligned["scientific_configuration_fingerprint"] = scientific_fingerprint
    aligned["eligibility_alignment_fingerprint"] = calculated_alignment
    return aligned.reset_index(drop=True), calculated_alignment, audit_payload


def _restrict_eligibility_mapping(
    mapping: pd.DataFrame,
    selected_cache: pd.DataFrame,
    *,
    restriction_reason: str,
    selection_label: str = "selected analysis",
) -> pd.DataFrame:
    _verify_frozen_frame(mapping, "parent eligibility_mapping")
    parent_audit = _extract_audit_payload(mapping, "eligibility_audit")
    _validate_declared_mapping_policy(mapping, parent_audit)
    calculated_parent = _alignment_fingerprint(mapping)
    selected_units = sorted(set(selected_cache["analysis_unit_key"].astype(str)))
    if not selected_units:
        raise ValueError("Cannot freeze an empty eligibility restriction")
    mapping_units = set(mapping["analysis_unit_key"].astype(str))
    missing_units = sorted(set(selected_units) - mapping_units)
    if missing_units:
        raise ValueError(
            f"Every {selection_label} unit must survive the eligibility restriction; "
            f"{len(missing_units)} unit(s) are absent"
        )
    parent_fingerprints = set(
        mapping["eligibility_alignment_fingerprint"].astype(str)
    )
    if len(parent_fingerprints) != 1:
        raise ValueError("Parent eligibility mapping has ambiguous provenance")
    parent_fingerprint = next(iter(parent_fingerprints))
    if parent_fingerprint != calculated_parent:
        raise ValueError("Parent eligibility alignment fingerprint does not verify")
    result = _without_hash_columns(mapping).copy()
    result = result[result["analysis_unit_key"].astype(str).isin(selected_units)].copy()
    if result.empty:
        raise ValueError("Eligibility restriction selected no mapping rows")
    realized_units = sorted(set(result["analysis_unit_key"].astype(str)))
    if realized_units != selected_units:
        raise ValueError(
            f"Every {selection_label} unit must survive the eligibility restriction"
        )
    old_audit_columns = [
        column
        for column in result
        if column.startswith("eligibility_audit_")
        or column
        in {"eligibility_alignment_fingerprint", "eligibility_mapping_policy"}
    ]
    result = result.drop(columns=old_audit_columns)
    result["parent_eligibility_alignment_fingerprint"] = parent_fingerprint
    result["eligibility_mapping_policy"] = _EXPLICIT_RESTRICTION_POLICY
    restricted_fingerprint = _alignment_fingerprint(result)
    result["eligibility_alignment_fingerprint"] = restricted_fingerprint
    payload = {
        "mapping_policy_id": _EXPLICIT_RESTRICTION_POLICY,
        "alignment_fingerprint": restricted_fingerprint,
        "parent_alignment_fingerprint": parent_fingerprint,
        "restriction_reason": restriction_reason,
        "selected_analysis_unit_keys": selected_units,
        "selected_analysis_unit_count": len(realized_units),
        "requested_analysis_unit_count": len(selected_units),
        "realized_analysis_unit_count": len(realized_units),
        "realized_mapping_row_count": len(result),
        "representation_ids": sorted(result["representation_id"].astype(str).unique()),
        "representation_unit_dispositions": {
            representation_id: {
                "input_eligible_analysis_unit_keys": selected_units,
                "retained_analysis_unit_keys": selected_units,
                "excluded_analysis_unit_keys": [],
            }
            for representation_id in sorted(
                result["representation_id"].astype(str).unique()
            )
        },
    }
    result = _stamp_analysis_provenance(
        result,
        analysis_scope="representation_alignment",
        analysis_role="design_context",
        estimand_id="restricted_representation_eligibility",
        multiplicity_family="not_applicable",
        analysis_mode="design",
        confirmatory_protocol=False,
    )
    result = _embed_audit_payload(result, "eligibility_audit", payload)
    return _finalize_manifest(result)


def _stamp_builder_provenance(
    frame: pd.DataFrame,
    *,
    scientific_fingerprint: str,
    eligibility_fingerprint: str,
    eligibility_audit: object,
) -> pd.DataFrame:
    result = frame.copy()
    result["scientific_configuration_fingerprint"] = scientific_fingerprint
    result["eligibility_alignment_fingerprint"] = eligibility_fingerprint
    return _embed_audit_payload(result, "eligibility_audit", eligibility_audit)


def _validate_qualified_keys(frame: pd.DataFrame) -> None:
    dataset = frame["dataset"].astype("string").str.strip()
    participant_key = frame["participant_key"].astype("string").str.strip()
    recording_key = frame["recording_key"].astype("string").str.strip()
    for column, values in (
        ("dataset", dataset),
        ("participant_key", participant_key),
        ("recording_key", recording_key),
    ):
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"{column} contains empty values")
    participant_parts = participant_key.astype(str).str.partition("::")
    recording_parts = recording_key.astype(str).str.partition("::")
    if not (
        participant_parts[1].eq("::")
        & participant_parts[0].eq(dataset.astype(str))
        & participant_parts[2].ne("")
    ).all():
        raise ValueError("Every row must use a dataset-qualified participant_key")
    if not (
        recording_parts[1].eq("::")
        & recording_parts[0].eq(dataset.astype(str))
        & recording_parts[2].ne("")
    ).all():
        raise ValueError("Every row must use a dataset-qualified recording_key")
    if "participant_id" in frame:
        expected = dataset.astype(str) + "::" + frame["participant_id"].astype(str)
        if not participant_key.astype(str).equals(expected):
            raise ValueError("qualified participant_key disagrees with participant_id")
    if "recording_id" in frame:
        expected = dataset.astype(str) + "::" + frame["recording_id"].astype(str)
        if not recording_key.astype(str).equals(expected):
            raise ValueError("qualified recording_key disagrees with recording_id")


def _eligible_cache(cache_index: pd.DataFrame) -> pd.DataFrame:
    _require_columns(cache_index, _CACHE_COLUMNS, "cache_index")
    _validate_qualified_keys(cache_index)
    boolean_values = cache_index["eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if not boolean_values.all():
        raise ValueError("cache_index eligible values must be booleans")
    result = cache_index.loc[cache_index["eligible"]].copy()
    if result.empty:
        raise ValueError("cache_index has no eligible rows")
    if not result["label_binary"].isin(_CLASS_TO_INDEX).all():
        raise ValueError("Eligible cache rows contain unknown or ambiguous labels")
    for column in ("tensor_sha256", "source_audio_sha256", "preprocessing_hash"):
        values = result[column].astype(str).str.casefold()
        if not values.map(lambda value: bool(_SHA256_PATTERN.fullmatch(value))).all():
            raise ValueError(f"Eligible cache rows contain invalid {column} values")
        result[column] = values
    identity = ["recording_key", "modality", "representation_id"]
    for optional in ("event_key", "label_source", "preprocessing_variant"):
        if optional in result:
            identity.append(optional)
    if result.duplicated(identity).any():
        raise ValueError(f"Eligible cache rows duplicate identity columns: {identity}")
    return result.reset_index(drop=True)


def _participant_table(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby("participant_key", sort=False)["label_binary"].nunique()
    mixed = counts[counts != 1]
    if not mixed.empty:
        examples = ", ".join(mixed.index.astype(str).tolist()[:5])
        raise ValueError(f"Participants with mixed labels cannot enter a manifest: {examples}")
    dataset_counts = frame.groupby("participant_key", sort=False)["dataset"].nunique()
    if (dataset_counts != 1).any():
        raise ValueError("A qualified participant_key maps to multiple datasets")
    people = (
        frame[["participant_key", "dataset", "label_binary"]]
        .drop_duplicates()
        .sort_values("participant_key", kind="mergesort")
        .reset_index(drop=True)
    )
    if set(people["label_binary"]) != set(_CLASS_TO_INDEX):
        raise ValueError("Participant cohort must contain positive and negative labels")
    return people


def _validate_fraction(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _resolve_temporal_analysis_mode(
    analysis_mode: str,
    *,
    train_fraction: float,
    validation_fraction: float,
    candidate_count: int,
    random_state: int,
    training_seed: int,
) -> str:
    mode = str(analysis_mode).strip().casefold()
    if mode not in {"confirmatory", "exploratory", "test_mode"}:
        raise ValueError(
            "analysis_mode must be confirmatory, exploratory, or test_mode"
        )
    integer_settings = {
        "candidate_count": candidate_count,
        "random_state": random_state,
        "training_seed": training_seed,
    }
    invalid_integer_settings = [
        name
        for name, value in integer_settings.items()
        if isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ]
    if invalid_integer_settings:
        raise ValueError(
            f"{mode} temporal integer settings must be exact integers: "
            f"{invalid_integer_settings}"
        )
    deviations: list[str] = []
    if not math.isclose(float(train_fraction), 0.6, rel_tol=0.0, abs_tol=1e-12):
        deviations.append("train_fraction")
    if not math.isclose(float(validation_fraction), 0.2, rel_tol=0.0, abs_tol=1e-12):
        deviations.append("validation_fraction")
    if int(candidate_count) != 1000:
        deviations.append("candidate_count")
    if int(random_state) != 42:
        deviations.append("random_state")
    if int(training_seed) != 42:
        deviations.append("training_seed")
    if mode == "confirmatory" and deviations:
        raise ValueError(
            "confirmatory temporal protocols are frozen to 60/20/20, "
            "candidate_count=1000, random_state=42, and training_seed=42; "
            f"deviations require explicit exploratory/test_mode labeling: {deviations}"
        )
    return mode


def _expand_assignments(
    cache: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    cohort: str,
) -> pd.DataFrame:
    result = cache.merge(
        assignments,
        on="participant_key",
        how="inner",
        validate="many_to_many",
        suffixes=("", "_assignment"),
    )
    if len(result) == 0:
        raise ValueError("Participant assignments selected no cache rows")
    result["protocol"] = protocol
    result["cohort"] = cohort
    _assert_no_participant_leakage(result)
    return result


def _audit_group_columns(manifest: pd.DataFrame) -> list[str]:
    # Cohort is descriptive, not a leakage boundary. Source and external cohorts
    # in the same fold must still be compared for participant/content overlap.
    columns = [column for column in ("protocol", "fold") if column in manifest]
    return columns or ["fold"]


def _assert_no_participant_leakage(manifest: pd.DataFrame) -> None:
    _require_columns(
        manifest,
        {"participant_key", "recording_key", "label_binary", "split"},
        "manifest",
    )
    groups = _audit_group_columns(manifest)
    for _, unit in manifest.groupby(groups, dropna=False, sort=False):
        labels = unit.groupby("participant_key")["label_binary"].nunique()
        if (labels != 1).any():
            raise ValueError("Manifest contains mixed labels within a participant")
        splits = unit.groupby("participant_key")["split"].nunique()
        if (splits != 1).any():
            raise ValueError("Manifest assigns one participant to multiple splits")
        for content_column in _CONTENT_HASH_COLUMNS:
            if content_column not in unit:
                continue
            content = unit.loc[
                unit[content_column].astype("string").notna()
                & unit[content_column].astype(str).ne("")
            ]
            split_counts = content.groupby(content_column)["split"].nunique()
            if split_counts.gt(1).any():
                raise ValueError(
                    f"Manifest contains content hash leakage in {content_column}"
                )
        identity = ["recording_key", "modality", "split", "representation_id"]
        for optional in ("event_key", "label_source", "preprocessing_variant"):
            if optional in unit:
                identity.append(optional)
        if unit.duplicated(identity).any():
            raise ValueError(f"Manifest duplicates recording identity: {identity}")
        representation_identity = ["recording_key", "modality", "split"]
        duplicated = unit.duplicated(representation_identity, keep=False)
        if duplicated.any():
            duplicate_rows = unit.loc[duplicated]
            required = {
                "paired_representation",
                "analysis_unit_key",
                "analysis_unit_weight",
            }
            if not required.issubset(duplicate_rows.columns):
                raise ValueError("Duplicate representation rows are not explicitly paired")
            if not duplicate_rows["paired_representation"].eq(True).all():
                raise ValueError("Duplicate representation rows are not explicitly paired")
            unit_counts = duplicate_rows.groupby(representation_identity)[
                "analysis_unit_key"
            ].nunique()
            if unit_counts.ne(1).any():
                raise ValueError("Paired representations disagree on analysis_unit_key")
        if "analysis_unit_key" in unit and "analysis_unit_weight" in unit:
            weights = unit.groupby(["split", "analysis_unit_key"])[
                "analysis_unit_weight"
            ].sum()
            if not np.allclose(weights.to_numpy(dtype=float), 1.0, atol=1e-12):
                raise ValueError("Paired representations inflate an analysis unit")


def build_protocol_matched_hst_manifest(
    cache_index: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    test_fraction: float = 0.2,
    validation_fraction_of_remaining: float = 0.125,
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
    eligibility_mapping: pd.DataFrame | None = None,
    eligibility_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Freeze ten literature-aligned, stratified participant holdouts."""
    supplied_seeds = tuple(int(seed) for seed in seeds)
    if supplied_seeds != PRESPECIFIED_HST_REPO_SEEDS:
        raise ValueError(
            "Track-A seeds must exactly match the prespecified released HST baseline seeds"
        )
    test_fraction = _validate_fraction(test_fraction, "test_fraction")
    validation_fraction = _validate_fraction(
        validation_fraction_of_remaining, "validation_fraction_of_remaining"
    )
    if not math.isclose(test_fraction, 0.2, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        validation_fraction, 0.125, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Track-A protocol is frozen to nominal 70/10/20 parameters")
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    cache, alignment, eligibility_audit = _apply_eligibility_alignment(
        cache_index,
        scientific_fingerprint=science,
        eligibility_mapping=eligibility_mapping,
        eligibility_fingerprint=eligibility_fingerprint,
    )
    people = _participant_table(cache)
    labels = people["label_binary"].map(_CLASS_TO_INDEX).to_numpy()
    class_counts = np.bincount(labels, minlength=2)
    if int(class_counts.min()) < 3:
        raise ValueError(
            "Each class needs at least three participants for nominal 70/10/20 splitting"
        )

    assignment_frames: list[pd.DataFrame] = []
    for fold, seed in enumerate(supplied_seeds, start=1):
        outer = StratifiedShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=seed
        )
        train_validation_index, test_index = next(outer.split(people, labels))
        train_validation = people.iloc[train_validation_index].reset_index(drop=True)
        train_validation_labels = labels[train_validation_index]
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed,
        )
        train_index, validation_index = next(
            inner.split(train_validation, train_validation_labels)
        )
        assignments = pd.concat(
            [
                train_validation.iloc[train_index][["participant_key"]].assign(split="train"),
                train_validation.iloc[validation_index][["participant_key"]].assign(
                    split="validation"
                ),
                people.iloc[test_index][["participant_key"]].assign(split="test"),
            ],
            ignore_index=True,
        )
        assignments["fold"] = fold
        assignments["split_seed"] = seed
        assignments["training_seed"] = seed
        assignments["test_fraction"] = test_fraction
        assignments["validation_fraction_of_remaining"] = validation_fraction
        realized = assignments["split"].value_counts()
        assignments["nominal_split_ratio"] = "70/10/20"
        assignments["split_fraction_semantics"] = (
            "nominal_parameters_with_sklearn_realized_counts"
        )
        for split in _SPLITS:
            count = int(realized.get(split, 0))
            assignments[f"realized_{split}_participant_count"] = count
            assignments[f"realized_{split}_fraction"] = count / len(people)
        assignment_frames.append(assignments)

    manifest = _expand_assignments(
        cache,
        pd.concat(assignment_frames, ignore_index=True),
        protocol=TRACK_A_PROTOCOL,
        cohort="project_target_all_eligible",
    )
    manifest["seed_provenance"] = "released_hst_baseline_scripts"
    manifest["evaluation_design"] = "ten_repeated_stratified_participant_holdouts"
    manifest = _stamp_analysis_provenance(
        manifest,
        analysis_scope="internal_performance",
        analysis_role="primary",
        estimand_id="track_a_internal_hst_vs_aligned_comparator",
        multiplicity_family="primary_internal_performance",
        analysis_mode="confirmatory",
        confirmatory_protocol=True,
    )
    manifest = _stamp_builder_provenance(
        manifest,
        scientific_fingerprint=science,
        eligibility_fingerprint=alignment,
        eligibility_audit=eligibility_audit,
    )
    result = _finalize_manifest(manifest)
    clean = audit_hst_manifest(result)
    integrity_columns = (
        "participant_overlap_count",
        "content_hash_leakage_count",
        "unpaired_duplicate_representation_count",
        "analysis_weight_violation_count",
        "invalid_audit_payload_count",
    )
    if any(not clean[column].eq(0).all() for column in integrity_columns):
        raise AssertionError("Track-A manifest failed leakage or provenance audit")
    return result


def _parse_explicit_boolean(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    token = " ".join(str(value).strip().casefold().replace("_", " ").split())
    if token in {"1", "true", "yes", "y", "present", "cough", "coughing"}:
        return True
    if token in {"0", "false", "no", "n", "absent", "none", "no cough"}:
        return False
    return None


def _parse_symptom_list(value: object) -> bool | None:
    explicit = _parse_explicit_boolean(value)
    if explicit is not None:
        return explicit
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    token = str(value).strip().casefold()
    if not token:
        return None
    values = {
        " ".join(item.strip().replace("_", " ").split())
        for item in re.split(r"[,;|]", token)
        if item.strip()
    }
    if "no cough" in values:
        return False
    if values & {"cough", "coughing", "dry cough", "wet cough"}:
        return True
    return None


def _symptom_fields(frame: pd.DataFrame) -> list[str]:
    explicit = [
        column
        for column in ("cough_symptom_present", "cough_symptom", "cough")
        if column in frame
    ]
    if explicit:
        return explicit
    listed = [
        column
        for column in ("symptoms", "symptom_list", "reported_symptoms")
        if column in frame
    ]
    if not listed:
        raise ValueError("Task-2-like cohort requires an explicit cough symptom field")
    return listed


def build_hst_task2_like_cough_manifest(
    cache_index: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
    eligibility_mapping: pd.DataFrame | None = None,
    eligibility_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Build the symptom-matched cough sensitivity without relabeling the task."""
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    cache, _, _ = _apply_eligibility_alignment(
        cache_index,
        scientific_fingerprint=science,
        eligibility_mapping=eligibility_mapping,
        eligibility_fingerprint=eligibility_fingerprint,
    )
    fields = _symptom_fields(cache)
    parsed = pd.DataFrame(index=cache.index)
    for field in fields:
        parser = _parse_symptom_list if "symptom" in field and field.endswith("s") else _parse_explicit_boolean
        parsed[field] = cache[field].map(parser)

    participant_status: dict[str, bool | None] = {}
    exclusion_reason: dict[str, str] = {}
    for participant_key, row_indices in cache.groupby("participant_key").groups.items():
        values = {
            bool(value)
            for value in parsed.loc[list(row_indices)].to_numpy().ravel().tolist()
            if value is not None and not pd.isna(value)
        }
        if values == {True}:
            participant_status[str(participant_key)] = True
            exclusion_reason[str(participant_key)] = "included"
        elif values == {False}:
            participant_status[str(participant_key)] = False
            exclusion_reason[str(participant_key)] = "cough_symptom_absent"
        elif values == {True, False}:
            participant_status[str(participant_key)] = None
            exclusion_reason[str(participant_key)] = "conflicting_cough_symptom"
        else:
            participant_status[str(participant_key)] = None
            exclusion_reason[str(participant_key)] = "missing_or_ambiguous_cough_symptom"

    selected = cache[
        cache["participant_key"].astype(str).map(participant_status).eq(True)
        & cache["modality"].astype(str).eq("cough")
    ].copy()
    selected["cough_symptom_present"] = True
    selected["cough_symptom_source_fields"] = "|".join(fields)
    task2_eligibility = _restrict_eligibility_mapping(
        eligibility_mapping,
        selected,
        restriction_reason="explicit_cough_symptom_and_cough_modality",
    )
    manifest = build_protocol_matched_hst_manifest(
        selected,
        seeds=seeds,
        scientific_config=scientific_config,
        scientific_fingerprint=science,
        eligibility_mapping=task2_eligibility,
    )
    manifest = _without_hash_columns(manifest)
    manifest["cohort"] = "hst_task2_like_cough"
    manifest = _stamp_analysis_provenance(
        manifest,
        analysis_scope="symptom_matched_cough",
        analysis_role="exploratory",
        estimand_id="task2_like_cough_internal_performance",
        multiplicity_family="exploratory_task2_like",
        analysis_mode="exploratory",
        confirmatory_protocol=False,
    )

    participants = cache[["participant_key"]].drop_duplicates()
    participants["reason"] = participants["participant_key"].astype(str).map(exclusion_reason)
    symptom_audit = (
        participants.groupby("reason", dropna=False)
        .size()
        .rename("participant_count")
        .reset_index()
        .sort_values("reason")
        .reset_index(drop=True)
    )
    manifest = _embed_audit_payload(
        manifest,
        "symptom_exclusion_audit",
        _frame_payload(symptom_audit),
    )
    result = _finalize_manifest(manifest)
    result.attrs["symptom_exclusion_audit"] = symptom_audit.copy()
    result.attrs["symptom_source_fields"] = tuple(fields)
    return result


def _date_column(frame: pd.DataFrame) -> str:
    present = [column for column in _DATE_COLUMNS if column in frame]
    if not present:
        raise ValueError(f"Date protocol requires one of these columns: {list(_DATE_COLUMNS)}")
    return present[0]


def _date_eligible_participants(
    cache_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = _eligible_cache(cache_index)
    people = _participant_table(cache)
    column = _date_column(cache)
    dated = cache[["participant_key", "recording_key", column]].copy()
    dated["_timestamp"] = pd.to_datetime(dated[column], errors="coerce", utc=True)
    timestamp_counts = dated.groupby(
        ["participant_key", "recording_key"], dropna=False
    )["_timestamp"].nunique(dropna=False)
    if timestamp_counts.gt(1).any():
        raise ValueError("Representations disagree on a recording timestamp")
    dated = dated.drop_duplicates(
        ["participant_key", "recording_key"], keep="first"
    )
    dated["_session_date_utc"] = dated["_timestamp"].dt.normalize()
    date_counts = (
        dated.groupby("participant_key")["_timestamp"]
        .agg(
            valid_recording_count=lambda values: int(values.notna().sum()),
            invalid_recording_count=lambda values: int(values.isna().sum()),
            participant_recording_timestamp_min_utc="min",
            participant_recording_timestamp_max_utc="max",
        )
        .reset_index()
    )
    session_counts = (
        dated.groupby("participant_key")["_session_date_utc"]
        .nunique(dropna=True)
        .rename("utc_session_date_count")
        .reset_index()
    )
    date_counts = date_counts.merge(
        session_counts, on="participant_key", how="left", validate="one_to_one"
    )
    date_counts["participant_timestamp_utc"] = date_counts[
        "participant_recording_timestamp_min_utc"
    ]
    people = people.merge(
        date_counts, on="participant_key", how="left", validate="one_to_one"
    )
    reason = pd.Series("date_eligible", index=people.index, dtype="string")
    no_valid = people["valid_recording_count"].fillna(0).eq(0)
    partially_unparseable = (
        people["invalid_recording_count"].fillna(0).gt(0) & ~no_valid
    )
    reason.loc[no_valid] = "no_parseable_participant_date"
    reason.loc[partially_unparseable] = "partially_unparseable_recording_dates"
    people["date_eligibility_reason"] = reason
    eligible_people = people[people["date_eligibility_reason"].eq("date_eligible")].copy()
    if len(eligible_people) < 5:
        raise ValueError("Too few participants have a parseable recording timestamp")
    eligible_people["participant_month_ordinal"] = (
        eligible_people["participant_timestamp_utc"].dt.year * 12
        + eligible_people["participant_timestamp_utc"].dt.month
    ).astype(int)
    eligible_people = eligible_people.sort_values(
        ["participant_timestamp_utc", "participant_key"], kind="mergesort"
    ).reset_index(drop=True)

    audit_rows: list[dict[str, object]] = []
    partial_count = int(
        (people["invalid_recording_count"].fillna(0).gt(0) & ~no_valid).sum()
    )
    multiple_date_count = int(
        (people["utc_session_date_count"].fillna(0).gt(1) & ~no_valid).sum()
    )
    for audit_reason in (
        "date_eligible",
        "partially_unparseable_recording_dates",
        "no_parseable_participant_date",
    ):
        keys = sorted(
            people.loc[
                people["date_eligibility_reason"].eq(audit_reason),
                "participant_key",
            ].astype(str)
        )
        audit_rows.append(
            {
                "reason": audit_reason,
                "participant_count": len(keys),
                "participant_keys_json": _canonical_json(keys),
                "participant_with_unparseable_recording_count": partial_count,
                "participant_with_multiple_valid_dates_count": multiple_date_count,
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit["date_source_column"] = column
    eligible_cache = cache[
        cache["participant_key"].isin(eligible_people["participant_key"])
    ].copy()
    eligible_cache["date_source_column"] = column
    return eligible_cache, eligible_people, audit


def _assert_temporal_label_support(assignments: pd.DataFrame) -> None:
    people = assignments[
        ["participant_key", "label_binary", "split"]
    ].drop_duplicates()
    for split in _SPLITS:
        observed = set(people.loc[people["split"].eq(split), "label_binary"])
        if observed != set(_CLASS_TO_INDEX):
            raise ValueError(
                f"Temporal partition {split} must contain both labels"
            )


def _verify_temporal_recording_boundaries(
    cache: pd.DataFrame,
    assignments: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> pd.DataFrame:
    date_column = _date_column(cache)
    assignment_rows = assignments[["participant_key", "split"]].drop_duplicates()
    recordings = cache[["participant_key", "recording_key", date_column]].drop_duplicates(
        ["participant_key", "recording_key"], keep="first"
    ).merge(
        assignment_rows,
        on="participant_key",
        how="inner",
        validate="many_to_one",
    )
    recordings["_recording_timestamp_utc"] = pd.to_datetime(
        recordings[date_column], errors="coerce", utc=True
    )
    unparseable_count = int(recordings["_recording_timestamp_utc"].isna().sum())
    recordings = recordings[recordings["_recording_timestamp_utc"].notna()].copy()
    result = boundaries.copy()
    result["left_recording_timestamp_max_utc"] = ""
    result["right_recording_timestamp_min_utc"] = ""
    result["recording_timestamp_order_verified"] = False
    result["parseable_recordings_order_verified"] = False
    result["full_recording_order_verified"] = False
    result["unparseable_recording_timestamp_count"] = unparseable_count
    verification_status = (
        "partial_unknown_unparseable_recordings"
        if unparseable_count
        else "all_recordings_verified"
    )
    result["recording_order_verification_status"] = verification_status
    for index, boundary in result.iterrows():
        left_max = recordings.loc[
            recordings["split"].eq(boundary["left_split"]),
            "_recording_timestamp_utc",
        ].max()
        right_min = recordings.loc[
            recordings["split"].eq(boundary["right_split"]),
            "_recording_timestamp_utc",
        ].min()
        if pd.isna(left_max) or pd.isna(right_min) or not left_max < right_min:
            raise ValueError(
                "Strict temporal recording timestamps cross a split boundary"
            )
        result.at[index, "left_recording_timestamp_max_utc"] = left_max.isoformat()
        result.at[index, "right_recording_timestamp_min_utc"] = right_min.isoformat()
        result.at[index, "parseable_recordings_order_verified"] = True
        if unparseable_count == 0:
            result.at[index, "recording_timestamp_order_verified"] = True
            result.at[index, "full_recording_order_verified"] = True
    return result


def _split_counts(
    n_people: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int, int]:
    train_fraction = _validate_fraction(train_fraction, "train_fraction")
    validation_fraction = _validate_fraction(validation_fraction, "validation_fraction")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if n_people < 5:
        raise ValueError("At least five participants are required")
    train_count = int(math.floor(n_people * train_fraction))
    validation_count = int(math.floor(n_people * validation_fraction))
    test_count = n_people - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("Requested fractions create an empty split")
    return train_count, validation_count, test_count


def _chronological_assignments(
    people: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_count, validation_count, test_count = _split_counts(
        len(people), train_fraction, validation_fraction
    )
    return _tie_safe_temporal_assignments(
        people,
        split_order=("train", "validation", "test"),
        desired_counts=(train_count, validation_count, test_count),
        boundary_names=("train_to_validation", "validation_to_test"),
    )


def _tie_safe_temporal_assignments(
    people: pd.DataFrame,
    *,
    split_order: tuple[str, str, str],
    desired_counts: tuple[int, int, int],
    boundary_names: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = people.sort_values(
        ["participant_timestamp_utc", "participant_key"], kind="mergesort"
    ).reset_index(drop=True)
    n_people = len(ordered)
    desired_boundaries = (
        int(desired_counts[0]),
        int(desired_counts[0] + desired_counts[1]),
    )
    valid_boundaries = [
        index
        for index in range(1, n_people)
        if ordered.iloc[index - 1]["participant_timestamp_utc"]
        != ordered.iloc[index]["participant_timestamp_utc"]
    ]
    pairs = [
        (left, right)
        for left in valid_boundaries
        for right in valid_boundaries
        if left < right
    ]
    if not pairs:
        raise ValueError("Temporal cohort has fewer than three timestamp groups")
    first, second = min(
        pairs,
        key=lambda pair: (
            abs(pair[0] - desired_boundaries[0])
            + abs(pair[1] - desired_boundaries[1]),
            pair[0],
            pair[1],
        ),
    )
    result = ordered.copy()
    result["split"] = split_order[2]
    result.loc[: first - 1, "split"] = split_order[0]
    result.loc[first : second - 1, "split"] = split_order[1]
    _assert_temporal_label_support(result)
    realized_counts = (first, second - first, n_people - second)
    desired_json = _canonical_json(dict(zip(split_order, desired_counts)))
    realized_json = _canonical_json(dict(zip(split_order, realized_counts)))

    rows: list[dict[str, object]] = []
    for index, (name, desired, realized) in enumerate(
        zip(boundary_names, desired_boundaries, (first, second))
    ):
        desired_timestamp = ordered.iloc[desired - 1]["participant_timestamp_utc"]
        desired_tie_count = int(
            ordered["participant_timestamp_utc"].eq(desired_timestamp).sum()
        )
        rows.append(
            {
                "boundary_name": name,
                "left_split": split_order[index],
                "right_split": split_order[index + 1],
                "desired_boundary_index": desired,
                "realized_boundary_index": realized,
                "boundary_moved": bool(desired != realized),
                "boundary_tie_count": (
                    desired_tie_count
                    if desired not in valid_boundaries
                    else 0
                ),
                "left_timestamp_utc": result.iloc[realized - 1][
                    "participant_timestamp_utc"
                ],
                "right_timestamp_utc": result.iloc[realized][
                    "participant_timestamp_utc"
                ],
                "desired_left_partition_count": desired,
                "realized_left_partition_count": realized,
                "desired_split_counts_json": desired_json,
                "realized_split_counts_json": realized_json,
            }
        )
    return result, pd.DataFrame(rows)


def _boundary_crossing_participants(
    assignments: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> dict[str, list[str]]:
    crossing: dict[str, list[str]] = {}
    for _, boundary in boundaries.iterrows():
        right_min = assignments.loc[
            assignments["split"].eq(boundary["right_split"]),
            "participant_recording_timestamp_min_utc",
        ].min()
        left = assignments[assignments["split"].eq(boundary["left_split"])]
        keys = sorted(
            left.loc[
                left["participant_recording_timestamp_max_utc"].ge(right_min),
                "participant_key",
            ].astype(str)
        )
        for key in keys:
            crossing.setdefault(key, []).append(str(boundary["boundary_name"]))
    return crossing


def _temporal_assignments_excluding_crossers(
    people: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    split_order: tuple[str, str, str],
    boundary_names: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    remaining = people.copy()
    excluded: dict[str, list[str]] = {}
    while True:
        train_count, validation_count, test_count = _split_counts(
            len(remaining), train_fraction, validation_fraction
        )
        counts = {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
        assignments, boundaries = _tie_safe_temporal_assignments(
            remaining,
            split_order=split_order,
            desired_counts=tuple(counts[split] for split in split_order),
            boundary_names=boundary_names,
        )
        current_crossing = _boundary_crossing_participants(assignments, boundaries)
        if not current_crossing:
            return assignments, boundaries, excluded
        for key, names in current_crossing.items():
            excluded.setdefault(key, []).extend(names)
        remaining = remaining[
            ~remaining["participant_key"].astype(str).isin(current_crossing)
        ].copy()


def _audit_boundary_crossing_exclusions(
    date_audit: pd.DataFrame,
    assignments: pd.DataFrame,
    excluded: dict[str, list[str]],
) -> pd.DataFrame:
    result = date_audit.copy()
    included_keys = sorted(assignments["participant_key"].astype(str).unique())
    eligible = result["reason"].eq("date_eligible")
    result.loc[eligible, "participant_count"] = len(included_keys)
    result.loc[eligible, "participant_keys_json"] = _canonical_json(included_keys)
    if excluded:
        row = {
            "reason": "recording_span_crosses_temporal_boundary",
            "participant_count": len(excluded),
            "participant_keys_json": _canonical_json(sorted(excluded)),
            "boundary_crossings_json": _canonical_json(
                {
                    key: sorted(set(names))
                    for key, names in sorted(excluded.items())
                }
            ),
            "date_source_column": result["date_source_column"].iloc[0],
            "participant_with_unparseable_recording_count": int(
                result["participant_with_unparseable_recording_count"].iloc[0]
            ),
            "participant_with_multiple_valid_dates_count": int(
                result["participant_with_multiple_valid_dates_count"].iloc[0]
            ),
        }
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result.sort_values("reason", kind="mergesort").reset_index(drop=True)


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("inf")
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, values, side="right") / len(left_sorted)
    right_cdf = np.searchsorted(right_sorted, values, side="right") / len(right_sorted)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _assignment_score(
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
    splits: Sequence[str],
) -> dict[str, float]:
    full_month = reference["participant_month_ordinal"].to_numpy(dtype=float)
    full_time = (
        reference["participant_timestamp_utc"].astype("int64").to_numpy(dtype=float)
    )
    month_scale = float(np.std(full_month, ddof=1)) if len(full_month) > 1 else 0.0
    metrics: dict[str, float] = {}
    objective_values: list[float] = []
    for split in splits:
        current = assignments[assignments["split"].eq(split)]
        current_month = current["participant_month_ordinal"].to_numpy(dtype=float)
        current_time = (
            current["participant_timestamp_utc"].astype("int64").to_numpy(dtype=float)
        )
        mean_difference = abs(float(np.mean(current_month) - np.mean(full_month)))
        smd = mean_difference / month_scale if month_scale > 0 else float(mean_difference > 0)
        ks = _ks_distance(current_time, full_time)
        metrics[f"{split}_month_smd_abs"] = smd
        metrics[f"{split}_date_ks"] = ks
        objective_values.extend([smd, ks])
    metrics["objective"] = max(objective_values)
    return metrics


def _class_counts_by_split(assignments: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts = assignments.groupby(["split", "label_binary"]).size()
    return {(str(split), str(label)): int(count) for (split, label), count in counts.items()}


def _candidate_assignment(
    people: pd.DataFrame,
    *,
    target_counts: dict[tuple[str, str], int],
    splits: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for label in ("negative", "positive"):
        label_people = people[people["label_binary"].eq(label)].copy()
        order = rng.permutation(len(label_people))
        label_people = label_people.iloc[order].reset_index(drop=True)
        offset = 0
        for split in splits:
            count = int(target_counts.get((split, label), 0))
            chosen = label_people.iloc[offset : offset + count].copy()
            chosen["split"] = split
            frames.append(chosen)
            offset += count
        if offset != len(label_people):
            raise ValueError("Target class counts do not cover the candidate participant pool")
    return pd.concat(frames, ignore_index=True).sort_values(
        "participant_key", kind="mergesort"
    ).reset_index(drop=True)


def _select_date_balanced_assignment(
    people: pd.DataFrame,
    *,
    target_counts: dict[tuple[str, str], int],
    splits: Sequence[str],
    candidate_count: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(candidate_count) < 1:
        raise ValueError("candidate_count must be positive")
    rows: list[dict[str, object]] = []
    candidates: dict[int, pd.DataFrame] = {}
    for candidate_index in range(int(candidate_count)):
        seed = int(random_state) + candidate_index
        assignment = _candidate_assignment(
            people,
            target_counts=target_counts,
            splits=splits,
            seed=seed,
        )
        candidates[seed] = assignment
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_seed": seed,
                **_assignment_score(assignment, people, splits),
            }
        )
    scores = pd.DataFrame(rows).sort_values("candidate_seed").reset_index(drop=True)
    selected = scores.sort_values(
        ["objective", "candidate_seed"], kind="mergesort"
    ).iloc[0]
    return candidates[int(selected["candidate_seed"])], scores


def _split_summary(people: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in _SPLITS:
        current = people[people["split"].eq(split)]
        rows.append(
            {
                "split": split,
                "participant_count": int(len(current)),
                "positive_count": int(current["label_binary"].eq("positive").sum()),
                "negative_count": int(current["label_binary"].eq("negative").sum()),
                "prevalence": float(current["label_binary"].eq("positive").mean()),
                "date_min": current["participant_timestamp_utc"].min(),
                "date_max": current["participant_timestamp_utc"].max(),
            }
        )
    return pd.DataFrame(rows)


def _attach_temporal_attrs(
    manifest: pd.DataFrame,
    *,
    date_audit: pd.DataFrame,
    split_summary: pd.DataFrame,
    candidate_scores: pd.DataFrame | None = None,
    boundary_diagnostics: pd.DataFrame | None = None,
) -> None:
    manifest.attrs["date_eligibility_audit"] = date_audit.copy()
    manifest.attrs["split_summary"] = split_summary.copy()
    if candidate_scores is not None:
        manifest.attrs["candidate_scores"] = candidate_scores.copy()
    if boundary_diagnostics is not None:
        manifest.attrs["boundary_diagnostics"] = boundary_diagnostics.copy()


def _temporal_analysis_provenance(
    protocol: str,
    analysis_mode: str,
) -> dict[str, object]:
    if protocol in {SPLIT_POLICY_MIXED_PROTOCOL, SPLIT_POLICY_CHRONOLOGICAL_PROTOCOL}:
        estimand_id = "split_policy_temporal_contrast"
        analysis_scope = "reliability_evaluation"
        analysis_role = "secondary"
        multiplicity_family = "prespecified_reliability"
    elif protocol in {COMMON_LATE_BALANCED_PROTOCOL, COMMON_LATE_CHRONOLOGICAL_PROTOCOL}:
        estimand_id = "common_late_temporal_contrast"
        analysis_scope = "reliability_evaluation"
        analysis_role = "secondary"
        multiplicity_family = "prespecified_reliability"
    elif protocol == REVERSE_TEMPORAL_PROTOCOL:
        estimand_id = "reverse_temporal_direction"
        analysis_scope = "sensitivity_analysis"
        analysis_role = "sensitivity"
        multiplicity_family = "temporal_sensitivity"
    else:
        raise ValueError(f"Unknown temporal protocol provenance: {protocol}")
    return {
        "analysis_scope": analysis_scope,
        "analysis_role": analysis_role,
        "estimand_id": estimand_id,
        "multiplicity_family": multiplicity_family,
        "analysis_mode": analysis_mode,
        "confirmatory_protocol": (
            analysis_mode == "confirmatory" and protocol != REVERSE_TEMPORAL_PROTOCOL
        ),
    }


def _audit_frame_with_protocol_context(
    frame: pd.DataFrame,
    *,
    protocol: str,
    provenance: Mapping[str, object],
) -> pd.DataFrame:
    result = frame.copy()
    result["audit_protocol"] = protocol
    for column, value in provenance.items():
        result[column] = value
    return result


def _finalize_temporal_manifest(
    frame: pd.DataFrame,
    *,
    scientific_fingerprint: str,
    eligibility_fingerprint: str,
    eligibility_audit: object,
    date_audit: pd.DataFrame,
    split_summary: pd.DataFrame,
    analysis_mode: str,
    boundary_diagnostics: pd.DataFrame | None = None,
    candidate_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    protocols = set(frame["protocol"].astype(str))
    if len(protocols) != 1:
        raise ValueError("A temporal manifest must contain exactly one protocol")
    protocol = next(iter(protocols))
    provenance = _temporal_analysis_provenance(protocol, analysis_mode)
    frame = _stamp_analysis_provenance(frame, **provenance)
    contextual_date_audit = _audit_frame_with_protocol_context(
        date_audit, protocol=protocol, provenance=provenance
    )
    contextual_split_summary = _audit_frame_with_protocol_context(
        split_summary, protocol=protocol, provenance=provenance
    )
    contextual_candidates = (
        _audit_frame_with_protocol_context(
            candidate_scores, protocol=protocol, provenance=provenance
        )
        if candidate_scores is not None
        else None
    )
    contextual_boundaries = (
        _audit_frame_with_protocol_context(
            boundary_diagnostics, protocol=protocol, provenance=provenance
        )
        if boundary_diagnostics is not None
        else None
    )
    contextual_eligibility_audit = {
        "audit_protocol": protocol,
        **provenance,
        "eligibility_audit": eligibility_audit,
    }
    result = _stamp_builder_provenance(
        frame,
        scientific_fingerprint=scientific_fingerprint,
        eligibility_fingerprint=eligibility_fingerprint,
        eligibility_audit=contextual_eligibility_audit,
    )
    payloads = {
        "date_eligibility_audit": contextual_date_audit,
        "split_summary": contextual_split_summary,
    }
    if contextual_boundaries is not None:
        payloads["boundary_diagnostics"] = contextual_boundaries
    if contextual_candidates is not None:
        payloads["candidate_scores"] = contextual_candidates
    for prefix, payload in payloads.items():
        result = _embed_audit_payload(result, prefix, _frame_payload(payload))
    result = _finalize_manifest(result)
    _attach_temporal_attrs(
        result,
        date_audit=contextual_date_audit,
        split_summary=contextual_split_summary,
        candidate_scores=contextual_candidates,
        boundary_diagnostics=contextual_boundaries,
    )
    return result


def build_split_policy_contrast_manifests(
    cache_index: pd.DataFrame,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    candidate_count: int = 1000,
    random_state: int = 42,
    training_seed: int = 42,
    analysis_mode: str = "confirmatory",
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
    eligibility_mapping: pd.DataFrame | None = None,
    eligibility_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build matched calendar-mixed and chronological policy manifests."""
    mode = _resolve_temporal_analysis_mode(
        analysis_mode,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        candidate_count=candidate_count,
        random_state=random_state,
        training_seed=training_seed,
    )
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    aligned, alignment, eligibility_audit = _apply_eligibility_alignment(
        cache_index,
        scientific_fingerprint=science,
        eligibility_mapping=eligibility_mapping,
        eligibility_fingerprint=eligibility_fingerprint,
    )
    cache, people, date_audit = _date_eligible_participants(aligned)
    chronological, boundaries, crossing_exclusions = _temporal_assignments_excluding_crossers(
        people,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        split_order=("train", "validation", "test"),
        boundary_names=("train_to_validation", "validation_to_test"),
    )
    included_keys = set(chronological["participant_key"].astype(str))
    people = people[people["participant_key"].astype(str).isin(included_keys)].copy()
    cache = cache[cache["participant_key"].astype(str).isin(included_keys)].copy()
    date_audit = _audit_boundary_crossing_exclusions(
        date_audit, chronological, crossing_exclusions
    )
    boundaries = _verify_temporal_recording_boundaries(
        cache, chronological, boundaries
    )
    target_counts = _class_counts_by_split(chronological)
    mixed, scores = _select_date_balanced_assignment(
        people,
        target_counts=target_counts,
        splits=_SPLITS,
        candidate_count=candidate_count,
        random_state=random_state,
    )
    if _class_counts_by_split(mixed) != target_counts:
        raise AssertionError("Date-balanced assignment changed split class counts")
    score_hash = _content_digest(scores.to_dict(orient="records"))
    selected_seed = int(
        scores.sort_values(["objective", "candidate_seed"], kind="mergesort").iloc[0][
            "candidate_seed"
        ]
    )
    common = {
        "fold": 1,
        "training_seed": int(training_seed),
        "candidate_count": int(candidate_count),
        "random_state": int(random_state),
        "analysis_mode": mode,
        "confirmatory_protocol": mode == "confirmatory",
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": 1.0 - float(train_fraction) - float(validation_fraction),
        "candidate_scores_sha256": score_hash,
        "date_assignment_only": True,
    }
    mixed = mixed.assign(assignment_seed=selected_seed, **common)
    chronological = chronological.assign(assignment_seed=pd.NA, **common)
    mixed_frame = _expand_assignments(
        cache,
        mixed.drop(columns=["dataset", "label_binary"], errors="ignore"),
        protocol=SPLIT_POLICY_MIXED_PROTOCOL,
        cohort="date_eligible_project_target",
    )
    chronological_frame = _expand_assignments(
        cache,
        chronological.drop(columns=["dataset", "label_binary"], errors="ignore"),
        protocol=SPLIT_POLICY_CHRONOLOGICAL_PROTOCOL,
        cohort="date_eligible_project_target",
    )
    mixed_summary = _split_summary(mixed)
    chronological_summary = _split_summary(chronological)
    boundaries = boundaries.copy()
    boundaries["diagnostic_protocol"] = SPLIT_POLICY_CHRONOLOGICAL_PROTOCOL
    mixed_manifest = _finalize_temporal_manifest(
        mixed_frame,
        scientific_fingerprint=science,
        eligibility_fingerprint=alignment,
        eligibility_audit=eligibility_audit,
        date_audit=date_audit,
        split_summary=mixed_summary,
        analysis_mode=mode,
        candidate_scores=scores,
    )
    chronological_manifest = _finalize_temporal_manifest(
        chronological_frame,
        scientific_fingerprint=science,
        eligibility_fingerprint=alignment,
        eligibility_audit=eligibility_audit,
        date_audit=date_audit,
        split_summary=chronological_summary,
        analysis_mode=mode,
        candidate_scores=scores,
        boundary_diagnostics=boundaries,
    )
    return mixed_manifest, chronological_manifest


def build_common_late_test_manifests(
    cache_index: pd.DataFrame,
    *,
    candidate_count: int = 1000,
    random_state: int = 42,
    training_seed: int = 42,
    analysis_mode: str = "confirmatory",
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
    eligibility_mapping: pd.DataFrame | None = None,
    eligibility_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold the latest test cohort fixed while varying source assignment."""
    mode = _resolve_temporal_analysis_mode(
        analysis_mode,
        train_fraction=0.6,
        validation_fraction=0.2,
        candidate_count=candidate_count,
        random_state=random_state,
        training_seed=training_seed,
    )
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    aligned, alignment, eligibility_audit = _apply_eligibility_alignment(
        cache_index,
        scientific_fingerprint=science,
        eligibility_mapping=eligibility_mapping,
        eligibility_fingerprint=eligibility_fingerprint,
    )
    cache, people, date_audit = _date_eligible_participants(aligned)
    chronological, boundaries, crossing_exclusions = _temporal_assignments_excluding_crossers(
        people,
        train_fraction=0.6,
        validation_fraction=0.2,
        split_order=("train", "validation", "test"),
        boundary_names=("train_to_validation", "validation_to_test"),
    )
    included_keys = set(chronological["participant_key"].astype(str))
    people = people[people["participant_key"].astype(str).isin(included_keys)].copy()
    cache = cache[cache["participant_key"].astype(str).isin(included_keys)].copy()
    date_audit = _audit_boundary_crossing_exclusions(
        date_audit, chronological, crossing_exclusions
    )
    boundaries = _verify_temporal_recording_boundaries(
        cache, chronological, boundaries
    )
    source_people = chronological[~chronological["split"].eq("test")].drop(
        columns="split"
    )
    fixed_test = chronological[chronological["split"].eq("test")].copy()
    target_counts = _class_counts_by_split(
        chronological[chronological["split"].isin(("train", "validation"))]
    )
    balanced_source, scores = _select_date_balanced_assignment(
        source_people,
        target_counts=target_counts,
        splits=("train", "validation"),
        candidate_count=candidate_count,
        random_state=random_state,
    )
    balanced = pd.concat([balanced_source, fixed_test], ignore_index=True)
    selected_seed = int(
        scores.sort_values(["objective", "candidate_seed"], kind="mergesort").iloc[0][
            "candidate_seed"
        ]
    )
    score_hash = _content_digest(scores.to_dict(orient="records"))
    common = {
        "fold": 1,
        "training_seed": int(training_seed),
        "candidate_count": int(candidate_count),
        "random_state": int(random_state),
        "analysis_mode": mode,
        "confirmatory_protocol": mode == "confirmatory",
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "candidate_scores_sha256": score_hash,
        "common_late_test": True,
    }
    balanced = balanced.assign(assignment_seed=selected_seed, **common)
    chronological = chronological.assign(assignment_seed=pd.NA, **common)
    balanced_frame = _expand_assignments(
        cache,
        balanced.drop(columns=["dataset", "label_binary"], errors="ignore"),
        protocol=COMMON_LATE_BALANCED_PROTOCOL,
        cohort="date_eligible_project_target",
    )
    chronological_frame = _expand_assignments(
        cache,
        chronological.drop(columns=["dataset", "label_binary"], errors="ignore"),
        protocol=COMMON_LATE_CHRONOLOGICAL_PROTOCOL,
        cohort="date_eligible_project_target",
    )
    balanced_summary = _split_summary(balanced)
    chronological_summary = _split_summary(chronological)
    boundaries = boundaries.copy()
    boundaries.loc[
        boundaries["boundary_name"].eq("validation_to_test"), "boundary_name"
    ] = "validation_to_fixed_late_test"
    boundaries["diagnostic_protocol"] = COMMON_LATE_CHRONOLOGICAL_PROTOCOL
    balanced_manifest = _finalize_temporal_manifest(
        balanced_frame,
        scientific_fingerprint=science,
        eligibility_fingerprint=alignment,
        eligibility_audit=eligibility_audit,
        date_audit=date_audit,
        split_summary=balanced_summary,
        analysis_mode=mode,
        candidate_scores=scores,
    )
    chronological_manifest = _finalize_temporal_manifest(
        chronological_frame,
        scientific_fingerprint=science,
        eligibility_fingerprint=alignment,
        eligibility_audit=eligibility_audit,
        date_audit=date_audit,
        split_summary=chronological_summary,
        analysis_mode=mode,
        candidate_scores=scores,
        boundary_diagnostics=boundaries,
    )
    return balanced_manifest, chronological_manifest


def build_reverse_temporal_hst_manifest(
    cache_index: pd.DataFrame,
    *,
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
    eligibility_mapping: pd.DataFrame | None = None,
    eligibility_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Build the prespecified latest-to-earliest temporal sensitivity."""
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    aligned, alignment, eligibility_audit = _apply_eligibility_alignment(
        cache_index,
        scientific_fingerprint=science,
        eligibility_mapping=eligibility_mapping,
        eligibility_fingerprint=eligibility_fingerprint,
    )
    cache, people, date_audit = _date_eligible_participants(aligned)
    reverse, boundaries, crossing_exclusions = _temporal_assignments_excluding_crossers(
        people,
        train_fraction=0.6,
        validation_fraction=0.2,
        split_order=("test", "validation", "train"),
        boundary_names=(
            "earliest_test_to_validation",
            "validation_to_latest_train",
        ),
    )
    included_keys = set(reverse["participant_key"].astype(str))
    cache = cache[cache["participant_key"].astype(str).isin(included_keys)].copy()
    date_audit = _audit_boundary_crossing_exclusions(
        date_audit, reverse, crossing_exclusions
    )
    boundaries = _verify_temporal_recording_boundaries(cache, reverse, boundaries)
    reverse["fold"] = 1
    reverse["training_seed"] = 42
    reverse["analysis_mode"] = "sensitivity"
    reverse["confirmatory_protocol"] = False
    reverse["temporal_direction"] = "late_train_to_early_test"
    boundaries = boundaries.copy()
    boundaries["diagnostic_protocol"] = REVERSE_TEMPORAL_PROTOCOL
    frame = _expand_assignments(
        cache,
        reverse.drop(columns=["dataset", "label_binary"], errors="ignore"),
        protocol=REVERSE_TEMPORAL_PROTOCOL,
        cohort="date_eligible_project_target",
    )
    summary = _split_summary(
        reverse.sort_values("participant_timestamp_utc").reset_index(drop=True)
    )
    manifest = _finalize_temporal_manifest(
        frame,
        scientific_fingerprint=science,
        eligibility_fingerprint=alignment,
        eligibility_audit=eligibility_audit,
        date_audit=date_audit,
        split_summary=summary,
        analysis_mode="sensitivity",
        boundary_diagnostics=boundaries,
    )
    return manifest


def build_external_hst_manifest(
    cache_index: pd.DataFrame,
    source_manifest: pd.DataFrame,
    *,
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
    eligibility_mapping: pd.DataFrame | None = None,
    eligibility_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Attach each eligible COUGHVID cough cohort to every source fold."""
    _require_columns(
        source_manifest,
        {
            "dataset",
            "participant_key",
            "recording_key",
            "modality",
            "label_binary",
            "split",
            "fold",
            "protocol",
        }
        | _EXTERNAL_PROVENANCE_COLUMNS,
        "source_manifest",
    )
    _validate_qualified_keys(source_manifest)
    if source_manifest.empty:
        raise ValueError("source_manifest is empty")
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    _verify_track_a_source(source_manifest, science)
    source_hash = _manifest_digest(source_manifest)

    _require_columns(cache_index, {"dataset", "modality"}, "cache_index")
    target_rows = cache_index[
        cache_index["dataset"].astype(str).eq("coughvid")
        & cache_index["modality"].astype(str).eq("cough")
    ].copy()
    target_cache = _eligible_cache(target_rows)
    if target_cache.empty:
        raise ValueError("No eligible COUGHVID cough rows are available")
    target_cache["analysis_unit_key"] = target_cache.apply(
        lambda row: _analysis_unit_key(row["recording_key"], row["modality"]),
        axis=1,
    )
    if eligibility_mapping is None:
        raise ValueError("A frozen representation eligibility mapping is required")
    parent_alignment_values = set(
        eligibility_mapping["eligibility_alignment_fingerprint"].astype(str)
    )
    if len(parent_alignment_values) != 1:
        raise ValueError("Parent eligibility mapping has ambiguous provenance")
    if eligibility_fingerprint is not None and next(
        iter(parent_alignment_values)
    ) != str(eligibility_fingerprint).strip().casefold():
        raise ValueError("supplied eligibility fingerprint does not verify")
    target_mapping = _restrict_eligibility_mapping(
        eligibility_mapping,
        target_cache,
        restriction_reason="coughvid_cough_external_target_only",
        selection_label="selected target analysis",
    )
    target, target_alignment, target_eligibility_audit = _apply_eligibility_alignment(
        target_cache,
        scientific_fingerprint=science,
        eligibility_mapping=target_mapping,
        eligibility_fingerprint=None,
    )
    _require_columns(target, _EXTERNAL_PROVENANCE_COLUMNS, "COUGHVID cache")
    for column in _EXTERNAL_PROVENANCE_COLUMNS:
        values = target[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"COUGHVID cache has empty {column} provenance")
    target["cohort"] = (
        "coughvid::"
        + target["label_source"].astype(str)
        + "::"
        + target["preprocessing_variant"].astype(str)
    )
    for _, cohort in target.groupby("cohort", sort=False):
        _participant_table(cohort)

    signature_columns = ["representation_id", "preprocessing_hash"]
    source_signature = set(
        map(tuple, source_manifest[signature_columns].astype(str).to_numpy())
    )
    target_signature = set(
        map(tuple, target[signature_columns].astype(str).to_numpy())
    )
    if source_signature != target_signature:
        raise ValueError(
            "Source/target representation and preprocessing signatures do not match"
        )

    source_keys = set(source_manifest["participant_key"].astype(str))
    target_keys = set(target["participant_key"].astype(str))
    if not source_keys.isdisjoint(target_keys):
        raise ValueError("Source and target qualified participant keys collide")

    context_columns = [
        column
        for column in ("fold", "split_seed", "training_seed")
        if column in source_manifest
    ]
    contexts = source_manifest[context_columns].drop_duplicates().sort_values(
        context_columns, kind="mergesort"
    )
    target_frames: list[pd.DataFrame] = []
    for context in contexts.to_dict(orient="records"):
        current = target.copy()
        for column, value in context.items():
            current[column] = value
        current["split"] = "external_test"
        target_frames.append(current)

    source = _without_hash_columns(source_manifest).copy()
    audit_columns = [
        column
        for column in source
        if column.endswith("_payload_json")
        or (
            column.endswith("_sha256")
            and f"{column[:-7]}_payload_json" in source.columns
        )
    ]
    source = source.drop(columns=audit_columns)
    source_alignment_values = set(
        source_manifest["eligibility_alignment_fingerprint"].astype(str)
    )
    if len(source_alignment_values) != 1:
        raise ValueError("Coswara Track-A source has ambiguous eligibility provenance")
    source_alignment = next(iter(source_alignment_values))
    source["source_protocol"] = source["protocol"]
    source["protocol"] = EXTERNAL_PROTOCOL
    source["source_fold_manifest_sha256"] = source_hash
    external = pd.concat(target_frames, ignore_index=True)
    external["source_protocol"] = TRACK_A_PROTOCOL
    external["protocol"] = EXTERNAL_PROTOCOL
    external["source_fold_manifest_sha256"] = source_hash
    manifest = pd.concat([source, external], ignore_index=True, sort=False)
    manifest["scientific_configuration_fingerprint"] = science
    manifest["source_eligibility_alignment_fingerprint"] = source_alignment
    manifest["target_eligibility_alignment_fingerprint"] = target_alignment
    manifest = _stamp_analysis_provenance(
        manifest,
        analysis_scope="reliability_evaluation",
        analysis_role="secondary",
        estimand_id="coswara_to_coughvid_external_transfer",
        multiplicity_family="prespecified_reliability",
        analysis_mode="confirmatory",
        confirmatory_protocol=True,
    )
    for column in _EXTERNAL_PROVENANCE_COLUMNS:
        values = manifest[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"External manifest has empty {column} provenance")
    if not manifest.loc[manifest["dataset"].eq("coughvid"), "split"].eq(
        "external_test"
    ).all():
        raise AssertionError("An external target row entered a source split")
    eligibility_payload = {
        "source_alignment_fingerprint": source_alignment,
        "target_alignment_fingerprint": target_alignment,
        "target_alignment_audit": target_eligibility_audit,
    }
    manifest = _embed_audit_payload(
        manifest, "eligibility_audit", eligibility_payload
    )
    _assert_no_participant_leakage(manifest)
    return _finalize_manifest(manifest)


def _verify_track_a_source(
    source_manifest: pd.DataFrame, scientific_fingerprint: str
) -> None:
    _verify_frozen_frame(source_manifest, "source_manifest")
    if set(source_manifest["protocol"].astype(str)) != {TRACK_A_PROTOCOL}:
        raise ValueError("External source must be a frozen Coswara Track-A manifest")
    if set(source_manifest["dataset"].astype(str)) != {"coswara"}:
        raise ValueError("External source must be a frozen Coswara Track-A manifest")
    if set(source_manifest["modality"].astype(str)) != {"cough"}:
        raise ValueError("External source must be a cough-only Coswara Track-A manifest")
    if not source_manifest["split"].isin(_SPLITS).all():
        raise ValueError("Coswara Track-A source contains invalid split labels")
    _require_columns(
        source_manifest,
        {
            "fold",
            "split_seed",
            "training_seed",
            "test_fraction",
            "validation_fraction_of_remaining",
            "nominal_split_ratio",
            "split_fraction_semantics",
            "realized_train_participant_count",
            "realized_validation_participant_count",
            "realized_test_participant_count",
            "realized_train_fraction",
            "realized_validation_fraction",
            "realized_test_fraction",
            "cohort",
            "seed_provenance",
            "evaluation_design",
            "scientific_configuration_fingerprint",
            "eligibility_alignment_fingerprint",
        }
        | _ANALYSIS_PROVENANCE_COLUMNS,
        "Coswara Track-A source",
    )
    if set(source_manifest["cohort"].astype(str)) != {
        "project_target_all_eligible"
    } or set(source_manifest["seed_provenance"].astype(str)) != {
        "released_hst_baseline_scripts"
    } or set(source_manifest["evaluation_design"].astype(str)) != {
        "ten_repeated_stratified_participant_holdouts"
    }:
        raise ValueError("External source must be the primary Coswara Track-A cohort")
    expected_analysis_provenance = {
        "analysis_scope": "internal_performance",
        "analysis_role": "primary",
        "estimand_id": "track_a_internal_hst_vs_aligned_comparator",
        "multiplicity_family": "primary_internal_performance",
        "analysis_mode": "confirmatory",
    }
    for column, expected in expected_analysis_provenance.items():
        if set(source_manifest[column].astype(str)) != {expected}:
            raise ValueError("External source must preserve primary Track-A provenance")
    if not source_manifest["confirmatory_protocol"].eq(True).all():
        raise ValueError("External source must preserve primary Track-A provenance")
    if set(source_manifest["scientific_configuration_fingerprint"].astype(str)) != {
        scientific_fingerprint
    }:
        raise ValueError("Coswara Track-A scientific configuration does not match")
    observed_folds = sorted(source_manifest["fold"].astype(int).unique().tolist())
    if observed_folds != list(range(1, len(PRESPECIFIED_HST_REPO_SEEDS) + 1)):
        raise ValueError("Coswara Track-A must contain all prescribed folds")
    source_people = _participant_table(source_manifest)
    source_labels = source_people["label_binary"].map(_CLASS_TO_INDEX).to_numpy()
    for fold, seed in enumerate(PRESPECIFIED_HST_REPO_SEEDS, start=1):
        current = source_manifest[source_manifest["fold"].astype(int).eq(fold)]
        if set(current["split"].astype(str)) != set(_SPLITS):
            raise ValueError("Coswara Track-A fold has an invalid split structure")
        if set(current["split_seed"].astype(int)) != {seed} or set(
            current["training_seed"].astype(int)
        ) != {seed}:
            raise ValueError("Coswara Track-A fold/seed provenance does not match")
        if not np.allclose(current["test_fraction"].astype(float), 0.2) or not np.allclose(
            current["validation_fraction_of_remaining"].astype(float), 0.125
        ):
            raise ValueError(
                "Coswara Track-A split parameters are not nominal 70/10/20"
            )
        if set(current["nominal_split_ratio"].astype(str)) != {"70/10/20"} or set(
            current["split_fraction_semantics"].astype(str)
        ) != {"nominal_parameters_with_sklearn_realized_counts"}:
            raise ValueError("Coswara Track-A nominal split provenance does not match")
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_validation_index, test_index = next(
            outer.split(source_people, source_labels)
        )
        train_validation = source_people.iloc[train_validation_index].reset_index(
            drop=True
        )
        train_validation_labels = source_labels[train_validation_index]
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        train_index, validation_index = next(
            inner.split(train_validation, train_validation_labels)
        )
        expected = {
            "train": set(
                train_validation.iloc[train_index]["participant_key"].astype(str)
            ),
            "validation": set(
                train_validation.iloc[validation_index]["participant_key"].astype(str)
            ),
            "test": set(source_people.iloc[test_index]["participant_key"].astype(str)),
        }
        realized_counts = {split: len(keys) for split, keys in expected.items()}
        total_people = len(source_people)
        for split, count in realized_counts.items():
            count_column = f"realized_{split}_participant_count"
            fraction_column = f"realized_{split}_fraction"
            if set(current[count_column].astype(int)) != {count} or not np.allclose(
                current[fraction_column].astype(float), count / total_people
            ):
                raise ValueError("Coswara Track-A realized split provenance does not match")
        for split, expected_keys in expected.items():
            observed_keys = set(
                current.loc[current["split"].eq(split), "participant_key"].astype(str)
            )
            if observed_keys != expected_keys:
                raise ValueError(
                    "Coswara Track-A fold does not match the prescribed split structure"
                )
    _assert_no_participant_leakage(source_manifest)
    audit = audit_hst_manifest(source_manifest)
    required_zero = (
        "participant_overlap_count",
        "content_hash_leakage_count",
        "unpaired_duplicate_representation_count",
        "analysis_weight_violation_count",
        "invalid_audit_payload_count",
    )
    if any(not audit[column].eq(0).all() for column in required_zero):
        raise ValueError("Coswara Track-A source failed its integrity audit")


def intersect_representation_eligibility(
    *indices: pd.DataFrame,
    scientific_config: object | None = None,
    scientific_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Return the exact eligible recording/modality intersection across caches.

    A single cache is the identity case used before any optional aligned
    representation comparator is introduced. It retains the same frozen
    eligibility and exclusion provenance without claiming that rows are paired.
    """
    if not indices:
        raise ValueError("At least one representation index is required")
    science = _resolve_scientific_fingerprint(
        scientific_config, scientific_fingerprint
    )
    eligible: list[pd.DataFrame] = []
    complete: list[pd.DataFrame] = []
    representation_ids: list[str] = []
    identity = ["recording_key", "modality"]
    for position, index in enumerate(indices):
        current = _eligible_cache(index)
        representations = sorted(current["representation_id"].astype(str).unique())
        if len(representations) != 1:
            raise ValueError(
                f"Representation index {position} must contain exactly one representation_id"
            )
        if current.duplicated(identity).any():
            raise ValueError(f"Representation {representations[0]} duplicates {identity}")
        eligible.append(current)
        complete.append(index.copy())
        representation_ids.append(representations[0])
    if len(set(representation_ids)) != len(representation_ids):
        raise ValueError("Representation indices must have distinct representation_id values")

    key_sets = {
        representation_id: set(map(tuple, current[identity].to_numpy()))
        for representation_id, current in zip(representation_ids, eligible)
    }
    common = set.intersection(*key_sets.values())
    if not common:
        raise ValueError("Representations have no shared eligible recording/modality rows")
    union = set.union(*key_sets.values())
    aligned_frames: list[pd.DataFrame] = []
    for current in eligible:
        current = current.copy()
        current["_identity"] = list(map(tuple, current[identity].to_numpy()))
        aligned_frames.append(
            current[current["_identity"].isin(common)].drop(columns="_identity")
        )
    result = pd.concat(aligned_frames, ignore_index=True, sort=False)
    consistency = result.groupby(identity, dropna=False)[
        ["dataset", "participant_key", "label_binary"]
    ].nunique(dropna=False)
    if consistency.gt(1).any().any():
        raise ValueError(
            "Representations disagree on dataset, participant, or label"
        )
    _assert_representation_source_audio_consistency(
        result, name="representation intersection"
    )
    sorted_representations = sorted(representation_ids)
    result["representation_count"] = len(sorted_representations)
    result["representation_ids"] = "|".join(sorted_representations)
    result["analysis_unit_key"] = result.apply(
        lambda row: _analysis_unit_key(row["recording_key"], row["modality"]),
        axis=1,
    )
    result["paired_representation_count"] = len(sorted_representations)
    result["paired_representation"] = len(sorted_representations) > 1
    result["analysis_unit_weight"] = 1.0 / len(sorted_representations)
    result["scientific_configuration_fingerprint"] = science
    result["eligibility_mapping_policy"] = _SHARED_INTERSECTION_POLICY
    alignment = _alignment_fingerprint(result)
    result["eligibility_alignment_fingerprint"] = alignment
    exclusion_frames: list[pd.DataFrame] = []
    for representation_id, index in zip(representation_ids, complete):
        audited = index.copy()
        audited["_identity"] = list(map(tuple, audited[identity].to_numpy()))
        audited["representation_id"] = representation_id
        reason = pd.Series("representation_ineligible", index=audited.index, dtype="string")
        eligible_mask = audited["eligible"].astype(bool)
        shared_mask = audited["_identity"].isin(common)
        reason.loc[eligible_mask & shared_mask] = "included_shared_intersection"
        reason.loc[eligible_mask & ~shared_mask] = (
            "not_in_shared_representation_intersection"
        )
        if "reason" in audited:
            supplied_reason = audited["reason"].astype("string").str.strip()
            useful_reason = supplied_reason.notna() & supplied_reason.ne("")
            reason.loc[~eligible_mask & useful_reason] = supplied_reason.loc[
                ~eligible_mask & useful_reason
            ]
        audited["exclusion_reason"] = reason
        audited["_recording_modality_key"] = (
            audited["recording_key"].astype(str)
            + "||"
            + audited["modality"].astype(str)
        )
        dimensions = [
            column
            for column in (
                "dataset",
                "protocol",
                "split",
                "label_binary",
                "modality",
                "representation_id",
                "exclusion_reason",
            )
            if column in audited
        ]
        grouped = (
            audited.groupby(dimensions, dropna=False)
            .agg(
                recording_count=("_recording_modality_key", "nunique"),
                participant_count=("participant_key", "nunique"),
            )
            .reset_index()
        )
        exclusion_frames.append(grouped)
    exclusions = pd.concat(exclusion_frames, ignore_index=True, sort=False)
    exclusion_sort = [
        column
        for column in (
            "dataset",
            "protocol",
            "split",
            "label_binary",
            "modality",
            "representation_id",
            "exclusion_reason",
        )
        if column in exclusions
    ]
    exclusions = exclusions.sort_values(
        exclusion_sort, kind="mergesort"
    ).reset_index(drop=True)
    retained_unit_keys = sorted(
        _analysis_unit_key(recording_key, modality)
        for recording_key, modality in common
    )
    unit_dispositions = {
        representation_id: {
            "input_eligible_analysis_unit_keys": sorted(
                _analysis_unit_key(recording_key, modality)
                for recording_key, modality in key_sets[representation_id]
            ),
            "retained_analysis_unit_keys": retained_unit_keys,
            "excluded_analysis_unit_keys": sorted(
                _analysis_unit_key(recording_key, modality)
                for recording_key, modality in (
                    key_sets[representation_id] - common
                )
            ),
        }
        for representation_id in sorted_representations
    }
    audit_payload = {
        "mapping_policy_id": _SHARED_INTERSECTION_POLICY,
        "alignment_fingerprint": alignment,
        "representations": sorted_representations,
        "shared_recording_modality_count": len(common),
        "union_recording_modality_count": len(union),
        "representation_unit_dispositions": unit_dispositions,
        "exclusions": _frame_payload(exclusions),
    }
    result = _stamp_analysis_provenance(
        result,
        analysis_scope="representation_alignment",
        analysis_role="design_context",
        estimand_id="shared_representation_eligibility",
        multiplicity_family="not_applicable",
        analysis_mode="design",
        confirmatory_protocol=False,
    )
    result = _embed_audit_payload(result, "eligibility_audit", audit_payload)
    frozen = _finalize_manifest(result)
    frozen.attrs["representation_exclusions"] = exclusions.copy()
    return frozen


def audit_hst_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize participant leakage, label consistency, and content integrity."""
    _require_columns(
        manifest,
        {
            "dataset",
            "participant_key",
            "recording_key",
            "modality",
            "label_binary",
            "split",
        },
        "manifest",
    )
    _validate_qualified_keys(manifest)
    stored_manifest_hashes = (
        set(manifest["manifest_sha256"].astype(str))
        if "manifest_sha256" in manifest
        else set()
    )
    calculated_manifest_hash = _manifest_digest(manifest)
    manifest_hash_valid = stored_manifest_hashes == {calculated_manifest_hash}
    if "row_content_sha256" in manifest:
        valid_row_hash = manifest["row_content_sha256"].astype(str).eq(
            _row_hashes(manifest)
        )
    else:
        valid_row_hash = pd.Series(False, index=manifest.index)

    invalid_audit_payload_count = 0
    for payload_column in sorted(
        column for column in manifest if column.endswith("_payload_json")
    ):
        prefix = payload_column[: -len("_payload_json")]
        hash_column = f"{prefix}_sha256"
        if hash_column not in manifest:
            invalid_audit_payload_count += 1
            continue
        hashes = set(manifest[hash_column].astype(str))
        payloads = manifest.loc[
            manifest[payload_column].astype(str).ne(""), payload_column
        ]
        if len(hashes) != 1 or len(payloads) != 1:
            invalid_audit_payload_count += 1
            continue
        try:
            payload = json.loads(str(payloads.iloc[0]))
        except json.JSONDecodeError:
            invalid_audit_payload_count += 1
            continue
        if _content_digest(payload) != next(iter(hashes)):
            invalid_audit_payload_count += 1

    rows: list[dict[str, object]] = []
    group_columns = _audit_group_columns(manifest)
    for group_key, unit in manifest.groupby(group_columns, dropna=False, sort=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(group_columns, keys))
        label_counts = unit.groupby("participant_key")["label_binary"].nunique()
        split_counts = unit.groupby("participant_key")["split"].nunique()
        mixed_count = int(label_counts.gt(1).sum())
        multiple_split_count = int(split_counts.gt(1).sum())
        content_leakage_count = 0
        for content_column in _CONTENT_HASH_COLUMNS:
            if content_column not in unit:
                continue
            content = unit.loc[
                unit[content_column].astype("string").notna()
                & unit[content_column].astype(str).ne("")
            ]
            content_leakage_count += int(
                content.groupby(content_column)["split"].nunique().gt(1).sum()
            )
        representation_identity = ["recording_key", "modality", "split"]
        duplicate_groups = unit.groupby(representation_identity, dropna=False).size()
        duplicated_keys = duplicate_groups[duplicate_groups.gt(1)].index
        unpaired_duplicate_count = 0
        for duplicate_key in duplicated_keys:
            key_values = (
                duplicate_key if isinstance(duplicate_key, tuple) else (duplicate_key,)
            )
            mask = pd.Series(True, index=unit.index)
            for column, value in zip(representation_identity, key_values):
                mask &= unit[column].eq(value)
            duplicate_rows = unit.loc[mask]
            valid_pair = (
                "paired_representation" in duplicate_rows
                and duplicate_rows["paired_representation"].eq(True).all()
                and "analysis_unit_key" in duplicate_rows
                and duplicate_rows["analysis_unit_key"].nunique() == 1
            )
            if not valid_pair:
                unpaired_duplicate_count += 1
        analysis_weight_violation_count = 0
        if {"analysis_unit_key", "analysis_unit_weight"}.issubset(unit.columns):
            weights = unit.groupby(["split", "analysis_unit_key"], dropna=False)[
                "analysis_unit_weight"
            ].sum()
            analysis_weight_violation_count = int(
                (~np.isclose(weights.to_numpy(dtype=float), 1.0, atol=1e-12)).sum()
            )
        row: dict[str, object] = {
            **identity,
            "n_rows": int(len(unit)),
            "n_participants": int(unit["participant_key"].nunique()),
            "participant_overlap_count": multiple_split_count,
            "multiple_split_participant_count": multiple_split_count,
            "mixed_label_participant_count": mixed_count,
            "duplicate_recording_count": unpaired_duplicate_count,
            "unpaired_duplicate_representation_count": unpaired_duplicate_count,
            "analysis_weight_violation_count": analysis_weight_violation_count,
            "content_hash_leakage_count": content_leakage_count,
            "invalid_audit_payload_count": invalid_audit_payload_count,
            "manifest_hash_valid": bool(manifest_hash_valid),
            "invalid_row_hash_count": int((~valid_row_hash.loc[unit.index]).sum()),
            "calculated_manifest_sha256": calculated_manifest_hash,
        }
        for split in ("train", "validation", "test", "external_test"):
            current = unit[unit["split"].eq(split)]
            people = current[["participant_key", "label_binary"]].drop_duplicates()
            row[f"n_{split}_participants"] = int(len(people))
            row[f"n_{split}_positive"] = int(people["label_binary"].eq("positive").sum())
            row[f"n_{split}_negative"] = int(people["label_binary"].eq("negative").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _strip_non_scientific(value: object) -> object:
    canonical = _canonicalize(value)
    if isinstance(canonical, dict):
        result: dict[str, object] = {}
        for key, item in canonical.items():
            if key in _NON_SCIENTIFIC_CONFIG_KEYS:
                continue
            if key == "protocol":
                if not isinstance(item, dict):
                    continue
                item = {
                    nested_key: nested_item
                    for nested_key, nested_item in item.items()
                    if nested_key not in {"label", "protocol_label", "protocol_name"}
                }
            result[key] = item
        return result
    return canonical


def _scientific_section_has_content(value: object) -> bool:
    canonical = _canonicalize(value)
    if canonical is None:
        return False
    if isinstance(canonical, str):
        return bool(canonical.strip())
    if isinstance(canonical, dict):
        return bool(canonical) and any(
            _scientific_section_has_content(item) for item in canonical.values()
        )
    if isinstance(canonical, list):
        return bool(canonical) and any(
            _scientific_section_has_content(item) for item in canonical
        )
    return True


def scientific_configuration_fingerprint(config: object) -> str:
    """Hash scientific choices while excluding artifact and protocol-label identity."""
    canonical = _canonicalize(config)
    if not isinstance(canonical, dict):
        raise TypeError("Scientific configuration must be a mapping or dataclass-like object")
    required = {
        "architecture",
        "source_checkpoint",
        "preprocessing",
        "augmentation",
        "optimizer",
        "stopping_rule",
        "participant_aggregation",
        "fusion",
        "thresholding",
        "sampling",
        "calibration",
    }
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Scientific configuration missing sections: {missing}")
    for section in sorted(required):
        if not _scientific_section_has_content(canonical[section]):
            raise ValueError(f"Scientific configuration section {section} cannot be empty")
    metric_sections = [
        section for section in ("metrics", "metric_settings") if section in canonical
    ]
    if not metric_sections:
        raise ValueError("Scientific configuration requires metrics or metric_settings")
    for section in metric_sections:
        if not _scientific_section_has_content(canonical[section]):
            raise ValueError(
                f"Scientific configuration section {section} cannot be empty"
            )
    return _content_digest(_strip_non_scientific(canonical))
