from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pandas as pd

from .hst_data_contracts import normalize_coughvid_status


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hst_coughvid_metadata(
    *,
    cohort_path: Path,
    raw_metadata_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Bind the frozen processed cohort to COUGHVID-v3 labels by UUID."""
    cohort_path = Path(cohort_path).resolve()
    raw_metadata_path = Path(raw_metadata_path).resolve()
    output_path = Path(output_path).resolve()
    if not cohort_path.is_file():
        raise FileNotFoundError(cohort_path)
    if not raw_metadata_path.is_file():
        raise FileNotFoundError(raw_metadata_path)
    if output_path in {cohort_path, raw_metadata_path}:
        raise ValueError("HST COUGHVID metadata output must not replace an input")

    cohort = pd.read_csv(cohort_path, low_memory=False)
    raw = pd.read_csv(raw_metadata_path, low_memory=False)
    required_cohort = {
        "participant_id",
        "recording_id",
        "audio_path",
        "label_binary",
    }
    required_raw = {"uuid", "status_SSL", "status"}
    missing_cohort = sorted(required_cohort - set(cohort.columns))
    missing_raw = sorted(required_raw - set(raw.columns))
    if missing_cohort:
        raise ValueError(f"Processed COUGHVID cohort is missing columns: {missing_cohort}")
    if missing_raw:
        raise ValueError(f"Raw COUGHVID metadata is missing columns: {missing_raw}")
    if cohort["participant_id"].astype(str).duplicated().any():
        raise ValueError("Processed COUGHVID participant/UUID values are not unique")
    if raw["uuid"].astype(str).duplicated().any():
        raise ValueError("Raw COUGHVID UUID values are not unique")

    raw = raw.drop(
        columns=[column for column in raw if str(column).startswith("Unnamed:")],
        errors="ignore",
    )
    overlap = sorted(set(cohort.columns) & set(raw.columns))
    if overlap:
        raise ValueError(f"COUGHVID cohort/raw columns collide before binding: {overlap}")
    cohort = cohort.rename(columns={"label_binary": "legacy_label_binary"})
    cohort["_hst_cohort_order"] = range(len(cohort))
    bound = cohort.merge(
        raw,
        left_on="participant_id",
        right_on="uuid",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not bound["_merge"].eq("both").all():
        missing = int(bound["_merge"].ne("both").sum())
        raise ValueError(f"Processed COUGHVID cohort has {missing} UUIDs absent from raw metadata")
    if not bound["_hst_cohort_order"].eq(range(len(bound))).all():
        raise RuntimeError("COUGHVID UUID binding changed the processed cohort order")
    bound = bound.drop(columns=["_merge", "_hst_cohort_order"])
    if bound["status_SSL"].isna().any():
        raise ValueError("The processed COUGHVID cohort contains missing status_SSL labels")

    legacy = bound["legacy_label_binary"].map(normalize_coughvid_status)
    ssl = bound["status_SSL"].map(normalize_coughvid_status)
    supervised = {"negative", "positive"}
    if not ssl.isin(supervised).all():
        raise ValueError("The processed COUGHVID cohort contains non-binary status_SSL labels")
    disagreement = legacy.ne(ssl)
    if disagreement.any():
        raise ValueError(
            "Processed COUGHVID labels disagree with status_SSL for "
            f"{int(disagreement.sum())} rows"
        )
    missing_audio = bound["audio_path"].map(lambda value: not Path(str(value)).is_file())
    if missing_audio.any():
        raise FileNotFoundError(
            f"Processed COUGHVID cohort has {int(missing_audio.sum())} missing audio files"
        )

    cohort_hash = _sha256_file(cohort_path)
    raw_hash = _sha256_file(raw_metadata_path)
    bound["cohort_source_sha256"] = cohort_hash
    bound["label_metadata_source_sha256"] = raw_hash
    bound["label_join_contract"] = "processed_participant_id_equals_coughvid_v3_uuid"
    bound["hst_metadata_schema_version"] = "hst-coughvid-label-binding-v1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        bound.to_csv(temporary, index=False, lineterminator="\n")
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "schema_version": 1,
        "output_path": output_path.as_posix(),
        "output_sha256": _sha256_file(output_path),
        "row_count": int(len(bound)),
        "cohort_source_sha256": cohort_hash,
        "label_metadata_source_sha256": raw_hash,
        "status_ssl_negative": int(ssl.eq("negative").sum()),
        "status_ssl_positive": int(ssl.eq("positive").sum()),
        "legacy_label_disagreement_count": int(disagreement.sum()),
        "missing_audio_count": int(missing_audio.sum()),
    }
