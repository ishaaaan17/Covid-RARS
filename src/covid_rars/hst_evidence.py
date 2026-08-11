from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Iterable

import pandas as pd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_role(relative: str) -> str:
    path = Path(relative)
    token = relative.casefold()
    if path.suffix.casefold() in {".svg", ".png"}:
        return "figure"
    if "prediction" in token:
        return "prediction"
    if "metric" in token:
        return "metric"
    if path.suffix.casefold() in {".pt", ".pth", ".pkl", ".joblib"}:
        return "model"
    if "manifest" in token:
        return "manifest"
    if "audit" in token:
        return "audit"
    return "artifact"


def _validate_metric_table(path: Path) -> int:
    frame = pd.read_csv(path)
    required = {"model_name", "auroc", "analysis_scope"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"HST metric table {path.name} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"HST metric table {path.name} is empty")
    allowed_scope = {"confirmatory", "secondary", "exploratory", "sensitivity"}
    if not frame["analysis_scope"].astype(str).isin(allowed_scope).all():
        raise ValueError(f"HST metric table {path.name} has invalid analysis_scope")
    return int(len(frame))


def _validate_stage_receipt_identity(
    *,
    receipt: dict[str, object],
    receipt_path: Path,
    run_root: Path,
) -> str:
    if receipt.get("receipt_type") != "hst_stage":
        raise ValueError(f"Invalid HST stage receipt type: {receipt_path}")
    if receipt.get("run_id") != run_root.name:
        raise ValueError(f"HST stage receipt has the wrong run identity: {receipt_path}")
    stage = str(receipt.get("stage", ""))
    if not stage or stage != receipt_path.stem:
        raise ValueError(
            f"HST stage receipt stage does not match its filename: {receipt_path}"
        )
    return stage


def validate_hst_manifest_artifacts(
    *,
    run_root: Path,
    manifest: dict[str, object],
) -> None:
    """Revalidate every manifest artifact against the current filesystem state."""
    run_root = Path(run_root).resolve(strict=True)
    if manifest.get("run_id") != run_root.name:
        raise ValueError("HST evidence manifest has the wrong run identity")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("HST evidence manifest has no artifact list")
    if manifest.get("artifact_count") != len(artifacts):
        raise ValueError("HST evidence manifest artifact count mismatch")

    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("HST evidence manifest contains an invalid artifact")
        relative = Path(str(artifact.get("path", ""))).as_posix()
        if not relative or relative in seen:
            raise ValueError(f"Invalid or duplicate HST artifact path: {relative!r}")
        seen.add(relative)
        resolved = (run_root / relative).resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError as exc:
            raise ValueError(f"HST artifact escapes the run root: {relative}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual = _sha256_file(resolved)
        if actual != artifact.get("sha256"):
            raise ValueError(f"HST artifact checksum mismatch: {relative}")
        if resolved.stat().st_size != artifact.get("size_bytes"):
            raise ValueError(f"HST artifact size mismatch: {relative}")
        if artifact.get("role") == "metric":
            row_count = _validate_metric_table(resolved)
            if row_count != artifact.get("row_count"):
                raise ValueError(f"HST metric artifact row-count mismatch: {relative}")


def build_hst_evidence_manifest(
    *,
    run_root: Path,
    output_path: Path,
    required_stages: Iterable[str] | None = None,
) -> dict[str, object]:
    run_root = Path(run_root).resolve()
    output_path = Path(output_path).resolve()
    try:
        output_path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("HST evidence manifest must be written inside the run root") from exc
    stage_root = run_root / "runtime" / "stages"
    receipts: dict[str, dict[str, object]] = {}
    for receipt_path in sorted(stage_root.glob("*.json")):
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid HST stage receipt: {receipt_path}")
        claimed_record_hash = payload.get("record_hash")
        unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
        actual_record_hash = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if claimed_record_hash != actual_record_hash:
            raise ValueError(f"HST stage receipt record checksum mismatch: {receipt_path}")
        stage = _validate_stage_receipt_identity(
            receipt=payload,
            receipt_path=receipt_path,
            run_root=run_root,
        )
        if stage in receipts:
            raise ValueError(f"Duplicate HST stage identity: {receipt_path}")
        if payload.get("status") != "success":
            raise ValueError(f"HST stage {stage!r} is not successful")
        receipts[stage] = payload
    required = set(required_stages or ())
    missing_stages = sorted(required - set(receipts))
    if missing_stages:
        raise ValueError(f"HST evidence is missing successful stages: {missing_stages}")
    if not receipts:
        raise ValueError("HST evidence has no successful stage receipts")

    artifacts: dict[str, dict[str, object]] = {}
    for stage, receipt in sorted(receipts.items()):
        paths = receipt.get("output_paths")
        checksums = receipt.get("output_checksums")
        if not isinstance(paths, list) or not isinstance(checksums, dict):
            raise ValueError(f"HST stage {stage!r} has no auditable output contract")
        for value in paths:
            relative = Path(str(value)).as_posix()
            resolved = (run_root / relative).resolve()
            try:
                resolved.relative_to(run_root)
            except ValueError as exc:
                raise ValueError(f"HST stage output escapes the run root: {relative}") from exc
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            expected = checksums.get(relative)
            actual = _sha256_file(resolved)
            if expected != actual:
                raise ValueError(
                    f"HST stage output checksum mismatch for {relative}: "
                    f"expected {expected}, found {actual}"
                )
            role = _artifact_role(relative)
            row_count = _validate_metric_table(resolved) if role == "metric" else None
            prior = artifacts.get(relative)
            if prior is not None and prior["sha256"] != actual:
                raise ValueError(f"Conflicting receipted HST artifact: {relative}")
            artifacts[relative] = {
                "path": relative,
                "role": role,
                "sha256": actual,
                "size_bytes": resolved.stat().st_size,
                "row_count": row_count,
                "producer_stages": sorted(
                    set((prior or {}).get("producer_stages", [])) | {stage}
                ),
            }
    run_id = run_root.name
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "artifact_count": len(artifacts),
        "stage_count": len(receipts),
        "stages": sorted(receipts),
        "artifacts": [artifacts[key] for key in sorted(artifacts)],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json(output_path, manifest)
    return manifest


def register_hst_metrics(
    *,
    existing: pd.DataFrame,
    hst_metrics: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    frames = [existing.copy(), *(frame.copy() for frame in hst_metrics)]
    if any(frame.empty for frame in frames[1:]):
        raise ValueError("HST metric registration cannot include an empty metric table")
    columns = list(existing.columns)
    for frame in frames[1:]:
        columns.extend(column for column in frame.columns if column not in columns)
    return pd.concat(
        [frame.reindex(columns=columns) for frame in frames],
        ignore_index=True,
        sort=False,
    )


def publish_hst_latest(
    *,
    run_root: Path,
    evidence_manifest_path: Path,
    latest_path: Path,
) -> dict[str, object]:
    """Atomically publish a run only after its evidence stage is receipted."""
    run_root = Path(run_root).resolve(strict=True)
    manifest_path = Path(evidence_manifest_path).resolve(strict=True)
    try:
        manifest_relative = manifest_path.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ValueError("HST evidence manifest must be inside its run root") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("HST evidence manifest must be a JSON object")
    claimed_manifest_hash = manifest.get("manifest_sha256")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    actual_manifest_hash = hashlib.sha256(
        json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if claimed_manifest_hash != actual_manifest_hash:
        raise ValueError("HST evidence manifest record checksum mismatch")
    validate_hst_manifest_artifacts(run_root=run_root, manifest=manifest)

    receipt_path = run_root / "runtime" / "stages" / "evidence_pack.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("status") != "success":
        raise ValueError("A successful evidence_pack receipt is required for publication")
    _validate_stage_receipt_identity(
        receipt=receipt,
        receipt_path=receipt_path,
        run_root=run_root,
    )
    claimed_receipt_hash = receipt.get("record_hash")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "record_hash"
    }
    actual_receipt_hash = hashlib.sha256(
        json.dumps(unsigned_receipt, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if claimed_receipt_hash != actual_receipt_hash:
        raise ValueError("HST evidence_pack receipt record checksum mismatch")
    paths = receipt.get("output_paths")
    checksums = receipt.get("output_checksums")
    manifest_file_hash = _sha256_file(manifest_path)
    if (
        not isinstance(paths, list)
        or manifest_relative not in paths
        or not isinstance(checksums, dict)
        or checksums.get(manifest_relative) != manifest_file_hash
    ):
        raise ValueError("HST evidence manifest is not checksummed by evidence_pack")

    latest_path = Path(latest_path).resolve()
    relative_run_root = Path(
        os.path.relpath(run_root, start=latest_path.parent)
    ).as_posix()
    latest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_root.name,
        "run_root": relative_run_root,
        "run_root_path_base": "latest_pointer_directory",
        "evidence_manifest_path": manifest_relative,
        "evidence_manifest_path_base": "run_root",
        "evidence_manifest_sha256": manifest_file_hash,
        "evidence_manifest_record_hash": actual_manifest_hash,
        "evidence_pack_receipt_sha256": _sha256_file(receipt_path),
        "evidence_pack_receipt_record_hash": actual_receipt_hash,
    }
    latest["record_hash"] = hashlib.sha256(
        json.dumps(latest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(latest_path, latest)
    return latest
