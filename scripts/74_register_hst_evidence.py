#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_rars.hst_evidence import (
    register_hst_metrics,
    validate_hst_manifest_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.17g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _relative_portable(path: Path, *, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=start.resolve())).as_posix()


def _generation_id(
    *,
    hst_manifest_path: Path,
    hst_manifest_record_hash: object,
    existing_metrics_path: Path,
    existing_manifest_path: Path,
) -> str:
    identity = {
        "schema_version": 1,
        "hst_manifest_sha256": _sha256(hst_manifest_path),
        "hst_manifest_record_hash": str(hst_manifest_record_hash),
        "existing_metrics_sha256": _sha256(existing_metrics_path),
        "existing_manifest_sha256": _sha256(existing_manifest_path),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return f"registration-{hashlib.sha256(canonical).hexdigest()[:24]}"


def _publish_generation(*, staging: Path, generation: Path) -> None:
    expected_names = (
        "combined_paper_metric_table.csv",
        "combined_experiment_manifest.json",
    )
    generation.parent.mkdir(parents=True, exist_ok=True)
    if generation.exists():
        unchanged = generation.is_dir() and all(
            (generation / name).is_file()
            and _sha256(generation / name) == _sha256(staging / name)
            for name in expected_names
        )
        if not unchanged:
            raise FileExistsError(
                f"Refusing to overwrite non-identical evidence generation: {generation}"
            )
        shutil.rmtree(staging)
        return
    try:
        os.rename(staging, generation)
    except FileExistsError:
        _publish_generation(staging=staging, generation=generation)
        return
    _fsync_directory(generation.parent)


def _load_validated_hst_metrics(
    *,
    run_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, object], list[pd.DataFrame]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("HST evidence manifest must be a JSON object")
    claimed = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_manifest_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed != actual_manifest_hash:
        raise ValueError("HST evidence manifest checksum mismatch")
    validate_hst_manifest_artifacts(run_root=run_root, manifest=manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("HST evidence manifest has no artifact list")
    metric_frames: list[pd.DataFrame] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("role") != "metric":
            continue
        path = (run_root / str(artifact.get("path", ""))).resolve()
        metric_frames.append(pd.read_csv(path, low_memory=False))
    if not metric_frames:
        raise ValueError("HST evidence manifest contains no metric tables")
    return manifest, metric_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register checksum-validated HST evidence without overwriting prior reports."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hst-manifest", type=Path, required=True)
    parser.add_argument("--existing-metrics", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    hst_manifest_path = args.hst_manifest.resolve()
    existing_metrics_path = args.existing_metrics.resolve()
    existing_manifest_path = args.existing_manifest.resolve()
    output_dir = args.output_dir.resolve()
    hst_manifest, hst_metrics = _load_validated_hst_metrics(
        run_root=run_root,
        manifest_path=hst_manifest_path,
    )
    existing_metrics = pd.read_csv(existing_metrics_path, low_memory=False)
    combined = register_hst_metrics(existing=existing_metrics, hst_metrics=hst_metrics)
    generation_id = _generation_id(
        hst_manifest_path=hst_manifest_path,
        hst_manifest_record_hash=hst_manifest["manifest_sha256"],
        existing_metrics_path=existing_metrics_path,
        existing_manifest_path=existing_manifest_path,
    )
    generations_root = output_dir / "generations"
    generation = generations_root / generation_id
    staging = output_dir / f".staging-{generation_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    output_metrics = staging / "combined_paper_metric_table.csv"
    output_manifest = staging / "combined_experiment_manifest.json"
    try:
        _atomic_csv(combined, output_metrics)

        prior_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(prior_manifest, dict):
            raise ValueError("Existing experiment manifest must be a JSON object")
        combined_manifest = dict(prior_manifest)
        hst_runs = list(combined_manifest.get("hst_runs", []))
        hst_runs.append(
            {
                "run_id": hst_manifest["run_id"],
                "evidence_manifest_path": _relative_portable(
                    hst_manifest_path,
                    start=output_dir,
                ),
                "evidence_manifest_path_base": "registration_output_dir",
                "evidence_manifest_sha256": _sha256(hst_manifest_path),
                "evidence_manifest_record_hash": hst_manifest["manifest_sha256"],
                "metric_rows": int(sum(len(frame) for frame in hst_metrics)),
            }
        )
        combined_manifest["hst_runs"] = hst_runs
        combined_manifest["combined_paper_metric_table"] = {
            "path": "combined_paper_metric_table.csv",
            "path_base": "combined_manifest_directory",
            "sha256": _sha256(output_metrics),
            "rows": int(len(combined)),
            "prior_rows": int(len(existing_metrics)),
        }
        combined_manifest["registration_generation_id"] = generation_id
        _atomic_json(combined_manifest, output_manifest)
        _fsync_directory(staging)
        _publish_generation(staging=staging, generation=generation)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    published_metrics = generation / output_metrics.name
    published_manifest = generation / output_manifest.name
    latest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "generation_path": generation.relative_to(output_dir).as_posix(),
        "combined_paper_metric_table_sha256": _sha256(published_metrics),
        "combined_experiment_manifest_sha256": _sha256(published_manifest),
    }
    _atomic_json(latest, output_dir / "latest.json")
    print(published_metrics)
    print(published_manifest)


if __name__ == "__main__":
    main()
