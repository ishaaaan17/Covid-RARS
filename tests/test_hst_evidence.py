from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_complete_run(tmp_path: Path) -> Path:
    from covid_rars.hst_runtime import canonical_json_sha256

    run_root = tmp_path / "hst-run"
    metrics = run_root / "metrics" / "internal.csv"
    figure = run_root / "figures" / "performance.svg"
    metrics.parent.mkdir(parents=True)
    figure.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "protocol": ["internal"],
            "model_name": ["hst_base"],
            "metric_split": ["test"],
            "auroc": [0.8],
            "analysis_scope": ["confirmatory"],
        }
    ).to_csv(metrics, index=False)
    figure.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
    stage_root = run_root / "runtime" / "stages"
    stage_root.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "receipt_type": "hst_stage",
        "run_id": "hst-run",
        "stage": "evidence_pack",
        "status": "success",
        "fingerprint": "a" * 64,
        "output_paths": ["metrics/internal.csv", "figures/performance.svg"],
        "output_checksums": {
            "metrics/internal.csv": _sha256(metrics),
            "figures/performance.svg": _sha256(figure),
        },
    }
    receipt["record_hash"] = canonical_json_sha256(receipt)
    (stage_root / "evidence_pack.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    return run_root


def test_evidence_manifest_rejects_tampered_stage_receipt(tmp_path: Path) -> None:
    from covid_rars.hst_evidence import build_hst_evidence_manifest

    run_root = _fake_complete_run(tmp_path)
    receipt_path = run_root / "runtime" / "stages" / "evidence_pack.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "success"
    receipt["output_paths"] = ["metrics/internal.csv"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="record checksum"):
        build_hst_evidence_manifest(
            run_root=run_root,
            output_path=run_root / "evidence" / "manifest.json",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("receipt_type", "launch", "receipt type"),
        ("run_id", "different-run", "run identity"),
        ("stage", "different-stage", "filename"),
    ],
)
def test_evidence_manifest_binds_receipt_to_current_run_and_filename(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    from covid_rars.hst_evidence import build_hst_evidence_manifest
    from covid_rars.hst_runtime import canonical_json_sha256

    run_root = _fake_complete_run(tmp_path)
    receipt_path = run_root / "runtime" / "stages" / "evidence_pack.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_hst_evidence_manifest(
            run_root=run_root,
            output_path=run_root / "evidence" / "manifest.json",
        )


def test_evidence_manifest_accepts_only_receipted_checksum_valid_outputs(
    tmp_path: Path,
) -> None:
    from covid_rars.hst_evidence import build_hst_evidence_manifest

    run_root = _fake_complete_run(tmp_path)
    output = run_root / "evidence" / "hst_evidence_manifest.json"
    manifest = build_hst_evidence_manifest(run_root=run_root, output_path=output)

    assert manifest["run_id"] == "hst-run"
    assert manifest["artifact_count"] == 2
    assert {row["role"] for row in manifest["artifacts"]} == {"metric", "figure"}
    assert output.is_file()


def test_evidence_manifest_rejects_tampered_stage_output(tmp_path: Path) -> None:
    from covid_rars.hst_evidence import build_hst_evidence_manifest

    run_root = _fake_complete_run(tmp_path)
    (run_root / "metrics" / "internal.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        build_hst_evidence_manifest(
            run_root=run_root,
            output_path=run_root / "evidence" / "manifest.json",
        )


def test_combined_registration_preserves_every_existing_metric_row(
    tmp_path: Path,
) -> None:
    from covid_rars.hst_evidence import register_hst_metrics

    existing = pd.DataFrame({"model_name": ["old_a", "old_b"], "auroc": [0.7, 0.8]})
    hst = pd.DataFrame({"model_name": ["hst_base"], "auroc": [0.9]})
    combined = register_hst_metrics(existing=existing, hst_metrics=[hst])

    assert combined["model_name"].tolist() == ["old_a", "old_b", "hst_base"]
    pd.testing.assert_frame_equal(
        combined.iloc[: len(existing)][existing.columns].reset_index(drop=True),
        existing,
        check_dtype=False,
    )


def test_evidence_pack_cli_builds_manifest_from_receipts(tmp_path: Path) -> None:
    run_root = _fake_complete_run(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    output = run_root / "evidence" / "manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "73_make_hst_evidence_pack.py"),
            "--run-root",
            str(run_root),
            "--output",
            str(output),
            "--required-stage",
            "evidence_pack",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["artifact_count"] == 2


def test_registration_cli_preserves_prior_manifest_and_metrics(tmp_path: Path) -> None:
    from covid_rars.hst_evidence import build_hst_evidence_manifest

    run_root = _fake_complete_run(tmp_path)
    hst_manifest_path = run_root / "evidence" / "hst_evidence_manifest.json"
    build_hst_evidence_manifest(
        run_root=run_root,
        output_path=hst_manifest_path,
        required_stages=["evidence_pack"],
    )
    existing_metrics = tmp_path / "paper_metric_table.csv"
    pd.DataFrame({"model_name": ["old"], "auroc": [0.7]}).to_csv(
        existing_metrics,
        index=False,
    )
    existing_manifest = tmp_path / "experiment_manifest.json"
    existing_manifest.write_text(
        json.dumps({"schema_version": 9, "prior_key": "must-remain"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "combined"
    project_root = Path(__file__).resolve().parents[1]

    figure = run_root / "figures" / "performance.svg"
    original_figure = figure.read_bytes()
    figure.write_bytes(b"tampered\n")
    rejected = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "74_register_hst_evidence.py"),
            "--run-root",
            str(run_root),
            "--hst-manifest",
            str(hst_manifest_path),
            "--existing-metrics",
            str(existing_metrics),
            "--existing-manifest",
            str(existing_manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "artifact checksum mismatch" in rejected.stderr.casefold()
    figure.write_bytes(original_figure)

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "74_register_hst_evidence.py"),
            "--run-root",
            str(run_root),
            "--hst-manifest",
            str(hst_manifest_path),
            "--existing-metrics",
            str(existing_metrics),
            "--existing-manifest",
            str(existing_manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    latest_path = output_dir / "latest.json"
    assert latest_path.is_file()
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    generation = output_dir / latest["generation_path"]
    assert not Path(latest["generation_path"]).is_absolute()
    combined = pd.read_csv(generation / "combined_paper_metric_table.csv")
    assert combined["model_name"].tolist() == ["old", "hst_base"]
    manifest = json.loads(
        (generation / "combined_experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["prior_key"] == "must-remain"
    assert manifest["hst_runs"][0]["run_id"] == "hst-run"
    assert not Path(manifest["hst_runs"][0]["evidence_manifest_path"]).is_absolute()
    assert manifest["combined_paper_metric_table"]["path"] == (
        "combined_paper_metric_table.csv"
    )

    combined_path = generation / "combined_paper_metric_table.csv"
    original = combined_path.read_bytes()
    combined_path.write_bytes(b"tampered\n")
    repeated = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "74_register_hst_evidence.py"),
            "--run-root",
            str(run_root),
            "--hst-manifest",
            str(hst_manifest_path),
            "--existing-metrics",
            str(existing_metrics),
            "--existing-manifest",
            str(existing_manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr.casefold()
    assert combined_path.read_bytes() != original


def test_latest_publication_requires_successful_receipted_evidence(
    tmp_path: Path,
) -> None:
    from covid_rars.hst_evidence import (
        build_hst_evidence_manifest,
        publish_hst_latest,
    )
    from covid_rars.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _fake_complete_run(tmp_path)
    manifest_path = run_root / "evidence" / "hst_evidence_manifest.json"
    build_hst_evidence_manifest(run_root=run_root, output_path=manifest_path)
    receipt_path = run_root / "runtime" / "stages" / "evidence_pack.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    relative = manifest_path.relative_to(run_root).as_posix()
    receipt["output_paths"].append(relative)
    receipt["output_checksums"][relative] = stable_file_sha256(manifest_path)
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    latest_path = tmp_path / "reports" / "hst" / "latest.json"
    latest = publish_hst_latest(
        run_root=run_root,
        evidence_manifest_path=manifest_path,
        latest_path=latest_path,
    )
    assert latest["run_id"] == run_root.name
    assert latest["evidence_manifest_sha256"] == stable_file_sha256(manifest_path)
    assert not Path(latest["run_root"]).is_absolute()
    assert latest["run_root_path_base"] == "latest_pointer_directory"
    assert latest["evidence_manifest_path"] == (
        manifest_path.relative_to(run_root).as_posix()
    )
    assert latest["evidence_manifest_path_base"] == "run_root"
    assert json.loads(latest_path.read_text(encoding="utf-8")) == latest

    receipt["status"] = "failed"
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="successful evidence_pack"):
        publish_hst_latest(
            run_root=run_root,
            evidence_manifest_path=manifest_path,
            latest_path=latest_path,
        )


def test_latest_publication_rehashes_every_manifest_artifact(tmp_path: Path) -> None:
    from covid_rars.hst_evidence import (
        build_hst_evidence_manifest,
        publish_hst_latest,
    )
    from covid_rars.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _fake_complete_run(tmp_path)
    manifest_path = run_root / "evidence" / "hst_evidence_manifest.json"
    build_hst_evidence_manifest(run_root=run_root, output_path=manifest_path)
    receipt_path = run_root / "runtime" / "stages" / "evidence_pack.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    relative = manifest_path.relative_to(run_root).as_posix()
    receipt["output_paths"].append(relative)
    receipt["output_checksums"][relative] = stable_file_sha256(manifest_path)
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (run_root / "figures" / "performance.svg").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        publish_hst_latest(
            run_root=run_root,
            evidence_manifest_path=manifest_path,
            latest_path=tmp_path / "reports" / "hst" / "latest.json",
        )
