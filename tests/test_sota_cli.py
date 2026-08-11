from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_with_audio(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    sr = 16000
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    y = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for idx, split in enumerate(["train", "validation", "test"]):
        path = tmp_path / f"audio_{idx}.wav"
        sf.write(path, y, sr)
        rows.append(
            {
                "recording_id": f"r{idx}",
                "participant_id": f"p{idx}",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": "positive" if idx != 1 else "negative",
                "split": split,
                "quality_flag": "ok",
                "audio_path": str(path),
            }
        )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return metadata


def test_build_sota_segment_index_cli_writes_index_and_audit(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata_with_audio(tmp_path)
    index_output = tmp_path / "segments.csv"
    audit_output = tmp_path / "audit.csv"
    script = Path(__file__).parents[1] / "scripts" / "50_build_sota_segment_index.py"
    module = _load_script(script, "build_sota_segment_index_cli")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "50_build_sota_segment_index.py",
            "--metadata",
            str(metadata),
            "--output",
            str(index_output),
            "--audit-output",
            str(audit_output),
            "--modalities",
            "cough",
            "--window-sec",
            "1.0",
            "--augment-train-copies",
            "1",
        ],
    )
    module.main()

    index = pd.read_csv(index_output)
    audit = pd.read_csv(audit_output)
    assert not index.empty
    assert not audit.empty
    assert index[index["is_augmented"].astype(bool)]["split"].eq("train").all()


def test_fuse_sota_predictions_cli_writes_metrics_and_predictions(tmp_path: Path, monkeypatch) -> None:
    from covid_rars.sota_predictions import aggregate_sota_predictions, evaluate_sota_prediction_table

    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        for idx in range(10):
            label = "positive" if idx % 2 else "negative"
            probability = 0.75 if label == "positive" else 0.25
            for model_name in ("ssl", "spec"):
                rows.append(
                    {
                        "recording_id": f"{split}_{idx}",
                        "participant_id": f"{split}_p{idx}",
                        "dataset": "coswara",
                        "modality": "cough",
                        "submodality": "cough",
                        "label_binary": label,
                        "split": split,
                        "model_name": model_name,
                        "analysis_family": "sota_ssl_branch",
                        "evaluation_protocol": "sota_internal_protocol",
                        "probability": probability,
                    }
                )
    predictions = aggregate_sota_predictions(pd.DataFrame(rows), level="participant")
    metrics = evaluate_sota_prediction_table(predictions)
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    output_metrics = tmp_path / "fused_metrics.csv"
    output_predictions = tmp_path / "fused_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "52_fuse_sota_predictions.py"
    module = _load_script(script, "fuse_sota_predictions_cli")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "52_fuse_sota_predictions.py",
            "--metrics",
            str(metrics_path),
            "--predictions",
            str(predictions_path),
            "--metrics-output",
            str(output_metrics),
            "--predictions-output",
            str(output_predictions),
            "--top-k",
            "2",
        ],
    )
    module.main()

    assert pd.read_csv(output_metrics)["analysis_family"].eq("sota_prediction_fusion").all()
    assert not pd.read_csv(output_predictions).empty


def test_train_sota_ssl_branch_cli_debug_backend_writes_branch_outputs(tmp_path: Path, monkeypatch) -> None:
    from covid_rars.sota_segments import build_sota_segment_index

    metadata = pd.read_csv(_metadata_with_audio(tmp_path))
    segment_index = build_sota_segment_index(metadata, modalities=["cough"], window_sec=1.0)
    segment_index_path = tmp_path / "segments.csv"
    metrics_output = tmp_path / "ssl_metrics.csv"
    predictions_output = tmp_path / "ssl_predictions.csv"
    history_output = tmp_path / "ssl_history.csv"
    segment_index.to_csv(segment_index_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "51_train_sota_ssl_branch.py"
    module = _load_script(script, "train_sota_ssl_branch_cli")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "51_train_sota_ssl_branch.py",
            "--segment-index",
            str(segment_index_path),
            "--modality",
            "cough",
            "--backend",
            "debug_acoustic",
            "--metrics-output",
            str(metrics_output),
            "--predictions-output",
            str(predictions_output),
            "--history-output",
            str(history_output),
        ],
    )
    module.main()

    assert not pd.read_csv(metrics_output).empty
    assert not pd.read_csv(predictions_output).empty
    assert pd.read_csv(history_output)["backend"].eq("debug_acoustic").all()
