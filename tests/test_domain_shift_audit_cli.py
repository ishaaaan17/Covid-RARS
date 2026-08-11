from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def test_domain_shift_cli_writes_metrics_predictions_and_importance(tmp_path, monkeypatch) -> None:
    source = pd.DataFrame(
        {
            "recording_id": [f"s{i}" for i in range(12)],
            "participant_id": [f"ps{i}" for i in range(12)],
            "dataset": "coswara",
            "modality": "cough",
            "label_binary": ["positive", "negative"] * 6,
            "feat": [0.0, 0.1, 0.2, 0.1, 0.0, 0.2] * 2,
        }
    )
    external = pd.DataFrame(
        {
            "recording_id": [f"e{i}" for i in range(12)],
            "participant_id": [f"pe{i}" for i in range(12)],
            "dataset": "coughvid",
            "modality": "cough",
            "label_binary": ["negative", "positive"] * 6,
            "feat": [3.0, 3.1, 3.2, 3.1, 3.0, 3.2] * 2,
        }
    )
    source_path = tmp_path / "source.csv"
    external_path = tmp_path / "external.csv"
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    importance_path = tmp_path / "importance.csv"
    source.to_csv(source_path, index=False)
    external.to_csv(external_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "37_domain_shift_audit.py"
    spec = importlib.util.spec_from_file_location("domain_shift_audit_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "37_domain_shift_audit.py",
            "--source-features",
            str(source_path),
            "--external-features",
            str(external_path),
            "--representation",
            "toy",
            "--metrics-output",
            str(metrics_path),
            "--predictions-output",
            str(predictions_path),
            "--importance-output",
            str(importance_path),
            "--test-size",
            "0.5",
        ],
    )

    module.main()

    assert pd.read_csv(metrics_path).loc[0, "representation"] == "toy"
    assert "probability_external" in pd.read_csv(predictions_path).columns
    assert pd.read_csv(importance_path).loc[0, "feature"] == "feat"
