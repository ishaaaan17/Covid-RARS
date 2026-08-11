from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def test_ipw_sensitivity_cli_writes_all_outputs(tmp_path, monkeypatch) -> None:
    metadata = pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(12)],
            "recording_id": [f"r{i}" for i in range(12)],
            "label_binary": ["positive"] * 6 + ["negative"] * 6,
            "split": "test",
            "country": ["India"] * 5 + ["Canada"] + ["India"] + ["Canada"] * 5,
            "age": [50] * 6 + [30] * 6,
            "gender": ["male", "female"] * 6,
        }
    )
    predictions = pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(12)],
            "label_binary": ["positive"] * 6 + ["negative"] * 6,
            "split": "test",
            "probability": [0.8] * 6 + [0.2] * 6,
            "fusion_method": "quality_weighted_auprc",
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    balance_path = tmp_path / "balance.csv"
    weights_path = tmp_path / "weights.csv"
    metadata.to_csv(metadata_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "38_ipw_sensitivity_analysis.py"
    spec = importlib.util.spec_from_file_location("ipw_sensitivity_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "38_ipw_sensitivity_analysis.py",
            "--predictions",
            str(predictions_path),
            "--metadata",
            str(metadata_path),
            "--metrics-output",
            str(metrics_path),
            "--balance-output",
            str(balance_path),
            "--weights-output",
            str(weights_path),
            "--covariates",
            "country",
            "age",
            "gender",
            "--weight-caps",
            "2",
            "5",
            "--clip-quantiles",
            "1.0",
        ],
    )

    module.main()

    assert set(pd.read_csv(metrics_path)["weight_cap"].dropna()) == {2.0, 5.0}
    assert "after_abs_smd" in pd.read_csv(balance_path).columns
    assert "ipw_weight" in pd.read_csv(weights_path).columns
