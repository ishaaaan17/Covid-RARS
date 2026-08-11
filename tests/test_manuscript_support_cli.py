from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "42_manuscript_support_analyses.py"
    spec = importlib.util.spec_from_file_location("manuscript_support_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_project(root: Path) -> None:
    metadata_rows = []
    for split, n in [("train", 12), ("validation", 6), ("test", 6)]:
        for idx in range(n):
            positive = idx % 2 == 0
            metadata_rows.append(
                {
                    "participant_id": f"{split}_{idx}",
                    "recording_id": f"rec_{split}_{idx}",
                    "dataset": "coswara",
                    "split": split,
                    "label_binary": "positive" if positive else "negative",
                    "age": 50 if positive else 28,
                    "gender": "male" if positive else "female",
                    "country": "India" if positive else "Canada",
                    "recording_date": "2021-05-01" if positive else "2020-04-01",
                    "duration_sec": 8.0 if positive else 4.0,
                    "sample_rate_original": 48000 if positive else 44100,
                    "quality_flag": "ok" if positive else "short",
                    "symptoms_json": "{}",
                    "comorbidities_json": "{}",
                }
            )
    metadata_rows.append(
        {
            "participant_id": "unknown",
            "recording_id": "rec_unknown",
            "dataset": "coswara",
            "split": "unused",
            "label_binary": "unknown",
            "age": 65,
            "gender": "male",
            "country": "India",
            "recording_date": "2022-01-01",
            "duration_sec": 3.0,
            "sample_rate_original": 16000,
            "quality_flag": "mostly_silence",
            "symptoms_json": "{}",
            "comorbidities_json": "{}",
        }
    )

    processed = root / "data" / "processed"
    metrics = root / "data" / "outputs" / "metrics"
    tables = root / "reports" / "tables"
    processed.mkdir(parents=True)
    metrics.mkdir(parents=True)
    tables.mkdir(parents=True)
    pd.DataFrame(metadata_rows).to_csv(processed / "metadata_clean.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["age", "country_India"],
            "before_abs_smd": [0.5, 0.8],
            "after_abs_smd": [0.2, 0.724],
            "weight_config": ["ipw_cap_2_q_0.95", "ipw_cap_2_q_0.95"],
            "control_method": ["ipw_label_propensity", "ipw_label_propensity"],
        }
    ).to_csv(tables / "ipw_sensitivity_balance.csv", index=False)
    for name, auroc, auprc in [
        ("external_model_grid_metrics.csv", 0.534, 0.042),
        ("external_model_grid_opensmile_egemaps_metrics.csv", 0.552, 0.039),
        ("external_model_grid_beats_metrics.csv", 0.553, 0.039),
        ("external_model_grid_panns_metrics.csv", 0.502, 0.035),
    ]:
        pd.DataFrame({"model_name": ["lr"], "feature_strategy": ["all"], "auroc": [auroc], "auprc": [auprc]}).to_csv(metrics / name, index=False)
    pd.DataFrame(
        {
            "recording_id": [f"ext_{idx}" for idx in range(10)],
            "label_binary": ["positive"] + ["negative"] * 9,
        }
    ).to_csv(metrics / "external_model_grid_beats_predictions.csv", index=False)


def test_manuscript_support_cli_writes_expected_outputs(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    _write_synthetic_project(tmp_path)
    output_dir = tmp_path / "out"
    argv = [
        "42_manuscript_support_analyses.py",
        "--project-root",
        str(tmp_path),
        "--shap-output",
        str(output_dir / "shap.csv"),
        "--ipw-output",
        str(output_dir / "ipw.csv"),
        "--auprc-output",
        str(output_dir / "auprc.csv"),
        "--unknown-summary-output",
        str(output_dir / "unknown_summary.csv"),
        "--unknown-balance-output",
        str(output_dir / "unknown_balance.csv"),
        "--summary-output",
        str(output_dir / "summary.md"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    for filename in ["shap.csv", "ipw.csv", "auprc.csv", "unknown_summary.csv", "unknown_balance.csv", "summary.md"]:
        path = output_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0

    assert pd.read_csv(output_dir / "ipw.csv").iloc[0]["feature"] == "country_India"
    assert "External AUPRC Lift" in (output_dir / "summary.md").read_text()
