from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _frame(dataset: str, split: str, offset: float) -> pd.DataFrame:
    labels = ["positive", "negative"] * 4
    return pd.DataFrame(
        {
            "recording_id": [f"{dataset}_{split}_{idx}" for idx in range(len(labels))],
            "participant_id": [f"{dataset}_p_{idx}" for idx in range(len(labels))],
            "dataset": dataset,
            "modality": "cough",
            "split": split,
            "label_binary": labels,
            "feat_0": [offset + (1 if label == "positive" else -1) for label in labels],
            "feat_1": [offset * 0.2 + idx * 0.1 for idx in range(len(labels))],
        }
    )


def test_domain_adaptation_cli_writes_metrics_predictions_and_mmd(tmp_path, monkeypatch) -> None:
    source = pd.concat(
        [
            _frame("coswara", "train", 0.0),
            _frame("coswara", "validation", 0.2),
        ],
        ignore_index=True,
    )
    external = _frame("coughvid", "external", 2.0)

    source_path = tmp_path / "source.csv"
    external_path = tmp_path / "external.csv"
    metrics_path = tmp_path / "domain_adaptation_metrics.csv"
    predictions_path = tmp_path / "domain_adaptation_predictions.csv"
    mmd_path = tmp_path / "domain_adaptation_mmd.csv"
    source.to_csv(source_path, index=False)
    external.to_csv(external_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "41_domain_adaptation_baseline.py"
    spec = importlib.util.spec_from_file_location("domain_adaptation_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "41_domain_adaptation_baseline.py",
            "--source-features",
            str(source_path),
            "--external-features",
            str(external_path),
            "--metrics-output",
            str(metrics_path),
            "--predictions-output",
            str(predictions_path),
            "--mmd-output",
            str(mmd_path),
            "--models",
            "logistic_regression",
            "--feature-strategies",
            "all",
            "--representation",
            "mfcc",
            "--n-mmd-samples",
            "100",
        ],
    )

    module.main()

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    mmd = pd.read_csv(mmd_path)
    assert set(metrics["adaptation_method"]) == {"source_only", "coral"}
    assert set(predictions["adaptation_method"]) == {"source_only", "coral"}
    assert {"mmd_before", "mmd_after", "mmd_reduction"}.issubset(metrics.columns)
    assert len(mmd) == 1
    assert mmd["representation"].iloc[0] == "mfcc"

def test_domain_adaptation_cli_discovers_default_representation_runs(tmp_path) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "41_domain_adaptation_baseline.py"
    spec = importlib.util.spec_from_file_location("domain_adaptation_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "features_beats_coswara_cough.csv").write_text("x\n1\n")
    (tmp_path / "data" / "processed" / "features_beats_coughvid_cough.csv").write_text("x\n2\n")
    (tmp_path / "reports" / "tables" / "feature_shift_beats_cough.csv").write_text("feature,abs_standardized_mean_difference\nx,0.1\n")

    runs = module.default_representation_runs(project_root=tmp_path)

    assert len(runs) == 1
    assert runs[0].representation == "beats"
    assert runs[0].source_features.name == "features_beats_coswara_cough.csv"
    assert runs[0].external_features.name == "features_beats_coughvid_cough.csv"
