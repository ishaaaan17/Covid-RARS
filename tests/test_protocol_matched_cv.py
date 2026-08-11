from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _feature_frame(n_participants: int = 48, rows_per_participant: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant_idx in range(n_participants):
        label = "positive" if participant_idx % 4 in {1, 2} else "negative"
        y = 1.0 if label == "positive" else -1.0
        for row_idx in range(rows_per_participant):
            idx = participant_idx * rows_per_participant + row_idx
            rows.append(
                {
                    "recording_id": f"rec_{idx:03d}",
                    "participant_id": f"p_{participant_idx:03d}",
                    "dataset": "coswara",
                    "modality": "cough",
                    "submodality": "heavy_cough",
                    "label_binary": label,
                    "quality_flag": "ok",
                    "train_signal": y + idx * 0.001,
                    "weak_signal": y * 0.1 + (idx % 5) * 0.01,
                    "noise": (idx % 7) * 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_protocol_matched_cv_keeps_validation_and_test_participants_disjoint() -> None:
    from covid_rars.protocol_matched_cv import run_protocol_matched_cv

    result = run_protocol_matched_cv(
        _feature_frame(),
        modality="cough",
        n_splits=3,
        validation_fraction=0.25,
        top_k_values=[2],
        ranker="univariate",
        model_names=["logistic_l2_f80"],
        random_state=0,
    )

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert set(result.predictions["fold_unit"]) == {"participant"}
    assert set(result.metrics["fold_unit"].dropna()) == {"participant"}
    assert set(result.split_audit["overlap_count"]) == {0}
    aggregate = result.metrics[result.metrics["metric_split"].eq("test_aggregate")]
    assert len(aggregate) == 1
    assert aggregate.iloc[0]["evaluation_protocol"] == "protocol_matched_participant_10fold_cv"


def test_protocol_matched_cv_uses_hst_style_70_10_20_participant_ratios() -> None:
    from covid_rars.protocol_matched_cv import run_protocol_matched_cv

    result = run_protocol_matched_cv(
        _feature_frame(n_participants=50, rows_per_participant=1),
        modality="cough",
        n_splits=10,
        validation_fraction=0.125,
        test_fraction=0.2,
        top_k_values=[2],
        ranker="univariate",
        model_names=["logistic_l2_f80"],
        random_state=0,
    )

    assert len(result.split_audit) == 10
    assert set(result.split_audit["n_train_participants"]) == {35.0}
    assert set(result.split_audit["n_validation_participants"]) == {5.0}
    assert set(result.split_audit["n_test_participants"]) == {10.0}
    assert set(result.split_audit["overlap_count"]) == {0}


def test_protocol_matched_cv_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    features_path = tmp_path / "features.csv"
    _feature_frame().to_csv(features_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "69_run_protocol_matched_cv.py"
    spec = importlib.util.spec_from_file_location("protocol_matched_cv_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics_output = tmp_path / "metrics.csv"
    predictions_output = tmp_path / "predictions.csv"
    selection_output = tmp_path / "selection.csv"
    split_audit_output = tmp_path / "split_audit.csv"
    argv = [
        "69_run_protocol_matched_cv.py",
        "--features",
        str(features_path),
        "--n-splits",
        "3",
        "--validation-fraction",
        "0.25",
        "--top-k-values",
        "2",
        "--ranker",
        "univariate",
        "--model-names",
        "logistic_l2_f80",
        "--metrics-output",
        str(metrics_output),
        "--predictions-output",
        str(predictions_output),
        "--feature-selection-output",
        str(selection_output),
        "--split-audit-output",
        str(split_audit_output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    metrics = pd.read_csv(metrics_output)
    split_audit = pd.read_csv(split_audit_output)
    assert "test_aggregate" in set(metrics["metric_split"])
    assert set(split_audit["overlap_count"]) == {0}
