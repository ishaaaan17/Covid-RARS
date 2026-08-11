from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _feature_frame(n_rows: int = 72) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(n_rows):
        label = "positive" if idx % 4 in {1, 2} else "negative"
        y = 1.0 if label == "positive" else -1.0
        rows.append(
            {
                "recording_id": f"rec_{idx:03d}",
                "participant_id": f"p_{idx // 2:03d}",
                "dataset": "coswara",
                "modality": "cough" if idx < n_rows - 6 else "breath",
                "submodality": "heavy_cough",
                "label_binary": label,
                "split": "unused",
                "quality_flag": "ok",
                "train_signal": y + idx * 0.001,
                "weak_signal": y * 0.1 + (idx % 5) * 0.01,
                "noise": (idx % 7) * 0.02,
            }
        )
    return pd.DataFrame(rows)


def test_fold_feature_selection_uses_inner_training_rows_only() -> None:
    from covid_rars.paper_comparable_cv import select_fold_feature_columns

    frame = _feature_frame(30)
    frame["split"] = ["train"] * 20 + ["validation"] * 5 + ["test"] * 5
    frame["test_only_leak"] = 0.0
    test_mask = frame["split"].eq("test")
    frame.loc[test_mask, "test_only_leak"] = frame.loc[test_mask, "label_binary"].map(
        {"positive": 1.0, "negative": -1.0}
    )

    selected, ranking = select_fold_feature_columns(
        frame,
        k=1,
        ranker="univariate",
        random_state=0,
    )

    assert selected == ["train_signal"]
    leak = ranking[ranking["feature"].eq("test_only_leak")]
    assert not leak.empty
    assert float(leak.iloc[0]["importance"]) == 0.0
    assert ranking["selection_split"].eq("train").all()


def test_paper_comparable_cv_reports_fold_and_aggregate_metrics() -> None:
    from covid_rars.paper_comparable_cv import run_paper_comparable_cv

    features = _feature_frame(72)
    result = run_paper_comparable_cv(
        features,
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
    test_rows = result.metrics[result.metrics["metric_split"].eq("test")]
    aggregate = result.metrics[result.metrics["metric_split"].eq("test_aggregate")]
    assert len(test_rows) == 3
    assert len(aggregate) == 1
    assert aggregate.iloc[0]["evaluation_protocol"] == "paper_comparable_10fold_cv"
    assert aggregate.iloc[0]["fold_unit"] == "recording"
    assert aggregate.iloc[0]["feature_strategy"] == "compare_is10_top2_univariate"
    assert "auroc_std" in aggregate.columns
    assert result.feature_selection["fold"].nunique() == 3


def test_paper_comparable_cv_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    features = _feature_frame(72)
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "57_run_paper_comparable_cv.py"
    spec = importlib.util.spec_from_file_location("paper_comparable_cv_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics_output = tmp_path / "metrics.csv"
    predictions_output = tmp_path / "predictions.csv"
    selection_output = tmp_path / "selection.csv"
    argv = [
        "57_run_paper_comparable_cv.py",
        "--features",
        str(features_path),
        "--n-splits",
        "3",
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
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert metrics_output.exists()
    assert predictions_output.exists()
    assert selection_output.exists()
    metrics = pd.read_csv(metrics_output)
    assert "test_aggregate" in set(metrics["metric_split"])
