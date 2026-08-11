from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _metadata_frame(n_participants: int = 48) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(n_participants):
        label = "positive" if idx % 4 in {1, 2} else "negative"
        split = "train" if idx % 6 < 4 else "validation" if idx % 6 == 4 else "test"
        for modality in ("cough", "breath", "speech"):
            rows.append(
                {
                    "recording_id": f"rec_{idx:03d}_{modality}",
                    "participant_id": f"p_{idx:03d}",
                    "dataset": "coswara",
                    "modality": modality,
                    "submodality": modality,
                    "label_raw": "positive_mild" if label == "positive" else "healthy",
                    "label_binary": label,
                    "label_group": "positive_mild" if label == "positive" else "healthy",
                    "split": split,
                    "quality_flag": "ok",
                }
            )
    return pd.DataFrame(rows)


def _feature_table(metadata: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        idx = int(str(row["participant_id"]).split("_")[-1])
        y = 1.0 if row["label_binary"] == "positive" else -1.0
        is_test = row["split"] == "test"
        rows.append(
            {
                "recording_id": row["recording_id"],
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "modality": row["modality"],
                "submodality": row["submodality"],
                "label_binary": row["label_binary"],
                "split": row["split"],
                f"{prefix}_train_signal": y + idx * 0.001,
                f"{prefix}_weak_signal": (y * 0.1) + (idx % 3) * 0.01,
                f"{prefix}_test_only_leak": y if is_test else 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_merge_feature_tables_prefixes_features_and_keeps_recording_schema() -> None:
    from covid_rars.compare_is10_rescue import merge_feature_tables

    metadata = _metadata_frame(12)
    strong = _feature_table(metadata, "strong_raw")
    compare = _feature_table(metadata, "compare_raw")
    is10 = _feature_table(metadata, "is10_raw")

    merged = merge_feature_tables(
        {
            "strong": strong,
            "compare2016": compare,
            "is10": is10,
        }
    )

    assert len(merged) == len(metadata)
    assert {"recording_id", "participant_id", "modality", "label_binary", "split"}.issubset(merged.columns)
    assert "strong__strong_raw_train_signal" in merged.columns
    assert "compare2016__compare_raw_train_signal" in merged.columns
    assert "is10__is10_raw_train_signal" in merged.columns
    assert "strong_raw_train_signal" not in merged.columns


def test_lightweight_top_k_selection_uses_training_rows_only() -> None:
    from covid_rars.compare_is10_rescue import merge_feature_tables, select_top_k_feature_tables

    metadata = _metadata_frame(48)
    merged = merge_feature_tables({"strong": _feature_table(metadata, "strong_raw")})

    result = select_top_k_feature_tables(
        merged,
        k_values=[1, 2],
        ranker="univariate",
        random_state=0,
    )

    selected_k1 = result.tables[1]
    assert "strong__strong_raw_train_signal" in selected_k1.columns
    assert "strong__strong_raw_test_only_leak" not in selected_k1.columns
    assert result.summary["selection_split"].eq("train").all()
    leak_rows = result.importance[result.importance["feature"].eq("strong__strong_raw_test_only_leak")]
    assert not leak_rows.empty
    assert float(leak_rows.iloc[0]["importance"]) == 0.0


def test_compare_is10_rescue_cli_writes_selected_features_and_metrics(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata_frame(48)
    strong = _feature_table(metadata, "strong_raw")
    compare = _feature_table(metadata, "compare_raw")
    is10 = _feature_table(metadata, "is10_raw")

    metadata_path = tmp_path / "metadata.csv"
    strong_path = tmp_path / "strong.csv"
    compare_path = tmp_path / "compare.csv"
    is10_path = tmp_path / "is10.csv"
    metadata.to_csv(metadata_path, index=False)
    strong.to_csv(strong_path, index=False)
    compare.to_csv(compare_path, index=False)
    is10.to_csv(is10_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "56_run_compare_is10_rescue.py"
    spec = importlib.util.spec_from_file_location("compare_is10_rescue_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    outputs = {
        "--metrics-output": tmp_path / "metrics.csv",
        "--predictions-output": tmp_path / "predictions.csv",
        "--selection-output": tmp_path / "selection.csv",
        "--importance-output": tmp_path / "importance.csv",
        "--summary-output": tmp_path / "summary.csv",
        "--merged-output": tmp_path / "merged.csv",
    }
    argv = [
        "56_run_compare_is10_rescue.py",
        "--metadata",
        str(metadata_path),
        "--strong-features",
        str(strong_path),
        "--compare-features",
        str(compare_path),
        "--is10-features",
        str(is10_path),
        "--no-extract-missing",
        "--top-k-values",
        "2",
        "--ranker",
        "univariate",
        "--model-names",
        "logistic_l2_f80",
        "--optuna-trials",
        "0",
        "--ensemble-top-k",
        "0",
    ]
    for flag, path in outputs.items():
        argv.extend([flag, str(path)])
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    metrics = pd.read_csv(outputs["--metrics-output"])
    predictions = pd.read_csv(outputs["--predictions-output"])
    summary = pd.read_csv(outputs["--summary-output"])
    assert not metrics.empty
    assert not predictions.empty
    assert summary["k"].tolist() == [2]
    assert metrics["feature_strategy"].eq("compare_is10_top2_univariate").all()


def test_extract_if_missing_passes_target_path_to_streaming_extractor(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata_frame(2)
    script = Path(__file__).parents[1] / "scripts" / "56_run_compare_is10_rescue.py"
    spec = importlib.util.spec_from_file_location("compare_is10_rescue_cli_path_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "compare.csv"
    seen: dict[str, object] = {}

    def fake_extract(metadata_arg: pd.DataFrame, output_path: Path, **kwargs: object) -> int:
        seen["output_path"] = output_path
        pd.DataFrame(
            [
                {
                    "recording_id": "rec_000_cough",
                    "participant_id": "p_000",
                    "dataset": "coswara",
                    "modality": "cough",
                    "submodality": "cough",
                    "label_binary": "negative",
                    "split": "train",
                    "representation": "opensmile_compare2016",
                    "event_duration_sec": 1.0,
                    "opensmile_compare2016_feature": 0.5,
                }
            ]
        ).to_csv(output_path, index=False)
        return 1

    monkeypatch.setattr(module, "extract_opensmile_feature_csv", fake_extract)

    features = module._extract_if_missing(
        output,
        metadata,
        feature_set="compare2016",
        quality_mode="quality_ok_only",
        progress_interval=250,
        chunk_size=32,
    )

    assert seen["output_path"] == output
    assert output.exists()
    assert features["recording_id"].tolist() == ["rec_000_cough"]
