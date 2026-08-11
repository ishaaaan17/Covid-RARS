from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _metadata_frame(n_participants: int = 72) -> pd.DataFrame:
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
                    "recording_date": f"2020-{(idx // 6) + 1:02d}-{(idx % 26) + 1:02d}",
                }
            )
    return pd.DataFrame(rows)


def _feature_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    modality_offsets = {"cough": 0.0, "breath": 0.2, "speech": -0.2}
    for _, row in metadata.iterrows():
        idx = int(str(row["participant_id"]).split("_")[-1])
        y = 1.0 if row["label_binary"] == "positive" else -1.0
        offset = modality_offsets[str(row["modality"])]
        rows.append(
            {
                "recording_id": row["recording_id"],
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "modality": row["modality"],
                "submodality": row["submodality"],
                "label_binary": row["label_binary"],
                "split": row["split"],
                "quality_flag": row["quality_flag"],
                "strong__signal": y + offset + idx * 0.002,
                "compare2016__signal": y * 0.7 + offset,
                "is10__noise": (idx % 7) * 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_compare_is10_temporal_validation_reports_three_protocols_and_summary() -> None:
    from covid_rars.compare_is10_final_validation import run_compare_is10_final_validation

    metadata = _metadata_frame()
    features = _feature_frame(metadata)

    result = run_compare_is10_final_validation(
        features=features,
        metadata=metadata,
        feature_strategy="compare_is10_top3_univariate",
        selected_feature_k=3,
        modalities=["cough", "breath", "speech"],
        model_names=["logistic_l2_f80"],
        enable_feature_level_fusion=True,
        global_stack_top_k=0,
        random_state=0,
    )

    assert {
        "compare_is10_existing_participant_split",
        "compare_is10_time_stratified_participant_split",
        "compare_is10_temporal_early_to_late",
    }.issubset(set(result.metrics["evaluation_protocol"]))
    assert {"strong_audio_modality", "strong_multimodal_fusion"}.issubset(
        set(result.metrics["analysis_family"])
    )
    assert result.metrics["feature_strategy"].eq("compare_is10_top3_univariate").all()
    assert result.metrics["selected_feature_k"].eq(3.0).all()
    assert not result.predictions.empty
    assert not result.split_summary.empty
    assert not result.modality_coverage.empty
    assert not result.final_summary.empty
    assert {"auroc", "delta_auroc_from_existing"}.issubset(result.final_summary.columns)


def test_final_summary_selects_on_validation_before_reading_test() -> None:
    from covid_rars.compare_is10_final_validation import _build_final_summary

    metrics = pd.DataFrame(
        [
            {
                "evaluation_protocol": "protocol_a",
                "analysis_family": "family",
                "model_name": "validation_winner",
                "modality": "cough",
                "metric_split": "validation",
                "auroc": 0.82,
                "auprc": 0.75,
            },
            {
                "evaluation_protocol": "protocol_a",
                "analysis_family": "family",
                "model_name": "test_winner",
                "modality": "cough",
                "metric_split": "validation",
                "auroc": 0.78,
                "auprc": 0.72,
            },
            {
                "evaluation_protocol": "protocol_a",
                "analysis_family": "family",
                "model_name": "validation_winner",
                "modality": "cough",
                "metric_split": "test",
                "auroc": 0.70,
                "auprc": 0.66,
            },
            {
                "evaluation_protocol": "protocol_a",
                "analysis_family": "family",
                "model_name": "test_winner",
                "modality": "cough",
                "metric_split": "test",
                "auroc": 0.91,
                "auprc": 0.84,
            },
        ]
    )

    summary = _build_final_summary(metrics, "top800", 800)

    assert summary["model_name"].tolist() == ["validation_winner"]
    assert summary["auroc"].tolist() == [0.70]


def test_final_summary_excludes_in_sample_validation_stack() -> None:
    from covid_rars.compare_is10_final_validation import _build_final_summary

    common = {
        "evaluation_protocol": "protocol_a",
        "analysis_family": "strong_multimodal_fusion",
        "model_name": "fusion",
        "modality": "multimodal",
        "modality_combination": "cough+speech",
    }
    metrics = pd.DataFrame(
        [
            {**common, "fusion_method": "uniform_mean", "metric_split": "validation", "auroc": 0.82, "auprc": 0.75},
            {
                **common,
                "fusion_method": "stacked_logistic_validation",
                "metric_split": "validation",
                "auroc": 0.90,
                "auprc": 0.84,
            },
            {**common, "fusion_method": "uniform_mean", "metric_split": "test", "auroc": 0.80, "auprc": 0.74},
            {
                **common,
                "fusion_method": "stacked_logistic_validation",
                "metric_split": "test",
                "auroc": 0.91,
                "auprc": 0.85,
            },
        ]
    )

    summary = _build_final_summary(metrics, "top800", 800)

    assert summary["fusion_method"].tolist() == ["uniform_mean"]
    assert summary["auroc"].tolist() == [0.80]


def test_compare_is10_external_transfer_runs_when_target_features_match() -> None:
    from covid_rars.compare_is10_final_validation import run_compare_is10_external_transfer

    metadata = _metadata_frame()
    source = _feature_frame(metadata)
    target = source[source["modality"].eq("cough") & source["split"].eq("test")].copy()
    target["dataset"] = "coughvid"
    target["split"] = "external_test"

    metrics, predictions = run_compare_is10_external_transfer(
        source_features=source,
        target_features=target,
        feature_strategy="compare_is10_top3_univariate",
        selected_feature_k=3,
        model_names=["logistic_l2_f80"],
        random_state=0,
    )

    assert not metrics.empty
    assert not predictions.empty
    assert metrics["evaluation_protocol"].eq("coswara_to_coughvid_compare_is10_external").all()
    assert predictions["dataset"].eq("coughvid").all()
    assert metrics["metric_split"].eq("external_test").all()


def test_compare_is10_final_validation_cli_writes_tables_and_figures(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata_frame()
    features = _feature_frame(metadata)
    target = features[features["modality"].eq("cough") & features["split"].eq("test")].copy()
    target["dataset"] = "coughvid"
    target["split"] = "external_test"

    metadata_path = tmp_path / "metadata.csv"
    features_path = tmp_path / "features.csv"
    external_path = tmp_path / "external.csv"
    metadata.to_csv(metadata_path, index=False)
    features.to_csv(features_path, index=False)
    target.to_csv(external_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "58_run_compare_is10_final_validation.py"
    spec = importlib.util.spec_from_file_location("compare_is10_final_validation_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    outputs = {
        "--metrics-output": tmp_path / "metrics.csv",
        "--predictions-output": tmp_path / "predictions.csv",
        "--summary-output": tmp_path / "summary.csv",
        "--split-summary-output": tmp_path / "splits.csv",
        "--modality-coverage-output": tmp_path / "coverage.csv",
        "--external-metrics-output": tmp_path / "external_metrics.csv",
        "--external-predictions-output": tmp_path / "external_predictions.csv",
        "--figure-output": tmp_path / "degradation.svg",
        "--summary-figure-output": tmp_path / "summary.svg",
    }
    argv = [
        "58_run_compare_is10_final_validation.py",
        "--features",
        str(features_path),
        "--metadata",
        str(metadata_path),
        "--external-features",
        str(external_path),
        "--feature-strategy",
        "compare_is10_top3_univariate",
        "--selected-feature-k",
        "3",
        "--model-names",
        "logistic_l2_f80",
        "--enable-feature-level-fusion",
        "--global-stack-top-k",
        "0",
    ]
    for flag, path in outputs.items():
        argv.extend([flag, str(path)])
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
    summary = pd.read_csv(outputs["--summary-output"])
    assert "compare_is10_temporal_early_to_late" in set(summary["evaluation_protocol"])
    external = pd.read_csv(outputs["--external-metrics-output"])
    assert not external.empty
