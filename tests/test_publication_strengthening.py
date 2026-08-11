import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_paired_bootstrap_difference_detects_model_delta():
    from covid_rars.model_comparison import paired_bootstrap_difference

    predictions = pd.DataFrame(
        {
            "recording_id": ["r1", "r2", "r3", "r4", "r1", "r2", "r3", "r4"],
            "label_binary": ["negative", "negative", "positive", "positive"] * 2,
            "model_name": ["weak"] * 4 + ["strong"] * 4,
            "probability": [0.45, 0.40, 0.55, 0.60, 0.05, 0.15, 0.85, 0.95],
        }
    )

    result = paired_bootstrap_difference(
        predictions,
        baseline_name="weak",
        candidate_name="strong",
        model_column="model_name",
        metric="brier",
        n_bootstraps=100,
        random_state=7,
    )

    assert result["baseline_name"] == "weak"
    assert result["candidate_name"] == "strong"
    assert result["metric"] == "brier"
    assert result["difference"] < 0.0
    assert result["n_matched"] == 4
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]


def test_coarsened_matching_balances_label_groups_by_age_bucket_and_gender():
    from covid_rars.confounding import balance_table, coarsened_exact_match

    metadata = pd.DataFrame(
        {
            "recording_id": [f"r{i}" for i in range(8)],
            "participant_id": [f"p{i}" for i in range(8)],
            "label_binary": ["positive", "negative"] * 4,
            "age": [22, 23, 39, 38, 54, 55, 70, 72],
            "gender": ["female", "female", "male", "male", "female", "female", "male", "male"],
        }
    )

    matched = coarsened_exact_match(metadata, covariates=["age_bucket", "gender"])
    balance = balance_table(matched, covariates=["age_bucket", "gender"])

    assert len(matched) == 8
    assert set(matched["matched_set_id"]) == {"age_bucket=<30|gender=female", "age_bucket=30-44|gender=male", "age_bucket=45-59|gender=female", "age_bucket=60+|gender=male"}
    assert balance["max_abs_standardized_difference"].max() == 0.0


def test_feature_shift_report_ranks_shifted_numeric_features():
    from covid_rars.shift import feature_shift_report

    source = pd.DataFrame({"recording_id": ["s1", "s2", "s3"], "mfcc_1": [0.0, 0.1, 0.2], "mfcc_2": [1.0, 1.0, 1.0]})
    external = pd.DataFrame({"recording_id": ["e1", "e2", "e3"], "mfcc_1": [5.0, 5.1, 5.2], "mfcc_2": [1.0, 1.0, 1.0]})

    report = feature_shift_report(source, external, id_columns=["recording_id"])

    assert report.iloc[0]["feature"] == "mfcc_1"
    assert report.iloc[0]["abs_standardized_mean_difference"] > 1.0
    assert report.loc[report["feature"] == "mfcc_2", "abs_standardized_mean_difference"].iloc[0] == 0.0


def test_manifest_records_config_and_sha256(tmp_path):
    from covid_rars.manifest import build_experiment_manifest

    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nauroc,0.8\n", encoding="utf-8")

    manifest = build_experiment_manifest(
        config={"seed": 42, "run_name": "toy"},
        artifact_paths=[artifact],
        include_packages=["python"],
    )

    assert manifest["config"]["seed"] == 42
    assert manifest["artifacts"][0]["path"].endswith("metrics.csv")
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert "created_at_utc" in manifest



def test_participant_level_subgroup_merge_does_not_duplicate_fusion_rows():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "10_shift_and_confounding_checks.py"
    spec = importlib.util.spec_from_file_location("shift_and_confounding_checks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    predictions = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "label_binary": ["positive", "negative"],
            "probability": [0.8, 0.2],
            "fusion_method": ["uniform_mean", "uniform_mean"],
        }
    )
    metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "recording_id": ["r1", "r2", "r3", "r4"],
            "quality_flag": ["ok", "corrupt", "ok", "mostly_silence"],
            "age": [21, 21, 44, 44],
            "gender": ["male", "male", "female", "female"],
        }
    )

    merged = module.merge_predictions_with_metadata(predictions, metadata)

    assert len(merged) == len(predictions)
    assert merged.loc[merged["participant_id"] == "p1", "quality_flag"].iloc[0] == "corrupt"
    assert merged.loc[merged["participant_id"] == "p2", "quality_flag"].iloc[0] == "mostly_silence"



def test_matched_subset_filter_supports_participant_level_fusion_predictions():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "22_confounding_matching.py"
    spec = importlib.util.spec_from_file_location("confounding_matching_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    matched = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "recording_id": ["r1", "r2", "r3", "r4"],
            "label_binary": ["positive", "positive", "negative", "negative"],
        }
    )
    predictions = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "label_binary": ["positive", "negative", "positive"],
            "probability": [0.8, 0.2, 0.9],
            "fusion_method": ["quality_weighted_auprc"] * 3,
        }
    )

    subset = module.filter_predictions_to_matched_subset(predictions, matched)

    assert list(subset["participant_id"]) == ["p1", "p2"]
    assert "p3" not in set(subset["participant_id"])



def test_quality_weighted_fusion_uses_validation_threshold_below_half():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "16_run_quality_weighted_fusion.py"
    spec = importlib.util.spec_from_file_location("quality_weighted_fusion_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows = []
    for split, ids in [("validation", ["v1", "v2", "v3", "v4"]), ("test", ["t1", "t2", "t3", "t4"])]:
        labels = {ids[0]: "negative", ids[1]: "negative", ids[2]: "positive", ids[3]: "positive"}
        probs = {ids[0]: 0.05, ids[1]: 0.10, ids[2]: 0.25, ids[3]: 0.35}
        for pid in ids:
            for modality in ["cough", "breath", "speech"]:
                rows.append(
                    {
                        "recording_id": f"{pid}_{modality}",
                        "participant_id": pid,
                        "model_name": "logistic_regression",
                        "modality": modality,
                        "label_binary": labels[pid],
                        "split": split,
                        "probability": probs[pid],
                    }
                )
    predictions = pd.DataFrame(rows)
    quality = pd.DataFrame({"recording_id": predictions["recording_id"], "quality_flag": "ok"})
    validation_metrics = pd.DataFrame({"modality": ["cough", "breath", "speech"], "auprc": [0.9, 0.8, 0.7]})

    validation_fused = module._run_quality_weighted_fusion(
        predictions[predictions["split"] == "validation"], quality, validation_metrics, "auprc"
    )
    thresholds = module._thresholds_from_validation(validation_fused)
    test_fused = module._run_quality_weighted_fusion(
        predictions[predictions["split"] == "test"], quality, validation_metrics, "auprc"
    )
    metrics = module._evaluate_fused(test_fused, thresholds)

    assert thresholds["threshold"].iloc[0] < 0.5
    assert metrics["threshold"].iloc[0] < 0.5
    assert metrics["f1"].iloc[0] == 1.0


def test_paper_table_defaults_include_quality_weighted_fusion_metrics():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "20_make_paper_tables.py"
    spec = importlib.util.spec_from_file_location("paper_tables_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    paths = {path.as_posix() for path in module.DEFAULT_METRIC_PATHS}
    assert "data/outputs/metrics/quality_weighted_fusion_metrics.csv" in paths


def test_ml_baseline_script_writes_validation_metrics_for_fusion_weights(tmp_path):
    import subprocess
    import sys

    rows = []
    labels = ["negative", "positive", "negative", "positive"] * 3
    splits = ["train"] * 4 + ["validation"] * 4 + ["test"] * 4
    for idx, (label, split) in enumerate(zip(labels, splits)):
        is_pos = label == "positive"
        rows.append(
            {
                "recording_id": f"r{idx}",
                "participant_id": f"p{idx}",
                "dataset": "toy",
                "modality": "cough",
                "label_binary": label,
                "split": split,
                "mfcc_mean_0": float(idx) + (1.0 if is_pos else 0.0),
                "mfcc_std_0": float(idx % 3) + (0.5 if is_pos else 0.0),
                "spectral_centroid_mean": float(idx % 5),
            }
        )
    features = pd.DataFrame(rows)
    features_path = tmp_path / "features.csv"
    metrics_path = tmp_path / "test_metrics.csv"
    validation_metrics_path = tmp_path / "validation_metrics.csv"
    validation_predictions_path = tmp_path / "validation_predictions.csv"
    test_predictions_path = tmp_path / "test_predictions.csv"
    models_dir = tmp_path / "models"
    features.to_csv(features_path, index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "06_train_ml_baselines.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--features",
            str(features_path),
            "--models-dir",
            str(models_dir),
            "--metrics-output",
            str(metrics_path),
            "--validation-metrics-output",
            str(validation_metrics_path),
            "--validation-output",
            str(validation_predictions_path),
            "--test-output",
            str(test_predictions_path),
            "--modalities",
            "cough",
            "--model-names",
            "logistic_regression",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    test_metrics = pd.read_csv(metrics_path)
    validation_metrics = pd.read_csv(validation_metrics_path)

    assert test_metrics.loc[0, "n_samples"] == 4.0
    assert validation_metrics.loc[0, "n_samples"] == 4.0
    assert validation_metrics.loc[0, "metric_split"] == "validation"
    assert validation_metrics.loc[0, "model_name"] == "logistic_regression"
    assert validation_metrics.loc[0, "modality"] == "cough"


def test_fusion_scripts_reject_test_metrics_as_validation_weights(tmp_path):
    import importlib.util

    for script_name in ["09_run_fusion.py", "16_run_quality_weighted_fusion.py"]:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
        spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        bad_metrics = tmp_path / "ml_baseline_metrics.csv"
        bad_metrics.write_text("modality,auprc\ncough,0.9\n", encoding="utf-8")

        try:
            module._read_validation_metrics(bad_metrics)
        except ValueError as exc:
            assert "test-set output" in str(exc)
        else:
            raise AssertionError(f"{script_name} accepted test metrics as validation metrics")

        good_metrics = tmp_path / f"{script_name}_validation_metrics.csv"
        good_metrics.write_text("modality,auprc,metric_split\ncough,0.9,validation\n", encoding="utf-8")
        accepted = module._read_validation_metrics(good_metrics)
        assert accepted.loc[0, "metric_split"] == "validation"
