from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _metadata_frame(n_participants: int = 36) -> pd.DataFrame:
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


def _feature_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offsets = {"cough": 0.0, "breath": 0.2, "speech": -0.2}
    for _, row in metadata.iterrows():
        idx = int(str(row["participant_id"]).split("_")[-1])
        signal = 1.0 if row["label_binary"] == "positive" else -1.0
        offset = offsets[str(row["modality"])]
        rows.append(
            {
                "recording_id": row["recording_id"],
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "modality": row["modality"],
                "submodality": row["submodality"],
                "label_binary": row["label_binary"],
                "split": row["split"],
                "feat_signal": signal + offset + idx * 0.001,
                "feat_signal_2": signal * 0.5 + offset,
                "feat_noise": (idx % 4) * 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_clean_internal_protocol_excludes_ambiguous_and_reports_audits() -> None:
    from covid_rars.strong_baseline_protocol import build_clean_internal_protocol

    metadata = _metadata_frame(12)
    ambiguous = metadata.iloc[[0]].copy()
    ambiguous["recording_id"] = "ambiguous_rec"
    ambiguous["participant_id"] = "ambiguous_p"
    ambiguous["label_raw"] = "recovered_full"
    ambiguous["label_binary"] = "negative"
    metadata = pd.concat([metadata, ambiguous], ignore_index=True)

    result = build_clean_internal_protocol(metadata)

    assert "ambiguous_rec" not in set(result.metadata["recording_id"])
    assert result.metadata["label_binary"].isin(["positive", "negative"]).all()
    assert {"included", "ambiguous_raw_label"}.issubset(
        set(result.audit["protocol_exclusion_reason"].astype(str))
    )
    assert not result.participant_audit.empty
    assert result.metadata["evaluation_protocol"].eq("clean_internal_protocol").all()


def test_strong_baseline_selects_models_and_runs_all_fusion_combinations() -> None:
    from covid_rars.strong_baseline import run_strong_baseline

    metadata = _metadata_frame(48)
    features = _feature_frame(metadata)
    result = run_strong_baseline(
        features,
        metadata=metadata,
        model_names=["logistic_l2_f80"],
        random_state=0,
    )

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert set(result.selection["modality"]) == {"cough", "breath", "speech"}
    modality_metrics = result.metrics[result.metrics["analysis_family"].eq("strong_audio_modality")]
    assert {"validation", "test"}.issubset(set(modality_metrics["metric_split"]))
    fusion = result.metrics[result.metrics["analysis_family"].eq("strong_multimodal_fusion")]
    assert {
        "cough+breath",
        "cough+speech",
        "breath+speech",
        "cough+breath+speech",
    }.issubset(set(fusion["modality_combination"]))
    assert fusion["threshold_source"].eq("validation_balanced_accuracy").all()
    test_rows = result.metrics[result.metrics["metric_split"].eq("test")]
    assert test_rows["auroc"].notna().any()
    assert test_rows["n_participants"].notna().all()


def test_feature_level_fusion_trains_multimodal_models_before_prediction_fusion() -> None:
    from covid_rars.strong_baseline import run_strong_baseline

    metadata = _metadata_frame(48)
    features = _feature_frame(metadata)
    result = run_strong_baseline(
        features,
        metadata=metadata,
        model_names=["logistic_l2_f80"],
        enable_feature_level_fusion=True,
        global_stack_top_k=0,
        random_state=0,
    )

    early = result.metrics[result.metrics["analysis_family"].eq("strong_feature_level_fusion")]

    assert not early.empty
    assert {
        "cough+breath",
        "cough+speech",
        "breath+speech",
        "cough+breath+speech",
    }.issubset(set(early["modality_combination"]))
    assert {"validation", "test"}.issubset(set(early["metric_split"]))
    assert early["threshold_source"].eq("validation_balanced_accuracy").all()


def test_augmented_rows_survive_clean_protocol_merge_by_source_recording_id() -> None:
    from covid_rars.strong_baseline import _prepare_features_for_protocol

    metadata = _metadata_frame(18)
    features = _feature_frame(metadata)
    train_features = features[features["split"].eq("train")].head(2).copy()
    train_features["source_recording_id"] = train_features["recording_id"].astype(str)
    train_features["recording_id"] = train_features["recording_id"].astype(str) + "::aug1"
    train_features["is_augmented"] = True
    augmented = pd.concat([features, train_features], ignore_index=True)

    prepared, _ = _prepare_features_for_protocol(
        augmented,
        metadata=metadata,
        modalities=["cough", "breath", "speech"],
        require_quality_ok=False,
    )

    assert set(train_features["recording_id"].astype(str)).issubset(set(prepared["recording_id"].astype(str)))
    assert prepared[prepared["recording_id"].astype(str).str.contains("::aug1")]["split"].eq("train").all()


def test_global_stacker_uses_validation_predictions_and_reports_test_metrics() -> None:
    from covid_rars.strong_baseline import run_strong_baseline

    metadata = _metadata_frame(54)
    features = _feature_frame(metadata)
    result = run_strong_baseline(
        features,
        metadata=metadata,
        model_names=["logistic_l2_f80", "extra_trees_f100"],
        enable_feature_level_fusion=True,
        global_stack_top_k=4,
        random_state=0,
    )

    stacked = result.metrics[result.metrics["analysis_family"].eq("strong_global_stacking")]

    assert not stacked.empty
    assert {"validation", "test"}.issubset(set(stacked["metric_split"]))
    assert stacked["model_name"].astype(str).str.contains("global").all()
    assert stacked["threshold_source"].eq("validation_balanced_accuracy").all()


def test_strong_baseline_cli_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata_frame(36)
    features = _feature_frame(metadata)
    metadata_path = tmp_path / "metadata.csv"
    features_path = tmp_path / "features.csv"
    metadata.to_csv(metadata_path, index=False)
    features.to_csv(features_path, index=False)

    outputs = {
        "--metrics-output": tmp_path / "metrics.csv",
        "--predictions-output": tmp_path / "predictions.csv",
        "--selection-output": tmp_path / "selection.csv",
        "--protocol-audit-output": tmp_path / "protocol.csv",
        "--participant-audit-output": tmp_path / "participants.csv",
    }
    script = Path(__file__).parents[1] / "scripts" / "47_run_strong_baseline.py"
    spec = importlib.util.spec_from_file_location("strong_baseline_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "47_run_strong_baseline.py",
        "--features",
        str(features_path),
        "--metadata",
        str(metadata_path),
        "--model-names",
        "logistic_l2_f80",
        "--feature-level-fusion",
        "--global-stack-top-k",
        "4",
    ]
    for flag, path in outputs.items():
        argv.extend([flag, str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
    metrics = pd.read_csv(outputs["--metrics-output"])
    assert {"strong_audio_modality", "strong_multimodal_fusion"}.issubset(set(metrics["analysis_family"]))


def test_strong_baseline_e2e_runner_uses_existing_inputs(tmp_path: Path, monkeypatch) -> None:
    script = Path(__file__).parents[1] / "scripts" / "48_run_strong_baseline_e2e.py"
    spec = importlib.util.spec_from_file_location("strong_baseline_e2e", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    features = tmp_path / "data" / "processed" / "features_mfcc.csv"
    metadata = tmp_path / "data" / "processed" / "metadata_with_quality.csv"
    features.parent.mkdir(parents=True)
    pd.DataFrame({"recording_id": ["r1"]}).to_csv(features, index=False)
    pd.DataFrame({"recording_id": ["r1"]}).to_csv(metadata, index=False)

    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_run", lambda args: commands.append(list(args)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "48_run_strong_baseline_e2e.py",
            "--features",
            "data/processed/features_mfcc.csv",
            "--metadata",
            "data/processed/metadata_with_quality.csv",
            "--skip-validation",
        ],
    )

    module.main()

    command_text = [" ".join(cmd) for cmd in commands]
    assert any("47_run_strong_baseline.py" in cmd for cmd in command_text)
    assert any("20_make_paper_tables.py" in cmd for cmd in command_text)
    assert any("24_make_experiment_manifest.py" in cmd for cmd in command_text)
    assert not any("12_validate_artifacts.py" in cmd for cmd in command_text)


def test_strong_baseline_e2e_validates_split_metadata_after_rebuild(tmp_path: Path, monkeypatch) -> None:
    script = Path(__file__).parents[1] / "scripts" / "48_run_strong_baseline_e2e.py"
    spec = importlib.util.spec_from_file_location("strong_baseline_e2e", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "interim" / "coswara_index.csv").write_text("recording_id\n")
    (tmp_path / "data" / "processed" / "metadata_with_quality.csv").write_text("recording_id\n")
    (tmp_path / "data" / "processed" / "audio_quality.csv").write_text("recording_id\n")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_run", lambda args: commands.append(list(args)))
    monkeypatch.setattr(module, "_torch_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "48_run_strong_baseline_e2e.py",
            "--raw-dir",
            str(raw_dir),
            "--rebuild-features",
            "--skip-cnn",
        ],
    )

    module.main()

    validation_commands = [cmd for cmd in commands if "scripts/12_validate_artifacts.py" in cmd]
    assert validation_commands
    validation = validation_commands[-1]
    assert "--metadata" in validation
    assert validation[validation.index("--metadata") + 1] == "data/processed/metadata_with_quality.csv"


def test_extended_acoustic_feature_bank_contains_paper_style_features() -> None:
    from covid_rars.strong_features import extract_extended_acoustic_features

    sample_rate = 16_000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    features = extract_extended_acoustic_features(y, sample_rate=sample_rate, n_mfcc=20, n_mels=32)

    assert len(features) > 250
    assert {"mfcc_01_mean", "delta_mfcc_01_std", "mel_band_01_mean", "chroma_01_mean"}.issubset(features)
    assert {"spectral_contrast_01_mean", "rms_mean", "zcr_mean", "duration_sec"}.issubset(features)
    assert all(np.isfinite(value) for value in features.values())


def test_strong_feature_extraction_augments_training_rows_only(monkeypatch) -> None:
    import covid_rars.strong_features as strong_features

    metadata = pd.DataFrame(
        [
            {
                "recording_id": "train_a",
                "participant_id": "p_train_a",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": "positive",
                "split": "train",
                "audio_path": "dummy.wav",
            },
            {
                "recording_id": "train_b",
                "participant_id": "p_train_b",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": "negative",
                "split": "train",
                "audio_path": "dummy.wav",
            },
            {
                "recording_id": "validation_a",
                "participant_id": "p_validation",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": "positive",
                "split": "validation",
                "audio_path": "dummy.wav",
            },
            {
                "recording_id": "test_a",
                "participant_id": "p_test",
                "dataset": "coswara",
                "modality": "cough",
                "submodality": "cough",
                "label_binary": "negative",
                "split": "test",
                "audio_path": "dummy.wav",
            },
        ]
    )

    def fake_extract(row: pd.Series) -> dict[str, object]:
        return {
            "recording_id": row["recording_id"],
            "participant_id": row["participant_id"],
            "dataset": row["dataset"],
            "modality": row["modality"],
            "submodality": row["submodality"],
            "label_binary": row["label_binary"],
            "split": row["split"],
            "is_augmented": bool(row.get("is_augmented", False)),
            "augmentation_id": row.get("augmentation_id", "original"),
            "source_recording_id": row.get("source_recording_id", row["recording_id"]),
            "feat_signal": 1.0,
        }

    monkeypatch.setattr(strong_features, "extract_strong_features_for_row", fake_extract)

    table = strong_features.build_strong_feature_table(
        metadata,
        augment_train_copies=2,
        augmentation_seed=123,
    )

    assert len(table) == 8
    assert table["recording_id"].is_unique
    assert table[table["split"].eq("train")]["is_augmented"].sum() == 4
    assert table[table["split"].eq("validation")]["is_augmented"].sum() == 0
    assert table[table["split"].eq("test")]["is_augmented"].sum() == 0
    assert set(table[table["is_augmented"]]["source_recording_id"]) == {"train_a", "train_b"}


def test_optional_smote_model_is_registered_or_skips_cleanly() -> None:
    from covid_rars.strong_baseline import _make_model

    try:
        model = _make_model("extra_trees_smote_f100", random_state=0)
    except RuntimeError as exc:
        assert "imbalanced-learn" in str(exc)
    else:
        assert hasattr(model, "fit")


def test_strong_models_filter_constant_features_before_univariate_selection() -> None:
    from covid_rars.strong_baseline import _make_model

    model = _make_model("logistic_l2_f80", random_state=0)

    assert "variance" in model.named_steps
    assert list(model.named_steps).index("variance") < list(model.named_steps).index("select")


def test_strong_model_fit_suppresses_expected_constant_feature_warnings() -> None:
    from covid_rars.strong_baseline import _make_model

    x = pd.DataFrame(
        {
            "constant": [1.0] * 8,
            "perfect_split": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "useful": [-1.2, -0.9, -0.8, -0.7, 0.7, 0.8, 0.9, 1.2],
        }
    )
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    model = _make_model("logistic_l2_f80", random_state=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)

    warning_text = "\n".join(str(item.message) for item in caught)
    assert "constant" not in warning_text.lower()
    assert "divide" not in warning_text.lower()


def test_svc_uses_explicit_calibration_instead_of_deprecated_probability_mode() -> None:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    from covid_rars.strong_baseline import _make_model

    model = _make_model("svc_rbf_f60", random_state=0)
    calibrated = model.named_steps["model"]

    assert isinstance(calibrated, CalibratedClassifierCV)
    assert isinstance(calibrated.estimator, SVC)
    assert getattr(calibrated.estimator, "probability", None) is not True


def test_svc_fit_does_not_emit_deprecated_probability_warning() -> None:
    from covid_rars.strong_baseline import _make_model

    x = pd.DataFrame(
        {
            "signal": [-1.0, -0.8, -0.7, -0.6, 0.6, 0.7, 0.8, 1.0, -0.9, 0.9, -0.5, 0.5],
            "noise": [0.01, 0.02, 0.01, 0.03, 0.02, 0.04, 0.03, 0.01, 0.04, 0.02, 0.05, 0.05],
            "constant": [1.0] * 12,
        }
    )
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])
    model = _make_model("svc_rbf_f60", random_state=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)

    warning_text = "\n".join(str(item.message) for item in caught)
    assert "probability" not in warning_text.lower()


def test_top_k_validation_ensemble_is_added_as_modality_candidate() -> None:
    from covid_rars.strong_baseline import run_strong_baseline

    metadata = _metadata_frame(48)
    features = _feature_frame(metadata)
    result = run_strong_baseline(
        features,
        metadata=metadata,
        model_names=["logistic_l2_f80", "extra_trees_f100", "random_forest_f80"],
        ensemble_top_k=2,
        random_state=0,
    )

    assert "top_2_validation_ensemble" in set(result.metrics["model_name"].astype(str))
    ensemble_predictions = result.predictions[
        result.predictions["model_name"].astype(str).eq("top_2_validation_ensemble")
    ]
    assert not ensemble_predictions.empty
    assert ensemble_predictions["analysis_family"].eq("strong_audio_modality").all()
