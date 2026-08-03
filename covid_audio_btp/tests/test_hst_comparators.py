from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest


MODEL_NAMES = (
    "lightgbm_smote_f80",
    "svc_rbf_f60",
    "catboost_smote_f80",
    "xgboost_smote_f80",
)
SELECTED_CANDIDATE_NAME = "validation_selected_candidate"


class _DeterministicRanker:
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_DeterministicRanker":
        self.feature_importances_ = np.arange(len(x.columns), 0, -1, dtype=float)
        return self


class _RecordingEstimator:
    fit_rows: list[tuple[str, tuple[str, ...]]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_RecordingEstimator":
        assert np.isfinite(x.to_numpy(dtype=float)).all()
        self.__class__.fit_rows.append((self.model_name, tuple(x.index.astype(str))))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        score = np.clip(pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(0.0), -6.0, 6.0)
        probability = 1.0 / (1.0 + np.exp(-score.to_numpy(dtype=float)))
        return np.column_stack([1.0 - probability, probability])


def _manifest() -> pd.DataFrame:
    from covid_audio_btp.hst_protocols import _finalize_manifest

    rows: list[dict[str, object]] = []
    split_people = {
        "train": (("tr0", "negative"), ("tr1", "positive"), ("tr2", "negative"), ("tr3", "positive")),
        "validation": (("va0", "negative"), ("va1", "positive")),
        "test": (("te0", "negative"), ("te1", "positive")),
    }
    for split, people in split_people.items():
        for participant_id, label in people:
            n_recordings = 3 if participant_id == "te1" else 1
            for recording_index in range(n_recordings):
                recording_id = f"{participant_id}_r{recording_index}"
                rows.append(
                    {
                        "run_id": "ignored_manifest_run",
                        "protocol": "track_a",
                        "fold": 0,
                        "cohort": "all_eligible",
                        "split": split,
                        "dataset": "coswara",
                        "participant_key": f"coswara::{participant_id}",
                        "recording_key": f"coswara::{recording_id}",
                        "modality": "cough",
                        "label_binary": label,
                        "representation_id": "shared_eligible_v1",
                        "source_audio_sha256": sha256(recording_id.encode("ascii")).hexdigest(),
                        "scientific_configuration_fingerprint": sha256(
                            b"test-scientific-config"
                        ).hexdigest(),
                        "eligibility_alignment_fingerprint": sha256(
                            b"test-eligibility-alignment"
                        ).hexdigest(),
                        "analysis_scope": "internal_performance",
                        "analysis_role": "primary",
                        "estimand_id": "test_primary_participant_auroc",
                        "multiplicity_family": "test_primary_internal_performance",
                        "analysis_mode": "confirmatory",
                        "confirmatory_protocol": True,
                    }
                )
    return _finalize_manifest(pd.DataFrame(rows))


def _features(manifest: pd.DataFrame) -> pd.DataFrame:
    base = manifest[
        [
            "dataset",
            "participant_key",
            "recording_key",
            "modality",
            "label_binary",
            "source_audio_sha256",
        ]
    ].drop_duplicates().reset_index(drop=True)
    labels = base["label_binary"].map({"negative": -1.0, "positive": 1.0})
    recording_offset = base.groupby("participant_key").cumcount().astype(float) / 10.0
    base["compare__f0"] = labels + recording_offset
    base["compare__f1"] = labels * 2.0
    base["is10__f0"] = np.arange(len(base), dtype=float)
    base["constant_train"] = 1.0
    base["nonfinite_train"] = 0.0
    train_keys = set(manifest.loc[manifest["split"].eq("train"), "recording_key"])
    base.loc[base["recording_key"].isin(train_keys), "nonfinite_train"] = np.nan
    return base


def _feature_contract(features: pd.DataFrame):
    from covid_audio_btp.hst_comparators import build_compare_is10_feature_contract

    return build_compare_is10_feature_contract(
        features,
        ordered_feature_columns=(
            "compare__f0",
            "compare__f1",
            "is10__f0",
            "constant_train",
            "nonfinite_train",
        ),
    )


def _refreeze(manifest: pd.DataFrame) -> pd.DataFrame:
    from covid_audio_btp.hst_protocols import _finalize_manifest

    return _finalize_manifest(
        manifest.drop(columns=["row_content_sha256", "manifest_sha256"], errors="ignore")
    )


def _run(manifest: pd.DataFrame, features: pd.DataFrame, **kwargs: object):
    from covid_audio_btp.hst_comparators import run_aligned_compare_is10

    feature_contract = kwargs.pop("feature_contract") if "feature_contract" in kwargs else _feature_contract(features)
    return run_aligned_compare_is10(
        features,
        manifest,
        feature_contract=feature_contract,
        selected_feature_k=3,
        ensemble_top_k=5,
        test_mode=True,
        ranker_factory=lambda random_state: _DeterministicRanker(),
        estimator_factory=lambda model_name, random_state: _RecordingEstimator(model_name),
        **kwargs,
    )


def test_feature_ranking_is_fold_train_only_and_records_backend() -> None:
    manifest = _manifest()
    features = _features(manifest)

    first = _run(manifest, features)
    changed = features.copy()
    held_out = set(manifest.loc[~manifest["split"].eq("train"), "recording_key"])
    changed.loc[changed["recording_key"].isin(held_out), changed.columns.str.contains("__f")] = 1e12
    second = _run(manifest, changed)

    selected_first = first.feature_selection.loc[first.feature_selection["selected"], "feature"].tolist()
    selected_second = second.feature_selection.loc[second.feature_selection["selected"], "feature"].tolist()
    assert selected_first == selected_second == ["compare__f0", "compare__f1", "is10__f0"]
    assert set(first.feature_selection["selection_split"]) == {"train"}
    assert set(first.feature_selection["ranker_backend"]) == {"injected:_DeterministicRanker"}
    removed = first.feature_selection.set_index("feature")["removal_reason"].to_dict()
    assert removed["constant_train"] == "constant_in_train"
    assert removed["nonfinite_train"] == "no_finite_training_values"


def test_true_held_out_missing_values_use_frozen_training_imputation() -> None:
    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    baseline = _run(manifest, features, feature_contract=contract)
    missing = features.copy()
    missing["compare__f0"] = missing["compare__f0"].astype(object)
    held_out = set(manifest.loc[~manifest["split"].eq("train"), "recording_key"])
    missing.loc[missing["recording_key"].isin(held_out), "compare__f0"] = pd.NA

    changed = _run(manifest, missing, feature_contract=contract)

    selected = changed.feature_selection.loc[changed.feature_selection["selected"], "feature"].tolist()
    assert selected == baseline.feature_selection.loc[
        baseline.feature_selection["selected"], "feature"
    ].tolist()
    assert np.isfinite(changed.predictions["probability"]).all()
    assert set(changed.feature_selection["feature_schema_sha256"]) == {contract.schema_sha256}
    declared = changed.feature_selection.set_index("feature")["declared_dtype"].to_dict()
    assert declared["compare__f0"] == "float64"


def test_malformed_held_out_numeric_content_is_rejected_not_imputed() -> None:
    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    features["compare__f0"] = features["compare__f0"].astype(object)
    held_out = set(manifest.loc[~manifest["split"].eq("train"), "recording_key"])
    features.loc[features["recording_key"].isin(held_out), "compare__f0"] = "malformed"

    with pytest.raises(ValueError, match="malformed.*compare__f0"):
        _run(manifest, features, feature_contract=contract)


def test_feature_schema_hash_order_and_membership_are_verified() -> None:
    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    invalid_contract = {
        "ordered_feature_columns": list(contract.ordered_feature_columns),
        "feature_dtypes": list(contract.feature_dtypes),
        "schema_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="schema_sha256"):
        _run(manifest, features, feature_contract=invalid_contract)

    reordered_columns = list(features.columns)
    left = reordered_columns.index("compare__f0")
    right = reordered_columns.index("compare__f1")
    reordered_columns[left], reordered_columns[right] = (
        reordered_columns[right],
        reordered_columns[left],
    )
    reordered = features[reordered_columns]
    with pytest.raises(ValueError, match="ordered feature"):
        _run(manifest, reordered, feature_contract=contract)

    undeclared = features.assign(rogue_feature=1.0)
    with pytest.raises(ValueError, match="undeclared feature"):
        _run(manifest, undeclared, feature_contract=contract)


def test_declared_semantic_dtype_uses_lossless_conversion_and_train_median_policy() -> None:
    from covid_audio_btp.hst_comparators import build_compare_is10_feature_contract

    manifest = _manifest()
    features = _features(manifest)
    features["compare__f0"] = np.rint(features["compare__f0"]).astype(float)
    columns = (
        "compare__f0",
        "compare__f1",
        "is10__f0",
        "constant_train",
        "nonfinite_train",
    )
    contract = build_compare_is10_feature_contract(
        features,
        ordered_feature_columns=columns,
        declared_feature_dtypes={
            column: "float64" if column == "nonfinite_train" else "int64"
            for column in columns
        },
    )
    result = _run(manifest, features, feature_contract=contract)
    te1 = result.predictions[
        result.predictions["participant_key"].eq("coswara::te1")
        & result.predictions["model"].eq(MODEL_NAMES[0])
    ].sort_values("recording_key")
    assert te1["probability"].tolist() == pytest.approx(
        [1.0 / (1.0 + np.exp(-1.0))] * 3
    )
    assert set(result.feature_selection["missing_policy"]) == {"train_median"}

    lossy_train = features.copy()
    train_key = manifest.loc[manifest["split"].eq("train"), "recording_key"].iloc[0]
    lossy_train.loc[lossy_train["recording_key"].eq(train_key), "compare__f0"] = 1.25
    with pytest.raises(ValueError, match="malformed or lossy"):
        _run(manifest, lossy_train, feature_contract=contract)


def test_numeric_coercion_flags_integer_precision_loss_and_malformed_content() -> None:
    from covid_audio_btp.hst_comparators import _coerce_declared_numeric

    values = pd.Series(
        [np.uint64(9_007_199_254_740_993), pd.NA, "not-a-number"],
        dtype=object,
    )
    numeric, invalid = _coerce_declared_numeric(values, "uint64")

    assert invalid.tolist() == [True, False, True]
    assert numeric.isna().all()

    float_values = pd.Series(
        [
            np.uint64(9_007_199_254_740_993),
            9_007_199_254_740_993,
            "0.1",
            "0.5",
            "NaN",
            pd.NA,
            np.nan,
        ],
        dtype=object,
    )
    numeric, invalid = _coerce_declared_numeric(float_values, "float64")
    assert invalid.tolist() == [True, True, True, False, True, False, False]
    assert numeric.iloc[3] == 0.5
    assert numeric.iloc[[0, 1, 2, 4, 5, 6]].isna().all()


def test_lossy_values_fail_even_when_feature_would_not_be_selected() -> None:
    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    features["constant_train"] = features["constant_train"].astype(object)
    held_out_index = features.index[
        features["recording_key"].eq(
            manifest.loc[manifest["split"].eq("test"), "recording_key"].iloc[0]
        )
    ][0]
    features.at[held_out_index, "constant_train"] = "0.1"

    with pytest.raises(ValueError, match="malformed or lossy.*constant_train"):
        _run(manifest, features, feature_contract=contract)


def test_fixed_model_bank_thresholds_and_participant_aggregation_are_frozen() -> None:
    _RecordingEstimator.fit_rows.clear()
    manifest = _manifest()
    result = _run(manifest, _features(manifest), run_id="run-7")

    expected_models = {
        *MODEL_NAMES,
        "top_4_validation_ensemble",
        SELECTED_CANDIDATE_NAME,
    }
    assert set(result.predictions["model"]) == expected_models
    assert set(result.participant_predictions["model"]) == expected_models
    assert set(result.model_audit["model"]) == expected_models
    ensemble = result.model_audit[result.model_audit["model"].eq("top_4_validation_ensemble")].iloc[0]
    assert ensemble["ensemble_members"].split("|") == list(MODEL_NAMES)
    assert ensemble["requested_ensemble_cap"] == 5
    assert ensemble["effective_ensemble_size"] == 4
    assert set(result.model_audit["threshold_source"]) == {
        "validation_participant_balanced_accuracy"
    }
    selected = result.model_audit[
        result.model_audit["model"].eq(SELECTED_CANDIDATE_NAME)
    ].iloc[0]
    assert selected["selected_candidate_source_model"] == "catboost_smote_f80"
    assert selected["candidate_selection_split"] == "validation"
    assert selected["candidate_selection_primary_metric"] == "auroc"
    assert selected["candidate_selection_tiebreak_metric"] == "auprc"
    assert selected["candidate_selection_final_tiebreak"] == "model_name_ascending"
    for table in (result.predictions, result.participant_predictions, result.metrics):
        roles = table.groupby("model")["comparator_endpoint_role"].first().to_dict()
        assert roles[SELECTED_CANDIDATE_NAME] == "primary_validation_selected_endpoint"
        for model_name in (*MODEL_NAMES, "top_4_validation_ensemble"):
            assert roles[model_name] == "secondary_prespecified_model_bank"
        assert table["test_selection_use"].eq(False).all()  # noqa: E712
        assert table["held_out_evaluation_policy"].eq(
            "single_nonadaptive_pass_after_validation_freeze"
        ).all()

    audit_roles = result.model_audit.set_index("model")["comparator_endpoint_role"]
    assert audit_roles.loc[SELECTED_CANDIDATE_NAME] == "primary_validation_selected_endpoint"
    assert set(audit_roles.drop(SELECTED_CANDIDATE_NAME)) == {
        "secondary_prespecified_model_bank"
    }
    assert result.candidate_selection["test_selection_use"].eq(False).all()  # noqa: E712

    selected_predictions = result.participant_predictions[
        result.participant_predictions["model"].eq(SELECTED_CANDIDATE_NAME)
    ].sort_values(["split", "participant_key"], kind="mergesort")
    source_predictions = result.participant_predictions[
        result.participant_predictions["model"].eq("catboost_smote_f80")
    ].sort_values(["split", "participant_key"], kind="mergesort")
    pd.testing.assert_series_equal(
        selected_predictions["probability"].reset_index(drop=True),
        source_predictions["probability"].reset_index(drop=True),
        check_names=False,
    )

    train_recordings = tuple(
        manifest.loc[manifest["split"].eq("train"), "recording_key"].sort_values()
    )
    assert len(_RecordingEstimator.fit_rows) == 4
    assert all(tuple(sorted(rows)) == train_recordings for _, rows in _RecordingEstimator.fit_rows)

    participant = result.participant_predictions
    te1 = participant[
        participant["participant_key"].eq("coswara::te1")
        & participant["model"].eq("lightgbm_smote_f80")
    ].iloc[0]
    recordings = result.predictions[
        result.predictions["participant_key"].eq("coswara::te1")
        & result.predictions["model"].eq("lightgbm_smote_f80")
    ]
    assert te1["n_recordings"] == 3
    assert te1["probability"] == pytest.approx(recordings["probability"].mean())
    assert set(result.metrics["analysis_unit"]) == {"participant"}
    assert set(result.predictions["run_id"]) == {"run-7"}
    assert set(result.predictions["cohort"]) == {"all_eligible"}
    assert set(result.predictions["manifest_sha256"]) == {
        manifest["manifest_sha256"].iloc[0]
    }


def test_test_mode_marks_every_primary_table_and_confirmatory_guard_rejects_it() -> None:
    from covid_audio_btp.hst_comparators import assert_confirmatory_comparator_table

    result = _run(_manifest(), _features(_manifest()))
    tables = (
        result.predictions,
        result.participant_predictions,
        result.metrics,
        result.alignment_audit,
        result.feature_selection,
        result.model_audit,
        result.candidate_selection,
    )
    for table in tables:
        assert table["execution_class"].eq("exploratory_test_only").all()
        assert table["confirmatory_eligible"].eq(False).all()  # noqa: E712
        assert table["test_mode"].eq(True).all()  # noqa: E712
        assert table["reporting_guard"].eq(
            "EXPLORATORY_TEST_MODE_DO_NOT_USE_AS_CONFIRMATORY"
        ).all()
        with pytest.raises(TypeError, match="file path|DataFrame"):
            assert_confirmatory_comparator_table(table)


def test_confirmatory_ingestion_requires_checksummed_generation_not_dataframe(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_comparators import assert_confirmatory_comparator_table

    manifest = _manifest()
    result = _run(manifest, _features(manifest), audit_dir=tmp_path / "audit")
    with pytest.raises(TypeError, match="file path|DataFrame"):
        assert_confirmatory_comparator_table(result.metrics)

    receipt = json.loads((tmp_path / "audit" / "current.json").read_text(encoding="ascii"))
    generation = tmp_path / "audit" / "generations" / receipt["generation_id"]
    with pytest.raises(ValueError, match="exploratory|test.mode|confirmatory"):
        assert_confirmatory_comparator_table(
            generation / "comparator_metrics.csv",
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=tmp_path / "audit" / "current.json",
            approval_record_path=tmp_path / "unavailable.approved.json",
            trusted_project_repository_root=tmp_path,
            accepted_freezes_path=tmp_path / "unavailable-freezes.json",
            expected_accepted_freezes_sha256="b" * 64,
            runtime_random_state=42,
        )


def test_test_values_cannot_change_validation_threshold_or_ensemble_membership() -> None:
    manifest = _manifest()
    features = _features(manifest)
    first = _run(manifest, features)
    changed = features.copy()
    test_keys = set(manifest.loc[manifest["split"].eq("test"), "recording_key"])
    changed.loc[changed["recording_key"].isin(test_keys), "compare__f0"] *= -1000.0
    second = _run(manifest, changed)

    columns = [
        "model",
        "threshold",
        "threshold_source",
        "ensemble_members",
        "selected_candidate_source_model",
    ]
    pd.testing.assert_frame_equal(
        first.model_audit[columns].sort_values("model").reset_index(drop=True),
        second.model_audit[columns].sort_values("model").reset_index(drop=True),
    )


def test_generation_metric_verification_rejects_incomplete_required_schema(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    audit_dir = tmp_path / "audit"
    _run(manifest, _features(manifest), audit_dir=audit_dir)
    receipt = json.loads((audit_dir / "current.json").read_text(encoding="ascii"))
    generation = audit_dir / "generations" / receipt["generation_id"]
    metrics_path = generation / "comparator_metrics.csv"
    metrics = pd.read_csv(metrics_path).drop(columns="auroc")
    _replace_read_only_csv(metrics, metrics_path)

    with pytest.raises(ValueError, match="required metric schema.*auroc"):
        comparators._verify_generation_metrics(generation)


def test_generation_verification_recomputes_validation_selected_candidate(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    audit_dir = tmp_path / "audit"
    result = _run(manifest, _features(manifest), audit_dir=audit_dir)
    receipt = json.loads((audit_dir / "current.json").read_text(encoding="ascii"))
    generation = audit_dir / "generations" / receipt["generation_id"]
    selection_path = generation / "comparator_candidate_selection.csv"
    selection = pd.read_csv(selection_path)
    selected_source = result.model_audit.loc[
        result.model_audit["model"].eq(SELECTED_CANDIDATE_NAME),
        "selected_candidate_source_model",
    ].iloc[0]
    replacement = next(
        model for model in (*MODEL_NAMES, "top_4_validation_ensemble")
        if model != selected_source
    )
    selection["selected"] = selection["candidate_model"].eq(replacement)
    selection["selected_candidate_source_model"] = replacement
    _replace_read_only_csv(selection, selection_path)

    with pytest.raises(ValueError, match="candidate selection.*validation predictions"):
        comparators._verify_generation_metrics(generation)


def test_validation_selected_candidate_is_frozen_before_test_predictions_change() -> None:
    manifest = _manifest()
    features = _features(manifest)
    first = _run(manifest, features)
    changed = features.copy()
    test_keys = set(manifest.loc[manifest["split"].eq("test"), "recording_key"])
    changed.loc[changed["recording_key"].isin(test_keys), "compare__f0"] *= -1000.0
    second = _run(manifest, changed)

    first_selection = first.model_audit.loc[
        first.model_audit["model"].eq(SELECTED_CANDIDATE_NAME),
        [
            "selected_candidate_source_model",
            "candidate_selection_validation_auroc",
            "candidate_selection_validation_auprc",
        ],
    ].reset_index(drop=True)
    second_selection = second.model_audit.loc[
        second.model_audit["model"].eq(SELECTED_CANDIDATE_NAME),
        [
            "selected_candidate_source_model",
            "candidate_selection_validation_auroc",
            "candidate_selection_validation_auprc",
        ],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_selection, second_selection)


def test_same_manifest_parity_and_alignment_audit_reject_mismatch() -> None:
    from covid_audio_btp.hst_comparators import audit_comparator_alignment

    manifest = _manifest()
    result = _run(manifest, _features(manifest))
    comparator = result.predictions[result.predictions["model"].eq(MODEL_NAMES[0])].copy()
    hst = comparator.copy()
    hst["model"] = "hst"
    hst["checkpoint_hash"] = "b" * 64

    audit = audit_comparator_alignment(hst, comparator)
    assert audit["aligned"].all()
    assert audit["n_recordings"].sum() == len(comparator)

    wrong_fold = comparator.copy()
    wrong_fold["fold"] = 1
    with pytest.raises(ValueError, match="context|cohort"):
        audit_comparator_alignment(hst, wrong_fold)

    missing = comparator.iloc[:-1].copy()
    with pytest.raises(ValueError, match="cohort"):
        audit_comparator_alignment(hst, missing)

    with pytest.raises(ValueError, match="recording_key"):
        audit_comparator_alignment(hst.drop(columns="recording_key"), comparator)

    with pytest.raises(ValueError, match="cohort"):
        audit_comparator_alignment(hst.drop(columns="cohort"), comparator)


def test_rejects_manifest_leakage_feature_mismatch_and_cross_fold_rows() -> None:
    manifest = _manifest()
    features = _features(manifest)

    leaked = manifest.copy()
    leaked.loc[leaked["participant_key"].eq("coswara::tr0"), "split"] = ["train"] * (
        leaked["participant_key"].eq("coswara::tr0").sum()
    )
    duplicate = leaked[leaked["participant_key"].eq("coswara::tr0")].iloc[[0]].copy()
    duplicate["recording_key"] = "coswara::tr0_leaked"
    duplicate["split"] = "test"
    leaked = _refreeze(pd.concat([leaked, duplicate], ignore_index=True))
    leaked_features = pd.concat(
        [features, features.iloc[[0]].assign(recording_key="coswara::tr0_leaked")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="participant leakage"):
        _run(leaked, leaked_features)

    content_leaked = manifest.copy()
    train_index = content_leaked.index[content_leaked["split"].eq("train")][0]
    test_index = content_leaked.index[content_leaked["split"].eq("test")][0]
    content_leaked.loc[test_index, "source_audio_sha256"] = content_leaked.loc[
        train_index, "source_audio_sha256"
    ]
    content_leaked = _refreeze(content_leaked)
    with pytest.raises(ValueError, match="content leakage"):
        _run(content_leaked, features)

    with pytest.raises(ValueError, match="missing manifest recordings"):
        _run(manifest, features.iloc[:-1].copy())

    global_table = features.copy()
    global_table["protocol"] = "stale_global_value"
    global_table["fold"] = 99
    global_table["split"] = "unused"
    extra = global_table.iloc[[0]].copy()
    extra["participant_key"] = "coswara::outside"
    extra["recording_key"] = "coswara::outside_r0"
    global_table = pd.concat([global_table, extra], ignore_index=True)
    result = _run(manifest, global_table, feature_contract=_feature_contract(features))
    assert set(result.alignment_audit["excluded_global_row_count"]) == {1}

    duplicated_features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _run(manifest, duplicated_features, feature_contract=_feature_contract(features))

    conflicting_features = features.copy()
    conflicting_features.loc[0, "label_binary"] = "positive"
    with pytest.raises(ValueError, match="labels conflict"):
        _run(manifest, conflicting_features, feature_contract=_feature_contract(features))


def test_confirmatory_mode_requires_top800_full_schema_and_real_factories(tmp_path: Path) -> None:
    from covid_audio_btp.hst_comparators import run_aligned_compare_is10

    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    with pytest.raises(ValueError, match="confirmatory.*800"):
        run_aligned_compare_is10(
            features,
            manifest,
            feature_contract=contract,
            selected_feature_k=3,
            audit_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="test_mode"):
        run_aligned_compare_is10(
            features,
            manifest,
            feature_contract=contract,
            selected_feature_k=800,
            ranker_factory=lambda random_state: _DeterministicRanker(),
            estimator_factory=lambda model_name, random_state: _RecordingEstimator(model_name),
            audit_dir=tmp_path,
        )

    preselected = _full_feature_table(manifest, n_features=800)
    full_contract = _feature_contract_for_columns(
        preselected, tuple(f"f{index:04d}" for index in range(800))
    )
    with pytest.raises(ValueError, match="full.*schema|preselected"):
        run_aligned_compare_is10(
            preselected,
            manifest,
            feature_contract=full_contract,
            approval_record_path=tmp_path / "unused.approved.json",
            selected_feature_k=800,
            audit_dir=tmp_path,
        )

    full = _full_feature_table(manifest, n_features=801)
    full_contract = _feature_contract_for_columns(
        full, tuple(f"f{index:04d}" for index in range(801))
    )
    with pytest.raises(ValueError, match="ensemble.*exactly 5|frozen.*cap"):
        run_aligned_compare_is10(
            full,
            manifest,
            feature_contract=full_contract,
            approval_record_path=tmp_path / "unused.approved.json",
            trusted_project_repository_root=tmp_path,
            accepted_freezes_path=tmp_path / "unavailable-freezes.json",
            expected_accepted_freezes_sha256="b" * 64,
            ensemble_top_k=6,
            audit_dir=tmp_path / "audit",
        )


def _feature_contract_for_columns(features: pd.DataFrame, columns: tuple[str, ...]):
    from covid_audio_btp.hst_comparators import build_compare_is10_feature_contract

    return build_compare_is10_feature_contract(features, ordered_feature_columns=columns)


def _feature_artifact_sha256(features: pd.DataFrame) -> str:
    from covid_audio_btp.hst_comparators import compare_is10_feature_artifact_sha256

    return compare_is10_feature_artifact_sha256(features)


def test_feature_artifact_hash_streams_exact_typed_raw_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_comparators import compare_is10_feature_artifact_sha256

    identity = _features(_manifest()).iloc[:2].copy()
    identity["uint_feature"] = pd.Series(
        [np.uint64(9_007_199_254_740_993), np.uint64(18_446_744_073_709_551_615)],
        dtype="uint64",
    )
    identity["float_feature"] = np.array([0.0, -0.0], dtype=np.float64)
    identity["object_missing"] = pd.Series([None, pd.NA], dtype=object)
    identity["nullable_float"] = pd.Series([1.0, pd.NA], dtype="Float64")
    nan_bits = np.array(
        [0x7FF8000000000001, 0x7FF8000000000002], dtype=np.uint64
    )
    identity["nan_payload"] = nan_bits.view(np.float64)

    monkeypatch.setattr(
        pd.util,
        "hash_pandas_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("64-bit pandas hash must not be used")
        ),
    )
    baseline = compare_is10_feature_artifact_sha256(identity)

    signed_zero_changed = identity.copy()
    signed_zero_changed.loc[signed_zero_changed.index[1], "float_feature"] = 0.0
    assert compare_is10_feature_artifact_sha256(signed_zero_changed) != baseline

    uint_changed = identity.copy()
    uint_changed.loc[uint_changed.index[0], "uint_feature"] = np.uint64(
        9_007_199_254_740_994
    )
    assert compare_is10_feature_artifact_sha256(uint_changed) != baseline

    nan_changed = identity.copy()
    nan_changed["nan_payload"] = np.array(
        [0x7FF8000000000002, 0x7FF8000000000002], dtype=np.uint64
    ).view(np.float64)
    assert compare_is10_feature_artifact_sha256(nan_changed) != baseline

    missing_changed = identity.copy()
    missing_changed["object_missing"] = pd.Series([pd.NA, pd.NA], dtype=object)
    assert compare_is10_feature_artifact_sha256(missing_changed) != baseline


def _full_feature_table(manifest: pd.DataFrame, n_features: int = 801) -> pd.DataFrame:
    identity = _features(manifest)[
        [
            "dataset",
            "participant_key",
            "recording_key",
            "modality",
            "label_binary",
            "source_audio_sha256",
        ]
    ].copy()
    row = np.arange(len(identity), dtype=float)
    payload = pd.DataFrame(
        {
            f"f{index:04d}": row + float(index) / 10_000.0
            for index in range(n_features)
        },
        index=identity.index,
    )
    return pd.concat([identity, payload], axis=1)


def _approval_protocol_binding_sha256(manifest: pd.DataFrame) -> str:
    columns = [
        "protocol",
        "fold",
        "cohort",
        "split",
        "dataset",
        "participant_key",
        "recording_key",
        "modality",
        "label_binary",
        "source_audio_sha256",
        "analysis_scope",
        "analysis_role",
        "estimand_id",
        "multiplicity_family",
        "analysis_mode",
        "confirmatory_protocol",
    ]
    ordered = manifest[columns].astype(str).sort_values(columns, kind="mergesort")
    records = ordered.to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("ascii")).hexdigest()


def test_approval_protocol_binding_includes_live_analysis_provenance() -> None:
    from covid_audio_btp.hst_comparators import _approval_protocol_binding_sha256

    manifest = _manifest()
    changed = manifest.copy()
    changed.loc[0, "multiplicity_family"] = "different_family"
    assert _approval_protocol_binding_sha256(changed) != _approval_protocol_binding_sha256(
        manifest
    )


def _write_frozen_approval_record(
    path: Path,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    contract: object,
) -> None:
    payload = {
        "approval_record_version": 1,
        "approval_status": "approved",
        "approval_id": "test-independent-freeze",
        "approved_at_utc": "2026-08-02T00:00:00Z",
        "feature_schema_sha256": contract.schema_sha256,
        "feature_artifact_sha256": _feature_artifact_sha256(features),
        "manifest_sha256": str(manifest["manifest_sha256"].iloc[0]),
        "scientific_configuration_fingerprint": str(
            manifest["scientific_configuration_fingerprint"].iloc[0]
        ),
        "eligibility_alignment_fingerprint": str(
            manifest["eligibility_alignment_fingerprint"].iloc[0]
        ),
        "protocol_binding_sha256": _approval_protocol_binding_sha256(manifest),
        "comparator_configuration": {
            "selected_feature_k": 800,
            "ranker": "lightgbm",
            "selection_scope": "per_modality_mean",
            "model_names": list(MODEL_NAMES),
            "ensemble_policy": "uniform_probability_mean",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["approval_record_sha256"] = sha256(canonical.encode("ascii")).hexdigest()
    path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"))


def _commit_approval_record(path: Path) -> None:
    root = path.parents[1] if path.parent.name == "freeze" else path.parent
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "approval-test@example.invalid"),
        ("git", "config", "user.name", "Approval Test"),
        ("git", "add", path.relative_to(root).as_posix()),
        ("git", "commit", "-q", "-m", "Freeze comparator approval"),
    )
    for command in commands:
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)


def _git(tmp_path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _trusted_project_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    contract: object,
) -> tuple[Path, Path, str]:
    import covid_audio_btp.hst_comparators as comparators

    root = tmp_path / "trusted-project"
    source_root = root / "covid_audio_btp" / "src" / "covid_audio_btp"
    config_root = root / "covid_audio_btp" / "configs"
    source_root.mkdir(parents=True)
    config_root.mkdir(parents=True)
    live_source_root = Path(comparators.__file__).resolve().parent
    for name in (
        "hst_comparators.py",
        "hst_data_contracts.py",
        "hst_protocols.py",
        "metrics.py",
        "strong_baseline.py",
    ):
        shutil.copy2(live_source_root / name, source_root / name)
    shutil.copy2(
        live_source_root.parents[1] / "requirements-hst.txt",
        root / "covid_audio_btp" / "requirements-hst.txt",
    )
    approval_path = config_root / "hst_compare_is10_approval.approved.json"
    _write_frozen_approval_record(approval_path, manifest, features, contract)

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "approval-test@example.invalid")
    _git(root, "config", "user.name", "Approval Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "Accepted comparator freeze")
    accepted_commit = _git(root, "rev-parse", "HEAD")
    os.chmod(approval_path, stat.S_IREAD)
    monkeypatch.setattr(comparators, "__file__", str(source_root / "hst_comparators.py"))
    return root, approval_path, accepted_commit


def _trusted_project_repository_with_complete_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    contract: object,
) -> tuple[Path, Path, str, Path, str]:
    import covid_audio_btp.hst_comparators as comparators

    root = tmp_path / "trusted-complete-project"
    source_root = root / "covid_audio_btp" / "src" / "covid_audio_btp"
    config_root = root / "covid_audio_btp" / "configs"
    source_root.mkdir(parents=True)
    config_root.mkdir(parents=True)
    live_source_root = Path(comparators.__file__).resolve().parent
    for name in (
        "hst_comparators.py",
        "hst_data_contracts.py",
        "hst_protocols.py",
        "metrics.py",
        "strong_baseline.py",
    ):
        shutil.copy2(live_source_root / name, source_root / name)
    requirements_path = root / "covid_audio_btp" / "requirements-hst.txt"
    shutil.copy2(live_source_root.parents[1] / "requirements-hst.txt", requirements_path)
    shutil.copy2(
        live_source_root.parents[1] / "requirements-gpu.txt",
        root / "covid_audio_btp" / "requirements-gpu.txt",
    )
    monkeypatch.setattr(comparators, "__file__", str(source_root / "hst_comparators.py"))
    locked_versions = {
        "numpy": "2.4.6",
        "scipy": "1.17.0",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "imbalanced-learn": "0.14.2",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
        "lightgbm": "4.6.0",
        "xgboost": "3.2.0",
        "catboost": "1.2.10",
    }
    monkeypatch.setattr(
        comparators.importlib_metadata,
        "version",
        lambda distribution: locked_versions[distribution],
    )

    environment_lock_sha256 = sha256(requirements_path.read_bytes()).hexdigest()
    executable_recipe = comparators._build_compare_is10_executable_recipe(
        root,
        random_state=42,
        accepted_environment_lock_sha256=environment_lock_sha256,
    )
    approval_path = config_root / "hst_compare_is10_approval.approved.json"
    payload = {
        "approval_record_version": 2,
        "approval_status": "approved",
        "approval_id": "test-independent-complete-recipe-freeze",
        "approved_at_utc": "2026-08-02T00:00:00Z",
        "feature_schema_sha256": contract.schema_sha256,
        "feature_artifact_sha256": _feature_artifact_sha256(features),
        "manifest_sha256": str(manifest["manifest_sha256"].iloc[0]),
        "scientific_configuration_fingerprint": str(
            manifest["scientific_configuration_fingerprint"].iloc[0]
        ),
        "eligibility_alignment_fingerprint": str(
            manifest["eligibility_alignment_fingerprint"].iloc[0]
        ),
        "protocol_binding_sha256": _approval_protocol_binding_sha256(manifest),
        "comparator_configuration": {
            "selected_feature_k": 800,
            "ranker": "lightgbm",
            "selection_scope": "per_modality_mean",
            "model_names": list(MODEL_NAMES),
            "ensemble_policy": "uniform_probability_mean",
        },
        "executable_recipe": executable_recipe,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["approval_record_sha256"] = sha256(canonical.encode("ascii")).hexdigest()
    approval_path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    )

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "approval-test@example.invalid")
    _git(root, "config", "user.name", "Approval Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "Accepted complete comparator freeze")
    accepted_commit = _git(root, "rev-parse", "HEAD")
    os.chmod(approval_path, stat.S_IREAD)
    accepted_path, expected_hash = _install_authenticated_accepted_freezes(
        root, approval_path, accepted_commit, environment_lock_sha256
    )
    return root, approval_path, accepted_commit, accepted_path, expected_hash


def _install_authenticated_accepted_freezes(
    root: Path,
    approval_path: Path,
    accepted_commit: str,
    environment_lock_sha256: str,
) -> tuple[Path, str]:
    remote_url = "https://example.invalid/covid-rars.git"
    _git(root, "remote", "add", "origin", remote_url)
    accepted_path = (
        root
        / "covid_audio_btp"
        / "configs"
        / "hst_comparator_accepted_freezes.approved.json"
    )
    payload = {
        "accepted_freezes_version": 1,
        "project_identity": {
            "project_id": "covid-rars-test",
            "expected_remote_url": remote_url,
            "accepted_ancestor_commit": accepted_commit,
        },
        "compare_is10_approval": {
            "relative_path": approval_path.relative_to(root).as_posix(),
            "commit_sha": accepted_commit,
        },
        "environment_lock": {
            "relative_path": "covid_audio_btp/requirements-hst.txt",
            "sha256": environment_lock_sha256,
        },
        "accepted_generation_manifests": {},
    }
    accepted_path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    )
    _git(root, "add", accepted_path.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "Authenticate comparator accepted freezes")
    os.chmod(accepted_path, stat.S_IREAD)
    return accepted_path, sha256(accepted_path.read_bytes()).hexdigest()


def _accept_generation(
    root: Path,
    accepted_path: Path,
    generation_manifest_path: Path,
) -> str:
    manifest_bytes = generation_manifest_path.read_bytes()
    generation = json.loads(manifest_bytes.decode("ascii"))
    accepted = json.loads(accepted_path.read_text(encoding="ascii"))
    accepted["accepted_generation_manifests"][generation["generation_id"]] = sha256(
        manifest_bytes
    ).hexdigest()
    os.chmod(accepted_path, stat.S_IWRITE | stat.S_IREAD)
    accepted_path.write_bytes(
        (json.dumps(accepted, indent=2, sort_keys=True) + "\n").encode("ascii")
    )
    _git(root, "add", accepted_path.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", f"Accept generation {generation['generation_id']}")
    os.chmod(accepted_path, stat.S_IREAD)
    return sha256(accepted_path.read_bytes()).hexdigest()


def _replace_read_only_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    path.write_bytes(payload)
    os.chmod(path, stat.S_IREAD)


def _replace_read_only_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    frame.to_csv(path, index=False)
    os.chmod(path, stat.S_IREAD)


def _write_current_receipt(audit_dir: Path, generation_manifest_path: Path) -> None:
    generation = json.loads(generation_manifest_path.read_text(encoding="ascii"))
    receipt = {
        "generation_id": generation["generation_id"],
        "generation_manifest_sha256": sha256(
            generation_manifest_path.read_bytes()
        ).hexdigest(),
    }
    receipt["receipt_sha256"] = sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    _replace_read_only_bytes(
        audit_dir / "current.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii")
    )


def test_confirmatory_trust_requires_authenticated_canonical_accepted_freezes_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    features = _full_feature_table(manifest)
    columns = tuple(f"f{index:04d}" for index in range(801))
    contract = _feature_contract_for_columns(features, columns)
    root, approval_path, accepted_commit, accepted_path, expected_hash = (
        _trusted_project_repository_with_complete_recipe(
            tmp_path, monkeypatch, manifest, features, contract
        )
    )

    approval = comparators.load_frozen_compare_is10_approval(
        approval_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=expected_hash,
        runtime_random_state=42,
        feature_contract=contract,
        feature_artifact_sha256=_feature_artifact_sha256(features),
        manifest=manifest,
    )
    assert approval["approval_git_commit"] == accepted_commit

    with pytest.raises(ValueError, match="accepted-freezes.*hash"):
        comparators.load_frozen_compare_is10_approval(
            approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256="0" * 64,
            runtime_random_state=42,
            feature_contract=contract,
            feature_artifact_sha256=_feature_artifact_sha256(features),
            manifest=manifest,
        )

    _git(root, "remote", "set-url", "origin", "https://example.invalid/attacker.git")
    with pytest.raises(ValueError, match="remote|project identity"):
        comparators.load_frozen_compare_is10_approval(
            approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
            feature_contract=contract,
            feature_artifact_sha256=_feature_artifact_sha256(features),
            manifest=manifest,
        )


def test_confirmatory_trust_is_anchored_to_project_and_canonical_accepted_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_comparators import load_frozen_compare_is10_approval

    manifest = _manifest()
    features = _full_feature_table(manifest)
    columns = tuple(f"f{index:04d}" for index in range(801))
    contract = _feature_contract_for_columns(features, columns)
    root, approval_path, accepted_commit, accepted_path, expected_hash = (
        _trusted_project_repository_with_complete_recipe(
            tmp_path, monkeypatch, manifest, features, contract
        )
    )

    approval = load_frozen_compare_is10_approval(
        approval_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=expected_hash,
        runtime_random_state=42,
        feature_contract=contract,
        feature_artifact_sha256=_feature_artifact_sha256(features),
        manifest=manifest,
    )
    assert approval["approval_git_commit"] == accepted_commit

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _git(unrelated, "init", "-q")
    with pytest.raises(ValueError, match="trusted project repository"):
        load_frozen_compare_is10_approval(
            approval_path,
            trusted_project_repository_root=unrelated,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
            feature_contract=contract,
            feature_artifact_sha256=_feature_artifact_sha256(features),
            manifest=manifest,
        )

    alternate = approval_path.with_name("caller-created.approved.json")
    shutil.copy2(approval_path, alternate)
    os.chmod(alternate, stat.S_IREAD)
    with pytest.raises(ValueError, match="canonical approval path"):
        load_frozen_compare_is10_approval(
            alternate,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
            feature_contract=contract,
            feature_artifact_sha256=_feature_artifact_sha256(features),
            manifest=manifest,
        )


def test_confirmatory_approval_freezes_complete_executable_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    features = _full_feature_table(manifest)
    columns = tuple(f"f{index:04d}" for index in range(801))
    contract = _feature_contract_for_columns(features, columns)
    root, approval_path, accepted_commit, accepted_path, expected_hash = (
        _trusted_project_repository_with_complete_recipe(
            tmp_path, monkeypatch, manifest, features, contract
        )
    )
    arguments = {
        "trusted_project_repository_root": root,
        "accepted_freezes_path": accepted_path,
        "expected_accepted_freezes_sha256": expected_hash,
        "feature_contract": contract,
        "feature_artifact_sha256": _feature_artifact_sha256(features),
        "manifest": manifest,
    }

    approval = comparators.load_frozen_compare_is10_approval(
        approval_path,
        runtime_random_state=42,
        **arguments,
    )
    recipe = approval["executable_recipe"]
    assert recipe["random_state"] == 42
    assert recipe["model_names"] == list(MODEL_NAMES)
    assert recipe["selected_candidate_model"] == SELECTED_CANDIDATE_NAME
    assert recipe["selected_candidate_pool"] == [
        *MODEL_NAMES,
        "top_4_validation_ensemble",
    ]
    assert recipe["selected_candidate_policy"] == {
        "selection_split": "validation",
        "primary_metric": "auroc",
        "tiebreak_metric": "auprc",
        "final_tiebreak": "model_name_ascending",
    }
    assert set(recipe["model_hyperparameters"]) == set(MODEL_NAMES)
    for model_name, specification in recipe["model_hyperparameters"].items():
        preprocessing = specification["preprocessing"]
        assert "imputer" not in preprocessing
        assert preprocessing["variance_threshold"] == 0.0
        assert preprocessing["score_function"] == "strong_baseline._safe_f_classif"
    assert set(recipe["executable_source_sha256"]) == {
        "hst_comparators.py",
        "hst_data_contracts.py",
        "hst_protocols.py",
        "metrics.py",
        "strong_baseline.py",
    }
    assert recipe["environment_lock_sha256"] == sha256(
        (root / "covid_audio_btp" / "requirements-hst.txt").read_bytes()
    ).hexdigest()
    assert recipe["package_versions"]["scipy"] == "1.17.0"
    assert {"joblib", "threadpoolctl"} <= set(recipe["package_versions"])
    assert recipe["python_runtime"]["implementation"]
    assert recipe["python_runtime"]["version"]
    assert recipe["python_runtime"]["cache_tag"]
    assert set(recipe["thread_environment"]) == {
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
    assert recipe["blas_runtime_sha256"] == sha256(
        json.dumps(
            recipe["blas_runtime"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="random_state|recipe"):
        comparators.load_frozen_compare_is10_approval(
            approval_path,
            runtime_random_state=43,
            **arguments,
        )

    monkeypatch.setenv("OMP_NUM_THREADS", "987")
    with pytest.raises(ValueError, match="environment|recipe"):
        comparators.load_frozen_compare_is10_approval(
            approval_path,
            runtime_random_state=42,
            **arguments,
        )
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    original_ranker = comparators._lightgbm_ranker
    monkeypatch.setattr(comparators, "_lightgbm_ranker", lambda seed: original_ranker(seed))
    with pytest.raises(ValueError, match="factory|recipe|executable"):
        comparators.load_frozen_compare_is10_approval(
            approval_path,
            runtime_random_state=42,
            **arguments,
        )
    monkeypatch.setattr(comparators, "_lightgbm_ranker", original_ranker)

    original_estimator = comparators._default_estimator_factory
    monkeypatch.setattr(
        comparators,
        "_default_estimator_factory",
        lambda model_name, seed: original_estimator(model_name, seed),
    )
    with pytest.raises(ValueError, match="factory|recipe|executable"):
        comparators.load_frozen_compare_is10_approval(
            approval_path,
            runtime_random_state=42,
            **arguments,
        )

def test_confirmatory_approval_must_be_independent_read_only_and_fully_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_comparators import (
        load_frozen_compare_is10_approval,
        run_aligned_compare_is10,
    )

    manifest = _manifest()
    features = _full_feature_table(manifest)
    columns = tuple(f"f{index:04d}" for index in range(801))
    contract = _feature_contract_for_columns(features, columns)

    with pytest.raises(TypeError, match="approved_feature_schema_sha256"):
        run_aligned_compare_is10(
            features,
            manifest,
            feature_contract=contract,
            approved_feature_schema_sha256=contract.schema_sha256,
            approved_feature_artifact_sha256=_feature_artifact_sha256(features),
            selected_feature_k=800,
            audit_dir=tmp_path / "audit",
        )

    root, approval_path, accepted_commit, accepted_path, expected_hash = (
        _trusted_project_repository_with_complete_recipe(
            tmp_path, monkeypatch, manifest, features, contract
        )
    )
    common = {
        "trusted_project_repository_root": root,
        "accepted_freezes_path": accepted_path,
        "expected_accepted_freezes_sha256": expected_hash,
        "runtime_random_state": 42,
        "feature_contract": contract,
        "feature_artifact_sha256": _feature_artifact_sha256(features),
        "manifest": manifest,
    }
    approval = load_frozen_compare_is10_approval(
        approval_path,
        **common,
    )
    assert approval["approval_id"] == "test-independent-complete-recipe-freeze"

    os.chmod(approval_path, stat.S_IWRITE | stat.S_IREAD)
    with pytest.raises(ValueError, match="immutable|read-only"):
        load_frozen_compare_is10_approval(approval_path, **common)
    os.chmod(approval_path, stat.S_IREAD)

    changed = features.copy()
    changed.loc[0, "f0000"] += 1.0
    with pytest.raises(ValueError, match="feature_artifact_sha256"):
        load_frozen_compare_is10_approval(
            approval_path,
            **{**common, "feature_artifact_sha256": _feature_artifact_sha256(changed)},
        )


def test_confirmatory_generation_carries_approval_binding_in_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    features = _full_feature_table(manifest)
    columns = tuple(f"f{index:04d}" for index in range(801))
    contract = _feature_contract_for_columns(features, columns)
    monkeypatch.setattr(
        comparators,
        "_lightgbm_ranker",
        lambda random_state: _DeterministicRanker(),
    )
    monkeypatch.setattr(
        comparators,
        "_default_estimator_factory",
        lambda model_name, random_state: _RecordingEstimator(model_name),
    )
    root, approval_path, accepted_commit, accepted_path, expected_hash = (
        _trusted_project_repository_with_complete_recipe(
            tmp_path, monkeypatch, manifest, features, contract
        )
    )
    approval = json.loads(approval_path.read_text(encoding="ascii"))
    audit_dir = tmp_path / "audit"
    result = comparators.run_aligned_compare_is10(
        features,
        manifest,
        feature_contract=contract,
        approval_record_path=approval_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=expected_hash,
        selected_feature_k=800,
        audit_dir=audit_dir,
    )

    for table in (
        result.predictions,
        result.participant_predictions,
        result.metrics,
        result.alignment_audit,
        result.feature_selection,
        result.model_audit,
        result.candidate_selection,
    ):
        assert table["approval_id"].eq(approval["approval_id"]).all()
        assert table["approval_record_sha256"].eq(
            approval["approval_record_sha256"]
        ).all()
        assert table["approval_git_commit"].astype(str).str.fullmatch(r"[0-9a-f]{40,64}").all()
        assert table["approval_git_blob"].astype(str).str.fullmatch(r"[0-9a-f]{40,64}").all()
        assert table["executable_recipe_sha256"].astype(str).str.fullmatch(
            r"[0-9a-f]{64}"
        ).all()

    receipt = json.loads((audit_dir / "current.json").read_text(encoding="ascii"))
    generation = audit_dir / "generations" / receipt["generation_id"]
    generation_manifest = json.loads(
        (generation / "manifest.json").read_text(encoding="ascii")
    )
    with pytest.raises(ValueError, match="accepted-freezes"):
        comparators.assert_confirmatory_comparator_table(
            generation / "comparator_metrics.csv",
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
        )
    expected_hash = _accept_generation(
        root, accepted_path, generation / "manifest.json"
    )
    metrics_path = generation / "comparator_metrics.csv"
    os.chmod(metrics_path, stat.S_IREAD | stat.S_IWRITE)
    with pytest.raises(ValueError, match="generation artifact.*read-only|immutable"):
        comparators.assert_confirmatory_comparator_table(
            metrics_path,
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
        )
    os.chmod(metrics_path, stat.S_IREAD)
    original_manifest_bytes = (generation / "manifest.json").read_bytes()
    original_receipt_bytes = (audit_dir / "current.json").read_bytes()
    incomplete_bank = json.loads(original_manifest_bytes.decode("ascii"))
    incomplete_bank["model_names"] = list(MODEL_NAMES[:-1])
    _replace_read_only_bytes(
        generation / "manifest.json",
        (json.dumps(incomplete_bank, indent=2, sort_keys=True) + "\n").encode("ascii")
    )
    _write_current_receipt(audit_dir, generation / "manifest.json")
    incomplete_hash = _accept_generation(root, accepted_path, generation / "manifest.json")
    with pytest.raises(ValueError, match="model bank|frozen models"):
        comparators.assert_confirmatory_comparator_table(
            generation / "comparator_metrics.csv",
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=incomplete_hash,
            runtime_random_state=42,
        )
    _replace_read_only_bytes(generation / "manifest.json", original_manifest_bytes)
    _replace_read_only_bytes(audit_dir / "current.json", original_receipt_bytes)
    expected_hash = _accept_generation(root, accepted_path, generation / "manifest.json")
    table_paths = {
        "comparator_predictions.csv",
        "comparator_participant_predictions.csv",
        "comparator_metrics.csv",
        "comparator_alignment_audit.csv",
        "comparator_feature_selection.csv",
        "comparator_model_audit.csv",
        "comparator_candidate_selection.csv",
    }
    for relative_path in table_paths:
        loaded = comparators.assert_confirmatory_comparator_table(
            generation / relative_path,
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
        )
        assert not loaded.empty
    metrics_path = generation / "comparator_metrics.csv"
    original_metrics = metrics_path.read_bytes()
    _replace_read_only_bytes(metrics_path, original_metrics + b"\n")
    with pytest.raises(ValueError, match="checksum|size"):
        comparators.assert_confirmatory_comparator_table(
            metrics_path,
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
        )
    _replace_read_only_bytes(metrics_path, original_metrics)
    assert generation_manifest["approval_record_sha256"] == approval[
        "approval_record_sha256"
    ]
    assert generation_manifest["approval_git_commit"] == result.metrics[
        "approval_git_commit"
    ].iloc[0]
    first_model = result.model_audit[result.model_audit["model"].eq(MODEL_NAMES[0])].iloc[0]
    bundle = pickle.loads((generation / first_model["model_artifact"]).read_bytes())
    assert bundle["approval_record_sha256"] == approval["approval_record_sha256"]
    assert bundle["approval_git_commit"] == result.metrics["approval_git_commit"].iloc[0]
    ensemble_row = result.model_audit[
        result.model_audit["model"].eq("top_4_validation_ensemble")
    ].iloc[0]
    verified_ensemble = comparators.load_verified_compare_is10_bundle(
        generation / ensemble_row["model_artifact"],
        generation_manifest_path=generation / "manifest.json",
        current_receipt_path=audit_dir / "current.json",
        approval_record_path=approval_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=expected_hash,
        runtime_random_state=42,
    )
    assert verified_ensemble["verified_generation_manifest_sha256"] == receipt[
        "generation_manifest_sha256"
    ]
    assert set(verified_ensemble["verified_member_bundles"]) == set(MODEL_NAMES)
    assert all(
        member["bundle_version"] == 3
        for member in verified_ensemble["verified_member_bundles"].values()
    )
    selected_row = result.model_audit[
        result.model_audit["model"].eq(SELECTED_CANDIDATE_NAME)
    ].iloc[0]
    verified_selected = comparators.load_verified_compare_is10_bundle(
        generation / selected_row["model_artifact"],
        generation_manifest_path=generation / "manifest.json",
        current_receipt_path=audit_dir / "current.json",
        approval_record_path=approval_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=expected_hash,
        runtime_random_state=42,
    )
    assert set(verified_selected["verified_member_bundles"]) == {
        selected_row["selected_candidate_source_model"]
    }

    tampered_metrics = pd.read_csv(metrics_path)
    tampered_metrics.loc[0, "auroc"] = float(tampered_metrics.loc[0, "auroc"]) + 0.05
    _replace_read_only_csv(tampered_metrics, metrics_path)
    rewritten_manifest = json.loads(
        (generation / "manifest.json").read_text(encoding="ascii")
    )
    rewritten_manifest["files"]["comparator_metrics.csv"] = {
        "sha256": sha256(metrics_path.read_bytes()).hexdigest(),
        "size_bytes": metrics_path.stat().st_size,
    }
    _replace_read_only_bytes(
        generation / "manifest.json",
        (json.dumps(rewritten_manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    )
    rewritten_receipt = {
        "generation_id": receipt["generation_id"],
        "generation_manifest_sha256": sha256(
            (generation / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    rewritten_receipt["receipt_sha256"] = sha256(
        json.dumps(
            rewritten_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    _replace_read_only_bytes(
        audit_dir / "current.json",
        (json.dumps(rewritten_receipt, indent=2, sort_keys=True) + "\n").encode("ascii")
    )
    with pytest.raises(ValueError, match="accepted-freezes"):
        comparators.assert_confirmatory_comparator_table(
            metrics_path,
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
        )

    accepted_tampered_hash = _accept_generation(
        root, accepted_path, generation / "manifest.json"
    )
    with pytest.raises(ValueError, match="recomputation"):
        comparators.assert_confirmatory_comparator_table(
            metrics_path,
            generation_manifest_path=generation / "manifest.json",
            current_receipt_path=audit_dir / "current.json",
            approval_record_path=approval_path,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=accepted_tampered_hash,
            runtime_random_state=42,
        )


def test_frozen_approval_rejects_symlink_indirection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_comparators import load_frozen_compare_is10_approval

    manifest = _manifest()
    features = _full_feature_table(manifest)
    columns = tuple(f"f{index:04d}" for index in range(801))
    contract = _feature_contract_for_columns(features, columns)
    root, approval_path, accepted_commit, accepted_path, expected_hash = (
        _trusted_project_repository_with_complete_recipe(
            tmp_path, monkeypatch, manifest, features, contract
        )
    )
    link = tmp_path / "approval-link.json"
    real_resolve = Path.resolve
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, strict=False: approval_path
        if self == link
        else real_resolve(self, strict=strict),
    )
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == link or real_is_symlink(self),
    )

    with pytest.raises(ValueError, match="regular file|symlink"):
        load_frozen_compare_is10_approval(
            link,
            trusted_project_repository_root=root,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=expected_hash,
            runtime_random_state=42,
            feature_contract=contract,
            feature_artifact_sha256=_feature_artifact_sha256(features),
            manifest=manifest,
        )


def test_confirmatory_mode_requires_independent_frozen_approval_record(tmp_path: Path) -> None:
    from covid_audio_btp.hst_comparators import run_aligned_compare_is10

    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    with pytest.raises(ValueError, match="independent frozen approval record"):
        run_aligned_compare_is10(
            features,
            manifest,
            feature_contract=contract,
            selected_feature_k=800,
            audit_dir=tmp_path,
        )


def test_confirmatory_ranker_is_lightgbm_without_fallback(tmp_path: Path) -> None:
    from covid_audio_btp.hst_comparators import run_aligned_compare_is10

    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    common = {
        "feature_contract": contract,
        "approval_record_path": tmp_path / "unused.approved.json",
        "trusted_project_repository_root": tmp_path,
        "accepted_freezes_path": tmp_path / "unavailable-freezes.json",
        "expected_accepted_freezes_sha256": "b" * 64,
        "selected_feature_k": 800,
        "audit_dir": tmp_path,
    }
    with pytest.raises(ValueError, match="confirmatory ranker.*lightgbm"):
        run_aligned_compare_is10(features, manifest, ranker="sklearn_extra_trees", **common)
    with pytest.raises(ValueError, match="fallback"):
        run_aligned_compare_is10(
            features,
            manifest,
            ranker="lightgbm",
            allow_sklearn_fallback=True,
            **common,
        )


def test_manifest_hash_is_recomputed_and_content_hash_is_required() -> None:
    manifest = _manifest()
    features = _features(manifest)
    tampered = manifest.copy()
    tampered.loc[0, "source_audio_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest_sha256"):
        _run(tampered, features)

    with pytest.raises(ValueError, match="source_audio_sha256"):
        _refreeze(manifest.drop(columns="source_audio_sha256"))


def test_invocation_requires_one_complete_canonical_manifest() -> None:
    first = _manifest()
    second = first.drop(columns=["row_content_sha256", "manifest_sha256"]).copy()
    second["protocol"] = "track_b"
    second = _refreeze(second)
    combined = pd.concat([first, second], ignore_index=True)

    with pytest.raises(ValueError, match="one canonical manifest"):
        _run(combined, _features(first))


def test_feature_rows_require_complete_matching_source_hashes() -> None:
    manifest = _manifest()
    features = _features(manifest)
    contract = _feature_contract(features)
    _run(manifest, features, feature_contract=contract)

    with pytest.raises(ValueError, match="features.*source.*hash"):
        _run(
            manifest,
            features.drop(columns="source_audio_sha256"),
            feature_contract=contract,
        )

    missing = features.copy()
    missing.loc[0, "source_audio_sha256"] = pd.NA
    with pytest.raises(ValueError, match="every feature row.*source.*hash"):
        _run(manifest, missing, feature_contract=contract)

    conflicting = features.copy()
    conflicting.loc[0, "source_audio_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source hash"):
        _run(manifest, conflicting, feature_contract=contract)


def test_metrics_are_partitioned_by_dataset() -> None:
    manifest = _manifest()
    target = manifest["participant_key"].eq("coswara::te1")
    manifest.loc[target, "dataset"] = "coughvid"
    manifest.loc[target, "participant_key"] = "coughvid::te1"
    manifest.loc[target, "recording_key"] = manifest.loc[target, "recording_key"].str.replace(
        "coswara::", "coughvid::", regex=False
    )
    manifest = _refreeze(manifest)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _run(manifest, _features(manifest))

    test_metrics = result.metrics[result.metrics["split"].eq("test")]
    assert set(test_metrics["dataset"]) == {"coswara", "coughvid"}
    assert test_metrics.groupby("model")["dataset"].nunique().eq(2).all()


def test_atomic_audit_exports_and_explicit_sklearn_fallback(tmp_path: Path) -> None:
    from covid_audio_btp.hst_comparators import run_aligned_compare_is10

    manifest = _manifest()
    result = run_aligned_compare_is10(
        _features(manifest),
        manifest,
        feature_contract=_feature_contract(_features(manifest)),
        selected_feature_k=3,
        ranker="sklearn_extra_trees",
        allow_sklearn_fallback=True,
        test_mode=True,
        estimator_factory=lambda model_name, random_state: _RecordingEstimator(model_name),
        audit_dir=tmp_path,
    )

    assert set(result.feature_selection["ranker_backend"]) == {"sklearn_extra_trees"}
    receipt = json.loads((tmp_path / "current.json").read_text(encoding="ascii"))
    generation = tmp_path / "generations" / receipt["generation_id"]
    audit_manifest = json.loads((generation / "manifest.json").read_text(encoding="ascii"))
    expected_evidence = {
        "comparator_predictions.csv",
        "comparator_participant_predictions.csv",
        "comparator_metrics.csv",
        "comparator_alignment_audit.csv",
        "comparator_feature_selection.csv",
        "comparator_model_audit.csv",
        "comparator_candidate_selection.csv",
    }
    assert expected_evidence <= set(audit_manifest["files"])
    assert audit_manifest["execution_class"] == "exploratory_test_only"
    assert audit_manifest["confirmatory_eligible"] is False
    for relative_path, descriptor in audit_manifest["files"].items():
        payload = (generation / relative_path).read_bytes()
        assert sha256(payload).hexdigest() == descriptor["sha256"]
        assert len(payload) == descriptor["size_bytes"]
    model_rows = result.model_audit[result.model_audit["model"].isin(MODEL_NAMES)]
    assert set(model_rows["estimator_class"]) == {"_RecordingEstimator"}
    assert set(model_rows["estimator_module"]) == {__name__}
    for row in model_rows.itertuples(index=False):
        artifact = generation / row.model_artifact
        assert sha256(artifact.read_bytes()).hexdigest() == row.checkpoint_hash
        bundle = pickle.loads(artifact.read_bytes())
        assert isinstance(bundle["estimator"], _RecordingEstimator)
        assert bundle["selected_feature_columns"] == (
            "compare__f0",
            "compare__f1",
            "is10__f0",
        )
        assert tuple(bundle["training_medians"]) == bundle["selected_feature_columns"]
        assert bundle["feature_schema_sha256"] == _feature_contract(_features(manifest)).schema_sha256
        assert bundle["feature_artifact_sha256"] == _feature_artifact_sha256(_features(manifest))
        assert bundle["threshold"] == pytest.approx(row.threshold)
        assert bundle["threshold_source"] == "validation_participant_balanced_accuracy"
        assert bundle["model_identity"]["name"] == row.model
        assert bundle["protocol"] == row.protocol
        assert bundle["fold"] == row.fold
        assert bundle["modality"] == row.modality
        assert bundle["cohort"] == row.cohort
        assert bundle["model_seed"] == row.random_state
        assert bundle["manifest_sha256"] == str(manifest["manifest_sha256"].iloc[0])
        assert bundle["datasets"] == ("coswara",)
        assert bundle["splits"] == ("test", "train", "validation")
        assert bundle["label_mapping"] == {"negative": 0, "positive": 1}
        assert "executable_source_sha256" in bundle
        assert "dependency_lock_sha256" in bundle
        assert "environment_lock_sha256" in bundle

    ensemble_row = result.model_audit[
        result.model_audit["model"].eq("top_4_validation_ensemble")
    ].iloc[0]
    ensemble_bundle = pickle.loads(
        (generation / ensemble_row["model_artifact"]).read_bytes()
    )
    assert ensemble_bundle["threshold"] == pytest.approx(ensemble_row["threshold"])
    assert set(ensemble_bundle["member_artifacts"]) == set(MODEL_NAMES)
    for member in ensemble_bundle["member_artifacts"].values():
        member_path = generation / member["path"]
        assert member_path.is_file()
        assert sha256(member_path.read_bytes()).hexdigest() == member["sha256"]
    assert not list(tmp_path.rglob("*.tmp"))

    with pytest.raises(ValueError, match="fallback"):
        run_aligned_compare_is10(
            _features(manifest),
            manifest,
            feature_contract=_feature_contract(_features(manifest)),
            selected_feature_k=3,
            ranker="sklearn_extra_trees",
            allow_sklearn_fallback=False,
            test_mode=True,
            estimator_factory=lambda model_name, random_state: _RecordingEstimator(model_name),
        )


def test_atomic_generation_is_read_only_before_manual_acceptance(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    audit_dir = tmp_path / "audit"
    _run(manifest, _features(manifest), audit_dir=audit_dir)

    current_path = audit_dir / "current.json"
    receipt = json.loads(current_path.read_text(encoding="ascii"))
    generation = audit_dir / "generations" / receipt["generation_id"]
    generation_manifest_path = generation / "manifest.json"
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="ascii")
    )

    assert comparators._path_is_read_only(current_path)
    assert comparators._path_is_read_only(generation_manifest_path)
    for relative_path in generation_manifest["files"]:
        assert comparators._path_is_read_only(generation / relative_path)


def test_artifact_paths_hash_canonical_context_and_reject_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    left = comparators._model_artifact_path("track/a", 0, "cough", MODEL_NAMES[0], "pkl")
    right = comparators._model_artifact_path("track_a", 0, "cough", MODEL_NAMES[0], "pkl")
    assert left != right
    assert len(Path(left).stem) == 64
    assert len(Path(right).stem) == 64

    monkeypatch.setattr(
        comparators,
        "_model_artifact_path",
        lambda *args, **kwargs: "models/duplicate.pkl",
    )
    with pytest.raises(ValueError, match="duplicate model artifact path"):
        _run(_manifest(), _features(_manifest()))


def test_failed_audit_generation_does_not_replace_current_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    features = _features(manifest)
    _run(manifest, features, audit_dir=tmp_path)
    original_receipt = (tmp_path / "current.json").read_bytes()
    original_generations = set((tmp_path / "generations").iterdir())

    def fail_export(frame: pd.DataFrame, path: Path) -> None:
        raise OSError("simulated audit failure")

    monkeypatch.setattr(comparators, "_atomic_csv", fail_export)
    with pytest.raises(OSError, match="simulated audit failure"):
        _run(manifest, features, audit_dir=tmp_path)

    assert (tmp_path / "current.json").read_bytes() == original_receipt
    assert set((tmp_path / "generations").iterdir()) == original_generations


def test_failed_current_pointer_removes_unpublished_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_comparators as comparators

    manifest = _manifest()
    features = _features(manifest)
    _run(manifest, features, audit_dir=tmp_path)
    original_receipt = (tmp_path / "current.json").read_bytes()
    original_generations = set((tmp_path / "generations").iterdir())
    real_atomic_json = comparators._atomic_json

    def fail_current(payload: dict[str, object], path: Path) -> None:
        if path.name == "current.json":
            raise OSError("simulated current-pointer failure")
        real_atomic_json(payload, path)

    monkeypatch.setattr(comparators, "_atomic_json", fail_current)
    with pytest.raises(OSError, match="current-pointer failure"):
        _run(manifest, features, audit_dir=tmp_path)

    assert (tmp_path / "current.json").read_bytes() == original_receipt
    assert set((tmp_path / "generations").iterdir()) == original_generations
