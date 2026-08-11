import numpy as np
import pandas as pd


def test_scale_pos_weight_uses_negative_to_positive_ratio():
    from covid_rars.rescue_experiments import scale_pos_weight_from_labels

    labels = pd.Series(["negative", "negative", "negative", "positive"])

    assert scale_pos_weight_from_labels(labels) == 3.0


def test_feature_strategy_drops_high_shift_features_and_keeps_common_columns():
    from covid_rars.rescue_experiments import select_feature_columns_for_strategy

    source = pd.DataFrame(
        {
            "recording_id": ["s1", "s2"],
            "mfcc_a": [0.1, 0.2],
            "mfcc_b": [1.0, 2.0],
            "mfcc_c": [5.0, 6.0],
            "source_only": [9.0, 9.5],
        }
    )
    external = pd.DataFrame(
        {
            "recording_id": ["e1", "e2"],
            "mfcc_a": [0.1, 0.2],
            "mfcc_b": [10.0, 11.0],
            "mfcc_c": [5.0, 6.0],
            "external_only": [3.0, 4.0],
        }
    )
    shift = pd.DataFrame(
        {
            "feature": ["mfcc_b", "mfcc_c", "mfcc_a"],
            "abs_standardized_mean_difference": [1.2, 0.4, 0.1],
        }
    )

    selected = select_feature_columns_for_strategy(
        source,
        external,
        strategy="drop_high_shift",
        shift_report=shift,
        smd_threshold=0.5,
        id_columns=["recording_id"],
    )

    assert selected == ["mfcc_a", "mfcc_c"]


def test_feature_strategy_selects_top_stable_features_in_shift_order():
    from covid_rars.rescue_experiments import select_feature_columns_for_strategy

    source = pd.DataFrame({"f1": [1, 2], "f2": [2, 3], "f3": [3, 4]})
    external = pd.DataFrame({"f1": [1, 2], "f2": [20, 30], "f3": [3, 4]})
    shift = pd.DataFrame(
        {
            "feature": ["f2", "f3", "f1"],
            "abs_standardized_mean_difference": [2.0, 0.2, 0.1],
        }
    )

    selected = select_feature_columns_for_strategy(
        source,
        external,
        strategy="top_stable",
        shift_report=shift,
        top_k=2,
    )

    assert selected == ["f1", "f3"]


def test_stratified_external_splits_keep_positive_and_negative_in_each_split():
    from covid_rars.rescue_experiments import make_stratified_external_splits

    labels = ["negative"] * 12 + ["positive"] * 12
    features = pd.DataFrame(
        {
            "recording_id": [f"r{i}" for i in range(24)],
            "participant_id": [f"p{i}" for i in range(24)],
            "label_binary": labels,
            "modality": ["cough"] * 24,
            "f1": np.arange(24, dtype=float),
        }
    )

    split = make_stratified_external_splits(
        features,
        train_size=0.5,
        validation_size=0.25,
        random_state=7,
    )

    assert set(split["split"]) == {"train", "validation", "test"}
    counts = split.groupby(["split", "label_binary"]).size().unstack(fill_value=0)
    assert (counts[["negative", "positive"]] > 0).all().all()


def test_source_to_external_uses_source_validation_for_calibration_metadata():
    from covid_rars.rescue_experiments import evaluate_source_to_external

    source = pd.DataFrame(
        {
            "recording_id": [f"s{i}" for i in range(12)],
            "participant_id": [f"sp{i}" for i in range(12)],
            "dataset": ["coswara"] * 12,
            "modality": ["cough"] * 12,
            "label_binary": ["negative", "positive"] * 6,
            "split": ["train"] * 8 + ["validation"] * 4,
            "f1": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2, 0.3, 1.3, 0.4, 1.4, 0.5, 1.5],
        }
    )
    external = pd.DataFrame(
        {
            "recording_id": ["e1", "e2", "e3", "e4"],
            "participant_id": ["ep1", "ep2", "ep3", "ep4"],
            "dataset": ["coughvid"] * 4,
            "modality": ["cough"] * 4,
            "label_binary": ["negative", "positive", "negative", "positive"],
            "split": ["external"] * 4,
            "f1": [0.0, 1.0, 0.2, 1.2],
        }
    )

    result = evaluate_source_to_external(
        source,
        external,
        model_name="logistic_regression",
        feature_strategy="all",
    )

    assert result.metrics["source_rows"] == 8
    assert result.metrics["validation_rows"] == 4
    assert result.metrics["calibration_method"] == "platt"
    assert "raw_probability" in result.predictions.columns
    assert result.predictions["probability"].between(0, 1).all()
