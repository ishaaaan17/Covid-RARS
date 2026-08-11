from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_frame(
    *,
    dataset: str,
    split: str,
    labels: list[str],
    offset: float,
) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(labels):
        signal = 1.0 if label == "positive" else -1.0
        rows.append(
            {
                "recording_id": f"{dataset}_{split}_{idx}",
                "participant_id": f"{dataset}_p_{idx}",
                "dataset": dataset,
                "modality": "cough",
                "split": split,
                "label_binary": label,
                "feat_0": offset + signal + idx * 0.01,
                "feat_1": offset * 0.5 + signal * 0.3 + idx * 0.02,
                "feat_2": offset - signal * 0.2 + idx * 0.03,
            }
        )
    return pd.DataFrame(rows)


def test_coral_alignment_reduces_distribution_gap() -> None:
    from covid_rars.domain_adaptation import coral_align_source_to_target, rbf_mmd_squared

    rng = np.random.default_rng(42)
    source = rng.normal(size=(160, 3))
    target = rng.normal(size=(160, 3)) @ np.array(
        [
            [1.8, 0.2, 0.0],
            [0.1, 0.6, 0.0],
            [0.0, 0.0, 1.3],
        ]
    ) + np.array([2.0, -1.0, 0.5])

    aligned = coral_align_source_to_target(source, target)

    before_mean_gap = np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))
    after_mean_gap = np.linalg.norm(aligned.mean(axis=0) - target.mean(axis=0))
    before_mmd = rbf_mmd_squared(source, target)
    after_mmd = rbf_mmd_squared(aligned, target)

    assert after_mean_gap < before_mean_gap
    assert after_mmd < before_mmd


def test_domain_adaptation_baseline_reports_source_and_coral_metrics() -> None:
    from covid_rars.domain_adaptation import run_domain_adaptation_baseline

    source = pd.concat(
        [
            _feature_frame(dataset="coswara", split="train", labels=["positive", "negative"] * 8, offset=0.0),
            _feature_frame(dataset="coswara", split="validation", labels=["positive", "negative"] * 4, offset=0.2),
        ],
        ignore_index=True,
    )
    external = _feature_frame(
        dataset="coughvid",
        split="external",
        labels=["positive", "negative"] * 5,
        offset=2.0,
    )

    result = run_domain_adaptation_baseline(
        source,
        external,
        model_name="logistic_regression",
        feature_strategy="all",
        representation="mfcc",
        n_mmd_samples=100,
    )

    methods = set(result.metrics["adaptation_method"])
    assert methods == {"source_only", "coral"}
    assert len(result.predictions) == len(external) * 2
    assert set(result.predictions["adaptation_method"]) == methods
    assert result.metrics["mmd_before"].notna().all()
    assert result.metrics["mmd_after"].notna().all()
    coral = result.metrics[result.metrics["adaptation_method"].eq("coral")].iloc[0]
    assert coral["mmd_after"] <= coral["mmd_before"]
    assert coral["n_features"] == 3


def test_coral_transform_applies_train_fitted_shift_to_validation() -> None:
    from covid_rars.domain_adaptation import apply_coral_transform, fit_coral_transform

    source_train = np.array([[0.0], [2.0]])
    target = np.array([[10.0], [14.0]])
    validation = np.array([[4.0]])

    transform = fit_coral_transform(source_train, target, regularization=1e-12)
    aligned_validation = apply_coral_transform(validation, transform)

    assert aligned_validation.shape == validation.shape
    assert np.isclose(aligned_validation[0, 0], 18.0, atol=1e-6)
