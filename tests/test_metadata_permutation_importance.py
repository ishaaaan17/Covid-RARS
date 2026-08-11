from __future__ import annotations

import pandas as pd

from covid_rars.metadata_permutation_importance import run_metadata_permutation_importance
from covid_rars.shuffle_retrain_sanity import run_metadata_shuffle_retrain_sanity


def _metadata_frame() -> pd.DataFrame:
    rows = []
    for split, n in [("train", 40), ("validation", 20), ("test", 30)]:
        for idx in range(n):
            positive = idx % 2 == 0
            rows.append(
                {
                    "participant_id": f"{split}_{idx:03d}",
                    "recording_id": f"{split}_{idx:03d}_rec",
                    "label_binary": "positive" if positive else "negative",
                    "split": split,
                    "age": 60 if positive else 25,
                    "gender": "female" if idx % 3 == 0 else "male",
                    "country": "mixed_country",
                    "recording_date": "2020-05-01",
                    "duration_sec": 5.0,
                    "sample_rate_original": 44100,
                    "quality_flag": "ok",
                    "symptoms_json": "{}",
                    "comorbidities_json": "{}",
                }
            )
    return pd.DataFrame(rows)


def test_metadata_permutation_importance_identifies_predictive_metadata_field() -> None:
    result = run_metadata_permutation_importance(
        _metadata_frame(),
        feature_sets=["full_safe_metadata"],
        n_repeats=5,
        random_state=3,
    )

    assert not result.importance.empty
    assert not result.group_summary.empty
    top = result.importance.sort_values("importance_mean", ascending=False).iloc[0]
    assert top["feature_group"] in {"demographic", "recording_protocol"}
    assert top["importance_mean"] > 0
    assert result.metrics.iloc[0]["audit_model"] == "full_safe_metadata"


def test_metadata_shuffle_retrain_sanity_collapses_shortcut_model() -> None:
    result = run_metadata_shuffle_retrain_sanity(
        _metadata_frame(),
        feature_sets=["full_safe_metadata"],
        n_permutations=5,
        random_state=10,
    )

    assert not result.empty
    row = result.iloc[0]
    assert row["sanity_check"] == "metadata_retrain_with_shuffled_labels"
    assert row["observed_auroc"] > 0.95
    assert 0.25 < row["shuffled_auroc_mean"] < 0.75
