from __future__ import annotations

import pandas as pd


def _metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        label = "positive" if idx % 3 else "negative"
        rows.append(
            {
                "recording_id": f"rec_{idx}",
                "participant_id": f"p_{idx}",
                "label_binary": label,
                "split": "test" if idx >= 12 else "train",
                "country": "India" if idx % 2 else "United States",
                "device": "android" if idx % 4 else "ios",
                "recording_date": f"2021-{(idx % 4) + 1:02d}-15",
            }
        )
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(12, 24):
        label = "positive" if idx % 3 else "negative"
        rows.append(
            {
                "recording_id": f"rec_{idx}",
                "participant_id": f"p_{idx}",
                "label_binary": label,
                "probability": 0.7 if label == "positive" else 0.25,
                "split": "test",
                "audit_model": "full_safe_metadata",
                "model_name": "metadata_confounding_logistic_regression",
            }
        )
    return pd.DataFrame(rows)


def test_metadata_subgroup_breakdown_derives_temporal_and_device_tables() -> None:
    from covid_rars.metadata_confounding_subgroups import build_metadata_confounding_subgroup_tables

    availability, breakdown, metrics = build_metadata_confounding_subgroup_tables(
        metadata=_metadata(),
        predictions=_predictions(),
        subgroup_columns=["country", "device", "recording_year", "recording_month"],
        min_samples=2,
    )

    assert set(availability["subgroup"]) == {"country", "device", "recording_year", "recording_month"}
    assert availability["available"].all()
    assert {"country", "device", "recording_year", "recording_month"}.issubset(set(breakdown["subgroup"]))
    assert {"n_participants", "n_positive", "positive_prevalence"}.issubset(breakdown.columns)
    assert not metrics.empty
    assert metrics["audit_model"].eq("full_safe_metadata").all()


def test_metadata_subgroup_availability_reports_missing_requested_columns() -> None:
    from covid_rars.metadata_confounding_subgroups import build_metadata_confounding_subgroup_tables

    availability, breakdown, metrics = build_metadata_confounding_subgroup_tables(
        metadata=_metadata().drop(columns=["device"]),
        predictions=_predictions(),
        subgroup_columns=["device", "recording_year"],
        min_samples=2,
    )

    device = availability[availability["subgroup"].eq("device")].iloc[0]
    assert bool(device["available"]) is False
    assert "recording_year" in set(breakdown["subgroup"])
    assert not metrics.empty
