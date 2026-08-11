from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedShuffleSplit


HST_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)


def _digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


def _scientific_config(*, learning_rate: float = 1e-5) -> dict[str, object]:
    return {
        "architecture": {"name": "hst_base", "classes": 2},
        "source_checkpoint": {"sha256": _digest("checkpoint")},
        "preprocessing": {"representation": "paper_logmel_224"},
        "augmentation": {"rotation_degrees": 5},
        "optimizer": {"name": "adamw", "learning_rate": learning_rate},
        "stopping_rule": {"metric": "participant_f1", "patience": 100},
        "participant_aggregation": "recording_probability_mean",
        "fusion": {"primary": "uniform_mean"},
        "thresholding": {"primary": 0.5},
        "sampling": {"unit": "participant", "class_balance": "natural"},
        "calibration": {"method": "none", "fit_split": "validation"},
        "metric_settings": {"ece_bins": 10},
    }


def _cache_index(
    n_participants: int = 80,
    *,
    dataset: str = "coswara",
    modalities: tuple[str, ...] = ("cough", "speech"),
    representation_id: str = "paper_logmel_224",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2020-04-01T00:00:00Z")
    for participant_index in range(n_participants):
        participant_id = f"p{participant_index:03d}"
        label = "positive" if participant_index % 2 else "negative"
        timestamp = start + pd.Timedelta(days=participant_index)
        for modality in modalities:
            recording_id = f"{participant_id}-{modality}"
            rows.append(
                {
                    "dataset": dataset,
                    "participant_id": participant_id,
                    "participant_key": f"{dataset}::{participant_id}",
                    "recording_id": recording_id,
                    "recording_key": f"{dataset}::{recording_id}",
                    "modality": modality,
                    "label_binary": label,
                    "eligible": True,
                    "tensor_sha256": _digest(
                        f"tensor::{dataset}::{representation_id}::{recording_id}"
                    ),
                    "source_audio_sha256": _digest(
                        f"source-audio::{dataset}::{recording_id}"
                    ),
                    "preprocessing_hash": _digest(f"preprocess::{representation_id}"),
                    "representation_id": representation_id,
                    "recording_timestamp": timestamp.isoformat(),
                    "cough_symptom": "yes" if participant_index % 3 else "no",
                    "label_source": "project_label",
                    "label_provenance": "frozen_project_contract",
                    "dataset_release_id": "coswara-test-release",
                    "source_manifest_sha256": _digest("coswara-source-manifest"),
                    "preprocessing_variant": "full_recording",
                }
            )
    return pd.DataFrame(rows)


def _eligibility_mapping(
    cache: pd.DataFrame,
    *,
    scientific_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    from covid_rars.hst_protocols import intersect_representation_eligibility

    comparator = cache.copy()
    comparator["representation_id"] = "compare_is10_top800"
    comparator["tensor_sha256"] = comparator["recording_key"].map(
        lambda value: _digest(f"comparator::{value}")
    )
    comparator["preprocessing_hash"] = _digest("compare-is10-preprocessing")
    return intersect_representation_eligibility(
        cache,
        comparator,
        scientific_config=scientific_config or _scientific_config(),
    )


def _paired_representation_cache(cache: pd.DataFrame) -> pd.DataFrame:
    comparator = cache.copy()
    comparator["representation_id"] = "compare_is10_top800"
    comparator["tensor_sha256"] = comparator["recording_key"].map(
        lambda value: _digest(f"comparator::{value}")
    )
    comparator["preprocessing_hash"] = _digest("compare-is10-preprocessing")
    return pd.concat([cache, comparator], ignore_index=True)


def _participants(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[
        ["fold", "participant_key", "label_binary", "split", "split_seed"]
    ].drop_duplicates()


def _expected_repeated_holdout(
    cache_index: pd.DataFrame, seed: int
) -> dict[str, set[str]]:
    people = (
        cache_index[["participant_key", "label_binary"]]
        .drop_duplicates()
        .sort_values("participant_key")
        .reset_index(drop=True)
    )
    labels = people["label_binary"].map({"negative": 0, "positive": 1}).to_numpy()
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_validation_index, test_index = next(outer.split(people, labels))
    train_validation = people.iloc[train_validation_index].reset_index(drop=True)
    train_validation_labels = labels[train_validation_index]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_index, validation_index = next(
        inner.split(train_validation, train_validation_labels)
    )
    return {
        "train": set(train_validation.iloc[train_index]["participant_key"]),
        "validation": set(
            train_validation.iloc[validation_index]["participant_key"]
        ),
        "test": set(people.iloc[test_index]["participant_key"]),
    }


def test_protocol_matched_manifest_uses_exact_seeds_proportions_and_splitter() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index()
    manifest = build_protocol_matched_hst_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    people = _participants(manifest)

    assert tuple(
        people.sort_values("fold").drop_duplicates("fold")["split_seed"].astype(int)
    ) == HST_SEEDS
    assert manifest["protocol"].eq("hst_literature_aligned_repeated_holdout").all()
    assert manifest["seed_provenance"].eq("released_hst_baseline_scripts").all()
    assert manifest["fold"].nunique() == 10
    assert manifest["analysis_scope"].eq("internal_performance").all()
    assert manifest["analysis_role"].eq("primary").all()
    assert manifest["estimand_id"].eq(
        "track_a_internal_hst_vs_aligned_comparator"
    ).all()
    assert manifest["multiplicity_family"].eq(
        "primary_internal_performance"
    ).all()
    assert manifest["confirmatory_protocol"].eq(True).all()

    for fold, seed in enumerate(HST_SEEDS, start=1):
        current = people[people["fold"].eq(fold)]
        expected = _expected_repeated_holdout(cache, seed)
        assert current["participant_key"].is_unique
        assert current["split"].value_counts().to_dict() == {
            "train": 56,
            "test": 16,
            "validation": 8,
        }
        for split, expected_keys in expected.items():
            observed = set(current.loc[current["split"].eq(split), "participant_key"])
            assert observed == expected_keys


def test_protocol_manifest_is_deterministic_and_content_addressed() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40)
    alignment = _eligibility_mapping(cache)
    first = build_protocol_matched_hst_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=alignment,
    )
    second = build_protocol_matched_hst_manifest(
        cache.sample(frac=1.0, random_state=991).reset_index(drop=True),
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=alignment,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["manifest_sha256"].nunique() == 1
    assert first["manifest_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert first["row_content_sha256"].is_unique
    assert first["row_content_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_manifest_requires_scientific_configuration_and_eligibility_mapping() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40)
    with pytest.raises(ValueError, match="scientific"):
        build_protocol_matched_hst_manifest(cache, seeds=HST_SEEDS)
    with pytest.raises(ValueError, match="eligibility"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
        )


def test_primary_track_a_rejects_mapping_that_omits_one_eligible_cache_unit() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40, modalities=("cough",))
    incomplete_mapping = _eligibility_mapping(
        cache[~cache["participant_key"].eq("coswara::p039")].copy()
    )

    with pytest.raises(ValueError, match="omits eligible cache analysis units"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=incomplete_mapping,
        )


def test_track_a_marks_hst_and_aligned_comparator_rows_as_primary() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40, modalities=("cough",))
    paired_cache = _paired_representation_cache(cache)
    manifest = build_protocol_matched_hst_manifest(
        paired_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )

    assert set(manifest["representation_id"]) == {
        "paper_logmel_224",
        "compare_is10_top800",
    }
    assert manifest["analysis_role"].eq("primary").all()
    assert manifest["estimand_id"].eq(
        "track_a_internal_hst_vs_aligned_comparator"
    ).all()


def test_scientific_fingerprint_is_in_every_hashed_manifest_row() -> None:
    from covid_rars.hst_protocols import (
        build_protocol_matched_hst_manifest,
        scientific_configuration_fingerprint,
    )

    cache = _cache_index(40)
    first_config = _scientific_config(learning_rate=1e-5)
    second_config = _scientific_config(learning_rate=2e-5)
    first = build_protocol_matched_hst_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=first_config,
        eligibility_mapping=_eligibility_mapping(
            cache, scientific_config=first_config
        ),
    )
    second = build_protocol_matched_hst_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=second_config,
        eligibility_mapping=_eligibility_mapping(
            cache, scientific_config=second_config
        ),
    )

    assert first["scientific_configuration_fingerprint"].nunique() == 1
    assert first["scientific_configuration_fingerprint"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert first["manifest_sha256"].iloc[0] != second["manifest_sha256"].iloc[0]
    assert set(first["row_content_sha256"]).isdisjoint(
        set(second["row_content_sha256"])
    )
    with pytest.raises(ValueError, match="configuration.*required"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_fingerprint=scientific_configuration_fingerprint(first_config),
            eligibility_mapping=_eligibility_mapping(
                cache, scientific_config=first_config
            ),
        )


def test_protocol_identity_rejects_non_nominal_70_10_20_fractions() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40)
    alignment = _eligibility_mapping(cache)
    with pytest.raises(ValueError, match="nominal 70/10/20"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            test_fraction=0.25,
            scientific_config=_scientific_config(),
            eligibility_mapping=alignment,
        )
    with pytest.raises(ValueError, match="nominal 70/10/20"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            validation_fraction_of_remaining=0.2,
            scientific_config=_scientific_config(),
            eligibility_mapping=alignment,
        )


def test_track_a_reports_nominal_ratio_and_realized_fold_counts() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(42, modalities=("cough",))
    manifest = build_protocol_matched_hst_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    expected_columns = {
        "nominal_split_ratio",
        "realized_train_participant_count",
        "realized_validation_participant_count",
        "realized_test_participant_count",
        "realized_train_fraction",
        "realized_validation_fraction",
        "realized_test_fraction",
    }
    assert expected_columns.issubset(manifest.columns)
    assert manifest["nominal_split_ratio"].eq("70/10/20").all()
    for _, fold in manifest.groupby("fold"):
        people = fold[["participant_key", "split"]].drop_duplicates()
        counts = people["split"].value_counts()
        assert fold["realized_train_participant_count"].eq(counts["train"]).all()
        assert fold["realized_validation_participant_count"].eq(
            counts["validation"]
        ).all()
        assert fold["realized_test_participant_count"].eq(counts["test"]).all()
        assert fold["realized_train_fraction"].eq(counts["train"] / 42).all()
        assert fold["realized_validation_fraction"].eq(
            counts["validation"] / 42
        ).all()
        assert fold["realized_test_fraction"].eq(counts["test"] / 42).all()
        assert counts.to_dict() != {"train": 29, "validation": 4, "test": 9}


@pytest.mark.parametrize("mutation", ("label", "participant"))
def test_builder_binds_scientific_identity_to_eligibility_mapping(
    mutation: str,
) -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40, modalities=("cough",))
    alignment = _eligibility_mapping(cache)
    if mutation == "label":
        cache.loc[0, "label_binary"] = "positive"
    else:
        cache.loc[0, "participant_id"] = "replacement"
        cache.loc[0, "participant_key"] = "coswara::replacement"

    with pytest.raises(ValueError, match="eligibility"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=alignment,
        )


@pytest.mark.parametrize("defect", ("missing", "invalid"))
def test_scientific_manifest_requires_valid_source_audio_hash(defect: str) -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40, modalities=("cough",))
    alignment = _eligibility_mapping(cache)
    if defect == "missing":
        cache = cache.drop(columns="source_audio_sha256")
    else:
        cache.loc[0, "source_audio_sha256"] = "not-a-sha256"

    with pytest.raises(ValueError, match="source_audio_sha256"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=alignment,
        )


@pytest.mark.parametrize(
    "content_column",
    ("tensor_sha256", "source_sha256", "source_audio_sha256"),
)
def test_builder_rejects_content_hash_crossing_splits(content_column: str) -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40, modalities=("cough",))
    if content_column == "source_sha256":
        cache[content_column] = cache["recording_key"].map(
            lambda value: _digest(f"source::{value}")
        )
    expected = _expected_repeated_holdout(cache, HST_SEEDS[0])
    train_key = sorted(expected["train"])[0]
    test_key = sorted(expected["test"])[0]
    shared_hash = _digest(f"shared::{content_column}")
    cache.loc[
        cache["participant_key"].isin({train_key, test_key}), content_column
    ] = shared_hash

    with pytest.raises(ValueError, match="content hash leakage"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(cache),
        )


def test_builder_rejects_cache_content_not_present_in_frozen_alignment() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40)
    alignment = _eligibility_mapping(cache)
    cache.loc[0, "tensor_sha256"] = _digest("changed-after-alignment")
    with pytest.raises(ValueError, match="eligibility"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=alignment,
        )


def test_builder_rejects_incomplete_cache_for_frozen_alignment() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    cache = _cache_index(40, modalities=("cough",))
    alignment = _eligibility_mapping(cache)
    incomplete = cache.iloc[:-1].copy()

    with pytest.raises(ValueError, match="eligibility"):
        build_protocol_matched_hst_manifest(
            incomplete,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=alignment,
        )


def test_paired_representation_rows_have_one_analysis_unit_weight() -> None:
    from covid_rars.hst_protocols import (
        audit_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    base = _cache_index(40, modalities=("cough",))
    paired_cache = _paired_representation_cache(base)
    alignment = _eligibility_mapping(base)
    manifest = build_protocol_matched_hst_manifest(
        paired_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=alignment,
    )
    weights = manifest.groupby(
        ["fold", "split", "recording_key", "modality"]
    )["analysis_unit_weight"].sum()
    assert weights.eq(1.0).all()
    assert manifest["paired_representation"].eq(True).all()
    audit = audit_hst_manifest(manifest)
    assert audit["unpaired_duplicate_representation_count"].eq(0).all()
    assert audit["analysis_weight_violation_count"].eq(0).all()


def test_manifest_build_rejects_mixed_labels_and_unqualified_keys() -> None:
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest

    mixed = _cache_index(40)
    participant = mixed.iloc[0]["participant_key"]
    rows = mixed.index[mixed["participant_key"].eq(participant)]
    mixed.loc[rows[-1], "label_binary"] = "positive"
    with pytest.raises(ValueError, match="mixed labels"):
        build_protocol_matched_hst_manifest(
            mixed,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(mixed),
        )

    unqualified = _cache_index(40)
    unqualified.loc[0, "participant_key"] = "p000"
    with pytest.raises(ValueError, match="qualified participant_key"):
        build_protocol_matched_hst_manifest(
            unqualified,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(unqualified),
        )


def test_audit_reports_zero_leakage_and_detects_tampering() -> None:
    from covid_rars.hst_protocols import (
        audit_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    cache = _cache_index(40)
    manifest = build_protocol_matched_hst_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    clean = audit_hst_manifest(manifest)
    assert clean["participant_overlap_count"].eq(0).all()
    assert clean["mixed_label_participant_count"].eq(0).all()
    assert clean["multiple_split_participant_count"].eq(0).all()
    assert clean["manifest_hash_valid"].eq(True).all()

    tampered = manifest.copy()
    first_fold = tampered["fold"].eq(1)
    participant = tampered.loc[first_fold, "participant_key"].iloc[0]
    duplicate = tampered.loc[
        first_fold & tampered["participant_key"].eq(participant)
    ].copy()
    duplicate["split"] = "test" if duplicate["split"].iloc[0] != "test" else "train"
    tampered = pd.concat([tampered, duplicate], ignore_index=True)
    audit = audit_hst_manifest(tampered)
    fold_one = audit[audit["fold"].eq(1)].iloc[0]
    assert fold_one["participant_overlap_count"] == 1
    assert fold_one["multiple_split_participant_count"] == 1
    assert not bool(fold_one["manifest_hash_valid"])


def test_task2_like_cough_cohort_requires_explicit_cough_in_both_classes() -> None:
    from covid_rars.hst_protocols import build_hst_task2_like_cough_manifest

    cache = _cache_index(60)
    manifest = build_hst_task2_like_cough_manifest(
        cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    eligible_people = {
        f"coswara::p{index:03d}" for index in range(60) if index % 3
    }

    assert manifest["cohort"].eq("hst_task2_like_cough").all()
    assert manifest["modality"].eq("cough").all()
    assert manifest["cough_symptom_present"].eq(True).all()
    assert manifest["analysis_scope"].eq("symptom_matched_cough").all()
    assert manifest["analysis_role"].eq("exploratory").all()
    assert manifest["estimand_id"].eq(
        "task2_like_cough_internal_performance"
    ).all()
    assert manifest["multiplicity_family"].eq("exploratory_task2_like").all()
    assert manifest["confirmatory_protocol"].eq(False).all()
    assert set(manifest["participant_key"]) == eligible_people
    assert set(manifest["label_binary"]) == {"negative", "positive"}
    audit = manifest.attrs["symptom_exclusion_audit"]
    assert audit.set_index("reason").loc["cough_symptom_absent", "participant_count"] == 20
    assert manifest["symptom_exclusion_audit_sha256"].nunique() == 1
    assert manifest["symptom_exclusion_audit_sha256"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert manifest["symptom_exclusion_audit_payload_json"].ne("").sum() == 1


def test_split_policy_pair_uses_one_date_eligible_cohort_and_matched_counts() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    cache = _cache_index(60)
    invalid_person = "coswara::p059"
    cache.loc[cache["participant_key"].eq(invalid_person), "recording_timestamp"] = "bad-date"
    mixed, chronological = build_split_policy_contrast_manifests(
        cache,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    mixed_people = mixed[["participant_key", "label_binary", "split"]].drop_duplicates()
    chronological_people = chronological[
        ["participant_key", "label_binary", "split", "participant_timestamp_utc"]
    ].drop_duplicates()

    assert invalid_person not in set(mixed_people["participant_key"])
    assert set(mixed_people["participant_key"]) == set(
        chronological_people["participant_key"]
    )
    for split in ("train", "validation", "test"):
        left = mixed_people[mixed_people["split"].eq(split)][
            "label_binary"
        ].value_counts().to_dict()
        right = chronological_people[chronological_people["split"].eq(split)][
            "label_binary"
        ].value_counts().to_dict()
        assert left == right

    chronological_people = chronological_people.sort_values("participant_timestamp_utc")
    assert chronological_people["split"].drop_duplicates().tolist() == [
        "train",
        "validation",
        "test",
    ]
    scores = mixed.attrs["candidate_scores"]
    assert len(scores) == 1000
    assert scores["candidate_seed"].tolist() == list(range(42, 1042))
    chosen = int(mixed["assignment_seed"].iloc[0])
    expected = scores.sort_values(["objective", "candidate_seed"]).iloc[0]
    assert chosen == int(expected["candidate_seed"])
    assert mixed["candidate_scores_sha256"].nunique() == 1
    date_audit = chronological.attrs["date_eligibility_audit"]
    assert date_audit.set_index("reason").loc["no_parseable_participant_date", "participant_count"] == 1
    for prefix in (
        "date_eligibility_audit",
        "split_summary",
        "candidate_scores",
    ):
        assert mixed[f"{prefix}_sha256"].nunique() == 1
        assert mixed[f"{prefix}_payload_json"].ne("").sum() == 1
    assert "boundary_diagnostics_sha256" not in mixed
    assert "boundary_diagnostics" not in mixed.attrs
    diagnostics = chronological.attrs["boundary_diagnostics"]
    assert diagnostics["diagnostic_protocol"].eq(
        "hst_chronological_split_policy"
    ).all()
    assert diagnostics["parseable_recordings_order_verified"].eq(True).all()
    assert diagnostics["full_recording_order_verified"].eq(True).all()
    assert diagnostics["recording_order_verification_status"].eq(
        "all_recordings_verified"
    ).all()


@pytest.mark.parametrize("builder_name", ("split_policy", "common_late"))
def test_confirmatory_temporal_builders_reject_unfrozen_search_settings(
    builder_name: str,
) -> None:
    from covid_rars.hst_protocols import (
        build_common_late_test_manifests,
        build_split_policy_contrast_manifests,
    )

    cache = _cache_index(60)
    kwargs = {
        "candidate_count": 5,
        "scientific_config": _scientific_config(),
        "eligibility_mapping": _eligibility_mapping(cache),
    }
    with pytest.raises(ValueError, match="confirmatory"):
        if builder_name == "split_policy":
            build_split_policy_contrast_manifests(cache, **kwargs)
        else:
            build_common_late_test_manifests(cache, **kwargs)


@pytest.mark.parametrize("builder_name", ("split_policy", "common_late"))
def test_temporal_exploratory_overrides_are_labeled_in_rows_and_audits(
    builder_name: str,
) -> None:
    from covid_rars.hst_protocols import (
        build_common_late_test_manifests,
        build_split_policy_contrast_manifests,
    )

    cache = _cache_index(60)
    kwargs = {
        "candidate_count": 5,
        "random_state": 7,
        "training_seed": 9,
        "analysis_mode": "test_mode",
        "scientific_config": _scientific_config(),
        "eligibility_mapping": _eligibility_mapping(cache),
    }
    if builder_name == "split_policy":
        manifests = build_split_policy_contrast_manifests(
            cache,
            train_fraction=0.5,
            validation_fraction=0.25,
            **kwargs,
        )
    else:
        manifests = build_common_late_test_manifests(cache, **kwargs)

    for manifest in manifests:
        assert manifest["analysis_mode"].eq("test_mode").all()
        assert manifest["confirmatory_protocol"].eq(False).all()
        assert manifest["analysis_scope"].eq("reliability_evaluation").all()
        assert manifest["analysis_role"].eq("secondary").all()
        assert manifest["multiplicity_family"].eq(
            "prespecified_reliability"
        ).all()
        expected_estimand = {
            "hst_calendar_mixed_split_policy": "split_policy_temporal_contrast",
            "hst_chronological_split_policy": "split_policy_temporal_contrast",
            "hst_common_late_test_date_balanced_source": (
                "common_late_temporal_contrast"
            ),
            "hst_common_late_test_chronological_source": (
                "common_late_temporal_contrast"
            ),
        }
        assert manifest["estimand_id"].eq(
            expected_estimand[manifest["protocol"].iloc[0]]
        ).all()
        assert manifest["candidate_count"].eq(5).all()
        assert manifest["random_state"].eq(7).all()
        assert manifest["training_seed"].eq(9).all()
        for column in manifest.columns:
            if not column.endswith("_payload_json"):
                continue
            values = manifest.loc[manifest[column].ne(""), column]
            assert len(values) == 1
            payload = json.loads(values.iloc[0])
            if isinstance(payload, list):
                assert payload
                assert all(row["analysis_mode"] == "test_mode" for row in payload)
            else:
                assert payload["analysis_mode"] == "test_mode"


@pytest.mark.parametrize("builder_name", ("split_policy", "common_late"))
def test_confirmatory_temporal_defaults_are_frozen_in_every_manifest_row(
    builder_name: str,
) -> None:
    from covid_rars.hst_protocols import (
        build_common_late_test_manifests,
        build_split_policy_contrast_manifests,
    )

    cache = _cache_index(60)
    kwargs = {
        "scientific_config": _scientific_config(),
        "eligibility_mapping": _eligibility_mapping(cache),
    }
    if builder_name == "split_policy":
        manifests = build_split_policy_contrast_manifests(cache, **kwargs)
    else:
        manifests = build_common_late_test_manifests(cache, **kwargs)

    for manifest in manifests:
        assert manifest["analysis_mode"].eq("confirmatory").all()
        assert manifest["confirmatory_protocol"].eq(True).all()
        assert manifest["analysis_scope"].eq("reliability_evaluation").all()
        assert manifest["analysis_role"].eq("secondary").all()
        assert manifest["analysis_role"].ne("primary").all()
        assert manifest["multiplicity_family"].eq(
            "prespecified_reliability"
        ).all()
        assert manifest["train_fraction"].eq(0.6).all()
        assert manifest["validation_fraction"].eq(0.2).all()
        assert manifest["test_fraction"].eq(0.2).all()
        assert manifest["candidate_count"].eq(1000).all()
        assert manifest["random_state"].eq(42).all()
        assert manifest["training_seed"].eq(42).all()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_count", 1000.5),
        ("random_state", 42.5),
        ("training_seed", 42.5),
    ),
)
def test_confirmatory_temporal_integer_settings_reject_fractional_aliases(
    field: str,
    value: float,
) -> None:
    from covid_rars.hst_protocols import _resolve_temporal_analysis_mode

    settings = {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "candidate_count": 1000,
        "random_state": 42,
        "training_seed": 42,
    }
    settings[field] = value

    with pytest.raises(ValueError, match="confirmatory"):
        _resolve_temporal_analysis_mode("confirmatory", **settings)


def test_temporal_boundaries_never_split_identical_timestamps() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    cache = _cache_index(60)
    shared_timestamp = cache.loc[
        cache["participant_key"].eq("coswara::p035"), "recording_timestamp"
    ].iloc[0]
    cache.loc[
        cache["participant_key"].eq("coswara::p036"), "recording_timestamp"
    ] = shared_timestamp
    _, chronological = build_split_policy_contrast_manifests(
        cache,
        candidate_count=25,
        analysis_mode="test_mode",
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    people = chronological[
        ["participant_key", "participant_timestamp_utc", "split"]
    ].drop_duplicates()
    tied = people["participant_timestamp_utc"].eq(
        pd.Timestamp(shared_timestamp)
    )

    assert people.loc[tied, "split"].nunique() == 1
    diagnostics = chronological.attrs["boundary_diagnostics"]
    first = diagnostics[
        diagnostics["boundary_name"].eq("train_to_validation")
    ].iloc[0]
    assert first["boundary_moved"]
    assert first["desired_boundary_index"] == 36
    assert first["realized_boundary_index"] in {35, 37}
    assert chronological["boundary_diagnostics_sha256"].nunique() == 1
    assert chronological["boundary_diagnostics_payload_json"].ne("").sum() == 1


def test_temporal_eligibility_excludes_any_participant_with_an_undated_recording() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    cache = _cache_index(60)
    participant = "coswara::p000"
    speech = cache["participant_key"].eq(participant) & cache["modality"].eq("speech")
    cache.loc[speech, "recording_timestamp"] = "bad-date"
    mixed, chronological = build_split_policy_contrast_manifests(
        cache,
        candidate_count=25,
        analysis_mode="test_mode",
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )

    assert participant not in set(mixed["participant_key"])
    assert participant not in set(chronological["participant_key"])
    audit = chronological.attrs["date_eligibility_audit"].set_index("reason")
    assert audit.loc["partially_unparseable_recording_dates", "participant_count"] == 1
    diagnostics = chronological.attrs["boundary_diagnostics"]
    assert diagnostics["parseable_recordings_order_verified"].eq(True).all()
    assert diagnostics["full_recording_order_verified"].eq(True).all()
    assert diagnostics["recording_timestamp_order_verified"].eq(True).all()
    assert diagnostics["recording_order_verification_status"].eq(
        "all_recordings_verified"
    ).all()


def test_temporal_eligibility_keeps_multi_date_recordings_inside_one_split() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    cache = _cache_index(60)
    participant = "coswara::p000"
    speech = cache["participant_key"].eq(participant) & cache["modality"].eq("speech")
    cache.loc[speech, "recording_timestamp"] = "2020-04-02T00:00:00Z"
    _, chronological = build_split_policy_contrast_manifests(
        cache,
        candidate_count=25,
        analysis_mode="test_mode",
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )

    rows = chronological[chronological["participant_key"].eq(participant)]
    assert not rows.empty
    assert rows["participant_timestamp_utc"].eq(
        pd.Timestamp("2020-04-01T00:00:00Z")
    ).all()
    assert rows["utc_session_date_count"].eq(2).all()


def test_chronological_protocol_excludes_true_boundary_crossing_participants() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    cache = _cache_index(60)
    participant = "coswara::p000"
    speech = cache["participant_key"].eq(participant) & cache["modality"].eq("speech")
    cache.loc[speech, "recording_timestamp"] = "2021-01-01T00:00:00Z"
    _, chronological = build_split_policy_contrast_manifests(
        cache,
        candidate_count=25,
        analysis_mode="test_mode",
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )

    assert participant not in set(chronological["participant_key"])
    date_audit = chronological.attrs["date_eligibility_audit"].set_index("reason")
    assert date_audit.loc[
        "recording_span_crosses_temporal_boundary", "participant_count"
    ] == 1
    people = chronological[
        ["participant_key", "split", "recording_timestamp"]
    ].drop_duplicates()
    people["timestamp"] = pd.to_datetime(
        people["recording_timestamp"], utc=True
    )
    assert people.loc[people["split"].eq("train"), "timestamp"].max() < people.loc[
        people["split"].eq("validation"), "timestamp"
    ].min()
    assert people.loc[
        people["split"].eq("validation"), "timestamp"
    ].max() < people.loc[people["split"].eq("test"), "timestamp"].min()


@pytest.mark.parametrize("protocol", ("split_policy", "common_late", "reverse"))
def test_temporal_protocols_require_both_labels_in_every_partition(
    protocol: str,
) -> None:
    from covid_rars.hst_protocols import (
        build_common_late_test_manifests,
        build_reverse_temporal_hst_manifest,
        build_split_policy_contrast_manifests,
    )

    cache = _cache_index(60)
    for participant_index in range(60):
        participant = f"coswara::p{participant_index:03d}"
        cache.loc[cache["participant_key"].eq(participant), "label_binary"] = (
            "negative" if participant_index < 36 else "positive"
        )
    kwargs = {
        "scientific_config": _scientific_config(),
        "eligibility_mapping": _eligibility_mapping(cache),
    }
    with pytest.raises(ValueError, match="both labels"):
        if protocol == "split_policy":
            build_split_policy_contrast_manifests(
                cache, candidate_count=5, analysis_mode="test_mode", **kwargs
            )
        elif protocol == "common_late":
            build_common_late_test_manifests(
                cache, candidate_count=5, analysis_mode="test_mode", **kwargs
            )
        else:
            build_reverse_temporal_hst_manifest(cache, **kwargs)


def test_temporal_split_summary_describes_its_own_manifest() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    cache = _cache_index(60)
    mixed, chronological = build_split_policy_contrast_manifests(
        cache,
        candidate_count=25,
        analysis_mode="test_mode",
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )

    for manifest in (mixed, chronological):
        people = manifest[
            ["participant_key", "label_binary", "split", "participant_timestamp_utc"]
        ].drop_duplicates()
        summary = manifest.attrs["split_summary"].set_index("split")
        for split in ("train", "validation", "test"):
            current = people[people["split"].eq(split)]
            assert summary.loc[split, "participant_count"] == len(current)
            assert pd.Timestamp(summary.loc[split, "date_min"]) == current[
                "participant_timestamp_utc"
            ].min()
            assert pd.Timestamp(summary.loc[split, "date_max"]) == current[
                "participant_timestamp_utc"
            ].max()


def test_paired_representations_do_not_inflate_date_eligibility_counts() -> None:
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    base = _cache_index(40, modalities=("cough",))
    paired = _paired_representation_cache(base)
    _, chronological = build_split_policy_contrast_manifests(
        paired,
        candidate_count=25,
        analysis_mode="test_mode",
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(base),
    )

    assert chronological["valid_recording_count"].eq(1).all()
    assert chronological["invalid_recording_count"].eq(0).all()


def test_common_late_test_control_fixes_identical_latest_test_participants() -> None:
    from covid_rars.hst_protocols import build_common_late_test_manifests

    cache = _cache_index(60)
    balanced, chronological = build_common_late_test_manifests(
        cache,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    balanced_people = balanced[
        ["participant_key", "label_binary", "split"]
    ].drop_duplicates()
    chronological_people = chronological[
        ["participant_key", "label_binary", "split", "participant_timestamp_utc"]
    ].drop_duplicates()
    balanced_test = set(
        balanced_people.loc[balanced_people["split"].eq("test"), "participant_key"]
    )
    chronological_test = set(
        chronological_people.loc[
            chronological_people["split"].eq("test"), "participant_key"
        ]
    )
    latest = set(
        chronological_people.sort_values("participant_timestamp_utc")
        .tail(12)["participant_key"]
    )

    assert balanced_test == chronological_test == latest
    for split in ("train", "validation", "test"):
        left = balanced_people[balanced_people["split"].eq(split)][
            "label_binary"
        ].value_counts().to_dict()
        right = chronological_people[chronological_people["split"].eq(split)][
            "label_binary"
        ].value_counts().to_dict()
        assert left == right


def test_reverse_temporal_manifest_uses_latest_train_and_earliest_test() -> None:
    from covid_rars.hst_protocols import build_reverse_temporal_hst_manifest

    cache = _cache_index(60)
    manifest = build_reverse_temporal_hst_manifest(
        cache,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )
    people = manifest[
        ["participant_key", "participant_timestamp_utc", "split", "training_seed"]
    ].drop_duplicates().sort_values("participant_timestamp_utc")

    assert people.iloc[:12]["split"].eq("test").all()
    assert people.iloc[12:24]["split"].eq("validation").all()
    assert people.iloc[24:]["split"].eq("train").all()
    assert people["training_seed"].eq(42).all()
    assert manifest["protocol"].eq("hst_reverse_temporal_sensitivity").all()
    assert manifest["analysis_scope"].eq("sensitivity_analysis").all()
    assert manifest["analysis_role"].eq("sensitivity").all()
    assert manifest["estimand_id"].eq("reverse_temporal_direction").all()
    assert manifest["multiplicity_family"].eq("temporal_sensitivity").all()
    assert manifest["confirmatory_protocol"].eq(False).all()
    boundaries = manifest.attrs["boundary_diagnostics"]
    assert set(boundaries["boundary_name"]) == {
        "earliest_test_to_validation",
        "validation_to_latest_train",
    }


def test_common_late_calendar_balanced_manifest_has_no_chronological_boundary_audit() -> None:
    from covid_rars.hst_protocols import build_common_late_test_manifests

    cache = _cache_index(60)
    balanced, chronological = build_common_late_test_manifests(
        cache,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(cache),
    )

    assert "boundary_diagnostics_sha256" not in balanced
    assert "boundary_diagnostics" not in balanced.attrs
    diagnostics = chronological.attrs["boundary_diagnostics"]
    assert diagnostics["diagnostic_protocol"].eq(
        "hst_common_late_test_chronological_source"
    ).all()


def test_external_manifest_attaches_target_only_as_external_test_with_provenance() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    external_cache = _cache_index(
        12, dataset="coughvid", modalities=("cough",)
    )
    external_cache["label_source"] = "status_SSL"
    external_cache["label_provenance"] = "coughvid_metadata_compiled"
    external_cache["dataset_release_id"] = "coughvid-test-release"
    external_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    combined_cache = pd.concat([source_cache, external_cache], ignore_index=True)
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )

    manifest = build_external_hst_manifest(
        combined_cache,
        source_manifest,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(combined_cache),
    )
    target = manifest[manifest["dataset"].eq("coughvid")]
    source = manifest[manifest["dataset"].eq("coswara")]

    assert target["split"].eq("external_test").all()
    assert target["fold"].nunique() == 10
    assert len(target) == 12 * 10
    assert set(target["label_source"]) == {"status_SSL"}
    assert target["source_fold_manifest_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(source["split"]) <= {"train", "validation", "test"}
    assert set(source["participant_key"]).isdisjoint(set(target["participant_key"]))
    assert manifest["analysis_scope"].eq("reliability_evaluation").all()
    assert manifest["analysis_role"].eq("secondary").all()
    assert manifest["estimand_id"].eq(
        "coswara_to_coughvid_external_transfer"
    ).all()
    assert manifest["multiplicity_family"].eq(
        "prespecified_reliability"
    ).all()
    assert manifest["confirmatory_protocol"].eq(True).all()
    for column in (
        "label_source",
        "label_provenance",
        "dataset_release_id",
        "source_manifest_sha256",
        "preprocessing_variant",
    ):
        assert manifest[column].astype(str).str.len().gt(0).all()


def test_external_builder_rejects_non_track_a_source_manifest() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_reverse_temporal_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    arbitrary_source = build_reverse_temporal_hst_manifest(
        source_cache,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    combined = pd.concat([source_cache, target_cache], ignore_index=True)

    with pytest.raises(ValueError, match="Coswara Track-A"):
        build_external_hst_manifest(
            combined,
            arbitrary_source,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(combined),
        )


def test_external_builder_requires_cough_only_track_a_source() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40)
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )
    combined = pd.concat([source_cache, target_cache], ignore_index=True)

    with pytest.raises(ValueError, match="cough-only"):
        build_external_hst_manifest(
            combined,
            source_manifest,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(combined),
        )


def test_external_builder_requires_matching_representation_and_preprocessing() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["preprocessing_hash"] = _digest("different-target-preprocessing")
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )
    combined = pd.concat([source_cache, target_cache], ignore_index=True)

    with pytest.raises(ValueError, match="representation.*preprocessing"):
        build_external_hst_manifest(
            combined,
            source_manifest,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(combined),
        )


def test_external_target_fingerprint_is_frozen_after_target_filtering() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    combined = pd.concat([source_cache, target_cache], ignore_index=True)
    combined_alignment = _eligibility_mapping(combined)
    target_alignment = _eligibility_mapping(target_cache)
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )
    manifest = build_external_hst_manifest(
        combined,
        source_manifest,
        scientific_config=_scientific_config(),
        eligibility_mapping=combined_alignment,
    )

    expected = target_alignment["eligibility_alignment_fingerprint"].iloc[0]
    assert manifest["target_eligibility_alignment_fingerprint"].eq(expected).all()
    assert expected != combined_alignment[
        "eligibility_alignment_fingerprint"
    ].iloc[0]


def test_external_restriction_rejects_selected_target_unit_missing_from_mapping() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    combined = pd.concat([source_cache, target_cache], ignore_index=True)
    missing_recording = str(target_cache.iloc[-1]["recording_key"])
    incomplete_alignment = _eligibility_mapping(
        combined[~combined["recording_key"].eq(missing_recording)].copy()
    )
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )

    with pytest.raises(ValueError, match="selected target.*survive"):
        build_external_hst_manifest(
            combined,
            source_manifest,
            scientific_config=_scientific_config(),
            eligibility_mapping=incomplete_alignment,
        )


def test_external_restriction_audits_requested_and_realized_target_units() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    combined = pd.concat([source_cache, target_cache], ignore_index=True)
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )
    manifest = build_external_hst_manifest(
        combined,
        source_manifest,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(combined),
    )

    payload_json = manifest.loc[
        manifest["eligibility_audit_payload_json"].ne(""),
        "eligibility_audit_payload_json",
    ].iloc[0]
    target_audit = json.loads(payload_json)["target_alignment_audit"]
    assert target_audit["mapping_policy_id"] == "explicit_frozen_restriction_v1"
    assert target_audit["restriction_reason"] == (
        "coughvid_cough_external_target_only"
    )
    assert len(target_audit["parent_alignment_fingerprint"]) == 64
    assert target_audit["requested_analysis_unit_count"] == 12
    assert target_audit["realized_analysis_unit_count"] == 12
    assert target_audit["realized_mapping_row_count"] == 24


def test_external_builder_rejects_content_collision_across_cohorts() -> None:
    from covid_rars.hst_protocols import (
        build_external_hst_manifest,
        build_protocol_matched_hst_manifest,
    )

    source_cache = _cache_index(40, modalities=("cough",))
    target_cache = _cache_index(12, dataset="coughvid", modalities=("cough",))
    target_cache["label_source"] = "status_SSL"
    target_cache["label_provenance"] = "coughvid_metadata_compiled"
    target_cache["dataset_release_id"] = "coughvid-test-release"
    target_cache["source_manifest_sha256"] = _digest("coughvid-source-manifest")
    target_cache.loc[target_cache.index[0], "tensor_sha256"] = source_cache.loc[
        source_cache.index[0], "tensor_sha256"
    ]
    source_manifest = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=HST_SEEDS,
        scientific_config=_scientific_config(),
        eligibility_mapping=_eligibility_mapping(source_cache),
    )
    combined = pd.concat([source_cache, target_cache], ignore_index=True)

    with pytest.raises(ValueError, match="content hash leakage"):
        build_external_hst_manifest(
            combined,
            source_manifest,
            scientific_config=_scientific_config(),
            eligibility_mapping=_eligibility_mapping(combined),
        )


def test_representation_intersection_uses_recording_and_modality_keys() -> None:
    from covid_rars.hst_protocols import intersect_representation_eligibility

    paper = _cache_index(20, modalities=("cough",))
    released = _cache_index(
        20,
        modalities=("cough",),
        representation_id="released_linear_specgram_224",
    )
    missing_key = released.iloc[-1]["recording_key"]
    released = released[~released["recording_key"].eq(missing_key)].copy()
    shared = intersect_representation_eligibility(
        paper,
        released,
        scientific_config=_scientific_config(),
    )
    reordered = intersect_representation_eligibility(
        released.sample(frac=1.0, random_state=91).reset_index(drop=True),
        paper.sample(frac=1.0, random_state=92).reset_index(drop=True),
        scientific_config=_scientific_config(),
    )

    assert len(shared) == 38
    assert missing_key not in set(shared["recording_key"])
    assert shared["representation_count"].eq(2).all()
    assert shared["representation_ids"].eq(
        "paper_logmel_224|released_linear_specgram_224"
    ).all()
    assert shared["eligibility_alignment_fingerprint"].nunique() == 1
    assert shared["eligibility_alignment_fingerprint"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert shared["analysis_unit_weight"].eq(0.5).all()
    assert shared["eligibility_audit_payload_json"].ne("").sum() == 1
    assert shared["analysis_scope"].eq("representation_alignment").all()
    assert shared["analysis_role"].eq("design_context").all()
    assert shared["estimand_id"].eq("shared_representation_eligibility").all()
    assert shared["multiplicity_family"].eq("not_applicable").all()
    assert shared["confirmatory_protocol"].eq(False).all()
    for (_, _), unit in shared.groupby(["recording_key", "modality"]):
        assert set(unit["representation_id"]) == {
            "paper_logmel_224",
            "released_linear_specgram_224",
        }
    exclusions = shared.attrs["representation_exclusions"]
    assert {
        "dataset",
        "label_binary",
        "modality",
        "representation_id",
        "exclusion_reason",
        "recording_count",
        "participant_count",
    }.issubset(exclusions.columns)
    paper_excluded = exclusions[
        exclusions["representation_id"].eq("paper_logmel_224")
        & exclusions["exclusion_reason"].eq(
            "not_in_shared_representation_intersection"
        )
    ].iloc[0]
    assert paper_excluded["recording_count"] == 1
    pd.testing.assert_frame_equal(shared, reordered)


def test_representation_intersection_supports_audited_single_cache_identity() -> None:
    from covid_rars.hst_protocols import intersect_representation_eligibility

    paper = _cache_index(20, modalities=("cough",))
    excluded_index = paper.index[-1]
    excluded_key = str(paper.loc[excluded_index, "recording_key"])
    paper.loc[excluded_index, "eligible"] = False
    paper.loc[excluded_index, "reason"] = "too_short"

    shared = intersect_representation_eligibility(
        paper,
        scientific_config=_scientific_config(),
    )

    assert len(shared) == 19
    assert excluded_key not in set(shared["recording_key"].astype(str))
    assert shared["representation_count"].eq(1).all()
    assert shared["representation_ids"].eq("paper_logmel_224").all()
    assert shared["paired_representation_count"].eq(1).all()
    assert shared["paired_representation"].eq(False).all()
    assert shared["analysis_unit_weight"].eq(1.0).all()
    assert shared["eligibility_alignment_fingerprint"].nunique() == 1
    assert shared["eligibility_alignment_fingerprint"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert shared["eligibility_audit_payload_json"].ne("").sum() == 1
    exclusions = shared.attrs["representation_exclusions"]
    excluded = exclusions.loc[
        exclusions["representation_id"].eq("paper_logmel_224")
        & exclusions["exclusion_reason"].eq("too_short")
    ]
    assert len(excluded) == 1
    assert int(excluded.iloc[0]["recording_count"]) == 1


def test_representation_intersection_rejects_missing_cache_indices() -> None:
    from covid_rars.hst_protocols import intersect_representation_eligibility

    with pytest.raises(ValueError, match="At least one representation index"):
        intersect_representation_eligibility(
            scientific_config=_scientific_config(),
        )


def test_representation_intersection_binds_source_audio_across_representations() -> None:
    from covid_rars.hst_protocols import intersect_representation_eligibility

    paper = _cache_index(20, modalities=("cough",))
    released = _cache_index(
        20,
        modalities=("cough",),
        representation_id="released_linear_specgram_224",
    )
    released.loc[released.index[0], "source_audio_sha256"] = _digest(
        "different-original-audio"
    )

    with pytest.raises(ValueError, match="source_audio_sha256"):
        intersect_representation_eligibility(
            paper,
            released,
            scientific_config=_scientific_config(),
        )


def test_manifest_builder_rejects_frozen_mapping_with_cross_representation_audio_mismatch() -> None:
    from covid_rars.hst_protocols import (
        _alignment_fingerprint,
        _finalize_manifest,
        _without_hash_columns,
        build_protocol_matched_hst_manifest,
    )

    cache = _cache_index(40, modalities=("cough",))
    mapping = _without_hash_columns(_eligibility_mapping(cache))
    comparator = mapping["representation_id"].eq("compare_is10_top800")
    mapping.loc[mapping.index[comparator][0], "source_audio_sha256"] = _digest(
        "forged-comparator-source-audio"
    )
    mapping["eligibility_alignment_fingerprint"] = _alignment_fingerprint(mapping)
    forged_mapping = _finalize_manifest(mapping)

    with pytest.raises(ValueError, match="source_audio_sha256"):
        build_protocol_matched_hst_manifest(
            cache,
            seeds=HST_SEEDS,
            scientific_config=_scientific_config(),
            eligibility_mapping=forged_mapping,
        )


@dataclass(frozen=True)
class _ScientificConfig:
    architecture: dict[str, object]
    source_checkpoint: dict[str, object]
    preprocessing: dict[str, object]
    augmentation: dict[str, object]
    optimizer: dict[str, object]
    stopping_rule: dict[str, object]
    participant_aggregation: str
    fusion: dict[str, object]
    thresholding: dict[str, object]
    sampling: dict[str, object]
    calibration: dict[str, object]
    metric_settings: dict[str, object]
    manifest_path: Path
    protocol_label: str


def test_scientific_fingerprint_excludes_protocol_identity_but_tracks_methods() -> None:
    from covid_rars.hst_protocols import scientific_configuration_fingerprint

    first = _ScientificConfig(
        architecture={"name": "hst_base", "classes": 2},
        source_checkpoint={"sha256": _digest("checkpoint")},
        preprocessing={"representation": "paper_logmel_224"},
        augmentation={"rotation_degrees": 5},
        optimizer={"name": "adamw", "learning_rate": 1e-5},
        stopping_rule={"metric": "participant_f1", "patience": 100},
        participant_aggregation="recording_probability_mean",
        fusion={"primary": "uniform_mean"},
        thresholding={"primary": 0.5},
        sampling={"unit": "participant", "class_balance": "natural"},
        calibration={"method": "none", "fit_split": "validation"},
        metric_settings={"ece_bins": 10},
        manifest_path=Path("manifests/internal.csv"),
        protocol_label="internal",
    )
    second = replace(
        first,
        manifest_path=Path("manifests/temporal.csv"),
        protocol_label="temporal",
    )
    changed_method = replace(first, optimizer={"name": "adamw", "learning_rate": 2e-5})

    first_hash = scientific_configuration_fingerprint(first)
    assert first_hash == scientific_configuration_fingerprint(second)
    assert first_hash != scientific_configuration_fingerprint(changed_method)
    assert len(first_hash) == 64


def test_scientific_fingerprint_preserves_nested_scientific_protocol_fields() -> None:
    from covid_rars.hst_protocols import scientific_configuration_fingerprint

    first = _scientific_config()
    first["preprocessing"] = {
        "representation": "paper_logmel_224",
        "protocol": "amplitude_then_log",
    }
    second = _scientific_config()
    second["preprocessing"] = {
        "representation": "paper_logmel_224",
        "protocol": "log_then_amplitude",
    }

    assert scientific_configuration_fingerprint(first) != (
        scientific_configuration_fingerprint(second)
    )


def test_scientific_fingerprint_preserves_top_level_protocol_mapping_content() -> None:
    from covid_rars.hst_protocols import scientific_configuration_fingerprint

    first = _scientific_config()
    first["protocol"] = {
        "split_policy": "participant_chronological",
        "train_fraction": 0.6,
    }
    second = _scientific_config()
    second["protocol"] = {
        "split_policy": "participant_chronological",
        "train_fraction": 0.7,
    }

    assert scientific_configuration_fingerprint(first) != (
        scientific_configuration_fingerprint(second)
    )

    identity_only = _scientific_config()
    identity_only["protocol"] = {
        "protocol_label": "temporal_sensitivity",
        "split_policy": "participant_chronological",
        "train_fraction": 0.6,
    }
    assert scientific_configuration_fingerprint(first) == (
        scientific_configuration_fingerprint(identity_only)
    )


@pytest.mark.parametrize(
    "section",
    (
        "architecture",
        "source_checkpoint",
        "preprocessing",
        "augmentation",
        "optimizer",
        "stopping_rule",
        "participant_aggregation",
        "fusion",
        "thresholding",
        "sampling",
        "calibration",
        "metric_settings",
    ),
)
def test_scientific_fingerprint_rejects_empty_required_sections(section: str) -> None:
    from covid_rars.hst_protocols import scientific_configuration_fingerprint

    config = _scientific_config()
    config[section] = {}

    with pytest.raises(ValueError, match=section):
        scientific_configuration_fingerprint(config)


@pytest.mark.parametrize("section", ("sampling", "calibration"))
def test_scientific_fingerprint_requires_sampling_and_calibration(
    section: str,
) -> None:
    from covid_rars.hst_protocols import scientific_configuration_fingerprint

    config = _scientific_config()
    del config[section]

    with pytest.raises(ValueError, match=section):
        scientific_configuration_fingerprint(config)
