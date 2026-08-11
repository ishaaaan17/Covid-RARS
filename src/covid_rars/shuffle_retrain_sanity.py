from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.metadata_confounding import run_metadata_confounding_audit


def shuffle_labels_by_participant(metadata: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = metadata.copy()
    labeled = out[out["label_binary"].isin(["positive", "negative"])].copy()
    if labeled.empty:
        return out
    participant_labels = (
        labeled.groupby("participant_id", dropna=False)["label_binary"]
        .agg(lambda values: str(values.value_counts().index[0]))
        .reset_index()
    )
    shuffled = participant_labels["label_binary"].to_numpy(dtype=object).copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(participant_labels["participant_id"].astype(str), shuffled))
    mask = out["participant_id"].astype(str).isin(mapping)
    out.loc[mask, "label_binary"] = out.loc[mask, "participant_id"].astype(str).map(mapping)
    return out


def _best_auroc(metrics: pd.DataFrame, feature_set: str) -> float:
    if metrics.empty:
        return float("nan")
    work = metrics[metrics.get("audit_model", pd.Series(index=metrics.index, dtype=object)).astype(str).eq(str(feature_set))].copy()
    if work.empty:
        work = metrics[metrics.get("feature_strategy", pd.Series(index=metrics.index, dtype=object)).astype(str).eq(str(feature_set))].copy()
    if work.empty:
        return float("nan")
    auroc = pd.to_numeric(work.get("auroc"), errors="coerce")
    return float(auroc.max()) if auroc.notna().any() else float("nan")


def run_metadata_shuffle_retrain_sanity(
    metadata: pd.DataFrame,
    feature_sets: list[str] | None = None,
    n_permutations: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Retrain metadata-only models after participant-level label shuffling."""
    feature_sets = feature_sets or ["full_safe_metadata", "symptoms_only", "demographic_protocol_only"]
    observed = run_metadata_confounding_audit(metadata, feature_sets=feature_sets, random_state=random_state)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    shuffled_values: dict[str, list[float]] = {feature_set: [] for feature_set in feature_sets}
    for permutation_index in range(max(1, int(n_permutations))):
        shuffled = shuffle_labels_by_participant(metadata, rng)
        try:
            result = run_metadata_confounding_audit(
                shuffled,
                feature_sets=feature_sets,
                random_state=random_state + permutation_index + 1,
            )
        except Exception:
            continue
        for feature_set in feature_sets:
            value = _best_auroc(result.metrics, feature_set)
            if np.isfinite(value):
                shuffled_values[feature_set].append(value)

    for feature_set in feature_sets:
        values = np.asarray(shuffled_values[feature_set], dtype=float)
        observed_auroc = _best_auroc(observed.metrics, feature_set)
        if values.size:
            low, high = np.quantile(values, [0.025, 0.975])
            mean = float(values.mean())
        else:
            low = high = mean = float("nan")
        rows.append(
            {
                "sanity_check": "metadata_retrain_with_shuffled_labels",
                "audit_model": feature_set,
                "observed_auroc": float(observed_auroc),
                "shuffled_auroc_mean": mean,
                "shuffled_auroc_ci_low": float(low),
                "shuffled_auroc_ci_high": float(high),
                "n_permutations": int(values.size),
                "random_state": int(random_state),
            }
        )
    return pd.DataFrame(rows)
