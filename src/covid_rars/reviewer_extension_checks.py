from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.compare_is10_rescue import _rank_features_for_frame
from covid_rars.features import feature_columns
from covid_rars.metadata_confounding import build_audit_feature_frame
from covid_rars.metrics import binary_metric_bundle, labels_to_binary
from covid_rars.temporal_month_causal import build_matched_temporal_participants


DEFAULT_SELECTOR_COLUMNS = [
    "prediction_source",
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "submodality",
    "modality_combination",
    "fusion_method",
    "feature_strategy",
    "selected_feature_k",
    "metric_split",
    "split",
    "audit_model",
]


def _safe_numeric(series: pd.Series, fill_value: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def _valid_predictions(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    required = {probability_column, label_column}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Predictions missing required columns: {sorted(missing)}")
    frame = predictions[predictions[label_column].isin(["positive", "negative"])].copy()
    frame[probability_column] = _safe_numeric(frame[probability_column])
    frame = frame[np.isfinite(frame[probability_column])].copy()
    frame[probability_column] = frame[probability_column].clip(0.0, 1.0)
    if "metric_split" not in frame.columns and "split" in frame.columns:
        frame["metric_split"] = frame["split"].astype(str)
    if "split" not in frame.columns and "metric_split" in frame.columns:
        frame["split"] = frame["metric_split"].astype(str)
    return frame


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in frame.columns]


def _iter_groups(frame: pd.DataFrame, group_columns: list[str]) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    if not group_columns:
        yield {}, frame
        return
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        yield dict(zip(group_columns, key)), group


def _aggregate_unit_predictions(
    predictions: pd.DataFrame,
    id_columns: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    frame = _valid_predictions(predictions, probability_column=probability_column, label_column=label_column)
    if frame.empty:
        return frame
    ids = _available(id_columns or ["recording_id", "participant_id"], frame)
    if not ids:
        raise KeyError("Predictions need recording_id or participant_id for unit-level analysis")
    groups = [*ids, *_available(DEFAULT_SELECTOR_COLUMNS, frame)]
    aggregated = (
        frame.groupby(groups, dropna=False)
        .agg(
            label_binary=(label_column, "first"),
            probability=(probability_column, "mean"),
            threshold=("threshold", "first") if "threshold" in frame.columns else (probability_column, lambda _: 0.5),
        )
        .reset_index()
    )
    aggregated["threshold"] = _safe_numeric(aggregated["threshold"], fill_value=0.5)
    return aggregated


def build_specification_curve(
    metrics: pd.DataFrame,
    metric_split: str = "test",
    metric_column: str = "auroc",
) -> pd.DataFrame:
    """Rank all available model specifications for cherry-picking checks."""
    if metrics.empty:
        return pd.DataFrame()
    work = metrics.copy()
    if "skipped" in work.columns:
        work = work[~work["skipped"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    split_col = "metric_split" if "metric_split" in work.columns else "split" if "split" in work.columns else None
    if split_col is not None:
        work = work[work[split_col].astype(str).eq(str(metric_split))].copy()
    if metric_column not in work.columns:
        raise KeyError(f"metrics missing {metric_column}")
    work[metric_column] = _safe_numeric(work[metric_column])
    work = work[np.isfinite(work[metric_column])].copy()
    if work.empty:
        return pd.DataFrame()
    sort_cols = [metric_column]
    if "auprc" in work.columns and metric_column != "auprc":
        work["auprc"] = _safe_numeric(work["auprc"])
        sort_cols.append("auprc")
    group_cols = _available(["evaluation_protocol", "dataset", "metric_split", "split"], work)
    if group_cols:
        ranked = []
        for _, group in work.groupby(group_cols, dropna=False):
            group = group.sort_values(sort_cols, ascending=False).reset_index(drop=True)
            group["specification_rank"] = np.arange(1, len(group) + 1, dtype=int)
            group["n_specifications_in_protocol"] = int(len(group))
            ranked.append(group)
        out = pd.concat(ranked, ignore_index=True, sort=False)
    else:
        out = work.sort_values(sort_cols, ascending=False).reset_index(drop=True)
        out["specification_rank"] = np.arange(1, len(out) + 1, dtype=int)
        out["n_specifications_in_protocol"] = int(len(out))
    selector_cols = _available(DEFAULT_SELECTOR_COLUMNS, out)
    cols = [*selector_cols, "specification_rank", "n_specifications_in_protocol", metric_column]
    for col in ["auprc", "balanced_accuracy", "f1", "n_samples", "n_participants", "table_source"]:
        if col in out.columns and col not in cols:
            cols.append(col)
    return out[cols].reset_index(drop=True)


def build_quality_label_month_table(
    metadata: pd.DataFrame,
    date_column: str = "recording_date",
    quality_column: str = "quality_flag",
) -> pd.DataFrame:
    """Crosstab recording quality against labels and collection month."""
    if metadata.empty or "label_binary" not in metadata.columns:
        return pd.DataFrame()
    frame = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    if frame.empty:
        return pd.DataFrame()
    if quality_column not in frame.columns:
        frame[quality_column] = "unknown"
    date = pd.to_datetime(frame[date_column], errors="coerce") if date_column in frame.columns else pd.Series(pd.NaT, index=frame.index)
    frame["recording_year_month"] = date.dt.to_period("M").astype(str).where(date.notna(), "unknown")
    group_cols = _available(["dataset", "split", "modality", quality_column, "recording_year_month"], frame)
    rows: list[dict[str, object]] = []
    for values, group in _iter_groups(frame, group_cols):
        labels = group["label_binary"].astype(str)
        n_positive = int(labels.eq("positive").sum())
        n_negative = int(labels.eq("negative").sum())
        total = n_positive + n_negative
        rows.append(
            {
                **values,
                "quality_flag": values.get(quality_column, "unknown"),
                "n_rows": int(len(group)),
                "n_participants": int(group["participant_id"].nunique()) if "participant_id" in group.columns else int(len(group)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_prevalence": float(n_positive / total) if total else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    sort_cols = _available(["dataset", "split", "modality", "recording_year_month", "quality_flag"], out)
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def build_decision_curve_table(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Vickers-style decision curve analysis for binary screening endpoints."""
    frame = _valid_predictions(predictions)
    groups = _available(group_columns or DEFAULT_SELECTOR_COLUMNS, frame)
    thresholds = thresholds or [round(x, 2) for x in np.arange(0.05, 0.501, 0.05)]
    rows: list[dict[str, object]] = []
    for group_values, group in _iter_groups(frame, groups):
        if group["label_binary"].nunique() < 2:
            continue
        y_true = labels_to_binary(group["label_binary"])
        y_prob = group["probability"].astype(float).to_numpy()
        n = len(group)
        prevalence = float(np.mean(y_true))
        for threshold in thresholds:
            threshold = float(threshold)
            if threshold <= 0.0 or threshold >= 1.0:
                continue
            y_pred = (y_prob >= threshold).astype(int)
            tp = float(np.sum((y_pred == 1) & (y_true == 1)))
            fp = float(np.sum((y_pred == 1) & (y_true == 0)))
            odds = threshold / (1.0 - threshold)
            model_nb = tp / n - fp / n * odds
            treat_all_nb = prevalence - (1.0 - prevalence) * odds
            rows.append(
                {
                    **group_values,
                    "threshold_probability": threshold,
                    "model_net_benefit": float(model_nb),
                    "treat_all_net_benefit": float(treat_all_nb),
                    "treat_none_net_benefit": 0.0,
                    "net_benefit_minus_treat_all": float(model_nb - treat_all_nb),
                    "net_benefit_minus_treat_none": float(model_nb),
                    "prevalence": prevalence,
                    "n_samples": int(n),
                    "n_positive": int(np.sum(y_true == 1)),
                    "n_negative": int(np.sum(y_true == 0)),
                }
            )
    return pd.DataFrame(rows)


def _metadata_unit_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame()
    id_col = "recording_id" if "recording_id" in metadata.columns else "participant_id" if "participant_id" in metadata.columns else None
    if id_col is None:
        return metadata.copy()
    aggregations: dict[str, object] = {}
    for col in metadata.columns:
        if col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(metadata[col]):
            aggregations[col] = "mean"
        else:
            aggregations[col] = "first"
    return metadata.groupby(id_col, as_index=False, dropna=False).agg(aggregations)


def _merge_predictions_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    frame = _aggregate_unit_predictions(predictions)
    if frame.empty or metadata.empty:
        return pd.DataFrame()
    metadata_units = _metadata_unit_frame(metadata)
    if "recording_id" in frame.columns and "recording_id" in metadata_units.columns:
        return frame.merge(metadata_units, on="recording_id", how="left", suffixes=("", "_metadata"))
    if "participant_id" in frame.columns and "participant_id" in metadata_units.columns:
        return frame.merge(metadata_units, on="participant_id", how="left", suffixes=("", "_metadata"))
    return pd.DataFrame()


def _corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    l = _safe_numeric(left)
    r = _safe_numeric(right)
    mask = np.isfinite(l.to_numpy()) & np.isfinite(r.to_numpy())
    if int(mask.sum()) < 2:
        return float("nan")
    return float(l[mask].corr(r[mask], method=method))


def build_duration_shortcut_table(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    group_columns: list[str] | None = None,
    duration_column: str = "duration_sec",
) -> pd.DataFrame:
    merged = _merge_predictions_metadata(predictions, metadata)
    if merged.empty or duration_column not in merged.columns:
        return pd.DataFrame(
            [{"analysis": "duration_shortcut", "skipped": True, "skip_reason": f"{duration_column} not available or no aligned rows"}]
        )
    groups = _available(group_columns or DEFAULT_SELECTOR_COLUMNS, merged)
    rows: list[dict[str, object]] = []
    for values, group in _iter_groups(merged, groups):
        group = group[np.isfinite(_safe_numeric(group[duration_column]))].copy()
        if group.empty:
            continue
        rows.append(
            {
                **values,
                "analysis": "duration_shortcut",
                "n_aligned": int(len(group)),
                "duration_mean": float(_safe_numeric(group[duration_column]).mean()),
                "duration_std": float(_safe_numeric(group[duration_column]).std(ddof=0)),
                "probability_duration_pearson": _corr(group["probability"], group[duration_column]),
                "probability_duration_spearman": _corr(group["probability"], group[duration_column], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def _age_band(series: pd.Series) -> pd.Series:
    numeric = _safe_numeric(series)
    bins = [-np.inf, 30, 45, 60, np.inf]
    labels = ["<=30", "31-45", "46-60", ">60"]
    return pd.cut(numeric, bins=bins, labels=labels).astype(str).where(numeric.notna(), "unknown")


def build_performance_equity_table(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    subgroup_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
    min_subgroup_size: int = 20,
) -> pd.DataFrame:
    merged = _merge_predictions_metadata(predictions, metadata)
    if merged.empty:
        return pd.DataFrame([{"analysis": "performance_equity", "skipped": True, "skip_reason": "no aligned rows"}])
    subgroups = subgroup_columns or ["age_band", "gender", "country", "recording_year_month"]
    if "age_band" in subgroups and "age_band" not in merged.columns and "age" in merged.columns:
        merged["age_band"] = _age_band(merged["age"])
    if "recording_year_month" in subgroups and "recording_year_month" not in merged.columns and "recording_date" in merged.columns:
        date = pd.to_datetime(merged["recording_date"], errors="coerce")
        merged["recording_year_month"] = date.dt.to_period("M").astype(str).where(date.notna(), "unknown")
    groups = _available(group_columns or DEFAULT_SELECTOR_COLUMNS, merged)
    rows: list[dict[str, object]] = []
    for values, group in _iter_groups(merged, groups):
        for subgroup in subgroups:
            if subgroup not in group.columns:
                continue
            for value, subgroup_frame in group.groupby(subgroup, dropna=False):
                subgroup_frame = subgroup_frame[subgroup_frame["label_binary"].isin(["positive", "negative"])].copy()
                if len(subgroup_frame) < int(min_subgroup_size) or subgroup_frame["label_binary"].nunique() < 2:
                    continue
                threshold = float(_safe_numeric(subgroup_frame.get("threshold", pd.Series([0.5])), fill_value=0.5).median())
                bundle = binary_metric_bundle(
                    labels_to_binary(subgroup_frame["label_binary"]),
                    subgroup_frame["probability"].astype(float).to_numpy(),
                    threshold=threshold,
                )
                bundle.update(
                    {
                        **values,
                        "analysis": "performance_equity",
                        "subgroup_column": subgroup,
                        "subgroup_value": str(value),
                        "n_participants": int(subgroup_frame["participant_id"].nunique()) if "participant_id" in subgroup_frame.columns else int(len(subgroup_frame)),
                        "positive_prevalence": float(subgroup_frame["label_binary"].eq("positive").mean()),
                    }
                )
                rows.append(bundle)
    return pd.DataFrame(rows)


def _participant_split_predictions(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = _valid_predictions(predictions)
    if "participant_id" not in frame.columns:
        raise KeyError("participant_id is required for nested comparison")
    split_mask = frame["split"].astype(str).eq(str(split)) if "split" in frame.columns else frame["metric_split"].astype(str).eq(str(split))
    frame = frame[split_mask].copy()
    if frame.empty:
        return frame
    return (
        frame.groupby("participant_id", dropna=False)
        .agg(label_binary=("label_binary", "first"), probability=("probability", "mean"))
        .reset_index()
    )


def build_metadata_predictions_for_nested_comparison(
    metadata: pd.DataFrame,
    feature_set: str = "full_safe_metadata",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a metadata-only model and emit validation/test probabilities.

    This is intentionally separate from the historical metadata-confounding CSV
    because older artifacts may only contain test predictions. The nested
    metadata+audio comparison needs a validation fold for the combiner.
    """
    required = {"participant_id", "label_binary", "split"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"metadata missing required columns for nested comparison: {sorted(missing)}")
    frame = metadata[
        metadata["label_binary"].isin(["positive", "negative"])
        & metadata["split"].isin(["train", "validation", "test"])
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    # Use participant-level metadata so repeated recordings do not dominate the
    # nested comparison.
    participant_rows = (
        frame.sort_values(["participant_id", "split"])
        .groupby("participant_id", as_index=False, dropna=False)
        .first()
    )
    train = participant_rows[participant_rows["split"].eq("train")].copy()
    validation = participant_rows[participant_rows["split"].eq("validation")].copy()
    test = participant_rows[participant_rows["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty or train["label_binary"].nunique() < 2:
        return pd.DataFrame()
    train_x_raw, _ = build_audit_feature_frame(train, feature_set=feature_set)
    validation_x_raw, _ = build_audit_feature_frame(validation, feature_set=feature_set)
    test_x_raw, _ = build_audit_feature_frame(test, feature_set=feature_set)
    columns = sorted(set(train_x_raw.columns) | set(validation_x_raw.columns) | set(test_x_raw.columns))
    train_x = train_x_raw.reindex(columns=columns, fill_value=0.0)
    validation_x = validation_x_raw.reindex(columns=columns, fill_value=0.0)
    test_x = test_x_raw.reindex(columns=columns, fill_value=0.0)
    varying = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
    if not varying:
        return pd.DataFrame()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state)),
        ]
    )
    model.fit(train_x[varying], labels_to_binary(train["label_binary"]))
    frames: list[pd.DataFrame] = []
    for split_name, split_frame, split_x in [
        ("validation", validation, validation_x),
        ("test", test, test_x),
    ]:
        probability = model.predict_proba(split_x[varying])[:, 1]
        out = split_frame[["participant_id", "label_binary", "split"]].copy()
        if "recording_id" in split_frame.columns:
            out["recording_id"] = split_frame["recording_id"].to_numpy()
        out["probability"] = probability
        out["model_name"] = "metadata_nested_logistic_regression"
        out["audit_model"] = feature_set
        out["feature_strategy"] = feature_set
        out["split"] = split_name
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_nested_metadata_audio_comparison(
    audio_predictions: pd.DataFrame,
    metadata_predictions: pd.DataFrame,
    validation_split: str = "validation",
    test_split: str = "test",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether audio adds discrimination after metadata-only predictions."""
    try:
        audio_val = _participant_split_predictions(audio_predictions, validation_split)
        audio_test = _participant_split_predictions(audio_predictions, test_split)
        meta_val = _participant_split_predictions(metadata_predictions, validation_split)
        meta_test = _participant_split_predictions(metadata_predictions, test_split)
    except KeyError as exc:
        return pd.DataFrame([{"analysis": "nested_metadata_audio", "skipped": True, "skip_reason": str(exc)}]), pd.DataFrame()
    val = audio_val.merge(meta_val, on="participant_id", suffixes=("_audio", "_metadata"), how="inner")
    test = audio_test.merge(meta_test, on="participant_id", suffixes=("_audio", "_metadata"), how="inner")
    for frame in [val, test]:
        if not frame.empty:
            frame.drop(frame[~frame["label_binary_audio"].astype(str).eq(frame["label_binary_metadata"].astype(str))].index, inplace=True)
    if val.empty or test.empty or val["label_binary_audio"].nunique() < 2 or test["label_binary_audio"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "analysis": "nested_metadata_audio",
                    "skipped": True,
                    "skip_reason": "validation/test prediction pairs unavailable or single-class",
                    "n_validation_aligned": int(len(val)),
                    "n_test_aligned": int(len(test)),
                }
            ]
        ), pd.DataFrame()
    x_val = val[["probability_metadata", "probability_audio"]].astype(float)
    y_val = labels_to_binary(val["label_binary_audio"])
    x_test = test[["probability_metadata", "probability_audio"]].astype(float)
    y_test = labels_to_binary(test["label_binary_audio"])
    combiner = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    combiner.fit(x_val, y_val)
    combined_prob = combiner.predict_proba(x_test)[:, 1]
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    probs = {
        "metadata_only": x_test["probability_metadata"].to_numpy(dtype=float),
        "audio_only": x_test["probability_audio"].to_numpy(dtype=float),
        "metadata_plus_audio": combined_prob,
    }
    metadata_auc = float(roc_auc_score(y_test, probs["metadata_only"]))
    for name, prob in probs.items():
        bundle = binary_metric_bundle(y_test, prob, threshold=0.5)
        if name == "metadata_only":
            delong = {
                "paired_delong_delta_vs_metadata": 0.0,
                "paired_delong_delta_ci_low_vs_metadata": 0.0,
                "paired_delong_delta_ci_high_vs_metadata": 0.0,
                "paired_delong_p_value_vs_metadata": float("nan"),
                "paired_delong_n_vs_metadata": int(len(y_test)),
            }
        else:
            delong_result = paired_delong_auc_comparison(y_test, prob, probs["metadata_only"])
            delong = {
                "paired_delong_delta_vs_metadata": delong_result.get("delta", float("nan")),
                "paired_delong_delta_ci_low_vs_metadata": delong_result.get("delta_ci_low", float("nan")),
                "paired_delong_delta_ci_high_vs_metadata": delong_result.get("delta_ci_high", float("nan")),
                "paired_delong_p_value_vs_metadata": delong_result.get("p_value", float("nan")),
                "paired_delong_n_vs_metadata": delong_result.get("n_paired", 0),
            }
        bundle.update(
            {
                "analysis": "nested_metadata_audio",
                "nested_model": name,
                "comparison_level": "participant_prediction_level",
                "metric_split": test_split,
                "n_validation_aligned": int(len(val)),
                "n_test_aligned": int(len(test)),
                "incremental_auroc_over_metadata": float(bundle["auroc"] - metadata_auc),
                "skipped": False,
                **delong,
            }
        )
        metric_rows.append(bundle)
        out = test[["participant_id", "label_binary_audio"]].rename(columns={"label_binary_audio": "label_binary"}).copy()
        out["probability"] = prob
        out["nested_model"] = name
        out["analysis"] = "nested_metadata_audio"
        out["split"] = test_split
        prediction_frames.append(out)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True, sort=False)


def build_support_overlap_diagnostic(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    max_features: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate positivity/support overlap through a source-vs-external classifier."""
    if "split" in source_features.columns:
        source_train = source_features[source_features["split"].astype(str).eq("train")].copy()
        if not source_train.empty:
            source_features = source_train
            source_split_scope = "source_train"
        else:
            source_split_scope = "source_all_available"
    else:
        source_split_scope = "source_all_available"
    common = sorted(set(feature_columns(source_features)) & set(feature_columns(external_features)))
    if not common:
        return pd.DataFrame([{"analysis": "support_overlap_positivity", "skipped": True, "skip_reason": "no common numeric features"}])
    variance = source_features[common].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).var(axis=0)
    common = variance.sort_values(ascending=False).head(int(max_features)).index.astype(str).tolist()
    source_x = source_features[common].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    external_x = external_features[common].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    x = pd.concat([source_x, external_x], ignore_index=True)
    y = np.r_[np.zeros(len(source_x), dtype=int), np.ones(len(external_x), dtype=int)]
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
        return pd.DataFrame([{"analysis": "support_overlap_positivity", "skipped": True, "skip_reason": "insufficient source/external rows"}])
    n_splits = min(5, int(np.min(np.bincount(y))))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state)),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prob_external = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    source_prob = prob_external[: len(source_x)]
    external_prob = prob_external[len(source_x) :]
    source_q05, source_q95 = np.quantile(source_prob, [0.05, 0.95])
    external_in_source = (external_prob >= source_q05) & (external_prob <= source_q95)
    row = {
        "analysis": "support_overlap_positivity",
        "domain_classifier_auroc": float(roc_auc_score(y, prob_external)),
        "n_source": int(len(source_x)),
        "n_external": int(len(external_x)),
        "source_split_scope": source_split_scope,
        "n_common_features": int(len(common)),
        "source_domain_probability_mean": float(source_prob.mean()),
        "external_domain_probability_mean": float(external_prob.mean()),
        "source_domain_probability_q05": float(source_q05),
        "source_domain_probability_q95": float(source_q95),
        "external_within_source_domain_probability_band_fraction": float(np.mean(external_in_source)),
        "external_probably_outside_source_support_fraction": float(np.mean(~external_in_source)),
        "external_high_external_probability_fraction": float(np.mean(external_prob >= 0.95)),
        "source_high_external_probability_fraction": float(np.mean(source_prob >= 0.95)),
        "skipped": False,
    }
    return pd.DataFrame([row])


def build_feature_selection_stability(
    features: pd.DataFrame,
    top_k: int = 800,
    ranker: str = "univariate",
    date_column: str = "recording_date",
    random_state: int = 42,
) -> pd.DataFrame:
    cols = feature_columns(features)
    if not cols:
        return pd.DataFrame([{"analysis": "feature_selection_stability", "skipped": True, "skip_reason": "no numeric features"}])
    frame = features[features["label_binary"].isin(["positive", "negative"])].copy()
    if "split" in frame.columns:
        train = frame[frame["split"].astype(str).eq("train")].copy()
        if not train.empty:
            frame = train
    if date_column not in frame.columns:
        return pd.DataFrame([{"analysis": "feature_selection_stability", "skipped": True, "skip_reason": f"{date_column} missing"}])
    frame["_date"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame[frame["_date"].notna()].copy()
    if frame.empty:
        return pd.DataFrame([{"analysis": "feature_selection_stability", "skipped": True, "skip_reason": "no dated rows"}])
    cutoff = frame["_date"].median()
    early = frame[frame["_date"] <= cutoff].copy()
    late = frame[frame["_date"] > cutoff].copy()
    if early["label_binary"].nunique() < 2 or late["label_binary"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "analysis": "feature_selection_stability",
                    "skipped": True,
                    "skip_reason": "early or late subset is single-class",
                    "n_early_rows": int(len(early)),
                    "n_late_rows": int(len(late)),
                }
            ]
        )
    early_scores = _rank_features_for_frame(early, cols, ranker=ranker, random_state=random_state)
    late_scores = _rank_features_for_frame(late, cols, ranker=ranker, random_state=random_state + 1)
    k = min(int(top_k), len(cols))
    early_top = set(pd.Series(cols).iloc[np.argsort(np.nan_to_num(early_scores, nan=0.0))[::-1][:k]].astype(str))
    late_top = set(pd.Series(cols).iloc[np.argsort(np.nan_to_num(late_scores, nan=0.0))[::-1][:k]].astype(str))
    union = early_top | late_top
    intersection = early_top & late_top
    return pd.DataFrame(
        [
            {
                "analysis": "feature_selection_stability",
                "ranker": ranker,
                "top_k": int(k),
                "date_cutoff": str(cutoff.date()) if pd.notna(cutoff) else "",
                "n_early_rows": int(len(early)),
                "n_late_rows": int(len(late)),
                "n_early_positive": int(early["label_binary"].eq("positive").sum()),
                "n_late_positive": int(late["label_binary"].eq("positive").sum()),
                "overlap_count": int(len(intersection)),
                "union_count": int(len(union)),
                "jaccard_overlap": float(len(intersection) / len(union)) if union else float("nan"),
                "early_only_count": int(len(early_top - late_top)),
                "late_only_count": int(len(late_top - early_top)),
                "top_shared_features": ";".join(sorted(intersection)[:50]),
                "skipped": False,
            }
        ]
    )


def build_matched_cohort_quality_table(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame()
    pairs = build_matched_temporal_participants(metadata)
    if pairs.empty:
        return pd.DataFrame([{"analysis": "matched_cohort_quality", "skipped": True, "skip_reason": "no matched temporal pairs"}])
    quality_cols = [col for col in ["quality_flag", "age", "gender", "country", "duration_sec"] if col in metadata.columns]
    participant_meta = metadata.groupby("participant_id", as_index=False).agg(
        {col: ("mean" if pd.api.types.is_numeric_dtype(metadata[col]) else "first") for col in quality_cols}
    )
    train = pairs[["matched_pair_id", "train_participant_id"]].rename(columns={"train_participant_id": "participant_id"}).merge(
        participant_meta, on="participant_id", how="left"
    )
    test = pairs[["matched_pair_id", "test_participant_id"]].rename(columns={"test_participant_id": "participant_id"}).merge(
        participant_meta, on="participant_id", how="left"
    )
    rows: list[dict[str, object]] = []
    for col in quality_cols:
        if col not in train.columns or col not in test.columns:
            continue
        if pd.api.types.is_numeric_dtype(participant_meta[col]):
            train_values = _safe_numeric(train[col])
            test_values = _safe_numeric(test[col])
            pooled = float(np.sqrt((train_values.var(ddof=0) + test_values.var(ddof=0)) / 2.0))
            smd = float((test_values.mean() - train_values.mean()) / pooled) if pooled > 0 else 0.0
            rows.append(
                {
                    "analysis": "matched_cohort_quality",
                    "covariate": col,
                    "comparison_type": "numeric_smd_test_minus_train",
                    "train_value": float(train_values.mean()),
                    "test_value": float(test_values.mean()),
                    "standardized_mean_difference": smd,
                    "abs_standardized_mean_difference": abs(smd),
                    "n_matched_pairs": int(len(pairs)),
                    "skipped": False,
                }
            )
        else:
            train_counts = train[col].fillna("unknown").astype(str).value_counts(normalize=True)
            test_counts = test[col].fillna("unknown").astype(str).value_counts(normalize=True)
            levels = sorted(set(train_counts.index) | set(test_counts.index))
            for level in levels:
                diff = float(test_counts.get(level, 0.0) - train_counts.get(level, 0.0))
                rows.append(
                    {
                        "analysis": "matched_cohort_quality",
                        "covariate": col,
                        "level": level,
                        "comparison_type": "proportion_difference_test_minus_train",
                        "train_value": float(train_counts.get(level, 0.0)),
                        "test_value": float(test_counts.get(level, 0.0)),
                        "proportion_difference": diff,
                        "abs_proportion_difference": abs(diff),
                        "n_matched_pairs": int(len(pairs)),
                        "skipped": False,
                    }
                )
    return pd.DataFrame(rows)


def build_context_control_exposure_table(
    metadata: pd.DataFrame,
    candidate_columns: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit one-column context models to test broad label-context entanglement."""
    candidate_columns = candidate_columns or [
        "sample_rate_original",
        "quality_flag",
        "manual_quality_label",
        "country",
        "device",
        "browser",
        "platform",
        "language",
        "preferred_language",
    ]
    required = {"participant_id", "label_binary", "split"}
    if metadata.empty or not required.issubset(metadata.columns):
        return pd.DataFrame([{"analysis": "context_control_exposure", "skipped": True, "skip_reason": "metadata lacks participant_id/label_binary/split"}])
    rows: list[dict[str, object]] = []
    for column in [col for col in candidate_columns if col in metadata.columns]:
        frame = metadata[
            metadata["label_binary"].isin(["positive", "negative"])
            & metadata["split"].isin(["train", "validation", "test"])
        ].copy()
        participant_rows = frame.sort_values(["participant_id", "split"]).groupby("participant_id", as_index=False).first()
        train = participant_rows[participant_rows["split"].eq("train")].copy()
        test = participant_rows[participant_rows["split"].eq("test")].copy()
        if train.empty or test.empty or train["label_binary"].nunique() < 2 or test["label_binary"].nunique() < 2:
            rows.append(
                {
                    "analysis": "context_control_exposure",
                    "context_column": column,
                    "skipped": True,
                    "skip_reason": "missing two-class train/test rows",
                }
            )
            continue
        train_x_raw, _ = build_audit_feature_frame(train, feature_set="context_control", feature_columns=[column])
        test_x_raw, _ = build_audit_feature_frame(test, feature_set="context_control", feature_columns=[column])
        columns = sorted(set(train_x_raw.columns) | set(test_x_raw.columns))
        train_x = train_x_raw.reindex(columns=columns, fill_value=0.0)
        test_x = test_x_raw.reindex(columns=columns, fill_value=0.0)
        varying = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
        if not varying:
            rows.append(
                {
                    "analysis": "context_control_exposure",
                    "context_column": column,
                    "skipped": True,
                    "skip_reason": "column has no train-varying features",
                }
            )
            continue
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
            ]
        )
        model.fit(train_x[varying], labels_to_binary(train["label_binary"]))
        prob = model.predict_proba(test_x[varying])[:, 1]
        bundle = binary_metric_bundle(labels_to_binary(test["label_binary"]), prob, threshold=0.5)
        bundle.update(
            {
                "analysis": "context_control_exposure",
                "context_column": column,
                "n_features": int(len(varying)),
                "n_train_participants": int(len(train)),
                "n_test_participants": int(len(test)),
                "skipped": False,
            }
        )
        rows.append(bundle)
    if not rows:
        return pd.DataFrame([{"analysis": "context_control_exposure", "skipped": True, "skip_reason": "no candidate context columns present"}])
    return pd.DataFrame(rows)


def build_label_construction_audit_table(
    metadata: pd.DataFrame,
    external_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize the label columns actually available in processed artifacts."""
    frames: list[tuple[str, pd.DataFrame]] = []
    if metadata is not None and not metadata.empty:
        frames.append(("coswara_processed_metadata", metadata))
    if external_metadata is not None and not external_metadata.empty:
        frames.append(("external_processed_metadata", external_metadata))
    rows: list[dict[str, object]] = []
    candidate_columns = [
        "label_binary",
        "label_raw",
        "label_group",
        "covid_status",
        "status",
        "test_status",
        "label",
    ]
    for dataset_name, frame in frames:
        dataset_column = frame["dataset"].astype(str) if "dataset" in frame.columns else pd.Series([dataset_name] * len(frame), index=frame.index)
        for dataset, dataset_frame in frame.groupby(dataset_column, dropna=False):
            row: dict[str, object] = {
                "dataset": str(dataset),
                "source_table": dataset_name,
                "n_rows": int(len(dataset_frame)),
                "n_participants": int(dataset_frame["participant_id"].nunique()) if "participant_id" in dataset_frame.columns else int(len(dataset_frame)),
                "available_label_columns": ",".join([col for col in candidate_columns if col in dataset_frame.columns]),
                "interpretation": "processed analytic labels; verify raw source-definition text in manuscript methods",
            }
            if "label_binary" in dataset_frame.columns:
                labels = dataset_frame["label_binary"].astype(str)
                row["n_positive"] = int(labels.eq("positive").sum())
                row["n_negative"] = int(labels.eq("negative").sum())
                row["n_unknown_or_other"] = int((~labels.isin(["positive", "negative"])).sum())
                known = labels.isin(["positive", "negative"])
                row["positive_prevalence_known_labels"] = float(labels[known].eq("positive").mean()) if bool(known.any()) else float("nan")
            for col in [c for c in candidate_columns if c in dataset_frame.columns and c != "label_binary"]:
                counts = dataset_frame[col].astype(str).value_counts(dropna=False).head(10)
                row[f"{col}_top_values"] = ";".join(f"{key}:{int(value)}" for key, value in counts.items())
            rows.append(row)
    if not rows:
        return pd.DataFrame([{"source_table": "none", "skipped": True, "skip_reason": "no metadata tables available"}])
    return pd.DataFrame(rows)


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def paired_delong_auc_comparison(
    y_true: np.ndarray | pd.Series,
    left_probability: np.ndarray | pd.Series,
    right_probability: np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> dict[str, float | int | bool | str]:
    """Paired DeLong comparison with AUROC delta and normal-approximation CI."""
    y = np.asarray(y_true).astype(int)
    left = np.asarray(left_probability, dtype=float)
    right = np.asarray(right_probability, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    y = y[mask]
    left = left[mask]
    right = right[mask]
    if y.size < 4 or len(np.unique(y)) < 2:
        return {"skipped": True, "skip_reason": "insufficient paired two-class rows", "n_paired": int(y.size)}
    order = np.argsort(-y)
    label_1_count = int(np.sum(y == 1))
    predictions = np.vstack([left, right])[:, order]
    aucs, covariance = _fast_delong(predictions, label_1_count)
    contrast = np.array([1.0, -1.0])
    delta = float(aucs[0] - aucs[1])
    variance = float(contrast @ covariance @ contrast.T)
    se = float(np.sqrt(max(variance, 0.0)))
    z = delta / se if se > 0 else float("inf") if delta != 0 else 0.0
    p_value = float(2.0 * (1.0 - NormalDist().cdf(abs(z)))) if np.isfinite(z) else 0.0
    z_alpha = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return {
        "left_auc": float(aucs[0]),
        "right_auc": float(aucs[1]),
        "delta": delta,
        "standard_error": se,
        "delta_ci_low": float(delta - z_alpha * se),
        "delta_ci_high": float(delta + z_alpha * se),
        "p_value": max(0.0, min(1.0, p_value)),
        "confidence": float(confidence),
        "n_paired": int(y.size),
        "n_positive": int(label_1_count),
        "n_negative": int(y.size - label_1_count),
        "skipped": False,
    }


def build_paired_delong_table(
    predictions: pd.DataFrame,
    left_selector: dict[str, object],
    right_selector: dict[str, object],
    comparison_id: str,
    paired_on: str = "participant_id",
) -> pd.DataFrame:
    frame = _valid_predictions(predictions)
    if paired_on not in frame.columns:
        return pd.DataFrame([{"comparison_id": comparison_id, "skipped": True, "skip_reason": f"{paired_on} missing"}])

    def _filter(selector: dict[str, object]) -> pd.DataFrame:
        out = frame.copy()
        for col, value in selector.items():
            if col in out.columns and value is not None and not pd.isna(value):
                out = out[out[col].astype(str).eq(str(value))]
        return (
            out.groupby(paired_on, as_index=False)
            .agg(label_binary=("label_binary", "first"), probability=("probability", "mean"))
        )

    left = _filter(left_selector)
    right = _filter(right_selector)
    merged = left.merge(right, on=paired_on, suffixes=("_left", "_right"), how="inner")
    merged = merged[merged["label_binary_left"].astype(str).eq(merged["label_binary_right"].astype(str))].copy()
    if merged.empty:
        return pd.DataFrame([{"comparison_id": comparison_id, "skipped": True, "skip_reason": "no paired rows"}])
    result = paired_delong_auc_comparison(
        labels_to_binary(merged["label_binary_left"]),
        merged["probability_left"].astype(float).to_numpy(),
        merged["probability_right"].astype(float).to_numpy(),
    )
    result.update(
        {
            "comparison_id": comparison_id,
            "left_selector": str(left_selector),
            "right_selector": str(right_selector),
            "paired_on": paired_on,
        }
    )
    return pd.DataFrame([result])


def write_label_construction_note(output: Path, dataset_notes: pd.DataFrame | None = None) -> str:
    rows = []
    if dataset_notes is not None and not dataset_notes.empty:
        for _, row in dataset_notes.iterrows():
            if bool(row.get("skipped", False)):
                continue
            dataset = row.get("dataset", "dataset")
            available = row.get("available_label_columns", "")
            n_pos = row.get("n_positive", "NA")
            n_neg = row.get("n_negative", "NA")
            n_unknown = row.get("n_unknown_or_other", "NA")
            rows.append(
                f"- {dataset}: processed table `{row.get('source_table', 'metadata')}` contains label columns "
                f"`{available}`; analytic `label_binary` counts are positive={n_pos}, negative={n_neg}, "
                f"unknown/other={n_unknown}."
            )
    if not rows:
        rows = [
            "- Coswara: analytic labels are taken from the processed project metadata `label_binary` column after the repository's positive/negative mapping.",
            "- COUGHVID: analytic labels are taken from the processed external metadata `label_binary` column used for transfer evaluation.",
            "- These labels are not assumed to be clinically identical across datasets; cross-dataset transfer is therefore interpreted as an external stress test affected by label-construction and collection-protocol differences.",
        ]
    else:
        rows = [
            "- Coswara: analytic labels are taken from the processed project metadata `label_binary` column after the repository's positive/negative mapping.",
            "- COUGHVID: analytic labels are taken from the processed external metadata `label_binary` column used for transfer evaluation.",
            "- These labels are not assumed to be clinically identical across datasets; cross-dataset transfer is therefore interpreted as an external stress test affected by label-construction and collection-protocol differences.",
        ]
    text = "\n".join(
        [
            "# Label Construction Audit",
            "",
            "This note records the label source used by the project artifacts and separates label-definition mismatch from acoustic/domain shift.",
            "",
            *rows,
            "",
            "Manuscript implication: external-transfer failure should be described as a real-world dataset-transfer failure, not as proof that the same clinical label has identical annotation semantics across corpora.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def write_multiplicity_scope_note(output: Path) -> str:
    text = "\n".join(
        [
            "# Multiplicity and Analysis Scope",
            "",
            "Primary confirmatory endpoint: the validation ladder comparing existing participant split, time-stratified participant split, early-to-late temporal validation, and COUGHVID external transfer.",
            "",
            "Exploratory reviewer extensions: subgroup equity, decision-curve, recalibration-only, quality-label/month crosstabs, duration-shortcut, nested metadata+audio, support-overlap, feature-selection stability, and context-control analyses.",
            "",
            "P-values from exploratory analyses should be interpreted as diagnostic evidence rather than family-wise confirmatory claims. Effect sizes, confidence intervals, and consistency across analyses are prioritized over isolated significance tests.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text
