from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.metadata_confounding import build_audit_feature_frame
from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary
from covid_rars.reviewer_extension_checks import paired_delong_auc_comparison


AUDIO_SOURCE_COLUMNS = [
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
]


@dataclass(frozen=True)
class _AlignedSplit:
    validation: pd.DataFrame
    test: pd.DataFrame


def _safe_numeric(series: pd.Series, fill_value: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in frame.columns]


def _source_key(row: pd.Series | dict[str, object], columns: list[str] | None = None) -> str:
    cols = columns or AUDIO_SOURCE_COLUMNS
    parts: list[str] = []
    for col in cols:
        value = row.get(col, "") if isinstance(row, dict) else row[col] if col in row.index else ""
        if pd.isna(value):
            value = ""
        parts.append(f"{col}={value}")
    return "||".join(parts)


def _valid_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = {"participant_id", "label_binary", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"prediction table missing required columns: {sorted(missing)}")
    work = frame[frame["label_binary"].isin(["positive", "negative"])].copy()
    work["probability"] = _safe_numeric(work["probability"])
    work = work[np.isfinite(work["probability"])].copy()
    work["probability"] = work["probability"].clip(0.0, 1.0)
    if "split" not in work.columns and "metric_split" in work.columns:
        work["split"] = work["metric_split"].astype(str)
    if "metric_split" not in work.columns and "split" in work.columns:
        work["metric_split"] = work["split"].astype(str)
    return work


def _participant_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "label_binary", "split"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"metadata missing required columns: {sorted(missing)}")
    frame = metadata[
        metadata["label_binary"].isin(["positive", "negative"])
        & metadata["split"].isin(["train", "validation", "test"])
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    sort_cols = [col for col in ["participant_id", "split", "recording_id"] if col in frame.columns]
    return frame.sort_values(sort_cols).groupby("participant_id", as_index=False, dropna=False).first()


def _fit_metadata_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    random_state: int,
) -> tuple[Pipeline, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_x_raw, _ = build_audit_feature_frame(train, feature_set=feature_set)
    validation_x_raw, _ = build_audit_feature_frame(validation, feature_set=feature_set)
    test_x_raw, _ = build_audit_feature_frame(test, feature_set=feature_set)
    columns = sorted(set(train_x_raw.columns) | set(validation_x_raw.columns) | set(test_x_raw.columns))
    train_x = train_x_raw.reindex(columns=columns, fill_value=0.0)
    validation_x = validation_x_raw.reindex(columns=columns, fill_value=0.0)
    test_x = test_x_raw.reindex(columns=columns, fill_value=0.0)
    varying = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
    if not varying:
        raise ValueError(f"metadata feature set {feature_set!r} has no train-varying columns")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state)),
        ]
    )
    model.fit(train_x[varying], labels_to_binary(train["label_binary"]))
    return model, varying, train_x, validation_x, test_x


def build_metadata_probability_table(
    metadata: pd.DataFrame,
    feature_set: str = "symptoms_only",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a metadata-only model and emit validation/test participant probabilities."""
    participants = _participant_rows(metadata)
    if participants.empty:
        return pd.DataFrame()
    train = participants[participants["split"].eq("train")].copy()
    validation = participants[participants["split"].eq("validation")].copy()
    test = participants[participants["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty or train["label_binary"].nunique() < 2:
        return pd.DataFrame()
    model, columns, _, validation_x, test_x = _fit_metadata_model(
        train,
        validation,
        test,
        feature_set=feature_set,
        random_state=random_state,
    )

    frames: list[pd.DataFrame] = []
    for split_name, split_frame, split_x in [
        ("validation", validation, validation_x),
        ("test", test, test_x),
    ]:
        probability = model.predict_proba(split_x[columns])[:, 1]
        out_cols = [col for col in ["participant_id", "recording_id", "label_binary"] if col in split_frame.columns]
        out = split_frame[out_cols].copy()
        out["split"] = split_name
        out["metric_split"] = split_name
        out["probability"] = probability
        out["analysis_family"] = "incremental_metadata_model"
        out["model_name"] = "metadata_nested_logistic_regression"
        out["modality"] = "metadata"
        out["feature_strategy"] = feature_set
        out["metadata_feature_set"] = feature_set
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False)


def _participant_audio_predictions(audio_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _valid_labeled_rows(audio_predictions)
    if frame.empty:
        return pd.DataFrame()
    if "split" not in frame.columns:
        raise KeyError("audio predictions need split or metric_split")
    source_cols = _available(AUDIO_SOURCE_COLUMNS, frame)
    group_cols = [*source_cols, "participant_id", "split"]
    aggregated = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            label_binary=("label_binary", "first"),
            probability=("probability", "mean"),
            n_prediction_rows=("probability", "size"),
        )
        .reset_index()
    )
    aggregated["audio_source_key"] = aggregated.apply(lambda row: _source_key(row, source_cols), axis=1)
    return aggregated


def _split_metadata_predictions(metadata_predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = _valid_labeled_rows(metadata_predictions)
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame["split"].astype(str).eq(str(split))].copy()
    if frame.empty:
        return frame
    return (
        frame.groupby("participant_id", as_index=False, dropna=False)
        .agg(label_binary=("label_binary", "first"), probability=("probability", "mean"))
        .rename(columns={"probability": "probability_metadata"})
    )


def _aligned_for_candidate(
    participant_audio: pd.DataFrame,
    metadata_predictions: pd.DataFrame,
    candidate: pd.Series,
    validation_split: str,
    test_split: str,
) -> _AlignedSplit:
    source_cols = _available(AUDIO_SOURCE_COLUMNS, participant_audio)
    candidate_key = str(candidate["audio_source_key"])
    audio = participant_audio[participant_audio["audio_source_key"].astype(str).eq(candidate_key)].copy()
    val_audio = audio[audio["split"].astype(str).eq(validation_split)].copy()
    test_audio = audio[audio["split"].astype(str).eq(test_split)].copy()
    val_meta = _split_metadata_predictions(metadata_predictions, validation_split)
    test_meta = _split_metadata_predictions(metadata_predictions, test_split)

    def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        merged = left.merge(right, on="participant_id", how="inner", suffixes=("_audio", "_metadata"))
        if merged.empty:
            return merged
        merged = merged[merged["label_binary_audio"].astype(str).eq(merged["label_binary_metadata"].astype(str))].copy()
        merged = merged.rename(
            columns={
                "label_binary_audio": "label_binary",
                "probability": "probability_audio",
            }
        )
        for col in source_cols:
            if col not in merged.columns and col in candidate.index:
                merged[col] = candidate[col]
        return merged

    return _AlignedSplit(validation=_merge(val_audio, val_meta), test=_merge(test_audio, test_meta))


def build_audio_source_candidates(
    audio_predictions: pd.DataFrame,
    metadata_predictions: pd.DataFrame,
    top_k: int = 5,
    validation_split: str = "validation",
    test_split: str = "test",
) -> pd.DataFrame:
    """Rank audio prediction sources by aligned validation AUROC.

    Sources without both aligned validation and aligned test participants are
    excluded because they cannot support an honest nested comparison.
    """
    participant_audio = _participant_audio_predictions(audio_predictions)
    if participant_audio.empty:
        return pd.DataFrame()
    source_cols = _available(AUDIO_SOURCE_COLUMNS, participant_audio)
    meta_val = _split_metadata_predictions(metadata_predictions, validation_split)
    meta_test = _split_metadata_predictions(metadata_predictions, test_split)
    rows: list[dict[str, object]] = []
    for key, group in participant_audio.groupby("audio_source_key", dropna=False):
        first = group.iloc[0]
        val = group[group["split"].astype(str).eq(validation_split)].merge(meta_val, on="participant_id", how="inner")
        test = group[group["split"].astype(str).eq(test_split)].merge(meta_test, on="participant_id", how="inner")
        val = val[val["label_binary_x"].astype(str).eq(val["label_binary_y"].astype(str))].copy()
        test = test[test["label_binary_x"].astype(str).eq(test["label_binary_y"].astype(str))].copy()
        if val.empty or test.empty or val["label_binary_x"].nunique() < 2 or test["label_binary_x"].nunique() < 2:
            continue
        y_val = labels_to_binary(val["label_binary_x"])
        prob_val = val["probability"].astype(float).to_numpy()
        row: dict[str, object] = {col: first[col] for col in source_cols}
        row.update(
            {
                "audio_source_key": str(key),
                "validation_auroc": float(roc_auc_score(y_val, prob_val)),
                "validation_auprc": float(average_precision_score(y_val, prob_val)),
                "n_validation_aligned": int(len(val)),
                "n_test_aligned": int(len(test)),
                "n_validation_positive": int(np.sum(y_val == 1)),
                "n_test_positive": int(test["label_binary_x"].eq("positive").sum()),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["validation_auroc", "validation_auprc"], ascending=False).reset_index(drop=True)
    out["candidate_rank"] = np.arange(1, len(out) + 1, dtype=int)
    if top_k > 0:
        out = out.head(int(top_k)).reset_index(drop=True)
    return out


def _model_probability(train_x: pd.DataFrame, y_train: np.ndarray, eval_x: pd.DataFrame, random_state: int) -> np.ndarray:
    if train_x.shape[1] == 1:
        train_values = train_x.iloc[:, 0].to_numpy(dtype=float)
        if len(np.unique(train_values)) > 1:
            corr = np.corrcoef(train_values, y_train)[0, 1]
            if np.isfinite(corr) and corr < 0:
                return 1.0 - eval_x.iloc[:, 0].to_numpy(dtype=float)
        return eval_x.iloc[:, 0].to_numpy(dtype=float)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(train_x, y_train)
    return model.predict_proba(eval_x)[:, 1]


def _bootstrap_delta_ci(
    y_true: np.ndarray,
    candidate_prob: np.ndarray,
    baseline_prob: np.ndarray,
    metric: str,
    n_bootstraps: int,
    random_state: int,
) -> tuple[float, float]:
    point = _metric_value(y_true, candidate_prob, metric) - _metric_value(y_true, baseline_prob, metric)
    if n_bootstraps <= 0 or y_true.size < 4 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    deltas: list[float] = [float(point)]
    indices = np.arange(y_true.size)
    for _ in range(int(n_bootstraps)):
        sample = rng.choice(indices, size=indices.size, replace=True)
        if len(np.unique(y_true[sample])) < 2:
            continue
        delta = _metric_value(y_true[sample], candidate_prob[sample], metric) - _metric_value(
            y_true[sample],
            baseline_prob[sample],
            metric,
        )
        if np.isfinite(delta):
            deltas.append(float(delta))
    if len(deltas) < 2:
        return float("nan"), float("nan")
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(min(low, point)), float(max(high, point))


def _metric_value(y_true: np.ndarray, probability: np.ndarray, metric: str) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(y_true, probability))
    if metric == "auprc":
        return float(average_precision_score(y_true, probability))
    raise ValueError(f"Unsupported metric for bootstrap delta: {metric}")


def _nested_rows_for_candidate(
    aligned: _AlignedSplit,
    candidate: pd.Series,
    metadata_feature_set: str,
    n_bootstraps: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = aligned.validation.copy()
    test = aligned.test.copy()
    if val.empty or test.empty or val["label_binary"].nunique() < 2 or test["label_binary"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    y_val = labels_to_binary(val["label_binary"])
    y_test = labels_to_binary(test["label_binary"])
    train_features = {
        "metadata_only": val[["probability_metadata"]].astype(float),
        "audio_only": val[["probability_audio"]].astype(float),
        "metadata_plus_audio": val[["probability_metadata", "probability_audio"]].astype(float),
    }
    test_features = {
        "metadata_only": test[["probability_metadata"]].astype(float),
        "audio_only": test[["probability_audio"]].astype(float),
        "metadata_plus_audio": test[["probability_metadata", "probability_audio"]].astype(float),
    }
    val_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for model_name in ["metadata_only", "audio_only", "metadata_plus_audio"]:
        val_probabilities[model_name] = _model_probability(
            train_features[model_name],
            y_val,
            train_features[model_name],
            random_state=random_state,
        )
        test_probabilities[model_name] = _model_probability(
            train_features[model_name],
            y_val,
            test_features[model_name],
            random_state=random_state,
        )

    metadata_prob = test_probabilities["metadata_only"]
    metadata_auroc = _metric_value(y_test, metadata_prob, "auroc")
    metadata_auprc = _metric_value(y_test, metadata_prob, "auprc")
    source_cols = _available(AUDIO_SOURCE_COLUMNS, test)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for nested_model, prob in test_probabilities.items():
        threshold = best_threshold_by_balanced_accuracy(y_val, val_probabilities[nested_model])
        bundle = binary_metric_bundle(y_test, prob, threshold=threshold)
        auroc_delta = float(bundle["auroc"] - metadata_auroc) if np.isfinite(metadata_auroc) else float("nan")
        auprc_delta = float(bundle["auprc"] - metadata_auprc) if np.isfinite(metadata_auprc) else float("nan")
        if nested_model == "metadata_only":
            ci = {
                "delta_auroc_ci_low_vs_metadata": 0.0,
                "delta_auroc_ci_high_vs_metadata": 0.0,
                "delta_auprc_ci_low_vs_metadata": 0.0,
                "delta_auprc_ci_high_vs_metadata": 0.0,
                "paired_delong_delta_vs_metadata": 0.0,
                "paired_delong_delta_ci_low_vs_metadata": 0.0,
                "paired_delong_delta_ci_high_vs_metadata": 0.0,
                "paired_delong_p_value_vs_metadata": float("nan"),
            }
        else:
            auroc_low, auroc_high = _bootstrap_delta_ci(
                y_test,
                prob,
                metadata_prob,
                metric="auroc",
                n_bootstraps=n_bootstraps,
                random_state=random_state,
            )
            auprc_low, auprc_high = _bootstrap_delta_ci(
                y_test,
                prob,
                metadata_prob,
                metric="auprc",
                n_bootstraps=n_bootstraps,
                random_state=random_state + 17,
            )
            delong = paired_delong_auc_comparison(y_test, prob, metadata_prob)
            ci = {
                "delta_auroc_ci_low_vs_metadata": auroc_low,
                "delta_auroc_ci_high_vs_metadata": auroc_high,
                "delta_auprc_ci_low_vs_metadata": auprc_low,
                "delta_auprc_ci_high_vs_metadata": auprc_high,
                "paired_delong_delta_vs_metadata": delong.get("delta", float("nan")),
                "paired_delong_delta_ci_low_vs_metadata": delong.get("delta_ci_low", float("nan")),
                "paired_delong_delta_ci_high_vs_metadata": delong.get("delta_ci_high", float("nan")),
                "paired_delong_p_value_vs_metadata": delong.get("p_value", float("nan")),
            }
        row = {
            **bundle,
            "analysis": "incremental_audio_metadata_value",
            "nested_model": nested_model,
            "metadata_feature_set": metadata_feature_set,
            "audio_source_key": candidate["audio_source_key"],
            "metric_split": "test",
            "threshold_source": "validation_balanced_accuracy",
            "n_validation_aligned": int(len(val)),
            "n_test_aligned": int(len(test)),
            "n_bootstraps": int(n_bootstraps),
            "delta_auroc_vs_metadata": auroc_delta,
            "delta_auprc_vs_metadata": auprc_delta,
            "skipped": False,
            **ci,
        }
        for col in source_cols:
            row[col] = candidate[col] if col in candidate.index else test[col].iloc[0]
        metric_rows.append(row)
        pred = test[["participant_id", "label_binary"]].copy()
        pred["probability"] = prob
        pred["split"] = "test"
        pred["metric_split"] = "test"
        pred["analysis"] = "incremental_audio_metadata_value"
        pred["nested_model"] = nested_model
        pred["metadata_feature_set"] = metadata_feature_set
        pred["audio_source_key"] = candidate["audio_source_key"]
        for col in source_cols:
            pred[col] = candidate[col] if col in candidate.index else test[col].iloc[0]
        prediction_frames.append(pred)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True, sort=False)


def build_incremental_audio_metadata_value(
    metadata: pd.DataFrame,
    audio_predictions: pd.DataFrame,
    metadata_feature_sets: list[str] | None = None,
    top_k_audio_sources: int = 5,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare metadata-only, audio-only, and metadata+audio on aligned participants."""
    feature_sets = metadata_feature_sets or ["symptoms_only", "full_safe_metadata"]
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []
    participant_audio = _participant_audio_predictions(audio_predictions)
    if participant_audio.empty:
        skipped = pd.DataFrame(
            [
                {
                    "analysis": "incremental_audio_metadata_value",
                    "skipped": True,
                    "skip_reason": "no usable audio predictions",
                }
            ]
        )
        return skipped, pd.DataFrame(), pd.DataFrame()
    for feature_set in feature_sets:
        try:
            metadata_predictions = build_metadata_probability_table(metadata, feature_set=feature_set, random_state=random_state)
        except (KeyError, ValueError) as exc:
            all_metrics.append(
                pd.DataFrame(
                    [
                        {
                            "analysis": "incremental_audio_metadata_value",
                            "metadata_feature_set": feature_set,
                            "skipped": True,
                            "skip_reason": str(exc),
                        }
                    ]
                )
            )
            continue
        candidates = build_audio_source_candidates(
            participant_audio,
            metadata_predictions,
            top_k=top_k_audio_sources,
        )
        if candidates.empty:
            all_metrics.append(
                pd.DataFrame(
                    [
                        {
                            "analysis": "incremental_audio_metadata_value",
                            "metadata_feature_set": feature_set,
                            "skipped": True,
                            "skip_reason": "no audio source has aligned validation and test predictions",
                        }
                    ]
                )
            )
            continue
        candidates["metadata_feature_set"] = feature_set
        all_candidates.append(candidates)
        for _, candidate in candidates.iterrows():
            aligned = _aligned_for_candidate(participant_audio, metadata_predictions, candidate, "validation", "test")
            metrics, predictions = _nested_rows_for_candidate(
                aligned,
                candidate,
                metadata_feature_set=feature_set,
                n_bootstraps=n_bootstraps,
                random_state=random_state,
            )
            if metrics.empty:
                continue
            all_metrics.append(metrics)
            all_predictions.append(predictions)
    metrics_out = pd.concat(all_metrics, ignore_index=True, sort=False) if all_metrics else pd.DataFrame()
    predictions_out = pd.concat(all_predictions, ignore_index=True, sort=False) if all_predictions else pd.DataFrame()
    candidates_out = pd.concat(all_candidates, ignore_index=True, sort=False) if all_candidates else pd.DataFrame()
    return metrics_out, predictions_out, candidates_out


def _best_metric_row(frame: pd.DataFrame, filters: dict[str, object]) -> pd.Series | None:
    if frame.empty:
        return None
    work = frame.copy()
    if "skipped" in work.columns:
        work = work[~work["skipped"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    for col, value in filters.items():
        if col in work.columns and value is not None:
            work = work[work[col].astype(str).eq(str(value))].copy()
    if work.empty or "auroc" not in work.columns:
        return None
    work["auroc"] = _safe_numeric(work["auroc"])
    if "auprc" in work.columns:
        work["auprc"] = _safe_numeric(work["auprc"])
        return work.sort_values(["auroc", "auprc"], ascending=False).iloc[0]
    return work.sort_values("auroc", ascending=False).iloc[0]


def _summary_row(
    family_model: str,
    model_family: str,
    internal: pd.Series | None,
    external: pd.Series | None,
) -> dict[str, object]:
    if internal is None or external is None:
        return {
            "family_model": family_model,
            "model_family": model_family,
            "skipped": True,
            "skip_reason": "internal or external row unavailable",
        }
    internal_auroc = float(internal.get("auroc", np.nan))
    external_auroc = float(external.get("auroc", np.nan))
    internal_auprc = float(internal.get("auprc", np.nan))
    external_auprc = float(external.get("auprc", np.nan))
    return {
        "family_model": family_model,
        "model_family": model_family,
        "internal_evaluation_protocol": internal.get("evaluation_protocol", ""),
        "external_evaluation_protocol": external.get("evaluation_protocol", ""),
        "internal_metric_split": internal.get("metric_split", ""),
        "external_metric_split": external.get("metric_split", ""),
        "internal_auroc": internal_auroc,
        "external_auroc": external_auroc,
        "delta_auroc_internal_minus_external": round(internal_auroc - external_auroc, 12),
        "internal_auprc": internal_auprc,
        "external_auprc": external_auprc,
        "delta_auprc_internal_minus_external": round(internal_auprc - external_auprc, 12),
        "internal_n_samples": internal.get("n_samples", np.nan),
        "external_n_samples": external.get("n_samples", np.nan),
        "skipped": False,
    }


def build_external_model_family_transfer_summary(
    compare_internal_metrics: pd.DataFrame,
    compare_external_metrics: pd.DataFrame,
    wavlm_metrics: pd.DataFrame | None = None,
    cnn_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize internal-to-COUGHVID transfer across handcrafted and deep families."""
    rows: list[dict[str, object]] = []
    if compare_external_metrics is not None and not compare_external_metrics.empty:
        compare_external = compare_external_metrics.copy()
        if "metric_split" in compare_external.columns:
            compare_external = compare_external[compare_external["metric_split"].astype(str).eq("external_test")].copy()
        for model_name in sorted(compare_external.get("model_name", pd.Series(dtype=object)).dropna().astype(str).unique()):
            external = _best_metric_row(compare_external, {"model_name": model_name, "modality": "cough"})
            internal = _best_metric_row(
                compare_internal_metrics,
                {
                    "model_name": model_name,
                    "modality": "cough",
                    "metric_split": "test",
                },
            )
            rows.append(_summary_row(f"compare_is10_{model_name}", "compare_is10_handcrafted", internal, external))
    for metrics, family_model, model_family in [
        (wavlm_metrics, "wavlm_base_plus_pooled_cough", "wavlm_transformer"),
        (cnn_metrics, "cnn_bigru_cough", "cnn_bigru_spectrogram"),
    ]:
        if metrics is None or metrics.empty:
            continue
        model_names = metrics["model_name"].dropna().astype(str).unique().tolist() if "model_name" in metrics.columns else []
        selected_name = family_model if family_model in model_names else model_names[0] if model_names else None
        filters = {"model_name": selected_name} if selected_name is not None else {}
        external = _best_metric_row(metrics, {**filters, "metric_split": "external_test"})
        internal = _best_metric_row(metrics, {**filters, "metric_split": "test"})
        if internal is None:
            work = metrics.copy()
            if "metric_split" in work.columns:
                work = work[~work["metric_split"].astype(str).eq("external_test")].copy()
            internal = _best_metric_row(work, filters)
        rows.append(_summary_row(selected_name or family_model, model_family, internal, external))
    return pd.DataFrame(rows)
