from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from covid_rars.features import feature_columns
from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary
from covid_rars.strong_baseline import (
    DEFAULT_MODEL_NAMES,
    DEFAULT_MODALITIES,
    _make_model,
    _participant_average,
    _predict_probability,
    _selected_predictions,
    run_global_prediction_stacker,
    run_strong_fusion,
    train_feature_level_fusion_models,
    train_strong_modality_models,
)
from covid_rars.temporal_holdout import (
    REQUIRED_SPLITS,
    _apply_split_to_features,
    _existing_split_summary,
    _modality_coverage,
    build_temporal_split_assignments,
    build_time_stratified_split_assignments,
)


COMPARE_EXISTING_PROTOCOL = "compare_is10_existing_participant_split"
COMPARE_TIME_STRATIFIED_PROTOCOL = "compare_is10_time_stratified_participant_split"
COMPARE_TEMPORAL_PROTOCOL = "compare_is10_temporal_early_to_late"
COMPARE_EXTERNAL_PROTOCOL = "coswara_to_coughvid_compare_is10_external"


@dataclass(frozen=True)
class CompareIS10FinalValidationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    split_summary: pd.DataFrame
    modality_coverage: pd.DataFrame
    final_summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: set[str], frame_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise KeyError(f"{frame_name} missing required columns: {sorted(missing)}")


def _with_feature_strategy(frame: pd.DataFrame, feature_strategy: str, selected_feature_k: int | float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["feature_strategy"] = feature_strategy
    out["selected_feature_k"] = float(selected_feature_k)
    return out


def _relabel_protocol(frame: pd.DataFrame, protocol: str, feature_strategy: str, selected_feature_k: int | float) -> pd.DataFrame:
    out = _with_feature_strategy(frame, feature_strategy=feature_strategy, selected_feature_k=selected_feature_k)
    if not out.empty:
        out["evaluation_protocol"] = protocol
    return out


def _has_skipped_column(frame: pd.DataFrame) -> pd.Series:
    if "skipped" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["skipped"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _run_strong_protocol(
    features: pd.DataFrame,
    protocol: str,
    feature_strategy: str,
    selected_feature_k: int | float,
    modalities: Iterable[str],
    model_names: Iterable[str],
    random_state: int,
    optuna_trials: int,
    ensemble_top_k: int,
    enable_feature_level_fusion: bool,
    global_stack_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modality_metrics, modality_predictions, selection, _ = train_strong_modality_models(
        features,
        modalities=modalities,
        model_names=model_names,
        random_state=random_state,
        optuna_trials=optuna_trials,
        ensemble_top_k=ensemble_top_k,
    )
    selected = _selected_predictions(modality_predictions, selection)
    fusion_metrics, fusion_predictions = run_strong_fusion(selected, selection, modalities=modalities)

    metric_frames = [modality_metrics, fusion_metrics]
    prediction_frames = [modality_predictions, fusion_predictions]

    if enable_feature_level_fusion:
        early_metrics, early_predictions = train_feature_level_fusion_models(
            features,
            modalities=modalities,
            model_names=model_names,
            random_state=random_state,
            optuna_trials=optuna_trials,
        )
        metric_frames.append(early_metrics)
        prediction_frames.append(early_predictions)

    metrics = pd.concat([frame for frame in metric_frames if not frame.empty], ignore_index=True, sort=False)
    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True, sort=False)

    if global_stack_top_k >= 2 and not metrics.empty and not predictions.empty:
        stack_metrics, stack_predictions = run_global_prediction_stacker(
            metrics,
            predictions,
            top_k=global_stack_top_k,
        )
        if not stack_metrics.empty:
            metrics = pd.concat([metrics, stack_metrics], ignore_index=True, sort=False)
        if not stack_predictions.empty:
            predictions = pd.concat([predictions, stack_predictions], ignore_index=True, sort=False)

    return (
        _relabel_protocol(metrics, protocol, feature_strategy, selected_feature_k),
        _relabel_protocol(predictions, protocol, feature_strategy, selected_feature_k),
    )


def _existing_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out = out[out["label_binary"].isin(["positive", "negative"])].copy()
    return out[out["split"].isin(REQUIRED_SPLITS)].copy()


def _rename_split_summary(summary: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = summary.copy()
    if not out.empty:
        out["evaluation_protocol"] = protocol
    return out


def _build_final_summary(metrics: pd.DataFrame, feature_strategy: str, selected_feature_k: int | float) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    work = metrics.copy()
    work = work[~_has_skipped_column(work)].copy()
    if work.empty:
        return pd.DataFrame()
    work["auroc"] = pd.to_numeric(work.get("auroc"), errors="coerce")
    work["auprc"] = pd.to_numeric(work.get("auprc"), errors="coerce")

    candidate_columns = [
        column
        for column in (
            "analysis_family",
            "model_name",
            "modality",
            "submodality",
            "modality_combination",
            "fusion_method",
            "ensemble_members",
        )
        if column in work.columns
    ]

    def candidate_mask(frame: pd.DataFrame, candidate: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        for column in candidate_columns:
            value = candidate[column]
            mask &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
        return mask

    rows: list[pd.Series] = []
    for protocol, group in work.groupby("evaluation_protocol", dropna=False):
        validation = group[group["metric_split"].astype(str).eq("validation")].copy()
        test = group[group["metric_split"].astype(str).eq("test")].copy()
        if {"analysis_family", "fusion_method"}.issubset(validation.columns):
            in_sample_stack = validation["analysis_family"].astype(str).eq("strong_multimodal_fusion") & validation[
                "fusion_method"
            ].astype(str).eq("stacked_logistic_validation")
            validation = validation[~in_sample_stack].copy()
        if validation.empty or test.empty:
            continue
        ranked = validation.sort_values(
            ["auroc", "auprc", "analysis_family", "model_name"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        selected_validation = ranked.iloc[0]
        selected_test = test[candidate_mask(test, selected_validation)]
        if len(selected_test) != 1:
            raise ValueError(
                f"Expected exactly one test row for the validation-selected candidate in {protocol!r}; "
                f"found {len(selected_test)}"
            )
        rows.append(selected_test.iloc[0])
    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows).reset_index(drop=True)
    existing = summary[summary["evaluation_protocol"].astype(str).eq(COMPARE_EXISTING_PROTOCOL)]
    existing_auroc = float(pd.to_numeric(existing["auroc"], errors="coerce").max()) if not existing.empty else float("nan")
    summary["delta_auroc_from_existing"] = pd.to_numeric(summary["auroc"], errors="coerce") - existing_auroc
    summary["feature_strategy"] = feature_strategy
    summary["selected_feature_k"] = float(selected_feature_k)
    sort_key = {
        COMPARE_EXISTING_PROTOCOL: 0,
        COMPARE_TIME_STRATIFIED_PROTOCOL: 1,
        COMPARE_TEMPORAL_PROTOCOL: 2,
    }
    summary["_sort"] = summary["evaluation_protocol"].map(sort_key).fillna(99)
    summary = summary.sort_values(["_sort", "auroc"], ascending=[True, False]).drop(columns=["_sort"]).reset_index(drop=True)
    return summary


def run_compare_is10_final_validation(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_strategy: str,
    selected_feature_k: int | float,
    modalities: Iterable[str] = DEFAULT_MODALITIES,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    random_state: int = 42,
    optuna_trials: int = 0,
    ensemble_top_k: int = 5,
    enable_feature_level_fusion: bool = False,
    global_stack_top_k: int = 0,
) -> CompareIS10FinalValidationResult:
    _require_columns(features, {"recording_id", "participant_id", "modality", "label_binary", "split"}, "features")
    _require_columns(metadata, {"participant_id", "label_binary", "split", "recording_date"}, "metadata")

    protocol_specs: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    existing_features = _existing_features(features)
    protocol_specs.append(
        (
            COMPARE_EXISTING_PROTOCOL,
            existing_features,
            _rename_split_summary(_existing_split_summary(metadata), COMPARE_EXISTING_PROTOCOL),
        )
    )

    time_assignments, time_summary = build_time_stratified_split_assignments(metadata, random_state=random_state)
    protocol_specs.append(
        (
            COMPARE_TIME_STRATIFIED_PROTOCOL,
            _apply_split_to_features(features, time_assignments, "time_stratified_split"),
            _rename_split_summary(time_summary, COMPARE_TIME_STRATIFIED_PROTOCOL),
        )
    )

    temporal_assignments, temporal_summary = build_temporal_split_assignments(metadata)
    protocol_specs.append(
        (
            COMPARE_TEMPORAL_PROTOCOL,
            _apply_split_to_features(features, temporal_assignments, "temporal_split"),
            _rename_split_summary(temporal_summary, COMPARE_TEMPORAL_PROTOCOL),
        )
    )

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []

    for protocol, protocol_features, split_summary in protocol_specs:
        split_frames.append(split_summary)
        coverage_frames.append(_modality_coverage(protocol_features, protocol))
        metrics, predictions = _run_strong_protocol(
            protocol_features,
            protocol=protocol,
            feature_strategy=feature_strategy,
            selected_feature_k=selected_feature_k,
            modalities=modalities,
            model_names=model_names,
            random_state=random_state,
            optuna_trials=optuna_trials,
            ensemble_top_k=ensemble_top_k,
            enable_feature_level_fusion=enable_feature_level_fusion,
            global_stack_top_k=global_stack_top_k,
        )
        metric_frames.append(metrics)
        prediction_frames.append(predictions)

    metrics = pd.concat([frame for frame in metric_frames if not frame.empty], ignore_index=True, sort=False)
    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True, sort=False)
    split_summary = pd.concat([frame for frame in split_frames if not frame.empty], ignore_index=True, sort=False)
    modality_coverage = pd.concat([frame for frame in coverage_frames if not frame.empty], ignore_index=True, sort=False)
    final_summary = _build_final_summary(metrics, feature_strategy=feature_strategy, selected_feature_k=selected_feature_k)

    return CompareIS10FinalValidationResult(
        metrics=metrics,
        predictions=predictions,
        split_summary=split_summary,
        modality_coverage=modality_coverage,
        final_summary=final_summary,
    )


def _external_prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    modality: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["recording_id"].astype(str).to_numpy(),
            "participant_id": source["participant_id"].astype(str).to_numpy(),
            "dataset": source.get("dataset", pd.Series(["external"] * len(source), index=source.index)).astype(str).to_numpy(),
            "modality": modality,
            "submodality": source.get("submodality", pd.Series(["unknown"] * len(source), index=source.index)).astype(str).to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": "external_test",
            "model_name": model_name,
            "analysis_family": "compare_is10_external_transfer",
            "evaluation_protocol": COMPARE_EXTERNAL_PROTOCOL,
            "probability": probabilities,
        }
    )


def run_compare_is10_external_transfer(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    feature_strategy: str,
    selected_feature_k: int | float,
    model_names: Iterable[str],
    modality: str = "cough",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(source_features, {"recording_id", "participant_id", "modality", "label_binary", "split"}, "source_features")
    _require_columns(target_features, {"recording_id", "participant_id", "modality", "label_binary"}, "target_features")

    source = source_features[
        source_features["modality"].astype(str).eq(str(modality))
        & source_features["label_binary"].isin(["positive", "negative"])
        & source_features["split"].isin(REQUIRED_SPLITS)
    ].copy()
    target = target_features[
        target_features["modality"].astype(str).eq(str(modality))
        & target_features["label_binary"].isin(["positive", "negative"])
    ].copy()

    source_cols = feature_columns(source)
    target_cols = set(feature_columns(target))
    cols = [col for col in source_cols if col in target_cols]
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    train = source[source["split"].eq("train")].copy()
    validation = source[source["split"].eq("validation")].copy()
    if train.empty or validation.empty or target.empty or train["label_binary"].nunique() < 2 or not cols:
        return (
            pd.DataFrame(
                [
                    {
                        "evaluation_protocol": COMPARE_EXTERNAL_PROTOCOL,
                        "analysis_family": "compare_is10_external_transfer",
                        "model_name": "not_run",
                        "modality": modality,
                        "metric_split": "external_test",
                        "feature_strategy": feature_strategy,
                        "selected_feature_k": float(selected_feature_k),
                        "skipped": True,
                        "skip_reason": "missing train/validation/target rows, two train classes, or common numeric feature columns",
                    }
                ]
            ),
            pd.DataFrame(),
        )

    x_train = train[cols].fillna(0.0)
    y_train = labels_to_binary(train["label_binary"])
    x_validation = validation[cols].fillna(0.0)
    y_validation = labels_to_binary(validation["label_binary"])
    x_target = target[cols].fillna(0.0)

    for model_name in model_names:
        model_name = str(model_name)
        try:
            if model_name == "optuna_validation_search":
                raise ValueError("optuna_validation_search is not used in external transfer")
            model = _make_model(model_name, random_state=random_state)
            model.fit(x_train, y_train)
            validation_probability = _predict_probability(model, x_validation)
            threshold = best_threshold_by_balanced_accuracy(y_validation, validation_probability)
            target_probability = _predict_probability(model, x_target)
        except Exception as exc:
            metric_rows.append(
                {
                    "evaluation_protocol": COMPARE_EXTERNAL_PROTOCOL,
                    "analysis_family": "compare_is10_external_transfer",
                    "model_name": model_name,
                    "modality": modality,
                    "metric_split": "external_test",
                    "feature_strategy": feature_strategy,
                    "selected_feature_k": float(selected_feature_k),
                    "skipped": True,
                    "skip_reason": str(exc),
                }
            )
            continue

        predictions = _external_prediction_frame(target, target_probability, model_name=model_name, modality=modality)
        predictions = _with_feature_strategy(predictions, feature_strategy, selected_feature_k)
        participant_predictions = _participant_average(predictions)
        row = binary_metric_bundle(
            labels_to_binary(participant_predictions["label_binary"]),
            participant_predictions["probability"].astype(float).to_numpy(),
            threshold=threshold,
        )
        row.update(
            {
                "evaluation_protocol": COMPARE_EXTERNAL_PROTOCOL,
                "analysis_family": "compare_is10_external_transfer",
                "model_name": model_name,
                "modality": modality,
                "metric_split": "external_test",
                "threshold_source": "source_validation_balanced_accuracy",
                "feature_strategy": feature_strategy,
                "selected_feature_k": float(selected_feature_k),
                "n_features": float(len(cols)),
                "n_participants": float(participant_predictions["participant_id"].nunique()),
                "skipped": False,
            }
        )
        metric_rows.append(row)
        prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _write_svg_bars(rows: list[tuple[str, float]], output_path: Path, title: str, subtitle: str = "") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    row_height = 46
    top = 88
    height = top + max(1, len(rows)) * row_height + 50
    max_value = max([value for _, value in rows] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="28" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{escape(subtitle)}</text>'
        )
    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_height
        bar_width = int(620 * (value / max_value)) if max_value > 0 else 0
        parts.extend(
            [
                f'<text x="28" y="{y + 20}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{escape(label)}</text>',
                f'<rect x="300" y="{y}" width="620" height="24" fill="#e5e7eb" rx="3"/>',
                f'<rect x="300" y="{y}" width="{bar_width}" height="24" fill="#2563eb" rx="3"/>',
                f'<text x="930" y="{y + 18}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{value:.3f}</text>',
            ]
        )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_temporal_degradation_figure(summary: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        label = str(row.get("evaluation_protocol", "unknown")).replace("compare_is10_", "")
        rows.append((label, _numeric(row.get("auroc"))))
    _write_svg_bars(
        rows,
        output_path,
        title="ComParE+IS10 Final Validation AUROC",
        subtitle="Best validation-selected test row per protocol; lower temporal AUROC indicates time-shift sensitivity.",
    )


def write_final_validation_summary_figure(
    summary: pd.DataFrame,
    external_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    rows: list[tuple[str, float]] = []
    for _, row in summary.iterrows():
        label = str(row.get("evaluation_protocol", "unknown")).replace("compare_is10_", "")
        rows.append((label, _numeric(row.get("auroc"))))
    if not external_metrics.empty:
        external = external_metrics[~_has_skipped_column(external_metrics)].copy()
        if not external.empty:
            external["auroc"] = pd.to_numeric(external.get("auroc"), errors="coerce")
            best = external.sort_values(["auroc", "model_name"], ascending=[False, True]).iloc[0]
            rows.append(("external_transfer:" + str(best.get("model_name", "model")), _numeric(best.get("auroc"))))
    _write_svg_bars(
        rows,
        output_path,
        title="Final Validation Summary",
        subtitle="Internal split, time-aware stress tests, and optional external-transfer reference.",
    )
