from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.fusion import uniform_fusion, validation_weighted_fusion
from covid_rars.metadata_confounding import build_audit_feature_frame, run_metadata_confounding_audit
from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary


TEMPORAL_PROTOCOL = "temporal_early_to_late"
EXISTING_PROTOCOL = "existing_participant_split"
TIME_STRATIFIED_PROTOCOL = "time_stratified_participant_split"
REQUIRED_SPLITS = ("train", "validation", "test")
NON_FEATURE_COLUMNS = {
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
    "segmentation_method",
    "quality_flag",
}


@dataclass
class TemporalHoldoutResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    split_summary: pd.DataFrame
    modality_coverage: pd.DataFrame
    metadata_feature_importance: pd.DataFrame
    metadata_group_summary: pd.DataFrame
    metadata_ablation: pd.DataFrame
    stability_summary: pd.DataFrame
    bootstrap_ci: pd.DataFrame
    external_unification: pd.DataFrame


def _feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in feature_df.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[col])
    ]


def _make_model(model_name: str, random_state: int) -> object:
    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=random_state)
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown temporal holdout model name: {model_name}")


def _predict_prob(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not expose predict_proba or decision_function")


def _metric_bundle_with_prevalence_lift(
    labels: pd.Series,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = labels_to_binary(labels)
    y_prob = np.asarray(probabilities, dtype=float)
    metrics = binary_metric_bundle(y_true, y_prob, threshold=threshold)
    prevalence = float(np.mean(y_true)) if len(y_true) else float("nan")
    metrics["test_positive_prevalence"] = prevalence
    auprc = metrics.get("auprc", float("nan"))
    metrics["auprc_lift_over_prevalence"] = (
        float(auprc) - prevalence if np.isfinite(float(auprc)) and np.isfinite(prevalence) else float("nan")
    )
    return metrics


def _add_lift_to_existing_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if "test_positive_prevalence" in out.columns and "auprc" in out.columns:
        prevalence = pd.to_numeric(out["test_positive_prevalence"], errors="coerce")
        auprc = pd.to_numeric(out["auprc"], errors="coerce")
        out["auprc_lift_over_prevalence"] = auprc - prevalence
    return out


def _combination_name(modalities: tuple[str, ...] | list[str]) -> str:
    return "+".join(sorted(str(modality) for modality in modalities))


def _required_columns(frame: pd.DataFrame, columns: set[str], frame_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise KeyError(f"{frame_name} missing required columns: {sorted(missing)}")


def _participant_label(labels: pd.Series) -> str:
    counts = labels.astype(str).value_counts()
    if counts.empty:
        return "unknown"
    return str(counts.index[0])


def _participant_dates(metadata: pd.DataFrame, date_column: str) -> pd.DataFrame:
    _required_columns(metadata, {"participant_id", "label_binary", date_column}, "metadata")
    df = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    df["_recording_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=["_recording_date"])
    if df.empty:
        raise ValueError("No labeled rows with valid recording dates are available for temporal holdout")
    participants = (
        df.groupby("participant_id", dropna=False)
        .agg(
            recording_date=("_recording_date", "min"),
            label_binary=("label_binary", _participant_label),
            n_rows=("label_binary", "size"),
        )
        .reset_index()
    )
    return participants.sort_values(["recording_date", "participant_id"]).reset_index(drop=True)


def _split_counts(n_items: int, train_fraction: float, validation_fraction: float) -> tuple[int, int, int]:
    if n_items < 3:
        raise ValueError("Need at least three participants for train/validation/test temporal holdout")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    train_n = max(1, int(np.floor(n_items * train_fraction)))
    validation_n = max(1, int(np.floor(n_items * validation_fraction)))
    test_n = n_items - train_n - validation_n
    if test_n < 1:
        test_n = 1
        if validation_n > 1:
            validation_n -= 1
        else:
            train_n -= 1
    if train_n < 1 or validation_n < 1 or test_n < 1:
        raise ValueError("Temporal split fractions produced an empty split")
    return train_n, validation_n, test_n


def _summarize_split_participants(
    participants: pd.DataFrame,
    split_column: str,
    evaluation_protocol: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in REQUIRED_SPLITS:
        group = participants[participants[split_column].astype(str).eq(split)].copy()
        labels = group["label_binary"].astype(str)
        n_positive = int(labels.eq("positive").sum())
        n_negative = int(labels.eq("negative").sum())
        total = n_positive + n_negative
        rows.append(
            {
                "evaluation_protocol": evaluation_protocol,
                "temporal_split": split,
                "n_participants": int(group["participant_id"].nunique()),
                "n_rows": int(group.get("n_rows", pd.Series([len(group)])).sum()) if not group.empty else 0,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_prevalence": float(n_positive / total) if total else float("nan"),
                "date_min": group["recording_date"].min() if not group.empty else pd.NaT,
                "date_max": group["recording_date"].max() if not group.empty else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def build_temporal_split_assignments(
    metadata: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    date_column: str = "recording_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = _participant_dates(metadata, date_column=date_column)
    train_n, validation_n, _ = _split_counts(len(participants), train_fraction, validation_fraction)
    split_values = np.array(["test"] * len(participants), dtype=object)
    split_values[:train_n] = "train"
    split_values[train_n : train_n + validation_n] = "validation"
    assignments = participants.copy()
    assignments["temporal_split"] = split_values
    assignments["temporal_order"] = np.arange(len(assignments), dtype=int)
    assignments["temporal_train_fraction"] = float(train_fraction)
    assignments["temporal_validation_fraction"] = float(validation_fraction)
    summary = _summarize_split_participants(assignments, "temporal_split", TEMPORAL_PROTOCOL)
    return assignments, summary


def build_time_stratified_split_assignments(
    metadata: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    date_column: str = "recording_date",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a participant split that preserves recording-month coverage across splits.

    This is not a chronological holdout. It is the calendar-balanced reference used
    to test whether the early-to-late collapse is mainly a time-distribution effect.
    """
    participants = _participant_dates(metadata, date_column=date_column).copy()
    participants["recording_year_month"] = participants["recording_date"].dt.to_period("M").astype(str)
    rng = np.random.default_rng(random_state)
    assigned: list[pd.DataFrame] = []
    small_group_counter = 0

    for _, group in participants.groupby("recording_year_month", sort=True, dropna=False):
        group = group.copy().reset_index(drop=True)
        if len(group) > 1:
            group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        n_items = len(group)
        if n_items >= 3:
            train_n, validation_n, test_n = _split_counts(n_items, train_fraction, validation_fraction)
            splits = ["train"] * train_n + ["validation"] * validation_n + ["test"] * test_n
        elif n_items == 2:
            splits = ["train", "validation"] if small_group_counter % 2 == 0 else ["train", "test"]
            small_group_counter += 1
        else:
            splits = [["train", "validation", "test"][small_group_counter % 3]]
            small_group_counter += 1
        group["time_stratified_split"] = splits[:n_items]
        assigned.append(group)

    assignments = pd.concat(assigned, ignore_index=True, sort=False) if assigned else participants.iloc[0:0].copy()
    if assignments.empty:
        raise ValueError("No participants available for time-stratified split")

    missing_splits = [split for split in REQUIRED_SPLITS if split not in set(assignments["time_stratified_split"])]
    for split in missing_splits:
        donor_counts = assignments["time_stratified_split"].value_counts()
        donor_split = str(donor_counts.idxmax())
        donor_index = assignments[assignments["time_stratified_split"].eq(donor_split)].index[-1]
        assignments.loc[donor_index, "time_stratified_split"] = split

    assignments = assignments.sort_values(["recording_date", "participant_id"]).reset_index(drop=True)
    assignments["time_stratified_order"] = np.arange(len(assignments), dtype=int)
    assignments["temporal_train_fraction"] = float(train_fraction)
    assignments["temporal_validation_fraction"] = float(validation_fraction)
    summary = _summarize_split_participants(assignments, "time_stratified_split", TIME_STRATIFIED_PROTOCOL)
    return assignments, summary


def _existing_split_summary(metadata: pd.DataFrame, date_column: str = "recording_date") -> pd.DataFrame:
    participants = _participant_dates(metadata, date_column=date_column)
    split_map = (
        metadata[metadata["label_binary"].isin(["positive", "negative"])]
        .groupby("participant_id", dropna=False)["split"]
        .agg(lambda values: str(values.value_counts().index[0]))
        .reset_index(name="existing_split")
    )
    participants = participants.merge(split_map, on="participant_id", how="left")
    participants = participants[participants["existing_split"].isin(REQUIRED_SPLITS)].copy()
    return _summarize_split_participants(participants, "existing_split", EXISTING_PROTOCOL)


def _apply_split(metadata: pd.DataFrame, assignments: pd.DataFrame, split_column: str) -> pd.DataFrame:
    out = metadata.merge(assignments[["participant_id", split_column]], on="participant_id", how="left")
    out["split"] = out[split_column].fillna("unused")
    return out.drop(columns=[split_column])


def _apply_split_to_features(features: pd.DataFrame, assignments: pd.DataFrame, split_column: str) -> pd.DataFrame:
    _required_columns(features, {"participant_id", "label_binary", "modality", "split"}, "features")
    out = features.merge(assignments[["participant_id", split_column]], on="participant_id", how="left")
    out["split"] = out[split_column].fillna("unused")
    return out.drop(columns=[split_column])


def _apply_temporal_split_to_features(features: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    return _apply_split_to_features(features, assignments, "temporal_split")


def _prediction_frame(source: pd.DataFrame, probabilities: np.ndarray, model_name: str, threshold: float) -> pd.DataFrame:
    recording_id = source["recording_id"] if "recording_id" in source.columns else pd.Series([""] * len(source), index=source.index)
    submodality = source["submodality"] if "submodality" in source.columns else pd.Series([""] * len(source), index=source.index)
    return pd.DataFrame(
        {
            "recording_id": recording_id.to_numpy(),
            "participant_id": source["participant_id"].to_numpy(),
            "dataset": source.get("dataset", pd.Series(["coswara"] * len(source), index=source.index)).to_numpy(),
            "modality": source["modality"].to_numpy(),
            "submodality": submodality.to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": source["split"].to_numpy(),
            "model_name": model_name,
            "probability": probabilities,
            "threshold": threshold,
        }
    )


def _train_audio_model(
    features: pd.DataFrame,
    model_name: str,
    modality: str,
    random_state: int,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame]:
    df = features[
        features["label_binary"].isin(["positive", "negative"])
        & features["modality"].astype(str).eq(str(modality))
        & features["split"].isin(REQUIRED_SPLITS)
    ].copy()
    if df.empty:
        raise ValueError(f"No labeled feature rows available for modality={modality}")
    cols = _feature_columns(df)
    if not cols:
        raise ValueError(f"No numeric feature columns available for modality={modality}")
    train = df[df["split"].eq("train")].copy()
    validation = df[df["split"].eq("validation")].copy()
    test = df[df["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Need train/validation/test feature rows for modality={modality}")
    if train["label_binary"].nunique() < 2:
        raise ValueError(f"Training split for modality={modality} has only one class")

    model = _make_model(model_name, random_state=random_state)
    model.fit(train[cols].fillna(0.0), labels_to_binary(train["label_binary"]))
    validation_prob = _predict_prob(model, validation[cols].fillna(0.0))
    threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)
    test_prob = _predict_prob(model, test[cols].fillna(0.0))

    validation_metrics = _metric_bundle_with_prevalence_lift(validation["label_binary"], validation_prob, threshold=threshold)
    validation_metrics.update(
        {
            "analysis_family": "audio_modality",
            "model_name": model_name,
            "modality": modality,
            "metric_split": "validation",
            "n_features": float(len(cols)),
        }
    )
    test_metrics = _metric_bundle_with_prevalence_lift(test["label_binary"], test_prob, threshold=threshold)
    test_metrics.update(
        {
            "analysis_family": "audio_modality",
            "model_name": model_name,
            "modality": modality,
            "metric_split": "test",
            "n_features": float(len(cols)),
        }
    )
    validation_predictions = _prediction_frame(validation, validation_prob, model_name=model_name, threshold=threshold)
    test_predictions = _prediction_frame(test, test_prob, model_name=model_name, threshold=threshold)
    return test_metrics, validation_metrics, validation_predictions, test_predictions


def _safe_validation_metrics(validation_metrics: pd.DataFrame, metric_column: str = "auprc") -> pd.DataFrame:
    safe = validation_metrics.copy()
    if metric_column in safe.columns:
        safe[metric_column] = pd.to_numeric(safe[metric_column], errors="coerce")
        safe[metric_column] = safe[metric_column].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    return safe


def _evaluate_fusion(
    prediction_frames: list[pd.DataFrame],
    validation_metrics: pd.DataFrame,
    model_name: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    branch_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    branch_predictions = branch_predictions[branch_predictions["model_name"].astype(str).eq(str(model_name))].copy()
    available_modalities = sorted(str(item) for item in branch_predictions["modality"].dropna().unique())
    if branch_predictions.empty or len(available_modalities) < 2:
        return [], []
    metric_rows: list[dict[str, object]] = []
    prediction_outputs: list[pd.DataFrame] = []
    safe_validation_metrics = _safe_validation_metrics(validation_metrics, metric_column="auprc")

    modality_combinations: list[tuple[str, ...]] = []
    for size in range(2, len(available_modalities) + 1):
        modality_combinations.extend(combinations(available_modalities, size))

    for combo in modality_combinations:
        combo_name = _combination_name(combo)
        combo_predictions = branch_predictions[branch_predictions["modality"].astype(str).isin(combo)].copy()
        combo_validation_metrics = safe_validation_metrics[
            safe_validation_metrics["modality"].astype(str).isin(combo)
        ].copy()
        fused_frames = [
            uniform_fusion(combo_predictions),
            validation_weighted_fusion(combo_predictions, combo_validation_metrics, metric_column="auprc"),
        ]
        for fused in fused_frames:
            validation = fused[fused["split"].eq("validation")].copy()
            test = fused[fused["split"].eq("test")].copy()
            validation = validation[np.isfinite(validation["probability"].astype(float))].copy()
            test = test[np.isfinite(test["probability"].astype(float))].copy()
            if validation.empty or test.empty:
                continue
            threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation["probability"])
            for split_name, group in (("validation", validation), ("test", test)):
                pred = group.copy()
                pred["model_name"] = model_name
                pred["modality"] = "multimodal"
                pred["modality_combination"] = combo_name
                pred["threshold"] = threshold
                pred["analysis_family"] = "multimodal_fusion"
                prediction_outputs.append(pred)
                if split_name != "test":
                    continue
                metrics = _metric_bundle_with_prevalence_lift(
                    group["label_binary"],
                    group["probability"],
                    threshold=threshold,
                )
                metrics.update(
                    {
                        "analysis_family": "multimodal_fusion",
                        "model_name": model_name,
                        "modality": "multimodal",
                        "modality_combination": combo_name,
                        "n_modalities": float(len(combo)),
                        "fusion_method": str(group["fusion_method"].iloc[0]),
                        "available_modalities": str(group["available_modalities"].mode().iloc[0]),
                        "metric_split": "test",
                    }
                )
                metric_rows.append(metrics)
    return metric_rows, prediction_outputs

def _modality_coverage(features: pd.DataFrame, evaluation_protocol: str) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, modality), group in features.groupby(["split", "modality"], dropna=False):
        labels = group["label_binary"].astype(str)
        rows.append(
            {
                "evaluation_protocol": evaluation_protocol,
                "split": split,
                "modality": modality,
                "n_rows": int(len(group)),
                "n_participants": int(group["participant_id"].nunique()),
                "n_positive": int(labels.eq("positive").sum()),
                "n_negative": int(labels.eq("negative").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["evaluation_protocol", "split", "modality"]).reset_index(drop=True)


def _run_audio_protocol(
    features: pd.DataFrame,
    evaluation_protocol: str,
    modalities: list[str],
    model_names: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_metric_rows: list[dict[str, object]] = []
    validation_prediction_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        for modality in modalities:
            try:
                metrics, validation_metrics, validation_predictions, test_predictions = _train_audio_model(
                    features,
                    model_name=model_name,
                    modality=modality,
                    random_state=random_state,
                )
            except Exception as exc:
                metric_rows.append(
                    {
                        "analysis_family": "audio_modality",
                        "evaluation_protocol": evaluation_protocol,
                        "model_name": model_name,
                        "modality": modality,
                        "metric_split": "test",
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                )
                continue
            for row in (metrics, validation_metrics):
                row["evaluation_protocol"] = evaluation_protocol
                row["skipped"] = False
            metric_rows.append(metrics)
            validation_metric_rows.append(validation_metrics)
            for frame in (validation_predictions, test_predictions):
                frame["analysis_family"] = "audio_modality"
                frame["evaluation_protocol"] = evaluation_protocol
                prediction_frames.append(frame)
            validation_prediction_frames.append(validation_predictions)

        usable_predictions = [
            frame
            for frame in prediction_frames
            if "model_name" in frame.columns and frame["model_name"].astype(str).eq(str(model_name)).any()
        ]
        if usable_predictions:
            fusion_metrics, fusion_predictions = _evaluate_fusion(
                usable_predictions,
                pd.DataFrame(validation_metric_rows),
                model_name=model_name,
            )
            for row in fusion_metrics:
                row["evaluation_protocol"] = evaluation_protocol
                row["skipped"] = False
                metric_rows.append(row)
            for frame in fusion_predictions:
                frame["evaluation_protocol"] = evaluation_protocol
                prediction_frames.append(frame)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def _run_metadata_protocol(
    metadata: pd.DataFrame,
    evaluation_protocol: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = run_metadata_confounding_audit(metadata, random_state=random_state)
    metrics = _add_lift_to_existing_metric_rows(result.metrics.copy())
    predictions = result.predictions.copy()
    importance = result.feature_importance.copy()
    group_summary = result.group_summary.copy()
    for frame in (metrics, predictions, importance, group_summary):
        frame["evaluation_protocol"] = evaluation_protocol
    metrics["analysis_family"] = "metadata_confounding"
    predictions["analysis_family"] = "metadata_confounding"
    return metrics, predictions, importance, group_summary



def _align_metadata_frames(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    columns = sorted(set().union(*(frame.columns for frame in frames))) if frames else []
    aligned = [frame.reindex(columns=columns, fill_value=0.0) for frame in frames]
    return aligned, columns


def _metadata_ablation_specs() -> list[tuple[str, str, list[str]]]:
    return [
        ("demographic_protocol_only", "demographic_protocol_full", []),
        ("demographic_protocol_only", "demographic_protocol_no_recording_year", ["recording_year"]),
        ("demographic_protocol_only", "demographic_protocol_no_recording_month", ["recording_month"]),
        (
            "demographic_protocol_only",
            "demographic_protocol_no_recording_year_month",
            ["recording_year", "recording_month"],
        ),
        ("recording_date_only", "recording_year_only", ["recording_month"]),
        ("recording_date_only", "recording_month_only", ["recording_year"]),
        ("recording_date_only", "recording_year_month_only", []),
        ("full_safe_metadata", "full_safe_metadata_full", []),
        ("full_safe_metadata", "full_safe_metadata_no_recording_year", ["recording_year"]),
        ("full_safe_metadata", "full_safe_metadata_no_recording_month", ["recording_month"]),
        (
            "full_safe_metadata",
            "full_safe_metadata_no_recording_year_month",
            ["recording_year", "recording_month"],
        ),
    ]


def _fit_metadata_ablation(
    metadata: pd.DataFrame,
    feature_set: str,
    ablation_name: str,
    removed_features: list[str],
    evaluation_protocol: str,
    random_state: int,
) -> dict[str, object]:
    df = metadata[
        metadata["label_binary"].isin(["positive", "negative"])
        & metadata["split"].isin(REQUIRED_SPLITS)
    ].copy()
    train = df[df["split"].eq("train")].copy()
    validation = df[df["split"].eq("validation")].copy()
    test = df[df["split"].eq("test")].copy()
    base_row: dict[str, object] = {
        "evaluation_protocol": evaluation_protocol,
        "analysis_family": "metadata_temporal_ablation",
        "model_name": "metadata_confounding_logistic_regression",
        "modality": "metadata",
        "base_feature_set": feature_set,
        "audit_model": ablation_name,
        "ablation_name": ablation_name,
        "removed_features": ";".join(removed_features) if removed_features else "none",
        "skipped": False,
    }
    try:
        if train.empty or validation.empty or test.empty:
            raise ValueError("Need train/validation/test rows for metadata temporal ablation")
        if train["label_binary"].nunique() < 2:
            raise ValueError("Training split has only one class")

        feature_columns = ["recording_date"] if feature_set == "recording_date_only" else None
        builder_feature_set = "demographic_protocol_only" if feature_set == "recording_date_only" else feature_set
        train_raw, _ = build_audit_feature_frame(
            train,
            feature_set=builder_feature_set,
            feature_columns=feature_columns,
        )
        validation_raw, _ = build_audit_feature_frame(
            validation,
            feature_set=builder_feature_set,
            feature_columns=feature_columns,
        )
        test_raw, _ = build_audit_feature_frame(
            test,
            feature_set=builder_feature_set,
            feature_columns=feature_columns,
        )
        for frame in (train_raw, validation_raw, test_raw):
            drop_cols = [col for col in removed_features if col in frame.columns]
            if drop_cols:
                frame.drop(columns=drop_cols, inplace=True)
        (train_x, validation_x, test_x), columns = _align_metadata_frames([train_raw, validation_raw, test_raw])
        varying_columns = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
        if not varying_columns:
            raise ValueError(f"Ablation {ablation_name} selected no train-varying columns")
        train_x = train_x[varying_columns]
        validation_x = validation_x[varying_columns]
        test_x = test_x[varying_columns]

        model = _make_model("logistic_regression", random_state=random_state)
        model.fit(train_x, labels_to_binary(train["label_binary"]))
        validation_prob = _predict_prob(model, validation_x)
        threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)
        test_prob = _predict_prob(model, test_x)
        metrics = _metric_bundle_with_prevalence_lift(test["label_binary"], test_prob, threshold=threshold)
        base_row.update(metrics)
        base_row.update(
            {
                "train_rows": float(len(train)),
                "validation_rows": float(len(validation)),
                "test_rows": float(len(test)),
                "n_features": float(len(varying_columns)),
            }
        )
    except Exception as exc:
        base_row.update(
            {
                "skipped": True,
                "skip_reason": str(exc),
                "train_rows": float(len(train)),
                "validation_rows": float(len(validation)),
                "test_rows": float(len(test)),
                "n_features": float("nan"),
            }
        )
    return base_row


def _run_metadata_ablation_protocol(
    metadata: pd.DataFrame,
    evaluation_protocol: str,
    random_state: int,
) -> pd.DataFrame:
    rows = [
        _fit_metadata_ablation(
            metadata,
            feature_set=feature_set,
            ablation_name=ablation_name,
            removed_features=removed_features,
            evaluation_protocol=evaluation_protocol,
            random_state=random_state,
        )
        for feature_set, ablation_name, removed_features in _metadata_ablation_specs()
    ]
    return pd.DataFrame(rows)


def _usable_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if "metric_split" in out.columns:
        out = out[out["metric_split"].fillna("test").astype(str).eq("test")].copy()
    if "skipped" in out.columns:
        out = out[~out["skipped"].fillna(False).astype(str).str.lower().eq("true")].copy()
    return out


def _identity_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "analysis_family",
        "model_name",
        "modality",
        "modality_combination",
        "fusion_method",
        "audit_model",
        "ablation_name",
    ]
    return [col for col in candidates if col in frame.columns]


def _key_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def build_temporal_stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    work = _usable_metric_rows(metrics)
    if work.empty or "evaluation_protocol" not in work.columns:
        return pd.DataFrame()
    key_cols = _identity_columns(work)
    existing = work[work["evaluation_protocol"].astype(str).eq(EXISTING_PROTOCOL)].copy()
    temporal = work[work["evaluation_protocol"].astype(str).eq(TEMPORAL_PROTOCOL)].copy()
    time_stratified = work[work["evaluation_protocol"].astype(str).eq(TIME_STRATIFIED_PROTOCOL)].copy()
    if existing.empty or temporal.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    metric_cols = ["auroc", "auprc", "balanced_accuracy", "brier", "ece", "auprc_lift_over_prevalence"]
    for _, existing_row in existing.iterrows():
        mask = pd.Series(True, index=temporal.index)
        time_mask = pd.Series(True, index=time_stratified.index)
        for col in key_cols:
            key = _key_value(existing_row.get(col))
            mask &= temporal[col].map(_key_value).eq(key)
            if not time_stratified.empty:
                time_mask &= time_stratified[col].map(_key_value).eq(key)
        matching_temporal = temporal[mask]
        if matching_temporal.empty:
            continue
        temporal_row = matching_temporal.iloc[0]
        matching_time = time_stratified[time_mask] if not time_stratified.empty else pd.DataFrame()
        time_row = matching_time.iloc[0] if not matching_time.empty else None
        row = {col: existing_row.get(col) for col in key_cols}
        for metric in metric_cols:
            existing_value = pd.to_numeric(pd.Series([existing_row.get(metric)]), errors="coerce").iloc[0]
            temporal_value = pd.to_numeric(pd.Series([temporal_row.get(metric)]), errors="coerce").iloc[0]
            row[f"existing_{metric}"] = existing_value
            row[f"temporal_{metric}"] = temporal_value
            row[f"delta_{metric}_temporal_minus_existing"] = temporal_value - existing_value
            if time_row is not None:
                time_value = pd.to_numeric(pd.Series([time_row.get(metric)]), errors="coerce").iloc[0]
                row[f"time_stratified_{metric}"] = time_value
                row[f"delta_{metric}_time_stratified_minus_existing"] = time_value - existing_value
        row["existing_n_samples"] = existing_row.get("n_samples")
        row["temporal_n_samples"] = temporal_row.get("n_samples")
        if time_row is not None:
            row["time_stratified_n_samples"] = time_row.get("n_samples")
        rows.append(row)
    summary = pd.DataFrame(rows)
    alias_map = {
        "delta_auprc_lift_over_prevalence_temporal_minus_existing": "delta_auprc_lift_temporal_minus_existing",
        "delta_auprc_lift_over_prevalence_time_stratified_minus_existing": "delta_auprc_lift_time_stratified_minus_existing",
        "existing_auprc_lift_over_prevalence": "existing_auprc_lift",
        "temporal_auprc_lift_over_prevalence": "temporal_auprc_lift",
        "time_stratified_auprc_lift_over_prevalence": "time_stratified_auprc_lift",
    }
    for source, target in alias_map.items():
        if source in summary.columns and target not in summary.columns:
            summary[target] = summary[source]
    return summary


def bootstrap_temporal_metric_ci(
    predictions: pd.DataFrame,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    if predictions.empty or n_bootstraps <= 0:
        return pd.DataFrame()
    required = {"participant_id", "label_binary", "probability", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing required bootstrap columns: {sorted(missing)}")
    work = predictions[
        predictions["split"].astype(str).eq("test")
        & predictions["label_binary"].isin(["positive", "negative"])
    ].copy()
    work["probability"] = pd.to_numeric(work["probability"], errors="coerce")
    work = work[np.isfinite(work["probability"])].copy()
    if work.empty:
        return pd.DataFrame()

    group_cols = [
        col
        for col in [
            "evaluation_protocol",
            "analysis_family",
            "model_name",
            "modality",
            "modality_combination",
            "fusion_method",
            "audit_model",
        ]
        if col in work.columns
    ]
    for col in group_cols:
        work[col] = work[col].fillna("").astype(str)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    alpha = (1.0 - confidence) / 2.0
    metric_names = ["auroc", "auprc", "balanced_accuracy", "brier", "ece"]

    iterator = work.groupby(group_cols, dropna=False) if group_cols else [((), work)]
    for group_key, group in iterator:
        if len(group) < 2 or group["label_binary"].nunique() < 2:
            continue
        threshold = 0.5
        if "threshold" in group.columns:
            threshold_values = pd.to_numeric(group["threshold"], errors="coerce").dropna()
            if not threshold_values.empty:
                threshold = float(threshold_values.iloc[0])
        y = labels_to_binary(group["label_binary"])
        prob = group["probability"].to_numpy(dtype=float)
        point = binary_metric_bundle(y, prob, threshold=threshold)
        participant_codes, participant_ids = pd.factorize(group["participant_id"].astype(str), sort=False)
        n_participants = len(participant_ids)
        if n_participants < 2:
            continue
        participant_indices = [np.flatnonzero(participant_codes == idx) for idx in range(n_participants)]
        bootstrap_values: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for _ in range(n_bootstraps):
            sampled_participants = rng.integers(0, n_participants, size=n_participants)
            sampled_indices = np.concatenate([participant_indices[idx] for idx in sampled_participants])
            y_sample = y[sampled_indices]
            if len(np.unique(y_sample)) < 2:
                continue
            sample_metrics = binary_metric_bundle(y_sample, prob[sampled_indices], threshold=threshold)
            for metric in metric_names:
                value = float(sample_metrics.get(metric, float("nan")))
                if np.isfinite(value):
                    bootstrap_values[metric].append(value)
        if group_cols:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            key_payload = dict(zip(group_cols, group_key))
        else:
            key_payload = {}
        for metric in metric_names:
            values = np.asarray(bootstrap_values[metric], dtype=float)
            if values.size == 0:
                continue
            row = dict(key_payload)
            row.update(
                {
                    "metric": metric,
                    "point": float(point.get(metric, float("nan"))),
                    "mean": float(np.mean(values)),
                    "ci_low": float(np.quantile(values, alpha)),
                    "ci_high": float(np.quantile(values, 1.0 - alpha)),
                    "confidence": float(confidence),
                    "n_bootstraps": int(values.size),
                    "n_samples": int(len(group)),
                    "n_participants": int(n_participants),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)

def _select_unification_row(metrics: pd.DataFrame, protocol: str) -> pd.Series | None:
    work = _usable_metric_rows(metrics)
    work = work[work["evaluation_protocol"].astype(str).eq(protocol)].copy()
    if work.empty:
        return None
    preferred = work[
        work.get("analysis_family", pd.Series(index=work.index, dtype=object)).astype(str).eq("multimodal_fusion")
        & work.get("modality_combination", pd.Series(index=work.index, dtype=object)).astype(str).eq("breath+cough+speech")
        & work.get("fusion_method", pd.Series(index=work.index, dtype=object)).astype(str).eq("uniform_mean")
    ].copy()
    if preferred.empty:
        preferred = work[work.get("analysis_family", pd.Series(index=work.index, dtype=object)).astype(str).eq("multimodal_fusion")].copy()
    if preferred.empty:
        preferred = work.copy()
    auroc = pd.to_numeric(preferred.get("auroc"), errors="coerce")
    if auroc.notna().any():
        return preferred.loc[auroc.idxmax()]
    return preferred.iloc[0]


def build_temporal_external_unification(
    metrics: pd.DataFrame,
    external_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    protocol_specs = [
        ("participant_internal", EXISTING_PROTOCOL, "random participant split internal reference"),
        ("time_stratified_internal", TIME_STRATIFIED_PROTOCOL, "calendar-balanced participant split reference"),
        ("temporal_holdout", TEMPORAL_PROTOCOL, "early-to-late chronological stress test"),
    ]
    for stress_test, protocol, interpretation in protocol_specs:
        row = _select_unification_row(metrics, protocol)
        if row is None:
            continue
        rows.append(
            {
                "stress_test": stress_test,
                "evaluation_protocol": protocol,
                "analysis_family": row.get("analysis_family"),
                "model_name": row.get("model_name"),
                "modality": row.get("modality"),
                "modality_combination": row.get("modality_combination"),
                "fusion_method": row.get("fusion_method"),
                "auroc": row.get("auroc"),
                "auprc": row.get("auprc"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "brier": row.get("brier"),
                "ece": row.get("ece"),
                "n_samples": row.get("n_samples"),
                "interpretation": interpretation,
            }
        )
    if external_reference is not None and not external_reference.empty:
        external = external_reference.copy()
        if {"claim_id", "primary_metric", "primary_value"}.issubset(external.columns):
            external = external[
                external["claim_id"].astype(str).str.startswith("external_transfer_")
                & external["primary_metric"].astype(str).eq("auroc")
            ].copy()
            external["primary_value"] = pd.to_numeric(external["primary_value"], errors="coerce")
            if external["primary_value"].notna().any():
                best = external.loc[external["primary_value"].idxmax()]
                rows.append(
                    {
                        "stress_test": "external_transfer",
                        "evaluation_protocol": "coswara_to_coughvid_external",
                        "analysis_family": "external_transfer",
                        "model_name": best.get("claim_id"),
                        "modality": "cough",
                        "modality_combination": "cough",
                        "fusion_method": best.get("comparison"),
                        "auroc": float(best["primary_value"]),
                        "auprc": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "brier": float("nan"),
                        "ece": float("nan"),
                        "n_samples": best.get("n_samples"),
                        "interpretation": "independent dataset-transfer stress test",
                    }
                )
    return pd.DataFrame(rows)

def run_temporal_holdout_audit(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    modalities: list[str] | None = None,
    model_names: list[str] | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    include_existing_split_reference: bool = True,
    include_time_stratified_reference: bool = True,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> TemporalHoldoutResult:
    modalities = modalities or ["cough", "breath", "speech"]
    model_names = model_names or ["logistic_regression"]
    assignments, temporal_summary = build_temporal_split_assignments(
        metadata,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    protocols: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    temporal_metadata = _apply_split(metadata, assignments, "temporal_split")
    temporal_features = _apply_temporal_split_to_features(features, assignments)
    protocols.append((TEMPORAL_PROTOCOL, temporal_metadata, temporal_features, temporal_summary))

    if include_time_stratified_reference:
        time_assignments, time_summary = build_time_stratified_split_assignments(
            metadata,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        time_metadata = _apply_split(metadata, time_assignments, "time_stratified_split")
        time_features = _apply_split_to_features(features, time_assignments, "time_stratified_split")
        protocols.append((TIME_STRATIFIED_PROTOCOL, time_metadata, time_features, time_summary))

    if include_existing_split_reference and "split" in metadata.columns and "split" in features.columns:
        existing_summary = _existing_split_summary(metadata)
        protocols.append((EXISTING_PROTOCOL, metadata.copy(), features.copy(), existing_summary))

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    split_summaries: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    group_summary_frames: list[pd.DataFrame] = []
    metadata_ablation_frames: list[pd.DataFrame] = []

    for evaluation_protocol, protocol_metadata, protocol_features, split_summary in protocols:
        split_summaries.append(split_summary)
        coverage_frames.append(_modality_coverage(protocol_features, evaluation_protocol))
        audio_metrics, audio_predictions = _run_audio_protocol(
            protocol_features,
            evaluation_protocol=evaluation_protocol,
            modalities=modalities,
            model_names=model_names,
            random_state=random_state,
        )
        if not audio_metrics.empty:
            metric_frames.append(audio_metrics)
        if not audio_predictions.empty:
            prediction_frames.append(audio_predictions)
        metadata_metrics, metadata_predictions, importance, group_summary = _run_metadata_protocol(
            protocol_metadata,
            evaluation_protocol=evaluation_protocol,
            random_state=random_state,
        )
        metric_frames.append(metadata_metrics)
        prediction_frames.append(metadata_predictions)
        importance_frames.append(importance)
        group_summary_frames.append(group_summary)
        metadata_ablation = _run_metadata_ablation_protocol(
            protocol_metadata,
            evaluation_protocol=evaluation_protocol,
            random_state=random_state,
        )
        metadata_ablation_frames.append(metadata_ablation)

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    metadata_ablation = (
        pd.concat(metadata_ablation_frames, ignore_index=True, sort=False)
        if metadata_ablation_frames
        else pd.DataFrame()
    )
    stability_summary = build_temporal_stability_summary(metrics)
    bootstrap_ci = bootstrap_temporal_metric_ci(
        predictions,
        n_bootstraps=bootstrap_samples,
        random_state=random_state,
    )
    external_unification = build_temporal_external_unification(metrics)

    return TemporalHoldoutResult(
        metrics=metrics,
        predictions=predictions,
        split_summary=pd.concat(split_summaries, ignore_index=True, sort=False) if split_summaries else pd.DataFrame(),
        modality_coverage=pd.concat(coverage_frames, ignore_index=True, sort=False) if coverage_frames else pd.DataFrame(),
        metadata_feature_importance=pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame(),
        metadata_group_summary=pd.concat(group_summary_frames, ignore_index=True, sort=False) if group_summary_frames else pd.DataFrame(),
        metadata_ablation=metadata_ablation,
        stability_summary=stability_summary,
        bootstrap_ci=bootstrap_ci,
        external_unification=external_unification,
    )
