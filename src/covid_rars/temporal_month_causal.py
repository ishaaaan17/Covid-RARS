from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.metadata_baseline import _parse_json_dict
from covid_rars.metadata_confounding import build_audit_feature_frame
from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary
from covid_rars.temporal_holdout import (
    REQUIRED_SPLITS,
    TEMPORAL_PROTOCOL,
    _apply_split_to_features,
    _feature_columns,
    _make_model,
    _predict_prob,
    build_temporal_split_assignments,
)

FULL_MULTIMODAL = "breath+cough+speech"
UNIFORM_FUSION = "uniform_mean"


@dataclass
class TemporalMonthCausalAuditResult:
    month_label_shift: pd.DataFrame
    month_covariate_shift: pd.DataFrame
    matched_cohort_metrics: pd.DataFrame
    failure_modes: pd.DataFrame
    uncertainty_summary: pd.DataFrame
    month_ablation_effect_sizes: pd.DataFrame
    causal_dag_markdown: str
    theory_markdown: str


def _as_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _metadata_truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(float(value)) and float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "present", "positive"}:
        return True
    if text in {"false", "0", "no", "n", "nan", "none", "null", "absent", "negative", ""}:
        return False
    return bool(text)


def _recording_month(metadata: pd.DataFrame, date_column: str = "recording_date") -> pd.Series:
    date = pd.to_datetime(metadata[date_column], errors="coerce")
    return date.dt.to_period("M").astype(str).where(date.notna(), "unknown")


def _participant_frame(metadata: pd.DataFrame, date_column: str = "recording_date") -> pd.DataFrame:
    labeled = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    if labeled.empty:
        return pd.DataFrame()
    labeled["recording_year_month"] = _recording_month(labeled, date_column=date_column)
    labeled["_recording_date"] = pd.to_datetime(labeled[date_column], errors="coerce")

    def _mode(values: pd.Series) -> object:
        counts = values.dropna().astype(str).value_counts()
        return counts.index[0] if not counts.empty else np.nan

    out = (
        labeled.groupby("participant_id", dropna=False)
        .agg(
            label_binary=("label_binary", _mode),
            recording_year_month=("recording_year_month", _mode),
            recording_date=("_recording_date", "min"),
            age=("age", "median") if "age" in labeled.columns else ("label_binary", "size"),
            gender=("gender", _mode) if "gender" in labeled.columns else ("label_binary", _mode),
            country=("country", _mode) if "country" in labeled.columns else ("label_binary", _mode),
            quality_flag=("quality_flag", _mode) if "quality_flag" in labeled.columns else ("label_binary", _mode),
            duration_sec=("duration_sec", "mean") if "duration_sec" in labeled.columns else ("label_binary", "size"),
            symptoms_json=("symptoms_json", _mode) if "symptoms_json" in labeled.columns else ("label_binary", _mode),
            n_rows=("label_binary", "size"),
        )
        .reset_index()
    )
    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")
    out["duration_sec"] = pd.to_numeric(out.get("duration_sec"), errors="coerce")
    out["symptom_count"] = out.get("symptoms_json", pd.Series(["{}"] * len(out))).map(
        lambda text: float(sum(_metadata_truthy(v) for v in _parse_json_dict(text).values()))
    )
    return out


def build_month_label_shift(metadata: pd.DataFrame, date_column: str = "recording_date") -> pd.DataFrame:
    participants = _participant_frame(metadata, date_column=date_column)
    if participants.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for month, group in participants.groupby("recording_year_month", dropna=False, sort=True):
        labels = group["label_binary"].astype(str)
        n_positive = int(labels.eq("positive").sum())
        n_negative = int(labels.eq("negative").sum())
        total = n_positive + n_negative
        rows.append(
            {
                "recording_year_month": month,
                "n_participants": int(group["participant_id"].nunique()),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_prevalence": float(n_positive / total) if total else float("nan"),
                "age_mean": float(group["age"].mean()) if "age" in group else float("nan"),
                "symptom_count_mean": float(group["symptom_count"].mean()) if "symptom_count" in group else float("nan"),
                "duration_sec_mean": float(group["duration_sec"].mean()) if "duration_sec" in group else float("nan"),
                "quality_ok_share": float(group["quality_flag"].astype(str).str.lower().eq("ok").mean()) if "quality_flag" in group else float("nan"),
                "female_share": float(group["gender"].astype(str).str.lower().eq("female").mean()) if "gender" in group else float("nan"),
                "top_country": str(group["country"].astype(str).mode().iloc[0]) if "country" in group and not group.empty else "",
            }
        )
    return pd.DataFrame(rows)


def build_month_covariate_shift(metadata: pd.DataFrame, date_column: str = "recording_date") -> pd.DataFrame:
    participants = _participant_frame(metadata, date_column=date_column)
    if participants.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for month, group in participants.groupby("recording_year_month", dropna=False, sort=True):
        base = {"recording_year_month": month, "n_participants": int(group["participant_id"].nunique())}
        numeric_specs = {
            "age_mean": group.get("age"),
            "duration_sec_mean": group.get("duration_sec"),
            "symptom_count_mean": group.get("symptom_count"),
        }
        for covariate, series in numeric_specs.items():
            rows.append({**base, "covariate": covariate, "value": float(pd.to_numeric(series, errors="coerce").mean())})
        if "gender" in group:
            rows.append({**base, "covariate": "female_share", "value": float(group["gender"].astype(str).str.lower().eq("female").mean())})
        if "quality_flag" in group:
            rows.append({**base, "covariate": "quality_ok_share", "value": float(group["quality_flag"].astype(str).str.lower().eq("ok").mean())})
        if "country" in group:
            country_counts = group["country"].astype(str).value_counts(normalize=True)
            for country, share in country_counts.head(5).items():
                rows.append({**base, "covariate": f"country_share_{country}", "value": float(share)})
        if "symptoms_json" in group:
            parsed = group["symptoms_json"].map(_parse_json_dict)
            symptom_keys = sorted({str(key) for item in parsed for key in item.keys()})
            for key in symptom_keys[:20]:
                rows.append(
                    {
                        **base,
                        "covariate": f"symptom_share_{key}",
                        "value": float(parsed.map(lambda item, k=key: _metadata_truthy(item.get(k, False))).mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_month_ablation_effect_sizes(month_ablation: pd.DataFrame) -> pd.DataFrame:
    if month_ablation.empty:
        return pd.DataFrame()
    work = month_ablation.copy()
    if "metadata_configuration" not in work.columns:
        label_map = {
            "full_safe_metadata_full": "Full metadata",
            "full_safe_metadata_no_recording_year": "Remove year",
            "full_safe_metadata_no_recording_month": "Remove month",
            "full_safe_metadata_no_recording_year_month": "Remove year + month",
        }
        work["metadata_configuration"] = work["ablation_name"].map(label_map).fillna(work["ablation_name"])
    metric_col = "temporal_auroc" if "temporal_auroc" in work.columns else "auroc"
    values = work.set_index("metadata_configuration")[metric_col].map(_as_float).to_dict()
    full = values.get("Full metadata", float("nan"))
    rows = []
    for label in ["Remove year", "Remove month", "Remove year + month"]:
        value = values.get(label, float("nan"))
        rows.append(
            {
                "comparison": f"{label} minus full metadata",
                "reference": "Full metadata",
                "comparison_configuration": label,
                "reference_auroc": full,
                "comparison_auroc": value,
                "delta_auroc": value - full if np.isfinite(value) and np.isfinite(full) else float("nan"),
                "interpretation": "improves_temporal_generalization" if np.isfinite(value) and np.isfinite(full) and value > full else "no_improvement",
            }
        )
    return pd.DataFrame(rows)


def _covariate_matrix(participants: pd.DataFrame) -> pd.DataFrame:
    if participants.empty:
        return pd.DataFrame(index=participants.index)
    frame = pd.DataFrame(index=participants.index)
    frame["age"] = pd.to_numeric(participants.get("age"), errors="coerce").fillna(pd.to_numeric(participants.get("age"), errors="coerce").median())
    frame["symptom_count"] = pd.to_numeric(participants.get("symptom_count"), errors="coerce").fillna(0.0)
    frame["duration_sec"] = pd.to_numeric(participants.get("duration_sec"), errors="coerce").fillna(0.0)
    for col in ["gender", "quality_flag", "country"]:
        if col in participants.columns:
            dummies = pd.get_dummies(participants[col].fillna("unknown").astype(str), prefix=col)
            frame = pd.concat([frame, dummies], axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def build_matched_temporal_participants(metadata: pd.DataFrame, date_column: str = "recording_date") -> pd.DataFrame:
    assignments, _ = build_temporal_split_assignments(metadata, date_column=date_column)
    participants = _participant_frame(metadata, date_column=date_column).merge(
        assignments[["participant_id", "temporal_split"]], on="participant_id", how="inner"
    )
    train = participants[participants["temporal_split"].eq("train")].copy().reset_index(drop=True)
    test = participants[participants["temporal_split"].eq("test")].copy().reset_index(drop=True)
    if train.empty or test.empty:
        return pd.DataFrame()
    train_x = _covariate_matrix(train)
    test_x = _covariate_matrix(test)
    columns = sorted(set(train_x.columns) | set(test_x.columns))
    train_x = train_x.reindex(columns=columns, fill_value=0.0)
    test_x = test_x.reindex(columns=columns, fill_value=0.0)
    scale = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    train_z = (train_x - train_x.mean(axis=0)) / scale
    test_z = (test_x - train_x.mean(axis=0)) / scale
    available_train = set(range(len(train)))
    rows: list[dict[str, object]] = []
    for test_idx, test_row in test_z.iterrows():
        if not available_train:
            break
        train_indices = np.array(sorted(available_train), dtype=int)
        distances = ((train_z.iloc[train_indices] - test_row) ** 2).sum(axis=1).to_numpy(dtype=float)
        best_pos = int(np.argmin(distances))
        train_idx = int(train_indices[best_pos])
        available_train.remove(train_idx)
        rows.append(
            {
                "matched_pair_id": len(rows),
                "train_participant_id": train.loc[train_idx, "participant_id"],
                "test_participant_id": test.loc[test_idx, "participant_id"],
                "distance": float(distances[best_pos]),
                "train_label": train.loc[train_idx, "label_binary"],
                "test_label": test.loc[test_idx, "label_binary"],
                "train_month": train.loc[train_idx, "recording_year_month"],
                "test_month": test.loc[test_idx, "recording_year_month"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_matched_temporal_audio(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    modalities: list[str] | None = None,
    model_name: str = "logistic_regression",
    random_state: int = 42,
) -> pd.DataFrame:
    modalities = modalities or ["cough", "breath", "speech"]
    assignments, _ = build_temporal_split_assignments(metadata)
    pairs = build_matched_temporal_participants(metadata)
    if pairs.empty:
        return pd.DataFrame()
    matched_train = set(pairs["train_participant_id"].astype(str))
    matched_test = set(pairs["test_participant_id"].astype(str))
    validation = set(assignments[assignments["temporal_split"].eq("validation")]["participant_id"].astype(str))
    split_map = {participant: "train" for participant in matched_train}
    split_map.update({participant: "validation" for participant in validation})
    split_map.update({participant: "test" for participant in matched_test})
    matched_features = features.copy()
    matched_features["split"] = matched_features["participant_id"].astype(str).map(split_map).fillna("unused")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_metric_rows: list[dict[str, object]] = []
    for modality in modalities:
        df = matched_features[
            matched_features["label_binary"].isin(["positive", "negative"])
            & matched_features["modality"].astype(str).eq(str(modality))
            & matched_features["split"].isin(REQUIRED_SPLITS)
        ].copy()
        if df.empty:
            continue
        cols = _feature_columns(df)
        train = df[df["split"].eq("train")].copy()
        validation_df = df[df["split"].eq("validation")].copy()
        test = df[df["split"].eq("test")].copy()
        if train.empty or validation_df.empty or test.empty or train["label_binary"].nunique() < 2:
            continue
        model = _make_model(model_name, random_state=random_state)
        model.fit(train[cols].fillna(0.0), labels_to_binary(train["label_binary"]))
        val_prob = _predict_prob(model, validation_df[cols].fillna(0.0))
        threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation_df["label_binary"]), val_prob)
        test_prob = _predict_prob(model, test[cols].fillna(0.0))
        metrics = binary_metric_bundle(labels_to_binary(test["label_binary"]), test_prob, threshold=threshold)
        metrics.update(
            {
                "evaluation_protocol": "matched_temporal_train_test",
                "analysis_family": "matched_audio_modality",
                "model_name": model_name,
                "modality": modality,
                "n_matched_pairs": float(len(pairs)),
                "n_train_participants": float(len(matched_train)),
                "n_test_participants": float(len(matched_test)),
                "threshold": threshold,
                "n_features": float(len(cols)),
            }
        )
        metric_rows.append(metrics)
        validation_metric_rows.append({"modality": modality, "auprc": metrics.get("auprc", np.nan)})
        pred = test[["recording_id", "participant_id", "label_binary", "modality", "split"]].copy()
        pred["probability"] = test_prob
        prediction_frames.append(pred)

    if prediction_frames:
        joined = pd.concat(prediction_frames, ignore_index=True, sort=False)
        pivot = joined.pivot_table(index=["participant_id", "label_binary"], columns="modality", values="probability", aggfunc="mean")
        available = [modality for modality in modalities if modality in pivot.columns]
        if len(available) >= 2:
            probability = pivot[available].mean(axis=1)
            labels = labels_to_binary(pd.Series(pivot.index.get_level_values("label_binary")))
            threshold = 0.5
            metrics = binary_metric_bundle(labels, probability.to_numpy(dtype=float), threshold=threshold)
            metrics.update(
                {
                    "evaluation_protocol": "matched_temporal_train_test",
                    "analysis_family": "matched_multimodal_fusion",
                    "model_name": model_name,
                    "modality": "multimodal",
                    "modality_combination": "+".join(sorted(available)),
                    "fusion_method": "uniform_mean",
                    "n_matched_pairs": float(len(pairs)),
                    "n_train_participants": float(len(matched_train)),
                    "n_test_participants": float(len(matched_test)),
                    "threshold": threshold,
                    "n_features": float("nan"),
                }
            )
            metric_rows.append(metrics)
    return pd.DataFrame(metric_rows)


def _selected_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    work = predictions[
        predictions.get("split", pd.Series(index=predictions.index, dtype=object)).astype(str).eq("test")
        & predictions.get("evaluation_protocol", pd.Series(index=predictions.index, dtype=object)).astype(str).isin(
            ["existing_participant_split", "time_stratified_participant_split", TEMPORAL_PROTOCOL]
        )
        & predictions.get("analysis_family", pd.Series(index=predictions.index, dtype=object)).astype(str).eq("multimodal_fusion")
        & predictions.get("modality_combination", pd.Series(index=predictions.index, dtype=object)).astype(str).eq(FULL_MULTIMODAL)
        & predictions.get("fusion_method", pd.Series(index=predictions.index, dtype=object)).astype(str).eq(UNIFORM_FUSION)
        & predictions.get("label_binary", pd.Series(index=predictions.index, dtype=object)).isin(["positive", "negative"])
    ].copy()
    work["probability"] = pd.to_numeric(work.get("probability"), errors="coerce")
    work["threshold"] = pd.to_numeric(work.get("threshold", 0.5), errors="coerce").fillna(0.5)
    return work[np.isfinite(work["probability"])].copy()


def build_failure_mode_summary(metadata: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_predictions(predictions)
    if selected.empty:
        return pd.DataFrame()
    participants = _participant_frame(metadata)[["participant_id", "recording_year_month", "age", "gender", "country", "symptom_count", "quality_flag"]]
    selected = selected.merge(participants, on="participant_id", how="left")
    y_true = labels_to_binary(selected["label_binary"])
    y_pred = (selected["probability"].to_numpy(dtype=float) >= selected["threshold"].to_numpy(dtype=float)).astype(int)
    selected["prediction_outcome"] = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )
    rows: list[dict[str, object]] = []
    group_cols = ["evaluation_protocol", "prediction_outcome"]
    for (protocol, outcome), group in selected.groupby(group_cols, dropna=False):
        rows.append(
            {
                "evaluation_protocol": protocol,
                "prediction_outcome": outcome,
                "n_rows": int(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "mean_age": float(pd.to_numeric(group["age"], errors="coerce").mean()),
                "mean_symptom_count": float(pd.to_numeric(group["symptom_count"], errors="coerce").mean()),
                "top_month": str(group["recording_year_month"].astype(str).mode().iloc[0]) if not group.empty else "",
                "top_country": str(group["country"].astype(str).mode().iloc[0]) if not group.empty else "",
                "top_quality_flag": str(group["quality_flag"].astype(str).mode().iloc[0]) if not group.empty else "",
            }
        )
    return pd.DataFrame(rows)


def build_uncertainty_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_predictions(predictions)
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for protocol, group in selected.groupby("evaluation_protocol", dropna=False):
        prob = group["probability"].to_numpy(dtype=float)
        threshold = group["threshold"].to_numpy(dtype=float)
        y_true = labels_to_binary(group["label_binary"])
        y_pred = (prob >= threshold).astype(int)
        confidence = np.maximum(prob, 1.0 - prob)
        error = y_true != y_pred
        rows.append(
            {
                "evaluation_protocol": protocol,
                "n_samples": int(len(group)),
                "mean_probability": float(np.mean(prob)),
                "mean_confidence": float(np.mean(confidence)),
                "high_confidence_share": float(np.mean(confidence >= 0.8)),
                "error_rate": float(np.mean(error)),
                "high_confidence_error_rate": float(np.mean(error[confidence >= 0.8])) if np.any(confidence >= 0.8) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_causal_dag_markdown(output: Path | None = None) -> str:
    text = """# Temporal Month Causal DAG\n\n```text\nCalendar month / collection period\n  |\n  |--> population prevalence and testing behavior --> observed COVID label\n  |\n  |--> symptom mix and healthcare-seeking behavior --> symptom metadata --> observed COVID label\n  |\n  |--> recruitment geography and demographics --> participant mix --> observed COVID label\n  |\n  |--> device, environment, prompt compliance, recording quality --> audio features\n  |\n  |--> shortcut variable used by source-domain models\n\nCOVID pathophysiology --> respiratory audio features --> desired COVID prediction signal\n```\n\nThe month variable is not interpreted as a biological cause of COVID audio changes. It is a proxy for changing prevalence, recruitment, symptom mix, and recording conditions. A model can therefore exploit month-linked structure during random participant splits, while the same shortcut becomes harmful under early-to-late evaluation.\n"""
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
    return text


def write_temporal_theory_markdown(
    month_label_shift: pd.DataFrame,
    month_ablation_effects: pd.DataFrame,
    matched_metrics: pd.DataFrame,
    output: Path | None = None,
) -> str:
    best_delta = float("nan")
    if not month_ablation_effects.empty:
        row = month_ablation_effects[month_ablation_effects["comparison_configuration"].astype(str).eq("Remove month")]
        if not row.empty:
            best_delta = _as_float(row.iloc[0].get("delta_auroc"))
    prevalence_min = _as_float(month_label_shift.get("positive_prevalence", pd.Series(dtype=float)).min()) if not month_label_shift.empty else float("nan")
    prevalence_max = _as_float(month_label_shift.get("positive_prevalence", pd.Series(dtype=float)).max()) if not month_label_shift.empty else float("nan")
    matched_text = "matched temporal evaluation was not estimable"
    if not matched_metrics.empty and "auroc" in matched_metrics.columns:
        fusion = matched_metrics[matched_metrics.get("analysis_family", pd.Series(dtype=object)).astype(str).eq("matched_multimodal_fusion")]
        if not fusion.empty:
            matched_text = f"matched multimodal temporal AUROC was {float(fusion.iloc[0]['auroc']):.3f}"
    text = "\n".join(
        [
            "# Temporal Shortcut Theory",
            "",
            f"Observed monthly positive prevalence ranges from {prevalence_min:.3f} to {prevalence_max:.3f} across available months.",
            f"Removing recording month from full safe metadata changes strict temporal AUROC by {best_delta:.3f}.",
            f"In the matched-cohort check, {matched_text}.",
            "",
            "Interpretation: recording month behaves as a collection-period shortcut. It reflects changing prevalence, symptom composition, recruitment geography, and recording conditions. This shortcut can support high apparent internal performance under participant-level random splitting, but it does not transfer cleanly to future months or external datasets.",
            "",
        ]
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
    return text


def run_temporal_month_causal_audit(
    metadata: pd.DataFrame,
    features: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    month_ablation: pd.DataFrame | None = None,
) -> TemporalMonthCausalAuditResult:
    features = features if features is not None else pd.DataFrame()
    predictions = predictions if predictions is not None else pd.DataFrame()
    month_ablation = month_ablation if month_ablation is not None else pd.DataFrame()
    month_label_shift = build_month_label_shift(metadata)
    month_covariate_shift = build_month_covariate_shift(metadata)
    matched_metrics = evaluate_matched_temporal_audio(metadata, features) if not features.empty else pd.DataFrame()
    failure_modes = build_failure_mode_summary(metadata, predictions)
    uncertainty = build_uncertainty_summary(predictions)
    effect_sizes = build_month_ablation_effect_sizes(month_ablation)
    dag = write_causal_dag_markdown()
    theory = write_temporal_theory_markdown(month_label_shift, effect_sizes, matched_metrics)
    return TemporalMonthCausalAuditResult(
        month_label_shift=month_label_shift,
        month_covariate_shift=month_covariate_shift,
        matched_cohort_metrics=matched_metrics,
        failure_modes=failure_modes,
        uncertainty_summary=uncertainty,
        month_ablation_effect_sizes=effect_sizes,
        causal_dag_markdown=dag,
        theory_markdown=theory,
    )
