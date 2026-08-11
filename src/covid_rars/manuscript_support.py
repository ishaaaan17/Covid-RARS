from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.metadata_confounding import (
    build_audit_feature_frame,
    feature_group_for_column,
)
from covid_rars.metrics import labels_to_binary


@dataclass(frozen=True)
class ManuscriptSupportResult:
    metadata_shap: pd.DataFrame
    ipw_residual_smd: pd.DataFrame
    auprc_lift: pd.DataFrame
    unknown_label_summary: pd.DataFrame
    unknown_label_balance: pd.DataFrame


def _align_feature_frames(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    columns = sorted(set().union(*(frame.columns for frame in frames))) if frames else []
    return [frame.reindex(columns=columns, fill_value=0.0) for frame in frames], columns


def linear_metadata_shap_table(
    metadata: pd.DataFrame,
    *,
    feature_set: str = "demographic_protocol_only",
    split: str = "test",
    top_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank metadata drivers with exact linear-logit attributions.

    The metadata confounding audit uses a StandardScaler followed by logistic
    regression. For this additive linear model, each split-row contribution is
    simply standardized_feature_value * coefficient in logit space. Aggregating
    absolute contributions gives a deterministic SHAP-style ranking without
    adding the optional external ``shap`` dependency.
    """
    df = metadata[
        metadata["label_binary"].isin(["positive", "negative"])
        & metadata["split"].isin(["train", "validation", "test"])
    ].copy()
    train = df[df["split"].eq("train")].copy()
    target = df[df["split"].eq(split)].copy()
    if train.empty or target.empty:
        raise ValueError("Need non-empty train and target split rows for metadata attribution")

    train_raw, train_groups = build_audit_feature_frame(train, feature_set=feature_set)
    target_raw, target_groups = build_audit_feature_frame(target, feature_set=feature_set)
    (train_x, target_x), columns = _align_feature_frames([train_raw, target_raw])
    varying_columns = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
    if not varying_columns:
        raise ValueError(f"Feature set {feature_set} selected no train-varying columns")
    train_x = train_x[varying_columns]
    target_x = target_x[varying_columns]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ]
    )
    model.fit(train_x, labels_to_binary(train["label_binary"]))
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    target_z = scaler.transform(target_x)
    coefficients = classifier.coef_[0]
    contributions = target_z * coefficients
    groups = {**target_groups, **train_groups}

    table = pd.DataFrame(
        {
            "audit_model": feature_set,
            "split": split,
            "feature": varying_columns,
            "feature_group": [groups.get(col, feature_group_for_column(col)) for col in varying_columns],
            "coefficient": coefficients,
            "mean_shap": contributions.mean(axis=0),
            "mean_abs_shap": np.abs(contributions).mean(axis=0),
            "max_abs_shap": np.abs(contributions).max(axis=0),
            "positive_label_share": (contributions > 0).mean(axis=0),
        }
    )
    table["direction_by_mean_shap"] = np.where(table["mean_shap"] >= 0, "positive_label", "negative_label")
    table = table.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    if top_n is not None:
        table = table.head(top_n).reset_index(drop=True)
    return table


def _severity(abs_smd: float) -> str:
    if abs_smd >= 0.5:
        return "severe_residual_imbalance"
    if abs_smd >= 0.25:
        return "moderate_residual_imbalance"
    if abs_smd >= 0.1:
        return "minor_residual_imbalance"
    return "well_balanced"


def ipw_residual_smd_table(
    balance: pd.DataFrame,
    *,
    weight_config: str = "ipw_cap_2_q_0.95",
    top_n: int | None = None,
) -> pd.DataFrame:
    if balance.empty:
        return pd.DataFrame()
    frame = balance.copy()
    if "weight_config" in frame.columns:
        frame = frame[frame["weight_config"].astype(str).eq(weight_config)].copy()
    if "control_method" in frame.columns:
        controlled = frame[frame["control_method"].astype(str).eq("ipw_label_propensity")].copy()
        if not controlled.empty:
            frame = controlled
    if frame.empty:
        return frame
    frame["before_abs_smd"] = pd.to_numeric(frame.get("before_abs_smd", np.nan), errors="coerce")
    frame["after_abs_smd"] = pd.to_numeric(frame["after_abs_smd"], errors="coerce")
    frame["smd_reduction"] = frame["before_abs_smd"] - frame["after_abs_smd"]
    frame["balance_severity"] = frame["after_abs_smd"].fillna(0.0).map(_severity)
    frame = frame.sort_values(["after_abs_smd", "feature"], ascending=[False, True]).reset_index(drop=True)
    frame["rank_after_weighting"] = np.arange(1, len(frame) + 1)
    if top_n is not None:
        frame = frame.head(top_n).reset_index(drop=True)
    return frame


def auprc_lift_over_prevalence_table(
    metrics_by_representation: dict[str, pd.DataFrame],
    *,
    target_prevalence: float,
    select_by: str = "auroc",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for representation, metrics in metrics_by_representation.items():
        if metrics.empty:
            continue
        frame = metrics.copy()
        frame["auroc"] = pd.to_numeric(frame["auroc"], errors="coerce")
        frame["auprc"] = pd.to_numeric(frame["auprc"], errors="coerce")
        sort_col = select_by if select_by in frame.columns else "auroc"
        row = frame.sort_values(sort_col, ascending=False).iloc[0]
        auprc = float(row["auprc"])
        prevalence = float(target_prevalence)
        absolute_lift = auprc - prevalence
        relative_lift = auprc / prevalence if prevalence > 0 else np.nan
        rows.append(
            {
                "representation": representation,
                "model_name": row.get("model_name", ""),
                "feature_strategy": row.get("feature_strategy", ""),
                "selection_metric": sort_col,
                "auroc": float(row.get("auroc", np.nan)),
                "auprc": auprc,
                "target_prevalence": prevalence,
                "absolute_auprc_lift": round(float(absolute_lift), 6),
                "relative_auprc_lift": round(float(relative_lift), 6) if np.isfinite(relative_lift) else np.nan,
                "pr_lift_interpretation": "near_prevalence" if absolute_lift < 0.003 else "limited_lift",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("auprc", ascending=False).reset_index(drop=True)


def _top_value(series: pd.Series) -> tuple[str, float]:
    clean = series.fillna("unknown").astype(str)
    if clean.empty:
        return "", 0.0
    counts = clean.value_counts(normalize=True)
    return str(counts.index[0]), float(counts.iloc[0])


def _numeric_smd(known: pd.Series, unknown: pd.Series) -> float:
    known_num = pd.to_numeric(known, errors="coerce").dropna()
    unknown_num = pd.to_numeric(unknown, errors="coerce").dropna()
    if known_num.empty or unknown_num.empty:
        return np.nan
    pooled = np.sqrt((known_num.var(ddof=0) + unknown_num.var(ddof=0)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((unknown_num.mean() - known_num.mean()) / pooled)


def _categorical_differences(metadata: pd.DataFrame, column: str) -> pd.DataFrame:
    known = metadata[metadata["label_availability"].eq("known")][column].fillna("unknown").astype(str)
    unknown = metadata[metadata["label_availability"].eq("unknown")][column].fillna("unknown").astype(str)
    values = sorted(set(known.unique()).union(set(unknown.unique())))
    rows = []
    for value in values:
        known_prop = float((known == value).mean()) if len(known) else np.nan
        unknown_prop = float((unknown == value).mean()) if len(unknown) else np.nan
        difference = unknown_prop - known_prop
        rows.append(
            {
                "feature": f"{column}_{value}",
                "comparison_type": "categorical_proportion_difference",
                "known_value": known_prop,
                "unknown_value": unknown_prop,
                "difference_unknown_minus_known": difference,
                "abs_difference": abs(difference),
            }
        )
    return pd.DataFrame(rows)


def unknown_label_audit_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = metadata.copy()
    frame["label_availability"] = np.where(frame["label_binary"].isin(["positive", "negative"]), "known", "unknown")
    if "recording_date" in frame.columns:
        date = pd.to_datetime(frame["recording_date"], errors="coerce")
        frame["recording_year"] = date.dt.year
        frame["recording_month"] = date.dt.month

    summary_rows = []
    for label_availability, group in frame.groupby("label_availability", dropna=False):
        top_country, top_country_share = _top_value(group.get("country", pd.Series(dtype=object)))
        top_quality, top_quality_share = _top_value(group.get("quality_flag", pd.Series(dtype=object)))
        top_gender, top_gender_share = _top_value(group.get("gender", pd.Series(dtype=object)))
        summary_rows.append(
            {
                "label_availability": label_availability,
                "n_rows": int(len(group)),
                "n_participants": int(group["participant_id"].nunique()) if "participant_id" in group.columns else int(len(group)),
                "age_mean": float(pd.to_numeric(group.get("age", pd.Series(dtype=float)), errors="coerce").mean()),
                "age_median": float(pd.to_numeric(group.get("age", pd.Series(dtype=float)), errors="coerce").median()),
                "duration_sec_mean": float(pd.to_numeric(group.get("duration_sec", pd.Series(dtype=float)), errors="coerce").mean()),
                "recording_year_min": float(pd.to_numeric(group.get("recording_year", pd.Series(dtype=float)), errors="coerce").min()),
                "recording_year_max": float(pd.to_numeric(group.get("recording_year", pd.Series(dtype=float)), errors="coerce").max()),
                "top_country": top_country,
                "top_country_share": top_country_share,
                "top_quality_flag": top_quality,
                "top_quality_flag_share": top_quality_share,
                "top_gender": top_gender,
                "top_gender_share": top_gender_share,
            }
        )
    summary = pd.DataFrame(summary_rows)

    known = frame[frame["label_availability"].eq("known")]
    unknown = frame[frame["label_availability"].eq("unknown")]
    balance_rows: list[dict[str, object]] = []
    for col in ["age", "duration_sec", "sample_rate_original", "recording_year", "recording_month"]:
        if col not in frame.columns:
            continue
        known_mean = pd.to_numeric(known[col], errors="coerce").mean()
        unknown_mean = pd.to_numeric(unknown[col], errors="coerce").mean()
        smd = _numeric_smd(known[col], unknown[col])
        balance_rows.append(
            {
                "feature": col,
                "comparison_type": "numeric_smd_unknown_vs_known",
                "known_value": float(known_mean),
                "unknown_value": float(unknown_mean),
                "difference_unknown_minus_known": float(unknown_mean - known_mean),
                "abs_difference": abs(smd) if np.isfinite(smd) else np.nan,
                "smd_unknown_vs_known": smd,
            }
        )

    categorical_frames = [pd.DataFrame(balance_rows)]
    for col in ["gender", "country", "quality_flag"]:
        if col in frame.columns:
            categorical_frames.append(_categorical_differences(frame, col))
    balance = pd.concat(categorical_frames, ignore_index=True, sort=False) if categorical_frames else pd.DataFrame()
    if not balance.empty:
        balance = balance.sort_values(["abs_difference", "feature"], ascending=[False, True]).reset_index(drop=True)
    return summary, balance


def load_external_metric_tables(project_root: Path) -> dict[str, pd.DataFrame]:
    metrics_dir = project_root / "data" / "outputs" / "metrics"
    paths = {
        "mfcc": metrics_dir / "external_model_grid_metrics.csv",
        "opensmile_egemaps": metrics_dir / "external_model_grid_opensmile_egemaps_metrics.csv",
        "beats": metrics_dir / "external_model_grid_beats_metrics.csv",
        "panns": metrics_dir / "external_model_grid_panns_metrics.csv",
    }
    return {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}


def infer_target_prevalence(project_root: Path) -> float:
    candidates = [
        project_root / "data" / "outputs" / "metrics" / "external_model_grid_beats_predictions.csv",
        project_root / "data" / "outputs" / "metrics" / "external_model_grid_predictions.csv",
        project_root / "data" / "processed" / "features_beats_coughvid_cough.csv",
        project_root / "data" / "processed" / "features_panns_coughvid_cough.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda col: col in {"recording_id", "label_binary"})
        if "label_binary" not in frame.columns:
            continue
        if "recording_id" in frame.columns:
            frame = frame.drop_duplicates("recording_id")
        labels = frame["label_binary"]
        labels = labels[labels.isin(["positive", "negative"])]
        if not labels.empty:
            return float((labels == "positive").mean())
    raise FileNotFoundError("Could not infer external target prevalence from predictions or COUGHVID feature files")


def run_manuscript_support_analyses(project_root: Path) -> ManuscriptSupportResult:
    metadata = pd.read_csv(project_root / "data" / "processed" / "metadata_clean.csv")
    metadata_shap = linear_metadata_shap_table(metadata, top_n=None)
    balance_path = project_root / "reports" / "tables" / "ipw_sensitivity_balance.csv"
    ipw_balance = pd.read_csv(balance_path) if balance_path.exists() else pd.DataFrame()
    ipw_residual = ipw_residual_smd_table(ipw_balance)
    target_prevalence = infer_target_prevalence(project_root)
    auprc_lift = auprc_lift_over_prevalence_table(
        load_external_metric_tables(project_root),
        target_prevalence=target_prevalence,
    )
    unknown_summary, unknown_balance = unknown_label_audit_tables(metadata)
    return ManuscriptSupportResult(
        metadata_shap=metadata_shap,
        ipw_residual_smd=ipw_residual,
        auprc_lift=auprc_lift,
        unknown_label_summary=unknown_summary,
        unknown_label_balance=unknown_balance,
    )
