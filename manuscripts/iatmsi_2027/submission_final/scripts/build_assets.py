from __future__ import annotations

from pathlib import Path
import sys
import tarfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_TABLES = REPO_ROOT / "reports" / "tables"
OUTPUT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
PREDICTION_PATH = REPO_ROOT / "data" / "outputs" / "metrics" / "compare_is10_final_validation_predictions.csv"
PREDICTION_BUNDLE = REPO_ROOT / "artifacts" / "bundles" / "covid_btp_final_doc_artifacts.tar.gz"
PREDICTION_MEMBER = "data/outputs/metrics/compare_is10_final_validation_predictions.csv"

SOURCE_TABLE = "compare_is10_final_validation_metrics"
PROTOCOL = "compare_is10_existing_participant_split"
FEATURE_STRATEGY = "compare_is10_top800_lightgbm"
SELECTED_K = 800
PRIMARY_FUSION = "uniform_mean"
INDEXED_PARTICIPANTS = 2746
RESOLVED_LABEL_PARTICIPANTS = 2114
COMPARE_FEATURES = 6373
IS10_FEATURES = 1582
PROJECT_FEATURES = 2177
OPENSMILE_EVENT_FEATURES = 8
CANDIDATE_FEATURES = COMPARE_FEATURES + IS10_FEATURES + PROJECT_FEATURES + OPENSMILE_EVENT_FEATURES
EXPECTED_PARTITIONS = {
    "train": (1460, 476, 984, 12685),
    "validation": (312, 101, 211, 2702),
    "test": (316, 103, 213, 2719),
}


def _one(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected one row for {description}, found {len(frame)}")
    return frame.iloc[0]


def _same_key(frame: pd.DataFrame, row: pd.Series, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in columns:
        value = row.get(column)
        mask &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
    return mask


def _metric(value: object) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Non-finite metric encountered: {value!r}")
    return number


def load_evidence() -> pd.DataFrame:
    raw = pd.read_csv(REPORT_TABLES / "paper_metric_table_raw.csv", low_memory=False)
    required = {
        "table_source",
        "evaluation_protocol",
        "feature_strategy",
        "selected_feature_k",
        "analysis_family",
        "model_name",
        "modality",
        "modality_combination",
        "fusion_method",
        "metric_split",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "f1",
        "n_samples",
        "threshold",
    }
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"paper_metric_table_raw.csv is missing: {sorted(missing)}")

    evidence = raw[
        raw["table_source"].eq(SOURCE_TABLE)
        & raw["evaluation_protocol"].eq(PROTOCOL)
        & raw["feature_strategy"].eq(FEATURE_STRATEGY)
        & pd.to_numeric(raw["selected_feature_k"], errors="coerce").eq(float(SELECTED_K))
        & raw["metric_split"].isin(["validation", "test"])
    ].copy()
    if evidence.empty:
        raise ValueError("No rows match the frozen conference evidence definition")
    for column in (
        "auroc",
        "auprc",
        "balanced_accuracy",
        "f1",
        "brier",
        "ece",
        "sensitivity",
        "specificity",
        "threshold",
        "n_samples",
    ):
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    return evidence


def select_modality_branches(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    key_columns = ["analysis_family", "model_name", "modality"]
    for label, modality in (("Breathing", "breath"), ("Cough", "cough"), ("Speech", "speech")):
        candidates = evidence[
            evidence["analysis_family"].eq("strong_audio_modality")
            & evidence["modality"].eq(modality)
            & evidence["metric_split"].eq("validation")
        ].sort_values(["auroc", "auprc", "model_name"], ascending=[False, False, True])
        selected = candidates.iloc[0]
        for split in ("validation", "test"):
            row = _one(
                evidence[
                    evidence["metric_split"].eq(split)
                    & _same_key(evidence, selected, key_columns)
                ],
                f"selected {modality} branch on {split}",
            )
            rows.append(
                {
                    "system": label,
                    "system_type": "single modality",
                    "configuration": selected["model_name"],
                    "split": split,
                    **{
                        column: row[column]
                        for column in (
                            "auroc",
                            "auprc",
                            "balanced_accuracy",
                            "f1",
                            "sensitivity",
                            "specificity",
                            "threshold",
                            "n_samples",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def validation_model_bank(evidence: pd.DataFrame) -> pd.DataFrame:
    models = {
        "lightgbm_smote_f80",
        "xgboost_smote_f80",
        "catboost_smote_f80",
        "svc_rbf_f60",
        "top_4_validation_ensemble",
    }
    modalities = {"breath", "cough", "speech"}
    rows = evidence[
        evidence["analysis_family"].eq("strong_audio_modality")
        & evidence["metric_split"].eq("validation")
        & evidence["model_name"].isin(models)
        & evidence["modality"].isin(modalities)
    ][
        [
            "modality",
            "model_name",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "n_samples",
        ]
    ].copy()
    if len(rows) != 15 or set(rows["model_name"]) != models or set(rows["modality"]) != modalities:
        raise ValueError("Validation model bank is incomplete or ambiguous")
    if rows.duplicated(["modality", "model_name"]).any():
        raise ValueError("Validation model bank contains duplicate modality/model rows")
    return rows.sort_values(["model_name", "modality"])


def write_validation_model_bank_table(rows: pd.DataFrame) -> None:
    indexed = rows.set_index(["model_name", "modality"])
    order = [
        ("LightGBM", "lightgbm_smote_f80"),
        ("XGBoost", "xgboost_smote_f80"),
        ("CatBoost", "catboost_smote_f80"),
        ("RBF-SVC", "svc_rbf_f60"),
        ("Four-model mean", "top_4_validation_ensemble"),
    ]
    selected = {
        "breath": "top_4_validation_ensemble",
        "cough": "top_4_validation_ensemble",
        "speech": "lightgbm_smote_f80",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Validation AUROC Across the Modality Candidate Bank}",
        r"\label{tab:model-bank}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Candidate & Breathing & Cough & Speech \\",
        r"\midrule",
    ]
    for label, model_name in order:
        values: list[str] = []
        for modality in ("breath", "cough", "speech"):
            value = _metric(indexed.loc[(model_name, modality), "auroc"])
            rendered = f"{value:.3f}"
            if selected[modality] == model_name:
                rendered = rf"\textbf{{{rendered}}}"
            values.append(rendered)
        lines.append(f"{label} & {' & '.join(values)} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{1pt}\scriptsize Bold indicates the validation-selected branch. The cough mean and SVC tie in AUROC to six decimals; AUPRC (0.752 versus 0.737) selects the mean.",
            r"\end{table}",
        ]
    )
    (TABLE_DIR / "validation_model_bank.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def select_fusion(evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    uniform = evidence[
        evidence["analysis_family"].eq("strong_multimodal_fusion")
        & evidence["fusion_method"].eq(PRIMARY_FUSION)
    ].copy()
    validation = uniform[uniform["metric_split"].eq("validation")].sort_values(
        ["auroc", "auprc", "modality_combination"],
        ascending=[False, False, True],
    )
    selected_validation = validation.iloc[0]
    if selected_validation["modality_combination"] != "cough+speech":
        raise ValueError("The validation-selected uniform fusion is no longer cough+speech")
    selected_test = _one(
        uniform[
            uniform["metric_split"].eq("test")
            & uniform["modality_combination"].eq(selected_validation["modality_combination"])
        ],
        "validation-selected uniform fusion test row",
    )

    combinations = uniform[
        [
            "modality_combination",
            "metric_split",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "n_samples",
            "threshold",
        ]
    ].sort_values(["metric_split", "auroc"], ascending=[True, False])
    return combinations, selected_validation, selected_test


def fusion_sensitivity_rows(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = evidence[
        evidence["analysis_family"].eq("strong_multimodal_fusion")
        & evidence["modality_combination"].eq("cough+speech")
        & evidence["metric_split"].isin(["validation", "test"])
    ][
        [
            "fusion_method",
            "metric_split",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "brier",
            "ece",
            "sensitivity",
            "specificity",
            "threshold",
            "n_samples",
        ]
    ].copy()
    expected = {"uniform_mean", "validation_weighted_auprc", "stacked_logistic_validation"}
    if set(rows["fusion_method"]) != expected or len(rows) != 6:
        raise ValueError("Unexpected cough+speech fusion sensitivity rows")
    return rows.sort_values(["metric_split", "fusion_method"])


def feature_level_fusion_comparator(evidence: pd.DataFrame) -> pd.DataFrame:
    candidates = evidence[
        evidence["analysis_family"].eq("strong_feature_level_fusion")
        & evidence["modality_combination"].eq("cough+speech")
        & evidence["metric_split"].eq("validation")
    ].sort_values(["auroc", "auprc", "model_name"], ascending=[False, False, True])
    selected = candidates.iloc[0]
    if selected["model_name"] != "xgboost_smote_f80":
        raise ValueError("Validation no longer selects XGBoost for feature-level cough+speech fusion")
    rows = evidence[
        evidence["analysis_family"].eq("strong_feature_level_fusion")
        & evidence["modality_combination"].eq("cough+speech")
        & evidence["model_name"].eq(selected["model_name"])
        & evidence["metric_split"].isin(["validation", "test"])
    ][
        [
            "model_name",
            "metric_split",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "brier",
            "ece",
            "n_samples",
        ]
    ].copy()
    if len(rows) != 2:
        raise ValueError("Feature-level cough+speech comparator is incomplete")
    return rows.sort_values("metric_split")


def multiseed_uniform_fusion() -> pd.DataFrame:
    raw = pd.read_csv(REPORT_TABLES / "paper_metric_table_raw.csv", low_memory=False)
    rows = raw[
        raw["table_source"].eq("compare_is10_multiseed_metrics")
        & raw["evaluation_protocol"].eq(PROTOCOL)
        & raw["feature_strategy"].eq(FEATURE_STRATEGY)
        & pd.to_numeric(raw["selected_feature_k"], errors="coerce").eq(float(SELECTED_K))
        & raw["analysis_family"].eq("strong_multimodal_fusion")
        & raw["model_name"].eq("strong_baseline_selected_fusion")
        & raw["modality_combination"].eq("cough+speech")
        & raw["fusion_method"].eq(PRIMARY_FUSION)
        & raw["metric_split"].eq("test")
    ][
        [
            "random_state",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "brier",
            "ece",
            "n_samples",
        ]
    ].copy()
    for column in rows.columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.sort_values("random_state")
    if len(rows) != 5 or rows["random_state"].astype(int).tolist() != [42, 43, 44, 45, 46]:
        raise ValueError("Expected exactly the five retained random-state runs 42--46")
    if not rows["n_samples"].eq(314).all():
        raise ValueError("Multi-seed internal test cohort is not fixed at 314 participants")
    return rows


def validate_cohort() -> pd.DataFrame:
    participant = pd.read_csv(REPORT_TABLES / "strong_baseline_participant_audit.csv")
    participant_counts = (
        participant.groupby(["split", "label_binary"], as_index=False)["n_participants"]
        .sum()
        .pivot(index="split", columns="label_binary", values="n_participants")
        .fillna(0)
    )
    protocol = pd.read_csv(REPORT_TABLES / "strong_baseline_protocol_audit.csv")
    recording_counts = (
        protocol[protocol["protocol_exclusion_reason"].eq("included")]
        .groupby("split")["n_rows"]
        .sum()
    )
    rows: list[dict[str, int | str]] = []
    for split, expected in EXPECTED_PARTITIONS.items():
        positive = int(participant_counts.loc[split, "positive"])
        negative = int(participant_counts.loc[split, "negative"])
        observed = (positive + negative, positive, negative, int(recording_counts.loc[split]))
        if observed != expected:
            raise ValueError(f"Unexpected {split} cohort: expected {expected}, found {observed}")
        rows.append(
            {
                "split": split,
                "n_participants": observed[0],
                "n_positive": observed[1],
                "n_negative": observed[2],
                "n_recordings": observed[3],
            }
        )
    cohort = pd.DataFrame(rows)
    if int(cohort["n_participants"].sum()) != 2088:
        raise ValueError("Quality-passing participant total is not 2,088")
    if not (INDEXED_PARTICIPANTS > RESOLVED_LABEL_PARTICIPANTS > int(cohort["n_participants"].sum())):
        raise ValueError("Participant-flow totals are not monotonically decreasing")
    return cohort


def selected_fusion_ci(selected_test: pd.Series, metric: str) -> pd.Series:
    bootstrap = pd.read_csv(REPORT_TABLES / "final_validation_bootstrap_ci.csv", low_memory=False)
    selected = bootstrap[
        bootstrap["prediction_source"].eq("compare_is10_final_validation_predictions")
        & bootstrap["evaluation_protocol"].eq(PROTOCOL)
        & bootstrap["feature_strategy"].eq(FEATURE_STRATEGY)
        & pd.to_numeric(bootstrap["selected_feature_k"], errors="coerce").eq(float(SELECTED_K))
        & bootstrap["analysis_family"].eq("strong_multimodal_fusion")
        & bootstrap["model_name"].eq("strong_baseline_selected_fusion")
        & bootstrap["modality_combination"].eq("cough+speech")
        & bootstrap["fusion_method"].eq(PRIMARY_FUSION)
        & bootstrap["metric_split"].eq("test")
        & bootstrap["metric"].eq(metric)
    ]
    row = _one(selected, f"primary fusion participant-bootstrap {metric.upper()} interval")
    if not np.isclose(_metric(row["point"]), _metric(selected_test[metric])):
        raise ValueError(f"Bootstrap point estimate disagrees with the primary test {metric.upper()}")
    if int(row["n_samples"]) != 314 or int(row["n_bootstraps"]) != 1000:
        raise ValueError("Unexpected bootstrap sample or replicate count")
    return row


def _load_final_predictions() -> pd.DataFrame:
    if PREDICTION_PATH.exists():
        predictions = pd.read_csv(PREDICTION_PATH, low_memory=False)
    elif PREDICTION_BUNDLE.exists():
        with tarfile.open(PREDICTION_BUNDLE, mode="r:gz") as archive:
            handle = archive.extractfile(PREDICTION_MEMBER)
            if handle is None:
                raise FileNotFoundError(f"{PREDICTION_MEMBER} is missing from {PREDICTION_BUNDLE}")
            predictions = pd.read_csv(handle, low_memory=False)
    else:
        raise FileNotFoundError(
            "Paired comparison requires compare_is10_final_validation_predictions.csv "
            "or the frozen final-artifact bundle"
        )
    required = {
        "participant_id",
        "label_binary",
        "split",
        "probability",
        "analysis_family",
        "model_name",
        "modality",
        "modality_combination",
        "fusion_method",
        "evaluation_protocol",
        "feature_strategy",
        "selected_feature_k",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Final prediction artifact is missing: {sorted(missing)}")
    return predictions


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_branch_comparisons(branches: pd.DataFrame, selected_test: pd.Series) -> pd.DataFrame:
    package_src = REPO_ROOT / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    from covid_rars.model_comparison import paired_bootstrap_difference
    from covid_rars.reviewer_extension_checks import paired_delong_auc_comparison

    predictions = _load_final_predictions()
    base = (
        predictions["evaluation_protocol"].eq(PROTOCOL)
        & predictions["feature_strategy"].eq(FEATURE_STRATEGY)
        & pd.to_numeric(predictions["selected_feature_k"], errors="coerce").eq(float(SELECTED_K))
        & predictions["split"].eq("test")
        & predictions["label_binary"].isin(["positive", "negative"])
    )
    fusion = predictions[
        base
        & predictions["analysis_family"].eq("strong_multimodal_fusion")
        & predictions["model_name"].eq("strong_baseline_selected_fusion")
        & predictions["modality_combination"].eq("cough+speech")
        & predictions["fusion_method"].eq(PRIMARY_FUSION)
    ][["participant_id", "label_binary", "probability"]].copy()
    if len(fusion) != int(selected_test["n_samples"]) or fusion["participant_id"].nunique() != len(fusion):
        raise ValueError("Primary fusion predictions are not one row per expected test participant")

    selected_models = (
        branches[branches["split"].eq("test")]
        .set_index("system")["configuration"]
        .to_dict()
    )
    result_rows: list[dict[str, object]] = []
    for display_name, modality in (("Cough", "cough"), ("Speech", "speech")):
        model_name = str(selected_models[display_name])
        branch_rows = predictions[
            base
            & predictions["analysis_family"].eq("strong_audio_modality")
            & predictions["modality"].eq(modality)
            & predictions["model_name"].eq(model_name)
        ][["participant_id", "label_binary", "probability"]].copy()
        branch = (
            branch_rows.groupby(["participant_id", "label_binary"], as_index=False)["probability"]
            .mean()
            .rename(columns={"probability": "branch_probability"})
        )
        aligned = fusion.rename(columns={"probability": "fusion_probability"}).merge(
            branch,
            on=["participant_id", "label_binary"],
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != 314 or aligned["label_binary"].nunique() != 2:
            raise ValueError(f"Expected 314 paired two-class rows for fusion versus {modality}")
        y_true = aligned["label_binary"].eq("positive").astype(int).to_numpy()
        fusion_probability = aligned["fusion_probability"].to_numpy(dtype=float)
        branch_probability = aligned["branch_probability"].to_numpy(dtype=float)
        delong = paired_delong_auc_comparison(y_true, fusion_probability, branch_probability)
        if bool(delong.get("skipped", False)):
            raise ValueError(f"Paired DeLong was skipped for fusion versus {modality}")
        result_rows.append(
            {
                "comparison": f"fusion_minus_{modality}",
                "baseline_system": display_name,
                "candidate_system": "Cough+speech mean",
                "metric": "auroc",
                "n_paired": int(delong["n_paired"]),
                "n_positive": int(delong["n_positive"]),
                "n_negative": int(delong["n_negative"]),
                "baseline_value": float(delong["right_auc"]),
                "candidate_value": float(delong["left_auc"]),
                "difference": float(delong["delta"]),
                "ci_low": float(delong["delta_ci_low"]),
                "ci_high": float(delong["delta_ci_high"]),
                "p_value": float(delong["p_value"]),
                "method": "paired_delong",
                "n_resamples": 0,
            }
        )

        long_predictions = pd.concat(
            [
                aligned[["participant_id", "label_binary"]]
                .assign(system=modality, probability=branch_probability),
                aligned[["participant_id", "label_binary"]]
                .assign(system="fusion", probability=fusion_probability),
            ],
            ignore_index=True,
        )
        bootstrap = paired_bootstrap_difference(
            long_predictions,
            baseline_name=modality,
            candidate_name="fusion",
            model_column="system",
            id_column="participant_id",
            metric="auprc",
            n_bootstraps=2000,
            random_state=20270808,
        )
        result_rows.append(
            {
                "comparison": f"fusion_minus_{modality}",
                "baseline_system": display_name,
                "candidate_system": "Cough+speech mean",
                "metric": "auprc",
                "n_paired": int(bootstrap["n_matched"]),
                "n_positive": int(y_true.sum()),
                "n_negative": int(len(y_true) - y_true.sum()),
                "baseline_value": float(bootstrap["baseline_value"]),
                "candidate_value": float(bootstrap["candidate_value"]),
                "difference": float(bootstrap["difference"]),
                "ci_low": float(bootstrap["ci_low"]),
                "ci_high": float(bootstrap["ci_high"]),
                "p_value": float(bootstrap["p_two_sided_bootstrap"]),
                "method": "paired_participant_bootstrap",
                "n_resamples": int(bootstrap["n_bootstraps"]),
            }
        )

    result = pd.DataFrame(result_rows)
    result["p_value_holm"] = np.nan
    for metric, indices in result.groupby("metric").groups.items():
        adjusted = _holm_adjust(result.loc[indices, "p_value"].astype(float).tolist())
        result.loc[indices, "p_value_holm"] = adjusted
    return result


def heldout_auroc_intervals(
    branches: pd.DataFrame,
    selected_test: pd.Series,
    feature_level: pd.DataFrame,
    primary_ci: pd.Series,
) -> pd.DataFrame:
    package_src = REPO_ROOT / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    from covid_rars.statistics import bootstrap_metric_ci

    predictions = _load_final_predictions()
    base = (
        predictions["evaluation_protocol"].eq(PROTOCOL)
        & predictions["feature_strategy"].eq(FEATURE_STRATEGY)
        & pd.to_numeric(predictions["selected_feature_k"], errors="coerce").eq(float(SELECTED_K))
        & predictions["split"].eq("test")
        & predictions["label_binary"].isin(["positive", "negative"])
    )
    selected_models = (
        branches[branches["split"].eq("test")]
        .set_index("system")["configuration"]
        .to_dict()
    )
    expected_points = (
        branches[branches["split"].eq("test")]
        .set_index("system")["auroc"]
        .astype(float)
        .to_dict()
    )

    systems: list[tuple[str, pd.DataFrame, float]] = []
    for system, modality in (("Breathing", "breath"), ("Cough", "cough"), ("Speech", "speech")):
        rows = predictions[
            base
            & predictions["analysis_family"].eq("strong_audio_modality")
            & predictions["modality"].eq(modality)
            & predictions["model_name"].eq(str(selected_models[system]))
        ][["participant_id", "label_binary", "probability"]].copy()
        pooled = rows.groupby(["participant_id", "label_binary"], as_index=False)["probability"].mean()
        systems.append((system, pooled, float(expected_points[system])))

    feature_test = _one(feature_level[feature_level["metric_split"].eq("test")], "feature-level test interval row")
    feature_predictions = predictions[
        base
        & predictions["analysis_family"].eq("strong_feature_level_fusion")
        & predictions["model_name"].eq(str(feature_test["model_name"]))
        & predictions["modality_combination"].eq("cough+speech")
    ][["participant_id", "label_binary", "probability"]].copy()
    systems.append(("Feature concat.", feature_predictions, _metric(feature_test["auroc"])))

    fusion_predictions = predictions[
        base
        & predictions["analysis_family"].eq("strong_multimodal_fusion")
        & predictions["model_name"].eq("strong_baseline_selected_fusion")
        & predictions["modality_combination"].eq("cough+speech")
        & predictions["fusion_method"].eq(PRIMARY_FUSION)
    ][["participant_id", "label_binary", "probability"]].copy()
    systems.append(("Score fusion", fusion_predictions, _metric(selected_test["auroc"])))

    output: list[dict[str, object]] = []
    for system, frame, expected_point in systems:
        if frame.empty or frame["participant_id"].duplicated().any() or frame["label_binary"].nunique() != 2:
            raise ValueError(f"Invalid participant-level prediction rows for {system}")
        interval = bootstrap_metric_ci(
            frame["label_binary"].eq("positive").astype(int).to_numpy(),
            frame["probability"].astype(float).to_numpy(),
            metric="auroc",
            n_bootstraps=1000,
            random_state=42,
        )
        if not np.isclose(float(interval["point"]), expected_point):
            raise ValueError(f"Participant-level AUROC disagrees with the frozen {system} result")
        interval["system"] = system
        output.append(interval)

    result = pd.DataFrame(output)
    fusion_interval = _one(result[result["system"].eq("Score fusion")], "recomputed score-fusion interval")
    for column in ("point", "ci_low", "ci_high"):
        if not np.isclose(float(fusion_interval[column]), _metric(primary_ci[column])):
            raise ValueError(f"Recomputed score-fusion {column} disagrees with the frozen bootstrap record")
    return result


def feature_selection_record() -> pd.DataFrame:
    summary = pd.read_csv(REPORT_TABLES / "sota_compare_is10_feature_selection_summary.csv")
    selected = summary[pd.to_numeric(summary["k"], errors="coerce").eq(float(SELECTED_K))].copy()
    row = _one(selected, "top-800 feature-selection record")
    if (
        int(row["n_selected_features"]) != SELECTED_K
        or row["ranker"] != "lightgbm"
        or row["selection_split"] != "train"
        or row["selection_scope"] != "per_modality_mean"
    ):
        raise ValueError("The top-800 selection record does not match the manuscript protocol")
    return selected


def _tex_number(value: float, digits: int = 3) -> str:
    rounded = round(float(value), digits)
    if abs(rounded) < 0.5 * (10.0 ** (-digits)):
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def write_values(
    branches: pd.DataFrame,
    combinations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    feature_level: pd.DataFrame,
    multiseed: pd.DataFrame,
    cohort: pd.DataFrame,
    feature_record: pd.DataFrame,
    selected_validation: pd.Series,
    selected_test: pd.Series,
    auroc_ci: pd.Series,
    auprc_ci: pd.Series,
    paired: pd.DataFrame,
) -> None:
    if CANDIDATE_FEATURES != 10140:
        raise ValueError(f"Unexpected candidate feature total: {CANDIDATE_FEATURES}")
    test_branches = branches[branches["split"].eq("test")].set_index("system")
    val_branches = branches[branches["split"].eq("validation")].set_index("system")
    weighted_test = _one(
        sensitivity[
            sensitivity["metric_split"].eq("test")
            & sensitivity["fusion_method"].eq("validation_weighted_auprc")
        ],
        "weighted fusion test",
    )
    stack_test = _one(
        sensitivity[
            sensitivity["metric_split"].eq("test")
            & sensitivity["fusion_method"].eq("stacked_logistic_validation")
        ],
        "stacked fusion test",
    )
    uniform_test = _one(
        sensitivity[
            sensitivity["metric_split"].eq("test")
            & sensitivity["fusion_method"].eq("uniform_mean")
        ],
        "uniform fusion test sensitivity row",
    )
    feature_validation = _one(
        feature_level[feature_level["metric_split"].eq("validation")],
        "feature-level fusion validation comparator",
    )
    feature_test = _one(
        feature_level[feature_level["metric_split"].eq("test")],
        "feature-level fusion test comparator",
    )
    all_val = _one(
        combinations[
            combinations["metric_split"].eq("validation")
            & combinations["modality_combination"].eq("cough+breath+speech")
        ],
        "three-modality validation row",
    )
    combination_rows = combinations.set_index(["modality_combination", "metric_split"])
    raw_weights = np.asarray(
        [max(_metric(val_branches.loc[name, "auprc"]) - 0.5, 0.01) for name in ("Cough", "Speech")]
    )
    weights = raw_weights / raw_weights.sum()
    cohort_by_split = cohort.set_index("split")
    selected_names = str(_one(feature_record, "top-800 feature-selection record")["selected_features"]).split(";")
    if len(selected_names) != SELECTED_K or len(set(selected_names)) != SELECTED_K:
        raise ValueError("Selected feature list does not contain 800 unique names")
    selected_active = sum("__event_" in name for name in selected_names)
    selected_compare = sum(name.startswith("compare2016__") and "__event_" not in name for name in selected_names)
    selected_is10 = sum(name.startswith("is10__") and "__event_" not in name for name in selected_names)
    selected_project = sum(name.startswith("strong__") and "__event_" not in name for name in selected_names)
    selected_counts = [selected_compare, selected_is10, selected_project, selected_active]
    if selected_active != OPENSMILE_EVENT_FEATURES or sum(selected_counts) != SELECTED_K:
        raise ValueError(f"Unexpected selected feature composition: {selected_counts}")

    def paired_row(comparison: str, metric: str) -> pd.Series:
        return _one(
            paired[paired["comparison"].eq(comparison) & paired["metric"].eq(metric)],
            f"{comparison} {metric} paired comparison",
        )

    cough_auc = paired_row("fusion_minus_cough", "auroc")
    cough_auprc = paired_row("fusion_minus_cough", "auprc")
    speech_auc = paired_row("fusion_minus_speech", "auroc")
    speech_auprc = paired_row("fusion_minus_speech", "auprc")

    values = {
        "BreathValidationAUROC": _tex_number(val_branches.loc["Breathing", "auroc"]),
        "CoughValidationAUROC": _tex_number(val_branches.loc["Cough", "auroc"]),
        "SpeechValidationAUROC": _tex_number(val_branches.loc["Speech", "auroc"]),
        "BreathTestAUROC": _tex_number(test_branches.loc["Breathing", "auroc"]),
        "CoughTestAUROC": _tex_number(test_branches.loc["Cough", "auroc"]),
        "SpeechTestAUROC": _tex_number(test_branches.loc["Speech", "auroc"]),
        "SpeechTestAUPRC": _tex_number(test_branches.loc["Speech", "auprc"]),
        "SpeechTestBalancedAccuracy": _tex_number(test_branches.loc["Speech", "balanced_accuracy"]),
        "SpeechTestFOne": _tex_number(test_branches.loc["Speech", "f1"]),
        "FusionValidationAUROC": _tex_number(selected_validation["auroc"]),
        "FusionValidationAUPRC": _tex_number(selected_validation["auprc"]),
        "AllModalitiesValidationAUROC": _tex_number(all_val["auroc"]),
        "AllModalitiesValidationAUPRC": _tex_number(all_val["auprc"]),
        "AllModalitiesTestAUROC": _tex_number(combination_rows.loc[("cough+breath+speech", "test"), "auroc"]),
        "AllModalitiesTestAUPRC": _tex_number(combination_rows.loc[("cough+breath+speech", "test"), "auprc"]),
        "BreathSpeechValidationAUROC": _tex_number(combination_rows.loc[("breath+speech", "validation"), "auroc"]),
        "BreathSpeechValidationAUPRC": _tex_number(combination_rows.loc[("breath+speech", "validation"), "auprc"]),
        "BreathSpeechTestAUROC": _tex_number(combination_rows.loc[("breath+speech", "test"), "auroc"]),
        "BreathSpeechTestAUPRC": _tex_number(combination_rows.loc[("breath+speech", "test"), "auprc"]),
        "CoughBreathValidationAUROC": _tex_number(combination_rows.loc[("cough+breath", "validation"), "auroc"]),
        "CoughBreathValidationAUPRC": _tex_number(combination_rows.loc[("cough+breath", "validation"), "auprc"]),
        "CoughBreathTestAUROC": _tex_number(combination_rows.loc[("cough+breath", "test"), "auroc"]),
        "CoughBreathTestAUPRC": _tex_number(combination_rows.loc[("cough+breath", "test"), "auprc"]),
        "FusionTestAUROC": _tex_number(selected_test["auroc"]),
        "FusionTestAUPRC": _tex_number(selected_test["auprc"]),
        "FusionTestBalancedAccuracy": _tex_number(selected_test["balanced_accuracy"]),
        "FusionTestFOne": _tex_number(selected_test["f1"]),
        "FusionTestSensitivity": _tex_number(selected_test["sensitivity"]),
        "FusionTestSpecificity": _tex_number(selected_test["specificity"]),
        "FusionTestN": str(int(selected_test["n_samples"])),
        "FusionThreshold": _tex_number(selected_test["threshold"]),
        "FusionAUCILow": _tex_number(auroc_ci["ci_low"]),
        "FusionAUCIHigh": _tex_number(auroc_ci["ci_high"]),
        "FusionAUPRCILow": _tex_number(auprc_ci["ci_low"]),
        "FusionAUPRCIHigh": _tex_number(auprc_ci["ci_high"]),
        "WeightedTestAUROC": _tex_number(weighted_test["auroc"]),
        "StackTestAUROC": _tex_number(stack_test["auroc"]),
        "UniformTestBrier": _tex_number(uniform_test["brier"]),
        "UniformTestECE": _tex_number(uniform_test["ece"]),
        "WeightedTestBrier": _tex_number(weighted_test["brier"]),
        "WeightedTestECE": _tex_number(weighted_test["ece"]),
        "StackTestBrier": _tex_number(stack_test["brier"]),
        "StackTestECE": _tex_number(stack_test["ece"]),
        "FeatureFusionValidationAUROC": _tex_number(feature_validation["auroc"]),
        "FeatureFusionValidationAUPRC": _tex_number(feature_validation["auprc"]),
        "FeatureFusionTestAUROC": _tex_number(feature_test["auroc"]),
        "FeatureFusionTestAUPRC": _tex_number(feature_test["auprc"]),
        "ScoreMinusFeatureAUROC": _tex_number(_metric(selected_test["auroc"]) - _metric(feature_test["auroc"])),
        "ScoreMinusFeatureAUPRC": _tex_number(_metric(selected_test["auprc"]) - _metric(feature_test["auprc"])),
        "MultiSeedAUROCMean": _tex_number(multiseed["auroc"].mean(), 4),
        "MultiSeedAUROCSD": _tex_number(multiseed["auroc"].std(ddof=1), 4),
        "MultiSeedAUROCLow": _tex_number(multiseed["auroc"].min(), 4),
        "MultiSeedAUROCHigh": _tex_number(multiseed["auroc"].max(), 4),
        "MultiSeedAUPRCMean": _tex_number(multiseed["auprc"].mean(), 4),
        "MultiSeedAUPRCSD": _tex_number(multiseed["auprc"].std(ddof=1), 4),
        "CoughFusionWeight": _tex_number(weights[0]),
        "SpeechFusionWeight": _tex_number(weights[1]),
        "FusionMinusCoughAUROC": _tex_number(cough_auc["difference"]),
        "FusionMinusCoughAUCILow": _tex_number(cough_auc["ci_low"]),
        "FusionMinusCoughAUCIHigh": _tex_number(cough_auc["ci_high"]),
        "FusionMinusCoughAUROCPValue": _tex_number(cough_auc["p_value"]),
        "FusionMinusCoughAUROCHolmPValue": _tex_number(cough_auc["p_value_holm"]),
        "FusionMinusCoughAUPRC": _tex_number(cough_auprc["difference"]),
        "FusionMinusCoughAUPRCILow": _tex_number(cough_auprc["ci_low"]),
        "FusionMinusCoughAUPRCIHigh": _tex_number(cough_auprc["ci_high"]),
        "FusionMinusSpeechAUROC": _tex_number(speech_auc["difference"]),
        "FusionMinusSpeechAUCILow": _tex_number(speech_auc["ci_low"]),
        "FusionMinusSpeechAUCIHigh": _tex_number(speech_auc["ci_high"]),
        "FusionMinusSpeechAUROCPValue": _tex_number(speech_auc["p_value"]),
        "FusionMinusSpeechAUPRC": _tex_number(speech_auprc["difference"]),
        "FusionMinusSpeechAUPRCILow": _tex_number(speech_auprc["ci_low"]),
        "FusionMinusSpeechAUPRCIHigh": _tex_number(speech_auprc["ci_high"]),
        "CandidateFeatureCount": f"{CANDIDATE_FEATURES:,}".replace(",", "{,}"),
        "CompareFeatureCount": f"{COMPARE_FEATURES:,}".replace(",", "{,}"),
        "ISFeatureCount": f"{IS10_FEATURES:,}".replace(",", "{,}"),
        "ProjectFeatureCount": f"{PROJECT_FEATURES:,}".replace(",", "{,}"),
        "OpenSmileEventFeatureCount": str(OPENSMILE_EVENT_FEATURES),
        "SelectedCompareFeatureCount": str(selected_compare),
        "SelectedISFeatureCount": str(selected_is10),
        "SelectedProjectFeatureCount": str(selected_project),
        "SelectedEventFeatureCount": str(selected_active),
        "SelectedFeatureCount": str(SELECTED_K),
        "IndexedParticipants": f"{INDEXED_PARTICIPANTS:,}".replace(",", "{,}"),
        "ResolvedParticipants": f"{RESOLVED_LABEL_PARTICIPANTS:,}".replace(",", "{,}"),
        "QualityParticipants": f"{int(cohort['n_participants'].sum()):,}".replace(",", "{,}"),
        "TrainParticipants": f"{int(cohort_by_split.loc['train', 'n_participants']):,}".replace(",", "{,}"),
        "ValidationParticipants": str(int(cohort_by_split.loc["validation", "n_participants"])),
        "TestParticipants": str(int(cohort_by_split.loc["test", "n_participants"])),
        "TrainPositive": str(int(cohort_by_split.loc["train", "n_positive"])),
        "ValidationPositive": str(int(cohort_by_split.loc["validation", "n_positive"])),
        "TestPositive": str(int(cohort_by_split.loc["test", "n_positive"])),
        "TrainNegative": str(int(cohort_by_split.loc["train", "n_negative"])),
        "ValidationNegative": str(int(cohort_by_split.loc["validation", "n_negative"])),
        "TestNegative": str(int(cohort_by_split.loc["test", "n_negative"])),
        "TrainRecordings": f"{int(cohort_by_split.loc['train', 'n_recordings']):,}".replace(",", "{,}"),
        "ValidationRecordings": f"{int(cohort_by_split.loc['validation', 'n_recordings']):,}".replace(",", "{,}"),
        "TestRecordings": f"{int(cohort_by_split.loc['test', 'n_recordings']):,}".replace(",", "{,}"),
    }
    lines = ["% Generated by scripts/build_assets.py. Do not edit by hand."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    (OUTPUT_ROOT / "manuscript_values.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def write_claim_ledger(
    selected_test: pd.Series,
    auroc_ci: pd.Series,
    auprc_ci: pd.Series,
    paired: pd.DataFrame,
    feature_level: pd.DataFrame,
    multiseed: pd.DataFrame,
) -> None:
    cough_auc = _one(
        paired[paired["comparison"].eq("fusion_minus_cough") & paired["metric"].eq("auroc")],
        "fusion-minus-cough AUROC evidence",
    )
    speech_auc = _one(
        paired[paired["comparison"].eq("fusion_minus_speech") & paired["metric"].eq("auroc")],
        "fusion-minus-speech AUROC evidence",
    )
    feature_test = _one(
        feature_level[feature_level["metric_split"].eq("test")],
        "feature-level fusion test evidence",
    )
    rows = [
        {
            "claim_id": "conference_estimand",
            "displayed_value": "participant-disjoint Coswara source evaluation",
            "evidence_file": "paper_metric_table_raw.csv",
            "row_selector": f"table_source={SOURCE_TABLE};evaluation_protocol={PROTOCOL}",
            "boundary": "Internal source-domain evaluation; no temporal, external, or clinical-validity claim.",
        },
        {
            "claim_id": "primary_fusion_auroc",
            "displayed_value": _tex_number(selected_test["auroc"]),
            "evidence_file": "paper_metric_table_raw.csv",
            "row_selector": "strong_multimodal_fusion;cough+speech;uniform_mean;test",
            "boundary": "Validation-selected fixed uniform fusion on 314 complete-case test participants.",
        },
        {
            "claim_id": "primary_fusion_auroc_ci",
            "displayed_value": f"[{_tex_number(auroc_ci['ci_low'])}, {_tex_number(auroc_ci['ci_high'])}]",
            "evidence_file": "final_validation_bootstrap_ci.csv",
            "row_selector": "compare_is10_final_validation_predictions;cough+speech;uniform_mean;test;auroc",
            "boundary": "Participant-resampling uncertainty for a fixed test cohort; excludes model-refit uncertainty.",
        },
        {
            "claim_id": "primary_fusion_auprc_ci",
            "displayed_value": f"[{_tex_number(auprc_ci['ci_low'])}, {_tex_number(auprc_ci['ci_high'])}]",
            "evidence_file": "final_validation_bootstrap_ci.csv",
            "row_selector": "compare_is10_final_validation_predictions;cough+speech;uniform_mean;test;auprc",
            "boundary": "Participant-resampling uncertainty for a fixed test cohort; excludes model-refit uncertainty.",
        },
        {
            "claim_id": "feature_selection",
            "displayed_value": "top 800 of 10,140 numeric candidates",
            "evidence_file": "sota_compare_is10_feature_selection_summary.csv and feature construction code",
            "row_selector": "k=800;ranker=lightgbm;selection_split=train;selection_scope=per_modality_mean",
            "boundary": "The 800-feature budget was one of 500, 800, and 1,200 explored during development.",
        },
        {
            "claim_id": "fusion_minus_cough_paired",
            "displayed_value": (
                f"+{_tex_number(cough_auc['difference'])} AUROC "
                f"[{_tex_number(cough_auc['ci_low'])}, {_tex_number(cough_auc['ci_high'])}]"
            ),
            "evidence_file": "compare_is10_final_validation_predictions.csv",
            "row_selector": "paired participants;uniform cough+speech minus selected cough;test",
            "boundary": "Paired DeLong comparison on the 314 complete-case participants; fixed predictions only.",
        },
        {
            "claim_id": "fusion_minus_speech",
            "displayed_value": (
                f"+{_tex_number(speech_auc['difference'])} AUROC "
                f"[{_tex_number(speech_auc['ci_low'])}, {_tex_number(speech_auc['ci_high'])}]"
            ),
            "evidence_file": "compare_is10_final_validation_predictions.csv",
            "row_selector": "paired participants;uniform cough+speech minus selected speech;test",
            "boundary": "Paired DeLong CI crosses zero; do not claim superiority over speech.",
        },
        {
            "claim_id": "score_level_vs_feature_level",
            "displayed_value": (
                f"+{_tex_number(_metric(selected_test['auroc']) - _metric(feature_test['auroc']))} AUROC; "
                f"+{_tex_number(_metric(selected_test['auprc']) - _metric(feature_test['auprc']))} AUPRC"
            ),
            "evidence_file": "paper_metric_table_raw.csv",
            "row_selector": "validation-selected XGBoost feature concatenation versus uniform score fusion; cough+speech; test",
            "boundary": "Descriptive held-out ablation; paired uncertainty was not computed for this contrast.",
        },
        {
            "claim_id": "multiseed_workflow_stability",
            "displayed_value": (
                f"AUROC {_tex_number(multiseed['auroc'].mean(), 4)} +/- "
                f"{_tex_number(multiseed['auroc'].std(ddof=1), 4)}"
            ),
            "evidence_file": "paper_metric_table_raw.csv",
            "row_selector": "compare_is10_multiseed_metrics;random_state=42--46;existing participant split;cough+speech;uniform_mean;test",
            "boundary": "Same 314-participant test cohort; SD captures fitting and validation-selection variability, not population sampling.",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "claim_evidence_ledger.csv", index=False)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )


def build_study_design_figure(cohort: pd.DataFrame) -> None:
    _set_plot_style()
    figure = plt.figure(figsize=(7.08, 2.12))
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=[4.25, 0.95],
        hspace=0.16,
        left=0.015,
        right=0.995,
        top=0.90,
        bottom=0.06,
    )
    stage_grid = outer[0].subgridspec(
        1,
        5,
        width_ratios=[1.05, 1.25, 1.25, 1.15, 0.88],
        wspace=0.22,
    )

    ink = "#222222"
    accent = "#116B80"
    muted = "#777777"
    rule = "#C9C9C9"
    stage_titles = [
        "Prompted recordings",
        "Acoustic representation",
        "Modality branches",
        "Validation selection",
        "Held-out test",
    ]
    stage_axes: list[plt.Axes] = []
    for index, title in enumerate(stage_titles):
        stage = figure.add_subplot(stage_grid[0, index])
        stage.set_xlim(0, 1)
        stage.set_ylim(0, 1)
        stage.set_xticks([])
        stage.set_yticks([])
        stage.set_title(title, fontsize=7.4, weight="bold", pad=4)
        for side in ("top", "bottom", "left"):
            stage.spines[side].set_visible(False)
        stage.spines["right"].set_visible(index < len(stage_titles) - 1)
        stage.spines["right"].set_color(rule)
        stage.spines["right"].set_linewidth(0.55)
        stage_axes.append(stage)

    prompt, feature, branches, selection, locked = stage_axes

    # Schematic waveforms identify the three prompted tasks. All artists are
    # clipped to this stage, so labels cannot enter the feature panel.
    waveform_x = np.linspace(0.43, 0.94, 180)
    phase = np.linspace(0, 1, waveform_x.size)
    waveforms = [
        (0.73, "Cough", np.sin(20 * np.pi * phase) * np.exp(-3.2 * phase)),
        (0.52, "Breathing", 0.65 * np.sin(5 * np.pi * phase)),
        (0.31, "Speech", 0.55 * np.sin(25 * np.pi * phase) + 0.2 * np.sin(8 * np.pi * phase)),
    ]
    for center, label, signal in waveforms:
        prompt.text(0.02, center, label, ha="left", va="center", fontsize=6.9, clip_on=True)
        prompt.plot(waveform_x, center + 0.045 * signal, color=ink, linewidth=0.75, clip_on=True)
    prompt.text(0.48, 0.09, "16 kHz\nquality screened", ha="center", va="center", fontsize=6.1, color=muted, linespacing=1.05, clip_on=True)

    feature.text(0.50, 0.79, "10,140", ha="center", va="center", fontsize=11.8, weight="bold", color=accent, clip_on=True)
    feature.text(0.50, 0.67, "recording-level variables", ha="center", va="center", fontsize=6.6, clip_on=True)
    feature.text(0.50, 0.49, "ComParE 6,373\nIS10 1,582", ha="center", va="center", fontsize=6.4, linespacing=1.25, clip_on=True)
    feature.text(0.50, 0.30, "Signal/timing 2,185", ha="center", va="center", fontsize=6.4, clip_on=True)
    feature.plot([0.16, 0.84], [0.21, 0.21], color=rule, linewidth=0.6, clip_on=True)
    feature.text(0.50, 0.09, "Training-only ranking\n800 retained", ha="center", va="center", fontsize=6.2, color=muted, linespacing=1.05, clip_on=True)

    branches.text(0.04, 0.84, "Sound", ha="left", va="center", fontsize=6.1, color=muted, clip_on=True)
    branches.text(0.96, 0.84, "Selected branch", ha="right", va="center", fontsize=6.1, color=muted, clip_on=True)
    for y, modality, model, color in [
        (0.65, "Cough", "Mean of four", accent),
        (0.46, "Breathing", "Mean of four", muted),
        (0.27, "Speech", "LightGBM", accent),
    ]:
        branches.text(0.04, y, modality, ha="left", va="center", fontsize=6.7, color=color, clip_on=True)
        branches.text(0.96, y, model, ha="right", va="center", fontsize=6.5, color=color, clip_on=True)
    branches.text(0.50, 0.08, "Participant-level pooling", ha="center", va="center", fontsize=6.1, color=muted, clip_on=True)

    selection.text(0.50, 0.80, "Selected", ha="center", va="center", fontsize=6.1, color=muted, clip_on=True)
    selection.text(0.50, 0.66, "Cough + speech", ha="center", va="center", fontsize=7.2, weight="bold", color=accent, clip_on=True)
    selection.text(0.50, 0.46, r"$p=(p_c+p_s)/2$", ha="center", va="center", fontsize=7.8, clip_on=True)
    selection.text(0.50, 0.25, "Threshold 0.325", ha="center", va="center", fontsize=6.4, clip_on=True)

    locked.text(0.50, 0.79, "n = 314", ha="center", va="center", fontsize=6.7, color=muted, clip_on=True)
    locked.text(0.50, 0.61, "AUROC", ha="center", va="center", fontsize=6.4, clip_on=True)
    locked.text(0.50, 0.47, "0.895", ha="center", va="center", fontsize=12.5, weight="bold", color=accent, clip_on=True)
    locked.text(0.50, 0.23, "AUPRC 0.862", ha="center", va="center", fontsize=6.5, clip_on=True)

    role_grid = outer[1].subgridspec(1, 4, width_ratios=[0.72, 2.80, 1.45, 1.05], wspace=0.03)
    role_specs = [
        ("Data use", "", "white"),
        ("TRAIN", "rank | fit | calibrate", "#E4F0F4"),
        ("VALIDATION", "select system | threshold", "#F2ECD9"),
        ("TEST", "estimate | compare | CI", "#ECECEC"),
    ]
    for index, (label, operations, facecolor) in enumerate(role_specs):
        role = figure.add_subplot(role_grid[0, index])
        role.set_facecolor(facecolor)
        role.set_xticks([])
        role.set_yticks([])
        for spine in role.spines.values():
            spine.set_visible(False)
        role.text(0.50, 0.68 if operations else 0.50, label, ha="center", va="center", fontsize=6.6, weight="bold", clip_on=True)
        if operations:
            role.text(0.50, 0.27, operations, ha="center", va="center", fontsize=6.0, clip_on=True)

    # Draw arrows only in the gutters between clipped stages.
    figure.canvas.draw()
    for left_axis, right_axis in zip(stage_axes[:-1], stage_axes[1:]):
        left_box = left_axis.get_position()
        right_box = right_axis.get_position()
        y = left_box.y0 + 0.49 * left_box.height
        figure.add_artist(
            FancyArrowPatch(
                (left_box.x1 + 0.003, y),
                (right_box.x0 - 0.003, y),
                transform=figure.transFigure,
                arrowstyle="-|>",
                mutation_scale=6.0,
                linewidth=0.85,
                color=accent,
                clip_on=False,
            )
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(FIGURE_DIR / f"study_design.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def build_results_figure(
    branches: pd.DataFrame,
    combinations: pd.DataFrame,
    selected_test: pd.Series,
    feature_level: pd.DataFrame,
    paired: pd.DataFrame,
    heldout_intervals: pd.DataFrame,
) -> None:
    _set_plot_style()
    figure = plt.figure(figsize=(7.08, 2.34), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.06, 1.10, 0.86], wspace=0.20)
    accent = "#116B80"
    ink = "#2A2A2A"
    grid_color = "#D4D4D4"

    selection = figure.add_subplot(grid[0, 0])
    validation = combinations[combinations["metric_split"].eq("validation")].copy()
    display_names = {
        "cough+speech": "Cough + speech",
        "cough+breath+speech": "Cough + breathing + speech",
        "breath+speech": "Breathing + speech",
        "cough+breath": "Cough + breathing",
    }
    order = ["cough+breath", "breath+speech", "cough+breath+speech", "cough+speech"]
    validation = validation.set_index("modality_combination").loc[order]
    y_left = np.arange(len(order))
    selection.hlines(y_left, 0.81, validation["auroc"], color=grid_color, linewidth=1.0, zorder=1)
    selection.scatter(validation["auroc"].iloc[:-1], y_left[:-1], marker="o", s=22, color="#777777", zorder=3)
    selection.scatter(validation["auroc"].iloc[-1], y_left[-1], marker="o", s=26, color=accent, zorder=4)
    selection.set_yticks(y_left, [display_names[name] for name in order])
    selection.get_yticklabels()[-1].set_weight("bold")
    selection.set_xlim(0.81, 0.85)
    selection.set_ylim(-0.48, 3.48)
    selection.set_xticks([0.81, 0.82, 0.83, 0.84, 0.85])
    selection.set_xlabel("Validation AUROC")
    selection.set_title("(a) Sound-set ranking", loc="left", fontsize=8.0, weight="bold")
    selection.grid(axis="x", color=grid_color, linewidth=0.45)
    selection.spines[["top", "right", "left"]].set_visible(False)
    selection.tick_params(axis="y", length=0)

    result = figure.add_subplot(grid[0, 1])
    labels = ["Breathing", "Cough", "Speech", "Feature concat.", "Score fusion"]
    intervals = heldout_intervals.set_index("system").loc[labels]
    y = np.arange(5)[::-1]
    for index, (system, row) in enumerate(intervals.iterrows()):
        point = _metric(row["point"])
        low = _metric(row["ci_low"])
        high = _metric(row["ci_high"])
        color = accent if system == "Score fusion" else "#666666"
        result.errorbar(
            point,
            y[index],
            xerr=np.asarray([[point - low], [high - point]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.2 if system == "Score fusion" else 3.8,
            elinewidth=1.25 if system == "Score fusion" else 1.0,
            capsize=2.0,
            zorder=3,
        )
    result.set_yticks(y, labels)
    result.get_yticklabels()[-1].set_weight("bold")
    result.set_xlim(0.75, 0.95)
    result.set_ylim(-0.48, 4.48)
    result.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95])
    result.set_xlabel("Held-out AUROC (95% CI)")
    result.set_title("(b) Held-out discrimination", loc="left", fontsize=8.0, weight="bold")
    result.grid(axis="x", color=grid_color, linewidth=0.45)
    result.spines[["top", "right", "left"]].set_visible(False)
    result.tick_params(axis="y", length=0)

    effect = figure.add_subplot(grid[0, 2])
    effect_rows = []
    for comparison, label in (("fusion_minus_cough", "Fusion - cough"), ("fusion_minus_speech", "Fusion - speech")):
        row = _one(
            paired[paired["comparison"].eq(comparison) & paired["metric"].eq("auroc")],
            f"{comparison} AUROC effect figure row",
        )
        effect_rows.append((label, row))
    effect_y = np.asarray([1, 0])
    for y_value, (label, row) in zip(effect_y, effect_rows):
        point = _metric(row["difference"])
        low = _metric(row["ci_low"])
        high = _metric(row["ci_high"])
        effect.errorbar(
            point,
            y_value,
            xerr=np.asarray([[point - low], [high - point]]),
            fmt="o",
            color=accent,
            ecolor=accent,
            markersize=4.2,
            elinewidth=1.25,
            capsize=2.2,
            zorder=3,
        )
    effect.axvline(0.0, color=ink, linewidth=0.7, linestyle="--", zorder=1)
    effect.set_yticks(effect_y, [label for label, _ in effect_rows])
    effect.set_xlim(-0.02, 0.065)
    effect.set_ylim(-0.58, 1.50)
    effect.set_xticks([-0.02, 0.00, 0.03, 0.06])
    effect.set_xlabel(r"Paired $\Delta$AUROC")
    effect.set_title("(c) Matched effects", loc="left", fontsize=8.0, weight="bold")
    effect.grid(axis="x", color=grid_color, linewidth=0.45)
    effect.spines[["top", "right", "left"]].set_visible(False)
    effect.tick_params(axis="y", length=0)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(FIGURE_DIR / f"selection_and_results.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    branches = select_modality_branches(evidence)
    model_bank = validation_model_bank(evidence)
    combinations, selected_validation, selected_test = select_fusion(evidence)
    sensitivity = fusion_sensitivity_rows(evidence)
    feature_level = feature_level_fusion_comparator(evidence)
    multiseed = multiseed_uniform_fusion()
    cohort = validate_cohort()
    auroc_ci = selected_fusion_ci(selected_test, "auroc")
    auprc_ci = selected_fusion_ci(selected_test, "auprc")
    feature_record = feature_selection_record()
    paired = paired_branch_comparisons(branches, selected_test)
    heldout_intervals = heldout_auroc_intervals(branches, selected_test, feature_level, auroc_ci)

    branches.to_csv(TABLE_DIR / "selected_modality_branches.csv", index=False)
    model_bank.to_csv(TABLE_DIR / "validation_model_bank.csv", index=False)
    write_validation_model_bank_table(model_bank)
    combinations.to_csv(TABLE_DIR / "uniform_fusion_ablation.csv", index=False)
    sensitivity.to_csv(TABLE_DIR / "cough_speech_fusion_sensitivity.csv", index=False)
    feature_level.to_csv(TABLE_DIR / "feature_level_fusion_comparator.csv", index=False)
    multiseed.to_csv(TABLE_DIR / "multiseed_uniform_fusion.csv", index=False)
    cohort.to_csv(TABLE_DIR / "cohort_partition_record.csv", index=False)
    pd.DataFrame([auroc_ci]).to_csv(TABLE_DIR / "selected_auroc_bootstrap_record.csv", index=False)
    pd.DataFrame([auprc_ci]).to_csv(TABLE_DIR / "selected_auprc_bootstrap_record.csv", index=False)
    paired.to_csv(TABLE_DIR / "paired_fusion_branch_comparisons.csv", index=False)
    heldout_intervals.to_csv(TABLE_DIR / "heldout_auroc_intervals.csv", index=False)
    feature_record.to_csv(TABLE_DIR / "feature_selection_record.csv", index=False)
    pd.DataFrame(
        [
            {
                "selection_split": "validation",
                "modality_combination": selected_validation["modality_combination"],
                "fusion_method": selected_validation["fusion_method"],
                "validation_auroc": selected_validation["auroc"],
                "validation_auprc": selected_validation["auprc"],
                "test_auroc": selected_test["auroc"],
                "test_auprc": selected_test["auprc"],
            }
        ]
    ).to_csv(TABLE_DIR / "validation_selection_record.csv", index=False)

    write_values(
        branches,
        combinations,
        sensitivity,
        feature_level,
        multiseed,
        cohort,
        feature_record,
        selected_validation,
        selected_test,
        auroc_ci,
        auprc_ci,
        paired,
    )
    write_claim_ledger(selected_test, auroc_ci, auprc_ci, paired, feature_level, multiseed)
    build_study_design_figure(cohort)
    build_results_figure(branches, combinations, selected_test, feature_level, paired, heldout_intervals)
    print(f"Validation-selected uniform fusion AUROC: {_metric(selected_test['auroc']):.6f}")
    print(
        "Bootstrap AUROC 95% CI: "
        f"[{_metric(auroc_ci['ci_low']):.6f}, {_metric(auroc_ci['ci_high']):.6f}]"
    )
    print(
        "Bootstrap AUPRC 95% CI: "
        f"[{_metric(auprc_ci['ci_low']):.6f}, {_metric(auprc_ci['ci_high']):.6f}]"
    )
    cough_delta = _one(
        paired[paired["comparison"].eq("fusion_minus_cough") & paired["metric"].eq("auroc")],
        "fusion-minus-cough AUROC output",
    )
    print(
        "Paired fusion-minus-cough AUROC: "
        f"{_metric(cough_delta['difference']):.6f} "
        f"[{_metric(cough_delta['ci_low']):.6f}, {_metric(cough_delta['ci_high']):.6f}]"
    )
    print(f"Wrote evidence-audited conference assets under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
