from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_TABLES = REPO_ROOT / "reports" / "tables"
OUTPUT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"

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


def selected_fusion_ci(selected_test: pd.Series) -> pd.Series:
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
        & bootstrap["metric"].eq("auroc")
    ]
    row = _one(selected, "primary fusion participant-bootstrap AUROC interval")
    if not np.isclose(_metric(row["point"]), _metric(selected_test["auroc"])):
        raise ValueError("Bootstrap point estimate disagrees with the primary test AUROC")
    if int(row["n_samples"]) != 314 or int(row["n_bootstraps"]) != 1000:
        raise ValueError("Unexpected bootstrap sample or replicate count")
    return row


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
    return f"{float(value):.{digits}f}"


def write_values(
    branches: pd.DataFrame,
    combinations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cohort: pd.DataFrame,
    selected_validation: pd.Series,
    selected_test: pd.Series,
    ci: pd.Series,
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
        "FusionAUCILow": _tex_number(ci["ci_low"]),
        "FusionAUCIHigh": _tex_number(ci["ci_high"]),
        "WeightedTestAUROC": _tex_number(weighted_test["auroc"]),
        "StackTestAUROC": _tex_number(stack_test["auroc"]),
        "CoughFusionWeight": _tex_number(weights[0]),
        "SpeechFusionWeight": _tex_number(weights[1]),
        "FusionMinusSpeechAUROC": _tex_number(_metric(selected_test["auroc"]) - _metric(test_branches.loc["Speech", "auroc"])),
        "FusionMinusSpeechAUPRC": _tex_number(_metric(selected_test["auprc"]) - _metric(test_branches.loc["Speech", "auprc"])),
        "CandidateFeatureCount": f"{CANDIDATE_FEATURES:,}".replace(",", "{,}"),
        "CompareFeatureCount": f"{COMPARE_FEATURES:,}".replace(",", "{,}"),
        "ISFeatureCount": f"{IS10_FEATURES:,}".replace(",", "{,}"),
        "ProjectFeatureCount": f"{PROJECT_FEATURES:,}".replace(",", "{,}"),
        "OpenSmileEventFeatureCount": str(OPENSMILE_EVENT_FEATURES),
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


def write_claim_ledger(selected_test: pd.Series, ci: pd.Series) -> None:
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
            "displayed_value": f"[{_tex_number(ci['ci_low'])}, {_tex_number(ci['ci_high'])}]",
            "evidence_file": "final_validation_bootstrap_ci.csv",
            "row_selector": "compare_is10_final_validation_predictions;cough+speech;uniform_mean;test;auroc",
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
            "claim_id": "fusion_minus_speech",
            "displayed_value": "descriptive +0.007 AUROC",
            "evidence_file": "paper_metric_table_raw.csv",
            "row_selector": "uniform cough+speech test minus selected speech test",
            "boundary": "No paired CI or superiority test is available; do not call the difference significant.",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "claim_evidence_ledger.csv", index=False)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _role_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    facecolor: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=facecolor,
            edgecolor="#2F3B43",
            linewidth=0.85,
        )
    )
    ax.text(x + width / 2, y + height * 0.69, title, ha="center", va="center", fontsize=6.6, weight="bold")
    ax.text(x + width / 2, y + height * 0.31, detail, ha="center", va="center", fontsize=5.45, linespacing=1.08)


def _horizontal_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=0.9,
            color="#4A555C",
            shrinkA=1,
            shrinkB=1,
        )
    )


def build_study_design_figure(cohort: pd.DataFrame) -> None:
    _set_plot_style()
    figure, ax = plt.subplots(figsize=(7.08, 2.30))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_boxes = [
        (0.02, "Indexed resource", f"{INDEXED_PARTICIPANTS:,} participants"),
        (0.37, "Resolved binary labels", f"{RESOLVED_LABEL_PARTICIPANTS:,} participants"),
        (0.72, "Quality-passing audio", f"{int(cohort['n_participants'].sum()):,} participants"),
    ]
    for x, title, detail in top_boxes:
        _role_box(ax, x, 0.68, 0.26, 0.20, title, detail, "#F2F4F5")
    _horizontal_arrow(ax, (0.285, 0.78), (0.365, 0.78))
    _horizontal_arrow(ax, (0.635, 0.78), (0.715, 0.78))
    ax.text(0.325, 0.655, "632 unresolved labels", ha="center", va="center", fontsize=5.25, color="#4A555C")
    ax.text(0.675, 0.655, "26 without eligible audio", ha="center", va="center", fontsize=5.25, color="#4A555C")

    cohort_by_split = cohort.set_index("split")
    roles = [
        (
            0.04,
            "TRAIN",
            f"n={int(cohort_by_split.loc['train', 'n_participants']):,}\nrank features; fit models\nSMOTE within training only",
            "#DDEBF4",
        ),
        (
            0.375,
            "VALIDATION",
            f"n={int(cohort_by_split.loc['validation', 'n_participants']):,}\nselect branches and fusion\nfreeze decision threshold",
            "#E5F0E6",
        ),
        (
            0.71,
            "TEST",
            f"n={int(cohort_by_split.loc['test', 'n_participants']):,}\none locked evaluation\nparticipant-level metrics",
            "#F2F4F5",
        ),
    ]
    split_y = 0.53
    source_x = 0.85
    ax.plot([source_x, source_x], [0.68, split_y], color="#4A555C", linewidth=0.85)
    ax.plot([0.165, source_x], [split_y, split_y], color="#4A555C", linewidth=0.85)
    for x, title, detail, color in roles:
        center = x + 0.125
        ax.add_patch(
            FancyArrowPatch(
                (center, split_y),
                (center, 0.40),
                arrowstyle="-|>",
                mutation_scale=7.5,
                linewidth=0.85,
                color="#4A555C",
            )
        )
        _role_box(ax, x, 0.08, 0.25, 0.31, title, detail, color)
    ax.text(0.50, 0.015, "All recordings from a participant remain in the same partition", ha="center", fontsize=5.8)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(FIGURE_DIR / f"study_design.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def build_results_figure(
    branches: pd.DataFrame,
    combinations: pd.DataFrame,
    selected_test: pd.Series,
    ci: pd.Series,
) -> None:
    _set_plot_style()
    figure = plt.figure(figsize=(7.08, 2.65), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.17)

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
    selection.axhspan(2.58, 3.42, color="#EAF3F8", zorder=0)
    selection.scatter(validation["auroc"], y_left + 0.10, marker="o", s=23, color="#0072B2", label="AUROC", zorder=3)
    selection.scatter(
        validation["auprc"],
        y_left - 0.10,
        marker="s",
        s=19,
        facecolor="white",
        edgecolor="#333333",
        linewidth=0.9,
        label="AUPRC",
        zorder=3,
    )
    for yi, (_, row) in zip(y_left, validation.iterrows()):
        selection.text(float(row["auroc"]) + 0.002, yi + 0.10, f"{float(row['auroc']):.3f}", va="center", fontsize=6.1, color="#005A8C")
        pr = float(row["auprc"])
        pr_offset = 0.002 if pr < 0.785 else -0.002
        pr_align = "left" if pr < 0.785 else "right"
        selection.text(pr + pr_offset, yi - 0.10, f"{pr:.3f}", va="center", ha=pr_align, fontsize=6.0, color="#333333")
    selection.set_yticks(y_left, [display_names[name] for name in order])
    selection.get_yticklabels()[-1].set_weight("bold")
    selection.set_xlim(0.765, 0.852)
    selection.set_ylim(-0.55, 3.55)
    selection.set_xlabel("Validation performance")
    selection.set_title("(a) Modality-set selection", loc="left", fontsize=7.2, weight="bold")
    selection.grid(axis="x", color="#D8D8D8", linewidth=0.5)
    selection.spines[["top", "right", "left"]].set_visible(False)
    selection.tick_params(axis="y", length=0)
    selection.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)

    result = figure.add_subplot(grid[0, 1])
    selected = branches[branches["split"].eq("test")].set_index("system")
    order = ["Breathing", "Cough", "Speech"]
    labels = ["Breathing", "Cough", "Speech", "Cough+speech\nmean"]
    auroc = np.asarray([_metric(selected.loc[name, "auroc"]) for name in order] + [_metric(selected_test["auroc"])])
    auprc = np.asarray([_metric(selected.loc[name, "auprc"]) for name in order] + [_metric(selected_test["auprc"])])
    y = np.arange(4)[::-1]
    result.hlines(
        y[-1] + 0.10,
        _metric(ci["ci_low"]),
        _metric(ci["ci_high"]),
        color="#0072B2",
        linewidth=1.3,
        zorder=2,
    )
    result.scatter(auroc, y + 0.10, marker="o", s=23, color="#0072B2", label="AUROC", zorder=3)
    result.scatter(
        auprc,
        y - 0.10,
        marker="s",
        s=19,
        facecolor="white",
        edgecolor="#333333",
        linewidth=0.9,
        label="AUPRC",
        zorder=3,
    )
    for yi, roc, pr in zip(y, auroc, auprc):
        result.text(
            roc + 0.004,
            yi + 0.10,
            f"{roc:.3f}",
            va="center",
            fontsize=6.5,
            color="#005A8C",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35},
        )
        result.text(pr - 0.004, yi - 0.10, f"{pr:.3f}", va="center", ha="right", fontsize=6.3, color="#333333")
    result.set_yticks(y, labels)
    result.set_xlim(0.68, 0.945)
    result.set_ylim(-0.60, 3.40)
    result.set_xlabel("Participant-level held-out performance")
    result.set_title("(b) Selected systems", loc="left", fontsize=7.2, weight="bold")
    result.grid(axis="x", color="#D8D8D8", linewidth=0.5)
    result.spines[["top", "right", "left"]].set_visible(False)
    result.tick_params(axis="y", length=0)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(FIGURE_DIR / f"selection_and_results.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    branches = select_modality_branches(evidence)
    combinations, selected_validation, selected_test = select_fusion(evidence)
    sensitivity = fusion_sensitivity_rows(evidence)
    cohort = validate_cohort()
    ci = selected_fusion_ci(selected_test)
    feature_record = feature_selection_record()

    branches.to_csv(TABLE_DIR / "selected_modality_branches.csv", index=False)
    combinations.to_csv(TABLE_DIR / "uniform_fusion_ablation.csv", index=False)
    sensitivity.to_csv(TABLE_DIR / "cough_speech_fusion_sensitivity.csv", index=False)
    cohort.to_csv(TABLE_DIR / "cohort_partition_record.csv", index=False)
    pd.DataFrame([ci]).to_csv(TABLE_DIR / "selected_auroc_bootstrap_record.csv", index=False)
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

    write_values(branches, combinations, sensitivity, cohort, selected_validation, selected_test, ci)
    write_claim_ledger(selected_test, ci)
    build_study_design_figure(cohort)
    build_results_figure(branches, combinations, selected_test, ci)
    print(f"Validation-selected uniform fusion AUROC: {_metric(selected_test['auroc']):.6f}")
    print(f"Bootstrap 95% CI: [{_metric(ci['ci_low']):.6f}, {_metric(ci['ci_high']):.6f}]")
    print(f"Wrote evidence-audited conference assets under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
