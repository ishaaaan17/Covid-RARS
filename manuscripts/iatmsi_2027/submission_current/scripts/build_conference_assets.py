from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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
EXPECTED_PARTITIONS = {
    "train": (1460, 476, 984, 12685),
    "validation": (312, 101, 211, 2702),
    "test": (316, 103, 213, 2719),
}


def _one(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected one row for {description}, found {len(frame)}")
    return frame.iloc[0]


def _same_key(frame: pd.DataFrame, row: pd.Series, key_columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in key_columns:
        value = row[column]
        mask &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
    return mask


def validate_supporting_evidence(selected_test: pd.Series) -> None:
    participant_audit = pd.read_csv(REPORT_TABLES / "strong_baseline_participant_audit.csv")
    participant_counts = (
        participant_audit.groupby(["split", "label_binary"], as_index=False)["n_participants"].sum()
        .pivot(index="split", columns="label_binary", values="n_participants")
        .fillna(0)
    )
    recording_audit = pd.read_csv(REPORT_TABLES / "strong_baseline_protocol_audit.csv")
    recording_counts = (
        recording_audit[recording_audit["protocol_exclusion_reason"].eq("included")]
        .groupby("split", as_index=True)["n_rows"]
        .sum()
    )
    cohort_rows: list[dict[str, object]] = []
    for split, expected in EXPECTED_PARTITIONS.items():
        positive = int(participant_counts.loc[split, "positive"])
        negative = int(participant_counts.loc[split, "negative"])
        participants = positive + negative
        recordings = int(recording_counts.loc[split])
        observed = (participants, positive, negative, recordings)
        if observed != expected:
            raise ValueError(f"Unexpected {split!r} cohort counts: expected {expected}, found {observed}")
        cohort_rows.append(
            {
                "split": split,
                "n_participants": participants,
                "n_positive": positive,
                "n_negative": negative,
                "n_recordings": recordings,
            }
        )

    bootstrap = pd.read_csv(REPORT_TABLES / "final_validation_bootstrap_ci.csv", low_memory=False)
    selected_ci = bootstrap[
        bootstrap["prediction_source"].eq("compare_is10_final_validation_predictions")
        & bootstrap["evaluation_protocol"].eq(PROTOCOL)
        & bootstrap["analysis_family"].eq(selected_test["analysis_family"])
        & bootstrap["model_name"].eq(selected_test["model_name"])
        & bootstrap["modality_combination"].eq(selected_test["modality_combination"])
        & bootstrap["fusion_method"].eq(selected_test["fusion_method"])
        & bootstrap["metric_split"].eq("test")
        & bootstrap["metric"].eq("auroc")
    ].copy()
    selected_ci_row = _one(selected_ci, "selected test AUROC bootstrap interval")
    if not np.isclose(float(selected_ci_row["point"]), float(selected_test["auroc"])):
        raise ValueError("The selected test AUROC and its bootstrap point estimate disagree")
    if int(selected_ci_row["n_bootstraps"]) != 1000 or int(selected_ci_row["n_samples"]) != 314:
        raise ValueError("Unexpected bootstrap resample or participant count")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cohort_rows).to_csv(TABLE_DIR / "cohort_partition_record.csv", index=False)
    selected_ci.to_csv(TABLE_DIR / "selected_auroc_bootstrap_record.csv", index=False)


def build_evidence_tables() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(REPORT_TABLES / "paper_metric_table_raw.csv", low_memory=False)
    evidence = raw[
        raw["table_source"].eq(SOURCE_TABLE)
        & raw["evaluation_protocol"].eq(PROTOCOL)
        & raw["feature_strategy"].eq(FEATURE_STRATEGY)
        & raw["metric_split"].isin(["validation", "test"])
    ].copy()

    numeric_columns = [
        "n_samples",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
    ]
    for column in numeric_columns:
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")

    branch_specs = [("Breathing", "breath"), ("Cough", "cough"), ("Speech", "speech")]
    rows: list[dict[str, object]] = []
    key_columns = [
        "analysis_family",
        "model_name",
        "modality",
        "modality_combination",
        "fusion_method",
    ]
    for label, modality in branch_specs:
        validation_candidates = evidence[
            evidence["analysis_family"].eq("strong_audio_modality")
            & evidence["modality"].eq(modality)
            & evidence["metric_split"].eq("validation")
        ].sort_values(["auroc", "auprc", "model_name"], ascending=[False, False, True])
        selected_branch = validation_candidates.iloc[0]
        for split in ("validation", "test"):
            row = _one(
                evidence[
                    evidence["metric_split"].eq(split)
                    & _same_key(evidence, selected_branch, key_columns)
                ],
                f"{label} {split}",
            )
            rows.append(
                {
                    "system": label,
                    "system_type": "single modality",
                    "configuration": selected_branch["model_name"],
                    "split": split,
                    **{column: row[column] for column in numeric_columns},
                }
            )

    fusion_methods = [
        ("Cough+speech mean", "uniform_mean"),
        ("Cough+speech weighted", "validation_weighted_auprc"),
        ("Cough+speech stack", "stacked_logistic_validation"),
    ]
    for label, method in fusion_methods:
        for split in ("validation", "test"):
            row = _one(
                evidence[
                    evidence["analysis_family"].eq("strong_multimodal_fusion")
                    & evidence["modality_combination"].eq("cough+speech")
                    & evidence["fusion_method"].eq(method)
                    & evidence["metric_split"].eq(split)
                ],
                f"{label} {split}",
            )
            rows.append(
                {
                    "system": label,
                    "system_type": "probability fusion",
                    "configuration": method,
                    "split": split,
                    **{column: row[column] for column in numeric_columns},
                }
            )

    comparison = pd.DataFrame(rows)

    validation_fusions = evidence[
        evidence["analysis_family"].eq("strong_multimodal_fusion")
        & evidence["metric_split"].eq("validation")
        & evidence["fusion_method"].eq("uniform_mean")
    ].sort_values(["auroc", "auprc"], ascending=[False, False])
    selected_validation = validation_fusions.iloc[0]
    selected_test = _one(
        evidence[
            evidence["metric_split"].eq("test")
            & _same_key(evidence, selected_validation, key_columns)
        ],
        "validation-selected fusion test result",
    )

    if not (
        selected_validation["modality_combination"] == "cough+speech"
        and selected_validation["fusion_method"] == "uniform_mean"
    ):
        raise ValueError("The frozen validation-selected fusion no longer matches the audited configuration")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLE_DIR / "selected_internal_systems.csv", index=False)

    selection_record = pd.DataFrame(
        [
            {
                "selection_split": "validation",
                "analysis_family": selected_validation["analysis_family"],
                "modality_combination": selected_validation["modality_combination"],
                "fusion_method": selected_validation["fusion_method"],
                "validation_auroc": selected_validation["auroc"],
                "validation_auprc": selected_validation["auprc"],
                "test_auroc": selected_test["auroc"],
                "test_auprc": selected_test["auprc"],
            }
        ]
    )
    selection_record.to_csv(TABLE_DIR / "validation_selection_record.csv", index=False)
    validate_supporting_evidence(selected_test)
    return comparison, selected_test


def _node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    face: str,
    fontsize: float = 6.5,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=face,
            edgecolor="#333333",
            linewidth=0.8,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.12,
        zorder=3,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color="#4A4A4A",
            shrinkA=1,
            shrinkB=1,
            zorder=1,
        )
    )


def build_figure(comparison: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.1, 2.35), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.14)

    flow = figure.add_subplot(grid[0, 0])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    flow.text(0.0, 0.98, "(a)", weight="bold", va="top")

    _arrow(flow, (0.19, 0.52), (0.26, 0.52))
    _arrow(flow, (0.47, 0.52), (0.54, 0.52))
    _arrow(flow, (0.78, 0.52), (0.85, 0.52))

    _node(flow, 0.02, 0.37, 0.17, 0.30, "Coswara\ncohort", "#F2F2F2", fontsize=6.8)
    _node(
        flow,
        0.26,
        0.37,
        0.21,
        0.30,
        "Training only\nrank features\nfit three branches",
        "#E8F1F8",
        fontsize=6.2,
    )
    _node(
        flow,
        0.54,
        0.37,
        0.24,
        0.30,
        "Validation only\nselect branches\nfusion + threshold",
        "#E8F1F8",
        fontsize=6.2,
    )
    _node(flow, 0.85, 0.37, 0.13, 0.30, "Held-out\ntest", "#F2F2F2", fontsize=6.4)
    flow.text(0.50, 0.96, "Fixed participant-disjoint development", ha="center", va="top", fontsize=7.0)
    flow.text(0.365, 0.29, "cough  |  breathing  |  speech", ha="center", va="top", fontsize=5.8)

    result = figure.add_subplot(grid[0, 1])
    result.text(-0.12, 1.02, "(b)", transform=result.transAxes, weight="bold", va="top")
    labels = ["Breathing", "Cough", "Speech", "Cough+speech\nmean"]
    selected = comparison[(comparison["split"] == "test") & comparison["system"].isin([
        "Breathing", "Cough", "Speech", "Cough+speech mean"
    ])].set_index("system")
    order = ["Breathing", "Cough", "Speech", "Cough+speech mean"]
    y = np.arange(len(order))[::-1]
    auroc = selected.loc[order, "auroc"].to_numpy(dtype=float)
    auprc = selected.loc[order, "auprc"].to_numpy(dtype=float)

    roc_y = y + 0.10
    pr_y = y - 0.10
    result.scatter(auroc, roc_y, marker="o", s=24, color="#00629B", label="AUROC", zorder=3)
    result.scatter(auprc, pr_y, marker="s", s=20, facecolor="white", edgecolor="#333333", label="AUPRC", zorder=3)
    for yi, roc, pr in zip(y, auroc, auprc):
        result.text(roc + 0.004, yi + 0.10, f"{roc:.3f}", va="center", fontsize=6.8, color="#004B76")
        result.text(pr - 0.004, yi - 0.10, f"{pr:.3f}", va="center", ha="right", fontsize=6.5, color="#333333")
    result.set_yticks(y, labels)
    result.set_xlim(0.68, 0.925)
    result.set_ylim(-0.65, len(order) - 0.35)
    result.set_xlabel("Participant-level test performance")
    result.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    result.spines[["top", "right", "left"]].set_visible(False)
    result.tick_params(axis="y", length=0)
    result.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 1.02),
        frameon=False,
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    for suffix in ("pdf", "svg"):
        figure.savefig(FIGURE_DIR / f"conference_design_and_results.{suffix}", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    comparison, selected_test = build_evidence_tables()
    build_figure(comparison)
    print(f"Validation-selected test AUROC: {float(selected_test['auroc']):.6f}")
    print(f"Wrote conference assets under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
