#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_DIR = SCRIPT_DIR.parent
REPO_DIR = MASTER_DIR.parents[2]
RESULTS_DIR = REPO_DIR / "reports" / "tables"
METRICS_DIR = REPO_DIR / "data" / "outputs" / "metrics"
OUTPUT_DIR = MASTER_DIR / "figures"

COLORS = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "grid": "#D9E2EC",
    "source": "#2F6B9A",
    "process": "#4F7D4A",
    "model": "#7B5D8E",
    "internal": "#2F6B9A",
    "calendar": "#D08A28",
    "temporal": "#B44C43",
    "external": "#4F7D4A",
    "shuffle": "#9AA5B1",
    "accent": "#B44C43",
}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + height - 0.045, title, weight="bold", color=color, va="top", fontsize=7.7)
    ax.text(
        x + 0.018,
        y + height - 0.102,
        body,
        color=COLORS["ink"],
        va="top",
        linespacing=1.22,
        fontsize=6.75,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=COLORS["muted"],
            shrinkA=2,
            shrinkB=2,
        )
    )


def build_study_design() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    width = 0.22
    height = 0.24
    xs = [0.02, 0.27, 0.52, 0.77]
    _box(ax, xs[0], 0.69, width, height, "Coswara source", "2,114 participants\n19,024 recordings\nThree modality groups", COLORS["source"])
    _box(ax, xs[1], 0.69, width, height, "Preprocessing", "Quality audit + decoding\n16 kHz mono audio\nActive-region crop/pad", COLORS["process"])
    _box(ax, xs[2], 0.69, width, height, "Representations", "10,140 to top 800\nWavLM\nCNN-BiGRU", COLORS["model"])
    _box(ax, xs[3], 0.69, width, height, "Models + fusion", "SVC + boosted trees\nMean / weighted fusion\nLogistic stack", COLORS["accent"])
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(ax, (left + width, 0.81), (right, 0.81))

    eval_specs = [
        (0.02, "Participant-disjoint", "Random participant split\nCough+speech stack\nn=314", COLORS["internal"]),
        (0.27, "Calendar-aware", "Calendar-balanced split\nThree-modality mean\nn=431", COLORS["calendar"]),
        (0.52, "Early-to-late", "Earlier to later split\nFrozen feature bank\nn=411", COLORS["temporal"]),
        (0.77, "External transfer", "Coswara cough to COUGHVID\nNo target fitting\nn=8,331 recordings", COLORS["external"]),
    ]
    ax.plot([0.13, 0.88], [0.62, 0.62], color=COLORS["muted"], linewidth=1.0)
    _arrow(ax, (0.88, 0.69), (0.88, 0.62))
    ax.text(0.505, 0.64, "Frozen source workflow evaluated under four protocols", ha="center", va="bottom", fontsize=7.4, color=COLORS["muted"])
    for x, title, body, color in eval_specs:
        _box(ax, x, 0.30, width, 0.25, title, body, color)
        _arrow(ax, (x + width / 2, 0.62), (x + width / 2, 0.55))

    guardrail = FancyBboxPatch(
        (0.02, 0.06),
        0.97,
        0.14,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        linewidth=0.9,
        edgecolor=COLORS["muted"],
        facecolor="#F5F7FA",
    )
    ax.add_patch(guardrail)
    ax.text(0.04, 0.165, "Evaluation guardrails", weight="bold", color=COLORS["ink"], va="center")
    ax.text(
        0.04,
        0.105,
        "participant separation   |   training-only SMOTE/ranking   |   "
        "source-validation selection   |   no target-label fitting",
        color=COLORS["muted"],
        va="center",
        fontsize=7.05,
    )
    ax.text(0.02, 0.98, "Study design and evaluation boundaries", weight="bold", fontsize=10.5, va="top")
    _save(fig, "fig01_study_design")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_validation_results() -> None:
    selected = _read(RESULTS_DIR / "compare_is10_final_validation_summary.csv")
    transfer = _read(RESULTS_DIR / "reviewer_external_model_family_transfer_summary.csv")
    deltas = _read(RESULTS_DIR / "final_validation_delta_bootstrap_ci.csv")

    selected_specs = [
        ("compare_is10_existing_participant_split", "Participant-disjoint\ncough+speech stack (n=314)", COLORS["internal"]),
        ("compare_is10_time_stratified_participant_split", "Calendar-aware\n3-modality mean (n=431)", COLORS["calendar"]),
        ("compare_is10_temporal_early_to_late", "Early-to-late\nbreath ensemble (n=411)", COLORS["temporal"]),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 4.65), gridspec_kw={"width_ratios": [0.88, 1.42]})

    values: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    for protocol, label, color in selected_specs:
        row = selected[selected["evaluation_protocol"].astype(str).eq(protocol)]
        if len(row) != 1:
            raise ValueError(f"Expected one selected row for {protocol}, found {len(row)}")
        values.append(float(row.iloc[0]["auroc"]))
        labels.append(label)
        colors.append(color)

    y = np.arange(len(values))[::-1]
    ax1.hlines(y, 0.5, values, color=COLORS["grid"], linewidth=2.2)
    ax1.scatter(values, y, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for value, yi in zip(values, y):
        ax1.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=7.8)
    ax1.axvline(0.5, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0.48, 1.02)
    ax1.set_xlabel("AUROC")
    ax1.set_title("A. Protocol-specific selections", loc="left", weight="bold", fontsize=8.8)
    ax1.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    ordered_models = [
        ("compare_is10_lightgbm_smote_f80", "LightGBM"),
        ("compare_is10_svc_rbf_f60", "SVC-RBF"),
        ("compare_is10_catboost_smote_f80", "CatBoost"),
        ("compare_is10_xgboost_smote_f80", "XGBoost"),
        ("wavlm_base_plus_pooled_cough", "WavLM"),
        ("cnn_bigru", "CNN-BiGRU"),
    ]
    y2 = np.arange(len(ordered_models))[::-1]
    for yi, (key, label) in zip(y2, ordered_models):
        row = transfer[transfer["family_model"].astype(str).eq(key)]
        if len(row) != 1:
            raise ValueError(f"Expected one transfer row for {key}, found {len(row)}")
        internal = float(row.iloc[0]["internal_auroc"])
        external = float(row.iloc[0]["external_auroc"])
        ax2.plot([external, internal], [yi, yi], color=COLORS["grid"], linewidth=2.4, zorder=1)
        ax2.scatter(internal, yi, s=40, color=COLORS["internal"], edgecolor="white", linewidth=0.7, zorder=3)
        ax2.scatter(external, yi, s=40, color=COLORS["external"], edgecolor="white", linewidth=0.7, zorder=3)
        annotation = f"delta {internal - external:.3f}"
        comparison_key = f"existing_cough_{key.removeprefix('compare_is10_')}_minus_coughvid_external"
        delta_row = deltas[deltas["comparison_id"].astype(str).eq(comparison_key) & deltas["metric"].astype(str).eq("auroc")]
        if not delta_row.empty:
            lo = float(delta_row.iloc[0]["ci_low"])
            hi = float(delta_row.iloc[0]["ci_high"])
            annotation += f" [{lo:.3f}, {hi:.3f}]"
        ax2.text(1.005, yi, annotation, ha="right", va="center", fontsize=6.9, color=COLORS["muted"])

    ax2.axvline(0.5, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    ax2.set_xlim(0.44, 1.02)
    ax2.set_yticks(y2, [label for _, label in ordered_models])
    ax2.set_xlabel("AUROC")
    ax2.set_title("B. Cough-only internal-to-external transfer", loc="left", weight="bold", fontsize=8.8)
    ax2.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.scatter([], [], color=COLORS["internal"], label="Coswara internal")
    ax2.scatter([], [], color=COLORS["external"], label="COUGHVID external")
    ax2.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1.0, -0.20))
    fig.text(0.055, 0.055, "A: protocol-specific selections are not a frozen-model degradation curve.", color=COLORS["muted"], fontsize=6.8)
    fig.text(0.055, 0.025, "B: brackets are 95% intervals for conventional AUROC differences (316 source participants; 8,331 target recordings).", color=COLORS["muted"], fontsize=6.8)
    fig.subplots_adjust(wspace=0.57, bottom=0.25, top=0.90)
    _save(fig, "fig02_validation_and_transfer")


def build_mechanism_results() -> None:
    shuffle = _read(RESULTS_DIR / "metadata_confounding_shuffle_retrain_sanity.csv")
    incremental = _read(METRICS_DIR / "reviewer_incremental_audio_metadata_metrics.csv")
    candidates = _read(RESULTS_DIR / "reviewer_incremental_audio_metadata_candidates.csv")
    stability = _read(RESULTS_DIR / "reviewer_feature_selection_stability.csv").iloc[0]
    overlap = _read(RESULTS_DIR / "reviewer_support_overlap_positivity.csv").iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.55))
    ax1, ax2, ax3, ax4 = axes.ravel()
    order = ["full_safe_metadata", "symptoms_only", "demographic_protocol_only"]
    labels = ["Full metadata", "Symptoms", "Demographic + protocol"]
    rows = shuffle.set_index("audit_model").loc[order]
    x = np.arange(len(order))
    width = 0.34
    ax1.bar(x - width / 2, rows["observed_auroc"], width, color=COLORS["source"], label="Observed")
    shuffle_mean = rows["shuffled_auroc_mean"].to_numpy(dtype=float)
    shuffle_low = rows["shuffled_auroc_ci_low"].to_numpy(dtype=float)
    shuffle_high = rows["shuffled_auroc_ci_high"].to_numpy(dtype=float)
    ax1.bar(x + width / 2, shuffle_mean, width, color=COLORS["shuffle"], label="Labels shuffled")
    ax1.errorbar(x + width / 2, shuffle_mean, yerr=np.vstack([shuffle_mean - shuffle_low, shuffle_high - shuffle_mean]), fmt="none", ecolor=COLORS["ink"], capsize=2.5, linewidth=0.8)
    ax1.axhline(0.5, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax1.set_ylim(0.42, 1.01)
    ax1.set_xticks(x, labels, rotation=16, ha="right")
    ax1.set_ylabel("AUROC")
    ax1.set_title("A. Metadata association\nand shuffle control", loc="left", weight="bold", fontsize=8.8)
    ax1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.34))
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)

    forest_rows = []
    for feature_set, label in [("full_safe_metadata", "Full metadata + audio"), ("symptoms_only", "Symptoms + audio")]:
        candidate = candidates[candidates["metadata_feature_set"].astype(str).eq(feature_set) & pd.to_numeric(candidates["candidate_rank"], errors="coerce").eq(1)]
        if len(candidate) != 1:
            raise ValueError(f"Expected one rank-1 incremental candidate for {feature_set}")
        source_key = str(candidate.iloc[0]["audio_source_key"])
        row = incremental[
            incremental["metadata_feature_set"].astype(str).eq(feature_set)
            & incremental["nested_model"].astype(str).eq("metadata_plus_audio")
            & incremental["audio_source_key"].astype(str).eq(source_key)
            & incremental["metric_split"].astype(str).eq("test")
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one incremental metric row for {feature_set}")
        forest_rows.append((label, float(row.iloc[0]["delta_auroc_vs_metadata"]), float(row.iloc[0]["delta_auroc_ci_low_vs_metadata"]), float(row.iloc[0]["delta_auroc_ci_high_vs_metadata"]), int(float(row.iloc[0]["n_samples"]))))
    y_forest = np.arange(len(forest_rows))[::-1]
    for yi, (label, estimate, low, high, n) in zip(y_forest, forest_rows):
        ax2.plot([low, high], [yi, yi], color=COLORS["source"], linewidth=1.5)
        ax2.scatter(estimate, yi, color=COLORS["source"], s=42, zorder=3)
        ax2.text(high + 0.008, yi, f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]", va="center", fontsize=7.3)
        ax2.text(-0.055, yi + 0.18, label, va="bottom", ha="left", fontsize=7.0, color=COLORS["muted"])
    ax2.axvline(0.0, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax2.set_xlim(-0.06, 0.18)
    ax2.set_yticks([])
    ax2.set_ylim(-0.35, 1.35)
    ax2.set_xlabel("AUROC difference")
    ax2.set_title("B. Incremental audio value\n(rank-1 validation candidate)", loc="left", weight="bold", fontsize=8.8)
    ax2.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.text(0.02, 0.05, f"Aligned test cohort: n={forest_rows[0][4]}", transform=ax2.transAxes, fontsize=7.0, color=COLORS["muted"])

    early_only = int(stability["early_only_count"])
    shared = int(stability["overlap_count"])
    late_only = int(stability["late_only_count"])
    total = early_only + shared + late_only
    left = 0
    for value, color, label in [(early_only, COLORS["calendar"], "Early only"), (shared, COLORS["source"], "Shared"), (late_only, COLORS["temporal"], "Late only")]:
        ax3.barh([0], [value], left=left, color=color, height=0.42, label=label)
        if value / total > 0.09:
            ax3.text(left + value / 2, 0, str(value), ha="center", va="center", color="white", weight="bold")
        left += value
    ax3.text(early_only + shared / 2, 0.29, str(shared), ha="center", va="bottom", color=COLORS["source"], weight="bold")
    ax3.set_xlim(0, total)
    ax3.set_yticks([])
    ax3.set_xlabel("Features in the early/late top-800 union")
    ax3.set_title("C. Temporal feature-selection stability", loc="left", weight="bold")
    ax3.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax3.spines[["top", "right", "left"]].set_visible(False)
    ax3.text(0.5, 0.84, f"Jaccard overlap = {float(stability['jaccard_overlap']):.3f}", transform=ax3.transAxes, ha="center", weight="bold", color=COLORS["ink"])

    within = 100.0 * float(overlap["external_within_source_domain_probability_band_fraction"])
    outside = 100.0 * float(overlap["external_probably_outside_source_support_fraction"])
    ax4.barh([0], [within], color=COLORS["external"], height=0.42, label="Within source band")
    ax4.barh([0], [outside], left=[within], color=COLORS["accent"], height=0.42, label="Outside source band")
    ax4.text(within / 2, 0, f"{within:.1f}%", color="white", ha="center", va="center", weight="bold")
    ax4.text(within + outside / 2, 0, f"{outside:.1f}%", color="white", ha="center", va="center", weight="bold")
    ax4.set_xlim(0, 100)
    ax4.set_yticks([])
    ax4.set_xlabel("COUGHVID recordings")
    ax4.set_title("D. Source-target overlap diagnostic", loc="left", weight="bold")
    ax4.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax4.spines[["top", "right", "left"]].set_visible(False)
    ax4.text(0.5, 0.84, f"Domain-classifier AUROC = {float(overlap['domain_classifier_auroc']):.3f}", transform=ax4.transAxes, ha="center", weight="bold", color=COLORS["ink"])

    fig.subplots_adjust(wspace=0.50, hspace=0.82, bottom=0.14, top=0.94)
    _save(fig, "fig03_mechanism_and_incremental_value")


def main() -> None:
    _configure()
    build_study_design()
    build_validation_results()
    build_mechanism_results()
    print(f"Wrote publication figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
