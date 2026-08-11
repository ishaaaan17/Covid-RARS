from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_class_distribution(metadata: pd.DataFrame, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = metadata.groupby(["split", "label_binary"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(8, 4))
    ax.set_title("Class distribution by split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Recordings")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_reliability_diagram(
    predictions: pd.DataFrame,
    output: Path,
    probability_column: str = "probability",
    n_bins: int = 10,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = predictions[predictions["label_binary"].isin(["positive", "negative"])].copy()
    df["y"] = (df["label_binary"] == "positive").astype(int)
    df["bin"] = pd.cut(df[probability_column], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        confidence=(probability_column, "mean"),
        accuracy=("y", "mean"),
        n=("y", "size"),
    )
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(grouped["confidence"], grouped["accuracy"], marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical positive rate")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def write_report_outline(output: Path) -> None:
    text = """# Final Report Outline

1. Introduction
2. Literature Review
3. Research Gap
4. Dataset And Ethics
5. Methodology
6. Experiments
7. Results
8. Calibration And Reliability
9. Quality, Confounding, And Shift Analysis
10. Demo
11. Limitations
12. Conclusion

Required disclaimer: Research prototype only. Not a clinical diagnostic tool.
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def write_slides_outline(output: Path) -> None:
    text = """# Slides Outline

1. Title and disclaimer
2. Problem and motivation
3. Literature gap
4. Dataset: Coswara
5. Pipeline overview
6. Quality and participant split controls
7. Models
8. Calibration and uncertainty
9. Results
10. Confounding and shift checks
11. Demo screenshot
12. Limitations and future work
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

DEFAULT_PAPER_METRICS = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "brier",
    "ece",
    "nll",
]


def _is_missing_metric(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _format_float(value: object, digits: int = 3) -> str:
    if _is_missing_metric(value):
        return ""
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _matching_ci_row(
    ci_table: pd.DataFrame,
    metric_name: str,
    metric_row: pd.Series,
    group_columns: list[str],
) -> pd.Series | None:
    if ci_table is None or ci_table.empty or "metric" not in ci_table.columns:
        return None
    mask = ci_table["metric"].astype(str) == metric_name
    for col in group_columns:
        if col == "table_source":
            continue
        if col in ci_table.columns and col in metric_row.index:
            mask &= ci_table[col].astype(str) == str(metric_row[col])
    matches = ci_table[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_paper_metric_table(
    metrics: pd.DataFrame,
    ci_table: pd.DataFrame | None = None,
    group_columns: list[str] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a compact, paper-ready metric table with optional bootstrap CIs."""
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    group_columns = [col for col in (group_columns or []) if col in metrics.columns]
    metric_columns = metric_columns or DEFAULT_PAPER_METRICS
    rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        out: dict[str, object] = {col: row[col] for col in group_columns}
        if "n_samples" in metrics.columns:
            out["n_samples"] = int(row["n_samples"]) if not _is_missing_metric(row["n_samples"]) else ""
        for metric_name in metric_columns:
            if metric_name not in metrics.columns:
                continue
            ci_row = _matching_ci_row(ci_table, metric_name, row, group_columns) if ci_table is not None else None
            if ci_row is not None and {"ci_low", "ci_high"}.issubset(ci_row.index):
                point = ci_row["point"] if "point" in ci_row.index and not _is_missing_metric(ci_row["point"]) else row[metric_name]
                point_text = _format_float(point)
                low_text = _format_float(ci_row["ci_low"])
                high_text = _format_float(ci_row["ci_high"])
                out[metric_name] = f"{point_text} [{low_text}, {high_text}]" if point_text and low_text and high_text else point_text
            else:
                out[metric_name] = _format_float(row[metric_name])
        rows.append(out)
    return pd.DataFrame(rows)


def read_existing_csvs(paths: list[str | Path], source_column: str = "table_source") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df[source_column] = p.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
