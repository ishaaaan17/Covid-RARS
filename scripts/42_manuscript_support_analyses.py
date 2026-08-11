#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from covid_rars.manuscript_support import run_manuscript_support_analyses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript-support analyses for reviewer-facing robustness claims.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--shap-output", type=Path, default=Path("reports/tables/manuscript_demographic_protocol_linear_shap.csv"))
    parser.add_argument("--ipw-output", type=Path, default=Path("reports/tables/manuscript_ipw_residual_smd.csv"))
    parser.add_argument("--auprc-output", type=Path, default=Path("reports/tables/manuscript_external_auprc_lift.csv"))
    parser.add_argument("--unknown-summary-output", type=Path, default=Path("reports/tables/manuscript_unknown_label_summary.csv"))
    parser.add_argument("--unknown-balance-output", type=Path, default=Path("reports/tables/manuscript_unknown_label_balance.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/final/MANUSCRIPT_SUPPORT_ANALYSES.md"))
    return parser.parse_args()


def _write_table(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {path} ({len(frame)} rows)")


def _markdown_table(frame) -> str:
    if frame.empty:
        return "_No rows available._"
    table = frame.copy()
    columns = [str(col) for col in table.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        values = []
        for col in table.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run_manuscript_support_analyses(args.project_root)
    _write_table(result.metadata_shap, args.shap_output)
    _write_table(result.ipw_residual_smd, args.ipw_output)
    _write_table(result.auprc_lift, args.auprc_output)
    _write_table(result.unknown_label_summary, args.unknown_summary_output)
    _write_table(result.unknown_label_balance, args.unknown_balance_output)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        "# Manuscript Support Analyses\n\n"
        "## Top Demographic/Protocol Linear Attribution Drivers\n\n"
        + _markdown_table(result.metadata_shap.head(5))
        + "\n\n## Worst Residual IPW Balance Rows\n\n"
        + _markdown_table(result.ipw_residual_smd.head(5))
        + "\n\n## External AUPRC Lift Over Prevalence\n\n"
        + _markdown_table(result.auprc_lift)
        + "\n\n## Unknown Label Summary\n\n"
        + _markdown_table(result.unknown_label_summary)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
