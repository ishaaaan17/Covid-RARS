#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.final_report import build_final_report, build_summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final evidence-driven Covid-RARS publication report."
    )
    parser.add_argument("--evidence", type=Path, default=Path("reports/tables/publication_evidence_matrix.csv"))
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/final/COVID_RARS_PUBLICATION_RESULTS_REPORT.md"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/final/COVID_RARS_PUBLICATION_RESULTS_SUMMARY.md"),
    )
    parser.add_argument("--related-paper-comparison", type=Path, default=Path("reports/tables/related_paper_comparison.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.evidence.exists() or args.evidence.stat().st_size == 0:
        raise FileNotFoundError(f"Publication evidence matrix not found: {args.evidence}")
    evidence = pd.read_csv(args.evidence)
    related_markdown = None
    if args.related_paper_comparison.exists() and args.related_paper_comparison.stat().st_size > 0:
        related_markdown = args.related_paper_comparison.read_text(encoding="utf-8")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(build_final_report(evidence, related_paper_markdown=related_markdown), encoding="utf-8")
    args.summary_output.write_text(build_summary_report(evidence), encoding="utf-8")
    print(f"Wrote final publication report: {args.report_output}")
    print(f"Wrote final publication summary: {args.summary_output}")


if __name__ == "__main__":
    main()
