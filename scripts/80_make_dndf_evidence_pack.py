#!/usr/bin/env python3
"""Build and format publication-ready evidence pack and comparative tables for DNDT/DNDF."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from covid_rars.dndf_reporting import build_comparative_table_dndf_vs_hst_vs_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("dndf_evidence_pack")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DNDT/DNDF comparative evidence pack")
    parser.add_argument("--dndf-dir", type=str, default="reports/dndf", help="Directory containing DNDF outputs")
    parser.add_argument("--output-csv", type=str, default="reports/tables/dndf_comparative_publication_matrix.csv", help="Output CSV path")
    args = parser.parse_args()

    dndf_path = Path(args.dndf_dir) / "dndf_final_validation_summary.csv"
    if not dndf_path.exists():
        logger.warning(f"DNDF summary not found at {dndf_path}. Run 79_run_dndf_reliability.py first.")
        return 1

    dndf_df = pd.read_csv(dndf_path)
    out_table = build_comparative_table_dndf_vs_hst_vs_baselines(dndf_df)

    out_file = Path(args.output_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_table.to_csv(out_file, index=False)
    logger.info(f"Comparative publication matrix saved to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
