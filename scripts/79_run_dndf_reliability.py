#!/usr/bin/env python3
"""Run DNDT and DNDF Reliability Evaluation across all Tracks (A, B, C) and Audits.

Inspired by:
    Rofiqul Islam, Nihad Karim Chowdhury, and Muhammad Ashad Kabir,
    "Robust COVID-19 detection from cough sounds using deep neural decision tree and forest:
    A comprehensive cross-datasets evaluation", Expert Systems with Applications, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from covid_rars.dndf_stages import run_dndf_reliability_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dndf_reliability_runner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DNDT / DNDF integrated reliability study")
    parser.add_argument("--features", type=str, default="data/processed/features_compare_is10_top800.csv", help="Path to source features CSV")
    parser.add_argument("--external-features", type=str, default="data/processed/features_compare_is10_coughvid_cough_top800.csv", help="Path to external COUGHVID features CSV")
    parser.add_argument("--config", type=str, default="configs/dndf_reliability.json", help="Path to DNDF config JSON")
    parser.add_argument("--output-dir", type=str, default="reports/dndf", help="Output directory for results")
    parser.add_argument("--modalities", nargs="+", default=["cough", "speech", "breath"], help="Modalities to evaluate")
    parser.add_argument("--num-trees", type=int, default=20, help="Number of trees in DNDF")
    parser.add_argument("--depth", type=int, default=4, help="Tree depth")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")

    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        logger.error(f"Features file not found at: {feat_path}")
        return 1

    logger.info(f"Loading source features from: {feat_path}")
    features_df = pd.read_csv(feat_path)

    ext_df = None
    ext_path = Path(args.external_features) if args.external_features else None
    if ext_path and ext_path.exists():
        logger.info(f"Loading external features from: {ext_path}")
        ext_df = pd.read_csv(ext_path)

    artifacts = run_dndf_reliability_pipeline(
        features_df=features_df,
        external_features_df=ext_df,
        modalities=args.modalities,
        seeds=(1, 2, 5, 12, 40),
        num_trees=args.num_trees,
        depth=args.depth,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        device=args.device,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("DNDT / DNDF FINAL VALIDATION SUMMARY TABLE:")
    print("=" * 60)
    print(artifacts.final_summary_table.to_string(index=False))
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
