#!/usr/bin/env python3
"""Script 80: Run Exact Paper Replication Benchmark (Islam et al. ESWA 2026).

Executes DNDF on 80/20 stratified recording-level split matching the exact
methodology from Islam et al. (Expert Systems with Applications, 2026).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from covid_rars.dndf_replication import run_paper_replication_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Exact Paper Replication Benchmark for DNDF.")
    parser.add_argument("--features", type=str, default="data/processed/features_compare_is10_merged.csv", help="Path to features CSV")
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath"], help="Modalities to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--trees", type=int, default=80, help="Number of trees in forest")
    parser.add_argument("--depth", type=int, default=5, help="Tree depth")
    parser.add_argument("--k-features", type=int, default=140, help="Top-k features to select")
    parser.add_argument("--lr", type=float, default=0.00132, help="Learning rate")
    parser.add_argument("--output", type=str, default="reports/dndf/paper_replication_summary.csv", help="Output summary CSV")
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        # Try finding in parent directory
        alt_path = Path("D:/BTP/processed/features_compare_is10_merged.csv")
        if alt_path.exists():
            feat_path = alt_path
        else:
            print(f"Error: Features file not found at {feat_path}")
            return 1

    print(f"Loading features from {feat_path}...")
    df = pd.read_csv(feat_path, low_memory=False)

    summary_df = run_paper_replication_benchmark(
        features_df=df,
        modalities=args.modalities,
        test_size=0.20,
        random_state=42,
        num_trees=args.trees,
        depth=args.depth,
        n_selected_features=args.k_features,
        learning_rate=args.lr,
        device=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"✅ Saved paper replication summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
