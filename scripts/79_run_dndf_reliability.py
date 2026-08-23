#!/usr/bin/env python3
"""Run DNDT and DNDF Reliability Evaluation across Track 1, Track 2, and Track 3.

Inspired by & benchmarking against:
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

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from covid_rars.dndf_tracks import (
    run_track1_author_exact_reproduction,
    run_track2_corrected_leak_free_reproduction,
    run_track3_covid_rars_reliability_suite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dndf_runner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DNDT / DNDF 3-Track Benchmark Suite")
    parser.add_argument("--features", type=str, default="data/processed/features_compare_is10_merged.csv", help="Path to features CSV")
    parser.add_argument("--external-features", type=str, default="data/processed/features_compare_is10_coughvid_cough_top800.csv", help="Path to COUGHVID CSV")
    parser.add_argument("--config", type=str, default="configs/dndf_reliability.json", help="Path to JSON config")
    parser.add_argument("--track", type=str, default="all", choices=["1", "2", "3", "all"], help="Track to execute: 1 (Paper), 2 (Corrected), 3 (Reliability), or all")
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"], help="Modalities to evaluate")
    parser.add_argument("--output-dir", type=str, default="reports/dndf", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--smoke-test", action="store_true", help="Fast smoke test on small subset (runs in <10s)")

    args = parser.parse_args()

    # Resolve config defaults if exists
    config_data = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    dndf_cfg = config_data.get("architecture", {}).get("dndf", {})
    train_cfg = config_data.get("training", {})

    num_trees = dndf_cfg.get("num_trees", 25)
    depth = dndf_cfg.get("depth", 11)
    lr = train_cfg.get("learning_rate", 0.01)
    batch_size = train_cfg.get("batch_size", 16)
    max_epochs = train_cfg.get("max_epochs", 14)
    n_selected_features = train_cfg.get("n_selected_features", 33)

    if args.smoke_test:
        print("\n⚡ RUNNING IN FAST SMOKE-TEST MODE (50 samples, 2 epochs)...")
        num_trees = 5
        depth = 3
        max_epochs = 2
        batch_size = 16

    feat_path = Path(args.features)
    if not feat_path.exists():
        alt_path = Path("D:/BTP/processed/features_compare_is10_merged.csv")
        if alt_path.exists():
            feat_path = alt_path
        else:
            logger.error(f"Features file not found at: {feat_path}")
            return 1

    logger.info(f"Loading features from {feat_path}...")
    features_df = pd.read_csv(feat_path, low_memory=False)

    if args.smoke_test:
        # Take small subset per modality for fast end-to-end check
        sub_dfs = []
        for m in args.modalities:
            m_sub = features_df[features_df["modality"] == m].head(30)
            sub_dfs.append(m_sub)
        features_df = pd.concat(sub_dfs, ignore_index=True)

    ext_df = None
    ext_path = Path(args.external_features) if args.external_features else None
    if ext_path and ext_path.exists():
        ext_df = pd.read_csv(ext_path, low_memory=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Track 1: Authors' Exact Paper Reproduction
    if args.track in ("1", "all"):
        t1_df = run_track1_author_exact_reproduction(
            features_df=features_df,
            modality="cough",
            n_splits=2 if args.smoke_test else 10,
            num_trees=num_trees,
            depth=depth,
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            n_selected_features=n_selected_features,
            feature_selection="f_classif" if args.smoke_test else "rfecv_extratrees",
            device=args.device,
        )
        if not t1_df.empty:
            t1_df.to_csv(out_dir / "track1_author_paper_reproduction.csv", index=False)

    # 2. Track 2: Methodologically Corrected Leak-Free Reproduction
    if args.track in ("2", "all"):
        t2_df = run_track2_corrected_leak_free_reproduction(
            features_df=features_df,
            modality="cough",
            n_splits=2 if args.smoke_test else 10,
            num_trees=num_trees,
            depth=depth,
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            n_selected_features=n_selected_features,
            feature_selection="f_classif" if args.smoke_test else "rfecv_extratrees",
            device=args.device,
        )
        if not t2_df.empty:
            t2_df.to_csv(out_dir / "track2_corrected_leak_free_reproduction.csv", index=False)

    # 3. Track 3: COVID-RARS Extended Reliability Suite
    if args.track in ("3", "all"):
        t3_results = run_track3_covid_rars_reliability_suite(
            features_df=features_df,
            external_features_df=ext_df,
            modalities=args.modalities,
            seeds=(1, 2) if args.smoke_test else (1, 2, 5, 12, 40),
            num_trees=num_trees,
            depth=depth,
            learning_rate=lr,
            max_epochs=max_epochs,
            device=args.device,
        )

    print("\n✅ All requested tracks executed and saved to:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
