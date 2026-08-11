#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.representation_features import validate_feature_table, write_feature_table
from covid_rars.ssl_extractors import create_ssl_extractor
from covid_rars.torch_embedding_features import extract_torch_embedding_feature_table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract learned audio embeddings from wav2vec2, BEATs, or PANNs into a validated feature table."
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        required=True,
        help="Embedding backend: wav2vec2, beats, or panns. BEATs/PANNs require local checkpoint/source paths.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quality-mode", default="all_samples", choices=["all_samples", "quality_ok_only"])
    parser.add_argument("--modality", default=None, help="Optional modality filter such as cough, breath, or speech.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit before full extraction.")
    parser.add_argument("--split-name", default=None, help="Force all input rows to this split, e.g. external.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--strict", action="store_true", help="Fail immediately on the first unreadable audio file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    if args.split_name is not None:
        metadata = metadata.copy()
        metadata["split"] = args.split_name
    extractor = create_ssl_extractor(
        backend=args.backend,
        checkpoint_path=args.checkpoint_path,
        source_dir=args.source_dir,
        device=args.device,
    )
    features = extract_torch_embedding_feature_table(
        metadata,
        extractor=extractor,
        quality_mode=args.quality_mode,
        modality=args.modality,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        strict=args.strict,
    )
    validate_feature_table(features)
    write_feature_table(features, args.output)
    print(
        f"Wrote {extractor.representation_name} features: "
        f"{args.output} ({len(features)} rows, {len(features.columns)} columns)"
    )


if __name__ == "__main__":
    main()
