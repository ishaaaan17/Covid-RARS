#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_rars.hst_coughvid_metadata import build_hst_coughvid_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind the processed COUGHVID cohort to frozen COUGHVID-v3 labels."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_hst_coughvid_metadata(
        cohort_path=args.cohort,
        raw_metadata_path=args.raw_metadata,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
