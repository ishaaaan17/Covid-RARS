#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_rars.hst_acceptance import build_pilot_acceptance_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a review-only HST pilot acceptance candidate."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_pilot_acceptance_candidate(
        run_root=args.run_root,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("This is not an approval. Review it before manual promotion to accepted_freezes.json.")


if __name__ == "__main__":
    main()
