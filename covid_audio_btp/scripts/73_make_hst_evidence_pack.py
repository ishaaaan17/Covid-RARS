#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_audio_btp.hst_evidence import build_hst_evidence_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a checksum-validated HST evidence manifest."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-stage", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_root / "evidence" / "hst_evidence_manifest.json"
    )
    manifest = build_hst_evidence_manifest(
        run_root=run_root,
        output_path=output,
        required_stages=args.required_stage,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
