#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_audio_btp.hst_comparator_approval import (
    build_comparator_generation_acceptance_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate one comparator generation and prepare a review-only accepted-"
            "freezes update. All paths, including --output, are explicit and manual "
            "review/edit/commit is required."
        ),
        epilog="Manual review required. This tool never approves a generation.",
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--approval-record", type=Path, required=True)
    parser.add_argument("--accepted-freezes", type=Path, required=True)
    parser.add_argument("--expected-accepted-freezes-sha256", required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--current-receipt", type=Path, required=True)
    parser.add_argument("--runtime-random-state", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_comparator_generation_acceptance_candidate(
        project_root=args.project_root,
        approval_record_path=args.approval_record,
        accepted_freezes_path=args.accepted_freezes,
        expected_accepted_freezes_sha256=args.expected_accepted_freezes_sha256,
        generation_manifest_path=args.generation_manifest,
        current_receipt_path=args.current_receipt,
        runtime_random_state=args.runtime_random_state,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(
        "REVIEW-ONLY CANDIDATE: manual review, explicit promotion, and a Git commit "
        "are required. The canonical accepted-freezes file was not modified."
    )


if __name__ == "__main__":
    main()
