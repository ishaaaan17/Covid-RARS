#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_rars.hst_comparator_approval import (
    build_comparator_accepted_freezes_candidate,
    build_comparator_approval_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare review-only comparator approval candidates. Every path, including "
            "--output, must be explicit; manual review/edit/commit is always required."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    approval = subparsers.add_parser(
        "approval-record",
        help="prepare the exact ComParE+IS10 approval-record candidate",
    )
    approval.add_argument("--project-root", type=Path, required=True)
    approval.add_argument("--run-root", type=Path, required=True)
    approval.add_argument("--manifests-receipt", type=Path, required=True)
    approval.add_argument("--feature-table", type=Path, required=True)
    approval.add_argument("--pilot-accepted-freezes", type=Path, required=True)
    approval.add_argument("--environment-lock", type=Path, required=True)
    approval.add_argument("--runtime-random-state", type=int, required=True)
    approval.add_argument("--output", type=Path, required=True)

    accepted = subparsers.add_parser(
        "accepted-freezes",
        help="prepare accepted-freezes after the approval record was manually committed",
    )
    accepted.add_argument("--project-root", type=Path, required=True)
    accepted.add_argument("--approval-record", type=Path, required=True)
    accepted.add_argument("--pilot-accepted-freezes", type=Path, required=True)
    accepted.add_argument("--environment-lock", type=Path, required=True)
    accepted.add_argument("--project-id", required=True)
    accepted.add_argument("--expected-remote-url", required=True)
    accepted.add_argument("--runtime-random-state", type=int, required=True)
    accepted.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "approval-record":
        payload = build_comparator_approval_candidate(
            project_root=args.project_root,
            run_root=args.run_root,
            manifests_receipt_path=args.manifests_receipt,
            manifest_name="aligned_comparator",
            feature_table_path=args.feature_table,
            pilot_accepted_freezes_path=args.pilot_accepted_freezes,
            environment_lock_path=args.environment_lock,
            runtime_random_state=args.runtime_random_state,
            output_path=args.output,
        )
    else:
        payload = build_comparator_accepted_freezes_candidate(
            project_root=args.project_root,
            approval_record_path=args.approval_record,
            pilot_accepted_freezes_path=args.pilot_accepted_freezes,
            environment_lock_path=args.environment_lock,
            project_id=args.project_id,
            expected_remote_url=args.expected_remote_url,
            runtime_random_state=args.runtime_random_state,
            output_path=args.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(
        "REVIEW-ONLY CANDIDATE: manual review, explicit promotion/edit, and a Git "
        "commit are required. No canonical approved file was written."
    )


if __name__ == "__main__":
    main()
