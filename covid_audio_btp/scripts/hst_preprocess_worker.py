#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import pandas as pd

from covid_audio_btp.hst_spectrograms import HSTSpectrogramConfig, build_hst_spectrogram_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process one isolated HST spectrogram-cache job")
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    config = HSTSpectrogramConfig(**job["config"])
    result = build_hst_spectrogram_cache(
        pd.DataFrame([job["metadata"]]),
        output_dir=Path(job["output_dir"]),
        config=config,
        force=bool(job.get("force", False)),
    )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result_json.with_suffix(args.result_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"request_id": job["request_id"], "result": result.iloc[0].to_dict()},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.result_json)


if __name__ == "__main__":
    main()
