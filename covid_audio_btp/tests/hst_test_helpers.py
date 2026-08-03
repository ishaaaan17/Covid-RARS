from __future__ import annotations

from pathlib import Path

import pandas as pd


PRESPECIFIED_HST_REPO_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)


def expected_base_architecture() -> dict[str, object]:
    return {
        "img_size": 224,
        "h": 4,
        "img_channel": 3,
        "num_labels": 2,
        "d": 96,
        "num_blocks": [1, 1, 9, 1],
        "num_attention_heads": [3, 6, 12, 24],
        "win_size": 7,
        "mlp_ratio": 4.0,
        "use_bias": True,
        "dropout_rate": 0.0,
        "attn_dropout_rate": 0.0,
        "drop_path_rate": 0.1,
        "use_checkpoint": False,
    }


def expected_backbone_parameter_count() -> int:
    return 48_837_258


def official_hst_paths() -> tuple[Path, Path]:
    package_root = Path(__file__).resolve().parents[1]
    repo = package_root.parent / "HST"
    checkpoint = (
        package_root
        / ".cache"
        / "hst"
        / "checkpoints"
        / "hst_base_imagenet.pth"
    )
    return checkpoint, repo


def make_recording_predictions() -> pd.DataFrame:
    """Return two participants with unequal recording counts."""
    rows = []
    for recording_id, participant_id, label, probability in (
        ("r1", "p1", "positive", 0.9),
        ("r2", "p1", "positive", 0.7),
        ("r3", "p2", "negative", 0.2),
    ):
        rows.append(
            {
                "run_id": "run-a",
                "protocol": "internal",
                "fold": 1,
                "dataset": "coswara",
                "participant_id": participant_id,
                "participant_key": f"coswara::{participant_id}",
                "recording_id": recording_id,
                "recording_key": f"coswara::{recording_id}",
                "split": "test",
                "modality": "cough",
                "model": "hst_base",
                "checkpoint_hash": "a" * 64,
                "representation": "hst_spectrogram",
                "label_binary": label,
                "probability": probability,
            }
        )
    return pd.DataFrame(rows)
