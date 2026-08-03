from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import subprocess
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any


PINNED_HST_COMMIT = "7f94ad81e392da856c7aac6d364d036c28e26c32"
EXPECTED_HST_MODEL_SOURCE_SHA256 = (
    "44c1688afb00ee3f7632577d011ca3857200d042818bc2ac28b3b8d18288479f"
)

ARCHITECTURES: dict[str, dict[str, object]] = {
    "hst_small": {
        "img_size": 224,
        "h": 4,
        "img_channel": 3,
        "num_labels": 2,
        "d": 96,
        "num_blocks": [1, 1, 3, 1],
        "num_attention_heads": [3, 6, 12, 24],
        "win_size": 7,
        "mlp_ratio": 4.0,
        "use_bias": True,
        "dropout_rate": 0.0,
        "attn_dropout_rate": 0.0,
        "drop_path_rate": 0.1,
        "use_checkpoint": False,
    },
    "hst_base": {
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
    },
}

CHECKPOINTS = {
    "hst_small": {
        "size_bytes": 111_266_629,
        "sha256": "e7086d1b87d598120296b9a1b5f094c7587cb06f50bf609a4ca13badc95e3112",
    },
    "hst_base": {
        "size_bytes": 197_063_145,
        "sha256": "f39f001d5f8cd90cb78d45612486202a4ea280e23df0b2c1d6ce35d96b30cce4",
    },
}

EXPECTED_BACKBONE_PARAMETERS = {
    "hst_small": 27_769_058,
    "hst_base": 49_173_398,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular_file(path: Path) -> bool:
    path = Path(path)
    if path.is_symlink():
        return False
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def verify_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    path = Path(path)
    if not _is_regular_file(path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Checkpoint must be a regular file, not a symlink or partial: {path}")
        raise FileNotFoundError(path)
    actual_size = path.lstat().st_size
    if actual_size != int(expected_size):
        raise ValueError(f"Checkpoint size mismatch: expected {expected_size}, found {actual_size}")
    actual_hash = sha256_file(path)
    if actual_hash.casefold() != str(expected_sha256).casefold():
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, found {actual_hash}")


def verify_hst_source(hst_repo: Path, expected_commit: str) -> str:
    hst_repo = Path(hst_repo)
    model_source = hst_repo / "model" / "hst_model.py"
    if not model_source.is_file():
        raise FileNotFoundError(f"Official HST model source is missing: {model_source}")
    result = subprocess.run(
        ["git", "-C", str(hst_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = result.stdout.strip().casefold()
    if actual_commit != str(expected_commit).strip().casefold():
        raise ValueError(f"HST source commit mismatch: expected {expected_commit}, found {actual_commit}")
    tracked_diff = subprocess.run(
        ["git", "-C", str(hst_repo), "diff", "--quiet", actual_commit, "--", "model/hst_model.py"],
        check=False,
    )
    if tracked_diff.returncode != 0:
        raise ValueError("Official HST model source has tracked modifications")
    source_hash = sha256_file(model_source)
    if source_hash != EXPECTED_HST_MODEL_SOURCE_SHA256:
        raise ValueError(
            "Official HST model source SHA-256 mismatch: "
            f"expected {EXPECTED_HST_MODEL_SOURCE_SHA256}, found {source_hash}"
        )
    return actual_commit


def download_verified_checkpoint(
    *,
    google_drive_file_id: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    destination = Path(destination)
    if _is_regular_file(destination):
        try:
            verify_file(
                destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except ValueError:
            pass
        else:
            return destination

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required to download official HST checkpoints") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
    try:
        result = gdown.download(id=google_drive_file_id, output=str(temporary), quiet=False)
        if result is None:
            raise RuntimeError("Google Drive checkpoint download did not produce a file")
        if not _is_regular_file(temporary):
            raise ValueError("Downloaded checkpoint must be a regular file, not a symlink or partial")
        with temporary.open("rb") as handle:
            prefix = handle.read(512).lstrip().lower()
        if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
            raise ValueError("Downloaded checkpoint is an HTML response, not model weights")
        verify_file(temporary, expected_size=expected_size, expected_sha256=expected_sha256)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    verify_file(destination, expected_size=expected_size, expected_sha256=expected_sha256)
    return destination


def _load_official_module(hst_repo: Path) -> ModuleType:
    source_path = Path(hst_repo) / "model" / "hst_model.py"
    module_name = f"covid_rars_official_hst_{sha256_file(source_path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official HST source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unwrap_state_dict(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Official HST checkpoint must contain a state dictionary")
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not raw or not all(isinstance(key, str) for key in raw):
        raise ValueError("Official HST checkpoint has invalid tensor keys")
    return dict(raw)


def load_verified_hst_model(
    *,
    model_name: str,
    checkpoint_path: Path,
    hst_repo: Path,
    seed: int,
) -> tuple[object, dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load HST") from exc

    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unsupported HST model {model_name!r}; expected one of {sorted(ARCHITECTURES)}")
    checkpoint_spec = CHECKPOINTS[model_name]
    verify_file(
        checkpoint_path,
        expected_size=int(checkpoint_spec["size_bytes"]),
        expected_sha256=str(checkpoint_spec["sha256"]),
    )
    commit = verify_hst_source(hst_repo, PINNED_HST_COMMIT)
    official = _load_official_module(Path(hst_repo))
    architecture = dict(ARCHITECTURES[model_name])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = official.HSTModel(**architecture)

    raw = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(raw)
    head_keys = {"head.weight", "head.bias"}
    if not head_keys.issubset(state):
        raise ValueError(f"Official checkpoint is missing expected classification head: {sorted(head_keys - set(state))}")
    for key in head_keys:
        state.pop(key)
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("Official HST checkpoint contains non-tensor state values")

    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != head_keys or unexpected:
        raise ValueError(
            f"HST backbone key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        torch.nn.init.trunc_normal_(model.head.weight, std=0.02)
        torch.nn.init.zeros_(model.head.bias)

    if tuple(model.patch_parts.projection.weight.shape) != (96, 3, 4, 4):
        raise ValueError("Unexpected HST patch-projection shape")
    if tuple(model.head.weight.shape) != (2, 768):
        raise ValueError("Unexpected HST classification-head shape")
    backbone_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if not name.startswith("head.")
    )
    expected_backbone = EXPECTED_BACKBONE_PARAMETERS[model_name]
    if backbone_parameters != expected_backbone:
        raise ValueError(
            f"HST backbone parameter mismatch: expected {expected_backbone}, found {backbone_parameters}"
        )

    audit: dict[str, object] = {
        "source_commit": commit,
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "checkpoint_size_bytes": Path(checkpoint_path).stat().st_size,
        "checkpoint_tensor_count": len(state) + len(head_keys),
        "checkpoint_element_count_without_head": int(sum(value.numel() for value in state.values())),
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "backbone_parameter_count": int(backbone_parameters),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "head_reinitialized": True,
        "head_initialization_seed": int(seed),
        "architecture": architecture,
    }
    return model, audit
