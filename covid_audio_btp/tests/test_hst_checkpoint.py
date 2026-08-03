from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_official_hst_paths_use_prepared_checkpoint_cache() -> None:
    from tests.hst_test_helpers import official_hst_paths

    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parent
    checkpoint, hst_repo = official_hst_paths()

    assert checkpoint == (
        package_root / ".cache" / "hst" / "checkpoints" / "hst_base_imagenet.pth"
    )
    assert hst_repo == repository_root / "HST"
    assert "imagenet_weights" not in checkpoint.parts


def test_hst_config_pins_source_and_primary_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "hst_reliability.json").read_text(encoding="utf-8"))
    assert config["source"]["commit"] == "7f94ad81e392da856c7aac6d364d036c28e26c32"
    small = config["checkpoints"]["hst_small_imagenet"]
    assert small["sha256"] == "e7086d1b87d598120296b9a1b5f094c7587cb06f50bf609a4ca13badc95e3112"
    assert small["size_bytes"] == 111266629
    assert small["google_drive_file_id"] == "1MHSIBpM3-pa2xXKSrk5oEDTvlhIaC_M3"
    base = config["checkpoints"]["hst_base_imagenet"]
    assert base["sha256"] == "f39f001d5f8cd90cb78d45612486202a4ea280e23df0b2c1d6ce35d96b30cce4"
    assert base["size_bytes"] == 197063145
    assert base["google_drive_file_id"] == "1jol7869ixS77FyoAXzb_m3oJGTtKuOVO"
    assert config["experiment"]["primary_model"] == "hst_base"
    assert config["class_to_index"] == {"negative": 0, "positive": 1}
    assert config["training"]["selection_metric"] == "participant_auroc"
    assert config["training"]["train_all_epochs"] is True


def test_verify_file_rejects_wrong_hash(tmp_path: Path) -> None:
    from covid_audio_btp.hst_checkpoint import verify_file

    path = tmp_path / "weights.pth"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_file(path, expected_size=16, expected_sha256="0" * 64)


def test_download_verified_checkpoint_reuses_valid_regular_file_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_checkpoint import download_verified_checkpoint

    payload = b"verified checkpoint bytes"
    destination = tmp_path / "weights.pth"
    destination.write_bytes(payload)

    monkeypatch.setitem(__import__("sys").modules, "gdown", None)

    result = download_verified_checkpoint(
        google_drive_file_id="unused",
        destination=destination,
        expected_size=len(payload),
        expected_sha256=_sha256(payload),
    )

    assert result == destination
    assert result.read_bytes() == payload


def test_download_verified_checkpoint_atomically_replaces_invalid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_checkpoint import download_verified_checkpoint

    payload = b"new verified checkpoint"
    destination = tmp_path / "weights.pth"
    destination.write_bytes(b"corrupt cache")
    observed_destination_bytes: list[bytes] = []

    def download(*, id: str, output: str, quiet: bool) -> str:
        assert id == "checkpoint-id"
        assert quiet is False
        observed_destination_bytes.append(destination.read_bytes())
        Path(output).write_bytes(payload)
        observed_destination_bytes.append(destination.read_bytes())
        return output

    monkeypatch.setitem(
        __import__("sys").modules,
        "gdown",
        SimpleNamespace(download=download),
    )

    result = download_verified_checkpoint(
        google_drive_file_id="checkpoint-id",
        destination=destination,
        expected_size=len(payload),
        expected_sha256=_sha256(payload),
    )

    assert observed_destination_bytes == [b"corrupt cache", b"corrupt cache"]
    assert result.read_bytes() == payload
    assert not list(tmp_path.glob(".weights.pth.*.download"))


def test_download_verified_checkpoint_rejects_downloader_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_checkpoint import download_verified_checkpoint

    payload = b"valid bytes behind an unsafe link"
    destination = tmp_path / "weights.pth"
    original_is_symlink = Path.is_symlink

    def download(*, output: str, **_: object) -> str:
        Path(output).write_bytes(payload)
        return output

    def classify_download_as_symlink(path: Path) -> bool:
        if path.name.startswith(".weights.pth.") and path.name.endswith(".download"):
            return True
        return original_is_symlink(path)

    monkeypatch.setitem(
        __import__("sys").modules,
        "gdown",
        SimpleNamespace(download=download),
    )
    monkeypatch.setattr(Path, "is_symlink", classify_download_as_symlink)

    with pytest.raises(ValueError, match="regular file"):
        download_verified_checkpoint(
            google_drive_file_id="checkpoint-id",
            destination=destination,
            expected_size=len(payload),
            expected_sha256=_sha256(payload),
        )

    assert not destination.exists()


def test_download_verified_checkpoint_never_reuses_stale_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_audio_btp.hst_checkpoint import download_verified_checkpoint

    payload = b"verified checkpoint bytes"
    destination = tmp_path / "weights.pth"
    stale_partial = tmp_path / ".weights.pth.stale.download"
    stale_partial.write_bytes(payload)
    calls = 0

    def download(*, output: str, **_: object) -> str:
        nonlocal calls
        calls += 1
        Path(output).write_bytes(payload)
        return output

    monkeypatch.setitem(
        __import__("sys").modules,
        "gdown",
        SimpleNamespace(download=download),
    )

    result = download_verified_checkpoint(
        google_drive_file_id="checkpoint-id",
        destination=destination,
        expected_size=len(payload),
        expected_sha256=_sha256(payload),
    )

    assert calls == 1
    assert result.read_bytes() == payload
    assert stale_partial.read_bytes() == payload


def test_verify_hst_source_accepts_only_pinned_commit() -> None:
    from covid_audio_btp.hst_checkpoint import verify_hst_source

    repo = Path(__file__).resolve().parents[2] / "HST"
    commit = verify_hst_source(repo, "7f94ad81e392da856c7aac6d364d036c28e26c32")
    assert commit == "7f94ad81e392da856c7aac6d364d036c28e26c32"
    with pytest.raises(ValueError, match="commit"):
        verify_hst_source(repo, "0" * 40)


def test_official_model_source_hash_is_pinned() -> None:
    from covid_audio_btp.hst_checkpoint import EXPECTED_HST_MODEL_SOURCE_SHA256

    assert EXPECTED_HST_MODEL_SOURCE_SHA256 == (
        "44c1688afb00ee3f7632577d011ca3857200d042818bc2ac28b3b8d18288479f"
    )


def test_official_checkpoint_files_match_frozen_hashes() -> None:
    from covid_audio_btp.hst_checkpoint import verify_file

    checkpoint_root = (
        Path(__file__).resolve().parents[1] / ".cache" / "hst" / "checkpoints"
    )
    verify_file(
        checkpoint_root / "hst_small_imagenet.pth",
        expected_size=111266629,
        expected_sha256="e7086d1b87d598120296b9a1b5f094c7587cb06f50bf609a4ca13badc95e3112",
    )
    verify_file(
        checkpoint_root / "hst_base_imagenet.pth",
        expected_size=197063145,
        expected_sha256="f39f001d5f8cd90cb78d45612486202a4ea280e23df0b2c1d6ce35d96b30cce4",
    )


def test_load_verified_model_reinitializes_only_head() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from covid_audio_btp.hst_checkpoint import load_verified_hst_model
    from tests.hst_test_helpers import (
        expected_backbone_parameter_count,
        expected_base_architecture,
        official_hst_paths,
    )

    checkpoint, hst_repo = official_hst_paths()
    model, audit = load_verified_hst_model(
        model_name="hst_base",
        checkpoint_path=checkpoint,
        hst_repo=hst_repo,
        seed=42,
    )
    assert set(audit["missing_keys"]) == {"head.bias", "head.weight"}
    assert set(audit["unexpected_keys"]) == set()
    assert audit["head_reinitialized"] is True
    assert audit["architecture"] == expected_base_architecture()
    assert audit["backbone_parameter_count"] == expected_backbone_parameter_count()
    logits = model(torch.zeros(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 2)
    assert torch.isfinite(logits).all()
