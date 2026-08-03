#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from covid_audio_btp.hst_checkpoint import load_verified_hst_model
from covid_audio_btp.hst_training import make_hst_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one isolated HST-Base resource trial")
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _atomic_result(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite_parameters(model: object) -> bool:
    import torch

    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())  # type: ignore[attr-defined]


def execute(job: dict[str, object]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("HST resource pilot requires CUDA")
    seed = int(job["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    model, model_audit = load_verified_hst_model(
        model_name=str(job["model_name"]),
        checkpoint_path=Path(str(job["checkpoint_path"])),
        hst_repo=Path(str(job["hst_repo"])),
        seed=seed,
    )
    model = model.to("cuda")  # type: ignore[attr-defined]
    cache = pd.read_csv(str(job["cache_index_path"]))
    manifest = pd.read_csv(str(job["manifest_path"]))
    batch_size = int(job["physical_batch_size"])
    accumulation = int(job["gradient_accumulation"])
    precision = str(job["precision"])
    requested_updates = int(job["optimizer_updates"])
    loaders = make_hst_dataloaders(
        cache,
        manifest,
        fold=int(job["fold"]),
        modality=str(job["modality"]),
        physical_batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )
    representative_loader = loaders["train_factory"](1)
    optimizer_updates_per_epoch = int(math.ceil(len(representative_loader) / accumulation))
    if optimizer_updates_per_epoch <= 0:
        raise RuntimeError("Resource pilot produced no optimizer updates per epoch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)  # type: ignore[attr-defined]
    use_amp = precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    skipped = 0
    finite_loss = True
    finite_gradients = True
    completed_updates = 0
    epoch = 1
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    accumulated_samples = 0

    while completed_updates < requested_updates:
        loader = loaders["train_factory"](epoch)
        for batch_index, (images, labels, _) in enumerate(loader, start=1):
            images = images.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            sample_count = int(labels.shape[0])
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                mean_loss = torch.nn.functional.cross_entropy(logits, labels)
                summed_loss = mean_loss * sample_count
            finite_loss = finite_loss and bool(torch.isfinite(mean_loss))
            scaler.scale(summed_loss).backward()
            accumulated_samples += sample_count
            boundary = batch_index % accumulation == 0 or batch_index == len(loader)
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            gradients_finite = True
            for parameter in model.parameters():  # type: ignore[attr-defined]
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
                    gradients_finite = gradients_finite and bool(torch.isfinite(parameter.grad).all())
            finite_gradients = finite_gradients and gradients_finite
            if gradients_finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # type: ignore[attr-defined]
            old_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            if use_amp and float(scaler.get_scale()) < old_scale:
                skipped += 1
            else:
                completed_updates += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
            if completed_updates >= requested_updates or skipped > 0:
                break
        epoch += 1
        if epoch > 1000:
            raise RuntimeError("Resource pilot could not reach requested optimizer updates")
        if skipped > 0:
            break

    model.eval()  # type: ignore[attr-defined]
    validation_batch = next(iter(loaders["validation"]))
    images, labels, _ = validation_batch
    images = images.to("cuda", non_blocking=True)
    labels = labels.to("cuda", non_blocking=True)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(images)
        evaluation_loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
    torch.cuda.synchronize()
    free_vram, total_vram = torch.cuda.mem_get_info()
    elapsed = time.perf_counter() - started
    valid = (
        completed_updates >= requested_updates
        and skipped == 0
        and finite_loss
        and finite_gradients
        and _finite_parameters(model)
        and bool(torch.isfinite(probabilities).all())
        and math.isfinite(float(evaluation_loss))
    )
    return {
        "schema_version": 1,
        "physical_batch_size": batch_size,
        "precision": precision,
        "valid": bool(valid),
        "completed_optimizer_updates": completed_updates,
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "skipped_optimizer_updates": skipped,
        "seconds": elapsed,
        "free_vram_bytes": int(free_vram),
        "total_vram_bytes": int(total_vram),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "finite_loss": bool(finite_loss and math.isfinite(float(evaluation_loss))),
        "finite_gradients": bool(finite_gradients),
        "finite_parameters": bool(_finite_parameters(model)),
        "finite_predictions": bool(torch.isfinite(probabilities).all()),
        "evaluation_loss": float(evaluation_loss),
        "evaluation_probabilities": [float(value) for value in probabilities.cpu().tolist()],
        "model_source_commit": model_audit["source_commit"],
        "source_checkpoint_sha256": model_audit["checkpoint_sha256"],
    }


def main() -> None:
    args = parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    try:
        result = execute(job)
    except BaseException as exc:
        result = {
            **job,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _atomic_result(args.result_json, result)
    if not bool(result.get("valid")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
