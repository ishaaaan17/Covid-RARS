from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_rars.metrics import labels_to_binary
from covid_rars.sota_predictions import aggregate_sota_predictions, evaluate_sota_prediction_table
from covid_rars.sota_segments import load_sota_segment_waveform


@dataclass(frozen=True)
class SOTASSLResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    history: pd.DataFrame


def _filter_branch_segments(segment_index: pd.DataFrame, modality: str) -> pd.DataFrame:
    df = segment_index.copy()
    df = df[df["modality"].astype(str).eq(str(modality))]
    df = df[df["label_binary"].isin(["positive", "negative"])]
    df = df[df["split"].astype(str).isin(["train", "validation", "test", "external_test"])]
    if df.empty:
        raise ValueError(f"No SOTA segments available for modality={modality}")
    return df.reset_index(drop=True)


def _waveform_stats(row: pd.Series, target_samples: int) -> dict[str, float]:
    y = load_sota_segment_waveform(row, target_samples=target_samples)
    abs_y = np.abs(y)
    zcr = np.mean(np.abs(np.diff(np.signbit(y).astype(np.int8)))) if y.size > 1 else 0.0
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "abs_mean": float(np.mean(abs_y)),
        "abs_max": float(np.max(abs_y)) if y.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0,
        "zcr": float(zcr),
        "q25": float(np.percentile(y, 25)),
        "q75": float(np.percentile(y, 75)),
    }


def train_debug_acoustic_branch(
    segment_index: pd.DataFrame,
    modality: str,
    target_samples: int = 48000,
    model_name: str = "debug_acoustic",
) -> SOTASSLResult:
    """Fast smoke backend for validating SOTA branch plumbing without heavy models."""
    df = _filter_branch_segments(segment_index, modality)
    feature_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        feature_rows.append({**row.to_dict(), **_waveform_stats(row, target_samples=target_samples)})
    features = pd.DataFrame(feature_rows)
    train = features[features["split"].eq("train")].copy()
    eval_df = features[~features["split"].eq("train")].copy()
    feature_cols = ["mean", "std", "abs_mean", "abs_max", "rms", "zcr", "q25", "q75"]
    if train.empty or eval_df.empty:
        raise ValueError(f"Need train and evaluation segments for modality={modality}")

    model = None
    if train["label_binary"].nunique() >= 2:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(train[feature_cols], labels_to_binary(train["label_binary"]))
        probabilities = model.predict_proba(eval_df[feature_cols])[:, 1]
    else:
        fallback = float(labels_to_binary(train["label_binary"]).mean()) if not train.empty else 0.5
        probabilities = np.full(len(eval_df), fallback, dtype=float)

    predictions = eval_df[
        [
            "segment_id",
            "recording_id",
            "participant_id",
            "dataset",
            "modality",
            "submodality",
            "label_binary",
            "split",
        ]
    ].copy()
    predictions["model_name"] = f"{model_name}_{modality}"
    predictions["analysis_family"] = "sota_ssl_branch"
    predictions["evaluation_protocol"] = "sota_internal_protocol"
    predictions["probability"] = probabilities
    participant_predictions = aggregate_sota_predictions(predictions, level="participant")
    metrics = evaluate_sota_prediction_table(participant_predictions)
    history = pd.DataFrame(
        [
            {
                "backend": "debug_acoustic",
                "modality": modality,
                "model_name": f"{model_name}_{modality}",
                "n_train_segments": float(len(train)),
                "n_eval_segments": float(len(eval_df)),
                "trained_classifier": bool(model is not None),
            }
        ]
    )
    return SOTASSLResult(metrics=metrics, predictions=participant_predictions, history=history)


def _require_hf_stack():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, get_cosine_schedule_with_warmup
    except Exception as exc:  # pragma: no cover - optional heavy dependencies
        raise RuntimeError(
            "The hf_ssl backend requires torch and transformers. Install the GPU requirements "
            "before running the serious SOTA branch."
        ) from exc
    return torch, nn, DataLoader, Dataset, AutoModel, get_cosine_schedule_with_warmup


def _freeze_ssl_layers(model: object, unfreeze_top_layers: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
    feature_extractor = getattr(model, "feature_extractor", None)
    if feature_extractor is not None:
        for param in feature_extractor.parameters():
            param.requires_grad = False
    encoder = getattr(model, "encoder", None)
    layers = list(getattr(encoder, "layers", []) or [])
    if not layers:
        for param in model.parameters():
            param.requires_grad = True
        return
    keep = max(0, min(int(unfreeze_top_layers), len(layers)))
    for layer in layers[-keep:]:
        for param in layer.parameters():
            param.requires_grad = True


def train_hf_ssl_branch(
    segment_index: pd.DataFrame,
    modality: str,
    model_name: str = "microsoft/wavlm-base-plus",
    max_epochs: int = 8,
    batch_size: int = 2,
    gradient_accumulation: int = 16,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    unfreeze_top_layers: int = 4,
    target_samples: int = 48000,
    patience: int = 3,
    device: str | None = None,
    model_output: Path | None = None,
) -> SOTASSLResult:
    """Fine-tune a waveform SSL encoder on segment rows for one modality."""
    torch, nn, DataLoader, Dataset, AutoModel, get_cosine_schedule_with_warmup = _require_hf_stack()

    class SegmentDataset(Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int):
            row = self.frame.iloc[idx]
            y = load_sota_segment_waveform(row, target_samples=target_samples)
            label = 1.0 if row["label_binary"] == "positive" else 0.0
            return torch.from_numpy(y), torch.tensor(label, dtype=torch.float32), row.to_dict()

    class SSLClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            _freeze_ssl_layers(self.encoder, unfreeze_top_layers=unfreeze_top_layers)
            hidden = int(getattr(self.encoder.config, "hidden_size", getattr(self.encoder.config, "d_model", 768)))
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(0.25),
                nn.Linear(hidden, 256),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(256, 1),
            )

        def forward(self, input_values):
            outputs = self.encoder(input_values=input_values)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
            return self.classifier(pooled).squeeze(-1)

    def collate(batch):
        audio, labels, rows = zip(*batch)
        return torch.stack(audio, dim=0), torch.stack(labels, dim=0), list(rows)

    def collect(model, loader, device_name: str) -> pd.DataFrame:
        model.eval()
        pred_rows: list[dict[str, object]] = []
        with torch.no_grad():
            for audio, _, rows in loader:
                audio = audio.to(device_name)
                logits = model(audio).detach().float().cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                for row, prob, logit in zip(rows, probs, logits):
                    pred_rows.append(
                        {
                            "segment_id": row["segment_id"],
                            "recording_id": row["recording_id"],
                            "participant_id": row["participant_id"],
                            "dataset": row.get("dataset", "unknown"),
                            "modality": row.get("modality", modality),
                            "submodality": row.get("submodality", modality),
                            "label_binary": row["label_binary"],
                            "split": row["split"],
                            "model_name": model_name,
                            "analysis_family": "sota_ssl_branch",
                            "evaluation_protocol": "sota_internal_protocol",
                            "logit": float(logit),
                            "probability": float(prob),
                        }
                    )
        return pd.DataFrame(pred_rows)

    df = _filter_branch_segments(segment_index, modality)
    train = df[df["split"].eq("train")].copy()
    validation = df[df["split"].eq("validation")].copy()
    test = df[df["split"].eq("test")].copy()
    external = df[df["split"].eq("external_test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Need train/validation/test segments for modality={modality}")
    if train["label_binary"].nunique() < 2:
        raise ValueError(f"Train split for modality={modality} does not contain both classes")

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLClassifier().to(device_name)
    train_loader = DataLoader(SegmentDataset(train), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SegmentDataset(validation), batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(SegmentDataset(test), batch_size=batch_size, shuffle=False, collate_fn=collate)
    external_loader = (
        DataLoader(SegmentDataset(external), batch_size=batch_size, shuffle=False, collate_fn=collate)
        if not external.empty
        else None
    )

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate},
            {"params": head_params, "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    total_steps = max(1, int(np.ceil(len(train_loader) / max(1, gradient_accumulation))) * int(max_epochs))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    labels = labels_to_binary(train["label_binary"])
    pos_weight = torch.tensor([(len(labels) - labels.sum()) / max(1, labels.sum())], dtype=torch.float32, device=device_name)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val = -np.inf
    bad_epochs = 0
    history_rows: list[dict[str, object]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, (audio, y, _) in enumerate(train_loader, start=1):
            audio = audio.to(device_name)
            y = y.to(device_name)
            logits = model(audio)
            loss = criterion(logits, y) / max(1, gradient_accumulation)
            loss.backward()
            losses.append(float(loss.detach().cpu()) * max(1, gradient_accumulation))
            if step % max(1, gradient_accumulation) == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_pred = aggregate_sota_predictions(collect(model, val_loader, device_name), level="participant")
        val_metrics = evaluate_sota_prediction_table(
            pd.concat([val_pred, val_pred.assign(split="test")], ignore_index=True, sort=False)
        )
        val_auroc = float(pd.to_numeric(val_metrics.get("auroc"), errors="coerce").max()) if not val_metrics.empty else float("nan")
        history_rows.append(
            {
                "backend": "hf_ssl",
                "model_name": model_name,
                "modality": modality,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "validation_auroc": val_auroc,
            }
        )
        improved = np.isfinite(val_auroc) and val_auroc > best_val
        if improved:
            best_val = val_auroc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "model_name": model_name, "modality": modality}, model_output)

    prediction_frames = [collect(model, val_loader, device_name), collect(model, test_loader, device_name)]
    if external_loader is not None:
        prediction_frames.append(collect(model, external_loader, device_name))
    segment_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    participant_predictions = aggregate_sota_predictions(segment_predictions, level="participant")
    metrics = evaluate_sota_prediction_table(participant_predictions)
    history = pd.DataFrame(history_rows)
    return SOTASSLResult(metrics=metrics, predictions=participant_predictions, history=history)


def train_sota_ssl_branch(
    segment_index: pd.DataFrame,
    modality: str,
    backend: str = "hf_ssl",
    model_name: str = "microsoft/wavlm-base-plus",
    target_samples: int = 48000,
    **kwargs,
) -> SOTASSLResult:
    if backend == "debug_acoustic":
        return train_debug_acoustic_branch(
            segment_index,
            modality=modality,
            target_samples=target_samples,
            model_name="debug_acoustic",
        )
    if backend == "hf_ssl":
        return train_hf_ssl_branch(
            segment_index,
            modality=modality,
            model_name=model_name,
            target_samples=target_samples,
            **kwargs,
        )
    raise ValueError(f"Unknown SOTA SSL backend: {backend}")
