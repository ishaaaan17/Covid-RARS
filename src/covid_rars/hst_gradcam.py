from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image


GRADCAM_SELECTION_RULE = (
    "within one frozen protocol/fold/split and outcome cell, select descending "
    "classification confidence then participant_key and recording_key"
)


def token_gradients_to_heatmap(
    activations: np.ndarray,
    gradients: np.ndarray,
    *,
    spatial_shape: tuple[int, int] = (7, 7),
    output_shape: tuple[int, int] = (224, 224),
) -> np.ndarray:
    activations = np.asarray(activations, dtype=np.float32)
    gradients = np.asarray(gradients, dtype=np.float32)
    if activations.ndim != 2 or gradients.ndim != 2:
        raise ValueError("HST Grad-CAM activations and gradients must be token-by-channel matrices")
    if activations.shape != gradients.shape:
        raise ValueError("HST Grad-CAM activation and gradient shapes must match")
    expected_tokens = int(spatial_shape[0]) * int(spatial_shape[1])
    if activations.shape[0] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} HST tokens for spatial shape {spatial_shape}, "
            f"received {activations.shape[0]}"
        )
    if not np.isfinite(activations).all() or not np.isfinite(gradients).all():
        raise ValueError("HST Grad-CAM inputs must be finite")
    if min(*spatial_shape, *output_shape) <= 0:
        raise ValueError("Grad-CAM spatial dimensions must be positive")

    channel_weights = gradients.mean(axis=0)
    camera = np.maximum(activations @ channel_weights, 0.0).reshape(spatial_shape)
    minimum = float(camera.min())
    maximum = float(camera.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros(output_shape, dtype=np.float32)
    normalized = ((camera - minimum) / (maximum - minimum)).astype(np.float32)
    resized = Image.fromarray(normalized, mode="F").resize(
        (int(output_shape[1]), int(output_shape[0])),
        resample=Image.Resampling.BILINEAR,
    )
    heatmap = np.asarray(resized, dtype=np.float32)
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32, copy=False)


def _binary_labels(values: pd.Series) -> pd.Series:
    aliases = {
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
        "negative": 0,
        "positive": 1,
    }
    normalized = values.map(aliases)
    if normalized.isna().any():
        invalid = sorted(values.loc[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Grad-CAM labels must be binary; invalid values: {invalid}")
    return normalized.astype(int)


def select_gradcam_examples(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    per_cell: int = 1,
) -> pd.DataFrame:
    required = {
        "participant_key",
        "recording_key",
        "label_binary",
        "probability",
        "split",
        "fold",
        "protocol",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Grad-CAM predictions are missing columns: {missing}")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if int(per_cell) <= 0:
        raise ValueError("per_cell must be positive")
    frame = predictions.copy()
    for column in ("protocol", "fold", "split"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"Grad-CAM example selection requires exactly one {column}")
    if frame.duplicated(["participant_key", "recording_key"]).any():
        raise ValueError("Grad-CAM prediction keys must be unique")
    frame["label_binary"] = _binary_labels(frame["label_binary"])
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    if not np.isfinite(frame["probability"]).all() or not frame["probability"].between(0, 1).all():
        raise ValueError("Grad-CAM probabilities must be finite and within [0, 1]")
    predicted = frame["probability"].ge(float(threshold)).astype(int)
    frame["outcome"] = np.select(
        [
            frame["label_binary"].eq(1) & predicted.eq(1),
            frame["label_binary"].eq(0) & predicted.eq(0),
            frame["label_binary"].eq(0) & predicted.eq(1),
            frame["label_binary"].eq(1) & predicted.eq(0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="invalid",
    )
    frame["selection_confidence"] = np.where(
        predicted.eq(1), frame["probability"], 1.0 - frame["probability"]
    )
    rows: list[pd.DataFrame] = []
    for outcome in ("TP", "TN", "FP", "FN"):
        cell = frame.loc[frame["outcome"].eq(outcome)].sort_values(
            ["selection_confidence", "participant_key", "recording_key"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        if not cell.empty:
            rows.append(cell.head(int(per_cell)))
    if not rows:
        return pd.DataFrame(columns=[*predictions.columns, "outcome", "selection_confidence", "selection_rule"])
    selected = pd.concat(rows, ignore_index=True, sort=False)
    selected["selection_rule"] = GRADCAM_SELECTION_RULE
    return selected


@dataclass(frozen=True)
class GradCAMGroupSummary:
    participant_heatmaps: pd.DataFrame
    negative_mean: np.ndarray
    positive_mean: np.ndarray
    mean_difference: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    bootstrap_replicates: int
    bootstrap_seed: int
    resampling_unit: str = "participant_key"


@dataclass(frozen=True)
class StageEmbeddingFigure:
    output_path: Path
    coordinates: pd.DataFrame
    method: str
    seed: int


def build_participant_gradcam_summary(
    heatmaps: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> GradCAMGroupSummary:
    required = {
        "participant_key",
        "recording_key",
        "label_binary",
        "outcome",
        "heatmap",
        "threshold",
        "threshold_source",
        "protocol",
        "fold",
        "split",
    }
    missing = sorted(required - set(heatmaps.columns))
    if missing:
        raise ValueError(f"Grad-CAM group summary is missing columns: {missing}")
    if int(bootstrap_replicates) <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    frame = heatmaps.copy()
    for column in ("threshold", "threshold_source", "protocol", "fold", "split"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(
                f"Grad-CAM group summary requires one frozen {column} context"
            )
    frame["label_binary"] = _binary_labels(frame["label_binary"])
    frame = frame.loc[frame["outcome"].isin(["TP", "TN"])].copy()
    if frame.empty:
        raise ValueError("Grad-CAM group summary has no correctly classified examples")
    if frame.duplicated(["participant_key", "recording_key"]).any():
        raise ValueError("Grad-CAM heatmaps must be unique by participant and recording")
    participant_labels = frame.groupby("participant_key", sort=False)[
        "label_binary"
    ].nunique(dropna=False)
    if participant_labels.gt(1).any():
        raise ValueError("A Grad-CAM participant has conflicting labels")

    expected_shape: tuple[int, ...] | None = None
    normalized_maps: list[np.ndarray] = []
    for value in frame["heatmap"]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Grad-CAM heatmaps must be finite two-dimensional arrays")
        if (array < 0.0).any() or (array > 1.0).any():
            raise ValueError("Grad-CAM heatmaps must lie within [0, 1]")
        expected_shape = expected_shape or array.shape
        if array.shape != expected_shape:
            raise ValueError("Grad-CAM heatmaps must share one spatial shape")
        normalized_maps.append(array)
    frame["heatmap"] = normalized_maps

    participant_rows: list[dict[str, object]] = []
    for participant_key, group in frame.groupby("participant_key", sort=True):
        participant_rows.append(
            {
                "participant_key": participant_key,
                "label_binary": int(group["label_binary"].iloc[0]),
                "n_recordings": int(group["recording_key"].nunique()),
                "heatmap": np.mean(np.stack(group["heatmap"].tolist()), axis=0).astype(
                    np.float32
                ),
            }
        )
    participants = pd.DataFrame(participant_rows)
    class_arrays: dict[int, np.ndarray] = {}
    for label in (0, 1):
        values = participants.loc[participants["label_binary"].eq(label), "heatmap"]
        if values.empty:
            raise ValueError("Grad-CAM group summary requires both outcome classes")
        class_arrays[label] = np.stack(values.tolist()).astype(np.float32)
    negative_mean = class_arrays[0].mean(axis=0).astype(np.float32)
    positive_mean = class_arrays[1].mean(axis=0).astype(np.float32)
    difference = (positive_mean - negative_mean).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    replicates = np.empty(
        (int(bootstrap_replicates), *difference.shape),
        dtype=np.float32,
    )
    for index in range(int(bootstrap_replicates)):
        negative = class_arrays[0][
            rng.integers(0, len(class_arrays[0]), size=len(class_arrays[0]))
        ].mean(axis=0)
        positive = class_arrays[1][
            rng.integers(0, len(class_arrays[1]), size=len(class_arrays[1]))
        ].mean(axis=0)
        replicates[index] = positive - negative
    ci_low = np.quantile(replicates, 0.025, axis=0, method="linear").astype(np.float32)
    ci_high = np.quantile(replicates, 0.975, axis=0, method="linear").astype(np.float32)
    return GradCAMGroupSummary(
        participant_heatmaps=participants,
        negative_mean=negative_mean,
        positive_mean=positive_mean,
        mean_difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=int(seed),
    )


def _safe_artifact_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not token:
        raise ValueError("Grad-CAM artifact key is empty after normalization")
    return token[:120]


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.png")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlay_image(image: object, heatmap: np.ndarray) -> Image.Image:
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 3:
        source = source.mean(axis=2)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("Grad-CAM source images must be finite two-dimensional arrays")
    source_min = float(source.min())
    source_max = float(source.max())
    if source_max > source_min:
        source = (source - source_min) / (source_max - source_min)
    else:
        source = np.zeros_like(source)
    base = Image.fromarray(np.uint8(np.clip(source, 0.0, 1.0) * 255), mode="L").convert("RGB")
    if base.size != (heatmap.shape[1], heatmap.shape[0]):
        base = base.resize(
            (heatmap.shape[1], heatmap.shape[0]),
            resample=Image.Resampling.BILINEAR,
        )
    red = np.zeros((*heatmap.shape, 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = np.uint8(np.clip(heatmap, 0.0, 1.0) * 150)
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(red, mode="RGBA")).convert("RGB")


def build_gradcam_evidence(
    model: object,
    examples: pd.DataFrame,
    *,
    output_dir: Path,
    camera_factory: Callable[[object], object] | None = None,
) -> pd.DataFrame:
    """Generate deterministic, file-backed Grad-CAM evidence for frozen examples."""
    required = {
        "participant_key",
        "recording_key",
        "label_binary",
        "probability",
        "outcome",
        "threshold",
        "threshold_source",
        "protocol",
        "fold",
        "split",
        "model_input",
        "image",
    }
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError(f"Grad-CAM examples are missing columns: {missing}")
    if examples.empty:
        raise ValueError("Grad-CAM evidence requires at least one selected example")
    frame = examples.copy()
    for column in ("threshold", "threshold_source", "protocol", "fold", "split"):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"Grad-CAM evidence requires one frozen {column} context")
    if frame.duplicated(["participant_key", "recording_key"]).any():
        raise ValueError("Grad-CAM evidence keys must be unique")
    frame["label_binary"] = _binary_labels(frame["label_binary"])
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    if not frame["outcome"].isin(["TP", "TN", "FP", "FN"]).all():
        raise ValueError("Grad-CAM outcomes must be TP, TN, FP, or FN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = camera_factory or HSTGradCAM
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values(
        ["outcome", "participant_key", "recording_key"],
        kind="mergesort",
    )
    with factory(model) as camera:  # type: ignore[attr-defined]
        for row in ordered.itertuples(index=False):
            target_class = int(float(row.probability) >= float(row.threshold))
            heatmap = np.asarray(
                camera.generate(row.model_input, target_class=target_class),  # type: ignore[attr-defined]
                dtype=np.float32,
            )
            if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
                raise ValueError("Generated Grad-CAM heatmaps must be finite 2-D arrays")
            if (heatmap < 0.0).any() or (heatmap > 1.0).any():
                raise ValueError("Generated Grad-CAM heatmaps must lie within [0, 1]")
            stem = "__".join(
                (
                    _safe_artifact_token(row.outcome),
                    _safe_artifact_token(row.participant_key),
                    _safe_artifact_token(row.recording_key),
                )
            )
            heatmap_path = output_dir / "heatmaps" / f"{stem}.npy"
            overlay_path = output_dir / "overlays" / f"{stem}.png"
            _atomic_save_array(heatmap_path, heatmap)
            _atomic_save_image(overlay_path, _overlay_image(row.image, heatmap))
            row_values = row._asdict()
            row_values.pop("model_input", None)
            row_values.pop("image", None)
            rows.append(
                {
                    **row_values,
                    "target_class": target_class,
                    "zero_map": bool(float(heatmap.max()) <= 0.0),
                    "heatmap_path": heatmap_path.relative_to(output_dir).as_posix(),
                    "overlay_path": overlay_path.relative_to(output_dir).as_posix(),
                    "heatmap_sha256": _sha256_file(heatmap_path),
                    "overlay_sha256": _sha256_file(overlay_path),
                    "target_layer": "layers[-1].HSTblocks[-1].attn2:input",
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["outcome", "participant_key", "recording_key"], kind="mergesort"
    ).reset_index(drop=True)
    manifest_path = output_dir / "gradcam_evidence_manifest.csv"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        manifest.to_csv(temporary, index=False)
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def build_stage_embedding_figure(
    embeddings: pd.DataFrame,
    *,
    output_path: Path,
    method: str,
    seed: int,
) -> StageEmbeddingFigure:
    """Project participant-level held-out stage embeddings without using labels to fit."""
    required = {
        "participant_key",
        "label_binary",
        "split",
        "stage",
        "dimension",
        "value",
    }
    missing = sorted(required - set(embeddings.columns))
    if missing:
        raise ValueError(f"Stage embeddings are missing columns: {missing}")
    frame = embeddings.copy()
    if frame.empty:
        raise ValueError("Stage embedding figure requires non-empty embeddings")
    if frame["split"].astype(str).str.casefold().eq("train").any():
        raise ValueError("Stage embedding figures are restricted to held-out rows")
    if frame["stage"].nunique(dropna=False) != 1:
        raise ValueError("Stage embedding figure requires one HST stage")
    frame["label_binary"] = _binary_labels(frame["label_binary"])
    frame["dimension"] = pd.to_numeric(frame["dimension"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if not np.isfinite(frame["value"]).all():
        raise ValueError("Stage embedding values must be finite")
    if frame.duplicated(["participant_key", "dimension"]).any():
        raise ValueError("Participant-stage embedding dimensions must be unique")
    labels = frame.groupby("participant_key", sort=True)["label_binary"].nunique()
    if labels.gt(1).any():
        raise ValueError("A participant has conflicting embedding labels")
    matrix = frame.pivot(
        index="participant_key", columns="dimension", values="value"
    ).sort_index()
    if matrix.isna().any().any() or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("Embedding projection requires a complete matrix with at least 3 participants and 2 dimensions")
    method = str(method).casefold()
    if method == "pca":
        from sklearn.decomposition import PCA

        projected = PCA(n_components=2, svd_solver="full").fit_transform(matrix.to_numpy())
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(2.0, float(matrix.shape[0] - 1) / 3.0))
        projected = TSNE(
            n_components=2,
            random_state=int(seed),
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(matrix.to_numpy())
    else:
        raise ValueError("Embedding method must be pca or tsne")
    label_map = (
        frame[["participant_key", "label_binary"]]
        .drop_duplicates()
        .set_index("participant_key")["label_binary"]
    )
    coordinates = pd.DataFrame(
        {
            "participant_key": matrix.index.astype(str),
            "label_binary": [int(label_map.loc[key]) for key in matrix.index],
            "component_1": projected[:, 0],
            "component_2": projected[:, 1],
            "stage": str(frame["stage"].iloc[0]),
            "method": method,
        }
    ).sort_values("participant_key", kind="mergesort").reset_index(drop=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    palette = {0: "#2a6fbb", 1: "#c73e32"}
    for label, name in ((0, "Negative"), (1, "Positive")):
        group = coordinates.loc[coordinates["label_binary"].eq(label)]
        axis.scatter(
            group["component_1"],
            group["component_2"],
            s=24,
            alpha=0.8,
            color=palette[label],
            label=name,
        )
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.set_title(f"Held-out {coordinates['stage'].iloc[0]} embeddings ({method.upper()})")
    axis.legend(frameon=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}{output_path.suffix}")
    try:
        figure.savefig(temporary, dpi=200)
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return StageEmbeddingFigure(
        output_path=output_path,
        coordinates=coordinates,
        method=method,
        seed=int(seed),
    )


def extract_stage_participant_embeddings(
    model: object,
    loader: object,
) -> pd.DataFrame:
    """Extract held-out HST stage outputs and average recordings per participant."""
    import torch

    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("HST stage embedding extraction requires model.layers")
    captured: dict[int, object] = {}
    handles = []
    was_training = bool(getattr(model, "training", False))

    def capture(stage_index: int) -> Callable[[object, object, object], None]:
        def hook(_module: object, _inputs: object, output: object) -> None:
            captured[stage_index] = output[0] if isinstance(output, (tuple, list)) else output

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(capture(index)))
    try:
        try:
            device = next(model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            device = torch.device("cpu")
        model.eval()  # type: ignore[attr-defined]
        recording_rows: list[dict[str, object]] = []
        with torch.no_grad():
            for batch in loader:  # type: ignore[union-attr]
                if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                    raise ValueError("Embedding loader batches must be (inputs, labels, metadata)")
                inputs, labels, metadata = batch
                if not isinstance(metadata, list) or len(metadata) != int(inputs.shape[0]):
                    raise ValueError("Embedding metadata must contain one row per input")
                captured.clear()
                model(inputs.to(device))  # type: ignore[operator, union-attr]
                if set(captured) != set(range(len(layers))):
                    raise RuntimeError("Not every HST stage emitted an embedding")
                label_values = labels.detach().cpu().numpy().astype(int)
                if not np.isin(label_values, [0, 1]).all():
                    raise ValueError("Embedding labels must be binary")
                for stage_index in range(len(layers)):
                    value = captured[stage_index]
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("HST stage outputs must be tensors")
                    stage_value = value.detach().float().cpu()
                    if stage_value.ndim == 3:
                        stage_value = stage_value.mean(dim=1)
                    elif stage_value.ndim > 2:
                        stage_value = stage_value.flatten(start_dim=1)
                    if stage_value.ndim != 2 or stage_value.shape[0] != len(metadata):
                        raise ValueError("HST stage embeddings have an invalid batch shape")
                    vectors = stage_value.numpy()
                    if not np.isfinite(vectors).all():
                        raise ValueError("HST stage embeddings must be finite")
                    for row_index, meta in enumerate(metadata):
                        if not isinstance(meta, dict):
                            raise ValueError("Embedding metadata rows must be dictionaries")
                        split = str(meta.get("split", ""))
                        if split.casefold() == "train" or not split:
                            raise ValueError("Stage embeddings are restricted to held-out rows")
                        participant_key = str(meta.get("participant_key", "")).strip()
                        recording_key = str(meta.get("recording_key", "")).strip()
                        if not participant_key or not recording_key:
                            raise ValueError("Embedding metadata requires participant and recording keys")
                        for dimension, scalar in enumerate(vectors[row_index]):
                            recording_rows.append(
                                {
                                    "participant_key": participant_key,
                                    "recording_key": recording_key,
                                    "label_binary": int(label_values[row_index]),
                                    "split": split,
                                    "stage": f"stage_{stage_index + 1}",
                                    "dimension": int(dimension),
                                    "value": float(scalar),
                                }
                            )
    finally:
        model.train(was_training)  # type: ignore[attr-defined]
        for handle in handles:
            handle.remove()
    recordings = pd.DataFrame(recording_rows)
    if recordings.empty:
        raise ValueError("Stage embedding loader produced no held-out rows")
    participant_labels = recordings.groupby("participant_key")["label_binary"].nunique()
    if participant_labels.gt(1).any():
        raise ValueError("A participant has conflicting embedding labels")
    recording_counts = (
        recordings[["participant_key", "recording_key"]]
        .drop_duplicates()
        .groupby("participant_key")
        .size()
        .rename("n_recordings")
    )
    participants = (
        recordings.groupby(
            ["participant_key", "label_binary", "split", "stage", "dimension"],
            as_index=False,
            sort=True,
        )["value"]
        .mean()
        .merge(recording_counts, on="participant_key", validate="many_to_one")
        .sort_values(["stage", "participant_key", "dimension"], kind="mergesort")
        .reset_index(drop=True)
    )
    return participants


def resolve_official_hst_gradcam_layer(model: object) -> object:
    try:
        layers = getattr(model, "layers")
        layer = layers[-1]
        blocks = getattr(layer, "HSTblocks")
        block = blocks[-1]
        target = getattr(block, "attn2")
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "Model does not expose official HST layers[-1].HSTblocks[-1].attn2"
        ) from exc
    if target is None:
        raise ValueError("Official HST Grad-CAM target layer is missing")
    return target


@dataclass
class HSTGradCAM:
    model: object
    target_layer: object | None = None

    def __post_init__(self) -> None:
        if self.target_layer is None:
            self.target_layer = resolve_official_hst_gradcam_layer(self.model)
        self._activation: Any | None = None
        self._forward_pre_handle: Any | None = None

    @staticmethod
    def _tensor_output(value: object) -> object:
        if isinstance(value, (tuple, list)):
            if not value:
                raise RuntimeError("HST Grad-CAM hook received an empty output")
            return value[0]
        return value

    def _capture_activation(self, _module: object, inputs: object) -> None:
        self._activation = self._tensor_output(inputs)

    def __enter__(self) -> "HSTGradCAM":
        if self._forward_pre_handle is not None:
            raise RuntimeError("HST Grad-CAM hooks are already registered")
        self._forward_pre_handle = self.target_layer.register_forward_pre_hook(  # type: ignore[union-attr]
            self._capture_activation
        )
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._forward_pre_handle is not None:
            self._forward_pre_handle.remove()
        self._forward_pre_handle = None

    def generate(
        self,
        model_input: object,
        *,
        target_class: int = 1,
        spatial_shape: tuple[int, int] = (7, 7),
        output_shape: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        if self._forward_pre_handle is None:
            raise RuntimeError("Use HSTGradCAM as a context manager before generate")
        import torch

        self._activation = None
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()  # type: ignore[attr-defined]
        try:
            if isinstance(model_input, torch.Tensor) and not model_input.requires_grad:
                model_input = model_input.detach().requires_grad_(True)
            device = model_input.device if isinstance(model_input, torch.Tensor) else torch.device("cpu")
            with torch.enable_grad(), torch.autocast(
                device_type=device.type,
                enabled=False,
            ):
                logits = self.model(model_input)  # type: ignore[operator]
                if getattr(logits, "ndim", None) != 2 or int(logits.shape[0]) != 1:
                    raise ValueError(
                        "Grad-CAM requires one HST sample and two-dimensional logits"
                    )
                if target_class < 0 or target_class >= int(logits.shape[1]):
                    raise ValueError("target_class is outside the model output range")
                if self._activation is None:
                    raise RuntimeError(
                        "HST Grad-CAM hook did not capture the attention input"
                    )
                gradient = torch.autograd.grad(
                    logits[0, int(target_class)],
                    self._activation,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
        finally:
            self.model.train(was_training)  # type: ignore[attr-defined]
        activation = self._activation.detach().float().cpu().numpy()
        gradient = gradient.detach().float().cpu().numpy()
        if activation.ndim == 3 and activation.shape[0] == 1:
            activation = activation[0]
        if gradient.ndim == 3 and gradient.shape[0] == 1:
            gradient = gradient[0]
        return token_gradients_to_heatmap(
            activation,
            gradient,
            spatial_shape=spatial_shape,
            output_shape=output_shape,
        )
