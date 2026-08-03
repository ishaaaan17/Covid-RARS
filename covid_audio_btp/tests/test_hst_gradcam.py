from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_gradcam_implementation_uses_pre_attention_autograd_contract() -> None:
    from covid_audio_btp.hst_gradcam import HSTGradCAM

    source = inspect.getsource(HSTGradCAM)
    assert "register_forward_pre_hook" in source
    assert "torch.autograd.grad" in source
    assert "zero_grad" not in source


def test_token_gradients_form_normalized_224_heatmap() -> None:
    from covid_audio_btp.hst_gradcam import token_gradients_to_heatmap

    activations = np.arange(49 * 3, dtype=np.float32).reshape(49, 3)
    gradients = np.ones((49, 3), dtype=np.float32)

    heatmap = token_gradients_to_heatmap(
        activations,
        gradients,
        spatial_shape=(7, 7),
        output_shape=(224, 224),
    )

    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) == pytest.approx(0.0)
    assert float(heatmap.max()) == pytest.approx(1.0)
    assert np.isfinite(heatmap).all()


def test_constant_or_negative_cam_is_zero_not_nan() -> None:
    from covid_audio_btp.hst_gradcam import token_gradients_to_heatmap

    activations = -np.ones((49, 2), dtype=np.float32)
    gradients = np.ones((49, 2), dtype=np.float32)
    heatmap = token_gradients_to_heatmap(activations, gradients)
    assert np.array_equal(heatmap, np.zeros((224, 224), dtype=np.float32))


def test_heatmap_rejects_non_hst_token_geometry() -> None:
    from covid_audio_btp.hst_gradcam import token_gradients_to_heatmap

    with pytest.raises(ValueError, match="token"):
        token_gradients_to_heatmap(
            np.ones((48, 2), dtype=np.float32),
            np.ones((48, 2), dtype=np.float32),
        )


def test_example_selection_is_deterministic_and_covers_error_cells() -> None:
    from covid_audio_btp.hst_gradcam import select_gradcam_examples

    predictions = pd.DataFrame(
        {
            "participant_key": [f"coswara::p{i}" for i in range(8)],
            "recording_key": [f"coswara::r{i}" for i in range(8)],
            "label_binary": [1, 1, 0, 0, 1, 1, 0, 0],
            "probability": [0.95, 0.80, 0.05, 0.20, 0.10, 0.40, 0.90, 0.60],
            "split": ["test"] * 8,
            "fold": [0] * 8,
            "protocol": ["internal"] * 8,
        }
    )

    first = select_gradcam_examples(predictions, threshold=0.5, per_cell=1)
    second = select_gradcam_examples(
        predictions.sample(frac=1.0, random_state=4), threshold=0.5, per_cell=1
    )

    assert first["outcome"].tolist() == ["TP", "TN", "FP", "FN"]
    pd.testing.assert_frame_equal(first, second)
    assert first["selection_rule"].nunique() == 1


def test_example_selection_rejects_mixed_protocol_or_fold() -> None:
    from covid_audio_btp.hst_gradcam import select_gradcam_examples

    predictions = pd.DataFrame(
        {
            "participant_key": ["coswara::p1", "coswara::p2"],
            "recording_key": ["coswara::r1", "coswara::r2"],
            "label_binary": [1, 0],
            "probability": [0.8, 0.2],
            "split": ["test", "test"],
            "fold": [0, 1],
            "protocol": ["internal", "internal"],
        }
    )
    with pytest.raises(ValueError, match="fold"):
        select_gradcam_examples(predictions)


def test_resolve_official_layer_uses_last_attn2() -> None:
    from covid_audio_btp.hst_gradcam import resolve_official_hst_gradcam_layer

    class Block:
        attn2 = object()

    class Layer:
        HSTblocks = [Block(), Block()]

    class Model:
        layers = [Layer(), Layer()]

    assert resolve_official_hst_gradcam_layer(Model()) is Model.layers[-1].HSTblocks[-1].attn2


def test_gradcam_uses_attention_input_without_mutating_model_state() -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_gradcam import HSTGradCAM

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn2 = torch.nn.Linear(3, 3, bias=False)

        def forward(self, value: object) -> object:
            return self.attn2(value)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.HSTblocks = torch.nn.ModuleList([Block()])

        def forward(self, value: object) -> object:
            return self.HSTblocks[0](value)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])
            self.head = torch.nn.Linear(3, 2)

        def forward(self, value: object) -> object:
            encoded = self.layers[0](value)
            return self.head(encoded.mean(dim=1))

    model = Model().train()
    sentinel_gradients = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
        sentinel_gradients[name] = parameter.grad.detach().clone()
    target = model.layers[-1].HSTblocks[-1].attn2
    before_hooks = (len(target._forward_pre_hooks), len(target._forward_hooks))

    with HSTGradCAM(model) as camera:
        heatmap = camera.generate(torch.randn(1, 49, 3), output_shape=(14, 14))

    assert heatmap.shape == (14, 14)
    assert model.training is True
    assert before_hooks == (len(target._forward_pre_hooks), len(target._forward_hooks))
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, sentinel_gradients[name])


def test_participant_gradcam_summary_uses_correct_participants_as_clusters() -> None:
    from covid_audio_btp.hst_gradcam import build_participant_gradcam_summary

    heatmaps = pd.DataFrame(
        {
            "participant_key": ["p0", "p0", "p1", "p2", "p3"],
            "recording_key": ["r0", "r1", "r2", "r3", "r4"],
            "label_binary": [0, 0, 0, 1, 1],
            "outcome": ["TN", "TN", "TN", "TP", "TP"],
            "heatmap": [
                np.zeros((2, 2), dtype=np.float32),
                np.full((2, 2), 0.2, dtype=np.float32),
                np.full((2, 2), 0.3, dtype=np.float32),
                np.full((2, 2), 0.8, dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            ],
            "threshold": [0.5] * 5,
            "threshold_source": ["fixed_0.5"] * 5,
            "protocol": ["internal"] * 5,
            "fold": [0] * 5,
            "split": ["test"] * 5,
        }
    )

    summary = build_participant_gradcam_summary(
        heatmaps,
        bootstrap_replicates=30,
        seed=42,
    )

    assert len(summary.participant_heatmaps) == 4
    assert summary.participant_heatmaps.loc[
        summary.participant_heatmaps["participant_key"].eq("p0"), "n_recordings"
    ].item() == 2
    assert np.allclose(summary.negative_mean, 0.2)
    assert np.allclose(summary.positive_mean, 0.9)
    assert np.allclose(summary.mean_difference, 0.7)
    assert summary.ci_low.shape == (2, 2)
    assert summary.ci_high.shape == (2, 2)
    assert summary.resampling_unit == "participant_key"


def test_gradcam_evidence_writes_auditable_heatmaps_and_overlays(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_gradcam import build_gradcam_evidence

    class Camera:
        def __init__(self, _model: object) -> None:
            self.index = 0

        def __enter__(self) -> "Camera":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def generate(self, _value: object, *, target_class: int) -> np.ndarray:
            self.index += 1
            return np.full((4, 4), 0.1 * self.index + 0.1 * target_class, dtype=np.float32)

    examples = pd.DataFrame(
        {
            "participant_key": ["coswara::p0", "coswara::p1"],
            "recording_key": ["coswara::r0", "coswara::r1"],
            "label_binary": [0, 1],
            "probability": [0.1, 0.9],
            "outcome": ["TN", "TP"],
            "threshold": [0.5, 0.5],
            "threshold_source": ["validation", "validation"],
            "protocol": ["internal", "internal"],
            "fold": [0, 0],
            "split": ["test", "test"],
            "model_input": [object(), object()],
            "image": [
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            ],
        }
    )

    manifest = build_gradcam_evidence(
        object(),
        examples,
        output_dir=tmp_path,
        camera_factory=Camera,
    )

    assert manifest["target_class"].tolist() == [0, 1]
    assert not manifest["zero_map"].any()
    assert all((tmp_path / path).is_file() for path in manifest["heatmap_path"])
    assert all((tmp_path / path).is_file() for path in manifest["overlay_path"])
    assert all(len(value) == 64 for value in manifest["heatmap_sha256"])
    assert (tmp_path / "gradcam_evidence_manifest.csv").is_file()


def test_embedding_figure_rejects_training_rows_and_is_deterministic(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_gradcam import build_stage_embedding_figure

    rows = []
    for participant_index in range(6):
        for dimension in range(3):
            rows.append(
                {
                    "participant_key": f"p{participant_index}",
                    "label_binary": participant_index % 2,
                    "split": "test",
                    "stage": "stage_4",
                    "dimension": dimension,
                    "value": float(participant_index + dimension),
                }
            )
    embeddings = pd.DataFrame(rows)
    first = build_stage_embedding_figure(
        embeddings,
        output_path=tmp_path / "embedding.png",
        method="pca",
        seed=42,
    )
    second = build_stage_embedding_figure(
        embeddings.sample(frac=1.0, random_state=7),
        output_path=tmp_path / "embedding_again.png",
        method="pca",
        seed=42,
    )

    assert first.coordinates.equals(second.coordinates)
    assert first.output_path.is_file()
    with pytest.raises(ValueError, match="held-out"):
        build_stage_embedding_figure(
            embeddings.assign(split="train"),
            output_path=tmp_path / "invalid.png",
            method="pca",
            seed=42,
        )


def test_stage_embedding_extraction_averages_recordings_within_participant() -> None:
    torch = pytest.importorskip("torch")
    from covid_audio_btp.hst_gradcam import extract_stage_participant_embeddings

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
            )
            for layer in self.layers:
                layer.weight.data.copy_(torch.eye(2))

        def forward(self, value: object) -> object:
            for layer in self.layers:
                value = layer(value)
            return value.mean(dim=1)

    loader = [
        (
            torch.tensor(
                [
                    [[1.0, 3.0], [1.0, 3.0]],
                    [[3.0, 5.0], [3.0, 5.0]],
                    [[8.0, 10.0], [8.0, 10.0]],
                ]
            ),
            torch.tensor([0, 0, 1]),
            [
                {"participant_key": "p0", "recording_key": "r0", "split": "test"},
                {"participant_key": "p0", "recording_key": "r1", "split": "test"},
                {"participant_key": "p1", "recording_key": "r2", "split": "test"},
            ],
        )
    ]

    result = extract_stage_participant_embeddings(Model(), loader)

    p0_stage1 = result.loc[
        result["participant_key"].eq("p0") & result["stage"].eq("stage_1")
    ].sort_values("dimension")
    assert p0_stage1["value"].tolist() == pytest.approx([2.0, 4.0])
    assert p0_stage1["n_recordings"].unique().tolist() == [2]
    assert set(result["split"]) == {"test"}
