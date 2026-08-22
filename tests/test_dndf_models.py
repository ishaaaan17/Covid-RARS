from __future__ import annotations

import numpy as np
import pytest
import torch

from covid_rars.dndf_models import DNDFClassifier, NeuralDecisionForest, NeuralDecisionTree


def test_neural_decision_tree_forward():
    batch_size = 8
    num_features = 50
    depth = 3
    num_classes = 2

    tree = NeuralDecisionTree(
        depth=depth,
        num_features=num_features,
        used_features_rate=0.8,
        num_classes=num_classes,
        temperature=1.0,
        random_state=42,
    )

    x = torch.randn(batch_size, num_features)
    out = tree(x)

    assert out.shape == (batch_size, num_classes)
    # Check probabilities sum to 1
    sums = out.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_neural_decision_forest_forward():
    batch_size = 10
    num_features = 40
    num_trees = 5
    depth = 3
    num_classes = 2

    forest = NeuralDecisionForest(
        num_trees=num_trees,
        depth=depth,
        num_features=num_features,
        used_features_rate=0.7,
        num_classes=num_classes,
        temperature=1.0,
        random_state=42,
    )

    x = torch.randn(batch_size, num_features)
    out = forest(x)

    assert out.shape == (batch_size, num_classes)
    sums = out.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    assert len(forest.trees) == num_trees


def test_dndf_classifier_fit_predict():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 20).astype(np.float32)
    y = (rng.rand(60) > 0.5).astype(int)

    X_val = rng.randn(20, 20).astype(np.float32)
    y_val = (rng.rand(20) > 0.5).astype(int)

    # Test DNDT
    dndt = DNDFClassifier(
        model_type="dndt",
        depth=3,
        max_epochs=5,
        batch_size=16,
        use_smote=False,
        random_state=42,
        device="cpu",
    )
    dndt.fit(X, y, X_val=X_val, y_val=y_val)

    probs = dndt.predict_proba(X_val)
    preds = dndt.predict(X_val)

    assert probs.shape == (20, 2)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Test DNDF
    dndf = DNDFClassifier(
        model_type="dndf",
        num_trees=3,
        depth=3,
        max_epochs=5,
        batch_size=16,
        use_smote=False,
        random_state=42,
        device="cpu",
    )
    dndf.fit(X, y, X_val=X_val, y_val=y_val)

    probs_f = dndf.predict_proba(X_val)
    preds_f = dndf.predict(X_val)

    assert probs_f.shape == (20, 2)
    assert preds_f.shape == (20,)
    assert set(np.unique(preds_f)).issubset({0, 1})
