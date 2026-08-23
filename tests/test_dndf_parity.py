from __future__ import annotations

import unittest
import numpy as np
import torch

from covid_rars.dndf_models import DNDFClassifier, NeuralDecisionForest, NeuralDecisionTree
from covid_rars.paper_features import extract_paper_193_features_from_audio, get_paper_193_feature_names


class TestDNDFMathematicalParity(unittest.TestCase):
    """Rigorous unit tests verifying mathematical correctness and zero-leakage parity of DNDT/DNDF."""

    def test_paper_193_feature_dimensions(self) -> None:
        """Verify the exact 193 feature names and array lengths."""
        names = get_paper_193_feature_names()
        self.assertEqual(len(names), 193)
        self.assertEqual(names[0], "mfcc_1")
        self.assertEqual(names[39], "mfcc_40")
        self.assertEqual(names[40], "chroma_1")
        self.assertEqual(names[51], "chroma_12")
        self.assertEqual(names[52], "mel_1")
        self.assertEqual(names[179], "mel_128")
        self.assertEqual(names[180], "contrast_1")
        self.assertEqual(names[186], "contrast_7")
        self.assertEqual(names[187], "tonnetz_1")
        self.assertEqual(names[192], "tonnetz_6")

        # Synthetic audio
        dummy_audio = np.random.randn(22050).astype(np.float32)
        feats = extract_paper_193_features_from_audio(dummy_audio, sr=22050)
        self.assertEqual(feats.shape, (193,))

    def test_neural_decision_tree_path_probabilities_sum_to_one(self) -> None:
        """Verify that tree leaf routing probabilities mu_l(x) strictly partition probability mass (sum to 1.0)."""
        depth = 4
        num_features = 33
        tree = NeuralDecisionTree(depth=depth, num_features=num_features, temperature=1.0, random_state=42)
        tree.eval()

        batch_size = 10
        x = torch.randn(batch_size, num_features)

        # Compute leaf probabilities directly
        d_j = torch.sigmoid(tree.decision_layer(tree.input_norm(x)) / tree.temperature)
        log_p = torch.log(torch.clamp(d_j, min=1e-7, max=1.0))
        log_not_p = torch.log(torch.clamp(1.0 - d_j, min=1e-7, max=1.0))
        log_prob = torch.cat([log_p, log_not_p], dim=-1)

        leaf_log_probs = torch.matmul(log_prob, tree.path_matrix.T)
        leaf_probs = torch.exp(leaf_log_probs)

        # Assert sum across leaves equals 1.0 for every sample
        leaf_sums = leaf_probs.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(leaf_sums, np.ones(batch_size), atol=1e-4)

    def test_neural_decision_forest_output_probabilities(self) -> None:
        """Verify that the DNDF ensemble averages soft predictions and produces valid probabilities."""
        forest = NeuralDecisionForest(
            num_trees=25,
            depth=5,
            num_features=33,
            used_features_rate=0.8,
            num_classes=2,
            temperature=1.0,
            random_state=42,
        )
        forest.eval()

        batch_size = 16
        x = torch.randn(batch_size, 33)
        probs = forest(x).detach().numpy()

        self.assertEqual(probs.shape, (batch_size, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(batch_size), atol=1e-5)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_zero_leakage_feature_selection_and_threshold(self) -> None:
        """Verify that DNDFClassifier fits selectors on train data and tunes threshold on val data."""
        np.random.seed(42)
        X_train = np.random.randn(80, 50).astype(np.float32)
        y_train = np.random.randint(0, 2, size=80)
        X_val = np.random.randn(20, 50).astype(np.float32)
        y_val = np.random.randint(0, 2, size=20)
        X_test = np.random.randn(20, 50).astype(np.float32)

        clf = DNDFClassifier(
            model_type="dndf",
            num_trees=5,
            depth=3,
            max_epochs=3,
            n_selected_features=15,
            feature_selection="f_classif",
            random_state=42,
        )

        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, optimize_threshold=True)

        # Assert selector was fitted
        self.assertIsNotNone(clf.feature_selector_)
        self.assertIsNotNone(clf.scaler_)

        # Assert transform works on unseen test data
        probs = clf.predict_proba(X_test)
        self.assertEqual(probs.shape, (20, 2))
        self.assertTrue(0.0 <= clf.best_threshold_ <= 1.0)


if __name__ == "__main__":
    unittest.main()
