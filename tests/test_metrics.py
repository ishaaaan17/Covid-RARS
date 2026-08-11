import numpy as np

from covid_rars.metrics import expected_calibration_error


def test_ece_perfect_predictions_is_low():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.01, 0.99, 0.02, 0.98])

    assert expected_calibration_error(y_true, y_prob, n_bins=2) < 0.05




def test_best_threshold_by_balanced_accuracy_can_move_below_half():
    from covid_rars.metrics import best_threshold_by_balanced_accuracy

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.25, 0.35])

    threshold = best_threshold_by_balanced_accuracy(y_true, y_prob)

    assert threshold < 0.5
    assert 0.10 <= threshold <= 0.25
