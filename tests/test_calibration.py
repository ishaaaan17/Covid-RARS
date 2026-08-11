import numpy as np

from covid_rars.calibration import PlattCalibrator, TemperatureScaler, uncertainty_label


def test_platt_calibrator_requires_fit():
    calibrator = PlattCalibrator()
    try:
        calibrator.transform(np.array([0.2, 0.8]))
    except RuntimeError as exc:
        assert "fit" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError")


def test_temperature_scaler_outputs_probabilities():
    scaler = TemperatureScaler(temperatures=np.array([1.0, 2.0]))
    scaler.fit(np.array([-2.0, 2.0]), np.array([0, 1]))
    probs = scaler.transform_logits(np.array([-1.0, 1.0]))
    assert probs.shape == (2,)
    assert np.all((probs > 0) & (probs < 1))


def test_uncertainty_label():
    assert uncertainty_label(0.5) == "uncertain"
    assert uncertainty_label(0.7) == "screening_signal_higher"
    assert uncertainty_label(0.2) == "screening_signal_lower"
