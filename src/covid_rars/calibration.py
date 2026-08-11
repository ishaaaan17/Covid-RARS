from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs")
        self.is_fit = False

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        x = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        self.model.fit(x, y)
        self.is_fit = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Calibrator must be fit before transform")
        x = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.is_fit = False

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(probabilities, dtype=float), np.asarray(y_true, dtype=int))
        self.is_fit = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Calibrator must be fit before transform")
        return self.model.transform(np.asarray(probabilities, dtype=float))


class TemperatureScaler:
    def __init__(self, temperatures: np.ndarray | None = None) -> None:
        self.temperatures = temperatures if temperatures is not None else np.linspace(0.5, 5.0, 91)
        self.temperature_: float | None = None

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        return 1.0 / (1.0 + np.exp(-logits))

    @staticmethod
    def _nll(y_true: np.ndarray, probs: np.ndarray) -> float:
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return float(-np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        best_temp = 1.0
        best_loss = float("inf")
        for temp in self.temperatures:
            probs = self._sigmoid(logits / temp)
            loss = self._nll(y_true, probs)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        self.temperature_ = best_temp
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler must be fit before transform")
        return self._sigmoid(np.asarray(logits, dtype=float) / self.temperature_)


def uncertainty_label(probability: float, low: float = 0.40, high: float = 0.60) -> str:
    if low <= probability <= high:
        return "uncertain"
    if probability > high:
        return "screening_signal_higher"
    return "screening_signal_lower"
