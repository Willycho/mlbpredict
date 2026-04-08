"""v2 Calibration Layer — Isotonic + Platt scaling.

확률 예측의 보정(calibration)과 평가 지표 계산.
"""

import os
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ============================================================
# Calibration Models
# ============================================================

class IsotonicCalibrator:
    """Isotonic Regression 기반 calibrator."""

    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.01, y_max=0.99,
            out_of_bounds="clip",
        )
        self.fitted = False

    def fit(self, predicted: np.ndarray, actual: np.ndarray):
        self.model.fit(predicted, actual)
        self.fitted = True

    def predict(self, predicted: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return predicted
        return self.model.predict(predicted)


class PlattCalibrator:
    """Platt Scaling (Logistic Regression on logit)."""

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.fitted = False

    def fit(self, predicted: np.ndarray, actual: np.ndarray):
        # logit 변환
        eps = 1e-6
        predicted = np.clip(predicted, eps, 1 - eps)
        logits = np.log(predicted / (1 - predicted)).reshape(-1, 1)
        self.model.fit(logits, actual.astype(int))
        self.fitted = True

    def predict(self, predicted: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return predicted
        eps = 1e-6
        predicted = np.clip(predicted, eps, 1 - eps)
        logits = np.log(predicted / (1 - predicted)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class EnsembleCalibrator:
    """Isotonic + Platt 앙상블."""

    def __init__(self, isotonic_weight: float = 0.6):
        self.isotonic = IsotonicCalibrator()
        self.platt = PlattCalibrator()
        self.isotonic_weight = isotonic_weight
        self.fitted = False

    def fit(self, predicted: np.ndarray, actual: np.ndarray):
        self.isotonic.fit(predicted, actual)
        self.platt.fit(predicted, actual)
        self.fitted = True

    def predict(self, predicted: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return predicted
        iso_pred = self.isotonic.predict(predicted)
        platt_pred = self.platt.predict(predicted)
        return (
            self.isotonic_weight * iso_pred
            + (1 - self.isotonic_weight) * platt_pred
        )


# ============================================================
# Evaluation Metrics
# ============================================================

def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Brier Score: mean((p - y)^2). 낮을수록 좋음."""
    return float(np.mean((predicted - actual) ** 2))


def log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Log Loss: -mean(y*log(p) + (1-y)*log(1-p)). 낮을수록 좋��."""
    eps = 1e-6
    p = np.clip(predicted, eps, 1 - eps)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))


def expected_calibration_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE (Expected Calibration Error).

    확률 구간별 |예측 평균 - 실제 빈도| 가중 평균.
    낮을수록 좋음 (0 = 완벽한 calibration).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(predicted)

    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_pred = predicted[mask].mean()
        avg_actual = actual[mask].mean()
        ece += (n_bin / total) * abs(avg_pred - avg_actual)

    return float(ece)


def reliability_bins(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Reliability Curve 데이터 생성.

    Returns:
        list of dicts: [{bin_center, avg_predicted, avg_actual, n_samples}, ...]
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (predicted >= bin_edges[i]) & (predicted <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bins.append({
            "bin_center": round((bin_edges[i] + bin_edges[i + 1]) / 2, 2),
            "avg_predicted": round(float(predicted[mask].mean()), 4),
            "avg_actual": round(float(actual[mask].mean()), 4),
            "n_samples": int(n_bin),
        })

    return bins


def hit_rate(predicted: np.ndarray, actual: np.ndarray, threshold: float = 0.5) -> float:
    """단순 적중률 (보조 지표)."""
    picks = (predicted >= threshold).astype(int)
    return float((picks == actual).mean())


def evaluate_all(
    predicted: np.ndarray,
    actual: np.ndarray,
    label: str = "",
) -> dict:
    """전체 평가 지표 한번에 계산."""
    metrics = {
        "brier_score": brier_score(predicted, actual),
        "log_loss": log_loss(predicted, actual),
        "ece": expected_calibration_error(predicted, actual),
        "hit_rate": hit_rate(predicted, actual),
        "n_samples": len(predicted),
        "reliability_bins": reliability_bins(predicted, actual),
    }

    if label:
        metrics["label"] = label

    return metrics


# ============================================================
# Save / Load
# ============================================================

def save_calibrator(calibrator, path: str):
    """Calibrator를 pickle로 저장."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: str):
    """저장된 calibrator 로드."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
