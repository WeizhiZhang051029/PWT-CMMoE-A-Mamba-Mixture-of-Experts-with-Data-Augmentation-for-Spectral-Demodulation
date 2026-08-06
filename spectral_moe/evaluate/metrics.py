from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, dict[str, float]]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    result: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(target_names):
        err = y_pred[:, idx] - y_true[:, idx]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        nonzero = np.abs(y_true[:, idx]) > 1e-8
        mape = (
            float(np.mean(np.abs(err[nonzero] / y_true[nonzero, idx])) * 100.0)
            if np.any(nonzero)
            else float("nan")
        )
        denom = float(np.sum((y_true[:, idx] - np.mean(y_true[:, idx])) ** 2))
        r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else float("nan")
        result[name] = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    return result


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {f"{target}_{name}": value for target, values in metrics.items() for name, value in values.items()}
