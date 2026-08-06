from __future__ import annotations

import re

import numpy as np

_TRACKED_TROUGH = re.compile(r"^tracked_dip_(?P<center>-?\d+(?:\.\d+)?)_wavelength_nm$")


def design_matrix(temperature_salinity: np.ndarray) -> np.ndarray:


    values = np.asarray(temperature_salinity, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("expected [n, 2] temperature/salinity values")
    t, sal = values[:, 0], values[:, 1]
    return np.column_stack([np.ones(len(values)), t, t**2, sal, t * sal])


def tracked_trough_metadata(feature_names: list[str]) -> tuple[list[int], np.ndarray]:

    indices: list[int] = []
    centers: list[float] = []
    for index, name in enumerate(feature_names):
        match = _TRACKED_TROUGH.match(name)
        if match:
            indices.append(index)
            centers.append(float(match.group("center")))
    if not indices:
        raise ValueError("no tracked trough wavelength feature is available")
    return indices, np.asarray(centers, dtype=np.float64)


def trough_reliability(
    wavelengths_nm: np.ndarray,
    centers_nm: np.ndarray,
    half_window_nm: float,
    edge_margin_nm: float,
    minimum_weight: float = 0.05,
) -> np.ndarray:

    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    centers = np.asarray(centers_nm, dtype=np.float64)
    if wavelengths.ndim != 2:
        raise ValueError("wavelengths_nm must have shape [n_samples, n_troughs]")
    if centers.shape != (wavelengths.shape[1],):
        raise ValueError("centers_nm must match the trough-feature dimension")
    if half_window_nm <= 0 or edge_margin_nm <= 0:
        raise ValueError("half_window_nm and edge_margin_nm must be positive")
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in [0, 1]")
    distance_to_edge = half_window_nm - np.abs(wavelengths - centers[None, :])
    confidence = np.clip(distance_to_edge / edge_margin_nm, 0.0, 1.0)
    return (minimum_weight + (1.0 - minimum_weight) * confidence).astype(np.float32)


def fit_forward_trough_calibrator(labels, physics, feature_names, ridge_alpha=1e-3, sample_weight=None):

    indices, centers = tracked_trough_metadata(feature_names)
    x = design_matrix(labels)
    y = np.asarray(physics, dtype=np.float64)[:, indices]
    if y.shape[0] != x.shape[0]:
        raise ValueError("labels and physics must contain the same number of samples")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    if sample_weight is None:
        weights = np.ones_like(y)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights[:, None]
        if weights.shape != y.shape:
            raise ValueError("sample_weight must have shape [n] or [n, n_troughs]")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite and non-negative")
    penalty = ridge_alpha * np.eye(x.shape[1]); penalty[0, 0] = 0.0
    coefficients, residuals = [], []
    for col in range(y.shape[1]):
        root_weight = np.sqrt(np.maximum(weights[:, col], 1e-12))
        wx, wy = x * root_weight[:, None], y[:, col] * root_weight
        coef = np.linalg.solve(wx.T @ wx + penalty, wx.T @ wy)
        coefficients.append(coef)
        residuals.append(y[:, col] - x @ coef)
    coef_array = np.asarray(coefficients, dtype=np.float64)
    scale = np.maximum(np.column_stack(residuals).std(axis=0), 1.0)
    return {"indices": indices, "centers_nm": centers.astype(np.float32), "coefficients": coef_array.astype(np.float32), "scale_nm": scale.astype(np.float32)}


def predict_forward_troughs(labels, coefficients):
    return design_matrix(labels) @ np.asarray(coefficients, dtype=np.float64).T


def fit_forward_feature_calibrator(labels, features, ridge_alpha=1e-3):

    x = design_matrix(labels)
    y = np.asarray(features, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("features must have shape [n_samples, n_features]")
    penalty = ridge_alpha * np.eye(x.shape[1]); penalty[0, 0] = 0.0
    coef, residuals, scores = [], [], []
    for col in range(y.shape[1]):
        current = np.linalg.solve(x.T @ x + penalty, x.T @ y[:, col])
        residual = y[:, col] - x @ current
        total = np.sum((y[:, col] - y[:, col].mean()) ** 2)
        coef.append(current); residuals.append(residual)
        scores.append(1.0 - np.sum(residual ** 2) / total if total > 1e-12 else 0.0)
    return {"coefficients": np.asarray(coef, dtype=np.float32), "scale": np.maximum(np.column_stack(residuals).std(axis=0), 1e-6).astype(np.float32), "r2": np.asarray(scores, dtype=np.float32)}
