from __future__ import annotations

import re

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) * 0.5))


def _safe_stat(values: np.ndarray, fn, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(fn(values))


def _top_k_dips(y: np.ndarray, k: int, min_separation: int) -> list[int]:
    order = np.argsort(y)
    selected: list[int] = []
    for idx in order:
        idx_int = int(idx)
        if all(abs(idx_int - prev) >= min_separation for prev in selected):
            selected.append(idx_int)
        if len(selected) >= k:
            break
    return sorted(selected)


def extract_physics_features(
    x: np.ndarray,
    wavelength_nm: np.ndarray,
    *,
    num_dips: int = 6,
    num_bands: int = 8,
    tracked_centers_nm: list[float] | tuple[float, ...] | None = None,
    tracked_half_window_nm: float = 35.0,
) -> tuple[np.ndarray, list[str]]:
    if x.ndim != 2:
        raise ValueError("x must have shape [n_samples, n_points]")
    if x.shape[1] != len(wavelength_nm):
        raise ValueError("wavelength length must match x.shape[1]")

    names: list[str] = [
        "global_mean",
        "global_std",
        "global_min",
        "global_max",
        "global_range",
        "argmin_wavelength_nm",
        "argmax_wavelength_nm",
        "area_trapz",
        "first_derivative_mean",
        "first_derivative_std",
        "second_derivative_std",
    ]
    # The manuscript's physics constraint is defined jointly on dip 1 and dip 2.
    tracked_centers = [float(v) for v in (tracked_centers_nm or [1516.0, 1599.0])]
    tracked_half_window = float(tracked_half_window_nm)
    for center in tracked_centers:
        label = int(round(center))
        names.extend(
            [
                f"tracked_dip_{label}_wavelength_nm",
                f"tracked_dip_{label}_intensity",
                f"tracked_dip_{label}_local_slope",
            ]
        )
    for i in range(num_dips):
        names.extend([f"dip_{i+1}_wavelength_nm", f"dip_{i+1}_intensity", f"dip_{i+1}_local_slope"])
    for i in range(num_bands):
        names.extend([f"band_{i+1}_mean", f"band_{i+1}_std", f"band_{i+1}_min"])

    rows: list[list[float]] = []
    dx = np.gradient(wavelength_nm)
    band_edges = np.linspace(wavelength_nm[0], wavelength_nm[-1], num_bands + 1)
    min_separation = max(1, len(wavelength_nm) // max(20, num_dips * 10))

    for y in x:
        dy = np.gradient(y, wavelength_nm)
        ddy = np.gradient(dy, wavelength_nm)
        argmin = int(np.argmin(y))
        argmax = int(np.argmax(y))
        row: list[float] = [
            float(np.mean(y)),
            float(np.std(y)),
            float(np.min(y)),
            float(np.max(y)),
            float(np.max(y) - np.min(y)),
            float(wavelength_nm[argmin]),
            float(wavelength_nm[argmax]),
            _trapezoid(y, wavelength_nm),
            float(np.mean(dy)),
            float(np.std(dy)),
            float(np.std(ddy)),
        ]

        for center in tracked_centers:
            mask = (wavelength_nm >= center - tracked_half_window) & (wavelength_nm <= center + tracked_half_window)
            if np.any(mask):
                indices = np.flatnonzero(mask)
                local_idx = indices[int(np.argmin(y[mask]))]
                row.extend([float(wavelength_nm[local_idx]), float(y[local_idx]), float(dy[local_idx])])
            else:
                row.extend([0.0, 0.0, 0.0])

        dips = _top_k_dips(y, num_dips, min_separation=min_separation)
        for idx in dips:
            row.extend([float(wavelength_nm[idx]), float(y[idx]), float(dy[idx])])
        for _ in range(num_dips - len(dips)):
            row.extend([0.0, 0.0, 0.0])

        for band_idx in range(num_bands):
            lo = band_edges[band_idx]
            hi = band_edges[band_idx + 1]
            if band_idx == num_bands - 1:
                mask = (wavelength_nm >= lo) & (wavelength_nm <= hi)
            else:
                mask = (wavelength_nm >= lo) & (wavelength_nm < hi)
            values = y[mask]
            row.extend(
                [
                    _safe_stat(values, np.mean),
                    _safe_stat(values, np.std),
                    _safe_stat(values, np.min),
                ]
            )
        rows.append(row)

    return np.asarray(rows, dtype=np.float32), names


def fit_feature_standardizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:


    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_feature_standardizer(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)
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
