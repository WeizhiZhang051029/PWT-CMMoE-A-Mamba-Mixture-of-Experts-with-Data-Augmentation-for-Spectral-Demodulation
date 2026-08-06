from __future__ import annotations

import numpy as np


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
    tracked_centers = [float(v) for v in (tracked_centers_nm or [1599.0])]
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
            float(np.trapezoid(y, wavelength_nm)),
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


def standardize_features(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:


    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return tuple(((arr - mean) / std).astype(np.float32) for arr in (train, *others))


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
