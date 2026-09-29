from __future__ import annotations

import numpy as np


def _covariance(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("expected at least two samples with shape [n, d]")
    return np.atleast_2d(np.cov(x, rowvar=False))


def frechet_distance(reference: np.ndarray, generated: np.ndarray) -> float:

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if reference.ndim != 2 or generated.ndim != 2 or reference.shape[1] != generated.shape[1]:
        raise ValueError("reference and generated must be 2-D with matching dimensions")
    mean_delta = reference.mean(axis=0) - generated.mean(axis=0)
    cov_r, cov_g = _covariance(reference), _covariance(generated)
    product = cov_r @ cov_g
    eigenvalues = np.linalg.eigvals(product).real
    covariance_mean_trace = float(np.sqrt(np.clip(eigenvalues, 0.0, None)).sum())
    return float(mean_delta @ mean_delta + np.trace(cov_r) + np.trace(cov_g) - 2.0 * covariance_mean_trace)


def quantile_l1_distance(reference: np.ndarray, generated: np.ndarray, quantiles: int = 101) -> float:

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if reference.ndim != 2 or generated.ndim != 2 or reference.shape[1] != generated.shape[1]:
        raise ValueError("reference and generated must be 2-D with matching dimensions")
    grid = np.linspace(0.0, 1.0, quantiles)
    return float(np.mean(np.abs(np.quantile(reference, grid, axis=0) - np.quantile(generated, grid, axis=0))))


def _nearest_distances(query: np.ndarray, reference: np.ndarray, *, exclude_self: bool = False) -> np.ndarray:
    distances = np.empty(len(query), dtype=np.float64)
    reference_norm = np.sum(reference * reference, axis=1)
    for start in range(0, len(query), 256):
        stop = min(start + 256, len(query))
        chunk = query[start:stop]
        distance_sq = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + reference_norm[None, :]
            - 2.0 * chunk @ reference.T
        )
        np.maximum(distance_sq, 0.0, out=distance_sq)
        if exclude_self:
            rows = np.arange(stop - start)
            distance_sq[rows, np.arange(start, stop)] = np.inf
        distances[start:stop] = np.sqrt(distance_sq.min(axis=1))
    return distances


def nearest_reference_coverage(reference: np.ndarray, generated: np.ndarray, percentile: float = 95.0) -> float:

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(reference) < 3:
        raise ValueError("reference needs at least three samples")
    scale = reference.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    ref = reference / scale
    gen = generated / scale
    radius = np.percentile(_nearest_distances(ref, ref, exclude_self=True), percentile)
    gen_distance = _nearest_distances(gen, ref)
    return float(np.mean(gen_distance <= radius))


def real_manifold_distances(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """Return normalized nearest-real distances and the leave-one-out radius.

    This is the common manifold metric used by PCST screening, confidence
    scoring, and diversity-aware selection.  Keeping it in one function
    prevents those stages from using subtly different distance scales.
    """
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("reference and query must be 2-D with matching feature dimensions")
    if len(reference) < 3:
        raise ValueError("reference needs at least three samples")
    scale = reference.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    ref = reference / scale
    qry = query / scale
    radius = float(np.percentile(
        _nearest_distances(ref, ref, exclude_self=True), percentile
    ))
    return _nearest_distances(qry, ref), radius


def synthetic_acceptance_mask(
    reference_features: np.ndarray,
    synthetic_features: np.ndarray,
    observed_synthetic_wavelengths: np.ndarray,
    expected_synthetic_wavelengths: np.ndarray,
    *,
    max_conditional_trough_mae_nm: float,
    manifold_percentile: float = 95.0,
) -> tuple[np.ndarray, dict]:


    if max_conditional_trough_mae_nm <= 0:
        raise ValueError("max_conditional_trough_mae_nm must be positive")
    reference = np.asarray(reference_features, dtype=np.float64)
    synthetic = np.asarray(synthetic_features, dtype=np.float64)
    observed = np.asarray(observed_synthetic_wavelengths, dtype=np.float64)
    expected = np.asarray(expected_synthetic_wavelengths, dtype=np.float64)
    if reference.ndim != 2 or synthetic.ndim != 2 or reference.shape[1] != synthetic.shape[1]:
        raise ValueError("feature arrays must be 2-D with matching feature dimensions")
    if observed.shape != expected.shape or observed.shape[0] != synthetic.shape[0]:
        raise ValueError("trough arrays must match and align with synthetic_features")
    nearest, radius = real_manifold_distances(
        reference, synthetic, percentile=manifold_percentile
    )
    trough_mae = np.mean(np.abs(observed - expected), axis=1)
    accepted = (trough_mae <= max_conditional_trough_mae_nm) & (nearest <= radius)

    return accepted, {
        "max_conditional_trough_mae_nm": float(max_conditional_trough_mae_nm),
        "manifold_radius": radius,
        "accepted_fraction": float(accepted.mean()),
        "accepted_count": int(accepted.sum()),
        "mean_conditional_trough_mae_nm": float(trough_mae.mean()),
        "median_conditional_trough_mae_nm": float(np.median(trough_mae)),
        "p95_conditional_trough_mae_nm": float(np.percentile(trough_mae, 95)),
        "min_conditional_trough_mae_nm": float(trough_mae.min()),
        "mean_nearest_manifold_distance": float(nearest.mean()),
        "median_nearest_manifold_distance": float(np.median(nearest)),
        "trough_only_pass_fraction": float((trough_mae <= max_conditional_trough_mae_nm).mean()),
        "manifold_only_pass_fraction": float((nearest <= radius).mean()),
    }


def synthetic_quality_report(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    real_trough_features: np.ndarray,
    synthetic_trough_features: np.ndarray,
    observed_synthetic_wavelengths: np.ndarray,
    expected_synthetic_wavelengths: np.ndarray,
) -> dict:

    observed = np.asarray(observed_synthetic_wavelengths, dtype=np.float64)
    expected = np.asarray(expected_synthetic_wavelengths, dtype=np.float64)
    if observed.shape != expected.shape:
        raise ValueError("expected and observed synthetic trough arrays must match")
    return {
        "spectral_frechet_distance": frechet_distance(real_features, synthetic_features),
        "spectral_real_manifold_coverage": nearest_reference_coverage(real_features, synthetic_features),
        "trough_distribution_quantile_l1": quantile_l1_distance(real_trough_features, synthetic_trough_features),
        "conditional_trough_mae_nm": float(np.mean(np.abs(observed - expected))),
    }


