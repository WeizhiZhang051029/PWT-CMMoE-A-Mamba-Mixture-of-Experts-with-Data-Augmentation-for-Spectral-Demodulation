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


from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CMIPCQDReport:
    selected_idx: np.ndarray
    confidence: np.ndarray
    audit: dict[str, Any] = field(default_factory=dict)


def _pairwise_sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    A64 = A.astype(np.float64, copy=False)
    B64 = B.astype(np.float64, copy=False)
    A2 = np.sum(A64 * A64, axis=1, keepdims=True)
    B2 = np.sum(B64 * B64, axis=1, keepdims=True).T
    return np.clip(A2 + B2 - 2.0 * A64 @ B64.T, 0.0, None)


def _knn_indices_and_dists(query: np.ndarray, ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:

    sqd = _pairwise_sq_dist(query, ref)
    k = min(k, ref.shape[0])
    idx = np.argpartition(sqd, kth=k - 1, axis=1)[:, :k]

    rows = np.arange(query.shape[0])[:, None]
    dists_partial = sqd[rows, idx]
    order = np.argsort(dists_partial, axis=1)
    idx_sorted = idx[rows, order]
    dists_sorted = np.sqrt(dists_partial[rows, order])
    return idx_sorted, dists_sorted


def _knn_inverse_labels(
    query_features: np.ndarray,
    ref_features: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    self_excluded: bool = False,
) -> tuple[np.ndarray, np.ndarray]:


    idx, dists = _knn_indices_and_dists(query_features, ref_features, k)
    if self_excluded:

        idx = idx[:, 1:]
        dists = dists[:, 1:]


    med_scale = np.median(dists, axis=1, keepdims=True)
    med_scale = np.where(med_scale < 1e-9, 1.0, med_scale)
    weights = np.exp(-dists / med_scale)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum = np.where(weights_sum < 1e-12, 1.0, weights_sum)
    weights = weights / weights_sum

    y_hat = np.einsum("nk,nkt->nt", weights, ref_labels[idx])
    mean_dist = dists.mean(axis=1)
    return y_hat, mean_dist


def calibrate_inverse_tau(
    real_features: np.ndarray,
    real_labels: np.ndarray,
    k: int = 7,
    quantile: float = 0.90,
) -> dict[str, float]:


    n = real_features.shape[0]
    if n <= k:
        raise ValueError(f"real training samples too few (n={n}) for k={k} LOO calibration")
    idx, dists = _knn_indices_and_dists(real_features, real_features, k + 1)

    idx = idx[:, 1:]
    dists = dists[:, 1:]
    med_scale = np.median(dists, axis=1, keepdims=True)
    med_scale = np.where(med_scale < 1e-9, 1.0, med_scale)
    weights = np.exp(-dists / med_scale)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum = np.where(weights_sum < 1e-12, 1.0, weights_sum)
    weights = weights / weights_sum
    y_hat = np.einsum("nk,nkt->nt", weights, real_labels[idx])
    e = np.abs(y_hat - real_labels)
    tau_T = float(np.quantile(e[:, 0], quantile))
    tau_S = float(np.quantile(e[:, 1], quantile))
    return {
        "tau_T": max(tau_T, 1e-6),
        "tau_S": max(tau_S, 1e-6),
        "loo_temp_p50": float(np.median(e[:, 0])),
        "loo_temp_p90": tau_T,
        "loo_sal_p50": float(np.median(e[:, 1])),
        "loo_sal_p90": tau_S,
    }


def _compute_condition_bin_ids(
    y: np.ndarray, y_train: np.ndarray, bins: tuple[int, int]
) -> np.ndarray:

    lo = y_train.min(axis=0)
    hi = y_train.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scaled = (y - lo) / span
    scaled = np.clip(scaled, 0.0, 1.0 - 1e-9)
    ti = np.floor(scaled[:, 0] * bins[0]).astype(np.int32)
    si = np.floor(scaled[:, 1] * bins[1]).astype(np.int32)
    return ti * bins[1] + si


def _train_bin_ratios(
    y_train: np.ndarray, bins: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    ids = _compute_condition_bin_ids(y_train, y_train, bins)
    n_bins = bins[0] * bins[1]
    counts = np.bincount(ids, minlength=n_bins).astype(np.float64)
    ratios = counts / max(counts.sum(), 1.0)
    return ratios, counts


def _mmr_select_within_bin(
    z_syn_bin: np.ndarray,
    Q_bin: np.ndarray,
    quota: int,
    diversity_weight: float,
) -> np.ndarray:


    n = z_syn_bin.shape[0]
    quota = min(quota, n)
    if quota <= 0:
        return np.empty(0, dtype=np.int64)
    selected = [int(np.argmax(Q_bin))]
    remaining = set(range(n)) - {selected[0]}
    Q_max = max(float(Q_bin.max()), 1e-12)
    while len(selected) < quota and remaining:
        rem = np.asarray(sorted(remaining))

        z_sel = z_syn_bin[selected]
        z_rem = z_syn_bin[rem]
        sqd = _pairwise_sq_dist(z_rem, z_sel)
        min_d = np.sqrt(sqd.min(axis=1))
        max_d = max(float(min_d.max()), 1e-12)
        novelty_norm = min_d / max_d
        Q_norm = Q_bin[rem] / Q_max
        score = (1.0 - diversity_weight) * Q_norm + diversity_weight * novelty_norm
        pick = int(rem[np.argmax(score)])
        selected.append(pick)
        remaining.discard(pick)
    return np.asarray(selected, dtype=np.int64)


def select_cmi_pcqd(
    *,

    F_real: np.ndarray,
    y_real: np.ndarray,
    z_real_norm: np.ndarray,

    F_syn: np.ndarray,
    y_syn: np.ndarray,
    z_syn_norm: np.ndarray,
    trough_mae_syn_nm: np.ndarray,
    nearest_manifold_dist_syn: np.ndarray,

    manifold_radius: float,
    max_conditional_trough_mae_nm: float = 8.0,
    hard_reject_trough_mae_nm: float = 16.0,

    target_count: int = 2000,
    min_accepted_samples: int = 1200,
    condition_bins: tuple[int, int] = (4, 4),
    diversity_weight: float = 0.30,
    min_per_nonempty_bin: int = 50,

    knn_k: int = 7,
    tau_quantile: float = 0.90,

    w_phys: float = 0.20,
    w_manifold: float = 0.15,
    w_temp: float = 0.20,
    w_sal: float = 0.45,

    confidence_clip: tuple[float, float] = (0.25, 2.0),
    seed: int = 0,
) -> CMIPCQDReport:


    n_syn = F_syn.shape[0]
    if n_syn == 0:
        raise RuntimeError("no synthetic candidates provided to CMI-PCQD selector")

    rng = np.random.default_rng(seed + 22001)


    manifold_ok = nearest_manifold_dist_syn <= manifold_radius
    trough_ok = trough_mae_syn_nm <= hard_reject_trough_mae_nm
    safe_mask = manifold_ok & trough_ok
    safe_count = int(safe_mask.sum())

    if safe_count < min_accepted_samples:
        raise RuntimeError(
            f"CMI-PCQD: safe candidates {safe_count} < min_accepted_samples "
            f"{min_accepted_samples}. synthetic_only=true refuses fallback."
        )

    safe_idx = np.flatnonzero(safe_mask)


    F_safe = F_syn[safe_idx]
    y_safe = y_syn[safe_idx]
    y_hat, mean_neighbor_dist = _knn_inverse_labels(F_safe, F_real, y_real, k=knn_k)
    e_T = np.abs(y_hat[:, 0] - y_safe[:, 0])
    e_S = np.abs(y_hat[:, 1] - y_safe[:, 1])

    tau = calibrate_inverse_tau(F_real, y_real, k=knn_k, quantile=tau_quantile)
    tau_T = tau["tau_T"]
    tau_S = tau["tau_S"]

    q_T = np.exp(-e_T / tau_T)
    q_S = np.exp(-e_S / tau_S)


    trough_safe = trough_mae_syn_nm[safe_idx]
    manifold_safe = nearest_manifold_dist_syn[safe_idx]
    q_phys = np.exp(-trough_safe / max(max_conditional_trough_mae_nm, 1e-6))
    q_manifold = np.exp(-manifold_safe / max(manifold_radius, 1e-6))

    Q = (
        w_phys * q_phys
        + w_manifold * q_manifold
        + w_temp * q_T
        + w_sal * q_S
    )


    train_ratios, train_counts = _train_bin_ratios(y_real, condition_bins)
    n_bins = condition_bins[0] * condition_bins[1]
    bin_ids_safe = _compute_condition_bin_ids(y_safe, y_real, condition_bins)


    quotas = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        if train_counts[b] == 0:
            continue
        base = int(round(train_ratios[b] * target_count))
        quotas[b] = max(base, min_per_nonempty_bin)

    total = int(quotas.sum())
    if total > target_count:

        for _ in range(50):
            excess = int(quotas.sum()) - target_count
            if excess <= 0:
                break

            over = np.where(quotas > min_per_nonempty_bin)[0]
            if over.size == 0:
                break
            step = min(excess, over.size)

            order = np.argsort(-quotas[over])
            for i in range(step):
                quotas[over[order[i]]] -= 1
    elif total < target_count:

        while int(quotas.sum()) < target_count:
            deficit = target_count - int(quotas.sum())
            non_empty = np.where(train_counts > 0)[0]
            if non_empty.size == 0:
                break
            take = min(deficit, non_empty.size)
            order = np.argsort(-train_ratios[non_empty])
            for i in range(take):
                quotas[non_empty[order[i]]] += 1


    selected_in_safe: list[int] = []
    z_syn_safe = z_syn_norm[safe_idx]
    per_bin_selected: dict[int, int] = {}
    for b in range(n_bins):
        if quotas[b] == 0:
            per_bin_selected[b] = 0
            continue
        mask = bin_ids_safe == b
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            per_bin_selected[b] = 0
            continue
        quota = int(min(quotas[b], cand.size))
        if quota == cand.size:
            picks_local = np.arange(cand.size)
        else:
            picks_local = _mmr_select_within_bin(
                z_syn_safe[cand], Q[cand], quota, diversity_weight
            )
        picks_global = cand[picks_local]
        selected_in_safe.extend(picks_global.tolist())
        per_bin_selected[b] = int(len(picks_global))

    selected_in_safe_arr = np.asarray(sorted(set(selected_in_safe)), dtype=np.int64)


    if selected_in_safe_arr.size < target_count:
        remaining_pool = np.setdiff1d(
            np.arange(safe_idx.size), selected_in_safe_arr, assume_unique=False
        )
        if remaining_pool.size > 0:
            need = target_count - selected_in_safe_arr.size
            order = np.argsort(-Q[remaining_pool])
            fillers = remaining_pool[order[:need]]
            selected_in_safe_arr = np.sort(
                np.concatenate([selected_in_safe_arr, fillers])
            )


    selected_count = int(selected_in_safe_arr.size)
    if selected_count < min_accepted_samples:
        raise RuntimeError(
            f"CMI-PCQD: after selection got {selected_count} < min {min_accepted_samples}."
        )
    shortfall = max(target_count - selected_count, 0)


    Q_selected = Q[selected_in_safe_arr]
    Q_mean = float(Q_selected.mean())
    if Q_mean < 1e-9:
        Q_mean = 1e-9
    confidence = Q_selected / Q_mean
    lo, hi = confidence_clip
    confidence = np.clip(confidence, lo, hi).astype(np.float32)

    c_mean = float(confidence.mean())
    if c_mean > 1e-9:
        confidence = confidence / c_mean


    final_idx = safe_idx[selected_in_safe_arr]


    def _pcts(x):
        if x.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        return {
            "mean": float(np.mean(x)),
            "p50": float(np.median(x)),
            "p95": float(np.quantile(x, 0.95)),
        }

    ess = float((confidence.sum() ** 2) / (np.sum(confidence ** 2) + 1e-12))
    audit = {
        "method": "cmi_pcqd",
        "candidate_count": int(n_syn),
        "safe_candidate_count": int(safe_count),
        "selected_count": selected_count,
        "selection_shortfall_count": int(shortfall),
        "tau_T": tau_T,
        "tau_S": tau_S,
        "tau_calibration": tau,
        "trough_mae_nm": _pcts(trough_safe),
        "inverse_temperature_error": _pcts(e_T),
        "inverse_salinity_error": _pcts(e_S),
        "q_phys_mean": float(q_phys.mean()),
        "q_manifold_mean": float(q_manifold.mean()),
        "q_temp_mean": float(q_T.mean()),
        "q_sal_mean": float(q_S.mean()),
        "Q_mean_selected": Q_mean,
        "confidence_mean_after_norm": float(confidence.mean()),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "effective_sample_size": ess,
        "manifold_radius": float(manifold_radius),
        "condition_bins": list(condition_bins),
        "per_bin_selected": {str(k): int(v) for k, v in per_bin_selected.items()},
        "quotas": [int(x) for x in quotas],
        "train_bin_counts": [int(x) for x in train_counts],
        "diversity_weight": float(diversity_weight),
        "knn_k": int(knn_k),
        "Q_weights": {
            "phys": float(w_phys),
            "manifold": float(w_manifold),
            "temp": float(w_temp),
            "sal": float(w_sal),
        },
    }

    return CMIPCQDReport(
        selected_idx=final_idx.astype(np.int64),
        confidence=confidence.astype(np.float32),
        audit=audit,
    )
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
