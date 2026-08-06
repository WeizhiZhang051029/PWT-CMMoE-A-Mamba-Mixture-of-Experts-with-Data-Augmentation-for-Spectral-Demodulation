from __future__ import annotations

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


def build_quality_features(
    z_norm: np.ndarray,
    tracked_wavelengths_norm: np.ndarray,
    tracked_intensities_norm: np.ndarray,
    dip_features_norm: np.ndarray,
) -> np.ndarray:

    parts = [z_norm]
    if tracked_wavelengths_norm.size > 0:
        parts.append(tracked_wavelengths_norm)
    if tracked_intensities_norm.size > 0:
        parts.append(tracked_intensities_norm)
    if dip_features_norm.size > 0:
        parts.append(dip_features_norm)
    return np.concatenate(parts, axis=1).astype(np.float64)


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


def _spectral_novelty(z_norm: np.ndarray, ref_norm: np.ndarray) -> np.ndarray:

    if ref_norm.shape[0] == 0:
        return np.zeros(z_norm.shape[0], dtype=np.float64)
    idx, dists = _knn_indices_and_dists(z_norm, ref_norm, k=1)
    return dists[:, 0]


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
