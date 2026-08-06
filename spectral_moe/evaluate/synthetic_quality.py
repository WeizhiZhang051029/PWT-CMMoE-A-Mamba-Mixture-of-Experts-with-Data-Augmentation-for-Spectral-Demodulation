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
    scale = reference.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    ref = reference / scale
    syn = synthetic / scale
    radius = float(np.percentile(
        _nearest_distances(ref, ref, exclude_self=True), manifold_percentile
    ))
    nearest = _nearest_distances(syn, ref)
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


def select_physics_quality_diverse_indices(
    candidate_indices: np.ndarray, synthetic_features: np.ndarray,
    synthetic_labels: np.ndarray, reference_labels: np.ndarray,
    trough_mae_nm: np.ndarray, manifold_distance: np.ndarray, *,
    target_count: int, max_conditional_trough_mae_nm: float,
    manifold_radius: float, condition_bins: tuple[int, int] = (4, 4),
    diversity_weight: float = 0.35,
) -> tuple[np.ndarray, dict]:

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    x = np.asarray(synthetic_features, dtype=np.float64)
    y = np.asarray(synthetic_labels, dtype=np.float64)
    ref_y = np.asarray(reference_labels, dtype=np.float64)
    trough = np.asarray(trough_mae_nm, dtype=np.float64)
    nearest = np.asarray(manifold_distance, dtype=np.float64)
    if target_count <= 0 or x.ndim != 2 or y.shape != (len(x), 2) or ref_y.ndim != 2 or ref_y.shape[1] != 2:
        raise ValueError('invalid PCQD selector inputs')
    if len(candidate_indices) <= target_count:
        return np.sort(candidate_indices), {'selector': 'pcqd', 'selected_count': int(len(candidate_indices)), 'candidate_count': int(len(candidate_indices)), 'selection_skipped': 'candidate_count_not_above_target'}
    b = tuple(int(v) for v in condition_bins)
    if len(b) != 2 or min(b) < 1:
        raise ValueError('condition_bins must contain two positive integers')
    dweight = float(np.clip(diversity_weight, 0.0, 1.0))
    quality = 0.5 * (np.clip(1.0 - trough / max(max_conditional_trough_mae_nm, 1e-8), 0.0, 1.0) + np.clip(1.0 - nearest / max(manifold_radius, 1e-8), 0.0, 1.0))
    edges=[]
    for d,count in enumerate(b):
        lo,hi=np.quantile(ref_y[:,d],[0.0,1.0]); hi=max(hi,lo+1e-6)
        edges.append(np.linspace(lo,hi,count+1)[1:-1])
    groups={}
    for i in candidate_indices:
        key=tuple(int(np.digitize(y[i,d],edges[d])) for d in range(2))
        groups.setdefault(key,[]).append(int(i))
    groups=[np.asarray(groups[k],dtype=np.int64) for k in sorted(groups)]
    scale=x[candidate_indices].std(axis=0,keepdims=True); scale[scale<1e-8]=1.0; x=x/scale
    def choose(pool,budget):
        picked=[]; remain=pool.copy()
        for _ in range(min(budget,len(pool))):
            if not picked: best=int(remain[np.argmax(quality[remain])])
            else:
                dist=np.sqrt(((x[remain,None,:]-x[np.asarray(picked)][None,:,:])**2).sum(axis=2)).min(axis=1)
                score=(1.0-dweight)*quality[remain]+dweight*dist/max(float(dist.max()),1e-8)
                best=int(remain[np.argmax(score)])
            picked.append(best); remain=remain[remain!=best]
        return picked
    base,extra=divmod(target_count,len(groups)); selected=[]; rest=[]
    for j,g in enumerate(groups):
        picked=choose(g,base+(j<extra)); selected+=picked; rest+=np.setdiff1d(g,np.asarray(picked,dtype=np.int64)).tolist()
    if len(selected)<target_count:
        rest=np.asarray(rest,dtype=np.int64); selected+=rest[np.argsort(-quality[rest],kind='stable')[:target_count-len(selected)]].tolist()
    selected=np.sort(np.asarray(selected[:target_count],dtype=np.int64))
    return selected, {'selector':'pcqd','selected_count':int(len(selected)),'candidate_count':int(len(candidate_indices)),'occupied_condition_bins':int(len(groups)),'condition_bins':list(b),'diversity_weight':dweight,'selected_quality_mean':float(quality[selected].mean()),'candidate_quality_mean':float(quality[candidate_indices].mean())}
