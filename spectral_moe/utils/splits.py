from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def resolve_split_seed(split_cfg: dict, default_seed: int) -> int:


    if "seed" in split_cfg:
        return int(split_cfg["seed"])
    if "split_seed" in split_cfg:
        return int(split_cfg["split_seed"])
    return int(default_seed)


def random_split(
    n_samples: int,
    *,
    seed: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_fraction))
    n_val = int(round(n_samples * val_fraction))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def group_split(
    groups: np.ndarray,
    *,
    seed: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(groups) == 0:
        raise ValueError("groups must not be empty")
    unique_groups = np.unique(groups.astype(str))
    rng = np.random.default_rng(seed)
    unique_groups = unique_groups[rng.permutation(len(unique_groups))]
    n_test = int(round(len(unique_groups) * test_fraction))
    n_val = int(round(len(unique_groups) * val_fraction))
    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val :])
    group_values = groups.astype(str)
    train_idx = np.flatnonzero(np.isin(group_values, list(train_groups)))
    val_idx = np.flatnonzero(np.isin(group_values, list(val_groups)))
    test_idx = np.flatnonzero(np.isin(group_values, list(test_groups)))
    return train_idx, val_idx, test_idx


def _hash_spectrum(row: np.ndarray) -> str:
    rounded = np.round(row.astype(np.float32), 6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def split_groups(labels: pd.DataFrame, x_raw_dbm: np.ndarray | None, group_by: str) -> np.ndarray:
    if group_by in labels.columns:
        return labels[group_by].fillna("").astype(str).to_numpy()
    if group_by == "condition":
        return (
            "T"
            + labels["temperature_c"].round(3).astype(str)
            + "_S"
            + labels["salinity_ppt"].round(3).astype(str)
        ).to_numpy()
    if group_by == "spectrum_hash":
        if x_raw_dbm is None:
            raise ValueError("x_raw_dbm is required for spectrum_hash grouping")
        return np.asarray([_hash_spectrum(row) for row in x_raw_dbm])
    if group_by == "run_condition":
        condition = split_groups(labels, x_raw_dbm, "condition")
        return (
            labels["experiment_type"].fillna("").astype(str)
            + "|"
            + labels["temp_direction"].fillna("").astype(str)
            + "|"
            + labels["salinity_direction"].fillna("").astype(str)
            + "|"
            + labels["repeat_index"].fillna("").astype(str)
            + "|"
            + pd.Series(condition, index=labels.index).astype(str)
        ).to_numpy()
    if group_by in {"leakage_safe", "condition_or_spectrum_hash"}:
        condition = split_groups(labels, x_raw_dbm, "condition")
        spectrum = split_groups(labels, x_raw_dbm, "spectrum_hash")
        return connected_component_groups([condition, spectrum])
    raise ValueError(f"Unknown split group: {group_by}")


def connected_component_groups(group_arrays: list[np.ndarray]) -> np.ndarray:
    n_samples = len(group_arrays[0])
    parent = np.arange(n_samples)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for groups in group_arrays:
        if len(groups) != n_samples:
            raise ValueError("all group arrays must have the same length")
        first_seen: dict[str, int] = {}
        for index, group in enumerate(groups.astype(str)):
            if group in first_seen:
                union(first_seen[group], index)
            else:
                first_seen[group] = index

    roots = np.asarray([find(index) for index in range(n_samples)])
    root_to_id = {root: group_id for group_id, root in enumerate(np.unique(roots))}
    return np.asarray([f"component_{root_to_id[root]}" for root in roots])


def split_from_config(
    n_samples: int,
    *,
    seed: int,
    split_cfg: dict,
    labels: pd.DataFrame | None = None,
    x_raw_dbm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    method = str(split_cfg.get("method", "random")).lower()
    val_fraction = float(split_cfg.get("val_fraction", 0.2))
    test_fraction = float(split_cfg.get("test_fraction", 0.2))
    if method == "random":
        train_idx, val_idx, test_idx = random_split(
            n_samples,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        meta = {"method": method, "group_by": None}
        return train_idx, val_idx, test_idx, meta

    if method in {"group", "grouped"}:
        if labels is None:
            raise ValueError("labels are required for grouped split")
        group_by = str(split_cfg.get("group_by", "spectrum_hash_group"))
        groups = split_groups(labels, x_raw_dbm, group_by)
        train_idx, val_idx, test_idx = group_split(
            groups,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        meta = {
            "method": method,
            "group_by": group_by,
            "n_groups": int(len(np.unique(groups))),
            "n_train_groups": int(len(np.unique(groups[train_idx]))),
            "n_val_groups": int(len(np.unique(groups[val_idx]))),
            "n_test_groups": int(len(np.unique(groups[test_idx]))),
        }
        return train_idx, val_idx, test_idx, meta

    raise ValueError(f"Unknown split method: {method}")


def subsample_train_indices(train_idx: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("train fraction must be in (0, 1]")
    if fraction >= 1:
        return train_idx
    rng = np.random.default_rng(seed)
    selected_count = max(1, int(round(len(train_idx) * fraction)))
    selected = rng.choice(train_idx, size=selected_count, replace=False)
    return np.sort(selected)


def split_assignment_labels(
    n_samples: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    full_train_idx: np.ndarray | None = None,
) -> np.ndarray:
    split = np.full(n_samples, "unused_train", dtype=object)
    if full_train_idx is None:
        full_train_idx = train_idx
    split[full_train_idx] = "train"
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    if full_train_idx is not None:
        unused = np.setdiff1d(full_train_idx, train_idx, assume_unique=False)
        split[unused] = "unused_train"
    return split


def split_audit(
    labels: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    x_raw_dbm: np.ndarray | None = None,
) -> dict:
    split = np.full(len(labels), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    audit: dict[str, object] = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    for group_name in ["condition_group", "spectrum_hash_group", "run_condition_group"]:
        if group_name in labels.columns:
            groups = labels[group_name].fillna("").astype(str).to_numpy()
        elif group_name == "condition_group":
            groups = split_groups(labels, x_raw_dbm, "condition")
        elif group_name == "spectrum_hash_group" and x_raw_dbm is not None:
            groups = split_groups(labels, x_raw_dbm, "spectrum_hash")
        else:
            continue
        test_groups = set(groups[test_idx])
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        audit[f"{group_name}_test_overlap_train_groups"] = int(len(test_groups & train_groups))
        audit[f"{group_name}_val_overlap_train_groups"] = int(len(val_groups & train_groups))
        audit[f"{group_name}_test_samples_with_train_group"] = int(np.sum(np.isin(groups[test_idx], list(train_groups))))
    if "experiment_type" in labels.columns:
        table = pd.crosstab(labels["experiment_type"], pd.Series(split, name="split")).astype(int)
        audit["experiment_type_by_split"] = {
            str(index): {str(column): int(table.loc[index, column]) for column in table.columns}
            for index in table.index
        }
    return audit


def kfold_indices(n_samples: int, folds: int, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_parts = np.array_split(indices, folds)
    result = []
    for i in range(folds):
        val_idx = fold_parts[i]
        train_idx = np.concatenate([fold_parts[j] for j in range(folds) if j != i])
        result.append((train_idx, val_idx))
    return result
