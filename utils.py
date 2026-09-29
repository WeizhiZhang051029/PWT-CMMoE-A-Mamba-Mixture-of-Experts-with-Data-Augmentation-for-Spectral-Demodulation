from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(config_path.resolve())
    data["_config_dir"] = str(config_path.resolve().parent)
    return data

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX = PROJECT_ROOT / "data" / "spectra.npz"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels.csv"


def resolve_project_path(path: str | Path | None, default: Path) -> Path:
    if path is None or str(path).strip() == "":
        return default
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass

def resolve_split_seed(split_cfg: dict, default_seed: int) -> int:
    """Resolve the reproducible seed used by the train/validation split."""
    if "seed" in split_cfg:
        return int(split_cfg["seed"])
    if "split_seed" in split_cfg:
        return int(split_cfg["split_seed"])
    return int(default_seed)


def random_split(n_samples: int, *, seed: int, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    indices = np.random.default_rng(seed).permutation(n_samples)
    n_val = int(round(n_samples * val_fraction))
    return indices[n_val:], indices[:n_val]


def group_split(groups: np.ndarray, *, seed: int, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    if len(groups) == 0:
        raise ValueError("groups must not be empty")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    unique_groups = np.unique(groups.astype(str))
    unique_groups = unique_groups[np.random.default_rng(seed).permutation(len(unique_groups))]
    n_val = int(round(len(unique_groups) * val_fraction))
    val_groups = set(unique_groups[:n_val])
    train_groups = set(unique_groups[n_val:])
    group_values = groups.astype(str)
    train_idx = np.flatnonzero(np.isin(group_values, list(train_groups)))
    val_idx = np.flatnonzero(np.isin(group_values, list(val_groups)))
    return train_idx, val_idx


def _hash_spectrum(row: np.ndarray) -> str:
    rounded = np.round(row.astype(np.float32), 6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def split_groups(labels: pd.DataFrame, x_raw_dbm: np.ndarray | None, group_by: str) -> np.ndarray:
    if group_by in labels.columns:
        return labels[group_by].fillna("").astype(str).to_numpy()
    if group_by == "condition":
        return (
            "T" + labels["temperature_c"].round(3).astype(str)
            + "_S" + labels["salinity_ppt"].round(3).astype(str)
        ).to_numpy()
    if group_by == "spectrum_hash":
        if x_raw_dbm is None:
            raise ValueError("x_raw_dbm is required for spectrum_hash grouping")
        return np.asarray([_hash_spectrum(row) for row in x_raw_dbm])
    if group_by == "run_condition":
        condition = split_groups(labels, x_raw_dbm, "condition")
        return (
            labels["experiment_type"].fillna("").astype(str) + "|"
            + labels["temp_direction"].fillna("").astype(str) + "|"
            + labels["salinity_direction"].fillna("").astype(str) + "|"
            + labels["repeat_index"].fillna("").astype(str) + "|"
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
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the train/validation partition used by every training stage."""
    method = str(split_cfg.get("method", "random")).lower()
    val_fraction = float(split_cfg.get("val_fraction", 0.2))
    if method == "random":
        train_idx, val_idx = random_split(n_samples, seed=seed, val_fraction=val_fraction)
        return train_idx, val_idx, {"method": method, "group_by": None}
    if method in {"group", "grouped"}:
        if labels is None:
            raise ValueError("labels are required for grouped split")
        group_by = str(split_cfg.get("group_by", "spectrum_hash_group"))
        groups = split_groups(labels, x_raw_dbm, group_by)
        train_idx, val_idx = group_split(groups, seed=seed, val_fraction=val_fraction)
        return train_idx, val_idx, {
            "method": method,
            "group_by": group_by,
            "n_groups": int(len(np.unique(groups))),
            "n_train_groups": int(len(np.unique(groups[train_idx]))),
            "n_val_groups": int(len(np.unique(groups[val_idx]))),
        }
    raise ValueError(f"Unknown split method: {method}")


def subsample_train_indices(train_idx: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("train fraction must be in (0, 1]")
    if fraction >= 1:
        return train_idx
    selected_count = max(1, int(round(len(train_idx) * fraction)))
    selected = np.random.default_rng(seed).choice(train_idx, size=selected_count, replace=False)
    return np.sort(selected)
