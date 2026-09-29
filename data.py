from __future__ import annotations

from dataclasses import dataclass
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


"""Dataset loading and wavelength-grid utilities."""

@dataclass
class SpectrumBundle:
    """Aligned spectra, labels, and metadata used by the training pipeline."""

    x: np.ndarray
    x_raw_dbm: np.ndarray
    wavelength_nm: np.ndarray
    input_wavelength_nm: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    labels: pd.DataFrame
    target_names: list[str]


def align_wavelength_grid(
    x_source: np.ndarray,
    wl_source: np.ndarray,
    wl_target: np.ndarray,
    *,
    fill_mode: str = "edge",
) -> np.ndarray:
    """Interpolate spectra from one wavelength grid onto another."""


    if x_source.ndim != 2:
        raise ValueError(f"x_source must be 2-D, got {x_source.ndim}-D")
    if x_source.shape[1] != len(wl_source):
        raise ValueError(
            f"x_source length {x_source.shape[1]} != wl_source length {len(wl_source)}"
        )
    wl_source = np.asarray(wl_source, dtype=np.float64)
    wl_target = np.asarray(wl_target, dtype=np.float64)
    if wl_source[0] > wl_source[-1]:
        wl_source = wl_source[::-1]
        x_source = x_source[:, ::-1]

    reversed_target = wl_target[0] > wl_target[-1]
    if reversed_target:
        wl_target = wl_target[::-1]

    out = np.empty((x_source.shape[0], len(wl_target)), dtype=np.float32)
    left_val = np.nan if fill_mode == "nan" else (0.0 if fill_mode == "zero" else None)
    right_val = left_val
    for i in range(x_source.shape[0]):
        if fill_mode == "edge":
            out[i] = np.interp(wl_target, wl_source, x_source[i]).astype(np.float32)
        else:
            out[i] = np.interp(
                wl_target, wl_source, x_source[i], left=left_val, right=right_val
            ).astype(np.float32)

    if reversed_target:
        out = out[:, ::-1]
    return out


def load_spectrum_bundle(config: dict[str, Any]) -> SpectrumBundle:


    data_cfg = config.get("data", {})
    npz_path = resolve_project_path(data_cfg.get("npz_path"), DEFAULT_MATRIX)
    labels_path = resolve_project_path(data_cfg.get("labels_path"), DEFAULT_LABELS)
    use_zscore = bool(data_cfg.get("use_zscore", True))
    target_names = list(data_cfg.get("target_names", ["temperature_c", "salinity_ppt"]))
    align_npz = data_cfg.get("align_to_wavelength_npz")

    if not npz_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {npz_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    payload = np.load(npz_path, allow_pickle=False)
    x_raw = payload["X_raw_dbm"].astype(np.float32)
    if "X_zscore" in payload:
        x_zscore = payload["X_zscore"].astype(np.float32)
    else:
        x_zscore = x_raw - x_raw.mean(axis=1, keepdims=True)
        x_zscore /= np.maximum(x_raw.std(axis=1, keepdims=True), 1e-8)
        x_zscore = x_zscore.astype(np.float32)
    sample_id = payload["sample_id"].astype(str)
    wavelength_nm = payload["wavelength_nm"].astype(np.float64)
    if x_raw.ndim != 2:
        raise ValueError("X_raw_dbm must have shape [n_samples, n_points]")
    if x_zscore.shape != x_raw.shape:
        raise ValueError("X_zscore must have the same shape as X_raw_dbm")
    if wavelength_nm.ndim != 1 or len(wavelength_nm) != x_raw.shape[1]:
        raise ValueError("wavelength_nm must be a 1-D axis matching the spectrum width")
    if not np.isfinite(wavelength_nm).all() or np.any(np.diff(wavelength_nm) == 0):
        raise ValueError("wavelength_nm must be finite with unique sample positions")
    if len(sample_id) != x_raw.shape[0]:
        raise ValueError("sample_id length must match the number of spectra")

    if align_npz:


        align_candidate = Path(str(align_npz))
        if not align_candidate.is_absolute():
            align_candidate = (PROJECT_ROOT / align_candidate).resolve()
        if not align_candidate.exists():
            raise FileNotFoundError(f"align_to_wavelength_npz not found: {align_candidate}")
        ref = np.load(align_candidate, allow_pickle=False)
        wl_target = ref["wavelength_nm"].astype(np.float64)
        fill_mode = str(data_cfg.get("align_fill_mode", "edge"))
        x_raw_aligned = align_wavelength_grid(x_raw, wavelength_nm, wl_target, fill_mode=fill_mode)


        x_zscore_aligned = align_wavelength_grid(
            x_zscore, wavelength_nm, wl_target, fill_mode=fill_mode
        )
        x_raw = x_raw_aligned
        x_zscore = x_zscore_aligned
        wavelength_nm = wl_target


        use_zscore = False

    x_model_raw = x_raw
    x_model_zscore = x_zscore
    input_wavelength_nm = wavelength_nm
    downsample_to = data_cfg.get("downsample_to")
    if downsample_to is not None:
        method = str(data_cfg.get("resample_method", "linear")).lower()
        if method != "linear":
            raise ValueError("data.resample_method currently supports only 'linear'")
        target_length = int(downsample_to)
        if target_length < 2:
            raise ValueError("data.downsample_to must be at least 2")
        if target_length != len(wavelength_nm):
            wl_target = np.linspace(
                float(wavelength_nm[0]), float(wavelength_nm[-1]), target_length
            )
            x_model_raw = align_wavelength_grid(x_raw, wavelength_nm, wl_target)
            x_model_zscore = align_wavelength_grid(x_zscore, wavelength_nm, wl_target)
            input_wavelength_nm = wl_target

    x = x_model_zscore if use_zscore else x_model_raw

    labels = pd.read_csv(labels_path)
    if len(labels) != len(sample_id):
        raise ValueError("labels row count must match the number of spectra")
    if "sample_id" in labels.columns:
        label_ids = labels["sample_id"].astype(str).to_numpy()
        if not np.all(label_ids == sample_id):
            raise ValueError("sample_id order mismatch between matrix and labels")

    missing_targets = [name for name in target_names if name not in labels.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in labels: {missing_targets}")

    include_types = data_cfg.get("include_experiment_types")
    exclude_types = data_cfg.get("exclude_experiment_types", [])
    keep = np.ones(len(labels), dtype=bool)
    if include_types:
        if "experiment_type" not in labels.columns:
            raise ValueError("include_experiment_types requires an experiment_type column")
        include = {str(value) for value in include_types}
        keep &= labels["experiment_type"].astype(str).isin(include).to_numpy()
    if exclude_types:
        if "experiment_type" in labels.columns:
            exclude = {str(value) for value in exclude_types}
            keep &= ~labels["experiment_type"].astype(str).isin(exclude).to_numpy()

    if not np.all(keep):
        x_raw = x_raw[keep]
        x_zscore = x_zscore[keep]
        x = x[keep]
        sample_id = sample_id[keep]
        labels = labels.loc[keep].reset_index(drop=True)

    y = labels[target_names].to_numpy(dtype=np.float32)

    return SpectrumBundle(
        x=x,
        x_raw_dbm=x_raw,
        wavelength_nm=wavelength_nm,
        input_wavelength_nm=input_wavelength_nm,
        y=y,
        sample_id=sample_id,
        labels=labels,
        target_names=target_names,
    )


class TorchSpectrumDataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchSpectrumDataset requires PyTorch") from exc

        self.torch = torch
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        x = self.x[index]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        item = {"x": x}
        if self.y is not None:
            item["y"] = self.y[index]
        return item


class TorchRegressionDataset:
    """Tensor dataset shared by MoE pretraining and adapter fine-tuning."""

    def __init__(
        self,
        spectrum_features: np.ndarray,
        physics: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        raw_spectrum: np.ndarray | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchRegressionDataset requires PyTorch") from exc

        self.z = torch.from_numpy(spectrum_features.astype(np.float32))
        self.phy = torch.from_numpy(physics.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
        self.sample_weight = None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))
        self.raw = None if raw_spectrum is None else torch.from_numpy(raw_spectrum.astype(np.float32))

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, index: int) -> dict:
        item = {"z": self.z[index], "physics": self.phy[index]}
        if self.y is not None:
            item["y"] = self.y[index]
        if self.sample_weight is not None:
            item["sample_weight"] = self.sample_weight[index]
        if self.raw is not None:
            item["raw"] = self.raw[index]
        return item
