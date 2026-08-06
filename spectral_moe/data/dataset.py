"""Dataset loading and wavelength-grid utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spectral_moe.utils.io import DEFAULT_LABELS, DEFAULT_MATRIX, resolve_project_path


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
    x_zscore = payload["X_zscore"].astype(np.float32)
    sample_id = payload["sample_id"].astype(str)
    wavelength_nm = payload["wavelength_nm"].astype(np.float64)

    if align_npz:


        align_candidate = Path(str(align_npz))
        if not align_candidate.is_absolute():
            from spectral_moe.utils.io import PROJECT_ROOT
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
        if target_length < len(wavelength_nm):
            wl_target = np.linspace(
                float(wavelength_nm[0]), float(wavelength_nm[-1]), target_length
            )
            x_model_raw = align_wavelength_grid(x_raw, wavelength_nm, wl_target)
            x_model_zscore = align_wavelength_grid(x_zscore, wavelength_nm, wl_target)
            input_wavelength_nm = wl_target

    x = x_model_zscore if use_zscore else x_model_raw

    labels = pd.read_csv(labels_path)
    if "sample_id" in labels.columns:
        label_ids = labels["sample_id"].astype(str).to_numpy()
        if len(label_ids) == len(sample_id) and not np.all(label_ids == sample_id):
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
        physics: np.ndarray | None = None,
        prior_prediction: np.ndarray | None = None,
        temp_context: np.ndarray | None = None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchSpectrumDataset requires PyTorch") from exc

        self.torch = torch
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
        self.physics = None if physics is None else torch.from_numpy(physics.astype(np.float32))
        self.prior_prediction = (
            None if prior_prediction is None else torch.from_numpy(prior_prediction.astype(np.float32))
        )
        self.temp_context = None if temp_context is None else torch.from_numpy(temp_context.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        x = self.x[index]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        item = {"x": x}
        if self.y is not None:
            item["y"] = self.y[index]
        if self.physics is not None:
            item["physics"] = self.physics[index]
        if self.prior_prediction is not None:
            item["prior_prediction"] = self.prior_prediction[index]
        if self.temp_context is not None:
            item["temp_context"] = self.temp_context[index]
        return item
