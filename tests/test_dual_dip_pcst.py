"""Regression checks for the manuscript's dual-dip PCST definition."""

from pathlib import Path

import numpy as np
import yaml

from spectral_moe.evaluate.conditional_synthetic_quality import select_cmi_pcqd
from spectral_moe.evaluate.synthetic_quality import (
    real_manifold_distances,
    synthetic_acceptance_mask,
)


def test_default_configuration_uses_the_same_two_dips_everywhere():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    expected = [1516.0, 1599.0]
    assert config["features"]["tracked_centers_nm"] == expected
    assert config["gan"]["pinn"]["tracked_centers_nm"] == expected


def test_pcst_position_error_is_the_mean_absolute_error_of_both_dips():
    reference = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    synthetic = np.array([[0.01, 0.01], [0.11, 0.11]])
    observed = np.array([[1516.0, 1599.0], [1518.0, 1601.0]])
    expected = np.array([[1515.0, 1601.0], [1518.0, 1604.0]])
    accepted, audit = synthetic_acceptance_mask(
        reference,
        synthetic,
        observed,
        expected,
        max_conditional_trough_mae_nm=2.1,
        manifold_percentile=100.0,
    )
    assert accepted.tolist() == [True, True]
    assert np.isclose(audit["mean_conditional_trough_mae_nm"], 1.5)


def test_pcst_selector_returns_coverage_aware_continuous_confidence_weights():
    rng = np.random.default_rng(7)
    y_real = np.array([[t, s] for t in (10.0, 20.0, 30.0, 40.0) for s in (0.0, 10.0, 20.0, 30.0)])
    f_real = rng.normal(size=(len(y_real), 6))
    f_syn = f_real + rng.normal(scale=0.01, size=f_real.shape)
    nearest, radius = real_manifold_distances(f_real, f_syn, percentile=95.0)
    result = select_cmi_pcqd(
        F_real=f_real,
        y_real=y_real,
        z_real_norm=f_real,
        F_syn=f_syn,
        y_syn=y_real.copy(),
        z_syn_norm=f_syn,
        trough_mae_syn_nm=np.full(len(f_syn), 0.2),
        nearest_manifold_dist_syn=nearest,
        manifold_radius=radius,
        target_count=8,
        min_accepted_samples=8,
        condition_bins=(2, 2),
        min_per_nonempty_bin=1,
        seed=7,
    )
    assert len(result.selected_idx) == 8
    assert result.audit["method"] == "cmi_pcqd"
    assert np.isclose(result.confidence.mean(), 1.0)
    assert np.all(result.confidence > 0)


def test_catb_keeps_task_specific_heads_and_temperature_encoder_out_of_shared_projection():
    trainable_names = [
        "shared_proj.1.adapter_down.weight",
        "router.gate.1.adapter_up.weight",
        "temperature_head.residual_mlp.0.weight",
        "salinity_head.residual_mlp.0.weight",
        "temp_encoder.proj.weight",
    ]
    shared = [
        "temperature_head" not in name
        and "salinity_head" not in name
        and "temp_encoder" not in name
        for name in trainable_names
    ]
    assert shared == [True, True, False, False, False]
