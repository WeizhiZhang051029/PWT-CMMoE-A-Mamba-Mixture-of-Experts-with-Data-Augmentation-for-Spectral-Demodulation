"""Regression checks for the manuscript's dual-dip PCST definition."""

from pathlib import Path

import numpy as np
import yaml

from spectral_moe.evaluate.conditional_synthetic_quality import select_cmi_pcqd
from spectral_moe.evaluate.synthetic_quality import (
    real_manifold_distances,
    synthetic_acceptance_mask,
)
from spectral_moe.train.finetune_adapter import AdaptiveMTLBalancer


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


def test_catb_keeps_only_task_specific_heads_out_of_shared_projection():
    trainable_names = [
        "shared_proj.1.adapter_down.weight",
        "router.gate.1.adapter_up.weight",
        "temperature_head.residual_mlp.0.weight",
        "salinity_head.residual_mlp.0.weight",
    ]
    shared = [
        "temperature_head" not in name
        and "salinity_head" not in name
        for name in trainable_names
    ]
    assert shared == [True, True, False, False]


def test_default_configuration_has_no_temperature_encoder():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert "temp_context_out_dim" not in config["heterogeneous_moe"]
    assert all("temp_encoder" not in item for item in config["adapter"]["exclude_modules"])
    assert all("temp_encoder" not in item for item in config["adapter_finetune"]["trainable_name_filters"])


def test_catb_uses_equal_weight_calibration_before_adaptive_updates():
    balancer = AdaptiveMTLBalancer(alpha=1.0, beta=1.0, gamma=0.5, ema_span=10)
    assert (balancer.lambda_T, balancer.lambda_S) == (0.5, 0.5)

    balancer.initialize_from_calibration(L_T=3.0, L_S=1.0, epoch=1)
    assert np.isclose(balancer.lambda_T, 0.75)
    assert np.isclose(balancer.lambda_S, 0.25)
    assert balancer.L_ema == [3.0, 1.0]
    assert balancer.L_prev_norm == [1.0, 1.0]
    assert balancer.history[-1]["phase"] == "equal_weight_calibration"
