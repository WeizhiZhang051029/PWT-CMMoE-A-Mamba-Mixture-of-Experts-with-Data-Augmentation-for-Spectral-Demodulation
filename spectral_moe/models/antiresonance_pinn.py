from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


QUAN_FRY_COEFFICIENTS = (
    1.31405, 1.779e-4, -1.05e-6, 1.6e-8, -2.02e-6,
    15.868, 0.01155, -0.00423, -4382.0, 1.1455e6,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("antiresonance_pinn requires PyTorch.")


def quan_fry_refractive_index(salinity_percent, temperature_c, wavelength_nm):

    _require_torch()
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9 = QUAN_FRY_COEFFICIENTS
    salinity_percent, temperature_c, wavelength_nm = torch.broadcast_tensors(
        salinity_percent, temperature_c, wavelength_nm
    )
    inv_lambda = wavelength_nm.reciprocal()
    return (
        k0
        + (k1 + k2 * temperature_c + k3 * temperature_c.square()) * salinity_percent
        + k4 * temperature_c.square()
        + (k5 + k6 * salinity_percent + k7 * temperature_c) * inv_lambda
        + k8 * inv_lambda.square()
        + k9 * inv_lambda.pow(3)
    )


def fused_silica_index(wavelength_nm, temperature_c, thermo_optic_per_c=1.0e-5):

    _require_torch()
    wavelength_um = wavelength_nm * 1.0e-3
    lam2 = wavelength_um.square()

    n2 = 1.0
    for b, c in ((0.6961663, 0.0684043**2), (0.4079426, 0.1162414**2), (0.8974794, 9.896161**2)):
        n2 = n2 + b * lam2 / (lam2 - c)
    return torch.sqrt(n2.clamp_min(1.0)) + thermo_optic_per_c * (temperature_c - 20.0)


if nn is not None:

    @dataclass(frozen=True)
    class AntiResonanceConfig:
        wall_thickness_um: float = 26.5
        resonance_orders: tuple[int, ...] = (18, 19)
        salinity_to_percent: float = 0.1
        reference_temperature_c: float = 20.0
        max_thickness_correction_um: float = 1.5
        max_cladding_index_correction: float = 0.01
        fixed_point_steps: int = 12
        thermo_optic_per_c: float = 1.0e-5


    class AntiResonancePINN(nn.Module):


        def __init__(self, config: AntiResonanceConfig | None = None) -> None:
            super().__init__()
            self.config = config or AntiResonanceConfig()
            if not self.config.resonance_orders or any(n <= 0 for n in self.config.resonance_orders):
                raise ValueError("resonance_orders must contain positive integers")
            if self.config.wall_thickness_um <= 0 or self.config.fixed_point_steps < 1:
                raise ValueError("wall thickness must be positive and fixed_point_steps >= 1")
            self.register_buffer("orders", torch.tensor(self.config.resonance_orders, dtype=torch.float32))
            self.raw_thickness_correction = nn.Parameter(torch.zeros(()))
            self.raw_cladding_correction = nn.Parameter(torch.zeros(()))

        @property
        def effective_thickness_um(self):
            return self.config.wall_thickness_um + self.config.max_thickness_correction_um * torch.tanh(
                self.raw_thickness_correction
            )

        @property
        def effective_cladding_index_correction(self):
            return self.config.max_cladding_index_correction * torch.tanh(self.raw_cladding_correction)

        def predicted_troughs_nm(self, temperature_c, salinity_ppt):

            temperature_c = temperature_c.reshape(-1, 1)
            salinity_percent = salinity_ppt.reshape(-1, 1) * self.config.salinity_to_percent
            orders = self.orders.to(device=temperature_c.device, dtype=temperature_c.dtype).reshape(1, -1)

            wavelength_nm = torch.full_like(temperature_c * orders, 1550.0)
            thickness_nm = self.effective_thickness_um.to(dtype=temperature_c.dtype) * 1000.0
            for _ in range(self.config.fixed_point_steps):
                liquid = quan_fry_refractive_index(salinity_percent, temperature_c, wavelength_nm)
                cladding = fused_silica_index(
                    wavelength_nm, temperature_c, self.config.thermo_optic_per_c
                ) + self.effective_cladding_index_correction.to(dtype=temperature_c.dtype)
                radicand = (cladding.square() - liquid.square()).clamp_min(1.0e-8)
                wavelength_nm = 2.0 * thickness_nm * torch.sqrt(radicand) / orders
            return wavelength_nm

        def physical_derivatives(self, temperature_c, salinity_ppt):

            t = temperature_c.reshape(-1).detach().clone().requires_grad_(True)
            s = salinity_ppt.reshape(-1).detach().clone().requires_grad_(True)
            troughs = self.predicted_troughs_nm(t, s)
            grad_t, grad_s = [], []
            for column in range(troughs.shape[1]):
                dt, ds = torch.autograd.grad(
                    troughs[:, column].sum(), (t, s), create_graph=True, retain_graph=True
                )
                grad_t.append(dt)
                grad_s.append(ds)
            return torch.stack(grad_t, dim=1), torch.stack(grad_s, dim=1)


    def calibrate_antiresonance_prior(
        tracked_centers_nm: Iterable[float],
        reference_temperature_c: float,
        reference_salinity_ppt: float,
        nominal_wall_thickness_um: float,
        candidate_orders: Iterable[int] = range(10, 31),
        fixed_point_steps: int = 12,
    ) -> dict:


        centers = [float(v) for v in tracked_centers_nm]
        orders = [int(v) for v in candidate_orders if int(v) > 0]
        if not centers or not orders or nominal_wall_thickness_um <= 0:
            raise ValueError("centers, positive candidate_orders and nominal wall thickness are required")
        t = torch.tensor([float(reference_temperature_c)], dtype=torch.float32)
        s = torch.tensor([float(reference_salinity_ppt)], dtype=torch.float32)
        required_thickness = []
        selected_orders = []
        for center in centers:
            candidates = []
            for order in orders:
                def predicted_at(thickness_um: float) -> float:
                    probe = AntiResonancePINN(AntiResonanceConfig(
                        wall_thickness_um=thickness_um,
                        resonance_orders=(order,), fixed_point_steps=fixed_point_steps,
                        max_thickness_correction_um=0.0,
                        max_cladding_index_correction=0.0,
                    ))
                    return float(probe.predicted_troughs_nm(t, s).item())


                lower, upper = 0.1, 100.0
                for _ in range(36):
                    middle = (lower + upper) / 2.0
                    if predicted_at(middle) < center:
                        lower = middle
                    else:
                        upper = middle
                implied = (lower + upper) / 2.0
                candidates.append((abs(implied - nominal_wall_thickness_um), order, implied))
            _, order, implied = min(candidates, key=lambda row: row[0])
            selected_orders.append(int(order))
            required_thickness.append(float(implied))
        calibrated_thickness = float(torch.tensor(required_thickness).median().item())
        calibrated = AntiResonancePINN(AntiResonanceConfig(
            wall_thickness_um=calibrated_thickness,
            resonance_orders=tuple(selected_orders), fixed_point_steps=fixed_point_steps,
            max_thickness_correction_um=0.0,
            max_cladding_index_correction=0.0,
        ))
        predicted = calibrated.predicted_troughs_nm(t, s).detach().cpu().reshape(-1).tolist()
        errors = [abs(a - b) for a, b in zip(predicted, centers)]
        return {
            "orders": selected_orders,
            "wall_thickness_um": calibrated_thickness,
            "reference_temperature_c": float(reference_temperature_c),
            "reference_salinity_ppt": float(reference_salinity_ppt),
            "predicted_centers_nm": predicted,
            "center_errors_nm": errors,
            "max_center_error_nm": max(errors),
            "nominal_thickness_deviation_um": calibrated_thickness - float(nominal_wall_thickness_um),
        }


    def soft_trough_locations(spectrum, wavelength_nm, centers_nm: Iterable[float], half_window_nm=35.0, temperature=0.08):

        if temperature <= 0 or half_window_nm <= 0:
            raise ValueError("temperature and half_window_nm must be positive")
        if spectrum.ndim == 3:
            spectrum = spectrum.squeeze(1)
        if spectrum.ndim != 2:
            raise ValueError("spectrum must have shape [B, L] or [B, 1, L]")
        wavelength_nm = wavelength_nm.to(device=spectrum.device, dtype=spectrum.dtype).reshape(-1)
        if wavelength_nm.numel() != spectrum.shape[1]:
            raise ValueError("wavelength grid length must equal spectrum length")
        result = []
        for center in centers_nm:
            mask = (wavelength_nm - float(center)).abs() <= half_window_nm
            if int(mask.sum()) < 3:
                raise ValueError("trough window contains fewer than three wavelength points")
            local_wavelength = wavelength_nm[mask]
            local_spectrum = spectrum[:, mask]
            weights = torch.softmax(-local_spectrum / temperature, dim=-1)
            result.append((weights * local_wavelength).sum(dim=-1))
        return torch.stack(result, dim=1)


    def antiresonance_trough_loss(observed_troughs_nm, expected_troughs_nm, scale_nm=1.0):

        if scale_nm <= 0:
            raise ValueError("scale_nm must be positive")
        return ((observed_troughs_nm - expected_troughs_nm) / scale_nm).square().mean()


else:
    class AntiResonanceConfig:
        def __init__(self, *args, **kwargs):
            _require_torch()

    class AntiResonancePINN:
        def __init__(self, *args, **kwargs):
            _require_torch()
