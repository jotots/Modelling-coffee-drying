"""Behavioral tests for Strategy C's coupled drying physics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from strategy_c.physics import (
    CoupledInversePINN,
    PhysicsBatch,
    PhysicsConfig,
    TransferCoefficients,
    bulk_predictions,
    flat_plate_transfer,
    loss_components,
    physics_residuals,
    saturation_pressure_water,
    surface_fluxes,
)


def _conditions(count: int = 1) -> tuple[Tensor, Tensor]:
    """Return standardized features and their corresponding SI metadata."""
    features = torch.zeros((count, 4), dtype=torch.float64)
    physical = torch.tensor(
        [[0.22, 450e-6, 0.8, 500.0]] * count, dtype=torch.float64
    )
    return features, physical


def test_notebook_flat_plate_reference_examples() -> None:
    config = PhysicsConfig()

    first = flat_plate_transfer(0.22, 450e-6, config)
    second = flat_plate_transfer(0.48, 450e-6, config)

    assert first.reynolds == pytest.approx(553.8771399798591)
    assert second.reynolds == pytest.approx(1208.4592145015106)
    assert first.prandtl == pytest.approx(0.6986436314363144)
    assert first.nusselt == pytest.approx(13.866282206670709)
    assert second.nusselt == pytest.approx(20.481856213911477)
    assert first.sherwood > 0
    assert first.heat_transfer_coefficient_w_m2_k > 0
    assert first.mass_transfer_coefficient_m_s > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bed_depth_m": 0.0}, "bed_depth_m"),
        ({"air_temperature_K": 373.16}, "air_temperature_K"),
        ({"relative_humidity": 1.1}, "relative_humidity"),
        ({"air_vapour_diffusivity_m2_s": -1e-5}, "air_vapour_diffusivity"),
    ],
)
def test_physics_configuration_rejects_invalid_values(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PhysicsConfig(**kwargs)


@pytest.mark.parametrize("speed", [0.0, -0.1, float("nan"), float("inf")])
def test_transfer_correlation_rejects_invalid_air_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="air_speed_m_s"):
        flat_plate_transfer(speed, 450e-6, PhysicsConfig())


def test_saturation_pressure_is_differentiable_and_rejects_extrapolation() -> None:
    assert saturation_pressure_water(273.15) == pytest.approx(611.21, rel=2e-4)
    assert saturation_pressure_water(373.15) == pytest.approx(101417.98, rel=2e-5)
    temperature = torch.tensor([298.15], dtype=torch.float64, requires_grad=True)

    pressure = saturation_pressure_water(temperature)
    derivative = torch.autograd.grad(pressure.sum(), temperature)[0]

    assert torch.isfinite(pressure).all()
    assert torch.isfinite(derivative).all()
    assert derivative.item() > 0
    with pytest.raises(ValueError, match="273.15"):
        saturation_pressure_water(torch.tensor([272.99]))
    with pytest.raises(ValueError, match="finite"):
        saturation_pressure_water(float("nan"))


def test_surface_flux_keeps_evaporation_and_latent_heat_signs() -> None:
    config = PhysicsConfig(relative_humidity=0.0)
    features, physical = _conditions()
    moisture_ratio = torch.tensor([[0.8]], dtype=torch.float64)
    surface_temperature = torch.tensor([[310.0]], dtype=torch.float64)

    fluxes = surface_fluxes(
        moisture_ratio, surface_temperature, features, physical, config
    )

    assert fluxes.mass_flux_kg_m2_s.item() > 0
    assert fluxes.convective_heat_flux_w_m2.item() > 0
    assert fluxes.required_conductive_heat_flux_w_m2.item() < (
        fluxes.convective_heat_flux_w_m2.item()
    )
    assert fluxes.required_conductive_heat_flux_w_m2.item() == pytest.approx(
        fluxes.convective_heat_flux_w_m2.item()
        - config.latent_heat_j_kg * fluxes.mass_flux_kg_m2_s.item()
    )

    humid = PhysicsConfig(relative_humidity=1.0)
    ambient_only = surface_fluxes(
        torch.tensor([[1e-12]], dtype=torch.float64),
        torch.tensor([[343.15]], dtype=torch.float64),
        features,
        physical,
        humid,
    )
    assert ambient_only.mass_flux_kg_m2_s.item() < 0


def test_surface_boundary_residuals_use_the_declared_signed_fluxes() -> None:
    config = PhysicsConfig()
    model = _AnalyticModel()
    batch = _batch()

    residuals = physics_residuals(model, batch, config)

    tau = torch.tensor([[0.2]], dtype=torch.float64)
    xi = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    features, physical = _conditions()
    concentration, temperature = model(tau, xi, features)
    concentration_gradient = torch.autograd.grad(
        concentration.sum(), xi, create_graph=True
    )[0]
    theta = (
        temperature - config.initial_temperature_K
    ) / config.temperature_difference_K
    temperature_gradient = torch.autograd.grad(theta.sum(), xi)[0]
    surface = surface_fluxes(concentration, temperature, features, physical, config)
    diffusivity = model.diffusivity(temperature, physical[:, 2:3] * concentration)
    expected_mass = (
        -physical[:, 3:4]
        * diffusivity
        * physical[:, 2:3]
        / config.bed_depth_m
        * concentration_gradient
        - surface.mass_flux_kg_m2_s
    ) / config.mass_flux_scale_kg_m2_s
    expected_heat = (
        config.product_thermal_conductivity_w_m_k
        * config.temperature_difference_K
        / config.bed_depth_m
        * temperature_gradient
        - surface.required_conductive_heat_flux_w_m2
    ) / config.heat_flux_scale_w_m2

    assert residuals.surface_mass.item() == pytest.approx(expected_mass.item())
    assert residuals.surface_heat.item() == pytest.approx(expected_heat.item())
    assert surface.mass_flux_kg_m2_s.item() > 0
    assert surface.required_conductive_heat_flux_w_m2.item() < (
        surface.convective_heat_flux_w_m2.item()
    )


class _LinearSurfaceTemperatureModel(nn.Module):
    """Manufacture a steady linear temperature profile with a chosen surface value."""

    def __init__(
        self,
        surface_temperature_K: float,
        temperature_rise_K: float,
        config: PhysicsConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.surface_temperature_K = surface_temperature_K
        self.temperature_rise_K = temperature_rise_K

    def forward(
        self, tau: Tensor, xi: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        del features
        # A linear profile gives a constant physical temperature gradient. The
        # configured rise is T(surface) - T(base), so its sign directly records
        # the direction of conductive heat flow for base-to-surface depth.
        concentration = 0.8 + 0.0 * (tau.square() + xi.square())
        temperature = (
            self.surface_temperature_K
            + self.temperature_rise_K * (xi - 1.0)
            + 0.0 * (tau.square() + xi.square())
        )
        return concentration, temperature

    def diffusivity(self, temperature: Tensor, moisture: Tensor) -> Tensor:
        """Supply a smooth positive diffusivity for the unrelated mass residual."""
        del moisture
        return 1e-10 * torch.exp(
            -30_000.0
            / self.config.gas_constant_j_mol_k
            * (temperature.reciprocal() - 1.0 / self.config.reference_temperature_K)
        )


def _independent_surface_mass_flux(
    *,
    moisture_ratio: float,
    surface_temperature_K: float,
    relative_humidity: float,
    physical: Tensor,
    config: PhysicsConfig,
) -> tuple[float, float]:
    """Calculate evaporation from transfer data and the stated vapour-pressure law."""
    transfer = flat_plate_transfer(
        physical[0, 0].item(), physical[0, 1].item(), config
    )
    water_activity = (
        physical[0, 2].item() * moisture_ratio
        / (physical[0, 2].item() * moisture_ratio + 0.0667)
    )
    surface_vapour_term = (
        water_activity
        * saturation_pressure_water(surface_temperature_K)
        / surface_temperature_K
    )
    ambient_vapour_term = (
        relative_humidity
        * saturation_pressure_water(config.air_temperature_K)
        / config.air_temperature_K
    )
    mass_flux = (
        transfer.mass_transfer_coefficient_m_s
        * config.water_molecular_weight_kg_mol
        / config.gas_constant_j_mol_k
        * (surface_vapour_term - ambient_vapour_term)
    )
    return transfer.heat_transfer_coefficient_w_m2_k, mass_flux


def test_surface_heat_sign_for_hot_air_without_evaporation() -> None:
    """Hot air drives heat inward, so temperature rises toward the exposed surface."""
    _, physical = _conditions()
    moisture_ratio = 0.8
    surface_temperature = 320.0
    reference_config = PhysicsConfig()
    water_activity = (
        physical[0, 2].item() * moisture_ratio
        / (physical[0, 2].item() * moisture_ratio + 0.0667)
    )
    surface_vapour_term = (
        water_activity
        * saturation_pressure_water(surface_temperature)
        / surface_temperature
    )
    ambient_vapour_term = (
        saturation_pressure_water(reference_config.air_temperature_K)
        / reference_config.air_temperature_K
    )
    relative_humidity = surface_vapour_term / ambient_vapour_term
    config = PhysicsConfig(
        relative_humidity=relative_humidity,
        bed_depth_m=1e-4,
    )
    heat_transfer, mass_flux = _independent_surface_mass_flux(
        moisture_ratio=moisture_ratio,
        surface_temperature_K=surface_temperature,
        relative_humidity=relative_humidity,
        physical=physical,
        config=config,
    )
    assert abs(mass_flux) < 1e-12

    inward_heat_flux = heat_transfer * (config.air_temperature_K - surface_temperature)
    temperature_rise = (
        inward_heat_flux
        * config.bed_depth_m
        / config.product_thermal_conductivity_w_m_k
    )
    model = _LinearSurfaceTemperatureModel(surface_temperature, temperature_rise, config)
    residual = physics_residuals(model, _batch(), config).surface_heat.item()

    assert temperature_rise > 0
    assert residual == pytest.approx(0.0, abs=1e-10)


def test_surface_heat_sign_when_evaporative_loss_exceeds_convection() -> None:
    """Latent cooling exceeding convection requires temperature to fall toward surface."""
    config = PhysicsConfig(relative_humidity=0.0, bed_depth_m=1e-4)
    features, physical = _conditions()
    moisture_ratio = 0.8
    surface_temperature = 320.0
    heat_transfer, mass_flux = _independent_surface_mass_flux(
        moisture_ratio=moisture_ratio,
        surface_temperature_K=surface_temperature,
        relative_humidity=config.relative_humidity,
        physical=physical,
        config=config,
    )
    convective_gain = heat_transfer * (
        config.air_temperature_K - surface_temperature
    )
    latent_loss = config.latent_heat_j_kg * mass_flux
    assert latent_loss > convective_gain

    net_inward_heat_flux = convective_gain - latent_loss
    temperature_rise = (
        net_inward_heat_flux
        * config.bed_depth_m
        / config.product_thermal_conductivity_w_m_k
    )
    model = _LinearSurfaceTemperatureModel(surface_temperature, temperature_rise, config)
    residual = physics_residuals(model, _batch(), config).surface_heat.item()

    assert temperature_rise < 0
    assert residual == pytest.approx(0.0, abs=1e-10)


class _AnalyticModel(nn.Module):
    """Smooth manufactured fields for derivative and quadrature checks."""

    def __init__(self, mode: str = "polynomial") -> None:
        super().__init__()
        self.mode = mode
        self.config = PhysicsConfig()

    def forward(
        self, tau: Tensor, xi: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        del features
        if self.mode == "polynomial":
            concentration = 1.0 + 0.2 * tau + 0.3 * xi.square()
            temperature = 298.15 + 8.0 * xi + 1.5 * tau
        elif self.mode == "constant_temperature":
            concentration = 1.0 + 0.2 * tau + 0.3 * xi.square()
            temperature = torch.full_like(xi, 298.15)
        else:
            concentration = 1.0 + 0.5 * xi.pow(4)
            temperature = torch.full_like(xi, 298.15)
        return concentration, temperature

    def diffusivity(self, temperature: Tensor, moisture: Tensor) -> Tensor:
        del moisture
        return 1e-10 * torch.exp(
            -30_000.0
            / self.config.gas_constant_j_mol_k
            * (temperature.reciprocal() - 1.0 / self.config.reference_temperature_K)
        )


def _batch() -> PhysicsBatch:
    """Construct independent domain and boundary samples for residual tests."""
    features, physical = _conditions()
    tau = torch.tensor([[0.2]], dtype=torch.float64)
    xi = torch.tensor([[0.4]], dtype=torch.float64)
    return PhysicsBatch(
        domain_tau=tau,
        domain_xi=xi,
        domain_features=features,
        domain_conditions=physical,
        initial_xi=torch.tensor([[0.2]], dtype=torch.float64),
        initial_features=features,
        initial_conditions=physical,
        base_tau=tau,
        base_features=features,
        base_conditions=physical,
        surface_tau=tau,
        surface_features=features,
        surface_conditions=physical,
    )


def test_mass_equation_differentiates_the_full_temperature_dependent_flux() -> None:
    config = PhysicsConfig()
    batch = _batch()
    model = _AnalyticModel()

    residuals = physics_residuals(model, batch, config)

    # For c=1+0.2*tau+0.3*xi^2 and T=298.15+8*xi+1.5*tau,
    # d(D*c_xi)/dxi includes both D*c_xixi and D_T*T_xi*c_xi.
    xi = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor([[0.2]], dtype=torch.float64)
    concentration = 1.0 + 0.2 * tau + 0.3 * xi.square()
    temperature = 298.15 + 8.0 * xi + 1.5 * tau
    d_ref = torch.tensor(1.0e-10, dtype=torch.float64)
    activation = torch.tensor(30_000.0, dtype=torch.float64)
    gas_constant = config.gas_constant_j_mol_k
    diffusivity = d_ref * torch.exp(
        -activation
        / gas_constant
        * (temperature.reciprocal() - 1.0 / config.reference_temperature_K)
    )
    full_flux = diffusivity * (0.6 * xi)
    full_derivative = torch.autograd.grad(full_flux.sum(), xi)[0]
    time_derivative = 0.2
    expected = time_derivative - (
        config.time_scale_s / config.bed_depth_m**2 * full_derivative
    )
    actual = residuals.mass.item()

    assert actual == pytest.approx(expected.item(), rel=1e-10, abs=1e-10)
    without_diffusivity_derivative = time_derivative - (
        config.time_scale_s
        / config.bed_depth_m**2
        * (diffusivity * 0.6).item()
    )
    assert actual != pytest.approx(without_diffusivity_derivative, rel=1e-4)


@pytest.mark.parametrize(
    ("quadrature_points", "expected_error"),
    [(2, 0.8 / 360.0), (4, 0.0)],
)
def test_fixed_gauss_quadrature_converges_on_depth_integral(
    quadrature_points: int, expected_error: float
) -> None:
    features, physical = _conditions()
    prediction = bulk_predictions(
        _AnalyticModel("quartic"),
        torch.zeros((1, 1), dtype=torch.float64),
        features,
        physical,
        PhysicsConfig(),
        quadrature_points=quadrature_points,
    )

    assert abs(prediction.item() - 0.88) == pytest.approx(expected_error, abs=1e-12)
    if quadrature_points == 2:
        assert prediction.item() < 0.88
    else:
        assert prediction.item() == pytest.approx(0.88, abs=1e-12)


def test_integrated_mass_balance_matches_boundary_flux_difference() -> None:
    """The divergence-form PDE integrates to its physical endpoint fluxes."""
    config = PhysicsConfig()
    batch = _batch()
    model = _AnalyticModel("constant_temperature")
    points = 32
    quadrature_xi, quadrature_weights = np.polynomial.legendre.leggauss(points)
    xi = torch.tensor(
        ((quadrature_xi + 1) / 2).reshape(-1, 1), dtype=torch.float64
    )
    tau = torch.full_like(xi, 0.2)
    features, physical = _conditions()
    residual_batch = PhysicsBatch(
        domain_tau=tau,
        domain_xi=xi,
        domain_features=features.expand(points, -1),
        domain_conditions=physical.expand(points, -1),
        initial_xi=batch.initial_xi,
        initial_features=features,
        initial_conditions=physical,
        base_tau=batch.base_tau,
        base_features=features,
        base_conditions=physical,
        surface_tau=batch.surface_tau,
        surface_features=features,
        surface_conditions=physical,
    )
    residuals = physics_residuals(model, residual_batch, config)
    integrated_residual = (
        residuals.mass.reshape(-1)
        * torch.tensor(quadrature_weights / 2, dtype=torch.float64)
    ).sum()
    concentration_left, _ = model(
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
        features,
    )
    concentration_right, _ = model(
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
        features,
    )
    expected_bulk_rate = 0.2
    top_gradient = 0.6
    diffusivity = model.diffusivity(
        torch.tensor([[298.15]], dtype=torch.float64),
        torch.tensor([[0.8]], dtype=torch.float64),
    )
    integrated_pde = expected_bulk_rate - (
        config.time_scale_s
        / config.bed_depth_m**2
        * diffusivity.item()
        * top_gradient
    )
    assert concentration_right.item() > concentration_left.item()
    assert integrated_residual.item() == pytest.approx(integrated_pde, rel=1e-8)

    surface = surface_fluxes(
        torch.ones((1, 1), dtype=torch.float64),
        torch.full((1, 1), 298.15, dtype=torch.float64),
        features,
        physical,
        config,
    )
    top_outward_diffusive_flux = -(
        physical[0, 3]
        * diffusivity
        * physical[0, 2]
        / config.bed_depth_m
        * top_gradient
    )
    assert surface.mass_flux_kg_m2_s.item() > 0
    assert top_outward_diffusive_flux.item() < 0
    bulk_inventory_rate = (
        physical[0, 3]
        * config.bed_depth_m
        * physical[0, 2]
        / config.time_scale_s
        * expected_bulk_rate
    )
    integrated_balance_defect = (
        physical[0, 3]
        * config.bed_depth_m
        * physical[0, 2]
        / config.time_scale_s
        * integrated_residual
    )
    assert (bulk_inventory_rate + top_outward_diffusive_flux).item() == pytest.approx(
        integrated_balance_defect.item(), rel=1e-8
    )
    assert batch.domain_xi.item() == pytest.approx(0.4)


def test_transfer_coefficients_are_typed_and_physical() -> None:
    result = flat_plate_transfer(0.22, 450e-6, PhysicsConfig())
    assert isinstance(result, TransferCoefficients)
    assert result.schmidt > 0
    assert result.mass_transfer_coefficient_m_s > 0
    assert result.heat_transfer_coefficient_w_m2_k > 0


def test_model_outputs_and_physics_parameters_have_finite_nonzero_gradients() -> None:
    torch.manual_seed(17)
    config = PhysicsConfig()
    model = CoupledInversePINN(config).double()
    features, physical = _conditions(count=3)
    tau = torch.tensor([[0.0], [0.4], [1.0]], dtype=torch.float64)
    xi = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)

    moisture_ratio, temperature = model(tau, xi, features)
    diffusivity = model.diffusivity(temperature, moisture_ratio)
    batch = _batch()
    components = loss_components(physics_residuals(model, batch, config))
    objective = sum(components.values())
    objective.backward()

    assert torch.all(moisture_ratio > 0)
    assert torch.all(temperature > 0)
    assert torch.allclose(
        moisture_ratio,
        torch.ones_like(moisture_ratio),
        atol=0.1,
        rtol=0.0,
    )
    assert torch.allclose(
        temperature,
        torch.full_like(temperature, config.initial_temperature_K),
        atol=1.0,
        rtol=0.0,
    )
    assert torch.all(diffusivity > 0)
    assert torch.isfinite(diffusivity).all()
    assert model.moisture_network[-1].weight.grad is not None
    assert model.temperature_network[-1].weight.grad is not None
    assert torch.isfinite(model.moisture_network[-1].weight.grad).all()
    assert torch.isfinite(model.temperature_network[-1].weight.grad).all()
    assert model.log_reference_diffusivity.grad is not None
    assert model.log_activation_energy.grad is not None
    assert torch.isfinite(model.log_reference_diffusivity.grad).all()
    assert torch.isfinite(model.log_activation_energy.grad).all()
    assert torch.count_nonzero(model.moisture_network[-1].weight.grad).item() > 0
    assert torch.count_nonzero(model.temperature_network[-1].weight.grad).item() > 0
    assert torch.count_nonzero(model.log_reference_diffusivity.grad).item() > 0
    assert torch.count_nonzero(model.log_activation_energy.grad).item() > 0
    assert physical.shape == (3, 4)
