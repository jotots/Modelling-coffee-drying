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
    conservation_diagnostics,
    flat_plate_transfer,
    loss_components,
    physics_residuals,
    saturation_pressure_water,
    surface_fluxes,
    vaporization_enthalpy_barnard,
    water_activity_barnard,
)


def _barnard_activity(moisture_db: float) -> float:
    """Independently restate Barnard et al. (2024) Table 1 water activity."""
    x = moisture_db
    if x < 0.22:
        return 1232.3 * x**4 - 646.14 * x**3 + 98.541 * x**2 - 0.3406 * x - 0.0008
    if x <= 0.6:
        return -0.5129 * x**2 + 0.6398 * x + 0.5995
    return 0.0013 * x**3 - 0.0197 * x**2 + 0.1164 * x + 0.7448


def _barnard_vaporization_enthalpy(temperature_K: float, moisture_db: float) -> float:
    """Independently restate Barnard Table 1 h_v = h_fg + dh_w in J/kg."""
    x = moisture_db
    latent = -2.394 * (temperature_K - 273.15) + 2502.1
    if x <= 0.02:
        wetting = 1420.95
    elif x <= 0.2:
        wetting = (
            -8.2068e6 * x**4
            + 4.0003e6 * x**3
            - 6.1614e5 * x**2
            + 2.3682e4 * x
            + 1.1631e3
        )
    else:
        wetting = -22.583 * x + 130.18
    return 1000.0 * (latent + wetting)


def _conditions(count: int = 1) -> tuple[Tensor, Tensor]:
    """Return standardized features and their corresponding SI metadata."""
    features = torch.zeros((count, 4), dtype=torch.float64)
    physical = torch.tensor([[0.22, 450e-6, 0.8, 500.0]] * count, dtype=torch.float64)
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
        ({"water_specific_heat_j_kg_k": 0.0}, "water_specific_heat_j_kg_k"),
        ({"water_specific_heat_j_kg_k": -4180.0}, "water_specific_heat_j_kg_k"),
        ({"water_specific_heat_j_kg_k": float("nan")}, "water_specific_heat_j_kg_k"),
        ({"water_specific_heat_j_kg_k": float("inf")}, "water_specific_heat_j_kg_k"),
        ({"activation_energy_j_mol": 0.0}, "activation_energy_j_mol"),
        ({"activation_energy_j_mol": float("nan")}, "activation_energy_j_mol"),
    ],
)
def test_physics_configuration_rejects_invalid_values(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PhysicsConfig(**kwargs)


def test_physics_configuration_declares_water_heat_capacity_without_flux_scale() -> (
    None
):
    """Wet-bed capacity needs liquid-water cp; the ad hoc heat-flux scale is gone."""
    config = PhysicsConfig()

    assert config.water_specific_heat_j_kg_k == pytest.approx(4180.0)
    assert not hasattr(config, "heat_flux_scale_w_m2")
    # Unpack a mapping so the rejected legacy keyword is checked at runtime.
    legacy_keywords = {"heat_flux_scale_w_m2": 1000.0}
    with pytest.raises(TypeError):
        PhysicsConfig(**legacy_keywords)


@pytest.mark.parametrize(
    "legacy_keyword",
    ["latent_heat_j_kg", "initial_activation_energy_j_mol"],
)
def test_physics_configuration_rejects_removed_latent_heat_and_learned_ea(
    legacy_keyword: str,
) -> None:
    """No stale constant latent heat or initial learned Ea can be configured."""
    config = PhysicsConfig()

    assert not hasattr(config, legacy_keyword)
    with pytest.raises(TypeError):
        PhysicsConfig(**{legacy_keyword: 1.0})


def test_fixed_activation_energy_is_barnard_2044_kelvin_times_gas_constant() -> None:
    config = PhysicsConfig()

    assert config.activation_energy_j_mol == pytest.approx(
        2044.0 * config.gas_constant_j_mol_k, rel=1e-15
    )
    assert config.activation_energy_j_mol == pytest.approx(16_994.76, abs=0.01)
    assert config.activation_energy_j_mol / config.gas_constant_j_mol_k == (
        pytest.approx(2044.0, rel=1e-15)
    )


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


@pytest.mark.parametrize(
    ("moisture_db", "expected"),
    [
        # X < 0.22 quartic, including the unclipped negative value at X = 0.
        (0.0, -0.0008),
        (0.1, 0.42764),
        # 0.22 <= X <= 0.6 quadratic, inclusive at both ends.
        (0.22, 0.71543164),
        (0.4, 0.773356),
        (0.6, 0.798736),
        # X > 0.6 cubic.
        (1.0, 0.8428),
    ],
)
def test_water_activity_barnard_matches_table_1_branches(
    moisture_db: float, expected: float
) -> None:
    activity = water_activity_barnard(
        torch.tensor([[moisture_db]], dtype=torch.float64)
    )

    assert activity.item() == pytest.approx(expected, abs=1e-12)
    assert activity.item() == pytest.approx(_barnard_activity(moisture_db), abs=1e-12)


def test_water_activity_barnard_is_differentiable_and_unclipped() -> None:
    moisture = torch.tensor(
        [[0.0], [0.1], [0.3], [0.9]], dtype=torch.float64, requires_grad=True
    )

    activity = water_activity_barnard(moisture)
    (gradient,) = torch.autograd.grad(activity.sum(), moisture)

    assert activity[0].item() == pytest.approx(-0.0008, abs=1e-15)
    assert torch.isfinite(gradient).all()
    # d/dX of each selected branch, written independently.
    assert gradient[0].item() == pytest.approx(-0.3406, abs=1e-12)
    assert gradient[1].item() == pytest.approx(
        4 * 1232.3 * 0.1**3 - 3 * 646.14 * 0.1**2 + 2 * 98.541 * 0.1 - 0.3406,
        abs=1e-12,
    )
    assert gradient[2].item() == pytest.approx(-2 * 0.5129 * 0.3 + 0.6398, abs=1e-12)
    assert gradient[3].item() == pytest.approx(
        3 * 0.0013 * 0.81 - 2 * 0.0197 * 0.9 + 0.1164, abs=1e-12
    )
    with pytest.raises(ValueError, match="finite"):
        water_activity_barnard(torch.tensor([[float("nan")]], dtype=torch.float64))
    with pytest.raises(ValueError, match="floating-point"):
        water_activity_barnard(torch.tensor([[1]]))


def test_negative_activity_near_dry_surface_condenses_without_flooring() -> None:
    """At X -> 0 the published activity is -0.0008, so dry-air flux is negative."""
    features, physical = _conditions()

    fluxes = surface_fluxes(
        torch.tensor([[1e-12]], dtype=torch.float64),
        torch.tensor([[320.0]], dtype=torch.float64),
        features,
        physical,
        PhysicsConfig(relative_humidity=0.0),
    )

    assert fluxes.mass_flux_kg_m2_s.item() < 0.0


def test_vaporization_enthalpy_barnard_matches_table_1() -> None:
    # Ts = 298.15 K, X = 0.5: h_fg = 2442.25 and dh_w = 118.8885 kJ/kg.
    enthalpy = vaporization_enthalpy_barnard(
        torch.tensor([[298.15]], dtype=torch.float64),
        torch.tensor([[0.5]], dtype=torch.float64),
    )
    assert enthalpy.item() == pytest.approx(1000.0 * (2442.25 + 118.8885), rel=1e-12)

    # At 273.15 K, h_fg = 2502.1 kJ/kg isolates the heat of wetting.
    temperatures = torch.full((4, 1), 273.15, dtype=torch.float64)
    moisture = torch.tensor([[0.01], [0.02], [0.1], [0.2]], dtype=torch.float64)
    wetting = vaporization_enthalpy_barnard(temperatures, moisture) / 1000.0 - 2502.1
    assert wetting[0].item() == pytest.approx(1420.95, abs=1e-9)
    assert wetting[1].item() == pytest.approx(1420.95, abs=1e-9)
    assert wetting[2].item() == pytest.approx(
        _barnard_vaporization_enthalpy(273.15, 0.1) / 1000.0 - 2502.1, abs=1e-9
    )
    assert wetting[3].item() == pytest.approx(125.42, abs=1e-6)


@pytest.mark.parametrize(
    ("boundary", "tolerance_kj_kg"),
    # Published-coefficient rounding leaves gaps of about 0.023 and 0.243 kJ/kg.
    [(0.02, 0.03), (0.2, 0.25)],
)
def test_heat_of_wetting_pieces_are_continuous_at_branch_boundaries(
    boundary: float, tolerance_kj_kg: float
) -> None:
    offset = 1e-9
    temperatures = torch.full((2, 1), 273.15, dtype=torch.float64)
    moisture = torch.tensor(
        [[boundary - offset], [boundary + offset]], dtype=torch.float64
    )

    wetting = vaporization_enthalpy_barnard(temperatures, moisture) / 1000.0 - 2502.1

    assert wetting[0].item() == pytest.approx(wetting[1].item(), abs=tolerance_kj_kg)
    expected = 1421.0 if boundary == 0.02 else 125.5
    assert wetting[0].item() == pytest.approx(expected, abs=0.1)


def test_vaporization_enthalpy_rejects_invalid_inputs() -> None:
    temperature = torch.tensor([[300.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="share a shape"):
        vaporization_enthalpy_barnard(
            temperature, torch.tensor([0.5, 0.5], dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="finite"):
        vaporization_enthalpy_barnard(
            torch.tensor([[float("inf")]], dtype=torch.float64),
            torch.tensor([[0.5]], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="floating-point"):
        vaporization_enthalpy_barnard(
            torch.tensor([[300]]), torch.tensor([[0.5]], dtype=torch.float64)
        )


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
        - fluxes.vaporization_enthalpy_j_kg.item() * fluxes.mass_flux_kg_m2_s.item()
    )
    # X = X0*u = 0.8*0.8 = 0.64 at 310 K selects the X > 0.2 heat-of-wetting
    # branch; the per-point enthalpy replaces the removed constant latent heat.
    assert fluxes.vaporization_enthalpy_j_kg.item() == pytest.approx(
        _barnard_vaporization_enthalpy(310.0, 0.64), rel=1e-12
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
    theta = (temperature - config.air_temperature_K) / config.signed_temperature_scale_K
    temperature_gradient = torch.autograd.grad(theta.sum(), xi)[0]
    surface = surface_fluxes(concentration, temperature, features, physical, config)
    diffusivity = model.diffusivity(temperature, physical)
    expected_mass = (
        -physical[:, 3:4]
        * diffusivity
        * physical[:, 2:3]
        / config.bed_depth_m
        * concentration_gradient
        - surface.mass_flux_kg_m2_s
    ) / config.mass_flux_scale_kg_m2_s
    # The surface heat residual is the SI flux balance divided by the
    # conductive scale k*(T_initial - T_air)/L (Incropera nondimensional form).
    expected_heat = (
        config.product_thermal_conductivity_w_m_k
        * config.signed_temperature_scale_K
        / config.bed_depth_m
        * temperature_gradient
        - surface.required_conductive_heat_flux_w_m2
    ) / (
        config.product_thermal_conductivity_w_m_k
        * config.signed_temperature_scale_K
        / config.bed_depth_m
    )

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

    def diffusivity(self, temperature: Tensor, conditions: Tensor) -> Tensor:
        """Supply a smooth positive diffusivity for the unrelated mass residual."""
        del conditions
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
    transfer = flat_plate_transfer(physical[0, 0].item(), physical[0, 1].item(), config)
    water_activity = _barnard_activity(physical[0, 2].item() * moisture_ratio)
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
    water_activity = _barnard_activity(physical[0, 2].item() * moisture_ratio)
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
    model = _LinearSurfaceTemperatureModel(
        surface_temperature, temperature_rise, config
    )
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
    convective_gain = heat_transfer * (config.air_temperature_K - surface_temperature)
    latent_loss = (
        _barnard_vaporization_enthalpy(
            surface_temperature, physical[0, 2].item() * moisture_ratio
        )
        * mass_flux
    )
    assert latent_loss > convective_gain

    net_inward_heat_flux = convective_gain - latent_loss
    temperature_rise = (
        net_inward_heat_flux
        * config.bed_depth_m
        / config.product_thermal_conductivity_w_m_k
    )
    model = _LinearSurfaceTemperatureModel(
        surface_temperature, temperature_rise, config
    )
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

    def diffusivity(self, temperature: Tensor, conditions: Tensor) -> Tensor:
        del conditions
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
        config.time_scale_s / config.bed_depth_m**2 * (diffusivity * 0.6).item()
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
    xi = torch.tensor(((quadrature_xi + 1) / 2).reshape(-1, 1), dtype=torch.float64)
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
        physical,
    )
    integrated_pde = expected_bulk_rate - (
        config.time_scale_s / config.bed_depth_m**2 * diffusivity.item() * top_gradient
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
    diffusivity = model.diffusivity(temperature, physical)
    batch = _batch()
    components = loss_components(physics_residuals(model, batch, config))
    objective = sum(components.values())
    objective.backward()

    assert torch.all(moisture_ratio > 0)
    assert torch.all(temperature > 0)
    assert moisture_ratio[0].item() == pytest.approx(1.0, abs=1e-12)
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
    assert model.diameter_exponent.grad is not None
    assert model.speed_exponent.grad is not None
    assert torch.isfinite(model.log_reference_diffusivity.grad).all()
    assert torch.count_nonzero(model.moisture_network[-1].weight.grad).item() > 0
    assert torch.count_nonzero(model.temperature_network[-1].weight.grad).item() > 0
    assert torch.count_nonzero(model.log_reference_diffusivity.grad).item() > 0
    # The fixed Barnard activation energy is a buffer and never accrues grad.
    assert not model.activation_energy_j_mol.requires_grad
    assert model.activation_energy_j_mol.grad is None
    assert physical.shape == (3, 4)


def test_direct_fields_have_exact_initial_states_and_positive_readouts() -> None:
    """The exponential readouts preserve ICs without prescribing later kinetics."""
    torch.manual_seed(29)
    model = CoupledInversePINN().double()
    for network in (model.moisture_network, model.temperature_network):
        assert isinstance(network[-1], nn.Linear)
        assert torch.count_nonzero(network[-1].weight).item() > 0
        assert network[-1].weight.abs().max().item() < 1.0e-2
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.2, 0.2)
    features, _ = _conditions(count=4)
    tau = torch.tensor([[0.0], [0.0], [0.4], [1.0]], dtype=torch.float64)
    xi = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float64)

    moisture, temperature = model(tau, xi, features)

    torch.testing.assert_close(moisture[:2], torch.ones_like(moisture[:2]))
    torch.testing.assert_close(
        temperature[:2],
        torch.full_like(temperature[:2], model.config.initial_temperature_K),
    )
    assert torch.all(moisture > 0)
    assert torch.all(temperature > 0)
    assert model.moisture_network[0].in_features == 6
    assert model.temperature_network[0].in_features == 6
    assert model.moisture_network[-1].out_features == 1
    assert model.temperature_network[-1].out_features == 1


def test_direct_fields_have_no_imposed_bulk_monotonicity_or_convexity() -> None:
    """A positive-depth moisture history may increase after its exact IC."""
    model = CoupledInversePINN().double()
    with torch.no_grad():
        final = model.moisture_network[-1]
        assert isinstance(final, nn.Linear)
        final.weight.zero_()
        final.bias.fill_(0.2)
    features, physical = _conditions()
    tau = torch.tensor([[0.0], [0.5]], dtype=torch.float64)
    xi = torch.tensor([[0.5], [0.5]], dtype=torch.float64)

    local_moisture, _ = model(tau, xi, features.expand(2, -1))
    bulk = bulk_predictions(
        model,
        tau,
        features.expand(2, -1),
        physical.expand(2, -1),
        quadrature_points=8,
    )

    assert local_moisture[1].item() > local_moisture[0].item()
    assert bulk[1].item() > bulk[0].item()


def test_depth_quadrature_integrates_manufactured_fields_and_time_derivatives() -> None:
    """Gauss integration and autograd agree for manufactured local fields."""
    from strategy_c.physics import depth_average_fields

    features, _ = _conditions()
    tau = torch.tensor([[0.37]], dtype=torch.float64, requires_grad=True)
    moisture_mean, temperature_mean = depth_average_fields(
        _AnalyticModel(),
        tau,
        features,
        quadrature_points=2,
    )
    moisture_rate = torch.autograd.grad(moisture_mean.sum(), tau)[0]
    temperature_rate = torch.autograd.grad(temperature_mean.sum(), tau)[0]

    assert moisture_mean.item() == pytest.approx(1.0 + 0.2 * 0.37 + 0.1)
    assert temperature_mean.item() == pytest.approx(298.15 + 4.0 + 1.5 * 0.37)
    assert moisture_rate.item() == pytest.approx(0.2)
    assert temperature_rate.item() == pytest.approx(1.5)


def test_incropera_temperature_ratio_uses_signed_unbounded_air_reference() -> None:
    """Incropera theta preserves its sign and overshoots without clamping."""
    config = PhysicsConfig()

    def theta(temperature: float) -> float:
        return (temperature - config.air_temperature_K) / (
            config.initial_temperature_K - config.air_temperature_K
        )

    assert theta(config.initial_temperature_K) == pytest.approx(1.0)
    assert theta(config.air_temperature_K) == pytest.approx(0.0)
    assert theta(290.0) > 1.0
    assert theta(350.0) < 0.0
    model = CoupledInversePINN(config).double()
    features, _ = _conditions()
    tau = torch.tensor([[0.0], [0.4]], dtype=torch.float64)
    xi = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    _, temperature = model(tau, xi, features.expand(2, -1))
    temperature.sum().backward()
    assert temperature[0].item() == pytest.approx(config.initial_temperature_K)
    assert model.temperature_network[-1].weight.grad is not None
    assert torch.isfinite(model.temperature_network[-1].weight.grad).all()
    assert torch.count_nonzero(model.temperature_network[-1].weight.grad).item() > 0


def test_initial_temperature_residual_is_zero_for_the_hard_initial_condition() -> None:
    """The IC loss must compare the Incropera ratio with one, not zero."""
    torch.manual_seed(41)
    config = PhysicsConfig()
    model = CoupledInversePINN(config).double()
    batch = _batch()

    residuals = physics_residuals(model, batch, config)

    torch.testing.assert_close(
        residuals.initial_temperature,
        torch.zeros_like(residuals.initial_temperature),
        atol=1e-14,
        rtol=0.0,
    )
    initial_loss = loss_components(residuals)["initial_temperature"]
    assert initial_loss.item() < 1e-28


def test_shared_diffusivity_uses_physical_diameter_speed_and_learned_gradients() -> (
    None
):
    """Raw SI condition tensors drive each learned law parameter; Ea is fixed."""
    config = PhysicsConfig()
    model = CoupledInversePINN(config).double()
    conditions = torch.tensor([[0.22, 610e-6, 0.8, 500.0]], dtype=torch.float64)
    temperature = torch.tensor([[320.0]], dtype=torch.float64)

    diffusivity = model.diffusivity(temperature, conditions)
    diffusivity.sum().backward()
    diameter_star = conditions[:, 1:2] / config.reference_particle_diameter_m
    speed_star = conditions[:, 0:1] / config.reference_air_speed_m_s
    temperature_star = temperature / config.reference_temperature_K
    activation_energy_star = model.activation_energy_j_mol.detach() / (
        config.gas_constant_j_mol_k * config.reference_temperature_K
    )
    expected_dimensionless = (
        model.reference_diffusivity_m2_s.detach()
        * diameter_star ** model.diameter_exponent.detach()
        * speed_star ** model.speed_exponent.detach()
        * torch.exp(activation_energy_star * (1.0 - temperature_star.reciprocal()))
    )
    # Keep an independently written Kelvin/SI form as an algebraic
    # equivalence check; the implementation must not take reciprocals of
    # standardized or Celsius temperatures.
    expected_kelvin = (
        model.reference_diffusivity_m2_s.detach()
        * (conditions[:, 1:2] / config.reference_particle_diameter_m)
        ** model.diameter_exponent.detach()
        * (conditions[:, 0:1] / config.reference_air_speed_m_s)
        ** model.speed_exponent.detach()
        * torch.exp(
            model.activation_energy_j_mol.detach()
            / config.gas_constant_j_mol_k
            * (1.0 / config.reference_temperature_K - temperature.reciprocal())
        )
    )

    torch.testing.assert_close(diffusivity.detach(), expected_dimensionless)
    torch.testing.assert_close(diffusivity.detach(), expected_kelvin)
    reference_conditions = conditions.clone()
    reference_conditions[:, 0:1] = config.reference_air_speed_m_s
    reference_conditions[:, 1:2] = config.reference_particle_diameter_m
    reference_temperature = torch.full_like(temperature, config.reference_temperature_K)
    torch.testing.assert_close(
        model.diffusivity(reference_temperature, reference_conditions),
        model.reference_diffusivity_m2_s.detach().expand_as(temperature),
    )
    for parameter in (
        model.log_reference_diffusivity,
        model.diameter_exponent,
        model.speed_exponent,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert model.activation_energy_j_mol.grad is None
    assert config.reference_particle_diameter_m == pytest.approx(450e-6)
    assert config.reference_air_speed_m_s == pytest.approx(0.4)


def test_fixed_activation_energy_is_a_saved_non_trainable_buffer() -> None:
    """Ea lives in state_dict for reconstruction but is not a model parameter."""
    config = PhysicsConfig()
    model = CoupledInversePINN(config).double()

    parameter_ids = {id(parameter) for parameter in model.parameters()}
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}

    assert id(model.activation_energy_j_mol) not in parameter_ids
    assert not any("activation" in name for name in parameter_names)
    assert "fixed_activation_energy_j_mol" in buffer_names
    assert "fixed_activation_energy_j_mol" in model.state_dict()
    assert not hasattr(model, "log_activation_energy")
    assert isinstance(model.activation_energy_j_mol, Tensor)
    assert not model.activation_energy_j_mol.requires_grad
    assert model.activation_energy_j_mol.dtype == torch.float64
    assert model.activation_energy_j_mol.item() == config.activation_energy_j_mol


def test_diffusivity_temperature_ratio_matches_barnard_exponent() -> None:
    """D(343.15 K)/D(298.15 K) = exp(2044*(1/298.15 - 1/343.15)) with fixed Ea."""
    config = PhysicsConfig()
    model = CoupledInversePINN(config).double()
    reference_conditions = torch.tensor(
        [
            [
                config.reference_air_speed_m_s,
                config.reference_particle_diameter_m,
                0.8,
                500.0,
            ]
        ]
        * 2,
        dtype=torch.float64,
    )
    temperatures = torch.tensor([[343.15], [298.15]], dtype=torch.float64)

    with torch.no_grad():
        diffusivity = model.diffusivity(temperatures, reference_conditions)
    ratio = (diffusivity[0] / diffusivity[1]).item()
    expected = float(np.exp(2044.0 * (1.0 / 298.15 - 1.0 / 343.15)))

    assert ratio == pytest.approx(expected, rel=1e-12)
    assert ratio == pytest.approx(2.457, abs=5e-4)


def test_surface_boundary_excludes_only_exact_initial_corner() -> None:
    """An exact-zero surface sample is dropped while positive times remain."""
    features, physical = _conditions(count=2)
    batch = _batch()
    mixed = PhysicsBatch(
        domain_tau=batch.domain_tau.expand(2, -1),
        domain_xi=batch.domain_xi.expand(2, -1),
        domain_features=features,
        domain_conditions=physical,
        initial_xi=batch.initial_xi,
        initial_features=batch.initial_features,
        initial_conditions=batch.initial_conditions,
        base_tau=batch.base_tau,
        base_features=batch.base_features,
        base_conditions=batch.base_conditions,
        surface_tau=torch.tensor([[0.0], [0.2]], dtype=torch.float64),
        surface_features=features,
        surface_conditions=physical,
    )

    residuals = physics_residuals(CoupledInversePINN().double(), mixed)

    assert residuals.surface_mass.shape == (1, 1)
    assert residuals.surface_heat.shape == (1, 1)


def test_si_conservation_diagnostics_preserve_signed_fluxes_and_units() -> None:
    """Water/energy diagnostics report dimensional balances without claiming exactness."""
    model = CoupledInversePINN().double()
    features, physical = _conditions()

    records = conservation_diagnostics(
        model,
        features,
        physical,
        time_samples=(0.2,),
        quadrature_points=8,
    )

    assert len(records) == 1
    record = records[0]
    assert record["water_inventory_kg_m2"] > 0
    assert record["surface_evaporation_flux_kg_m2_s"] > 0
    assert record["latent_heat_flux_w_m2"] > 0
    assert record["latent_heat_flux_w_m2"] == pytest.approx(
        record["surface_vaporization_enthalpy_j_kg"]
        * record["surface_evaporation_flux_kg_m2_s"]
    )
    # h_v = h_fg(T_s) + dh_w(X) stays near the 2.4-2.6 MJ/kg range here.
    assert 2.3e6 < record["surface_vaporization_enthalpy_j_kg"] < 2.7e6
    assert record["net_heat_flux_w_m2"] == pytest.approx(
        record["convective_heat_flux_w_m2"] - record["latent_heat_flux_w_m2"]
    )
    assert record["water_balance_defect_kg_m2_s"] == pytest.approx(
        record["water_storage_rate_kg_m2_s"]
        + record["surface_evaporation_flux_kg_m2_s"]
    )
    assert record["energy_storage_minus_net_w_m2"] == pytest.approx(
        record["sensible_storage_rate_w_m2"] - record["net_heat_flux_w_m2"]
    )
    assert record["top_conduction_minus_net_w_m2"] == pytest.approx(
        record["top_conductive_heat_flux_w_m2"] - record["net_heat_flux_w_m2"]
    )
    assert all(np.isfinite(value) for value in record.values())
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        conservation_diagnostics(
            model,
            features,
            torch.tensor([[0.22, 450e-6, 0.8, 500.0]], dtype=torch.float64),
            time_samples=(0.0,),
        )


class _CurvedHeatModel(nn.Module):
    """Manufactured fields with nonzero temperature curvature and varying moisture.

    ``u = 0.9 - 0.2 tau + 0.1 xi^2`` departs from one at the test points so a
    heat capacity evaluated with the initial moisture X0 (instead of the local
    ``X = X0 u``) is detectable. ``T = 300 + 6 xi^2 + 3 tau`` has a constant
    second depth derivative so the Fourier residual has two distinct terms.
    """

    def __init__(self, config: PhysicsConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self, tau: Tensor, xi: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        del features
        concentration = 0.9 - 0.2 * tau + 0.1 * xi.square()
        temperature = 300.0 + 6.0 * xi.square() + 3.0 * tau
        return concentration, temperature

    def diffusivity(self, temperature: Tensor, conditions: Tensor) -> Tensor:
        """Supply a smooth positive diffusivity for the unrelated mass residual."""
        del conditions
        return 1e-10 * torch.exp(
            -30_000.0
            / self.config.gas_constant_j_mol_k
            * (temperature.reciprocal() - 1.0 / self.config.reference_temperature_K)
        )


def test_heat_residual_is_incropera_fourier_form_with_local_wet_bed_capacity() -> None:
    """R_heat = (1/Fo_s) dtheta/dtau - d2theta/dxi2 with alpha from local X."""
    config = PhysicsConfig()
    _, physical = _conditions()
    residuals = physics_residuals(_CurvedHeatModel(config), _batch(), config)

    # Hand-derived values at the domain sample tau=0.2, xi=0.4.
    tau_value, xi_value = 0.2, 0.4
    moisture_ratio = 0.9 - 0.2 * tau_value + 0.1 * xi_value**2
    signed_scale = config.initial_temperature_K - config.air_temperature_K
    theta_tau = 3.0 / signed_scale
    theta_xixi = 12.0 / signed_scale
    initial_moisture = physical[0, 2].item()
    dry_bulk_density = physical[0, 3].item()

    def expected_residual(local_moisture_db: float) -> float:
        """Evaluate the Incropera residual for a chosen moisture in rho*c."""
        volumetric_capacity = dry_bulk_density * (
            config.product_specific_heat_j_kg_k
            + local_moisture_db * config.water_specific_heat_j_kg_k
        )
        alpha = config.product_thermal_conductivity_w_m_k / volumetric_capacity
        heat_fourier_scale = alpha * config.time_scale_s / config.bed_depth_m**2
        return theta_tau / heat_fourier_scale - theta_xixi

    local_expected = expected_residual(initial_moisture * moisture_ratio)
    assert residuals.heat.shape == (1, 1)
    assert residuals.heat.item() == pytest.approx(local_expected, rel=1e-12)
    # Using the initial moisture X0 or a dry-only capacity must be detectable.
    assert residuals.heat.item() != pytest.approx(
        expected_residual(initial_moisture), rel=1e-6
    )
    assert residuals.heat.item() != pytest.approx(expected_residual(0.0), rel=1e-6)


def test_surface_heat_residual_is_incropera_biot_and_evaporation_number_form() -> None:
    """R_surface = dtheta/dxi + Bi*theta - E_evap equals raw/(k*S/L)."""
    config = PhysicsConfig()
    features, physical = _conditions()
    residuals = physics_residuals(_CurvedHeatModel(config), _batch(), config)

    tau_value = 0.2
    surface_moisture = 0.9 - 0.2 * tau_value + 0.1
    surface_temperature = 300.0 + 6.0 + 3.0 * tau_value
    signed_scale = config.initial_temperature_K - config.air_temperature_K
    theta_surface = (surface_temperature - config.air_temperature_K) / signed_scale
    theta_xi_surface = 12.0 / signed_scale
    fluxes = surface_fluxes(
        torch.tensor([[surface_moisture]], dtype=torch.float64),
        torch.tensor([[surface_temperature]], dtype=torch.float64),
        features,
        physical,
        config,
    )
    heat_transfer = fluxes.heat_transfer_coefficient_w_m2_k.item()
    mass_flux = fluxes.mass_flux_kg_m2_s.item()
    conductivity = config.product_thermal_conductivity_w_m_k
    depth = config.bed_depth_m
    heat_biot = heat_transfer * depth / conductivity
    evaporation_number = (
        fluxes.vaporization_enthalpy_j_kg.item()
        * mass_flux
        * depth
        / (conductivity * (config.air_temperature_K - config.initial_temperature_K))
    )
    expected = theta_xi_surface + heat_biot * theta_surface - evaporation_number

    assert mass_flux > 0.0
    assert evaporation_number > 0.0
    assert residuals.surface_heat.item() == pytest.approx(expected, rel=1e-12)

    # Regression: identical to the SI balance over the conductive scale k*S/L
    # and different from the removed fixed 1000 W m-2 normalisation.
    raw_balance = (
        conductivity * signed_scale / depth * theta_xi_surface
        - fluxes.required_conductive_heat_flux_w_m2.item()
    )
    conductive_scale = conductivity * signed_scale / depth
    assert residuals.surface_heat.item() == pytest.approx(
        raw_balance / conductive_scale, rel=1e-12
    )
    assert residuals.surface_heat.item() != pytest.approx(
        raw_balance / 1000.0, rel=1e-3
    )


def test_energy_diagnostic_storage_uses_local_wet_bed_heat_capacity() -> None:
    """Sensible storage integrates rho_d*(cp_s + X0*u*cp_w)*dT/dt over depth."""
    torch.manual_seed(53)
    config = PhysicsConfig()
    model = CoupledInversePINN(config).double()
    with torch.no_grad():
        final_layer = model.temperature_network[-1]
        assert isinstance(final_layer, nn.Linear)
        # A positive exponent rate makes dT/dt clearly nonzero at tau=0.3.
        final_layer.bias.fill_(0.05)
    features, physical = _conditions()
    points = 8
    tau_value = 0.3

    record = conservation_diagnostics(
        model,
        features,
        physical,
        time_samples=(tau_value,),
        quadrature_points=points,
    )[0]

    nodes, weights = np.polynomial.legendre.leggauss(points)
    xi = torch.tensor(((nodes + 1.0) / 2.0).reshape(-1, 1), dtype=torch.float64)
    tau = torch.full_like(xi, tau_value).requires_grad_(True)
    moisture, temperature = model(tau, xi, features.expand(points, -1))
    temperature_rate = (
        torch.autograd.grad(temperature.sum(), tau)[0] / config.time_scale_s
    )
    half_weights = torch.tensor(weights / 2.0, dtype=torch.float64).reshape(-1, 1)

    def storage(local_capacity: Tensor) -> float:
        """Depth-integrate the volumetric sensible-heat storage rate."""
        return float(
            (local_capacity * temperature_rate * half_weights).sum()
            * config.bed_depth_m
        )

    dry_bulk_density = physical[0, 3]
    initial_moisture = physical[0, 2]
    wet_capacity = dry_bulk_density * (
        config.product_specific_heat_j_kg_k
        + initial_moisture * moisture.detach() * config.water_specific_heat_j_kg_k
    )
    dry_capacity = torch.full_like(
        wet_capacity, float(dry_bulk_density * config.product_specific_heat_j_kg_k)
    )

    assert storage(wet_capacity) > 0.0
    assert record["sensible_storage_rate_w_m2"] == pytest.approx(
        storage(wet_capacity), rel=1e-10
    )
    assert record["sensible_storage_rate_w_m2"] != pytest.approx(
        storage(dry_capacity), rel=1e-3
    )
