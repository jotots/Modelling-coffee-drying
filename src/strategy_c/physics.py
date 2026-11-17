"""Coupled, dimensionless drying physics and an inverse PINN model.

The neural networks receive dimensionless time/depth and train-only
standardized experiment features. Physics calculations separately receive SI
conditions in the order ``(air speed, particle diameter, X0, dry bulk
density)`` so feature scaling can never alter physical coefficients.

Thermal, transport, and sorption values inherited from the notebook are
provisional assumptions, not measured wet-bed properties. The flat-plate
correlation follows the notebook's laminar average Nusselt expression and its
heat/mass-transfer analogy. Water saturation pressure uses the IAPWS-95
Wagner formulation (Wagner & Pruss, 2002, DOI: 10.1063/1.1461829), evaluated
only over the configured 0--100 degC interval; values outside that interval
are rejected rather than extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Final, Protocol, overload
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional


_CONDITION_COUNT: Final[int] = 4
_MIN_SATURATION_TEMPERATURE_K: Final[float] = 273.15
_MAX_SATURATION_TEMPERATURE_K: Final[float] = 373.15
_WATER_CRITICAL_TEMPERATURE_K: Final[float] = 647.096
_WATER_CRITICAL_PRESSURE_PA: Final[float] = 22.064e6


@dataclass(frozen=True)
class PhysicsConfig:
    """Validated SI constants, scales, and explicitly provisional properties.

    The product heat capacity (600 J/kg/K), conductivity (0.02 W/m/K), latent
    heat, air properties, length/time scales, and ambient state preserve the
    existing notebook assumptions. The volumetric heat capacity is evaluated
    as the measured dry-solids bulk density times this dry-solids heat
    capacity; the notebook's particle density is not substituted for bed
    bulk density. Moisture-dependent diffusivity is optional and defaults to
    a constant moisture factor because no verified constitutive law is
    supplied.
    """

    bed_depth_m: float = 0.01
    dish_diameter_m: float = 0.05
    time_scale_s: float = 14_400.0
    initial_temperature_K: float = 298.15
    temperature_difference_K: float = 45.0
    air_temperature_K: float = 343.15
    relative_humidity: float = 0.0
    product_specific_heat_j_kg_k: float = 600.0
    product_thermal_conductivity_w_m_k: float = 0.02
    latent_heat_j_kg: float = 2_261_000.0
    air_dynamic_viscosity_pa_s: float = 20.44e-6
    air_kinematic_viscosity_m2_s: float = 19.86e-6
    air_specific_heat_j_kg_k: float = 1009.0
    air_thermal_conductivity_w_m_k: float = 0.02952
    air_density_kg_m3: float = 1.029
    air_vapour_diffusivity_m2_s: float = 3.0e-5
    ambient_pressure_pa: float = 101_325.0
    water_molecular_weight_kg_mol: float = 0.01801528
    gas_constant_j_mol_k: float = 8.31446261815324
    reference_temperature_K: float = 298.15
    reference_moisture_db: float = 0.8
    moisture_diffusivity_exponent: float = 0.0
    initial_reference_diffusivity_m2_s: float = 1.0e-10
    initial_activation_energy_j_mol: float = 30_000.0
    mass_flux_scale_kg_m2_s: float = 0.01
    heat_flux_scale_w_m2: float = 1000.0

    def __post_init__(self) -> None:
        """Reject invalid configuration and unsupported saturation states."""
        positive_values = (
            "bed_depth_m",
            "dish_diameter_m",
            "time_scale_s",
            "initial_temperature_K",
            "temperature_difference_K",
            "air_temperature_K",
            "product_specific_heat_j_kg_k",
            "product_thermal_conductivity_w_m_k",
            "latent_heat_j_kg",
            "air_dynamic_viscosity_pa_s",
            "air_kinematic_viscosity_m2_s",
            "air_specific_heat_j_kg_k",
            "air_thermal_conductivity_w_m_k",
            "air_density_kg_m3",
            "air_vapour_diffusivity_m2_s",
            "ambient_pressure_pa",
            "water_molecular_weight_kg_mol",
            "gas_constant_j_mol_k",
            "reference_temperature_K",
            "reference_moisture_db",
            "initial_reference_diffusivity_m2_s",
            "initial_activation_energy_j_mol",
            "mass_flux_scale_kg_m2_s",
            "heat_flux_scale_w_m2",
        )
        for field_name in positive_values:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive.")
        if (
            isinstance(self.relative_humidity, bool)
            or not isinstance(self.relative_humidity, Real)
            or not math.isfinite(self.relative_humidity)
            or not 0.0 <= self.relative_humidity <= 1.0
        ):
            raise ValueError("relative_humidity must be finite and between 0 and 1.")
        if (
            isinstance(self.moisture_diffusivity_exponent, bool)
            or not isinstance(self.moisture_diffusivity_exponent, Real)
            or not math.isfinite(self.moisture_diffusivity_exponent)
        ):
            raise ValueError("moisture_diffusivity_exponent must be finite.")
        for name in (
            "air_temperature_K",
            "initial_temperature_K",
            "reference_temperature_K",
        ):
            temperature = getattr(self, name)
            if not _MIN_SATURATION_TEMPERATURE_K <= temperature <= (
                _MAX_SATURATION_TEMPERATURE_K
            ):
                raise ValueError(
                    f"{name} must be within 273.15--373.15 K for the "
                    "supported saturation-pressure model."
                )


@dataclass(frozen=True)
class TransferCoefficients:
    """Flat-plate dimensionless numbers and SI transfer coefficients."""

    air_speed_m_s: float
    particle_diameter_m: float
    characteristic_length_m: float
    reynolds: float
    prandtl: float
    nusselt: float
    schmidt: float
    sherwood: float
    heat_transfer_coefficient_w_m2_k: float
    mass_transfer_coefficient_m_s: float
    applicability_warnings: tuple[str, ...] = ()


class PhysicsModel(Protocol):
    """Typed model contract required by residual and depth-integration APIs."""

    config: PhysicsConfig

    def __call__(
        self, tau: Tensor, xi: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return normalized moisture and absolute temperature predictions."""
        ...

    def diffusivity(self, temperature_K: Tensor, moisture_db: Tensor) -> Tensor:
        """Return positive bed diffusivity for physical state tensors."""
        ...


@dataclass(frozen=True)
class SurfaceFluxes:
    """Signed surface exchange terms in SI units.

    Positive mass flux is evaporation out of the bean; a negative value is
    condensation. The required conductive heat flux follows the specified
    convention ``h(Tair-Tsurface) - latent_heat * J``.
    """

    mass_flux_kg_m2_s: Tensor
    convective_heat_flux_w_m2: Tensor
    required_conductive_heat_flux_w_m2: Tensor
    heat_transfer_coefficient_w_m2_k: Tensor
    mass_transfer_coefficient_m_s: Tensor


@dataclass(frozen=True)
class PhysicsBatch:
    """Fixed domain, initial, base, and exposed-surface training samples.

    Every feature tensor has shape ``(N, 4)`` and holds train-fitted
    standardized features. Every conditions tensor has shape ``(N, 4)`` and
    holds unscaled SI values in the documented condition order.
    """

    domain_tau: Tensor
    domain_xi: Tensor
    domain_features: Tensor
    domain_conditions: Tensor
    initial_xi: Tensor
    initial_features: Tensor
    initial_conditions: Tensor
    base_tau: Tensor
    base_features: Tensor
    base_conditions: Tensor
    surface_tau: Tensor
    surface_features: Tensor
    surface_conditions: Tensor


@dataclass(frozen=True)
class PhysicsResiduals:
    """Dimensionless governing-equation, initial, and boundary residuals."""

    mass: Tensor
    heat: Tensor
    initial_moisture: Tensor
    initial_temperature: Tensor
    base_moisture: Tensor
    base_temperature: Tensor
    surface_mass: Tensor
    surface_heat: Tensor
    surface_mass_flux_kg_m2_s: Tensor


class CoupledInversePINN(nn.Module):
    """Two independent tanh networks with positive physical scalar outputs.

    Each network has widths ``64-32-32-64`` before its scalar readout. The
    moisture network predicts the positive normalized field ``X/X0``; the
    temperature network predicts positive absolute Kelvin. Output biases are
    centered on the initial conditions so the model starts near a physically
    meaningful state instead of emitting hundreds of kelvin of arbitrary
    positive values. Diffusivity and activation energy are trainable in log
    space, making both strictly positive without clipping.
    """

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        super().__init__()
        self.config = PhysicsConfig() if config is None else config
        if not isinstance(self.config, PhysicsConfig):
            raise TypeError("config must be a PhysicsConfig instance.")
        input_width = 2 + _CONDITION_COUNT
        self.moisture_network = self._build_network(input_width)
        self.temperature_network = self._build_network(input_width)
        self._initialize_scalar_output(
            self.moisture_network, self._inverse_softplus(1.0)
        )
        self._initialize_scalar_output(
            self.temperature_network,
            self._inverse_softplus(self.config.initial_temperature_K),
        )
        self.log_reference_diffusivity = nn.Parameter(
            torch.tensor(math.log(self.config.initial_reference_diffusivity_m2_s))
        )
        self.log_activation_energy = nn.Parameter(
            torch.tensor(math.log(self.config.initial_activation_energy_j_mol))
        )

    @staticmethod
    def _build_network(input_width: int) -> nn.Sequential:
        """Create one independent tanh MLP with the approved hidden widths."""
        widths = (input_width, 64, 32, 32, 64, 1)
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        """Choose a finite pre-activation that maps to a desired positive value."""
        return value + math.log(-math.expm1(-value))

    @staticmethod
    def _initialize_scalar_output(network: nn.Sequential, bias: float) -> None:
        """Center the scalar head while retaining its normal small weights."""
        final_layer = network[-1]
        if not isinstance(final_layer, nn.Linear) or final_layer.bias is None:
            raise RuntimeError("The scalar network must end in a biased Linear layer.")
        with torch.no_grad():
            final_layer.bias.fill_(bias)

    def forward(
        self, tau: Tensor, xi: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return positive moisture ratio ``X/X0`` and absolute temperature K."""
        _validate_coordinate_inputs(tau, xi, features)
        combined = torch.cat((tau, xi, features), dim=1)
        moisture_ratio = torch_functional.softplus(self.moisture_network(combined))
        temperature_K = torch_functional.softplus(self.temperature_network(combined))
        return moisture_ratio, temperature_K

    @property
    def reference_diffusivity_m2_s(self) -> Tensor:
        """Return the positive trainable reference diffusivity in m²/s."""
        return self.log_reference_diffusivity.exp()

    @property
    def activation_energy_j_mol(self) -> Tensor:
        """Return the positive trainable Arrhenius activation energy in J/mol."""
        return self.log_activation_energy.exp()

    def diffusivity(self, temperature_K: Tensor, moisture_db: Tensor) -> Tensor:
        """Evaluate trainable Arrhenius bed diffusivity without state clipping.

        ``reference_diffusivity_m2_s`` is defined at ``reference_temperature_K``
        and ``reference_moisture_db``. The optional moisture exponent is zero
        unless selected explicitly in the configuration.
        """
        _validate_state_tensor(temperature_K, "temperature_K", positive=True)
        _validate_state_tensor(moisture_db, "moisture_db", positive=True)
        if temperature_K.shape != moisture_db.shape:
            raise ValueError("temperature_K and moisture_db must have matching shapes.")
        temperature_factor = torch.exp(
            -self.activation_energy_j_mol
            / self.config.gas_constant_j_mol_k
            * (
                temperature_K.reciprocal()
                - 1.0 / self.config.reference_temperature_K
            )
        )
        moisture_factor = (
            moisture_db / self.config.reference_moisture_db
        ).pow(self.config.moisture_diffusivity_exponent)
        result = self.reference_diffusivity_m2_s * temperature_factor * moisture_factor
        _require_finite_tensor(result, "bed diffusivity")
        return result


def _validate_coordinate_inputs(tau: Tensor, xi: Tensor, features: Tensor) -> None:
    """Validate PINN inputs at the public model boundary."""
    for name, value in (("tau", tau), ("xi", xi)):
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 1:
            raise ValueError(f"{name} must be a tensor with shape (N, 1).")
        _require_finite_tensor(value, name)
    if tau.shape != xi.shape or tau.device != xi.device or tau.dtype != xi.dtype:
        raise ValueError("tau and xi must have matching shapes, devices, and dtypes.")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 2
        or features.shape != (tau.shape[0], _CONDITION_COUNT)
    ):
        raise ValueError("features must have shape (N, 4), matching tau and xi.")
    if features.device != tau.device or features.dtype != tau.dtype:
        raise ValueError("features must match tau and xi devices and dtypes.")
    _require_finite_tensor(features, "features")


def _require_finite_tensor(value: Tensor, name: str) -> None:
    """Reject non-finite tensor values rather than propagating invalid physics."""
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_state_tensor(
    value: Tensor, name: str, *, positive: bool = False
) -> None:
    """Validate state tensors without mutating or silently clipping them."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    _require_finite_tensor(value, name)
    if positive and not bool((value > 0).all().detach().cpu()):
        raise ValueError(f"{name} must be strictly positive.")


@overload
def saturation_pressure_water(temperature_K: float) -> float: ...


@overload
def saturation_pressure_water(temperature_K: Tensor) -> Tensor: ...


def saturation_pressure_water(temperature_K: float | Tensor) -> float | Tensor:
    """Return liquid-water saturation pressure in Pa for 273.15--373.15 K.

    The six-term Wagner equation from IAPWS-95 is differentiable with respect
    to tensor temperature. Both scalar and tensor inputs are rejected if
    non-finite or outside the documented 0--100 degC operating interval;
    there is no extrapolation or clipping.
    """
    if isinstance(temperature_K, Tensor):
        temperature = temperature_K
        if not temperature.is_floating_point():
            raise ValueError("temperature_K tensor must use a floating-point dtype.")
        _require_finite_tensor(temperature, "temperature_K")
        if not bool(
            (
                (temperature >= _MIN_SATURATION_TEMPERATURE_K)
                & (temperature <= _MAX_SATURATION_TEMPERATURE_K)
            )
            .all()
            .detach()
            .cpu()
        ):
            raise ValueError(
                "temperature_K must be within 273.15--373.15 K "
                "for water saturation pressure."
            )
    else:
        if (
            isinstance(temperature_K, bool)
            or not isinstance(temperature_K, Real)
            or not math.isfinite(temperature_K)
        ):
            raise ValueError("temperature_K must be finite.")
        if not _MIN_SATURATION_TEMPERATURE_K <= temperature_K <= (
            _MAX_SATURATION_TEMPERATURE_K
        ):
            raise ValueError(
                "temperature_K must be within 273.15--373.15 K "
                "for water saturation pressure."
            )
        temperature = torch.tensor(float(temperature_K), dtype=torch.float64)

    reduced_temperature = 1.0 - temperature / _WATER_CRITICAL_TEMPERATURE_K
    coefficients = (
        -7.85951783,
        1.84408259,
        -11.7866497,
        22.6807411,
        -15.9618719,
        1.80122502,
    )
    exponent = (_WATER_CRITICAL_TEMPERATURE_K / temperature) * (
        coefficients[0] * reduced_temperature
        + coefficients[1] * reduced_temperature.pow(1.5)
        + coefficients[2] * reduced_temperature.pow(3.0)
        + coefficients[3] * reduced_temperature.pow(3.5)
        + coefficients[4] * reduced_temperature.pow(4.0)
        + coefficients[5] * reduced_temperature.pow(7.5)
    )
    pressure = _WATER_CRITICAL_PRESSURE_PA * torch.exp(exponent)
    _require_finite_tensor(pressure, "saturation pressure")
    if isinstance(temperature_K, Tensor):
        return pressure
    return float(pressure.item())


def flat_plate_transfer(
    air_speed_m_s: float, particle_diameter_m: float, config: PhysicsConfig
) -> TransferCoefficients:
    """Compute flat-plate Nu/Sh and SI coefficients using notebook air data.

    The characteristic length is the 50 mm exposed dish diameter, not the
    particle diameter. ``particle_diameter_m`` is validated and retained for
    per-condition reporting. The notebook correlation is laminar; a warning
    accompanies Reynolds numbers beyond its conventional transition range.
    """
    if not isinstance(config, PhysicsConfig):
        raise TypeError("config must be a PhysicsConfig instance.")
    for name, value in (
        ("air_speed_m_s", air_speed_m_s),
        ("particle_diameter_m", particle_diameter_m),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")
    characteristic_length = config.dish_diameter_m
    reynolds = (
        air_speed_m_s
        * characteristic_length
        / config.air_kinematic_viscosity_m2_s
    )
    prandtl = (
        config.air_dynamic_viscosity_pa_s
        * config.air_specific_heat_j_kg_k
        / config.air_thermal_conductivity_w_m_k
    )
    schmidt = (
        config.air_kinematic_viscosity_m2_s / config.air_vapour_diffusivity_m2_s
    )
    nusselt = 0.664 * math.sqrt(reynolds) * prandtl ** (1.0 / 3.0)
    sherwood = 0.664 * math.sqrt(reynolds) * schmidt ** (1.0 / 3.0)
    warnings_for_correlation: list[str] = []
    if reynolds > 500_000.0:
        warnings_for_correlation.append(
            "Reynolds number exceeds the laminar flat-plate correlation range."
        )
    coefficients = TransferCoefficients(
        air_speed_m_s=float(air_speed_m_s),
        particle_diameter_m=float(particle_diameter_m),
        characteristic_length_m=characteristic_length,
        reynolds=reynolds,
        prandtl=prandtl,
        nusselt=nusselt,
        schmidt=schmidt,
        sherwood=sherwood,
        heat_transfer_coefficient_w_m2_k=(
            nusselt * config.air_thermal_conductivity_w_m_k / characteristic_length
        ),
        mass_transfer_coefficient_m_s=(
            sherwood * config.air_vapour_diffusivity_m2_s / characteristic_length
        ),
        applicability_warnings=tuple(warnings_for_correlation),
    )
    numeric = (
        coefficients.reynolds,
        coefficients.prandtl,
        coefficients.nusselt,
        coefficients.schmidt,
        coefficients.sherwood,
        coefficients.heat_transfer_coefficient_w_m2_k,
        coefficients.mass_transfer_coefficient_m_s,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Transfer correlation produced non-finite coefficients.")
    if coefficients.applicability_warnings:
        for message in coefficients.applicability_warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return coefficients


def _validate_physical_conditions(conditions: Tensor, rows: int) -> None:
    """Validate the unscaled SI tensor used by all physical boundary terms."""
    if (
        not isinstance(conditions, Tensor)
        or conditions.ndim != 2
        or conditions.shape != (rows, _CONDITION_COUNT)
    ):
        raise ValueError("physical conditions must have shape (N, 4).")
    _require_finite_tensor(conditions, "physical conditions")
    for index, name in enumerate(
        ("air speed", "particle diameter", "initial moisture", "dry bulk density")
    ):
        if not bool((conditions[:, index] > 0).all().detach().cpu()):
            raise ValueError(f"physical {name} must be strictly positive.")


def _transfer_tensors(
    conditions: Tensor, config: PhysicsConfig
) -> tuple[Tensor, Tensor]:
    """Vectorize the selected flat-plate correlation while preserving autograd."""
    speed = conditions[:, 0:1]
    length = config.dish_diameter_m
    reynolds = speed * length / config.air_kinematic_viscosity_m2_s
    prandtl = (
        config.air_dynamic_viscosity_pa_s
        * config.air_specific_heat_j_kg_k
        / config.air_thermal_conductivity_w_m_k
    )
    schmidt = config.air_kinematic_viscosity_m2_s / config.air_vapour_diffusivity_m2_s
    nusselt = 0.664 * torch.sqrt(reynolds) * prandtl ** (1.0 / 3.0)
    sherwood = 0.664 * torch.sqrt(reynolds) * schmidt ** (1.0 / 3.0)
    heat_coefficient = nusselt * config.air_thermal_conductivity_w_m_k / length
    mass_coefficient = sherwood * config.air_vapour_diffusivity_m2_s / length
    _require_finite_tensor(heat_coefficient, "heat transfer coefficient")
    _require_finite_tensor(mass_coefficient, "mass transfer coefficient")
    return heat_coefficient, mass_coefficient


def surface_fluxes(
    moisture_ratio: Tensor,
    surface_temperature_K: Tensor,
    features: Tensor,
    physical_conditions: Tensor,
    config: PhysicsConfig,
) -> SurfaceFluxes:
    """Calculate signed evaporation and convective/latent heat exchange.

    The water activity assumption is ``a_w = X/(X + 0.0667)`` with ``X`` on a
    dry basis. This is a provisional single-reference-point assumption, not a
    published sorption equation. No flux or activity clipping is performed.
    """
    if not isinstance(config, PhysicsConfig):
        raise TypeError("config must be a PhysicsConfig instance.")
    if (
        not isinstance(moisture_ratio, Tensor)
        or moisture_ratio.ndim != 2
        or moisture_ratio.shape[1] != 1
    ):
        raise ValueError("moisture_ratio must have shape (N, 1).")
    if (
        not isinstance(surface_temperature_K, Tensor)
        or surface_temperature_K.shape != moisture_ratio.shape
    ):
        raise ValueError("surface_temperature_K must match moisture_ratio shape.")
    _validate_state_tensor(moisture_ratio, "moisture_ratio", positive=True)
    _validate_state_tensor(
        surface_temperature_K, "surface_temperature_K", positive=True
    )
    if (
        not isinstance(features, Tensor)
        or features.shape != (moisture_ratio.shape[0], _CONDITION_COUNT)
        or features.dtype != moisture_ratio.dtype
        or features.device != moisture_ratio.device
    ):
        raise ValueError(
            "features must have shape (N, 4) and match state dtype/device."
        )
    _require_finite_tensor(features, "features")
    _validate_physical_conditions(physical_conditions, moisture_ratio.shape[0])
    if (
        physical_conditions.dtype != moisture_ratio.dtype
        or physical_conditions.device != moisture_ratio.device
    ):
        raise ValueError("physical conditions must match state dtype and device.")

    actual_moisture_db = physical_conditions[:, 2:3] * moisture_ratio
    activity = actual_moisture_db / (actual_moisture_db + 0.0667)
    surface_vapour_pressure = saturation_pressure_water(surface_temperature_K)
    ambient_vapour_pressure = saturation_pressure_water(config.air_temperature_K)
    heat_transfer, mass_transfer = _transfer_tensors(physical_conditions, config)
    mass_flux = (
        mass_transfer
        * config.water_molecular_weight_kg_mol
        / config.gas_constant_j_mol_k
        * (
            activity * surface_vapour_pressure / surface_temperature_K
            - config.relative_humidity
            * ambient_vapour_pressure
            / config.air_temperature_K
        )
    )
    convective_heat = heat_transfer * (config.air_temperature_K - surface_temperature_K)
    required_conduction = convective_heat - config.latent_heat_j_kg * mass_flux
    for name, value in (
        ("surface mass flux", mass_flux),
        ("convective heat flux", convective_heat),
        ("required conductive heat flux", required_conduction),
    ):
        _require_finite_tensor(value, name)
    return SurfaceFluxes(
        mass_flux_kg_m2_s=mass_flux,
        convective_heat_flux_w_m2=convective_heat,
        required_conductive_heat_flux_w_m2=required_conduction,
        heat_transfer_coefficient_w_m2_k=heat_transfer,
        mass_transfer_coefficient_m_s=mass_transfer,
    )


def _coordinate_with_gradients(value: Tensor, name: str) -> Tensor:
    """Ensure derivatives are available without severing model gradients."""
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape (N, 1).")
    _require_finite_tensor(value, name)
    if value.requires_grad:
        return value
    return value.clone().requires_grad_(True)


def _derivative(output: Tensor, coordinate: Tensor) -> Tensor:
    """Differentiate independent batch rows with higher-order graph retention."""
    # Manufactured solutions can be exactly constant in a coordinate. PyTorch
    # then omits the derivative graph entirely; its mathematical derivative
    # is the correctly shaped zero tensor, not a differentiation failure.
    if not output.requires_grad:
        return torch.zeros_like(coordinate)
    return torch.autograd.grad(
        output.sum(),
        coordinate,
        create_graph=True,
        retain_graph=True,
    )[0]


def _validate_batch(batch: PhysicsBatch) -> None:
    """Check all independent sample groups before expensive AD evaluation."""
    if not isinstance(batch, PhysicsBatch):
        raise TypeError("batch must be a PhysicsBatch instance.")
    groups = (
        (
            "domain",
            batch.domain_tau,
            batch.domain_xi,
            batch.domain_features,
            batch.domain_conditions,
        ),
        (
            "initial",
            None,
            batch.initial_xi,
            batch.initial_features,
            batch.initial_conditions,
        ),
        ("base", batch.base_tau, None, batch.base_features, batch.base_conditions),
        (
            "surface",
            batch.surface_tau,
            None,
            batch.surface_features,
            batch.surface_conditions,
        ),
    )
    for label, tau, xi, features, conditions in groups:
        reference = tau if tau is not None else xi
        if reference is None or not isinstance(reference, Tensor):
            raise ValueError(f"{label} samples must provide coordinates.")
        count = reference.shape[0]
        if count == 0:
            raise ValueError(f"{label} samples must contain at least one point.")
        if tau is not None:
            _validate_coordinate_inputs(
                tau,
                batch.domain_xi if label == "domain" else torch.zeros_like(tau),
                features,
            )
        if xi is not None:
            if xi.shape != (count, 1):
                raise ValueError(f"{label} xi must have shape (N, 1).")
            _require_finite_tensor(xi, f"{label} xi")
        if conditions.dtype != reference.dtype or conditions.device != reference.device:
            raise ValueError(f"{label} conditions must match coordinates.")
        _validate_physical_conditions(conditions, count)
        if features.shape != (count, _CONDITION_COUNT):
            raise ValueError(f"{label} features must have shape (N, 4).")
        if features.dtype != reference.dtype or features.device != reference.device:
            raise ValueError(f"{label} features must match coordinates.")
        _require_finite_tensor(features, f"{label} features")


def physics_residuals(
    model: PhysicsModel,
    batch: PhysicsBatch,
    config: PhysicsConfig | None = None,
) -> PhysicsResiduals:
    """Evaluate dimensionless Fick/Fourier, IC, and signed Robin residuals.

    The moisture equation differentiates the complete variable flux
    ``D(T,X) * d(X/X0)/dξ``. The dry-basis inventory factor ``rho_d*X0`` is
    retained in the physical surface condition. PDE residuals are dimensionless
    through the configured time/length scales; flux residuals are divided by
    explicit SI mass- and heat-flux scales.
    """
    selected_config = model.config if config is None else config
    if not isinstance(selected_config, PhysicsConfig):
        raise TypeError("config must be a PhysicsConfig instance.")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module implementing forward().")
    if hasattr(model, "config") and model.config != selected_config:
        raise ValueError("The model and residuals must use the same PhysicsConfig.")
    _validate_batch(batch)

    domain_tau = _coordinate_with_gradients(batch.domain_tau, "domain_tau")
    domain_xi = _coordinate_with_gradients(batch.domain_xi, "domain_xi")
    concentration, temperature = model(
        domain_tau, domain_xi, batch.domain_features
    )
    _validate_state_tensor(concentration, "domain moisture ratio", positive=True)
    _validate_state_tensor(temperature, "domain temperature", positive=True)
    concentration_tau = _derivative(concentration, domain_tau)
    concentration_xi = _derivative(concentration, domain_xi)
    actual_moisture = batch.domain_conditions[:, 2:3] * concentration
    diffusivity = model.diffusivity(temperature, actual_moisture)
    full_moisture_flux = diffusivity * concentration_xi
    full_flux_xi = _derivative(full_moisture_flux, domain_xi)
    mass_residual = concentration_tau - (
        selected_config.time_scale_s / selected_config.bed_depth_m**2 * full_flux_xi
    )

    dimensionless_temperature = (
        temperature - selected_config.initial_temperature_K
    ) / selected_config.temperature_difference_K
    temperature_tau = _derivative(dimensionless_temperature, domain_tau)
    temperature_xi = _derivative(dimensionless_temperature, domain_xi)
    temperature_flux_xi = _derivative(temperature_xi, domain_xi)
    volumetric_heat_capacity = (
        batch.domain_conditions[:, 3:4]
        * selected_config.product_specific_heat_j_kg_k
    )
    heat_residual = temperature_tau - (
        selected_config.time_scale_s
        * selected_config.product_thermal_conductivity_w_m_k
        / (volumetric_heat_capacity * selected_config.bed_depth_m**2)
        * temperature_flux_xi
    )

    initial_tau = torch.zeros_like(batch.initial_xi)
    initial_moisture, initial_temperature = model(
        initial_tau, batch.initial_xi, batch.initial_features
    )
    initial_temperature_theta = (
        initial_temperature - selected_config.initial_temperature_K
    ) / selected_config.temperature_difference_K
    initial_moisture_residual = initial_moisture - 1.0

    base_tau = _coordinate_with_gradients(batch.base_tau, "base_tau")
    base_xi = torch.zeros_like(base_tau).requires_grad_(True)
    base_moisture, base_temperature = model(
        base_tau, base_xi, batch.base_features
    )
    del base_temperature
    base_moisture_xi = _derivative(base_moisture, base_xi)
    base_temperature_field = model(base_tau, base_xi, batch.base_features)[1]
    base_temperature_theta = (
        base_temperature_field - selected_config.initial_temperature_K
    ) / selected_config.temperature_difference_K
    base_temperature_xi = _derivative(base_temperature_theta, base_xi)

    surface_tau = _coordinate_with_gradients(batch.surface_tau, "surface_tau")
    surface_xi = torch.ones_like(surface_tau).requires_grad_(True)
    surface_moisture, surface_temperature = model(
        surface_tau, surface_xi, batch.surface_features
    )
    surface_moisture_xi = _derivative(surface_moisture, surface_xi)
    surface_temperature_theta = (
        surface_temperature - selected_config.initial_temperature_K
    ) / selected_config.temperature_difference_K
    surface_temperature_xi = _derivative(surface_temperature_theta, surface_xi)
    fluxes = surface_fluxes(
        surface_moisture,
        surface_temperature,
        batch.surface_features,
        batch.surface_conditions,
        selected_config,
    )
    surface_diffusivity = model.diffusivity(
        surface_temperature,
        batch.surface_conditions[:, 2:3] * surface_moisture,
    )
    surface_mass_raw = (
        -batch.surface_conditions[:, 3:4]
        * surface_diffusivity
        * batch.surface_conditions[:, 2:3]
        / selected_config.bed_depth_m
        * surface_moisture_xi
        - fluxes.mass_flux_kg_m2_s
    )
    surface_heat_raw = (
        selected_config.product_thermal_conductivity_w_m_k
        * selected_config.temperature_difference_K
        / selected_config.bed_depth_m
        * surface_temperature_xi
        - fluxes.required_conductive_heat_flux_w_m2
    )
    residuals = PhysicsResiduals(
        mass=mass_residual,
        heat=heat_residual,
        initial_moisture=initial_moisture_residual,
        initial_temperature=initial_temperature_theta,
        base_moisture=base_moisture_xi,
        base_temperature=base_temperature_xi,
        surface_mass=surface_mass_raw / selected_config.mass_flux_scale_kg_m2_s,
        surface_heat=surface_heat_raw / selected_config.heat_flux_scale_w_m2,
        surface_mass_flux_kg_m2_s=fluxes.mass_flux_kg_m2_s,
    )
    for name, value in residuals.__dict__.items():
        _require_finite_tensor(value, f"{name} residual")
    return residuals


def loss_components(residuals: PhysicsResiduals) -> dict[str, Tensor]:
    """Return separately inspectable mean-square terms for every physics loss."""
    if not isinstance(residuals, PhysicsResiduals):
        raise TypeError("residuals must be a PhysicsResiduals instance.")
    return {
        "physics_mass": residuals.mass.square().mean(),
        "physics_heat": residuals.heat.square().mean(),
        "initial_moisture": residuals.initial_moisture.square().mean(),
        "initial_temperature": residuals.initial_temperature.square().mean(),
        "base_moisture": residuals.base_moisture.square().mean(),
        "base_temperature": residuals.base_temperature.square().mean(),
        "surface_mass": residuals.surface_mass.square().mean(),
        "surface_heat": residuals.surface_heat.square().mean(),
    }


def bulk_predictions(
    model: PhysicsModel,
    tau: Tensor,
    features: Tensor,
    physical_conditions: Tensor,
    config: PhysicsConfig | None = None,
    *,
    quadrature_points: int = 64,
) -> Tensor:
    """Integrate absolute dry-basis moisture through depth by Gauss quadrature.

    The returned ``(N, 1)`` values are ``∫ X(τ, ξ) dξ``. With the constant dry
    bulk density used by the one-dimensional slab model, this equals the
    predicted bulk dry-basis moisture; quadrature and network evaluation stay
    differentiable with respect to model parameters.
    """
    selected_config = model.config if config is None else config
    if not isinstance(selected_config, PhysicsConfig):
        raise TypeError("config must be a PhysicsConfig instance.")
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, int)
        or quadrature_points < 2
    ):
        raise ValueError("quadrature_points must be an integer of at least two.")
    if not isinstance(tau, Tensor) or tau.ndim != 2 or tau.shape[1] != 1:
        raise ValueError("tau must have shape (N, 1).")
    _require_finite_tensor(tau, "tau")
    _validate_coordinate_inputs(tau, torch.zeros_like(tau), features)
    _validate_physical_conditions(physical_conditions, tau.shape[0])
    if (
        physical_conditions.dtype != tau.dtype
        or physical_conditions.device != tau.device
    ):
        raise ValueError("physical conditions must match tau dtype and device.")
    nodes_numpy, weights_numpy = np.polynomial.legendre.leggauss(quadrature_points)
    nodes = torch.as_tensor(
        (nodes_numpy + 1.0) / 2.0, dtype=tau.dtype, device=tau.device
    ).reshape(1, quadrature_points, 1)
    weights = torch.as_tensor(
        weights_numpy / 2.0, dtype=tau.dtype, device=tau.device
    ).reshape(1, quadrature_points, 1)
    row_count = tau.shape[0]
    quadrature_xi = nodes.expand(row_count, -1, -1).reshape(-1, 1)
    quadrature_tau = (
        tau.unsqueeze(1).expand(-1, quadrature_points, -1).reshape(-1, 1)
    )
    quadrature_features = (
        features.unsqueeze(1)
        .expand(-1, quadrature_points, -1)
        .reshape(-1, _CONDITION_COUNT)
    )
    moisture_ratio, _ = model(quadrature_tau, quadrature_xi, quadrature_features)
    moisture_by_depth = moisture_ratio.reshape(row_count, quadrature_points, 1)
    normalized_integral = (moisture_by_depth * weights).sum(dim=1)
    return physical_conditions[:, 2:3] * normalized_integral

