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


_CONDITION_COUNT: Final[int] = 4
_FIELD_INPUT_COUNT: Final[int] = 2 + _CONDITION_COUNT
_ACTIVE_PHYSICS_TERMS: Final[tuple[str, ...]] = (
    "physics_mass",
    "physics_heat",
    "base_moisture",
    "base_temperature",
    "surface_mass",
    "surface_heat",
)
MODEL_SCHEMA: Final[str] = "strategy-c-local-fields-v1"
# Version 6 introduces direct local fields and changes model state keys.
# Previous schemas are deliberately rejected rather than silently migrated.
ARTIFACT_FORMAT_VERSION: Final[int] = 6
_MIN_SATURATION_TEMPERATURE_K: Final[float] = 273.15
_MAX_SATURATION_TEMPERATURE_K: Final[float] = 373.15
_WATER_CRITICAL_TEMPERATURE_K: Final[float] = 647.096
_WATER_CRITICAL_PRESSURE_PA: Final[float] = 22.064e6
_GAS_CONSTANT_J_MOL_K: Final[float] = 8.31446261815324
# Barnard et al. (2024), Table 1, writes D_w proportional to exp(-2044/T).
# Ea = 2044 K * R makes the reference-form exponent identical to that term.
_BARNARD_ACTIVATION_TEMPERATURE_K: Final[float] = 2044.0
_CELSIUS_OFFSET_K: Final[float] = 273.15


@dataclass(frozen=True)
class PhysicsConfig:
    """Validated SI constants, scales, and explicitly provisional properties.

    The product heat capacity (600 J/kg/K), conductivity (0.02 W/m/K), air
    properties, length/time scales, and ambient state preserve the existing
    notebook assumptions. There is no constant latent heat: the vaporisation
    enthalpy ``h_v(T_s, X)`` is evaluated per surface point from Barnard et
    al. (2024) Table 1 (see ``vaporization_enthalpy_barnard``). The Arrhenius
    activation energy is fixed at ``2044 K * R`` (about 16,995 J/mol), also
    from Barnard et al. (2024) Table 1. Volumetric heat capacity is the wet-bed
    value ``rho_d * (cp_s + X * cp_w)``: measured dry-solids bulk density times
    the dry-solids heat capacity plus the liquid water held at the local
    dry-basis moisture ``X = X0 * u`` (``cp_w`` = 4180 J/kg/K). The
    shared bed-diffusivity law references physical diameter, speed, and
    temperature at 450 µm, 0.4 m/s, and 298.15 K, respectively; fitted law
    parameters (D_ref, alpha, beta) are model state rather than configuration
    constants. The
    surface heat condition is scaled by the conductive flux ``k*(T_initial -
    T_air)/L`` (Incropera Biot form), so no separate heat-flux scale exists.
    """

    bed_depth_m: float = 0.01
    dish_diameter_m: float = 0.05
    time_scale_s: float = 14_400.0
    initial_temperature_K: float = 298.15
    temperature_difference_K: float = 45.0
    air_temperature_K: float = 343.15
    relative_humidity: float = 0.0
    product_specific_heat_j_kg_k: float = 600.0
    water_specific_heat_j_kg_k: float = 4180.0
    product_thermal_conductivity_w_m_k: float = 0.02
    air_dynamic_viscosity_pa_s: float = 20.44e-6
    air_kinematic_viscosity_m2_s: float = 19.86e-6
    air_specific_heat_j_kg_k: float = 1009.0
    air_thermal_conductivity_w_m_k: float = 0.02952
    air_density_kg_m3: float = 1.029
    air_vapour_diffusivity_m2_s: float = 3.0e-5
    ambient_pressure_pa: float = 101_325.0
    water_molecular_weight_kg_mol: float = 0.01801528
    gas_constant_j_mol_k: float = _GAS_CONSTANT_J_MOL_K
    reference_temperature_K: float = 298.15
    reference_particle_diameter_m: float = 450.0e-6
    reference_air_speed_m_s: float = 0.4
    initial_reference_diffusivity_m2_s: float = 1.0e-7
    activation_energy_j_mol: float = (
        _BARNARD_ACTIVATION_TEMPERATURE_K * _GAS_CONSTANT_J_MOL_K
    )
    mass_flux_scale_kg_m2_s: float = 0.01

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
            "water_specific_heat_j_kg_k",
            "product_thermal_conductivity_w_m_k",
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
            "reference_particle_diameter_m",
            "reference_air_speed_m_s",
            "initial_reference_diffusivity_m2_s",
            "activation_energy_j_mol",
            "mass_flux_scale_kg_m2_s",
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
        for name in (
            "air_temperature_K",
            "initial_temperature_K",
            "reference_temperature_K",
        ):
            temperature = getattr(self, name)
            if (
                not _MIN_SATURATION_TEMPERATURE_K
                <= temperature
                <= (_MAX_SATURATION_TEMPERATURE_K)
            ):
                raise ValueError(
                    f"{name} must be within 273.15--373.15 K for the "
                    "supported saturation-pressure model."
                )
        if not math.isclose(
            self.temperature_difference_K,
            abs(self.initial_temperature_K - self.air_temperature_K),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "temperature_difference_K must equal the magnitude of "
                "initial_temperature_K - air_temperature_K."
            )

    @property
    def signed_temperature_scale_K(self) -> float:
        """Return Incropera's signed scale, negative when air exceeds Tinitial."""
        return self.initial_temperature_K - self.air_temperature_K


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

    def diffusivity(self, temperature_K: Tensor, conditions: Tensor) -> Tensor:
        """Return positive diffusivity using temperature and raw SI conditions."""
        ...


@dataclass(frozen=True)
class SurfaceFluxes:
    """Signed surface exchange terms in SI units.

    Positive mass flux is evaporation out of the bean; a negative value is
    condensation. The required conductive heat flux follows
    ``q_cond = q_conv - h_v J``, where ``h_v`` is the per-point Barnard
    vaporisation enthalpy in J/kg at the surface temperature and moisture.
    """

    mass_flux_kg_m2_s: Tensor
    convective_heat_flux_w_m2: Tensor
    required_conductive_heat_flux_w_m2: Tensor
    heat_transfer_coefficient_w_m2_k: Tensor
    mass_transfer_coefficient_m_s: Tensor
    vaporization_enthalpy_j_kg: Tensor


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
    """Predict direct positive local fields with exact initial values.

    Two independent six-input tanh MLPs predict scalar log fields from
    ``(tau, xi, standardized conditions)``. Exponential readouts
    ``u=exp(tau*f_X)`` and ``T=T_initial*exp(tau*f_T)`` make both initial
    conditions exact and both states positive without prescribing their later
    temporal or spatial shape. In particular, bulk monotonicity and convexity
    are not architectural guarantees; observations and the coupled PDE/BC
    objective determine the fields.

    The small nonzero output-head initialization avoids disconnecting early
    gradients through the depth-dependent networks. Shared trainable
    diffusivity parameters remain a positive log-parameterized D_ref and
    unconstrained signed alpha/beta exponents. Barnard activation energy
    remains a fixed, serialized buffer.
    """

    fixed_activation_energy_j_mol: Tensor

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        super().__init__()
        self.config = PhysicsConfig() if config is None else config
        if not isinstance(self.config, PhysicsConfig):
            raise TypeError("config must be a PhysicsConfig instance.")
        self.moisture_network = self._build_network(_FIELD_INPUT_COUNT, 1)
        self.temperature_network = self._build_network(_FIELD_INPUT_COUNT, 1)
        self._initialize_output_head(self.moisture_network)
        self._initialize_output_head(self.temperature_network)
        self.log_reference_diffusivity = nn.Parameter(
            torch.tensor(math.log(self.config.initial_reference_diffusivity_m2_s))
        )
        # Build in float64 so .double() keeps the exact configured value;
        # .to(float32) rounds it like every other tensor in the module.
        self.register_buffer(
            "fixed_activation_energy_j_mol",
            torch.tensor(self.config.activation_energy_j_mol, dtype=torch.float64),
        )
        # A zero exponent means the initial law is independent of bed depth
        # and air speed; measured SI values still drive both learned powers.
        self.diameter_exponent = nn.Parameter(torch.tensor(0.0))
        self.speed_exponent = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _build_network(input_width: int, output_width: int) -> nn.Sequential:
        """Create one tanh MLP with the approved hidden widths and readout."""
        widths = (input_width, 64, 32, 32, 64, output_width)
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    @staticmethod
    def _initialize_output_head(network: nn.Sequential) -> None:
        """Start both log fields near zero with nonzero backpropagation paths."""
        final_layer = network[-1]
        if not isinstance(final_layer, nn.Linear) or final_layer.bias is None:
            raise RuntimeError("Field networks must end in a biased linear layer.")
        with torch.no_grad():
            final_layer.weight.mul_(1.0e-3)
            final_layer.bias.zero_()

    def forward(
        self, tau: Tensor, xi: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(X/X0, T_K)`` with exact initial values and positive states."""
        _validate_coordinate_inputs(tau, xi, features)
        field_inputs = torch.cat((tau, xi, features), dim=1)
        moisture_log_field = self.moisture_network(field_inputs)
        temperature_log_field = self.temperature_network(field_inputs)
        _require_finite_tensor(moisture_log_field, "moisture network log-field")
        _require_finite_tensor(temperature_log_field, "temperature network log-field")
        moisture_ratio = torch.exp(tau * moisture_log_field)
        temperature_K = self.config.initial_temperature_K * torch.exp(
            tau * temperature_log_field
        )
        _validate_state_tensor(moisture_ratio, "moisture ratio", positive=True)
        _validate_state_tensor(temperature_K, "temperature_K", positive=True)
        return moisture_ratio, temperature_K

    @property
    def reference_diffusivity_m2_s(self) -> Tensor:
        """Return the positive trainable reference diffusivity in m²/s."""
        return self.log_reference_diffusivity.exp()

    @property
    def activation_energy_j_mol(self) -> Tensor:
        """Return the fixed (non-trainable) Arrhenius activation energy in J/mol."""
        return self.fixed_activation_energy_j_mol

    @property
    def alpha(self) -> Tensor:
        """Signed particle-diameter exponent in the shared diffusivity law."""
        return self.diameter_exponent

    @property
    def beta(self) -> Tensor:
        """Signed air-speed exponent in the shared diffusivity law."""
        return self.speed_exponent

    def diffusivity(self, temperature_K: Tensor, conditions: Tensor) -> Tensor:
        """Evaluate the SI diffusivity law through explicit dimensionless groups.

        Conditions and temperature intentionally remain SI-valued at this
        boundary: standardization belongs to neural features, not constitutive
        physics or surface fluxes. Ratios below make the dimensionless groups
        explicit while the returned diffusivity retains units of m²/s.
        """
        if (
            not isinstance(temperature_K, Tensor)
            or temperature_K.ndim != 2
            or temperature_K.shape[1] != 1
        ):
            raise ValueError("temperature_K must have shape (N, 1).")
        _validate_state_tensor(temperature_K, "temperature_K", positive=True)
        _validate_physical_conditions(conditions, temperature_K.shape[0])
        if (
            conditions.dtype != temperature_K.dtype
            or conditions.device != temperature_K.device
        ):
            raise ValueError("physical conditions must match temperature dtype/device.")
        diameter = conditions[:, 1:2]
        speed = conditions[:, 0:1]
        diameter_star = diameter / self.config.reference_particle_diameter_m
        speed_star = speed / self.config.reference_air_speed_m_s
        temperature_star = temperature_K / self.config.reference_temperature_K
        activation_energy_star = self.activation_energy_j_mol / (
            self.config.gas_constant_j_mol_k * self.config.reference_temperature_K
        )
        # This is algebraically the Kelvin Arrhenius expression:
        # Ea_star*(1 - 1/T_star) = (Ea/R)*(1/T_ref - 1/T).
        # Keeping absolute Kelvin in T_star avoids an invalid reciprocal of a
        # standardized or Celsius temperature.
        temperature_factor = torch.exp(
            activation_energy_star * (1.0 - temperature_star.reciprocal())
        )
        diameter_factor = diameter_star.pow(self.diameter_exponent)
        speed_factor = speed_star.pow(self.speed_exponent)
        result = (
            self.reference_diffusivity_m2_s
            * diameter_factor
            * speed_factor
            * temperature_factor
        )
        _require_finite_tensor(result, "bed diffusivity")
        if not bool((result > 0).all().detach().cpu()):
            raise FloatingPointError("Bed diffusivity must remain strictly positive.")
        return result


def _validate_coordinate_inputs(tau: Tensor, xi: Tensor, features: Tensor) -> None:
    """Validate PINN inputs at the public model boundary."""
    for name, value in (("tau", tau), ("xi", xi)):
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 1:
            raise ValueError(f"{name} must be a tensor with shape (N, 1).")
        _require_finite_tensor(value, name)
    if tau.shape != xi.shape or tau.device != xi.device or tau.dtype != xi.dtype:
        raise ValueError("tau and xi must have matching shapes, devices, and dtypes.")
    if not tau.is_floating_point():
        raise ValueError("tau and xi must use a floating-point dtype.")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 2
        or features.shape != (tau.shape[0], _CONDITION_COUNT)
    ):
        raise ValueError("features must have shape (N, 4), matching tau and xi.")
    if features.device != tau.device or features.dtype != tau.dtype:
        raise ValueError("features must match tau and xi devices and dtypes.")
    _require_finite_tensor(features, "features")
    for name, value in (("tau", tau), ("xi", xi)):
        if not bool(((value >= 0.0) & (value <= 1.0)).all().detach().cpu()):
            raise ValueError(f"{name} must lie in the modeled interval [0, 1].")


def _validate_condition_inputs(tau: Tensor, features: Tensor) -> None:
    """Validate the temporal coordinate and standardized condition features."""
    if not isinstance(tau, Tensor) or tau.ndim != 2 or tau.shape[1] != 1:
        raise ValueError("tau must have shape (N, 1).")
    if (
        not isinstance(features, Tensor)
        or features.shape != (tau.shape[0], _CONDITION_COUNT)
        or features.device != tau.device
        or features.dtype != tau.dtype
    ):
        raise ValueError("features must have shape (N, 4) and match tau.")
    _require_finite_tensor(tau, "tau")
    _require_finite_tensor(features, "features")
    if not tau.is_floating_point():
        raise ValueError("tau and features must use a floating-point dtype.")
    if not bool(((tau >= 0.0) & (tau <= 1.0)).all().detach().cpu()):
        raise ValueError("tau must lie in the modeled interval [0, 1].")


def _require_finite_tensor(value: Tensor, name: str) -> None:
    """Reject non-finite tensor values rather than propagating invalid physics."""
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_state_tensor(value: Tensor, name: str, *, positive: bool = False) -> None:
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
        if (
            not _MIN_SATURATION_TEMPERATURE_K
            <= temperature_K
            <= (_MAX_SATURATION_TEMPERATURE_K)
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


def _require_floating_tensor(value: Tensor, name: str) -> None:
    """Reject non-tensor, integer, or non-finite inputs to property correlations."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype.")
    _require_finite_tensor(value, name)


def water_activity_barnard(moisture_db: Tensor) -> Tensor:
    """Return Barnard et al. (2024) Table 1 water activity at dry-basis ``X``.

    The piecewise polynomial is used exactly as published:

    * ``X < 0.22``: ``1232.3 X^4 - 646.14 X^3 + 98.541 X^2 - 0.3406 X - 0.0008``
    * ``0.22 <= X <= 0.6``: ``-0.5129 X^2 + 0.6398 X + 0.5995``
    * ``X > 0.6``: ``0.0013 X^3 - 0.0197 X^2 + 0.1164 X + 0.7448``

    The isotherm was fitted to potato in the source paper and is assumed for
    spent coffee grounds. By the user's decision it is neither clipped nor
    floored, so it is slightly negative near ``X = 0`` (``-0.0008`` at zero).
    ``torch.where`` keeps every branch differentiable; each branch is a
    finite polynomial, so the unselected branches cannot inject NaN gradients.
    """
    _require_floating_tensor(moisture_db, "moisture_db")
    x = moisture_db
    low = (
        1232.3 * x.pow(4)
        - 646.14 * x.pow(3)
        + 98.541 * x.square()
        - 0.3406 * x
        - 0.0008
    )
    middle = -0.5129 * x.square() + 0.6398 * x + 0.5995
    high = 0.0013 * x.pow(3) - 0.0197 * x.square() + 0.1164 * x + 0.7448
    return torch.where(x < 0.22, low, torch.where(x <= 0.6, middle, high))


def vaporization_enthalpy_barnard(
    surface_temperature_K: Tensor, moisture_db: Tensor
) -> Tensor:
    """Return Barnard et al. (2024) Table 1 ``h_v = h_fg + dh_w`` in J/kg.

    Barnard tabulates both terms in kJ/kg with temperature in degC:

    * ``h_fg = -2.394 (T_s - 273.15) + 2502.1``
    * ``dh_w = 1420.95`` for ``X <= 0.02``;
      ``-8.2068e6 X^4 + 4.0003e6 X^3 - 6.1614e5 X^2 + 2.3682e4 X + 1.1631e3``
      for ``0.02 < X <= 0.2``; and ``-22.583 X + 130.18`` for ``X > 0.2``.

    The pieces of the heat of wetting meet within rounding of the published
    coefficients (about 1421 kJ/kg at 0.02 and 125.5 kJ/kg at 0.2). The sum
    is multiplied by 1000 to return SI J/kg.
    """
    _require_floating_tensor(surface_temperature_K, "surface_temperature_K")
    _require_floating_tensor(moisture_db, "moisture_db")
    if surface_temperature_K.shape != moisture_db.shape:
        raise ValueError("surface_temperature_K and moisture_db must share a shape.")
    x = moisture_db
    latent_kj_kg = -2.394 * (surface_temperature_K - _CELSIUS_OFFSET_K) + 2502.1
    dry_wetting = torch.full_like(x, 1420.95)
    middle_wetting = (
        -8.2068e6 * x.pow(4)
        + 4.0003e6 * x.pow(3)
        - 6.1614e5 * x.square()
        + 2.3682e4 * x
        + 1.1631e3
    )
    wet_wetting = -22.583 * x + 130.18
    wetting_kj_kg = torch.where(
        x <= 0.02, dry_wetting, torch.where(x <= 0.2, middle_wetting, wet_wetting)
    )
    enthalpy = 1000.0 * (latent_kj_kg + wetting_kj_kg)
    _require_finite_tensor(enthalpy, "vaporization enthalpy")
    return enthalpy


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
        air_speed_m_s * characteristic_length / config.air_kinematic_viscosity_m2_s
    )
    prandtl = (
        config.air_dynamic_viscosity_pa_s
        * config.air_specific_heat_j_kg_k
        / config.air_thermal_conductivity_w_m_k
    )
    schmidt = config.air_kinematic_viscosity_m2_s / config.air_vapour_diffusivity_m2_s
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

    Water activity follows ``water_activity_barnard`` (Barnard et al. 2024,
    Table 1; fitted to potato and assumed for spent coffee grounds) at the
    absolute dry-basis surface moisture ``X = X0 * u``. The latent term uses
    the per-point ``vaporization_enthalpy_barnard(T_s, X)``. No flux or
    activity clipping is performed, so a slightly negative activity near
    ``X = 0`` yields a small negative (condensing) flux.
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
    if (
        surface_temperature_K.dtype != moisture_ratio.dtype
        or surface_temperature_K.device != moisture_ratio.device
    ):
        raise ValueError("surface temperature must match moisture dtype and device.")
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
    activity = water_activity_barnard(actual_moisture_db)
    vaporization_enthalpy = vaporization_enthalpy_barnard(
        surface_temperature_K, actual_moisture_db
    )
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
    required_conduction = convective_heat - vaporization_enthalpy * mass_flux
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
        vaporization_enthalpy_j_kg=vaporization_enthalpy,
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
            if not bool(((xi >= 0.0) & (xi <= 1.0)).all().detach().cpu()):
                raise ValueError(f"{label} xi must lie in the modeled interval [0, 1].")
        if conditions.dtype != reference.dtype or conditions.device != reference.device:
            raise ValueError(f"{label} conditions must match coordinates.")
        _validate_physical_conditions(conditions, count)
        if features.shape != (count, _CONDITION_COUNT):
            raise ValueError(f"{label} features must have shape (N, 4).")
        if features.dtype != reference.dtype or features.device != reference.device:
            raise ValueError(f"{label} features must match coordinates.")
        _require_finite_tensor(features, f"{label} features")


def _wet_bed_volumetric_heat_capacity(
    physical_conditions: Tensor,
    moisture_ratio: Tensor,
    config: PhysicsConfig,
) -> Tensor:
    """Return ``rho_d * (cp_s + X * cp_w)`` in J m-3 K-1 at local moisture.

    ``X = X0 * u`` is the local dry-basis moisture, so the stored liquid water
    contributes its sensible heat alongside the dry solids. ``physical_conditions``
    may hold one row per sample or a single row broadcast over a depth grid.
    """
    local_moisture_db = physical_conditions[:, 2:3] * moisture_ratio
    return physical_conditions[:, 3:4] * (
        config.product_specific_heat_j_kg_k
        + local_moisture_db * config.water_specific_heat_j_kg_k
    )


def physics_residuals(
    model: PhysicsModel,
    batch: PhysicsBatch,
    config: PhysicsConfig | None = None,
) -> PhysicsResiduals:
    """Evaluate dimensionless Fick/Fourier, IC, and signed Robin residuals.

    The moisture equation differentiates the complete variable flux
    ``D(T,X) * d(X/X0)/dξ``. The dry-basis inventory factor ``rho_d*X0`` is
    retained in the physical surface condition. PDE residuals are dimensionless
    through the configured time/length scales. The heat equation uses
    Incropera's Fourier-time form with a local wet-bed diffusivity, and the
    surface heat condition uses the Biot/evaporation-number form; the surface
    mass residual is divided by the explicit SI mass-flux scale.
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
    concentration, temperature = model(domain_tau, domain_xi, batch.domain_features)
    _validate_state_tensor(concentration, "domain moisture ratio", positive=True)
    _validate_state_tensor(temperature, "domain temperature", positive=True)
    concentration_tau = _derivative(concentration, domain_tau)
    concentration_xi = _derivative(concentration, domain_xi)
    diffusivity = model.diffusivity(temperature, batch.domain_conditions)
    full_moisture_flux = diffusivity * concentration_xi
    full_flux_xi = _derivative(full_moisture_flux, domain_xi)
    mass_residual = concentration_tau - (
        selected_config.time_scale_s / selected_config.bed_depth_m**2 * full_flux_xi
    )

    dimensionless_temperature = (
        temperature - selected_config.air_temperature_K
    ) / selected_config.signed_temperature_scale_K
    temperature_tau = _derivative(dimensionless_temperature, domain_tau)
    temperature_xi = _derivative(dimensionless_temperature, domain_xi)
    temperature_flux_xi = _derivative(temperature_xi, domain_xi)
    # Incropera's 1-D conduction equation in Fourier time is
    #   dtheta/dFo_heat = d2theta/dxi2,  Fo_heat = alpha * t / L^2.
    # The network's time input stays tau = t / t_s, so dFo_heat = Fo_s * dtau
    # with Fo_s = alpha * t_s / L^2 and alpha = k / (rho_d*(cp_s + X*cp_w))
    # evaluated at the local moisture X = X0*u of each collocation point.
    volumetric_heat_capacity = _wet_bed_volumetric_heat_capacity(
        batch.domain_conditions, concentration, selected_config
    )
    thermal_diffusivity = (
        selected_config.product_thermal_conductivity_w_m_k / volumetric_heat_capacity
    )
    heat_fourier_scale = (
        thermal_diffusivity
        * selected_config.time_scale_s
        / selected_config.bed_depth_m**2
    )
    heat_residual = temperature_tau / heat_fourier_scale - temperature_flux_xi

    initial_tau = torch.zeros_like(batch.initial_xi)
    initial_moisture, initial_temperature = model(
        initial_tau, batch.initial_xi, batch.initial_features
    )
    initial_temperature_theta = (
        initial_temperature - selected_config.air_temperature_K
    ) / selected_config.signed_temperature_scale_K
    initial_moisture_residual = initial_moisture - 1.0

    base_tau = _coordinate_with_gradients(batch.base_tau, "base_tau")
    base_xi = torch.zeros_like(base_tau).requires_grad_(True)
    base_moisture, base_temperature = model(base_tau, base_xi, batch.base_features)
    base_moisture_xi = _derivative(base_moisture, base_xi)
    base_temperature_theta = (
        base_temperature - selected_config.air_temperature_K
    ) / selected_config.signed_temperature_scale_K
    base_temperature_xi = _derivative(base_temperature_theta, base_xi)

    # The prescribed uniform initial state and evaporating/heated Robin
    # condition are corner-incompatible. Remove only exact t=0 surface rows;
    # every positive-time surface sample remains in the loss.
    positive_surface = batch.surface_tau[:, 0] > 0.0
    if not bool(positive_surface.any().detach().cpu()):
        raise ValueError(
            "Surface boundary samples must include at least one positive time; "
            "the exact initial corner is excluded."
        )
    surface_tau_values = batch.surface_tau[positive_surface]
    surface_features = batch.surface_features[positive_surface]
    surface_conditions = batch.surface_conditions[positive_surface]
    surface_tau = _coordinate_with_gradients(surface_tau_values, "surface_tau")
    surface_xi = torch.ones_like(surface_tau).requires_grad_(True)
    surface_moisture, surface_temperature = model(
        surface_tau, surface_xi, surface_features
    )
    surface_moisture_xi = _derivative(surface_moisture, surface_xi)
    surface_temperature_theta = (
        surface_temperature - selected_config.air_temperature_K
    ) / selected_config.signed_temperature_scale_K
    surface_temperature_xi = _derivative(surface_temperature_theta, surface_xi)
    fluxes = surface_fluxes(
        surface_moisture,
        surface_temperature,
        surface_features,
        surface_conditions,
        selected_config,
    )
    surface_diffusivity = model.diffusivity(
        surface_temperature,
        surface_conditions,
    )
    surface_mass_raw = (
        -surface_conditions[:, 3:4]
        * surface_diffusivity
        * surface_conditions[:, 2:3]
        / selected_config.bed_depth_m
        * surface_moisture_xi
        - fluxes.mass_flux_kg_m2_s
    )
    # Incropera nondimensional surface energy balance at xi = 1 (exposed top):
    #   dtheta/dxi + Bi_heat * theta_s - E_evap = 0
    # with Bi_heat = h_T*L/k and E_evap = h_v*J*L/(k*(T_air - T_initial)).
    # This is the SI balance k*dT/dz = h_T*(T_air - T_s) - h_v*J divided by the
    # conductive scale k*(T_initial - T_air)/L. It is a pure flux balance with
    # no accumulation term: a boundary surface has no volume to store heat.
    conductivity = selected_config.product_thermal_conductivity_w_m_k
    heat_biot = (
        fluxes.heat_transfer_coefficient_w_m2_k
        * selected_config.bed_depth_m
        / conductivity
    )
    air_minus_initial_temperature = -selected_config.signed_temperature_scale_K
    evaporation_number = (
        fluxes.vaporization_enthalpy_j_kg
        * fluxes.mass_flux_kg_m2_s
        * selected_config.bed_depth_m
        / (conductivity * air_minus_initial_temperature)
    )
    surface_heat_residual = (
        surface_temperature_xi
        + heat_biot * surface_temperature_theta
        - evaporation_number
    )
    residuals = PhysicsResiduals(
        mass=mass_residual,
        heat=heat_residual,
        initial_moisture=initial_moisture_residual,
        # Incropera theta is one at the prescribed initial product state and
        # zero at air temperature. Subtract the prescribed value to form the
        # actual initial-condition residual (the ansatz enforces it exactly).
        initial_temperature=initial_temperature_theta - 1.0,
        base_moisture=base_moisture_xi,
        base_temperature=base_temperature_xi,
        surface_mass=surface_mass_raw / selected_config.mass_flux_scale_kg_m2_s,
        surface_heat=surface_heat_residual,
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


def _unit_interval_gauss_legendre(
    points: int, reference: Tensor
) -> tuple[Tensor, Tensor]:
    """Create differentiable-use Gauss nodes/weights in the model's placement."""
    nodes_numpy, weights_numpy = np.polynomial.legendre.leggauss(points)
    nodes = torch.as_tensor(
        (nodes_numpy + 1.0) / 2.0,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape(1, points, 1)
    weights = torch.as_tensor(
        weights_numpy / 2.0,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape(1, points, 1)
    return nodes, weights


def _depth_quadrature_fields(
    model: PhysicsModel,
    tau: Tensor,
    features: Tensor,
    quadrature_points: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate both local fields on one shared unit-depth quadrature grid."""
    _validate_condition_inputs(tau, features)
    nodes, weights = _unit_interval_gauss_legendre(quadrature_points, tau)
    row_count = tau.shape[0]
    quadrature_xi = nodes.expand(row_count, -1, -1).reshape(-1, 1)
    quadrature_tau = tau.unsqueeze(1).expand(-1, quadrature_points, -1).reshape(-1, 1)
    quadrature_features = (
        features.unsqueeze(1)
        .expand(-1, quadrature_points, -1)
        .reshape(-1, _CONDITION_COUNT)
    )
    moisture, temperature = model(quadrature_tau, quadrature_xi, quadrature_features)
    expected_shape = (row_count * quadrature_points, 1)
    if moisture.shape != expected_shape or temperature.shape != expected_shape:
        raise ValueError("Model fields must have shape (N, 1) at quadrature points.")
    _validate_state_tensor(moisture, "quadrature moisture ratio", positive=True)
    _validate_state_tensor(temperature, "quadrature temperature", positive=True)
    return (
        moisture.reshape(row_count, quadrature_points, 1),
        temperature.reshape(row_count, quadrature_points, 1),
        weights,
    )


def depth_average_fields(
    model: PhysicsModel,
    tau: Tensor,
    features: Tensor,
    *,
    quadrature_points: int = 64,
) -> tuple[Tensor, Tensor]:
    """Return differentiable depth means of moisture ratio and Kelvin temperature.

    The same Gauss-Legendre rule is used for fitted bulk data, validation,
    evaluation temperature summaries, and conservation diagnostics. The
    quadrature is part of the reported objective/evaluation configuration, not
    an analytic shortcut.
    """
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, int)
        or not 2 <= quadrature_points <= 1024
    ):
        raise ValueError("quadrature_points must be an integer from 2 through 1024.")
    moisture_grid, temperature_grid, weights = _depth_quadrature_fields(
        model, tau, features, quadrature_points
    )
    return (
        (moisture_grid * weights).sum(dim=1),
        (temperature_grid * weights).sum(dim=1),
    )


def bulk_predictions(
    model: PhysicsModel,
    tau: Tensor,
    features: Tensor,
    physical_conditions: Tensor,
    config: PhysicsConfig | None = None,
    *,
    quadrature_points: int = 64,
) -> Tensor:
    """Return absolute bulk dry-basis moisture in kg water/kg dry solids.

    The local moisture field is integrated using the configured differentiable
    Gauss-Legendre rule. Its result therefore responds to learned depth
    dependence, including through gradients to the depth input weights.
    """
    selected_config = model.config if config is None else config
    if not isinstance(selected_config, PhysicsConfig):
        raise TypeError("config must be a PhysicsConfig instance.")
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, int)
        or not 2 <= quadrature_points <= 1024
    ):
        raise ValueError("quadrature_points must be an integer from 2 through 1024.")
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
    normalized_integral, _ = depth_average_fields(
        model, tau, features, quadrature_points=quadrature_points
    )
    bulk = physical_conditions[:, 2:3] * normalized_integral
    _validate_state_tensor(bulk, "quadrature bulk moisture", positive=True)
    return bulk


def conservation_diagnostics(
    model: CoupledInversePINN,
    features: Tensor,
    physical_conditions: Tensor,
    config: PhysicsConfig | None = None,
    *,
    time_samples: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 1.0),
    quadrature_points: int = 32,
) -> list[dict[str, float]]:
    """Evaluate SI water/energy balance diagnostics without altering losses.

    Moisture inventory and its time derivative are both computed by
    Gauss-Legendre integration of the same direct local field used by the
    observation objective. Sensible heat storage integrates
    ``rho_d*(cp_s + X0*u*cp_w)*dT/dt`` with the identical rule.
    Every requested time must be strictly positive because the initial
    uniform state is corner-incompatible with the surface Robin condition.
    Base fluxes are reported explicitly: their training constraints are soft,
    so conservation defects are diagnostics rather than exact identities.
    """
    if not isinstance(model, CoupledInversePINN):
        raise TypeError("conservation diagnostics require CoupledInversePINN.")
    selected_config = model.config if config is None else config
    if (
        not isinstance(selected_config, PhysicsConfig)
        or selected_config != model.config
    ):
        raise ValueError("The diagnostic PhysicsConfig must match the model.")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 2
        or features.shape[1] != _CONDITION_COUNT
    ):
        raise ValueError("features must have shape (N, 4).")
    _require_finite_tensor(features, "diagnostic features")
    _validate_physical_conditions(physical_conditions, features.shape[0])
    if (
        physical_conditions.shape != features.shape
        or physical_conditions.dtype != features.dtype
        or physical_conditions.device != features.device
    ):
        raise ValueError("physical_conditions must match feature shape and placement.")
    if not time_samples:
        raise ValueError("time_samples must be nonempty.")
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, int)
        or quadrature_points < 2
        or quadrature_points > 1024
    ):
        raise ValueError("quadrature_points must be an integer from 2 through 1024.")
    for tau_value in time_samples:
        if (
            isinstance(tau_value, bool)
            or not isinstance(tau_value, Real)
            or not math.isfinite(tau_value)
            or not 0.0 < tau_value <= 1.0
        ):
            raise ValueError("Diagnostic times must be finite and in (0, 1].")
    if features.dtype not in (torch.float32, torch.float64):
        raise ValueError("Diagnostic tensors must use float32 or float64.")

    xi_grid, xi_weights_grid = _unit_interval_gauss_legendre(
        quadrature_points, features
    )
    xi_nodes = xi_grid.reshape(quadrature_points, 1)
    xi_weights = xi_weights_grid.reshape(quadrature_points, 1)
    diagnostics: list[dict[str, float]] = []

    for condition_index in range(features.shape[0]):
        condition = physical_conditions[condition_index : condition_index + 1]
        feature = features[condition_index : condition_index + 1]
        for tau_value in time_samples:
            with torch.enable_grad():
                tau_for_bulk = torch.tensor(
                    [[tau_value]],
                    device=features.device,
                    dtype=features.dtype,
                    requires_grad=True,
                )
                moisture_grid, _, bulk_weights = _depth_quadrature_fields(
                    model, tau_for_bulk, feature, quadrature_points
                )
                bulk = (moisture_grid * bulk_weights).sum(dim=1)
                bulk_rate = _derivative(bulk, tau_for_bulk)
                moisture_storage = (
                    condition[:, 3:4]
                    * condition[:, 2:3]
                    * selected_config.bed_depth_m
                    / selected_config.time_scale_s
                    * bulk_rate
                )
                water_inventory = (
                    condition[:, 3:4]
                    * condition[:, 2:3]
                    * selected_config.bed_depth_m
                    * bulk
                )

                tau_grid = torch.full(
                    (quadrature_points, 1),
                    tau_value,
                    device=features.device,
                    dtype=features.dtype,
                    requires_grad=True,
                )
                xi_grid = xi_nodes.clone().requires_grad_(True)
                feature_grid = feature.expand(quadrature_points, -1)
                moisture_grid, temperature_grid = model(tau_grid, xi_grid, feature_grid)
                temperature_rate = (
                    _derivative(temperature_grid, tau_grid)
                    / selected_config.time_scale_s
                )
                volumetric_capacity = _wet_bed_volumetric_heat_capacity(
                    condition, moisture_grid, selected_config
                )
                sensible_storage = selected_config.bed_depth_m * (
                    volumetric_capacity * temperature_rate * xi_weights
                ).sum(dim=0, keepdim=True)

                def endpoint_state(
                    depth_value: float,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                    """Return surface/base state and physical depth derivatives."""
                    endpoint_tau = torch.tensor(
                        [[tau_value]],
                        device=features.device,
                        dtype=features.dtype,
                        requires_grad=True,
                    )
                    endpoint_xi = torch.tensor(
                        [[depth_value]],
                        device=features.device,
                        dtype=features.dtype,
                        requires_grad=True,
                    )
                    local_moisture, local_temperature = model(
                        endpoint_tau, endpoint_xi, feature
                    )
                    moisture_gradient = _derivative(local_moisture, endpoint_xi)
                    temperature_gradient = _derivative(local_temperature, endpoint_xi)
                    return (
                        local_moisture,
                        local_temperature,
                        moisture_gradient,
                        temperature_gradient,
                    )

                (
                    surface_moisture,
                    surface_temperature,
                    surface_moisture_xi,
                    surface_temperature_xi,
                ) = endpoint_state(1.0)
                (
                    base_moisture,
                    base_temperature,
                    base_moisture_xi,
                    base_temperature_xi,
                ) = endpoint_state(0.0)
                fluxes = surface_fluxes(
                    surface_moisture,
                    surface_temperature,
                    feature,
                    condition,
                    selected_config,
                )
                base_diffusivity = model.diffusivity(base_temperature, condition)
                # Positive base flux is defined outward through the z=0 face.
                base_mass_outward = (
                    condition[:, 3:4]
                    * condition[:, 2:3]
                    * base_diffusivity
                    / selected_config.bed_depth_m
                    * base_moisture_xi
                )
                top_conduction = (
                    selected_config.product_thermal_conductivity_w_m_k
                    / selected_config.bed_depth_m
                    * surface_temperature_xi
                )
                base_conduction = (
                    selected_config.product_thermal_conductivity_w_m_k
                    / selected_config.bed_depth_m
                    * base_temperature_xi
                )
                latent_flux = (
                    fluxes.vaporization_enthalpy_j_kg * fluxes.mass_flux_kg_m2_s
                )
                net_heat_flux = fluxes.convective_heat_flux_w_m2 - latent_flux
                values = {
                    "tau": torch.tensor(tau_value, dtype=features.dtype),
                    "water_inventory_kg_m2": water_inventory.squeeze(),
                    "water_storage_rate_kg_m2_s": moisture_storage.squeeze(),
                    "surface_evaporation_flux_kg_m2_s": (
                        fluxes.mass_flux_kg_m2_s.squeeze()
                    ),
                    "water_balance_defect_kg_m2_s": (
                        moisture_storage + fluxes.mass_flux_kg_m2_s
                    ).squeeze(),
                    "base_mass_outward_flux_kg_m2_s": base_mass_outward.squeeze(),
                    "sensible_storage_rate_w_m2": sensible_storage.squeeze(),
                    "convective_heat_flux_w_m2": (
                        fluxes.convective_heat_flux_w_m2.squeeze()
                    ),
                    "latent_heat_flux_w_m2": latent_flux.squeeze(),
                    "surface_vaporization_enthalpy_j_kg": (
                        fluxes.vaporization_enthalpy_j_kg.squeeze()
                    ),
                    "net_heat_flux_w_m2": net_heat_flux.squeeze(),
                    "top_conductive_heat_flux_w_m2": top_conduction.squeeze(),
                    "base_conductive_heat_flux_w_m2": base_conduction.squeeze(),
                    "energy_storage_minus_net_w_m2": (
                        sensible_storage - net_heat_flux
                    ).squeeze(),
                    "top_conduction_minus_net_w_m2": (
                        top_conduction - net_heat_flux
                    ).squeeze(),
                }
                record = {
                    name: float(value.detach().cpu()) for name, value in values.items()
                }
                if not all(math.isfinite(value) for value in record.values()):
                    raise FloatingPointError(
                        "Conservation diagnostics contain non-finite SI values."
                    )
                diagnostics.append(record)
                # Keep the field graph local to one diagnostic sample.
                del moisture_grid, temperature_grid
    return diagnostics


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "MODEL_SCHEMA",
    "CoupledInversePINN",
    "PhysicsBatch",
    "PhysicsConfig",
    "PhysicsModel",
    "PhysicsResiduals",
    "SurfaceFluxes",
    "TransferCoefficients",
    "bulk_predictions",
    "conservation_diagnostics",
    "depth_average_fields",
    "flat_plate_transfer",
    "loss_components",
    "physics_residuals",
    "saturation_pressure_water",
    "surface_fluxes",
    "vaporization_enthalpy_barnard",
    "water_activity_barnard",
]
