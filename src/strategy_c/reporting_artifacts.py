"""Artifact construction helpers for Strategy C reports.

The public API remains in ``strategy_c.reporting``. This module isolates
filesystem export, manifest assembly, quality-flag auditing, and plotting from
the final-test metric calculation so each stage remains independently
maintainable.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import pickle
from typing import Final

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from strategy_c.data import (
    CurveData,
    ExperimentSplit,
    FeatureTransform,
    fit_feature_transform,
)
from strategy_c.field_visualization import (
    InternalFieldResult,
    write_internal_field_artifacts,
)
from strategy_c.loss_visualization import plot_loss_history
from strategy_c.physics import (
    ARTIFACT_FORMAT_VERSION,
    MODEL_SCHEMA,
    CoupledInversePINN,
    PhysicsConfig,
    flat_plate_transfer,
)
from strategy_c.reporting import (
    EvaluationResult,
    _PREDICTION_COLUMNS,
    _REQUIRED_EXPERIMENT_COLUMNS,
    _REQUIRED_OBSERVATION_COLUMNS,
    _finite_array,
    _require_columns,
)
from strategy_c.training import (
    SamplingConfig,
    TrainingConfig,
    TrainingResult,
    _schedule_boundaries,
)


_CONFIG_UNITS: Final[dict[str, str]] = {
    "bed_depth_m": "m",
    "dish_diameter_m": "m",
    "time_scale_s": "s",
    "initial_temperature_K": "K",
    "temperature_difference_K": "K",
    "air_temperature_K": "K",
    "relative_humidity": "fraction (0-1)",
    "product_specific_heat_j_kg_k": "J kg-1 K-1",
    "water_specific_heat_j_kg_k": "J kg-1 K-1",
    "product_thermal_conductivity_w_m_k": "W m-1 K-1",
    "air_dynamic_viscosity_pa_s": "Pa s",
    "air_kinematic_viscosity_m2_s": "m2 s-1",
    "air_specific_heat_j_kg_k": "J kg-1 K-1",
    "air_thermal_conductivity_w_m_k": "W m-1 K-1",
    "air_density_kg_m3": "kg m-3",
    "air_vapour_diffusivity_m2_s": "m2 s-1",
    "ambient_pressure_pa": "Pa",
    "water_molecular_weight_kg_mol": "kg mol-1",
    "gas_constant_j_mol_k": "J mol-1 K-1",
    "reference_temperature_K": "K",
    "reference_particle_diameter_m": "m",
    "reference_air_speed_m_s": "m s-1",
    "initial_reference_diffusivity_m2_s": "m2 s-1",
    "activation_energy_j_mol": "J mol-1",
    "mass_flux_scale_kg_m2_s": "kg m-2 s-1",
}
_ACTIVATION_ENERGY_PROVENANCE: Final[str] = (
    "fixed, Barnard et al. 2024 Table 1 (2044 K x R)"
)
_ACTIVE_RESIDUALS: Final[tuple[str, ...]] = (
    "physics_mass",
    "physics_heat",
    "base_moisture",
    "base_temperature",
    "surface_mass",
    "surface_heat",
)
_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "One-dimensional slab of 0.01 m depth with a 0.05 m exposed dish "
    "diameter, modeled as a uniform bed; heat capacity includes local liquid "
    "water (cp_s + X*cp_w). The bottom is insulated and impermeable.",
    "Air temperature is 343.15 K (70 degC), initial product temperature is "
    "298.15 K, and the modeled drying time scale is 14400 s.",
    "Ambient relative humidity is set to zero as an explicit approximation; "
    "zero humidity does not imply zero surface resistance.",
    "Water activity follows the Barnard et al. (2024) Table 1 piecewise "
    "polynomial in dry-basis X, used exactly as published without clipping "
    "or flooring (a_w = -0.0008 at X = 0). The isotherm was fitted to potato "
    "in that paper and is assumed for spent coffee grounds; it is not a "
    "verified coffee sorption equation.",
    "The vaporisation enthalpy is h_v = h_fg(T_s) + dh_w(X) per surface point "
    "from Barnard et al. (2024) Table 1 (kJ/kg converted to J/kg), assumed "
    "for spent coffee grounds; no constant latent heat is used.",
    "Dry mass, initial sample mass, initial dry-basis moisture, and dry-solid "
    "bulk density are treated as known retrospectively from measured records.",
    "Thermal properties, air properties, vapor-air diffusivity, and the "
    "transfer correlation are provisional inputs, not verified wet-bed "
    "measurements. Vapor-air diffusivity is not the learned bed diffusivity.",
    "The 70 degC air condition and predictions near the supported saturation "
    "pressure range may limit applicability; predicted temperature fields "
    "have not been independently validated.",
    "The configured Adam and bounded L-BFGS budgets are optimization limits, "
    "not evidence of convergence.",
    "An Arrhenius temperature dependence is imposed for bed diffusivity with "
    "a fixed activation energy of 2044 K x R (Barnard et al. 2024 Table 1), "
    "but the shared D(T,d,v) constitutive model is unvalidated for this material "
    "and experiment; fitted exponents do not establish causal mechanisms.",
    "Two independent 6-64-32-32-64-1 tanh MLPs predict local log moisture "
    "and temperature fields from time, depth, and four standardized conditions. "
    "Exponential readouts enforce exact initial values and positivity but do "
    "not enforce monotonicity or convexity.",
    "The Incropera temperature coordinate is theta=(T-T_air)/(T_initial-T_air), "
    "with theta(initial)=1 and theta(air)=0; theta is unbounded and is never clipped.",
    "Effective reference diffusivity and the condition exponents may be "
    "non-identifiable from the available bulk drying curves; the activation "
    "energy is fixed rather than learned.",
    "Measured mass increases are retained as observations with their quality "
    "flags. Workbook tx values and observation-derived flux targets are not "
    "used for fitting or evaluation.",
    "The data-only warmup and physics ramp are optimization choices, not "
    "guarantees against data deterioration. Residual RMS scales are frozen "
    "once from training collocation after warmup; validation and test data "
    "do not enter calibration.",
)


def _activation_energy_matches(
    actual_j_mol: float, configured_j_mol: float, dtype: torch.dtype
) -> bool:
    """Compare the model's fixed Ea buffer with config at its storage precision.

    The buffer is cast with the module, so a float32 model holds the configured
    value rounded to float32; one machine epsilon of that dtype is tolerated.
    """
    return math.isfinite(actual_j_mol) and math.isclose(
        actual_j_mol,
        configured_j_mol,
        rel_tol=float(torch.finfo(dtype).eps),
        abs_tol=0.0,
    )


def _json_safe_training_config(config: TrainingConfig) -> dict[str, object]:
    """Serialize torch dtype by name while retaining every optimizer setting."""
    values = asdict(config)
    values["dtype"] = str(config.dtype)
    return values


def _history_records(training_result: TrainingResult) -> list[dict[str, object]]:
    """Flatten the separate optimizer histories for portable CSV/JSON output."""
    records: list[dict[str, object]] = []
    for stage_name, history in training_result.stage_history.items():
        if not math.isfinite(history.elapsed_seconds) or history.elapsed_seconds < 0:
            raise ValueError(
                f"{stage_name} elapsed time must be finite and nonnegative."
            )
        for index, losses in enumerate(history.losses):
            safe_losses: dict[str, float] = {}
            for name, value in losses.items():
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"{stage_name} loss history contains non-finite data."
                    )
                safe_losses[name] = numeric
            records.append(
                {
                    "stage": stage_name,
                    "step": index + 1,
                    "elapsed_seconds_total": float(history.elapsed_seconds),
                    **safe_losses,
                }
            )
    return records


def _validate_conditioned_loss_snapshot(
    snapshot: dict[str, float],
    training_config: TrainingConfig,
    label: str,
) -> None:
    """Verify that reported raw, scaled, and weighted terms reconstruct total."""
    required = {"data", "total", "physics_ramp"}
    for name in _ACTIVE_RESIDUALS:
        required.update(
            (name, f"scale_{name}", f"scaled_{name}", f"contribution_{name}")
        )
    if not required.issubset(snapshot):
        raise ValueError(f"{label} is missing direct-objective contribution fields.")
    ramp = snapshot["physics_ramp"]
    if not 0.0 <= ramp <= 1.0:
        raise ValueError(f"{label} physics ramp must be between zero and one.")
    contribution_sum = snapshot["data"]
    for name in _ACTIVE_RESIDUALS:
        raw = snapshot[name]
        scale = snapshot[f"scale_{name}"]
        scaled = snapshot[f"scaled_{name}"]
        contribution = snapshot[f"contribution_{name}"]
        if scale < 1.0 or scaled < 0.0 or contribution < 0.0:
            raise ValueError(f"{label} has an invalid physics scale or contribution.")
        expected_scaled = raw / (scale * scale)
        group_weight = (
            training_config.physics_weight
            if name in ("physics_mass", "physics_heat")
            else training_config.boundary_weight
        )
        expected_contribution = (
            expected_scaled * group_weight * ramp / len(_ACTIVE_RESIDUALS)
        )
        if not math.isclose(
            scaled, expected_scaled, rel_tol=1.0e-10, abs_tol=1.0e-14
        ) or not math.isclose(
            contribution,
            expected_contribution,
            rel_tol=1.0e-10,
            abs_tol=1.0e-14,
        ):
            raise ValueError(f"{label} conditioned terms do not match the objective.")
        contribution_sum += contribution
    if not math.isclose(
        snapshot["total"], contribution_sum, rel_tol=1.0e-10, abs_tol=1.0e-14
    ):
        raise ValueError(f"{label} contributions do not reconstruct its total.")


def _physics_report(
    data: CurveData,
    config: PhysicsConfig,
    reference_diffusivity_m2_s: float,
    diameter_exponent: float,
    speed_exponent: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Document each experimental condition and its provisional transfer scales."""
    inputs: list[dict[str, object]] = []
    for _, row in data.experiments.iterrows():
        speed = float(row["air_speed_m_s"])
        diameter = float(row["particle_diameter_m"])
        dry_mass = float(row["dry_mass_kg"])
        initial_mass = float(row["initial_sample_mass_kg"])
        initial_moisture = float(row["initial_moisture_db"])
        bulk_density = float(row["dry_bulk_density_kg_m3"])
        physical_inputs = np.asarray(
            [
                speed,
                diameter,
                dry_mass,
                initial_mass,
                initial_moisture,
                bulk_density,
            ],
            dtype=float,
        )
        if not np.isfinite(physical_inputs).all() or (physical_inputs <= 0).any():
            raise ValueError("Experimental input values must be finite and positive.")
        coefficient = flat_plate_transfer(speed, diameter, config)
        # Condition tables carry no local moisture field, so the wet-bed heat
        # capacity rho_d*(cp_s + X*cp_w) is evaluated at the initial moisture
        # content X = X0; the training residual uses the local X = X0*u.
        thermal_diffusivity = config.product_thermal_conductivity_w_m_k / (
            bulk_density
            * (
                config.product_specific_heat_j_kg_k
                + initial_moisture * config.water_specific_heat_j_kg_k
            )
        )
        thermal_fourier = (
            thermal_diffusivity * config.time_scale_s / config.bed_depth_m**2
        )
        thermal_biot = (
            coefficient.heat_transfer_coefficient_w_m2_k
            * config.bed_depth_m
            / config.product_thermal_conductivity_w_m_k
        )
        condition_diffusivity = (
            reference_diffusivity_m2_s
            * (diameter / config.reference_particle_diameter_m) ** diameter_exponent
            * (speed / config.reference_air_speed_m_s) ** speed_exponent
        )
        moisture_fourier = (
            condition_diffusivity * config.time_scale_s / config.bed_depth_m**2
        )
        moisture_biot = (
            coefficient.mass_transfer_coefficient_m_s
            * config.bed_depth_m
            / condition_diffusivity
        )
        inputs.append(
            {
                "experiment_id": str(row["experiment_id"]),
                "condition_id": str(row["condition_id"]),
                "air_speed_m_s": speed,
                "particle_diameter_m": diameter,
                "dry_mass_kg": dry_mass,
                "initial_sample_mass_kg": initial_mass,
                "initial_moisture_db_kg_water_kg_dry": float(initial_moisture),
                "dry_bulk_density_kg_m3": bulk_density,
                "effective_diffusivity_at_reference_temperature_m2_s": (
                    condition_diffusivity
                ),
                "bed_depth_m": config.bed_depth_m,
                "dish_diameter_m": config.dish_diameter_m,
                "transfer_correlation": (
                    "laminar flat-plate average Nu/Sh = 0.664 Re^0.5 Pr^1/3 "
                    "and 0.664 Re^0.5 Sc^1/3; characteristic length is dish diameter"
                ),
                "reynolds": coefficient.reynolds,
                "prandtl": coefficient.prandtl,
                "nusselt": coefficient.nusselt,
                "schmidt": coefficient.schmidt,
                "sherwood": coefficient.sherwood,
                "heat_transfer_coefficient_w_m2_k": (
                    coefficient.heat_transfer_coefficient_w_m2_k
                ),
                "mass_transfer_coefficient_m_s": (
                    coefficient.mass_transfer_coefficient_m_s
                ),
                "thermal_diffusivity_m2_s_provisional": thermal_diffusivity,
                "thermal_fourier_number": thermal_fourier,
                "thermal_biot_number": thermal_biot,
                "moisture_fourier_number_at_fitted_condition_diffusivity": (
                    moisture_fourier
                ),
                "moisture_biot_number_at_fitted_condition_diffusivity": (moisture_biot),
                "applicability_warnings": list(coefficient.applicability_warnings),
            }
        )
    config_values = asdict(config)
    if set(config_values) != set(_CONFIG_UNITS):
        raise RuntimeError("Physics configuration unit documentation is incomplete.")
    property_values = {
        name: {"value": value, "unit": _CONFIG_UNITS[name]}
        for name, value in config_values.items()
    }
    scales = {
        "thermal_fourier_number": (
            "Fo_T = alpha_T * t_scale / bed_depth^2 with "
            "alpha_T = k_product / (rho_d * (cp_s + X0 * cp_w)), evaluated at "
            "the initial moisture content X0"
        ),
        "thermal_biot_number": "Bi_T = h_T * bed_depth / k_product",
        "moisture_fourier_number": "Fo_X = D(T_ref,d,v) * t_scale / bed_depth^2",
        "moisture_biot_number": "Bi_X = h_m * bed_depth / D(T_ref,d,v)",
    }
    return inputs, {
        "properties_and_scales": property_values,
        "dimensionless_scale_definitions": scales,
        "condition_coefficients": inputs,
    }


def _build_comparison_table(
    metrics: pd.DataFrame,
    transfer_inputs: list[dict[str, object]],
    config: PhysicsConfig,
) -> pd.DataFrame:
    """Join final-test scores to existing physics metadata without pooling.

    Keep the metric order and experiment key (not condition key): replicates
    can share all conditions while representing different final-test curves.
    Missing R2 remains nullable in memory and becomes a blank CSV cell.
    """
    metric_columns = (
        "experiment_id",
        "partition",
        "observation_count",
        "r2",
        "r2_status",
    )
    _require_columns(metrics, metric_columns, "per_curve_metrics")
    if metrics.empty or not metrics["partition"].eq("test").all():
        raise ValueError("Comparison rows must be nonempty final-test metrics.")
    if (
        metrics["experiment_id"].isna().any()
        or metrics["experiment_id"].duplicated().any()
    ):
        raise ValueError(
            "Comparison experiment identifiers must be unique and present."
        )
    available = metrics["r2_status"].eq("ok")
    undefined = metrics["r2_status"].isin(
        ("insufficient_observations", "constant_targets", "non_finite")
    )
    if (
        not (available | undefined).all()
        or metrics.loc[~available, "r2"].notna().any()
        or metrics.loc[available, "r2"].isna().any()
    ):
        raise ValueError(
            "R2 values must agree with their explicit availability status."
        )
    _finite_array(
        metrics.loc[available, "r2"].to_numpy(dtype=float), "Available per-curve R2"
    )
    condition_columns = [
        "experiment_id",
        "condition_id",
        "air_speed_m_s",
        "particle_diameter_m",
        "thermal_biot_number",
        "effective_diffusivity_at_reference_temperature_m2_s",
    ]
    conditions = pd.DataFrame(transfer_inputs).loc[:, condition_columns]
    comparison = metrics.loc[:, metric_columns].merge(
        conditions, on="experiment_id", how="left", sort=False, validate="one_to_one"
    )
    if comparison[condition_columns].isna().any(axis=None):
        raise ValueError("Every comparison curve requires matching physics metadata.")
    comparison["particle_diameter_um"] = comparison.pop("particle_diameter_m") * 1e6
    comparison["reference_temperature_K"] = config.reference_temperature_K
    numeric_columns = [
        "air_speed_m_s",
        "particle_diameter_um",
        "thermal_biot_number",
        "effective_diffusivity_at_reference_temperature_m2_s",
        "reference_temperature_K",
    ]
    values = comparison[numeric_columns].to_numpy(dtype=float)
    _finite_array(values, "Comparison physics values")
    if (values <= 0).any():
        raise ValueError("Comparison physics values must be positive.")
    return comparison.loc[
        :,
        [
            "experiment_id",
            "condition_id",
            "partition",
            "air_speed_m_s",
            "particle_diameter_um",
            "thermal_biot_number",
            "effective_diffusivity_at_reference_temperature_m2_s",
            "reference_temperature_K",
            "observation_count",
            "r2",
            "r2_status",
        ],
    ]


def _write_curve_plot(predictions: pd.DataFrame, destination: Path) -> None:
    """Render only actual final-test measurements and model predictions."""
    figure = Figure(figsize=(8.5, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    for identifier, group in predictions.groupby("experiment_id", sort=False):
        # Audit rows outside the modeled interval remain in the CSV, but must
        # not be presented as model output or measurements on the drying plot.
        normalized_time = pd.to_numeric(group["time_normalized"], errors="coerce")
        normalized_values = normalized_time.to_numpy(dtype=float)
        inside_domain = (
            np.isfinite(normalized_values)
            & (normalized_values >= 0.0)
            & (normalized_values <= 1.0)
        )
        ordered = group.loc[inside_domain].copy()
        observed = ordered.loc[
            ordered["target_available"]
            & np.isfinite(
                pd.to_numeric(ordered["moisture_observed_normalized"], errors="coerce")
            )
        ]
        predicted = ordered.loc[
            np.isfinite(
                pd.to_numeric(ordered["moisture_predicted_normalized"], errors="coerce")
            )
        ]
        if not predicted.empty:
            axes.plot(
                predicted["time_normalized"],
                predicted["moisture_predicted_normalized"],
                label=f"{identifier}: PINN prediction",
            )
        if not observed.empty:
            axes.scatter(
                observed["time_normalized"],
                observed["moisture_observed_normalized"],
                marker="o",
                s=30,
                label=f"{identifier}: measured",
            )
    axes.set_xlabel("Normalized time, t / 14400 s")
    axes.set_ylabel("Normalized bulk moisture, Xbulk / X0")
    axes.set_title("Locked final-test drying curves (no empirical comparator supplied)")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.savefig(destination, dpi=160, format="png")


def _quality_flag_report(data: CurveData) -> tuple[pd.DataFrame, dict[str, int]]:
    """Preserve parser quality strings for every experiment and count flags."""
    _require_columns(data.observations, _REQUIRED_OBSERVATION_COLUMNS, "observations")
    selected_columns = [
        column
        for column in ("experiment_id", "time_s", "target_available", "quality_flags")
        if column in data.observations.columns
    ]
    quality_table = data.observations.loc[:, selected_columns].copy()
    if "quality_flags" not in quality_table:
        quality_table["quality_flags"] = ""
    quality_table["quality_flags"] = quality_table["quality_flags"].fillna("").map(str)
    flag_counts: dict[str, int] = {}
    for flags in quality_table["quality_flags"]:
        for flag in flags.split(";"):
            normalized_flag = flag.strip()
            if normalized_flag:
                flag_counts[normalized_flag] = flag_counts.get(normalized_flag, 0) + 1
    return quality_table, flag_counts


def load_model_artifacts(
    run_directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CoupledInversePINN:
    """Reconstruct only the current trusted model schema from an artifact run.

    The manifest is checked before tensor deserialization. Older or unversioned
    state dictionaries are rejected explicitly; strict tensor loading is not
    used as an implicit migration mechanism.
    """
    directory = Path(run_directory)
    if not directory.is_dir():
        raise ValueError(f"Artifact run directory does not exist: {directory}.")
    manifest_path = directory / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read artifact manifest at {manifest_path}."
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Artifact manifest must be a JSON object.")
    model_metadata = manifest.get("model")
    if not isinstance(model_metadata, dict):
        raise ValueError(
            "Incompatible or missing model schema metadata; state cannot be loaded."
        )
    if model_metadata.get("schema") != MODEL_SCHEMA:
        raise ValueError(
            "Incompatible or missing model schema; direct-field Strategy C "
            f"artifacts require {MODEL_SCHEMA!r}. Older state dictionaries "
            "cannot be loaded or migrated."
        )
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            "Incompatible artifact format version; this loader requires "
            f"version {ARTIFACT_FORMAT_VERSION}."
        )
    if model_metadata.get("class") != "strategy_c.physics.CoupledInversePINN":
        raise ValueError("Artifact model class is not the trusted CoupledInversePINN.")
    architecture = model_metadata.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Artifact architecture metadata is missing.")
    config_values = model_metadata.get("physics_config")
    if not isinstance(config_values, dict):
        raise ValueError("Artifact PhysicsConfig is missing.")
    if (
        architecture.get("independent_networks")
        != ["moisture_network", "temperature_network"]
        or architecture.get("input_width") != 6
        or architecture.get("hidden_widths") != [64, 32, 32, 64]
        or architecture.get("output_width") != 1
        or architecture.get("activation") != "tanh"
        or architecture.get("output_head_initialization")
        != "small_nonzero_weights_zero_bias"
    ):
        raise ValueError(
            "Artifact architecture does not match the current model schema."
        )
    transform = manifest.get("feature_transform")
    if not isinstance(transform, dict):
        raise ValueError("Artifact train-only feature transform is missing.")
    feature_names = transform.get("feature_names")
    expected_feature_names = [
        "air_speed_m_s",
        "particle_diameter_m",
        "initial_moisture_db",
        "dry_bulk_density_kg_m3",
    ]
    if feature_names != expected_feature_names:
        raise ValueError("Artifact feature order must contain four neural features.")
    moisture_readout = architecture.get("moisture_readout")
    temperature_readout = architecture.get("temperature_readout")
    expected_input_order = [
        "time_normalized",
        "depth_normalized",
        *expected_feature_names,
    ]
    if (
        not isinstance(moisture_readout, dict)
        or moisture_readout.get("input_order") != expected_input_order
        or moisture_readout.get("formula") != "u = exp(tau*f_X)"
        or moisture_readout.get("pointwise_initial_moisture_ratio") != 1.0
        or moisture_readout.get("strictly_positive") is not True
        or moisture_readout.get("monotonicity_or_convexity_enforced") is not False
        or not isinstance(temperature_readout, dict)
        or temperature_readout.get("input_order") != expected_input_order
        or temperature_readout.get("formula") != "T_K = T_initial_K * exp(tau*f_T)"
        or temperature_readout.get("pointwise_initial_temperature_K")
        != config_values.get("initial_temperature_K")
        or temperature_readout.get("strictly_positive") is not True
    ):
        raise ValueError("Artifact readout metadata does not match the current schema.")
    try:
        config = PhysicsConfig(**config_values)
    except (TypeError, ValueError) as error:
        raise ValueError("Artifact PhysicsConfig is invalid.") from error
    normalization = model_metadata.get("temperature_normalization")
    expected_scale = config.signed_temperature_scale_K
    if (
        not isinstance(normalization, dict)
        or normalization.get("name") != "Incropera theta"
        or normalization.get("definition")
        != "(T_K - T_air_K) / (T_initial_K - T_air_K)"
        or normalization.get("initial_theta") != 1.0
        or normalization.get("air_theta") != 0.0
        or normalization.get("signed_scale_K") != expected_scale
        or normalization.get("unbounded_and_unclipped") is not True
    ):
        raise ValueError(
            "Artifact temperature normalization does not match PhysicsConfig."
        )
    objective = manifest.get("objective_conditioning")
    if not isinstance(objective, dict):
        raise ValueError("Artifact direct-objective conditioning metadata is missing.")
    calibration = objective.get("calibration_mse")
    scales = objective.get("frozen_scales")
    schedule = objective.get("schedule")
    expected_schedule_keys = {
        "adam_epochs",
        "warmup_epochs",
        "ramp_epochs",
        "full_objective_epochs",
    }
    if (
        objective.get("physics_residual_order") != list(_ACTIVE_RESIDUALS)
        or objective.get("scale_formula") != "max(1, sqrt(calibration_mse))"
        or objective.get("data_relative_physics_discount") is not False
        or objective.get("validation_or_test_used_for_calibration") is not False
        or objective.get("alternating_training") is not False
        or not isinstance(calibration, dict)
        or not isinstance(scales, dict)
        or set(calibration) != set(_ACTIVE_RESIDUALS)
        or set(scales) != set(_ACTIVE_RESIDUALS)
        or not isinstance(schedule, dict)
        or set(schedule) != expected_schedule_keys
    ):
        raise ValueError("Artifact direct-objective conditioning metadata is invalid.")
    for name in _ACTIVE_RESIDUALS:
        raw_calibration = calibration[name]
        frozen_scale = scales[name]
        if (
            isinstance(raw_calibration, bool)
            or not isinstance(raw_calibration, Real)
            or not math.isfinite(raw_calibration)
            or raw_calibration < 0.0
            or isinstance(frozen_scale, bool)
            or not isinstance(frozen_scale, Real)
            or not math.isfinite(frozen_scale)
            or frozen_scale < 1.0
            or not math.isclose(
                float(frozen_scale),
                max(1.0, math.sqrt(float(raw_calibration))),
                rel_tol=1.0e-12,
            )
        ):
            raise ValueError(f"Artifact frozen scale for {name} is invalid.")
    if (
        any(
            isinstance(schedule[name], bool)
            or not isinstance(schedule[name], int)
            or schedule[name] < 0
            for name in expected_schedule_keys
        )
        or schedule["adam_epochs"]
        != schedule["warmup_epochs"]
        + schedule["ramp_epochs"]
        + schedule["full_objective_epochs"]
        or schedule["full_objective_epochs"] < 1
    ):
        raise ValueError("Artifact objective schedule is invalid.")
    quadrature = manifest.get("quadrature")
    optimizer_metadata = manifest.get("optimizer")
    conservation_metadata = manifest.get("conservation_diagnostics")
    if not isinstance(optimizer_metadata, dict) or not isinstance(
        conservation_metadata, dict
    ):
        raise ValueError("Artifact optimizer or conservation metadata is missing.")
    training_configuration = optimizer_metadata.get("training_configuration")
    if (
        not isinstance(quadrature, dict)
        or quadrature.get("rule") != "Gauss-Legendre on normalized depth [0, 1]"
        or not isinstance(training_configuration, dict)
        or quadrature.get("training_points")
        != training_configuration.get("quadrature_points")
        or quadrature.get("validation_points")
        != training_configuration.get("quadrature_points")
        or quadrature.get("evaluation_points")
        != manifest.get("evaluation_quadrature_points")
        or quadrature.get("conservation_diagnostic_points")
        != manifest.get("conservation_quadrature_points")
        or conservation_metadata.get("quadrature_points_depth")
        != quadrature.get("conservation_diagnostic_points")
    ):
        raise ValueError("Artifact quadrature metadata is missing or inconsistent.")
    training_dtype = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }.get(training_configuration.get("dtype"))
    if training_dtype is None:
        raise ValueError("Artifact training dtype is invalid.")
    training_config_values = dict(training_configuration)
    training_config_values["dtype"] = training_dtype
    try:
        training_config = TrainingConfig(**training_config_values)
    except (TypeError, ValueError) as error:
        raise ValueError("Artifact training configuration is invalid.") from error
    if (
        schedule != _schedule_boundaries(training_config)
        or objective.get("warmup_fraction") != training_config.warmup_fraction
        or objective.get("ramp_fraction") != training_config.ramp_fraction
        or objective.get("physics_weight") != training_config.physics_weight
        or objective.get("boundary_weight") != training_config.boundary_weight
        or objective.get("initial_weight") != training_config.initial_weight
    ):
        raise ValueError(
            "Artifact objective schedule/weights differ from training configuration."
        )
    lbfgs_metadata = optimizer_metadata.get("lbfgs")
    if (
        not isinstance(lbfgs_metadata, dict)
        or not isinstance(lbfgs_metadata.get("accepted_state"), dict)
        or not isinstance(lbfgs_metadata.get("observed_termination_flags"), dict)
        or lbfgs_metadata.get("termination_reason_inferred") is not False
    ):
        raise ValueError("Artifact accepted-state L-BFGS diagnostics are missing.")
    accepted_state = lbfgs_metadata["accepted_state"]
    accepted_losses = accepted_state.get("losses")
    accepted_gradient_norm = accepted_state.get("gradient_l2_norm")
    termination_flags = lbfgs_metadata["observed_termination_flags"]
    if (
        not isinstance(accepted_losses, dict)
        or isinstance(accepted_gradient_norm, bool)
        or not isinstance(accepted_gradient_norm, Real)
        or not math.isfinite(accepted_gradient_norm)
        or accepted_gradient_norm < 0.0
        or set(termination_flags)
        != {"iteration_limit_reached", "evaluation_limit_reached"}
        or any(not isinstance(value, bool) for value in termination_flags.values())
        or not all(
            isinstance(name, str)
            and not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
            for name, value in accepted_losses.items()
        )
    ):
        raise ValueError("Artifact accepted-state L-BFGS values are invalid.")
    _validate_conditioned_loss_snapshot(
        {name: float(value) for name, value in accepted_losses.items()},
        training_config,
        "accepted L-BFGS state",
    )
    if accepted_losses.get("physics_ramp") != 1.0 or any(
        accepted_losses.get(f"scale_{name}") != scales[name]
        for name in _ACTIVE_RESIDUALS
    ):
        raise ValueError("Accepted L-BFGS state does not use the frozen objective.")
    learned = manifest.get("learned_parameters")
    if not isinstance(learned, dict):
        raise ValueError("Artifact learned-parameter metadata is missing.")
    for metadata_name, config_name in (
        ("reference_temperature_K", "reference_temperature_K"),
        ("reference_particle_diameter_m", "reference_particle_diameter_m"),
        ("reference_air_speed_m_s", "reference_air_speed_m_s"),
        (
            "reference_diffusivity_initial_m2_s",
            "initial_reference_diffusivity_m2_s",
        ),
    ):
        if learned.get(metadata_name) != getattr(config, config_name):
            raise ValueError(
                f"Artifact {metadata_name} differs from its PhysicsConfig reference."
            )
    if any(name.startswith("activation_energy") for name in learned):
        raise ValueError(
            "Artifact must not report the fixed activation energy as learned."
        )
    fixed = manifest.get("fixed_parameters")
    if (
        not isinstance(fixed, dict)
        or fixed.get("activation_energy_j_mol") != config.activation_energy_j_mol
        or fixed.get("activation_energy_provenance") != _ACTIVATION_ENERGY_PROVENANCE
        or fixed.get("activation_energy_trainable") is not False
    ):
        raise ValueError(
            "Artifact fixed activation energy does not match PhysicsConfig."
        )
    if learned.get("constitutive_law") != (
        "D=D_ref*(d/d_ref)^alpha*(v/v_ref)^beta*exp[(Ea/R)*(1/T_ref-1/T)]"
    ):
        raise ValueError("Artifact diffusivity law metadata does not match the schema.")
    dtype_name = model_metadata.get("dtype")
    dtype_by_name = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError("Artifact model dtype must be torch.float32 or torch.float64.")
    state_filename = model_metadata.get("state_dict_file")
    if state_filename != "model_state.pt":
        raise ValueError(
            "Artifact state_dict_file must be the expected model_state.pt."
        )
    model = CoupledInversePINN(config).to(
        device=map_location, dtype=dtype_by_name[dtype_name]
    )
    try:
        state = torch.load(
            directory / state_filename,
            map_location=map_location,
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError("Could not load the tensor-only model state.") from error
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("Model state must be a tensor-only state dictionary.")
    if any(
        not bool(torch.isfinite(value).all().detach().cpu()) for value in state.values()
    ):
        raise ValueError("Model state contains non-finite tensor values.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Model state is incompatible with the declared current schema."
        ) from error
    recorded_diffusivity = learned.get("reference_diffusivity_m2_s")
    if not _activation_energy_matches(
        float(model.activation_energy_j_mol.detach().cpu()),
        config.activation_energy_j_mol,
        dtype_by_name[dtype_name],
    ):
        raise ValueError(
            "Model state activation energy differs from the fixed PhysicsConfig value."
        )
    for label, recorded, actual in (
        (
            "reference diffusivity",
            recorded_diffusivity,
            float(model.reference_diffusivity_m2_s.detach().cpu()),
        ),
        (
            "diameter exponent",
            learned.get("diameter_exponent_alpha"),
            float(model.diameter_exponent.detach().cpu()),
        ),
        (
            "speed exponent",
            learned.get("speed_exponent_beta"),
            float(model.speed_exponent.detach().cpu()),
        ),
    ):
        if (
            isinstance(recorded, bool)
            or not isinstance(recorded, (int, float))
            or not math.isfinite(float(recorded))
            or not math.isclose(float(recorded), actual, rel_tol=1.0e-12, abs_tol=0.0)
        ):
            raise ValueError(f"Manifest {label} does not match the strict model state.")
    model.eval()
    return model


def export_results(
    evaluation: EvaluationResult,
    training_result: TrainingResult,
    data: CurveData,
    output_root: str | Path,
    physics_config: PhysicsConfig | None = None,
    training_config: TrainingConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    *,
    feature_transform: FeatureTransform,
    run_timestamp: datetime | None = None,
    internal_fields: InternalFieldResult | None = None,
) -> Path:
    """Write reports and model reconstruction inputs into a fresh run folder.

    The caller supplies the fitted transform because the neural weights alone
    do not encode how raw experimental features were standardized.
    """
    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult.")
    if not isinstance(training_result, TrainingResult):
        raise TypeError("training_result must be a TrainingResult.")
    if not isinstance(data, CurveData):
        raise TypeError("data must be CurveData.")
    if not isinstance(feature_transform, FeatureTransform):
        raise TypeError("feature_transform must be a FeatureTransform.")
    if internal_fields is not None and not isinstance(
        internal_fields, InternalFieldResult
    ):
        raise TypeError("internal_fields must be InternalFieldResult or None.")
    _require_columns(data.experiments, _REQUIRED_EXPERIMENT_COLUMNS, "experiments")
    quality_flags, quality_flag_counts = _quality_flag_report(data)
    config = PhysicsConfig() if physics_config is None else physics_config
    optimizer_config = TrainingConfig() if training_config is None else training_config
    sampler_config = SamplingConfig() if sampling_config is None else sampling_config
    if not isinstance(config, PhysicsConfig):
        raise TypeError("physics_config must be PhysicsConfig.")
    if not isinstance(optimizer_config, TrainingConfig):
        raise TypeError("training_config must be TrainingConfig.")
    if not isinstance(sampler_config, SamplingConfig):
        raise TypeError("sampling_config must be SamplingConfig.")
    if (
        evaluation.train_ids != training_result.provenance.train_ids
        or evaluation.validation_ids != training_result.provenance.validation_ids
        or evaluation.final_test_ids != training_result.provenance.final_test_ids
    ):
        raise ValueError("Evaluation IDs do not match training provenance.")
    expected_transform = fit_feature_transform(
        data,
        ExperimentSplit(
            train_ids=evaluation.train_ids,
            validation_ids=evaluation.validation_ids,
            test_ids=evaluation.final_test_ids,
        ),
    )
    if feature_transform != expected_transform:
        raise ValueError(
            "feature_transform must match train-only statistics for the "
            "evaluation's training IDs."
        )
    required_feature_order = (
        "air_speed_m_s",
        "particle_diameter_m",
        "initial_moisture_db",
        "dry_bulk_density_kg_m3",
    )
    if feature_transform.feature_names != required_feature_order:
        raise ValueError(
            "feature_transform feature order must match the model's four inputs."
        )
    if internal_fields is not None:
        if internal_fields.selected_experiment_id not in evaluation.train_ids:
            raise ValueError(
                "The internal-field representative must be one of the training IDs."
            )
        if internal_fields.physics_config != (
            PhysicsConfig() if physics_config is None else physics_config
        ):
            raise ValueError(
                "Internal-field scales must match the report PhysicsConfig."
            )
    required_stages = {"adam", "lbfgs"}
    if not required_stages.issubset(training_result.stage_history):
        raise ValueError("Training history must include Adam and L-BFGS stages.")
    adam_history = training_result.stage_history["adam"]
    lbfgs_history = training_result.stage_history["lbfgs"]
    if optimizer_config.adam_epochs != adam_history.epochs:
        raise ValueError(
            "Configured Adam epochs do not match the recorded training run."
        )
    if lbfgs_history.iterations > optimizer_config.lbfgs_max_iter:
        raise ValueError("Recorded L-BFGS iterations exceed the configured bound.")
    if set(evaluation.final_test_ids).difference(
        evaluation.predictions["experiment_id"].astype(str)
    ):
        raise ValueError(
            "The prediction table does not include every final-test curve."
        )
    if set(evaluation.final_test_ids).difference(
        evaluation.per_curve_metrics["experiment_id"].astype(str)
    ):
        raise ValueError("The metric table does not include every final-test curve.")
    if not evaluation.predictions["partition"].eq("test").all():
        raise ValueError("Prediction rows must be labeled as final-test records.")
    if not evaluation.per_curve_metrics["partition"].eq("test").all():
        raise ValueError("Per-curve metrics must be labeled as final-test records.")
    _require_columns(
        evaluation.predictions,
        _PREDICTION_COLUMNS,
        "evaluation.predictions",
    )
    prediction_values = evaluation.predictions.loc[
        :,
        [
            "time_normalized",
            "moisture_observed_normalized",
            "moisture_predicted_normalized",
            "temperature_predicted_K",
            "temperature_predicted_theta",
        ],
    ].to_numpy(dtype=float)
    if np.isinf(prediction_values).any():
        raise ValueError("Prediction records must not contain infinite values.")
    prediction_times = pd.to_numeric(
        evaluation.predictions["time_normalized"], errors="coerce"
    ).to_numpy(dtype=float)
    in_modeled_domain = (
        np.isfinite(prediction_times)
        & (prediction_times >= 0.0)
        & (prediction_times <= 1.0)
    )
    availability_values = evaluation.predictions["target_available"]
    if (
        not availability_values.dropna()
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .all()
    ):
        raise ValueError(
            "Prediction target_available values must be boolean or missing."
        )
    availability = availability_values.fillna(False).to_numpy(dtype=bool)
    available_without_modeled_time = availability & ~in_modeled_domain
    if available_without_modeled_time.any():
        if (availability & np.isfinite(prediction_times) & ~in_modeled_domain).any():
            raise ValueError(
                "target_available is true outside modeled domain [0, 1] in "
                "evaluation.predictions."
            )
        raise ValueError(
            "target_available is true without a finite time in the modeled domain."
        )
    valid_time_predictions = evaluation.predictions.loc[
        in_modeled_domain,
        [
            "moisture_predicted_normalized",
            "temperature_predicted_K",
            "temperature_predicted_theta",
        ],
    ].to_numpy(dtype=float)
    _finite_array(valid_time_predictions, "Valid-time prediction records")
    outside_domain_predictions = evaluation.predictions.loc[
        ~in_modeled_domain,
        [
            "moisture_predicted_normalized",
            "temperature_predicted_K",
            "temperature_predicted_theta",
        ],
    ]
    if outside_domain_predictions.notna().any(axis=None):
        raise ValueError("Out-of-domain audit rows must not contain model predictions.")
    metrics = evaluation.per_curve_metrics.loc[:, ["mae", "rmse"]].to_numpy(dtype=float)
    _finite_array(metrics, "Per-curve metrics")
    if metrics.size == 0:
        raise ValueError("Per-curve metrics cannot be empty.")
    aggregate_values: dict[str, float | int] = dict(evaluation.aggregate_metrics)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (Real, np.integer))
        or not math.isfinite(float(value))
        for value in aggregate_values.values()
    ):
        raise ValueError("Aggregate metrics must be finite numeric values.")
    learned_model = training_result.model
    if not isinstance(learned_model, CoupledInversePINN):
        raise ValueError(
            "Only CoupledInversePINN instances can be exported for reconstruction."
        )
    if not hasattr(learned_model, "reference_diffusivity_m2_s") or not hasattr(
        learned_model, "activation_energy_j_mol"
    ):
        raise ValueError("The trained model does not expose inverse parameters.")
    if hasattr(learned_model, "config") and learned_model.config != config:
        raise ValueError("The report PhysicsConfig differs from the trained model.")
    reference_diffusivity = float(
        learned_model.reference_diffusivity_m2_s.detach().cpu()
    )
    activation_energy = float(learned_model.activation_energy_j_mol.detach().cpu())
    diameter_exponent = float(learned_model.diameter_exponent.detach().cpu())
    speed_exponent = float(learned_model.speed_exponent.detach().cpu())
    if (
        not math.isfinite(reference_diffusivity)
        or reference_diffusivity <= 0
        or not math.isfinite(diameter_exponent)
        or not math.isfinite(speed_exponent)
    ):
        raise ValueError("Learned positive scales and signed exponents must be finite.")
    model_parameters = tuple(learned_model.parameters())
    if not model_parameters:
        raise ValueError("The trained model must contain parameters.")
    model_dtype = model_parameters[0].dtype
    if model_dtype not in (torch.float32, torch.float64) or any(
        parameter.dtype != model_dtype or parameter.device != model_parameters[0].device
        for parameter in model_parameters
    ):
        raise ValueError(
            "Model parameters must share float32/float64 dtype and device."
        )
    if not _activation_energy_matches(
        activation_energy, config.activation_energy_j_mol, model_dtype
    ):
        raise ValueError(
            "The model's fixed activation energy differs from PhysicsConfig."
        )
    model_state: dict[str, torch.Tensor] = {}
    for name, value in learned_model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Model state must contain tensors only.")
        if not bool(torch.isfinite(value).all().detach().cpu()):
            raise ValueError(f"Model state tensor {name!r} contains non-finite values.")
        model_state[name] = value.detach().cpu().clone()
    history_records = _history_records(training_result)
    if not history_records:
        raise ValueError("Training histories must contain at least one loss record.")
    for record in history_records:
        loss_snapshot = {
            name: float(value)
            for name, value in record.items()
            if name not in ("stage", "step", "elapsed_seconds_total")
        }
        _validate_conditioned_loss_snapshot(
            loss_snapshot,
            optimizer_config,
            f"{record['stage']} snapshot {record['step']}",
        )
    conditioning_keys = set(_ACTIVE_RESIDUALS)
    if (
        set(training_result.physics_scales) != conditioning_keys
        or set(training_result.physics_calibration_mse) != conditioning_keys
    ):
        raise ValueError("Training result must include six frozen residual scales.")
    for name in _ACTIVE_RESIDUALS:
        calibration = float(training_result.physics_calibration_mse[name])
        scale = float(training_result.physics_scales[name])
        if (
            not math.isfinite(calibration)
            or calibration < 0.0
            or not math.isfinite(scale)
            or scale < 1.0
            or not math.isclose(
                scale, max(1.0, math.sqrt(calibration)), rel_tol=1.0e-12
            )
        ):
            raise ValueError(f"Training result has an invalid frozen scale for {name}.")
    schedule = dict(training_result.schedule)
    if (
        set(schedule)
        != {
            "adam_epochs",
            "warmup_epochs",
            "ramp_epochs",
            "full_objective_epochs",
        }
        or schedule["adam_epochs"] != optimizer_config.adam_epochs
        or schedule["warmup_epochs"] < 0
        or schedule["ramp_epochs"] < 0
        or schedule["full_objective_epochs"] < 1
        or schedule["warmup_epochs"]
        + schedule["ramp_epochs"]
        + schedule["full_objective_epochs"]
        != optimizer_config.adam_epochs
    ):
        raise ValueError("Training result schedule is inconsistent with its config.")
    if (
        not training_result.stage_history["lbfgs"].accepted_losses
        or training_result.stage_history["lbfgs"].accepted_gradient_norm is None
        or not math.isfinite(
            training_result.stage_history["lbfgs"].accepted_gradient_norm
        )
        or training_result.stage_history["lbfgs"].accepted_gradient_norm < 0.0
        or set(training_result.stage_history["lbfgs"].termination_flags)
        != {"iteration_limit_reached", "evaluation_limit_reached"}
        or any(
            not isinstance(value, bool)
            for value in training_result.stage_history[
                "lbfgs"
            ].termination_flags.values()
        )
    ):
        raise ValueError("Training result lacks accepted-state L-BFGS diagnostics.")
    _validate_conditioned_loss_snapshot(
        training_result.stage_history["lbfgs"].accepted_losses,
        optimizer_config,
        "accepted L-BFGS state",
    )
    transfer_inputs, physics_report = _physics_report(
        data,
        config,
        reference_diffusivity,
        diameter_exponent,
        speed_exponent,
    )
    comparison = _build_comparison_table(
        evaluation.per_curve_metrics, transfer_inputs, config
    )
    if tuple(comparison["experiment_id"].astype(str)) != evaluation.final_test_ids:
        raise ValueError(
            "Comparison rows must match the locked final-test IDs in order."
        )
    # Export the same table exposed to notebook consumers, but detect stale or
    # edited summaries before creating a timestamp directory.
    try:
        pd.testing.assert_frame_equal(
            evaluation.comparison_table, comparison, check_dtype=False
        )
    except AssertionError as error:
        raise ValueError(
            "Comparison table differs from evaluation and physics metadata."
        ) from error
    required_diagnostic_columns = {
        "experiment_id",
        "tau",
        "water_inventory_kg_m2",
        "water_storage_rate_kg_m2_s",
        "surface_evaporation_flux_kg_m2_s",
        "water_balance_defect_kg_m2_s",
        "base_mass_outward_flux_kg_m2_s",
        "sensible_storage_rate_w_m2",
        "convective_heat_flux_w_m2",
        "latent_heat_flux_w_m2",
        "surface_vaporization_enthalpy_j_kg",
        "net_heat_flux_w_m2",
        "top_conductive_heat_flux_w_m2",
        "base_conductive_heat_flux_w_m2",
        "energy_storage_minus_net_w_m2",
        "top_conduction_minus_net_w_m2",
    }
    if not required_diagnostic_columns.issubset(
        evaluation.conservation_diagnostics.columns
    ):
        missing_diagnostics = required_diagnostic_columns.difference(
            evaluation.conservation_diagnostics.columns
        )
        raise ValueError(
            f"Conservation diagnostics are missing columns: "
            f"{sorted(missing_diagnostics)}."
        )
    if evaluation.conservation_diagnostics.empty:
        raise ValueError("Conservation diagnostics cannot be empty.")
    numeric_diagnostics = evaluation.conservation_diagnostics.drop(
        columns=["experiment_id"]
    ).to_numpy(dtype=float)
    _finite_array(numeric_diagnostics, "SI conservation diagnostics")
    diagnostic_summary = {
        column: {
            "minimum": float(evaluation.conservation_diagnostics[column].min()),
            "maximum": float(evaluation.conservation_diagnostics[column].max()),
            "mean": float(evaluation.conservation_diagnostics[column].mean()),
        }
        for column in evaluation.conservation_diagnostics.columns
        if column != "experiment_id"
    }
    timestamp = datetime.now(timezone.utc) if run_timestamp is None else run_timestamp
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("run_timestamp must be a timezone-aware datetime.")
    timestamp_label = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = Path(output_root).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError("output_root must be a directory path.")
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / timestamp_label
    run_directory.mkdir(exist_ok=False)

    # Persist tensors only. Consumers reconstruct the trusted model class
    # explicitly from the manifest and load this state with weights_only=True.
    torch.save(model_state, run_directory / "model_state.pt")
    evaluation.predictions.to_csv(run_directory / "predictions.csv", index=False)
    evaluation.per_curve_metrics.to_csv(
        run_directory / "per_curve_metrics.csv", index=False
    )
    evaluation.comparison_table.to_csv(
        run_directory / "comparison_table.csv", index=False
    )
    evaluation.conservation_diagnostics.to_csv(
        run_directory / "conservation_diagnostics.csv", index=False
    )
    pd.DataFrame.from_records(history_records).to_csv(
        run_directory / "loss_history.csv", index=False
    )
    quality_flags.to_csv(run_directory / "quality_flags.csv", index=False)
    _write_curve_plot(
        evaluation.predictions, run_directory / "normalized_drying_curves.png"
    )
    loss_figure = plot_loss_history(
        pd.DataFrame.from_records(history_records),
        physics_weight=optimizer_config.physics_weight,
        initial_weight=optimizer_config.initial_weight,
        boundary_weight=optimizer_config.boundary_weight,
    )
    try:
        FigureCanvasAgg(loss_figure).print_png(str(run_directory / "loss_history.png"))
    finally:
        # The notebook owns its separate interactive figure; exports must not
        # accumulate pyplot-managed figures across repeated runs.
        plt.close(loss_figure)
    field_paths = None
    if internal_fields is not None:
        field_paths = write_internal_field_artifacts(internal_fields, run_directory)
    artifact_files = [
        "predictions.csv",
        "per_curve_metrics.csv",
        "comparison_table.csv",
        "conservation_diagnostics.csv",
        "loss_history.csv",
        "loss_history.png",
        "quality_flags.csv",
        "normalized_drying_curves.png",
        "model_state.pt",
    ]
    if field_paths is not None:
        artifact_files.extend([field_paths.figure.name, field_paths.grid.name])
    manifest: dict[str, object] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "run_directory": timestamp_label,
        "train_ids": list(evaluation.train_ids),
        "validation_ids": list(evaluation.validation_ids),
        "test_ids": list(evaluation.final_test_ids),
        # Retain this legacy key as an explicit alias of the new final-test
        # role. Old manifests with this key are not migrated or reinterpreted.
        "held_out_ids": list(evaluation.final_test_ids),
        "held_out_ids_role": (
            "Legacy compatibility alias for test_ids (final test only); "
            "use validation_ids and test_ids for current partition roles."
        ),
        "loss_reporting": {
            "units": "dimensionless squared residuals",
            "partitions": "train, validation, and final test",
            "validation_data": (
                "Equal-curve mean MSE of depth-averaged X/X0; monitored during "
                "Adam and L-BFGS only, excluded from objective and optimization."
            ),
            "legacy_held_out_data": (
                "Historical CSV column only; it is not mapped to validation and "
                "does not establish the old run's partition sizes."
            ),
            "temperature_observations": False,
            "temperature_loss": (
                "Frozen-scaled heat PDE and boundary residual contributions; "
                "the exact initial-condition residual is diagnostic only."
            ),
            "snapshot_coordinates": {
                "adam": "pre-update epoch",
                "lbfgs": "closure evaluation, including line-search trials; not epoch",
            },
            "validation_quadrature_points": optimizer_config.quadrature_points,
            "objective": (
                "equal-curve data MSE plus the mean of six frozen-scaled "
                "physics/boundary residual MSEs multiplied by the Adam ramp "
                "and their configured group weights"
            ),
            "initial_condition_terms": (
                "exact zero diagnostics; excluded from objective"
            ),
        },
        "objective_conditioning": {
            "physics_residual_order": list(_ACTIVE_RESIDUALS),
            "calibration_partition": "training collocation after data-only warmup",
            "calibration_mse": dict(training_result.physics_calibration_mse),
            "frozen_scales": dict(training_result.physics_scales),
            "scale_formula": "max(1, sqrt(calibration_mse))",
            "schedule": schedule,
            "warmup_fraction": optimizer_config.warmup_fraction,
            "ramp_fraction": optimizer_config.ramp_fraction,
            "fraction_rounding": "nearest integer, half upward; final full step reserved",
            "physics_weight": optimizer_config.physics_weight,
            "boundary_weight": optimizer_config.boundary_weight,
            "initial_weight": optimizer_config.initial_weight,
            "data_relative_physics_discount": False,
            "validation_or_test_used_for_calibration": False,
            "alternating_training": False,
        },
        "quadrature": {
            "rule": "Gauss-Legendre on normalized depth [0, 1]",
            "training_points": optimizer_config.quadrature_points,
            "validation_points": optimizer_config.quadrature_points,
            "evaluation_points": evaluation.quadrature_points,
            "conservation_diagnostic_points": (
                evaluation.conservation_quadrature_points
            ),
        },
        "quality_flag_counts_all_curves": quality_flag_counts,
        "training_provenance": {
            "seed": training_result.provenance.seed,
            "training_observation_count": (
                training_result.provenance.training_observation_count
            ),
            "collocation_point_count": (
                training_result.provenance.collocation_point_count
            ),
            "device": training_result.provenance.device,
            "dtype": training_result.provenance.dtype,
        },
        "model": {
            "class": "strategy_c.physics.CoupledInversePINN",
            "schema": MODEL_SCHEMA,
            "architecture": {
                "independent_networks": ["moisture_network", "temperature_network"],
                "input_width": 6,
                "hidden_widths": [64, 32, 32, 64],
                "output_width": 1,
                "activation": "tanh",
                "output_head_initialization": "small_nonzero_weights_zero_bias",
                "moisture_readout": {
                    "input_order": [
                        "time_normalized",
                        "depth_normalized",
                        *feature_transform.feature_names,
                    ],
                    "formula": "u = exp(tau*f_X)",
                    "pointwise_initial_moisture_ratio": 1.0,
                    "strictly_positive": True,
                    "monotonicity_or_convexity_enforced": False,
                },
                "temperature_readout": {
                    "input_order": [
                        "time_normalized",
                        "depth_normalized",
                        *feature_transform.feature_names,
                    ],
                    "formula": "T_K = T_initial_K * exp(tau*f_T)",
                    "strictly_positive": True,
                    "pointwise_initial_temperature_K": config.initial_temperature_K,
                },
                "output_order": ["moisture_ratio_X_over_X0", "temperature_K"],
            },
            "state_dict_file": "model_state.pt",
            "state_dict_format": (
                "tensor-only PyTorch state_dict; load with "
                "torch.load(weights_only=True)"
            ),
            "dtype": str(model_dtype),
            "physics_config": asdict(config),
            "temperature_normalization": {
                "name": "Incropera theta",
                "definition": "(T_K - T_air_K) / (T_initial_K - T_air_K)",
                "initial_theta": 1.0,
                "air_theta": 0.0,
                "signed_scale_K": config.signed_temperature_scale_K,
                "unbounded_and_unclipped": True,
            },
        },
        "feature_transform": {
            "method": "training-only population mean and standard deviation",
            "fitted_on_ids": list(evaluation.train_ids),
            "feature_names": list(feature_transform.feature_names),
            "mean": list(feature_transform.mean),
            "scale": list(feature_transform.scale),
        },
        "final_test_metrics_normalized_moisture": aggregate_values,
        "held_out_metrics_normalized_moisture": aggregate_values,
        "held_out_metrics_role": (
            "Legacy compatibility alias for final_test_metrics_normalized_moisture."
        ),
        "evaluation_quadrature_points": evaluation.quadrature_points,
        "conservation_quadrature_points": (evaluation.conservation_quadrature_points),
        "temperature_reporting": {
            "temperature_kelvin_column": "temperature_predicted_K",
            "normalized_temperature_column": "temperature_predicted_theta",
            "normalization": ("theta=(T_K-T_air_K)/(T_initial_K-T_air_K)"),
            "initial_theta": 1.0,
            "air_theta": 0.0,
            "signed_scale_K": config.signed_temperature_scale_K,
            "unbounded_and_unclipped": True,
        },
        "per_curve_metric_definitions": {
            "mae": "mean absolute error in Xbulk/X0 units per final-test curve",
            "rmse": "root mean square error in Xbulk/X0 units per final-test curve",
            "r2": (
                "Predictive, unadjusted 1-SSE/SST on exactly the RMSE pairs; "
                "SST centers observed Xbulk/X0 within each final-test curve. "
                "Negative scores are valid; no aggregate R2 is reported."
            ),
            "r2_status": {
                "ok": "finite predictive R2",
                "insufficient_observations": "unavailable: fewer than two scored pairs",
                "constant_targets": "unavailable: constant observed targets (SST=0)",
                "non_finite": "unavailable: numerical overflow or underflow in R2",
                "missing_value_encoding": "nullable in memory; blank CSV cell, not a finite score",
            },
            "equal_curve_mae": "mean of per-curve MAE; each final-test curve has equal weight",
            "equal_curve_rmse": "mean of per-curve RMSE; each final-test curve has equal weight",
            "pooled_mae": "point-weighted MAE across final-test observations",
            "pooled_rmse": "point-weighted RMSE across final-test observations",
        },
        "comparison_table": {
            "artifact": "comparison_table.csv",
            "row_scope": "one row per locked final-test experiment; partition=test",
            "provenance_columns": ["experiment_id", "condition_id", "partition"],
            "units": {
                "air_speed_m_s": "m s-1",
                "particle_diameter_um": "um",
                "thermal_biot_number": "dimensionless",
                "effective_diffusivity_at_reference_temperature_m2_s": "m2 s-1",
                "reference_temperature_K": "K",
                "observation_count": "count",
                "r2": "dimensionless",
            },
            "thermal_biot_number": (
                "Fixed calculated Bi_T = h_T * bed_depth / k_product, reused "
                "from condition transfer coefficients; not a learned Biot number."
            ),
            "effective_diffusivity_at_reference_temperature_m2_s": (
                "Learned D(T_ref,d,v) from the existing physics report, evaluated "
                "at reference_temperature_K, not final or time-averaged temperature."
            ),
            "reference_temperature_K": config.reference_temperature_K,
            "r2_and_status_definitions": "per_curve_metric_definitions",
        },
        "learned_parameters": {
            "reference_diffusivity_m2_s": reference_diffusivity,
            "reference_diffusivity_initial_m2_s": (
                config.initial_reference_diffusivity_m2_s
            ),
            "diameter_exponent_alpha": diameter_exponent,
            "speed_exponent_beta": speed_exponent,
            "reference_temperature_K": config.reference_temperature_K,
            "reference_particle_diameter_m": (config.reference_particle_diameter_m),
            "reference_air_speed_m_s": config.reference_air_speed_m_s,
            "constitutive_law": (
                "D=D_ref*(d/d_ref)^alpha*(v/v_ref)^beta*exp[(Ea/R)*(1/T_ref-1/T)]"
            ),
        },
        "fixed_parameters": {
            "activation_energy_j_mol": config.activation_energy_j_mol,
            "activation_energy_kj_mol": config.activation_energy_j_mol / 1000.0,
            "activation_energy_over_gas_constant_K": (
                config.activation_energy_j_mol / config.gas_constant_j_mol_k
            ),
            "activation_energy_provenance": _ACTIVATION_ENERGY_PROVENANCE,
            "activation_energy_trainable": False,
        },
        "optimizer": {
            "adam": {
                "epochs": adam_history.epochs,
                "configured_epochs": optimizer_config.adam_epochs,
                "completed_epochs": adam_history.epochs,
                "learning_rate": optimizer_config.adam_learning_rate,
                "physics_parameter_learning_rate": (
                    optimizer_config.physics_parameter_learning_rate
                ),
                "elapsed_seconds": adam_history.elapsed_seconds,
                "loss_record_count": len(adam_history.losses),
            },
            "lbfgs": {
                "iterations": lbfgs_history.iterations,
                "configured_max_iterations": optimizer_config.lbfgs_max_iter,
                "configured_max_evaluations": optimizer_config.lbfgs_max_eval,
                "history_size": optimizer_config.lbfgs_history_size,
                "completed_iterations": lbfgs_history.iterations,
                "closure_evaluations": lbfgs_history.closure_calls,
                "elapsed_seconds": lbfgs_history.elapsed_seconds,
                "loss_record_count": len(lbfgs_history.losses),
                "accepted_state": {
                    "losses": lbfgs_history.accepted_losses,
                    "gradient_l2_norm": lbfgs_history.accepted_gradient_norm,
                },
                "observed_termination_flags": lbfgs_history.termination_flags,
                "termination_reason_inferred": False,
            },
            "training_configuration": _json_safe_training_config(optimizer_config),
            "sampling_configuration": asdict(sampler_config),
            "sequential_same_parameter_objects": True,
            "convergence_claimed": False,
        },
        "physics_configuration": physics_report["properties_and_scales"],
        "dimensionless_scale_definitions": physics_report[
            "dimensionless_scale_definitions"
        ],
        "transfer_coefficients": transfer_inputs,
        "conservation_diagnostics": {
            "file": "conservation_diagnostics.csv",
            "times_dimensionless": sorted(
                evaluation.conservation_diagnostics["tau"].unique().tolist()
            ),
            "conditions": "one row group per prepared experiment; SI physical inputs",
            "water_balance_definition": (
                "rho_d*X0*L/time_scale*dB/dtau + J_surface; base outward "
                "mass flux is reported separately because the zero-flux condition is soft"
            ),
            "energy_balance_definition": (
                "integral(rho_d*cp*dT/dt dz) - (q_conv-h_v*J), alongside "
                "q_cond(top)-(q_conv-h_v*J), with h_v(T_s,X) per surface point; "
                "base conductive flux is also reported"
            ),
            "units": {
                "water_inventory_kg_m2": "kg m-2",
                "water_storage_rate_kg_m2_s": "kg m-2 s-1",
                "surface_evaporation_flux_kg_m2_s": "kg m-2 s-1",
                "water_balance_defect_kg_m2_s": "kg m-2 s-1",
                "base_mass_outward_flux_kg_m2_s": "kg m-2 s-1",
                "sensible_storage_rate_w_m2": "W m-2",
                "convective_heat_flux_w_m2": "W m-2",
                "latent_heat_flux_w_m2": "W m-2",
                "surface_vaporization_enthalpy_j_kg": "J kg-1",
                "net_heat_flux_w_m2": "W m-2",
                "top_conductive_heat_flux_w_m2": "W m-2",
                "base_conductive_heat_flux_w_m2": "W m-2",
                "energy_storage_minus_net_w_m2": "W m-2",
                "top_conduction_minus_net_w_m2": "W m-2",
            },
            "quadrature_points_depth": evaluation.conservation_quadrature_points,
            "results_summary": {
                **diagnostic_summary,
            },
            "not_an_exact_conservation_claim": True,
        },
        "assumptions_and_limitations": list(_ASSUMPTIONS),
        "empirical_comparison": (
            "No compatible empirical reference curve or equation was supplied; "
            "no empirical comparator is fabricated."
        ),
        "artifacts": artifact_files,
    }
    if internal_fields is not None and field_paths is not None:
        manifest["internal_field_visualization"] = {
            "selected_experiment_id": internal_fields.selected_experiment_id,
            "selected_training_features": internal_fields.selected_features,
            "scales": {
                "time_scale_s": internal_fields.physics_config.time_scale_s,
                "bed_depth_m": internal_fields.physics_config.bed_depth_m,
                "initial_temperature_K": (
                    internal_fields.physics_config.initial_temperature_K
                ),
                "air_temperature_K": (internal_fields.physics_config.air_temperature_K),
                "temperature_difference_K": (
                    internal_fields.physics_config.temperature_difference_K
                ),
                "signed_temperature_scale_K": (
                    internal_fields.physics_config.signed_temperature_scale_K
                ),
            },
            "temperature_normalization": ("(T_K - T_air_K) / (T_initial_K - T_air_K)"),
            "profile_times_normalized": (
                internal_fields.profile_times_normalized.tolist()
            ),
            "profile_times_s": (
                internal_fields.profile_times_normalized
                * internal_fields.physics_config.time_scale_s
            ).tolist(),
            "figure_file": field_paths.figure.name,
            "grid_file": field_paths.grid.name,
            "validation_note": (
                "PINN-inferred internal fields; not validated against internal "
                "measurements."
            ),
            "bounded_run_caveat": (
                "Bounded smoke-run training does not establish convergence."
            ),
        }
    with (run_directory / "manifest.json").open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, allow_nan=False)
        manifest_file.write("\n")
    return run_directory


__all__ = ["export_results"]
