"""Artifact construction helpers for Strategy C reports.

The public API remains in ``strategy_c.reporting``. This module isolates
filesystem export, manifest assembly, quality-flag auditing, and plotting from
the held-out metric calculation so each stage remains independently
maintainable.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from strategy_c.data import (
    CurveData,
    ExperimentSplit,
    FeatureTransform,
    fit_feature_transform,
)
from strategy_c.physics import CoupledInversePINN, PhysicsConfig, flat_plate_transfer
from strategy_c.reporting import (
    EvaluationResult,
    _PREDICTION_COLUMNS,
    _REQUIRED_EXPERIMENT_COLUMNS,
    _REQUIRED_OBSERVATION_COLUMNS,
    _finite_array,
    _require_columns,
)
from strategy_c.training import SamplingConfig, TrainingConfig, TrainingResult


_CONFIG_UNITS: Final[dict[str, str]] = {
    "bed_depth_m": "m",
    "dish_diameter_m": "m",
    "time_scale_s": "s",
    "initial_temperature_K": "K",
    "temperature_difference_K": "K",
    "air_temperature_K": "K",
    "relative_humidity": "fraction (0-1)",
    "product_specific_heat_j_kg_k": "J kg-1 K-1",
    "product_thermal_conductivity_w_m_k": "W m-1 K-1",
    "latent_heat_j_kg": "J kg-1",
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
    "reference_moisture_db": "kg water kg-1 dry solids",
    "moisture_diffusivity_exponent": "dimensionless",
    "initial_reference_diffusivity_m2_s": "m2 s-1",
    "initial_activation_energy_j_mol": "J mol-1",
    "mass_flux_scale_kg_m2_s": "kg m-2 s-1",
    "heat_flux_scale_w_m2": "W m-2",
}
_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "One-dimensional, uniform, dry-bed slab of 0.01 m depth with a 0.05 m "
    "exposed dish diameter; the bottom is insulated and impermeable.",
    "Air temperature is 343.15 K (70 degC), initial product temperature is "
    "298.15 K, and the modeled drying time scale is 14400 s.",
    "Ambient relative humidity is set to zero as an explicit approximation; "
    "zero humidity does not imply zero surface resistance.",
    "Water activity is provisionally a_w = X / (X + 0.0667), with X on a dry "
    "basis. This is a user-specified single-reference-point assumption, not "
    "a verified or published coffee sorption equation.",
    "Dry mass, initial sample mass, initial dry-basis moisture, and dry-solid "
    "bulk density are treated as known retrospectively from measured records.",
    "Thermal properties, air properties, vapor-air diffusivity, and the "
    "transfer correlation are provisional inputs, not verified wet-bed "
    "measurements. Vapor-air diffusivity is not the learned bed diffusivity.",
    "The 70 degC air condition and predictions near the supported saturation "
    "pressure range may limit applicability; predicted temperature fields "
    "have not been independently validated.",
    "The 50-epoch Adam stage is a smoke-test budget, not evidence of "
    "convergence; bounded L-BFGS likewise does not guarantee convergence.",
    "An Arrhenius temperature dependence is imposed for bed diffusivity, but "
    "that constitutive model is unvalidated for this material and experiment.",
    "Effective reference diffusivity and activation energy may be "
    "non-identifiable from the available bulk drying curves.",
    "Measured mass increases are retained as observations with their quality "
    "flags. Workbook tx values and observation-derived flux targets are not "
    "used for fitting or evaluation.",
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


def _physics_report(
    data: CurveData, config: PhysicsConfig, reference_diffusivity_m2_s: float
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
        thermal_diffusivity = config.product_thermal_conductivity_w_m_k / (
            bulk_density * config.product_specific_heat_j_kg_k
        )
        thermal_fourier = (
            thermal_diffusivity * config.time_scale_s / config.bed_depth_m**2
        )
        thermal_biot = (
            coefficient.heat_transfer_coefficient_w_m2_k
            * config.bed_depth_m
            / config.product_thermal_conductivity_w_m_k
        )
        moisture_fourier = (
            reference_diffusivity_m2_s * config.time_scale_s / config.bed_depth_m**2
        )
        moisture_biot = (
            coefficient.mass_transfer_coefficient_m_s
            * config.bed_depth_m
            / reference_diffusivity_m2_s
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
                "moisture_fourier_number_at_fitted_Dref": moisture_fourier,
                "moisture_biot_number_at_fitted_Dref": moisture_biot,
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
        "thermal_fourier_number": "Fo_T = alpha_T * t_scale / bed_depth^2",
        "thermal_biot_number": "Bi_T = h_T * bed_depth / k_product",
        "moisture_fourier_number": "Fo_X = D_ref * t_scale / bed_depth^2",
        "moisture_biot_number": "Bi_X = h_m * bed_depth / D_ref",
    }
    return inputs, {
        "properties_and_scales": property_values,
        "dimensionless_scale_definitions": scales,
        "condition_coefficients": inputs,
    }


def _write_curve_plot(predictions: pd.DataFrame, destination: Path) -> None:
    """Render only actual held-out measurements and model predictions."""
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
    axes.set_title("Locked held-out drying curves (no empirical comparator supplied)")
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
        or evaluation.held_out_ids != training_result.provenance.held_out_ids
    ):
        raise ValueError("Evaluation IDs do not match training provenance.")
    expected_transform = fit_feature_transform(
        data, ExperimentSplit(evaluation.train_ids, evaluation.held_out_ids)
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
    if set(evaluation.held_out_ids).difference(
        evaluation.predictions["experiment_id"].astype(str)
    ):
        raise ValueError("The prediction table does not include every held-out curve.")
    if set(evaluation.held_out_ids).difference(
        evaluation.per_curve_metrics["experiment_id"].astype(str)
    ):
        raise ValueError("The metric table does not include every held-out curve.")
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
        ["moisture_predicted_normalized", "temperature_predicted_K"],
    ].to_numpy(dtype=float)
    _finite_array(valid_time_predictions, "Valid-time prediction records")
    outside_domain_predictions = evaluation.predictions.loc[
        ~in_modeled_domain,
        ["moisture_predicted_normalized", "temperature_predicted_K"],
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
    if (
        not math.isfinite(reference_diffusivity)
        or reference_diffusivity <= 0
        or not math.isfinite(activation_energy)
        or activation_energy <= 0
    ):
        raise ValueError(
            "Learned diffusivity and activation energy must be positive and finite."
        )
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
    transfer_inputs, physics_report = _physics_report(
        data, config, reference_diffusivity
    )
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
    pd.DataFrame.from_records(history_records).to_csv(
        run_directory / "loss_history.csv", index=False
    )
    quality_flags.to_csv(run_directory / "quality_flags.csv", index=False)
    _write_curve_plot(
        evaluation.predictions, run_directory / "normalized_drying_curves.png"
    )
    manifest: dict[str, object] = {
        "format_version": 2,
        "run_directory": timestamp_label,
        "train_ids": list(evaluation.train_ids),
        "held_out_ids": list(evaluation.held_out_ids),
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
            "architecture": {
                "independent_networks": ["moisture_network", "temperature_network"],
                "hidden_widths": [64, 32, 32, 64],
                "activation": "tanh",
                "scalar_output_activation": "softplus",
                "input_order": ["time_normalized", "depth_normalized"]
                + list(feature_transform.feature_names),
                "output_order": ["moisture_ratio_X_over_X0", "temperature_K"],
            },
            "state_dict_file": "model_state.pt",
            "state_dict_format": (
                "tensor-only PyTorch state_dict; load with "
                "torch.load(weights_only=True)"
            ),
            "dtype": str(model_dtype),
            "physics_config": asdict(config),
        },
        "feature_transform": {
            "method": "training-only population mean and standard deviation",
            "fitted_on_ids": list(evaluation.train_ids),
            "feature_names": list(feature_transform.feature_names),
            "mean": list(feature_transform.mean),
            "scale": list(feature_transform.scale),
        },
        "held_out_metrics_normalized_moisture": aggregate_values,
        "evaluation_quadrature_points": evaluation.quadrature_points,
        "per_curve_metric_definitions": {
            "mae": "mean absolute error in Xbulk/X0 units, computed per held-out curve",
            "rmse": "root mean square error in Xbulk/X0 units, computed per held-out curve",
            "equal_curve_mae": "arithmetic mean of per-curve MAE; each curve has equal weight",
            "equal_curve_rmse": "arithmetic mean of per-curve RMSE; each curve has equal weight",
            "pooled_mae": "point-weighted MAE across all held-out observations",
            "pooled_rmse": "point-weighted RMSE across all held-out observations",
        },
        "learned_parameters": {
            "reference_diffusivity_m2_s": reference_diffusivity,
            "reference_diffusivity_initial_m2_s": (
                config.initial_reference_diffusivity_m2_s
            ),
            "activation_energy_j_mol": activation_energy,
            "activation_energy_kj_mol": activation_energy / 1000.0,
            "activation_energy_initial_j_mol": config.initial_activation_energy_j_mol,
            "reference_temperature_K": config.reference_temperature_K,
            "reference_moisture_db_kg_water_kg_dry": config.reference_moisture_db,
        },
        "optimizer": {
            "adam": {
                "epochs": adam_history.epochs,
                "configured_epochs": optimizer_config.adam_epochs,
                "completed_epochs": adam_history.epochs,
                "learning_rate": optimizer_config.adam_learning_rate,
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
        "assumptions_and_limitations": list(_ASSUMPTIONS),
        "empirical_comparison": (
            "No compatible empirical reference curve or equation was supplied; "
            "no empirical comparator is fabricated."
        ),
        "artifacts": [
            "predictions.csv",
            "per_curve_metrics.csv",
            "loss_history.csv",
            "quality_flags.csv",
            "normalized_drying_curves.png",
            "model_state.pt",
        ],
    }
    with (run_directory / "manifest.json").open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, allow_nan=False)
        manifest_file.write("\n")
    return run_directory


__all__ = ["export_results"]
