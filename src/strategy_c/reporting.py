"""Locked held-out evaluation and exclusive Strategy C artifact export.

Evaluation is deliberately downstream of ``TrainingResult``: the experiment
split is checked against the optimizer's recorded provenance before any
held-out observation is scored. Metrics operate on bulk moisture normalized by
the curve-specific initial dry-basis moisture, and curve-level averages give
each complete held-out curve equal weight regardless of its observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from strategy_c.data import CurveData, ExperimentSplit, FeatureTransform
from strategy_c.physics import PhysicsConfig, bulk_predictions
from strategy_c.training import (
    SamplingConfig,
    TrainingConfig,
    TrainingResult,
)


_REQUIRED_EXPERIMENT_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "condition_id",
    "air_speed_m_s",
    "particle_diameter_m",
    "dry_mass_kg",
    "initial_sample_mass_kg",
    "initial_moisture_db",
    "dry_bulk_density_kg_m3",
)
_REQUIRED_OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "time_s",
    "moisture_normalized",
    "target_available",
)
_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "condition_id",
    "time_s",
    "time_normalized",
    "moisture_observed_normalized",
    "moisture_predicted_normalized",
    "temperature_predicted_K",
    "target_available",
    "quality_flags",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Held-out predictions and normalized moisture-error summaries."""

    predictions: pd.DataFrame
    per_curve_metrics: pd.DataFrame
    aggregate_metrics: Mapping[str, float | int]
    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    quadrature_points: int


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], label: str
) -> None:
    """Fail at the reporting boundary when upstream parser fields are absent."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer tensor placement from model parameters without moving user state."""
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("The trained model must contain parameters.")
    parameter = parameters[0]
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("The trained model must use float32 or float64 parameters.")
    if any(
        item.device != parameter.device or item.dtype != parameter.dtype
        for item in parameters
    ):
        raise ValueError("All trained model parameters must share device and dtype.")
    return parameter.device, parameter.dtype


def _finite_array(values: np.ndarray, label: str) -> None:
    """Reject model outputs that would otherwise contaminate scores or files."""
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values.")


def _prediction_at_times(
    model: nn.Module,
    tau_values: np.ndarray,
    normalized_features: np.ndarray,
    physical_conditions: np.ndarray,
    config: PhysicsConfig,
    quadrature_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict normalized bulk moisture and depth-mean absolute temperature."""
    tau = torch.as_tensor(tau_values, device=device, dtype=dtype).reshape(-1, 1)
    features = torch.as_tensor(
        np.repeat(normalized_features.reshape(1, -1), len(tau_values), axis=0),
        device=device,
        dtype=dtype,
    )
    conditions = torch.as_tensor(
        np.repeat(physical_conditions.reshape(1, -1), len(tau_values), axis=0),
        device=device,
        dtype=dtype,
    )

    # Bulk moisture uses the same differentiable depth integral as the
    # training objective; division by curve-specific X0 yields Xbulk/X0.
    with torch.inference_mode():
        moisture_bulk = bulk_predictions(
            model,
            tau,
            features,
            conditions,
            config,
            quadrature_points=quadrature_points,
        )
        moisture_normalized = moisture_bulk / conditions[:, 2:3]

        # Temperature is summarized as its depth average using the same
        # Gauss-Legendre quadrature, not misrepresented as a surface sensor.
        nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_points)
        depth_nodes = torch.as_tensor(
            (nodes_np + 1.0) / 2.0, device=device, dtype=dtype
        ).reshape(1, quadrature_points, 1)
        depth_weights = torch.as_tensor(
            weights_np / 2.0, device=device, dtype=dtype
        ).reshape(1, quadrature_points, 1)
        expanded_tau = tau[:, None, :].expand(-1, quadrature_points, -1).reshape(-1, 1)
        expanded_features = (
            features[:, None, :]
            .expand(-1, quadrature_points, -1)
            .reshape(-1, features.shape[1])
        )
        expanded_depth = depth_nodes.expand(len(tau_values), -1, -1).reshape(-1, 1)
        _, temperature = model(expanded_tau, expanded_depth, expanded_features)
        temperature_mean = (
            temperature.reshape(len(tau_values), quadrature_points, 1) * depth_weights
        ).sum(dim=1)

    moisture_values = moisture_normalized.detach().cpu().numpy().reshape(-1)
    temperature_values = temperature_mean.detach().cpu().numpy().reshape(-1)
    _finite_array(moisture_values, "Predicted normalized moisture")
    _finite_array(temperature_values, "Predicted temperature")
    if (moisture_values < 0).any() or (temperature_values <= 0).any():
        raise ValueError(
            "Predicted moisture and absolute temperature must be positive."
        )
    return moisture_values, temperature_values


def evaluate_locked_model(
    training_result: TrainingResult,
    data: CurveData,
    split: ExperimentSplit,
    feature_transform: FeatureTransform,
    physics_config: PhysicsConfig | None = None,
    *,
    quadrature_points: int = 64,
) -> EvaluationResult:
    """Score only the previously locked held-out curves after optimization.

    Training provenance must exactly match the supplied complete-curve split.
    The evaluation reads held-out observations only after training has already
    returned a ``TrainingResult``; they never enter the optimizer or transform.
    Every available measured point is preserved, including flagged increases.
    """
    if not isinstance(training_result, TrainingResult):
        raise TypeError("training_result must be a TrainingResult.")
    if not isinstance(data, CurveData):
        raise TypeError("data must be CurveData.")
    if not isinstance(split, ExperimentSplit):
        raise TypeError("split must be ExperimentSplit.")
    if not isinstance(feature_transform, FeatureTransform):
        raise TypeError("feature_transform must be FeatureTransform.")
    selected_config = PhysicsConfig() if physics_config is None else physics_config
    if not isinstance(selected_config, PhysicsConfig):
        raise TypeError("physics_config must be PhysicsConfig.")
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, int)
        or quadrature_points < 2
        or quadrature_points > 1024
    ):
        raise ValueError("quadrature_points must be an integer from 2 through 1024.")
    provenance = training_result.provenance
    if (
        provenance.train_ids != split.train_ids
        or provenance.held_out_ids != split.test_ids
    ):
        raise ValueError("The supplied split does not match training provenance.")
    if not split.test_ids:
        raise ValueError("At least one locked held-out curve is required.")
    _require_columns(data.experiments, _REQUIRED_EXPERIMENT_COLUMNS, "experiments")
    _require_columns(data.observations, _REQUIRED_OBSERVATION_COLUMNS, "observations")
    if "quality_flags" not in data.observations:
        # The notebook parser supplies this field; a transparent empty string
        # remains appropriate for compatible data sources without diagnostics.
        observations = data.observations.copy(deep=True)
        observations["quality_flags"] = ""
    else:
        observations = data.observations

    experiments = data.experiments.copy(deep=False)
    experiments["_identifier"] = experiments["experiment_id"].astype(str)
    if experiments["_identifier"].duplicated().any():
        raise ValueError("Experiment identifiers must be unique for evaluation.")
    experiment_ids = set(experiments["_identifier"])
    if set(split.train_ids).union(split.test_ids) != experiment_ids:
        raise ValueError("The locked split must cover every experiment exactly.")
    if set(split.train_ids).intersection(split.test_ids):
        raise ValueError("Training and held-out experiment IDs must be disjoint.")
    if observations["experiment_id"].isna().any() or set(
        observations["experiment_id"].astype(str)
    ).difference(experiment_ids):
        raise ValueError("Observations must reference known experiment identifiers.")
    held_out_experiments = experiments.loc[
        experiments["_identifier"].isin(split.test_ids)
    ].set_index("_identifier")
    held_out_experiments = held_out_experiments.reindex(split.test_ids)
    if held_out_experiments.isna().any(axis=None):
        raise ValueError("Every held-out experiment must have complete metadata.")
    normalized_features = feature_transform.transform(
        held_out_experiments.loc[:, feature_transform.feature_names]
    )
    conditions_by_id = {
        str(row["experiment_id"]): np.asarray(
            [
                row["air_speed_m_s"],
                row["particle_diameter_m"],
                row["initial_moisture_db"],
                row["dry_bulk_density_kg_m3"],
            ],
            dtype=float,
        )
        for _, row in held_out_experiments.iterrows()
    }
    _finite_array(normalized_features, "Standardized held-out features")
    for identifier, values in conditions_by_id.items():
        _finite_array(values, f"Physical conditions for {identifier}")
        if (values <= 0).any():
            raise ValueError(f"Physical conditions for {identifier} must be positive.")

    model = training_result.model
    if not isinstance(model, nn.Module):
        raise TypeError("TrainingResult.model must be a torch.nn.Module.")
    if hasattr(model, "config") and model.config != selected_config:
        raise ValueError("The evaluation PhysicsConfig differs from the trained model.")
    device, dtype = _model_device_dtype(model)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for curve_index, identifier in enumerate(split.test_ids):
        curve = observations.loc[
            observations["experiment_id"].astype(str).eq(identifier)
        ].copy()
        if curve.empty:
            raise ValueError(f"Held-out curve {identifier!r} has no observations.")
        times = pd.to_numeric(curve["time_s"], errors="coerce").to_numpy(dtype=float)
        valid_times = (
            np.isfinite(times)
            & (times >= 0.0)
            & (times <= selected_config.time_scale_s)
        )
        observed_values = pd.to_numeric(
            curve["moisture_normalized"], errors="coerce"
        ).to_numpy(dtype=float)
        raw_availability = curve["target_available"]
        if (
            not raw_availability.dropna()
            .map(lambda value: isinstance(value, (bool, np.bool_)))
            .all()
        ):
            raise ValueError("target_available values must be boolean or missing.")
        availability = raw_availability.fillna(False).to_numpy(dtype=bool)
        if (observed_values[availability & np.isfinite(observed_values)] < 0).any():
            raise ValueError(
                f"Available normalized moisture for {identifier!r} cannot be negative."
            )
        available_outside_domain = availability & np.isfinite(times) & ~valid_times
        if available_outside_domain.any():
            invalid_times = times[available_outside_domain].tolist()
            raise ValueError(
                f"target_available is true outside modeled domain "
                f"[0, {selected_config.time_scale_s:g}] s for held-out curve "
                f"{identifier!r}; out-of-domain times: {invalid_times}."
            )
        if (availability & (~valid_times | ~np.isfinite(observed_values))).any():
            raise ValueError(
                f"Available targets for {identifier!r} must have finite moisture "
                "and finite times in the modeled domain."
            )
        if not valid_times.any():
            raise ValueError(f"Held-out curve {identifier!r} has no modeled times.")
        curve_indices = np.flatnonzero(valid_times)
        tau_values = times[valid_times] / selected_config.time_scale_s
        moisture_values, temperature_values = _prediction_at_times(
            model,
            tau_values,
            normalized_features[curve_index],
            conditions_by_id[identifier],
            selected_config,
            quadrature_points,
            device,
            dtype,
        )
        score_mask = availability & np.isfinite(observed_values) & valid_times
        if not score_mask.any():
            raise ValueError(
                f"Held-out curve {identifier!r} has no available finite moisture targets."
            )
        scored_locations = np.flatnonzero(score_mask)
        prediction_for_scores = np.full(len(curve), np.nan, dtype=float)
        prediction_for_scores[curve_indices] = moisture_values
        errors = prediction_for_scores[score_mask] - observed_values[score_mask]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if not math.isfinite(mae) or not math.isfinite(rmse):
            raise ValueError(
                f"Metrics for held-out curve {identifier!r} are non-finite."
            )
        metric_rows.append(
            {
                "experiment_id": identifier,
                "observation_count": int(len(scored_locations)),
                "mae": mae,
                "rmse": rmse,
            }
        )

        # Preserve every scheduled record for audit, including invalid source
        # rows. Out-of-domain/missing times keep NaN predictions and cannot
        # enter metrics; available mass increases remain ordinary score points.
        local_prediction = np.full(len(curve), np.nan, dtype=float)
        local_temperature = np.full(len(curve), np.nan, dtype=float)
        local_prediction[curve_indices] = moisture_values
        local_temperature[curve_indices] = temperature_values
        experiment = held_out_experiments.loc[identifier]
        flags = curve["quality_flags"].fillna("").map(str).to_numpy()
        for position, (_, observation) in enumerate(curve.iterrows()):
            time_value = times[position]
            row_time_normalized = (
                float(time_value / selected_config.time_scale_s)
                if math.isfinite(time_value)
                else math.nan
            )
            target_value = observed_values[position]
            prediction_rows.append(
                {
                    "experiment_id": identifier,
                    "condition_id": str(experiment["condition_id"]),
                    "time_s": float(time_value)
                    if math.isfinite(time_value)
                    else math.nan,
                    "time_normalized": row_time_normalized,
                    "moisture_observed_normalized": (
                        float(target_value) if math.isfinite(target_value) else math.nan
                    ),
                    "moisture_predicted_normalized": float(local_prediction[position])
                    if math.isfinite(local_prediction[position])
                    else math.nan,
                    "temperature_predicted_K": float(local_temperature[position])
                    if math.isfinite(local_temperature[position])
                    else math.nan,
                    "target_available": bool(observation["target_available"])
                    if not pd.isna(observation["target_available"])
                    else False,
                    "quality_flags": flags[position],
                }
            )

    prediction_frame = pd.DataFrame(prediction_rows, columns=_PREDICTION_COLUMNS)
    metrics_frame = pd.DataFrame(
        metric_rows, columns=("experiment_id", "observation_count", "mae", "rmse")
    )
    if prediction_frame.empty or metrics_frame.empty:
        raise ValueError("Locked evaluation produced no predictions or metrics.")
    score_values = metrics_frame.loc[:, ["mae", "rmse"]].to_numpy(dtype=float)
    _finite_array(score_values, "Held-out per-curve metrics")
    normalized_prediction_times = pd.to_numeric(
        prediction_frame["time_normalized"], errors="coerce"
    ).to_numpy(dtype=float)
    expected_prediction_mask = (
        np.isfinite(normalized_prediction_times)
        & (normalized_prediction_times >= 0.0)
        & (normalized_prediction_times <= 1.0)
    )
    valid_prediction_values = prediction_frame.loc[
        expected_prediction_mask,
        ["moisture_predicted_normalized", "temperature_predicted_K"],
    ].to_numpy(dtype=float)
    _finite_array(valid_prediction_values, "Held-out predictions")
    aggregate_mask = (
        prediction_frame["target_available"]
        & prediction_frame["moisture_observed_normalized"].notna()
        & prediction_frame["moisture_predicted_normalized"].notna()
    )
    observed_all = prediction_frame.loc[
        aggregate_mask,
        "moisture_observed_normalized",
    ].to_numpy(dtype=float)
    all_errors = (
        prediction_frame.loc[
            aggregate_mask,
            "moisture_predicted_normalized",
        ].to_numpy(dtype=float)
        - observed_all
    )
    aggregate: dict[str, float | int] = {
        "curve_count": int(len(metrics_frame)),
        "observation_count": int(metrics_frame["observation_count"].sum()),
        "equal_curve_mae": float(metrics_frame["mae"].mean()),
        "equal_curve_rmse": float(metrics_frame["rmse"].mean()),
        "pooled_mae": float(np.mean(np.abs(all_errors))),
        "pooled_rmse": float(np.sqrt(np.mean(np.square(all_errors)))),
    }
    if any(
        not math.isfinite(value)
        for key, value in aggregate.items()
        if key != "curve_count" and key != "observation_count"
    ):
        raise ValueError("Held-out aggregate metrics must be finite.")
    return EvaluationResult(
        predictions=prediction_frame,
        per_curve_metrics=metrics_frame,
        aggregate_metrics=aggregate,
        train_ids=tuple(split.train_ids),
        held_out_ids=tuple(split.test_ids),
        quadrature_points=quadrature_points,
    )


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
    """Export evaluation and reconstruction inputs to a new exclusive folder.

    The explicit feature transform is required because the trained networks
    consume standardized features and that train-only mapping is not otherwise
    recoverable from their weights.
    """
    from strategy_c.reporting_artifacts import export_results as export_artifacts

    return export_artifacts(
        evaluation,
        training_result,
        data,
        output_root,
        physics_config,
        training_config,
        sampling_config,
        feature_transform=feature_transform,
        run_timestamp=run_timestamp,
    )


__all__ = ["EvaluationResult", "evaluate_locked_model", "export_results"]
