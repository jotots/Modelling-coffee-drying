"""Locked final-test evaluation and exclusive Strategy C artifact export.

Evaluation is deliberately downstream of ``TrainingResult``: the experiment
split is checked against the optimizer's recorded provenance before any
final-test observation is scored. Metrics operate on bulk moisture normalized by
the curve-specific initial dry-basis moisture, and curve-level averages give
each complete final-test curve equal weight regardless of its observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, Mapping, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from strategy_c.data import CurveData, ExperimentSplit, FeatureTransform
from strategy_c.physics import (
    CoupledInversePINN,
    PhysicsConfig,
    PhysicsModel,
    bulk_predictions,
    conservation_diagnostics,
    depth_average_fields,
)
from strategy_c.training import (
    SamplingConfig,
    TrainingConfig,
    TrainingResult,
)

if TYPE_CHECKING:
    from strategy_c.field_visualization import InternalFieldResult


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
    "partition",
    "experiment_id",
    "condition_id",
    "time_s",
    "time_normalized",
    "moisture_observed_normalized",
    "moisture_predicted_normalized",
    "temperature_predicted_K",
    "temperature_predicted_theta",
    "target_available",
    "quality_flags",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Final-test predictions and normalized moisture-error summaries."""

    predictions: pd.DataFrame
    per_curve_metrics: pd.DataFrame
    aggregate_metrics: Mapping[str, float | int]
    conservation_diagnostics: pd.DataFrame
    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    quadrature_points: int
    validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    conservation_quadrature_points: int = 32

    @property
    def final_test_ids(self) -> tuple[str, ...]:
        """Resolve final-test IDs, preserving the historic held-out field."""
        if self.test_ids or self.validation_ids:
            return self.test_ids
        return self.held_out_ids


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict normalized bulk moisture and depth-mean Kelvin/theta values."""
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

    # Bulk moisture and the reported temperature summary share the same
    # differentiable unit-depth quadrature used by the training objective.
    with torch.inference_mode():
        moisture_bulk = bulk_predictions(
            cast(PhysicsModel, model),
            tau,
            features,
            conditions,
            config,
            quadrature_points=quadrature_points,
        )
        moisture_normalized = moisture_bulk / conditions[:, 2:3]

        _, temperature_mean = depth_average_fields(
            cast(PhysicsModel, model),
            tau,
            features,
            quadrature_points=quadrature_points,
        )
        temperature_theta_mean = (
            temperature_mean - config.air_temperature_K
        ) / config.signed_temperature_scale_K

    moisture_values = moisture_normalized.detach().cpu().numpy().reshape(-1)
    temperature_values = temperature_mean.detach().cpu().numpy().reshape(-1)
    temperature_theta_values = temperature_theta_mean.detach().cpu().numpy().reshape(-1)
    _finite_array(moisture_values, "Predicted normalized moisture")
    _finite_array(temperature_values, "Predicted temperature")
    _finite_array(temperature_theta_values, "Predicted normalized temperature")
    if (moisture_values <= 0).any() or (temperature_values <= 0).any():
        raise ValueError(
            "Predicted moisture and absolute temperature must be positive."
        )
    return moisture_values, temperature_values, temperature_theta_values


def evaluate_locked_model(
    training_result: TrainingResult,
    data: CurveData,
    split: ExperimentSplit,
    feature_transform: FeatureTransform,
    physics_config: PhysicsConfig | None = None,
    *,
    quadrature_points: int = 64,
    conservation_quadrature_points: int = 32,
) -> EvaluationResult:
    """Score only previously locked final-test curves after optimization.

    Training provenance must exactly match the supplied complete-curve split.
    Validation observations may be monitored by training, but this final
    evaluator scores only the disjoint test IDs after training has returned a
    ``TrainingResult``. Neither role enters the optimizer or feature transform.
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
    if (
        isinstance(conservation_quadrature_points, bool)
        or not isinstance(conservation_quadrature_points, int)
        or not 2 <= conservation_quadrature_points <= 1024
    ):
        raise ValueError(
            "conservation_quadrature_points must be an integer from 2 through 1024."
        )
    provenance = training_result.provenance
    if (
        provenance.train_ids != split.train_ids
        or provenance.validation_ids != split.validation_ids
        or provenance.final_test_ids != split.test_ids
    ):
        raise ValueError("The supplied split does not match training provenance.")
    if not split.test_ids:
        raise ValueError("At least one locked final-test curve is required.")
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
    if (
        set(split.train_ids).union(split.validation_ids, split.test_ids)
        != experiment_ids
    ):
        raise ValueError("The locked split must cover every experiment exactly.")
    if (
        set(split.train_ids).intersection(split.validation_ids)
        or set(split.train_ids).intersection(split.test_ids)
        or set(split.validation_ids).intersection(split.test_ids)
    ):
        raise ValueError(
            "Training, validation, and test experiment IDs must be disjoint."
        )
    if observations["experiment_id"].isna().any() or set(
        observations["experiment_id"].astype(str)
    ).difference(experiment_ids):
        raise ValueError("Observations must reference known experiment identifiers.")
    test_experiments = experiments.loc[
        experiments["_identifier"].isin(split.test_ids)
    ].set_index("_identifier")
    test_experiments = test_experiments.reindex(split.test_ids)
    if test_experiments.isna().any(axis=None):
        raise ValueError("Every final-test experiment must have complete metadata.")
    normalized_features = feature_transform.transform(
        test_experiments.loc[:, feature_transform.feature_names]
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
        for _, row in test_experiments.iterrows()
    }
    _finite_array(normalized_features, "Standardized final-test features")
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
            raise ValueError(f"Final-test curve {identifier!r} has no observations.")
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
                f"[0, {selected_config.time_scale_s:g}] s for final-test curve "
                f"{identifier!r}; out-of-domain times: {invalid_times}."
            )
        if (availability & (~valid_times | ~np.isfinite(observed_values))).any():
            raise ValueError(
                f"Available targets for {identifier!r} must have finite moisture "
                "and finite times in the modeled domain."
            )
        if not valid_times.any():
            raise ValueError(f"Final-test curve {identifier!r} has no modeled times.")
        curve_indices = np.flatnonzero(valid_times)
        tau_values = times[valid_times] / selected_config.time_scale_s
        moisture_values, temperature_values, temperature_theta_values = (
            _prediction_at_times(
                model,
                tau_values,
                normalized_features[curve_index],
                conditions_by_id[identifier],
                selected_config,
                quadrature_points,
                device,
                dtype,
            )
        )
        score_mask = availability & np.isfinite(observed_values) & valid_times
        if not score_mask.any():
            raise ValueError(
                f"Final-test curve {identifier!r} has no available finite moisture targets."
            )
        scored_locations = np.flatnonzero(score_mask)
        prediction_for_scores = np.full(len(curve), np.nan, dtype=float)
        prediction_for_scores[curve_indices] = moisture_values
        errors = prediction_for_scores[score_mask] - observed_values[score_mask]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if not math.isfinite(mae) or not math.isfinite(rmse):
            raise ValueError(
                f"Metrics for final-test curve {identifier!r} are non-finite."
            )
        metric_rows.append(
            {
                "partition": "test",
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
        local_temperature_theta = np.full(len(curve), np.nan, dtype=float)
        local_prediction[curve_indices] = moisture_values
        local_temperature[curve_indices] = temperature_values
        local_temperature_theta[curve_indices] = temperature_theta_values
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
                    "partition": "test",
                    "experiment_id": identifier,
                    "condition_id": str(
                        test_experiments.loc[identifier, "condition_id"]
                    ),
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
                    "temperature_predicted_theta": float(
                        local_temperature_theta[position]
                    )
                    if math.isfinite(local_temperature_theta[position])
                    else math.nan,
                    "target_available": bool(observation["target_available"])
                    if not pd.isna(observation["target_available"])
                    else False,
                    "quality_flags": flags[position],
                }
            )

    prediction_frame = pd.DataFrame(prediction_rows, columns=_PREDICTION_COLUMNS)
    metrics_frame = pd.DataFrame(
        metric_rows,
        columns=("partition", "experiment_id", "observation_count", "mae", "rmse"),
    )
    if prediction_frame.empty or metrics_frame.empty:
        raise ValueError("Locked evaluation produced no predictions or metrics.")
    score_values = metrics_frame.loc[:, ["mae", "rmse"]].to_numpy(dtype=float)
    _finite_array(score_values, "Final-test per-curve metrics")
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
        [
            "moisture_predicted_normalized",
            "temperature_predicted_K",
            "temperature_predicted_theta",
        ],
    ].to_numpy(dtype=float)
    _finite_array(valid_prediction_values, "Final-test predictions")
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
        raise ValueError("Final-test aggregate metrics must be finite.")
    if not isinstance(model, CoupledInversePINN):
        raise TypeError("Conservation diagnostics require CoupledInversePINN.")
    all_experiments = experiments.set_index("_identifier").reindex(
        (*split.train_ids, *split.validation_ids, *split.test_ids)
    )
    diagnostic_feature_values = feature_transform.transform(
        all_experiments.loc[:, feature_transform.feature_names]
    )
    diagnostic_conditions = all_experiments.loc[
        :,
        (
            "air_speed_m_s",
            "particle_diameter_m",
            "initial_moisture_db",
            "dry_bulk_density_kg_m3",
        ),
    ].to_numpy(dtype=float)
    _finite_array(diagnostic_feature_values, "Conservation diagnostic features")
    _finite_array(diagnostic_conditions, "Conservation diagnostic conditions")
    feature_tensor = torch.as_tensor(
        diagnostic_feature_values, device=device, dtype=dtype
    )
    conditions_tensor = torch.as_tensor(
        diagnostic_conditions, device=device, dtype=dtype
    )
    diagnostic_rows = conservation_diagnostics(
        model,
        feature_tensor,
        conditions_tensor,
        selected_config,
        quadrature_points=conservation_quadrature_points,
    )
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    diagnostic_count_per_curve = len(diagnostic_rows) // len(all_experiments)
    diagnostic_frame.insert(
        0,
        "experiment_id",
        np.repeat(
            all_experiments.index.astype(str).to_numpy(), diagnostic_count_per_curve
        ),
    )
    for column in diagnostic_frame.columns:
        if column != "experiment_id":
            _finite_array(
                pd.to_numeric(diagnostic_frame[column], errors="coerce").to_numpy(
                    dtype=float
                ),
                f"Conservation diagnostic {column}",
            )
    return EvaluationResult(
        predictions=prediction_frame,
        per_curve_metrics=metrics_frame,
        aggregate_metrics=aggregate,
        conservation_diagnostics=diagnostic_frame,
        train_ids=tuple(split.train_ids),
        held_out_ids=tuple(split.test_ids),
        quadrature_points=quadrature_points,
        validation_ids=tuple(split.validation_ids),
        test_ids=tuple(split.test_ids),
        conservation_quadrature_points=conservation_quadrature_points,
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
    internal_fields: InternalFieldResult | None = None,
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
        internal_fields=internal_fields,
    )


def load_model_artifacts(
    run_directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CoupledInversePINN:
    """Load a version-checked model state from one exported artifact run."""
    from strategy_c.reporting_artifacts import (
        load_model_artifacts as load_model_artifact_files,
    )

    return load_model_artifact_files(run_directory, map_location=map_location)


__all__ = [
    "EvaluationResult",
    "evaluate_locked_model",
    "export_results",
    "load_model_artifacts",
]
