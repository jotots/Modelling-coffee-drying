"""Reproducible, conditioned Adam/L-BFGS training for Strategy C.

All fitting is performed on complete training curves only. The observation
batch and every collocation coordinate are materialized before optimization,
so both Adam and every L-BFGS closure evaluate the same deterministic,
dimensionless objective. Validation moisture is monitored separately without
gradients; final-test observations are excluded until post-training evaluation.
Adam first fits data alone, then ramps a fixed, calibrated six-residual physics
objective to its full configured weight. L-BFGS sees that same frozen objective
in every closure; line-search trials and the final accepted state are reported
separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
import time
from typing import Callable, Final, Mapping, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from strategy_c.data import CurveData, ExperimentSplit, FeatureTransform
from strategy_c.physics import (
    CoupledInversePINN,
    PhysicsBatch,
    PhysicsConfig,
    PhysicsModel,
    bulk_predictions,
    loss_components,
    physics_residuals,
)


_PHYSICAL_CONDITION_COLUMNS: Final[tuple[str, ...]] = (
    "air_speed_m_s",
    "particle_diameter_m",
    "initial_moisture_db",
    "dry_bulk_density_kg_m3",
)
# Attribute names of the shared, learned diffusivity-law parameters on
# CoupledInversePINN. They receive TrainingConfig.physics_parameter_learning_rate.
# The activation energy is a fixed buffer (Barnard et al. 2024), so it is absent.
_DIFFUSIVITY_LAW_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "log_reference_diffusivity",
    "diameter_exponent",
    "speed_exponent",
)
_ACTIVE_PHYSICS_TERMS: Final[tuple[str, ...]] = (
    "physics_mass",
    "physics_heat",
    "base_moisture",
    "base_temperature",
    "surface_mass",
    "surface_heat",
)


@dataclass(frozen=True)
class SamplingConfig:
    """Fixed collocation sizes and seed used to build one full training batch."""

    domain_points: int = 64
    initial_points: int = 16
    boundary_points: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject empty or unbounded sampling requests before allocating tensors."""
        for name in ("domain_points", "initial_points", "boundary_points"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 1
                or int(value) > 100_000
            ):
                raise ValueError(f"{name} must be an integer from 1 through 100000.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")


@dataclass(frozen=True)
class TrainingConfig:
    """Bounded settings for staged Adam followed by fixed-objective L-BFGS.

    ``adam_learning_rate`` drives the neural-network weights, while
    ``physics_parameter_learning_rate`` drives the three learned shared
    diffusivity-law parameters (log D_ref, diameter and speed exponents); the
    activation energy is fixed. At 0.001 Adam moves log D_ref only ~0.001 per
    step, so the mass PDE can otherwise lower diffusivity by cooling the bed
    through the Arrhenius term; a larger, separate rate lets law parameters
    respond. The fractional data-only warmup and physics ramp are rounded to
    integer epochs with halves upward; one final full-objective epoch is
    reserved. L-BFGS has no learning rate and uses frozen post-warmup scales.
    """

    adam_epochs: int = 1000
    adam_learning_rate: float = 0.001
    physics_parameter_learning_rate: float = 0.05
    lbfgs_max_iter: int = 1000
    lbfgs_max_eval: int = 1000
    lbfgs_history_size: int = 50
    seed: int = 0
    device: str = "auto"
    dtype: torch.dtype = torch.float64
    physics_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    quadrature_points: int = 32
    warmup_fraction: float = 0.2
    ramp_fraction: float = 0.4

    def __post_init__(self) -> None:
        """Validate optimizer work bounds and dimensionless objective weights."""
        limits = (
            ("adam_epochs", self.adam_epochs, 1, 10_000),
            ("lbfgs_max_iter", self.lbfgs_max_iter, 1, 1_000),
            ("lbfgs_max_eval", self.lbfgs_max_eval, 1, 2_000),
            ("lbfgs_history_size", self.lbfgs_history_size, 1, 1_000),
            ("quadrature_points", self.quadrature_points, 2, 1_024),
            ("seed", self.seed, 0, 2**32 - 1),
        )
        for name, value, minimum, maximum in limits:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < minimum
                or int(value) > maximum
            ):
                raise ValueError(
                    f"{name} must be an integer from {minimum} through {maximum}."
                )
        if self.lbfgs_max_eval < self.lbfgs_max_iter:
            raise ValueError("lbfgs_max_eval must be at least lbfgs_max_iter.")
        for name in ("adam_learning_rate", "physics_parameter_learning_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("physics_weight", "initial_weight", "boundary_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("warmup_fraction", "ramp_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 1.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64.")


@dataclass(frozen=True)
class TrainingBatch:
    """One immutable observation batch and fixed physics collocation batch."""

    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    observation_experiment_ids: tuple[str, ...]
    observation_tau: Tensor
    observation_features: Tensor
    observation_conditions: Tensor
    observation_targets: Tensor
    physics: PhysicsBatch
    validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check observation alignment and enforce disjoint curve provenance."""
        if not self.train_ids:
            raise ValueError("TrainingBatch requires at least one training curve.")
        if len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError("Training identifiers must be unique.")
        if len(set(self.held_out_ids)) != len(self.held_out_ids):
            raise ValueError("Held-out identifiers must be unique.")
        if set(self.train_ids).intersection(self.held_out_ids):
            raise ValueError("Training and held-out identifiers must be disjoint.")
        if len(set(self.validation_ids)) != len(self.validation_ids):
            raise ValueError("Validation identifiers must be unique.")
        if len(set(self.test_ids)) != len(self.test_ids):
            raise ValueError("Final-test identifiers must be unique.")
        if (
            set(self.train_ids).intersection(self.validation_ids)
            or set(self.train_ids).intersection(self.test_ids)
            or set(self.validation_ids).intersection(self.test_ids)
        ):
            raise ValueError("Training, validation, and test IDs must be disjoint.")
        if (self.validation_ids or self.test_ids) and set(self.held_out_ids) != set(
            self.validation_ids
        ).union(self.test_ids):
            raise ValueError(
                "held_out_ids must equal the explicit validation/test union."
            )
        if set(self.observation_experiment_ids).difference(self.train_ids):
            raise ValueError("Observation data include a non-training experiment ID.")
        count = len(self.observation_experiment_ids)
        tensors = (
            self.observation_tau,
            self.observation_features,
            self.observation_conditions,
            self.observation_targets,
        )
        if count == 0:
            raise ValueError("Training observations must not be empty.")
        if any(
            not isinstance(value, Tensor) or value.shape[0] != count
            for value in tensors
        ):
            raise ValueError(
                "Observation tensors must align with their experiment IDs."
            )
        if self.observation_tau.shape != (count, 1):
            raise ValueError("observation_tau must have shape (N, 1).")
        if self.observation_features.shape != (count, 4):
            raise ValueError("observation_features must have shape (N, 4).")
        if self.observation_conditions.shape != (count, 4):
            raise ValueError("observation_conditions must have shape (N, 4).")
        if self.observation_targets.shape != (count, 1):
            raise ValueError("observation_targets must have shape (N, 1).")
        reference = self.observation_tau
        for tensor in tensors[1:]:
            if tensor.device != reference.device or tensor.dtype != reference.dtype:
                raise ValueError("Observation tensors must share dtype and device.")
            if not bool(torch.isfinite(tensor).all().detach().cpu()):
                raise ValueError("Training observations must be finite.")
        if not bool(torch.isfinite(reference).all().detach().cpu()):
            raise ValueError("Training observation times must be finite.")


@dataclass(frozen=True)
class TrainingProvenance:
    """Record complete-curve roles plus a legacy held-out ID alias.

    ``held_out_ids`` preserves the former train-versus-held-out provenance
    field. New runs additionally record validation and final-test IDs
    separately; readers must not infer a validation subset from legacy IDs.
    """

    seed: int
    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    training_observation_count: int
    collocation_point_count: int
    device: str
    dtype: str
    validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate explicit role provenance without guessing legacy roles."""
        for label, identifiers in (
            ("training", self.train_ids),
            ("validation", self.validation_ids),
            ("final-test", self.test_ids),
            ("legacy held-out", self.held_out_ids),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{label} identifiers must be unique.")
        if self.validation_ids or self.test_ids:
            training = set(self.train_ids)
            validation = set(self.validation_ids)
            test = set(self.test_ids)
            if (
                training.intersection(validation)
                or training.intersection(test)
                or validation.intersection(test)
            ):
                raise ValueError(
                    "Training, validation, and final-test provenance must be disjoint."
                )
            if set(self.held_out_ids) != validation.union(test):
                raise ValueError(
                    "Legacy held_out_ids must equal the explicit validation/test union."
                )

    @property
    def final_test_ids(self) -> tuple[str, ...]:
        """Resolve final-test IDs, falling back only for legacy provenance."""
        if self.test_ids or self.validation_ids:
            return self.test_ids
        return self.held_out_ids


@dataclass
class StageHistory:
    """Loss snapshots and timing/evaluation counts for one optimizer stage."""

    epochs: int = 0
    iterations: int = 0
    closure_calls: int = 0
    elapsed_seconds: float = 0.0
    losses: list[dict[str, float]] = field(default_factory=list)
    accepted_losses: dict[str, float] = field(default_factory=dict)
    accepted_gradient_norm: float | None = None
    termination_flags: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """Trained model with stage histories and data-use provenance."""

    model: nn.Module
    stage_history: Mapping[str, StageHistory]
    provenance: TrainingProvenance
    physics_scales: Mapping[str, float] = field(default_factory=dict)
    physics_calibration_mse: Mapping[str, float] = field(default_factory=dict)
    schedule: Mapping[str, int] = field(default_factory=dict)


def _resolve_device(requested: str) -> torch.device:
    """Use CUDA only when requested or automatically available and usable."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _validate_split(data: CurveData, split: ExperimentSplit) -> pd.DataFrame:
    """Ensure the split covers precisely the prepared curves before training."""
    if not isinstance(data, CurveData):
        raise TypeError("data must be a CurveData instance.")
    if not isinstance(split, ExperimentSplit):
        raise TypeError("split must be an ExperimentSplit instance.")
    required_experiments = (
        "experiment_id",
        *_PHYSICAL_CONDITION_COLUMNS,
    )
    missing = [
        column for column in required_experiments if column not in data.experiments
    ]
    if missing:
        raise ValueError(f"Prepared experiments are missing columns: {missing}.")
    if data.experiments["experiment_id"].isna().any():
        raise ValueError("Prepared experiment identifiers must be present and unique.")
    identifiers = data.experiments["experiment_id"].astype(str)
    if identifiers.duplicated().any():
        raise ValueError("Prepared experiment identifiers must be present and unique.")
    known = set(identifiers)
    training = set(split.train_ids)
    validation = set(split.validation_ids)
    test = set(split.test_ids)
    if not training:
        raise ValueError("At least one training experiment is required.")
    if (
        training.intersection(validation)
        or training.intersection(test)
        or validation.intersection(test)
    ):
        raise ValueError("Training, validation, and test IDs must be disjoint.")
    if training.union(validation, test) != known:
        raise ValueError(
            "The split must cover every prepared curve exactly once across "
            "training, validation, and test."
        )
    conditions = data.experiments.loc[:, _PHYSICAL_CONDITION_COLUMNS].to_numpy(
        dtype=float
    )
    if not np.isfinite(conditions).all():
        raise ValueError("Prepared physical conditions must be finite.")
    if (conditions[:, 0] <= 0.0).any():
        zero_speed_ids = (
            data.experiments.loc[conditions[:, 0] <= 0.0, "experiment_id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Training requires strictly positive air_speed_m_s for the signed "
            f"diffusivity power law; invalid curves: {zero_speed_ids}."
        )
    if (conditions[:, 1:] <= 0.0).any():
        raise ValueError(
            "Training particle diameter, initial moisture, and dry bulk density "
            "must be strictly positive."
        )
    return data.experiments.copy(deep=False).assign(_identifier=identifiers)


def _tensor(
    values: np.ndarray | list[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a fresh tensor with the chosen precision and target device."""
    return torch.as_tensor(np.asarray(values), device=device, dtype=dtype)


def sample_training_batch(
    data: CurveData,
    split: ExperimentSplit,
    feature_transform: FeatureTransform,
    sampling_config: SamplingConfig | None = None,
    physics_config: PhysicsConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> TrainingBatch:
    """Select train-only measured observations and draw fixed collocation points.

    Data rows flagged unavailable are excluded from supervision. Valid measured
    increases are retained without smoothing or clipping. Sampling is seeded
    once here; no random coordinate generation occurs inside optimizer closures.
    """
    selected_sampling = SamplingConfig() if sampling_config is None else sampling_config
    selected_physics = PhysicsConfig() if physics_config is None else physics_config
    if not isinstance(selected_sampling, SamplingConfig):
        raise TypeError("sampling_config must be a SamplingConfig instance.")
    if not isinstance(selected_physics, PhysicsConfig):
        raise TypeError("physics_config must be a PhysicsConfig instance.")
    if not isinstance(feature_transform, FeatureTransform):
        raise TypeError("feature_transform must be a FeatureTransform instance.")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA sampling was requested but is not available.")
    experiments = _validate_split(data, split)
    required_observations = (
        "experiment_id",
        "time_s",
        "moisture_normalized",
        "target_available",
    )
    missing = [
        column for column in required_observations if column not in data.observations
    ]
    if missing:
        raise ValueError(f"Prepared observations are missing columns: {missing}.")
    availability = data.observations["target_available"]
    if (
        not availability.dropna()
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .all()
    ):
        raise ValueError("target_available values must be boolean or missing.")

    train_ids = tuple(str(identifier) for identifier in split.train_ids)
    validation_ids = tuple(str(identifier) for identifier in split.validation_ids)
    test_ids = tuple(str(identifier) for identifier in split.test_ids)
    train_experiments = experiments.loc[
        experiments["_identifier"].isin(train_ids)
    ].copy()
    if len(train_experiments) != len(train_ids):
        raise ValueError("Every training identifier must resolve to one experiment.")
    # This application deliberately transforms training rows alone. Validation
    # and final-test features are not transformed or used in the objective.
    train_features = feature_transform.transform(
        train_experiments.loc[:, feature_transform.feature_names]
    )
    train_experiments["_feature_row"] = list(train_features)
    observations = data.observations.loc[
        data.observations["experiment_id"].astype(str).isin(train_ids)
        & data.observations["target_available"].fillna(False).astype(bool)
    ].copy()
    if observations.empty:
        raise ValueError(
            "Training observations are empty after availability filtering."
        )
    observations["time_s"] = pd.to_numeric(observations["time_s"], errors="coerce")
    observations["moisture_normalized"] = pd.to_numeric(
        observations["moisture_normalized"], errors="coerce"
    )
    values = observations.loc[:, ["time_s", "moisture_normalized"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Available training observations must contain finite values.")
    if (values[:, 0] < 0).any() or (values[:, 0] > selected_physics.time_scale_s).any():
        raise ValueError(
            "Training observation times must lie within the modeled interval."
        )
    if (values[:, 1] < 0).any():
        raise ValueError(
            "Available training normalized moisture targets must be nonnegative."
        )
    if observations["experiment_id"].isna().any():
        raise ValueError("Available training observations must have experiment IDs.")

    condition_table = train_experiments.set_index("_identifier")
    features_by_id = {
        str(identifier): np.asarray(feature_row, dtype=float)
        for identifier, feature_row in zip(
            train_experiments["_identifier"], train_experiments["_feature_row"]
        )
    }
    conditions_by_id = {
        str(identifier): row
        for identifier, row in zip(
            condition_table.index,
            condition_table.loc[:, _PHYSICAL_CONDITION_COLUMNS].to_numpy(dtype=float),
        )
    }
    observations["_identifier"] = observations["experiment_id"].astype(str)
    # The per-curve completeness check avoids silently changing equal-curve
    # weighting when a training replicate has no valid measured masses.
    counts = observations.groupby("_identifier", sort=False).size().to_dict()
    missing_curves = [
        identifier for identifier in train_ids if not counts.get(identifier)
    ]
    if missing_curves:
        raise ValueError(
            "Each training curve requires available training observations; "
            f"missing for {missing_curves}."
        )
    observation_ids = tuple(observations["_identifier"].tolist())
    observation_tau = values[:, 0:1] / selected_physics.time_scale_s
    observation_features = np.stack(
        [features_by_id[identifier] for identifier in observation_ids]
    )
    observation_conditions = np.stack(
        [conditions_by_id[identifier] for identifier in observation_ids]
    )
    observation_targets = values[:, 1:2]
    generator = np.random.default_rng(int(selected_sampling.seed))
    count = len(train_ids)
    domain_shape = (count, selected_sampling.domain_points)
    initial_shape = (count, selected_sampling.initial_points)
    boundary_shape = (count, selected_sampling.boundary_points)

    def draw_rows(
        identifiers: tuple[str, ...], shape: tuple[int, int], *, time: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw coordinates while repeating one curve's fixed condition row."""
        sampled_tau = generator.random(shape) if time else np.zeros(shape)
        sampled_xi = generator.random(shape)
        row_features = np.stack([features_by_id[item] for item in identifiers])
        row_conditions = np.stack([conditions_by_id[item] for item in identifiers])
        expanded_features = np.repeat(row_features[:, None, :], shape[1], axis=1)
        expanded_conditions = np.repeat(row_conditions[:, None, :], shape[1], axis=1)
        return (
            sampled_tau.reshape(-1, 1),
            sampled_xi.reshape(-1, 1),
            expanded_features.reshape(-1, 4),
            expanded_conditions.reshape(-1, 4),
        )

    domain_tau, domain_xi, domain_features, domain_conditions = draw_rows(
        train_ids, domain_shape, time=True
    )
    initial_tau, initial_xi, initial_features, initial_conditions = draw_rows(
        train_ids, initial_shape, time=False
    )
    del initial_tau
    base_tau, _, base_features, base_conditions = draw_rows(
        train_ids, boundary_shape, time=True
    )
    surface_tau, _, surface_features, surface_conditions = draw_rows(
        train_ids, boundary_shape, time=True
    )
    # Keep every positive-time surface sample but exclude the incompatible
    # uniform-initial-state/Robin corner at exactly tau=0.
    surface_keep = surface_tau[:, 0] > 0.0
    if not surface_keep.any():
        raise ValueError(
            "Sampling produced no positive-time surface points after excluding "
            "the exact initial corner."
        )
    surface_tau = surface_tau[surface_keep]
    surface_features = surface_features[surface_keep]
    surface_conditions = surface_conditions[surface_keep]
    physics = PhysicsBatch(
        domain_tau=_tensor(domain_tau, device=selected_device, dtype=dtype),
        domain_xi=_tensor(domain_xi, device=selected_device, dtype=dtype),
        domain_features=_tensor(domain_features, device=selected_device, dtype=dtype),
        domain_conditions=_tensor(
            domain_conditions, device=selected_device, dtype=dtype
        ),
        initial_xi=_tensor(initial_xi, device=selected_device, dtype=dtype),
        initial_features=_tensor(initial_features, device=selected_device, dtype=dtype),
        initial_conditions=_tensor(
            initial_conditions, device=selected_device, dtype=dtype
        ),
        base_tau=_tensor(base_tau, device=selected_device, dtype=dtype),
        base_features=_tensor(base_features, device=selected_device, dtype=dtype),
        base_conditions=_tensor(base_conditions, device=selected_device, dtype=dtype),
        surface_tau=_tensor(surface_tau, device=selected_device, dtype=dtype),
        surface_features=_tensor(surface_features, device=selected_device, dtype=dtype),
        surface_conditions=_tensor(
            surface_conditions, device=selected_device, dtype=dtype
        ),
    )
    return TrainingBatch(
        train_ids=train_ids,
        held_out_ids=validation_ids + test_ids,
        observation_experiment_ids=observation_ids,
        observation_tau=_tensor(observation_tau, device=selected_device, dtype=dtype),
        observation_features=_tensor(
            observation_features, device=selected_device, dtype=dtype
        ),
        observation_conditions=_tensor(
            observation_conditions, device=selected_device, dtype=dtype
        ),
        observation_targets=_tensor(
            observation_targets, device=selected_device, dtype=dtype
        ),
        physics=physics,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )


def _validate_model_tensors(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Require a trainable model whose parameters are all finite."""
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("The model must contain trainable parameters.")
    for parameter in parameters:
        if not bool(torch.isfinite(parameter).all().detach().cpu()):
            raise FloatingPointError("Model parameters must be finite before training.")
    return parameters


def _finite_scalar(value: Tensor, label: str) -> None:
    """Fail immediately when a component or aggregate objective is invalid."""
    if value.numel() != 1 or not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{label} must be a finite scalar.")


def _finite_gradients(parameters: tuple[nn.Parameter, ...]) -> None:
    """Reject missing or non-finite trainable gradients before optimizer steps."""
    for index, parameter in enumerate(parameters):
        if parameter.grad is None:
            raise FloatingPointError(f"Trainable parameter {index} has no gradient.")
        if not bool(torch.isfinite(parameter.grad).all().detach().cpu()):
            raise FloatingPointError(
                f"Trainable parameter {index} has a non-finite gradient."
            )


def _check_finite_parameters(parameters: tuple[nn.Parameter, ...]) -> None:
    """Check every optimizer update before a later closure can obscure failure."""
    for index, parameter in enumerate(parameters):
        if not bool(torch.isfinite(parameter).all().detach().cpu()):
            raise FloatingPointError(
                f"Optimizer produced a non-finite parameter {index}."
            )


def _rounded_schedule_steps(fraction: float, epochs: int) -> int:
    """Round fractional stage lengths to nearest integer, with halves upward."""
    return int(math.floor(fraction * epochs + 0.5))


def _schedule_boundaries(config: TrainingConfig) -> dict[str, int]:
    """Resolve fractional warmup/ramp while reserving a full-objective epoch."""
    epochs = config.adam_epochs
    warmup = min(_rounded_schedule_steps(config.warmup_fraction, epochs), epochs - 1)
    ramp = min(
        _rounded_schedule_steps(config.ramp_fraction, epochs),
        max(0, epochs - warmup - 1),
    )
    return {
        "adam_epochs": epochs,
        "warmup_epochs": warmup,
        "ramp_epochs": ramp,
        "full_objective_epochs": epochs - warmup - ramp,
    }


def _physics_ramp_for_epoch(epoch_index: int, schedule: Mapping[str, int]) -> float:
    """Return the deterministic zero-to-one ramp at a zero-based Adam epoch."""
    if epoch_index < schedule["warmup_epochs"]:
        return 0.0
    ramp_epochs = schedule["ramp_epochs"]
    ramp_index = epoch_index - schedule["warmup_epochs"]
    if ramp_epochs > 0 and ramp_index < ramp_epochs:
        return (ramp_index + 1) / ramp_epochs
    return 1.0


def _validate_physics_scales(scales: Mapping[str, float]) -> dict[str, float]:
    """Require one finite, non-amplifying frozen scale for each active residual."""
    if set(scales) != set(_ACTIVE_PHYSICS_TERMS):
        raise ValueError("Physics scales must cover exactly the six active residuals.")
    validated: dict[str, float] = {}
    for name in _ACTIVE_PHYSICS_TERMS:
        value = scales[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 1.0
        ):
            raise ValueError(f"Physics scale for {name} must be finite and at least 1.")
        validated[name] = float(value)
    return validated


def _calibrate_physics_scales(
    model: nn.Module,
    batch: TrainingBatch,
    config: PhysicsConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Freeze residual RMS scales from the current training collocation only."""
    # Autograd is required inside the PDE residual even though this pass never
    # differentiates calibration losses into model parameters.
    with torch.enable_grad():
        residuals = physics_residuals(cast(PhysicsModel, model), batch.physics, config)
        raw_losses = loss_components(residuals)
    calibration_mse: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in _ACTIVE_PHYSICS_TERMS:
        loss = raw_losses[name].detach()
        _finite_scalar(loss, f"{name} calibration MSE")
        value = float(loss.cpu())
        calibration_mse[name] = value
        scales[name] = max(1.0, math.sqrt(value))
    return scales, calibration_mse


def _objective_components(
    model: nn.Module,
    batch: TrainingBatch,
    config: PhysicsConfig,
    training_config: TrainingConfig,
    physics_scales: Mapping[str, float],
    *,
    physics_ramp: float = 1.0,
) -> dict[str, Tensor]:
    """Evaluate data plus the mean frozen-scaled, group-weighted residual loss."""
    if (
        isinstance(physics_ramp, bool)
        or not isinstance(physics_ramp, Real)
        or not math.isfinite(physics_ramp)
        or not 0.0 <= physics_ramp <= 1.0
    ):
        raise ValueError("physics_ramp must be finite and between 0 and 1.")
    selected_scales = _validate_physics_scales(physics_scales)
    predicted_bulk_ratio = (
        bulk_predictions(
            cast(PhysicsModel, model),
            batch.observation_tau,
            batch.observation_features,
            batch.observation_conditions,
            config,
            quadrature_points=training_config.quadrature_points,
        )
        / batch.observation_conditions[:, 2:3]
    )
    data_by_curve: list[Tensor] = []
    for identifier in batch.train_ids:
        positions = [
            index
            for index, observed_id in enumerate(batch.observation_experiment_ids)
            if observed_id == identifier
        ]
        if not positions:
            raise ValueError(f"Training curve {identifier!r} has no observations.")
        indices = torch.as_tensor(
            positions, dtype=torch.long, device=predicted_bulk_ratio.device
        )
        curve_error = predicted_bulk_ratio.index_select(
            0, indices
        ) - batch.observation_targets.index_select(0, indices)
        data_by_curve.append(curve_error.square().mean())
    data_loss = torch.stack(data_by_curve).mean()
    residuals = physics_residuals(cast(PhysicsModel, model), batch.physics, config)
    components = {"data": data_loss, **loss_components(residuals)}
    ramp_tensor = data_loss.new_tensor(float(physics_ramp))
    components["physics_ramp"] = ramp_tensor
    objective_contributions: list[Tensor] = []
    for name in _ACTIVE_PHYSICS_TERMS:
        scale = selected_scales[name]
        scaled_loss = components[name] / (scale * scale)
        group_weight = (
            training_config.physics_weight
            if name in ("physics_mass", "physics_heat")
            else training_config.boundary_weight
        )
        contribution = (
            scaled_loss * group_weight * ramp_tensor / len(_ACTIVE_PHYSICS_TERMS)
        )
        components[f"scale_{name}"] = data_loss.new_tensor(scale)
        components[f"scaled_{name}"] = scaled_loss
        components[f"contribution_{name}"] = contribution
        objective_contributions.append(contribution)
    components["physics_contribution"] = torch.stack(objective_contributions).sum()
    components["total"] = data_loss + components["physics_contribution"]
    for name, value in components.items():
        _finite_scalar(value, f"{name} loss")
    return components


class _DeterministicLBFGSClosure:
    """Callable full-batch closure retaining detached diagnostics per evaluation."""

    def __init__(
        self,
        model: nn.Module,
        batch: TrainingBatch,
        physics_config: PhysicsConfig,
        training_config: TrainingConfig,
        physics_scales: Mapping[str, float] | None = None,
    ) -> None:
        self.model = model
        self.batch = batch
        self.physics_config = physics_config
        self.training_config = training_config
        self.physics_scales = _validate_physics_scales(
            _calibrate_physics_scales(model, batch, physics_config)[0]
            if physics_scales is None
            else physics_scales
        )
        self.parameters = _validate_model_tensors(model)
        self.calls = 0
        self.last_components: dict[str, Tensor] = {}
        self.losses: list[dict[str, float]] = []
        self.validation_monitor: Callable[[], Tensor] | None = None
        # Current Strategy C networks have no dropout or batch normalization.
        # Evaluation mode also prevents stochastic behavior if the model is
        # extended later, while autograd remains fully enabled.
        self.model.eval()

    def __call__(self) -> Tensor:
        """Recompute one identical objective and its gradients for L-BFGS."""
        self.model.zero_grad(set_to_none=True)
        components = _objective_components(
            self.model,
            self.batch,
            self.physics_config,
            self.training_config,
            self.physics_scales,
        )
        objective = components["total"]
        _finite_scalar(objective, "L-BFGS objective")
        objective.backward()
        _finite_gradients(self.parameters)
        self.calls += 1
        self.last_components = {
            name: value.detach().clone() for name, value in components.items()
        }
        snapshot = _as_float_losses(components)
        if self.validation_monitor is not None:
            snapshot["validation_data"] = float(self.validation_monitor().cpu())
        self.losses.append(snapshot)
        return objective


def make_lbfgs_closure(
    model: nn.Module,
    batch: TrainingBatch,
    physics_config: PhysicsConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    physics_scales: Mapping[str, float] | None = None,
) -> _DeterministicLBFGSClosure:
    """Create a fixed-data L-BFGS closure with finite objective/gradient checks."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(batch, TrainingBatch):
        raise TypeError("batch must be a TrainingBatch instance.")
    selected_physics = PhysicsConfig() if physics_config is None else physics_config
    selected_training = TrainingConfig() if training_config is None else training_config
    if not isinstance(selected_physics, PhysicsConfig):
        raise TypeError("physics_config must be a PhysicsConfig instance.")
    if not isinstance(selected_training, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig instance.")
    return _DeterministicLBFGSClosure(
        model, batch, selected_physics, selected_training, physics_scales
    )


def _seed_process(seed: int) -> None:
    """Seed supported framework generators before any internally made model."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_float_losses(components: Mapping[str, Tensor]) -> dict[str, float]:
    """Store finite, JSON-friendly loss snapshots without retaining graphs."""
    snapshot = {name: float(value.detach().cpu()) for name, value in components.items()}
    if not all(math.isfinite(value) for value in snapshot.values()):
        raise FloatingPointError("Training loss history contains a non-finite value.")
    return snapshot


def _make_validation_monitor(
    model: nn.Module,
    data: CurveData,
    split: ExperimentSplit,
    feature_transform: FeatureTransform,
    config: PhysicsConfig,
    quadrature_points: int,
) -> Callable[[], Tensor] | None:
    """Prepare validation-only moisture scoring outside the fitted objective.

    Monitoring uses the already train-fitted feature transform and the same
    quadrature/normalization as training. No temperature targets exist. Empty
    validation partitions produce no invented history; each validation curve
    must have an available, in-domain moisture target. Final-test IDs are never
    iterated here or read from the observation table during optimization.
    """
    if not split.validation_ids:
        return None
    parameter = next(model.parameters())
    experiments = data.experiments.set_index("experiment_id")
    curves: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for identifier in split.validation_ids:
        rows = data.observations.loc[
            data.observations["experiment_id"].eq(identifier)
            & data.observations["target_available"].fillna(False).astype(bool)
        ]
        values = rows.loc[:, ["time_s", "moisture_normalized"]].to_numpy(dtype=float)
        if (
            len(values) == 0
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values[:, 0] > config.time_scale_s).any()
        ):
            raise ValueError(
                f"Validation curve {identifier!r} requires finite, nonnegative, "
                "available targets within the modeled interval."
            )
        experiment = experiments.loc[[identifier]]
        features = feature_transform.transform(
            experiment.loc[:, feature_transform.feature_names]
        )
        conditions = experiment.loc[:, _PHYSICAL_CONDITION_COLUMNS].to_numpy(
            dtype=float
        )
        arrays = (
            values[:, 0:1] / config.time_scale_s,
            np.repeat(features, len(values), axis=0),
            np.repeat(conditions, len(values), axis=0),
            values[:, 1:2],
        )
        tau, feature_rows, condition_rows, targets = (
            torch.as_tensor(array, device=parameter.device, dtype=parameter.dtype)
            for array in arrays
        )
        curves.append((tau, feature_rows, condition_rows, targets))

    def monitor() -> Tensor:
        # Never backpropagate validation errors, select checkpoints, or stop.
        with torch.no_grad():
            errors = []
            for tau, features, conditions, targets in curves:
                predicted = (
                    bulk_predictions(
                        cast(PhysicsModel, model),
                        tau,
                        features,
                        conditions,
                        config,
                        quadrature_points=quadrature_points,
                    )
                    / conditions[:, 2:3]
                )
                errors.append((predicted - targets).square().mean())
            loss = torch.stack(errors).mean()
            _finite_scalar(loss, "Validation moisture MSE")
            return loss

    return monitor


def build_adam_optimizer(
    model: nn.Module, training_config: TrainingConfig
) -> torch.optim.Adam:
    """Create Adam with a separate learning rate for diffusivity-law parameters.

    The trainable parameters named in ``_DIFFUSIVITY_LAW_PARAMETER_NAMES`` are
    selected by object identity and placed in a group at
    ``physics_parameter_learning_rate``; every other trainable parameter uses
    ``adam_learning_rate``. The law group comes first so the flattened group
    order matches ``model.parameters()`` for CoupledInversePINN, where the
    law scalars are direct parameters (yielded before submodule weights).
    Models without any law attribute
    (e.g. test doubles) receive a single group. Frozen parameters are omitted,
    and each trainable parameter appears in exactly one group.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig instance.")
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise ValueError("The model must contain trainable parameters.")
    trainable_ids = {id(parameter) for parameter in trainable}
    law_ids: set[int] = set()
    for name in _DIFFUSIVITY_LAW_PARAMETER_NAMES:
        candidate = getattr(model, name, None)
        # Only registered, trainable Parameters join the law group; anything
        # else would either be absent from Adam or duplicated across groups.
        if isinstance(candidate, nn.Parameter) and id(candidate) in trainable_ids:
            law_ids.add(id(candidate))
    law_parameters = [parameter for parameter in trainable if id(parameter) in law_ids]
    network_parameters = [
        parameter for parameter in trainable if id(parameter) not in law_ids
    ]
    groups: list[dict[str, object]] = []
    if law_parameters:
        groups.append(
            {
                "params": law_parameters,
                "lr": training_config.physics_parameter_learning_rate,
            }
        )
    if network_parameters:
        groups.append(
            {"params": network_parameters, "lr": training_config.adam_learning_rate}
        )
    # Looked up on torch.optim at call time so instrumentation can wrap Adam.
    return torch.optim.Adam(groups, lr=training_config.adam_learning_rate)


def train_sequentially(
    model: nn.Module | None,
    data: CurveData,
    split: ExperimentSplit,
    feature_transform: FeatureTransform,
    *,
    sampling_config: SamplingConfig | None = None,
    training_config: TrainingConfig | None = None,
    physics_config: PhysicsConfig | None = None,
) -> TrainingResult:
    """Run staged full-batch Adam and continue on the same model with L-BFGS.

    Adam uses the configured epoch count at 0.001 for network weights and
    0.05 for the three learned diffusivity-law parameters (see
    ``build_adam_optimizer``). Its data-only warmup is followed by a linear
    ramp of six calibrated PDE/boundary residuals to the full objective.
    Frozen scales are fitted once from training collocation after warmup and
    remain unchanged through subsequent bounded L-BFGS. That optimizer
    receives the *same* model object and parameter instances; no checkpoint
    reset, reinitialization, split resampling, or held-out fitting occurs
    between stages.
    """
    selected_training = TrainingConfig() if training_config is None else training_config
    selected_sampling = SamplingConfig() if sampling_config is None else sampling_config
    selected_physics = PhysicsConfig() if physics_config is None else physics_config
    if not isinstance(selected_training, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig instance.")
    if not isinstance(selected_sampling, SamplingConfig):
        raise TypeError("sampling_config must be a SamplingConfig instance.")
    if not isinstance(selected_physics, PhysicsConfig):
        raise TypeError("physics_config must be a PhysicsConfig instance.")
    _seed_process(int(selected_training.seed))
    selected_device = _resolve_device(selected_training.device)
    if model is None:
        model = CoupledInversePINN(selected_physics)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module or None.")
    if hasattr(model, "config") and model.config != selected_physics:
        raise ValueError("The model and training must use the same PhysicsConfig.")
    model.to(device=selected_device, dtype=selected_training.dtype)
    model.eval()
    parameters = _validate_model_tensors(model)
    batch = sample_training_batch(
        data,
        split,
        feature_transform,
        selected_sampling,
        selected_physics,
        device=selected_device,
        dtype=selected_training.dtype,
    )
    validation_monitor = _make_validation_monitor(
        model,
        data,
        split,
        feature_transform,
        selected_physics,
        selected_training.quadrature_points,
    )

    schedule = _schedule_boundaries(selected_training)
    unit_scales = {name: 1.0 for name in _ACTIVE_PHYSICS_TERMS}
    physics_scales: dict[str, float] | None = None
    physics_calibration_mse: dict[str, float] | None = None

    def apply_frozen_scales_to_history(
        history: StageHistory, frozen_scales: Mapping[str, float]
    ) -> None:
        """Record the same post-warmup scales beside every Adam snapshot."""
        for snapshot in history.losses:
            for name in _ACTIVE_PHYSICS_TERMS:
                scale = frozen_scales[name]
                snapshot[f"scale_{name}"] = scale
                snapshot[f"scaled_{name}"] = snapshot[name] / (scale * scale)

    adam_history = StageHistory()
    adam = build_adam_optimizer(model, selected_training)
    adam_started = time.perf_counter()
    for epoch_index in range(selected_training.adam_epochs):
        if epoch_index == schedule["warmup_epochs"]:
            # Scales are fit once, after data-only warmup, exclusively from the
            # fixed training collocation. Validation/test data are never read.
            physics_scales, physics_calibration_mse = _calibrate_physics_scales(
                model, batch, selected_physics
            )
            apply_frozen_scales_to_history(adam_history, physics_scales)
        active_scales = unit_scales if physics_scales is None else physics_scales
        physics_ramp = _physics_ramp_for_epoch(epoch_index, schedule)
        adam.zero_grad(set_to_none=True)
        components = _objective_components(
            model,
            batch,
            selected_physics,
            selected_training,
            active_scales,
            physics_ramp=physics_ramp,
        )
        # Validation refers to the same pre-update Adam state, is detached,
        # and is deliberately absent from components["total"].
        if validation_monitor is not None:
            components["validation_data"] = validation_monitor()
        objective = components["total"]
        _finite_scalar(objective, "Adam objective")
        objective.backward()
        _finite_gradients(parameters)
        adam.step()
        _check_finite_parameters(parameters)
        adam_history.losses.append(_as_float_losses(components))
        adam_history.epochs += 1
    if physics_scales is None or physics_calibration_mse is None:
        raise RuntimeError("The Adam schedule did not reach physics-scale calibration.")
    apply_frozen_scales_to_history(adam_history, physics_scales)
    adam_history.elapsed_seconds = time.perf_counter() - adam_started

    # Constructing the second optimizer over the same parameter references is
    # deliberate: L-BFGS starts from the post-Adam state, including both
    # trainable inverse-physics scalars.
    lbfgs_history = StageHistory()
    closure = make_lbfgs_closure(
        model,
        batch,
        selected_physics,
        selected_training,
        physics_scales=physics_scales,
    )
    closure.validation_monitor = validation_monitor
    optimizer = torch.optim.LBFGS(
        parameters,
        max_iter=selected_training.lbfgs_max_iter,
        max_eval=selected_training.lbfgs_max_eval,
        history_size=selected_training.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    lbfgs_started = time.perf_counter()
    optimizer.step(closure)
    lbfgs_history.elapsed_seconds = time.perf_counter() - lbfgs_started
    lbfgs_history.closure_calls = closure.calls
    optimizer_state = optimizer.state.get(parameters[0], {})
    lbfgs_history.iterations = int(optimizer_state.get("n_iter", 0))
    lbfgs_history.losses.extend(closure.losses)
    _check_finite_parameters(parameters)
    if not lbfgs_history.losses:
        raise RuntimeError("L-BFGS completed without evaluating its closure.")
    # L-BFGS closure snapshots can be line-search trials rather than accepted
    # states. Re-evaluate the exact fixed objective and gradient at the
    # parameters that are actually returned to the caller.
    model.zero_grad(set_to_none=True)
    accepted_components = _objective_components(
        model,
        batch,
        selected_physics,
        selected_training,
        physics_scales,
    )
    accepted_objective = accepted_components["total"]
    _finite_scalar(accepted_objective, "Accepted L-BFGS objective")
    accepted_objective.backward()
    _finite_gradients(parameters)
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().square().sum().cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
    )
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("Accepted L-BFGS gradient norm must be finite.")
    lbfgs_history.accepted_losses = _as_float_losses(accepted_components)
    if validation_monitor is not None:
        lbfgs_history.accepted_losses["validation_data"] = float(
            validation_monitor().cpu()
        )
    lbfgs_history.accepted_gradient_norm = gradient_norm
    function_evaluations = int(optimizer_state.get("func_evals", closure.calls))
    lbfgs_history.termination_flags = {
        "iteration_limit_reached": (
            lbfgs_history.iterations >= selected_training.lbfgs_max_iter
        ),
        "evaluation_limit_reached": (
            function_evaluations >= selected_training.lbfgs_max_eval
        ),
    }
    provenance = TrainingProvenance(
        seed=int(selected_training.seed),
        train_ids=batch.train_ids,
        held_out_ids=batch.held_out_ids,
        training_observation_count=len(batch.observation_experiment_ids),
        collocation_point_count=(
            batch.physics.domain_tau.shape[0]
            + batch.physics.initial_xi.shape[0]
            + batch.physics.base_tau.shape[0]
            + batch.physics.surface_tau.shape[0]
        ),
        device=str(selected_device),
        dtype=str(selected_training.dtype),
        validation_ids=batch.validation_ids,
        test_ids=batch.test_ids,
    )
    return TrainingResult(
        model=model,
        stage_history={"adam": adam_history, "lbfgs": lbfgs_history},
        provenance=provenance,
        physics_scales=physics_scales,
        physics_calibration_mse=physics_calibration_mse,
        schedule=schedule,
    )


__all__ = [
    "SamplingConfig",
    "StageHistory",
    "TrainingBatch",
    "TrainingConfig",
    "TrainingProvenance",
    "TrainingResult",
    "build_adam_optimizer",
    "make_lbfgs_closure",
    "sample_training_batch",
    "train_sequentially",
]
