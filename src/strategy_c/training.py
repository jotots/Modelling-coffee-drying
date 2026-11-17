"""Reproducible, sequential Adam/L-BFGS training for Strategy C.

All fitting is performed on complete training curves only. The observation
batch and every collocation coordinate are materialized before optimization,
so both Adam and every L-BFGS closure evaluate the same deterministic,
dimensionless objective. Held-out curve data are excluded at the sampling
boundary rather than masked later in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
import time
from typing import Final, Mapping, cast

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
                or value < 1
                or value > 100_000
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
    """Bounded settings for the single Adam-then-L-BFGS optimization run."""

    adam_epochs: int = 50
    adam_learning_rate: float = 0.001
    lbfgs_max_iter: int = 20
    lbfgs_max_eval: int = 40
    lbfgs_history_size: int = 50
    seed: int = 0
    device: str = "auto"
    dtype: torch.dtype = torch.float64
    physics_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    quadrature_points: int = 32

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
                or value < minimum
                or value > maximum
            ):
                raise ValueError(
                    f"{name} must be an integer from {minimum} through {maximum}."
                )
        if self.lbfgs_max_eval < self.lbfgs_max_iter:
            raise ValueError("lbfgs_max_eval must be at least lbfgs_max_iter.")
        if (
            isinstance(self.adam_learning_rate, bool)
            or not isinstance(self.adam_learning_rate, Real)
            or not math.isfinite(self.adam_learning_rate)
            or self.adam_learning_rate <= 0
        ):
            raise ValueError("adam_learning_rate must be finite and positive.")
        for name in ("physics_weight", "initial_weight", "boundary_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
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
    """Explicit record of the curves and coordinates used for optimization."""

    seed: int
    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    training_observation_count: int
    collocation_point_count: int
    device: str
    dtype: str


@dataclass
class StageHistory:
    """Loss snapshots and timing/evaluation counts for one optimizer stage."""

    epochs: int = 0
    iterations: int = 0
    closure_calls: int = 0
    elapsed_seconds: float = 0.0
    losses: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingResult:
    """Trained model with stage histories and data-use provenance."""

    model: nn.Module
    stage_history: Mapping[str, StageHistory]
    provenance: TrainingProvenance


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
    held_out = set(split.test_ids)
    if not training:
        raise ValueError("At least one training experiment is required.")
    if training.intersection(held_out):
        raise ValueError("Training and held-out IDs must be disjoint.")
    if training.union(held_out) != known:
        raise ValueError("The split must cover every prepared curve exactly once.")
    conditions = data.experiments.loc[:, _PHYSICAL_CONDITION_COLUMNS].to_numpy(
        dtype=float
    )
    if not np.isfinite(conditions).all() or (conditions <= 0).any():
        raise ValueError("Prepared physical conditions must be finite and positive.")
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
    held_out_ids = tuple(str(identifier) for identifier in split.test_ids)
    train_experiments = experiments.loc[
        experiments["_identifier"].isin(train_ids)
    ].copy()
    if len(train_experiments) != len(train_ids):
        raise ValueError("Every training identifier must resolve to one experiment.")
    # This application deliberately transforms training rows alone. Test
    # features are not read, transformed, or used in any optimizer computation.
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
        held_out_ids=held_out_ids,
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


def _objective_components(
    model: nn.Module,
    batch: TrainingBatch,
    config: PhysicsConfig,
    training_config: TrainingConfig,
) -> dict[str, Tensor]:
    """Evaluate equal-curve data MSE and each separately named physics MSE."""
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
    physics_loss = components["physics_mass"] + components["physics_heat"]
    initial_loss = components["initial_moisture"] + components["initial_temperature"]
    boundary_loss = (
        components["base_moisture"]
        + components["base_temperature"]
        + components["surface_mass"]
        + components["surface_heat"]
    )
    components["total"] = (
        data_loss
        + training_config.physics_weight * physics_loss
        + training_config.initial_weight * initial_loss
        + training_config.boundary_weight * boundary_loss
    )
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
    ) -> None:
        self.model = model
        self.batch = batch
        self.physics_config = physics_config
        self.training_config = training_config
        self.parameters = _validate_model_tensors(model)
        self.calls = 0
        self.last_components: dict[str, Tensor] = {}
        self.losses: list[dict[str, float]] = []
        # Current Strategy C networks have no dropout or batch normalization.
        # Evaluation mode also prevents stochastic behavior if the model is
        # extended later, while autograd remains fully enabled.
        self.model.eval()

    def __call__(self) -> Tensor:
        """Recompute one identical objective and its gradients for L-BFGS."""
        self.model.zero_grad(set_to_none=True)
        components = _objective_components(
            self.model, self.batch, self.physics_config, self.training_config
        )
        objective = components["total"]
        _finite_scalar(objective, "L-BFGS objective")
        objective.backward()
        _finite_gradients(self.parameters)
        self.calls += 1
        self.last_components = {
            name: value.detach().clone() for name, value in components.items()
        }
        self.losses.append(
            {name: float(value.detach().cpu()) for name, value in components.items()}
        )
        return objective


def make_lbfgs_closure(
    model: nn.Module,
    batch: TrainingBatch,
    physics_config: PhysicsConfig | None = None,
    training_config: TrainingConfig | None = None,
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
    return _DeterministicLBFGSClosure(model, batch, selected_physics, selected_training)


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
    """Run one full-batch Adam stage then continue on the exact same model.

    Adam defaults to the approved 50 epochs at 0.001. The subsequent bounded
    L-BFGS optimizer receives the *same* model object and parameter instances;
    no checkpoint reset, reinitialization, split resampling, or held-out fitting
    occurs between stages.
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

    adam_history = StageHistory()
    adam = torch.optim.Adam(parameters, lr=selected_training.adam_learning_rate)
    adam_started = time.perf_counter()
    for _ in range(selected_training.adam_epochs):
        adam.zero_grad(set_to_none=True)
        components = _objective_components(
            model, batch, selected_physics, selected_training
        )
        objective = components["total"]
        _finite_scalar(objective, "Adam objective")
        objective.backward()
        _finite_gradients(parameters)
        adam.step()
        _check_finite_parameters(parameters)
        adam_history.losses.append(_as_float_losses(components))
        adam_history.epochs += 1
    adam_history.elapsed_seconds = time.perf_counter() - adam_started

    # Constructing the second optimizer over the same parameter references is
    # deliberate: L-BFGS starts from the post-Adam state, including both
    # trainable inverse-physics scalars.
    lbfgs_history = StageHistory()
    closure = make_lbfgs_closure(model, batch, selected_physics, selected_training)
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
    lbfgs_history.iterations = int(optimizer.state[parameters[0]].get("n_iter", 0))
    lbfgs_history.losses.extend(closure.losses)
    _check_finite_parameters(parameters)
    if not lbfgs_history.losses:
        raise RuntimeError("L-BFGS completed without evaluating its closure.")
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
    )
    return TrainingResult(
        model=model,
        stage_history={"adam": adam_history, "lbfgs": lbfgs_history},
        provenance=provenance,
    )


__all__ = [
    "SamplingConfig",
    "StageHistory",
    "TrainingBatch",
    "TrainingConfig",
    "TrainingProvenance",
    "TrainingResult",
    "make_lbfgs_closure",
    "sample_training_batch",
    "train_sequentially",
]
