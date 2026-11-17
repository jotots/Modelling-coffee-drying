"""Validated preparation of measured coffee-drying curves.

The notebook remains responsible for workbook parsing. This module accepts its
DataFrame results, retains every scheduled observation for auditability, and
derives supervision exclusively from measured sample masses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd


_EXPERIMENT_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "condition_id",
    "air_speed_m_s",
    "particle_diameter_m",
    "dry_mass_kg",
    "initial_sample_mass_kg",
)
_OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "time_s",
    "sample_mass_kg",
    "target_available",
)
_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "air_speed_m_s",
    "particle_diameter_m",
    "initial_moisture_db",
    "dry_bulk_density_kg_m3",
)


@dataclass(frozen=True)
class Scales:
    """Physical scales shared by prepared curves and subsequent PINN stages."""

    bed_depth_m: float = 0.01
    dish_diameter_m: float = 0.05
    time_scale_s: float = 14_400.0
    initial_temperature_K: float = 298.15
    temperature_difference_K: float = 45.0

    def __post_init__(self) -> None:
        """Reject invalid physical scales before they enter normalized data."""
        for field_name, value in (
            ("bed_depth_m", self.bed_depth_m),
            ("dish_diameter_m", self.dish_diameter_m),
            ("time_scale_s", self.time_scale_s),
            ("initial_temperature_K", self.initial_temperature_K),
            ("temperature_difference_K", self.temperature_difference_K),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive.")


@dataclass(frozen=True)
class CurveData:
    """Prepared metadata and scheduled mass observations for every experiment."""

    experiments: pd.DataFrame
    observations: pd.DataFrame


@dataclass(frozen=True)
class ExperimentSplit:
    """Immutable experiment identifiers partitioned by complete drying curve."""

    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Guarantee that identifiers occur once and never cross partitions."""
        if len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError("Training experiment identifiers must be unique.")
        if len(set(self.test_ids)) != len(self.test_ids):
            raise ValueError("Test experiment identifiers must be unique.")
        if set(self.train_ids).intersection(self.test_ids):
            raise ValueError("Training and test experiment identifiers must be disjoint.")


@dataclass(frozen=True)
class FeatureTransform:
    """Train-fitted standardization for the four experiment-level conditions."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        """Keep the transform self-consistent and safe to apply to new curves."""
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("Feature names must be nonempty and unique.")
        if len(self.mean) != len(self.feature_names) or len(self.scale) != len(
            self.feature_names
        ):
            raise ValueError("Feature statistics must match the feature names.")
        if not np.isfinite(self.mean).all():
            raise ValueError("Feature means must be finite.")
        if not np.isfinite(self.scale).all() or any(value <= 0 for value in self.scale):
            raise ValueError("Feature scales must be finite and positive.")

    def transform(self, experiments: pd.DataFrame) -> np.ndarray:
        """Return standardized features without fitting on the supplied rows."""
        missing = [name for name in self.feature_names if name not in experiments]
        if missing:
            raise ValueError(f"Experiment features are missing columns: {missing}.")
        values = experiments.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Experiment features must be finite before transformation.")
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    """Report absent parser outputs at the boundary with a useful message."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _valid_source_flags(values: pd.Series) -> pd.Series:
    """Interpret parser availability flags strictly instead of truthiness casting."""
    def parse(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        raise ValueError("target_available values must be boolean, 0/1, or missing.")

    return values.map(parse).astype(bool)


def prepare_curve_data(
    experiments: pd.DataFrame,
    observations: pd.DataFrame,
    scales: Scales,
) -> CurveData:
    """Derive mass-based targets and preserve invalid rows for transparent audit.

    A valid quality flag such as ``mass_increase`` is descriptive, not a reason
    to discard a measured mass. Invalid masses stay in the returned table, but
    their target flag is false and their derived moisture is missing.
    """
    if not isinstance(experiments, pd.DataFrame) or not isinstance(
        observations, pd.DataFrame
    ):
        raise TypeError("experiments and observations must be pandas DataFrames.")
    if not isinstance(scales, Scales):
        raise TypeError("scales must be a Scales instance.")
    _require_columns(experiments, _EXPERIMENT_COLUMNS, "experiments")
    _require_columns(observations, _OBSERVATION_COLUMNS, "observations")
    if experiments.empty:
        raise ValueError("experiments must contain at least one drying curve.")
    if experiments["experiment_id"].isna().any() or experiments[
        "experiment_id"
    ].duplicated().any():
        raise ValueError("Experiment identifiers must be present and unique.")
    if experiments["condition_id"].isna().any():
        raise ValueError("Every experiment must have a condition_id.")

    prepared_experiments = experiments.copy(deep=True)
    for column in (
        "air_speed_m_s",
        "particle_diameter_m",
        "dry_mass_kg",
        "initial_sample_mass_kg",
    ):
        try:
            prepared_experiments[column] = pd.to_numeric(
                prepared_experiments[column], errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain numeric metadata.") from error
        if not np.isfinite(prepared_experiments[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite metadata.")
    if (prepared_experiments["air_speed_m_s"] < 0).any():
        raise ValueError("air_speed_m_s cannot be negative.")
    if (prepared_experiments["particle_diameter_m"] <= 0).any():
        raise ValueError("particle_diameter_m must be positive.")
    if (prepared_experiments["dry_mass_kg"] <= 0).any():
        raise ValueError("dry_mass_kg must be positive.")
    if (
        prepared_experiments["initial_sample_mass_kg"]
        <= prepared_experiments["dry_mass_kg"]
    ).any():
        raise ValueError("Initial sample mass must exceed dry mass for X0 > 0.")

    unknown_ids = set(observations["experiment_id"].dropna()).difference(
        prepared_experiments["experiment_id"]
    )
    if unknown_ids:
        raise ValueError(f"Observations reference unknown experiment IDs: {unknown_ids}.")
    if observations["experiment_id"].isna().any():
        raise ValueError("Every observation must reference an experiment_id.")

    prepared_observations = observations.copy(deep=True)
    try:
        prepared_observations["time_s"] = pd.to_numeric(
            prepared_observations["time_s"], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("time_s must contain numeric values or missing values.") from error
    time_values = prepared_observations["time_s"].to_numpy(dtype=float)
    if np.isinf(time_values).any() or (prepared_observations["time_s"].dropna() < 0).any():
        raise ValueError("time_s values must be finite and nonnegative.")
    if prepared_observations.duplicated(["experiment_id", "time_s"]).any():
        raise ValueError("Observation times must be unique within each experiment.")
    available_from_source = _valid_source_flags(
        prepared_observations["target_available"]
    )
    try:
        sample_mass = pd.to_numeric(
            prepared_observations["sample_mass_kg"], errors="coerce"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("sample_mass_kg values must be numeric or missing.") from error

    curve_fields = prepared_experiments.set_index("experiment_id")
    dry_mass = prepared_observations["experiment_id"].map(
        curve_fields["dry_mass_kg"]
    ).astype(float)
    initial_sample_mass = prepared_observations["experiment_id"].map(
        curve_fields["initial_sample_mass_kg"]
    ).astype(float)
    initial_moisture = (
        initial_sample_mass - dry_mass
    ) / dry_mass
    valid_mass = sample_mass.notna() & np.isfinite(sample_mass) & (sample_mass >= dry_mass)
    valid_time = prepared_observations["time_s"].notna()
    target_available = available_from_source & valid_mass & valid_time
    available_by_curve = pd.Series(
        target_available.to_numpy(),
        index=prepared_observations["experiment_id"],
    ).groupby(level=0).any()
    curves_without_targets = [
        identifier
        for identifier in prepared_experiments["experiment_id"]
        if not bool(available_by_curve.get(identifier, False))
    ]
    if curves_without_targets:
        raise ValueError(
            "Each experiment must have at least one available mass target; "
            f"missing targets for {curves_without_targets}."
        )

    # Keep physical calculations separate from source availability: every
    # physically valid measurement is inspectable even when the parser flags it
    # unavailable, while only explicitly available rows can supervise a model.
    prepared_observations["moisture_db"] = np.where(
        valid_mass, (sample_mass - dry_mass) / dry_mass, np.nan
    )
    prepared_observations["initial_moisture_db"] = initial_moisture
    prepared_observations["moisture_normalized"] = (
        prepared_observations["moisture_db"] / initial_moisture
    )
    prepared_observations["time_normalized"] = (
        prepared_observations["time_s"] / scales.time_scale_s
    )
    prepared_observations["target_available"] = target_available.astype(bool)

    reasons = np.full(len(prepared_observations), "", dtype=object)
    reasons[~valid_time.to_numpy()] = "missing_time"
    reasons[(sample_mass.isna() | ~np.isfinite(sample_mass)).to_numpy()] = (
        "missing_or_nonfinite_sample_mass"
    )
    reasons[(sample_mass.notna() & (sample_mass < dry_mass)).to_numpy()] = (
        "sample_mass_below_dry_mass"
    )
    reasons[(valid_mass & valid_time & ~available_from_source).to_numpy()] = (
        "source_target_unavailable"
    )
    prepared_observations["invalid_reason"] = reasons

    area = math.pi * (scales.dish_diameter_m / 2.0) ** 2
    volume = area * scales.bed_depth_m
    prepared_experiments["initial_moisture_db"] = (
        prepared_experiments["initial_sample_mass_kg"]
        - prepared_experiments["dry_mass_kg"]
    ) / prepared_experiments["dry_mass_kg"]
    prepared_experiments["dry_bulk_density_kg_m3"] = (
        prepared_experiments["dry_mass_kg"] / volume
    )
    return CurveData(prepared_experiments, prepared_observations)


def split_experiments(experiments: pd.DataFrame, seed: int) -> ExperimentSplit:
    """Hold out one of exactly three complete replicate curves per condition."""
    _require_columns(experiments, ("experiment_id", "condition_id"), "experiments")
    if experiments.empty:
        raise ValueError("experiments must contain at least one drying curve.")
    if experiments["experiment_id"].isna().any() or experiments[
        "experiment_id"
    ].duplicated().any():
        raise ValueError("Experiment identifiers must be present and unique.")
    if experiments["condition_id"].isna().any():
        raise ValueError("Every experiment must have a condition_id.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    groups = list(experiments.groupby("condition_id", sort=True, dropna=False))
    for condition, group in groups:
        if len(group) != 3:
            raise ValueError(
                f"Condition {condition!r} must have exactly three replicates; "
                f"found {len(group)}."
            )

    # A single seeded generator makes the held-out replicate reproducible while
    # the complete condition group remains represented in both partitions.
    generator = np.random.default_rng(int(seed))
    train_ids: list[str] = []
    test_ids: list[str] = []
    for _, group in groups:
        ids = tuple(str(value) for value in group["experiment_id"].tolist())
        held_out_index = int(generator.integers(0, len(ids)))
        test_ids.append(ids[held_out_index])
        train_ids.extend(
            identifier for index, identifier in enumerate(ids) if index != held_out_index
        )
    return ExperimentSplit(tuple(train_ids), tuple(test_ids))


def fit_feature_transform(
    data: CurveData, split: ExperimentSplit
) -> FeatureTransform:
    """Fit feature statistics using only the training experiment metadata."""
    if not isinstance(data, CurveData) or not isinstance(split, ExperimentSplit):
        raise TypeError("data and split must be CurveData and ExperimentSplit instances.")
    _require_columns(data.experiments, _FEATURE_COLUMNS + ("experiment_id",), "data")
    known_ids = set(data.experiments["experiment_id"].astype(str))
    train_ids = set(split.train_ids)
    test_ids = set(split.test_ids)
    if not train_ids:
        raise ValueError("At least one training experiment is required.")
    if train_ids.intersection(test_ids):
        raise ValueError("Training and test experiment identifiers must be disjoint.")
    if not train_ids.issubset(known_ids) or not test_ids.issubset(known_ids):
        raise ValueError("The split contains experiment identifiers absent from data.")
    if train_ids.union(test_ids) != known_ids:
        raise ValueError("The split must assign every experiment to train or test.")
    training = data.experiments.loc[
        data.experiments["experiment_id"].astype(str).isin(train_ids),
        _FEATURE_COLUMNS,
    ].to_numpy(dtype=float)
    if not np.isfinite(training).all():
        raise ValueError("Training features must be finite.")
    means = training.mean(axis=0)
    standard_deviations = training.std(axis=0)
    # Constant features remain usable and map to zero rather than dividing by
    # zero; the replacement is learned exclusively from training rows.
    standard_deviations[standard_deviations == 0] = 1.0
    return FeatureTransform(
        feature_names=_FEATURE_COLUMNS,
        mean=tuple(float(value) for value in means),
        scale=tuple(float(value) for value in standard_deviations),
    )
