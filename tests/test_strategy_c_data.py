"""Behavioral tests for Strategy C's curve preparation and grouped split."""

from dataclasses import FrozenInstanceError
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# The repository uses a source layout without an installed package manifest.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

_data_module = importlib.import_module("strategy_c.data")
Scales = _data_module.Scales
fit_feature_transform = _data_module.fit_feature_transform
prepare_curve_data = _data_module.prepare_curve_data
split_experiments = _data_module.split_experiments


def _experiment_rows(condition_count: int = 2) -> pd.DataFrame:
    """Create three independent replicates for each drying condition."""
    rows = []
    for condition_index in range(condition_count):
        for replicate in (1, 2, 3):
            experiment_id = f"condition_{condition_index}_replicate_{replicate}"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "condition_id": f"condition_{condition_index}",
                    "replicate": replicate,
                    "air_speed_m_s": 0.4 + condition_index * 0.2,
                    "particle_diameter_m": 450e-6,
                    "dry_mass_kg": 0.010 + replicate * 0.001,
                    "initial_sample_mass_kg": 0.020 + replicate * 0.001,
                }
            )
    return pd.DataFrame(rows)


def _observation_rows(
    experiments: pd.DataFrame,
    *,
    initial_mass_offset_kg: float = 0.0,
) -> pd.DataFrame:
    """Build measured mass observations; tx is intentionally untrustworthy."""
    rows = []
    for row in experiments.itertuples(index=False):
        initial_mass = row.initial_sample_mass_kg + initial_mass_offset_kg
        rows.extend(
            [
                {
                    "experiment_id": row.experiment_id,
                    "time_s": 0.0,
                    "sample_mass_kg": initial_mass,
                    "target_available": True,
                    "quality_flags": "",
                    "tx": -99999.0,
                },
                {
                    "experiment_id": row.experiment_id,
                    "time_s": 3600.0,
                    "sample_mass_kg": initial_mass + 0.001,
                    "target_available": True,
                    "quality_flags": "mass_increase",
                    "tx": 99999.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_preparation_computes_dry_basis_and_initial_normalization() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:1].copy()
    observations = _observation_rows(experiments)

    data = prepare_curve_data(experiments, observations, Scales())

    curve = data.experiments.iloc[0]
    samples = data.observations.sort_values("time_s")
    expected_initial_x = (
        experiments.iloc[0]["initial_sample_mass_kg"]
        - experiments.iloc[0]["dry_mass_kg"]
    ) / experiments.iloc[0]["dry_mass_kg"]
    expected_x = (
        samples["sample_mass_kg"] - experiments.iloc[0]["dry_mass_kg"]
    ) / experiments.iloc[0]["dry_mass_kg"]
    assert curve["initial_moisture_db"] == pytest.approx(expected_initial_x)
    np.testing.assert_allclose(samples["moisture_db"], expected_x)
    np.testing.assert_allclose(
        samples["moisture_normalized"], expected_x / expected_initial_x
    )
    assert samples["time_normalized"].tolist() == pytest.approx([0.0, 0.25])


def test_dry_density_is_calculated_independently_for_each_curve() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:2].copy()
    observations = _observation_rows(experiments)

    data = prepare_curve_data(experiments, observations, Scales())

    expected = experiments["dry_mass_kg"].to_numpy() / (
        np.pi * 0.025**2 * 0.01
    )
    np.testing.assert_allclose(data.experiments["dry_bulk_density_kg_m3"], expected)
    assert data.experiments["dry_bulk_density_kg_m3"].nunique() == 2


def test_valid_mass_increase_is_kept_as_an_available_measurement() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:1].copy()
    observations = _observation_rows(experiments)

    data = prepare_curve_data(experiments, observations, Scales())

    increased = data.observations.loc[data.observations["quality_flags"].eq("mass_increase")]
    assert len(increased) == 1
    assert bool(increased.iloc[0]["target_available"])
    assert increased.iloc[0]["moisture_db"] > data.observations.iloc[0]["moisture_db"]


def test_invalid_observations_are_preserved_but_not_targets() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:1].copy()
    observations = _observation_rows(experiments)
    invalid_rows = pd.DataFrame(
        [
            {
                "experiment_id": experiments.iloc[0]["experiment_id"],
                "time_s": 7200.0,
                "sample_mass_kg": np.nan,
                "target_available": True,
                "quality_flags": "missing_mass",
                "tx": 0.0,
            },
            {
                "experiment_id": experiments.iloc[0]["experiment_id"],
                "time_s": 10800.0,
                "sample_mass_kg": 0.009,
                "target_available": True,
                "quality_flags": "below_dry_mass",
                "tx": 0.0,
            },
        ]
    )
    observations = pd.concat([observations, invalid_rows], ignore_index=True)

    data = prepare_curve_data(experiments, observations, Scales())

    invalid = data.observations.loc[data.observations["quality_flags"].ne("")]
    assert len(invalid) == 3
    assert invalid["target_available"].tolist() == [True, False, False]
    assert invalid["moisture_db"].iloc[1:].isna().all()
    assert invalid["invalid_reason"].iloc[1:].notna().all()


def test_source_tx_never_changes_moisture_targets() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:1].copy()
    observations = _observation_rows(experiments)
    first = prepare_curve_data(experiments, observations, Scales())
    observations["tx"] = [1e30, -1e30]
    second = prepare_curve_data(experiments, observations, Scales())

    np.testing.assert_allclose(
        first.observations["moisture_db"], second.observations["moisture_db"]
    )


def test_split_holds_out_complete_replicates_within_each_condition() -> None:
    experiments = _experiment_rows(condition_count=8)

    split = split_experiments(experiments, seed=23)
    repeated = split_experiments(experiments, seed=23)

    assert split == repeated
    assert len(split.train_ids) == 16
    assert len(split.test_ids) == 8
    assert set(split.train_ids).isdisjoint(split.test_ids)
    indexed = experiments.set_index("experiment_id")
    for _, group in experiments.groupby("condition_id"):
        ids = set(group["experiment_id"])
        assert len(ids.intersection(split.train_ids)) == 2
        assert len(ids.intersection(split.test_ids)) == 1
        assert len(ids.intersection(split.train_ids + split.test_ids)) == 3
    assert set(split.train_ids + split.test_ids) == set(indexed.index)


def test_split_rejects_conditions_without_exactly_three_replicates() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:2]

    with pytest.raises(ValueError, match="exactly three"):
        split_experiments(experiments, seed=0)


def test_feature_transform_uses_training_curves_only() -> None:
    experiments = _experiment_rows(condition_count=2)
    observations = _observation_rows(experiments)
    data = prepare_curve_data(experiments, observations, Scales())
    split = split_experiments(experiments, seed=23)
    changed = data.experiments.copy()
    held_out = changed["experiment_id"].isin(split.test_ids)
    changed.loc[held_out, "air_speed_m_s"] = 1e9
    changed.loc[held_out, "initial_moisture_db"] = 1e9
    changed.loc[held_out, "dry_bulk_density_kg_m3"] = 1e9
    changed_observations = data.observations.copy()
    held_out_observations = changed_observations["experiment_id"].isin(split.test_ids)
    changed_observations.loc[held_out_observations, "moisture_db"] = 1e12
    changed_observations.loc[held_out_observations, "moisture_normalized"] = 1e12
    altered_data = type(data)(
        experiments=changed,
        observations=changed_observations,
    )

    original_transform = fit_feature_transform(data, split)
    altered_transform = fit_feature_transform(altered_data, split)

    np.testing.assert_allclose(original_transform.mean, altered_transform.mean)
    np.testing.assert_allclose(original_transform.scale, altered_transform.scale)
    transformed = original_transform.transform(data.experiments)
    assert transformed.shape == (len(experiments), 4)
    assert np.isfinite(transformed).all()
    assert original_transform.feature_names == (
        "air_speed_m_s",
        "particle_diameter_m",
        "initial_moisture_db",
        "dry_bulk_density_kg_m3",
    )


def test_split_types_are_immutable_and_malformed_inputs_fail_visibly() -> None:
    split = split_experiments(_experiment_rows(condition_count=1), seed=1)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        split.train_ids = ()

    experiments = _experiment_rows(condition_count=1)
    observations = _observation_rows(experiments)
    observations.loc[0, "time_s"] = -1.0
    with pytest.raises(ValueError, match="time"):
        prepare_curve_data(experiments, observations, Scales())

    with pytest.raises(ValueError, match="positive"):
        Scales(bed_depth_m=0.0)


def test_curve_without_any_available_mass_target_fails_explicitly() -> None:
    experiments = _experiment_rows(condition_count=1).iloc[:1].copy()
    observations = _observation_rows(experiments)
    observations["target_available"] = False

    with pytest.raises(ValueError, match="at least one available mass target"):
        prepare_curve_data(experiments, observations, Scales())
