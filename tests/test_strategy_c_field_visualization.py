"""Focused tests for train-only inferred internal-field visualization."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

_data = importlib.import_module("strategy_c.data")
_physics = importlib.import_module("strategy_c.physics")
_fields = importlib.import_module("strategy_c.field_visualization")
CurveData = _data.CurveData
FeatureTransform = _data.FeatureTransform
PhysicsConfig = _physics.PhysicsConfig
infer_internal_fields = _fields.infer_internal_fields
select_representative_training_experiment = (
    _fields.select_representative_training_experiment
)
write_internal_field_artifacts = _fields.write_internal_field_artifacts


class AnalyticFieldModel(nn.Module):
    """Analytic forward pass makes axis orientation and scaling testable."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(
        self, tau: torch.Tensor, xi: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moisture_ratio = self.anchor + tau + 3.0 * xi
        initial_temperature = 298.15
        temperature = initial_temperature + 45.0 * (tau + 2.0 * xi)
        return moisture_ratio, temperature


def _data() -> tuple[CurveData, tuple[str, ...]]:
    """Create deliberately unsorted training metadata and irrelevant targets."""
    experiments = pd.DataFrame(
        [
            {
                "experiment_id": "train-z",
                "condition_id": "slow",
                "air_speed_m_s": 0.4,
                "particle_diameter_m": 500e-6,
                "replicate": 2,
                "initial_moisture_db": 0.8,
                "dry_bulk_density_kg_m3": 300.0,
            },
            {
                "experiment_id": "held-out",
                "condition_id": "fast",
                "air_speed_m_s": 0.2,
                "particle_diameter_m": 400e-6,
                "replicate": 1,
                "initial_moisture_db": 0.9,
                "dry_bulk_density_kg_m3": 310.0,
            },
            {
                "experiment_id": "train-a",
                "condition_id": "slow",
                "air_speed_m_s": 0.2,
                "particle_diameter_m": 400e-6,
                "replicate": 1,
                "initial_moisture_db": 0.7,
                "dry_bulk_density_kg_m3": 290.0,
            },
            {
                "experiment_id": "train-b",
                "condition_id": "slow",
                "air_speed_m_s": 0.2,
                "particle_diameter_m": 400e-6,
                "replicate": 2,
                "initial_moisture_db": 0.75,
                "dry_bulk_density_kg_m3": 295.0,
            },
        ]
    )
    observations = pd.DataFrame(
        {
            "experiment_id": ["held-out"],
            "moisture_normalized": [np.nan],
            "target_available": [False],
        }
    )
    return CurveData(experiments, observations), ("train-z", "train-b", "train-a")


def _transform() -> FeatureTransform:
    """Provide an explicit train-fitted transform for model feature inputs."""
    return FeatureTransform(
        feature_names=(
            "air_speed_m_s",
            "particle_diameter_m",
            "initial_moisture_db",
            "dry_bulk_density_kg_m3",
        ),
        mean=(0.2, 400e-6, 0.7, 290.0),
        scale=(0.1, 100e-6, 0.1, 10.0),
    )


def test_field_grid_orientation_dimensionless_conversion_and_model_state() -> None:
    """Depth rows/time columns preserve model values without changing its state."""
    data, train_ids = _data()
    model = AnalyticFieldModel()
    model.train()
    original_parameter = model.anchor.detach().clone()
    config = PhysicsConfig()

    result = infer_internal_fields(
        model,
        data,
        train_ids,
        _transform(),
        config,
        time_points=3,
        depth_points=3,
    )

    np.testing.assert_allclose(result.time_normalized, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result.depth_normalized, [0.0, 0.5, 1.0])
    assert result.moisture_ratio.shape == (3, 3)
    assert result.moisture_ratio[2, 0] == pytest.approx(4.0)
    assert result.moisture_ratio[0, 2] == pytest.approx(2.0)
    assert result.temperature_normalized[2, 0] == pytest.approx(-1.0)
    assert result.temperature_normalized[0, 2] == pytest.approx(0.0)
    moisture_map = result.figure.axes[0].collections[0]
    temperature_map = result.figure.axes[1].collections[0]
    assert moisture_map.get_clim() == (0.0, 5.0)
    assert temperature_map.get_clim() == (-2.0, 1.0)
    assert result.figure.axes[0].get_ylim() == (0.0, 1.0)
    assert result.figure.axes[1].get_title() == "temperature field"
    assert result.figure.axes[3].get_xlabel() == "Normalized temperature, θ"
    assert result.figure.axes[3].get_legend().get_title().get_text() == "Time (min)"
    result.figure.canvas.draw()
    caption_bounds = result.figure._suptitle.get_window_extent()
    assert caption_bounds.x0 >= 0
    assert caption_bounds.x1 <= result.figure.bbox.width
    assert result.selected_experiment_id == "train-a"
    assert model.training
    torch.testing.assert_close(model.anchor, original_parameter)


def test_selection_is_deterministic_and_restricted_to_training_metadata() -> None:
    """Selection follows speed, size, replicate, and excludes held-out IDs."""
    data, train_ids = _data()

    selected = select_representative_training_experiment(data, train_ids)

    assert selected["experiment_id"] == "train-a"
    assert "held-out" not in set(train_ids)


def test_field_inference_is_invariant_to_held_out_moisture_targets() -> None:
    """Held-out target changes cannot affect metadata-only field predictions."""
    data, train_ids = _data()
    changed_observations = data.observations.copy(deep=True)
    changed_observations["moisture_normalized"] = [1.0e12]
    changed_data = CurveData(data.experiments, changed_observations)

    original = infer_internal_fields(
        AnalyticFieldModel(),
        data,
        train_ids,
        _transform(),
        PhysicsConfig(),
        time_points=3,
        depth_points=3,
    )
    changed = infer_internal_fields(
        AnalyticFieldModel(),
        changed_data,
        train_ids,
        _transform(),
        PhysicsConfig(),
        time_points=3,
        depth_points=3,
    )

    np.testing.assert_array_equal(original.moisture_ratio, changed.moisture_ratio)
    np.testing.assert_array_equal(
        original.temperature_normalized, changed.temperature_normalized
    )


def test_field_artifact_writer_persists_figure_and_numeric_grids(
    tmp_path: Path,
) -> None:
    """Saved data includes physical coordinates and normalized predicted fields."""
    data, train_ids = _data()
    result = infer_internal_fields(
        AnalyticFieldModel(),
        data,
        train_ids,
        _transform(),
        PhysicsConfig(),
        time_points=3,
        depth_points=3,
    )

    paths = write_internal_field_artifacts(result, tmp_path)

    assert paths.figure.is_file()
    assert paths.grid.is_file()
    with np.load(paths.grid, allow_pickle=False) as archive:
        assert archive["moisture_ratio"].shape == (3, 3)
        assert archive["temperature_normalized"].shape == (3, 3)
        assert archive["temperature_kelvin"].shape == (3, 3)
        np.testing.assert_allclose(archive["time_s"], [0.0, 7200.0, 14400.0])
        np.testing.assert_allclose(archive["depth_m"], [0.0, 0.005, 0.01])
        reconstructed_kelvin = 343.15 + (-45.0) * archive["temperature_normalized"]
        np.testing.assert_allclose(archive["temperature_kelvin"], reconstructed_kelvin)
        assert archive["temperature_normalization_definition"].item() == (
            "(T_K - T_air_K) / (T_initial_K - T_air_K)"
        )
        assert archive["signed_temperature_scale_K"].item() == -45.0
    assert "θ=1 initially, θ=0 at air temperature" in (
        result.figure._suptitle.get_text()
    )


class OvershootFieldModel(AnalyticFieldModel):
    """Manufacture both cooler-than-initial and hotter-than-air field states."""

    def forward(
        self, tau: torch.Tensor, xi: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moisture = torch.ones_like(tau) + 0.0 * features[:, :1]
        temperature = 290.0 + 80.0 * tau + 5.0 * xi
        return moisture, temperature


def test_field_maps_and_archive_preserve_theta_outside_zero_to_one(
    tmp_path: Path,
) -> None:
    data, train_ids = _data()
    result = infer_internal_fields(
        OvershootFieldModel(),
        data,
        train_ids,
        _transform(),
        PhysicsConfig(),
        time_points=3,
        depth_points=3,
    )
    assert result.temperature_normalized.max() > 1.0
    assert result.temperature_normalized.min() < 0.0
    temperature_map = result.figure.axes[1].collections[0]
    assert temperature_map.get_clim() == (
        float(result.temperature_normalized.min()),
        float(result.temperature_normalized.max()),
    )

    paths = write_internal_field_artifacts(result, tmp_path)
    with np.load(paths.grid, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["temperature_normalized"], result.temperature_normalized
        )
        np.testing.assert_allclose(
            archive["temperature_kelvin"],
            343.15 - 45.0 * archive["temperature_normalized"],
        )
