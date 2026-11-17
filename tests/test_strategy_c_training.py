"""Behavioral tests for sequential Strategy C inverse-PINN training."""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

_data = importlib.import_module("strategy_c.data")
_physics = importlib.import_module("strategy_c.physics")
_training = importlib.import_module("strategy_c.training")
CurveData = _data.CurveData
ExperimentSplit = _data.ExperimentSplit
fit_feature_transform = _data.fit_feature_transform
prepare_curve_data = _data.prepare_curve_data
CoupledInversePINN = _physics.CoupledInversePINN
PhysicsConfig = _physics.PhysicsConfig
SamplingConfig = _training.SamplingConfig
TrainingConfig = _training.TrainingConfig
make_lbfgs_closure = _training.make_lbfgs_closure
sample_training_batch = _training.sample_training_batch
train_sequentially = _training.train_sequentially


def _prepared_data() -> tuple[CurveData, ExperimentSplit]:
    """Create three measured curves, with the third curve held out completely."""
    experiments = pd.DataFrame(
        [
            {
                "experiment_id": f"curve_{index}",
                "condition_id": "condition",
                "air_speed_m_s": 0.2 + 0.1 * index,
                "particle_diameter_m": 450e-6,
                "dry_mass_kg": 0.01 + 0.001 * index,
                "initial_sample_mass_kg": 0.02 + 0.002 * index,
            }
            for index in range(3)
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "experiment_id": f"curve_{curve}",
                "time_s": time_s,
                "sample_mass_kg": 0.02 + 0.002 * curve - 0.001 * time_index,
                "target_available": True,
            }
            for curve in range(3)
            for time_index, time_s in enumerate((0.0, 3600.0, 7200.0))
        ]
    )
    data = prepare_curve_data(experiments, observations, scales=_scales())
    return data, ExperimentSplit(("curve_0", "curve_1"), ("curve_2",))


def _scales():
    """Import the existing shared scale type without obscuring fixture setup."""
    from strategy_c.data import Scales

    return Scales()


def _training_batch():
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    batch = sample_training_batch(
        data,
        split,
        transform,
        SamplingConfig(
            domain_points=2,
            initial_points=2,
            boundary_points=2,
            seed=13,
        ),
    )
    return data, split, transform, batch


def test_training_batch_excludes_held_out_ids_and_observations() -> None:
    data, split, transform, batch = _training_batch()

    assert batch.train_ids == split.train_ids
    assert batch.held_out_ids == split.test_ids
    assert set(batch.observation_experiment_ids) == set(split.train_ids)
    assert set(batch.observation_experiment_ids).isdisjoint(split.test_ids)
    assert batch.observation_tau.shape[0] == 6
    assert len(data.observations) == 9
    assert len(transform.feature_names) == 4
    expected_targets = data.observations.loc[
        data.observations["experiment_id"].isin(split.train_ids),
        "moisture_normalized",
    ]
    np.testing.assert_allclose(
        batch.observation_targets.cpu().numpy().reshape(-1), expected_targets
    )


def test_held_out_measurement_changes_cannot_change_training_batch() -> None:
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    original_batch = sample_training_batch(
        data,
        split,
        transform,
        SamplingConfig(domain_points=2, initial_points=2, boundary_points=2, seed=9),
    )
    changed_observations = data.observations.copy()
    held_out_rows = changed_observations["experiment_id"].isin(split.test_ids)
    changed_observations.loc[held_out_rows, "sample_mass_kg"] = 100.0
    changed_observations.loc[held_out_rows, "moisture_normalized"] = 500.0
    changed_batch = sample_training_batch(
        CurveData(data.experiments, changed_observations),
        split,
        transform,
        SamplingConfig(domain_points=2, initial_points=2, boundary_points=2, seed=9),
    )

    assert torch.equal(
        original_batch.observation_targets, changed_batch.observation_targets
    )
    assert torch.equal(
        original_batch.physics.domain_features, changed_batch.physics.domain_features
    )


def test_available_noisy_mass_increase_remains_a_training_target() -> None:
    data, split = _prepared_data()
    noisy_observations = data.observations.copy()
    noisy_index = noisy_observations.index[
        noisy_observations["experiment_id"].eq(split.train_ids[0])
    ][-1]
    noisy_observations.loc[noisy_index, "moisture_normalized"] = 1.2
    noisy_observations.loc[noisy_index, "sample_mass_kg"] += 0.001
    batch = sample_training_batch(
        CurveData(data.experiments, noisy_observations),
        split,
        fit_feature_transform(data, split),
        SamplingConfig(domain_points=2, initial_points=2, boundary_points=2),
    )

    assert 1.2 in batch.observation_targets.cpu().numpy()


def test_sampled_data_loss_weights_each_curve_equally() -> None:
    _, _, _, batch = _training_batch()
    model = CoupledInversePINN(PhysicsConfig()).double()
    closure = make_lbfgs_closure(model, batch)

    closure()
    observed_losses = closure.last_components
    assert observed_losses["data"].isfinite()
    assert observed_losses["total"].isfinite()

    # Duplicating observations from one curve must not give that curve extra
    # weight in the data component because the reduction is curve-wise first.
    duplicated = replace(
        batch,
        observation_tau=torch.cat(
            (batch.observation_tau, batch.observation_tau[:3]), dim=0
        ),
        observation_features=torch.cat(
            (batch.observation_features, batch.observation_features[:3]), dim=0
        ),
        observation_conditions=torch.cat(
            (batch.observation_conditions, batch.observation_conditions[:3]), dim=0
        ),
        observation_targets=torch.cat(
            (batch.observation_targets, batch.observation_targets[:3]), dim=0
        ),
        observation_experiment_ids=(
            batch.observation_experiment_ids + batch.observation_experiment_ids[:3]
        ),
    )
    duplicated_closure = make_lbfgs_closure(model, duplicated)
    duplicated_closure()
    assert torch.allclose(
        observed_losses["data"], duplicated_closure.last_components["data"]
    )


def test_lbfgs_closure_repeats_identical_loss_and_gradients() -> None:
    _, _, _, batch = _training_batch()
    model = CoupledInversePINN(PhysicsConfig()).double()
    closure = make_lbfgs_closure(model, batch)
    parameters = tuple(model.parameters())

    first_value = closure().detach().clone()
    first_gradients = tuple(parameter.grad.detach().clone() for parameter in parameters)
    second_value = closure().detach().clone()
    second_gradients = tuple(
        parameter.grad.detach().clone() for parameter in parameters
    )

    assert closure.calls == 2
    assert torch.equal(first_value, second_value)
    for first, second in zip(first_gradients, second_gradients):
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()


def test_sequential_training_reuses_post_adam_model_and_inverse_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    config = TrainingConfig(
        adam_epochs=2,
        lbfgs_max_iter=1,
        seed=17,
        device="cpu",
    )
    model = CoupledInversePINN(PhysicsConfig()).double()
    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    adam = torch.optim.Adam
    lbfgs = torch.optim.LBFGS
    adam_parameters: list[
        tuple[tuple[torch.nn.Parameter, ...], tuple[torch.Tensor, ...]]
    ] = []
    lbfgs_initial_parameters: list[
        tuple[tuple[torch.nn.Parameter, ...], tuple[torch.Tensor, ...]]
    ] = []

    class RecordingAdam(adam):
        """Capture the exact model state after the final Adam update."""

        def step(self, closure=None):
            result = super().step(closure)
            parameters = tuple(self.param_groups[0]["params"])
            adam_parameters.append(
                (
                    parameters,
                    tuple(parameter.detach().clone() for parameter in parameters),
                )
            )
            return result

    class RecordingLBFGS(lbfgs):
        """Capture parameter objects and values at the L-BFGS handoff."""

        def __init__(self, parameters, *args, **kwargs):
            parameter_tuple = tuple(parameters)
            lbfgs_initial_parameters.append(
                (
                    parameter_tuple,
                    tuple(parameter.detach().clone() for parameter in parameter_tuple),
                )
            )
            super().__init__(parameter_tuple, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    monkeypatch.setattr(torch.optim, "LBFGS", RecordingLBFGS)

    result = train_sequentially(
        model,
        data,
        split,
        transform,
        sampling_config=SamplingConfig(
            domain_points=2,
            initial_points=2,
            boundary_points=2,
            seed=17,
        ),
        training_config=config,
    )

    assert result.model is model
    assert result.provenance.train_ids == split.train_ids
    assert result.provenance.held_out_ids == split.test_ids
    assert {
        id(parameter) for parameter in result.model.parameters()
    } == parameter_ids_before
    adam_parameters_tuple, adam_post_update_values = adam_parameters[-1]
    lbfgs_parameters_tuple, lbfgs_initial_values = lbfgs_initial_parameters[0]
    assert tuple(map(id, adam_parameters_tuple)) == tuple(
        map(id, lbfgs_parameters_tuple)
    )
    for after_adam, at_lbfgs_start in zip(
        adam_post_update_values, lbfgs_initial_values
    ):
        assert torch.equal(after_adam, at_lbfgs_start)
    assert result.stage_history["adam"].epochs == 2
    assert result.stage_history["lbfgs"].iterations == 1
    assert result.stage_history["adam"].elapsed_seconds >= 0
    assert result.stage_history["lbfgs"].elapsed_seconds >= 0
    assert result.stage_history["lbfgs"].closure_calls >= 1
    assert result.stage_history["adam"].losses[-1]["total"] >= 0
    assert result.stage_history["lbfgs"].losses[-1]["total"] >= 0
    assert all(
        torch.isfinite(parameter).all() for parameter in result.model.parameters()
    )
    assert result.model.reference_diffusivity_m2_s.item() > 0
    assert result.model.activation_energy_j_mol.item() > 0


def test_seeded_internal_model_initialization_is_reproducible() -> None:
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    sampling = SamplingConfig(
        domain_points=1,
        initial_points=1,
        boundary_points=1,
        seed=23,
    )
    training = TrainingConfig(
        adam_epochs=1,
        lbfgs_max_iter=1,
        seed=23,
        device="cpu",
        quadrature_points=2,
    )

    first = train_sequentially(
        None,
        data,
        split,
        transform,
        sampling_config=sampling,
        training_config=training,
    )
    second = train_sequentially(
        None,
        data,
        split,
        transform,
        sampling_config=sampling,
        training_config=training,
    )

    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters()
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_training_fails_explicitly_for_empty_or_nonfinite_observations() -> None:
    data, split = _prepared_data()
    invalid_observations = data.observations.copy()
    invalid_observations.loc[
        invalid_observations["experiment_id"].isin(split.train_ids),
        "target_available",
    ] = False
    no_targets = CurveData(data.experiments, invalid_observations)
    transform = fit_feature_transform(data, split)

    with pytest.raises(ValueError, match="(?i)training observations"):
        sample_training_batch(
            no_targets,
            split,
            transform,
            SamplingConfig(domain_points=2, initial_points=2, boundary_points=2),
        )

    nonfinite_observations = data.observations.copy()
    selected_index = nonfinite_observations.index[0]
    nonfinite_observations.loc[selected_index, "moisture_normalized"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sample_training_batch(
            CurveData(data.experiments, nonfinite_observations),
            split,
            transform,
            SamplingConfig(domain_points=2, initial_points=2, boundary_points=2),
        )

    zero_targets = data.observations.copy()
    zero_targets.loc[selected_index, "moisture_normalized"] = 0.0
    sample_training_batch(
        CurveData(data.experiments, zero_targets),
        split,
        transform,
        SamplingConfig(domain_points=2, initial_points=2, boundary_points=2),
    )
    negative_targets = data.observations.copy()
    negative_targets.loc[selected_index, "moisture_normalized"] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        sample_training_batch(
            CurveData(data.experiments, negative_targets),
            split,
            transform,
            SamplingConfig(domain_points=2, initial_points=2, boundary_points=2),
        )


@pytest.mark.parametrize(
    ("config_type", "kwargs"),
    [
        (SamplingConfig, {"domain_points": 0}),
        (TrainingConfig, {"adam_epochs": 0}),
        (TrainingConfig, {"lbfgs_max_iter": 0}),
        (SamplingConfig, {"seed": -1}),
        (TrainingConfig, {"seed": -1}),
    ],
)
def test_training_configurations_reject_invalid_work(config_type, kwargs) -> None:
    with pytest.raises(ValueError):
        config_type(**kwargs)
