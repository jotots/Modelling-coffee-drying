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
build_adam_optimizer = _training.build_adam_optimizer
sample_training_batch = _training.sample_training_batch
train_sequentially = _training.train_sequentially


def _bounded_training_config(
    *,
    adam_epochs: int = 1,
    adam_learning_rate: float = 0.001,
    physics_parameter_learning_rate: float = 0.05,
    lbfgs_max_iter: int = 1,
    lbfgs_max_eval: int = 2,
    seed: int = 0,
    device: str = "cpu",
    quadrature_points: int = 2,
) -> TrainingConfig:
    """Use valid small bounds without changing the project's global defaults."""
    return TrainingConfig(
        adam_epochs=adam_epochs,
        adam_learning_rate=adam_learning_rate,
        physics_parameter_learning_rate=physics_parameter_learning_rate,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_max_eval=lbfgs_max_eval,
        seed=seed,
        device=device,
        quadrature_points=quadrature_points,
    )


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
    return data, ExperimentSplit(
        train_ids=("curve_0",),
        test_ids=("curve_2",),
        validation_ids=("curve_1",),
    )


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


def test_validation_monitoring_changes_only_diagnostics_not_fitted_parameters() -> None:
    """Validation is diagnostic only; final-test observations remain unread."""
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    observations = data.observations.copy(deep=True)
    observations.loc[
        observations["experiment_id"].isin(split.validation_ids), "moisture_normalized"
    ] = 20.0
    observations.loc[observations["experiment_id"].isin(split.test_ids), "time_s"] = (
        14_401.0
    )
    observations.loc[
        observations["experiment_id"].isin(split.test_ids), "moisture_normalized"
    ] = 500.0
    settings = dict(
        sampling_config=SamplingConfig(
            domain_points=2, initial_points=2, boundary_points=2
        ),
        training_config=TrainingConfig(
            adam_epochs=2,
            lbfgs_max_iter=1,
            lbfgs_max_eval=2,
            quadrature_points=2,
            device="cpu",
        ),
    )
    original = train_sequentially(None, data, split, transform, **settings)
    changed = train_sequentially(
        None, CurveData(data.experiments, observations), split, transform, **settings
    )
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(
            value, changed.model.state_dict()[name], rtol=0, atol=0
        )
    for stage in ("adam", "lbfgs"):
        for first, second in zip(
            original.stage_history[stage].losses, changed.stage_history[stage].losses
        ):
            assert first["validation_data"] != second["validation_data"]
            assert first["total"] == second["total"]
            assert "held_out_data" not in second
    assert original.provenance.validation_ids == split.validation_ids
    assert original.provenance.test_ids == split.test_ids
    assert original.provenance.held_out_ids == split.validation_ids + split.test_ids
    assert original.physics_scales == changed.physics_scales
    assert original.physics_calibration_mse == changed.physics_calibration_mse


@pytest.mark.parametrize("bad_value", [float("nan"), -1.0, 14401.0])
def test_validation_monitor_rejects_invalid_available_times(bad_value: float) -> None:
    data, split = _prepared_data()
    observations = data.observations.copy(deep=True)
    observations.loc[observations["experiment_id"].eq("curve_1"), "time_s"] = bad_value
    with pytest.raises(ValueError, match="Validation curve"):
        _training._make_validation_monitor(
            CoupledInversePINN().double(),
            CurveData(data.experiments, observations),
            split,
            fit_feature_transform(data, split),
            PhysicsConfig(),
            2,
        )


def test_validation_monitor_handles_empty_partition_and_missing_targets() -> None:
    data, split = _prepared_data()
    model = CoupledInversePINN().double()
    transform = fit_feature_transform(data, split)
    assert (
        _training._make_validation_monitor(
            model,
            data,
            ExperimentSplit(
                train_ids=tuple(data.experiments["experiment_id"]),
                test_ids=(),
                validation_ids=(),
            ),
            transform,
            PhysicsConfig(),
            2,
        )
        is None
    )
    observations = data.observations.copy(deep=True)
    observations.loc[
        observations["experiment_id"].eq("curve_1"), "target_available"
    ] = False
    with pytest.raises(ValueError, match="Validation curve"):
        _training._make_validation_monitor(
            model,
            CurveData(data.experiments, observations),
            split,
            transform,
            PhysicsConfig(),
            2,
        )


def test_validation_monitor_uses_equal_curve_mse_and_no_gradients() -> None:
    """Unequal observation counts must not turn monitoring into pooled MSE."""
    data, _ = _prepared_data()
    split = ExperimentSplit(
        train_ids=("curve_0",),
        test_ids=(),
        validation_ids=("curve_1", "curve_2"),
    )
    observations = data.observations.loc[
        ~data.observations["experiment_id"].eq("curve_2")
        | data.observations["time_s"].eq(0.0)
    ].copy()
    observations.loc[
        observations["experiment_id"].eq("curve_2"), "moisture_normalized"
    ] = 10.0
    data = CurveData(data.experiments, observations)
    model = CoupledInversePINN().double()
    transform = fit_feature_transform(data, split)
    monitor = _training._make_validation_monitor(
        model,
        data,
        split,
        transform,
        PhysicsConfig(),
        2,
    )
    assert monitor is not None
    loss = monitor()
    assert not loss.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    # Score each validation curve separately with the same helper. Averaging these
    # scores must agree with the multi-curve score despite the 3:1 row counts.
    scores = []
    for identifier in split.validation_ids:
        single_split = ExperimentSplit(
            train_ids=tuple(
                value
                for value in ("curve_0", "curve_1", "curve_2")
                if value != identifier
            ),
            test_ids=(),
            validation_ids=(identifier,),
        )
        single = _training._make_validation_monitor(
            model,
            data,
            single_split,
            transform,
            PhysicsConfig(),
            2,
        )
        assert single is not None
        scores.append(float(single()))
    assert float(loss) == pytest.approx(sum(scores) / 2)
    assert float(loss) != pytest.approx((3 * scores[0] + scores[1]) / 4)


def test_training_batch_excludes_validation_and_test_ids_and_observations() -> None:
    data, split, transform, batch = _training_batch()

    assert batch.train_ids == split.train_ids
    assert batch.validation_ids == split.validation_ids
    assert batch.test_ids == split.test_ids
    assert batch.held_out_ids == split.validation_ids + split.test_ids
    assert set(batch.observation_experiment_ids) == set(split.train_ids)
    assert set(batch.observation_experiment_ids).isdisjoint(
        split.validation_ids + split.test_ids
    )
    assert batch.observation_tau.shape[0] == 3
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
    nontraining_rows = changed_observations["experiment_id"].isin(
        split.validation_ids + split.test_ids
    )
    changed_observations.loc[nontraining_rows, "sample_mass_kg"] = 100.0
    changed_observations.loc[nontraining_rows, "moisture_normalized"] = 500.0
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
    closure = make_lbfgs_closure(
        model, batch, training_config=_bounded_training_config()
    )

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
    duplicated_closure = make_lbfgs_closure(
        model, duplicated, training_config=_bounded_training_config()
    )
    duplicated_closure()
    assert torch.allclose(
        observed_losses["data"], duplicated_closure.last_components["data"]
    )


def test_integrated_moisture_data_loss_reaches_first_layer_depth_weights() -> None:
    """Data fitting differentiates the local field through bulk quadrature."""
    _, _, _, batch = _training_batch()
    model = CoupledInversePINN(PhysicsConfig()).double()
    predicted = (
        _physics.bulk_predictions(
            model,
            batch.observation_tau,
            batch.observation_features,
            batch.observation_conditions,
            quadrature_points=8,
        )
        / batch.observation_conditions[:, 2:3]
    )
    data_loss = (predicted - batch.observation_targets).square().mean()
    data_loss.backward()

    depth_column_gradient = model.moisture_network[0].weight.grad[:, 1]
    assert torch.isfinite(depth_column_gradient).all()
    assert torch.count_nonzero(depth_column_gradient).item() > 0


def test_data_only_warmup_isolates_physics_only_parameters() -> None:
    """At zero ramp, one Adam update changes data-field weights only."""
    _, _, _, batch = _training_batch()
    model = CoupledInversePINN(PhysicsConfig()).double()
    config = _bounded_training_config()
    scales, _ = _training._calibrate_physics_scales(model, batch, model.config)
    before_temperature = tuple(
        parameter.detach().clone()
        for parameter in model.temperature_network.parameters()
    )
    before_physics = tuple(
        parameter.detach().clone()
        for parameter in (
            model.log_reference_diffusivity,
            model.diameter_exponent,
            model.speed_exponent,
        )
    )
    optimizer = build_adam_optimizer(model, config)
    objective = _training._objective_components(
        model, batch, model.config, config, scales, physics_ramp=0.0
    )
    objective["total"].backward()
    optimizer.step()

    assert objective["physics_contribution"].item() == 0.0
    assert any(
        torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.moisture_network.parameters()
    )
    for previous, current in zip(
        before_temperature, model.temperature_network.parameters()
    ):
        torch.testing.assert_close(previous, current, rtol=0, atol=0)
    for previous, current in zip(
        before_physics,
        (
            model.log_reference_diffusivity,
            model.diameter_exponent,
            model.speed_exponent,
        ),
    ):
        torch.testing.assert_close(previous, current, rtol=0, atol=0)


def test_fractional_schedule_rounds_and_reserves_full_objective_step() -> None:
    """Default fractions resolve predictably for notebook and bounded budgets."""
    default = TrainingConfig(adam_epochs=50, device="cpu")
    schedule = _training._schedule_boundaries(default)
    assert schedule == {
        "adam_epochs": 50,
        "warmup_epochs": 10,
        "ramp_epochs": 20,
        "full_objective_epochs": 20,
    }
    one_step = _training._schedule_boundaries(
        TrainingConfig(adam_epochs=1, device="cpu")
    )
    assert one_step == {
        "adam_epochs": 1,
        "warmup_epochs": 0,
        "ramp_epochs": 0,
        "full_objective_epochs": 1,
    }
    assert _training._physics_ramp_for_epoch(0, one_step) == 1.0


def test_lbfgs_closure_repeats_identical_loss_and_gradients() -> None:
    _, _, _, batch = _training_batch()
    model = CoupledInversePINN(PhysicsConfig()).double()
    closure = make_lbfgs_closure(
        model, batch, training_config=_bounded_training_config()
    )
    parameters = tuple(model.parameters())

    first_value = closure().detach().clone()
    first_gradients = tuple(parameter.grad.detach().clone() for parameter in parameters)
    second_value = closure().detach().clone()
    second_gradients = tuple(
        parameter.grad.detach().clone() for parameter in parameters
    )

    assert closure.calls == 2
    assert torch.equal(first_value, second_value)
    assert observed_total_is_reconstructible(closure.last_components)
    for first, second in zip(first_gradients, second_gradients):
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()


def observed_total_is_reconstructible(components: dict[str, torch.Tensor]) -> bool:
    """Check that the reported explicit contributions recover the objective."""
    contributions = [
        components[f"contribution_{name}"] for name in _training._ACTIVE_PHYSICS_TERMS
    ]
    reconstructed = components["data"] + torch.stack(contributions).sum()
    return bool(torch.allclose(reconstructed, components["total"]))


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
            # Adam now holds network and diffusivity-law parameter groups;
            # flatten them in group order to recover every updated tensor.
            parameters = tuple(
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            )
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
    assert result.provenance.validation_ids == split.validation_ids
    assert result.provenance.test_ids == split.test_ids
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
    assert result.stage_history["lbfgs"].accepted_losses["total"] >= 0
    assert result.stage_history["lbfgs"].accepted_gradient_norm >= 0
    assert set(result.stage_history["lbfgs"].termination_flags) == {
        "iteration_limit_reached",
        "evaluation_limit_reached",
    }
    assert result.stage_history["adam"].losses[-1]["total"] >= 0
    assert result.stage_history["lbfgs"].losses[-1]["total"] >= 0
    assert all(
        torch.isfinite(parameter).all() for parameter in result.model.parameters()
    )
    assert result.model.reference_diffusivity_m2_s.item() > 0
    # Ea is fixed: training must leave the Barnard value untouched.
    assert result.model.activation_energy_j_mol.item() == pytest.approx(
        result.model.config.activation_energy_j_mol, rel=1e-15
    )


def test_lbfgs_accepted_state_is_recomputed_after_closure_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted-state report follows returned parameters, not a trial log."""
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    model = CoupledInversePINN().double()

    class TrialOnlyLBFGS:
        """Simulate an optimizer state change after its final closure trial."""

        def __init__(self, parameters, **kwargs):
            del kwargs
            self.parameters = tuple(parameters)
            self.state = {self.parameters[0]: {"n_iter": 1, "func_evals": 1}}

        def step(self, closure):
            trial_loss = closure().detach().clone()
            self.trial_loss = float(trial_loss.cpu())
            with torch.no_grad():
                model.moisture_network[-1].bias.add_(0.1)
            return trial_loss

    monkeypatch.setattr(torch.optim, "LBFGS", TrialOnlyLBFGS)
    result = train_sequentially(
        model,
        data,
        split,
        transform,
        sampling_config=SamplingConfig(
            domain_points=2, initial_points=2, boundary_points=2, seed=7
        ),
        training_config=TrainingConfig(
            adam_epochs=1,
            lbfgs_max_iter=1,
            lbfgs_max_eval=1,
            device="cpu",
            quadrature_points=2,
            physics_weight=0.0,
            boundary_weight=0.0,
        ),
    )
    history = result.stage_history["lbfgs"]

    assert history.accepted_losses["total"] != pytest.approx(history.losses[0]["total"])
    assert history.accepted_gradient_norm is not None
    assert history.termination_flags == {
        "iteration_limit_reached": True,
        "evaluation_limit_reached": True,
    }


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


def test_training_rejects_zero_air_speed_for_signed_diffusivity_power() -> None:
    data, split = _prepared_data()
    experiments = data.experiments.copy()
    experiments.loc[experiments["experiment_id"].eq("curve_0"), "air_speed_m_s"] = 0.0
    invalid_data = CurveData(experiments, data.observations)
    transform = fit_feature_transform(invalid_data, split)

    with pytest.raises(ValueError, match="strictly positive air_speed_m_s"):
        sample_training_batch(
            invalid_data,
            split,
            transform,
            SamplingConfig(domain_points=1, initial_points=1, boundary_points=1),
        )


def test_training_objective_propagates_gradients_to_all_shared_diffusivity_parameters() -> (
    None
):
    data, split = _prepared_data()
    experiments = data.experiments.copy()
    experiments.loc[:, "particle_diameter_m"] = [400e-6, 500e-6, 600e-6]
    data = CurveData(experiments, data.observations)
    transform = fit_feature_transform(data, split)
    batch = sample_training_batch(
        data,
        split,
        transform,
        SamplingConfig(domain_points=2, initial_points=2, boundary_points=2, seed=3),
    )
    model = CoupledInversePINN().double()
    closure = make_lbfgs_closure(
        model, batch, training_config=_bounded_training_config()
    )

    closure()

    for parameter in (
        model.log_reference_diffusivity,
        model.diameter_exponent,
        model.speed_exponent,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert model.activation_energy_j_mol.grad is None
    assert not model.activation_energy_j_mol.requires_grad


@pytest.mark.parametrize(
    ("config_type", "kwargs"),
    [
        (SamplingConfig, {"domain_points": 0}),
        (TrainingConfig, {"adam_epochs": 0}),
        (TrainingConfig, {"lbfgs_max_iter": 0}),
        (SamplingConfig, {"seed": -1}),
        (TrainingConfig, {"seed": -1}),
        (TrainingConfig, {"physics_parameter_learning_rate": 0.0}),
        (TrainingConfig, {"physics_parameter_learning_rate": -0.05}),
        (TrainingConfig, {"physics_parameter_learning_rate": float("nan")}),
        (TrainingConfig, {"physics_parameter_learning_rate": float("inf")}),
        (TrainingConfig, {"physics_parameter_learning_rate": True}),
        (TrainingConfig, {"warmup_fraction": -0.01}),
        (TrainingConfig, {"ramp_fraction": 1.01}),
        (TrainingConfig, {"warmup_fraction": float("nan")}),
    ],
)
def test_training_configurations_reject_invalid_work(config_type, kwargs) -> None:
    with pytest.raises(ValueError):
        config_type(**kwargs)


def test_physics_parameter_learning_rate_validation_names_the_field() -> None:
    learning_rate_default = TrainingConfig.__dataclass_fields__[
        "physics_parameter_learning_rate"
    ].default
    assert learning_rate_default == pytest.approx(0.05)
    with pytest.raises(ValueError, match="physics_parameter_learning_rate"):
        _bounded_training_config(physics_parameter_learning_rate=0.0)


_LAW_PARAMETER_NAMES = (
    "log_reference_diffusivity",
    "diameter_exponent",
    "speed_exponent",
)


def _group_ids(optimizer: torch.optim.Optimizer) -> list[tuple[float, set[int]]]:
    """Summarize each Adam parameter group as (learning rate, parameter ids)."""
    return [
        (group["lr"], {id(parameter) for parameter in group["params"]})
        for group in optimizer.param_groups
    ]


def test_adam_uses_separate_learning_rate_for_diffusivity_law_parameters() -> None:
    model = CoupledInversePINN(PhysicsConfig()).double()
    law_ids = {id(getattr(model, name)) for name in _LAW_PARAMETER_NAMES}
    trainable_ids = [id(parameter) for parameter in model.parameters()]

    optimizer = build_adam_optimizer(model, _bounded_training_config())

    groups = _group_ids(optimizer)
    assert len(groups) == 2
    law_group = [ids for rate, ids in groups if rate == pytest.approx(0.05)]
    network_group = [ids for rate, ids in groups if rate == pytest.approx(0.001)]
    assert law_group == [law_ids]
    assert len(network_group) == 1
    assert network_group[0] == set(trainable_ids) - law_ids
    # Every trainable tensor appears in exactly one group.
    flattened = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert sorted(flattened) == sorted(trainable_ids)
    assert len(flattened) == len(set(flattened))
    # The fixed activation energy buffer is never optimized.
    assert id(model.activation_energy_j_mol) not in flattened


def test_fixed_activation_energy_is_not_a_diffusivity_law_parameter() -> None:
    assert "log_activation_energy" not in _training._DIFFUSIVITY_LAW_PARAMETER_NAMES
    assert _training._DIFFUSIVITY_LAW_PARAMETER_NAMES == _LAW_PARAMETER_NAMES


def test_adam_group_rates_follow_the_training_configuration() -> None:
    model = CoupledInversePINN(PhysicsConfig()).double()
    config = _bounded_training_config(
        adam_learning_rate=0.002, physics_parameter_learning_rate=0.3
    )

    groups = _group_ids(build_adam_optimizer(model, config))

    rates = sorted(rate for rate, _ in groups)
    assert rates == [pytest.approx(0.002), pytest.approx(0.3)]


def test_adam_uses_single_group_for_models_without_diffusivity_law() -> None:
    model = torch.nn.Linear(2, 1).double()

    optimizer = build_adam_optimizer(model, _bounded_training_config())

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in model.parameters()
    }


def test_adam_excludes_frozen_law_parameters_from_every_group() -> None:
    model = CoupledInversePINN(PhysicsConfig()).double()
    model.diameter_exponent.requires_grad_(False)

    optimizer = build_adam_optimizer(model, _bounded_training_config())

    flattened = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert id(model.diameter_exponent) not in flattened
    law_group = [
        ids for rate, ids in _group_ids(optimizer) if rate == pytest.approx(0.05)
    ]
    assert law_group == [
        {
            id(getattr(model, name))
            for name in _LAW_PARAMETER_NAMES
            if name != "diameter_exponent"
        }
    ]
    assert sorted(flattened) == sorted(
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    )


def test_adam_optimizer_rejects_models_without_trainable_parameters() -> None:
    model = torch.nn.Linear(2, 1)
    model.requires_grad_(False)

    with pytest.raises(ValueError, match="trainable parameters"):
        build_adam_optimizer(model, _bounded_training_config())
    with pytest.raises(TypeError, match="TrainingConfig"):
        build_adam_optimizer(torch.nn.Linear(2, 1), object())


def test_sequential_training_builds_adam_with_physics_parameter_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, split = _prepared_data()
    transform = fit_feature_transform(data, split)
    model = CoupledInversePINN(PhysicsConfig()).double()
    law_ids = {id(getattr(model, name)) for name in _LAW_PARAMETER_NAMES}
    recorded_groups: list[list[tuple[float, set[int]]]] = []
    adam = torch.optim.Adam

    class RecordingAdam(adam):
        """Record the parameter-group layout chosen by train_sequentially."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded_groups.append(_group_ids(self))

    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)

    train_sequentially(
        model,
        data,
        split,
        transform,
        sampling_config=SamplingConfig(
            domain_points=1, initial_points=1, boundary_points=1, seed=5
        ),
        training_config=TrainingConfig(
            adam_epochs=1,
            # This assertion is about parameter-group membership; full-weight
            # staged training is exercised by the reporting integration test.
            physics_weight=0.0,
            boundary_weight=0.0,
            lbfgs_max_iter=1,
            seed=5,
            device="cpu",
            quadrature_points=2,
        ),
    )

    assert len(recorded_groups) == 1
    groups = recorded_groups[0]
    assert len(groups) == 2
    assert any(rate == pytest.approx(0.05) and ids == law_ids for rate, ids in groups)
    assert any(
        rate == pytest.approx(0.001)
        and ids == {id(parameter) for parameter in model.parameters()} - law_ids
        for rate, ids in groups
    )
