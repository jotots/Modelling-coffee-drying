"""Contract tests for locked Strategy C evaluation and artifact export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
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
_reporting = importlib.import_module("strategy_c.reporting")
_field_visualization = importlib.import_module("strategy_c.field_visualization")
CurveData = _data.CurveData
ExperimentSplit = _data.ExperimentSplit
Scales = _data.Scales
fit_feature_transform = _data.fit_feature_transform
prepare_curve_data = _data.prepare_curve_data
CoupledInversePINN = _physics.CoupledInversePINN
PhysicsConfig = _physics.PhysicsConfig
evaluate_locked_model = _reporting.evaluate_locked_model
export_results = _reporting.export_results
load_model_artifacts = _reporting.load_model_artifacts
SamplingConfig = _training.SamplingConfig
StageHistory = _training.StageHistory
TrainingConfig = _training.TrainingConfig
TrainingProvenance = _training.TrainingProvenance
TrainingResult = _training.TrainingResult
train_sequentially = _training.train_sequentially
infer_internal_fields = _field_visualization.infer_internal_fields


def _prepared_data() -> tuple[CurveData, ExperimentSplit]:
    """Make two fit curves and one immutable held-out curve with parser flags."""
    experiment_rows = [
        {
            "experiment_id": f"curve_{index}",
            "condition_id": "condition",
            "air_speed_m_s": 0.22 + 0.01 * index,
            "particle_diameter_m": 450e-6,
            "dry_mass_kg": 0.010,
            "initial_sample_mass_kg": 0.020,
        }
        for index in range(3)
    ]
    experiments = pd.DataFrame(experiment_rows)
    observation_rows = []
    for curve_index in range(3):
        for time_s, mass_kg in (
            (0.0, 0.020),
            (3600.0, 0.018),
            (7200.0, 0.019 if curve_index == 2 else 0.016),
        ):
            observation_rows.append(
                {
                    "experiment_id": f"curve_{curve_index}",
                    "time_s": time_s,
                    "sample_mass_kg": mass_kg,
                    "target_available": True,
                    "quality_flags": "mass_increase" if time_s == 7200 else "",
                }
            )
    observations = pd.DataFrame(observation_rows)
    data = prepare_curve_data(experiments, observations, Scales())
    return data, ExperimentSplit(
        train_ids=("curve_0",),
        test_ids=("curve_2",),
        validation_ids=("curve_1",),
    )


def _balanced_prepared_data() -> CurveData:
    """Make the complete 2-speed by 4-diameter replicated study for integration."""
    experiment_rows = []
    observation_rows = []
    for speed_index, air_speed in enumerate((0.22, 0.48)):
        for diameter_index, diameter in enumerate((213e-6, 236e-6, 450e-6, 610e-6)):
            condition_index = speed_index * 4 + diameter_index
            for replicate in range(3):
                identifier = (
                    f"speed_{speed_index}_diameter_{diameter_index}_rep_{replicate}"
                )
                dry_mass = 0.010 + replicate * 0.0001
                initial_mass = 0.020 + replicate * 0.0001
                experiment_rows.append(
                    {
                        "experiment_id": identifier,
                        "condition_id": f"condition_{condition_index}",
                        "air_speed_m_s": air_speed,
                        "particle_diameter_m": diameter,
                        "dry_mass_kg": dry_mass,
                        "initial_sample_mass_kg": initial_mass,
                    }
                )
                for time_s, mass_kg in (
                    (0.0, initial_mass),
                    (3600.0, initial_mass - 0.002),
                    (7200.0, initial_mass - 0.004),
                ):
                    observation_rows.append(
                        {
                            "experiment_id": identifier,
                            "time_s": time_s,
                            "sample_mass_kg": mass_kg,
                            "target_available": True,
                            "quality_flags": "",
                        }
                    )
    return prepare_curve_data(
        pd.DataFrame(experiment_rows),
        pd.DataFrame(observation_rows),
        Scales(),
    )


def _bounded_training_config(
    *,
    adam_epochs: int = 1,
    lbfgs_max_iter: int = 2,
    lbfgs_max_eval: int = 2,
    quadrature_points: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> TrainingConfig:
    """Provide valid small optimizer bounds for deterministic artifact tests."""
    return TrainingConfig(
        adam_epochs=adam_epochs,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_max_eval=lbfgs_max_eval,
        quadrature_points=quadrature_points,
        seed=seed,
        device=device,
    )


def _training_result(
    data: CurveData, split: ExperimentSplit
) -> tuple[TrainingResult, object]:
    """Build deterministic test parameters with strict train/test provenance."""
    transform = fit_feature_transform(data, split)
    model = CoupledInversePINN(PhysicsConfig()).double()
    recorded_config = _bounded_training_config()
    residuals = _training._ACTIVE_PHYSICS_TERMS
    frozen_scales = {name: 1.0 for name in residuals}
    calibration_mse = {name: 0.25 for name in residuals}
    schedule = _training._schedule_boundaries(recorded_config)

    def snapshot(data_loss: float) -> dict[str, float]:
        values = {
            "data": data_loss,
            "physics_ramp": 1.0,
            "initial_moisture": 0.0,
            "initial_temperature": 0.0,
        }
        for name in residuals:
            values[name] = 0.25
            values[f"scale_{name}"] = frozen_scales[name]
            values[f"scaled_{name}"] = 0.25
            values[f"contribution_{name}"] = 0.25 / len(residuals)
        values["physics_contribution"] = sum(
            values[f"contribution_{name}"] for name in residuals
        )
        values["total"] = data_loss + values["physics_contribution"]
        return values

    adam_snapshot = snapshot(0.1)
    lbfgs_snapshot = snapshot(0.08)
    result = TrainingResult(
        model=model,
        stage_history={
            "adam": StageHistory(
                epochs=recorded_config.adam_epochs,
                elapsed_seconds=1.25,
                losses=[adam_snapshot],
            ),
            "lbfgs": StageHistory(
                iterations=2,
                closure_calls=2,
                elapsed_seconds=0.5,
                losses=[lbfgs_snapshot],
                accepted_losses=snapshot(0.07),
                accepted_gradient_norm=0.04,
                termination_flags={
                    "iteration_limit_reached": False,
                    "evaluation_limit_reached": True,
                },
            ),
        },
        provenance=TrainingProvenance(
            seed=13,
            train_ids=split.train_ids,
            held_out_ids=split.validation_ids + split.test_ids,
            training_observation_count=3,
            collocation_point_count=8,
            device="cpu",
            dtype="torch.float64",
            validation_ids=split.validation_ids,
            test_ids=split.test_ids,
        ),
        physics_scales=frozen_scales,
        physics_calibration_mse=calibration_mse,
        schedule=schedule,
    )
    return result, transform


def test_notebook_result_display_runs_with_bounded_real_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the actual integration cell with tiny work bounds and temporary exports."""
    import matplotlib.pyplot as plt

    data = _balanced_prepared_data()
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "strategy_c.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def run_strategy_c_inverse_pinn" in "".join(cell["source"])
    )
    namespace = {
        "Path": Path,
        "sys": sys,
        "torch": torch,
        "experiments": data.experiments,
        "observations": data.observations,
    }
    exec(compile(source, "strategy_c.ipynb integration", "exec"), namespace)
    namespace["STRATEGY_C_REPOSITORY_ROOT"] = tmp_path
    namespace["SamplingConfig"] = lambda **settings: SamplingConfig(
        domain_points=2,
        initial_points=2,
        boundary_points=2,
    )
    namespace["TrainingConfig"] = lambda **settings: _bounded_training_config(
        adam_epochs=5,
        lbfgs_max_iter=1,
        lbfgs_max_eval=2,
        quadrature_points=2,
        device="cpu",
    )
    namespace["infer_internal_fields"] = lambda *args: infer_internal_fields(
        *args,
        time_points=3,
        depth_points=3,
    )
    displays = []
    monkeypatch.setattr(plt, "show", lambda: displays.append(True))
    result = namespace["run_strategy_c_inverse_pinn"]()
    try:
        assert len(displays) == 2
        assert result["loss_figure"].axes[0].get_xlabel() == "Adam epoch"
        directory = result["output_directory"]
        history = pd.read_csv(directory / "loss_history.csv")
        manifest = json.loads((directory / "manifest.json").read_text())
        predictions = pd.read_csv(directory / "predictions.csv")
        metrics = pd.read_csv(directory / "per_curve_metrics.csv")
        assert history["validation_data"].notna().all()
        assert "held_out_data" not in history
        assert set(history["stage"]) == {"adam", "lbfgs"}
        adam_history = history.loc[history["stage"].eq("adam")]
        assert adam_history["physics_ramp"].tolist() == [0.0, 0.5, 1.0, 1.0, 1.0]
        contribution_columns = [
            f"contribution_{name}" for name in _training._ACTIVE_PHYSICS_TERMS
        ]
        np.testing.assert_allclose(
            adam_history["data"] + adam_history[contribution_columns].sum(axis=1),
            adam_history["total"],
            rtol=1e-10,
            atol=1e-14,
        )
        assert manifest["objective_conditioning"]["schedule"] == {
            "adam_epochs": 5,
            "warmup_epochs": 1,
            "ramp_epochs": 2,
            "full_objective_epochs": 2,
        }
        assert manifest["objective_conditioning"]["calibration_partition"].startswith(
            "training collocation"
        )
        for name in _training._ACTIVE_PHYSICS_TERMS:
            calibration_mse = manifest["objective_conditioning"]["calibration_mse"][
                name
            ]
            scale = manifest["objective_conditioning"]["frozen_scales"][name]
            assert scale == pytest.approx(max(1.0, np.sqrt(calibration_mse)))
            assert adam_history[f"scale_{name}"].nunique() == 1
            assert adam_history[f"scale_{name}"].iloc[0] == pytest.approx(scale)
        assert np.all(adam_history["initial_moisture"] == 0.0)
        assert np.all(adam_history["initial_temperature"] == 0.0)
        assert manifest["optimizer"]["lbfgs"]["accepted_state"]["losses"]["total"] >= 0
        assert manifest["optimizer"]["lbfgs"]["accepted_state"]["gradient_l2_norm"] >= 0
        assert manifest["optimizer"]["lbfgs"]["iterations"] <= 3
        assert manifest["optimizer"]["lbfgs"]["closure_evaluations"] <= 10
        assert manifest["train_ids"] == list(result["split"].train_ids)
        assert manifest["validation_ids"] == list(result["split"].validation_ids)
        assert manifest["test_ids"] == list(result["split"].test_ids)
        assert manifest["held_out_ids"] == list(result["split"].test_ids)
        assert "test_ids" in manifest["held_out_ids_role"]
        assert manifest["held_out_metrics_role"].startswith("Legacy compatibility")
        assert predictions["partition"].eq("test").all()
        assert metrics["partition"].eq("test").all()
        assert set(metrics["experiment_id"]) == set(result["split"].test_ids)
        assert (directory / "loss_history.png").is_file()
        assert (directory / "inferred_internal_fields.png").is_file()
        assert len(result["split"].train_ids) == 16
        assert len(result["split"].validation_ids) == 4
        assert len(result["split"].test_ids) == 4
        assert len(result["evaluation"].predictions["experiment_id"].unique()) == 4
        output = capsys.readouterr().out
        assert "16 training, 4 validation, and 4 final-test curves" in output
        assert "Final-test normalized metrics" in output
        assert "No temperature observations" in output
    finally:
        plt.close(result["loss_figure"])
        plt.close(result["internal_fields"].figure)


def test_locked_evaluation_reports_normalized_per_curve_and_equal_curve_metrics() -> (
    None
):
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)

    evaluation = evaluate_locked_model(trained, data, split, transform)

    assert evaluation.per_curve_metrics["experiment_id"].tolist() == ["curve_2"]
    assert evaluation.per_curve_metrics["partition"].tolist() == ["test"]
    assert evaluation.predictions["partition"].eq("test").all()
    assert evaluation.aggregate_metrics["curve_count"] == 1
    assert evaluation.aggregate_metrics["equal_curve_rmse"] == pytest.approx(
        evaluation.per_curve_metrics["rmse"].mean()
    )
    assert evaluation.predictions["moisture_observed_normalized"].notna().all()
    assert (
        evaluation.predictions["moisture_predicted_normalized"]
        .between(0.0, np.inf)
        .all()
    )
    assert "temperature_predicted_K" in evaluation.predictions
    expected_theta = (
        evaluation.predictions["temperature_predicted_K"]
        - PhysicsConfig().air_temperature_K
    ) / PhysicsConfig().signed_temperature_scale_K
    np.testing.assert_allclose(
        evaluation.predictions["temperature_predicted_theta"], expected_theta
    )


def test_locked_evaluation_preserves_quality_flags_and_mass_increase_points() -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)

    evaluation = evaluate_locked_model(trained, data, split, transform)

    increasing = evaluation.predictions.loc[
        evaluation.predictions["time_s"].eq(7200.0)
    ].iloc[0]
    assert increasing["quality_flags"] == "mass_increase"
    assert increasing["moisture_observed_normalized"] > 0.0
    assert len(evaluation.predictions) == 3


def test_evaluation_rejects_split_not_locked_by_training_provenance() -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    changed_split = ExperimentSplit(("curve_0", "curve_2"), ("curve_1",))

    with pytest.raises(ValueError, match="training provenance"):
        evaluate_locked_model(trained, data, changed_split, transform)


def test_evaluation_scores_after_training_with_equal_curve_weighting() -> None:
    data, _ = _prepared_data()
    observations = data.observations.copy(deep=True)
    fewer_points = observations["experiment_id"].eq("curve_1") & observations[
        "time_s"
    ].eq(7200.0)
    observations.loc[fewer_points, "target_available"] = False
    data = CurveData(data.experiments, observations)
    split = ExperimentSplit(("curve_0",), ("curve_1", "curve_2"))
    transform = fit_feature_transform(data, split)
    training = train_sequentially(
        None,
        data,
        split,
        transform,
        sampling_config=SamplingConfig(
            domain_points=2,
            initial_points=2,
            boundary_points=2,
            seed=13,
        ),
        training_config=_bounded_training_config(
            adam_epochs=1,
            lbfgs_max_iter=1,
            lbfgs_max_eval=2,
            seed=13,
            device="cpu",
            quadrature_points=2,
        ),
    )

    evaluation = evaluate_locked_model(training, data, split, transform)

    assert training.stage_history["adam"].epochs == 1
    assert training.provenance.held_out_ids == (split.validation_ids + split.test_ids)
    assert evaluation.per_curve_metrics["observation_count"].tolist() == [2, 3]
    assert evaluation.aggregate_metrics["equal_curve_mae"] == pytest.approx(
        evaluation.per_curve_metrics["mae"].mean()
    )
    assert evaluation.aggregate_metrics["equal_curve_rmse"] == pytest.approx(
        evaluation.per_curve_metrics["rmse"].mean()
    )


def test_evaluation_does_not_use_held_out_values_to_change_training_provenance() -> (
    None
):
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    changed_observations = data.observations.copy(deep=True)
    held_out = changed_observations["experiment_id"].isin(split.test_ids)
    changed_observations.loc[held_out, "sample_mass_kg"] = [0.020, 0.018, 0.015]
    changed_observations.loc[held_out, "moisture_db"] = [1.0, 0.8, 0.5]
    changed_observations.loc[held_out, "moisture_normalized"] = [1.0, 0.8, 0.5]
    changed_data = CurveData(data.experiments, changed_observations)

    evaluation = evaluate_locked_model(trained, changed_data, split, transform)

    assert evaluation.predictions["moisture_observed_normalized"].tolist() == [
        1.0,
        0.8,
        0.5,
    ]
    assert trained.provenance.train_ids == split.train_ids


def test_unavailable_out_of_domain_rows_are_audited_but_not_predicted_or_scored() -> (
    None
):
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    observations = data.observations.copy(deep=True)
    observations.loc[len(observations)] = {
        "experiment_id": "curve_2",
        "time_s": 18_000.0,
        "sample_mass_kg": 0.015,
        "target_available": False,
        "quality_flags": "outside_model_domain;invalid_measurement",
        "moisture_db": 0.5,
        "initial_moisture_db": 1.0,
        "moisture_normalized": 0.5,
        "time_normalized": 18_000.0 / 14_400.0,
        "invalid_reason": "outside_model_domain",
    }
    data = CurveData(data.experiments, observations)

    evaluation = evaluate_locked_model(trained, data, split, transform)

    unavailable = evaluation.predictions.loc[
        evaluation.predictions["time_s"].eq(18_000.0)
    ].iloc[0]
    assert not unavailable["target_available"]
    assert unavailable["quality_flags"] == ("outside_model_domain;invalid_measurement")
    assert np.isnan(unavailable["moisture_predicted_normalized"])
    assert np.isnan(unavailable["temperature_predicted_K"])
    assert evaluation.per_curve_metrics.iloc[0]["observation_count"] == 3


def test_available_out_of_domain_target_is_rejected_with_domain_reason() -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    observations = data.observations.copy(deep=True)
    observations.loc[len(observations)] = {
        "experiment_id": "curve_2",
        "time_s": 18_000.0,
        "sample_mass_kg": 0.015,
        "target_available": True,
        "quality_flags": "",
        "moisture_db": 0.5,
        "initial_moisture_db": 1.0,
        "moisture_normalized": 0.5,
        "time_normalized": 18_000.0 / 14_400.0,
        "invalid_reason": "",
    }

    with pytest.raises(ValueError, match="target_available.*outside modeled domain"):
        evaluate_locked_model(
            trained, CurveData(data.experiments, observations), split, transform
        )


def test_only_available_out_of_domain_targets_report_domain_reason() -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    observations = data.observations.copy(deep=True)
    held_out = observations["experiment_id"].eq("curve_2")
    observations.loc[held_out, "time_s"] = 18_000.0
    observations.loc[held_out, "target_available"] = False
    available_row = observations.index[held_out][0]
    observations.loc[available_row, "target_available"] = True

    with pytest.raises(ValueError, match="target_available.*outside modeled domain"):
        evaluate_locked_model(
            trained, CurveData(data.experiments, observations), split, transform
        )


def test_plot_excludes_out_of_domain_observations_and_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = importlib.import_module("strategy_c.reporting_artifacts")
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    observations = data.observations.copy(deep=True)
    observations.loc[len(observations)] = {
        "experiment_id": "curve_2",
        "time_s": 18_000.0,
        "sample_mass_kg": 0.015,
        "target_available": False,
        "quality_flags": "outside_model_domain",
        "moisture_db": 0.5,
        "initial_moisture_db": 1.0,
        "moisture_normalized": 0.5,
        "time_normalized": 18_000.0 / 14_400.0,
        "invalid_reason": "outside_model_domain",
    }
    evaluation = evaluate_locked_model(
        trained, CurveData(data.experiments, observations), split, transform
    )
    captured: dict[str, list[np.ndarray]] = {"lines": [], "points": []}

    def capture_figure(figure: object, destination: Path, **kwargs: object) -> None:
        axes = figure.axes[0]
        captured["lines"] = [
            np.asarray(line.get_xdata(), dtype=float) for line in axes.lines
        ]
        captured["points"] = [
            np.asarray(collection.get_offsets(), dtype=float)[:, 0]
            for collection in axes.collections
        ]

    monkeypatch.setattr(artifacts.Figure, "savefig", capture_figure)
    artifacts._write_curve_plot(evaluation.predictions, tmp_path / "plot.png")

    assert all(not np.any(values > 1.0) for values in captured["lines"])
    assert all(not np.any(values > 1.0) for values in captured["points"])

    monkeypatch.undo()
    run_directory = export_results(
        evaluation,
        trained,
        CurveData(data.experiments, observations),
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
        run_timestamp=datetime(2026, 11, 16, 23, 1, tzinfo=timezone.utc),
    )
    exported_predictions = pd.read_csv(run_directory / "predictions.csv")
    audit_row = exported_predictions.loc[
        exported_predictions["time_s"].eq(18_000.0)
    ].iloc[0]
    assert not audit_row["target_available"]
    assert audit_row["quality_flags"] == "outside_model_domain"
    assert pd.isna(audit_row["moisture_predicted_normalized"])
    assert pd.isna(audit_row["temperature_predicted_K"])
    assert (run_directory / "normalized_drying_curves.png").is_file()

    unavailable_index = evaluation.predictions.index[
        evaluation.predictions["time_s"].eq(18_000.0)
    ][0]
    evaluation.predictions.loc[unavailable_index, "target_available"] = True
    with pytest.raises(ValueError, match="target_available.*outside modeled domain"):
        export_results(
            evaluation,
            trained,
            CurveData(data.experiments, observations),
            tmp_path / "rejected_artifacts",
            training_config=_bounded_training_config(),
            feature_transform=transform,
        )
    assert not (tmp_path / "rejected_artifacts").exists()


def test_export_writes_manifest_curves_and_plot_without_fake_empirical_data(
    tmp_path: Path,
) -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    run_time = datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc)

    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        sampling_config=SamplingConfig(
            domain_points=2, initial_points=2, boundary_points=2
        ),
        feature_transform=transform,
        run_timestamp=run_time,
    )

    assert run_directory.is_dir()
    assert (run_directory / "normalized_drying_curves.png").is_file()
    assert (run_directory / "predictions.csv").is_file()
    assert (run_directory / "per_curve_metrics.csv").is_file()
    assert (run_directory / "quality_flags.csv").is_file()
    assert (run_directory / "conservation_diagnostics.csv").is_file()
    assert (run_directory / "model_state.pt").is_file()
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_ids"] == list(split.train_ids)
    assert manifest["held_out_ids"] == list(split.test_ids)
    assert manifest["validation_ids"] == list(split.validation_ids)
    assert manifest["test_ids"] == list(split.test_ids)
    assert (
        manifest["optimizer"]["adam"]["epochs"]
        == _bounded_training_config().adam_epochs
    )
    assert "loss_history.png" in manifest["artifacts"]
    assert (run_directory / "loss_history.png").is_file()
    assert manifest["loss_reporting"]["temperature_observations"] is False
    assert manifest["optimizer"]["adam"]["learning_rate"] == pytest.approx(0.001)
    assert manifest["optimizer"]["adam"][
        "physics_parameter_learning_rate"
    ] == pytest.approx(0.05)
    assert manifest["optimizer"]["training_configuration"][
        "physics_parameter_learning_rate"
    ] == pytest.approx(0.05)
    physics_units = manifest["physics_configuration"]
    assert physics_units["water_specific_heat_j_kg_k"] == {
        "value": 4180.0,
        "unit": "J kg-1 K-1",
    }
    assert "heat_flux_scale_w_m2" not in physics_units
    assert "latent_heat_j_kg" not in physics_units
    assert "initial_activation_energy_j_mol" not in physics_units
    assert physics_units["activation_energy_j_mol"]["unit"] == "J mol-1"
    assert manifest["learned_parameters"]["reference_diffusivity_m2_s"] > 0.0
    assert not any(
        name.startswith("activation_energy") for name in manifest["learned_parameters"]
    )
    fixed = manifest["fixed_parameters"]
    assert fixed["activation_energy_j_mol"] == PhysicsConfig().activation_energy_j_mol
    assert fixed["activation_energy_over_gas_constant_K"] == pytest.approx(2044.0)
    assert fixed["activation_energy_provenance"] == (
        "fixed, Barnard et al. 2024 Table 1 (2044 K x R)"
    )
    assert fixed["activation_energy_trainable"] is False
    assert (
        manifest["conservation_diagnostics"]["units"][
            "surface_vaporization_enthalpy_j_kg"
        ]
        == "J kg-1"
    )
    assert any("potato" in note for note in manifest["assumptions_and_limitations"])
    assert manifest["learned_parameters"]["diameter_exponent_alpha"] == pytest.approx(
        trained.model.diameter_exponent.item()
    )
    assert manifest["learned_parameters"]["speed_exponent_beta"] == pytest.approx(
        trained.model.speed_exponent.item()
    )
    assert manifest["format_version"] == 6
    assert manifest["model"]["schema"] == "strategy-c-local-fields-v1"
    assert manifest["model"]["architecture"]["moisture_readout"]["formula"] == (
        "u = exp(tau*f_X)"
    )
    assert manifest["objective_conditioning"]["data_relative_physics_discount"] is False
    assert manifest["objective_conditioning"]["schedule"]["full_objective_epochs"] == 1
    assert manifest["quadrature"]["training_points"] == 2
    assert manifest["quadrature"]["evaluation_points"] == 64
    assert manifest["quadrature"]["conservation_diagnostic_points"] == 32
    assert manifest["optimizer"]["lbfgs"]["termination_reason_inferred"] is False
    assert manifest["optimizer"]["lbfgs"]["accepted_state"]["gradient_l2_norm"] == 0.04
    assert manifest["model"]["temperature_normalization"]["signed_scale_K"] == -45.0
    assert manifest["temperature_reporting"]["unbounded_and_unclipped"]
    assert manifest["conservation_diagnostics"]["not_an_exact_conservation_claim"]
    assert manifest["transfer_coefficients"]
    assert manifest["quality_flag_counts_all_curves"]["mass_increase"] == 3
    assert not list(run_directory.glob("*empirical*"))
    assert "was supplied" in manifest["empirical_comparison"]
    assert manifest["feature_transform"]["feature_names"] == list(
        transform.feature_names
    )
    assert manifest["feature_transform"]["mean"] == list(transform.mean)
    assert manifest["feature_transform"]["scale"] == list(transform.scale)

    reconstructed = load_model_artifacts(run_directory)
    reconstructed_training = TrainingResult(
        model=reconstructed,
        stage_history=trained.stage_history,
        provenance=trained.provenance,
    )
    reconstructed_transform = _data.FeatureTransform(
        feature_names=tuple(manifest["feature_transform"]["feature_names"]),
        mean=tuple(manifest["feature_transform"]["mean"]),
        scale=tuple(manifest["feature_transform"]["scale"]),
    )
    reconstructed_evaluation = evaluate_locked_model(
        reconstructed_training, data, split, reconstructed_transform
    )
    np.testing.assert_array_equal(
        reconstructed_evaluation.predictions[
            [
                "moisture_predicted_normalized",
                "temperature_predicted_K",
                "temperature_predicted_theta",
            ]
        ].to_numpy(),
        evaluation.predictions[
            [
                "moisture_predicted_normalized",
                "temperature_predicted_K",
                "temperature_predicted_theta",
            ]
        ].to_numpy(),
    )


def test_reported_thermal_diffusivity_uses_wet_bed_capacity_at_initial_moisture() -> (
    None
):
    """Condition tables have no local field, so alpha_T is evaluated at X0."""
    artifacts = importlib.import_module("strategy_c.reporting_artifacts")
    data, _ = _prepared_data()
    config = PhysicsConfig()

    rows, report = artifacts._physics_report(data, config, 1.0e-7, 0.0, 0.0)

    for row, (_, experiment) in zip(rows, data.experiments.iterrows()):
        wet_capacity = float(experiment["dry_bulk_density_kg_m3"]) * (
            config.product_specific_heat_j_kg_k
            + float(experiment["initial_moisture_db"])
            * config.water_specific_heat_j_kg_k
        )
        expected_alpha = config.product_thermal_conductivity_w_m_k / wet_capacity
        assert row["thermal_diffusivity_m2_s_provisional"] == pytest.approx(
            expected_alpha, rel=1e-12
        )
        assert row["thermal_fourier_number"] == pytest.approx(
            expected_alpha * config.time_scale_s / config.bed_depth_m**2,
            rel=1e-12,
        )
    definition = report["dimensionless_scale_definitions"]["thermal_fourier_number"]
    assert "initial moisture" in definition
    assert "cp_w" in definition


@pytest.mark.parametrize("schema_value", [None, "strategy-c-v0"])
def test_model_loader_rejects_missing_or_old_schema_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_value: str | None,
) -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
    )
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if schema_value is None:
        manifest["model"].pop("schema")
    else:
        manifest["model"]["schema"] = schema_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    load_called = False

    def forbidden_load(*args: object, **kwargs: object) -> None:
        nonlocal load_called
        load_called = True
        raise AssertionError("tensor loading must not run for an incompatible schema")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="Incompatible or missing model schema"):
        load_model_artifacts(run_directory)
    assert not load_called


@pytest.mark.parametrize("old_version", [3, 4, 5])
def test_model_loader_rejects_previous_format_versions_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old_version: int,
) -> None:
    """Direct-field loader rejects every previous artifact format."""
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
    )
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Only the declared version changes; the schema stays valid so the
    # rejection is attributable to the format-version gate alone.
    manifest["format_version"] = old_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor loading must not run for an old format")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="Incompatible artifact format version"):
        load_model_artifacts(run_directory)


def test_model_loader_rejects_tampered_direct_field_readout_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct-field manifest cannot relabel a different readout as current."""
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
    )
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["architecture"]["moisture_readout"]["formula"] = (
        "degree-eight Bernstein"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    load_called = False

    def forbidden_load(*args: object, **kwargs: object) -> None:
        nonlocal load_called
        load_called = True
        raise AssertionError("tensor loading must not run for invalid readout metadata")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="readout metadata"):
        load_model_artifacts(run_directory)
    assert not load_called


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("fixed_parameters", "activation_energy_provenance", "learned", "fixed"),
        ("fixed_parameters", "activation_energy_j_mol", 30_000.0, "fixed"),
        ("fixed_parameters", "activation_energy_trainable", True, "fixed"),
        ("learned_parameters", "activation_energy_j_mol", 16_994.76, "learned"),
    ],
)
def test_model_loader_rejects_ea_reported_as_learned_or_altered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    """The fixed Barnard Ea must keep its value, provenance, and fixed label."""
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
    )
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[section][key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor loading must not run for invalid Ea metadata")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match=message):
        load_model_artifacts(run_directory)


def test_model_loader_rejects_state_with_a_different_activation_energy(
    tmp_path: Path,
) -> None:
    """A tampered Ea buffer cannot silently override the fixed configuration."""
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
    )
    state_path = run_directory / "model_state.pt"
    state = torch.load(state_path, weights_only=True)
    state["fixed_activation_energy_j_mol"] = torch.tensor(
        30_000.0, dtype=state["fixed_activation_energy_j_mol"].dtype
    )
    torch.save(state, state_path)

    with pytest.raises(ValueError, match="activation energy"):
        load_model_artifacts(run_directory)


def test_export_optionally_persists_training_internal_fields(
    tmp_path: Path,
) -> None:
    """The optional field addition records paths/scales without changing scores."""
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    internal_fields = infer_internal_fields(
        trained.model,
        data,
        split.train_ids,
        transform,
        trained.model.config,
        time_points=3,
        depth_points=3,
    )

    run_directory = export_results(
        evaluation,
        trained,
        data,
        tmp_path / "artifacts",
        training_config=_bounded_training_config(),
        feature_transform=transform,
        internal_fields=internal_fields,
    )

    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    field_manifest = manifest["internal_field_visualization"]
    assert field_manifest["selected_experiment_id"] in split.train_ids
    assert field_manifest["figure_file"] == "inferred_internal_fields.png"
    assert field_manifest["grid_file"] == "inferred_internal_fields.npz"
    assert field_manifest["scales"]["time_scale_s"] == 14_400.0
    assert field_manifest["profile_times_s"] == [0.0, 720.0, 3600.0, 7200.0, 14400.0]
    assert (run_directory / field_manifest["figure_file"]).is_file()
    assert (run_directory / field_manifest["grid_file"]).is_file()
    assert field_manifest["selected_experiment_id"] in manifest["train_ids"]


def test_export_rejects_existing_timestamp_directory_without_overwriting(
    tmp_path: Path,
) -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    root = tmp_path / "artifacts"
    root.mkdir()
    run_time = datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc)
    existing = root / "20261116T230000.000000Z"
    existing.mkdir()
    protected = existing / "user-file.txt"
    protected.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_results(
            evaluation,
            trained,
            data,
            root,
            training_config=_bounded_training_config(),
            feature_transform=transform,
            run_timestamp=run_time,
        )

    assert protected.read_text(encoding="utf-8") == "keep"


def test_export_rejects_nonfinite_predictions_before_creating_artifacts(
    tmp_path: Path,
) -> None:
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)
    evaluation = evaluate_locked_model(trained, data, split, transform)
    evaluation.predictions.loc[0, "moisture_predicted_normalized"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        export_results(
            evaluation,
            trained,
            data,
            tmp_path / "artifacts",
            training_config=_bounded_training_config(),
            feature_transform=transform,
        )

    assert not (tmp_path / "artifacts").exists()
