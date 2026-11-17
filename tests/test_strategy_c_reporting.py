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
CurveData = _data.CurveData
ExperimentSplit = _data.ExperimentSplit
Scales = _data.Scales
fit_feature_transform = _data.fit_feature_transform
prepare_curve_data = _data.prepare_curve_data
CoupledInversePINN = _physics.CoupledInversePINN
PhysicsConfig = _physics.PhysicsConfig
evaluate_locked_model = _reporting.evaluate_locked_model
export_results = _reporting.export_results
SamplingConfig = _training.SamplingConfig
StageHistory = _training.StageHistory
TrainingConfig = _training.TrainingConfig
TrainingProvenance = _training.TrainingProvenance
TrainingResult = _training.TrainingResult
train_sequentially = _training.train_sequentially


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
    return data, ExperimentSplit(("curve_0", "curve_1"), ("curve_2",))


def _training_result(
    data: CurveData, split: ExperimentSplit
) -> tuple[TrainingResult, object]:
    """Build deterministic test parameters with strict train/test provenance."""
    transform = fit_feature_transform(data, split)
    model = CoupledInversePINN(PhysicsConfig()).double()
    result = TrainingResult(
        model=model,
        stage_history={
            "adam": StageHistory(
                epochs=50,
                elapsed_seconds=1.25,
                losses=[{"total": 0.25}],
            ),
            "lbfgs": StageHistory(
                iterations=2,
                closure_calls=3,
                elapsed_seconds=0.5,
                losses=[{"total": 0.2}],
            ),
        },
        provenance=TrainingProvenance(
            seed=13,
            train_ids=split.train_ids,
            held_out_ids=split.test_ids,
            training_observation_count=6,
            collocation_point_count=8,
            device="cpu",
            dtype="torch.float64",
        ),
    )
    return result, transform


def test_locked_evaluation_reports_normalized_per_curve_and_equal_curve_metrics() -> (
    None
):
    data, split = _prepared_data()
    trained, transform = _training_result(data, split)

    evaluation = evaluate_locked_model(trained, data, split, transform)

    assert evaluation.per_curve_metrics["experiment_id"].tolist() == ["curve_2"]
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
        training_config=TrainingConfig(
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
    assert training.provenance.held_out_ids == split.test_ids
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
        training_config=TrainingConfig(device="cpu"),
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
    assert (run_directory / "model_state.pt").is_file()
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_ids"] == list(split.train_ids)
    assert manifest["held_out_ids"] == list(split.test_ids)
    assert manifest["optimizer"]["adam"]["epochs"] == 50
    assert manifest["learned_parameters"]["reference_diffusivity_m2_s"] > 0.0
    assert manifest["transfer_coefficients"]
    assert manifest["quality_flag_counts_all_curves"]["mass_increase"] == 3
    assert not list(run_directory.glob("*empirical*"))
    assert "was supplied" in manifest["empirical_comparison"]
    assert manifest["feature_transform"]["feature_names"] == list(
        transform.feature_names
    )
    assert manifest["feature_transform"]["mean"] == list(transform.mean)
    assert manifest["feature_transform"]["scale"] == list(transform.scale)

    model_metadata = manifest["model"]
    reconstructed = CoupledInversePINN(
        PhysicsConfig(**model_metadata["physics_config"])
    ).to(dtype=torch.float64)
    saved_state = torch.load(
        run_directory / "model_state.pt", map_location="cpu", weights_only=True
    )
    reconstructed.load_state_dict(saved_state, strict=True)
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
            ["moisture_predicted_normalized", "temperature_predicted_K"]
        ].to_numpy(),
        evaluation.predictions[
            ["moisture_predicted_normalized", "temperature_predicted_K"]
        ].to_numpy(),
    )


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
            feature_transform=transform,
        )

    assert not (tmp_path / "artifacts").exists()
