"""Train-only PINN field inference and Raissi-inspired internal-field plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from strategy_c.data import CurveData, FeatureTransform
from strategy_c.physics import PhysicsConfig


_REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    "air_speed_m_s",
    "particle_diameter_m",
    "initial_moisture_db",
    "dry_bulk_density_kg_m3",
)
_PROFILE_FRACTIONS: Final[tuple[float, ...]] = (0.0, 0.05, 0.25, 0.5, 1.0)
_REPLICATE_COLUMNS: Final[tuple[str, ...]] = (
    "replicate",
    "replicate_id",
    "replicate_number",
    "replicate_no",
)


@dataclass(frozen=True)
class InternalFieldResult:
    """Dimensionless fields, fixed-time profiles, and their explanatory figure."""

    selected_experiment_id: str
    selected_features: dict[str, float]
    time_normalized: np.ndarray
    depth_normalized: np.ndarray
    moisture_ratio: np.ndarray
    temperature_normalized: np.ndarray
    temperature_kelvin: np.ndarray
    profile_times_normalized: np.ndarray
    profile_moisture_ratio: np.ndarray
    profile_temperature_normalized: np.ndarray
    profile_temperature_kelvin: np.ndarray
    physics_config: PhysicsConfig
    figure: Figure


@dataclass(frozen=True)
class InternalFieldArtifactPaths:
    """Paths of the exported field figure and numeric archive."""

    figure: Path
    grid: Path


def _validate_training_ids(data: CurveData, train_ids: tuple[str, ...]) -> pd.DataFrame:
    """Return only metadata rows for the fitted IDs, never touching observations."""
    if not isinstance(data, CurveData):
        raise TypeError("data must be CurveData.")
    if not isinstance(train_ids, tuple) or not train_ids:
        raise ValueError("train_ids must be a nonempty tuple of experiment IDs.")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Training experiment IDs must be unique.")
    if any(
        not isinstance(identifier, str) or not identifier for identifier in train_ids
    ):
        raise ValueError("Training experiment IDs must be nonempty strings.")
    experiments = data.experiments
    required = ("experiment_id", *_REQUIRED_FEATURES)
    missing = [column for column in required if column not in experiments.columns]
    if missing:
        raise ValueError(f"Experiment metadata is missing required columns: {missing}.")
    identifiers = experiments["experiment_id"].astype(str)
    if identifiers.duplicated().any():
        raise ValueError("Experiment identifiers must be unique.")
    absent = set(train_ids).difference(identifiers)
    if absent:
        raise ValueError(f"Training IDs are absent from experiment metadata: {absent}.")
    training = experiments.loc[identifiers.isin(train_ids)].copy()
    numeric_features = training.loc[:, _REQUIRED_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Training experiment features must be finite numeric values.")
    training.loc[:, _REQUIRED_FEATURES] = numeric_features
    training["_selection_id"] = training["experiment_id"].astype(str)
    return training


def select_representative_training_experiment(
    data: CurveData, train_ids: tuple[str, ...]
) -> pd.Series:
    """Select the lowest-speed/size training replicate deterministically.

    Sorting on actual training metadata avoids averaging unlike conditions;
    held-out observations and moisture targets are not inspected or needed.
    """
    training = _validate_training_ids(data, train_ids)
    sort_columns = ["air_speed_m_s", "particle_diameter_m"]
    replicate_column = next(
        (name for name in _REPLICATE_COLUMNS if name in training.columns), None
    )
    if replicate_column is not None:
        replicate_values = training[replicate_column]
        numeric_replicates = pd.to_numeric(replicate_values, errors="coerce")
        if numeric_replicates.notna().equals(replicate_values.notna()):
            training["_selection_replicate"] = numeric_replicates
        else:
            training["_selection_replicate"] = replicate_values.fillna("").astype(str)
        sort_columns.append("_selection_replicate")
    sort_columns.append("_selection_id")
    return training.sort_values(sort_columns, kind="mergesort").iloc[0].copy()


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer uniform model tensor placement without moving trained parameters."""
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


def _predict_fields(
    model: nn.Module,
    tau_values: np.ndarray,
    xi_values: np.ndarray,
    normalized_features: np.ndarray,
    config: PhysicsConfig,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run bounded batches and return moisture, Incropera theta, and Kelvin."""
    tau_grid, xi_grid = np.meshgrid(tau_values, xi_values, indexing="xy")
    flat_tau = tau_grid.reshape(-1)
    flat_xi = xi_grid.reshape(-1)
    moisture_chunks: list[np.ndarray] = []
    temperature_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, flat_tau.size, chunk_size):
            stop = min(start + chunk_size, flat_tau.size)
            tau_tensor = torch.as_tensor(
                flat_tau[start:stop], device=device, dtype=dtype
            ).reshape(-1, 1)
            xi_tensor = torch.as_tensor(
                flat_xi[start:stop], device=device, dtype=dtype
            ).reshape(-1, 1)
            features_tensor = torch.as_tensor(
                np.repeat(normalized_features.reshape(1, -1), stop - start, axis=0),
                device=device,
                dtype=dtype,
            )
            outputs = model(tau_tensor, xi_tensor, features_tensor)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise ValueError("The trained model must return (X/X0, temperature_K).")
            moisture, temperature = outputs
            expected_shape = (stop - start, 1)
            if (
                not isinstance(moisture, Tensor)
                or not isinstance(temperature, Tensor)
                or moisture.shape != expected_shape
                or temperature.shape != expected_shape
            ):
                raise ValueError("Model outputs must each have shape (N, 1).")
            moisture_chunks.append(moisture.detach().cpu().numpy().reshape(-1))
            temperature_chunks.append(temperature.detach().cpu().numpy().reshape(-1))
    moisture_values = np.concatenate(moisture_chunks).reshape(len(xi_values), -1)
    temperature_kelvin = np.concatenate(temperature_chunks).reshape(len(xi_values), -1)
    temperature_values = (
        temperature_kelvin - config.air_temperature_K
    ) / config.signed_temperature_scale_K
    if (
        not np.isfinite(moisture_values).all()
        or not np.isfinite(temperature_values).all()
    ):
        raise ValueError("Inferred internal fields must contain only finite values.")
    return moisture_values, temperature_values, temperature_kelvin


def _field_figure(
    result_data: dict[str, object],
    config: PhysicsConfig,
) -> Figure:
    """Create aligned, two-variable field maps and depth-profile panels."""
    time_values = result_data["time_normalized"]
    depth_values = result_data["depth_normalized"]
    moisture = result_data["moisture_ratio"]
    temperature = result_data["temperature_normalized"]
    profile_times = result_data["profile_times_normalized"]
    profile_moisture = result_data["profile_moisture_ratio"]
    profile_temperature = result_data["profile_temperature_normalized"]
    temperature_kelvin = result_data["temperature_kelvin"]
    profile_temperature_kelvin = result_data["profile_temperature_kelvin"]
    assert isinstance(time_values, np.ndarray)
    assert isinstance(depth_values, np.ndarray)
    assert isinstance(moisture, np.ndarray)
    assert isinstance(temperature, np.ndarray)
    assert isinstance(profile_times, np.ndarray)
    assert isinstance(profile_moisture, np.ndarray)
    assert isinstance(profile_temperature, np.ndarray)
    assert isinstance(temperature_kelvin, np.ndarray)
    assert isinstance(profile_temperature_kelvin, np.ndarray)
    selected_id = str(result_data["selected_experiment_id"])
    features = result_data["selected_features"]
    assert isinstance(features, dict)

    # Include the 0 and 1 reference states while retaining every model overshoot.
    moisture_norm = Normalize(
        vmin=min(0.0, 1.0, float(np.min(moisture))),
        vmax=max(0.0, 1.0, float(np.max(moisture))),
    )
    temperature_norm = Normalize(
        vmin=min(0.0, 1.0, float(np.min(temperature))),
        vmax=max(0.0, 1.0, float(np.max(temperature))),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), layout="constrained")
    # pcolormesh places the supplied inclusive coordinates at the pixel
    # centers; imshow's extent would instead shift those boundary samples.
    moisture_map = axes[0, 0].pcolormesh(
        time_values,
        depth_values,
        moisture,
        shading="nearest",
        cmap="viridis",
        norm=moisture_norm,
    )
    temperature_map = axes[0, 1].pcolormesh(
        time_values,
        depth_values,
        temperature,
        shading="nearest",
        cmap="cividis",
        norm=temperature_norm,
    )
    axes[0, 0].set(
        title="Normalized moisture field",
        xlabel="Dimensionless time, t / time scale",
        ylabel="Dimensionless depth, z / bed depth",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axes[0, 1].set(
        title="temperature field",
        xlabel="Dimensionless time, t / time scale",
        ylabel="Dimensionless depth, z / bed depth",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    figure.colorbar(moisture_map, ax=axes[0, 0], label="Moisture ratio, X/X0")
    figure.colorbar(
        temperature_map,
        ax=axes[0, 1],
        label="Normalized temperature, θ",
    )

    # Color and line pattern both distinguish the five requested profile times.
    # This five-color subset of Matplotlib's tab10 palette is paired with
    # line patterns and markers so profiles remain distinguishable in print.
    profile_colors: tuple[str, ...] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    )
    profile_styles = ("-", "--", "-.", ":", (0, (5, 1)))
    for index, fraction in enumerate(profile_times):
        minutes = fraction * config.time_scale_s / 60.0
        label = f"{minutes:g} min"
        style = {
            "color": profile_colors[index],
            "linestyle": profile_styles[index],
            "linewidth": 1.9,
            "marker": ("o", "s", "^", "D", "v")[index],
            "markevery": max(1, len(depth_values) // 8),
            "markersize": 4,
            "label": label,
        }
        axes[1, 0].plot(profile_moisture[index], depth_values, **style)
        axes[1, 1].plot(profile_temperature[index], depth_values, **style)
    axes[1, 0].set(
        title="Fixed-time moisture depth profiles",
        xlabel="Moisture ratio, X/X0",
        ylabel="Dimensionless depth, z / bed depth",
        ylim=(0.0, 1.0),
    )
    axes[1, 1].set(
        title="Fixed-time temperature depth profiles",
        xlabel="Normalized temperature, θ",
        ylabel="Dimensionless depth, z / bed depth",
        ylim=(0.0, 1.0),
    )
    for axis in axes[1]:
        axis.grid(True, alpha=0.25)
        axis.legend(title="Time (min)", fontsize=8, ncol=2)
    caption = (
        "PINN-inferred internal fields; not validated against internal measurements\n"
        f"Training experiment {selected_id} · air speed {features['air_speed_m_s']:.6g} m/s · "
        f"particle diameter {features['particle_diameter_m'] * 1e6:.6g} µm\n"
        f"Scales: t / {config.time_scale_s:g} s; z / {config.bed_depth_m:g} m; "
        f"X/X0\n"
        f"θ=(T−{config.air_temperature_K:g} K)/"
        f"({config.initial_temperature_K:g}−{config.air_temperature_K:g}) K\n"
        "θ=1 initially, θ=0 at air temperature; overshoots are not clipped.\n"
        "Bounded smoke-run training "
        "does not establish convergence."
    )
    figure.suptitle(caption, fontsize=11)
    return figure


def infer_internal_fields(
    model: nn.Module,
    data: CurveData,
    train_ids: tuple[str, ...],
    feature_transform: FeatureTransform,
    physics_config: PhysicsConfig | None = None,
    *,
    time_points: int = 201,
    depth_points: int = 101,
    chunk_size: int = 8192,
) -> InternalFieldResult:
    """Infer and plot fields from one actual training curve, without targets.

    Inputs to the network use its existing train-fitted feature transform and
    model dtype/device. The model's individual train/eval flags are restored
    after inference, and inference mode prevents gradient or optimizer work.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(feature_transform, FeatureTransform):
        raise TypeError("feature_transform must be FeatureTransform.")
    config = (
        getattr(model, "config", None) if physics_config is None else physics_config
    )
    if config is None:
        raise ValueError(
            "physics_config is required when the model does not expose its config."
        )
    if not isinstance(config, PhysicsConfig):
        raise TypeError("physics_config must be PhysicsConfig.")
    for name, value in (
        ("time_points", time_points),
        ("depth_points", depth_points),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least two.")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer.")
    if feature_transform.feature_names != _REQUIRED_FEATURES:
        raise ValueError("feature_transform feature order must match the model inputs.")

    selected = select_representative_training_experiment(data, train_ids)
    selected_id = str(selected["experiment_id"])
    selected_features = {name: float(selected[name]) for name in _REQUIRED_FEATURES}
    normalized_features = feature_transform.transform(
        pd.DataFrame.from_records([selected_features])
    )[0]
    time_values = np.linspace(0.0, 1.0, time_points)
    depth_values = np.linspace(0.0, 1.0, depth_points)
    profile_times = np.asarray(_PROFILE_FRACTIONS, dtype=float)
    device, dtype = _model_device_dtype(model)

    # Preserve mixed child-module modes exactly; eval behavior avoids stateful
    # layers updating during a diagnostic forward pass.
    module_modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        moisture, temperature, temperature_kelvin = _predict_fields(
            model,
            time_values,
            depth_values,
            normalized_features,
            config,
            device,
            dtype,
            chunk_size,
        )
        profile_moisture, profile_temperature, profile_temperature_kelvin = (
            _predict_fields(
                model,
                profile_times,
                depth_values,
                normalized_features,
                config,
                device,
                dtype,
                chunk_size,
            )
        )
        # _predict_fields orders rows by depth and columns by time. The profile
        # archive and plotting loop use one profile per row for direct indexing.
        profile_moisture = profile_moisture.T
        profile_temperature = profile_temperature.T
        profile_temperature_kelvin = profile_temperature_kelvin.T
    finally:
        for module, was_training in module_modes:
            module.training = was_training

    result_data: dict[str, object] = {
        "selected_experiment_id": selected_id,
        "selected_features": selected_features,
        "time_normalized": time_values,
        "depth_normalized": depth_values,
        "moisture_ratio": moisture,
        "temperature_normalized": temperature,
        "temperature_kelvin": temperature_kelvin,
        "profile_times_normalized": profile_times,
        "profile_moisture_ratio": profile_moisture,
        "profile_temperature_normalized": profile_temperature,
        "profile_temperature_kelvin": profile_temperature_kelvin,
    }
    figure = _field_figure(result_data, config)
    return InternalFieldResult(
        selected_experiment_id=selected_id,
        selected_features=selected_features,
        time_normalized=time_values,
        depth_normalized=depth_values,
        moisture_ratio=moisture,
        temperature_normalized=temperature,
        temperature_kelvin=temperature_kelvin,
        profile_times_normalized=profile_times,
        profile_moisture_ratio=profile_moisture,
        profile_temperature_normalized=profile_temperature,
        profile_temperature_kelvin=profile_temperature_kelvin,
        physics_config=config,
        figure=figure,
    )


def write_internal_field_artifacts(
    result: InternalFieldResult, output_directory: str | Path
) -> InternalFieldArtifactPaths:
    """Save an explanatory PNG and replayable normalized/physical grid archive."""
    if not isinstance(result, InternalFieldResult):
        raise TypeError("result must be InternalFieldResult.")
    coordinate_arrays = (
        result.time_normalized,
        result.depth_normalized,
        result.profile_times_normalized,
        result.moisture_ratio,
        result.temperature_normalized,
        result.temperature_kelvin,
        result.profile_moisture_ratio,
        result.profile_temperature_normalized,
        result.profile_temperature_kelvin,
    )
    if any(not isinstance(values, np.ndarray) for values in coordinate_arrays):
        raise TypeError("Internal-field coordinates and fields must be NumPy arrays.")
    if any(
        not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()
        for values in coordinate_arrays
    ):
        raise ValueError(
            "Internal-field coordinates and fields must be finite numeric arrays."
        )
    if (
        ((result.time_normalized < 0.0) | (result.time_normalized > 1.0)).any()
        or ((result.depth_normalized < 0.0) | (result.depth_normalized > 1.0)).any()
        or (
            (result.profile_times_normalized < 0.0)
            | (result.profile_times_normalized > 1.0)
        ).any()
    ):
        raise ValueError("Internal-field coordinates must lie in the modeled domain.")
    grid_shape = (len(result.depth_normalized), len(result.time_normalized))
    if (
        result.moisture_ratio.shape != grid_shape
        or result.temperature_normalized.shape != grid_shape
        or result.temperature_kelvin.shape != grid_shape
        or result.profile_moisture_ratio.shape
        != (len(result.profile_times_normalized), len(result.depth_normalized))
        or result.profile_temperature_normalized.shape
        != result.profile_moisture_ratio.shape
        or result.profile_temperature_kelvin.shape
        != result.profile_moisture_ratio.shape
    ):
        raise ValueError("Internal-field arrays do not match their coordinate grids.")
    if (
        (result.moisture_ratio <= 0.0).any()
        or (result.profile_moisture_ratio <= 0.0).any()
        or (result.temperature_kelvin <= 0.0).any()
        or (result.profile_temperature_kelvin <= 0.0).any()
    ):
        raise ValueError("Moisture ratios and absolute temperatures must be positive.")
    config = result.physics_config
    expected_theta = (
        result.temperature_kelvin - config.air_temperature_K
    ) / config.signed_temperature_scale_K
    expected_profile_theta = (
        result.profile_temperature_kelvin - config.air_temperature_K
    ) / config.signed_temperature_scale_K
    if not np.allclose(
        result.temperature_normalized, expected_theta, rtol=1e-6, atol=1e-7
    ) or not np.allclose(
        result.profile_temperature_normalized,
        expected_profile_theta,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "Normalized temperature arrays do not match their Kelvin values."
        )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / "inferred_internal_fields.png"
    grid_path = destination / "inferred_internal_fields.npz"
    result.figure.savefig(figure_path, dpi=180, format="png")
    np.savez_compressed(
        grid_path,
        experiment_id=np.asarray(result.selected_experiment_id),
        time_normalized=result.time_normalized,
        depth_normalized=result.depth_normalized,
        time_s=result.time_normalized * config.time_scale_s,
        depth_m=result.depth_normalized * config.bed_depth_m,
        moisture_ratio=result.moisture_ratio,
        temperature_normalized=result.temperature_normalized,
        temperature_kelvin=result.temperature_kelvin,
        temperature_normalization_definition=np.asarray(
            "(T_K - T_air_K) / (T_initial_K - T_air_K)"
        ),
        air_temperature_K=np.asarray(config.air_temperature_K),
        initial_temperature_K=np.asarray(config.initial_temperature_K),
        signed_temperature_scale_K=np.asarray(config.signed_temperature_scale_K),
        profile_times_normalized=result.profile_times_normalized,
        profile_times_s=result.profile_times_normalized * config.time_scale_s,
        profile_moisture_ratio=result.profile_moisture_ratio,
        profile_temperature_normalized=result.profile_temperature_normalized,
        profile_temperature_kelvin=result.profile_temperature_kelvin,
    )
    return InternalFieldArtifactPaths(figure=figure_path, grid=grid_path)


__all__ = [
    "InternalFieldArtifactPaths",
    "InternalFieldResult",
    "infer_internal_fields",
    "select_representative_training_experiment",
    "write_internal_field_artifacts",
]
