"""Loss-history displays with explicit units, partition roles, and optimizer axes."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from strategy_c.training import TrainingResult


def plot_loss_history(
    history: pd.DataFrame | TrainingResult,
    *,
    physics_weight: float = 1.0,
    initial_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> Figure:
    """Plot objective contributions, validation MSE, and legacy test monitoring.

    New snapshots use their explicit per-residual objective contributions;
    this preserves the frozen calibration scales and warmup/ramp coefficient.
    Historical snapshots without those fields retain the legacy weight-based
    reconstruction. The two groups sum to the fitted objective, but share
    coupled physics and are not losses for independent fits. Validation MSE is
    unweighted, moisture-only, and never part of that sum. Historical
    ``held_out_data`` remains a legacy final-test label.
    L-BFGS line-search trials are closure evaluations, not epochs. Missing
    historic components are displayed as unavailable, never as zeros.
    """
    if isinstance(history, TrainingResult):
        history = pd.DataFrame.from_records(
            {"stage": stage, "step": index + 1, **losses}
            for stage, record in history.stage_history.items()
            for index, losses in enumerate(record.losses)
        )
    if history.empty or not {"stage", "step"}.issubset(history.columns):
        raise ValueError("Loss history requires nonempty stage and step columns.")
    if not history["stage"].isin(("adam", "lbfgs")).all():
        raise ValueError("Loss history stages must be adam or lbfgs.")
    weights = np.asarray([physics_weight, initial_weight, boundary_weight], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Loss weights must be finite and nonnegative.")
    if "contribution_physics_mass" in history:
        groups = {
            "Moisture objective contribution": {
                "data": 1.0,
                "contribution_physics_mass": 1.0,
                "contribution_base_moisture": 1.0,
                "contribution_surface_mass": 1.0,
            },
            "Temperature objective contribution (physics only)": {
                "contribution_physics_heat": 1.0,
                "contribution_base_temperature": 1.0,
                "contribution_surface_heat": 1.0,
            },
        }
    else:
        # Retain exact interpretation for historical run CSVs, whose raw
        # components predate frozen scaling and schedule-aware contributions.
        groups = {
            "Moisture objective contribution": {
                "data": 1.0,
                "physics_mass": physics_weight,
                "initial_moisture": initial_weight,
                "base_moisture": boundary_weight,
                "surface_mass": boundary_weight,
            },
            "Temperature objective contribution (physics only)": {
                "physics_heat": physics_weight,
                "initial_temperature": initial_weight,
                "base_temperature": boundary_weight,
                "surface_heat": boundary_weight,
            },
        }
    for name in {term for group in groups.values() for term in group} | {
        "validation_data",
        "held_out_data",
    }:
        if name in history:
            values = history[name].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Loss values must be finite and nonnegative.")
    for _, rows in history.groupby("stage", sort=False):
        steps = rows["step"].to_numpy(dtype=float)
        if (
            not np.isfinite(steps).all()
            or (steps < 1).any()
            or (np.diff(steps) <= 0).any()
        ):
            raise ValueError("History steps must be positive and strictly increasing.")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for axis, stage, xlabel in zip(
        axes,
        ("adam", "lbfgs"),
        ("Adam epoch", "L-BFGS closure evaluation (not epoch)"),
    ):
        rows = history.loc[history["stage"].eq(stage)]
        for (label, terms), color, style in zip(
            groups.items(), ("#1f77b4", "#d62728"), ("-", "--")
        ):
            if set(terms).issubset(rows.columns) and not rows.empty:
                # min_count retains missing snapshots as gaps, not partial sums.
                values = pd.concat(
                    [rows[name] * weight for name, weight in terms.items()], axis=1
                ).sum(axis=1, min_count=len(terms))
                axis.plot(
                    rows["step"], values, color=color, linestyle=style, label=label
                )
        if "validation_data" in rows and rows["validation_data"].notna().any():
            axis.plot(
                rows["step"],
                rows["validation_data"],
                color="#2ca02c",
                linestyle=":",
                label="Validation moisture MSE (unweighted)",
            )
        if "held_out_data" in rows and rows["held_out_data"].notna().any():
            axis.plot(
                rows["step"],
                rows["held_out_data"],
                color="#9467bd",
                linestyle="-.",
                label="Legacy held-out test moisture MSE (unweighted)",
            )
        if not axis.lines:
            axis.text(
                0.5,
                0.5,
                "Component history unavailable",
                ha="center",
                transform=axis.transAxes,
            )
        else:
            axis.legend(fontsize=8)
        # Linear scaling preserves exact zeros and does not silently clip them.
        axis.set(
            xlabel=xlabel, ylabel="Dimensionless squared loss", title=stage.upper()
        )
        axis.grid(True, alpha=0.25)
    has_validation = (
        "validation_data" in history and history["validation_data"].notna().any()
    )
    has_legacy_test_monitor = (
        "held_out_data" in history and history["held_out_data"].notna().any()
    )
    if has_validation and has_legacy_test_monitor:
        note = (
            "Validation is monitored; legacy held-out test history remains "
            "separate and is not mapped to validation."
        )
    elif has_validation:
        note = (
            "Validation monitoring only; final-test targets are reserved for "
            "post-training reporting."
        )
    elif has_legacy_test_monitor:
        note = (
            "Legacy held-out test history; historical split sizes are not "
            "inferred and this series is not validation."
        )
    else:
        note = (
            "Validation history unavailable; final-test targets are reserved "
            "for post-training reporting."
        )
    figure.suptitle(
        "Moisture and temperature loss history\n"
        "Temperature has no observation targets; objective contributions sum to total.\n"
        + note,
        fontsize=10,
    )
    return figure
