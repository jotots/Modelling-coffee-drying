"""Regression tests for honest optimizer axes and dimensionless loss displays."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def _plotter():
    """Import lazily so a missing implementation is a test failure."""
    return importlib.import_module("strategy_c.loss_visualization").plot_loss_history


def _history() -> pd.DataFrame:
    """Distinct residuals make weighted moisture/heat grouping unambiguous."""
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "step": step,
                "data": 1.0,
                "physics_mass": 2.0,
                "physics_heat": 3.0,
                "initial_moisture": 4.0,
                "initial_temperature": 5.0,
                "base_moisture": 6.0,
                "base_temperature": 7.0,
                "surface_mass": 8.0,
                "surface_heat": 9.0,
                "total": 158.0,
                "held_out_data": 0.25,
            }
            for stage in ("adam", "lbfgs")
            for step in (1, 2)
        ]
    )


def test_legacy_loss_plot_uses_explicit_test_label_and_honest_axes() -> None:
    """Old held-out history remains legacy test monitoring, never validation."""
    figure = _plotter()(
        _history(), physics_weight=2.0, initial_weight=3.0, boundary_weight=4.0
    )
    try:
        adam, lbfgs = figure.axes
        assert adam.get_xlabel() == "Adam epoch"
        assert lbfgs.get_xlabel() == "L-BFGS closure evaluation (not epoch)"
        np.testing.assert_array_equal(adam.lines[0].get_ydata(), [73.0, 73.0])
        np.testing.assert_array_equal(adam.lines[1].get_ydata(), [85.0, 85.0])
        np.testing.assert_array_equal(adam.lines[2].get_ydata(), [0.25, 0.25])
        np.testing.assert_array_equal(
            adam.lines[0].get_ydata() + adam.lines[1].get_ydata(),
            _history().loc[lambda rows: rows["stage"].eq("adam"), "total"],
        )
        assert adam.lines[2].get_label() == (
            "Legacy held-out test moisture MSE (unweighted)"
        )
        assert "not validation" in figure._suptitle.get_text()
        assert adam.get_yscale() == "linear"
    finally:
        plt.close(figure)


def test_old_history_does_not_invent_held_out_values() -> None:
    """Historic training CSVs can be displayed but cannot recover test history."""
    figure = _plotter()(_history().drop(columns="held_out_data"))
    try:
        assert len(figure.axes[0].lines) == 2
        assert "Validation history unavailable" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_missing_components_are_not_zero_filled_and_legacy_test_values_are_gaps() -> (
    None
):
    history = _history()
    history.loc[0, "held_out_data"] = np.nan
    history.loc[0, "physics_heat"] = np.nan
    figure = _plotter()(history)
    try:
        assert np.isnan(figure.axes[0].lines[1].get_ydata()[0])
        assert np.isnan(figure.axes[0].lines[2].get_ydata()[0])
    finally:
        plt.close(figure)


def test_validation_history_uses_current_validation_label() -> None:
    history = _history().drop(columns="held_out_data")
    history["validation_data"] = 0.125
    figure = _plotter()(history)
    try:
        assert figure.axes[0].lines[-1].get_label() == (
            "Validation moisture MSE (unweighted)"
        )
        assert "final-test targets are reserved" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_direct_objective_plot_uses_serialized_conditioned_contributions() -> None:
    """Display actual ramp/scaled contributions rather than recomputing weights."""
    history = pd.DataFrame(
        [
            {
                "stage": stage,
                "step": 1,
                "data": 0.2,
                "contribution_physics_mass": 0.1,
                "contribution_base_moisture": 0.03,
                "contribution_surface_mass": 0.02,
                "contribution_physics_heat": 0.04,
                "contribution_base_temperature": 0.01,
                "contribution_surface_heat": 0.02,
                "total": 0.42,
            }
            for stage in ("adam", "lbfgs")
        ]
    )
    figure = _plotter()(history, physics_weight=100.0, boundary_weight=100.0)
    try:
        moisture, temperature = figure.axes[0].lines[:2]
        assert moisture.get_ydata()[0] == pytest.approx(0.35)
        assert temperature.get_ydata()[0] == pytest.approx(0.07)
        assert moisture.get_ydata()[0] + temperature.get_ydata()[0] == pytest.approx(
            history.loc[history["stage"].eq("adam"), "total"].iloc[0]
        )
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "change, message",
    [
        ("empty", "nonempty"),
        ("stage", "stages"),
        ("steps", "strictly increasing"),
        ("weights", "weights"),
    ],
)
def test_invalid_history_coordinates_and_weights(change: str, message: str) -> None:
    history = _history()
    if change == "empty":
        history = history.iloc[:0]
    elif change == "stage":
        history.loc[0, "stage"] = "unknown"
    elif change == "steps":
        history.loc[1, "step"] = 1
    with pytest.raises(ValueError, match=message):
        _plotter()(history, physics_weight=-1 if change == "weights" else 1)


@pytest.mark.parametrize("bad_value", [np.inf, -1.0])
def test_invalid_losses_are_rejected(bad_value: float) -> None:
    history = _history()
    history.loc[0, "data"] = bad_value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _plotter()(history)


def test_notebook_displays_and_returns_loss_figure() -> None:
    """Check persistent notebook integration without running full training."""
    import ast
    import json

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
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    # Validate that the display path actually invokes the plot and plt.show,
    # rather than merely importing the helper into an otherwise unused cell.
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]
    assert "plot_loss_history" in calls
    assert calls.count("plt.show") >= 2
    assert '"loss_figure"' in source
    assert "first consumed here" not in source
    assert "locked_split.validation_ids" in source
    assert "locked_split.test_ids" in source
    assert "final-test targets" in source
