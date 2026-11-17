"""Focused tests for Strategy C's workbook and rate-interval cleaning."""

import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "strategy_c.ipynb"


def _strategy_c_functions() -> dict[str, object]:
    """Load notebook helpers without executing workbook reads or model setup."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    tree = ast.parse(source)
    desired = {"build_scheduled_intervals"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in desired
    ]
    namespace = {"np": np, "pd": pd, "total_time": 240 * 60}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK_PATH), "exec"
        ),
        namespace,
    )
    return namespace


def _observations(rows: list[tuple[str, float, float, bool]]) -> pd.DataFrame:
    """Build the minimal scheduled-observation columns used by rate cleaning."""
    return pd.DataFrame(
        rows,
        columns=["experiment_id", "time_s", "adm_recomputed", "target_available"],
    )


def test_scheduled_rates_distinguish_negative_from_zero() -> None:
    helpers = _strategy_c_functions()
    observations = _observations(
        [
            ("negative", 0.0, 1.0, True),
            ("negative", 60.0, 1.2, True),
            ("zero", 0.0, 1.0, True),
            ("zero", 60.0, 1.0, True),
        ]
    )
    # These source tx values intentionally disagree with the recomputed
    # moisture trajectory; the legacy formula must not determine interval sign.
    observations["tx_source"] = [5.0, 4.0, 0.0, 10.0]

    intervals = helpers["build_scheduled_intervals"](observations)

    rates = intervals.set_index("experiment_id")["rate_star"]
    assert rates["negative"] < 0
    assert rates["zero"] == 0
    available = intervals.set_index("experiment_id")["rate_target_available"]
    assert not available["negative"]
    assert available["zero"]


def test_invalid_scheduled_observation_breaks_rate_interval() -> None:
    helpers = _strategy_c_functions()
    observations = _observations(
        [
            ("gap", 0.0, 1.0, True),
            ("gap", 60.0, 1.2, False),
            ("gap", 120.0, 0.8, True),
        ]
    )

    intervals = helpers["build_scheduled_intervals"](observations)

    assert intervals.empty


def test_notebook_keeps_observations_and_excludes_only_negative_intervals(
    monkeypatch,
) -> None:
    sheet = pd.DataFrame(
        [
            [None, "450um", None, None, None, None],
            [None, 1, None, None, None, None],
            [None, 1.0, None, None, None, None],
            [None, None, None, None, None, None],
            ["Tempo", "borra +B", "m borra", "u", "adm", "tx"],
            [0, 11.0, 10.0, None, None, None],
            [1, 12.0, 11.0, None, None, None],
            [2, 10.0, 9.0, None, None, None],
            [3, 10.0, 9.0, None, None, None],
            ["Mseca", None, 5.0, None, None, None],
        ]
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: {"0.22ms": sheet})
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    namespace = {
        "Path": Path,
        "np": np,
        "pd": pd,
        "re": re,
        "bed_depth_m": 0.01,
        "T_air_K": 343.15,
        "total_time": 240 * 60,
    }

    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    assert len(namespace["experiments"]) == 1
    assert namespace["observations"]["time_s"].tolist() == [0, 60, 120, 180]
    assert namespace["observations"]["target_available"].all()
    intervals = namespace["scheduled_intervals"]
    assert intervals["rate_target_available"].tolist() == [False, True, True]
    assert namespace["negative_rate_intervals"]["left_time_s"].tolist() == [0]
    targets = namespace["rate_intervals"]
    assert list(zip(targets["left_time_s"], targets["right_time_s"])) == [
        (60, 120),
        (120, 180),
    ]
    assert (targets["rate_star"] >= 0).all()
