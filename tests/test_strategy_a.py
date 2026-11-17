"""Focused tests for the executable Strategy A notebook implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "strategy_a.ipynb"


def load_notebook_namespace():
    """Execute only tagged notebook setup and core cells, never workbook IO."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if cell["cell_type"] == "code" and {
            "strategy-a-test-core",
            "strategy-a-test-setup",
        }.intersection(tags):
            source = "".join(cell["source"])
            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def curves_fixture() -> pd.DataFrame:
    """Create scheduled curves containing a gap and a mass increase."""
    records = []
    for curve_id, masses in (
        ("curve-a", [1.0, 0.8, np.nan, 0.7, 0.72]),
        ("curve-b", [1.0, 0.9, 0.8, 0.7, 0.6]),
    ):
        for index, mass in enumerate(masses):
            records.append(
                {
                    "experiment_id": curve_id,
                    "condition_id": curve_id,
                    "time_s": index * 60.0,
                    "adm_recomputed": mass,
                    "target_available": bool(np.isfinite(mass)),
                    "quality_flags": "mass_increase" if index == 4 else "",
                    "particle_diameter_m": 450e-6,
                    "air_speed_m_s": 0.22,
                    "initial_water_concentration_kg_m3": 120.0,
                }
            )
    return pd.DataFrame(records)


def run_cell_source() -> str:
    """Return the tagged end-to-end run cell without executing it."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "strategy-a-run" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(sources) != 1:
        raise AssertionError(f"Expected one strategy-a-run cell, found {len(sources)}.")
    return sources[0]


def workbook_shaped_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build parsed-workbook-shaped experiments and observations for four runs."""
    experiment_records = []
    observation_records = []
    for index, (diameter, speed) in enumerate(
        ((450e-6, 0.22), (450e-6, 0.44), (900e-6, 0.22), (900e-6, 0.44))
    ):
        experiment_id = f"run-{index}"
        experiment_records.append(
            {
                "experiment_id": experiment_id,
                "condition_id": f"d{diameter:g}-v{speed:g}",
                "replicate": 1,
                "particle_diameter_m": diameter,
                "air_speed_m_s": speed,
                "initial_sample_mass_kg": 0.020,
                "dry_mass_kg": 0.012,
            }
        )
        for step, moisture_ratio in enumerate((1.0, 0.8, 0.6, 0.45, 0.3)):
            observation_records.append(
                {
                    "experiment_id": experiment_id,
                    "time_s": step * 600.0,
                    "adm_recomputed": moisture_ratio,
                    "target_available": True,
                    "quality_flags": "",
                }
            )
    return pd.DataFrame(experiment_records), pd.DataFrame(observation_records)


class _ForbiddenCall:
    """Fail loudly if preprocessing-only mode touches model-dependent work."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"{self.name} must not run when RUN_TRAINING=False.")

    def __getattr__(self, attribute: str) -> "_ForbiddenCall":
        return _ForbiddenCall(f"{self.name}.{attribute}")


class StrategyATests(unittest.TestCase):
    """Analytic, deterministic tests for physics and persistence helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = load_notebook_namespace()

    def test_joint_architecture_hard_initial_conditions_and_parameters(self) -> None:
        """Preserve the hidden Tanh architecture and unbounded state projection."""
        model = self.namespace["DryingPINN"]().to(dtype=torch.float64, device="cpu")
        layers = [
            module for module in model.network if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in layers],
            [(4, 64), (64, 32), (32, 32), (32, 32), (32, 64), (64, 2)],
        )
        self.assertIsInstance(model.network[-1], torch.nn.Linear)
        self.assertEqual(
            sum(isinstance(module, torch.nn.Tanh) for module in model.network), 5
        )

        # A controlled projection proves the physical state can leave the old
        # bounded-output range while the hard initial conditions remain exact.
        with torch.no_grad():
            model.network[-1].weight.zero_()
            model.network[-1].bias.copy_(
                torch.tensor([-1.822, 2.0], dtype=torch.float64)
            )
        coordinates = torch.zeros((2, 4), dtype=torch.float64)
        coordinates[:, 1] = torch.tensor([0.5, 0.1], dtype=torch.float64)
        fields = model(coordinates)
        torch.testing.assert_close(
            fields[:, 0],
            torch.tensor([0.089, 0.8178], dtype=torch.float64),
            rtol=0.0,
            atol=1e-14,
        )
        torch.testing.assert_close(
            fields[:, 1],
            torch.tensor([1.0, 0.2], dtype=torch.float64),
            rtol=0.0,
            atol=1e-14,
        )
        self.assertLess(fields[0, 0].item(), 1.0 - coordinates[0, 1].item())
        self.assertGreater(fields[1, 1].item(), coordinates[1, 1].item())

        coordinates = torch.rand((7, 4), dtype=torch.float64)
        coordinates[:, 1] = 0.0
        fields = model(coordinates)
        self.assertTrue(torch.equal(fields[:, 0], torch.ones(7, dtype=torch.float64)))
        self.assertTrue(torch.equal(fields[:, 1], torch.zeros(7, dtype=torch.float64)))
        self.assertEqual(
            set(model.learned_theta), {"theta0", "theta1", "theta2", "theta3"}
        )
        self.assertTrue(
            all(
                name in dict(model.named_parameters())
                for name in ("raw_theta0", "theta1", "theta2", "raw_theta3")
            )
        )
        self.assertGreater(model.theta0.item(), 0.0)
        self.assertGreater(model.theta3.item(), 0.0)
        self.assertTrue(
            all(parameter.dtype == torch.float64 for parameter in model.parameters())
        )
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in model.parameters())
        )
        inputs = torch.tensor([[0.5, 0.4, 1.0, 1.0]], dtype=torch.float64)
        cool = torch.tensor([[0.0]], dtype=torch.float64)
        warm = torch.tensor([[0.5]], dtype=torch.float64)
        diffusivity_cool = model.diffusivity_star(inputs, cool)
        diffusivity_warm = model.diffusivity_star(inputs, warm)
        self.assertGreater(diffusivity_warm.item(), diffusivity_cool.item())
        self.assertAlmostEqual(
            diffusivity_cool.item() * self.namespace["D_REF_M2_S"],
            (diffusivity_cool * self.namespace["D_REF_M2_S"]).item(),
        )
        with self.assertRaisesRegex(ValueError, "dtype"):
            model(torch.ones((1, 4), dtype=torch.float32))
        if torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "device"):
                model(
                    torch.ones((1, 4), dtype=torch.float64, device=torch.device("cuda"))
                )

    def test_coordinate_derivatives_and_four_constitutive_gradients_are_finite(
        self,
    ) -> None:
        """Keep field and physics derivatives connected through the real model."""
        model = self.namespace["DryingPINN"]().to(dtype=torch.float64, device="cpu")
        coordinates = torch.tensor(
            [
                [0.2, 0.3, 1.5, 0.7],
                [0.8, 0.6, 0.8, 1.4],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        fields = model(coordinates)
        first_derivative = torch.autograd.grad(
            fields.sum(), coordinates, create_graph=True
        )[0]
        second_derivative = torch.autograd.grad(
            first_derivative.sum(), coordinates, create_graph=True
        )[0]
        self.assertTrue(torch.isfinite(first_derivative).all())
        self.assertTrue(torch.isfinite(second_derivative).all())

        residuals = self.namespace["physics_residuals"](
            model, coordinates, fo_ref=5.76, fo_heat=0.177613
        )
        for residual_name in ("mass", "heat"):
            residual = residuals[residual_name]
            self.assertTrue(torch.isfinite(residual).all())
            residual_derivative = torch.autograd.grad(
                residual.sum(), coordinates, retain_graph=True
            )[0]
            self.assertTrue(torch.isfinite(residual_derivative).all())

        constitutive_inputs = coordinates.detach().clone()
        temperature_star = torch.full((2, 1), 0.4, dtype=torch.float64)
        diffusivity = model.diffusivity_star(constitutive_inputs, temperature_star)
        parameters = tuple(
            getattr(model, name)
            for name in ("raw_theta0", "theta1", "theta2", "raw_theta3")
        )
        gradients = torch.autograd.grad(diffusivity.sum(), parameters)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(torch.count_nonzero(gradient) for gradient in gradients))

    def test_dimensionless_groups_and_air_property_interpolation(self) -> None:
        """Verify groups come from cited properties and approved equations."""
        constants = self.namespace["PHYSICAL_CONSTANTS"]
        groups = self.namespace["build_physics_tables"](
            curves_fixture().drop_duplicates("experiment_id"), constants
        )
        self.assertAlmostEqual(
            constants["nu_air_m2_s"],
            constants["mu_air_pa_s"] / constants["rho_air_kg_m3"],
        )
        self.assertAlmostEqual(
            constants["pr_air"],
            constants["mu_air_pa_s"]
            * constants["cp_air_j_kg_k"]
            / constants["k_air_w_m_k"],
        )
        self.assertAlmostEqual(constants["cp_air_j_kg_k"], 1008.315)
        self.assertAlmostEqual(constants["fo_ref"], 5.76)
        self.assertAlmostEqual(constants["fo_heat"], 0.177613, places=5)
        self.assertEqual(groups["h_w_m2_k"].nunique(), 1)
        row = groups.iloc[0]
        reynolds = 0.22 * constants["dish_diameter_m"] / constants["nu_air_m2_s"]
        self.assertAlmostEqual(row["re"], reynolds)
        nusselt = 0.664 * math.sqrt(reynolds) * constants["pr_air"] ** (1 / 3)
        self.assertAlmostEqual(row["nusselt"], nusselt)
        self.assertAlmostEqual(
            row["h_w_m2_k"],
            row["nusselt"] * constants["k_air_w_m_k"] / constants["dish_diameter_m"],
        )
        self.assertFalse(math.isclose(row["h_w_m2_k"], 1.24, rel_tol=1e-3))
        self.assertAlmostEqual(
            constants["fo_ref"],
            constants["d_ref_m2_s"]
            * constants["time_scale_s"]
            / constants["bed_depth_m"] ** 2,
        )
        self.assertAlmostEqual(
            row["bi"],
            row["h_w_m2_k"] * constants["bed_depth_m"] / constants["product_k_w_m_k"],
        )

    def test_air_viscosity_and_reference_nusselt_have_distinct_units(self) -> None:
        """Keep dimensional viscosity separate from the reference correlation."""
        namespace = self.namespace
        viscosity = namespace["MU_AIR_PA_S"] / namespace["RHO_AIR_KG_M3"]
        self.assertEqual(namespace["NU_AIR_M2_S"], viscosity)
        self.assertEqual(namespace["PHYSICAL_CONSTANTS"]["nu_air_m2_s"], viscosity)
        reynolds = (
            namespace["SPEED_REFERENCE_M_S"] * namespace["DISH_DIAMETER_M"] / viscosity
        )
        self.assertEqual(namespace["RE_AIR"], reynolds)
        prandtl = (
            namespace["MU_AIR_PA_S"]
            * namespace["CP_AIR_J_KG_K"]
            / namespace["K_AIR_W_M_K"]
        )
        self.assertEqual(namespace["PR_AIR"], prandtl)
        self.assertAlmostEqual(
            namespace["NUSSELT_AIR_REF"],
            0.664 * math.sqrt(reynolds) * prandtl ** (1 / 3),
            places=12,
        )
        self.assertAlmostEqual(viscosity, 1.8858490566037734e-5, places=16)
        self.assertAlmostEqual(namespace["NUSSELT_AIR_REF"], 28.47605335199768)

    def test_per_speed_convection_uses_dynamic_viscosity_identity(self) -> None:
        """Compute expected Re/Nu/h independently of the potentially wrong nu."""
        constants = self.namespace["PHYSICAL_CONSTANTS"]
        curves = curves_fixture().drop_duplicates("experiment_id").copy()
        curves["air_speed_m_s"] = [0.22, 0.48]
        curves["particle_diameter_m"] = [213e-6, 900e-6]
        groups = self.namespace["build_physics_tables"](curves, constants)
        for row in groups.to_dict(orient="records"):
            with self.subTest(speed=row["velocity_m_s"]):
                # Use mu/rho directly so a dimensionless nu cannot mask the bug.
                reynolds = (
                    row["velocity_m_s"]
                    * constants["dish_diameter_m"]
                    * constants["rho_air_kg_m3"]
                    / constants["mu_air_pa_s"]
                )
                prandtl = (
                    constants["mu_air_pa_s"]
                    * constants["cp_air_j_kg_k"]
                    / constants["k_air_w_m_k"]
                )
                nusselt = 0.664 * math.sqrt(reynolds) * prandtl ** (1 / 3)
                h = nusselt * constants["k_air_w_m_k"] / constants["dish_diameter_m"]
                self.assertAlmostEqual(row["re"], reynolds, places=10)
                self.assertEqual(row["pr"], prandtl)
                self.assertAlmostEqual(row["nusselt"], nusselt, places=12)
                self.assertAlmostEqual(row["h_w_m2_k"], h, places=12)
                self.assertNotEqual(row["h_w_m2_k"], 1.24)
        self.assertAlmostEqual(groups.iloc[0]["h_w_m2_k"], 4.100551682687666)
        self.assertAlmostEqual(
            groups.iloc[1]["h_w_m2_k"] / groups.iloc[0]["h_w_m2_k"],
            math.sqrt(0.48 / 0.22),
            places=12,
        )
        self.assertEqual(self.namespace["COMPARISON_H_W_M2_K"], 1.24)

    def test_intervals_break_at_gaps_and_keep_negative_rates(self) -> None:
        """Never bridge an invalid scheduled row or clip a mass increase."""
        intervals = self.namespace["build_scheduled_intervals"](curves_fixture())
        curve_a = intervals.loc[intervals["experiment_id"].eq("curve-a")]
        self.assertEqual(
            list(zip(curve_a["left_time_s"], curve_a["right_time_s"])),
            [(0.0, 60.0), (180.0, 240.0)],
        )
        self.assertLess(curve_a.iloc[-1]["rate_star"], 0.0)
        self.assertTrue(curve_a.iloc[-1]["mass_increase"])
        self.assertEqual(
            len(intervals.loc[intervals["experiment_id"].eq("curve-b")]), 4
        )

    def test_equal_curve_weights_and_conservative_divergence(self) -> None:
        """Check curve weighting and full d(D rho_z)/dz differentiation."""
        intervals = self.namespace["build_scheduled_intervals"](curves_fixture())
        weights = self.namespace["equal_curve_weights"](intervals, "experiment_id")
        for _, curve_weights in intervals.assign(weight=weights).groupby(
            "experiment_id"
        ):
            self.assertAlmostEqual(curve_weights["weight"].sum(), 0.5)

        model = self.namespace["DryingPINN"]().to(dtype=torch.float64, device="cpu")
        points = torch.tensor(
            [[0.35, 0.2, 1.1, 0.8], [0.65, 0.4, 0.9, 1.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        constants = self.namespace["PHYSICAL_CONSTANTS"]
        residuals = self.namespace["physics_residuals"](
            model, points, constants["fo_ref"], constants["fo_heat"]
        )
        self.assertTrue(torch.isfinite(residuals["mass"]).all())
        self.assertTrue(torch.isfinite(residuals["heat"]).all())
        self.assertEqual(residuals["mass"].shape, points[:, :1].shape)
        fields = model(points)
        moisture_gradient = torch.autograd.grad(
            fields[:, 0:1].sum(), points, create_graph=True, retain_graph=True
        )[0][:, 0:1]
        diffusivity = model.diffusivity_star(points, fields[:, 1:2])
        flux_gradient = torch.autograd.grad(
            (diffusivity * moisture_gradient).sum(),
            points,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        diffusivity_gradient = torch.autograd.grad(
            diffusivity.sum(), points, create_graph=True, retain_graph=True
        )[0][:, 0:1]
        moisture_curvature = torch.autograd.grad(
            moisture_gradient.sum(),
            points,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]
        omitted_term_form = diffusivity * moisture_curvature
        self.assertTrue(torch.isfinite(flux_gradient).all())
        self.assertTrue(torch.isfinite(diffusivity_gradient).all())
        self.assertGreater(
            torch.max(torch.abs(flux_gradient - omitted_term_form)).item(),
            1e-12,
        )

    def test_signed_surface_balances_and_bounded_lbfgs_smoke(self) -> None:
        """Check surface signs and a finite fixed-batch constitutive update."""
        constants = self.namespace["PHYSICAL_CONSTANTS"]

        class ConstantFields:
            """Manufactured differentiable solution for zero-gradient BC tests."""

            def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
                zero = coordinates[:, 0:1] * 0.0
                return torch.cat((zero + 1.0, zero), dim=1)

            def diffusivity_star(
                self, coordinates: torch.Tensor, temperature: torch.Tensor
            ) -> torch.Tensor:
                return temperature * 0.0 + 1.0

        boundary_points = torch.tensor(
            [[0.0, 0.2, 1.0, 1.0], [1.0, 0.4, 1.0, 1.0]],
            dtype=torch.float64,
        )
        zero_base = self.namespace["boundary_residuals"](
            ConstantFields(),
            boundary_points[:1],
            boundary_points[1:],
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
        )
        self.assertTrue(
            torch.equal(
                zero_base["base_mass"], torch.zeros_like(zero_base["base_mass"])
            )
        )
        self.assertTrue(
            torch.equal(
                zero_base["base_heat"], torch.zeros_like(zero_base["base_heat"])
            )
        )

        model = self.namespace["DryingPINN"]().to(device=self.namespace["DEVICE"])
        scheduled = curves_fixture()
        curve_metadata = scheduled.drop_duplicates("experiment_id").drop(
            columns=[
                "time_s",
                "adm_recomputed",
                "target_available",
                "quality_flags",
            ]
        )
        scheduled_observations = scheduled.drop(
            columns=[
                "condition_id",
                "particle_diameter_m",
                "air_speed_m_s",
                "initial_water_concentration_kg_m3",
            ]
        )
        batch = self.namespace["build_training_batch"](
            scheduled_observations,
            curve_metadata,
            self.namespace["build_physics_tables"](curve_metadata, constants),
            collocation_count=4,
            seed=17,
        )
        positive_rate = torch.ones_like(batch["rate_targets"])
        positive_boundary = self.namespace["boundary_residuals"](
            model,
            batch["base"],
            batch["rate_features"],
            positive_rate,
            batch["surface_bi"],
            batch["surface_ev"],
        )
        negative_boundary = self.namespace["boundary_residuals"](
            model,
            batch["base"],
            batch["rate_features"],
            -positive_rate,
            batch["surface_bi"],
            batch["surface_ev"],
        )
        self.assertTrue(
            torch.allclose(
                positive_boundary["surface_mass"] - negative_boundary["surface_mass"],
                2.0 * positive_rate,
            )
        )
        self.assertTrue(
            torch.allclose(
                positive_boundary["surface_heat"] - negative_boundary["surface_heat"],
                2.0 * batch["surface_ev"] * positive_rate,
            )
        )
        fixed_points = batch["interior"].clone()
        components, objective = self.namespace["evaluate_loss_components"](model, batch)
        objective.backward()
        self.assertTrue(torch.isfinite(model.raw_theta0.grad).all())
        self.assertTrue(torch.isfinite(model.raw_theta3.grad).all())
        model.zero_grad(set_to_none=True)

        history, closure_count, selected_step = self.namespace["train_inverse_model"](
            model, batch, steps=1, learning_rate=1.0
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(selected_step, 1)
        self.assertGreaterEqual(closure_count, 1)
        self.assertTrue(
            np.isfinite(history.select_dtypes(include=[np.number])).all().all()
        )
        self.assertTrue(torch.equal(fixed_points, batch["interior"]))
        self.assertTrue(
            torch.isfinite(
                torch.stack([value.detach() for value in components.values()])
            ).all()
        )

    def test_export_is_strict_json_dimensional_and_non_overwriting(self) -> None:
        """Write auditable artifacts only to a fresh resolved run directory."""
        model = self.namespace["DryingPINN"]().to(dtype=torch.float64, device="cpu")
        history = pd.DataFrame([{"outer_step": 1, "loss": 0.5}])
        curves = curves_fixture().drop_duplicates("experiment_id")
        tables = self.namespace["make_diffusivity_table"](
            model, curves, depth_points=2, time_points=2
        )
        self.assertEqual(len(tables), 8)
        self.assertTrue((tables["dw_m2_s"] > 0.0).all())
        self.assertTrue(
            np.allclose(
                tables["dw_m2_s"],
                self.namespace["D_REF_M2_S"] * tables["diffusivity_star"],
            )
        )
        initial_temperature = tables.loc[tables["time_star"].eq(0.0)]
        self.assertTrue(
            np.allclose(
                initial_temperature["temperature_k_model_implied"],
                self.namespace["T0_K"],
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "run"
            output = self.namespace["export_strategy_a_run"](
                model=model,
                history=history,
                diffusivity_tables=tables,
                output_path=output_path,
                manifest={"purpose": "test"},
                selected_outer_step=1,
            )
            self.assertTrue(output.is_dir())
            payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["selected_outer_step"], 1)
            self.assertTrue((output / "joint_state.pt").is_file())
            checkpoint = torch.load(
                output / "joint_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["selected_outer_step"], 1)
            self.assertTrue(
                torch.equal(
                    checkpoint["model_state_dict"]["raw_theta0"],
                    model.raw_theta0.detach().cpu(),
                )
            )
            theta = json.loads((output / "theta.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(theta["theta0"], model.theta0.item())
            self.assertAlmostEqual(
                theta["activation_energy_j_mol"],
                theta["theta3"]
                * self.namespace["UNIVERSAL_GAS_CONSTANT_J_MOL_K"]
                * (self.namespace["T_AIR_K"] - self.namespace["T0_K"]),
            )
            self.assertAlmostEqual(
                pd.read_csv(output / "diffusivity.csv").loc[0, "dw_m2_s"],
                tables.loc[0, "dw_m2_s"],
            )
            self.assertEqual(
                self.namespace["resolve_output_path"]("src/run"),
                (Path.cwd() / "src" / "run").resolve(),
            )
            with self.assertRaises(FileExistsError):
                self.namespace["export_strategy_a_run"](
                    model=model,
                    history=history,
                    diffusivity_tables=tables,
                    output_path=output,
                    manifest={"purpose": "no overwrite"},
                    selected_outer_step=1,
                )
            invalid_destination = Path(temporary_directory) / "invalid"
            with self.assertRaises(ValueError):
                self.namespace["export_strategy_a_run"](
                    model=model,
                    history=history,
                    diffusivity_tables=tables,
                    output_path=invalid_destination,
                    manifest={"invalid_number": float("nan")},
                    selected_outer_step=1,
                )
            self.assertFalse(invalid_destination.exists())

    def test_invalid_temperature_denominator_is_rejected(self) -> None:
        """The Arrhenius denominator must describe positive absolute temperature."""
        model = self.namespace["DryingPINN"]().to(dtype=torch.float64, device="cpu")
        inputs = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "temperature"):
            model.diffusivity_star(inputs, torch.tensor([[-10.0]], dtype=torch.float64))

    def test_bulk_rate_is_true_mean_derivative(self) -> None:
        """A depth-independent rho=1-2*t* must give J*=2 at every resolution."""

        class LinearMoisture:
            def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    (1.0 - 2.0 * coordinates[:, 1:2], coordinates[:, 1:2] * 0.0),
                    dim=1,
                )

        coordinates = torch.tensor(
            [[0.2, 0.1, 1.0, 1.0], [0.8, 0.4, 2.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        original = coordinates.detach().clone()
        for depth_points in (1, 5, 50):
            with self.subTest(depth_points=depth_points):
                bulk, rate = self.namespace["bulk_state_and_rate"](
                    LinearMoisture(), coordinates, depth_points=depth_points
                )
                torch.testing.assert_close(
                    bulk, 1.0 - 2.0 * coordinates[:, 1:2], rtol=0.0, atol=1e-14
                )
                torch.testing.assert_close(
                    rate, torch.full_like(bulk, 2.0), rtol=0.0, atol=1e-14
                )
        torch.testing.assert_close(coordinates, original)
        self.assertIsNone(coordinates.grad)

    def test_bulk_rate_preserves_parameter_graph_and_validates_depths(self) -> None:
        """Pointwise AD averages depth-varying slopes without mixing batch rows."""

        class DepthSlope(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.slope = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

            def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
                moisture = 1.0 - self.slope * coordinates[:, 1:2] * (
                    coordinates[:, 0:1] + coordinates[:, 3:4]
                )
                return torch.cat((moisture, coordinates[:, 1:2] * 0.0), dim=1)

        model = DepthSlope()
        coordinates = torch.tensor(
            [[0.2, 0.1, 1.0, 1.0], [0.8, 0.4, 2.0, 2.0]], dtype=torch.float64
        )
        for depth_points in (1, 5, 50):
            bulk, rate = self.namespace["bulk_state_and_rate"](
                model, coordinates, depth_points=depth_points
            )
            expected = 0.5 + coordinates[:, 3:4]
            torch.testing.assert_close(rate, model.slope * expected)
            gradient = torch.autograd.grad(rate.sum(), model.slope)[0]
            torch.testing.assert_close(gradient, expected.sum())
            bulk_only, no_rate = self.namespace["bulk_state_and_rate"](
                model, coordinates, depth_points=depth_points, compute_time_rate=False
            )
            torch.testing.assert_close(bulk_only, bulk)
            self.assertIsNone(no_rate)
        for invalid_depth in (0, -1, True, 2.5):
            with self.assertRaisesRegex(ValueError, "depth_points"):
                self.namespace["bulk_state_and_rate"](
                    model, coordinates, depth_points=invalid_depth
                )

    def test_rate_loss_uses_unscaled_bulk_derivative(self) -> None:
        """The coupled objective must compare J*=2, not 2/50, to signed targets."""
        scheduled = curves_fixture()
        curves = scheduled.drop_duplicates("experiment_id").drop(
            columns=["time_s", "adm_recomputed", "target_available", "quality_flags"]
        )
        observations = scheduled.drop(
            columns=[
                "condition_id",
                "particle_diameter_m",
                "air_speed_m_s",
                "initial_water_concentration_kg_m3",
            ]
        )
        batch = self.namespace["build_training_batch"](
            observations,
            curves,
            self.namespace["build_physics_tables"](
                curves, self.namespace["PHYSICAL_CONSTANTS"]
            ),
            collocation_count=4,
            seed=17,
        )
        model = self.namespace["DryingPINN"]().to(device=self.namespace["DEVICE"])
        with torch.no_grad():
            # A constant network projection gives rho*=1-2*t* through the
            # existing hard initial condition, without changing any physics.
            model.network[-1].weight.zero_()
            model.network[-1].bias.copy_(
                torch.tensor(
                    [-2.0, 0.0], dtype=torch.float64, device=self.namespace["DEVICE"]
                )
            )
        components, total = self.namespace["evaluate_loss_components"](model, batch)
        expected_rate_loss = (
            (2.0 - batch["rate_targets"]) ** 2 * batch["rate_weights"]
        ).sum()
        torch.testing.assert_close(components["rate"], expected_rate_loss)
        expected_total = sum(
            self.namespace["LOSS_WEIGHTS"][name] * value
            for name, value in components.items()
        )
        torch.testing.assert_close(total, expected_total)

    def _rate_diagnostic_fixture(self):
        """Retain a missing endpoint, irregular times and a signed mass increase."""
        scheduled = curves_fixture()
        scheduled.loc[scheduled["experiment_id"].eq("curve-b"), "air_speed_m_s"] = 0.48
        scheduled.loc[scheduled["experiment_id"].eq("curve-b"), "time_s"] = [
            0.0,
            30.0,
            150.0,
            210.0,
            360.0,
        ]
        curves = scheduled.drop_duplicates("experiment_id")[
            ["experiment_id", "condition_id", "particle_diameter_m", "air_speed_m_s"]
        ].copy()
        curves["replicate"] = [1, 2]
        curves["split"] = ["train", "test"]
        observations = scheduled[
            [
                "experiment_id",
                "time_s",
                "adm_recomputed",
                "target_available",
                "quality_flags",
            ]
        ].copy()
        # Deliberately poisonous legacy columns must never enter the diagnostic.
        observations["tx_source"] = 999999.0
        observations["legacy_adm_per_min"] = -999999.0
        return observations, curves

    def test_rate_diagnostic_secants_ad_sign_and_units(self) -> None:
        """Check analytic AD, midpoint secants and the seconds-to-minutes factor."""
        observations, curves = self._rate_diagnostic_fixture()

        class LinearMoisture:
            """Depth-varying slope distinguishes bulk AD from a surface rate."""

            def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
                moisture = 1.0 - coordinates[:, 1:2] * (
                    coordinates[:, 0:1] + coordinates[:, 3:4]
                )
                return torch.cat((moisture, coordinates[:, 1:2] * 0.0), dim=1)

        for depth_points in (1, 5, 7, 50):
            with torch.no_grad():
                table = self.namespace["make_drying_rate_table"](
                    LinearMoisture(), observations, curves, depth_points=depth_points
                )
            expected = self.namespace["build_scheduled_intervals"](observations)
            pd.testing.assert_frame_equal(table[expected.columns], expected)
            curve_a = table.loc[table["experiment_id"].eq("curve-a")]
            np.testing.assert_allclose(curve_a["mid_time_s"], [30.0, 210.0])
            np.testing.assert_allclose(curve_a["rate_star"], [48.0, -4.8])
            np.testing.assert_allclose(curve_a["observed_mr_per_min"], [0.2, -0.02])
            np.testing.assert_allclose(
                table["model_rate_star"], 0.5 + table["air_speed_m_s"] / 0.22
            )
            np.testing.assert_allclose(
                table["model_mr_per_min"],
                table["model_rate_star"] * 60.0 / self.namespace["TIME_SCALE_S"],
            )
            np.testing.assert_allclose(
                table["model_training_rate_star"],
                table["model_rate_star"],
            )
            self.assertTrue(curve_a.iloc[-1]["mass_increase"])
            self.assertEqual(set(table["air_speed_m_s"]), {0.22, 0.48})
            self.assertEqual(set(table["split"]), {"train", "test"})

        # Reversing the manufactured slope must retain negative predicted rates.
        class IncreasingMoisture:
            def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    (1.0 + coordinates[:, 1:2], coordinates[:, 1:2] * 0.0), dim=1
                )

        table = self.namespace["make_drying_rate_table"](
            IncreasingMoisture(), observations, curves
        )
        np.testing.assert_allclose(table["model_rate_star"], -1.0)
        self.assertTrue((table["model_mr_per_min"] < 0.0).all())

    def test_rate_figures_preserve_both_speeds_gaps_and_replicas(self) -> None:
        """Inspect figure data rather than merely asserting a PNG was created."""
        observations, curves = self._rate_diagnostic_fixture()
        model = self.namespace["DryingPINN"]()
        table = self.namespace["make_drying_rate_table"](model, observations, curves)
        figures = self.namespace["plot_drying_rates"](table, curves)
        self.assertEqual(set(figures), {0.22, 0.48})
        plt = self.namespace["plt"]
        try:
            for speed, figure in figures.items():
                axis = figure.axes[0]
                rows = table.loc[table["air_speed_m_s"].eq(speed)]
                observed = axis.collections[0].get_offsets()
                np.testing.assert_allclose(observed[:, 0], rows["mid_time_s"] / 60.0)
                np.testing.assert_allclose(observed[:, 1], rows["observed_mr_per_min"])
                predicted = [line for line in axis.lines if line.get_marker() == "D"]
                np.testing.assert_allclose(
                    np.concatenate([line.get_xdata() for line in predicted]),
                    rows["mid_time_s"] / 60.0,
                )
                np.testing.assert_allclose(
                    np.concatenate([line.get_ydata() for line in predicted]),
                    rows["model_mr_per_min"],
                )
                self.assertTrue(all(line.get_linestyle() == "--" for line in predicted))
                zero_lines = [
                    line for line in axis.lines if line.get_label() == "Zero rate"
                ]
                np.testing.assert_allclose(zero_lines[0].get_ydata(), [0.0, 0.0])
                self.assertIn("MR/min", axis.get_ylabel())
                self.assertIn("inference-only", figure._suptitle.get_text())
                self.assertIn(f"{speed:g} m/s", figure._suptitle.get_text())
                if speed == 0.22:
                    self.assertEqual(len(predicted), 2)  # No line across the gap.
                    self.assertLess(float(observed[-1, 1]), 0.0)
                    self.assertIn("train", axis.get_title())
                else:
                    self.assertIn("test", axis.get_title())
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_rate_diagnostic_empty_curves_and_invalid_inputs(self) -> None:
        """No usable pairs remain auditable; malformed metadata fail explicitly."""
        observations, curves = self._rate_diagnostic_fixture()
        model = self.namespace["DryingPINN"]()
        observations["target_available"] = False
        table = self.namespace["make_drying_rate_table"](model, observations, curves)
        self.assertTrue(table.empty)
        figures = self.namespace["plot_drying_rates"](table, curves)
        try:
            self.assertEqual(set(figures), {0.22, 0.48})
            self.assertTrue(
                all(
                    "No valid consecutive pairs" in figure.axes[0].texts[0].get_text()
                    for figure in figures.values()
                )
            )
        finally:
            for figure in figures.values():
                self.namespace["plt"].close(figure)
        for invalid_depth in (0, 2.5, True):
            with self.assertRaisesRegex(ValueError, "depth_points"):
                self.namespace["make_drying_rate_table"](
                    model, observations, curves, depth_points=invalid_depth
                )
        for invalid_curves in (
            curves.drop(columns="split"),
            pd.concat([curves, curves.iloc[:1]]),
            curves.assign(air_speed_m_s=0.0),
            curves.assign(particle_diameter_m=np.nan),
            curves.assign(split="unknown"),
            curves.iloc[:0],
        ):
            with self.assertRaises(ValueError):
                self.namespace["make_drying_rate_table"](
                    model, observations, invalid_curves
                )
        with self.assertRaisesRegex(ValueError, "Unknown observation curves"):
            self.namespace["make_drying_rate_table"](
                model, observations.iloc[:5].assign(experiment_id="unknown"), curves
            )

    def test_rate_diagnostic_preserves_replicas_model_and_fails_nonfinite(self) -> None:
        """Every replica gets a panel; inference cannot mutate the trained state."""
        observations, curves = self._rate_diagnostic_fixture()
        for replica, curve_id in ((3, "curve-c"), (4, "curve-d"), (5, "curve-e")):
            extra_curve = curves.iloc[:1].assign(
                experiment_id=curve_id, replicate=replica
            )
            extra_observations = observations.loc[
                observations["experiment_id"].eq("curve-a")
            ].assign(experiment_id=curve_id, adm_recomputed=0.5)
            curves = pd.concat([curves, extra_curve], ignore_index=True)
            observations = pd.concat(
                [observations, extra_observations], ignore_index=True
            )
        model = self.namespace["DryingPINN"]()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        table = self.namespace["make_drying_rate_table"](model, observations, curves)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        figures = self.namespace["plot_drying_rates"](table, curves)
        try:
            self.assertEqual(len(figures[0.22].axes), 6)
            self.assertEqual(sum(axis.get_visible() for axis in figures[0.22].axes), 4)
            self.assertEqual(len(figures[0.22].axes[0].collections[0].get_offsets()), 2)
            np.testing.assert_allclose(
                figures[0.22].axes[1].collections[0].get_offsets()[:, 1], [0.0, 0.0]
            )
        finally:
            for figure in figures.values():
                self.namespace["plt"].close(figure)
        for invalid_table in (
            table.drop(columns="model_mr_per_min"),
            table.assign(model_mr_per_min=np.nan),
            table.assign(experiment_id="unknown"),
        ):
            with self.assertRaises(ValueError):
                self.namespace["plot_drying_rates"](invalid_table, curves)

        class NonfiniteMoisture:
            def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
                return coordinates[:, :2] * float("nan")

        with self.assertRaisesRegex(ValueError, "Model AD drying rates"):
            self.namespace["make_drying_rate_table"](
                NonfiniteMoisture(), observations, curves
            )

    def _disabled_run_namespace(
        self, export_directory: Path, *, export_diffusivity: bool
    ) -> dict[str, object]:
        """Prepare a fresh namespace for the run cell with training disabled."""
        namespace = load_notebook_namespace()
        experiments, observations = workbook_shaped_fixture()
        namespace.update(
            {
                "experiments": experiments,
                "observations": observations,
                "RUN_TRAINING": False,
                "EXPORT_DIFFUSIVITY": export_diffusivity,
                "EXPORT_DIRECTORY": export_directory,
                "COLLOCATION_COUNT": 8,
            }
        )
        for name in (
            "train_inverse_model",
            "make_diffusivity_table",
            "format_learned_parameters",
            "export_strategy_a_run",
            "make_drying_rate_table",
            "plot_drying_rates",
            "display",
            "plt",
        ):
            namespace[name] = _ForbiddenCall(name)
        return namespace

    def test_disabled_training_completes_preprocessing_without_artifacts(
        self,
    ) -> None:
        """RUN_TRAINING=False builds setup tables but skips inference and export."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            export_directory = Path(temporary_directory) / "training_results"
            namespace = self._disabled_run_namespace(
                export_directory, export_diffusivity=False
            )
            exec(compile(run_cell_source(), str(NOTEBOOK_PATH), "exec"), namespace)
            curve_table = namespace["curve_table"]
            self.assertEqual(
                curve_table["split"].value_counts().to_dict(),
                {"train": 2, "validation": 1, "test": 1},
            )
            self.assertTrue(
                (curve_table["initial_water_concentration_kg_m3"] > 0.0).all()
            )
            self.assertEqual(len(namespace["physics_tables"]), 4)
            batch = namespace["training_batch"]
            self.assertEqual(int(batch["interior"].shape[0]), 8)
            self.assertGreater(int(batch["rate_targets"].shape[0]), 0)
            self.assertIsInstance(namespace["model"], namespace["DryingPINN"])
            for downstream_name in (
                "history",
                "split_metrics",
                "learned_parameters",
                "diffusivity_results",
                "exported_run",
                "drying_rate_table",
                "drying_rate_figures",
            ):
                self.assertNotIn(downstream_name, namespace)
            self.assertFalse(export_directory.exists())

    def test_export_without_training_or_checkpoint_fails_before_setup(
        self,
    ) -> None:
        """EXPORT_DIFFUSIVITY=True cannot publish an untrained, unloaded model."""
        with self.assertRaisesRegex(TypeError, "RUN_TRAINING must be a bool"):
            self.namespace["validate_run_configuration"](
                run_training="False", export_diffusivity=False
            )
        self.namespace["validate_run_configuration"](
            run_training=True, export_diffusivity=True
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            export_directory = Path(temporary_directory) / "training_results"
            namespace = self._disabled_run_namespace(
                export_directory, export_diffusivity=True
            )
            with self.assertRaisesRegex(RuntimeError, "no saved Strategy A checkpoint"):
                exec(compile(run_cell_source(), str(NOTEBOOK_PATH), "exec"), namespace)
            self.assertNotIn("training_batch", namespace)
            self.assertFalse(export_directory.exists())


if __name__ == "__main__":
    unittest.main()
