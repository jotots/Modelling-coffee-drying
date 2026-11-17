"""Focused tests for the illustrative drying-curve generator script."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1] / "notebooks" / "Drying_curve_generator.py"
)


def load_generator():
    """Import the standalone script as a module without running main()."""
    spec = importlib.util.spec_from_file_location(
        "drying_curve_generator", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DryingCurveKineticsTests(unittest.TestCase):
    """Check the physical consistency of the schematic curves."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()
        cls.params = cls.gen.DryingCurveParameters()
        cls.curve = cls.gen.compute_drying_curve(cls.params, n_points=4001)

    def test_rate_is_negative_derivative_of_moisture(self):
        numerical_rate = -np.gradient(self.curve.moisture, self.curve.time)
        # Skip the two edge samples, where np.gradient is one-sided.
        np.testing.assert_allclose(
            numerical_rate[1:-1], self.curve.rate[1:-1], atol=2e-3
        )

    def test_moisture_and_rate_continuous_at_transitions(self):
        p = self.params
        for boundary in (p.warmup_end_time, self.curve.critical_time):
            for function in (
                self.gen.moisture_at,
                self.gen.drying_rate_at,
                self.gen.temperature_at,
            ):
                before = function(p, boundary - 1e-9)
                after = function(p, boundary + 1e-9)
                self.assertAlmostEqual(before, after, places=6)

    def test_critical_point_matches_phase_boundary(self):
        p = self.params
        self.assertAlmostEqual(
            self.gen.moisture_at(p, self.curve.critical_time),
            p.critical_moisture,
            places=10,
        )
        # The rate is constant up to the critical time and falls after it.
        self.assertAlmostEqual(
            self.gen.drying_rate_at(p, self.curve.critical_time),
            p.constant_rate,
            places=10,
        )
        self.assertLess(
            self.gen.drying_rate_at(p, self.curve.critical_time + 0.1),
            p.constant_rate,
        )

    def test_equilibrium_is_only_approached_asymptotically(self):
        falling = self.curve.time > self.curve.critical_time
        self.assertTrue(
            np.all(self.curve.moisture > self.params.equilibrium_moisture)
        )
        self.assertTrue(np.all(self.curve.rate[falling] > 0.0))
        self.assertTrue(np.all(np.diff(self.curve.rate[falling]) < 0.0))
        self.assertLess(self.curve.rate[-1], 0.02 * self.params.constant_rate)

    def test_temperature_phases(self):
        p = self.params
        t = self.curve.time
        temperature = self.curve.temperature
        warmup = t <= p.warmup_end_time
        constant = (t >= p.warmup_end_time) & (t <= self.curve.critical_time)
        falling = t >= self.curve.critical_time
        self.assertTrue(np.all(np.diff(temperature[warmup]) > 0.0))
        np.testing.assert_allclose(
            temperature[constant], p.wet_bulb_temperature
        )
        self.assertTrue(np.all(np.diff(temperature[falling]) > 0.0))
        self.assertTrue(np.all(temperature < p.air_temperature))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.gen.DryingCurveParameters(
                critical_moisture=0.05, equilibrium_moisture=0.08
            )
        with self.assertRaises(ValueError):
            self.gen.DryingCurveParameters(
                wet_bulb_temperature=70.0, air_temperature=60.0
            )
        with self.assertRaises(ValueError):
            self.gen.compute_drying_curve(self.params, n_points=1)


class DryingCurveOutputTests(unittest.TestCase):
    """Check output location, file formats and Portuguese annotation."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_default_output_dir_is_resolved_from_script_path(self):
        expected = (
            SCRIPT_PATH.resolve().parents[1] / "src" / "drying_curve"
        )
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(self.gen.default_output_dir(), expected)
            finally:
                os.chdir(previous)

    def test_figure_has_portuguese_labels_and_illustrative_note(self):
        figure = self.gen.build_figure(self.gen.DryingCurveParameters())
        labels = " ".join(axis.get_ylabel() for axis in figure.axes)
        self.assertEqual(len(figure.axes), 3)
        self.assertIn("Teor de umidade", labels)
        self.assertIn("Taxa de secagem", labels)
        self.assertIn("Temperatura do produto", labels)
        self.assertIn("Tempo de secagem", figure.axes[0].get_xlabel())
        texts = " ".join(text.get_text() for text in figure.texts)
        self.assertIn("dados arbitrários", texts)

    def test_save_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.gen.generate_drying_curve_figure(Path(directory))
            suffixes = sorted(path.suffix for path in paths)
            self.assertEqual(suffixes, [".pdf", ".png"])
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
            pdf = next(path for path in paths if path.suffix == ".pdf")
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
