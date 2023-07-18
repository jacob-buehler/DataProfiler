"""Validates that generator intakes DATAPROFILER_SEED properly."""
import os
import unittest

import numpy as np
from numpy.random import PCG64

from .. import utils_global


class TestOriginalFunction(unittest.TestCase):
    """Validates get_random_number_generator() is properly working."""

    @unittest.mock.patch("os.environ")
    def test_return_random_value(self, *mocks):
        """If DATAPROFILER_SEED not set, test that rng returns different random value."""
        rng = utils_global.get_random_number_generator()
        sample_value = rng.integers(0, 1e6)
        self.assertNotEqual(sample_value, 850624)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_return_default_rng_with_settings_seed(self):
        """If DATAPROFILER_SEED not set, test that rng uses seed = settings._seed."""
        rng = utils_global.get_random_number_generator()
        actual_value = rng.integers(0, 100)
        expected_value_generator = np.random.default_rng(123)
        expected_value = expected_value_generator.integers(0, 100)
        self.assertEqual(actual_value, expected_value)

    @unittest.mock.patch.dict(os.environ, {"DATAPROFILER_SEED": "0"}, clear=True)
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_dataprofiler_seed_true_settings_seed_false(self):
        """Verify that we get the expected result when DATAPROFILER_SEED in os.environ and settings._seed!=None."""
        rng = utils_global.get_random_number_generator()
        actual_value = rng.integers(0, 100)
        expected_value_generator = np.random.default_rng(123)
        expected_value = expected_value_generator.integers(0, 100)
        self.assertEqual(actual_value, expected_value)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    def test_dataprofiler_seed_false_settings_seed_true(self):
        """Verify that we get the expected result when DATAPROFILER_SEED not in os.environ and settings._seed==None."""
        with unittest.mock.patch.dict("os.environ"):
            del os.environ["DATAPROFILER_SEED"]
            rng = utils_global.get_random_number_generator()
            actual_value = rng.integers(0, 1e6)
        self.assertNotEqual(actual_value, 850624)

    @unittest.mock.patch.dict(os.environ, {"DATAPROFILER_SEED": "123"}, clear=True)
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    def test_dataprofiler_seed_true_settings_seed_true(self):
        """Verify that we get the expected result when DATAPROFILER_SEED in os.environ and settings._seed==None."""
        rng = utils_global.get_random_number_generator()
        actual_value = rng.integers(0, 1e6)
        expected_value_generator = np.random.default_rng(123)
        expected_value = expected_value_generator.integers(0, 1e6)
        self.assertEqual(actual_value, expected_value)

    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_dataprofiler_seed_false_settings_seed_false(self):
        """Verify that we get the expected result when DATAPROFILER_SEED not in os.environ and settings._seed!=None."""
        with unittest.mock.patch.dict("os.environ"):
            del os.environ["DATAPROFILER_SEED"]
            rng = utils_global.get_random_number_generator()
            actual_value = rng.integers(0, 1e6)
            expected_value_generator = np.random.default_rng(123)
            expected_value = expected_value_generator.integers(0, 1e6)
        self.assertEqual(actual_value, expected_value)

    @unittest.mock.patch.dict(
        os.environ, {"DATAPROFILER_SEED": "George Washington"}, clear=True
    )
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=None)
    def test_warning_raised(self):
        with self.assertWarnsRegex(RuntimeWarning, "Seed should be an integer"):
            rng = utils_global.get_random_number_generator()
            actual_value = rng.integers(0, 1e6)
        self.assertNotEqual(actual_value, 850624)

    @unittest.mock.patch.dict(os.environ, {"DATAPROFILER_SEED": "0"}, clear=True)
    @unittest.mock.patch("dataprofiler.utils_global.settings._seed", new=123)
    def test_try_returned(self):
        rng = utils_global.get_random_number_generator()
        actual_value = rng.integers(0, 100)
        expected_value_generator = np.random.default_rng(123)
        expected_value = expected_value_generator.integers(0, 100)
        self.assertEqual(actual_value, expected_value)
