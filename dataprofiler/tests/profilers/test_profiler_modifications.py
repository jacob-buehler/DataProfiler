### Begin importing packages
pass
pass
import os

pass
pass
import unittest

pass
pass

import networkx as nx
import numpy as np
import pandas as pd

import dataprofiler
import dataprofiler as dp
from dataprofiler import StructuredDataLabeler, UnstructuredDataLabeler
from dataprofiler.profilers.column_profile_compilers import (
    ColumnDataLabelerCompiler,
    ColumnPrimitiveTypeProfileCompiler,
    ColumnStatsProfileCompiler,
)
from dataprofiler.profilers.graph_profiler import GraphProfiler
from dataprofiler.profilers.helpers.report_helpers import _prepare_report
from dataprofiler.profilers.profile_builder import (
    Profiler,
    StructuredColProfiler,
    StructuredProfiler,
    UnstructuredCompiler,
    UnstructuredProfiler,
)
from dataprofiler.profilers.profiler_options import (
    ProfilerOptions,
    StructuredOptions,
    UnstructuredOptions,
)

from . import utils as test_utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
### End importing packages


class TestStructuredProfilerRowStatistics(unittest.TestCase):
    def setUp(self):
        test_utils.set_seed(0)

    @classmethod
    def setUpClass(cls):
        test_utils.set_seed(seed=0)

        data = {
            "names": [
                "orange",
                "green",
                "blue",
                "mexico",
                "france",
                "morocco",
                "chevy",
                "ford",
                "toyota",
                "apple",
            ]
            * 2,
            "numbers": [1, 2, 3, 4, 5] * 4,
            "tf_null": [None, 1, None, 2] * 5,
        }

        cls.data = pd.DataFrame(data)

    def test_correct_null_row_counts(self):
        file_path = os.path.join(test_root_path, "data", "csv/empty_rows.txt")
        data = pd.read_csv(file_path)
        profiler_options = ProfilerOptions()
        profiler_options.set(
            {
                "*.is_enabled": False,
                "row_statistics.is_enabled": True,
            }
        )
        profile = dp.StructuredProfiler(data, options=profiler_options)
        self.assertEqual(2, profile.row_has_null_count)
        self.assertEqual(0.25, profile._get_row_has_null_ratio())
        self.assertEqual(2, profile.row_is_null_count)
        self.assertEqual(0.25, profile._get_row_is_null_ratio())

        file_path = os.path.join(test_root_path, "data", "csv/iris-with-null-rows.csv")
        data = pd.read_csv(file_path)
        profile = dp.StructuredProfiler(data, options=profiler_options)
        self.assertEqual(13, profile.row_has_null_count)
        self.assertEqual(13 / 24, profile._get_row_has_null_ratio())
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(3 / 24, profile._get_row_is_null_ratio())

    def test_row_is_null_ratio_row_stats_disabled(self):
        profiler_options_1 = ProfilerOptions()
        profiler_options_1.set(
            {
                "*.is_enabled": False,
            }
        )
        profiler = StructuredProfiler(pd.DataFrame([]), options=profiler_options_1)
        self.assertIsNone(profiler._get_row_is_null_ratio())

    def test_row_has_null_ratio_row_stats_disabled(self):
        profiler_options_1 = ProfilerOptions()
        profiler_options_1.set(
            {
                "*.is_enabled": False,
            }
        )
        profiler = StructuredProfiler(pd.DataFrame([]), options=profiler_options_1)
        self.assertIsNone(profiler._get_row_has_null_ratio())

    def test_null_in_file(self):
        filename_null_in_file = os.path.join(
            test_root_path, "data", "csv/sparse-first-and-last-column.txt"
        )
        profiler_options = ProfilerOptions()
        profiler_options.set(
            {
                "*.is_enabled": False,
                "row_statistics.is_enabled": True,
            }
        )
        data = dp.Data(filename_null_in_file)
        profile = dp.StructuredProfiler(data, options=profiler_options)

        report = profile.report(report_options={"output_format": "pretty"})
        count_idx = report["global_stats"]["profile_schema"]["COUNT"][0]
        numbers_idx = report["global_stats"]["profile_schema"][" NUMBERS"][0]

        self.assertEqual(
            report["data_stats"][count_idx]["statistics"]["null_types_index"],
            {"": "[2, 3, 4, 5, 7, 8]"},
        )

        self.assertEqual(
            report["data_stats"][numbers_idx]["statistics"]["null_types_index"],
            {"": "[5, 6, 8]", " ": "[2, 4]"},
        )

    def test_null_calculation_with_differently_sampled_cols(self):
        opts = ProfilerOptions()
        opts.set(
            {
                "*.is_enabled": False,
                "row_statistics.is_enabled": True,
            }
        )
        data = pd.DataFrame(
            {
                "full": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "sparse": [1, None, 3, None, 5, None, 7, None, 9],
            }
        )
        profile = dp.StructuredProfiler(
            data, samples_per_update=5, min_true_samples=5, options=opts
        )
        # Rows 2, 4, 5, 6, 7 are sampled in first column
        # Therefore only those rows should be considered for null calculations
        # The only null in those rows in second column in that subset are 5, 7
        # Therefore only 2 rows have null according to row_has_null_count
        self.assertEqual(0, profile.row_is_null_count)
        self.assertEqual(2, profile.row_has_null_count)
        # Accordingly, make sure ratio of null rows accounts for the fact that
        # Only 5 total rows were sampled (5 in col 1, 9 in col 2)
        self.assertEqual(0, profile._get_row_is_null_ratio())
        self.assertEqual(0.4, profile._get_row_has_null_ratio())

        data2 = pd.DataFrame(
            {
                "sparse": [1, None, 3, None, 5, None, 7, None],
                "sparser": [1, None, None, None, None, None, None, 8],
            }
        )
        profile2 = dp.StructuredProfiler(
            data2, samples_per_update=2, min_true_samples=2, options=opts
        )
        # Rows are sampled as follows: [6, 5], [1, 4], [2, 3], [0, 7]
        # First column gets min true samples from ids 1, 4, 5, 6
        # Second column gets completely sampled (has a null in 1, 4, 5, 6)
        # rows 1 and 5 are completely null, 4 and 6 only null in col 2
        self.assertEqual(2, profile2.row_is_null_count)
        self.assertEqual(4, profile2.row_has_null_count)
        # Only 4 total rows sampled, ratio accordingly
        self.assertEqual(0.5, profile2._get_row_is_null_ratio())
        self.assertEqual(1, profile2._get_row_has_null_ratio())

    def test_null_row_stats_correct_after_updates(self, *mocks):
        data1 = pd.DataFrame([[1, None], [1, 1], [None, None], [None, 1]])
        data2 = pd.DataFrame([[None, None], [1, None], [None, None], [None, 1]])
        opts = ProfilerOptions()
        opts.set(
            {
                "*.is_enabled": False,
                "row_statistics.is_enabled": True,
            }
        )

        # When setting min true samples/samples per update
        profile = dp.StructuredProfiler(
            data1, min_true_samples=2, samples_per_update=2, options=opts
        )
        self.assertEqual(3, profile.row_has_null_count)
        self.assertEqual(1, profile.row_is_null_count)
        self.assertEqual(0.75, profile._get_row_has_null_ratio())
        self.assertEqual(0.25, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3}, profile._profile[0].null_types_index["nan"])
        self.assertSetEqual({0, 2}, profile._profile[1].null_types_index["nan"])

        profile.update_profile(data2, min_true_samples=2, sample_size=2)
        self.assertEqual(7, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(0.875, profile._get_row_has_null_ratio())
        self.assertEqual(0.375, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual(
            {2, 3, 4, 6, 7}, profile._profile[0].null_types_index["nan"]
        )
        self.assertSetEqual(
            {0, 2, 4, 5, 6}, profile._profile[1].null_types_index["nan"]
        )

        # When not setting min true samples/samples per update
        opts = ProfilerOptions()
        opts.set(
            {
                "*.is_enabled": False,
                "row_statistics.is_enabled": True,
            }
        )
        profile = dp.StructuredProfiler(data1, options=opts)
        self.assertEqual(3, profile.row_has_null_count)
        self.assertEqual(1, profile.row_is_null_count)
        self.assertEqual(0.75, profile._get_row_has_null_ratio())
        self.assertEqual(0.25, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual({2, 3}, profile._profile[0].null_types_index["nan"])
        self.assertSetEqual({0, 2}, profile._profile[1].null_types_index["nan"])

        profile.update_profile(data2)
        self.assertEqual(7, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(0.875, profile._get_row_has_null_ratio())
        self.assertEqual(0.375, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual(
            {2, 3, 4, 6, 7}, profile._profile[0].null_types_index["nan"]
        )
        self.assertSetEqual(
            {0, 2, 4, 5, 6}, profile._profile[1].null_types_index["nan"]
        )

        # Test that update with emtpy data doesn't change stats
        profile.update_profile(pd.DataFrame([]))
        self.assertEqual(7, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(0.875, profile._get_row_has_null_ratio())
        self.assertEqual(0.375, profile._get_row_is_null_ratio())
        self.assertEqual(4, profile._min_sampled_from_batch)
        self.assertSetEqual(
            {2, 3, 4, 6, 7}, profile._profile[0].null_types_index["nan"]
        )
        self.assertSetEqual(
            {0, 2, 4, 5, 6}, profile._profile[1].null_types_index["nan"]
        )

        # Test one row update
        profile.update_profile(pd.DataFrame([[1, None]]))
        self.assertEqual(8, profile.row_has_null_count)
        self.assertEqual(3, profile.row_is_null_count)
        self.assertEqual(8 / 9, profile._get_row_has_null_ratio())
        self.assertEqual(3 / 9, profile._get_row_is_null_ratio())
        self.assertEqual(1, profile._min_sampled_from_batch)
        self.assertSetEqual(
            {2, 3, 4, 6, 7}, profile._profile[0].null_types_index["nan"]
        )
        self.assertSetEqual(
            {0, 2, 4, 5, 6}, profile._profile[1].null_types_index["nan"]
        )
        # Weird pandas behavior makes this None since this column will be
        # recognized as object, not float64
        self.assertSetEqual({8}, profile._profile[1].null_types_index["None"])

        # Tests row stats disabled
        options = StructuredOptions()
        options.set(
            {
                "*.is_enabled": False,
                "row_statistics.is_enabled": False,
            }
        )
        profile2 = StructuredProfiler(data1, options=options)
        self.assertEqual(0, profile2.row_is_null_count)
        self.assertEqual(0, profile2.row_has_null_count)


if __name__ == "__main__":
    unittest.main()
