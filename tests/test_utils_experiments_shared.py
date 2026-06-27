"""
Tests for py/utils_experiments_shared.py
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
from utils_experiments_shared import stats_dict_to_df


def _make_binomial_dict(n=3):
    """Minimal valid binomial stats dict with n rows."""
    return {
        i: {
            "decision_iteration": i,
            "accept": False,
            "reject_below": False,
            "reject_above": False,
            "conclusive": False,
            "inconclusive": True,
            "successes": i,
            "failures": n - i,
            "hdi_min": 0.1 * i,
            "hdi_max": 0.1 * i + 0.2,
            "goal_achieved": False,
        }
        for i in range(1, n + 1)
    }


class TestStatsDictToDfBinomial:

    def test_returns_dataframe(self):
        df = stats_dict_to_df(_make_binomial_dict())
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self):
        df = stats_dict_to_df(_make_binomial_dict(n=5))
        assert len(df) == 5

    def test_index_name_is_experiment_number(self):
        df = stats_dict_to_df(_make_binomial_dict())
        assert df.index.name == "experiment_number"

    def test_reject_column_is_sum_of_below_and_above(self):
        d = _make_binomial_dict(n=3)
        d[1]["reject_below"] = True
        d[2]["reject_above"] = True
        df = stats_dict_to_df(d)
        assert df.loc[1, "reject"] == True
        assert df.loc[2, "reject"] == True
        assert df.loc[3, "reject"] == False

    def test_precision_is_hdi_width(self):
        df = stats_dict_to_df(_make_binomial_dict())
        expected = df["hdi_max"] - df["hdi_min"]
        pd.testing.assert_series_equal(df["precision"], expected, check_names=False)

    def test_success_rate_column_present_for_binomial(self):
        df = stats_dict_to_df(_make_binomial_dict())
        assert "success_rate" in df.columns

    def test_success_rate_values(self):
        """success_rate = successes / (successes + failures)."""
        df = stats_dict_to_df(_make_binomial_dict(n=4))
        expected = df["successes"] / (df["successes"] + df["failures"])
        pd.testing.assert_series_equal(df["success_rate"], expected, check_names=False)

    def test_hdi_columns_are_float(self):
        df = stats_dict_to_df(_make_binomial_dict())
        assert df["hdi_min"].dtype == float
        assert df["hdi_max"].dtype == float

    # TODO: I'm not sure that this is necessary.
    # Better practice may be have decision_iteration an `int` type.
    def test_decision_iteration_is_float(self):
        df = stats_dict_to_df(_make_binomial_dict())
        assert df["decision_iteration"].dtype == float

    def test_unknown_data_type_raises(self):
        with pytest.raises(ValueError, match="Unknown data_type"):
            stats_dict_to_df(_make_binomial_dict(), data_type="unknown")
