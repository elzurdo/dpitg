"""
Tests for py/utils_experiments_binomial.py
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
from utils_experiments_binomial import (
    BinaryAccounting,
    successes_failures_caculate_hdi_limits,
    sequence_to_iteration_stats,
    SEQUENCE_HANDPICKED
    )


# All results up to iteration 804 are sufficient.
# This reduces pytest by a factor of 2 (compared to 1,500 iterations ...)
SEQUENCE_LIMIT = 810 
sequence = SEQUENCE_HANDPICKED[:SEQUENCE_LIMIT]
class TestBinaryAccounting:

    def test_returns_two_floats(self):
        ba = BinaryAccounting()
        lo, hi = ba.successes_failures_to_hdi_limits(50, 50)
        assert isinstance(lo, float) and isinstance(hi, float)
        assert lo < hi

    def test_first_call_populates_cache(self):
        ba = BinaryAccounting()
        ba.successes_failures_to_hdi_limits(10, 20)
        assert (10, 20) in ba.dict_successes_failures_hdi_limits
        assert ba.dict_successes_failures_counter[(10, 20)] == 1

    def test_second_call_increments_counter(self):
        ba = BinaryAccounting()
        ba.successes_failures_to_hdi_limits(10, 20)
        ba.successes_failures_to_hdi_limits(10, 20)
        assert ba.dict_successes_failures_counter[(10, 20)] == 2

    def test_cached_result_is_identical(self):
        """Second call must return the exact same object, not a recomputed one."""
        ba = BinaryAccounting()
        result_first = ba.successes_failures_to_hdi_limits(30, 70)
        result_second = ba.successes_failures_to_hdi_limits(30, 70)
        assert result_first is result_second

    def test_different_pairs_cached_independently(self):
        ba = BinaryAccounting()
        ba.successes_failures_to_hdi_limits(10, 10)
        ba.successes_failures_to_hdi_limits(20, 20)
        ba.successes_failures_to_hdi_limits(10, 10)
        assert ba.dict_successes_failures_counter[(10, 10)] == 2
        assert ba.dict_successes_failures_counter[(20, 20)] == 1

    def test_result_matches_direct_calculation(self):
        """BinaryAccounting should return the same HDI as successes_failures_caculate_hdi_limits."""
        ba = BinaryAccounting()
        lo_ba, hi_ba = ba.successes_failures_to_hdi_limits(40, 60)
        lo_direct, hi_direct = successes_failures_caculate_hdi_limits(40, 60)
        assert lo_ba == pytest.approx(lo_direct, rel=1e-9)
        assert hi_ba == pytest.approx(hi_direct, rel=1e-9)

    def test_zero_failures_handled(self):
        """Zero failures triggers Laplace smoothing — should not raise."""
        ba = BinaryAccounting()
        lo, hi = ba.successes_failures_to_hdi_limits(10, 0)
        assert 0.0 < lo < hi <= 1.0

    def test_zero_successes_handled(self):
        """Zero successes triggers Laplace smoothing — should not raise."""
        ba = BinaryAccounting()
        lo, hi = ba.successes_failures_to_hdi_limits(0, 10)
        assert 0.0 <= lo < hi < 1.0


class TestSequenceToIterationStats:

    # Paper setup: fair coin, theta_null=0.5, ROPE=[0.45, 0.55], omega_goal=0.08
    ROPE_MIN = 0.45
    ROPE_MAX = 0.55
    OMEGA_GOAL = 0.08

    def _handpicked_df(self, binary_accounting=None):
        
        samples = np.array([int(c) for c in sequence])
        return sequence_to_iteration_stats(
            samples, self.OMEGA_GOAL, self.ROPE_MIN, self.ROPE_MAX,
            binary_accounting=binary_accounting,
        )

    def test_returns_dataframe_with_expected_columns(self):
        df = self._handpicked_df()
        assert isinstance(df, pd.DataFrame)
        for col in ["hdi_min", "hdi_max", "precision", "conclusive", "goal_achieved",
                    "accept", "reject_below", "reject_above", "successes", "failures"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_equals_sequence_length(self):
        df = self._handpicked_df()
        assert len(df) == len(sequence)

    def test_hdi_width_shrinks_over_time(self):
        """More data → narrower HDI on average; last quarter narrower than first."""
        df = self._handpicked_df()
        early_width = df["precision"].iloc[:100].mean()
        late_width = df["precision"].iloc[-100:].mean()
        assert late_width < early_width

    def test_conclusive_first_at_iteration_126(self):
        """Paper: HDI+ROPE stops at iteration 126 for the handpicked sequence."""
        df = self._handpicked_df()
        first_conclusive = df[df["conclusive"]].index[0]
        assert first_conclusive == 126

    def test_precision_goal_first_met_at_iteration_598(self):
        """Paper: PitG stops at iteration 598 for the handpicked sequence."""
        df = self._handpicked_df()
        first_goal = df[df["goal_achieved"]].index[0]
        assert first_goal == 598

    def test_dpitg_stop_at_iteration_804(self):
        """Paper: DPitG stops at iteration 804 (first iter where both criteria hold)."""
        df = self._handpicked_df()
        both = df[df["conclusive"] & df["goal_achieved"]]
        assert both.index[0] == 804

    def test_inconclusive_is_complement_of_conclusive(self):
        df = self._handpicked_df()
        assert (df["conclusive"] != df["inconclusive"]).all()

    def test_success_rate_bounded(self):
        df = self._handpicked_df()
        assert (df["success_rate"] >= 0).all() and (df["success_rate"] <= 1).all()

    def test_binary_accounting_gives_same_result(self):
        """Cached and uncached paths agree once both successes and failures are non-zero.
        They differ on zero-count rows: BinaryAccounting applies Laplace smoothing via
        successes_failures_caculate_hdi_limits, while the direct path passes zeros straight
        to Beta(a, 0) or Beta(0, b), which is undefined and yields nan."""
        df_no_cache = self._handpicked_df()
        df_cached = self._handpicked_df(binary_accounting=BinaryAccounting())
        mask = (df_no_cache["successes"] > 0) & (df_no_cache["failures"] > 0)
        pd.testing.assert_frame_equal(df_no_cache[mask], df_cached[mask])
