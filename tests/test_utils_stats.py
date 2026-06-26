"""
Tests for py/utils_stats.py
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
from scipy.stats import beta as scipy_beta
from utils_stats import (
    binomial_rate_ci_width_to_sample_size,
    HDIofICDF,
    successes_failures_to_hdi_ci_limits,
    CI_FRACTION,
)


class TestBinomialRateCiWidthToSampleSize:

    def test_matches_paper_equation(self):
        """n = 4*z*^2/omega^2 * p*(1-p)  (Eq. pitg_stop_iteration)."""
        p, omega, z = 0.5, 0.1, 1.96
        expected = 4 * z**2 / omega**2 * p * (1 - p)
        assert binomial_rate_ci_width_to_sample_size(p, omega, z_star=z) == pytest.approx(expected, rel=1e-9)

    def test_paper_example_omega_006(self):
        """Paper states N_goal ≈ 1067 for omega_goal=0.06, theta=0.5 (methods.tex)."""
        n = binomial_rate_ci_width_to_sample_size(0.5, 0.06)
        assert n == pytest.approx(1067.1, abs=0.5)

    def test_p_half_gives_largest_n(self):
        """p=0.5 maximises p(1-p), so it requires the most samples."""
        n_half = binomial_rate_ci_width_to_sample_size(0.5, 0.1)
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert n_half > binomial_rate_ci_width_to_sample_size(p, 0.1)

    def test_narrower_width_needs_more_samples(self):
        """Halving omega quadruples the required n (inverse-square law)."""
        n_wide = binomial_rate_ci_width_to_sample_size(0.5, 0.1)
        n_narrow = binomial_rate_ci_width_to_sample_size(0.5, 0.05)
        assert n_narrow == pytest.approx(4 * n_wide, rel=1e-9)

    def test_result_is_positive(self):
        assert binomial_rate_ci_width_to_sample_size(0.5, 0.1) > 0


class TestHDIofICDF:

    def test_symmetric_beta_centered_at_half(self):
        """Beta(100, 100) HDI should be symmetric around 0.5."""
        lo, hi = HDIofICDF(scipy_beta, a=100, b=100)
        assert lo < 0.5 < hi
        assert (lo + hi) / 2 == pytest.approx(0.5, abs=0.005)

    def test_returns_two_ordered_values(self):
        result = HDIofICDF(scipy_beta, a=10, b=10)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_bounds_within_0_1(self):
        lo, hi = HDIofICDF(scipy_beta, a=5, b=20)
        assert 0.0 <= lo < hi <= 1.0

    def test_width_shrinks_with_more_data(self):
        """Higher concentration (larger a+b) → narrower HDI."""
        lo_small, hi_small = HDIofICDF(scipy_beta, a=10, b=10)
        lo_large, hi_large = HDIofICDF(scipy_beta, a=100, b=100)
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_symmetric_beta_hdi_matches_equal_tailed(self):
        """For symmetric Beta, HDI equals the equal-tailed credible interval."""
        lo, hi = HDIofICDF(scipy_beta, a=100, b=100, ci_fraction=0.95)
        dist = scipy_beta(100, 100)
        assert lo == pytest.approx(dist.ppf(0.025), abs=0.002)
        assert hi == pytest.approx(dist.ppf(0.975), abs=0.002)

    def test_custom_ci_fraction_narrower(self):
        """90% HDI should be narrower than 95% HDI."""
        lo_90, hi_90 = HDIofICDF(scipy_beta, a=50, b=50, ci_fraction=0.90)
        lo_95, hi_95 = HDIofICDF(scipy_beta, a=50, b=50, ci_fraction=0.95)
        assert (hi_90 - lo_90) < (hi_95 - lo_95)

    def test_default_ci_fraction_matches_constant(self):
        """Default ci_fraction should equal the module-level CI_FRACTION."""
        lo_default, hi_default = HDIofICDF(scipy_beta, a=50, b=50)
        lo_explicit, hi_explicit = HDIofICDF(scipy_beta, a=50, b=50, ci_fraction=CI_FRACTION)
        assert lo_default == pytest.approx(lo_explicit, rel=1e-6)
        assert hi_default == pytest.approx(hi_explicit, rel=1e-6)


class TestSuccessesFailuresToHdiCiLimits:

    def test_symmetric_posterior_centered_at_half(self):
        """Equal successes and failures → HDI symmetric around 0.5."""
        lo, hi = successes_failures_to_hdi_ci_limits(100, 100)
        assert lo < 0.5 < hi
        assert (lo + hi) / 2 == pytest.approx(0.5, abs=0.005)

    def test_bounds_within_0_1(self):
        for a, b in [(1, 1), (2, 5), (50, 50), (1, 100), (100, 1)]:
            lo, hi = successes_failures_to_hdi_ci_limits(a, b)
            assert 0.0 <= lo < hi <= 1.0, f"Failed for Beta({a},{b})"

    def test_width_shrinks_with_more_data(self):
        lo_small, hi_small = successes_failures_to_hdi_ci_limits(10, 10)
        lo_large, hi_large = successes_failures_to_hdi_ci_limits(100, 100)
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_skewed_posterior_above_half(self):
        """Beta(90, 10): HDI lower bound should be well above 0.5."""
        lo, hi = successes_failures_to_hdi_ci_limits(90, 10)
        assert lo > 0.5

    def test_matches_hdIofICDF_directly(self):
        """successes_failures_to_hdi_ci_limits is a thin wrapper over HDIofICDF."""
        lo_wrap, hi_wrap = successes_failures_to_hdi_ci_limits(40, 60)
        lo_direct, hi_direct = HDIofICDF(scipy_beta, a=40, b=60)
        assert lo_wrap == pytest.approx(lo_direct, rel=1e-9)
        assert hi_wrap == pytest.approx(hi_direct, rel=1e-9)

    def test_custom_ci_fraction(self):
        """90% HDI should be narrower than 95% HDI."""
        lo_90, hi_90 = successes_failures_to_hdi_ci_limits(50, 50, ci_fraction=0.90)
        lo_95, hi_95 = successes_failures_to_hdi_ci_limits(50, 50, ci_fraction=0.95)
        assert (hi_90 - lo_90) < (hi_95 - lo_95)
