"""
Tests for py/utils_stats.py
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
from utils_stats import binomial_rate_ci_width_to_sample_size


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
