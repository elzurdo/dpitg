
from scipy.optimize import fmin
from scipy.stats import beta

CI_FRACTION = 0.95

def binomial_rate_ci_width_to_sample_size(p, credible_interval_width, z_star = 1.96):
    variance_ = (0.5 *  credible_interval_width / z_star) ** 2
    n_ = p * (1 - p) / variance_  #- 1
    return n_


def HDIofICDF(dist_name, ci_fraction=CI_FRACTION, **args):
    """
    This program finds the HDI of a probability density function that is specified
    mathematically in Python.

    Example usage: HDIofICDF(beta, a=100, b=100)

    The HDI is computed numerically by minimising the interval width over the lower tail probability ℓ using uncon-
strained Nelder-Mead optimisation. $F^{-1}(1-\alpha+\ell)-F^{-1}(\ell)$ over $\ell\geq 0$; the minimiser $\ell^*$
    gives $L^*=F^{-1}(\ell^*)$ and $U^*=F^{-1}(1-\alpha+\ell^*)$.
    Credit: aloctavodia@github: https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/HDIofICDF.py
    """
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass =  1.0 - ci_fraction


    def intervalWidth(lowTailPr):
        return distri.ppf(ci_fraction + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, ci_fraction + HDIlowTailPr])

def successes_failures_to_hdi_ci_limits(a, b, ci_fraction=CI_FRACTION):
    return HDIofICDF(beta, a=a, b=b, ci_fraction=ci_fraction)