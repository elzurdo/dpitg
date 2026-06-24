
CI_FRACTION = 0.95

def binomial_rate_ci_width_to_sample_size(p, credible_interval_width, z_star = 1.96):
    variance_ = (0.5 *  credible_interval_width / z_star) ** 2
    n_ = p * (1 - p) / variance_ - 1
    return n_