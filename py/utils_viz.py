import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from utils_stats import (CI_FRACTION,
                        #  successes_failures_to_hdi_ci_limits,
                        #  get_success_rates,
                        #  beta,
                        #  test_value,
                         binomial_rate_ci_width_to_sample_size,
                         )

FIG_WIDTH = 8
FIG_HEIGHT = 6

def plot_n_goal_by_parameter(z_star = 1.96 ):

    thetas = np.arange(0.01, 0.99, 0.01)
    goals = [0.1, 0.08, 0.06, 0.04]

    n_stop_goals = {goal: [binomial_rate_ci_width_to_sample_size(theta, goal, z_star=z_star)
                           for theta in thetas] for goal in goals}
    df_n_stop_goals = pd.DataFrame(n_stop_goals, index=thetas)

    plt.figure(figsize=(1 * FIG_WIDTH, FIG_HEIGHT))


    for idx, goal in enumerate(goals):
        plt.plot(df_n_stop_goals.index, df_n_stop_goals[goal],
                 label=f"{goal:.2f}", linewidth=idx + 1)
    plt.legend(title=r"$\omega_{\rm goal}$")
    plt.grid(alpha=0.3)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$N_{\rm goal}(\theta,\, \omega_{\rm goal})$")

    plt.title(
        r"Minimum $N_{\rm goal}$ to Achieve Precision Goal $\omega_{\rm goal}$",
        fontsize=20
    )
    plt.tight_layout()