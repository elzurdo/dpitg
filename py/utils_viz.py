import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as scipy_beta

from utils_stats import (CI_FRACTION,
                        successes_failures_to_hdi_ci_limits,
                        #  get_success_rates,
                        #  test_value,
                         binomial_rate_ci_width_to_sample_size,
                         )

FIG_WIDTH = 8
FIG_HEIGHT = 6

theta_str = r"$\theta$"
hat_theta_str = r"$\hat{\theta}$"  # was called theta_hat_str
theta_null_str = r"$\theta_{\rm null}$"
theta_true_str = r"$\theta_{\rm true}$"
p_theta_str = r"$p(\theta)$"
delta_rope_str = r"$\Delta_{\rm ROPE}$"
n_true_str = r"$N_{\rm goal}$"
n_stop_str = r"$N_{\rm stop}$"
ω_goal_str = r"$\omega_{\rm goal}$"  # was called goal_str
ω_hdi_str = r"$\omega_{\rm HDI}$"  # was called width_str and omega_hdi_str

# TODO: refactor to use plt.grid instead of this function
def plot_vhlines_lines(vertical=None, horizontal=0, color="black", ax=None, alpha=0.2, linestyle=None, linewidth=1, label=None):
    if ax is None:
        ax = plt.gca()

    if horizontal is not None:
	    ax.axhline(horizontal, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)
        
    if vertical is not None:
        ax.axvline(vertical, color=color, linewidth=linewidth, alpha=alpha,linestyle=linestyle, label=label)


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


def viz_sequence_stats(df_sample_results, precision_goal, rope_min, rope_max, θ_true=None):
    # Was called: viz_one_sample_results
    df_conclusive_accept = df_sample_results.query("conclusive").query("accept")
    df_conclusive_reject = df_sample_results.query("conclusive").query("reject")
    df_sample_goal = df_sample_results.query("goal_achieved")

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    plt.plot(df_sample_results["decision_iteration"], df_sample_results["hdi_min"], color="gray", label=None)
    plt.plot(df_sample_results["decision_iteration"], df_sample_results["hdi_max"], color="gray", label=None)
    plt.fill_between(df_sample_results["decision_iteration"], df_sample_results["hdi_max"], df_sample_results["hdi_min"], color='gray', alpha=0.2, label="HDI")

    # experiments which are conclusive to accept null hypothesis
    for idx, (iteration, row) in enumerate(df_conclusive_accept.iterrows()):
        if idx == 0:
            label = "conclusive: accept"
        else:
            label = None
        plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], color='lightgreen', alpha=0.7, linewidth=1, label=label)

    # experiments which are conclusive to reject null hypothesis
    for idx, (iteration, row) in enumerate(df_conclusive_reject.iterrows()):
        if idx == 0:
            label = "conclusive: reject"
        else:
            label = None
        plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], color='red', alpha=0.7, linewidth=1, label=label, linestyle=":")

    #for iteration, row in df_sample_goal.iterrows():
    #    plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], color='blue', alpha=0.1, linewidth=1)
    plt.scatter(df_sample_goal["decision_iteration"], df_sample_goal["hdi_min"], color="purple", label=f"{ω_hdi_str}≤{ω_goal_str}={precision_goal:0.3} met", marker="o", s=20)
    plt.scatter(df_sample_goal["decision_iteration"], df_sample_goal["hdi_max"], color="purple", label=None, marker="o", s=20)

    plot_vhlines_lines(vertical=None, label='ROPE', horizontal=rope_min, linestyle="--", color="orange", alpha=0.7)
    plot_vhlines_lines(vertical=None, horizontal=rope_max, linestyle="--", color="orange", alpha=0.7)

    plt.legend()
    plt.xlabel(r"$N$")
    plt.ylabel(f"{hat_theta_str}")

    if θ_true is not None:
        plt.title(f"{theta_true_str}={θ_true:0.3f}")


def plot_pdf(sr_experiment_stats, rope_min, rope_max, xlim=None, xtitle=r"success rate $\theta$"):
    pp = np.linspace(0, 1, 1000)
    pp_hdi = np.linspace(sr_experiment_stats["hdi_min"], sr_experiment_stats["hdi_max"], 1000)

    successes = sr_experiment_stats["successes"]
    failures = sr_experiment_stats["failures"]
    rate = successes / (successes + failures)
    n_ = successes + failures

    hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(successes, failures)
    print(hdi_min, hdi_max)

    pdf = scipy_beta.pdf(pp, successes, failures)
    pdf_hdi = scipy_beta.pdf(pp_hdi, successes, failures)

    plt.plot(pp, pdf, color="purple", label=f"pdf {hat_theta_str}={rate:0.3f}; n={n_:,}")
    label_hdi = f"{ω_hdi_str}={hdi_max - hdi_min:0.3f}"
    plt.fill_between(pp_hdi, pdf_hdi, color="purple", alpha=0.2, label=label_hdi)
    plot_vhlines_lines(vertical=rope_min, label='ROPE', horizontal=None, linestyle="--")
    plot_vhlines_lines(vertical=rope_max, horizontal=None, linestyle="--")
    plt.legend()

    if xtitle is not None:
        plt.xlabel(xtitle)
    plt.ylabel(p_theta_str)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([rope_min - 0.1, rope_max + 0.1])


def plot_sample_pdf_methods(method_df_stats, isample, rope_min, rope_max, xlim = (0.2, 0.6), method_names=None):

    if method_names is None:
        method_names = list(method_df_stats.keys())

    ncols, nrows = 1, len(method_names)

    plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, 1.2* FIG_HEIGHT))

    for imethod, method_name in enumerate(method_names):
        experiment_stats = method_df_stats[method_name].loc[isample]

        plt.subplot(nrows, ncols, imethod + 1)

        if imethod == len(method_names) - 1:
            xtitle = r"$\theta$"
        else:
            xtitle = None
        plot_pdf(experiment_stats, rope_min, rope_max, xlim=xlim, xtitle=xtitle)
        n_stop = experiment_stats["successes"] + experiment_stats["failures"]
        plt.title(f"Stop Iteration: {n_stop:,}")
        #plt.title(f"{METHOD_FULL[method_name]}")

    #plt.suptitle(f"Outcomes depending on Stop Criterion", fontsize=18)
    plt.tight_layout()