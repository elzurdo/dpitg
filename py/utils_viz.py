import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as scipy_beta

import matplotlib.gridspec as gridspec

from utils_stats import (CI_FRACTION,
                        successes_failures_to_hdi_ci_limits,
                        #  get_success_rates,
                        #  test_value,
                         binomial_rate_ci_width_to_sample_size,
                         )

DPI = 350  # dots per inch for saving figures

FIG_WIDTH = 8
FIG_HEIGHT = 6
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

ALGO_COLORS = {"pitg": "blue", "dpitg": "lightgreen", "hdi_rope": "red"}
ALGO_LINEWIDTH =  {"hdi_rope":1, "pitg": 2, "dpitg":3}

method_pretty_short_name = {
    "pitg": "PitG",
    "dpitg": "DPitG",
    "hdi_rope": "HDI+ROPE"
}

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


def plot_vhlines_lines(vertical=None, horizontal=0, color="black", ax=None, alpha=0.2, linestyle=None, linewidth=1, label=None):
    if ax is None:
        ax = plt.gca()

    if horizontal is not None:
	    ax.axhline(horizontal, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)
        
    if vertical is not None:
        ax.axvline(vertical, color=color, linewidth=linewidth, alpha=alpha,linestyle=linestyle, label=label)


# TODO: refactor to use plt.grid instead of this function
def plot_grid(with_y=True, with_x=False, alpha=0.3):
    ax = plt.gca()

    if with_y:
        ax.grid(axis="y", alpha=alpha)
    else:
        ax.grid(False, axis="y")

    if with_x:
        ax.grid(axis="x", alpha=alpha)
    else:
        ax.grid(False, axis="x")

FIGURE_PATH_ROOT = "./figures/"

def save_figure(fig, filename, dpi=DPI, bbox_inches="tight"):
    if filename is not None:
        filepath_png = f"{FIGURE_PATH_ROOT}png/{filename}_dpi{DPI}.png"
        filepath_tiff = f"{FIGURE_PATH_ROOT}tiff/{filename}_dpi{DPI}.tiff"
        fig.savefig(filepath_png, dpi=DPI)
        print(f"Saved figure to {filepath_png}")
        fig.savefig(filepath_tiff, bbox_inches=bbox_inches, dpi=DPI)
        print(f"Saved figure to {filepath_tiff}")

def plot_n_goal_by_parameter(z_star = 1.96, filename=None):

    thetas = np.arange(0.01, 0.99, 0.01)
    goals = [0.1, 0.08, 0.06, 0.04]

    n_stop_goals = {goal: [binomial_rate_ci_width_to_sample_size(theta, goal, z_star=z_star)
                           for theta in thetas] for goal in goals}
    df_n_stop_goals = pd.DataFrame(n_stop_goals, index=thetas)

    fig = plt.figure(figsize=(1 * FIG_WIDTH, FIG_HEIGHT))

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
    save_figure(fig, filename)




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

# TODO: rename viz_epitg to viz_dpitg
def plot_multiple_decision_rates_separate(method_df_iteration_counts, success_rate, experiments, viz_epitg="separate", iteration_values=None):
    print("viz_epitg", viz_epitg)
    plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
    xlabel = "iteration"

    if success_rate is not None:
        suptitle = f"{theta_true_str} = {success_rate:0.3f}"
    else:
        suptitle = None

    for method_name, df_counts in method_df_iteration_counts.items():
        if iteration_values is None:
            iteration_values = df_counts["iteration"]

        linestyle_accept, linewidth_accept = None, 5
        linestyle_reject, linewidth_reject = "--", 3
        linestyle_inconclusive, linewidth_inconclusive = "-.", 1
        alpha=0.7
        label_accept = "accept"
        label_reject = "reject"
        label_inconclusive = "inconclusive/\ncollect more"
        
        if "hdi_rope" == method_name:
            if viz_epitg == "together":
                plt.subplot(1, 2, 1)
            elif viz_epitg == "separate":
                plt.subplot(1, 3, 1)
            else:
                plt.subplot(1, 2, 1)
            title = "HDI + ROPE"
        else:
            if viz_epitg == "together":
                plt.subplot(1, 2, 2)
                if "pitg" == method_name:
                    title = "Precision is the Goal (thin), Decisive (thick)"
                if "dpitg" == method_name:
                    linewidth_accept, linewidth_reject, linewidth_inconclusive = 6, 6, 6
                    alpha = 0.3
                    label_accept, label_reject, label_inconclusive = None, None, None
            elif viz_epitg == "separate":
                if "pitg" == method_name:
                    plt.subplot(1, 3, 2)
                    title = "Precision is the Goal"
                elif "dpitg" == method_name:
                    plt.subplot(1, 3, 3)
                    title = "Decisive Precision is the Goal"
            else:
                if "pitg" == method_name:
                    plt.subplot(1, 2, 1)
                    title = "Precision is the Goal"

        # plotting HDI+ROPE
        plt.plot(iteration_values, df_counts['accept'] / experiments, color="green", linewidth=linewidth_accept, alpha=alpha, linestyle=linestyle_accept, label=label_accept)
        plt.plot(iteration_values, df_counts['reject'] / experiments, color="red", linewidth=linewidth_reject, alpha=alpha, linestyle=linestyle_reject, label=label_reject)
        plt.plot(iteration_values, df_counts['inconclusive'] / experiments, color="gray", linewidth=linewidth_inconclusive, alpha=alpha, linestyle=linestyle_inconclusive, label=label_inconclusive)

        plot_grid(with_y=True, with_x=False, alpha=0.3)
        plt.legend(title="decision")
        plt.xlabel(xlabel)
        plt.ylabel(f"proportion of {experiments:,} experiments")
        plt.title(title)


    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()


def scatter_stop_iter_sample_rate(method_df_stats, rope_min=None, rope_max=None, 
                                          success_rate_true=None, success_rate_hypothesis=None, 
                                          precision_goal=None, title=None, method_names=None,
                                          scatter_ratio=3, bins=30, imbalance_cutoff_ratio=3.0):
    """
    Creates a 3-panel plot: 
    - Main scatter plot (Top-Right)
    - Success rate histogram (Left, sharing y-axis)
    - Stop iteration histogram (Bottom, sharing x-axis)
    
    imbalance_cutoff_ratio: if the max density of one distribution > X * peak of another,
                            limit the view to help see the smaller one.
    """

    method_markers = {"pitg": "o", "dpitg": "x", "hdi_rope": "s"}
    method_mean_markers = {"pitg": "$\u25EF$", "dpitg": "x", "hdi_rope": "$\u25A1$"}

    if method_names is None:
        method_names = ["hdi_rope", "pitg", "dpitg"]

    if success_rate_true:
        #theta_true_str = r"$\theta_{\rm true}$"
        title = f" {theta_true_str} = {success_rate_true:0.2f}"
    else:
        title = ""

    fig = plt.figure(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 1.5))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, scatter_ratio], height_ratios=[scatter_ratio, 1], 
                           wspace=0.05, hspace=0.05)

    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_scatter)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_scatter)

    last_df_len = 0
    
    iteration_max_densities = []
    success_max_densities = []

    for method_name in method_names:
        if method_name not in method_df_stats:
            continue
        
        df_stats = method_df_stats[method_name].copy()
        last_df_len = len(df_stats)
        
        color, marker = ALGO_COLORS[method_name], method_markers[method_name]
        mean_marker = method_mean_markers[method_name]
        label = method_pretty_short_name[method_name]
        label_mean = f"{method_pretty_short_name[method_name]} mean"

        # --- Main Panel: Scatter ---
        ax_scatter.scatter(df_stats["decision_iteration"], df_stats["success_rate"], 
                           alpha=0.3, color=color, label=label, marker=marker, s=20)
        
        # --- Bottom Panel: Iteration Histogram ---
        # Calculate histogram first to capture densities
        iter_counts, iter_bins = np.histogram(df_stats["decision_iteration"], bins=bins, density=True)
        iteration_max_densities.append(np.max(iter_counts))
        
        ax_bottom.hist(df_stats["decision_iteration"], bins=bins, color=color, alpha=0.3, 
                       density=True, histtype='stepfilled')
        ax_bottom.hist(df_stats["decision_iteration"], bins=bins, color=color, alpha=0.8, 
                       density=True, histtype='step', linewidth=1.5)

        # --- Left Panel: Success Rate Histogram ---
        success_counts, success_bins = np.histogram(df_stats["success_rate"], bins=bins, density=True)
        success_max_densities.append(np.max(success_counts))

        ax_left.hist(df_stats["success_rate"], bins=bins, color=color, alpha=0.3, 
                     density=True, histtype='stepfilled', orientation='horizontal')
        ax_left.hist(df_stats["success_rate"], bins=bins, color=color, alpha=0.8, 
                     density=True, histtype='step', orientation='horizontal', linewidth=1.5)
    
    # --- Auto-Scaling Logic ---
    def get_limit(max_densities, ratio):
        if not max_densities: return None
        sorted_max = sorted(max_densities)
        if len(sorted_max) > 1:
            # Check if largest is outlier compared to second largest
            if sorted_max[-1] > ratio * sorted_max[-2]:
                return sorted_max[-2] * 1.5 # Show the second largest comfortably
        return None # Default scaling

    ylim_bottom = get_limit(iteration_max_densities, imbalance_cutoff_ratio)
    xlim_left = get_limit(success_max_densities, imbalance_cutoff_ratio)

    if ylim_bottom:
        ax_bottom.set_ylim(ylim_bottom, 0) # Inverted
    else:
        ax_bottom.invert_yaxis()

    if xlim_left:
        ax_left.set_xlim(xlim_left, 0) # Inverted
    else:
        ax_left.invert_xaxis()


    # --- Decorate Main Scatter Panel ---
    if success_rate_true is not None:
        plot_vhlines_lines(vertical=None, label=f'{theta_true_str}', horizontal=success_rate_true, alpha=0.7, ax=ax_scatter)

    if rope_min is not None:
        plot_vhlines_lines(vertical=None, label='ROPE', horizontal=rope_min, linestyle="--", ax=ax_scatter)
    if rope_max is not None:
        plot_vhlines_lines(vertical=None, horizontal=rope_max, linestyle="--", ax=ax_scatter)

    if precision_goal is not None:
        n_true_str = r"$N_{\theta_\mathrm{true}}$"
        n_hypo_str = r"$N_{\theta_\mathrm{null}}$"
        n_precision_goal_true, n_precision_goal_hypothesis = None, None
        if (success_rate_true is not None):
            n_precision_goal_true = binomial_rate_ci_width_to_sample_size(success_rate_true, precision_goal)
        if (success_rate_hypothesis is not None):
            n_precision_goal_hypothesis = binomial_rate_ci_width_to_sample_size(success_rate_hypothesis, precision_goal) 

        if (n_precision_goal_true == n_precision_goal_hypothesis) and (n_precision_goal_true is not None):
            label_n = f"{n_true_str}={n_hypo_str}={n_precision_goal_true:0.1f}"
            ax_scatter.axvline(n_precision_goal_true, color='gray', linestyle=':', label=label_n)
        else:
            if n_precision_goal_true:
                label_n_true = f"{n_true_str}={n_precision_goal_true:0.1f}"
                ax_scatter.axvline(n_precision_goal_true, color='gray', linestyle=':', label=label_n_true)
            if n_precision_goal_hypothesis:
                laben_n_hypo = f"{n_hypo_str}={n_precision_goal_hypothesis:0.1f}"
                ax_scatter.axvline(n_precision_goal_hypothesis, color='gray', linestyle='--', label=laben_n_hypo)

    # --- Labels & Legends ---
    
    # Hide ticks between panels
    plt.setp(ax_scatter.get_xticklabels(), visible=False)
    plt.setp(ax_scatter.get_yticklabels(), visible=False)

    # Bottom panel labels
    ax_bottom.set_xlabel("stop iteration")
    ax_bottom.set_ylabel("density")

    # Invert to have 0 near the scatter plot
    ax_bottom.invert_yaxis() 
    ax_left.invert_xaxis()   

    # Hide 0 label on bottom panel y-axis to avoid clash with left panel x-axis
    from matplotlib.ticker import FuncFormatter
    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{x:g}"))

    # Left panel labels
    theta_hat_str = r"$\hat{\theta}$"
    ax_left.set_ylabel(f"success rate at stop {theta_hat_str}")
    ax_left.set_xlabel("density")

    # Legend on Scatter
    ax_scatter.legend(title=None, loc="upper right", fontsize=10)

    # if title is not None:
    #     # Adjust title position to not overlap with top-left empty space if needed
    #     # but suptitle usually handles it well.
    title += f" ({last_df_len:,} experiments)"
    plt.suptitle(title, y=0.95)

    return fig