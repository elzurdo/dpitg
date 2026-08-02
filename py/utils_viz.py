import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
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
rope_str = r"${\rm ROPE}$"
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

def save_figure(fig, filename=None, dpi=DPI, bbox_inches="tight"):
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




def viz_sequence_stats(df_sample_results, precision_goal, rope_min, rope_max, θ_true=None, filename=None):
    # Was called: viz_one_sample_results
    df_conclusive_accept = df_sample_results.query("conclusive").query("accept")
    df_conclusive_reject = df_sample_results.query("conclusive").query("reject")
    df_sample_goal = df_sample_results.query("goal_achieved")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

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

    save_figure(fig, filename)


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
    plt.legend(framealpha=1)

    if xtitle is not None:
        plt.xlabel(xtitle)
    plt.ylabel(p_theta_str)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([rope_min - 0.1, rope_max + 0.1])


def plot_sample_pdf_methods(method_df_stats, isample, rope_min, rope_max, xlim = (0.2, 0.6), method_names=None, filename=None):

    if method_names is None:
        method_names = list(method_df_stats.keys())

    ncols, nrows = 1, len(method_names)

    fig, ax = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, 1.2* FIG_HEIGHT))

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
    save_figure(fig, filename)

# TODO: rename viz_epitg to viz_dpitg
# TODO: rename success_rate to theta_true (or consider removing)
def plot_multiple_decision_rates_separate(method_df_iteration_counts, success_rate, experiments, viz_epitg="separate", iteration_values=None, filename=None, suptitle=None):
    fig = plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
    xlabel = "iteration"

    if suptitle is None:
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
    save_figure(fig, filename)


def scatter_stop_iter_sample_rate(method_df_stats, rope_min=None, rope_max=None, 
                                          success_rate_true=None, success_rate_hypothesis=None, 
                                          precision_goal=None, title=None, method_names=None,
                                          scatter_ratio=3, bins=30, imbalance_cutoff_ratio=3.0, filename=None, suptitle=None):
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

    if suptitle is None:
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
    
    if suptitle is None:
        title += f" ({last_df_len:,} experiments)"
        suptitle = title
    plt.suptitle(suptitle, y=0.95)
    save_figure(fig, filename)

    return fig


METHOD_SHORT = {
    "hdi_rope": "HDI+ROPE",
    "pitg": "PitG",
    "dpitg": "DPitG",

}

ALGO_HATCH = {
    "hdi_rope": None,
    "pitg": "/",
    "dpitg": "\\"
}

def plot_success_by_truth(algo_stats_df, dsuccess_rate, subset_name = "conclusive", param_null=0.5):

    assert subset_name in ["conclusive", "inconclusive", "overall"]

    if  "conclusive" == subset_name:
        title = "Conclusive"
    elif "overall" == subset_name:
        title = "Conclusive + Inconclusive"
    elif "inconclusive" == subset_name:
        title = "Inconclusive"

    truth_values = np.array(algo_stats_df[subset_name]["dpitg"]["param_median"].index.tolist())

    rope_min = param_null - dsuccess_rate
    rope_max = param_null + dsuccess_rate

    algo_alpha = {
       "hdi_rope": 0.2,
        "pitg": 0.5,
        "dpitg": 0.5     
    }

    plt.title(title, fontsize=20)
    for algo_name in METHOD_SHORT.keys():
        #this_truths = algo_stats_df[subset_name][algo_name].query("param_p25 == param_p25").index.tolist()
        this_truths = algo_stats_df[subset_name][algo_name].query("count >= 20").index.tolist()

        label = f"{METHOD_SHORT[algo_name]}"

        try:
            plt.fill_between(
                this_truths, 
                algo_stats_df[subset_name][algo_name].loc[this_truths, "param_p25"].astype(float),
                algo_stats_df[subset_name][algo_name].loc[this_truths,"param_p75"].astype(float),
                color=ALGO_COLORS[algo_name], 
                alpha=algo_alpha[algo_name], 
                label=label,
                hatch=ALGO_HATCH[algo_name]
            )
            
        except Exception as e:
            print(f"Error plotting {algo_name}: {e}")
            df_aux = algo_stats_df[subset_name][algo_name].loc[this_truths]
            try:
                df_aux["diff_pcnt"] = (df_aux["param_p75"] - df_aux["param_p25"]) * 100.
                display(df_aux[["count", "stop_iter_median" ,"param_p25", "param_median","param_p75","diff_pcnt" ]])
            except:
                pass

        plt.plot(this_truths, algo_stats_df[subset_name][algo_name].loc[this_truths, "param_p25"], color=ALGO_COLORS[algo_name], alpha=1.)
        plt.plot(this_truths, algo_stats_df[subset_name][algo_name].loc[this_truths, "param_p75"], color=ALGO_COLORS[algo_name], alpha=1.)

    plt.axhline(rope_min, linestyle=":", color="gray")
    plt.axhline(rope_max, linestyle=":", color="gray")

    plt.plot(truth_values, truth_values, color="gray", linestyle=None, alpha=1)
    plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
    if param_null > 0:
        plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)

    plt.xlabel(r"$\theta_{\rm true}$")
    plt.ylabel(r"$\hat{\theta}$")
    plt.legend(title="IQR")

    all_true_vals = algo_stats_df[subset_name][algo_name].index.tolist()
    dyy = 0.1


    plt.grid(alpha=0.3)
    #plt.ylim(0.4, 0.75)
    plt.ylim(all_true_vals[0] - dyy, all_true_vals[-1] + dyy)


def plot_success_by_truth_diff(algo_stats_df, dsuccess_rate, subset_name="conclusive", success_metrics=["param_median"], param_null=0.5):
    METRIC_LINESTYLE = {"param_median": None, "param_mean": "--"}

    truth_values = np.array(algo_stats_df[subset_name]["dpitg"]["param_median"].index.tolist())

    for success_metric in success_metrics:
        for algo_name in METHOD_SHORT.keys():
            result_diff = algo_stats_df[subset_name][algo_name][success_metric] - truth_values

            label = f"{METHOD_SHORT[algo_name]}"
            plt.plot(truth_values, result_diff, color=ALGO_COLORS[algo_name], linestyle=METRIC_LINESTYLE[success_metric], alpha=0.7, label=label, linewidth=ALGO_LINEWIDTH[algo_name])


    plt.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    plt.xlabel(r"$\theta_{\rm true}$")
    plt.ylabel(r"$\hat{\theta} - \theta_{\rm true}$")
    plt.legend(title="median - true", framealpha=1)
    plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
    if param_null > 0.5:
        plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)

    plt.axhline(-dsuccess_rate, linestyle=":", color="gray")
    plt.axhline(dsuccess_rate, linestyle=":", color="gray")
    #plt.ylim(-dsuccess_rate,dsuccess_rate)
    plt.ylim(-0.1, 0.1)

    plt.grid(alpha=0.3)

def plot_success_by_truth_absolute_and_diff(algo_stats_df, dsuccess_rate, subset_name="conclusive",
                                            param_null=0.5, ncols = 2, nrows = 1, xlim=(0.498, 0.802), filename=None):
    plt.figure(figsize=(ncols * FIG_WIDTH, nrows * FIG_HEIGHT))

    # === Absolute Conclusive===
    plt.subplot(nrows, ncols, 1)
    plot_success_by_truth(algo_stats_df, dsuccess_rate, subset_name = subset_name, param_null=param_null)
    plt.title(r"$\hat{\theta}(\theta_{\rm true})$")
    plt.xlim(xlim)

    # === Relative Conclusive===
    plt.subplot(nrows, ncols, 2)
    plot_success_by_truth_diff(algo_stats_df, dsuccess_rate, subset_name=subset_name, success_metrics=["param_median"], param_null=param_null)
    plt.title(r"$\hat{\theta}(\theta_{\rm true}) - \theta_{\rm true}$")
    plt.xlim(xlim)

    plt.suptitle(f"Bias Stats of {subset_name.capitalize()} Experiments", fontsize=20)

    plt.tight_layout()

    save_figure(plt.gcf(), filename)


def plot_stop_and_conclusive_ratios(algo_stats_df, subset_name = "overall", param_null=0.5,
                                    dsuccess_rate=0.1, viz_mean=False, goal_val=0.08, xlim=(0.498, 0.702),
                                    denominator_type="goal", filename=None):
    
    
    dpitg_ = algo_stats_df[subset_name]["dpitg"]
    pitg_ = algo_stats_df[subset_name]["pitg"]
    theta_trues = dpitg_["stop_iter_median"].index.tolist()
    n_goals = np.array([binomial_rate_ci_width_to_sample_size(true_rate, goal_val) for true_rate in theta_trues])
    

    # stop ratio
    if denominator_type == "pitg":
        stop_ratio = dpitg_["stop_iter_median"] / pitg_["stop_iter_median"]
        stop_ratio_mean = dpitg_["stop_iter_mean"] / pitg_["stop_iter_mean"]
        stop_ratio_p25 = dpitg_["stop_iter_p25"] / pitg_["stop_iter_p25"]
        stop_ratio_p75 = dpitg_["stop_iter_p75"] / pitg_["stop_iter_p75"]
    elif denominator_type == "goal":
        print("Using goal as denominator")
        stop_ratio = dpitg_["stop_iter_median"] / n_goals
        stop_ratio_mean = dpitg_["stop_iter_mean"] / n_goals
        stop_ratio_p25 = dpitg_["stop_iter_p25"] / n_goals
        stop_ratio_p75 = dpitg_["stop_iter_p75"] / n_goals


    # conclusive ratio
    conclusive_ratio = algo_stats_df["overall"]["dpitg"]["conclusive_mean"] / algo_stats_df["overall"]["pitg"]["conclusive_mean"]

    n_stop_str = r"$N_{\rm goal}$" #(\theta_{\rm true},\omega)$"
    n_stop_epitg_str = r"$N_{\rm DPitG}$"
    n_stop_pitg_str = r"$N_{\rm PitG}$"
    ratio_str = f"{n_stop_epitg_str}/{n_stop_pitg_str}"
    plt.plot(stop_ratio, color="purple", linewidth=1, linestyle="-.", label=f"{ratio_str} Median")
    plt.fill_between(dpitg_.index, stop_ratio_p25, stop_ratio_p75, color="purple", alpha=0.2, label=f"{ratio_str} IQR")
    if viz_mean:
        plt.plot(stop_ratio_mean, color="gray", linewidth=0.5, label=f"{ratio_str} Mean")
    plt.plot(conclusive_ratio, color="purple", linewidth=2, label="Conclusiveness DPitG/PitG")
    plt.ylim(0,None)
    plt.grid(alpha=0.3)
    plt.xlabel(r"$\theta_{\rm true}$")
    plt.ylabel("Ratio")
    plt.xlim(xlim)

    # TODO: generalise plotting boundaries for binary vs. continuous
    plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5, label=r"ROPE$_{\rm max}$")
    if param_null > 0.5:
        plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5, label=r"ROPE$_{\rm min}$")
    plt.legend(framealpha=1)
    #title = f"{theta_null_str}={param_null}, {delta_rope_str}={2 * dsuccess_rate}, {ω_goal_str}={goal_val:0.2f}"
    title = f"{theta_null_str}={param_null}, ROPE=[{param_null - dsuccess_rate}, {param_null + dsuccess_rate}], {ω_goal_str}={goal_val:0.2f}"
    plt.title(title, fontsize=20)

    save_figure(plt.gcf(), filename)


def plot_conclusiveness_decisions_and_correctness_rates(algo_stats_df, df_correctness_rates,
                                                        dsuccess_rate, method_names=None,
                                                        n_experiments=None, subset_name = "overall",
                                                        param_null=0.5, xlim=(0.498, 0.702), ylim=(0,1), filename=None):
    ylabel = f"Fraction of all {n_experiments:,} Experiments" if n_experiments is not None else "Fraction of all Experiments"

    if method_names is None:
        method_names = list(METHOD_SHORT.keys())

    plt.figure(figsize=(3 * FIG_WIDTH, FIG_HEIGHT))

    # Conclusive Rates
    plt.subplot(1, 3, 1)
    for algo_name in METHOD_SHORT:
        plt.plot(algo_stats_df[subset_name][algo_name]["conclusive_mean"],
        color=ALGO_COLORS[algo_name], label=METHOD_SHORT[algo_name], linewidth=ALGO_LINEWIDTH[algo_name])

    plt.grid(alpha=0.3)
    plt.xlabel(r"$\theta_{\rm true}$")
    plt.legend()
    plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
    if param_null > 0.5:
        plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)

    plt.title("Conclusive Rates = Acceptence + Rejection")
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Acceptence + Rejection Rates
    plt.subplot(1, 3, 2)
    for algo_name in METHOD_SHORT:
        plt.plot(algo_stats_df[subset_name][algo_name]["accept_mean"],
        color=ALGO_COLORS[algo_name], label=METHOD_SHORT[algo_name], linewidth=ALGO_LINEWIDTH[algo_name])

        plt.plot(algo_stats_df[subset_name][algo_name]["reject_mean"],
            color=ALGO_COLORS[algo_name], linewidth=ALGO_LINEWIDTH[algo_name], linestyle="-.")

    plt.grid(alpha=0.3)
    plt.xlabel(r"$\theta_{\rm true}$")
    plt.ylabel(ylabel)
    plt.title("Acceptence (solid), Rejection (dashed)")
    plt.legend()
    plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
    if param_null > 0.5:
        plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)

    plt.xlim(xlim)
    plt.ylim(ylim)

    # Correctness Rates
    plt.subplot(1, 3, 3)
    for algo_name in method_names:
        plt.plot(df_correctness_rates[f"{algo_name}_decision_correct"], color=ALGO_COLORS[algo_name], label=f"{METHOD_SHORT[algo_name]}", linewidth=ALGO_LINEWIDTH[algo_name])

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.xlabel(r"$\theta_{\rm true}$")
    plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
    if param_null > 0.5:
        plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)
    plt.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    plt.title("Accuracy")
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()

    save_figure(plt.gcf(), filename)

def plot_stop_iterations_by_truth(algo_stats_df, dsuccess_rate, subset_name = "overall", param_null=0.5, ylim=(0,1500), xlim=(0.498, 0.652), precision_goal=0.08):

    algos_viz = list( METHOD_SHORT.keys())
    nrows = len(algos_viz)
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))


    for iplot, algo_name in enumerate(algos_viz): #  ["pitg", "epitg"]:
        plt.subplot(nrows, 1, iplot + 1)

        df_plot = algo_stats_df[subset_name][algo_name].query("count >= 20")

        plt.plot(df_plot["param_mean"],
                 df_plot["stop_iter_mean"],
                 color=ALGO_COLORS[algo_name], label="mean",
                 linewidth=ALGO_LINEWIDTH[algo_name])
        
        plt.fill_between(df_plot.index.tolist(),
                         df_plot["stop_iter_p25"].astype(float),
                         df_plot["stop_iter_p75"].astype(float),
                         color=ALGO_COLORS[algo_name], alpha=0.2, label="IQR")

        if iplot == 0:
            label_ntruths = f"{n_true_str}({theta_true_str}|{ω_goal_str}={precision_goal:0.2f})"
        else:
            label_ntruths = None
        n_truths = [binomial_rate_ci_width_to_sample_size(true_rate, precision_goal) for true_rate in df_plot.index.tolist()]
        plt.plot(df_plot.index.tolist(), n_truths, color="orange", linestyle=":", label=label_ntruths)

        plt.subplot(nrows, 1, iplot + 1)
        plt.axvline(x=param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
        if param_null > 0.5:
            plt.axvline(x=param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)

        if iplot == len(algos_viz) - 1:
            plt.xlabel(theta_true_str)
        
        plt.ylabel(n_stop_str)
        plt.legend(title=METHOD_SHORT[algo_name], loc="upper right", fontsize=10, bbox_to_anchor=(1.3, 1))
        plt.grid(alpha=0.3)
        plt.ylim(ylim)
        plt.xlim(xlim)

    plt.suptitle(f"Stop Iteration {n_stop_str}({theta_true_str}|{theta_null_str}={param_null},{ω_goal_str}={precision_goal:0.2f}) of {subset_name.capitalize() if subset_name is not 'overall' else 'All'} Experiments")
    plt.tight_layout()


def plot_stop_iterations_by_truth_two_panel(
        algo_stats_df_1, algo_stats_df_2,
        dsuccess_rate_1, dsuccess_rate_2,
        param_null_1=0.5, param_null_2=0.5,
        precision_goal_1=0.08, precision_goal_2=0.08,
        xlim_1=(0.498, 0.652), xlim_2=(0.498, 0.652),
        ylim=(0, 1500),
        subset_name="overall",
        average_name="mean",
        filename=None
        ):

    algos_viz = list(METHOD_SHORT.keys())
    nrows = len(algos_viz)
    ncols = 2
    plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))

    panels = [
        (algo_stats_df_1, dsuccess_rate_1, param_null_1, precision_goal_1, xlim_1),
        (algo_stats_df_2, dsuccess_rate_2, param_null_2, precision_goal_2, xlim_2),
    ]

    subset_label = "All" if subset_name == "overall" else subset_name.capitalize()

    for icol, (algo_stats_df, dsuccess_rate, param_null, precision_goal, xlim) in enumerate(panels):
        for iplot, algo_name in enumerate(algos_viz):
            subplot_idx = iplot * ncols + icol + 1
            plt.subplot(nrows, ncols, subplot_idx)

            df_plot = algo_stats_df[subset_name][algo_name].query("count >= 20")
            
            df_plot_low = algo_stats_df_1[subset_name][algo_name].query("count < 20")

            if df_plot_low.shape[0] > 0:
                print(f"Warning: {subset_name} rows with count < 20 for {METHOD_SHORT[algo_name]} in {subset_label} subset. These will be excluded from the plot.")
                display(df_plot_low.shape)

            theta_true_values = df_plot.index.tolist()
            
            plt.plot(theta_true_values,
                     df_plot[f"stop_iter_{average_name}"],
                     color=ALGO_COLORS[algo_name], label=average_name,
                     linewidth=ALGO_LINEWIDTH[algo_name])

            plt.fill_between(theta_true_values,
                             df_plot["stop_iter_p25"].astype(float),
                             df_plot["stop_iter_p75"].astype(float),
                             color=ALGO_COLORS[algo_name], alpha=0.2, label="IQR")

            if iplot == 0:
                label_ntruths = f"{n_true_str}({theta_true_str}|{ω_goal_str}={precision_goal:0.2f})"
            else:
                label_ntruths = None
            if icol == 0:
                this_param_null = float(param_null_1)
            else:
                this_param_null = float(param_null_2)
            n_null = binomial_rate_ci_width_to_sample_size(this_param_null, precision_goal)
            n_truths = [binomial_rate_ci_width_to_sample_size(true_rate, precision_goal) for true_rate in df_plot.index.tolist()]
            plt.plot(theta_true_values, n_truths, color="orange", linestyle=":", label=label_ntruths)

            plt.axhline(y=n_null, color="black", linestyle="--", alpha=0.2)
            n_null_str = r"$N_{\rm goal}(\theta_{\rm null})$"
            plt.annotate(f"{n_null_str}={n_null:0.0f}", xy=(xlim[-1] - 0.15, n_null+ 20), color="black", alpha=0.6, fontsize=10)
            plt.axvline(x=this_param_null + dsuccess_rate, color="black", linestyle="--", alpha=0.5)
            if this_param_null > 0.5:
                plt.axvline(x=this_param_null - dsuccess_rate, color="black", linestyle="--", alpha=0.5)

            if iplot == len(algos_viz) - 1:
                plt.xlabel(theta_true_str)

            plt.ylabel(n_stop_str)
            plt.legend(title=METHOD_SHORT[algo_name], loc="upper right", fontsize=10, bbox_to_anchor=(1.3, 1), framealpha=1)
            plt.grid(alpha=0.3)
            plt.ylim(ylim)
            plt.xlim(xlim)

            if iplot == 0:
                plt.title(f"{n_stop_str}({theta_true_str}|{theta_null_str}={param_null},{ω_goal_str}={precision_goal:0.2f})")

    plt.suptitle("Stop Iteration of Conclusive Experiments", fontsize=20)
    plt.tight_layout()
    save_figure(plt.gcf(), filename)

def plot_decision_rates_nhst(n_experiments, iteration_stopping_on_or_prior, fpr=None, success_rate_null=0.5, success_rate_true=0.5, filename=None):
    msize = 5
    fpr_str = r"$\alpha_{\rm FPR}$"
    xlabel = "iteration"
    ylabel = f"proportion of {n_experiments:,} experiments"
    title = f"{theta_true_str} = {success_rate_true:0.2f}, {theta_null_str} = {success_rate_null:0.2f}, {fpr_str}={fpr:0.2f}" if fpr is not None else f"{theta_true_str} = {success_rate_true:0.2f}, {theta_null_str} = {success_rate_null:0.2f}"
    # theta_null_str = r"$\theta_{\rm null}$"

    sr_iteration_stopping_on_or_prior = pd.Series(iteration_stopping_on_or_prior)
    sr_nhst_reject = sr_iteration_stopping_on_or_prior / n_experiments

    plt.plot(sr_nhst_reject.index, sr_nhst_reject + 0.01, alpha=0.7, color="red", linewidth=3, label=f"reject {theta_null_str}")
    plt.plot(sr_nhst_reject.index, 1. - sr_nhst_reject, alpha=0.7, color="gray", linewidth=3, linestyle='--', label="not reject / inconclusive")

    plt.legend()
    plt.xscale('log')

    
    if fpr is not None:
        plt.axhline(y=fpr, color="gray", linestyle='-.', alpha=0.3, label=f"{fpr_str}={fpr:0.2f}")
        last_ = list(iteration_stopping_on_or_prior.keys())[-1]
        x_annotate = last_ * 0.1 if last_ > 10 else last_ * 0.5
        plt.annotate(f"{fpr_str}={fpr:0.2f}", xy=(x_annotate, fpr + 0.02), color="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3, axis='y')
    save_figure(plt.gcf(), filename)