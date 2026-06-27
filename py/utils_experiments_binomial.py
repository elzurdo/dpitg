import numpy as np

from utils_stats import (
    successes_failures_to_hdi_ci_limits
)

from utils_experiments_shared import (
    stats_dict_to_df,
    # iteration_counts_to_df,
    # report_success_rates,
    # report_success_rates_multiple_algos,
    # create_decision_correctness_df,
)

SEQUENCE_HANDPICKED = "101101000110010000101111111110010101101110001111110010100110111111110111001111001110011110001010001011110101111110001111111111100000101001001100000001101000100010000000010010111001110100111000010010110011010000101011110011111111011100101011011100100101010011110101001111011100101110010011001010010001001011010101010100111100110011011011101110010100010110011001100101111001111101110101010001101110111100010110101010101010111100001000111011001010101100100110010001101101111100111000010011001000001010110010101101000001100101000110101110010101101000100110100100100110110100101011100001101000111111001001111100100011100011000101001010101110010000110111101111011100111011010010001001001111011100100000100011100000010010111111011110101000110110010001100101011110000001001101111100000001010011001001110001010100000101111100101110011011010111001000011110010011111110011111111100111011010000101110110001100111001000010011101100111000110010100000001101110000110011100111011100101001101010011001010100011000000011001100101100101000001101100111000000101010000110100100111110101101110010000100011101011011001110011100111011101010100101100001101100010111010010101000011000100111111010010111001100001001000110111011001011100100001001011111010011111101111001010000110011010101111001011110100001000100000010000011001110100110100100101000001100110111011011111010100111101111101010001010110010001000110111000101000010001011000100001101111011000000111010011000101001011110111101111010011101010111001111010101111011000110"

def successes_failures_caculate_hdi_limits(successes, failures):
    aa = int(successes)
    bb = int(failures)
    
    if not failures:
        aa += 1
        bb += 1
        
    if not successes:
        aa += 1
        bb += 1

    hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(aa, bb)

    return hdi_min, hdi_max

class BinaryAccounting():
    def __init__(self):
        self.dict_successes_failures_counter = {}
        self.dict_successes_failures_hdi_limits = {}
    
    def successes_failures_to_hdi_limits(self, successes, failures):
        pair = (successes, failures)
        if pair not in self.dict_successes_failures_hdi_limits:
            self.dict_successes_failures_hdi_limits[pair] =\
                successes_failures_caculate_hdi_limits(successes, failures)
            self.dict_successes_failures_counter[pair] = 1
        else:
            self.dict_successes_failures_counter[pair] += 1

        return self.dict_successes_failures_hdi_limits[pair]


def sequence_to_iteration_stats(samples, ω_goal, rope_min, rope_max, iteration_number=None, binary_accounting=None):
    # By all iterations it means that it doesn't stop, but does flag
    # when objectives are met: conclusiveness, percision goal

    # Previous function name: sample_all_iterations_results
    if iteration_number is None:
        iteration_number = np.arange(1, samples.shape[0] + 1)

    iteration_successes = samples.cumsum()
    iteration_failures = iteration_number - iteration_successes

    samples_results = {}
    for iteration, successes, failures in zip(iteration_number, iteration_successes, iteration_failures):
        # final_iteration = iteration == iteration_number[-1]
        
        # TODO: turn this part into a function (if works out well with other part)
        if binary_accounting is not None:
            hdi_min, hdi_max = binary_accounting.successes_failures_to_hdi_limits(successes, failures)
        else:
            hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(successes, failures)
        # has the precision goal been achieved?
        precision_goal_achieved = (hdi_max - hdi_min) < ω_goal

        # is the HDI conclusively within or outside the ROPE?
        decision_accept = (hdi_min >= rope_min) & (hdi_max <= rope_max)
        decision_reject_below = hdi_max < rope_min  
        decision_reject_above = rope_max < hdi_min
        conclusive = decision_accept | decision_reject_above | decision_reject_below


        iteration_results = {"decision_iteration": iteration,
                                                    "accept": decision_accept,
                                                    "reject_below": decision_reject_below,
                                                    "reject_above": decision_reject_above,
                                                    "conclusive": conclusive,
                                                    "inconclusive": not conclusive,
                                                    "successes": successes,
                                                    "failures": failures,
                                                    "hdi_min": hdi_min,
                                                    "hdi_max": hdi_max,
                                                    "goal_achieved": precision_goal_achieved,
                                                    }  
        # END TODO 

        samples_results[iteration] = iteration_results

    df_sample_results = stats_dict_to_df(samples_results)    

    return df_sample_results


def _update_iteration_tally(iteration_dict, iteration):
    for this_iteration in range(iteration, len(iteration_dict)+1):
        iteration_dict[this_iteration] += 1

def booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above):
    if decision_accept:
        return "within"
    elif decision_reject_below:
        return "below"
    elif decision_reject_above:
        return "above"

def stop_decision_multiple_experiments_multiple_methods(samples, rope_min, rope_max, precision_goal, binary_accounting=None, min_iter=30, viz=False):
    # For each method and rope result type creating tally of outcomes
    method_names = ["pitg", "epitg", "hdi_rope"]
    n_samples = samples.shape[1]

    # stats at sample stop iteration
    method_stats = {method_name: {} for method_name in method_names}

    # for each iteration the number of sample that stopped there or before
    # and the decision outcome
    method_roperesult_iteration = {}
    
    rope_results = ["within", "below", "above"]
    for method in method_names:
        method_roperesult_iteration[method] = {}
        for rope_result in rope_results:
            method_roperesult_iteration[method][rope_result] = {iteration: 0 for iteration in range(1, n_samples + 1)}


    iteration_number = np.arange(1, n_samples + 1)

    for isample, sample in enumerate(samples):
        pitg_stopped = False
        hdi_rope_stopped = False

        # the number of successes at each iteration for this sample
        iteration_successes = sample.cumsum()
        iteration_failures = iteration_number - iteration_successes

        # examining the specifics of each iteration
        for iteration, successes, failures in zip(iteration_number, iteration_successes, iteration_failures):
            final_iteration = iteration == iteration_number[-1]

            if binary_accounting is not None:
                hdi_min, hdi_max = binary_accounting.successes_failures_to_hdi_limits(successes, failures)
            else:
                hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(successes, failures)

            # has the precision goal been achieved?
            precision_goal_achieved = (hdi_max - hdi_min) < precision_goal

            # is the HDI conclusively within or outside the ROPE?
            decision_accept = (hdi_min >= rope_min) & (hdi_max <= rope_max)
            decision_reject_below = hdi_max < rope_min  
            decision_reject_above = rope_max < hdi_min
            conclusive = decision_accept | decision_reject_above | decision_reject_below

            if min_iter is not None:
                if iteration < min_iter:
                    #continue
                    decision_accept = False
                    decision_reject_below = False
                    decision_reject_above = False
                    conclusive = False
                    precision_goal_achieved = False


            iteration_results = {"decision_iteration": iteration,
                                                    "accept": decision_accept,
                                                        "reject_below": decision_reject_below,
                                                        "reject_above": decision_reject_above,
                                                        "conclusive": conclusive,
                                                        "inconclusive": not conclusive,
                                                        "successes": successes,
                                                        "failures": failures,
                                                        "hdi_min": hdi_min,
                                                        "hdi_max": hdi_max,
                                                        "precision_goal_achieved": precision_goal_achieved,
                                                    }   

            if precision_goal_achieved:

                # update Precision Is The Goal Stop
                if pitg_stopped is False:
                    # not applying `break` because we continue for ePiTG
                    if conclusive:
                        rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                        _update_iteration_tally(method_roperesult_iteration["pitg"][rope_result], iteration)
                    method_stats["pitg"][isample] = iteration_results
                    pitg_stopped = True  # sample does not continue with PITG (only ePiTG) 

                # continue with Enhance Precision Is The Goal
                if conclusive:
                    rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                    _update_iteration_tally(method_roperesult_iteration["epitg"][rope_result], iteration)

                    if hdi_rope_stopped is False:
                        _update_iteration_tally(method_roperesult_iteration["hdi_rope"][rope_result], iteration)
                        
                if conclusive | final_iteration:
                    method_stats["epitg"][isample] = iteration_results

                    if hdi_rope_stopped is False:
                        method_stats["hdi_rope"][isample] = iteration_results
                        hdi_rope_stopped = True

                    if final_iteration:
                        if viz:
                            print(f"Sample {isample} at final iteration")
                            print(method_stats["epitg"][isample])
                    break
            
            elif conclusive & (hdi_rope_stopped is False):
                # case in which precision not achieved yet but conclusive.
                # this is the HDI+ROPE approach which disregards precision
                rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                _update_iteration_tally(method_roperesult_iteration["hdi_rope"][rope_result], iteration)
                method_stats["hdi_rope"][isample] = iteration_results

                hdi_rope_stopped = True

            elif final_iteration:
                # ensures that if reached final iteration and not conclusive that
                # results are still recorded as inconclusive
                if isample not in method_stats["hdi_rope"]:
                    method_stats["hdi_rope"][isample] = iteration_results
                if isample not in method_stats["pitg"]:
                    method_stats["pitg"][isample] = iteration_results
                if isample not in method_stats["epitg"]:
                    method_stats["epitg"][isample] = iteration_results
                break

    return method_stats, method_roperesult_iteration