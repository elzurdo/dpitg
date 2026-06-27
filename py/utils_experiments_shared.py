import pandas as pd
import numpy as np



def stats_dict_to_df(method_stats, data_type='binomial'):
    """
    Convert method statistics dictionary to DataFrame.
    
    Generic function that handles both binomial and continuous data types.
    
    Parameters:
    -----------
    method_stats : dict
        Dictionary of experiment statistics
    data_type : str
        'binomial' or 'continuous'
    
    Returns:
    --------
    pd.DataFrame
        Statistics organized by experiment number with computed derivatives
        
    Notes:
    ------
    Shared columns (both types):
    - hdi_max, hdi_min, decision_iteration
    - reject (sum of reject_below and reject_above)
    - precision (HDI width)
    
    Binomial-specific columns:
    - success_rate: estimated parameter (successes / total)
    
    Continuous-specific columns:
    - sample_mean: estimated parameter
    - sample_std: sample standard deviation
    - n: sample size
    - se: standard error (sample_std / sqrt(n))
    - coefficient_of_variation: relative std (sample_std / |sample_mean|)
    - relative_precision: relative HDI width (precision / |sample_mean|)
    """
    df = pd.DataFrame(method_stats).T
    df.index.name = "experiment_number"
    df["hdi_max"] = df["hdi_max"].astype(float)
    df["hdi_min"] = df["hdi_min"].astype(float)
    df["decision_iteration"] = df["decision_iteration"].astype(float)
    df["reject"] = df["reject_below"] + df["reject_above"]
    df["precision"] = df["hdi_max"] - df["hdi_min"]
    
    if data_type == 'binomial':
        df["success_rate"] = df["successes"] / (df["successes"] + df["failures"])
    elif data_type == 'continuous':
        # Ensure proper types
        df["sample_mean"] = df["sample_mean"].astype(float)
        df["sample_std"] = df["sample_std"].astype(float)
        df["n"] = df["n"].astype(int)
        
        # Add continuous-specific derivatives
        df["se"] = df["sample_std"] / np.sqrt(df["n"])
        df["coefficient_of_variation"] = df["sample_std"] / df["sample_mean"].abs()
        df["relative_precision"] = df["precision"] / df["sample_mean"].abs()
    else:
        raise ValueError(f"Unknown data_type '{data_type}'. Choose 'binomial' or 'continuous'.")
    
    return df