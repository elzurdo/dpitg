import sys, platform, datetime, importlib.metadata as md
from collections import Counter
import numpy as np

def print_env_info():
    print(f"Last run: {datetime.datetime.now()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for pkg in ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "notebook"]:
        try:
            print(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            pass



def deep_sizeof(obj, seen=None):
    """Recursively calculate total memory size of an object."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid double-counting shared references
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(item, seen) for item in obj)

    return size

def report_accounting_size(binary_accounting):
    #size_bytes = sys.getsizeof(binary_accounting) 
    size_bytes = deep_sizeof(binary_accounting.dict_successes_failures_counter)
    size_mbytes = size_bytes / (1024 * 1024)
    
    n_dict = len(binary_accounting.dict_successes_failures_counter)
    print("Binary Accounting Report")
    print(f"{n_dict:,} elements in {size_mbytes:.2f} MBytes)")
    
    #counts = Counter(binary_accounting.dict_successes_failures_counter.values())
    sums = Counter(k[0] + k[1] for k in binary_accounting.dict_successes_failures_counter.keys())
    if len(sums) > 0:
        most_frequent_value, frequency = sums.most_common(1)[0]
        frac_ = frequency / n_dict
        print(f"Most frequent count value: {most_frequent_value:,} (appears {frequency:,} times (of {n_dict:,} {frac_:.2%}))")

        key_max_sum = max(binary_accounting.dict_successes_failures_counter.keys(), key=lambda t: t[0] + t[1])
        print(f"Max sum {np.sum(key_max_sum):,}")
        print(f"e.g, key {key_max_sum} appears {binary_accounting.dict_successes_failures_counter[key_max_sum]:,} time(s) yiedling {binary_accounting.dict_successes_failures_hdi_limits[key_max_sum]} HDI limits")
        print()
    else:
        print(f"The binary counter is empty")