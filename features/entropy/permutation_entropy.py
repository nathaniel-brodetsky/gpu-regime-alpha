import cudf
import cupy as cp
from cupy.lib.stride_tricks import as_strided
import math


def _build_sliding_windows(arr: cp.ndarray, window: int) -> cp.ndarray:
    n = arr.shape[0]
    n_windows = n - window + 1
    itemsize = arr.itemsize
    strided = as_strided(arr, shape=(n_windows, window), strides=(itemsize, itemsize))
    return strided


def _ordinal_pattern_rank(sub_windows: cp.ndarray, order: int) -> cp.ndarray:
    n_rows = sub_windows.shape[0]
    ranks = cp.argsort(sub_windows, axis=1)

    factorial_base = cp.array([math.factorial(order - 1 - i) for i in range(order)], dtype=cp.int64)

    pattern_id = cp.zeros(n_rows, dtype=cp.int64)
    for pos in range(order):
        remaining = ranks[:, pos].copy()
        for earlier in range(pos):
            remaining = cp.where(remaining > ranks[:, earlier], remaining - 1, remaining)
        pattern_id += remaining * factorial_base[pos]

    return pattern_id


def compute_permutation_entropy(series: cp.ndarray, order: int = 3, delay: int = 1) -> float:
    if delay > 1:
        series = series[::delay]

    sub_windows = _build_sliding_windows(series, order)
    pattern_ids = _ordinal_pattern_rank(sub_windows, order)

    n_patterns = math.factorial(order)
    counts = cp.bincount(pattern_ids, minlength=n_patterns).astype(cp.float64)
    probs = counts / cp.sum(counts)

    safe_probs = probs[probs > 0]
    entropy = -cp.sum(safe_probs * cp.log(safe_probs))

    max_entropy = math.log(n_patterns)
    normalized_entropy = float(entropy / max_entropy)

    return normalized_entropy


def rolling_permutation_entropy(series: cp.ndarray, window: int, order: int = 3, delay: int = 1) -> cp.ndarray:
    n = series.shape[0]
    n_windows = n - window + 1

    entropy_values = cp.full(n_windows, cp.nan, dtype=cp.float64)

    for w in range(n_windows):
        segment = series[w:w + window]
        entropy_values[w] = compute_permutation_entropy(segment, order, delay)

    return entropy_values