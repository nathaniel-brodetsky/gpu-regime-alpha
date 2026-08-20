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
    if delay > 1:
        series = series[::delay]

    n = series.shape[0]
    n_windows = n - window + 1
    n_patterns_per_window = window - order + 1

    outer_windows = _build_sliding_windows(series, window)

    itemsize = series.itemsize
    inner_strides = (outer_windows.strides[0], outer_windows.strides[1], outer_windows.strides[1])
    ordinal_windows = as_strided(
        outer_windows,
        shape=(n_windows, n_patterns_per_window, order),
        strides=inner_strides,
    )

    flat_ordinal = ordinal_windows.reshape(-1, order)
    pattern_ids_flat = _ordinal_pattern_rank(flat_ordinal, order)
    pattern_ids = pattern_ids_flat.reshape(n_windows, n_patterns_per_window)

    n_pattern_types = math.factorial(order)

    one_hot = cp.zeros((n_windows, n_patterns_per_window, n_pattern_types), dtype=cp.float64)
    row_idx = cp.repeat(cp.arange(n_windows), n_patterns_per_window)
    col_idx = cp.tile(cp.arange(n_patterns_per_window), n_windows)
    pattern_flat = pattern_ids.flatten()
    one_hot[row_idx, col_idx, pattern_flat] = 1.0

    counts = cp.sum(one_hot, axis=1)
    probs = counts / n_patterns_per_window

    safe_probs = cp.where(probs > 0, probs, 1.0)
    log_probs = cp.where(probs > 0, cp.log(safe_probs), 0.0)
    entropy = -cp.sum(probs * log_probs, axis=1)

    max_entropy = math.log(n_pattern_types)
    normalized_entropy = entropy / max_entropy

    return normalized_entropy