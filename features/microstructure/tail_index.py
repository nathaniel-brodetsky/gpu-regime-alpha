import cudf
import cupy as cp
from cupy.lib.stride_tricks import as_strided

from features.microstructure.realized_vol import compute_log_returns


def _build_sliding_windows(arr: cp.ndarray, window: int) -> cp.ndarray:
    n = arr.shape[0]
    n_windows = n - window + 1
    itemsize = arr.itemsize
    strided = as_strided(arr, shape=(n_windows, window), strides=(itemsize, itemsize))
    return strided


def compute_rolling_hill_estimator(df: cudf.DataFrame, window: int, tail_fraction: float = 0.1, price_col: str = "mid_price") -> cudf.Series:
    log_returns_arr = compute_log_returns(df, price_col).to_cupy()
    abs_returns = cp.abs(log_returns_arr)

    windows = _build_sliding_windows(abs_returns, window)
    sorted_windows = cp.sort(windows, axis=1)[:, ::-1]

    k = max(int(window * tail_fraction), 2)
    tail_values = sorted_windows[:, :k]
    threshold = sorted_windows[:, k - 1:k]

    safe_threshold = cp.where(threshold == 0, 1e-12, threshold)
    safe_tail = cp.where(tail_values == 0, 1e-12, tail_values)

    log_ratios = cp.log(safe_tail / safe_threshold)
    hill_sum = log_ratios.sum(axis=1)
    hill_estimator = hill_sum / k

    tail_index = cp.where(hill_estimator == 0, cp.nan, 1.0 / hill_estimator)

    n = log_returns_arr.shape[0]
    padded = cp.full(n, cp.nan, dtype=cp.float64)
    padded[window - 1:] = tail_index
    return cudf.Series(padded)