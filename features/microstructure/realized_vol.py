import cudf
import cupy as cp
from cupy.lib.stride_tricks import as_strided


def compute_log_returns(df: cudf.DataFrame, price_col: str = "mid_price") -> cudf.Series:
    log_price = cudf.Series(cp.log(df[price_col].to_cupy()))
    return log_price.diff().fillna(0)


def compute_realized_variance(df: cudf.DataFrame, window: int, price_col: str = "mid_price") -> cudf.Series:
    log_returns = compute_log_returns(df, price_col)
    squared_returns = log_returns ** 2
    return squared_returns.rolling(window=window, min_periods=1).sum()


def compute_realized_volatility(df: cudf.DataFrame, window: int, price_col: str = "mid_price") -> cudf.Series:
    realized_variance = compute_realized_variance(df, window, price_col)
    return cudf.Series(cp.sqrt(realized_variance.to_cupy()))


def _build_sliding_windows(arr: cp.ndarray, window: int) -> cp.ndarray:
    n = arr.shape[0]
    n_windows = n - window + 1
    itemsize = arr.itemsize
    strided = as_strided(arr, shape=(n_windows, window), strides=(itemsize, itemsize))
    return strided


def _rolling_moment_stats(arr: cp.ndarray, window: int):
    windows = _build_sliding_windows(arr, window)
    mean = windows.mean(axis=1)
    diff = windows - mean[:, None]
    variance = (diff ** 2).mean(axis=1)
    std = cp.sqrt(variance)
    return mean, std, diff


def compute_rolling_return_skew(df: cudf.DataFrame, window: int, price_col: str = "mid_price") -> cudf.Series:
    log_returns_arr = compute_log_returns(df, price_col).to_cupy()
    mean, std, diff = _rolling_moment_stats(log_returns_arr, window)

    safe_std = cp.where(std == 0, 1.0, std)
    z = diff / safe_std[:, None]
    skew = (z ** 3).mean(axis=1)
    skew = cp.where(std == 0, 0.0, skew)

    n = log_returns_arr.shape[0]
    padded = cp.full(n, cp.nan, dtype=cp.float64)
    padded[window - 1:] = skew
    return cudf.Series(padded)


def compute_rolling_return_kurtosis(df: cudf.DataFrame, window: int, price_col: str = "mid_price") -> cudf.Series:
    log_returns_arr = compute_log_returns(df, price_col).to_cupy()
    mean, std, diff = _rolling_moment_stats(log_returns_arr, window)

    safe_std = cp.where(std == 0, 1.0, std)
    z = diff / safe_std[:, None]
    kurt = (z ** 4).mean(axis=1) - 3.0
    kurt = cp.where(std == 0, 0.0, kurt)

    n = log_returns_arr.shape[0]
    padded = cp.full(n, cp.nan, dtype=cp.float64)
    padded[window - 1:] = kurt
    return cudf.Series(padded)