import cupy as cp
import math
from cupy.lib.stride_tricks import as_strided


def _build_sliding_windows(arr: cp.ndarray, window: int) -> cp.ndarray:
    n = arr.shape[0]
    n_windows = n - window + 1
    itemsize = arr.itemsize
    strided = as_strided(arr, shape=(n_windows, window), strides=(itemsize, itemsize))
    return strided


def _detrended_fluctuation(cumulative_series: cp.ndarray, box_size: int) -> float:
    n = cumulative_series.shape[0]
    n_boxes = n // box_size

    if n_boxes < 1:
        return cp.nan

    trimmed = cumulative_series[:n_boxes * box_size]
    boxes = trimmed.reshape(n_boxes, box_size)

    x = cp.arange(box_size, dtype=cp.float64)
    x_mean = cp.mean(x)
    x_centered = x - x_mean
    x_var = cp.sum(x_centered ** 2)

    y_mean = cp.mean(boxes, axis=1, keepdims=True)
    y_centered = boxes - y_mean

    slope = cp.sum(y_centered * x_centered[None, :], axis=1) / x_var
    intercept = y_mean.flatten() - slope * x_mean

    trend = slope[:, None] * x[None, :] + intercept[:, None]
    residuals = boxes - trend

    fluctuation = cp.sqrt(cp.mean(residuals ** 2, axis=1))
    return float(cp.mean(fluctuation))


def compute_hurst_exponent(series: cp.ndarray, min_box_size: int = 10, max_box_size: int = None, n_box_sizes: int = 15) -> float:
    n = series.shape[0]
    if max_box_size is None:
        max_box_size = n // 4

    if max_box_size <= min_box_size:
        return cp.nan

    demeaned = series - cp.mean(series)
    cumulative = cp.cumsum(demeaned)

    box_sizes = cp.unique(cp.logspace(cp.log10(min_box_size), cp.log10(max_box_size), n_box_sizes).astype(cp.int64))

    log_box_sizes = []
    log_fluctuations = []

    for box_size in box_sizes.tolist():
        fluctuation = _detrended_fluctuation(cumulative, box_size)
        if fluctuation is not None and not cp.isnan(fluctuation) and fluctuation > 0:
            log_box_sizes.append(math.log(box_size))
            log_fluctuations.append(float(cp.log(fluctuation)))

    if len(log_box_sizes) < 2:
        return cp.nan

    log_box_sizes_arr = cp.array(log_box_sizes, dtype=cp.float64)
    log_fluctuations_arr = cp.array(log_fluctuations, dtype=cp.float64)

    x_mean = cp.mean(log_box_sizes_arr)
    y_mean = cp.mean(log_fluctuations_arr)

    numerator = cp.sum((log_box_sizes_arr - x_mean) * (log_fluctuations_arr - y_mean))
    denominator = cp.sum((log_box_sizes_arr - x_mean) ** 2)

    hurst = float(numerator / denominator)
    return hurst


def _batched_detrended_fluctuation(cumulative_windows: cp.ndarray, box_size: int) -> cp.ndarray:
    n_windows, window_length = cumulative_windows.shape
    n_boxes = window_length // box_size

    if n_boxes < 1:
        return cp.full(n_windows, cp.nan, dtype=cp.float64)

    trimmed = cumulative_windows[:, :n_boxes * box_size]
    boxes = trimmed.reshape(n_windows, n_boxes, box_size)

    x = cp.arange(box_size, dtype=cp.float64)
    x_mean = cp.mean(x)
    x_centered = x - x_mean
    x_var = cp.sum(x_centered ** 2)

    y_mean = cp.mean(boxes, axis=2, keepdims=True)
    y_centered = boxes - y_mean

    slope = cp.sum(y_centered * x_centered[None, None, :], axis=2) / x_var
    intercept = y_mean[:, :, 0] - slope * x_mean

    trend = slope[:, :, None] * x[None, None, :] + intercept[:, :, None]
    residuals = boxes - trend

    fluctuation_per_box = cp.sqrt(cp.mean(residuals ** 2, axis=2))
    fluctuation = cp.mean(fluctuation_per_box, axis=1)

    return fluctuation


def rolling_hurst_exponent(series: cp.ndarray, window: int, min_box_size: int = 10, n_box_sizes: int = 15) -> cp.ndarray:
    max_box_size = window // 4
    if max_box_size <= min_box_size:
        n = series.shape[0]
        n_windows = n - window + 1
        return cp.full(n_windows, cp.nan, dtype=cp.float64)

    outer_windows = _build_sliding_windows(series, window)
    n_windows = outer_windows.shape[0]

    window_mean = cp.mean(outer_windows, axis=1, keepdims=True)
    demeaned = outer_windows - window_mean
    cumulative_windows = cp.cumsum(demeaned, axis=1)

    box_sizes = cp.unique(cp.logspace(cp.log10(min_box_size), cp.log10(max_box_size), n_box_sizes).astype(cp.int64))

    log_box_size_list = []
    log_fluctuation_columns = []

    for box_size in box_sizes.tolist():
        fluctuation = _batched_detrended_fluctuation(cumulative_windows, box_size)
        safe_fluctuation = cp.where(fluctuation > 0, fluctuation, cp.nan)
        log_fluctuation_columns.append(cp.log(safe_fluctuation))
        log_box_size_list.append(math.log(box_size))

    log_fluctuation_matrix = cp.stack(log_fluctuation_columns, axis=1)
    log_box_sizes_arr = cp.array(log_box_size_list, dtype=cp.float64)

    valid_mask = ~cp.isnan(log_fluctuation_matrix)
    valid_counts = cp.sum(valid_mask, axis=1)

    safe_log_fluctuation = cp.where(valid_mask, log_fluctuation_matrix, 0.0)

    x_broadcast = cp.broadcast_to(log_box_sizes_arr[None, :], log_fluctuation_matrix.shape)
    safe_x = cp.where(valid_mask, x_broadcast, 0.0)

    x_sum = cp.sum(safe_x, axis=1)
    y_sum = cp.sum(safe_log_fluctuation, axis=1)

    safe_counts = cp.where(valid_counts > 0, valid_counts, 1)
    x_mean = x_sum / safe_counts
    y_mean = y_sum / safe_counts

    x_centered = cp.where(valid_mask, x_broadcast - x_mean[:, None], 0.0)
    y_centered = cp.where(valid_mask, log_fluctuation_matrix - y_mean[:, None], 0.0)

    numerator = cp.sum(x_centered * y_centered, axis=1)
    denominator = cp.sum(x_centered ** 2, axis=1)

    safe_denominator = cp.where(denominator > 0, denominator, 1.0)
    hurst = numerator / safe_denominator

    hurst = cp.where(valid_counts >= 2, hurst, cp.nan)

    return hurst