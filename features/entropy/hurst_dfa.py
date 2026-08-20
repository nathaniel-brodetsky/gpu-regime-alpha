import cupy as cp


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
            log_box_sizes.append(math_log(box_size))
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


def math_log(x):
    import math
    return math.log(x)


def rolling_hurst_exponent(series: cp.ndarray, window: int, min_box_size: int = 10, n_box_sizes: int = 15) -> cp.ndarray:
    n = series.shape[0]
    n_windows = n - window + 1

    hurst_values = cp.full(n_windows, cp.nan, dtype=cp.float64)

    for w in range(n_windows):
        segment = series[w:w + window]
        hurst_values[w] = compute_hurst_exponent(segment, min_box_size=min_box_size, max_box_size=window // 4, n_box_sizes=n_box_sizes)

    return hurst_values