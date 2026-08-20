import cupy as cp

from features.embedding.takens import takens_embedding


def _find_nearest_non_temporal_neighbor(embedded: cp.ndarray, theiler_window: int) -> cp.ndarray:
    n = embedded.shape[0]
    diff = embedded[:, None, :] - embedded[None, :, :]
    dist = cp.sqrt(cp.sum(diff ** 2, axis=2))

    idx = cp.arange(n)
    temporal_mask = cp.abs(idx[:, None] - idx[None, :]) <= theiler_window
    dist = cp.where(temporal_mask, cp.inf, dist)

    nearest_idx = cp.argmin(dist, axis=1)
    return nearest_idx


def compute_local_lyapunov(embedded: cp.ndarray, theiler_window: int, evolution_steps: int) -> float:
    n = embedded.shape[0]
    usable_n = n - evolution_steps

    if usable_n <= theiler_window + 1:
        return cp.nan

    base = embedded[:usable_n]
    nearest_idx = _find_nearest_non_temporal_neighbor(base, theiler_window)

    valid_mask = nearest_idx < usable_n
    if not bool(cp.any(valid_mask)):
        return cp.nan

    log_divergence_sum = cp.zeros(usable_n, dtype=cp.float64)
    log_divergence_count = cp.zeros(usable_n, dtype=cp.float64)

    for step in range(1, evolution_steps + 1):
        point_now = embedded[step:usable_n + step]
        neighbor_now_idx = nearest_idx + step

        in_bounds = neighbor_now_idx < n
        neighbor_now_idx_clamped = cp.where(in_bounds, neighbor_now_idx, 0)
        neighbor_now = embedded[neighbor_now_idx_clamped]

        dist = cp.sqrt(cp.sum((point_now - neighbor_now) ** 2, axis=1))
        safe_dist = cp.where(dist == 0, 1e-12, dist)
        log_dist = cp.log(safe_dist)

        mask = in_bounds & valid_mask
        log_divergence_sum = cp.where(mask, log_divergence_sum + log_dist, log_divergence_sum)
        log_divergence_count = cp.where(mask, log_divergence_count + 1, log_divergence_count)

    safe_count = cp.where(log_divergence_count == 0, 1.0, log_divergence_count)
    mean_log_divergence = log_divergence_sum / safe_count

    valid_points = log_divergence_count > 0
    if not bool(cp.any(valid_points)):
        return cp.nan

    lyapunov_estimate = float(cp.mean(mean_log_divergence[valid_points])) / evolution_steps
    return lyapunov_estimate


def rolling_ftle(series: cp.ndarray, window: int, embedding_dim: int, delay: int, theiler_window: int, evolution_steps: int) -> cp.ndarray:
    n = series.shape[0]
    n_windows = n - window + 1

    ftle_values = cp.full(n_windows, cp.nan, dtype=cp.float64)

    for w in range(n_windows):
        segment = series[w:w + window]
        embedded = takens_embedding(segment, embedding_dim, delay)
        ftle_values[w] = compute_local_lyapunov(embedded, theiler_window, evolution_steps)

    return ftle_values