import time
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.lib.stride_tricks import as_strided
import math

import umap
import hdbscan
from ripser import ripser


def generate_synthetic_lob_cpu(n_ticks: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dt = np.full(n_ticks, 1e-3, dtype=np.float64)
    timestamps = np.cumsum(dt)

    mid_price_drift = rng.standard_normal(n_ticks) * 0.0001
    mid_price = 100.0 + np.cumsum(mid_price_drift)

    spread = np.abs(rng.standard_normal(n_ticks) * 0.005 + 0.02) + 1e-4
    best_bid = mid_price - spread / 2
    best_ask = mid_price + spread / 2

    bid_size = np.abs(rng.standard_normal(n_ticks) * 150 + 500) + 1.0
    ask_size = np.abs(rng.standard_normal(n_ticks) * 150 + 500) + 1.0

    order_flow_sign = np.where(rng.uniform(0, 1, n_ticks) > 0.5, 1.0, -1.0)
    trade_size = np.abs(rng.standard_normal(n_ticks) * 40 + 100) + 1.0
    trade_price = np.where(order_flow_sign > 0, best_ask, best_bid)

    n_shocks = max(n_ticks // 5000, 1)
    shock_indices = rng.integers(0, n_ticks, size=n_shocks)
    volatility_multiplier = np.ones(n_ticks)
    for shock_idx in shock_indices.tolist():
        window_end = min(shock_idx + 2000, n_ticks)
        volatility_multiplier[shock_idx:window_end] *= 4.0

    mid_price = mid_price * volatility_multiplier / volatility_multiplier[0]

    return pd.DataFrame({
        "timestamp": timestamps,
        "mid_price": mid_price,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "trade_price": trade_price,
        "trade_size": trade_size,
        "order_flow_sign": order_flow_sign,
    })


def compute_ofi_cpu(df: pd.DataFrame) -> pd.Series:
    bid_price_delta = df["best_bid"].diff().fillna(0)
    ask_price_delta = df["best_ask"].diff().fillna(0)
    bid_size_delta = df["bid_size"].diff().fillna(0)
    ask_size_delta = df["ask_size"].diff().fillna(0)

    bid_contribution = np.where(
        bid_price_delta > 0, df["bid_size"],
        np.where(bid_price_delta == 0, bid_size_delta, 0.0),
    )
    ask_contribution = np.where(
        ask_price_delta < 0, df["ask_size"],
        np.where(ask_price_delta == 0, ask_size_delta, 0.0),
    )

    return pd.Series(bid_contribution - ask_contribution, index=df.index)


def compute_spread_cpu(df: pd.DataFrame) -> pd.Series:
    return df["best_ask"] - df["best_bid"]


def compute_micro_price_cpu(df: pd.DataFrame) -> pd.Series:
    total_size = df["bid_size"] + df["ask_size"]
    weighted_bid = df["best_bid"] * df["ask_size"]
    weighted_ask = df["best_ask"] * df["bid_size"]
    return (weighted_bid + weighted_ask) / total_size


def compute_log_returns_cpu(df: pd.DataFrame, price_col: str = "mid_price") -> pd.Series:
    log_price = np.log(df[price_col])
    return log_price.diff().fillna(0)


def compute_realized_variance_cpu(df: pd.DataFrame, window: int, price_col: str = "mid_price") -> pd.Series:
    log_returns = compute_log_returns_cpu(df, price_col)
    return (log_returns ** 2).rolling(window=window, min_periods=1).sum()


def compute_realized_volatility_cpu(df: pd.DataFrame, window: int, price_col: str = "mid_price") -> pd.Series:
    return np.sqrt(compute_realized_variance_cpu(df, window, price_col))


def compute_rolling_hill_estimator_cpu(df: pd.DataFrame, window: int, tail_fraction: float = 0.1, price_col: str = "mid_price") -> pd.Series:
    log_returns = compute_log_returns_cpu(df, price_col).to_numpy()
    abs_returns = np.abs(log_returns)

    n = abs_returns.shape[0]
    n_windows = n - window + 1
    itemsize = abs_returns.itemsize
    windows = as_strided(abs_returns, shape=(n_windows, window), strides=(itemsize, itemsize))

    sorted_windows = np.sort(windows, axis=1)[:, ::-1]
    k = max(int(window * tail_fraction), 2)
    tail_values = sorted_windows[:, :k]
    threshold = sorted_windows[:, k - 1:k]

    safe_threshold = np.where(threshold == 0, 1e-12, threshold)
    safe_tail = np.where(tail_values == 0, 1e-12, tail_values)

    log_ratios = np.log(safe_tail / safe_threshold)
    hill_sum = log_ratios.sum(axis=1)
    hill_estimator = hill_sum / k

    tail_index = np.where(hill_estimator == 0, np.nan, 1.0 / hill_estimator)

    padded = np.full(n, np.nan, dtype=np.float64)
    padded[window - 1:] = tail_index
    return pd.Series(padded, index=df.index)


def rolling_permutation_entropy_cpu(series: np.ndarray, window: int, order: int = 3, delay: int = 1) -> np.ndarray:
    if delay > 1:
        series = series[::delay]

    n = series.shape[0]
    n_windows = n - window + 1

    entropy_values = np.full(n_windows, np.nan, dtype=np.float64)
    n_patterns = math.factorial(order)
    max_entropy = math.log(n_patterns)

    for w in range(n_windows):
        segment = series[w:w + window]
        n_sub = window - order + 1
        counts = np.zeros(n_patterns, dtype=np.float64)
        for i in range(n_sub):
            sub = segment[i:i + order]
            pattern = tuple(np.argsort(sub))
            idx = 0
            seen = []
            for p in pattern:
                rank = p - sum(1 for s in seen if s < p)
                idx = idx * (order - len(seen)) + rank
                seen.append(p)
            counts[idx % n_patterns] += 1
        probs = counts / n_sub
        safe_probs = probs[probs > 0]
        entropy = -np.sum(safe_probs * np.log(safe_probs))
        entropy_values[w] = entropy / max_entropy

    return entropy_values


def rolling_hurst_exponent_cpu(series: np.ndarray, window: int, min_box_size: int = 10, n_box_sizes: int = 15) -> np.ndarray:
    n = series.shape[0]
    n_windows = n - window + 1
    max_box_size = window // 4

    hurst_values = np.full(n_windows, np.nan, dtype=np.float64)

    if max_box_size <= min_box_size:
        return hurst_values

    box_sizes = np.unique(np.logspace(np.log10(min_box_size), np.log10(max_box_size), n_box_sizes).astype(np.int64))

    for w in range(n_windows):
        segment = series[w:w + window]
        demeaned = segment - np.mean(segment)
        cumulative = np.cumsum(demeaned)

        log_box_sizes = []
        log_fluctuations = []

        for box_size in box_sizes:
            n_boxes = window // box_size
            if n_boxes < 1:
                continue
            trimmed = cumulative[:n_boxes * box_size]
            boxes = trimmed.reshape(n_boxes, box_size)

            x = np.arange(box_size, dtype=np.float64)
            x_mean = np.mean(x)
            x_centered = x - x_mean
            x_var = np.sum(x_centered ** 2)

            y_mean = np.mean(boxes, axis=1, keepdims=True)
            y_centered = boxes - y_mean

            slope = np.sum(y_centered * x_centered[None, :], axis=1) / x_var
            intercept = y_mean.flatten() - slope * x_mean

            trend = slope[:, None] * x[None, :] + intercept[:, None]
            residuals = boxes - trend

            fluctuation = np.sqrt(np.mean(residuals ** 2, axis=1))
            mean_fluctuation = np.mean(fluctuation)

            if mean_fluctuation > 0:
                log_box_sizes.append(math.log(box_size))
                log_fluctuations.append(math.log(mean_fluctuation))

        if len(log_box_sizes) < 2:
            continue

        x_arr = np.array(log_box_sizes)
        y_arr = np.array(log_fluctuations)
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)

        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sum((x_arr - x_mean) ** 2)

        hurst_values[w] = numerator / denominator

    return hurst_values


def compute_rolling_correlation_matrices_cpu(matrix: np.ndarray, window: int) -> np.ndarray:
    n, n_features = matrix.shape
    n_windows = n - window + 1

    row_stride = matrix.strides[0]
    col_stride = matrix.strides[1]

    windows = as_strided(matrix, shape=(n_windows, window, n_features), strides=(row_stride, row_stride, col_stride))

    mean = windows.mean(axis=1, keepdims=True)
    centered = windows - mean
    std = windows.std(axis=1, keepdims=True)
    safe_std = np.where(std == 0, 1.0, std)
    normalized = centered / safe_std

    correlation_matrices = np.einsum('wtf,wtg->wfg', normalized, normalized) / window
    return correlation_matrices


def build_full_feature_matrix_cpu(df: pd.DataFrame, rolling_window: int = 100, tail_window: int = 200, entropy_window: int = 100, hurst_window: int = 300, spectral_window: int = 100) -> pd.DataFrame:
    n = len(df)

    ofi = compute_ofi_cpu(df).to_numpy()
    spread = compute_spread_cpu(df).to_numpy()
    micro_price = compute_micro_price_cpu(df).to_numpy()
    log_returns = compute_log_returns_cpu(df).to_numpy()
    realized_var = compute_realized_variance_cpu(df, window=rolling_window).to_numpy()
    realized_vol = compute_realized_volatility_cpu(df, window=rolling_window).to_numpy()
    hill = compute_rolling_hill_estimator_cpu(df, window=tail_window).to_numpy()

    perm_entropy = rolling_permutation_entropy_cpu(log_returns, window=entropy_window, order=3, delay=1)
    perm_entropy_padded = np.concatenate([np.full(n - perm_entropy.shape[0], np.nan), perm_entropy])

    hurst = rolling_hurst_exponent_cpu(log_returns, window=hurst_window, min_box_size=10)
    hurst_padded = np.concatenate([np.full(n - hurst.shape[0], np.nan), hurst])

    tick_features = pd.DataFrame({
        "ofi": ofi,
        "spread": spread,
        "micro_price": micro_price,
        "log_return": log_returns,
        "realized_var": realized_var,
        "realized_vol": realized_vol,
        "hill_tail_index": hill,
        "permutation_entropy": perm_entropy_padded,
        "hurst_exponent": hurst_padded,
    })

    spectral_cols = ["ofi", "spread", "micro_price", "realized_var"]
    spectral_matrix = tick_features[spectral_cols].fillna(0).to_numpy()
    corr_matrices = compute_rolling_correlation_matrices_cpu(spectral_matrix, window=spectral_window)

    eigenvalues = np.linalg.eigvalsh(corr_matrices)
    max_eig = eigenvalues[:, -1]
    spectral_gap = eigenvalues[:, -1] - eigenvalues[:, -2]

    max_eig_padded = np.concatenate([np.full(n - max_eig.shape[0], np.nan), max_eig])
    spectral_gap_padded = np.concatenate([np.full(n - spectral_gap.shape[0], np.nan), spectral_gap])

    tick_features["max_eigenvalue"] = max_eig_padded
    tick_features["spectral_gap"] = spectral_gap_padded
    tick_features["timestamp"] = df["timestamp"].to_numpy()

    return tick_features


def run_cpu_pipeline_benchmark(n_ticks: int, rolling_window: int = 100, tail_window: int = 200, entropy_window: int = 100, hurst_window: int = 300, spectral_window: int = 100, umap_components: int = 3, min_cluster_size: int = 50, horizon: int = 10) -> dict:
    timings = {}

    start = time.perf_counter()
    df = generate_synthetic_lob_cpu(n_ticks)
    timings["data_generation"] = time.perf_counter() - start
    print(f"[data_generation] {timings['data_generation']:.4f}s")

    start = time.perf_counter()
    features = build_full_feature_matrix_cpu(df, rolling_window, tail_window, entropy_window, hurst_window, spectral_window)
    timings["feature_engineering"] = time.perf_counter() - start
    print(f"[feature_engineering] {timings['feature_engineering']:.4f}s")

    feature_cols = [c for c in features.columns if c != "timestamp"]
    clean_features = features.dropna(subset=feature_cols).reset_index(drop=True)

    matrix = clean_features[feature_cols].to_numpy()
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    standardized = (matrix - mean) / safe_std

    start = time.perf_counter()
    reducer = umap.UMAP(n_components=umap_components, random_state=42)
    embedding = reducer.fit_transform(standardized)
    timings["umap_reduction"] = time.perf_counter() - start
    print(f"[umap_reduction] {timings['umap_reduction']:.4f}s")

    start = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding)
    timings["hdbscan_clustering"] = time.perf_counter() - start
    print(f"[hdbscan_clustering] {timings['hdbscan_clustering']:.4f}s")

    clustered = clean_features.copy()
    clustered["cluster_id"] = labels

    aligned_raw = df.iloc[-len(clustered):].reset_index(drop=True)
    future_price = aligned_raw["mid_price"].shift(-horizon)
    current_price = aligned_raw["mid_price"]
    target = (future_price - current_price) / current_price

    training_frame = clustered[feature_cols + ["cluster_id"]].copy()
    training_frame["target"] = target.reset_index(drop=True)
    training_frame = training_frame.dropna()

    split_idx = int(len(training_frame) * 0.7)
    train = training_frame.iloc[:split_idx]
    test = training_frame.iloc[split_idx:]

    train = train.copy()
    train["cluster_id"] = train["cluster_id"].astype("category")
    test = test.copy()
    test["cluster_id"] = test["cluster_id"].astype("category")

    all_features = feature_cols + ["cluster_id"]

    start = time.perf_counter()
    dtrain = xgb.DMatrix(train[all_features], label=train["target"], enable_categorical=True)
    params = {"tree_method": "hist", "max_depth": 6, "eta": 0.05, "objective": "reg:squarederror"}
    booster = xgb.train(params, dtrain, num_boost_round=100)
    timings["xgboost_train"] = time.perf_counter() - start
    print(f"[xgboost_train] {timings['xgboost_train']:.4f}s")

    start = time.perf_counter()
    dtest = xgb.DMatrix(test[all_features], enable_categorical=True)
    booster.predict(dtest)
    timings["xgboost_predict"] = time.perf_counter() - start
    print(f"[xgboost_predict] {timings['xgboost_predict']:.4f}s")

    total = sum(timings.values())
    timings["total"] = total
    print(f"\n[TOTAL] {total:.4f}s for n_ticks={n_ticks}")

    return timings