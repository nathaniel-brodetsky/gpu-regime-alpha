import cudf
import cupy as cp

from features.microstructure.ofi import compute_ofi
from features.microstructure.spread_features import compute_spread, compute_relative_spread, compute_micro_price, compute_rolling_micro_price_variance
from features.microstructure.realized_vol import compute_log_returns, compute_realized_variance, compute_realized_volatility
from features.microstructure.tail_index import compute_rolling_hill_estimator
from features.entropy.permutation_entropy import rolling_permutation_entropy
from features.entropy.hurst_dfa import rolling_hurst_exponent
from features.spectral.rolling_correlation import build_feature_matrix, compute_rolling_correlation_matrices
from features.spectral.rmt_marchenko_pastur import rolling_mp_deviation, rolling_max_eigenvalue, rolling_spectral_gap
from features.spectral.eigenvector_stability import rolling_eigenvector_stability
from features.spectral.eigenvalue_entropy import rolling_eigenvalue_entropy


def _pad_to_length(arr: cp.ndarray, target_length: int) -> cp.ndarray:
    current_length = arr.shape[0]
    if current_length == target_length:
        return arr
    pad_size = target_length - current_length
    padding = cp.full(pad_size, cp.nan, dtype=cp.float64)
    return cp.concatenate([padding, arr])


def build_tick_level_features(df: cudf.DataFrame, rolling_window: int, tail_window: int, entropy_window: int, hurst_window: int) -> cudf.DataFrame:
    n = len(df)

    ofi = compute_ofi(df).to_cupy()
    spread = compute_spread(df).to_cupy()
    relative_spread = compute_relative_spread(df).to_cupy()
    micro_price = compute_micro_price(df).to_cupy()
    micro_price_var = compute_rolling_micro_price_variance(df, window=rolling_window).to_cupy()
    log_returns = compute_log_returns(df).to_cupy()
    realized_var = compute_realized_variance(df, window=rolling_window).to_cupy()
    realized_vol = compute_realized_volatility(df, window=rolling_window).to_cupy()
    hill = compute_rolling_hill_estimator(df, window=tail_window).to_cupy(na_value=cp.nan)

    perm_entropy = rolling_permutation_entropy(log_returns, window=entropy_window, order=3, delay=1)
    perm_entropy_padded = _pad_to_length(perm_entropy, n)

    hurst = rolling_hurst_exponent(log_returns, window=hurst_window, min_box_size=10)
    hurst_padded = _pad_to_length(hurst, n)

    features = cudf.DataFrame({
        "ofi": ofi,
        "spread": spread,
        "relative_spread": relative_spread,
        "micro_price": micro_price,
        "micro_price_var": micro_price_var,
        "log_return": log_returns,
        "realized_var": realized_var,
        "realized_vol": realized_vol,
        "hill_tail_index": hill,
        "permutation_entropy": perm_entropy_padded,
        "hurst_exponent": hurst_padded,
    })

    return features


def build_spectral_features(tick_features: cudf.DataFrame, spectral_feature_cols: list, spectral_window: int) -> cudf.DataFrame:
    n = len(tick_features)

    matrix = build_feature_matrix(tick_features, spectral_feature_cols)
    matrix_clean = cp.nan_to_num(matrix, nan=0.0)

    corr_matrices = compute_rolling_correlation_matrices(matrix_clean, window=spectral_window)

    mp_dev = rolling_mp_deviation(corr_matrices, n_samples=spectral_window)
    max_eig = rolling_max_eigenvalue(corr_matrices)
    spectral_gap = rolling_spectral_gap(corr_matrices)

    eig_stability = rolling_eigenvector_stability(corr_matrices)
    eig_stability_padded = cp.concatenate([cp.array([cp.nan]), eig_stability])

    eig_entropy = rolling_eigenvalue_entropy(corr_matrices)

    mp_dev_padded = _pad_to_length(mp_dev, n)
    max_eig_padded = _pad_to_length(max_eig, n)
    spectral_gap_padded = _pad_to_length(spectral_gap, n)
    eig_stability_padded = _pad_to_length(eig_stability_padded, n)
    eig_entropy_padded = _pad_to_length(eig_entropy, n)

    spectral_df = cudf.DataFrame({
        "mp_deviation": mp_dev_padded,
        "max_eigenvalue": max_eig_padded,
        "spectral_gap": spectral_gap_padded,
        "eigenvector_stability": eig_stability_padded,
        "eigenvalue_entropy": eig_entropy_padded,
    })

    return spectral_df


def assign_blocks(n: int, block_size: int) -> cp.ndarray:
    block_ids = cp.arange(n) // block_size
    return block_ids


def build_full_feature_matrix(df: cudf.DataFrame, rolling_window: int = 100, tail_window: int = 200, entropy_window: int = 100, hurst_window: int = 300, spectral_window: int = 100) -> cudf.DataFrame:
    tick_features = build_tick_level_features(df, rolling_window, tail_window, entropy_window, hurst_window)

    spectral_cols = ["ofi", "spread", "micro_price", "realized_var"]
    spectral_features = build_spectral_features(tick_features, spectral_cols, spectral_window)

    combined = cudf.concat([tick_features, spectral_features], axis=1)
    combined["timestamp"] = df["timestamp"]

    return combined