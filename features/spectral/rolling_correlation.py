import cudf
import cupy as cp
from cupy.lib.stride_tricks import as_strided


def build_feature_matrix(df: cudf.DataFrame, feature_columns: list) -> cp.ndarray:
    columns = [df[col].to_cupy() for col in feature_columns]
    matrix = cp.stack(columns, axis=1)
    return matrix


def _build_sliding_windows_multivariate(matrix: cp.ndarray, window: int) -> cp.ndarray:
    n, n_features = matrix.shape
    n_windows = n - window + 1

    itemsize = matrix.itemsize
    row_stride = matrix.strides[0]
    col_stride = matrix.strides[1]

    windows = as_strided(
        matrix,
        shape=(n_windows, window, n_features),
        strides=(row_stride, row_stride, col_stride),
    )

    return windows


def compute_rolling_correlation_matrices(matrix: cp.ndarray, window: int) -> cp.ndarray:
    windows = _build_sliding_windows_multivariate(matrix, window)

    mean = windows.mean(axis=1, keepdims=True)
    centered = windows - mean

    std = windows.std(axis=1, keepdims=True)
    safe_std = cp.where(std == 0, 1.0, std)
    normalized = centered / safe_std

    n_windows = normalized.shape[0]
    n_features = normalized.shape[2]

    correlation_matrices = cp.empty((n_windows, n_features, n_features), dtype=cp.float64)
    for w in range(n_windows):
        correlation_matrices[w] = (normalized[w].T @ normalized[w]) / window

    return correlation_matrices