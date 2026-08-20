import cudf
import cupy as cp
from cupy.lib.stride_tricks import as_strided


def takens_embedding(series: cp.ndarray, embedding_dim: int, delay: int) -> cp.ndarray:
    n = series.shape[0]
    n_vectors = n - (embedding_dim - 1) * delay

    itemsize = series.itemsize
    embedded = as_strided(
        series,
        shape=(n_vectors, embedding_dim),
        strides=(itemsize, itemsize * delay),
    )

    return cp.ascontiguousarray(embedded)


def takens_embedding_from_series(series: cudf.Series, embedding_dim: int, delay: int) -> cp.ndarray:
    arr = series.to_cupy()
    return takens_embedding(arr, embedding_dim, delay)


def rolling_takens_embedding(series: cudf.Series, window: int, embedding_dim: int, delay: int) -> cp.ndarray:
    arr = series.to_cupy()
    n = arr.shape[0]
    n_windows = n - window + 1

    itemsize = arr.itemsize
    windows = as_strided(
        arr,
        shape=(n_windows, window),
        strides=(itemsize, itemsize),
    )

    n_vectors_per_window = window - (embedding_dim - 1) * delay
    if n_vectors_per_window <= 0:
        raise ValueError("window too small for given embedding_dim and delay")

    embedded_batches = cp.empty((n_windows, n_vectors_per_window, embedding_dim), dtype=arr.dtype)
    for d in range(embedding_dim):
        start = d * delay
        embedded_batches[:, :, d] = windows[:, start:start + n_vectors_per_window]

    return embedded_batches