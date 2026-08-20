import cupy as cp

from features.embedding.takens import takens_embedding


def compute_recurrence_matrix(embedded: cp.ndarray, threshold: float = None, threshold_quantile: float = 0.1) -> cp.ndarray:
    diff = embedded[:, None, :] - embedded[None, :, :]
    dist = cp.sqrt(cp.sum(diff ** 2, axis=2))

    if threshold is None:
        upper_triangle_idx = cp.triu_indices(dist.shape[0], k=1)
        upper_values = dist[upper_triangle_idx]
        threshold = float(cp.quantile(upper_values, threshold_quantile))

    recurrence = (dist <= threshold).astype(cp.float64)
    return recurrence


def compute_recurrence_matrix_from_series(series: cp.ndarray, embedding_dim: int, delay: int, threshold_quantile: float = 0.1) -> cp.ndarray:
    embedded = takens_embedding(series, embedding_dim, delay)
    return compute_recurrence_matrix(embedded, threshold_quantile=threshold_quantile)


def recurrence_rate(recurrence_matrix: cp.ndarray) -> float:
    n = recurrence_matrix.shape[0]
    total_points = n * n
    recurrent_points = cp.sum(recurrence_matrix)
    return float(recurrent_points / total_points)