import cupy as cp

from features.embedding.takens import takens_embedding


def _pairwise_nearest_neighbor(points: cp.ndarray) -> tuple:
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    dist_sq = cp.sum(diff ** 2, axis=2)

    cp.fill_diagonal(dist_sq, cp.inf)

    nearest_idx = cp.argmin(dist_sq, axis=1)
    nearest_dist = cp.sqrt(dist_sq[cp.arange(n), nearest_idx])

    return nearest_idx, nearest_dist


def false_nearest_neighbors_fraction(series: cp.ndarray, embedding_dim: int, delay: int, rtol: float = 15.0, atol: float = 2.0) -> float:
    embedded_low = takens_embedding(series, embedding_dim, delay)
    embedded_high = takens_embedding(series, embedding_dim + 1, delay)

    n_common = min(embedded_low.shape[0], embedded_high.shape[0])
    embedded_low = embedded_low[:n_common]
    embedded_high = embedded_high[:n_common]

    nearest_idx, nearest_dist_low = _pairwise_nearest_neighbor(embedded_low)

    series_std = cp.std(series)

    extra_coord_self = embedded_high[cp.arange(n_common), -1]
    extra_coord_neighbor = embedded_high[nearest_idx, -1]
    extra_dist = cp.abs(extra_coord_self - extra_coord_neighbor)

    safe_dist_low = cp.where(nearest_dist_low == 0, 1e-12, nearest_dist_low)

    criterion_1 = (extra_dist / safe_dist_low) > rtol
    criterion_2 = cp.sqrt(nearest_dist_low ** 2 + extra_dist ** 2) / series_std > atol

    is_false = criterion_1 | criterion_2
    fraction = cp.mean(is_false.astype(cp.float64))

    return float(fraction)


def find_optimal_embedding_dim(series: cp.ndarray, delay: int, max_dim: int = 10, fnn_threshold: float = 0.05) -> int:
    for dim in range(1, max_dim + 1):
        fraction = false_nearest_neighbors_fraction(series, dim, delay)
        if fraction < fnn_threshold:
            return dim
    return max_dim