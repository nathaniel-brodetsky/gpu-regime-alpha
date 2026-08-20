import cupy as cp
import numpy as np
from gph import ripser_parallel

from features.embedding.takens import takens_embedding


def compute_persistence_diagrams(embedded: cp.ndarray, maxdim: int = 1) -> dict:
    points_host = cp.asnumpy(embedded).astype(np.float64)
    result = ripser_parallel(points_host, maxdim=maxdim)
    return result["dgms"]


def compute_persistence_diagrams_from_series(series: cp.ndarray, embedding_dim: int, delay: int, maxdim: int = 1) -> dict:
    embedded = takens_embedding(series, embedding_dim, delay)
    return compute_persistence_diagrams(embedded, maxdim)


def diagram_to_cupy(diagram: np.ndarray) -> cp.ndarray:
    finite_mask = np.isfinite(diagram[:, 1])
    finite_diagram = diagram[finite_mask]
    return cp.asarray(finite_diagram)