import cupy as cp


def compute_top_eigenvectors(correlation_matrices: cp.ndarray) -> cp.ndarray:
    eigenvalues, eigenvectors = cp.linalg.eigh(correlation_matrices)
    top_eigenvectors = eigenvectors[:, :, -1]
    return top_eigenvectors


def rolling_eigenvector_stability(correlation_matrices: cp.ndarray) -> cp.ndarray:
    top_eigenvectors = compute_top_eigenvectors(correlation_matrices)

    current = top_eigenvectors[1:]
    previous = top_eigenvectors[:-1]

    dot_product = cp.sum(current * previous, axis=1)
    norm_current = cp.linalg.norm(current, axis=1)
    norm_previous = cp.linalg.norm(previous, axis=1)

    safe_denom = cp.where(norm_current * norm_previous == 0, 1e-12, norm_current * norm_previous)
    cosine_similarity = dot_product / safe_denom

    stability = cp.abs(cosine_similarity)

    return stability