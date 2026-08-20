import cupy as cp

from features.spectral.rmt_marchenko_pastur import compute_eigenvalue_spectrum


def rolling_eigenvalue_entropy(correlation_matrices: cp.ndarray) -> cp.ndarray:
    eigenvalues = compute_eigenvalue_spectrum(correlation_matrices)

    eigenvalues_clipped = cp.clip(eigenvalues, a_min=0.0, a_max=None)
    total = cp.sum(eigenvalues_clipped, axis=1, keepdims=True)

    safe_total = cp.where(total == 0, 1.0, total)
    probs = eigenvalues_clipped / safe_total

    safe_probs = cp.where(probs > 0, probs, 1.0)
    log_probs = cp.where(probs > 0, cp.log(safe_probs), 0.0)

    entropy = -cp.sum(probs * log_probs, axis=1)

    n_features = correlation_matrices.shape[1]
    max_entropy = cp.log(n_features)

    normalized_entropy = entropy / max_entropy

    return normalized_entropy