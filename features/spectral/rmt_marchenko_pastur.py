import cupy as cp


def marchenko_pastur_bounds(n_features: int, n_samples: int) -> tuple:
    q = n_features / n_samples
    lambda_min = (1 - cp.sqrt(q)) ** 2
    lambda_max = (1 + cp.sqrt(q)) ** 2
    return float(lambda_min), float(lambda_max)


def compute_eigenvalue_spectrum(correlation_matrices: cp.ndarray) -> cp.ndarray:
    eigenvalues = cp.linalg.eigvalsh(correlation_matrices)
    return eigenvalues


def rolling_mp_deviation(correlation_matrices: cp.ndarray, n_samples: int) -> cp.ndarray:
    n_windows, n_features, _ = correlation_matrices.shape
    lambda_min, lambda_max = marchenko_pastur_bounds(n_features, n_samples)

    eigenvalues = compute_eigenvalue_spectrum(correlation_matrices)

    outside_mp = (eigenvalues > lambda_max) | (eigenvalues < lambda_min)
    deviation_fraction = cp.mean(outside_mp.astype(cp.float64), axis=1)

    return deviation_fraction


def rolling_max_eigenvalue(correlation_matrices: cp.ndarray) -> cp.ndarray:
    eigenvalues = compute_eigenvalue_spectrum(correlation_matrices)
    return eigenvalues[:, -1]


def rolling_spectral_gap(correlation_matrices: cp.ndarray) -> cp.ndarray:
    eigenvalues = compute_eigenvalue_spectrum(correlation_matrices)
    return eigenvalues[:, -1] - eigenvalues[:, -2]