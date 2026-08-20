import cudf
import cupy as cp

from features.topology.vietoris_rips import compute_persistence_diagrams_from_series, diagram_to_cupy
from features.topology.wasserstein_pd import sinkhorn_divergence_diagrams
from features.spectral.rolling_correlation import build_feature_matrix, compute_rolling_correlation_matrices
from features.spectral.rmt_marchenko_pastur import compute_eigenvalue_spectrum


def _split_into_blocks(arr: cp.ndarray, block_size: int) -> list:
    n = arr.shape[0]
    n_blocks = n // block_size
    blocks = [arr[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    return blocks


def compute_topological_shift_series(log_returns: cp.ndarray, block_size: int, embedding_dim: int = 3, delay: int = 1, maxdim: int = 1) -> cp.ndarray:
    blocks = _split_into_blocks(log_returns, block_size)

    diagrams = []
    for block in blocks:
        dgms = compute_persistence_diagrams_from_series(block, embedding_dim, delay, maxdim)
        h1 = diagram_to_cupy(dgms[1])
        diagrams.append(h1)

    shift_scores = cp.full(len(diagrams), cp.nan, dtype=cp.float64)
    for i in range(1, len(diagrams)):
        shift_scores[i] = sinkhorn_divergence_diagrams(diagrams[i - 1], diagrams[i])

    return shift_scores


def compute_spectral_shift_series(feature_matrix: cp.ndarray, block_size: int) -> cp.ndarray:
    blocks = _split_into_blocks(feature_matrix, block_size)

    spectra = []
    for block in blocks:
        block_reshaped = block[None, :, :]
        corr = compute_rolling_correlation_matrices_single(block_reshaped)
        eigenvalues = compute_eigenvalue_spectrum(corr)
        spectra.append(eigenvalues[0])

    shift_scores = cp.full(len(spectra), cp.nan, dtype=cp.float64)
    for i in range(1, len(spectra)):
        diff = spectra[i] - spectra[i - 1]
        shift_scores[i] = float(cp.linalg.norm(diff))

    return shift_scores


def compute_rolling_correlation_matrices_single(block: cp.ndarray) -> cp.ndarray:
    single_window = block[0]
    mean = single_window.mean(axis=0, keepdims=True)
    centered = single_window - mean
    std = single_window.std(axis=0, keepdims=True)
    safe_std = cp.where(std == 0, 1.0, std)
    normalized = centered / safe_std

    n = normalized.shape[0]
    correlation = (normalized.T @ normalized) / n
    return correlation[None, :, :]


def build_block_level_shift_scores(df: cudf.DataFrame, log_returns: cp.ndarray, spectral_feature_cols: list, tick_features: cudf.DataFrame, block_size: int) -> cudf.DataFrame:
    topo_shift = compute_topological_shift_series(log_returns, block_size)

    feature_matrix = build_feature_matrix(tick_features, spectral_feature_cols)
    feature_matrix_clean = cp.nan_to_num(feature_matrix, nan=0.0)
    spectral_shift = compute_spectral_shift_series(feature_matrix_clean, block_size)

    n_blocks = min(topo_shift.shape[0], spectral_shift.shape[0])

    result = cudf.DataFrame({
        "block_id": cp.arange(n_blocks),
        "topological_shift": topo_shift[:n_blocks],
        "spectral_shift": spectral_shift[:n_blocks],
    })

    return result