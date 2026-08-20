import cudf
import cupy as cp
from cuml.manifold import UMAP


def drop_warmup_nan(features: cudf.DataFrame, feature_columns: list) -> cudf.DataFrame:
    clean = features.dropna(subset=feature_columns)
    return clean.reset_index(drop=True)


def standardize_features(features: cudf.DataFrame, feature_columns: list) -> cp.ndarray:
    matrix = features[feature_columns].to_cupy()

    mean = cp.mean(matrix, axis=0)
    std = cp.std(matrix, axis=0)
    safe_std = cp.where(std == 0, 1.0, std)

    standardized = (matrix - mean) / safe_std
    return standardized


def fit_umap_reduction(features: cudf.DataFrame, feature_columns: list, n_components: int = 3, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> cp.ndarray:
    clean_features = drop_warmup_nan(features, feature_columns)
    standardized = standardize_features(clean_features, feature_columns)

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(standardized)
    return embedding, clean_features