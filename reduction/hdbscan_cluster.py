import cudf
import cupy as cp
from cuml.cluster import HDBSCAN


def fit_hdbscan_clusters(embedding: cp.ndarray, min_cluster_size: int = 30, min_samples: int = None) -> cp.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=True,
    )

    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def cluster_summary(labels: cp.ndarray) -> dict:
    unique_labels = cp.unique(labels)
    n_clusters = int(cp.sum(unique_labels >= 0))
    n_noise = int(cp.sum(labels == -1))
    n_total = labels.shape[0]

    cluster_sizes = {}
    for label in unique_labels.tolist():
        if label == -1:
            continue
        size = int(cp.sum(labels == label))
        cluster_sizes[int(label)] = size

    return {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "noise_fraction": n_noise / n_total,
        "cluster_sizes": cluster_sizes,
    }


def attach_cluster_labels(clean_features: cudf.DataFrame, labels: cp.ndarray) -> cudf.DataFrame:
    result = clean_features.copy()
    result["cluster_id"] = labels
    return result