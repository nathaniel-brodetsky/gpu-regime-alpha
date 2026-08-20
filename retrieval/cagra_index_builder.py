import cupy as cp
from cuvs.neighbors import cagra


def build_cagra_index(embedding: cp.ndarray, intermediate_graph_degree: int = 128, graph_degree: int = 64) -> cagra.Index:
    embedding_f32 = embedding.astype(cp.float32)

    build_params = cagra.IndexParams(
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
    )

    index = cagra.build(build_params, embedding_f32)
    return index


def query_nearest_regime(index: cagra.Index, query_points: cp.ndarray, cluster_labels: cp.ndarray, k: int = 1) -> cp.ndarray:
    query_f32 = query_points.astype(cp.float32)

    search_params = cagra.SearchParams()
    distances, neighbor_indices = cagra.search(search_params, index, query_f32, k)

    neighbor_indices_arr = cp.asarray(neighbor_indices)
    nearest_idx = neighbor_indices_arr[:, 0]

    matched_labels = cluster_labels[nearest_idx]
    return matched_labels