import time
import cudf
import cupy as cp

from data.synthetic.lob_generator import generate_synthetic_lob
from fusion.feature_matrix_builder import build_full_feature_matrix
from reduction.umap_reduce import fit_umap_reduction
from reduction.hdbscan_cluster import fit_hdbscan_clusters, attach_cluster_labels
from retrieval.cagra_index_builder import build_cagra_index, query_nearest_regime
from model.xgb_conditioned_alpha import build_training_frame, walk_forward_split, train_conditioned_xgb, predict_alpha


def _sync_timer():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def benchmark_stage(name: str, fn, *args, **kwargs):
    start = _sync_timer()
    result = fn(*args, **kwargs)
    end = _sync_timer()
    elapsed = end - start
    print(f"[{name}] {elapsed:.4f}s")
    return result, elapsed


def run_full_pipeline_benchmark(n_ticks: int, rolling_window: int = 100, tail_window: int = 200, entropy_window: int = 100, hurst_window: int = 300, spectral_window: int = 100, umap_components: int = 3, min_cluster_size: int = 50, horizon: int = 10) -> dict:
    timings = {}

    df, t = benchmark_stage("data_generation", generate_synthetic_lob, n_ticks)
    timings["data_generation"] = t

    features, t = benchmark_stage(
        "feature_engineering",
        build_full_feature_matrix,
        df, rolling_window, tail_window, entropy_window, hurst_window, spectral_window,
    )
    timings["feature_engineering"] = t

    feature_cols = [c for c in features.columns if c != "timestamp"]

    (embedding, clean_features), t = benchmark_stage(
        "umap_reduction",
        fit_umap_reduction,
        features, feature_cols, umap_components,
    )
    timings["umap_reduction"] = t

    (labels, clusterer), t = benchmark_stage(
        "hdbscan_clustering",
        fit_hdbscan_clusters,
        embedding, min_cluster_size,
    )
    timings["hdbscan_clustering"] = t

    clustered = attach_cluster_labels(clean_features, labels)

    index, t = benchmark_stage("cagra_index_build", build_cagra_index, embedding)
    timings["cagra_index_build"] = t

    query_points = embedding[:100]
    _, t = benchmark_stage("cagra_query", query_nearest_regime, index, query_points, labels)
    timings["cagra_query"] = t

    training_frame, t = benchmark_stage(
        "training_frame_build",
        build_training_frame,
        clustered, df, horizon, feature_cols,
    )
    timings["training_frame_build"] = t

    (train, test), t = benchmark_stage("walk_forward_split", walk_forward_split, training_frame)
    timings["walk_forward_split"] = t

    booster, t = benchmark_stage("xgboost_train", train_conditioned_xgb, train, feature_cols)
    timings["xgboost_train"] = t

    _, t = benchmark_stage("xgboost_predict", predict_alpha, booster, test, feature_cols)
    timings["xgboost_predict"] = t

    total = sum(timings.values())
    timings["total"] = total

    print(f"\n[TOTAL] {total:.4f}s for n_ticks={n_ticks}")

    return timings