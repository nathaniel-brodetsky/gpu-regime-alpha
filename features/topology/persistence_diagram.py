import cupy as cp


def persistence_lifetimes(diagram: cp.ndarray) -> cp.ndarray:
    return diagram[:, 1] - diagram[:, 0]


def diagram_summary_stats(diagram: cp.ndarray) -> dict:
    if diagram.shape[0] == 0:
        return {
            "n_features": 0,
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "std_persistence": 0.0,
        }

    lifetimes = persistence_lifetimes(diagram)

    return {
        "n_features": int(diagram.shape[0]),
        "total_persistence": float(cp.sum(lifetimes)),
        "max_persistence": float(cp.max(lifetimes)),
        "mean_persistence": float(cp.mean(lifetimes)),
        "std_persistence": float(cp.std(lifetimes)) if diagram.shape[0] > 1 else 0.0,
    }


def top_k_persistent_features(diagram: cp.ndarray, k: int) -> cp.ndarray:
    lifetimes = persistence_lifetimes(diagram)
    n = diagram.shape[0]

    if n == 0:
        return cp.empty((0, 2), dtype=cp.float64)

    k_actual = min(k, n)
    top_indices = cp.argsort(lifetimes)[::-1][:k_actual]
    return diagram[top_indices]