import cupy as cp

from features.topology.persistence_diagram import persistence_lifetimes


def compute_persistent_entropy(diagram: cp.ndarray) -> float:
    if diagram.shape[0] == 0:
        return 0.0

    lifetimes = persistence_lifetimes(diagram)
    total_lifetime = cp.sum(lifetimes)

    if total_lifetime == 0:
        return 0.0

    probs = lifetimes / total_lifetime
    safe_probs = probs[probs > 0]

    entropy = -cp.sum(safe_probs * cp.log(safe_probs))
    return float(entropy)


def compute_normalized_persistent_entropy(diagram: cp.ndarray) -> float:
    n = diagram.shape[0]
    if n <= 1:
        return 0.0

    entropy = compute_persistent_entropy(diagram)
    max_entropy = cp.log(n)

    return float(entropy / max_entropy)