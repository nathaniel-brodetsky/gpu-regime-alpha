import cupy as cp


def _pad_diagrams_to_diagonal(diagram_a: cp.ndarray, diagram_b: cp.ndarray) -> tuple:
    diag_projection_a = cp.stack([
        (diagram_a[:, 0] + diagram_a[:, 1]) / 2,
        (diagram_a[:, 0] + diagram_a[:, 1]) / 2,
    ], axis=1)

    diag_projection_b = cp.stack([
        (diagram_b[:, 0] + diagram_b[:, 1]) / 2,
        (diagram_b[:, 0] + diagram_b[:, 1]) / 2,
    ], axis=1)

    padded_a = cp.concatenate([diagram_a, diag_projection_b], axis=0)
    padded_b = cp.concatenate([diagram_b, diag_projection_a], axis=0)

    return padded_a, padded_b


def _cost_matrix(points_a: cp.ndarray, points_b: cp.ndarray) -> cp.ndarray:
    diff = points_a[:, None, :] - points_b[None, :, :]
    return cp.sqrt(cp.sum(diff ** 2, axis=2))


def _sinkhorn_ot(cost_matrix: cp.ndarray, reg: float, n_iter: int) -> float:
    n, m = cost_matrix.shape

    K = cp.exp(-cost_matrix / reg)

    mu = cp.ones(n, dtype=cp.float64) / n
    nu = cp.ones(m, dtype=cp.float64) / m

    u = cp.ones(n, dtype=cp.float64) / n
    v = cp.ones(m, dtype=cp.float64) / m

    for _ in range(n_iter):
        u = mu / (K @ v + 1e-30)
        v = nu / (K.T @ u + 1e-30)

    transport_plan = cp.diag(u) @ K @ cp.diag(v)
    return float(cp.sum(transport_plan * cost_matrix))


def sinkhorn_divergence_diagrams(diagram_a: cp.ndarray, diagram_b: cp.ndarray, reg_fraction: float = 0.01, n_iter: int = 200) -> float:
    if diagram_a.shape[0] == 0 and diagram_b.shape[0] == 0:
        return 0.0

    if diagram_a.shape[0] == 0 or diagram_b.shape[0] == 0:
        non_empty = diagram_a if diagram_a.shape[0] > 0 else diagram_b
        lifetimes = non_empty[:, 1] - non_empty[:, 0]
        return float(cp.sum(lifetimes) / 2.0)

    padded_a, padded_b = _pad_diagrams_to_diagonal(diagram_a, diagram_b)

    cost_ab = _cost_matrix(padded_a, padded_b)
    cost_aa = _cost_matrix(padded_a, padded_a)
    cost_bb = _cost_matrix(padded_b, padded_b)

    max_cost = float(cp.max(cost_ab))
    if max_cost == 0:
        return 0.0
    reg = max_cost * reg_fraction

    ot_ab = _sinkhorn_ot(cost_ab, reg, n_iter)
    ot_aa = _sinkhorn_ot(cost_aa, reg, n_iter)
    ot_bb = _sinkhorn_ot(cost_bb, reg, n_iter)

    divergence = ot_ab - 0.5 * ot_aa - 0.5 * ot_bb
    return max(divergence, 0.0)


def structural_shift_score(diagram_t: cp.ndarray, diagram_t_plus_1: cp.ndarray, reg_fraction: float = 0.01, n_iter: int = 200) -> float:
    return