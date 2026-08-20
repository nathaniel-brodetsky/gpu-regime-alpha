import cupy as cp


def _diagonal_line_lengths(recurrence_matrix: cp.ndarray, min_length: int = 2) -> cp.ndarray:
    n = recurrence_matrix.shape[0]
    lengths = []

    for offset in range(1, n):
        diagonal = cp.diagonal(recurrence_matrix, offset=offset)
        diagonal_np = cp.asnumpy(diagonal)

        run_length = 0
        for val in diagonal_np:
            if val == 1:
                run_length += 1
            else:
                if run_length >= min_length:
                    lengths.append(run_length)
                run_length = 0
        if run_length >= min_length:
            lengths.append(run_length)

    return cp.array(lengths, dtype=cp.float64) if lengths else cp.array([], dtype=cp.float64)


def _vertical_line_lengths(recurrence_matrix: cp.ndarray, min_length: int = 2) -> cp.ndarray:
    n = recurrence_matrix.shape[0]
    matrix_np = cp.asnumpy(recurrence_matrix)
    lengths = []

    for col in range(n):
        run_length = 0
        for row in range(n):
            if matrix_np[row, col] == 1:
                run_length += 1
            else:
                if run_length >= min_length:
                    lengths.append(run_length)
                run_length = 0
        if run_length >= min_length:
            lengths.append(run_length)

    return cp.array(lengths, dtype=cp.float64) if lengths else cp.array([], dtype=cp.float64)


def determinism(recurrence_matrix: cp.ndarray, min_length: int = 2) -> float:
    diag_lengths = _diagonal_line_lengths(recurrence_matrix, min_length)
    total_recurrent = float(cp.sum(recurrence_matrix))

    if total_recurrent == 0:
        return 0.0

    points_in_diagonals = float(cp.sum(diag_lengths))
    return points_in_diagonals / total_recurrent


def laminarity(recurrence_matrix: cp.ndarray, min_length: int = 2) -> float:
    vert_lengths = _vertical_line_lengths(recurrence_matrix, min_length)
    total_recurrent = float(cp.sum(recurrence_matrix))

    if total_recurrent == 0:
        return 0.0

    points_in_verticals = float(cp.sum(vert_lengths))
    return points_in_verticals / total_recurrent


def average_diagonal_line_length(recurrence_matrix: cp.ndarray, min_length: int = 2) -> float:
    diag_lengths = _diagonal_line_lengths(recurrence_matrix, min_length)

    if diag_lengths.shape[0] == 0:
        return 0.0

    return float(cp.mean(diag_lengths))


def compute_rqa_summary(recurrence_matrix: cp.ndarray, min_length: int = 2) -> dict:
    n = recurrence_matrix.shape[0]
    total_points = n * n
    recurrence_rate = float(cp.sum(recurrence_matrix)) / total_points

    return {
        "recurrence_rate": recurrence_rate,
        "determinism": determinism(recurrence_matrix, min_length),
        "laminarity": laminarity(recurrence_matrix, min_length),
        "avg_diagonal_length": average_diagonal_line_length(recurrence_matrix, min_length),
    }