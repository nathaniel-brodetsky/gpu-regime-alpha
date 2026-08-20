import cudf
import cupy as cp
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def _find_dominant_label(labels_host: np.ndarray) -> int:
    unique, counts = np.unique(labels_host, return_counts=True)
    non_noise_mask = unique != -1
    if not non_noise_mask.any():
        return -1
    dominant_idx = np.argmax(counts[non_noise_mask])
    return int(unique[non_noise_mask][dominant_idx])


def _find_merged_runs(mask: np.ndarray, merge_gap: int) -> list:
    indices = np.where(mask)[0]
    if indices.shape[0] == 0:
        return []

    runs = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev > merge_gap:
            runs.append((start, prev))
            start = idx
        prev = idx

    runs.append((start, prev))
    return runs


def build_regime_timeline_figure(timestamps: cp.ndarray, prices: cp.ndarray, cluster_labels: cp.ndarray, title: str = "Price with Regime Overlay", max_shaded_regimes: int = 6, merge_gap: int = 50, max_shapes_per_regime: int = 15) -> go.Figure:
    timestamps_host = cp.asnumpy(timestamps)
    prices_host = cp.asnumpy(prices)
    labels_host = cp.asnumpy(cluster_labels)

    dominant_label = _find_dominant_label(labels_host)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps_host,
        y=prices_host,
        mode="lines",
        name="Mid Price",
        line=dict(color="rgba(60,60,60,0.6)", width=1),
    ))

    unique_labels = sorted(set(labels_host.tolist()))
    non_dominant_labels = [l for l in unique_labels if l != dominant_label and l != -1]

    label_counts = {l: int((labels_host == l).sum()) for l in non_dominant_labels}
    top_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:max_shaded_regimes]

    colors = px.colors.qualitative.Bold
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(top_labels)}

    for label in top_labels:
        mask = labels_host == label
        runs = _find_merged_runs(mask, merge_gap)

        runs_sorted = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:max_shapes_per_regime]

        color = color_map[label]

        for run_idx, (start, end) in enumerate(runs_sorted):
            x0 = timestamps_host[start]
            x1 = timestamps_host[end]
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=color,
                opacity=0.35,
                line_width=0,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Mid Price",
        showlegend=True,
    )

    for label in top_labels:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color_map[label]),
            name=f"Regime {label}",
        ))

    return fig


def save_regime_timeline_html(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path)