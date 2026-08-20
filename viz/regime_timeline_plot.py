import cudf
import cupy as cp
import plotly.graph_objects as go


def build_regime_timeline_figure(timestamps: cp.ndarray, prices: cp.ndarray, cluster_labels: cp.ndarray, title: str = "Price with Regime Overlay") -> go.Figure:
    timestamps_host = cp.asnumpy(timestamps)
    prices_host = cp.asnumpy(prices)
    labels_host = cp.asnumpy(cluster_labels)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps_host,
        y=prices_host,
        mode="lines",
        name="Mid Price",
        line=dict(color="rgba(100,100,100,0.4)", width=1),
    ))

    unique_labels = sorted(set(labels_host.tolist()))

    for label in unique_labels:
        mask = labels_host == label
        if not mask.any():
            continue

        name = "Noise" if label == -1 else f"Regime {label}"
        marker_size = 2 if label == -1 else 4

        fig.add_trace(go.Scatter(
            x=timestamps_host[mask],
            y=prices_host[mask],
            mode="markers",
            name=name,
            marker=dict(size=marker_size),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Mid Price",
        legend=dict(itemsizing="constant"),
    )

    return fig


def save_regime_timeline_html(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path)