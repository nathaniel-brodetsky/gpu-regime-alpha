import cudf
import cupy as cp
import plotly.graph_objects as go


def build_umap_3d_figure(embedding: cp.ndarray, cluster_labels: cp.ndarray, title: str = "UMAP Regime Clusters") -> go.Figure:
    embedding_host = cp.asnumpy(embedding)
    labels_host = cp.asnumpy(cluster_labels)

    unique_labels = sorted(set(labels_host.tolist()))

    fig = go.Figure()

    for label in unique_labels:
        mask = labels_host == label
        name = "Noise" if label == -1 else f"Regime {label}"
        marker_size = 2 if label == -1 else 3
        opacity = 0.3 if label == -1 else 0.8

        fig.add_trace(go.Scatter3d(
            x=embedding_host[mask, 0],
            y=embedding_host[mask, 1],
            z=embedding_host[mask, 2],
            mode="markers",
            name=name,
            marker=dict(size=marker_size, opacity=opacity),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        legend=dict(itemsizing="constant"),
    )

    return fig


def save_umap_3d_html(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path)