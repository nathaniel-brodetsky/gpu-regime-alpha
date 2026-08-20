import cupy as cp
import numpy as np
import plotly.graph_objects as go


def build_persistence_diagram_figure(diagram: cp.ndarray, title: str = "Persistence Diagram (H1)") -> go.Figure:
    diagram_host = cp.asnumpy(diagram) if isinstance(diagram, cp.ndarray) else np.asarray(diagram)

    if diagram_host.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no features found)")
        return fig

    births = diagram_host[:, 0]
    deaths = diagram_host[:, 1]
    lifetimes = deaths - births

    max_val = float(max(births.max(), deaths.max())) * 1.1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Diagonal (birth = death)",
        line=dict(color="gray", dash="dash", width=1),
    ))

    fig.add_trace(go.Scatter(
        x=births,
        y=deaths,
        mode="markers",
        name="Topological Features",
        marker=dict(
            size=6,
            color=lifetimes,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Lifetime", y=0.3, len=0.5),
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Birth",
        yaxis_title="Death",
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val]),
    )

    return fig


def save_persistence_diagram_html(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path)