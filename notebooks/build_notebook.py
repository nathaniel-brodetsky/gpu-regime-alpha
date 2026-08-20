import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# GPU Regime Alpha — Full Pipeline Walkthrough

End-to-end zero-copy GPU market regime classification, from raw order book
ticks to regime-conditioned alpha prediction. Every step below runs on a
single NVIDIA GPU via RAPIDS (cuDF/cuML/CuPy) and cuVS.

See `article/narrative_draft.md` for full technical writeup including
honest engineering failures and benchmark methodology."""))

cells.append(nbf.v4.new_markdown_cell("## 1. Generate synthetic multi-regime LOB data"))
cells.append(nbf.v4.new_code_cell("""from data.synthetic.lob_generator import generate_synthetic_lob

df = generate_synthetic_lob(n_ticks=100_000)
df.head()"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Feature engineering — 20 GPU-native modules in VRAM"))
cells.append(nbf.v4.new_code_cell("""from fusion.feature_matrix_builder import build_full_feature_matrix
import time

start = time.perf_counter()
features = build_full_feature_matrix(
    df,
    rolling_window=100,
    tail_window=200,
    entropy_window=100,
    hurst_window=300,
    spectral_window=100,
)
print(f"feature engineering: {time.perf_counter() - start:.3f}s")
features.head()"""))

cells.append(nbf.v4.new_markdown_cell("""Feature set spans market microstructure (OFI, spread, micro-price,
realized volatility, tail index), nonlinear dynamics (permutation entropy,
Hurst/DFA), and spectral/RMT (rolling correlation eigenstructure)."""))

cells.append(nbf.v4.new_markdown_cell("## 3. Topology — Vietoris-Rips persistence on phase-space embedding"))
cells.append(nbf.v4.new_code_cell("""from features.microstructure.realized_vol import compute_log_returns
from features.topology.vietoris_rips import compute_persistence_diagrams_from_series, diagram_to_cupy
from features.topology.wasserstein_pd import sinkhorn_divergence_diagrams

log_returns = compute_log_returns(df).to_cupy()

window_a = log_returns[:500]
window_b = log_returns[500:1000]

dgms_a = compute_persistence_diagrams_from_series(window_a, embedding_dim=3, delay=1, maxdim=1)
dgms_b = compute_persistence_diagrams_from_series(window_b, embedding_dim=3, delay=1, maxdim=1)

h1_a = diagram_to_cupy(dgms_a[1])
h1_b = diagram_to_cupy(dgms_b[1])

shift_score = sinkhorn_divergence_diagrams(h1_a, h1_b)
self_distance = sinkhorn_divergence_diagrams(h1_a, h1_a)

print(f"structural shift score (different windows): {shift_score:.4e}")
print(f"self-distance sanity check (should be ~0): {self_distance:.4e}")"""))

cells.append(nbf.v4.new_code_cell("""from viz.persistence_diagram_plot import build_persistence_diagram_figure

fig = build_persistence_diagram_figure(h1_a, title="Persistence Diagram (H1) — window A")
fig.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 4. UMAP reduction + HDBSCAN unsupervised regime discovery"))
cells.append(nbf.v4.new_code_cell("""from reduction.umap_reduce import fit_umap_reduction
from reduction.hdbscan_cluster import fit_hdbscan_clusters, cluster_summary, attach_cluster_labels

feature_cols = [c for c in features.columns if c != "timestamp"]
embedding, clean_features = fit_umap_reduction(features, feature_cols, n_components=3)

labels, clusterer = fit_hdbscan_clusters(embedding, min_cluster_size=100)
print(cluster_summary(labels))

clustered = attach_cluster_labels(clean_features, labels)"""))

cells.append(nbf.v4.new_code_cell("""from viz.umap_3d_plot import build_umap_3d_figure

fig = build_umap_3d_figure(embedding, labels, title="LOB Regime Clusters")
fig.show()"""))

cells.append(nbf.v4.new_code_cell("""from viz.regime_timeline_plot import build_regime_timeline_figure

timestamps = clean_features["timestamp"].to_cupy()
prices = clean_features["micro_price"].to_cupy()

fig = build_regime_timeline_figure(timestamps, prices, labels, title="Regime Timeline")
fig.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 5. cuVS CAGRA — microsecond live regime matching"))
cells.append(nbf.v4.new_code_cell("""from retrieval.cagra_index_builder import build_cagra_index, query_nearest_regime
import time

index = build_cagra_index(embedding)

query_points = embedding[:100]
start = time.perf_counter()
matched_labels = query_nearest_regime(index, query_points, labels)
elapsed = time.perf_counter() - start

print(f"CAGRA query for 100 points: {elapsed*1e6:.1f} microseconds total")
print(f"self-match rate (sanity check): {float((matched_labels == labels[:100]).mean()):.2%}")"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Regime-conditioned XGBoost alpha prediction"))
cells.append(nbf.v4.new_code_cell("""from model.xgb_conditioned_alpha import build_training_frame, walk_forward_split, train_conditioned_xgb, predict_alpha
import cupy as cp

model_feature_cols = feature_cols
training_frame = build_training_frame(clustered, df, horizon=10, feature_columns=model_feature_cols)
train, test = walk_forward_split(training_frame, train_fraction=0.7)

booster = train_conditioned_xgb(train, model_feature_cols, num_boost_round=100)
predictions = predict_alpha(booster, test, model_feature_cols)
actual = test["target"].to_cupy()

mse = float(cp.mean((predictions - actual) ** 2))
baseline_mse = float(cp.mean((actual - cp.mean(actual)) ** 2))

print(f"test MSE: {mse:.6f}")
print(f"baseline (mean predictor) MSE: {baseline_mse:.6f}")
print(f"improvement over baseline: {1 - mse/baseline_mse:.4%}")"""))

cells.append(nbf.v4.new_markdown_cell("""On purely synthetic Gaussian random-walk data, no improvement over baseline
is the *correct* result — there is no real alpha signal in pure noise. This
notebook demonstrates the infrastructure and mathematics, not a trading
strategy. See `article/narrative_draft.md` for the full discussion."""))

cells.append(nbf.v4.new_markdown_cell("## 7. End-to-end benchmark: CPU vs GPU"))
cells.append(nbf.v4.new_code_cell("""from benchmark.gpu_pipeline_bench import run_full_pipeline_benchmark

print("=== warm-up run (JIT compilation) ===")
run_full_pipeline_benchmark(n_ticks=5_000)

print()
print("=== measured run ===")
timings = run_full_pipeline_benchmark(n_ticks=100_000)
timings"""))

cells.append(nbf.v4.new_markdown_cell("""Full CPU vs GPU comparison across multiple scales (5K / 100K / 500K ticks),
including per-stage breakdown and verified GPU utilization logs, is in
`benchmark_report.md` and `article/narrative_draft.md`."""))

nb['cells'] = cells

with open('notebooks/01_full_pipeline_walkthrough.ipynb', 'w') as f:
    nbf.write(nb, f)

print("notebook written")