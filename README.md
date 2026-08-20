# GPU Regime Alpha

Zero-copy GPU pipeline for unsupervised market regime classification and
regime-conditioned alpha prediction, built entirely on NVIDIA RAPIDS
(cuDF, cuML, CuPy, cuVS) and XGBoost GPU-Hist.

Raw limit order book ticks go in. Twenty GPU-native feature engineering
modules — spanning market microstructure, nonlinear dynamics, algebraic
topology, random matrix theory, and information theory — run entirely in
VRAM. UMAP and HDBSCAN discover market regimes unsupervised. cuVS CAGRA
matches live ticks to historical regimes in microseconds. XGBoost predicts
forward returns conditioned on the discovered regime.

Full writeup: [`article/narrative_draft.md`](article/narrative_draft.md)

## Benchmark headline

| n_ticks | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|
| 5,000 | 26.48 | 1.76 | 15.0x |
| 100,000 | 190.98 | 5.13 | 37.2x |
| 500,000 | 1024.64 | 33.31 | 30.8x |

Hardware: single NVIDIA L40S (48GB VRAM) vs multi-core CPU baseline running
the identical mathematics on pandas/NumPy/scikit-learn-family libraries.
Full per-stage breakdown and methodology in the narrative doc above.

## Architecture

```
raw L2 LOB ticks (cuDF, VRAM)
        │
        ▼
┌───────────────────────────────────────────┐
│ features/microstructure/  OFI, spread,     │
│   micro-price, realized vol, tail index,   │
│   VPIN                                     │
├───────────────────────────────────────────┤
│ features/embedding/       Takens embedding,│
│   false nearest neighbors                  │
├───────────────────────────────────────────┤
│ features/chaos/           recurrence plots,│
│   RQA, finite-time Lyapunov exponent       │
├───────────────────────────────────────────┤
│ features/entropy/         permutation      │
│   entropy, Hurst/DFA                       │
├───────────────────────────────────────────┤
│ features/topology/        Vietoris-Rips →  │
│   persistence diagrams → persistent        │
│   entropy → debiased Sinkhorn divergence   │
├───────────────────────────────────────────┤
│ features/spectral/        rolling          │
│   correlation → RMT / Marchenko-Pastur →   │
│   eigenvector stability, eigenvalue entropy│
└───────────────────────────────────────────┘
        │
        ▼
   fusion/  (tick-level + block-level fusion)
        │
        ▼
   reduction/  UMAP → HDBSCAN (unsupervised regime discovery)
        │
        ▼
   retrieval/  cuVS CAGRA (live tick → nearest historical regime)
        │
        ▼
   model/  XGBoost GPU-Hist, regime as categorical feature
```

Data stays GPU-resident (CuPy/cuDF) end-to-end, with one deliberate,
documented exception: Vietoris-Rips persistence diagram computation runs
on CPU via `giotto-ph`, because the reference GPU-native library
(`ripser++`) no longer builds against modern CUDA/Thrust. Details in the
narrative doc.

## Setup

See [`environment.yml`](environment.yml) for the full, tested dependency
list (RAPIDS 25.08, CUDA 12.6, pinned scikit-learn, giotto-ph, plotly).

```bash
conda env create -f environment.yml
conda activate rapids
python -m pip install -r requirements-pip.txt
```

Requires an NVIDIA GPU, compute capability ≥ 7.0, Linux or WSL2. Tested on
RTX 4070 Laptop (8GB, local dev) and NVIDIA L40S (48GB, benchmark runs).

## Quick start

```bash
python -c "
from benchmark.gpu_pipeline_bench import run_full_pipeline_benchmark
timings = run_full_pipeline_benchmark(n_ticks=500_000)
print(timings)
"
```

See [`notebooks/01_full_pipeline_walkthrough.ipynb`](notebooks/01_full_pipeline_walkthrough.ipynb)
for a step-by-step walkthrough with inline visualizations.

## Repository layout

- `data/synthetic/` — synthetic multi-regime LOB tick generator
- `features/` — 20 GPU-native feature engineering modules (see architecture above)
- `fusion/` — combines tick-level and block-level features into training matrices
- `reduction/` — UMAP + HDBSCAN unsupervised regime discovery
- `retrieval/` — cuVS CAGRA real-time nearest-neighbor regime matching
- `model/` — XGBoost GPU-Hist conditioned alpha prediction
- `pipeline/` — DLPack zero-copy bridge utilities
- `benchmark/` — CPU baseline, GPU pipeline benchmark, GPU utilization logger, report generator
- `viz/` — 3D UMAP, regime timeline, persistence diagram visualizations
- `notebooks/` — presentation walkthrough
- `article/` — full technical writeup

## Known limitations

- Vietoris-Rips complex computation runs on CPU (see Architecture above)
- HDBSCAN clustering shows limited GPU speedup at scale (algorithmic property, not an implementation gap)
- XGBoost GPU training has fixed device-context overhead that doesn't amortize on small (<100K row) training sets
- Synthetic data generator produces pure Gaussian random-walk prices; the conditioned XGBoost model correctly shows no predictive edge over a mean baseline on this data, as expected — this is an infrastructure and methodology demonstration, not a trading strategy