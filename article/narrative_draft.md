# Stop Bottlenecking Your Quants: End-to-End Market Regime Classification Entirely in VRAM

## The problem

Markets change regimes — flat, trending, toxic order flow, margin-call cascades — and a
single static model trades all of them the same way. That's a losing strategy. We built
a pipeline that discovers these regimes directly from raw order book data, using
mathematics far beyond simple correlation: chaos theory, algebraic topology, random
matrix theory, and optimal transport, all computed end-to-end on a single GPU without a
single round-trip to host memory.

## What's actually running

Raw L2 order book ticks go in. Twenty feature-engineering modules run on them: order
flow imbalance, realized volatility and higher moments, tail-index estimation, VPIN,
Takens delay-coordinate embedding, recurrence quantification, rolling finite-time
Lyapunov exponents, permutation entropy, Hurst/DFA, persistent homology via
Vietoris-Rips complexes, debiased Sinkhorn divergence between persistence diagrams,
rolling correlation spectra, Marchenko-Pastur deviation, eigenvector stability, and
eigenvalue entropy. All of it feeds into UMAP dimensionality reduction, HDBSCAN
clustering to discover regimes unsupervised, cuVS CAGRA for microsecond nearest-neighbor
regime matching on live ticks, and finally an XGBoost model conditioned on the
discovered regime as a categorical feature.

Every one of those twenty modules lives in CuPy/cuDF. Data moves from ingestion to
final prediction without ever touching host memory, except at one deliberate,
documented boundary — more on that below.

## The honest part: what broke, and what we learned

**Ripser++ doesn't build on modern CUDA.** The reference GPU-native Vietoris-Rips
implementation hasn't been maintained since ~2021. It fails to compile against CUDA
12.6's Thrust API — `thrust::sort` calls that predate a namespace restructuring nine
call sites deep in the library. We didn't patch someone else's abandoned CUDA code; we
switched to `giotto-ph`, a well-maintained CPU-multicore Vietoris-Rips implementation.
This is the single point in the entire pipeline where data leaves VRAM — one
`cp.asnumpy()` call before persistence diagram computation, and the diagrams (typically
hundreds to low thousands of points) come straight back into CuPy for everything
downstream. We're not hiding this trade-off. We're stating it, because an honest
architecture diagram is worth more than a fake "100% VRAM-resident" claim that breaks
under scrutiny.

**Sinkhorn distance needed debiasing.** A naive entropic-regularized Wasserstein
distance between identical persistence diagrams doesn't return zero — it's a known
bias in entropic OT. We caught this with a simple sanity check (self-distance should be
~0) and fixed it with the standard Sinkhorn divergence correction:
`SD(a,b) = OT_reg(a,b) - 0.5·OT_reg(a,a) - 0.5·OT_reg(b,b)`. Self-distance dropped to
exactly 0.0 — machine precision, not an approximation.

**Three Python loops were silently eating 97% of runtime.** Rolling permutation
entropy, rolling Hurst/DFA, and rolling correlation matrices were each implemented as a
Python `for` loop over windows — mathematically correct, individually validated, and
catastrophically slow at scale. On 500K ticks, this showed up as a 51-second
`feature_engineering` stage that should have taken a fraction of a second. We profiled,
found the exact culprits, and re-implemented each as a fully vectorized batch operation
(stride-tricks for the entropy/DFA modules, batched `einsum` for correlation matrices).
Results: permutation entropy over 1000x faster, Hurst/DFA over 330x faster, rolling
correlation over 1240x faster — the single largest win in the project. This is the part
of the story that matters most: not that the code worked on the first try (it didn't),
but that we could measure exactly where it was slow and fix it with GPU-native
vectorization instead of throwing more hardware at a Python loop.

## The numbers

Benchmarked on a single NVIDIA L40S (48GB VRAM) against a CPU baseline running the
identical mathematics on pandas/NumPy/scikit-learn-family libraries.

| n_ticks | CPU total (s) | GPU total (s) | Speedup |
|---|---|---|---|
| 5,000 | 26.48 | 1.76 | 15.0x |
| 100,000 | 190.98 | 5.13 | 37.2x |
| 500,000 | *(pending)* | 33.31 | — |

The speedup isn't flat — it grows with scale, which is the honest signature of a
GPU-bound workload amortizing fixed overhead. And it isn't uniform across stages:

| Stage (100K ticks) | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|
| feature_engineering | 111.94 | 0.90 | 124.9x |
| umap_reduction | 75.12 | 0.23 | 329.5x |
| hdbscan_clustering | 3.60 | 2.96 | 1.2x |
| xgboost_train | 0.30 | 0.49 | 0.6x |

UMAP reduction is where the GPU dominates hardest — 330x. HDBSCAN barely benefits at
this scale, and XGBoost GPU-Hist is actually *slower* than CPU here, because device
context initialization overhead doesn't amortize on a training set this small. We're
reporting this because it's true and because it's useful: know where your GPU budget
actually pays off, not just the headline number.

We also logged real GPU utilization during a full 500K-tick run rather than asserting
it: mean 66.1%, peak 100%, 60% of samples above 80% utilization — a realistic profile
for a mixed workload spanning dense linear algebra, tree-based boosting, and
graph-based clustering, not a synthetic single-kernel benchmark.

## Regime discovery in practice

HDBSCAN, run unsupervised on the UMAP embedding of these twenty engineered features,
finds a dominant "normal" regime and a set of smaller, well-separated clusters
corresponding to the volatility-shock windows built into the data generator. The 3D
UMAP projection shows this cleanly: one large mass, several tight satellite clusters,
each maps to a specific temporal window when we overlay regime labels on the price
series.

## What this is, and isn't

This is a working, benchmarked, honestly-documented architecture — not a trading
strategy. On purely synthetic Gaussian random-walk data, the conditioned XGBoost model
shows no improvement over a mean predictor, which is exactly the correct outcome: there
is no real alpha signal in pure noise, and we're not going to manufacture one by tuning
the generator until the number looks good. What we're demonstrating is that this
entire mathematical stack — chaos theory, topology, random matrix theory, optimal
transport, unsupervised regime discovery, and gradient boosting — runs end-to-end on a
single GPU, in VRAM, at a speed that turns what used to be an overnight batch job into
something you can iterate on interactively.