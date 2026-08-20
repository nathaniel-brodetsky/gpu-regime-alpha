# GPU vs CPU Pipeline Benchmark Report

Hardware: NVIDIA L40S (48GB VRAM) vs multi-core CPU baseline

| n_ticks | CPU total (s) | GPU total (s) | Speedup |
|---|---|---|---|
| 5,000 | 26.48 | 1.76 | 15.0x |
| 100,000 | 190.98 | 5.13 | 37.2x |
| 500,000 | 1024.64 | 33.31 | 30.8x |

## Per-stage breakdown (100K ticks)

| Stage | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|
| feature_engineering | 111.94 | 0.90 | 124.9x |
| umap_reduction | 75.12 | 0.23 | 329.5x |
| hdbscan_clustering | 3.60 | 2.96 | 1.2x |
| xgboost_train | 0.30 | 0.49 | 0.6x |