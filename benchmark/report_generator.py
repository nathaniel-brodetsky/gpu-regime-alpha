def generate_benchmark_report(results: dict, output_path: str) -> None:
    lines = []
    lines.append("# GPU vs CPU Pipeline Benchmark Report\n")
    lines.append("Hardware: NVIDIA L40S (48GB VRAM) vs multi-core CPU baseline\n")
    lines.append("| n_ticks | CPU total (s) | GPU total (s) | Speedup |")
    lines.append("|---|---|---|---|")

    numeric_keys = [k for k in results.keys() if isinstance(k, int)]
    for n_ticks in sorted(numeric_keys):
        entry = results[n_ticks]
        cpu_t = entry.get("cpu_total")
        gpu_t = entry.get("gpu_total")
        if cpu_t is not None and gpu_t is not None:
            speedup = cpu_t / gpu_t
            lines.append(f"| {n_ticks:,} | {cpu_t:.2f} | {gpu_t:.2f} | {speedup:.1f}x |")
        else:
            lines.append(f"| {n_ticks:,} | {'pending' if cpu_t is None else f'{cpu_t:.2f}'} | {gpu_t:.2f} | - |")

    lines.append("")
    lines.append("## Per-stage breakdown (100K ticks)\n")
    lines.append("| Stage | CPU (s) | GPU (s) | Speedup |")
    lines.append("|---|---|---|---|")

    stage_data = results.get("stage_breakdown_100k", {})
    for stage, (cpu_s, gpu_s) in stage_data.items():
        speedup = cpu_s / gpu_s if gpu_s > 0 else float("inf")
        lines.append(f"| {stage} | {cpu_s:.2f} | {gpu_s:.2f} | {speedup:.1f}x |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


results = {
    5000: {"cpu_total": 26.485, "gpu_total": 1.7636},
    100000: {"cpu_total": 190.981, "gpu_total": 5.1337},
    500000: {"cpu_total": None, "gpu_total": 33.3106},
    "stage_breakdown_100k": {
        "feature_engineering": (111.942, 0.896),
        "umap_reduction": (75.116, 0.228),
        "hdbscan_clustering": (3.595, 2.965),
        "xgboost_train": (0.303, 0.492),
    },
}

if __name__ == "__main__":
    generate_benchmark_report(results, "benchmark_report.md")
    print("report generated")