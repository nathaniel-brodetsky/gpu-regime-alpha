import subprocess
import threading
import time
import csv


class GPUUtilizationLogger:
    def __init__(self, output_path: str, interval: float = 0.2):
        self.output_path = output_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._rows = []

    def _poll_loop(self):
        while not self._stop_event.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split(",")
                self._rows.append([v.strip() for v in line])
            time.sleep(self.interval)

    def start(self):
        self._stop_event.clear()
        self._rows = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if not self._rows:
            return {"n_samples": 0}

        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_util_pct", "mem_util_pct", "mem_used_mib", "mem_total_mib"])
            writer.writerows(self._rows)

        gpu_utils = [float(row[1]) for row in self._rows]
        mean_util = sum(gpu_utils) / len(gpu_utils)
        max_util = max(gpu_utils)
        pct_above_80 = sum(1 for u in gpu_utils if u >= 80) / len(gpu_utils) * 100

        summary = {
            "n_samples": len(self._rows),
            "mean_gpu_util_pct": round(mean_util, 1),
            "max_gpu_util_pct": max_util,
            "pct_time_above_80pct_util": round(pct_above_80, 1),
        }

        return summary