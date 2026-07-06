"""Background GPU telemetry sampler for the GB10 (DGX Spark).

Samples NVML counters on a thread and writes a CSV so each training
experiment gets a synchronized utilization/power/memory trace that you can
line up with what you see in nvtop.

Usage as a library:
    sampler = GPUSampler("results/exp1_gpu.csv", tag="exp1")
    sampler.start()
    ... train ...
    sampler.stop()

Usage standalone (log until Ctrl-C):
    python gpu_monitor.py --out results/manual_gpu.csv --tag manual
"""

import argparse
import csv
import threading
import time

import pynvml


FIELDS = [
    "ts",
    "tag",
    "util_gpu_pct",       # SM utilization as reported by NVML (what nvtop shows as GPU%)
    "util_mem_pct",       # memory controller utilization
    "mem_used_mib",
    "power_w",
    "sm_clock_mhz",
    "temp_c",
]


class GPUSampler:
    def __init__(self, out_csv: str, tag: str = "", interval_s: float = 0.2, device_index: int = 0):
        self.out_csv = out_csv
        self.tag = tag
        self.interval_s = interval_s
        self.device_index = device_index
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def _run(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        with open(self.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                row = {"ts": f"{time.time():.3f}", "tag": self.tag}
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    row["util_gpu_pct"] = util.gpu
                    row["util_mem_pct"] = util.memory
                except pynvml.NVMLError:
                    row["util_gpu_pct"] = row["util_mem_pct"] = ""
                try:
                    # GB10 has unified memory; NVML may not report per-process FB.
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    row["mem_used_mib"] = mem.used // (1024 * 1024)
                except pynvml.NVMLError:
                    row["mem_used_mib"] = ""
                try:
                    row["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    row["power_w"] = ""
                try:
                    row["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                except pynvml.NVMLError:
                    row["sm_clock_mhz"] = ""
                try:
                    row["temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    row["temp_c"] = ""
                writer.writerow(row)
                f.flush()
                self.samples.append(row)
                self._stop.wait(self.interval_s)
        pynvml.nvmlShutdown()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        """Mean/max of numeric columns over collected samples."""
        out = {}
        for key in ("util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w", "sm_clock_mhz"):
            vals = [float(s[key]) for s in self.samples if s.get(key) not in ("", None)]
            if vals:
                out[f"{key}_mean"] = sum(vals) / len(vals)
                out[f"{key}_max"] = max(vals)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu_trace.csv")
    ap.add_argument("--tag", default="manual")
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args()
    sampler = GPUSampler(args.out, tag=args.tag, interval_s=args.interval).start()
    print(f"logging to {args.out} every {args.interval}s, Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sampler.stop()
        print("\n", sampler.summary())


if __name__ == "__main__":
    main()
