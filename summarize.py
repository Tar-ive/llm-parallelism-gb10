"""Print results/summary.jsonl as an aligned comparison table."""

import json
from pathlib import Path

COLS = [
    ("tag", 22), ("precision", 9), ("batch_size", 3), ("grad_accum", 3),
    ("world_size", 2), ("step_ms", 8), ("tokens_per_s", 8), ("tflops", 7),
    ("mfu_pct", 7), ("util_gpu_pct_mean", 8), ("power_w_mean", 7),
    ("torch_mem_gib", 9),
]
HDR = ["tag", "prec", "bs", "ga", "ws", "step_ms", "tok/s", "TFLOPS",
       "MFU%", "GPU%avg", "W_avg", "mem_GiB"]


def main():
    path = Path(__file__).parent / "results/summary.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    latest = {r["tag"]: r for r in rows}  # last run per tag wins
    line = "  ".join(h.ljust(w) for h, (_, w) in zip(HDR, COLS))
    print(line)
    print("-" * len(line))
    for r in sorted(latest.values(), key=lambda r: r["tag"]):
        print("  ".join(str(r.get(k, "-")).ljust(w) for k, w in COLS))


if __name__ == "__main__":
    main()
