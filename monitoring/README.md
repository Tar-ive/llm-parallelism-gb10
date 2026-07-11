# GB10 training/NCCL monitoring stack (Prometheus + Grafana)

Real-time telemetry for the DGX Spark: GPU (DCGM), NCCL collectives
(NCCL Inspector profiler plugin), and per-step training metrics
(Pushgateway), all scraped by Prometheus and visualized in Grafana.

## Quick start

```bash
cd /root/vehicle_monitor/metrics/monitoring
docker compose up -d
```

| Service | URL | What it is |
|---|---|---|
| Grafana | http://localhost:3002 | dashboards (no login needed) |
| Prometheus | http://localhost:9090 | metrics DB + query UI |
| DCGM exporter | http://localhost:9400/metrics | GPU telemetry |
| node-exporter | http://localhost:9100/metrics | host + NCCL textfile metrics |
| Pushgateway | http://localhost:9091 | receives training metrics |

Port 3002 was chosen because 3000 (LibreChat proxy) and 3080 are taken.

## The three data paths

```
1. GPU:       DCGM exporter ──────────────────────────► Prometheus ► Grafana "GB10 GPU (DCGM)"
2. NCCL:      training/benchmark process
                └─ NCCL Inspector plugin (inside NCCL)
                     └─ writes .prom files to nccl_inspector/textfile/
                          └─ node-exporter textfile collector ► Prometheus ► "NCCL Inspector"
3. Training:  train_experiment.py ─ push per step ─► Pushgateway ► Prometheus ► "LLM Training (per-step)"
```

## Running things

GPU + training metrics need nothing special — DCGM is always on, and
`train_experiment.py` pushes automatically when the stack is up:

```bash
cd /root/vehicle_monitor/metrics/llm_parallelism
python train_experiment.py --tag demo --precision bf16 --batch-size 8 --steps 30
```

NCCL metrics need the workload wrapped so the Inspector plugin loads:

```bash
# any NCCL workload
./nccl_inspector/run_with_inspector.sh torchrun --nproc_per_node=2 your_script.py

# the provided collective benchmark (all_reduce/all_gather/reduce_scatter/broadcast/sendrecv)
./nccl_inspector/run_with_inspector.sh \
  torchrun --nproc_per_node=2 /root/vehicle_monitor/metrics/llm_parallelism/nccl_benchmark.py
```

## GB10-specific gotchas (learned the hard way)

- **PyTorch's bundled NCCL (2.28) is too old** for the Inspector's profiler
  interface (v5). The wrapper LD_PRELOADs the system NCCL 2.30
  (`/usr/lib/aarch64-linux-gnu/libnccl.so.2`, from `apt libnccl2`).
- **Two ranks on one GPU**: NCCL errors with "Duplicate GPU detected"
  unless `NCCL_MULTI_RANK_GPU_ENABLE=1` (the wrapper sets it). PyTorch's
  2.28 doesn't have that flag at all — another reason for the preload.
- **Prometheus dump mode flushes every 30s minimum** (enforced by the
  plugin) and **deletes its .prom files at process exit**. Runs shorter
  than ~30s therefore leave no Prometheus trace. For per-collective
  archaeology use JSON mode instead: `NCCL_INSPECTOR_PROM_DUMP=0`, which
  appends one JSON line per collective to `<dumpdir>/<host>-pid<pid>.log`.
- **Single-rank communicators emit nothing** — NCCL skips real kernels
  when nranks=1, so the inspector has nothing to record. Benchmarks must
  run ≥2 ranks.
- **DCGM profiling metrics (`DCGM_FI_PROF_*`) are unsupported on GB10**
  (no framebuffer fields either — unified memory). We're limited to the
  coarse fields: GPU%/mem-copy%, power, clocks, temperature, energy.
  This makes achieved-TFLOPS from the training loop (path 3) the only
  true utilization measure on this machine.
- **node-exporter's default collectors hang on this box** (a fuse
  "portal" mount blocks statfs, exhausting the 40-request limit). The
  compose file runs a curated allowlist: textfile, cpu, meminfo, loadavg,
  stat.

## Files

```
monitoring/
├── docker-compose.yml            # the whole stack
├── prometheus/prometheus.yml     # scrape configs (5s default, 2s hot paths)
├── dcgm/counters.csv             # DCGM fields that GB10 actually supports
├── grafana/
│   ├── provisioning/             # auto-wired datasource + dashboard loader
│   └── dashboards/
│       ├── gb10-gpu.json         # DCGM + host dashboard
│       ├── nccl-inspector.json   # collective/P2P busbw + exec time
│       └── llm-training.json     # tokens/s, TFLOPS, MFU, loss, memory
└── nccl_inspector/
    ├── run_with_inspector.sh     # wrapper: env vars + LD_PRELOAD
    └── textfile/                 # .prom drop-zone (node-exporter reads this)
```

The inspector plugin itself: built from NCCL master at
`/root/nccl-src/plugins/profiler/inspector/libnccl-profiler-inspector.so`
(`make` in that directory rebuilds it; needs only CUDA, headers vendored).
nccl-tests binaries (all_reduce_perf etc.): `/root/nccl-tests/build/`,
built for sm_121.

NVIDIA's own Grafana template for the inspector
(`/root/nccl-src/plugins/profiler/inspector/grafana/`) assumes a SLURM
cluster + MySQL metadata + multi-region Mimir, so a trimmed local
dashboard (`nccl-inspector.json`) was written instead; same metrics,
same panel semantics.
