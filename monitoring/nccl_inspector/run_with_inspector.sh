#!/usr/bin/env bash
# Wrap any NCCL workload so the NCCL Inspector profiler plugin streams
# per-collective metrics into the Prometheus stack.
#
#   ./run_with_inspector.sh torchrun --nproc_per_node=2 train.py ...
#   ./run_with_inspector.sh /root/nccl-tests/build/all_reduce_perf -b 8K -e 64M -f 2 -g 1
#
# Flow: plugin writes .prom files -> node-exporter textfile collector (:9100)
#       -> Prometheus (:9090) -> Grafana "NCCL Inspector" dashboard (:3002).
#
# Notes for this DGX Spark (GB10):
# - The plugin was built from NCCL master at
#   /root/nccl-src/plugins/profiler/inspector/libnccl-profiler-inspector.so
# - PyTorch bundles NCCL 2.28 whose profiler interface predates the plugin;
#   we LD_PRELOAD the system NCCL 2.30 (apt libnccl2) which supports it.
# - NCCL refuses 2 ranks on one GPU unless NCCL_MULTI_RANK_GPU_ENABLE=1.
#   That flag exists in NCCL >= 2.30 and makes single-box multi-rank
#   experiments (and their collectives) observable.
# - Prometheus dump mode enforces a minimum 30s flush interval, so short
#   runs (<30s) produce nothing. Either run longer or unset
#   NCCL_INSPECTOR_PROM_DUMP to get per-collective JSON logs instead.

set -e

INSPECTOR_SO=/root/nccl-src/plugins/profiler/inspector/libnccl-profiler-inspector.so
TEXTFILE_DIR=/root/vehicle_monitor/metrics/monitoring/nccl_inspector/textfile
SYSTEM_NCCL=/usr/lib/aarch64-linux-gnu/libnccl.so.2

export NCCL_PROFILER_PLUGIN="$INSPECTOR_SO"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_PROM_DUMP=${NCCL_INSPECTOR_PROM_DUMP:-1}
export NCCL_INSPECTOR_DUMP_DIR="${NCCL_INSPECTOR_DUMP_DIR:-$TEXTFILE_DIR}"
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=${NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS:-30000000}
export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=${NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING:-0}
export NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=${NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES:-0}
export NCCL_MULTI_RANK_GPU_ENABLE=1
export LD_PRELOAD="${SYSTEM_NCCL}${LD_PRELOAD:+:$LD_PRELOAD}"

mkdir -p "$NCCL_INSPECTOR_DUMP_DIR"
echo "[inspector] plugin=$NCCL_PROFILER_PLUGIN"
echo "[inspector] dumping to $NCCL_INSPECTOR_DUMP_DIR (prom=$NCCL_INSPECTOR_PROM_DUMP, every $((NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS/1000000))s)"

exec "$@"
