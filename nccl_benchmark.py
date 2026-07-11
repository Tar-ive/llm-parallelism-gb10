"""NCCL collective benchmark for the GB10, feeding the NCCL Inspector.

Runs the collectives that dominate real parallel training (all_reduce = DDP
gradient sync, all_gather + reduce_scatter = FSDP/ZeRO shard traffic,
broadcast = weight init, send/recv = pipeline parallelism) across a sweep of
message sizes, with 2 ranks timesharing the single GB10.

Launch through the inspector wrapper so metrics land in Prometheus/Grafana:

    /root/vehicle_monitor/metrics/monitoring/nccl_inspector/run_with_inspector.sh \
        torchrun --nproc_per_node=2 nccl_benchmark.py --seconds-per-phase 45

Note: with both ranks on one die, "bandwidth" here is device-memory copy
speed through SM timeslicing, not a link speed - useful as instrumentation
practice and as a lower bound, not as a fabric benchmark.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

SIZES_MB = [1, 16, 128]


def run_phase(name, fn, seconds, rank):
    """Run fn in lockstep on all ranks until `seconds` elapse on rank 0.

    Every rank must issue the exact same sequence of collectives or NCCL
    deadlocks, so only rank 0 watches the clock and broadcasts a stop flag
    at the end of each chunk (that broadcast is itself a collective all
    ranks agree on).
    """
    stop = torch.zeros(1, device="cuda")
    t0 = time.time()
    n = 0
    while True:
        for _ in range(10):
            fn()
        n += 10
        torch.cuda.synchronize()
        if rank == 0 and time.time() - t0 >= seconds:
            stop.fill_(1)
        dist.broadcast(stop, 0)
        if stop.item():
            break
    if rank == 0:
        print(f"  {name:<28} {n:6d} iters in {time.time()-t0:4.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds-per-phase", type=float, default=45,
                    help="inspector prom mode flushes every 30s; keep phases > 30s")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(0)

    if rank == 0:
        print(f"NCCL benchmark: world={world}, sizes={SIZES_MB} MB, "
              f"{args.seconds_per_phase:.0f}s per phase", flush=True)

    for mb in SIZES_MB:
        n = mb * 1024 * 1024 // 4
        x = torch.ones(n, device="cuda")
        gather_out = [torch.empty(n, device="cuda") for _ in range(world)]
        scatter_out = torch.empty(n // world, device="cuda")
        scatter_in = list(x.chunk(world))

        run_phase(f"all_reduce {mb}MB", lambda: dist.all_reduce(x), args.seconds_per_phase, rank)
        run_phase(f"all_gather {mb}MB", lambda: dist.all_gather(gather_out, x), args.seconds_per_phase, rank)
        run_phase(f"reduce_scatter {mb}MB", lambda: dist.reduce_scatter(scatter_out, scatter_in), args.seconds_per_phase, rank)
        run_phase(f"broadcast {mb}MB", lambda: dist.broadcast(x, 0), args.seconds_per_phase, rank)

        def sendrecv():
            if rank == 0:
                dist.send(x, 1)
                dist.recv(x, 1)
            else:
                dist.recv(x, 0)
                dist.send(x, 0)

        if world == 2:
            run_phase(f"send/recv {mb}MB", sendrecv, args.seconds_per_phase, rank)

    if rank == 0:
        print("benchmark complete", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
