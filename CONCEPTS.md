# From one GB10 to distributed training: the conceptual map

This ties together the pieces now running on this machine — the parallelism
experiments (`README.md` here), the NCCL Inspector, and the
Prometheus/Grafana stack (`monitoring/`) — into one mental model.

## 1. Why "parallelism" is really "communication"

Every distributed-training technique is a decision about **what to split
and what to communicate**:

| Technique | What's split | What must be communicated | NCCL collective |
|---|---|---|---|
| Data parallel (DDP) | the batch | gradients, every step | `AllReduce` |
| ZeRO / FSDP | optimizer state, grads, params | params before use, grads after | `AllGather` + `ReduceScatter` |
| Tensor parallel | individual matmuls | activations, twice per layer | `AllReduce` (or `AllGather`/`ReduceScatter`) |
| Pipeline parallel | layers | activations at stage borders | `Send`/`Recv` (P2P) |
| Expert parallel (MoE) | experts | tokens routed to experts | `AllToAll` |

The compute side is embarrassingly parallel; the communication is the whole
game. That's why the tool for understanding distributed training is a tool
that watches NCCL — which is exactly what the Inspector is.

**The single-GPU results in `README.md` are this table's degenerate case:**
grad accumulation gives you DP's math with zero communication; batch scaling
gives you TP's arithmetic-intensity effect with zero communication; and the
2-rank DDP experiment showed what happens when you pay communication cost
without adding hardware (43% slower).

## 2. What NCCL actually is

NCCL ("nickel") is the library every framework (PyTorch, JAX, DeepSpeed,
Megatron) uses to move tensors between GPUs. Key concepts you'll see as
labels in the Grafana NCCL dashboard:

- **Communicator** (`comm_name`, `nranks`, `n_nodes`): a group of ranks
  that talk to each other. Real jobs have several at once — e.g. one DP
  communicator across nodes plus one TP communicator inside each node.
- **Collective** (`collective`): AllReduce, AllGather, ReduceScatter,
  Broadcast, AllToAll — the operations from the table above.
- **Algorithm / protocol** (`algo_proto`, e.g. `RING_LL`, `TREE_SIMPLE`):
  NCCL picks how to route data (ring vs tree) and how to signal completion
  (LL/LL128 = low latency for small messages, Simple = high bandwidth for
  large ones). Watching this label change as message size grows is NCCL's
  autotuner working in real time.
- **algbw vs busbw**: algorithmic bandwidth = message size / time. Bus
  bandwidth corrects for the algorithm's redundancy (ring AllReduce moves
  2(n-1)/n bytes per byte of payload), so busbw is comparable across
  collectives and rank counts. The dashboard plots busbw.

## 3. What the Inspector adds

NCCL ships a **profiler plugin interface**: hand NCCL a `.so` via
`NCCL_PROFILER_PLUGIN` and it calls into the plugin at every collective
start/stop with kernel-level GPU timing. The Inspector plugin
(`/root/nccl-src/plugins/profiler/inspector/`) uses this to compute, per
collective, per communicator: execution time, algbw/busbw, message size,
algorithm — and writes it either as JSON lines (forensics) or as `.prom`
textfiles (live Prometheus, 30s cadence).

This is qualitatively different from DCGM/nvtop: those tell you *the GPU
was busy*; the Inspector tells you *the AllReduce for your gradient sync
ran at 3.6 GB/s busbw with RING_LL on 16MB messages*. When a training run
is slow, this is the layer that tells you whether compute or communication
is the bottleneck, and which collective in which communicator.

## 4. The monitoring philosophy (why three data paths)

The stack watches training at three timescales, because each layer can lie:

1. **DCGM (GPU counters, 1s)** — is the chip busy and drawing power?
   Coarse but always on. On GB10, remember: GPU% is "a kernel was
   resident", not throughput; power is the honest signal (README §5).
2. **NCCL Inspector (per-collective, 30s prom / instant JSON)** — is the
   *communication* healthy? Bandwidth collapse, algorithm flips, and
   straggler ranks show up here first.
3. **Trainer's own metrics (per-step, Pushgateway)** — tokens/s, TFLOPS,
   MFU, loss. The only numbers that measure *useful* work. Everything
   else exists to explain movements in these.

Prometheus is the time-series database that scrapes all three; Grafana is
the pane of glass. The layer-cake habit — check useful work first, then
communication, then raw GPU state — scales unchanged from this single Spark
to a thousand-GPU cluster; only the exporter endpoints multiply.

## 5. What this means for Bittensor

Bittensor miners/validators are, from the hardware's point of view,
inference or fine-tuning workloads — the same knobs measured in README.md
apply directly (bf16 always, compile, batch to fill memory, offload for
capacity). The monitoring stack transfers wholesale:

- Subnet workloads run as processes you now know how to instrument:
  DCGM for the GPU, Pushgateway for per-request/per-step metrics, and the
  NCCL Inspector if a subnet ever does multi-GPU sync.
- The GB10's superpower (119GB unified memory) fits Bittensor's common
  pattern of "big model, modest throughput" — offload-style serving of
  models that wouldn't fit a discrete 24-48GB card.
- What the GB10 won't give you: NVLink-class multi-GPU scaling. If a
  subnet rewards raw throughput, a Spark competes on model size and
  efficiency, not tokens/s.

## 6. Reading the dashboards during a run (cheat sheet)

| Symptom | Meaning |
|---|---|
| TFLOPS high, power high, GPU% whatever | healthy — ignore GPU% |
| GPU% high, TFLOPS low | overhead/contention (the DDP-on-one-die signature) |
| NCCL busbw drops at a specific message size | algorithm/protocol boundary — check `algo_proto` label |
| Step time up, NCCL exec time up, GPU power down | communication-bound: ranks waiting, not computing |
| Step time up, NCCL flat, power up | compute-bound: bigger batch/model than before |
| tokens/s sawtooth with period = checkpoint interval | checkpointing stalls, look at host CPU/disk panels |

## Where everything lives

- Parallelism experiments + results: this directory (`README.md`)
- Monitoring stack + how to run it: `monitoring/README.md`
- NCCL Inspector source + plugin: `/root/nccl-src/plugins/profiler/inspector/`
- nccl-tests (raw collective benchmarks): `/root/nccl-tests/build/`
- Collective benchmark feeding the dashboards: `nccl_benchmark.py` here
