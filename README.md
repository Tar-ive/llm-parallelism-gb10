# LLM training parallelism on a DGX Spark (1x GB10) — results & interpretation

Experiment: SFT-train a causal LM (`Qwen/Qwen2.5-0.5B-Instruct`, 494M params)
on `nvidia/Nemotron-Cascade-2-SFT-Data` — the SFT set behind
`nvidia/Nemotron-Cascade-2-30B-A3B`, whose FP8 quant is
`chankhavu/Nemotron-Cascade-2-30B-A3B-FP8` — under 10 different
parallelism/optimization configurations, with NVML telemetry sampled every
200ms (`results/<tag>_gpu.csv`) so each configuration can be correlated with
what nvtop shows. All runs: seq_len 1024, identical packed token blocks,
AdamW, measurements exclude 3 warmup steps.

Note: the linked FP8 checkpoint itself is an *inference* artifact
(compressed-tensors FP8 for vLLM/SGLang) — you can't backprop through it,
so training uses its open dataset with a small dense model.

## Measured results (Jul 6, 2026)

```
tag                  prec  bs  ga  ws  step_ms  tok/s  TFLOPS  MFU%   GPU%avg  W_avg  W_max
01_fp32_bs1          fp32  1   1   1   627.4    1632   4.84    15.61  72.0     48.2   66.5
02_tf32_bs1          tf32  1   1   1   504.2    2031   6.02    9.71   67.9     33.0   42.2
03_bf16_bs1          bf16  1   1   1   348.7    2937   8.71    6.96   60.6     27.6   37.9
04_bf16_bs8          bf16  8   1   1   1367.9   5989   17.75   14.20  82.2     42.2   51.6
05_bf16_bs8_ga4      bf16  8   4   1   4906.5   6678   19.80   15.84  86.4     46.7   56.6
06_bf16_bs8_ckpt     bf16  8   1   1   1441.7   5682   22.46*  17.97  83.1     45.9   54.8
07_bf16_bs8_compile  bf16  8   1   1   833.0    9834   29.15   23.32  51.5     39.1   69.4
08_ddp2_bf16_bs4     bf16  4   1   2   2407.4   3403   10.09   8.07   71.0     34.0   48.4
09_fsdp_bf16_bs8     bf16  8   1   1   1444.8   5670   16.81   13.45  81.6     41.3   51.9
10_fsdp_offload      bf16  8   1   1   1970.0   4158   12.33   9.86   56.8     29.5   50.3
```

TFLOPS = `(6 or 8*)·params·tokens/s`; MFU% is against assumed GB10 dense
peaks (fp32 31 / tf32 62 / bf16 125 TFLOPS — edit `PEAK_TFLOPS` in
`train_experiment.py` if you have vendor-confirmed numbers, MFU scales
linearly). `*` exp 06 uses the 8N factor because checkpointing re-runs the
forward pass: hardware FLOPs, of which 2N are recompute, not learning.

## Interpretation

### 1. Precision ladder (01 → 02 → 03): the biggest single lever

fp32 → tf32 → bf16 at identical work: **627 → 504 → 349 ms/step (1.8x)**.
This is tensor cores engaging. Two nvtop lessons here:

- **GPU% went *down* as speed went up** (72% → 61%): with bs=1 each step
  finishes faster, so fixed per-step overhead (Python, optimizer, launches)
  is a larger share of the timeline. SM-busy% measures "some kernel was
  resident", not work done.
- **Power went *down* too** (48W → 28W avg): at bs=1 the model is
  latency/bandwidth-bound; bf16 halves bytes moved, so the chip idles more
  per step. Don't read low watts as "bf16 is inefficient" — read it as "bs=1
  can't feed the tensor cores". Note MFU% *appears* to fall (15.6 → 7.0)
  only because the denominator (peak) quadruples; absolute TFLOPS rose 1.8x.

### 2. Batch scaling (03 → 04): the real single-GPU "parallelism"

bs 1 → 8: **8.7 → 17.8 TFLOPS (2x)**, GPU% 61 → 82, power 28 → 42W. Bigger
matmuls raise arithmetic intensity — this is what tensor parallelism does
*across* GPUs, achieved *within* one. All three signals (TFLOPS, GPU%,
watts) rising together is the healthy saturation signature in nvtop.

### 3. Gradient accumulation (05): data parallelism's math without the GPUs

ga=4 gives a 32k-token global batch (like 4-way DP) at flat memory. Slightly
*better* than 04 per token (19.8 vs 17.8 TFLOPS) because optimizer cost
amortizes over 4 micro-batches. In nvtop it's indistinguishable from 04 —
one long hump — which is the point: on one GPU, this is how you emulate DP's
statistical benefit at zero cost.

### 4. Gradient checkpointing (06): trade compute for memory

Throughput fell only 5% (5989 → 5682 tok/s) while hardware FLOPs rose to
22.5 TFLOPS — the recompute is nearly free because it's dense matmul work
the chip is good at. The memory it frees (not captured here; NVML can't read
the GB10's unified memory, later runs record `torch.cuda.max_memory_allocated`
as `mem_GiB`) is what you spend on bigger batches → higher *useful* MFU.

### 5. torch.compile (07): the overall winner

**9834 tok/s, 29.2 TFLOPS, 23.3% MFU — 1.64x over eager (04)** — while
GPU%avg *dropped* to 51.5 and peak power hit the run's maximum (69.4W).
Fused kernels finish the same math in fewer, denser launches: less time
"busy", more watts while running. The starkest proof in this table that
**nvtop's GPU% is not a throughput meter — watch power and measure TFLOPS.**

### 6. DDP, 2 ranks sharing the one GB10 (08): parallelism without parallel hardware

Same global batch as 04, **43% slower** (3403 vs 5989 tok/s). Two processes
timeslice one GPU and allreduce gradients over gloo on CPU every step. In
nvtop: two python ranks in the process list, jittery utilization, decent
GPU% (71) yet the lowest bf16 TFLOPS of any bs-8-equivalent run —
contention looks "busy". **Data parallelism only pays with more physical
GPUs** (e.g., a second Spark over its ConnectX 200GbE link).

### 7. FSDP and CPU offload (09, 10): capacity, not speed

FSDP at world=1 costs ~5% vs eager (16.8 vs 17.8 TFLOPS) — pure
flatten/gather wrapper overhead, sharding is degenerate with one rank.
Adding CPU offload costs only ~30% total (4158 tok/s) — remarkably cheap,
because "CPU memory" on Spark sits behind NVLink-C2C unified memory, not a
PCIe bus. This is the machine's actual superpower: **capacity**. ZeRO-style
offload here buys you models several times larger than GPU-resident
training would allow, at a modest slowdown that would be brutal on a
discrete-GPU workstation.

### Techniques that need >1 GPU (not measurable here)

- **Tensor parallelism** — splits individual matmuls; single-GPU stand-in is
  exp 04's batch scaling.
- **Pipeline parallelism** — splits layers; bubbles would appear as periodic
  GPU% gaps; exp 10's offload traffic is the closest analog.
- **Expert parallelism** — the linked 30B-A3B model is MoE (30B stored, ~3B
  active per token): at scale experts scatter across GPUs; on one Spark MoE
  just means "big memory, small compute" — exactly why its FP8 quant targets
  this class of box for *inference*.
- **Context/sequence parallelism** — splits the sequence; single-GPU analog
  is raising `--seq-len`.

## Bottom line: does parallelism get the most out of this GPU?

On a single-GPU DGX Spark, **no distributed-training technique makes
training faster — two of them (DDP-on-one-die, offload) make it slower.**
What actually raised utilization, in order of measured impact:

1. **bf16 tensor cores** (1.8x) — never train fp32 on this chip
2. **torch.compile** (1.64x on top)
3. **Batch size until memory is full** (2x from bs1→8, more headroom left)
4. **Grad accumulation** for large-batch statistics, free
5. **Checkpointing + offload** to fit *bigger models*, spending the cheap
   unified-memory bandwidth — the Spark-appropriate use of "parallelism"
   machinery

Best observed: 29 TFLOPS ≈ 23% of assumed bf16 peak at 0.5B scale, power
never past ~70W — the chip was never compute-saturated; a larger model
and/or bigger batches with compile+checkpointing is the path to higher MFU.
And nvtop's core lesson across all 10 runs: **GPU% tells you the GPU is
occupied, power tells you it's working, only tokens/s·FLOPs tells you it's
working *usefully*.**

## Reproduce

```bash
# terminal 1
nvtop
# terminal 2
cd /root/vehicle_monitor/metrics/llm_parallelism
bash run_all.sh          # full matrix (~15 min), 5s idle valleys between runs
python summarize.py      # re-print the comparison table
```

Single run: `python train_experiment.py --tag x --precision bf16 --batch-size 8 ...`
Knobs: `--model` (any cached HF causal LM), `--precision fp32|tf32|bf16`,
`--batch-size`, `--seq-len`, `--grad-accum`, `--steps`,
`--gradient-checkpointing`, `--compile`, `--mode single|ddp|fsdp`,
`--cpu-offload`, `--peak-tflops`. `MODEL=Qwen/Qwen2.5-3B-Instruct bash
run_all.sh` swaps the model.

Files: `train_experiment.py` (training loop), `gpu_monitor.py` (NVML
sampler), `run_all.sh` (matrix), `summarize.py` (table),
`results/*_gpu.csv` (per-run 200ms traces), `results/summary.jsonl` (all
run summaries), `run_all.log` (full console log), `data_cache/` (packed
token blocks so every run trains on identical data).

Environment: DGX Spark, 1x NVIDIA GB10 (unified 119GB), driver 580.95.05,
CUDA 13.0, torch 2.11.0+cu130, transformers 4.57.3. The GPU was shared with
resident services (tritonserver + CV pipelines, ~22GB) — idle baseline in
the traces is nonzero. GB10 NVML quirks: memory usage and memory-controller
utilization are not reported (unified memory).

HF auth: token already configured globally at `~/.cache/huggingface/token`
(works from any directory; do not commit token files into the repo).
