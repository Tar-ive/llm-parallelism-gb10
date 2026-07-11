"""One configurable SFT training step-loop to study GPU behavior per technique.

Trains a causal LM on nvidia/Nemotron-Cascade-2-SFT-Data (the dataset behind
Nemotron-Cascade-2-30B-A3B, whose FP8 quant you linked) while gpu_monitor.py
samples NVML. Run different flag combos and watch nvtop side by side.

Launch modes
  single process:  python train_experiment.py [flags]
  DDP (2 ranks sharing the one GB10, gloo allreduce):
                   torchrun --nproc_per_node=2 train_experiment.py --mode ddp [flags]
  FSDP (sharded params + optional CPU offload):
                   torchrun --nproc_per_node=1 train_experiment.py --mode fsdp [flags]

Reported per run: step time, tokens/s, achieved TFLOPS, MFU (% of peak),
plus NVML means/maxes. Summaries append to results/summary.jsonl.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu_monitor import GPUSampler
from prom_metrics import TrainingMetrics

HERE = Path(__file__).parent
DATASET = "nvidia/Nemotron-Cascade-2-SFT-Data"
DATA_FILE = "chat/chat_part_1.jsonl"

# Approximate dense peak TFLOPS for the GB10 in the DGX Spark. Edit if you
# have vendor-confirmed numbers; MFU scales linearly with this constant.
PEAK_TFLOPS = {"fp32": 31.0, "tf32": 62.0, "bf16": 125.0}


def rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg: str):
    if rank0():
        print(msg, flush=True)


def extract_text(row, tokenizer) -> str:
    msgs = row.get("messages") or row.get("conversations")
    if msgs:
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            return "\n".join(str(m.get("content", m)) for m in msgs)
    return str(row)


def build_batches(tokenizer, model_name: str, seq_len: int, n_blocks: int):
    """Stream the SFT data, tokenize, and pack into fixed-length blocks.

    Packing (no padding) keeps every position a real token, so tokens/s and
    MFU comparisons between experiments are apples-to-apples. The tokenized
    blocks are cached to disk so every experiment trains on identical data.
    """
    cache = HERE / f"data_cache/{model_name.split('/')[-1]}_L{seq_len}_N{n_blocks}.pt"
    if cache.exists():
        return torch.load(cache)
    log(f"tokenizing {DATASET}:{DATA_FILE} -> {n_blocks} blocks of {seq_len}")
    stream = load_dataset(DATASET, data_files={"train": DATA_FILE}, split="train", streaming=True)
    ids, blocks = [], []
    for row in stream:
        ids.extend(tokenizer(extract_text(row, tokenizer)).input_ids)
        ids.append(tokenizer.eos_token_id or 0)
        while len(ids) >= seq_len and len(blocks) < n_blocks:
            blocks.append(torch.tensor(ids[:seq_len], dtype=torch.long))
            ids = ids[seq_len:]
        if len(blocks) >= n_blocks:
            break
    data = torch.stack(blocks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="experiment name for logs")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--mode", choices=["single", "ddp", "fsdp"], default="single")
    ap.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32")
    ap.add_argument("--batch-size", type=int, default=1, help="micro-batch per rank")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30, help="optimizer steps")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true", help="FSDP CPU offload")
    ap.add_argument("--data-blocks", type=int, default=512)
    ap.add_argument("--peak-tflops", type=float, default=None)
    args = ap.parse_args()

    distributed = args.mode in ("ddp", "fsdp")
    if distributed:
        # gloo: both DDP ranks share the single GB10; NCCL refuses duplicate
        # devices, gloo allreduces on CPU (comm cost becomes clearly visible).
        dist.init_process_group("gloo")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = build_batches(tokenizer, args.model, args.seq_len, args.data_blocks)
    if distributed:
        dist.barrier()
        data = build_batches(tokenizer, args.model, args.seq_len, args.data_blocks)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.mode == "fsdp":
        model = FSDP(
            model.to(device),
            cpu_offload=CPUOffload(offload_params=True) if args.cpu_offload else None,
            device_id=device,
        )
    else:
        model = model.to(device)
        if args.mode == "ddp":
            model = DDP(model)
    if args.compile:
        model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0

    sampler = None
    prom = None
    if rank0():
        sampler = GPUSampler(str(results_dir / f"{args.tag}_gpu.csv"), tag=args.tag).start()
        prom = TrainingMetrics(args.tag, config={
            "model": args.model, "mode": args.mode, "precision": args.precision,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "grad_accum": args.grad_accum, "world_size": world,
            "grad_ckpt": args.gradient_checkpointing, "compile": args.compile,
        })
        time.sleep(3)  # capture idle baseline (other services share this GPU)

    autocast = torch.autocast("cuda", torch.bfloat16, enabled=args.precision == "bf16")
    # fwd 2N + bwd 4N per token; activation checkpointing re-runs fwd (+2N)
    flops_per_token = (8 if args.gradient_checkpointing else 6) * n_params
    peak = args.peak_tflops or PEAK_TFLOPS[args.precision]

    step_times, cursor = [], rank
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum * world
    log(f"[{args.tag}] params={n_params/1e6:.0f}M world={world} tokens/step={tokens_per_step}")

    for step in range(args.steps + 3):  # 3 warmup steps (cudagraphs/compile/allocator)
        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            batch = data[cursor % len(data)].unsqueeze(0).repeat(args.batch_size, 1).to(device)
            cursor += world
            with autocast:
                loss = model(input_ids=batch, labels=batch).loss / args.grad_accum
            loss.backward()
        optim.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= 3:
            step_times.append(dt)
        if rank0():
            tps = tokens_per_step / dt
            tfl = tps * flops_per_token / 1e12
            if step >= 3:
                prom.record_step(
                    step_ms=dt * 1000, tokens_per_s=tps, tflops=tfl,
                    mfu_pct=tfl / peak * 100,
                    loss=loss.item() * args.grad_accum,
                    mem_gib=torch.cuda.max_memory_allocated() / 2**30)
            if step % 5 == 0:
                log(f"  step {step:3d} loss={loss.item()*args.grad_accum:.3f} "
                    f"{dt*1000:6.0f}ms {tps:7.0f} tok/s "
                    f"{tfl:6.2f} TFLOPS ({tfl/peak*100:4.1f}% peak)")

    if rank0():
        time.sleep(3)
        sampler.stop()
        prom.finish()
        mean_dt = sum(step_times) / len(step_times)
        tps = tokens_per_step / mean_dt
        tflops = tps * flops_per_token / 1e12
        summary = {
            "tag": args.tag, "model": args.model, "mode": args.mode,
            "precision": args.precision, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "grad_accum": args.grad_accum,
            "grad_ckpt": args.gradient_checkpointing, "compile": args.compile,
            "cpu_offload": args.cpu_offload, "world_size": world,
            "params_m": round(n_params / 1e6),
            "step_ms": round(mean_dt * 1000, 1),
            "tokens_per_s": round(tps),
            "tflops": round(tflops, 2),
            "mfu_pct": round(tflops / peak * 100, 2),
            "peak_tflops_assumed": peak,
            "torch_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            **{k: round(v, 1) for k, v in sampler.summary().items()},
        }
        with open(results_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        log(json.dumps(summary, indent=2))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
