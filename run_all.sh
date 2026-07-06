#!/usr/bin/env bash
# Runs the parallelism/optimization matrix sequentially with idle gaps in
# between so each experiment is a distinct hump in nvtop.
# Watch alongside:   nvtop    (or: watch -n1 nvidia-smi)
set -e
cd "$(dirname "$0")"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
STEPS="${STEPS:-30}"
gap() { echo "--- idle gap ---"; sleep 5; }

# 1) precision ladder: same work, watch TFLOPS & power climb, step time drop
python train_experiment.py --tag 01_fp32_bs1        --model "$MODEL" --steps "$STEPS" --precision fp32 --batch-size 1;  gap
python train_experiment.py --tag 02_tf32_bs1        --model "$MODEL" --steps "$STEPS" --precision tf32 --batch-size 1;  gap
python train_experiment.py --tag 03_bf16_bs1        --model "$MODEL" --steps "$STEPS" --precision bf16 --batch-size 1;  gap

# 2) batch scaling: fill the SMs; GPU% may look similar, MFU tells the truth
python train_experiment.py --tag 04_bf16_bs8        --model "$MODEL" --steps "$STEPS" --precision bf16 --batch-size 8;  gap

# 3) gradient accumulation: data parallelism's math on one GPU (big global batch, flat memory)
python train_experiment.py --tag 05_bf16_bs8_ga4    --model "$MODEL" --steps 10       --precision bf16 --batch-size 8 --grad-accum 4;  gap

# 4) gradient checkpointing: memory down, extra re-forward compute
python train_experiment.py --tag 06_bf16_bs8_ckpt   --model "$MODEL" --steps "$STEPS" --precision bf16 --batch-size 8 --gradient-checkpointing;  gap

# 5) torch.compile: kernel fusion -> fewer/fatter kernels (slow first step = compile)
python train_experiment.py --tag 07_bf16_bs8_compile --model "$MODEL" --steps "$STEPS" --precision bf16 --batch-size 8 --compile;  gap

# 6) DDP, 2 ranks sharing the single GB10 (gloo allreduce on CPU):
#    same global batch as 04 -> pure overhead view of data parallelism on one die
torchrun --nproc_per_node=2 train_experiment.py --tag 08_ddp2_bf16_bs4 --model "$MODEL" --steps "$STEPS" --mode ddp --precision bf16 --batch-size 4;  gap

# 7) FSDP: ZeRO-style sharding mechanics (world=1), then with CPU offload
torchrun --nproc_per_node=1 train_experiment.py --tag 09_fsdp_bf16_bs8    --model "$MODEL" --steps "$STEPS" --mode fsdp --precision bf16 --batch-size 8;  gap
torchrun --nproc_per_node=1 train_experiment.py --tag 10_fsdp_offload    --model "$MODEL" --steps 10       --mode fsdp --precision bf16 --batch-size 8 --cpu-offload;  gap

python summarize.py
