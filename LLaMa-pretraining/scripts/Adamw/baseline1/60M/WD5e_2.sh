#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
torchrun --nproc_per_node=1 --master_port=21003 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_60m.json \
    --optimizer adamw \
    --lr 0.001 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --use_AWD_NeurIPS2023 \
    --weight_decay 0.05 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name baseline1new-adamw-60M-lr0.001-WD0.05-10000 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

    