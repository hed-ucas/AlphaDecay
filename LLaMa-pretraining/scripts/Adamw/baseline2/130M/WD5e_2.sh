#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=22013 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adamw \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.05 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb gradnorm \
    --assign_func layerwise_sigmoid \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name baseline2-adamw-130M-lr0.001-WD0.05-20000 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
