#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
torchrun --nproc_per_node=1 --master_port=20123 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_60m.json \
    --optimizer adamw \
    --lr 0.001 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adamw-60M-lr0.001-WD0.01-10000-ours-1.5-5unb \
    --target_eval_tokens 10_000_000 \
    --save_every 10000