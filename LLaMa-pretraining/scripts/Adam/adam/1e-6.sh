#!/bin/bash
# bash hed_scripts/130M/adam/adam_130M_lr1e_3_WD1e_5.sh

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20011 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 5 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-130M-lr0.001-WD0.000001-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20012 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 6 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-130M-lr0.001-WD0.000001-repeat2 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20013 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 7 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-130M-lr0.001-WD0.000001-repeat3 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20014 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 8 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-130M-lr0.001-WD0.000001-repeat4 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20015 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 9 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-130M-lr0.001-WD0.000001-repeat5 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20016 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 10 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-130M-lr0.001-WD0.000001-repeat6 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000