#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20002 --master_addr=localhost main.py \
    --model vit_deit_tiny_patch16_224 \
    --drop_path 0.1 \
    --batch_size 192 \
    --lr 4e-3 \
    --weight_decay 0.000001 \
    --use_AWD_NeurIPS2023 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /home/hedi/ImageNet \
    --output_dir /home/hedi/LLM/ViT-awd/outputs \
    --wandb_name vit_deit_tiny_patch16_224-baseline1-WD0.000001