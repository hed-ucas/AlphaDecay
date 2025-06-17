#!/bin/bash
# bash hed_scripts_temp/130M/baseline/baseline1.sh

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=20321 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 5 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-5unb-130M-lr0.001-WD0.000005-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=20322 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 6 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-5unb-130M-lr0.001-WD0.000005-repeat2 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=20323 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 7 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-5unb-130M-lr0.001-WD0.000005-repeat3 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=20324 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 8 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-5unb-130M-lr0.001-WD0.000005-repeat4 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=20325 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 9 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-5unb-130M-lr0.001-WD0.000005-repeat5 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --master_port=20326 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 10 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-5unb-130M-lr0.001-WD0.000005-repeat6 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000