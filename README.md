# AlphaDecay

This repo contains the pre-release version of AlphaDecay algorithm, proposed by [AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs].

AlphaDecay determines the weight decay parameter values of each module in LLM training through the spectral characteristics of the ESD distribution and makes dynamic adjustments during the training process, thereby improving the training performance of the model.

<div align="center">
  <img src="./images/AlphaDecay.png" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>

## Abstract

Weight decay is a standard regularization technique for training large language models (LLMs).  While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM.  Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify “heavy-tailedness.”  Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay.  Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance.  Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines. 

## Quick Start

### Setup

Our repository is built on top of [Galore](https://github.com/jiaweizzhao/GaLore) and [ConvNeXt](https://github.com/jiaweizzhao/GaLore). You can configure the environment using the following command lines:
```bash
conda create -n alphadecay python=3.9 -y
conda activate alphadecay
conda install -r requirements
```

### Prepare Dataset

We utilized the publicly available C4 dataset and ImageNet-1K dataset, both of which can be accessed and downloaded from their respective official websites.

#### Pretraining LLama-130M on C4
```bash
torchrun --nproc_per_node=2 --master_port=20301 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_130m.json \
    --optimizer adam \
    --seed 5 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.00001 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 3 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name ours-adam-1.5-3unb-130M-lr0.001-WD0.00001-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```
#### Pretraining ViT-tiny on Imagenet-1K
```bash
torchrun --nproc_per_node=2 --master_port=20023 --master_addr=localhost main.py \
    --model vit_deit_tiny_patch16_224 \
    --drop_path 0.1 \
    --batch_size 192 \
    --lr 4e-3 \
    --weight_decay 0.000005 \
    --use_modulewise_wd \
    --alpha_positively_with_WD \
    --unbalancedWD_every 500 \
    --esd_metric_for_tb alpha \
    --assign_func tb_linear_map\
    --wd_min_ratio 0.6666 \
    --wd_max_ratio 3 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /home/hedi/ImageNet \
    --output_dir /home/hedi/LLM/ViT-awd/outputs \
    --wandb_name vit_deit_tiny_patch16_224-Ours-WD0.000005-1.5-5unb
```

### Acknowledgement
This repository is build upon the [Galore](https://github.com/jiaweizzhao/GaLore) and [ConvNeXt](https://github.com/jiaweizzhao/GaLore) repositories. Thanks for their great work!

