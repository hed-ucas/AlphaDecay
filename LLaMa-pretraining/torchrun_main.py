import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from WeightDecayUnbalance import modulewise_AlphaDecay

from AWD_nips2023 import AdaptiveWeightDecay

transformers.logging.set_verbosity_error()
def configure_optimizers(param_dict, weight_decay, learning_rate, device_type, determined):
    if determined:
        optim_groups = param_dict
    else:
        optim_groups = [{'params': param_dict, 'weight_decay': weight_decay}] 

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer

def calculate_layer_gradnorms(model):
    """Calculate gradient norms for transformer layers only"""
    layer_gradnorms = {}
    
    # For DDP models, get the underlying model
    if hasattr(model, 'module'):
        model = model.module

    # Get all parameter names and split them into layers
    for name, param in model.named_parameters():
        if param.grad is not None and 'layers.' in name:
            # Extract layer index for transformer layers
            layer_idx = name.split('layers.')[1].split('.')[0]
            layer_name = f'layer{layer_idx}'
            
            if layer_name not in layer_gradnorms:
                layer_gradnorms[layer_name] = []
            
            layer_gradnorms[layer_name].append(param.grad.data.norm(2).item())
    
    # Calculate mean gradient norm for each layer
    layer_mean_gradnorms = {
        layer: np.mean(norms) 
        for layer, norms in layer_gradnorms.items()
    }
    
    return layer_mean_gradnorms

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default='configs/llama_60m.json')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_batch_size", type=int, default=64)
    parser.add_argument("--num_training_steps", type=int, default=20000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, default='code_test')
    parser.add_argument("--target_eval_tokens", type=int, default=10_000_000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--continue_from", type=str, default=None) 
    parser.add_argument("--optimizer", type=str, default="adam") 


    #parser.add_argument("--use_modulewise_wd", action="store_true", help="Whether to use unbalanced weight decay")
    parser.add_argument("--use_modulewise_wd", type=bool, default=True)

    parser.add_argument('--assign_func',         type=str,        default='tb_linear_map',       help='assignment function for layerwise lr')
    parser.add_argument('--wd_min_ratio',        type=float,    default=0.666)
    parser.add_argument('--wd_max_ratio',        type=float,    default=3)

    parser.add_argument("--unbalancedWD_every", type=int, default=500, help="Update unbalanced weight decay every n steps")
    parser.add_argument('--pl_fitting',     type=str,        default='median', choices=['median', 'goodness-of-fit', 'fix-finger'])
    parser.add_argument('--remove_last_layer',  default=True,   type=lambda x: (str(x).lower() == 'true'),  help='if remove the last layer')
    parser.add_argument('--remove_first_layer', default=True,   type=lambda x: (str(x).lower() == 'true'),  help='if remove the first layer')
    parser.add_argument('--batchnorm',          default=True,   type=lambda x: (str(x).lower() == 'true'),  help='balancing batch norm layer')
    parser.add_argument('--filter_zeros',       default=False,  type=lambda x: (str(x).lower() == 'true')   )
    parser.add_argument('--esd_metric_for_tb',   type=str,      default='alpha',  help='ww metric')
    parser.add_argument('--xmin_pos',            type=float,    default=2, help='xmin_index = size of eigs // xmin_pos')
    parser.add_argument('--batchnorm_type',      type=str,      default='name',  help='method to change batchnorm layer learning rate')

    # baseline1: AWD NeurIPS 2023
    parser.add_argument("--use_AWD_NeurIPS2023", action="store_true", help="Whether to test AWD from NeurIPS2023")
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")

    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    parser.add_argument("--single_cuda", default=False, action="store_true")
    
    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, target_eval_tokens):
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)

    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_cuda:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):

        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()} 
        labels = batch["input_ids"].clone() 
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss 
        total_loss += loss.detach() 

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size 

    total_loss = total_loss / total_batches 

    # Gather losses across all cudas
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)] 
    dist.all_gather(gathered_losses, total_loss) 
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens 


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not "LOCAL_RANK" in os.environ:
        os.environ['RANK'] = '0'
        os.environ["LOCAL_RANK"] = '0'
        os.environ["WORLD_SIZE"] = '1'
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = '26000'

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    if args.use_AWD_NeurIPS2023:
        args.weight_decay *= 100
    
    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="AlphaDecay", name=args.wandb_name)
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
    seed_for_shuffle = 42 
    
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_cuda:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from,'pytorch_model.bin')
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    if args.use_modulewise_wd:
        print("##############Enable and init AlphaDecay Balancing##################")
        wd_scheduler = modulewise_AlphaDecay(net=model, 
                    use_modulewise_wd=args.use_modulewise_wd,
                    pl_fitting=args.pl_fitting,
                    xmin_pos=args.xmin_pos, 
                    filter_zeros=args.filter_zeros,
                    remove_first_layer=args.remove_first_layer,
                    remove_last_layer=args.remove_last_layer,
                    esd_metric_for_tb=args.esd_metric_for_tb,
                    assign_func=args.assign_func,
                    wd_min_ratio=args.wd_min_ratio,
                    wd_max_ratio=args.wd_max_ratio,
                    batchnorm=args.batchnorm,
                    batchnorm_type=args.batchnorm_type,
                    wandb_name=args.wandb_name)
        trainable_params, _, _ = wd_scheduler.build_optimizer_param_group(untuned_wd=args.weight_decay, initialize=True)
    elif args.use_AWD_NeurIPS2023:
        print("##############Enable and init AWD NIPS 2023##################")
        wd_scheduler = AdaptiveWeightDecay(lambda_awd=args.weight_decay)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay) 
    elif args.optimizer.lower() == "adamw":
        optimizer = configure_optimizers(trainable_params, args.weight_decay, args.lr, 'cuda' if 'cuda' in device else 'cpu', args.use_modulewise_wd)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    print('*********************************')
    print(optimizer)
    print('*********************************')

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio)

    if not args.single_cuda:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False)

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    layermean_alpha = None

    for batch_idx, batch in enumerate(dataloader): 
        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps: 
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()} 

        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size 

        loss = model(**batch, labels=labels).loss 
        loss_temp = loss.detach().clone()

        scaled_loss = loss / args.gradient_accumulation 
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0: 
            continue

        ##################################################################################
        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
        if global_rank == 0: pbar.update(1) 

        wd = args.weight_decay                        
        if args.use_AWD_NeurIPS2023:
            wd = wd_scheduler.step(optimizer)
        elif args.use_modulewise_wd and (update_step+1) % args.unbalancedWD_every == 0:
            gradnorm = calculate_layer_gradnorms(model.module if hasattr(model, 'module') else model)
            print('----> One step of AlphaDecay Balancing')
            layermean_alpha = wd_scheduler.step(optimizer, wd, update_step, rank0=global_rank==0, gradnorm=gradnorm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}" 
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            model.module.generation_config.pad_token_id=0
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB', safe_serialization=False)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
                
            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, target_eval_tokens=args.target_eval_tokens
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before 

        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss_temp.item(),
                "lr": lr,
                "weight_decay": wd, 
                "update_step": update_step, 
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                },
                step=global_step,
            )
            if layermean_alpha != None:
                wandb.log({
                "alpha_Q": layermean_alpha['attn.q_proj'],
                "alpha_K": layermean_alpha['attn.k_proj'],
                "alpha_V": layermean_alpha['attn.v_proj'],
                "alpha_O": layermean_alpha['attn.o_proj'],
                "alpha_MLPgate": layermean_alpha['mlp.gate_proj'],
                "alpha_MLPup": layermean_alpha['mlp.up_proj'],
                "alpha_MLPdown": layermean_alpha['mlp.down_proj'],
                },
                step=global_step,)
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0:
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.generation_config.pad_token_id=0
        model.module.save_pretrained(current_model_directory, safe_serialization=False)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, loss_temp, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, target_eval_tokens=args.target_eval_tokens
    )

    if global_rank == 0:
        wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)