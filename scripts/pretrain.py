#!/usr/bin/env python3
"""
Causal LM pretraining on SMILES datasets (ZINC20, PubChem).

Examples:
    # Pretrain on ZINC20
    torchrun --nproc_per_node=4 scripts/pretrain.py configs/pretrain.yaml

    # Custom config
    python scripts/pretrain.py my_config.yaml
"""

import gc
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from peft import PeftModel
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from chemberta4.callbacks import WandbCallback
from chemberta4.data import PretrainingDataset
from chemberta4.trainer import OLMoPretrainer
from chemberta4.utils import is_main_process, print0

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_smiles_data(args):
    """Load SMILES dataset based on args."""
    if args.dataset == "zinc20":
        # ZINC20 dataset from HuggingFace
        dataset = load_dataset("zpn/zinc20", split="train", streaming=True, trust_remote_code=True).take(args.num_samples)
        smiles_col = "smiles"
    elif args.dataset == "pubchem":
        # PubChem dataset
        dataset = load_dataset("sagawa/pubchem-10m-smiles", split="train", streaming=True, trust_remote_code=True).take(args.num_samples)
        smiles_col = "smiles"
    elif args.dataset == "custom":
        if args.dataset_path is None:
            raise ValueError("--dataset_path required for custom dataset")
        dataset = load_dataset(args.dataset_path, split="train", streaming=True, trust_remote_code=True)
        smiles_col = args.smiles_column
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Take subset if specified
    if args.num_samples is not None:
        dataset = dataset.take(args.num_samples)

    # Extract SMILES strings
    smiles_list = [item[smiles_col] for item in dataset]
    return smiles_list


def run_pretraining_experiment(args, task_name):
    """Run pretraining for a single dataset."""
    # Set dataset in args for load_smiles_data compatibility
    args.dataset = task_name

    print0(f"Loading dataset: {args.dataset}")
    smiles_list = load_smiles_data(args)
    print0(f"Loaded {len(smiles_list)} SMILES strings")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Split into train/val
    val_size = int(len(smiles_list) * args.val_ratio)
    val_smiles = smiles_list[:val_size]
    train_smiles = smiles_list[val_size:]

    train_dataset = PretrainingDataset(train_smiles, tokenizer, args.max_len)
    val_dataset = PretrainingDataset(val_smiles, tokenizer, args.max_len)
    print0(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Model
    model = OLMoPretrainer(
        model_name=args.model_name,
        finetune_strategy=args.finetune_strategy,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        *([ WandbCallback() ] if args.wandb else []),
    ]

    num_devices = torch.cuda.device_count() or 1

    # Tracker init (rank 0 only)
    if is_main_process() and args.wandb:
        import wandb

        if args.wandb_key:
            wandb.login(key=args.wandb_key)

        project_name = args.wandb_project or f"chemberta4-pretrain-{args.dataset}"
        log_params = {k: v for k, v in vars(args).items() if k != "wandb_key"}
        wandb.init(
            entity=args.wandb_entity,
            project=project_name,
            name=f"pretrain-{args.dataset}_{timestamp}",
            config=log_params,
            reinit=True,
        )
        wandb.config.update({
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "num_devices": num_devices,
            "effective_batch_size": args.batch_size * args.gradient_accum * num_devices,
        })

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        precision="16-mixed",
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accum,
        gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=True,
        val_check_interval=args.val_check_interval,
    )

    # Train
    print0(f"Starting pretraining on {args.dataset}...")
    trainer.fit(model, train_dataloader, val_dataloader)

    # Finalize tracker
    if is_main_process() and args.wandb:
        import wandb

        for key, value in trainer.callback_metrics.items():
            wandb.log({f"final_{key.replace('/', '_')}": float(value)})
        wandb.finish()

    # Merge and push to hub
    if trainer.is_global_zero and args.hub_name:
        print0("Training complete. Saving adapter...")

        adapter_path = f"{args.output_dir}/{args.dataset}_adapter"
        model.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        print0("Freeing VRAM for merge...")
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        print0("Loading base model in FP16 for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print0("Loading adapter and merging...")
        model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)
        model_to_merge = model_to_merge.merge_and_unload()

        print0(f"Pushing merged model to Hub: {args.hub_name}")
        model_to_merge.push_to_hub(args.hub_name)
        tokenizer.push_to_hub(args.hub_name)

        print0(f"Done! Model available at: https://huggingface.co/{args.hub_name}")
    elif trainer.is_global_zero:
        # Just save locally
        adapter_path = f"{args.output_dir}/{args.dataset}_adapter"
        model.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print0(f"Adapter saved to: {adapter_path}")

    # Cleanup
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
