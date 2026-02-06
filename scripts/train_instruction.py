#!/usr/bin/env python3
"""
Instruction tuning on USPTO dataset.

Examples:
    # Basic instruction tuning
    torchrun --nproc_per_node=4 scripts/train_instruction.py --dataset uspto --num_samples 10000

    # Continue from pretrained model
    python scripts/train_instruction.py --dataset uspto --model_name harindhar10/OLMo-7B-ZINC20

    # Push to hub
    torchrun --nproc_per_node=4 scripts/train_instruction.py --dataset uspto --hub_name username/model
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from olmochem.data import InstructionDataset
from olmochem.trainer import OLMoPretrainer
from olmochem.utils import is_main_process, print0, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="OLMo Instruction Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset ----
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenMol/USPTO_1k_TPL-SFT",
        help="HuggingFace dataset name for instruction tuning",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to use",
    )

    # ---- Model ----
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMo-7B-hf",
        help="Base model name or path",
    )
    parser.add_argument(
        "--use_qlora",
        default=True,
        help="Use 4-bit QLoRA",
    )
    parser.add_argument(
        "--hub_name",
        type=str,
        default=None,
        help="HuggingFace Hub name for pushing model",
    )

    # ---- Training ----
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")

    # ---- LoRA ----
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # ---- Infrastructure ----
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    pl.seed_everything(args.seed)

    print0(f"Loading dataset: {args.dataset}")

    # Load dataset (streaming)
    dataset = (
        load_dataset(args.dataset, split="train", streaming=True, trust_remote_code=True)
        .take(args.num_samples)
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    instruction_dataset = InstructionDataset(dataset, tokenizer, args.max_len)
    print0(f"Loaded {len(instruction_dataset)} instruction samples")

    # DataLoader
    dataloader = DataLoader(
        instruction_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = OLMoPretrainer(
        model_name=args.model_name,
        use_qlora=args.use_qlora,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        precision="16-mixed",
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accum,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Train
    print0("Starting instruction tuning...")
    trainer.fit(model, dataloader)

    # Merge and push to hub
    if trainer.is_global_zero and args.hub_name:
        print0("Training complete. Saving adapter...")

        adapter_path = f"{args.output_dir}/instruction_adapter"
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
        adapter_path = f"{args.output_dir}/instruction_adapter"
        model.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print0(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
