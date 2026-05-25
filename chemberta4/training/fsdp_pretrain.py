import argparse
from functools import partial

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import ModelCard, ModelCardData
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class OLMoFSDP(pl.LightningModule):
    def __init__(self, model_id, save_name, lr=1e-5, weight_decay=0.01, warmup_ratio=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model_id = model_id
        self.save_name = save_name
        self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def configure_model(self):
        if self.model is not None:
            return

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            device_map=None,
            attn_implementation="flash_attention_2",
        )


    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        hp = self.hparams

        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "layer_norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": hp.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=hp.lr,
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * hp.warmup_ratio)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Use the passed value if set, otherwise default to 1.0
        clip_val = gradient_clip_val if gradient_clip_val is not None else 1.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)


# ---------------------------------------------------------------------------
# Non-streaming dataset: read SMILES from a plain text file
# ---------------------------------------------------------------------------
class PubChemMapDataset(Dataset):
    """Standard map-style dataset wrapping a list of SMILES strings."""

    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class PubChemDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size=4, num_samples=10_000, max_length=256, val_split=0.05):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_length = max_length
        self.val_split = val_split

    def setup(self, stage=None):
        records = []
        with open(self.data_file, "r") as f:
            for i, line in enumerate(f):
                if i >= self.num_samples:
                    break
                smiles = line.strip()
                if smiles:
                    records.append(smiles)

        split_idx = int(len(records) * (1 - self.val_split))
        self.train_dataset = PubChemMapDataset(records[:split_idx])
        self.val_dataset = PubChemMapDataset(records[split_idx:])

    def _collate_fn(self, batch):
        texts = [text + self.tokenizer.eos_token for text in batch]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )

        labels = encodings["input_ids"].clone()
        # labels[labels == self.tokenizer.pad_token_id] = -100
        labels[encodings["attention_mask"] == 0] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP full finetune on PubChem-10M SMILES (txt file)")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-7B-hf")
    parser.add_argument("--save_name", type=str, default="harindhar10/OLMo-7B-PubChem10M-full-finetune")
    parser.add_argument("--data_file", type=str, default="pubchem-10m.txt",
                        help="Path to text file with one SMILES string per line")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of samples to use for validation (default: 0.05)")
    parser.add_argument("--wandb_logging", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_key", type=str, default=None,
                        help="W&B API key (optional, uses cached login if omitted)")
    parser.add_argument("--wandb_notes", type=str, default=None,
                        help="Notes to attach to the W&B run")
    parser.add_argument("--run_id", type=str, required=True,
                        help="Unique run identifier, appended to checkpoint filename")
    parser.add_argument("--num_val_per_epoch", type=int, default=1,
                        help="Number of times to run validation per epoch")
    args = parser.parse_args()

    # ----------------------------
    # W&B Setup
    # ----------------------------
    wandb_logger = None
    if args.wandb_logging:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb_logger = WandbLogger(
            project="fsdp-pubchem10m-pretrain",
            log_model=False,
            config=vars(args),
            notes=args.wandb_notes,
        )

    # ----------------------------
    # Model + DataModule
    # ----------------------------
    pl_model = OLMoFSDP(
        model_id=args.model_id,
        save_name=args.save_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    dm = PubChemDataModule(
        pl_model.tokenizer,
        data_file=args.data_file,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_length=args.max_length,
        val_split=args.val_split,
    )

    # Force model init to get transformer layer class for FSDP wrapping
    pl_model.configure_model()

    total = sum(p.numel() for p in pl_model.parameters())
    trainable = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {trainable / total * 100:.2f}%")

    # ----------------------------
    # FSDP Strategy
    # ----------------------------
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            type(pl_model.model.model.layers[0])
        },
    )

    fsdp_strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy=auto_wrap_policy,
        cpu_offload=False,
        use_orig_params=True,
        sync_module_states=True,
        state_dict_type="full",
    )

    # ----------------------------
    # Callbacks
    # ----------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"olmo-pubchem10m-{args.run_id}-{{epoch:02d}}-{{step:06d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=4,
            save_weights_only=True,
        ),


        LearningRateMonitor(logging_interval="step"),
    ]

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=fsdp_strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        val_check_interval=1.0 / args.num_val_per_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger else True,
    )

    print(f"Starting Training on first {args.num_samples} samples from {args.data_file}...")
    trainer.fit(pl_model, datamodule=dm)

    # ---------------------------------------------------------
    # LOAD CHECKPOINT AND PUSH  (rank 0 only)
    # ---------------------------------------------------------
    trainer.strategy.barrier()

    if trainer.global_rank == 0:
        print("Loading checkpoint and pushing to HF Hub...")

        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(f"Checkpoint path: {ckpt_path}")

        print('Reloading model from pretrained')
        reload_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # reload_model.resize_token_embeddings(len(pl_model.tokenizer))

        print('Reloading model from checkpoint:', ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        cleaned = {k.replace("model.", "", 1): v for k, v in state.items()}
        reload_model.load_state_dict(cleaned, strict=True)

        reload_model.push_to_hub(args.save_name)

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.push_to_hub(args.save_name)

        card_data = ModelCardData(
            language="en",
            license="apache-2.0",
            base_model=args.model_id,
            tags=["chemistry", "SMILES", "pretraining", "fsdp", "pubchem"],
        )
        card = ModelCard.from_template(
            card_data,
            model_id=args.save_name,
            model_description="Pretrained model on PubChem-10M SMILES data.",
            training_details=f"""## Training Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | `{args.model_id}` |
| Data File | `{args.data_file}` |
| Learning Rate | `{args.lr}` |
| Weight Decay | `{args.weight_decay}` |
| Warmup Ratio | `{args.warmup_ratio}` |
| Max Epochs | `{args.max_epochs}` |
| Batch Size (per device) | `{args.batch_size}` |
| Gradient Accumulation Steps | `{args.accumulate_grad_batches}` |
| Max Sequence Length | `{args.max_length}` |
| Num Training Samples | `{args.num_samples}` |
| Precision | `bf16-mixed` |
| Gradient Clip Val | `1.0` |
| Optimizer | AdamW (betas=(0.9, 0.95), eps=1e-5) |
| LR Scheduler | Linear warmup + Cosine annealing (eta_min=1e-6) |
""",
        )
        card.push_to_hub(args.save_name)

        print(f"Model + tokenizer + model card pushed to {args.save_name}")

    if wandb_logger:
        import wandb

        wandb.finish()
