import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchmetrics import Accuracy, AUROC
import pandas as pd
import os
import json
import mlflow
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Configuration ----------------
MODEL_NAME = "allenai/OLMo-7B-hf"
MAX_LEN = 128
BATCH_SIZE = 8  # Reduced for T4 GPU memory constraints
GRADIENT_ACCUM_STEPS = 4  # Increased to maintain effective batch size
LR = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EPOCHS =1
PATIENCE = 7
MIN_DELTA = 0.0001
WARMUP_RATIO = 0.1
NUM_WORKERS = 4

# Classification settings
DATASET_NAME = 'hiv'  # Changeable: 'bbbp', 'bace', 'hiv', 'clintox'
NUM_CLASSES = 2
USE_LM_HEAD = True  # True = AutoModelForCausalLM, False = AutoModel + classification head

# Dataset-specific configuration
DATASET_CONFIG = {
    'bbbp': {
        'task_column': 'p_np',
        'prompt': 'Does this molecule permeate the blood-brain barrier?',
    },
    'bace_classification': {
        'task_column': 'Class',
        'prompt': 'Is this molecule a BACE-1 inhibitor?',
    },
    'hiv': {
        'task_column': 'HIV_active',
        'prompt': 'Does this molecule inhibit HIV replication?',
    },
    'clintox': {
        'task_column': 'CT_TOX',
        'prompt': 'Is this molecule clinically toxic?',
    },
    'tox21': {
        'task_column': 'SR-p53',
        'prompt': 'Is this molecule toxic on the target SR-p53?'
    }
}

# LoRA settings
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# ---------------- Dataset ----------------
class MolnetDataset(Dataset):
    def __init__(self, df, tokenizer, dataset_name, use_lm_head=False):
        """
        Generic MoleculeNet classification dataset.
        
        Args:
            df: DataFrame with 'smiles' and task column
            tokenizer: Hugging Face tokenizer
            dataset_name: Name of the dataset (e.g., 'bbbp', 'bace')
            use_lm_head: Whether to use LM head (affects prompt format)
        """
        config = DATASET_CONFIG[dataset_name]
        task_col = config['task_column']
        prompt_text = config['prompt']
        
        # Filter NaN labels
        df = df.dropna(subset=[task_col]).copy()
        
        # Build prompts based on model type
        if use_lm_head:
            # For LM head: prompt expects Yes/No answer
            texts = [f"Molecule: {s}\nQuestion: {prompt_text}\nAnswer:" 
                     for s in df['smiles']]
        else:
            # For classification head
            texts = [f"Molecule: {s}\n{prompt_text}" for s in df['smiles']]
        
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN, 
            return_tensors="pt"
        )
        
        # Labels as integers for classification
        self.labels = torch.tensor(df[task_col].values.astype(int), dtype=torch.long)
        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# ---------------- Model Wrappers ----------------
class OLMoClassification(nn.Module):
    """AutoModel + Classification Head approach."""
    
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)
        
        # Initialize classifier with small weights
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True,
        )
        
        # last_hidden_state = out.hidden_states[-1]
        last_hidden_state = out.last_hidden_state
        
        # Last token pooling for decoder-only model
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        
        last_token_indices = sequence_lengths.view(-1, 1, 1).expand(batch_size, 1, last_hidden_state.size(-1))
        last_token_indices = last_token_indices.to(last_hidden_state.device)
        
        pooled_output = torch.gather(last_hidden_state, 1, last_token_indices).squeeze(1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return logits, loss


class OLMoCausalClassification(nn.Module):
    """AutoModelForCausalLM approach - uses LM head to predict Yes/No tokens."""
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        
        # Get token IDs for Yes/No
        self.yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get logits for last non-padding token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = outputs.logits.shape[0]
        
        # Gather last token logits
        last_logits = outputs.logits[torch.arange(batch_size, device=outputs.logits.device), sequence_lengths]  # [batch, vocab_size]
        
        # Extract Yes/No logits only
        yes_logits = last_logits[:, self.yes_token_id]
        no_logits = last_logits[:, self.no_token_id]
        logits = torch.stack([no_logits, yes_logits], dim=-1)  # [batch, 2] - class 0=No, class 1=Yes

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return logits, loss

# ---------------- MLflow Callback ----------------
class MLflowCallback(Callback):
    """Logs training and validation metrics to MLflow at the end of each epoch."""

    def on_train_epoch_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            if "train" in key:
                mlflow.log_metric(key.replace("/", "_"), float(value), step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            if "val" in key:
                mlflow.log_metric(key.replace("/", "_"), float(value), step=trainer.current_epoch)

        # Log learning rate
        if trainer.lr_scheduler_configs:
            try:
                lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
                mlflow.log_metric("learning_rate", lr, step=trainer.current_epoch)
            except:
                pass

# ---------------- Lightning Module ----------------
class OLMoClassificationLightning(pl.LightningModule):
    def __init__(
        self,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        use_lm_head=USE_LM_HEAD,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_lm_head = use_lm_head
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        
        # Model will be initialized in configure_model
        self.model = None
        self.tokenizer = None
        
        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_auroc = AUROC(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.test_auroc = AUROC(task='binary')
        
    def configure_model(self):
        """Called before devices are set up - perfect for model initialization."""
        if self.model is not None:
            return
            
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # LoRA config
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM" if self.use_lm_head else "FEATURE_EXTRACTION",
        )
        
        if self.use_lm_head:
            # Use AutoModelForCausalLM with LM head
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": self.device.index if self.device.type == "cuda" else "cpu"}
            )
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
            base_model = get_peft_model(base_model, lora_cfg)
            
            if self.global_rank == 0:
                base_model.print_trainable_parameters()
            
            self.model = OLMoCausalClassification(base_model, self.tokenizer)
        else:
            # Use AutoModel with classification head
            base_model = AutoModel.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": self.device.index if self.device.type == "cuda" else "cpu"}
            )
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
            base_model = get_peft_model(base_model, lora_cfg)
            
            if self.global_rank == 0:
                base_model.print_trainable_parameters()
            
            self.model = OLMoClassification(base_model, self.num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask, labels)

    def training_step(self, batch, batch_idx):
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Calculate metrics
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        self.train_acc(preds, batch["labels"])
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Calculate metrics
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        
        self.val_acc(preds, batch["labels"])
        self.val_auroc(probs, batch["labels"])
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/roc_auc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Calculate metrics
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        
        self.test_acc(preds, batch["labels"])
        self.test_auroc(probs, batch["labels"])
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/roc_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"test_loss": loss}

    def configure_optimizers(self):
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=self.lr)
        
        # Calculate total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Learning rate schedulers
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps,
        )
        
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

# ---------------- Data Module ----------------
class MolnetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name=DATASET_NAME,
        use_lm_head=USE_LM_HEAD,
        data_dir='splits',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=MODEL_NAME,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.use_lm_head = use_lm_head
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.task_column = DATASET_CONFIG[dataset_name]['task_column']
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download or prepare data (called only on 1 GPU/process)."""
        pass

    def setup(self, stage=None):
        """Setup datasets (called on every GPU/process)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/train.csv')
            val_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/valid.csv')
            
            self.train_dataset = MolnetDataset(
                train_df, self.tokenizer, self.dataset_name, self.use_lm_head
            )
            self.val_dataset = MolnetDataset(
                val_df, self.tokenizer, self.dataset_name, self.use_lm_head
            )
            
            print(f"Dataset: {self.dataset_name}")
            print(f"Task column: {self.task_column}")
            print(f"Using LM head: {self.use_lm_head}")
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            test_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/test.csv')
            self.test_dataset = MolnetDataset(
                test_df, self.tokenizer, self.dataset_name, self.use_lm_head
            )
            print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

# ---------------- Main Training Script ----------------
def main():
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # MLflow setup
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(f"olmo-molnet-{DATASET_NAME}")
    run_name = f"{DATASET_NAME}_{'lm_head' if USE_LM_HEAD else 'cls_head'}"
    
    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters
        mlflow.log_params({
            "dataset_name": DATASET_NAME,
            "use_lm_head": USE_LM_HEAD,
            "model_name": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "gradient_accum_steps": GRADIENT_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUM_STEPS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "warmup_ratio": WARMUP_RATIO,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        })
        
        # Log dataset info as tags
        mlflow.set_tags({
            "task_type": "classification",
            "task_column": DATASET_CONFIG[DATASET_NAME]['task_column'],
            "prompt": DATASET_CONFIG[DATASET_NAME]['prompt'],
        })
        
        # Initialize data module
        data_module = MolnetDataModule(
            dataset_name=DATASET_NAME,
            use_lm_head=USE_LM_HEAD,
            data_dir='splits',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            model_name=MODEL_NAME,
        )
        
        # Setup data
        data_module.setup(stage="fit")
        
        # Initialize model
        model = OLMoClassificationLightning(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            use_lm_head=USE_LM_HEAD,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
        )
        
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor='val/roc_auc',
            min_delta=MIN_DELTA,
            patience=PATIENCE,
            verbose=True,
            mode='max'  # Maximize ROC-AUC
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{DATASET_NAME}/{timestamp}',
            filename='best-model',
            monitor='val/roc_auc',
            mode='max',  # Maximize ROC-AUC
            save_top_k=1,
            save_last=False,
            verbose=True,
            # save_weights_only=True,
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        mlflow_callback = MLflowCallback()
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator='gpu',
            devices=-1,
            strategy='ddp',
            precision='16-mixed',
            gradient_clip_val=MAX_GRAD_NORM,
            accumulate_grad_batches=GRADIENT_ACCUM_STEPS,
            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, mlflow_callback],
            logger=True,
            log_every_n_steps=10,
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
        )
        
        # Train
        print("Starting training...")
        print(f"Model approach: {'LM Head (Yes/No prediction)' if USE_LM_HEAD else 'Classification Head'}")
        trainer.fit(model, data_module)
        
        # Log training completion metrics
        mlflow.log_metrics({
            "epochs_trained": trainer.current_epoch + 1,
            "stopped_early": int(trainer.current_epoch < EPOCHS - 1),
        })
        
        # Test
        print("\nRunning test evaluation...")
        data_module.setup(stage="test")
        test_results = trainer.test(model, data_module)
        
        # Log final test metrics
        for key, value in trainer.callback_metrics.items():
            metric_name = key.replace("/", "_")
            mlflow.log_metric(f"final_{metric_name}", float(value))
        
        # Log best checkpoint as artifact
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt and os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt, artifact_path=f"checkpoints/{DATASET_NAME}/{timestamp}")
        
        # Save supplementary log
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"{log_dir}/{DATASET_NAME}_{timestamp}.json"
        
        log_data = {
            "config": {
                "dataset_name": DATASET_NAME,
                "use_lm_head": USE_LM_HEAD,
                "model_name": MODEL_NAME,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
            },
            "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
            "best_checkpoint": best_ckpt,
            "best_val_roc_auc": float(checkpoint_callback.best_model_score or 0),
            "epochs_trained": trainer.current_epoch + 1,
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        mlflow.log_artifact(log_path)
        
        print(f"\nTraining complete!")
        print(f"Best validation ROC-AUC: {checkpoint_callback.best_model_score:.4f}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Logs saved to: {log_path}")

if __name__ == "__main__":
    main()