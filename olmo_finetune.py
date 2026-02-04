import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Improved Config ----------------
MODEL_NAME = "allenai/OLMo-7B-hf"
MAX_LEN = 128
BATCH_SIZE = 8  
GRADIENT_ACCUM_STEPS = 2  # Effective batch = 8*2*4
LR = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EPOCHS = 5
PATIENCE = 7
MIN_DELTA = 0.0001
WARMUP_RATIO = 0.1
NUM_WORKERS = 4

# Dataset name
DATASET_NAME = 'clearance'

# ---------------- Dataset ----------------
class DelaneyDataset(Dataset):
    def __init__(self, df, tokenizer, normalize=True, label_stats=None):
        """
        Args:
            df: DataFrame with 'smiles' and 'measured log solubility in mols per litre' columns
            tokenizer: Hugging Face tokenizer
            normalize: Whether to normalize labels
            label_stats: Dict with 'mean' and 'std' for denormalization (used for val/test)
        """
        # Pre-tokenize everything at once
        texts = [f"Molecule: {s}\nPredict intrinsic hepatic clearance from SMILES" for s in df['smiles']]
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN, 
            return_tensors="pt"
        )
        
        # Handle label normalization
        labels = df['target'].values
        
        if normalize and label_stats is None:
            # Calculate stats from training data
            self.label_mean = float(labels.mean())
            self.label_std = float(labels.std())
            normalized_labels = (labels - self.label_mean) / self.label_std
        elif normalize and label_stats is not None:
            # Use provided stats (for val/test sets)
            self.label_mean = label_stats['mean']
            self.label_std = label_stats['std']
            normalized_labels = (labels - self.label_mean) / self.label_std
        else:
            # No normalization
            self.label_mean = 0.0
            self.label_std = 1.0
            normalized_labels = labels
        
        self.labels = torch.tensor(normalized_labels, dtype=torch.float32)
        self.raw_labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# ---------------- Model Wrapper ----------------
class OLMoRegression(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.regressor = nn.Linear(backbone.config.hidden_size, 1)
        hidden_size = backbone.config.hidden_size

        # MLP Head
        self.regressor = nn.Linear(backbone.config.hidden_size, 1)

        # Initialize regressor with small weights
        nn.init.normal_(self.regressor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        last_hidden_state = out.last_hidden_state
        
        # # Masked Mean Pooling logic
        # expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
        # sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        # h = sum_embeddings / sum_mask
        
        # preds = self.regressor(h).squeeze(-1)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        
        # Gather the vectors at the last valid index
        last_token_indices = sequence_lengths.view(-1, 1, 1).expand(batch_size, 1, last_hidden_state.size(-1))
        # Ensure indices are on the correct device
        last_token_indices = last_token_indices.to(last_hidden_state.device)
        
        # Shape: [batch, hidden_size]
        pooled_output = torch.gather(last_hidden_state, 1, last_token_indices).squeeze(1)
        
        # Regression head
        preds = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            # Using RMSE loss
            loss = torch.sqrt(nn.functional.mse_loss(preds, labels) + 1e-6)

        return preds, loss

# ---------------- Lightning Module ----------------
class OLMoRegressionLightning(pl.LightningModule):
    def __init__(
        self,
        model_name=MODEL_NAME,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        label_mean=0.0,
        label_std=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.label_mean = label_mean
        self.label_std = label_std
        
        # Model will be initialized in configure_model (for proper device placement)
        self.model = None
        
    def configure_model(self):
        """Called before devices are set up - perfect for model initialization"""
        if self.model is not None:
            return
            
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        base_model = AutoModel.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map={"": self.device.index if self.device.type == "cuda" else "cpu"}
        )
        
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Improved LoRA config
        lora_cfg = LoraConfig(
            r=32,  # Increased from 16
            lora_alpha=64,  # Keep 2:1 ratio
            target_modules=[
                "q_proj", "k_proj", "v_proj", 
                # "o_proj","gate_proj", "up_proj", "down_proj"  # Added MLP layers
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        base_model = get_peft_model(base_model, lora_cfg)
        
        if self.global_rank == 0:
            base_model.print_trainable_parameters()
        
        self.model = OLMoRegression(base_model)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask, labels)

    def training_step(self, batch, batch_idx):
        preds, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Calculate additional metrics (denormalized)
        preds_denorm = preds * self.label_std + self.label_mean
        labels_denorm = batch["labels"] * self.label_std + self.label_mean
        mae = torch.mean(torch.abs(preds_denorm - labels_denorm))
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Calculate metrics (denormalized)
        preds_denorm = preds * self.label_std + self.label_mean
        labels_denorm = batch["labels"] * self.label_std + self.label_mean
        
        mse = torch.mean((preds_denorm - labels_denorm) ** 2)
        rmse = torch.sqrt(mse + 1e-6)
        mae = torch.mean(torch.abs(preds_denorm - labels_denorm))
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        
        return {"val_loss": loss, "val_rmse": rmse, "val_mae": mae}

    def test_step(self, batch, batch_idx):
        preds, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Calculate metrics (denormalized)
        preds_denorm = preds * self.label_std + self.label_mean
        labels_denorm = batch["labels"] * self.label_std + self.label_mean
        
        mse = torch.mean((preds_denorm - labels_denorm) ** 2)
        rmse = torch.sqrt(mse + 1e-6)
        mae = torch.mean(torch.abs(preds_denorm - labels_denorm))
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        
        return {"test_loss": loss, "test_rmse": rmse, "test_mae": mae}

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
class DelaneyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name=DATASET_NAME,
        data_dir='splits',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=MODEL_NAME,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_mean = None
        self.label_std = None

    def prepare_data(self):
        """Download or prepare data (called only on 1 GPU/process)"""
        # This is where you'd download data if needed
        pass

    def setup(self, stage=None):
        """Setup datasets (called on every GPU/process)"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if stage == "fit" or stage is None:
            # Load data
            train_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/train.csv')
            val_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/valid.csv')
            
            # Create training dataset (computes normalization stats)
            self.train_dataset = DelaneyDataset(train_df, self.tokenizer, normalize=True)
            
            # Store normalization stats
            self.label_mean = self.train_dataset.label_mean
            self.label_std = self.train_dataset.label_std
            
            # Create validation dataset (uses training stats)
            label_stats = {'mean': self.label_mean, 'std': self.label_std}
            self.val_dataset = DelaneyDataset(val_df, self.tokenizer, normalize=True, label_stats=label_stats)
            
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
            print(f"Label normalization - Mean: {self.label_mean:.4f}, Std: {self.label_std:.4f}")
        
        if stage == "test" or stage is None:
            test_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/test.csv')
            
            # Use training stats if available, otherwise load train data to get stats
            if self.label_mean is None:
                train_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/train.csv')
                temp_dataset = DelaneyDataset(train_df, self.tokenizer, normalize=True)
                self.label_mean = temp_dataset.label_mean
                self.label_std = temp_dataset.label_std
            
            label_stats = {'mean': self.label_mean, 'std': self.label_std}
            self.test_dataset = DelaneyDataset(test_df, self.tokenizer, normalize=True, label_stats=label_stats)
            
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
    
    # Initialize data module
    data_module = DelaneyDataModule(
        dataset_name=DATASET_NAME,
        data_dir='splits',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=MODEL_NAME,
    )
    
    # Setup data to get normalization stats
    data_module.setup(stage="fit")
    
    # Initialize model with normalization stats
    model = OLMoRegressionLightning(
        model_name=MODEL_NAME,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        label_mean=data_module.label_mean,
        label_std=data_module.label_std,
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val/rmse',
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        verbose=True,
        mode='min'
    )
    
    # # SOLUTION 1: Save only LAST checkpoint (smallest disk usage)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath='/kaggle/working/checkpoints',
    #     filename='best-model',
    #     monitor='val/rmse',
    #     mode='min',
    #     save_top_k=1,  # Only save the single best model
    #     save_last=False,  # Don't save last separately
    #     verbose=True,
    #     save_weights_only=True,  # CRITICAL: Only save weights, not optimizer state (~10x smaller!)
    # )

    checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints_clearance',
    filename='olmo-qlora-{epoch:02d}-{val/rmse:.4f}',
    monitor='val/rmse',
    mode='min',
    save_top_k=1,
    save_last=True,
    verbose=True,
    save_weights_only=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='olmo_qlora_regression',
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=-1,  # Use all available GPUs
        strategy='ddp',  # Distributed Data Parallel
        precision='16-mixed',  # Mixed precision training
        gradient_clip_val=MAX_GRAD_NORM,
        accumulate_grad_batches=GRADIENT_ACCUM_STEPS,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # # Test with best model
    # print("\nLoading best model for testing...")
    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Best model path: {best_model_path}")
    
    # if best_model_path:
    #     # Load best model
    #     model = OLMoRegressionLightning.load_from_checkpoint(
    #         best_model_path,
    #         label_mean=data_module.label_mean,
    #         label_std=data_module.label_std,
    #     )
    
    # # Test
    # print("\nRunning test evaluation...")
    # data_module.setup(stage="test")
    # trainer.test(model, data_module)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()