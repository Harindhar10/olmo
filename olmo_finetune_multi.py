

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
import numpy as np
import os
import json
import mlflow
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Configuration ----------------
MODEL_NAME = "allenai/OLMo-7B-hf"
MAX_LEN = 128
BATCH_SIZE = 4  # Reduced for T4 GPU memory constraints
GRADIENT_ACCUM_STEPS = 24  # Increased to maintain effective batch size of ~96
LR = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EPOCHS = 5
PATIENCE = 7
MIN_DELTA = 0.0001
WARMUP_RATIO = 0.1
NUM_WORKERS = 4

# Classification settings
DATASET_NAME = 'sider'  # Options: 'bbbp', 'bace', 'hiv', 'clintox', 'sider', 'tox21'
USE_LM_HEAD = False  # True = AutoModelForCausalLM, False = AutoModel + classification head

# Dataset-specific configuration
# task_type: 'binary' (single label), 'multilabel' (multiple labels per sample), 'multitask' (multiple independent tasks)
DATASET_CONFIG = {
    'bbbp': {
        'task_columns': ['p_np'],
        'prompt': 'Does this molecule permeate the blood-brain barrier?',
        'task_type': 'binary',
    },
    'bace': {
        'task_columns': ['Class'],
        'prompt': 'Is this molecule a BACE-1 inhibitor?',
        'task_type': 'binary',
    },
    'hiv': {
        'task_columns': ['HIV_active'],
        'prompt': 'Does this molecule inhibit HIV replication?',
        'task_type': 'binary',
    },
    'clintox': {
        'task_columns': ['CT_TOX'],
        'prompt': 'Is this molecule clinically toxic?',
        'task_type': 'binary',
    },
    'sider': {
        'task_columns': [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders', 'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions',
            'Endocrine disorders', 'Surgical and medical procedures',
            'Vascular disorders', 'Blood and lymphatic system disorders',
            'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders',
            'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders',
            'Psychiatric disorders', 'Renal and urinary disorders',
            'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders',
            'Nervous system disorders',
            'Injury, poisoning and procedural complications'
        ],
        'prompt': 'What side effects does this drug cause?',
        'task_type': 'multilabel',
    },
    'tox21': {
        'task_columns': [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ],
        'prompt': 'Predict toxicity across multiple assays for this molecule.',
        'task_type': 'multitask',
    },
}

# LoRA settings
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Derived settings
NUM_TASKS = len(DATASET_CONFIG[DATASET_NAME]['task_columns'])
TASK_TYPE = DATASET_CONFIG[DATASET_NAME]['task_type']

# ---------------- Helper Functions ----------------
def is_main_process():
    """Check if this is the main process (rank 0) in distributed training."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

def get_log_file_path(dataset_name):
    """Get the common log file path for a dataset."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    return f"{log_dir}/{dataset_name}_runs.json"

def load_existing_logs(log_path):
    """Load existing logs from file, return empty list if not exists."""
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return []

def save_run_log(log_path, run_data):
    """Append a new run to the existing log file."""
    logs = load_existing_logs(log_path)
    logs.append(run_data)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

# ---------------- Dataset ----------------
class MolnetDataset(Dataset):
    def __init__(self, df, tokenizer, dataset_name, use_lm_head=False):
        """
        Generic MoleculeNet classification dataset supporting binary, multilabel, and multitask.
        
        Args:
            df: DataFrame with 'smiles' and task column(s)
            tokenizer: Hugging Face tokenizer
            dataset_name: Name of the dataset
            use_lm_head: Whether to use LM head (affects prompt format)
        """
        config = DATASET_CONFIG[dataset_name]
        task_cols = config['task_columns']
        prompt_text = config['prompt']
        task_type = config['task_type']
        
        self.task_type = task_type
        self.num_tasks = len(task_cols)
        
        # For binary classification, filter rows with NaN in the single task column
        # For multilabel/multitask, keep all rows but track NaN mask
        if task_type == 'binary':
            df = df.dropna(subset=task_cols).copy()
            self.labels = torch.tensor(df[task_cols[0]].values.astype(int), dtype=torch.long)
            self.label_mask = None
        else:
            # Keep all rows, create mask for valid labels (non-NaN)
            df = df.copy()
            labels_array = df[task_cols].values.astype(np.float32)
            self.label_mask = torch.tensor(~np.isnan(labels_array), dtype=torch.bool)
            # Replace NaN with 0 for computation (will be masked in loss)
            labels_array = np.nan_to_num(labels_array, nan=0.0)
            self.labels = torch.tensor(labels_array, dtype=torch.float32)
        
        # Build prompts
        if use_lm_head:
            texts = [f"Molecule: {s}\nQuestion: {prompt_text}\nAnswer:" 
                     for s in df['smiles']]
        else:
            texts = [f"Molecule: {s}\n{prompt_text}" for s in df['smiles']]
        
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN, 
            return_tensors="pt"
        )
        
        self.num_samples = len(df)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }
        if self.label_mask is not None:
            item["label_mask"] = self.label_mask[idx]
        return item

# ---------------- Model Wrappers ----------------
class OLMoClassification(nn.Module):
    """AutoModel + Classification Head approach for binary/multilabel/multitask."""
    
    def __init__(self, backbone, num_tasks=1, task_type='binary'):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_tasks = num_tasks
        
        # Output dimension: 2 for binary (CrossEntropy), num_tasks for multilabel/multitask (BCE)
        if task_type == 'binary':
            output_dim = 2
        else:
            output_dim = num_tasks
            
        self.classifier = nn.Linear(backbone.config.hidden_size, output_dim)
        
        # Initialize classifier with small weights
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        last_hidden_state = out.hidden_states[-1]
        
        # Last token pooling for decoder-only model
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        
        last_token_indices = sequence_lengths.view(-1, 1, 1).expand(batch_size, 1, last_hidden_state.size(-1))
        last_token_indices = last_token_indices.to(last_hidden_state.device)
        
        pooled_output = torch.gather(last_hidden_state, 1, last_token_indices).squeeze(1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.task_type == 'binary':
                loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                # BCEWithLogitsLoss for multilabel/multitask
                if label_mask is not None:
                    # Masked loss - only compute loss for non-NaN labels
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(logits, labels)
                    loss = (loss * label_mask.float()).sum() / label_mask.float().sum()
                else:
                    loss = nn.BCEWithLogitsLoss()(logits, labels)

        return logits, loss


class OLMoCausalClassification(nn.Module):
    """AutoModelForCausalLM approach - uses LM head to predict Yes/No tokens."""
    
    def __init__(self, model, tokenizer, num_tasks=1, task_type='binary'):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.num_tasks = num_tasks
        
        # Get token IDs for Yes/No
        self.yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
        
        # For multilabel/multitask, we need a projection layer from Yes/No logits to task outputs
        if task_type != 'binary':
            self.task_projector = nn.Linear(1, num_tasks)
            nn.init.normal_(self.task_projector.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.task_projector.bias)

    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get logits for last non-padding token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = outputs.logits.shape[0]
        
        # Gather last token logits
        last_logits = outputs.logits[torch.arange(batch_size, device=outputs.logits.device), sequence_lengths]
        
        # Extract Yes/No logits
        yes_logits = last_logits[:, self.yes_token_id]
        no_logits = last_logits[:, self.no_token_id]
        
        if self.task_type == 'binary':
            logits = torch.stack([no_logits, yes_logits], dim=-1)  # [batch, 2]
        else:
            # Use the difference (yes - no) as a "positive" signal, project to all tasks
            yes_no_diff = (yes_logits - no_logits).unsqueeze(-1)  # [batch, 1]
            logits = self.task_projector(yes_no_diff)  # [batch, num_tasks]

        loss = None
        if labels is not None:
            if self.task_type == 'binary':
                loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                if label_mask is not None:
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    loss = loss_fct(logits, labels)
                    loss = (loss * label_mask.float()).sum() / label_mask.float().sum()
                else:
                    loss = nn.BCEWithLogitsLoss()(logits, labels)

        return logits, loss

# ---------------- MLflow Callback ----------------
class MLflowCallback(Callback):
    """Logs training and validation metrics to MLflow at the end of each epoch.
    Only logs from main process (rank 0) to avoid duplicate runs."""

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        for key, value in trainer.callback_metrics.items():
            if "train" in key:
                mlflow.log_metric(key.replace("/", "_"), float(value), step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
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
        num_tasks=NUM_TASKS,
        task_type=TASK_TYPE,
        use_lm_head=USE_LM_HEAD,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.use_lm_head = use_lm_head
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        
        # Model will be initialized in configure_model
        self.model = None
        self.tokenizer = None
        
        # Metrics based on task type
        if task_type == 'binary':
            self.train_acc = Accuracy(task='binary')
            self.val_acc = Accuracy(task='binary')
            self.val_auroc = AUROC(task='binary')
            self.test_acc = Accuracy(task='binary')
            self.test_auroc = AUROC(task='binary')
        else:
            # For multilabel/multitask, use macro-averaged metrics
            self.train_acc = Accuracy(task='multilabel', num_labels=num_tasks, average='macro')
            self.val_acc = Accuracy(task='multilabel', num_labels=num_tasks, average='macro')
            self.val_auroc = AUROC(task='multilabel', num_labels=num_tasks, average='macro')
            self.test_acc = Accuracy(task='multilabel', num_labels=num_tasks, average='macro')
            self.test_auroc = AUROC(task='multilabel', num_labels=num_tasks, average='macro')
        
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
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": self.device.index if self.device.type == "cuda" else "cpu"}
            )
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
            base_model = get_peft_model(base_model, lora_cfg)
            
            if self.global_rank == 0:
                base_model.print_trainable_parameters()
            
            self.model = OLMoCausalClassification(
                base_model, self.tokenizer, self.num_tasks, self.task_type
            )
        else:
            base_model = AutoModel.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": self.device.index if self.device.type == "cuda" else "cpu"}
            )
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
            base_model = get_peft_model(base_model, lora_cfg)
            
            if self.global_rank == 0:
                base_model.print_trainable_parameters()
            
            self.model = OLMoClassification(base_model, self.num_tasks, self.task_type)

    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        return self.model(input_ids, attention_mask, labels, label_mask)

    def _compute_metrics(self, logits, labels, label_mask, stage_acc, stage_auroc):
        """Compute accuracy and AUROC based on task type."""
        if self.task_type == 'binary':
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            stage_acc(preds, labels)
            if stage_auroc is not None:
                stage_auroc(probs, labels)
        else:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            if label_mask is not None:
                # Only compute metrics on valid (non-NaN) labels
                valid_samples = label_mask.any(dim=1)
                if valid_samples.any():
                    stage_acc(preds[valid_samples], labels[valid_samples].int())
                    if stage_auroc is not None:
                        stage_auroc(probs[valid_samples], labels[valid_samples].int())
            else:
                stage_acc(preds, labels.int())
                if stage_auroc is not None:
                    stage_auroc(probs, labels.int())

    def training_step(self, batch, batch_idx):
        label_mask = batch.get("label_mask", None)
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            label_mask=label_mask
        )
        
        self._compute_metrics(logits, batch["labels"], label_mask, self.train_acc, None)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        label_mask = batch.get("label_mask", None)
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            label_mask=label_mask
        )
        
        self._compute_metrics(logits, batch["labels"], label_mask, self.val_acc, self.val_auroc)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/roc_auc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        label_mask = batch.get("label_mask", None)
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            label_mask=label_mask
        )
        
        self._compute_metrics(logits, batch["labels"], label_mask, self.test_acc, self.test_auroc)
        
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
        self.task_columns = DATASET_CONFIG[dataset_name]['task_columns']
        self.task_type = DATASET_CONFIG[dataset_name]['task_type']
        
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
            
            # Only print from main process
            if is_main_process():
                print(f"Dataset: {self.dataset_name}")
                print(f"Task type: {self.task_type}")
                print(f"Number of tasks: {len(self.task_columns)}")
                print(f"Task columns: {self.task_columns[:5]}{'...' if len(self.task_columns) > 5 else ''}")
                print(f"Using LM head: {self.use_lm_head}")
                print(f"Training samples: {len(self.train_dataset)}")
                print(f"Validation samples: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            test_df = pd.read_csv(f'{self.data_dir}/{self.dataset_name}/test.csv')
            self.test_dataset = MolnetDataset(
                test_df, self.tokenizer, self.dataset_name, self.use_lm_head
            )
            if is_main_process():
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
    
    # Only run MLflow and logging on main process
    is_main = is_main_process()
    
    # MLflow setup (only on main process)
    mlflow_run = None
    if is_main:
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment(f"olmo-molnet-{DATASET_NAME}")
        run_name = f"{DATASET_NAME}_{'lm_head' if USE_LM_HEAD else 'cls_head'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_run = mlflow.start_run(run_name=run_name)
        
        # Log all hyperparameters
        mlflow.log_params({
            "dataset_name": DATASET_NAME,
            "task_type": TASK_TYPE,
            "num_tasks": NUM_TASKS,
            "use_lm_head": USE_LM_HEAD,
            "model_name": MODEL_NAME,
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
            "task_type": TASK_TYPE,
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
        num_tasks=NUM_TASKS,
        task_type=TASK_TYPE,
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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{DATASET_NAME}/{datetime.now().isoformat()}',
        filename='best-model',
        monitor='val/roc_auc',
        mode='max',  # Maximize ROC-AUC
        save_top_k=1,
        save_last=False,
        verbose=True,
        save_weights_only=True,
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
    if is_main:
        print("Starting training...")
        print(f"Task type: {TASK_TYPE}")
        print(f"Number of tasks: {NUM_TASKS}")
        print(f"Model approach: {'LM Head (Yes/No prediction)' if USE_LM_HEAD else 'Classification Head'}")
    
    trainer.fit(model, data_module)
    
    # Test
    if is_main:
        print("\nRunning test evaluation...")
    data_module.setup(stage="test")
    test_results = trainer.test(model, data_module)
    
    # Only log final metrics and save files from main process
    if is_main and mlflow_run:
        # Log training completion metrics
        mlflow.log_metrics({
            "epochs_trained": trainer.current_epoch + 1,
            "stopped_early": int(trainer.current_epoch < EPOCHS - 1),
        })
        
        # Log final test metrics
        for key, value in trainer.callback_metrics.items():
            metric_name = key.replace("/", "_")
            mlflow.log_metric(f"final_{metric_name}", float(value))
        
        # Log best checkpoint as artifact
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt and os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
        
        # Prepare run data for common log file
        run_data = {
            "timestamp": datetime.now().isoformat(),
            "run_id": mlflow.active_run().info.run_id,
            "config": {
                "dataset_name": DATASET_NAME,
                "task_type": TASK_TYPE,
                "num_tasks": NUM_TASKS,
                "use_lm_head": USE_LM_HEAD,
                "model_name": MODEL_NAME,
                "batch_size": BATCH_SIZE,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUM_STEPS,
                "learning_rate": LR,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "max_len": MAX_LEN,
            },
            "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
            "best_checkpoint": best_ckpt,
            "best_val_roc_auc": float(checkpoint_callback.best_model_score or 0),
            "epochs_trained": trainer.current_epoch + 1,
            "stopped_early": trainer.current_epoch < EPOCHS - 1,
        }
        
        # Save to common log file for this dataset
        log_path = get_log_file_path(DATASET_NAME)
        save_run_log(log_path, run_data)
        mlflow.log_artifact(log_path)
        
        # End MLflow run
        mlflow.end_run()
        
        print(f"\nTraining complete!")
        print(f"Best validation ROC-AUC: {checkpoint_callback.best_model_score:.4f}")
        print(f"MLflow run ID: {run_data['run_id']}")
        print(f"Logs appended to: {log_path}")

if __name__ == "__main__":
    main()