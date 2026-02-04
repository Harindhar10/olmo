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
EPOCHS = 30
PATIENCE = 7
MIN_DELTA = 0.0001
WARMUP_RATIO = 0.1
NUM_WORKERS = 4

# Dataset name
DATASET_NAME = 'freesolv'
model_name = MODEL_NAME
#-------------------------------------------------------
class OLMoRegression(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
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


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

# Load base model with optimized device mapping
base_model = AutoModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map= {"": "cuda:0"}  # Change this based on your setup
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

base_model.print_trainable_parameters()

model = OLMoRegression(base_model)

