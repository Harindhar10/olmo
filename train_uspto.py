import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
# HF_TOKEN = "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"

MODEL_ID = "harindhar10/OLMo-7B-ZINC20-10k" 

NEW_ADAPTER_NAME = "harindhar10/OLMo-7B-USPTO-1k-ZINC"


class OLMoQLoRA(pl.LightningModule):
    def __init__(self, model_id, adapter_name):
        super().__init__()
        self.save_hyperparameters()
        self.model_id = model_id
        self.adapter_name = adapter_name
        
        # OLMo requires trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def configure_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        # QLoRA Config
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, peft_config)
        
        if self.trainer.is_global_zero:
            self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"] 
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(0.15 * total_steps)
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
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
                "interval": "step"
            }
        }


class USPTODataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = (
            load_dataset("OpenMol/USPTO_1k_TPL-SFT", split="train", streaming=True, trust_remote_code=True)
            .take(10_000)
        )

    def train_dataloader(self):
        def collate_fn(batch):
            text_samples = []
            for item in batch:

                prompt = (
                    f"Instruction: {item['instruction']}\n"
                    f"Input: {item['input']}\n"
                    f"Output: {item['output']}"
                )
                text_samples.append(prompt)
            
            encodings = self.tokenizer(
                text_samples,
                truncation=True,
                max_length=512, 
                padding="max_length",
                return_tensors="pt"
            )

            labels = encodings["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels
            }

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

if __name__ == "__main__":
    pl_model = OLMoQLoRA(MODEL_ID, NEW_ADAPTER_NAME)

    dm = USPTODataModule(pl_model.tokenizer, batch_size=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,                  
        strategy="ddp", 
        precision="16-mixed",       
        max_epochs=1,
        accumulate_grad_batches=4,  
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=True,
        gradient_clip_val=0.5
    )

    print(f"Starting Training on USPTO dataset (using base: {MODEL_ID})...")
    trainer.fit(pl_model, datamodule=dm)

    # ---------------------------------------------------------
    # MERGE AND PUSH LOGIC
    # ---------------------------------------------------------
    if trainer.is_global_zero:
        print("Training complete. Saving adapter to temporary local storage...")
        
        pl_model.model.save_pretrained("./temp_adapter_uspto")
        pl_model.tokenizer.save_pretrained("./temp_adapter_uspto")
        
        print("Freeing VRAM for merge process...")
        del pl_model
        del trainer
        del dm
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading Base Model in FP16 for merging...")
        # NOTE: We load the ZINC-trained model as the base here
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print("Loading Adapter and Merging...")
        model_to_merge = PeftModel.from_pretrained(base_model, "./temp_adapter_uspto")
        model_to_merge = model_to_merge.merge_and_unload()
        
        print(f"Pushing FULL MERGED model to Hub: {NEW_ADAPTER_NAME}")
        model_to_merge.push_to_hub(NEW_ADAPTER_NAME)
        
        # Push tokenizer as well
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.push_to_hub(NEW_ADAPTER_NAME)
        
        print("Done! Full merged USPTO model pushed.")
