chemberta4.trainer
==================

PyTorch Lightning training modules for OLMo fine-tuning.
Supports QLoRA (4-bit), LoRA, and full fine-tuning with DDP scaling.
All trainers use AdamW with linear warmup and cosine annealing.

.. class:: OLMoClassifier(pl.LightningModule)

   Lightning module for single-task and multi-task molecular classification.
   Supports both a linear classification head and Yes/No LM-head prediction.

   .. method:: __init__(model_name="allenai/OLMo-7B-hf", num_tasks=1, task_type="single_task", use_lm_head=False, finetune_strategy="qlora", lr=2e-4, weight_decay=0.01, warmup_ratio=0.1, lora_r=32, lora_alpha=64, lora_dropout=0.05)

      Configures the classifier with model, task, and LoRA hyperparameters.

   .. method:: configure_model()

      Initializes the OLMo backbone and applies LoRA or quantization based on
      ``finetune_strategy``. Called automatically by PyTorch Lightning.

   .. method:: forward(input_ids, attention_mask, labels=None, label_mask=None)

      Returns a ``(logits, loss)`` tuple from the underlying model head.

   .. method:: _shared_step(batch, stage)

      Computes loss, accuracy, and ROC-AUC. Logs metrics with the given stage prefix.

   .. method:: training_step(batch, batch_idx)
   .. method:: validation_step(batch, batch_idx)
   .. method:: test_step(batch, batch_idx)

      Standard PyTorch Lightning step methods delegating to ``_shared_step``.

   .. method:: configure_optimizers()

      Returns AdamW with separate weight decay groups and a cosine annealing
      scheduler with linear warmup.

.. class:: OLMoRegressor(pl.LightningModule)

   Lightning module for molecular property regression with label normalization.
   Trains on z-score normalized labels and reports denormalized RMSE/MAE.

   .. method:: __init__(model_name="allenai/OLMo-7B-hf", finetune_strategy="qlora", lr=2e-4, weight_decay=0.01, warmup_ratio=0.1, lora_r=32, lora_alpha=64, lora_dropout=0.05, label_mean=0.0, label_std=1.0)

      Configures the regressor with model, LoRA, and label normalization hyperparameters.

   .. method:: configure_model()

      Initializes the OLMo backbone with the configured fine-tuning strategy.

   .. method:: forward(input_ids, attention_mask, labels=None)

      Returns a ``(predictions, loss)`` tuple.

   .. method:: _denormalize(values)

      Converts normalized model outputs back to the original label scale.

   .. method:: _shared_step(batch, stage)

      Denormalizes predictions and labels, then computes and logs RMSE and MAE.

   .. method:: training_step(batch, batch_idx)
   .. method:: validation_step(batch, batch_idx)
   .. method:: test_step(batch, batch_idx)

      Standard PyTorch Lightning step methods delegating to ``_shared_step``.

   .. method:: configure_optimizers()

      Returns AdamW with cosine annealing and linear warmup.

.. class:: OLMoPretrainer(pl.LightningModule)

   Lightning module for causal LM pretraining on SMILES datasets or instruction tuning.
   Reports perplexity during validation.

   .. method:: __init__(model_name="allenai/OLMo-7B-hf", finetune_strategy="qlora", lr=1e-4, weight_decay=1e-4, warmup_ratio=0.15, lora_r=64, lora_alpha=128, lora_dropout=0.05, gradient_checkpointing=True)

      Configures the pretrainer with model, LoRA, and gradient checkpointing settings.

   .. method:: configure_model()

      Initializes the causal LM backbone and enables gradient checkpointing if requested.

   .. method:: forward(input_ids, attention_mask, labels=None)

      Returns the model output object with ``.loss`` and ``.logits`` attributes.

   .. method:: training_step(batch, batch_idx)

      Computes causal LM cross-entropy loss and logs per-step loss.

   .. method:: validation_step(batch, batch_idx)

      Computes loss and perplexity (``exp(loss)``).

   .. method:: configure_optimizers()

      Returns AdamW with linear warmup and cosine annealing.


