chemberta4.model
================

Lightweight task-specific model wrappers around an OLMo backbone with LoRA.
Each wrapper handles its own output head and loss computation.

.. function:: last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor

   Extracts the last non-padding token's hidden representation from a sequence.
   Used for decoder-only models (like OLMo) to obtain a fixed-size sentence embedding
   for classification or regression.

.. class:: ClassificationHead(nn.Module)

   Classification wrapper for single-task and multi-task molecular classification.
   Applies a linear projection on the last-token representation.
   Uses ``CrossEntropyLoss`` for single-task and ``BCEWithLogitsLoss`` for multi-task.

   .. method:: __init__(backbone, num_tasks=1, task_type="single_task")

      Initializes the classification head over the given OLMo backbone.

   .. method:: forward(input_ids, attention_mask, labels=None, label_mask=None)

      Runs the forward pass and returns a ``(logits, loss)`` tuple.
      Loss is ``None`` when labels are not provided.

   .. method:: _compute_loss(logits, labels, label_mask=None)

      Internal loss computation. Applies label masking for multi-task missing values.

.. class:: CausalLMClassificationHead(nn.Module)

   Classification via Yes/No token prediction from the causal LM head.
   Rather than a linear classifier, extracts logits for ``"Yes"`` and ``"No"`` tokens
   at the last sequence position. Suitable for instruction-style prompting.

   .. method:: __init__(model, tokenizer, num_tasks=1, task_type="single_task")

      Initializes the LM-head classifier with the given model and tokenizer.

   .. method:: forward(input_ids, attention_mask, labels=None, label_mask=None)

      Extracts Yes/No logits at the last token position and returns a ``(logits, loss)`` tuple.
      In multi-task mode, projects the Yes−No score difference to ``num_tasks`` outputs.

   .. method:: _compute_loss(logits, labels, label_mask=None)

      Mirrors ``ClassificationHead._compute_loss`` with label masking support.

.. class:: RegressionHead(nn.Module)

   Regression wrapper using last-token pooling and RMSE loss.

   .. method:: __init__(backbone)

      Initializes the regression head over the given OLMo backbone.

   .. method:: forward(input_ids, attention_mask, labels=None)

      Runs the forward pass and returns a ``(predictions, loss)`` tuple.
      Loss is RMSE if labels are provided, else ``None``.
