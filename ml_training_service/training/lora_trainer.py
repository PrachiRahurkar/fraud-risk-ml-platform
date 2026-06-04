"""
LoRA fine-tuning of Gemma-2B on fundraiser text for fraud classification.
Runs on GCP Compute Engine T4 GPU VM via Ray Train.
"""
import os
import logging
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "auc_roc": roc_auc_score(labels, probs),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def build_lora_model(base_model: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def tokenize_dataset(records: list[dict], tokenizer, max_length: int = 512) -> Dataset:
    ds = Dataset.from_list(records)

    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return ds.map(tokenize, batched=True, remove_columns=["input_text", "fund_id"])


def train(
    train_records: list[dict],
    eval_records: list[dict],
    base_model: str = "google/gemma-2b",
    output_dir: str = "models/lora-adapter",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    learning_rate: float = 2e-4,
    num_epochs: int = 10,
    batch_size: int = 8,
    max_length: int = 512,
    mlflow_tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "fraud-lora",
) -> str:
    """Fine-tune Gemma-2B with LoRA. Returns path to saved adapter."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    model, tokenizer = build_lora_model(base_model, lora_r, lora_alpha, lora_dropout)
    train_ds = tokenize_dataset(train_records, tokenizer, max_length)
    eval_ds = tokenize_dataset(eval_records, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc_roc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    with mlflow.start_run():
        mlflow.log_params({
            "base_model": base_model,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "train_size": len(train_records),
            "eval_size": len(eval_records),
        })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        eval_results = trainer.evaluate()
        mlflow.log_metrics({k.replace("eval_", ""): v for k, v in eval_results.items()})

        adapter_path = str(Path(output_dir) / "final-adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        mlflow.log_artifact(adapter_path, artifact_path="lora-adapter")
        logger.info("LoRA adapter saved to %s", adapter_path)

    return adapter_path
