import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score


def compute_metrics(pred):
    preds = torch.argmax(torch.tensor(pred.predictions), axis=1)
    labels = torch.tensor(pred.label_ids)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def apply_pruning(model, amount=0.4):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def apply_quantization(model):
    model.cpu()
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def train_model(model, dataset, tokenizer):
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()

dataset = load_dataset("glue", "sst2").select(range(2000))
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True)

dataset = dataset.map(tokenize_fn, batched=True).remove_columns(["sentence", "idx"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
base_metrics = train_model(model, dataset, tokenizer)

pruned_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
pruned_model = apply_pruning(pruned_model)
pruned_metrics = train_model(pruned_model, dataset, tokenizer)

quant_model = apply_quantization(pruned_model)
quant_metrics = train_model(quant_model, dataset, tokenizer)

results = pd.concat([
    pd.DataFrame([{"Model": "Original", **base_metrics}]),
    pd.DataFrame([{"Model": "Pruned", **pruned_metrics}]),
    pd.DataFrame([{"Model": "Pruned+Quantized", **quant_metrics}]),
], ignore_index=True)

print("\n=== SST-2 Evaluation Results ===")
print(results.to_string(index=False))
