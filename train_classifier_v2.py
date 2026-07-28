import json
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["simple", "medium", "complex"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

def load_jsonl(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {
        "text": [r["query"] for r in rows],
        "label": [LABEL2ID[r["label"]] for r in rows],
    }

train_data = load_jsonl("train_v2.jsonl")
val_data = load_jsonl("val_v2.jsonl")

train_ds = Dataset.from_dict(train_data)
val_ds = Dataset.from_dict(val_data)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

training_args = TrainingArguments(
    output_dir="./classifier_model_v2",
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nFinal evaluation:")
print(trainer.evaluate())

trainer.save_model("./classifier_model_v2/best")
tokenizer.save_pretrained("./classifier_model_v2/best")
print("\nModel saved to ./classifier_model_v2/best")
