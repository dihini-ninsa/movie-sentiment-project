"""
Step 4: Fine-tune DistilBERT with LoRA (PEFT)
-------------------------------------------------
This is the "efficiency" showcase piece of the project, mirroring your
food-review project: fine-tune a full transformer for sentiment
classification, but only train small LoRA adapter weights instead of the
whole model. Result: near-full-model accuracy, tiny saved adapter size.
 
Notes:
- Runs on CPU but will be slow. A GPU (even a free Colab T4) will be much
  faster for this step - consider running this particular script there if
  you don't have a local GPU.
- The adapter (not the full model) gets saved - that's what gives you the
  "260MB -> 3MB" style size reduction story for your write-up.
 
Run:
    python train_lora.py
"""
 
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
 
INPUT_CSV = "../data/reviews_labeled.csv"
BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = "../data/lora_sentiment_model"
 
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
 
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
 
 
def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["review_text", "sentiment"])
    df["label"] = df["sentiment"].map(LABEL2ID)
 
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
 
    train_ds = Dataset.from_pandas(train_df[["review_text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["review_text", "label"]].reset_index(drop=True))
 
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
 
    def tokenize(batch):
        return tokenizer(batch["review_text"], truncation=True, padding="max_length", max_length=256)
 
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
 
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
 
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
 
    # --- LoRA configuration ---
    # target_modules for DistilBERT's attention projection layers.
    # r = rank of the low-rank update matrices (higher = more capacity, bigger adapter)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # query/value projections in DistilBERT attention
    )
 
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # shows how few params are actually being trained
 
    training_args = TrainingArguments(
        output_dir="../data/lora_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,      # LoRA typically wants a higher LR than full fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
 
    print("Starting training...")
    trainer.train()
 
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)
 
    # Save only the LoRA adapter (this is the small ~MB file, not the full base model)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved LoRA adapter + tokenizer to {OUTPUT_DIR}")
 
    import os
    adapter_file = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        size_mb = os.path.getsize(adapter_file) / (1024 * 1024)
        print(f"Adapter size: {size_mb:.2f} MB (compare this to the ~260MB base model)")
 
 
if __name__ == "__main__":
    main()