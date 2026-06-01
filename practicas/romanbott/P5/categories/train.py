"""
Práctica 5: Fine-tuning y puesta en producción de modelos
Tarea: Clasificación temática de abstracts (Seq2Seq)
Modelo Base: google/flan-t5-base (Optimizado para 8GB VRAM AMD RX 7600)
"""

import os
import csv
import numpy as np
from datetime import datetime
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

MODEL_CHECKPOINT = "google/flan-t5-base"
DATASET_NAME = "gfissore/arxiv-abstracts-2021"
OUTPUT_DIR = "./modelo_arxiv_categorias_final"
CSV_LOG_FILE = "./training_metrics.csv"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 16
EPOCHS = 5

TRAIN_SAMPLES = 50000
EVAL_SAMPLES = 500
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4


class CSVLogCallback(TrainerCallback):
    """Intercepta la evaluación y escribe las métricas de clasificación en un CSV."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["datetime", "step", "epoch", "eval_loss", "accuracy", "f1_macro"]
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        state.global_step,
                        round(state.epoch or 0, 2),
                        round(metrics.get("eval_loss", 0), 4),
                        metrics.get("eval_accuracy", ""),
                        metrics.get("eval_f1_macro", ""),
                    ]
                )


def load_and_prepare_data():
    print(f"Descargando dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")

    total_samples = TRAIN_SAMPLES + EVAL_SAMPLES
    subset = dataset.select(range(total_samples))
    splits = subset.train_test_split(test_size=EVAL_SAMPLES, seed=42)

    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def get_preprocess_function(tokenizer):
    def preprocess_function(examples):
        inputs = ["classify abstract: " + str(doc) for doc in examples["abstract"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

        primary_categories = []
        for cat in examples["categories"]:
            if isinstance(cat, list) and len(cat) > 0:
                primary_categories.append(str(cat[0]).split()[0])
            else:
                primary_categories.append(str(cat).split()[0])

        labels = tokenizer(
            text_target=primary_categories,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def get_compute_metrics_function(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
        accuracy = (
            exact_matches / len(decoded_labels) if len(decoded_labels) > 0 else 0.0
        )

        f1_macro = f1_score(
            decoded_labels, decoded_preds, average="macro", zero_division=0
        )

        return {"accuracy": round(accuracy, 4), "f1_macro": round(f1_macro, 4)}

    return compute_metrics


def main():
    print(f"Cargando {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    model.config.tie_word_embeddings = False

    raw_datasets = load_and_prepare_data()
    preprocess_fn = get_preprocess_function(tokenizer)

    print("Tokenizando el dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_fn, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    effective_batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = TRAIN_SAMPLES // effective_batch_size
    total_steps = steps_per_epoch * EPOCHS
    eval_steps_freq = max(1, total_steps // 10)

    print(
        f"Total steps estimados: {total_steps}. Evaluando y guardando cada {eval_steps_freq} steps."
    )

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=eval_steps_freq,
        save_strategy="steps",
        save_steps=eval_steps_freq,
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        weight_decay=0.01,
        save_total_limit=10,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=False,
        bf16=True,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics_function(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            CSVLogCallback(CSV_LOG_FILE),
        ],
    )

    print("Ejecutando evaluación inicial ...")
    baseline_metrics = trainer.evaluate()
    print(f"Resultados iniciales (Step 0): {baseline_metrics}")

    print("Iniciando Fine-Tuning de clasificación...")
    trainer.train()

    print("Evaluando modelo final...")
    eval_results = trainer.evaluate()
    print(f"Resultados finales: {eval_results}")

    print(f"Guardando modelo en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Entrenamiento completado. Métricas en: {CSV_LOG_FILE}")


if __name__ == "__main__":
    main()
