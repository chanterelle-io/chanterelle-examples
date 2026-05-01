import json
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)


MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = Path("imdb_distilbert_model")
MODEL_INFO_PATH = Path("model_info.json")
DEFAULT_EPOCHS = 2


def get_runtime_config() -> dict:
    force_cpu = os.getenv("IMDB_TRAIN_USE_CPU", "0") == "1"
    mps_available = torch.backends.mps.is_available() and not force_cpu
    cuda_available = torch.cuda.is_available() and not force_cpu

    if mps_available:
        return {
            "device": "mps",
            "max_length": 256,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "use_cpu": False,
        }

    if cuda_available:
        return {
            "device": "cuda",
            "max_length": 512,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "use_cpu": False,
        }

    return {
        "device": "cpu",
        "max_length": 256,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "use_cpu": True,
    }


def compute_metrics_builder():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics


def build_training_args(runtime_config: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=runtime_config["train_batch_size"],
        per_device_eval_batch_size=runtime_config["eval_batch_size"],
        gradient_accumulation_steps=runtime_config["gradient_accumulation_steps"],
        num_train_epochs=DEFAULT_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        use_cpu=runtime_config["use_cpu"],
    )


def main() -> None:
    runtime_config = get_runtime_config()

    print("Loading IMDb dataset...")
    imdb = load_dataset("imdb")

    print(f"Training examples: {len(imdb['train'])}")
    print(f"Test examples: {len(imdb['test'])}")
    print(
        "Training configuration: "
        f"device={runtime_config['device']}, "
        f"max_length={runtime_config['max_length']}, "
        f"train_batch_size={runtime_config['train_batch_size']}, "
        f"eval_batch_size={runtime_config['eval_batch_size']}, "
        f"gradient_accumulation_steps={runtime_config['gradient_accumulation_steps']}"
    )

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {label: index for index, label in id2label.items()}

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=runtime_config["max_length"],
        )

    print("Tokenizing dataset...")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = build_training_args(runtime_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
    )

    print("Training model...")
    trainer.train()

    print("Evaluating model...")
    evaluation_results = trainer.evaluate()

    print("Saving model and tokenizer...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    classifier = pipeline(
        "sentiment-analysis",
        model=str(OUTPUT_DIR),
        tokenizer=str(OUTPUT_DIR),
    )

    test_reviews = [
        "This movie was absolutely fantastic! Great acting and amazing plot.",
        "Terrible movie. Waste of time. Poor acting and boring storyline.",
        "The movie was okay. Not great but not terrible either.",
    ]
    example_predictions = classifier(test_reviews)

    model_info = {
        "model_type": "IMDB Sentiment Classifier (Hugging Face DistilBERT)",
        "architecture": "DistilBERT sequence classification",
        "base_model": MODEL_NAME,
        "dataset": "imdb",
        "input": "raw_text",
        "output": "sentiment (NEGATIVE or POSITIVE)",
        "training_args": {
            "epochs": DEFAULT_EPOCHS,
            "device": runtime_config["device"],
            "train_batch_size": runtime_config["train_batch_size"],
            "eval_batch_size": runtime_config["eval_batch_size"],
            "gradient_accumulation_steps": runtime_config["gradient_accumulation_steps"],
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_length": runtime_config["max_length"],
        },
        "evaluation": evaluation_results,
        "example_predictions": example_predictions,
    }

    with MODEL_INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)

    print("Final evaluation:")
    for key, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nExample predictions:")
    for review, prediction in zip(test_reviews, example_predictions):
        print(f"\nReview: {review}")
        print(
            f"Prediction: {prediction['label']} "
            f"(confidence: {prediction['score']:.3f})"
        )

    print(f"\nModel saved to: {OUTPUT_DIR.resolve()}")
    print(f"Metadata saved to: {MODEL_INFO_PATH.resolve()}")


if __name__ == "__main__":
    main()