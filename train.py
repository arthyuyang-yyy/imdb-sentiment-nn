import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_PATH = Path("data/imdb_balanced_10k.csv")
METRICS_PATH = Path("metrics.json")
MODEL_DIR = Path("model")
MODEL_NAME = "textattack/bert-base-uncased-imdb"
EVALUATION_LIMIT = 50
BATCH_SIZE = 8


def detect_text_column(df: pd.DataFrame) -> str:
    preferred = ["text", "review", "content", "sentence", "comment"]
    lower_to_original = {column.lower(): column for column in df.columns}

    for name in preferred:
        if name in lower_to_original:
            return lower_to_original[name]

    object_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ]
    if not object_columns:
        raise ValueError("Could not detect a text column. Expected a text/review column.")

    return max(object_columns, key=lambda column: df[column].astype(str).str.len().mean())


def detect_label_column(df: pd.DataFrame, text_column: str) -> str:
    preferred = ["label", "sentiment", "target", "class", "rating"]
    lower_to_original = {column.lower(): column for column in df.columns}

    for name in preferred:
        if name in lower_to_original and lower_to_original[name] != text_column:
            return lower_to_original[name]

    candidates = []
    for column in df.columns:
        if column == text_column:
            continue
        unique_count = df[column].dropna().nunique()
        if 2 <= unique_count <= 10:
            candidates.append((unique_count, column))

    if not candidates:
        raise ValueError("Could not detect a label column. Expected a binary sentiment label.")

    candidates.sort()
    return candidates[0][1]


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "positive", "pos", "true", "yes", "good"}
    negative_values = {"0", "negative", "neg", "false", "no", "bad"}

    unique_values = sorted(cleaned.dropna().unique().tolist())
    if set(unique_values).issubset(positive_values | negative_values):
        mapped = cleaned.map(lambda value: 1 if value in positive_values else 0)
        return mapped.astype(int)

    if len(unique_values) != 2:
        raise ValueError(f"Expected binary labels, found {len(unique_values)} unique values.")

    label_to_int = {unique_values[0]: 0, unique_values[1]: 1}
    return cleaned.map(label_to_int).astype(int)


def model_label_to_int(label: str) -> int:
    normalized = label.strip().lower()
    positive_labels = {"1", "label_1", "positive", "pos"}
    negative_labels = {"0", "label_0", "negative", "neg"}

    if normalized in positive_labels:
        return 1
    if normalized in negative_labels:
        return 0

    raise ValueError(f"Unsupported model label: {label}")


def predict_sentiments(texts: list[str]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()

    id_to_label = model.config.id2label
    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            predicted_ids = outputs.logits.argmax(dim=-1).tolist()
            predictions.extend(
                model_label_to_int(id_to_label[predicted_id])
                for predicted_id in predicted_ids
            )

    return predictions


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    text_column = detect_text_column(df)
    label_column = detect_label_column(df, text_column)

    df = df[[text_column, label_column]].dropna()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""].head(EVALUATION_LIMIT)

    if df.empty:
        raise ValueError("No reviews available for evaluation.")

    labels = normalize_binary_labels(df[label_column])
    texts = df[text_column].tolist()
    predictions = predict_sentiments(texts)

    metrics = {
        "model_name": MODEL_NAME,
        "evaluated_samples": int(len(df)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }

    config = {
        "dataset": str(DATA_PATH),
        "text_column": text_column,
        "label_column": label_column,
        "model_name": MODEL_NAME,
        "evaluated_samples": int(len(df)),
        "evaluation_limit": EVALUATION_LIMIT,
        "device": "cpu",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (MODEL_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"model name: {metrics['model_name']}")
    print(f"number of evaluated samples: {metrics['evaluated_samples']}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
