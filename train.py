import json
import pickle
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


DATA_PATH = Path("data/imdb_balanced_10k.csv")
MODEL_DIR = Path("model")
RANDOM_STATE = 42


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


def normalize_binary_labels(series: pd.Series) -> tuple[pd.Series, dict]:
    cleaned = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "positive", "pos", "true", "yes", "good"}
    negative_values = {"0", "negative", "neg", "false", "no", "bad"}

    unique_values = sorted(cleaned.dropna().unique().tolist())
    if set(unique_values).issubset(positive_values | negative_values):
        mapped = cleaned.map(lambda value: 1 if value in positive_values else 0)
        return mapped.astype(int), {"0": "negative", "1": "positive"}

    if len(unique_values) != 2:
        raise ValueError(f"Expected binary labels, found {len(unique_values)} unique values.")

    label_to_int = {unique_values[0]: 0, unique_values[1]: 1}
    int_to_label = {str(value): label for label, value in label_to_int.items()}
    return cleaned.map(label_to_int).astype(int), int_to_label


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    text_column = detect_text_column(df)
    label_column = detect_label_column(df, text_column)

    df = df[[text_column, label_column]].dropna()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]

    labels, label_mapping = normalize_binary_labels(df[label_column])
    texts = df[text_column]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20_000,
        min_df=2,
        ngram_range=(1, 2),
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(x_train_tfidf, y_train)

    predictions = model.predict(x_test_tfidf)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    config = {
        "dataset": str(DATA_PATH),
        "text_column": text_column,
        "label_column": label_column,
        "label_mapping": label_mapping,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": 20_000,
            "min_df": 2,
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
        "model": {
            "type": "MLPClassifier",
            "hidden_layer_sizes": [64],
            "max_iter": 30,
            "early_stopping": True,
        },
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    with (MODEL_DIR / "vectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)
    with (MODEL_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Training complete")
    print(f"Text column: {text_column}")
    print(f"Label column: {label_column}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
