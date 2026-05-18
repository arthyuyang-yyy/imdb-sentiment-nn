import argparse
import json
import pickle
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_PATH = Path("data/imdb_balanced_10k.csv")
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
MODEL_PATH = Path("model.pt")
VECTORIZER_PATH = Path("vectorizer.pkl")
CONFIG_PATH = Path("config.json")
METRICS_PATH = Path("metrics.json")
MODEL_DIR = Path("model")
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_FEATURES = 10_000
DEFAULT_HIDDEN_DIM = 128
DEFAULT_TEST_SIZE = 0.2
SEED = 42


class SparseTextDataset(Dataset):
    def __init__(self, features, labels: pd.Series) -> None:
        self.features = features
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.features[index].toarray().squeeze(0)
        return torch.tensor(row, dtype=torch.float32), self.labels[index]


class SentimentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


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


def load_dataset(path: Path) -> tuple[list[str], pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {columns}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""]

    if df.empty:
        raise ValueError("No reviews available for training.")

    texts = df[TEXT_COLUMN].tolist()
    labels = normalize_binary_labels(df[LABEL_COLUMN])
    return texts, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1).cpu().tolist()

            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "precision": float(precision_score(all_labels, all_predictions, zero_division=0)),
        "recall": float(recall_score(all_labels, all_predictions, zero_division=0)),
        "f1": float(f1_score(all_labels, all_predictions, zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP sentiment classifier.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, labels = load_dataset(args.data_path)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=SEED,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    eval_features = vectorizer.transform(eval_texts)

    train_dataset = SparseTextDataset(train_features, train_labels.reset_index(drop=True))
    eval_dataset = SparseTextDataset(eval_features, eval_labels.reset_index(drop=True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    model = SentimentMLP(input_dim=train_features.shape[1], hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(float(train_loss))
        print(f"epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}")

    metrics = evaluate(model, eval_loader, device)
    metrics.update(
        {
            "train_samples": int(len(train_dataset)),
            "evaluated_samples": int(len(eval_dataset)),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "train_loss": float(train_losses[-1]),
        }
    )

    config = {
        "dataset": str(args.data_path),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "model_type": "tfidf_mlp",
        "vectorizer": "TfidfVectorizer",
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "input_dim": int(train_features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "num_classes": 2,
        "max_features": int(args.max_features),
        "test_size": float(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "seed": SEED,
        "device": str(device),
    }

    torch.save(model.state_dict(), MODEL_PATH)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"train samples: {metrics['train_samples']}")
    print(f"number of evaluated samples: {metrics['evaluated_samples']}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
