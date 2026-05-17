import argparse
import json
import pickle
from pathlib import Path

import joblib


MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"


def load_label_mapping() -> dict:
    if not CONFIG_PATH.exists():
        return {"0": "negative", "1": "positive"}

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("label_mapping", {"0": "negative", "1": "positive"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict IMDB review sentiment.")
    parser.add_argument("review", nargs="+", help="Review text to classify.")
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python train.py` first.")

    review_text = " ".join(args.review)

    model = joblib.load(MODEL_PATH)
    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)

    label_mapping = load_label_mapping()
    features = vectorizer.transform([review_text])
    prediction = model.predict(features)[0]
    sentiment = label_mapping.get(str(prediction), str(prediction))

    print(sentiment)


if __name__ == "__main__":
    main()
