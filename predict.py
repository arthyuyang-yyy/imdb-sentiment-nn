import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "textattack/bert-base-uncased-imdb"


def model_label_to_sentiment(label: str) -> str:
    normalized = label.strip().lower()
    positive_labels = {"1", "label_1", "positive", "pos"}
    negative_labels = {"0", "label_0", "negative", "neg"}

    if normalized in positive_labels:
        return "positive"
    if normalized in negative_labels:
        return "negative"

    return label


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict IMDB review sentiment.")
    parser.add_argument("review", nargs="+", help="Review text to classify.")
    args = parser.parse_args()

    review_text = " ".join(args.review)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()

    inputs = tokenizer(
        [review_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_id = int(outputs.logits.argmax(dim=-1).item())

    print(model_label_to_sentiment(model.config.id2label[predicted_id]))


if __name__ == "__main__":
    main()
