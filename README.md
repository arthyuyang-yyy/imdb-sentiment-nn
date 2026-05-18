# imdb-sentiment-nn

Lightweight IMDB sentiment analysis project for an ML CI/CD assignment.

This project evaluates the HuggingFace pretrained sentiment model
`textattack/bert-base-uncased-imdb` on the first 50 reviews from
`data/imdb_balanced_10k.csv`.

## Project Structure

```text
.
├── data/
│   └── imdb_balanced_10k.csv
├── .github/
│   └── workflows/
│       └── train-and-upload.yml
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No GPU is required. The scripts force CPU inference.

## Evaluate

```bash
python train.py
```

The evaluator reads `data/imdb_balanced_10k.csv`, automatically detects the text
and label columns when possible, evaluates only the first 50 non-empty reviews,
and writes metrics to `metrics.json` and `model/metrics.json`.

Console output format:

```text
model name: textattack/bert-base-uncased-imdb
number of evaluated samples: 50
accuracy: 0.0000
precision: 0.0000
recall: 0.0000
f1: 0.0000
```

`metrics.json` format:

```json
{
  "model_name": "textattack/bert-base-uncased-imdb",
  "evaluated_samples": 50,
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0
}
```

Generated artifacts:

- `metrics.json`
- `model/config.json`
- `model/metrics.json`

## Predict

```bash
python predict.py "This movie was excellent and very moving."
```

Example output:

```text
positive
```

## CI/CD

The GitHub Actions workflow runs on:

- push to `main`
- push to `master`
- manual `workflow_dispatch`

The workflow:

1. Installs Python dependencies.
2. Runs `python train.py`.
3. Uploads generated metrics artifacts to Hugging Face Hub when `HF_TOKEN` is configured.

If `HF_TOKEN` is not configured, the upload step prints a skip message and exits
successfully so GitHub Actions does not fail.

Optional GitHub repository secret for uploads:

```text
HF_TOKEN
```

Hugging Face repo:

```text
arthyuyang-2026/imdb-sentiment-nn
```
