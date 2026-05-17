# imdb-sentiment-nn

Lightweight IMDB sentiment analysis project for an ML CI/CD assignment.

This project trains a feedforward neural network using only scikit-learn:

- TF-IDF text features with `TfidfVectorizer`
- Neural network classifier with `MLPClassifier`
- CPU-only training
- GitHub Actions workflow for training and uploading artifacts to Hugging Face Hub

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

## Train

```bash
python train.py
```

Training reads `data/imdb_balanced_10k.csv`, detects the text and label columns, trains the TF-IDF vectorizer and MLP model, evaluates the model, and writes artifacts to `model/`.

Generated artifacts:

- `model/model.joblib`
- `model/vectorizer.pkl`
- `model/config.json`
- `model/metrics.json`

## Predict

After training:

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
3. Creates the Hugging Face Hub repo if needed.
4. Uploads the generated model artifacts.

Set this GitHub repository secret before running the upload step:

```text
HF_TOKEN
```

Hugging Face repo:

```text
arthyuyang-ai/imdb-sentiment-nn
```

## Notes

- PyTorch is not used.
- TensorFlow is not used.
- No GPU is required.
- The model is intentionally small so it can run reliably in GitHub Actions.
