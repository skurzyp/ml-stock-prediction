# NASDAQ LSTM Price Direction Predictor

Predicts the daily directional movement (UP/DOWN) of the NASDAQ Composite index using an LSTM neural network.

## Prerequisites

- **uv** — [installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Kaggle account** — needed for GPU training ([setup guide](https://www.kaggle.com/docs/api))

## Setup

```bash
# 1. Clone the repo and enter the project
git clone git@github.com:skurzyp/ml-stock-prediction.git
cd lstm-stock-prediction

# 2. Install dependencies (uv creates the virtualenv automatically)
uv sync
```

### Kaggle API token

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Running the project

All commands are run via `uv run` — no need to activate a virtualenv manually.

```bash
# Step 1: Download data and engineer features → data/nasdaq_features.json
uv run python src/data_pipeline.py

# Step 2: Train the model → data/best_model.pt
uv run python src/train.py

# Step 3: Evaluate on the held-out test set
uv run python src/evaluate.py
```

## Code quality

**Format** with black:
```bash
uv run black src/
```

**Type-check** with pyright (basic mode):
```bash
uv run pyright
```

**Pre-commit hooks** run both automatically on every `git commit`. Install once after cloning:
```bash
uv run pre-commit install
```

To run the hooks manually without committing:
```bash
uv run pre-commit run --all-files
```

## Local vs. Kaggle

The code automatically uses GPU if available (`cuda`) and falls back to CPU otherwise.

- **Local (CPU):** good for development and quick smoke-tests
- **Kaggle (GPU):** use for full training runs (~30h/week free)

```bash
# Push code to Kaggle for GPU training
uv run kaggle kernels push
```
