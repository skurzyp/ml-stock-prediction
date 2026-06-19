# NASDAQ LSTM Price Direction Predictor

Predicts the daily directional movement (UP/DOWN) of the NASDAQ Composite index using an LSTM neural network.

## Data record

Each row in `data/nasdaq_features.json` represents one trading day and contains the following fields:

| Field | Description |
|---|---|
| `Close` | Adjusted closing price |
| `rsi_14` | Relative Strength Index (14-day window) |
| `macd` | MACD line (12/26 EMA difference) |
| `macd_signal` | Signal line (9-day EMA of MACD) |
| `macd_diff` | MACD histogram (macd − signal) |
| `bb_high` | Bollinger Band upper band (20-day, 2σ) |
| `bb_low` | Bollinger Band lower band (20-day, 2σ) |
| `bb_mid` | Bollinger Band middle band (20-day SMA) |
| `bb_pband` | Bollinger Band %B — where Close sits within the band |
| `sma_20` | Simple Moving Average over 20 days |
| `sma_50` | Simple Moving Average over 50 days |
| `atr_14` | Average True Range over 14 days (volatility proxy) |
| `target` | Binary label: `1` if next day's close > today's, `0` otherwise |

Before training, all features (excluding `target`) are standardized to zero mean and unit variance using a `StandardScaler` fitted on the training split only.

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
# Run the complete pipeline (data -> train -> evaluate)
# Outputs (models, charts, metrics) will be saved to a timestamped directory: data/runs/<run_id>/
uv run python scripts/run_pipeline.py

# Alternatively, run stages individually (ensure RUN_ID is set in your environment if you want them grouped)
# uv run python -m src.data_pipeline
# uv run python -m src.train
# uv run python -m src.evaluate
```

## Tests

```bash
uv run pytest
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

- **Local (CPU):** Good for development and quick smoke-tests.
- **Kaggle (GPU):** Use for full training runs (~30 hours/week free GPU).

To run on Kaggle:

`src/` ships as a private Kaggle Dataset (`<username>/lstm-stock-prediction-src`) that the kernel mounts at `/kaggle/input/`. The entry point [kaggle_runner.py](file:///Users/stanislawkurzyp/Documents/private/topol/lstm-stock-prediction/kaggle_runner.py) puts that path on `sys.path` and runs each stage as `python -m src.<stage>`, so the code that runs on Kaggle is byte-identical to what you run locally.

1. **Set up Kaggle API Token:** Make sure `~/.kaggle/kaggle.json` is in place (see instructions above).
2. **Configure Username (one-time):** In [kernel-metadata.json](file:///Users/stanislawkurzyp/Documents/private/topol/lstm-stock-prediction/kernel-metadata.json), replace both occurrences of `YOUR_KAGGLE_USERNAME` (in `id` and `dataset_sources`) with your Kaggle username.
3. **Deploy:** Stage the dataset, version-or-create it, and push the kernel in one command:
   ```bash
   uv run python scripts/deploy_kaggle.py -m "what changed"
   ```
4. **View Status/Logs:**
   ```bash
   uv run kaggle kernels status
   uv run kaggle kernels output
   ```
