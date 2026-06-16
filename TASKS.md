# NASDAQ LSTM — Task Checklist

## Phase 0 — Environment Setup

- [x] 0.1 `uv init --no-readme` → `pyproject.toml` scaffolded
- [x] 0.2 `uv add torch yfinance pandas ta scikit-learn`
- [x] 0.3 `uv add --dev black`
- [x] 0.4 Create `data/` and `src/` directories
- [x] 0.5 Create `src/config.py` with all hyperparameters and paths
- [ ] 0.6 Verify CUDA: `uv run python -c "import torch; print(torch.cuda.is_available())"`

## Phase 1 — Data Pipeline (`src/data_pipeline.py`)

- [x] 1.1 `download_raw_data(ticker, start, end) -> pd.DataFrame` via `yfinance`
- [x] 1.2 `add_technical_indicators(df) -> pd.DataFrame` via `ta` (RSI, MACD, BB, SMA20/50, ATR)
- [x] 1.3 `create_target(df) -> pd.DataFrame` — binary UP/DOWN label from next-day close
- [x] 1.4 `clean_data(df) -> pd.DataFrame` — `dropna()`, log rows dropped
- [x] 1.5 `save_to_json(df, path)` — serialize to `data/nasdaq_features.json`
- [x] 1.6 `__main__` block — run full pipeline, confirm >2000 rows saved

## Phase 2 — Dataset & DataLoader (`src/dataset.py`)

- [ ] 2.1 `chronological_split(df, train_ratio, val_ratio) -> (train, val, test)` — no shuffling
- [ ] 2.2 `fit_scaler(train_df, feature_cols) -> scaler` — `StandardScaler` fit on train only
- [ ] 2.3 `transform_split(df, scaler, feature_cols) -> np.ndarray`
- [ ] 2.4 `NASDAQSequenceDataset(Dataset)` — `__getitem__` returns `(X: (60, F), y: scalar)`
- [ ] 2.5 `build_dataloaders(train_ds, val_ds, test_ds, batch_size)` — train shuffles, val/test don't

## Phase 3 — Model (`src/model.py`)

- [ ] 3.1 `LSTMClassifier(nn.Module)` — LSTM + Dropout + Linear(1); returns raw logit
- [ ] 3.2 Shape sanity-check: dummy `(4, 60, F)` input → output `(4,)`

## Phase 4 — Training Loop (`src/train.py`)

- [ ] 4.1 `train_one_epoch(model, loader, optimizer, criterion, device) -> float`
- [ ] 4.2 `evaluate_loss(model, loader, criterion, device) -> float`
- [ ] 4.3 Early stopping — track best val loss, restore weights after `patience` epochs
- [ ] 4.4 Main training loop — log `train_loss / val_loss` per epoch, save best weights
- [ ] 4.5 `__main__` block — smoke-test 5 epochs, confirm no crash

## Phase 5 — Evaluation (`src/evaluate.py`)

- [ ] 5.1 `predict(model, loader, device) -> (preds, targets)` — sigmoid → threshold 0.5
- [ ] 5.2 `compute_metrics(preds, targets, probs) -> dict` — Accuracy, F1, ROC-AUC
- [ ] 5.3 `naive_baseline_metrics(targets) -> dict` — always-UP strategy
- [ ] 5.4 `__main__` block — load weights, print comparison table (Model vs Naive Baseline)

## Phase 6 — Housekeeping

- [ ] 6.1 `uv run black src/` — format all source files
- [x] 6.2 `.gitignore` — add `data/`, `.venv/`, `__pycache__/`
- [ ] 6.3 Update `README.md` with spec + quick-start commands
