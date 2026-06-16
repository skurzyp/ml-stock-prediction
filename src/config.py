TICKER = "^IXIC"
START_DATE = "2016-01-01"
END_DATE = None

SEQ_LEN = 60

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10

DATA_PATH = "data/nasdaq_features.json"
MODEL_PATH = "data/best_model.pt"

FEATURE_COLS = [
    "Close",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high",
    "bb_low",
    "bb_mid",
    "bb_pband",
    "sma_20",
    "sma_50",
    "atr_14",
]
