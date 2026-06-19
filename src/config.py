import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import datetime
from pathlib import Path

SEED = 42

TICKER = "^IXIC"
START_DATE = "2016-01-01"
END_DATE = None

SEQ_LEN = int(os.environ.get("SEQ_LEN", 60))
TARGET_HORIZON = int(os.environ.get("TARGET_HORIZON", 1))

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 128))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 2))
DROPOUT = float(os.environ.get("DROPOUT", 0.2))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 100))
PATIENCE = int(os.environ.get("PATIENCE", 10))

RUN_ID = os.environ.get("RUN_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = Path("data") / "runs" / RUN_ID

DATA_PATH = "data/nasdaq_features.json"
MODEL_PATH = str(RUN_DIR / "best_model.pt")
CM_PATH = str(RUN_DIR / "confusion_matrices.png")
ROC_PATH = str(RUN_DIR / "roc_curves.png")
METRICS_PATH = str(RUN_DIR / "metrics.txt")
SETTINGS_PATH = str(RUN_DIR / "settings.txt")

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
