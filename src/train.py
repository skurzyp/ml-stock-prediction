import argparse
import logging
import subprocess
import sys
from typing import cast
import numpy as np
import pandas as pd
import torch

from . import config, model_lstm, model_cnn
from .dataset import chronological_split, fit_scaler, transform_split
from .strategy import ModelStrategy, PyTorchStrategy

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ALL")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.model == "ALL":
        for m in ["LSTM", "CNN"]:
            log.info("Spawning subprocess for %s training...", m)
            subprocess.run(
                [sys.executable, "-m", "src.train", "--model", m], check=True
            )
        return

    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[%s] Using device: %s", args.model, device)

    df = pd.read_json(config.DATA_PATH, orient="records")
    train_df, val_df, _ = chronological_split(df, config.TRAIN_RATIO, config.VAL_RATIO)

    scaler = fit_scaler(train_df, config.FEATURE_COLS)
    train_x = transform_split(train_df, scaler, config.FEATURE_COLS)
    val_x = transform_split(val_df, scaler, config.FEATURE_COLS)

    train_y = cast(np.ndarray, train_df["target"].values)
    val_y = cast(np.ndarray, val_df["target"].values)

    n_features = len(config.FEATURE_COLS)

    strategies: list[ModelStrategy] = [
        PyTorchStrategy(
            "LSTM",
            lambda: model_lstm.build_model(
                n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
            ),
        ),
        PyTorchStrategy(
            "CNN",
            lambda: model_cnn.build_model(
                n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
            ),
        ),
    ]

    for strategy in strategies:
        if strategy.name != args.model:
            continue

        log.info("=== Training %s ===", strategy.name)
        torch.manual_seed(config.SEED)

        path = config.MODEL_PATH.replace(
            "best_model.pt", f"best_model_{strategy.name.lower()}.pt"
        )

        train_data = strategy.prepare_train_data(
            train_x, train_y, val_x, val_y, config.SEQ_LEN, config.BATCH_SIZE
        )
        _ = strategy.fit(train_data, path, device)


if __name__ == "__main__":
    main()
