import logging
from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from . import config

log = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test in time order without shuffling."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler on training features to remove magnitude differences between indicators."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def transform_split(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> np.ndarray:
    """Apply a pre-fitted scaler to a split and return float32 array."""
    return np.asarray(scaler.transform(df[feature_cols].values), dtype=np.float32)


class NASDAQSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int) -> None:
        """Store scaled feature matrix, binary targets, and sliding-window length."""
        self.features = features
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Number of valid sliding windows (total rows minus warmup length)."""
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return (seq_len × features window, label at the step immediately after the window)."""
        x = torch.from_numpy(self.features[idx : idx + self.seq_len])
        y = torch.tensor(self.targets[idx + self.seq_len], dtype=torch.float32)
        return x, y


def build_dataloaders(
    train_ds: NASDAQSequenceDataset,
    val_ds: NASDAQSequenceDataset,
    test_ds: NASDAQSequenceDataset,
    batch_size: int,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    """Wrap datasets in DataLoaders; only train is shuffled to preserve val/test time order."""
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_json(config.DATA_PATH, orient="records")
    log.info("Loaded %d rows from %s", len(df), config.DATA_PATH)

    train_df, val_df, test_df = chronological_split(
        df, config.TRAIN_RATIO, config.VAL_RATIO
    )
    log.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    scaler = fit_scaler(train_df, config.FEATURE_COLS)

    train_x = transform_split(train_df, scaler, config.FEATURE_COLS)
    val_x = transform_split(val_df, scaler, config.FEATURE_COLS)
    test_x = transform_split(test_df, scaler, config.FEATURE_COLS)

    train_y = cast(np.ndarray, train_df["target"].values)
    val_y = cast(np.ndarray, val_df["target"].values)
    test_y = cast(np.ndarray, test_df["target"].values)

    train_ds = NASDAQSequenceDataset(train_x, train_y, config.SEQ_LEN)
    val_ds = NASDAQSequenceDataset(val_x, val_y, config.SEQ_LEN)
    test_ds = NASDAQSequenceDataset(test_x, test_y, config.SEQ_LEN)

    train_dl, val_dl, test_dl = build_dataloaders(
        train_ds, val_ds, test_ds, config.BATCH_SIZE
    )

    x_batch, y_batch = next(iter(train_dl))
    log.info(
        "Train batch X shape: %s  y shape: %s",
        tuple(x_batch.shape),
        tuple(y_batch.shape),
    )
