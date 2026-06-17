import logging
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config

log = logging.getLogger(__name__)


class ClassifierModel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self) -> ...: ...  # type: ignore[override]
    def train(self, mode: bool = True) -> "ClassifierModel": ...
    def eval(self) -> "ClassifierModel": ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / len(loader), correct / total


def fit(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    model_path: str,
    learning_rate: float = config.LEARNING_RATE,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.PATIENCE,
    max_grad_norm: float = 1.0,
) -> dict[str, list[float]]:
    """Train model with early stopping on val loss. Saves best checkpoint to model_path."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            model, train_dl, optimizer, criterion, device, max_grad_norm
        )
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log.info(
            "Epoch %3d | train_loss %.4f | val_loss %.4f | val_acc %.4f",
            epoch,
            train_loss,
            val_loss,
            val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            log.info("Saved best model (val_loss %.4f)", best_val_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    log.info("Restored best checkpoint from %s", model_path)
    return history


if __name__ == "__main__":
    import numpy as np
    from typing import cast

    from . import model_lstm, model_cnn
    from .dataset import (
        NASDAQSequenceDataset,
        build_dataloaders,
        chronological_split,
        fit_scaler,
        transform_split,
    )
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    df = pd.read_json(config.DATA_PATH, orient="records")
    train_df, val_df, test_df = chronological_split(
        df, config.TRAIN_RATIO, config.VAL_RATIO
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

    n_features = len(config.FEATURE_COLS)

    for name, mod in [
        (
            "lstm",
            model_lstm.build_model(
                n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
            ),
        ),
        (
            "cnn",
            model_cnn.build_model(
                n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
            ),
        ),
    ]:
        log.info("=== Training %s ===", name.upper())
        torch.manual_seed(config.SEED)
        path = config.MODEL_PATH.replace(".pt", f"_{name}.pt")
        history = fit(mod, train_dl, val_dl, device, path)

        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_acc = evaluate(mod, test_dl, criterion, device)
        log.info(
            "%s — test_loss %.4f | test_acc %.4f", name.upper(), test_loss, test_acc
        )
