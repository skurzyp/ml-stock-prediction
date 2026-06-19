import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import NASDAQSequenceDataset, build_dataloaders

log = logging.getLogger(__name__)


class ModelStrategy(Protocol):
    name: str

    def prepare_train_data(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        seq_len: int,
        batch_size: int,
    ) -> Any: ...

    def fit(
        self, train_data: Any, model_path: str, device: torch.device
    ) -> dict[str, list[float]]: ...

    def prepare_eval_data(
        self, test_x: np.ndarray, test_y: np.ndarray, seq_len: int, batch_size: int
    ) -> Any: ...

    def evaluate(
        self, eval_data: Any, model_path: str, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class PyTorchStrategy:
    def __init__(self, name: str, builder_fn: Callable[[], nn.Module]):
        self.name = name
        self.builder_fn = builder_fn

    def prepare_train_data(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        seq_len: int,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        train_ds = NASDAQSequenceDataset(train_x, train_y, seq_len)
        val_ds = NASDAQSequenceDataset(val_x, val_y, seq_len)
        train_dl, val_dl, _ = build_dataloaders(train_ds, val_ds, val_ds, batch_size)
        return train_dl, val_dl

    def prepare_eval_data(
        self, test_x: np.ndarray, test_y: np.ndarray, seq_len: int, batch_size: int
    ) -> DataLoader:
        test_ds = NASDAQSequenceDataset(test_x, test_y, seq_len)
        _, _, test_dl = build_dataloaders(test_ds, test_ds, test_ds, batch_size)
        return test_dl

    def fit(
        self,
        train_data: tuple[DataLoader, DataLoader],
        model_path: str,
        device: torch.device,
    ) -> dict[str, list[float]]:
        train_dl, val_dl = train_data
        model = self.builder_fn()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, config.MAX_EPOCHS + 1):
            # Train Epoch
            model.train()
            total_train_loss = 0.0
            for x_batch, y_batch in train_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)
                y_batch = y_batch.view_as(logits)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += loss.item()
            train_loss = total_train_loss / len(train_dl)

            # Val Epoch
            val_loss, val_acc = self._run_evaluate(model, val_dl, criterion, device)

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
                if epochs_without_improvement >= config.PATIENCE:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        model.load_state_dict(torch.load(model_path, map_location=device))
        log.info("Restored best checkpoint from %s", model_path)
        return history

    @torch.no_grad()
    def _run_evaluate(
        self,
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
            y_batch = y_batch.view_as(logits)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        return total_loss / len(loader), correct / total

    def evaluate(
        self, eval_data: DataLoader, model_path: str, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.builder_fn()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for x_batch, y_batch in eval_data:
                logits = model(x_batch.to(device))
                y_batch = y_batch.view_as(logits)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_labels.append(y_batch.cpu().numpy())
                all_probs.append(probs)

        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        preds = (probs >= 0.5).astype(int)
        return labels, preds, probs
